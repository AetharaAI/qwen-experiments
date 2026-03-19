from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException

from .config import ExperimentConfig
from .event_log import EventLogger
from .native_customvoice import NativeCustomVoiceRunner
from .provider_models import (
    ProviderModelInfo,
    ProviderSpeechRequest,
    ProviderSpeechResponse,
    ProviderStreamEndResponse,
    ProviderStreamStartRequest,
    ProviderTextChunkRequest,
    ProviderTextEventsResponse,
    ProviderVoiceInfo,
    ProviderWarmupResponse,
)

logger = logging.getLogger("qwen_provider")


@dataclass(slots=True)
class ProviderStreamState:
    session_id: str
    model: str
    voice: str
    sample_rate: int
    output_format: str
    context_mode: str
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.perf_counter)
    first_chunk_at: float | None = None
    chunk_count: int = 0
    audio_arrays: list[np.ndarray] = field(default_factory=list)
    text_fragments: list[str] = field(default_factory=list)
    cumulative_inference_ms: int = 0


class ProviderRuntime:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.event_logger = EventLogger(config.log_dir / "qwen-provider.jsonl")
        self.runner = NativeCustomVoiceRunner(config, self.event_logger)
        self.load_lock = asyncio.Lock()
        self.generate_lock = asyncio.Lock()
        self.loaded = False
        self.sessions: dict[str, ProviderStreamState] = {}

    async def ensure_loaded(self) -> None:
        if self.loaded:
            return
        async with self.load_lock:
            if self.loaded:
                return
            await asyncio.to_thread(self.runner.load)
            self.loaded = True

    def model_info(self) -> ProviderModelInfo:
        return ProviderModelInfo(
            id=self.config.provider_model_alias,
            label=f"{self.config.provider_model_alias} ({self.config.model_id})",
            default_voice=self.config.speaker,
        )

    def streaming_model_info(self) -> ProviderModelInfo:
        return ProviderModelInfo(
            id=self.config.provider_streaming_model_alias,
            label=f"{self.config.provider_streaming_model_alias} ({self.config.model_id})",
            default_voice=self.config.speaker,
            supports_batch=False,
            supports_streaming=True,
        )

    def voice_info(self) -> list[ProviderVoiceInfo]:
        return [
            ProviderVoiceInfo(id=voice, label=voice, language=self.config.provider_default_language, tags=["builtin", "qwen"])
            for voice in self.config.default_voices
        ]

    @staticmethod
    def _split_text(text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if not normalized:
            return []
        parts = re.split(r"([.!?]+\s+)", normalized)
        chunks: list[str] = []
        current = ""
        for index in range(0, len(parts), 2):
            sentence = parts[index].strip()
            punctuation = parts[index + 1] if index + 1 < len(parts) else ""
            candidate = f"{sentence}{punctuation}".strip()
            if candidate:
                chunks.append(candidate)
            elif sentence:
                current = sentence
        if current:
            chunks.append(current)
        return chunks or [normalized]

    @staticmethod
    def _decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim > 1:
            array = np.mean(array, axis=1, dtype=np.float32)
        return array, int(sample_rate)

    @staticmethod
    def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        return buffer.getvalue()

    @staticmethod
    def _instruction_from_metadata(metadata: dict[str, Any] | None) -> str | None:
        if not isinstance(metadata, dict):
            return None
        extra = metadata.get("extra")
        if not isinstance(extra, dict):
            return None
        explicit = extra.get("qwen_instructions")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        resolved_voice = extra.get("resolved_voice")
        if isinstance(resolved_voice, dict):
            generation_prompt = resolved_voice.get("generation_prompt")
            if isinstance(generation_prompt, str) and generation_prompt.strip():
                return generation_prompt.strip()
        return None

    async def warmup(self) -> ProviderWarmupResponse:
        started = time.perf_counter()
        await self.ensure_loaded()
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        self.event_logger.emit(
            "provider_warmup_completed",
            model=self.config.provider_model_alias,
            elapsed_ms=elapsed_ms,
        )
        return ProviderWarmupResponse(
            status="ready",
            model=self.config.provider_model_alias,
            ready=True,
            elapsed_ms=elapsed_ms,
        )

    async def synthesize(self, payload: ProviderSpeechRequest) -> ProviderSpeechResponse:
        if payload.model != self.config.provider_model_alias:
            raise HTTPException(status_code=400, detail=f"Unsupported model '{payload.model}'.")
        await self.ensure_loaded()
        async with self.generate_lock:
            started = time.perf_counter()
            audio_bytes, sample_rate, inference_ms = await asyncio.to_thread(
                self.runner.synthesize_to_bytes,
                text=payload.input,
                speaker=payload.voice,
                language=payload.language or self.config.provider_default_language,
                instruct=payload.instructions,
            )
            total_ms = round((time.perf_counter() - started) * 1000, 2)
        return ProviderSpeechResponse(
            model=self.config.provider_model_alias,
            format=payload.response_format,
            sample_rate=sample_rate,
            audio_b64=base64.b64encode(audio_bytes).decode("ascii"),
            timings={
                "inference_ms": int(inference_ms),
                "total_ms": int(total_ms),
            },
            artifacts={
                "runtime_path_used": self.config.provider_model_alias,
                "qwen_model_id": self.config.model_id,
                "qwen_voice": payload.voice,
                "qwen_language": payload.language or self.config.provider_default_language,
                "provider_public_base_url": self.config.provider_public_base_url,
                "supports_streaming_contract": False,
            },
        )

    async def start_stream(self, payload: ProviderStreamStartRequest) -> dict[str, Any]:
        if payload.model != self.config.provider_streaming_model_alias:
            raise HTTPException(status_code=400, detail=f"Unsupported streaming model '{payload.model}'.")
        await self.ensure_loaded()
        self.sessions[payload.session_id] = ProviderStreamState(
            session_id=payload.session_id,
            model=payload.model,
            voice=payload.voice,
            sample_rate=payload.sample_rate,
            output_format=payload.format,
            context_mode=payload.context_mode,
            metadata=dict(payload.metadata or {}),
        )
        self.event_logger.emit(
            "provider_stream_started",
            session_id=payload.session_id,
            model=payload.model,
            voice=payload.voice,
            sample_rate=payload.sample_rate,
        )
        return {
            "session_id": payload.session_id,
            "model": payload.model,
            "expires_in_seconds": 3600,
            "voice": payload.voice,
        }

    async def push_stream_text(self, session_id: str, text: str) -> list[dict[str, Any]]:
        state = self.sessions.get(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Unknown stream session '{session_id}'.")
        sentences = self._split_text(text)
        if not sentences:
            return []

        events: list[dict[str, Any]] = []
        instruction = self._instruction_from_metadata(state.metadata)
        for sentence in sentences:
            async with self.generate_lock:
                started = time.perf_counter()
                audio_bytes, sample_rate, inference_ms = await asyncio.to_thread(
                    self.runner.synthesize_to_bytes,
                    text=sentence,
                    speaker=state.voice,
                    language=self.config.provider_default_language,
                    instruct=instruction,
                )
                total_ms = round((time.perf_counter() - started) * 1000, 2)
            audio_array, decoded_sample_rate = self._decode_audio(audio_bytes)
            resolved_sample_rate = decoded_sample_rate or sample_rate or state.sample_rate
            state.audio_arrays.append(audio_array)
            state.text_fragments.append(sentence)
            state.chunk_count += 1
            state.cumulative_inference_ms += int(inference_ms)
            if state.first_chunk_at is None:
                state.first_chunk_at = time.perf_counter()
            events.append(
                {
                    "type": "audio_chunk",
                    "session_id": session_id,
                    "sequence": state.chunk_count,
                    "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
                    "format": state.output_format,
                    "metadata": {
                        "provider": self.config.provider_streaming_model_alias,
                        "voice": state.voice,
                        "chunk_text": sentence,
                        "timings": {
                            "inference_ms": int(inference_ms),
                            "total_ms": int(total_ms),
                        },
                    },
                }
            )
        self.event_logger.emit(
            "provider_stream_chunk_batch_completed",
            session_id=session_id,
            chunk_events=len(events),
            chunk_count=state.chunk_count,
        )
        return events

    async def complete_stream_text(self, session_id: str) -> list[dict[str, Any]]:
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail=f"Unknown stream session '{session_id}'.")
        self.event_logger.emit("provider_stream_text_completed", session_id=session_id)
        return []

    async def end_stream(self, session_id: str) -> ProviderStreamEndResponse:
        state = self.sessions.pop(session_id, None)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Unknown stream session '{session_id}'.")
        if state.audio_arrays:
            final_audio = np.concatenate(state.audio_arrays).astype(np.float32)
        else:
            final_audio = np.zeros(int(max(state.sample_rate, 1) * 0.1), dtype=np.float32)
        audio_bytes = self._encode_wav(final_audio, state.sample_rate)
        duration_ms = int((len(final_audio) / max(state.sample_rate, 1)) * 1000)
        total_ms = round((time.perf_counter() - state.started_at) * 1000, 2)
        first_chunk_ms = int((state.first_chunk_at - state.started_at) * 1000) if state.first_chunk_at is not None else 0
        self.event_logger.emit(
            "provider_stream_finished",
            session_id=session_id,
            model=state.model,
            voice=state.voice,
            duration_ms=duration_ms,
            total_ms=total_ms,
            first_chunk_ms=first_chunk_ms,
            chunk_count=state.chunk_count,
        )
        return ProviderStreamEndResponse(
            model=state.model,
            format="wav",
            duration_ms=duration_ms,
            sample_rate=state.sample_rate,
            audio_b64=base64.b64encode(audio_bytes).decode("ascii"),
            timings={
                "first_chunk_ms": first_chunk_ms,
                "inference_ms": int(state.cumulative_inference_ms),
                "total_ms": int(total_ms),
            },
            artifacts={
                "runtime_path_used": state.model,
                "qwen_model_id": self.config.model_id,
                "qwen_voice": state.voice,
                "qwen_language": self.config.provider_default_language,
                "supports_streaming_contract": True,
                "provider_public_base_url": self.config.provider_public_base_url,
            },
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = ExperimentConfig.from_env()
    app.state.runtime = ProviderRuntime(config)
    logger.info("qwen_provider_starting", extra={"model": config.model_id, "alias": config.provider_model_alias})
    yield


app = FastAPI(title="Qwen Provider", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    runtime: ProviderRuntime = app.state.runtime
    return {
        "status": "ok",
        "service": "qwen-provider",
        "model": runtime.config.provider_model_alias,
        "ready": runtime.loaded,
    }


@app.get("/v1/models")
async def models() -> dict:
    runtime: ProviderRuntime = app.state.runtime
    return {"models": [runtime.model_info().model_dump(), runtime.streaming_model_info().model_dump()]}


@app.get("/v1/voices")
async def voices() -> dict:
    runtime: ProviderRuntime = app.state.runtime
    return {"voices": [voice.model_dump() for voice in runtime.voice_info()]}


@app.post("/v1/warmup")
async def warmup() -> dict:
    runtime: ProviderRuntime = app.state.runtime
    return (await runtime.warmup()).model_dump()


@app.post("/v1/audio/speech")
async def audio_speech(payload: ProviderSpeechRequest) -> dict:
    runtime: ProviderRuntime = app.state.runtime
    return (await runtime.synthesize(payload)).model_dump()


@app.post("/v1/stream/start")
async def stream_start(payload: ProviderStreamStartRequest) -> dict:
    runtime: ProviderRuntime = app.state.runtime
    return await runtime.start_stream(payload)


@app.post("/v1/stream/{session_id}/text")
async def stream_text(session_id: str, payload: ProviderTextChunkRequest) -> dict:
    runtime: ProviderRuntime = app.state.runtime
    return ProviderTextEventsResponse(events=await runtime.push_stream_text(session_id, payload.text)).model_dump()


@app.post("/v1/stream/{session_id}/complete")
async def stream_complete(session_id: str) -> dict:
    runtime: ProviderRuntime = app.state.runtime
    return ProviderTextEventsResponse(events=await runtime.complete_stream_text(session_id)).model_dump()


@app.post("/v1/stream/{session_id}/end")
async def stream_end(session_id: str) -> dict:
    runtime: ProviderRuntime = app.state.runtime
    return (await runtime.end_stream(session_id)).model_dump()


@app.post("/v1/audio/speech/stream")
async def audio_speech_stream(payload: ProviderSpeechRequest) -> dict:
    runtime: ProviderRuntime = app.state.runtime
    session_id = f"speech_stream_{int(time.time() * 1000)}"
    await runtime.start_stream(
        ProviderStreamStartRequest(
            session_id=session_id,
            model=runtime.config.provider_streaming_model_alias,
            voice=payload.voice,
            sample_rate=24000,
            format=payload.response_format,
            context_mode="conversation",
            metadata=payload.metadata,
        )
    )
    events = await runtime.push_stream_text(session_id, payload.input)
    return ProviderTextEventsResponse(events=events).model_dump()


def main() -> None:
    config = ExperimentConfig.from_env()
    uvicorn.run(
        "qwen_experiments.provider_server:app",
        host=config.provider_host,
        port=config.provider_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
