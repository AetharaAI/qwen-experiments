from __future__ import annotations

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException

from .config import ExperimentConfig
from .event_log import EventLogger
from .native_customvoice import NativeCustomVoiceRunner
from .provider_models import (
    ProviderModelInfo,
    ProviderSpeechRequest,
    ProviderSpeechResponse,
    ProviderVoiceInfo,
    ProviderWarmupResponse,
)

logger = logging.getLogger("qwen_provider")


class ProviderRuntime:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.event_logger = EventLogger(config.log_dir / "qwen-provider.jsonl")
        self.runner = NativeCustomVoiceRunner(config, self.event_logger)
        self.load_lock = asyncio.Lock()
        self.generate_lock = asyncio.Lock()
        self.loaded = False

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

    def voice_info(self) -> list[ProviderVoiceInfo]:
        return [
            ProviderVoiceInfo(id=voice, label=voice, language=self.config.provider_default_language, tags=["builtin", "qwen"])
            for voice in self.config.default_voices
        ]

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
    return {"models": [runtime.model_info().model_dump()]}


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


@app.post("/v1/audio/speech/stream")
async def audio_speech_stream() -> dict:
    raise HTTPException(status_code=501, detail="Streaming is not implemented in the provider yet.")


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
