from __future__ import annotations

import io
import importlib.util
import time
from pathlib import Path

import soundfile as sf

from .config import ExperimentConfig
from .event_log import EventLogger


class NativeCustomVoiceRunner:
    def __init__(self, config: ExperimentConfig, logger: EventLogger) -> None:
        self.config = config
        self.logger = logger
        self.model = None

    def _resolve_attn_implementation(self) -> str:
        requested_attn = self.config.attn_implementation
        if requested_attn == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
            self.logger.emit(
                "attn_implementation_fallback",
                runner="native",
                model_id=self.config.model_id,
                requested_attn_implementation=requested_attn,
                resolved_attn_implementation="sdpa",
                reason="flash_attn_not_installed",
            )
            return "sdpa"
        return requested_attn

    def load(self) -> None:
        load_started = time.perf_counter()
        resolved_attn = self._resolve_attn_implementation()

        self.logger.emit(
            "model_load_started",
            runner="native",
            model_id=self.config.model_id,
            model_path=self.config.model_path,
            device=self.config.device,
            dtype=self.config.dtype,
            attn_implementation=resolved_attn,
        )
        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            dtype_name = self.config.dtype
            torch_dtype = getattr(torch, dtype_name)
            model_source = self.config.model_path or self.config.model_id

            self.model = Qwen3TTSModel.from_pretrained(
                model_source,
                device_map=self.config.device,
                dtype=torch_dtype,
                attn_implementation=resolved_attn,
            )
        except Exception as exc:
            self.logger.emit(
                "model_load_failed",
                runner="native",
                model_id=self.config.model_id,
                error=str(exc),
            )
            raise

        self.logger.emit(
            "model_load_completed",
            runner="native",
            model_id=self.config.model_id,
            attn_implementation=resolved_attn,
            elapsed_ms=round((time.perf_counter() - load_started) * 1000, 2),
        )

    def synthesize_to_bytes(
        self,
        *,
        text: str,
        speaker: str | None = None,
        language: str | None = None,
        instruct: str | None = None,
    ) -> tuple[bytes, int, float]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        speaker_value = speaker or self.config.speaker
        language_value = language or self.config.language
        instruct_value = self.config.instruct if instruct is None else instruct

        request_started = time.perf_counter()
        self.logger.emit(
            "request_started",
            runner="native",
            model_id=self.config.model_id,
            output_mode="batch_probe",
            text_chars=len(text),
            speaker=speaker_value,
            language=language_value,
            has_instruct=bool(instruct_value),
        )

        try:
            wavs, sample_rate = self.model.generate_custom_voice(
                text=text,
                language=language_value,
                speaker=speaker_value,
                instruct=instruct_value or None,
            )
        except Exception as exc:
            self.logger.emit(
                "request_failed",
                runner="native",
                model_id=self.config.model_id,
                error=str(exc),
            )
            raise

        elapsed_ms = round((time.perf_counter() - request_started) * 1000, 2)
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sample_rate, format="WAV")
        return buffer.getvalue(), sample_rate, elapsed_ms

    def synthesize(
        self,
        *,
        text: str,
        output_path: Path,
        speaker: str | None = None,
        language: str | None = None,
        instruct: str | None = None,
    ) -> Path:
        audio_bytes, sample_rate, elapsed_ms = self.synthesize_to_bytes(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio_bytes)

        self.logger.emit(
            "request_completed",
            runner="native",
            model_id=self.config.model_id,
            output_mode="batch_probe",
            output_path=str(output_path),
            sample_rate=sample_rate,
            elapsed_ms=elapsed_ms,
        )
        return output_path
