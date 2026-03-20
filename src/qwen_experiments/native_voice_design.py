from __future__ import annotations

import io
import importlib.util
import time

import soundfile as sf

from .config import ExperimentConfig
from .event_log import EventLogger


class NativeVoiceDesignRunner:
    def __init__(self, config: ExperimentConfig, logger: EventLogger) -> None:
        self.config = config
        self.logger = logger
        self.model = None

    def _resolve_attn_implementation(self) -> str:
        requested_attn = self.config.voice_design_attn_implementation
        if requested_attn == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
            self.logger.emit(
                "voice_design_attn_implementation_fallback",
                runner="native",
                model_id=self.config.voice_design_model_id,
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
            "voice_design_model_load_started",
            runner="native",
            model_id=self.config.voice_design_model_id,
            model_path=self.config.voice_design_model_path,
            device=self.config.voice_design_device,
            dtype=self.config.voice_design_dtype,
            attn_implementation=resolved_attn,
        )
        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            dtype_name = self.config.voice_design_dtype
            torch_dtype = getattr(torch, dtype_name)
            model_source = self.config.voice_design_model_path or self.config.voice_design_model_id

            self.model = Qwen3TTSModel.from_pretrained(
                model_source,
                device_map=self.config.voice_design_device,
                dtype=torch_dtype,
                attn_implementation=resolved_attn,
            )
        except Exception as exc:
            self.logger.emit(
                "voice_design_model_load_failed",
                runner="native",
                model_id=self.config.voice_design_model_id,
                error=str(exc),
            )
            raise

        self.logger.emit(
            "voice_design_model_load_completed",
            runner="native",
            model_id=self.config.voice_design_model_id,
            attn_implementation=resolved_attn,
            elapsed_ms=round((time.perf_counter() - load_started) * 1000, 2),
        )

    def synthesize_to_bytes(
        self,
        *,
        text: str,
        language: str | None = None,
        instruct: str | None = None,
    ) -> tuple[bytes, int, float]:
        if self.model is None:
            raise RuntimeError("VoiceDesign model is not loaded")

        language_value = language or self.config.provider_default_language
        instruct_value = (instruct or "").strip()
        if not instruct_value:
            raise ValueError("VoiceDesign requires a non-empty instruction prompt.")

        request_started = time.perf_counter()
        self.logger.emit(
            "voice_design_request_started",
            runner="native",
            model_id=self.config.voice_design_model_id,
            output_mode="batch_probe",
            text_chars=len(text),
            language=language_value,
            has_instruct=True,
        )

        try:
            wavs, sample_rate = self.model.generate_voice_design(
                text=text,
                language=language_value,
                instruct=instruct_value,
            )
        except Exception as exc:
            self.logger.emit(
                "voice_design_request_failed",
                runner="native",
                model_id=self.config.voice_design_model_id,
                error=str(exc),
            )
            raise

        elapsed_ms = round((time.perf_counter() - request_started) * 1000, 2)
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sample_rate, format="WAV")
        return buffer.getvalue(), sample_rate, elapsed_ms
