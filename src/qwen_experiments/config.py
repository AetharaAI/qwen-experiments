from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class ExperimentConfig:
    model_id: str
    model_path: str
    tokenizer_path: str
    device: str
    dtype: str
    attn_implementation: str
    language: str
    speaker: str
    instruct: str
    output_dir: Path
    log_dir: Path
    default_voices: list[str]
    provider_host: str
    provider_port: int
    provider_public_base_url: str
    provider_model_alias: str
    provider_default_language: str
    provider_default_response_format: str
    provider_timeout_seconds: float

    @classmethod
    def from_env(cls, env_file: str | None = None) -> "ExperimentConfig":
        if env_file:
            load_dotenv(env_file, override=False)
        else:
            load_dotenv(override=False)

        output_dir = Path(os.getenv("QWEN_EXPERIMENT_OUTPUT_DIR", "results/native_customvoice"))
        log_dir = Path(os.getenv("QWEN_EXPERIMENT_LOG_DIR", "logs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            model_id=os.getenv("QWEN_EXPERIMENT_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
            model_path=os.getenv(
                "QWEN_EXPERIMENT_MODEL_PATH",
                "/mnt/aetherpro/models/voice/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            ),
            tokenizer_path=os.getenv(
                "QWEN_EXPERIMENT_TOKENIZER_PATH",
                "/mnt/aetherpro/models/voice/Qwen3-TTS-Tokenizer-12Hz",
            ),
            device=os.getenv("QWEN_EXPERIMENT_DEVICE", "cuda:0"),
            dtype=os.getenv("QWEN_EXPERIMENT_DTYPE", "bfloat16"),
            attn_implementation=os.getenv(
                "QWEN_EXPERIMENT_ATTN_IMPLEMENTATION",
                "flash_attention_2",
            ),
            language=os.getenv("QWEN_EXPERIMENT_LANGUAGE", "English"),
            speaker=os.getenv("QWEN_EXPERIMENT_SPEAKER", "Ryan"),
            instruct=os.getenv("QWEN_EXPERIMENT_INSTRUCT", ""),
            output_dir=output_dir,
            log_dir=log_dir,
            default_voices=[
                voice.strip()
                for voice in os.getenv(
                    "QWEN_EXPERIMENT_DEFAULT_VOICES",
                    "Ryan,Aiden,Uncle_Fu,Serena,Vivian,Sohee",
                ).split(",")
                if voice.strip()
            ],
            provider_host=os.getenv("QWEN_PROVIDER_HOST", "0.0.0.0"),
            provider_port=int(os.getenv("QWEN_PROVIDER_PORT", "8072")),
            provider_public_base_url=os.getenv("QWEN_PROVIDER_PUBLIC_BASE_URL", "http://qwen-provider:8072"),
            provider_model_alias=os.getenv("QWEN_PROVIDER_MODEL_ALIAS", "qwen_customvoice"),
            provider_default_language=os.getenv("QWEN_PROVIDER_DEFAULT_LANGUAGE", "English"),
            provider_default_response_format=os.getenv("QWEN_PROVIDER_DEFAULT_RESPONSE_FORMAT", "wav"),
            provider_timeout_seconds=float(os.getenv("QWEN_PROVIDER_TIMEOUT_SECONDS", "180")),
        )
