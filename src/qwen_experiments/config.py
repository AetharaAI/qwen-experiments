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
        )
