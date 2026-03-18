from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .config import ExperimentConfig
from .event_log import EventLogger
from .native_customvoice import NativeCustomVoiceRunner
from .prompt_pack import DEFAULT_PROMPT_PACK, PromptCase, prompt_pack_manifest


def run_native_customvoice_pack(
    *,
    config: ExperimentConfig,
    logger: EventLogger,
    voices: list[str],
    language: str | None = None,
    instruct: str | None = None,
    output_subdir: str | None = None,
) -> Path:
    run_started = time.perf_counter()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = output_subdir or f"pack-{timestamp}"
    run_dir = config.output_dir / run_name
    structured_dir = run_dir / "structured"
    review_dir = run_dir / "review"
    structured_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    playlist_path = review_dir / "review.m3u"

    logger.emit(
        "pack_run_started",
        command="native-customvoice-pack",
        run_dir=str(run_dir),
        voices=voices,
        prompt_count=len(DEFAULT_PROMPT_PACK),
        language=language or config.language,
        has_instruct=bool(config.instruct if instruct is None else instruct),
    )

    runner = NativeCustomVoiceRunner(config, logger)
    runner.load()

    outputs: list[dict[str, object]] = []
    review_playlist_entries: list[str] = ["#EXTM3U"]
    for voice in voices:
        voice_dir = structured_dir / voice.lower()
        voice_dir.mkdir(parents=True, exist_ok=True)
        logger.emit(
            "pack_voice_started",
            command="native-customvoice-pack",
            voice=voice,
            voice_dir=str(voice_dir),
        )
        for prompt_case in DEFAULT_PROMPT_PACK:
            category_dir = voice_dir / prompt_case.category
            category_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{prompt_case.index:02d}_{prompt_case.slug}.wav"
            output_path = category_dir / filename
            started = time.perf_counter()
            runner.synthesize(
                text=prompt_case.text,
                output_path=output_path,
                speaker=voice,
                language=language,
                instruct=instruct,
            )
            review_filename = f"{prompt_case.index:02d}_{voice.lower()}_{prompt_case.category}_{prompt_case.slug}.wav"
            review_path = review_dir / review_filename
            shutil.copy2(output_path, review_path)
            review_playlist_entries.append(review_filename)
            outputs.append(
                {
                    "voice": voice,
                    "prompt": asdict(prompt_case),
                    "output_path": str(output_path),
                    "review_path": str(review_path),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
        logger.emit(
            "pack_voice_completed",
            command="native-customvoice-pack",
            voice=voice,
            generated_files=len(DEFAULT_PROMPT_PACK),
        )

    playlist_path.write_text("\n".join(review_playlist_entries) + "\n", encoding="utf-8")
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": "native-customvoice-pack",
        "model_id": config.model_id,
        "language": language or config.language,
        "instruct": config.instruct if instruct is None else instruct,
        "voices": voices,
        "prompt_pack": prompt_pack_manifest(),
        "structured_dir": str(structured_dir),
        "review_dir": str(review_dir),
        "playlist_path": str(playlist_path),
        "outputs": outputs,
        "elapsed_ms": round((time.perf_counter() - run_started) * 1000, 2),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.emit(
        "pack_run_completed",
        command="native-customvoice-pack",
        run_dir=str(run_dir),
        manifest_path=str(manifest_path),
        generated_files=len(outputs),
        elapsed_ms=manifest["elapsed_ms"],
    )
    return run_dir
