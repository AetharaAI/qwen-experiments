from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer

from .config import ExperimentConfig
from .event_log import EventLogger
from .native_customvoice import NativeCustomVoiceRunner

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Qwen experiment CLI root."""


@app.command("native-customvoice")
def native_customvoice(
    text: str = typer.Option(..., help="Text to synthesize."),
    output_name: str = typer.Option("sample.wav", help="Output filename inside the configured output dir."),
    speaker: str | None = typer.Option(None, help="Override default speaker."),
    language: str | None = typer.Option(None, help="Override default language."),
    instruct: str | None = typer.Option(None, help="Override default instruction."),
    env_file: str | None = typer.Option(None, help="Optional .env file path."),
) -> None:
    config = ExperimentConfig.from_env(env_file)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = config.log_dir / f"native-customvoice-{timestamp}.jsonl"
    logger = EventLogger(log_path)
    logger.emit(
        "cli_started",
        command="native-customvoice",
        env_file=env_file or ".env",
        output_name=output_name,
    )

    runner = NativeCustomVoiceRunner(config, logger)
    runner.load()
    output_path = config.output_dir / output_name
    runner.synthesize(
        text=text,
        output_path=output_path,
        speaker=speaker,
        language=language,
        instruct=instruct,
    )

    logger.emit(
        "cli_completed",
        command="native-customvoice",
        output_path=str(output_path),
        log_path=str(log_path),
    )


if __name__ == "__main__":
    app()
