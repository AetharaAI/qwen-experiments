# Native CustomVoice First Run

## Intent

This is the first runner harness for the Qwen spike.

It is not the final streaming answer.

It is the first controlled proof point for:

- environment correctness
- model load behavior
- generation success
- structured logging

## Current Truth

The harness currently validates the Python-native path as a generation probe.

It does not claim to prove true incremental streaming yet.

That is intentional.

First prove:

1. the model loads cleanly
2. the model generates cleanly
3. the logging surface is solid

Then extend or replace the harness for streaming-specific transport tests.

## Example Flow

1. copy [/.env.example](/home/cory/Aether-Voice-Platform/qwen-experiments/.env.example) to `.env`
2. adjust model paths for the VM if needed
3. install dependencies with:

```bash
uv sync --extra native
```

4. run a first probe:

```bash
uv run --extra native python -m qwen_experiments.cli native-customvoice \
  --text "Hello from the Qwen native custom voice spike." \
  --output-name first-run.wav
```

## Logs

Every run emits JSONL events into `logs/`.

The minimum events expected are:

- `cli_started`
- `model_load_started`
- `model_load_completed`
- `request_started`
- `request_completed`
- `cli_completed`
