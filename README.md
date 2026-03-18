# Qwen Experiments

This directory is intentionally outside of `Aether-Voice-X`.

It exists to test Qwen TTS runtime options without coupling early decisions to the production voice platform codebase.

Current purpose:

- validate `Qwen3-TTS-12Hz-1.7B-CustomVoice`
- validate `Qwen3-TTS-12Hz-1.7B-Base`
- compare runner strategies before integrating anything into `Aether-Voice-X`

Ground rules:

- stream first
- batch is fallback only
- log every meaningful handshake and state transition
- do not assume docs are truthful until runtime proves them
- do not edit the main platform to fit an unproven runner

Primary question:

Which runtime path is the best operational fit for Aether Voice?

- Python-native `qwen-tts`
- `vLLM-Omni`

The answer must come from runtime truth:

- boot reliability
- actual streaming behavior
- first-audio latency
- chunk cadence
- voice quality
- repeatability
- failure clarity
- GPU behavior

The main repo stays clean until one runner clearly wins.

## Environment

This workspace is managed with `uv`.

Target Python:

- `3.12`

The project file is:

- [pyproject.toml](/home/cory/Aether-Voice-Platform/qwen-experiments/pyproject.toml)

Expected dependency lanes:

- base harness and logging utilities
- optional `native` extras for Python-native `qwen-tts`
- optional `vllm` extras for `vLLM-Omni`
- `dev` tools for linting and tests
