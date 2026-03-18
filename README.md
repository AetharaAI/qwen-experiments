# Qwen Experiments

This directory is intentionally outside of `Aether-Voice-X`.

It exists to test Qwen TTS runtime options without coupling early decisions to the production voice platform codebase.

Current purpose:

- validate `Qwen3-TTS-12Hz-1.7B-CustomVoice`
- validate `Qwen3-TTS-12Hz-1.7B-Base`
- compare runner strategies before integrating anything into `Aether-Voice-X`
- generate reusable brand, site, telephony, and ad asset packs
- expose a modular HTTP provider boundary for the production stack

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

## Current outputs

The native prompt-pack runner writes:

- `structured/` voice and category folders for reusable assets
- `review/` flat copies for fast listening in VLC or Audacity
- `review/review.m3u` playlist for one-pass review
- `manifest.json` so every output stays tied to voice, prompt, and model truth

## Provider boundary

The experiment repo also contains the first modular Qwen provider container surface:

- HTTP contract doc: `docs/QWEN_PROVIDER_API_CONTRACT_2026-03-18.md`
- FastAPI app: `src/qwen_experiments/provider_server.py`
- container build: `Dockerfile`
- shared network compose: `docker-compose.yml`

This lets `Aether-Voice-X` call Qwen over a shared Docker network instead of embedding runner logic directly.

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
