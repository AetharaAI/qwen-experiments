# Qwen Provider API Contract - 2026-03-18

## Purpose

This contract defines the modular HTTP provider boundary between `qwen-experiments` and the production `Aether-Voice-X` stack.

The provider runs independently and joins the shared Docker network:

- `aether-voice-mesh`

The production stack talks to it over HTTP instead of embedding runner logic directly.

## Current Scope

Current implemented lane:

- `qwen_customvoice`

Current status:

- batch generation: implemented
- warmup: implemented
- model discovery: implemented
- voice discovery: implemented
- streaming endpoint: contract stub only

## Endpoints

### `GET /health`

Returns provider liveness and whether the model is already loaded.

### `GET /v1/models`

Returns available provider models.

Current result includes:

- `qwen_customvoice`

### `GET /v1/voices`

Returns built-in voice choices exposed by the provider.

### `POST /v1/warmup`

Forces model load and returns readiness metadata.

### `POST /v1/audio/speech`

Primary batch generation endpoint.

Request body:

```json
{
  "model": "qwen_customvoice",
  "input": "Thanks for calling Aether.",
  "voice": "Ryan",
  "response_format": "wav",
  "language": "English",
  "instructions": "Calm and confident.",
  "metadata": {}
}
```

Response body:

```json
{
  "model": "qwen_customvoice",
  "format": "wav",
  "sample_rate": 24000,
  "audio_b64": "...",
  "timings": {
    "inference_ms": 1455,
    "total_ms": 1560
  },
  "artifacts": {
    "runtime_path_used": "qwen_customvoice",
    "qwen_model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "qwen_voice": "Ryan",
    "qwen_language": "English"
  }
}
```

### `POST /v1/audio/speech/stream`

Current status:

- returns `501 Not Implemented`

This endpoint is reserved for future streaming work so the provider contract does not have to change later.

## Integration Rule

`Aether-Voice-X` should rely on:

- `GET /health`
- `POST /v1/warmup`
- `POST /v1/audio/speech`

The main repo should not call internal runner modules directly.
