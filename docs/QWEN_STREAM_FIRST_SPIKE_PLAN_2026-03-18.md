# Qwen Stream-First Spike Plan - 2026-03-18

## Objective

Run an isolated spike outside of `Aether-Voice-X` to prove the best way to serve Qwen TTS in real runtime conditions.

This spike exists to prevent another MOSS-style integration loop where runtime truth is discovered too late.

## Models In Scope

Initial models:

- `Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `Qwen3-TTS-12Hz-1.7B-Base`

Deferred model:

- `Qwen3-TTS-12Hz-1.7B-VoiceDesign`

Reason:

- `CustomVoice` has the highest immediate product value.
- `Base` covers enterprise custom voice and cloning workflows.
- `VoiceDesign` is useful, but it is not required to start making money.

## Runner Matrix

Each model must be tested on both runner paths before any platform integration:

1. Python-native `qwen-tts`
2. `vLLM-Omni`

Test order:

1. `1.7B-CustomVoice` on Python-native runner
2. `1.7B-CustomVoice` on `vLLM-Omni`
3. `1.7B-Base` on Python-native runner
4. `1.7B-Base` on `vLLM-Omni`

## Operating Principles

- streaming is the default expectation
- batch is only a fallback path
- logs must exist for every meaningful boundary
- no runner is trusted because docs say it works
- any runner that requires fragile hacks is a weaker candidate even if it produces good audio once

## Required Logging

For every runner, log all of the following:

- process start
- process ready
- model load start
- model load complete
- request accepted
- request rejected
- websocket connect start
- websocket connected
- first input accepted
- first audio chunk emitted
- final audio chunk emitted
- request complete
- client disconnect
- upstream disconnect
- timeout
- exception with stack trace

Also record:

- wall-clock timestamps
- request ID
- session ID
- model ID
- runner type
- host GPU ID
- output mode: stream or batch

## Metrics To Capture

Every test should capture:

- boot success or failure
- time to ready
- idle VRAM
- generation VRAM peak
- first-audio latency
- chunk cadence
- total generation time
- output duration
- repeat-run consistency
- subjective voice quality notes
- any transport glitches

## Pass Criteria

A runner is viable only if it:

1. boots reliably
2. streams without transport confusion
3. emits measurable first-audio events
4. produces repeatable output quality
5. fails clearly when something goes wrong
6. does not threaten the frozen `Voxtral + Kokoro` baseline

## Directory Intent

- `configs/`
  runner configs and example envs
- `logs/`
  raw boot logs and request traces
- `results/`
  benchmark summaries and generated sample outputs
- `scripts/`
  test harness scripts for spike runs
- `notes/`
  short operator notes during testing

## Decision Gate

No Qwen runtime code should be merged into `Aether-Voice-X` until this spike answers:

1. Which runner is more stable?
2. Which runner streams more truthfully?
3. Which runner gives the better latency-to-quality tradeoff?
4. Which runner is simpler to operate and debug?

## Expected Next Deliverable

The next deliverable from this directory should be:

- one runner comparison summary
- one recommended winner for `1.7B-CustomVoice`
- one clear follow-up plan for integrating that winner into `Aether-Voice-X`
