# Native CustomVoice Prompt Pack

## Intent

This command runs a fixed telephony-style prompt pack across a selected set of built-in Qwen voices.

The default voice set is:

- `Ryan`
- `Aiden`
- `Uncle_Fu`
- `Serena`
- `Vivian`
- `Sohee`

The default prompt pack is:

- `Thanks for calling Aether.`
- `Thanks for calling Aether Pro.`
- `Welcome to Aether Voice. How can I help you today?`
- `I can help with scheduling, support, billing, and general questions.`
- `Please hold for just a moment while I pull up your account.`

## Command

```bash
uv run --extra native python -m qwen_experiments.cli native-customvoice-pack
```

## Optional Voice Override

```bash
uv run --extra native python -m qwen_experiments.cli native-customvoice-pack \
  --voices "Ryan,Aiden,Serena,Vivian,Sohee,Ono_Anna"
```

## Outputs

The command writes a timestamped directory under:

- `results/native_customvoice/`

Inside that directory:

- one folder per voice
- one WAV per prompt
- `manifest.json` summarizing the run

Logs are written separately into:

- `logs/`
