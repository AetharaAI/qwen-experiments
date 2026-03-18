# Native CustomVoice Prompt Pack

## Intent

This command runs a reusable asset pack across a selected set of built-in Qwen voices.

The default voice set is:

- `Ryan`
- `Aiden`
- `Uncle_Fu`
- `Serena`
- `Vivian`
- `Sohee`

The default prompt pack is grouped as reusable assets:

- `brand`
- `site`
- `telephony`
- `ads`

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

- `structured/` grouped by voice and category
- `review/` with flat filenames for VLC or Audacity
- `review/review.m3u` playlist file
- `manifest.json` summarizing the run

Logs are written separately into:

- `logs/`
