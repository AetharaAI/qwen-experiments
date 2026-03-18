#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run --extra native python -m qwen_experiments.cli native-customvoice-pack "$@"
