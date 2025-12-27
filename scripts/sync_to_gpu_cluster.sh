#!/usr/bin/env bash

set -euo pipefail

REMOTE="atom7@192.168.24.9:/data/lsw/code/leggedrobotics/rsl_rl"

# Always run from the repo root so rsync includes the correct files.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but not installed" >&2
  exit 1
fi

rsync -av --delete \
  --exclude 'data/' \
  --exclude '.venv/' \
  --exclude 'artifacts/' \
  --exclude 'logs/' \
  --exclude 'wandb/' \
  --exclude '**/__pycache__/' \
  --exclude '*.mp4' \
  "$ROOT_DIR/" "$REMOTE"
