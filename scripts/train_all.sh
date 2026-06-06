#!/usr/bin/env bash
set -e

echo "==> Downloading datasets..."
uv run python training/scripts/download_routerbench.py
uv run python training/scripts/download_r2bench.py

echo "==> Training MLP heads..."
uv run python training/train_heads.py --all --epochs 30

echo "==> Fitting budget curves..."
uv run python training/train_curves.py

echo "==> Calibrating temperatures..."
uv run python training/calibrate.py

echo "==> Offline evaluation (val)..."
uv run python training/evaluate.py --split val

echno "Done. Checkpoints in ./checkpoints/"