#!/usr/bin/env bash
set -e

echo "==> Fitting budget curves..."
uv run python -m training.train_curves

echo "==> Training MLP heads..."
uv run python -m training.train_heads

echo "==> Calibrating temperatures..."
uv run python -m training.calibrate

echo "==> Offline evaluation..."
uv run python -m training.evaluate

echo "Done. Checkpoints in ./checkpoints/hybrid-router/"
