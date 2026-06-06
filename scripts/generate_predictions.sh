#!/usr/bin/env bash
set -e

ROUTER="hybrid-router"

echo "==> Generating full split predictions..."
uv run python ./router_inference/generate_prediction_file.py $ROUTER full

echo "==> Validating full predictions..."
uv run python ./router_inference/check_config_prediction_files.py $ROUTER full

echo "==> Generating robustness split predictions..."
uv run python ./router_inference/generate_prediction_file.py $ROUTER robustness

echo "==> Validating robustness predictions..."
uv run python ./router_inference/check_config_prediction_files.py $ROUTER robustness

echo "Prediction files ready in ./router_inference/predictions/"