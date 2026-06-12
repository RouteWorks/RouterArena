#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0
#
# Re-score routers affected by the issue #135 cost-accounting fix, using the real
# evaluation pipeline (llm_evaluation/run.py) with the patched cost logic. Saves
# each router's metrics.json to /tmp/metrics_<router>.json for leaderboard update.
set -euo pipefail
cd "$(dirname "$0")/.."

ROUTERS=("$@")
if [ ${#ROUTERS[@]} -eq 0 ]; then
  ROUTERS=(nadir-cascade-v2 azure-model-router sqwish-router vllm-sr)
fi

for r in "${ROUTERS[@]}"; do
  echo "=================================================================="
  echo ">>> Re-scoring ${r} ..."
  echo "=================================================================="
  PYTHONPATH=. .venv/bin/python llm_evaluation/run.py "${r}" full \
    --force --save-interval 0 --num-workers 16 --log-level WARNING
  cp metrics.json "/tmp/metrics_${r}.json"
  echo ">>> ${r} metrics:"
  cat "/tmp/metrics_${r}.json"
  echo
done
echo ">>> ALL DONE"
