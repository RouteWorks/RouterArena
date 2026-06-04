#!/bin/bash
# Run the real 2000-query batch inference for the 3 Claude models.
# Submits all 3 batches up front (parallel server-side), then waits + fetches each.
set -uo pipefail
cd /scratch/yl231/henry-shan/RouterArena

MODELS="claude-opus-4-8 claude-sonnet-4-6 claude-haiku-4-5"
BASE=/scratch/yl231/henry-shan/opus_batch
LOG=$BASE/run_claude_batches_$(date +%m%d_%H%M%S).log
echo "log: $LOG"

run() { uv run python llm_inference/anthropic_batch_inference.py "$@"; }

{
  echo "==== SUBMIT all 3 batches ($(date)) ===="
  for M in $MODELS; do
    run create --workdir "$BASE/opus_sample_2000__$M"
  done

  echo "==== WAIT + FETCH each ($(date)) ===="
  for M in $MODELS; do
    run status --workdir "$BASE/opus_sample_2000__$M" --wait
    run fetch  --workdir "$BASE/opus_sample_2000__$M"
  done

  echo "==== ALL DONE ($(date)) ===="
} 2>&1 | tee -a "$LOG"
