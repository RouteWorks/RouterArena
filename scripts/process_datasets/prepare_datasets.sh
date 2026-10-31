#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0
#
# Run prep_datasets.py with a time limit, retrying a few times. The Hugging Face
# downloads it makes occasionally stall without failing, which used to hang CI jobs
# until their own timeout; a stalled attempt is now stopped and retried.
#
# Usage: scripts/process_datasets/prepare_datasets.sh
#   PREP_TIMEOUT  seconds per attempt (default 900)
#   PREP_ATTEMPTS number of attempts (default 3)
set -euo pipefail

timeout_s="${PREP_TIMEOUT:-900}"
attempts="${PREP_ATTEMPTS:-3}"
for attempt in $(seq 1 "$attempts"); do
  if timeout --kill-after=30 "$timeout_s" uv run python scripts/process_datasets/prep_datasets.py; then
    exit 0
  fi
  echo "Dataset preparation attempt $attempt of $attempts failed or timed out." >&2
  if [[ "$attempt" -lt "$attempts" ]]; then
    sleep 15
  fi
done
echo "Dataset preparation failed after $attempts attempts." >&2
exit 1
