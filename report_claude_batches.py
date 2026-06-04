#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Report status of the 3 Claude batch jobs: server-side processing counts + fetched lines."""

import json
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

BASE = "/scratch/yl231/henry-shan/opus_batch"
MODELS = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]
client = Anthropic()

all_done = True
print(f"{'model':20s} {'status':12s} {'proc':>5} {'ok':>5} {'err':>4} {'fetched':>8}")
for m in MODELS:
    wd = f"{BASE}/opus_sample_2000__{m}"
    meta = json.load(open(f"{wd}/batch_meta.json"))
    bid = meta.get("batch_id")
    outp = os.path.join(meta["cache_dir"], meta["out_name"] + ".jsonl")
    nfetched = sum(1 for _ in open(outp)) if os.path.exists(outp) else 0
    if not bid:
        print(f"{m:20s} {'NOT-SUBMITTED':12s}")
        all_done = False
        continue
    b = client.messages.batches.retrieve(bid)
    rc = b.request_counts
    print(
        f"{m:20s} {b.processing_status:12s} {rc.processing:>5} {rc.succeeded:>5} {rc.errored:>4} {nfetched:>8}"
    )
    if b.processing_status != "ended" or nfetched < meta["n_requests"]:
        all_done = False

print(f"\nALL_DONE={all_done}")
