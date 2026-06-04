# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0
"""Retry the failed (success=false) records in the train_cached_results_300k_plus
per-model JSONLs via direct (synchronous) Anthropic calls, and merge in place.
For a handful of transient failures this is faster than re-batching."""

import json
import os
import glob
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

PLUS_DIR = "/scratch/yl231/henry-shan/prepare_data/train_cached_results_300k_plus"
MAX_TOKENS = 2048
client = Anthropic()


def retry_one(model_id, question):
    resp = client.messages.create(
        model=model_id,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": question}],
    )
    text = "".join(
        getattr(b, "text", "")
        for b in resp.content
        if getattr(b, "type", None) == "text"
    )
    u = resp.usage
    return {
        "generated_answer": text,
        "token_usage": {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "total_tokens": u.input_tokens + u.output_tokens,
        },
        "success": True,
        "error": None,
    }


for path in sorted(glob.glob(os.path.join(PLUS_DIR, "anthropic_*.jsonl"))):
    base = os.path.basename(path)[len("anthropic_") : -len(".jsonl")]  # -> model id
    recs = [json.loads(line) for line in open(path)]
    failed_idx = [i for i, r in enumerate(recs) if not r.get("success")]
    if not failed_idx:
        print(f"{base}: 0 failed, skip")
        continue
    print(
        f"{base}: retrying {len(failed_idx)} -> {[recs[i]['global_index'] for i in failed_idx]}"
    )
    n_fixed = 0
    for i in failed_idx:
        r = recs[i]
        try:
            out = retry_one(base, r["question"])
            r.update(out)
            n_fixed += 1
            print(
                f"  ✓ {r['global_index']}  out_tok={out['token_usage']['output_tokens']}"
            )
        except Exception as e:
            r["error"] = str(e)
            print(f"  ✗ {r['global_index']}  still failing: {str(e)[:100]}")
    # atomic rewrite
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in recs:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    os.replace(tmp, path)
    print(f"  rewrote {path}: fixed {n_fixed}/{len(failed_idx)}")
