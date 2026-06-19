# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""Run inference for the 3236 for_optimality entries in chuzom-router-v2.json.

These entries have routing decisions but no generated_result.
RouterArena uses them to compute Opt.Sel / Opt.Cost / Opt.Acc.

Usage:
    uv run python scripts/run_optimality_inference.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import httpx
from dotenv import load_dotenv

load_dotenv()

PRED_PATH = "router_inference/predictions/chuzom-router-v2.json"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
WORKERS = 20
TIMEOUT = 45

_lock = Lock()


def _call_openrouter(model: str, prompt: str, api_key: str) -> tuple[str | None, int, int]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/ypollak2/RouterArena",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    try:
        resp = httpx.post(OPENROUTER_URL, headers=headers, json=payload, timeout=TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            ans = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            in_tok = usage.get("prompt_tokens", 512)
            out_tok = usage.get("completion_tokens", max(1, len(ans) // 4))
            return ans, in_tok, out_tok
        if resp.status_code == 404 and model != "google/gemini-3.1-flash-lite":
            # Model unavailable on OpenRouter — retry with gemini-lite
            return _call_openrouter("google/gemini-3.1-flash-lite", prompt, api_key)
    except Exception as e:
        print(f"  Error [{model}]: {e}")
    return None, 512, 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.dry_run:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    pred_path = Path(PRED_PATH)
    with open(pred_path) as f:
        data = json.load(f)

    opt_entries = [(i, e) for i, e in enumerate(data) if e.get("for_optimality")]
    pending = [
        (i, e)
        for i, e in opt_entries
        if not (
            e.get("generated_result")
            and isinstance(e["generated_result"].get("generated_answer"), str)
        )
    ]

    print(f"Total for_optimality entries : {len(opt_entries)}")
    print(f"Already answered             : {len(opt_entries) - len(pending)}")
    print(f"Pending inference            : {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    model_dist = Counter(e.get("prediction", "unknown") for _, e in pending)
    print("\nModel distribution (pending):")
    for m, c in model_dist.most_common():
        print(f"  {m}: {c}")

    if args.dry_run:
        print(f"\n[dry-run] Would run inference for {len(pending)} entries.")
        return

    success = 0
    failed = 0
    start = time.time()

    def process(item):
        idx, e = item
        model = e.get("prediction", "")
        prompt = e.get("prompt", "")
        if not model or not prompt:
            return idx, None, 512, 1
        ans, in_tok, out_tok = _call_openrouter(model, prompt, api_key)
        return idx, ans, in_tok, out_tok

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process, item): item for item in pending}
        for i, fut in enumerate(as_completed(futures), 1):
            idx, ans, in_tok, out_tok = fut.result()
            with _lock:
                if ans:
                    data[idx]["generated_result"] = {
                        "generated_answer": ans,
                        "success": True,
                        "token_usage": {
                            "input_tokens": in_tok,
                            "output_tokens": out_tok,
                            "total_tokens": in_tok + out_tok,
                        },
                        "provider": "openrouter",
                        "error": None,
                    }
                    data[idx]["output_tokens"] = out_tok
                    data[idx]["input_tokens"] = in_tok
                    success += 1
                else:
                    failed += 1
            if i % 100 == 0 or i == len(pending):
                elapsed = time.time() - start
                rate = i / elapsed
                print(f"  {i}/{len(pending)} | ✅{success} ❌{failed} | {rate:.1f}/s")

    with open(pred_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {pred_path}")
    print(f"Done: {success} answered, {failed} failed")


if __name__ == "__main__":
    main()
