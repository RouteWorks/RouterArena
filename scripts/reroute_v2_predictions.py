# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""Re-route chuzom-router-v2 predictions using the v2.1.0 3-gate router.

v2.1.0 dropped the TF-IDF+LR Gate 1 (trained on RouterArena data).
This script applies the new routing to all 8400 entries and re-runs
inference only for entries whose assigned model changed.

Usage:
    uv run python scripts/reroute_v2_predictions.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import httpx
from dotenv import load_dotenv

load_dotenv()

PRED_PATH = "./router_inference/predictions/chuzom-router-v2.json"
ROB_PATH = "./router_inference/predictions/chuzom-router-v2-robustness.json"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_OUTPUT_TOKENS = 512
WORKERS = 20

MODEL_TO_CACHE_FILE = {
    "google/gemini-3.1-flash-lite": "google_gemini-3.1-flash-lite.jsonl",
    "deepseek/deepseek-v4-flash": "deepseek_deepseek-v4-flash.jsonl",
    "qwen/qwen3-235b-a22b-2507": "qwen_qwen3-235b-a22b-2507.jsonl",
    "qwen/qwen3-next-80b-a3b-instruct": "qwen_qwen3-next-80b-a3b-instruct.jsonl",
    "google/gemini-2.0-flash-001": "google_gemini-2.0-flash-001.jsonl",
}


def call_model(model: str, prompt: str, api_key: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = httpx.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return {
            "generated_answer": content,
            "success": bool(content),
            "token_usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            "provider": "openrouter",
            "error": None,
        }
    except Exception as e:
        return {
            "generated_answer": None,
            "success": False,
            "token_usage": {},
            "provider": "openrouter",
            "error": str(e)[:200],
        }


def reroute_file(
    pred_path: str,
    router,
    api_key: str | None,
    dry_run: bool,
    label: str,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Processing: {pred_path}  [{label}]")
    with open(pred_path, encoding="utf-8") as f:
        data = json.load(f)

    routing_entries = [e for e in data if not e.get("for_optimality")]
    changed = []
    for e in routing_entries:
        old_model = e["prediction"]
        new_model = router._get_prediction(e["prompt"])
        if new_model != old_model:
            changed.append((e, old_model, new_model))

    print(f"  Total routing entries : {len(routing_entries)}")
    print(f"  Routing changed       : {len(changed)}")
    print(f"  Routing unchanged     : {len(routing_entries) - len(changed)}")

    if changed:
        print("\n  Change breakdown (old → new):")
        breakdown = Counter(f"{o} → {n}" for _, o, n in changed)
        for pair, count in breakdown.most_common(10):
            print(f"    {count:4d}x  {pair}")

    if dry_run:
        print("\n  [dry-run] No changes written.")
        return

    if not api_key:
        print("\nERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    # Apply new routing
    for e, _old, new_model in changed:
        e["prediction"] = new_model
        e["generated_result"] = None  # clear stale answer

    # Run inference for changed entries
    to_infer = [(e, e["prediction"]) for e, _, _ in changed]
    if not to_infer:
        print("  Nothing to infer.")
    else:
        print(
            f"\n  Running inference for {len(to_infer)} changed entries ({WORKERS} workers)..."
        )
        lock = Lock()
        cache_updates: dict[str, list] = {}
        success = fail = 0
        start = time.time()

        def worker(item):
            e, model = item
            return e, model, call_model(model, e["prompt"], api_key)

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(worker, item): item for item in to_infer}
            for i, fut in enumerate(as_completed(futures), 1):
                e, model, result = fut.result()
                with lock:
                    e["generated_result"] = result
                    if result["success"]:
                        cache_updates.setdefault(model, []).append(
                            {
                                "global_index": e["global index"],
                                "question": e["prompt"],
                                "llm_selected": model,
                                **result,
                            }
                        )
                        success += 1
                    else:
                        fail += 1
                if i % 50 == 0 or i == len(to_infer):
                    elapsed = time.time() - start
                    rate = i / elapsed
                    print(
                        f"    {i}/{len(to_infer)} | ✅{success} ❌{fail} | {rate:.1f}/s"
                    )

        print(f"  Done | Success: {success} | Failed: {fail}")

        # Append to cache files
        for model, entries in cache_updates.items():
            cache_file = MODEL_TO_CACHE_FILE.get(model)
            if not cache_file:
                continue
            cache_path = os.path.join("./cached_results", cache_file)
            with open(cache_path, "a", encoding="utf-8") as f:
                for entry in entries:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write("\n")
            print(f"    Appended {len(entries)} entries to {cache_file}")

    # Save updated predictions
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  Saved: {pred_path}")

    # Final null check
    null_count = sum(
        1
        for e in data
        if not e.get("for_optimality")
        and not (e.get("generated_result") or {}).get("generated_answer")
    )
    print(f"  Remaining nulls: {null_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")

    # Import the updated router (3-gate, no TF-IDF)
    sys.path.insert(0, ".")
    from router_inference.router.chuzom_router_v2 import ChuzomRouterV2

    print("Loading ChuzomRouterV2 (v2.1.0 — 3 gates, no TF-IDF)...")
    router = ChuzomRouterV2("chuzom-router-v2", llm_judge_enabled=False)
    print("Router loaded.")

    reroute_file(PRED_PATH, router, api_key, args.dry_run, "main")
    reroute_file(ROB_PATH, router, api_key, args.dry_run, "robustness")


if __name__ == "__main__":
    main()
