#!/usr/bin/env python3
"""Run inference for entries missing from the cache in a prediction file.

Calls OpenRouter for each entry where generated_result is null or has no answer,
then updates both the prediction file and the relevant cached_results/ JSONL.

Usage:
    uv run python scripts/run_missing_inference.py chuzom-llm-router [--dry-run]
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import httpx
from dotenv import load_dotenv

load_dotenv()

PREDICTIONS_DIR = "./router_inference/predictions"
CACHED_RESULTS_DIR = "./cached_results"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL_TO_CACHE_FILE = {
    "google/gemini-3.1-flash-lite": "google_gemini-3.1-flash-lite.jsonl",
    "deepseek/deepseek-v4-flash": "deepseek_deepseek-v4-flash.jsonl",
    "qwen/qwen3-235b-a22b-2507": "qwen_qwen3-235b-a22b-2507.jsonl",
    "Qwen/Qwen3-Coder-Next": "Qwen_Qwen3-Coder-Next.jsonl",
    "gpt-4o-mini": "gpt-4o-mini.jsonl",
    "claude-3-haiku-20240307": "claude-3-haiku-20240307.jsonl",
    "qwen/qwen3-next-80b-a3b-instruct": "qwen_qwen3-next-80b-a3b-instruct.jsonl",
}

MAX_OUTPUT_TOKENS = 512


def call_model(model: str, prompt: str, api_key: str, timeout: int = 60) -> dict:
    """Call model via OpenRouter. Returns result dict."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = httpx.post(
            OPENROUTER_URL, json=payload, headers=headers, timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage", {})
        return {
            "generated_answer": content,
            "success": True,
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
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "provider": "openrouter",
            "error": str(e)[:200],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("router_name")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show counts but don't call APIs"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--optimality", action="store_true", help="Also fill optimality entries"
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: OPENROUTER_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    pred_path = os.path.join(PREDICTIONS_DIR, f"{args.router_name}.json")
    print(f"Loading: {pred_path}")
    with open(pred_path, encoding="utf-8") as f:
        predictions = json.load(f)

    # Find missing entries
    missing = [
        e
        for e in predictions
        if (not e.get("for_optimality") or args.optimality)
        and (
            not e.get("generated_result")
            or not e["generated_result"].get("generated_answer")
        )
    ]

    by_model = defaultdict(list)
    for e in missing:
        by_model[e["prediction"]].append(e)

    print(f"\nMissing entries requiring inference: {len(missing)}")
    for model, entries in sorted(by_model.items(), key=lambda x: -len(x[1])):
        print(f"  {model}: {len(entries)}")

    if args.dry_run or len(missing) == 0:
        print("\n[dry-run] Done.")
        return

    assert api_key is not None  # guaranteed by the earlier check
    print(f"\nRunning inference with {args.workers} workers...")
    cache_updates: dict[str, list] = defaultdict(list)  # model → new cache entries
    lock = Lock()
    success_count = 0
    fail_count = 0

    def worker(entry):
        model = entry["prediction"]
        prompt = entry["prompt"]
        gidx = entry["global index"]
        result = call_model(model, prompt, api_key)
        return entry, result, gidx, model

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, e): e for e in missing}
        for i, fut in enumerate(as_completed(futures), 1):
            entry, result, gidx, model = fut.result()
            with lock:
                entry["generated_result"] = result
                cache_updates[model].append(
                    {
                        "global_index": gidx,
                        "question": entry["prompt"],
                        "llm_selected": model,
                        **result,
                    }
                )
                if result["success"]:
                    success_count += 1
                else:
                    fail_count += 1

            if i % 50 == 0 or i == len(missing):
                elapsed = time.time() - start
                rate = i / elapsed
                eta = (len(missing) - i) / rate if rate > 0 else 0
                print(
                    f"  {i}/{len(missing)} | ✅{success_count} ❌{fail_count} | {rate:.1f}/s | ETA {eta:.0f}s"
                )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s | Success: {success_count} | Failed: {fail_count}")

    # Update prediction file
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)
    print(f"Updated predictions: {pred_path}")

    # Append new entries to cache files
    for model, entries in cache_updates.items():
        successful = [e for e in entries if e.get("success")]
        if not successful:
            continue
        cache_file = MODEL_TO_CACHE_FILE.get(model)
        if not cache_file:
            continue
        cache_path = os.path.join(CACHED_RESULTS_DIR, cache_file)
        with open(cache_path, "a", encoding="utf-8") as f:
            for e in successful:
                json.dump(e, f, ensure_ascii=False)
                f.write("\n")
        print(f"  Appended {len(successful)} new entries to {cache_file}")

    # Final null count
    null_count = sum(
        1
        for e in predictions
        if not e.get("generated_result")
        or not e.get("generated_result", {}).get("generated_answer")
    )
    print(f"\nFinal null count: {null_count}")


if __name__ == "__main__":
    main()
