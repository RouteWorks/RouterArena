"""Patch chuzom-router-v2.json: reroute unavailable/empty entries and rerun inference.

Problems found:
  - google/gemini-2.0-flash-001: 404 on OpenRouter (model retired)
  - deepseek/deepseek-v4-flash: 266 empty answers (content filter / token issue)
  - gemini-3.1-flash-lite: 2 transient errors

Fix: reroute those entries to gemini-3.1-flash-lite, then rerun inference.
"""

import json
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import httpx
from dotenv import load_dotenv

load_dotenv()

PRED_PATH = "./router_inference/predictions/chuzom-router-v2.json"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
FALLBACK_MODEL = "google/gemini-3.1-flash-lite"
CACHE_FILE = "./cached_results/google_gemini-3.1-flash-lite.jsonl"
MAX_OUTPUT_TOKENS = 512
WORKERS = 20


def call_model(model, prompt, api_key):
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


def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    print(f"Loading {PRED_PATH}...")
    with open(PRED_PATH) as f:
        data = json.load(f)

    # Find entries needing repair
    to_fix = []
    for e in data:
        if e.get("for_optimality"):
            continue
        gr = e.get("generated_result")
        needs_fix = (
            gr is None or not gr.get("generated_answer")  # null or empty string
        )
        if needs_fix:
            to_fix.append(e)

    print(f"Entries to fix: {len(to_fix)}")
    from collections import Counter

    print("  By model:", Counter(e["prediction"] for e in to_fix).most_common())

    # Reroute to fallback model
    for e in to_fix:
        e["prediction"] = FALLBACK_MODEL

    print(
        f"\nAll rerouted to {FALLBACK_MODEL}. Running inference ({WORKERS} workers)..."
    )
    lock = Lock()
    new_cache = []
    success = fail = 0

    def worker(e):
        return e, call_model(FALLBACK_MODEL, e["prompt"], api_key)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(worker, e): e for e in to_fix}
        for i, fut in enumerate(as_completed(futures), 1):
            e, result = fut.result()
            with lock:
                e["generated_result"] = result
                if result["success"]:
                    new_cache.append(
                        {
                            "global_index": e["global index"],
                            "question": e["prompt"],
                            "llm_selected": FALLBACK_MODEL,
                            **result,
                        }
                    )
                    success += 1
                else:
                    fail += 1
            if i % 50 == 0 or i == len(to_fix):
                print(f"  {i}/{len(to_fix)} | ✅{success} ❌{fail}")

    print(f"\nDone | Success: {success} | Failed: {fail}")

    # Save updated predictions
    with open(PRED_PATH, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Updated {PRED_PATH}")

    # Append to cache
    with open(CACHE_FILE, "a") as f:
        for entry in new_cache:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    print(f"Appended {len(new_cache)} entries to {CACHE_FILE}")

    # Final check
    remaining = sum(
        1
        for e in data
        if not e.get("for_optimality")
        and not (e.get("generated_result") or {}).get("generated_answer")
    )
    print(f"Remaining null routing entries: {remaining}")


if __name__ == "__main__":
    main()
