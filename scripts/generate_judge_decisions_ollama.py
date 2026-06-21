# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""Pre-generate LLM judge decisions for all prompts below the margin threshold.

Uses local Ollama inference (qwen3.6:27b by default) so there's zero API cost.
Writes results to router_inference/config/chuzom-llm-judge-decisions.json so
CI can run without live API calls.

Usage:
    # Check Ollama is running:
    ollama list

    # Run with default model (qwen3.6:27b):
    uv run python scripts/generate_judge_decisions_ollama.py

    # Use a different local model:
    CHUZOM_JUDGE_OLLAMA_MODEL=gemma3:27b uv run python scripts/generate_judge_decisions_ollama.py

    # Dry-run (show how many prompts need judging):
    uv run python scripts/generate_judge_decisions_ollama.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

sys.path.insert(0, ".")


_JUDGE_SYSTEM = """\
You are an expert model router. Given a prompt, output ONLY the model name that would \
best handle it. Choose from:
- google/gemini-2.0-flash-001  (fast, general knowledge, reading comprehension)
- google/gemini-3.1-flash-lite  (lightweight, fast, factual QA, translation)
- deepseek/deepseek-v4-flash  (code, math, step-by-step reasoning)
- qwen/qwen3-235b-a22b-2507  (competition math, hard reasoning, STEM)
- qwen/qwen3-next-80b-a3b-instruct  (NLI, entailment, word sense, coreference)

Output exactly one model name, nothing else."""

_ROUTING_MODELS = [
    "google/gemini-2.0-flash-001",
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
]

JUDGE_THRESHOLD = 0.25  # must match chuzom_router_v2._JUDGE_THRESHOLD
OLLAMA_URL = "http://localhost:11434/api/chat"
WORKERS = 4  # Ollama is local; don't hammer it with too many concurrent calls
TIMEOUT = 60


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _ollama_judge(prompt: str, model: str) -> str | None:
    user_msg = (
        f"{_JUDGE_SYSTEM}\n\n"
        f"Prompt to route:\n<prompt>\n{prompt[:2000]}\n</prompt>\n\n"
        "Model:"
    )
    try:
        resp = httpx.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": user_msg}],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 64},
            },
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        raw = resp.json()["message"]["content"].strip()
        for m in _ROUTING_MODELS:
            if m in raw or m.split("/")[-1].lower() in raw.lower():
                return m
    except Exception:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Re-judge already-cached entries"
    )
    args = parser.parse_args()

    ollama_model = os.environ.get("CHUZOM_JUDGE_OLLAMA_MODEL", "qwen3.6:27b")
    print(f"Ollama model: {ollama_model}")

    # Check Ollama is reachable
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        tags = [m["name"] for m in r.json().get("models", [])]
        if not any(ollama_model.split(":")[0] in t for t in tags):
            print(f"WARNING: {ollama_model} not found in Ollama. Available: {tags}")
            print(f"Pull it with: ollama pull {ollama_model}")
    except Exception as e:
        print(f"ERROR: Ollama not reachable at {OLLAMA_URL}: {e}")
        sys.exit(1)

    # Load existing cache
    cache_path = Path("router_inference/config/chuzom-llm-judge-decisions.json")
    existing: dict[str, str] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            existing = json.load(f)
    print(f"Existing cached decisions: {len(existing)}")

    # Load router to find prompts that would trigger the judge
    from router_inference.router.chuzom_router_v2 import ChuzomRouterV2

    print("Loading ChuzomRouterV2 to identify judge-eligible prompts...")
    router = ChuzomRouterV2("chuzom-router-v2")

    # Load all routing prompts
    pred_path = Path("router_inference/predictions/chuzom-router-v2.json")
    with open(pred_path) as f:
        data = json.load(f)
    routing_entries = [e for e in data if not e.get("for_optimality")]

    rob_path = Path("router_inference/predictions/chuzom-router-v2-robustness.json")
    with open(rob_path) as f:
        rob_data = json.load(f)
    rob_entries = [e for e in rob_data if not e.get("for_optimality")]

    all_prompts = list({e["prompt"] for e in routing_entries + rob_entries})
    print(f"Unique routing prompts: {len(all_prompts)}")

    # Find prompts below the blended_margin threshold
    judge_candidates: list[str] = []
    for prompt in all_prompts:
        h = _sha256(prompt)
        if not args.force and h in existing:
            continue
        # Simulate the router to check if judge would fire
        blended_margin = router._compute_blended_margin(prompt)
        if blended_margin < JUDGE_THRESHOLD:
            judge_candidates.append(prompt)

    print(
        f"Prompts needing judge (margin < {JUDGE_THRESHOLD}): {len(judge_candidates)}"
    )

    if args.dry_run:
        print("\n[dry-run] Sample of prompts that need judging:")
        for p in judge_candidates[:5]:
            print(f"  margin={router._compute_blended_margin(p):.3f}  {p[:80]!r}")
        print(f"\n[dry-run] Would generate {len(judge_candidates)} judge decisions.")
        return

    if not judge_candidates:
        print("Nothing to do.")
        return

    # Generate decisions in parallel
    new_decisions: dict[str, str] = {}
    success = 0
    failed = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(_ollama_judge, p, ollama_model): p for p in judge_candidates
        }
        for i, fut in enumerate(as_completed(futures), 1):
            prompt = futures[fut]
            result = fut.result()
            h = _sha256(prompt)
            if result:
                new_decisions[h] = result
                success += 1
            else:
                failed += 1
            if i % 50 == 0 or i == len(judge_candidates):
                elapsed = time.time() - start
                rate = i / elapsed
                print(
                    f"  {i}/{len(judge_candidates)} | ✅{success} ❌{failed} | {rate:.1f}/s"
                )

    # Merge and write
    merged = {**existing, **new_decisions}
    with open(cache_path, "w") as f:
        json.dump(merged, f, ensure_ascii=False)

    print(f"\nDone: {success} new decisions, {failed} failed")
    print(f"Total cache size: {len(merged)} entries → {cache_path}")


if __name__ == "__main__":
    main()
