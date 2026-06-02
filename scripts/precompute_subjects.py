"""Phase C: pre-classify MC prompts via local Ollama (qwen3.5).

Runs once per dataset split. Produces a `subject_cache.json` mapping
prompt-hash → subject label. The adapter's `_get_prediction` reads this
cache at routing time so the per-prompt path stays synchronous.

Why batch:
  RouterArena's `_get_prediction` is sync and called serially per prompt
  during `generate_prediction_file.py`. Adding a 1-second Ollama call
  inline would make sub_10 generation take ~13 min instead of <1s. Caching
  the classifications offline keeps the hot path instant.

Why only MC prompts:
  Other prompt families (LiveCodeBench, NarrativeQA, WMT19, etc.) are
  already correctly routed by prefix detection. Only the generic-MC
  bucket needs subject disambiguation — currently 500+ prompts default
  to qwen3-235b that may benefit from gemini/deepseek/Coder.

Cache format (subject_cache.json):
  {
    "<sha1 of prompt>": "medical" | "computer_science" | "history" | ...,
    ...
  }
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import aiohttp

# Explicit .env load — uv run doesn't auto-source .env, and
# OPENROUTER_API_KEY is needed for the openrouter backend.
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CACHE_PATH = PROJECT_ROOT / "router_inference" / "subject_cache.json"

# Subject categories — designed to match the per-dataset best-model mapping.
# Adding "other" as a sink for unmatched cases.
SUBJECTS = [
    "medical",  # MedMCQA — gemini wins
    "computer_science",  # MMLUPro_CS — gemini wins
    "history",  # MMLUPro_history — gemini wins
    "biology",  # MMLUPro_biology — gemini wins
    "geography",  # GeoBench — gemini wins
    "physics",  # MMLUPro_physics — deepseek wins
    "public_health",  # MMLUPro_health — deepseek wins
    "engineering",  # MMLUPro_engineering — qwen ties gemini (cheap)
    "mathematics",  # MMLUPro_math/MathQA — qwen ties (cheap)
    "law",  # MMLUPro_law — qwen ties (cheap)
    "psychology",  # MMLUPro_psychology — qwen ties (cheap)
    "economics",  # MMLUPro_economics — qwen ties (cheap)
    "business",  # MMLUPro_business — Coder wins narrowly
    "chemistry",  # MMLUPro_chemistry — qwen ties (cheap)
    "philosophy",  # MMLUPro_philosophy — qwen ties (cheap)
    "music_theory",  # MusicTheoryBench — qwen ties (cheap)
    "ethics",  # Ethics_* — qwen ties (cheap)
    "literature",  # QANTA_Literature — qwen ties (cheap)
    "trivia",  # OpenTDB_* — varies; cheap workhorse default
    "other",
]

CLASSIFIER_PROMPT = (
    "Classify the topic of the question below into exactly ONE category.\n"
    "Categories: medical, computer_science, history, biology, geography, "
    "physics, public_health, engineering, mathematics, law, psychology, "
    "economics, business, chemistry, philosophy, music_theory, ethics, "
    "literature, trivia, other.\n\n"
    "Reply with ONLY the category name. No explanation.\n\n"
    "Question:\n{prompt}\n\nCategory:"
)


def prompt_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def is_mc_default_prompt(text: str) -> bool:
    """Detect prompts that would currently fall through to qwen3-235b default."""
    stripped = text.lstrip()
    # Same prefix as _PFX_MC_GENERIC in the adapter
    return stripped.startswith(
        "Please read the following multiple-choice questions and provide "
        "the most likely correct answer"
    )


async def classify_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    prompt: str,
    model: str,
    timeout: int,
    backend: str = "ollama",
) -> Optional[str]:
    """Send one prompt to the configured backend. Returns category string or None."""
    classifier_input = CLASSIFIER_PROMPT.format(prompt=prompt[:3000])  # cap context

    if backend == "openrouter":
        import os
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": classifier_input}],
            "temperature": 0,
            "max_tokens": 30,
        }
        async with semaphore:
            try:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    raw = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                        .lower()
                    )
            except Exception:
                return None
    else:  # ollama
        payload = {
            "model": model,
            "prompt": classifier_input,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 10},
        }
        async with semaphore:
            try:
                async with session.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    raw = data.get("response", "").strip().lower()
            except Exception:
                return None

    # Parse response into category
    # Strip any extra words — keep first alphanumeric-underscore token
    for word in raw.replace(",", " ").split():
        word = word.strip("'\"`*.()-:")
        if word in SUBJECTS:
            return word
    # Fallback: try substring matching
    for s in SUBJECTS:
        if s in raw:
            return s
    return "other"


async def run(
    dataset_path: Path,
    cache_path: Path,
    model: str,
    concurrency: int,
    timeout: int,
    backend: str = "ollama",
) -> None:
    # Load dataset
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Load existing cache
    cache: dict[str, str] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
    print(f"Loaded {len(cache)} cached classifications", file=sys.stderr)

    # Find prompts to classify (MC defaults, not already cached)
    todo: list[tuple[str, str]] = []  # (hash, prompt_text)
    for entry in dataset:
        prompt = entry.get("prompt_formatted") or entry.get("prompt", "")
        if not prompt or not is_mc_default_prompt(prompt):
            continue
        h = prompt_hash(prompt)
        if h not in cache:
            todo.append((h, prompt))

    print(
        f"Classifying {len(todo)} MC prompts via {backend} "
        f"({model}, concurrency={concurrency})",
        file=sys.stderr,
    )
    if backend == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "ERROR: OPENROUTER_API_KEY not set. Add it to RouterArena/.env or export it.",
            file=sys.stderr,
        )
        return

    if not todo:
        print("Cache fully covers MC prompts; nothing to do.", file=sys.stderr)
        return

    # Classify in parallel — gather() preserves input order so results[i]
    # corresponds to todo[i]. Progress is tracked via a counter on each
    # task completion.
    semaphore = asyncio.Semaphore(concurrency)
    completed = [0]
    start = time.time()
    n = len(todo)

    async def _wrapped(session, prompt):
        r = await classify_one(session, semaphore, prompt, model, timeout, backend)
        completed[0] += 1
        if completed[0] % 50 == 0 or completed[0] == n:
            elapsed = time.time() - start
            rate = completed[0] / elapsed if elapsed > 0 else 0
            eta = (n - completed[0]) / rate if rate > 0 else float("inf")
            print(
                f"  {completed[0]}/{n} classified  "
                f"(rate {rate:.1f}/s  eta {eta:.0f}s)",
                file=sys.stderr,
            )
        return r

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *(_wrapped(session, prompt) for _, prompt in todo)
        )

    # Save results
    new_cnt = 0
    for (h, _), label in zip(todo, results):
        if label is not None:
            cache[h] = label
            new_cnt += 1

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(
        f"\nWrote {new_cnt} new classifications to {cache_path} "
        f"(total {len(cache)})",
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "split",
        choices=["sub_10", "full", "robustness"],
        help="Dataset split to classify",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "openrouter"],
        default="ollama",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5:latest",
        help="Model to use (Ollama tag or OpenRouter slug)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Parallel requests",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Per-request timeout (seconds)",
    )
    args = parser.parse_args()

    paths = {
        "sub_10": PROJECT_ROOT / "dataset" / "router_data_10.json",
        "full": PROJECT_ROOT / "dataset" / "router_data.json",
        "robustness": PROJECT_ROOT / "dataset" / "router_robustness.json",
    }
    dataset_path = paths[args.split]

    asyncio.run(
        run(
            dataset_path=dataset_path,
            cache_path=CACHE_PATH,
            model=args.model,
            concurrency=args.concurrency,
            timeout=args.timeout,
            backend=args.backend,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
