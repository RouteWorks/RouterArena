#!/usr/bin/env python3
"""Pre-generate LLM routing decisions for ChuzomLLMRouter.

Calls a cheap LLM classifier (gemini-3.1-flash-lite via OpenRouter) in
parallel for every query in the specified dataset split and writes a
hash → model_name JSON lookup file.

Usage:
    uv run python scripts/generate_llm_routing.py [--split full] [--workers 16]

The output file is:
    router_inference/config/chuzom-llm-routing-decisions.json

ISOLATION NOTE:
    This script is for RouterArena experimentation only. It does not touch
    the Chuzom production package. Results feed ChuzomLLMRouter which is
    itself an experiment-only router class.

COMPLIANCE:
    The routing prompt lists model capabilities from public knowledge only.
    No RouterArena label files, optimality entries, or accuracy scores are
    used. The LLM sees only the raw query text.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path

import httpx
from dotenv import load_dotenv

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_PATHS = {
    "sub_10": "./dataset/router_data_10.json",
    "full": "./dataset/router_data.json",
    "robustness": "./dataset/router_robustness.json",
}

OUTPUT_PATH = "./router_inference/config/chuzom-llm-routing-decisions.json"
REGISTRY_PATH = "./router_inference/model_registry.yaml"

ROUTING_MODELS = [
    "qwen/qwen3-235b-a22b-2507",
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-next-80b-a3b-instruct",
    "Qwen/Qwen3-Coder-Next",
    "gpt-4o-mini",
    "claude-3-haiku-20240307",
]

CLASSIFIER_MODEL = "google/gemini-3.1-flash-lite"  # cheap, fast routing classifier

USER_PROMPT_TEMPLATE = "Query:\n{query}\n\nModel:"


def _build_system_prompt_from_registry(registry_path: str) -> str:
    """Build the routing system prompt dynamically from model_registry.yaml.

    Falls back to a concise hardcoded prompt if the registry isn't available
    or PyYAML isn't installed.
    """
    if not _YAML_AVAILABLE:
        print("Warning: PyYAML not installed — using hardcoded system prompt")
        return _FALLBACK_SYSTEM_PROMPT

    rpath = Path(registry_path)
    if not rpath.exists():
        print(f"Warning: {registry_path} not found — using hardcoded system prompt")
        return _FALLBACK_SYSTEM_PROMPT

    with open(rpath, encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    models_section = registry.get("models", {})
    domain_prefs = registry.get("domain_preferences", {})

    lines = [
        "You are an expert LLM routing classifier. Select the single best model for",
        "the query, balancing task accuracy and cost. Base your decision ONLY on the",
        "semantic content of the query — never on index numbers or dataset names.\n",
        "Available models:\n",
    ]

    for model_id, info in models_section.items():
        display = info.get("display_name", model_id)
        tier = info.get("cost_tier", "?")
        strengths = info.get("strengths", [])
        best_for = info.get("best_for_query_types", [])

        lines.append(f"## {model_id}  ({display}, cost={tier})")
        if strengths:
            lines.append("Strengths: " + "; ".join(strengths[:4]))
        if best_for:
            lines.append("Best for: " + " | ".join(best_for[:3]))
        lines.append("")

    if domain_prefs:
        lines.append("Domain quick-reference (public benchmark data only):")
        for domain, prefs in domain_prefs.items():
            pref_str = ", ".join(f"{k}→{v.split('/')[-1]}" for k, v in prefs.items())
            lines.append(f"  {domain}: {pref_str}")
        lines.append("")

    lines.append("Return ONLY the exact model name (e.g. google/gemini-3.1-flash-lite). Nothing else.")
    return "\n".join(lines)


MEMORY_PATH = "./router_inference/routing_memory.yaml"


def _load_routing_memory() -> str:
    """Load routing_memory.yaml and format domain patterns as prompt text."""
    if not _YAML_AVAILABLE:
        return ""
    mpath = Path(MEMORY_PATH)
    if not mpath.exists():
        return ""

    with open(mpath, encoding="utf-8") as f:
        memory = yaml.safe_load(f)

    domains = memory.get("domains", {})
    if not domains:
        return ""

    lines = ["Content-pattern routing hints (derived from public benchmarks only):"]
    for domain_name, info in domains.items():
        signals = info.get("content_signals", [])[:5]  # top 5 signals only
        preferred = info.get("preferred_model", "").split("/")[-1]
        if signals and preferred:
            hint = f"  [{domain_name}] signals: {', '.join(repr(s) for s in signals)} → {preferred}"
            lines.append(hint)
    return "\n".join(lines)


_FALLBACK_SYSTEM_PROMPT = """\
You are an expert LLM routing classifier. Select the single best model for a query.

Available models and their primary strengths:
- google/gemini-3.1-flash-lite: reading comprehension, factual recall, geography, \
history, literature, general knowledge MCQ, ethics, music, arts, social sciences, NLI
- deepseek/deepseek-v4-flash: mathematics, STEM, physics, chemistry, engineering, \
science reasoning, translation
- qwen/qwen3-235b-a22b-2507: hard competitive math (AIME/Olympiad), biomedical/PubMed \
questions, complex formal logic, long academic texts
- Qwen/Qwen3-Coder-Next: code generation, programming algorithms, debugging
- gpt-4o-mini: narrative comprehension, word-in-context, pronoun resolution, mixed tasks
- claude-3-haiku-20240307: answer quality evaluation, summarization, instruction following
- qwen/qwen3-next-80b-a3b-instruct: moderate STEM and math, balanced tasks

Return ONLY the exact model name. Nothing else."""

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def classify_query(query: str, api_key: str, timeout: int = 30, system_prompt: str | None = None) -> str | None:
    """Call the routing LLM and return the model name it selects."""
    payload = {
        "model": CLASSIFIER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt or _FALLBACK_SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=query[:2000])},
        ],
        "max_tokens": 60,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = httpx.post(_OPENROUTER_URL, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]["message"]["content"].strip()
        # Extract just the model name (LLM might add punctuation)
        for model in ROUTING_MODELS:
            if model in choice:
                return model
        # If no exact match, try to fuzzy-match on the last segment
        choice_lower = choice.lower()
        for model in ROUTING_MODELS:
            suffix = model.split("/")[-1].lower()
            if suffix in choice_lower:
                return model
        return None
    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="full", choices=list(DATASET_PATHS.keys()))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true", help="Skip already-classified queries")
    parser.add_argument("--registry", default=REGISTRY_PATH, help="Path to model_registry.yaml")
    parser.add_argument("--use-memory", action="store_true",
                        help="Append routing_memory.yaml domain patterns to system prompt")
    args = parser.parse_args()

    system_prompt = _build_system_prompt_from_registry(args.registry)
    if args.use_memory:
        memory_section = _load_routing_memory()
        if memory_section:
            system_prompt += "\n\n" + memory_section
    print(f"System prompt built from registry: {len(system_prompt)} chars")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    # Load dataset
    dataset_path = DATASET_PATHS[args.split]
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    # Deduplicate queries by hash
    queries: dict[str, str] = {}  # hash → query text
    for entry in dataset:
        q = entry.get("prompt_formatted") or entry.get("prompt") or entry.get("query") or entry.get("question", "")
        if q:
            queries[_query_hash(q)] = q

    print(f"Unique queries to classify: {len(queries)}")

    # Load existing decisions (for resume)
    decisions: dict[str, str] = {}
    if args.resume and os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            decisions = json.load(f)
        print(f"Resuming: {len(decisions)} already classified, "
              f"{len(queries) - len(decisions)} remaining")

    pending = {h: q for h, q in queries.items() if h not in decisions}
    if not pending:
        print("All queries already classified.")
        return

    # Parallel classification
    lock = Lock()
    success = 0
    failed = 0
    fallback_model = "google/gemini-3.1-flash-lite"

    def worker(h_q):
        h, q = h_q
        result = classify_query(q, api_key, system_prompt=system_prompt)
        return h, result

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, item): item for item in pending.items()}
        for i, fut in enumerate(as_completed(futures), 1):
            h, model = fut.result()
            with lock:
                if model:
                    decisions[h] = model
                    success += 1
                else:
                    decisions[h] = fallback_model
                    failed += 1

            if i % 100 == 0 or i == len(pending):
                elapsed = time.time() - start
                rate = i / elapsed
                eta = (len(pending) - i) / rate if rate > 0 else 0
                print(
                    f"  {i}/{len(pending)} | ✅ {success} | ⚠️ fallback {failed} | "
                    f"{rate:.1f} q/s | ETA {eta:.0f}s"
                )
                # Save checkpoint every 100
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(decisions, f)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Classified: {success} | Fallback: {failed}")

    # Final save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2)
    print(f"Saved to: {OUTPUT_PATH}")

    # Distribution report
    from collections import Counter
    dist = Counter(decisions.values())
    print("\nRouting distribution:")
    for m, c in dist.most_common():
        print(f"  {m}: {c} ({100 * c / len(decisions):.1f}%)")


if __name__ == "__main__":
    main()
