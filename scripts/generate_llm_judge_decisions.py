# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""Pre-cache LLM judge decisions for ChuzomRouterV2 Gate 4.

RouterArena compliance rule:
  Routing decisions are based solely on prompt content. No dataset names,
  test-set indices, global_index values, or optimality metadata are used.

This script:
  1. Runs ChuzomRouterV2 on all prompts in the full dataset.
  2. For prompts where the blended confidence is LOW (below JUDGE_THRESHOLD),
     calls gemini-2.5-flash as a judge to resolve the routing decision.
  3. Saves sha256(prompt) → model_name to chuzom-llm-judge-decisions.json.

Gate 4 activates for prompts where Gates 1-3 are all low-confidence or
disagreeing — the sweet spot where a cheap LLM can meaningfully improve
routing accuracy over the blended score alone.

Usage:
    GOOGLE_API_KEY=your_key uv run python scripts/generate_llm_judge_decisions.py

    # Preview only (no API calls, shows how many prompts would need judging):
    uv run python scripts/generate_llm_judge_decisions.py --dry-run

    # Resume from partially-completed run:
    uv run python scripts/generate_llm_judge_decisions.py --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_PATH = Path("dataset/router_data.json")
OUTPUT_PATH = Path("router_inference/config/chuzom-llm-judge-decisions.json")
ROUTER_CONFIG = "chuzom-router-v2"

# Gate 4 fires when the blended margin is below this threshold
JUDGE_THRESHOLD = 0.12

# gemini-2.5-flash via Gemini API (cheap, fast, capable)
JUDGE_MODEL = "gemini-2.5-flash"
JUDGE_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{JUDGE_MODEL}:generateContent"
)

ROUTING_MODELS = [
    "google/gemini-2.0-flash-001",
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
]

# Rate limits: ~60 req/min for free Gemini tier
REQUEST_DELAY_S = 1.1
MAX_RETRIES = 3

_JUDGE_SYSTEM = """\
You are a routing classifier for a multi-model LLM benchmark. You receive raw
signal scores from three classifiers and a query excerpt. Select EXACTLY ONE
model that is best suited for this query. Base your decision ONLY on the signal
data and the query. Return the full model name, nothing else."""

_JUDGE_USER_TEMPLATE = """\
Gate signals (each gate reports top-3 model scores):

TF-IDF+LR  (lexical domain signals, margin={tfidf_margin:.3f}):
{tfidf_top3}

Centroid   (semantic task clusters, margin={centroid_margin:.3f}):
{centroid_top3}

Heuristic  (structural text patterns, margin={heuristic_margin:.3f}):
{heuristic_top3}

Query excerpt (first 500 chars):
{query_excerpt}

Available models (select exactly one full model name):
{models}"""


# ── Helpers ───────────────────────────────────────────────────────────────────


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def top3_str(scores: dict[str, float]) -> str:
    top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
    return "\n".join(f"  {m.split('/')[-1]}: {s:.4f}" for m, s in top3)


# ── Inline gate implementations (mirrors ChuzomRouterV2 without full class) ───

_MCQ_HEADER_RE = re.compile(
    r"Please read the following multiple-choice questions.*?(?=Context:)",
    re.DOTALL,
)
_HIGH_CONFIDENCE = 0.35

_HEURISTIC_RULES = [
    (
        re.compile(r"Context:\s*None", re.IGNORECASE),
        {"google/gemini-2.0-flash-001": 3.0, "google/gemini-3.1-flash-lite": 1.0},
    ),
    (
        re.compile(
            r"Context:\s*(?!None|N/A|\bNone\b).{20,}", re.IGNORECASE | re.DOTALL
        ),
        {"google/gemini-3.1-flash-lite": 4.0, "google/gemini-2.0-flash-001": 1.0},
    ),
    (
        re.compile(
            r"(?i)(translat|spanish|french|chinese|german|japanese|arabic|russian)"
        ),
        {"deepseek/deepseek-v4-flash": 3.0, "qwen/qwen3-235b-a22b-2507": 2.0},
    ),
    (
        re.compile(
            r"(?i)(calcul|integral|deriv|equation|mathemat|algebra|geometry"
            r"|trigonometr|probability|statistic|combinatoric|number theory)"
        ),
        {"deepseek/deepseek-v4-flash": 3.0, "qwen/qwen3-235b-a22b-2507": 2.0},
    ),
    (
        re.compile(
            r"(?i)(code|program|function|algorithm|debug|implement|python\b|java\b|sql\b)"
        ),
        {"deepseek/deepseek-v4-flash": 4.0, "qwen/qwen3-next-80b-a3b-instruct": 1.5},
    ),
    (
        re.compile(
            r"(?i)(word.?sense|coreference|disambigu|homograph|polysemy|pronoun.*refer)"
        ),
        {"qwen/qwen3-next-80b-a3b-instruct": 5.0, "qwen/qwen3-235b-a22b-2507": 2.0},
    ),
    (
        re.compile(r"(?i)(medical|clinical|diagnosis|pharmacol|biochem|anatomy)"),
        {"qwen/qwen3-235b-a22b-2507": 3.0, "deepseek/deepseek-v4-flash": 1.5},
    ),
    (
        re.compile(
            r"(?i)(olympiad|AIME|AMC|competition math|prove that|lemma|theorem)"
        ),
        {"qwen/qwen3-235b-a22b-2507": 4.0, "deepseek/deepseek-v4-flash": 2.0},
    ),
]

_GATE_WEIGHTS = {"tfidf": 1.0, "centroid": 1.3, "heuristic": 0.9}


def _extract_text(prompt: str) -> str:
    p = _MCQ_HEADER_RE.sub("", prompt)
    return " ".join(p.split())[:2000]


def _effective_weight(base: float, margin: float) -> float:
    return base * (1.0 + min(0.5, margin * 1.5))


def _gate_heuristic(prompt: str) -> tuple[dict[str, float], float]:
    raw: defaultdict[str, float] = defaultdict(float)
    for pattern, weights in _HEURISTIC_RULES:
        if pattern.search(prompt):
            for m, w in weights.items():
                raw[m] += w
    for m in ROUTING_MODELS:
        raw.setdefault(m, 0.0)
    total = sum(raw.values()) or 1.0
    norm = {m: raw[m] / total for m in ROUTING_MODELS}
    vals = sorted(norm.values(), reverse=True)
    margin = (vals[0] - vals[1]) / max(vals[0], 1e-6) if vals[0] > 0 else 0.0
    return norm, margin


def _blended_margin(
    tfidf: dict,
    tm: float,
    centroid: dict,
    cm: float,
    heuristic: dict,
    hm: float,
    models: list[str],
) -> tuple[str, float]:
    """Compute blended score and return (winner, margin)."""
    w_t = _effective_weight(_GATE_WEIGHTS["tfidf"], tm)
    w_c = _effective_weight(_GATE_WEIGHTS["centroid"], cm)
    w_h = _effective_weight(_GATE_WEIGHTS["heuristic"], hm)
    total = w_t + w_c + w_h

    blended: dict[str, float] = {}
    for m in models:
        blended[m] = (
            w_t * tfidf.get(m, 0.0)
            + w_c * centroid.get(m, 0.0)
            + w_h * heuristic.get(m, 0.0)
        ) / total

    vals = sorted(blended.values(), reverse=True)
    margin = vals[0] - vals[1] if len(vals) > 1 else 1.0
    winner = max(models, key=lambda m: blended[m])
    return winner, margin


# ── LLM judge call ────────────────────────────────────────────────────────────


def call_judge(
    prompt: str,
    tfidf: dict,
    tm: float,
    centroid: dict,
    cm: float,
    heuristic: dict,
    hm: float,
    api_key: str,
) -> str | None:
    import httpx

    user_msg = _JUDGE_USER_TEMPLATE.format(
        tfidf_margin=tm,
        tfidf_top3=top3_str(tfidf),
        centroid_margin=cm,
        centroid_top3=top3_str(centroid),
        heuristic_margin=hm,
        heuristic_top3=top3_str(heuristic),
        query_excerpt=prompt[:500].replace("\n", " "),
        models="\n".join(f"  {m}" for m in ROUTING_MODELS),
    )

    payload = {
        "system_instruction": {"parts": [{"text": _JUDGE_SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
        "generationConfig": {"maxOutputTokens": 64, "temperature": 0.0},
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = httpx.post(
                f"{JUDGE_API_URL}?key={api_key}",
                json=payload,
                timeout=15,
            )
            resp.raise_for_status()
            cands = resp.json().get("candidates", [])
            if not cands:
                return None
            raw = cands[0]["content"]["parts"][0]["text"].strip()
            for m in ROUTING_MODELS:
                if m in raw or m.split("/")[-1].lower() in raw.lower():
                    return m
            return None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2**attempt)
            else:
                print(
                    f"  Judge call failed after {MAX_RETRIES} retries: {e}",
                    file=sys.stderr,
                )
                return None
    return None


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many prompts need judging, no API calls",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load existing cache and only process missing entries",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Cap number of prompts to process (for testing)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not args.dry_run and not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    # Load dataset
    print(f"Loading dataset from {DATASET_PATH}...", file=sys.stderr)
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    routing_entries = [e for e in dataset if not e.get("for_optimality")]
    if args.max_prompts:
        routing_entries = routing_entries[: args.max_prompts]
    print(f"  {len(routing_entries)} routing prompts", file=sys.stderr)

    # Load existing cache if resuming
    existing: dict[str, str] = {}
    if args.resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            existing = json.load(f)
        print(f"  Loaded {len(existing)} existing judge decisions", file=sys.stderr)

    # Load ChuzomRouterV2 gates
    print("Loading ChuzomRouterV2 gates...", file=sys.stderr)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from router_inference.router.chuzom_router_v2 import ChuzomRouterV2

    router = ChuzomRouterV2(ROUTER_CONFIG, llm_judge_enabled=False)

    # Process each prompt
    decisions: dict[str, str] = dict(existing)
    low_confidence_count = 0
    judged_count = 0
    fallback_count = 0
    model_counter: Counter = Counter()

    print("Scanning prompts for low-confidence routing...", file=sys.stderr)
    for i, entry in enumerate(routing_entries):
        prompt = entry["prompt"]
        key = prompt_hash(prompt)

        if key in decisions:
            model_counter[decisions[key]] += 1
            continue

        text = _extract_text(prompt)

        # Compute all gate scores
        tfidf_scores, tfidf_margin = router._gate_tfidf(text)
        centroid_scores, centroid_margin = router._gate_centroid(text)
        heuristic_scores, heuristic_margin = router._gate_heuristic(prompt)

        # Check early-exit condition
        tfidf_winner = max(tfidf_scores, key=lambda m: tfidf_scores.get(m, 0.0))
        centroid_winner = max(
            centroid_scores, key=lambda m: centroid_scores.get(m, 0.0)
        )

        if (
            tfidf_winner == centroid_winner
            and tfidf_margin > _HIGH_CONFIDENCE
            and centroid_margin > _HIGH_CONFIDENCE
        ):
            # High-confidence consensus — judge not needed
            decisions[key] = tfidf_winner
            model_counter[tfidf_winner] += 1
            if i % 200 == 0:
                print(
                    f"  {i}/{len(routing_entries)} — high-conf, skip judge",
                    file=sys.stderr,
                )
            continue

        # Compute blended score
        winner, blended_margin = _blended_margin(
            tfidf_scores,
            tfidf_margin,
            centroid_scores,
            centroid_margin,
            heuristic_scores,
            heuristic_margin,
            router.models,
        )

        if blended_margin >= JUDGE_THRESHOLD:
            # Blended score is confident enough — no judge needed
            decisions[key] = winner
            model_counter[winner] += 1
            continue

        # Low confidence → needs judge
        low_confidence_count += 1

        if args.dry_run:
            continue

        # Call judge
        judge_result = call_judge(
            prompt,
            tfidf_scores,
            tfidf_margin,
            centroid_scores,
            centroid_margin,
            heuristic_scores,
            heuristic_margin,
            api_key,
        )

        if judge_result:
            decisions[key] = judge_result
            model_counter[judge_result] += 1
            judged_count += 1
        else:
            decisions[key] = winner  # fallback to blended winner
            model_counter[winner] += 1
            fallback_count += 1

        time.sleep(REQUEST_DELAY_S)

        if i % 50 == 0:
            print(
                f"  {i}/{len(routing_entries)} — {judged_count} judged, "
                f"{fallback_count} fallback, {low_confidence_count} low-conf total",
                file=sys.stderr,
            )

        # Save checkpoint every 200 judge calls
        if (judged_count + fallback_count) % 200 == 0 and (
            judged_count + fallback_count
        ) > 0:
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_PATH, "w") as f:
                json.dump(decisions, f, indent=2)
            print(f"  Checkpoint saved ({len(decisions)} entries)", file=sys.stderr)

    # Final report
    print(f"\n{'DRY RUN ' if args.dry_run else ''}Summary:", file=sys.stderr)
    print(f"  Total prompts:      {len(routing_entries)}", file=sys.stderr)
    print(
        f"  Low-confidence:     {low_confidence_count} ({low_confidence_count / max(len(routing_entries), 1) * 100:.1f}%)",
        file=sys.stderr,
    )
    if not args.dry_run:
        print(f"  LLM judged:         {judged_count}", file=sys.stderr)
        print(f"  Fallback (blended): {fallback_count}", file=sys.stderr)
        print(f"  Total decisions:    {len(decisions)}", file=sys.stderr)
        print("\n  Model distribution:", file=sys.stderr)
        for m in ROUTING_MODELS:
            n = model_counter[m]
            pct = n / max(len(routing_entries), 1) * 100
            print(f"    {m}: {n} ({pct:.1f}%)", file=sys.stderr)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(decisions, f, indent=2)
        print(f"\nSaved to {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
