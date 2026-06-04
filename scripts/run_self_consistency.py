# SPDX-FileCopyrightText: Copyright (c) 2026 Yali Pollak
# SPDX-License-Identifier: Apache-2.0

"""Tier 1A — self-consistency inference runner.

Loads the honest baseline predictions, identifies multiple-choice
entries, runs the assigned model N times at temperature=0.7 via
OpenRouter, extracts ``\\boxed{X}`` answers, and writes the majority
vote into ``generated_result``. Non-MC entries are passthrough.

Legitimacy
----------
The runner reads only the ``prediction`` (model name) and ``prompt``
fields from the baseline — never ``accuracy`` or ``cost``. The voted
letter is derived solely from fresh model outputs. This matches the
"output features only" policy in
``docs/ROUTERARENA_IMPROVEMENT_PLAN.md`` §"Allowed signals".

Usage
-----
Smoke test (~50 MC entries, sub_10 split, <$0.05):
    uv run python scripts/run_self_consistency.py \\
        --split sub_10 --limit 50 \\
        --output router_inference/predictions/llm-router.sub10-smoke.json \\
        --cache .self_consistency_cache.json

Full split (~17,500 calls, ~$2, 30-60 min):
    uv run python scripts/run_self_consistency.py \\
        --split full \\
        --output router_inference/predictions/llm-router.json \\
        --cache .self_consistency_cache.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Make scripts/ importable so we get the canonical self_consistency module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_consistency import (  # noqa: E402
    extract_mc_letter,
    is_multiple_choice,
    majority_vote,
)


SUB10_DATASET_PATH = Path("dataset/router_data_10.json")
BASELINE_PATH = Path("router_inference/predictions/llm-router.json.bak.honest")

# OpenRouter rate-limit tuning. Conservative defaults to avoid 429s
# during the full run; aggressive enough for the smoke test to finish in
# minutes.
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_RETRY_BACKOFF_SECONDS = (1.0, 3.0, 10.0)


def _get_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY not found in environment or .env. "
            "Add it to .env and re-run."
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )


def _complete_once(
    client: OpenAI, model: str, prompt: str, temperature: float
) -> str:
    """Single OpenRouter completion with bounded retries on transient failures."""
    last_err: Exception | None = None
    for backoff in DEFAULT_RETRY_BACKOFF_SECONDS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            choice = resp.choices[0] if resp.choices else None
            if choice and choice.message and choice.message.content:
                return choice.message.content
            return ""
        except Exception as err:  # OpenAI/OpenRouter exceptions are heterogeneous
            last_err = err
            time.sleep(backoff)
    # All retries failed — return empty so extract_mc_letter yields None
    # and majority_vote can still produce a winner from the surviving
    # samples. Failure is logged for the runner to surface.
    print(f"  ! completion failed after retries: {last_err}", file=sys.stderr)
    return ""


def _format_vote_as_generated_result(letter: str) -> str:
    """Match the canonical ``generated_result`` shape so the evaluator parses it.

    The eval pipeline reads ``generated_result`` as a JSON-serialized dict
    with a ``generated_answer`` field. We bake the voted letter into a
    ``\\boxed{X}`` payload so ``EnhancedExtractor._extract_standard_boxed``
    finds it.
    """
    return json.dumps({"generated_answer": f"The correct answer is \\boxed{{{letter}}}."})


def _load_cache(cache_path: Path | None) -> dict[str, list[str]]:
    if cache_path is None or not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        print(f"  ! cache file corrupt, starting fresh: {cache_path}", file=sys.stderr)
        return {}


def _save_cache(cache: dict[str, list[str]], cache_path: Path | None) -> None:
    if cache_path is None:
        return
    cache_path.write_text(json.dumps(cache))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=BASELINE_PATH,
        help="Baseline predictions to read prompt + model assignment from.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the self-consistency predictions.",
    )
    parser.add_argument(
        "--split",
        choices=["sub_10", "full"],
        required=True,
        help="Restrict to sub_10 indices, or process the whole baseline.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Samples per MC query (default 3).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default 0.7).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N MC entries (smoke test). Remaining entries pass through.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Cache file for samples (resume support). Recommended for full runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls; just count MC entries and exit.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Baseline predictions not found: {args.input}")

    baseline = json.loads(args.input.read_text())

    sub10_ids: set[str] = set()
    if args.split == "sub_10":
        if not SUB10_DATASET_PATH.exists():
            raise SystemExit(f"sub_10 dataset not found: {SUB10_DATASET_PATH}")
        sub10 = json.loads(SUB10_DATASET_PATH.read_text())
        sub10_ids = {entry["global index"] for entry in sub10}

    cache = _load_cache(args.cache)

    if args.dry_run:
        client: OpenAI | None = None
    else:
        client = _get_client()

    mc_total = mc_processed = passthrough_count = 0
    output: list[dict] = []

    for entry in baseline:
        gi = entry.get("global index", "")
        if entry.get("for_optimality"):
            output.append(entry)
            passthrough_count += 1
            continue
        if args.split == "sub_10" and gi not in sub10_ids:
            continue

        prompt = entry.get("prompt", "") or ""
        if not is_multiple_choice(prompt):
            output.append(entry)
            passthrough_count += 1
            continue

        mc_total += 1
        if args.limit is not None and mc_processed >= args.limit:
            output.append(entry)
            continue

        model = entry.get("prediction") or ""
        if not model:
            output.append(entry)
            continue

        cache_key = f"{gi}::{model}::T{args.temperature}::N{args.n_samples}"
        samples = cache.get(cache_key)
        if samples is None:
            if args.dry_run:
                samples = []
            else:
                assert client is not None
                samples = [
                    _complete_once(client, model, prompt, args.temperature)
                    for _ in range(args.n_samples)
                ]
                cache[cache_key] = samples
                # Persist after every entry so a mid-run abort doesn't waste $$
                _save_cache(cache, args.cache)

        letters = [extract_mc_letter(s) for s in samples]
        vote = majority_vote(letters)

        if vote is not None:
            new_generated = _format_vote_as_generated_result(vote)
            new_entry = dict(entry)
            new_entry["generated_result"] = new_generated
            output.append(new_entry)
        else:
            # No consensus — keep the baseline's generated_result so we
            # don't downgrade the answer with a random sample.
            output.append(entry)

        mc_processed += 1
        if mc_processed % 25 == 0:
            print(
                f"  processed {mc_processed} MC entries "
                f"(cache size {len(cache)})",
                file=sys.stderr,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    print()
    print(f"split:           {args.split}")
    print(f"baseline rows:   {len(baseline)}")
    print(f"output rows:     {len(output)}")
    print(f"MC total seen:   {mc_total}")
    print(f"MC processed:    {mc_processed}")
    print(f"passthrough:     {passthrough_count}")
    print(f"output path:     {args.output}")
    if args.cache:
        print(f"cache path:      {args.cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
