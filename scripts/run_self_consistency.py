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
    SYSTEM_PROMPT_VERSION,
    extract_mc_letter,
    is_multiple_choice,
    majority_vote,
    system_prompt_for,
)


SUB10_DATASET_PATH = Path("dataset/router_data_10.json")
BASELINE_PATH = Path("router_inference/predictions/llm-router.json.bak.honest")

# OpenRouter rate-limit tuning. Conservative defaults to avoid 429s
# during the full run; aggressive enough for the smoke test to finish in
# minutes.
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_RETRY_BACKOFF_SECONDS = (1.0, 3.0, 10.0)

# Substrings in an OpenRouter error response that indicate a hard cap —
# retries are pointless. The runner exits immediately when these appear
# so we don't burn 30 minutes hammering a dead key (as we did the first
# time around).
HARD_FAIL_ERROR_MARKERS = (
    "Key limit exceeded",
    "credit limit",
    "insufficient_quota",
    "402",
)

# Incremental checkpoint cadence. Output is rewritten every N MC entries
# so a mid-run abort leaves a usable partial submission rather than only
# the cache file.
OUTPUT_CHECKPOINT_EVERY = 100


class QuotaExhausted(SystemExit):
    """Raised when OpenRouter signals the key is over its limit."""


def _is_hard_quota_error(err: Exception) -> bool:
    message = str(err)
    return any(marker in message for marker in HARD_FAIL_ERROR_MARKERS)


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
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    system_prompt: str | None = None,
) -> str:
    """Single OpenRouter completion with bounded retries on transient failures.

    If ``system_prompt`` is provided it is sent as the first message in the
    chat. Otherwise we send just the user turn (matching the legacy
    behaviour, important so callers that omit the parameter get identical
    samples to the v0 runs).
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    last_err: Exception | None = None
    for backoff in DEFAULT_RETRY_BACKOFF_SECONDS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            choice = resp.choices[0] if resp.choices else None
            if choice and choice.message and choice.message.content:
                return choice.message.content
            return ""
        except Exception as err:  # OpenAI/OpenRouter exceptions are heterogeneous
            # Hard-cap errors (key limit exceeded, credit exhausted) are not
            # transient — retries cannot recover. Surface immediately so the
            # outer loop can abort, preserving the partial cache and any
            # checkpointed output.
            if _is_hard_quota_error(err):
                raise QuotaExhausted(
                    f"OpenRouter quota exhausted: {err}. Top up the key or "
                    f"rotate OPENROUTER_API_KEY and resume — the cache will "
                    f"skip already-completed entries."
                ) from err
            last_err = err
            time.sleep(backoff)
    # All retries failed — return empty so extract_mc_letter yields None
    # and majority_vote can still produce a winner from the surviving
    # samples. Failure is logged for the runner to surface.
    print(f"  ! completion failed after retries: {last_err}", file=sys.stderr)
    return ""


def _format_vote_as_generated_result(
    letter: str, prompt: str, samples: list[str]
) -> dict:
    """Match the canonical ``generated_result`` shape so the evaluator parses it.

    The evaluator's pre-check (``router_inference/check_config_prediction_files.py``)
    requires a dict with ``generated_answer: str``, ``success: bool``, and
    ``token_usage``. Token usage is estimated from prompt + observed sample
    texts (1 token ≈ 4 chars) so the multi-sample cost of self-consistency
    is honestly attributed — under-counting would artificially deflate cost
    and inflate Arena Score.
    """
    answer = f"The correct answer is \\boxed{{{letter}}}."
    input_tokens = len(samples) * max(1, len(prompt) // 4)
    output_tokens = sum(max(1, len(s) // 4) for s in samples) if samples else 0
    return {
        "generated_answer": answer,
        "success": True,
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


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
    parser.add_argument(
        "--no-system-prompts",
        dest="use_system_prompts",
        action="store_false",
        help=(
            "Disable Tier 1B task-family system prompts (debug only). "
            "Default is to send a tailored system prompt for each MC family."
        ),
    )
    parser.set_defaults(use_system_prompts=True)
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

    # Pre-build the output list in baseline order so every incremental
    # checkpoint is a complete, valid submission rather than a partial
    # prefix. MC rows are marked for processing; everything else is
    # already-final passthrough.
    mc_total = mc_processed = passthrough_count = 0
    output: list[dict] = []
    pending_indices: list[int] = []  # positions in `output` we still need to process

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
        # MC entry: append the baseline row as a starting point, record
        # its position for later mutation.
        output.append(entry)
        pending_indices.append(len(output) - 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    def _checkpoint() -> None:
        """Write the current `output` to disk so a kill leaves a usable file."""
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    # Initial checkpoint = baseline-shape file (so even a 0-sample abort
    # produces a valid submission).
    _checkpoint()

    for output_idx in pending_indices:
        if args.limit is not None and mc_processed >= args.limit:
            break

        entry = output[output_idx]
        gi = entry.get("global index", "")
        prompt = entry.get("prompt", "") or ""
        model = entry.get("prediction") or ""
        if not model:
            continue

        system_prompt = system_prompt_for(prompt) if args.use_system_prompts else None
        # Cache key fingerprints the system prompt version + whether one was
        # used at all. Bumping SYSTEM_PROMPT_VERSION invalidates cached
        # samples drawn from older prompts; the "SP0" tag preserves samples
        # from --no-system-prompts runs separately from v1.
        sp_tag = f"SP{SYSTEM_PROMPT_VERSION}" if system_prompt else "SP0"
        cache_key = f"{gi}::{model}::T{args.temperature}::N{args.n_samples}::{sp_tag}"
        samples = cache.get(cache_key)
        if samples is None:
            if args.dry_run:
                samples = []
            else:
                assert client is not None
                samples = [
                    _complete_once(
                        client, model, prompt, args.temperature, system_prompt
                    )
                    for _ in range(args.n_samples)
                ]
                cache[cache_key] = samples
                # Persist after every entry so a mid-run abort doesn't waste $$
                _save_cache(cache, args.cache)

        letters = [extract_mc_letter(s) for s in samples]
        vote = majority_vote(letters)

        if vote is not None:
            new_generated = _format_vote_as_generated_result(vote, prompt, samples)
            new_entry = dict(entry)
            new_entry["generated_result"] = new_generated
            output[output_idx] = new_entry
        # else: leave the baseline entry untouched — no consensus means no
        # downgrade.

        mc_processed += 1
        if mc_processed % 25 == 0:
            print(
                f"  processed {mc_processed} MC entries (cache size {len(cache)})",
                file=sys.stderr,
            )
        if mc_processed % OUTPUT_CHECKPOINT_EVERY == 0:
            _checkpoint()

    _checkpoint()

    print()
    print(f"split:           {args.split}")
    print(f"baseline rows:   {len(baseline)}")
    print(f"output rows:     {len(output)}")
    print(f"MC total seen:   {mc_total}")
    print(f"MC processed:    {mc_processed}")
    print(f"passthrough:     {passthrough_count}")
    print(
        f"system prompts:  {'on (' + SYSTEM_PROMPT_VERSION + ')' if args.use_system_prompts else 'off'}"
    )
    print(f"output path:     {args.output}")
    if args.cache:
        print(f"cache path:      {args.cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
