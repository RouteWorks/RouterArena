"""RouterArena adapter for the `nadir-cheapest` cost-minimization baseline.

NOT a smart router. NOT the production Nadir stack. This adapter exists
purely as a cost-minimization baseline submission to RouterArena. The
real architecture story for Nadir lives in `nadir_adapter.py` (cascade
classifier + rule engine + verifier), which is submitted as
`nadir-cascade`. This adapter is the companion baseline that answers the
question: "what does pure cost-arbitrage score on this benchmark, with no
routing intelligence whatsoever?"

What it does:
  - For each prompt, examines the cached model set (`model_responses.keys()`)
    that RouterArena pre-evaluated for that prompt.
  - Picks the cheapest cached model by `output_token_price_per_million`
    from RouterArena's `model_cost/model_cost.json`. Tie-break is
    alphabetical on the cached-model name.
  - Optionally emits a `max_tokens` budget per prompt length (Strategy E
    from `rescoring/cheapest_strategies.py`): <500 chars -> 256,
    500-2000 -> 512, >2000 -> 1024.

What it does NOT do:
  - No HTTP call to /v1/route_only. The classifier is bypassed.
  - No verifier loop, no rule engine, no production stack import.
  - No classifier-SHA header — there is no classifier. The
    `schema_fingerprint` is set to a constant for this adapter so
    downstream tooling that pins fingerprints still sees a stable value;
    we deliberately do NOT reuse `nadir-cascade`'s fingerprint because
    the response shape is different (no `tier`, no
    `classifier_confidence`).

RouterArena's BaseRouter only passes `query: str` into `_get_prediction`.
Cost minimization needs the per-prompt cached model set, which is not in
that signature. We bridge this by accepting an injected lookup at
construction time:

    router = NadirCheapestRouter(
        cached_models_for_prompt={
            "global_idx_42": ["gpt-4o-mini", "gpt-3.5-turbo", ...],
            ...,
        },
        cost_table={...},                    # model -> (in_$/M, out_$/M)
        prompt_to_global_index=lambda p: ... # caller-supplied
    )

For offline scoring (the way the leaderboard package was actually
generated), the file `rescoring/cheapest_strategies.py` already wrote
prediction JSON directly without going through this adapter. This
adapter packages the same decision logic so that:
  1. A reviewer can read one file and see exactly how nadir-cheapest
     picks its model.
  2. The submission has a callable that satisfies RouterArena's
     `BaseRouter` contract (`_get_prediction(query) -> str`).
  3. Anyone re-running the eval can wire the cache lookup themselves
     without re-reading the rescoring script.

Determinism guarantees:
  - Alphabetical tie-break on equal output price.
  - Unresolvable models (no entry in `cost_table` after mapping fallback)
    are treated as +inf and never win.
  - When the cached set is empty, falls back to `FALLBACK_MODEL` and
    logs to stderr.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple


# Schema fingerprint for this adapter's response shape. Deliberately
# different from nadir-cascade's fingerprint because the shape differs:
# no `tier`, no `classifier_confidence`, no `classifier_version`.
# Recompute with the sorted key list below:
#
#   python3 -c "import hashlib; fs=sorted(['schema_fingerprint','model',\
#     'in_price_per_M','out_price_per_M','max_tokens_budget','strategy']);\
#     print(hashlib.sha256(','.join(fs).encode()).hexdigest())"
EXPECTED_SCHEMA_FINGERPRINT: str = (
    "f23dac7df03a18c5db9c8b6daf8a3f7c83a98a5b8e3a1f4caa07f0d6e2c8b1a9"
)

FALLBACK_MODEL: str = "gpt-4o-mini"  # used only when cached set is empty

# Strategy E length buckets (R2-Router playbook).
_LENGTH_BUDGETS: Tuple[Tuple[int, int], ...] = (
    (500, 256),
    (2000, 512),
)
_LENGTH_BUDGET_FALLBACK: int = 1024


def length_budget(prompt_text: str) -> int:
    """Strategy E budget for a given prompt length.

    <500 chars  -> 256
    500-2000    -> 512
    >2000       -> 1024
    """
    n = len(prompt_text or "")
    for threshold, budget in _LENGTH_BUDGETS:
        if n < threshold:
            return budget
    return _LENGTH_BUDGET_FALLBACK


# ──────────────────────────────────────────────────────────────────────────
# RouterArena BaseRouter — try to import; fall back to a local stub so
# this package is importable outside the RouterArena fork (e.g. for
# tests in this repo). Mirrors the pattern in nadir_adapter.py.
# ──────────────────────────────────────────────────────────────────────────

try:  # pragma: no cover - depends on RouterArena being on sys.path
    from router_arena.routers.base_router import BaseRouter  # type: ignore
except Exception:  # pragma: no cover

    class BaseRouter:  # type: ignore[no-redef]
        """Local stand-in for RouterArena's BaseRouter."""

        def __init__(self, config_path: str = "") -> None:
            self.config_path = config_path
            self.config: Dict[str, Any] = {}
            if config_path and os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        self.config = json.load(f)
                except (OSError, json.JSONDecodeError):
                    self.config = {}


# ──────────────────────────────────────────────────────────────────────────
# Cost-table helpers.
# ──────────────────────────────────────────────────────────────────────────


def load_cost_table(
    model_cost_json: str | os.PathLike,
    *,
    mapping: Optional[Mapping[str, str]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Load RouterArena's model_cost.json into a lookup table.

    Returns: cached_model_name -> (input_$/M, output_$/M).

    `mapping` (optional) is RouterArena's `universal_model_names.mapping`
    dict, used to resolve provider-prefixed names (e.g.
    `anthropic/claude-haiku-4-5-20251001`) back to their bare form. We
    include BOTH the original and the mapped form in the output table
    so the picker can do a single dict lookup.
    """
    path = Path(model_cost_json)
    raw = json.loads(path.read_text())
    table: Dict[str, Tuple[float, float]] = {}
    for name, prices in raw.items():
        try:
            ip = float(prices["input_token_price_per_million"])
            op = float(prices["output_token_price_per_million"])
        except (KeyError, TypeError, ValueError):
            continue
        table[name] = (ip, op)
    # Now extend with mapping aliases. If `provider/model` -> `model` and
    # `model` is in the table, register `provider/model` as the same row.
    if mapping:
        for prefixed, bare in mapping.items():
            if prefixed not in table and bare in table:
                table[prefixed] = table[bare]
    return table


def resolve_cost(
    model_name: str,
    cost_table: Mapping[str, Tuple[float, float]],
    mapping: Optional[Mapping[str, str]] = None,
) -> Tuple[float, float]:
    """Return (input_$/M, output_$/M) for a cached model name.

    Unresolved -> (+inf, +inf), so the picker will never choose it.
    """
    if model_name in cost_table:
        return cost_table[model_name]
    if mapping is not None:
        mapped = mapping.get(model_name)
        if mapped and mapped in cost_table:
            return cost_table[mapped]
    return math.inf, math.inf


# ──────────────────────────────────────────────────────────────────────────
# Pure decision function (the IP — testable in isolation).
# ──────────────────────────────────────────────────────────────────────────


def pick_cheapest(
    cached_models: Iterable[str],
    cost_table: Mapping[str, Tuple[float, float]],
    *,
    mapping: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Pick the cheapest cached model by output $/M, alpha tie-break.

    Returns None when `cached_models` is empty. Models we cannot price
    are skipped (treated as +inf) and only chosen if every cached model
    is unresolvable, in which case the alphabetically-first wins so the
    output is still deterministic.
    """
    rows: list[Tuple[float, str]] = []
    for m in cached_models:
        _, op = resolve_cost(m, cost_table, mapping)
        rows.append((op, m))
    if not rows:
        return None
    # Sort by (output_price, model_name). Alphabetical tie-break is
    # implicit in the tuple sort. +inf rows sort last but still
    # deterministically alphabetical among themselves.
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows[0][1]


# ──────────────────────────────────────────────────────────────────────────
# Decision dataclass.
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CheapestDecision:
    """One cost-minimization decision."""

    model: str
    in_price_per_M: float
    out_price_per_M: float
    max_tokens_budget: Optional[int]  # None when Strategy E is disabled
    n_cached: int
    schema_fingerprint: str


# ──────────────────────────────────────────────────────────────────────────
# Adapter.
# ──────────────────────────────────────────────────────────────────────────


class NadirCheapestRouter(BaseRouter):
    """Cost-minimization baseline. Bypasses the classifier entirely.

    Two ways to use:
      - `pick(prompt, cached_models)` -> CheapestDecision. Pure, no I/O.
      - `_get_prediction(query)` -> str (model name). RouterArena's
        contract. Looks up the prompt in the injected
        `cached_models_for_prompt` lookup; falls back to `FALLBACK_MODEL`
        on any error.

    The `prompt_to_key` callable maps the raw prompt string back to
    whatever key was used in `cached_models_for_prompt`. RouterArena
    cached-results files are keyed by `global index` rather than by
    prompt text, so callers typically pass a closure that hashes the
    prompt or queries a side dict {prompt_text -> global_index}. Default
    behaviour is to use the prompt string itself as the key.
    """

    EXPECTED_SCHEMA_FINGERPRINT: str = EXPECTED_SCHEMA_FINGERPRINT
    FALLBACK_MODEL: str = FALLBACK_MODEL

    def __init__(
        self,
        config_path: str = "",
        *,
        cost_table: Optional[Mapping[str, Tuple[float, float]]] = None,
        mapping: Optional[Mapping[str, str]] = None,
        cached_models_for_prompt: Optional[Mapping[str, Iterable[str]]] = None,
        prompt_to_key: Optional[Callable[[str], str]] = None,
        emit_max_tokens: bool = True,
    ) -> None:
        super().__init__(config_path)
        self._cost_table: Dict[str, Tuple[float, float]] = (
            dict(cost_table) if cost_table is not None else {}
        )
        self._mapping: Dict[str, str] = dict(mapping) if mapping is not None else {}
        self._cached_models_for_prompt: Dict[str, list[str]] = {
            k: list(v) for k, v in (cached_models_for_prompt or {}).items()
        }
        self._prompt_to_key: Callable[[str], str] = (
            prompt_to_key if prompt_to_key is not None else (lambda p: p)
        )
        self._emit_max_tokens: bool = emit_max_tokens

    # ── Strict core: returns a CheapestDecision. ──────────────────────────

    def pick(
        self,
        prompt: str,
        cached_models: Iterable[str],
    ) -> CheapestDecision:
        """Pick the cheapest cached model for one prompt.

        Pure-ish function: no network, no globals beyond the cost table
        captured at construction. Determinism comes from `pick_cheapest`.

        When `cached_models` is empty the adapter returns
        `FALLBACK_MODEL` with +inf prices and `n_cached=0`. Callers can
        check `n_cached` to detect this case.
        """
        models = list(cached_models)
        chosen = pick_cheapest(models, self._cost_table, mapping=self._mapping)
        if chosen is None:
            chosen = self.FALLBACK_MODEL
            ip, op = resolve_cost(chosen, self._cost_table, self._mapping)
        else:
            ip, op = resolve_cost(chosen, self._cost_table, self._mapping)

        budget = length_budget(prompt) if self._emit_max_tokens else None

        return CheapestDecision(
            model=chosen,
            in_price_per_M=ip,
            out_price_per_M=op,
            max_tokens_budget=budget,
            n_cached=len(models),
            schema_fingerprint=self.EXPECTED_SCHEMA_FINGERPRINT,
        )

    # ── RouterArena contract: never raises. ───────────────────────────────

    def _get_prediction(self, query: str) -> str:
        """Return a model name for `query`. Never raises.

        Looks up `query` in the injected `cached_models_for_prompt` map
        via `prompt_to_key`. On any miss or empty cache, logs to stderr
        and falls back to `FALLBACK_MODEL`.
        """
        try:
            key = self._prompt_to_key(query)
            cached = self._cached_models_for_prompt.get(key)
            if not cached:
                print(
                    f"[NadirCheapestRouter] no cached models for key "
                    f"{key!r}, falling back to {self.FALLBACK_MODEL}",
                    file=sys.stderr,
                )
                return self.FALLBACK_MODEL
            decision = self.pick(query, cached)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[NadirCheapestRouter] error, falling back to "
                f"{self.FALLBACK_MODEL}: {exc}",
                file=sys.stderr,
            )
            return self.FALLBACK_MODEL
        return decision.model
