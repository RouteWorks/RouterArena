# SPDX-FileCopyrightText: Copyright contributors to the cruq.ai project
# SPDX-License-Identifier: Apache-2.0

"""
cruq router (Phase 1: training-free heuristic).

The router sees only the raw prompt string and must return one model from its
configured pool. This first version estimates a per-query difficulty from cheap
lexical signals (length, math/code markers, reasoning cues, whether the question
is multiple-choice) and maps that difficulty onto a pool sorted cheapest ->
strongest by price. Easy queries go to the cheapest model; harder queries escalate.

Design constraints honored here:
  - Prompt-only: no metadata, no ground-truth, no RouterArena labels are read.
  - Training-free: thresholds are priors set in the config, not fit on the
    benchmark. This keeps the entry compliant with the "no training/fitting/tuning
    on RouterArena data" rule while we develop the Phase 2 learned predictor.
  - Deterministic and local: no network call at decision time, so latency is
    sub-millisecond and robustness comes from stable, coarse features.

Pool order is derived from model_cost/model_cost.json (blended input+output price)
so the router works for any pool without code changes. Tunable knobs live in the
config's pipeline_params:
  - "difficulty_thresholds": ascending cut points in [0,1]; a pool of N models
    uses N-1 thresholds. Defaults to evenly spaced if absent.
  - "price_weight_output": weight on the output-token price when ranking models
    by cost (default 0.5).
"""

import json
import os
import re
from typing import Dict, List, Tuple

from router_inference.router.base_router import BaseRouter


def _load_model_costs() -> Dict[str, float]:
    """Blended $/1M price per model from model_cost/model_cost.json."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    path = os.path.join(root, "model_cost", "model_cost.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    costs: Dict[str, float] = {}
    for name, entry in raw.items():
        inp = entry.get("input_token_price_per_million")
        out = entry.get("output_token_price_per_million")
        if inp is None and out is None:
            continue
        inp = float(inp) if inp is not None else float(out)
        out = float(out) if out is not None else float(inp)
        costs[name] = (inp, out)
    return costs


_WORD_RE = re.compile(r"\w+")
_MATH_RE = re.compile(r"[=+\-*/^%]|\\frac|\\sqrt|\\int|\\sum|\bsolve\b|\bcalculate\b|\bprove\b|\bderivative\b|\bintegral\b|\btheorem\b", re.I)
_CODE_RE = re.compile(r"```|\bdef\b|\bclass\b|\bimport\b|\bfunction\b|\breturn\b|\balgorithm\b|\bcomplexity\b|#include|public\s+static", re.I)
_REASON_RE = re.compile(r"\bwhy\b|\bexplain\b|step[-\s]by[-\s]step|\btherefore\b|\bderive\b|\banalyze\b|\bcompare\b|\bevaluate\b", re.I)
_MCQ_RE = re.compile(r"(^|\n)\s*[A-E][\.\):]\s", re.M)


def _difficulty(query: str) -> float:
    """Coarse difficulty score in [0,1] from lexical signals. Higher = harder."""
    if not query:
        return 0.0
    words = _WORD_RE.findall(query)
    n = len(words)

    # Length: saturating. ~15 words -> 0, ~250 words -> ~1.
    length_sig = min(1.0, max(0.0, (n - 15) / 235.0))

    math_hits = len(_MATH_RE.findall(query))
    code_hits = len(_CODE_RE.findall(query))
    reason_hits = len(_REASON_RE.findall(query))

    math_sig = min(1.0, math_hits / 4.0)
    code_sig = min(1.0, code_hits / 3.0)
    reason_sig = min(1.0, reason_hits / 2.0)

    # Multiple-choice questions are constrained and tend to be easier: a discount.
    mcq = 1.0 if _MCQ_RE.search(query) else 0.0

    score = (
        0.30 * length_sig
        + 0.30 * math_sig
        + 0.25 * code_sig
        + 0.15 * reason_sig
        - 0.20 * mcq
    )
    return min(1.0, max(0.0, score))


class CruqRouter(BaseRouter):
    """Difficulty-tiered cheapest-sufficient router (Phase 1 heuristic)."""

    def __init__(self, router_name: str):
        super().__init__(router_name)
        params = self.config["pipeline_params"]
        w_out = float(params.get("price_weight_output", 0.5))
        costs = _load_model_costs()

        def blended(model: str) -> float:
            inp, out = costs.get(model, (1.0, 2.0))
            return (1.0 - w_out) * inp + w_out * out

        # Sort the configured pool cheapest -> strongest (price as a capability proxy).
        self._ordered: List[str] = sorted(self.models, key=blended)
        self._prices: List[float] = [blended(m) for m in self._ordered]

        n = len(self._ordered)
        thresholds = params.get("difficulty_thresholds")
        if thresholds is None:
            # Evenly spaced cut points: n models -> n-1 interior thresholds.
            thresholds = [i / n for i in range(1, n)]
        self._thresholds: List[float] = sorted(float(t) for t in thresholds)

    def _tier_for(self, difficulty: float) -> int:
        idx = 0
        for t in self._thresholds:
            if difficulty >= t:
                idx += 1
            else:
                break
        return min(idx, len(self._ordered) - 1)

    def _get_prediction(self, query: str) -> str:
        d = _difficulty(query)
        return self._ordered[self._tier_for(d)]
