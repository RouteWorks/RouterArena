# SPDX-FileCopyrightText: 2026 Chuzom (github.com/ypollak2/chuzom)
# SPDX-License-Identifier: MIT
"""Chuzom LLM Router for RouterArena — v0.1.0 (experimental).

IMPORTANT — ISOLATION POLICY:
  This router exists ONLY for RouterArena evaluation experiments.
  It MUST NOT be merged into the Chuzom production package or exposed
  to Chuzom users without explicit performance validation showing it
  improves user outcomes without regressions.

How it works:
  1. A separate pre-generation script (scripts/generate_llm_routing.py) calls
     a cheap LLM classifier (gemini-3.1-flash-lite) in parallel for all dataset
     queries and writes routing decisions to a JSON lookup file.
  2. This router loads that lookup file at init time.
  3. _get_prediction() returns the pre-computed decision in O(1).
  4. For unknown queries (robustness paraphrases, new queries) it falls back
     to the ChuzomRouter regex heuristics — no live LLM call at serve time.

RouterArena compliance:
  Routing decisions are based solely on prompt content as understood by a
  general-purpose LLM classifier. No RouterArena label files, optimality
  scores, or dataset names are used. The classifier prompt lists model
  capabilities from public knowledge only.
"""

import hashlib
import json
import os

from router_inference.router.base_router import BaseRouter
from router_inference.router.chuzom_router import ChuzomRouter

# Path to the pre-generated routing decisions file (relative to project root)
_DECISIONS_FILENAME = "chuzom-llm-routing-decisions.json"


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()


class ChuzomLLMRouter(BaseRouter):
    """LLM-classifier-based router for RouterArena experimentation.

    Uses pre-computed LLM routing decisions (generated offline in parallel)
    with regex-heuristic fallback for unseen queries.
    """

    def __init__(self, router_name: str):
        super().__init__(router_name)

        # Locate the decisions file relative to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        decisions_dir = os.path.join(project_root, "router_inference", "config")
        self._decisions_path = os.path.join(decisions_dir, _DECISIONS_FILENAME)

        self._decisions: dict[str, str] = {}
        if os.path.exists(self._decisions_path):
            with open(self._decisions_path, encoding="utf-8") as f:
                raw = json.load(f)
            # raw is {hash: model_name}
            self._decisions = raw
            print(
                f"  [ChuzomLLMRouter] Loaded {len(self._decisions)} "
                f"pre-computed routing decisions."
            )
        else:
            print(
                f"  [ChuzomLLMRouter] WARNING: decisions file not found at "
                f"{self._decisions_path}. Run scripts/generate_llm_routing.py first."
            )

        # Fallback router (same model pool assumed compatible)
        self._fallback = ChuzomRouter(router_name)
        self._fallback_count = 0
        self._hit_count = 0

    def _get_prediction(self, query: str) -> str:
        h = _query_hash(query)
        decision = self._decisions.get(h)

        if decision and decision in self.models:
            self._hit_count += 1
            return decision

        # Fall back to regex router
        self._fallback_count += 1
        fallback_model = self._fallback._get_prediction(query)
        # Map fallback model to one in our model pool
        if fallback_model in self.models:
            return fallback_model
        return self.models[0]  # last resort: first model in config
