# SPDX-FileCopyrightText: Copyright contributors to the cruq.ai project
# SPDX-License-Identifier: Apache-2.0

"""
cruq learned router (Phase 2): cheapest-sufficient via a capability predictor.

For each query it embeds the prompt (local MiniLM) and, using per-model logistic
heads trained on EXTERNAL data (phase2/data/predictor.pkl), predicts P(correct)
for every pool model. It then picks the CHEAPEST model whose predicted success
probability clears a threshold tau; if none clear it, it falls back to the model
with the highest predicted probability. This is the selector that closes the gap
between the lexical router (~0.62 optimal-selection) and the pool's oracle.

Rule compliance: the heads are fit only on the external corpus (public benchmark
splits deduped against RouterArena), never on RouterArena items or labels.

Config pipeline_params:
  - "tau": success-probability threshold (default 0.5)
  - "predictor_path": artifact path (default phase2/data/predictor.pkl)
  - "price_weight_output": output-price weight for the cost ranking (default 0.5)

Heavy deps (torch, sentence-transformers) are imported lazily in __init__ so that
importing this class never slows routers that do not use it.
"""

import json
import os
import pickle
from typing import List

from router_inference.router.base_router import BaseRouter
from router_inference.router.cruq_router import _load_model_costs


class CruqLearnedRouter(BaseRouter):
    def __init__(self, router_name: str):
        super().__init__(router_name)
        params = self.config["pipeline_params"]
        self._tau = float(params.get("tau", 0.5))
        w = float(params.get("price_weight_output", 0.5))
        path = params.get("predictor_path", "phase2/data/predictor.pkl")

        costs = _load_model_costs()

        def blended(m):
            i, o = costs.get(m, (1.0, 2.0))
            return (1.0 - w) * i + w * o

        self._ordered: List[str] = sorted(self.models, key=blended)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"predictor artifact not found: {path}. Run phase2/train_predictor.py first."
            )
        with open(path, "rb") as f:
            art = pickle.load(f)
        self._heads = {m: art["heads"][m] for m in self._ordered if m in art.get("heads", {})}

        # lazy heavy import, once
        import torch
        torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(art["embedder"])

    def _get_prediction(self, query: str) -> str:
        v = self._embedder.encode([query], normalize_embeddings=True)
        best_m, best_p = self._ordered[0], -1.0
        for m in self._ordered:
            head = self._heads.get(m)
            if head is None:
                continue
            p = float(head.predict_proba(v)[0][1])
            if p >= self._tau:
                return m  # cheapest model clearing tau
            if p > best_p:
                best_m, best_p = m, p
        return best_m  # none cleared tau -> highest predicted P(correct)
