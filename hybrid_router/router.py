# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

import math

class HybridRouter:
    def __init__(
        self, encoder, heads, scaler, curves, cost_model, budget_candidates=None
    ):
        """
        Takes the parameters encoder, heads, scaler, curves, cost_model, and budget_candidates in order.
        Budget candidates is set to be [80, 150, 200, 400, 800, 1500] by default.
        Stores them all as instance attributes.
        """
        self.encoder = encoder
        self.heads = heads
        self.scaler = scaler
        self.curves = curves
        self.cost_model = cost_model
        self.budget_candidates = budget_candidates or [80, 150, 200, 400, 800, 1500]

    def route(self, query: str) -> tuple[str, int]:
        """
        Encodes the query using self.encoder.encode(query) to get the embedding.
        For each model in self.heads.heads and for each budget in self.budget_candidates, it computes a score for (model, budget).

        Gets the raw logit from self.heads.predict(model, embedding) as base_quality.
        Gets the curve quality from self.curves.quality_at_budget(model, budget).
        Blends these values together as 0.5 * base_quality + 0.5 * curve_quality.
        Gets the cost from self.cost_model.estimate(mode, len(query.split()), budget).
        The score is blended / (cost + 1e-9)

        Stores the score in a dict keyed by (model, budget) tuples.
        Finds the key with the highest score.
        Returns that (model, budget) tuple.
        """
        encoded_query = self.encoder.encode(query)

        results = {}
        for model in self.heads.heads:
            sigmoid_out = self.heads.predict(model, encoded_query)
            raw_logit = math.log(max(sigmoid_out, 1e-9) / max(1 - sigmoid_out, 1e-9))
            calibrated = self.scaler.apply(model, raw_logit)

            for budget in self.budget_candidates:
                curve_quality = self.curves.quality_at_budget(model, budget)
                blended = 0.5 * calibrated + 0.5 * curve_quality
                cost = self.cost_model.estimate(model, len(query.split()), budget)
                score = blended / (cost + 1e-9)
                results[(model, budget)] = score

        max_key = max(results, key=results.get)
        return max_key
