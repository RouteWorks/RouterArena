class HybridRouter:
    def __init__(self, encoder, head_collection, calibration, 
                 budget_curves, cost_model, budget_candidates):
        pass

    def route(self, query: str) -> tuple[str, int]:
        """
        Returns (model_name, budget) for the given query.
        Selects argmax of quality(model, budget) / cost(model, budget) across all (model x budget_candidate) pairs.
        """

        embedding = self.encoder.encode(query)
        scores = {}

        for model in self.models:
            raw_logit = self.heads.predict(model, embedding)
            base_quality = self.calibration.apply(model, raw_logit)
            for budget in self.budget_candidates:
                quality = self.curves.quality_at_budget(model, embedding, budget)

                # Blend base qualuty prediction with curve estimate.
                blended = 0.5 * base_quality + 0.5 * quality

                cost = self.cost_model.estimate(model, len(query.split()), budget)
                scores[(model, budget)] = blended / (cost + 1e-9)

        best_model, best_budget = max(scores, key=scores.__getitem__)
        return best_model, best_budget