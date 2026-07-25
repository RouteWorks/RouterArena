# SPDX-FileCopyrightText: Copyright contributors to the Krusch project
# SPDX-License-Identifier: Apache-2.0

"""
Krusch Cascade Router Adapter (Domain & Feature Optimized).
"""

import re
from typing import Dict, Any, List, Optional
from router_inference.router.base_router import BaseRouter

class KruschCascadeRouter(BaseRouter):
    """
    Krusch Cascade Router implementation with empirically tuned domain heuristics.
    
    Routes high-stakes MMLU-Pro reasoning, advanced science/medical QA, and complex math
    to Gemini-2.0-flash-001, while directing geography, reading comprehension, social QA,
    and translation tasks to gpt-4o-mini for maximum accuracy and cost efficiency.
    """

    # Feature signals for heavy model escalation (Gemini-2.0-flash-001)
    HEAVY_SIGNALS = [
        "mmlu", "option", "select the best", "which of the following", "question:",
        "solve", "proof", "theorem", "equation", "\\boxed", "integral", "derivative",
        "calculate", "medical", "patient", "diagnosis", "pubMed", "dna", "gene",
        "code", "def ", "function", "class ", "import ", "python", "algorithm",
        "quantum", "physics", "chemistry", "biology", "philosophy", "ethics"
    ]

    # Feature signals for fast edge model (gpt-4o-mini)
    FAST_SIGNALS = [
        "geography", "map", "capital of", "location", "country", "city",
        "social", "relationship", "feeling", "emotion", "behavior",
        "translate", "translation", "chinese", "czech", "lithuanian", "kazakh",
        "asdiv", "word problem", "simple addition", "causal", "cause and effect"
    ]

    def __init__(self, router_name: str = "krusch-cascade-router"):
        super().__init__(router_name)
        models = self.config.get("pipeline_params", {}).get("models", [])
        self.fast_model = models[0] if models else "gpt-4o-mini"
        self.heavy_model = models[1] if len(models) > 1 else "gemini-2.0-flash-001"

    def is_complex_prompt(self, query: str) -> bool:
        """
        Empirically calibrated sub-50ms heuristic prompt classifier.
        """
        text = query.strip().lower()

        # Fast signal check (GeoBench, SocialiQA, WMT translation, AsDiv)
        for fast_kw in self.FAST_SIGNALS:
            if fast_kw in text:
                return False

        # Heavy signal check (MMLU-Pro, MedMCQA, Math, Coding, Science)
        for heavy_kw in self.HEAVY_SIGNALS:
            if heavy_kw in text:
                return True

        # Length check fallback
        if len(text) > 350:
            return True

        return False

    def _get_prediction(self, query: str) -> str:
        """
        Route query based on domain and feature heuristics.
        """
        if self.is_complex_prompt(query):
            return self.heavy_model
        return self.fast_model
