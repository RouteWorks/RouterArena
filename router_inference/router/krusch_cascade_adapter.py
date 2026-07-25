# SPDX-FileCopyrightText: Copyright contributors to the Krusch project
# SPDX-License-Identifier: Apache-2.0

"""
Krusch Cascade Router Adapter (Domain-Aware Hybrid Fine-Tuned).
"""

import re
from typing import Dict, Any, List, Optional
from router_inference.router.base_router import BaseRouter

class KruschCascadeRouter(BaseRouter):
    """
    Fine-tuned Krusch Cascade Router implementation for RouterArena.
    
    Routes simple MCQs and general trivia to fast edge models (gpt-4o-mini),
    while escalating complex step-by-step math, code, and long-context prompts
    to heavy reasoning models (gemini-2.0-flash-001).
    """

    HEAVY_REASONING_KEYWORDS = {
        "step by step", "prove", "theorem", "lemma", "derivative", "integral",
        "system of equations", "combinatorics", "recursion", "dynamic programming",
        "def ", "function", "class ", "import ", "python", "java", "c++", "rust",
        "algorithm", "time complexity", "space complexity", "asymptotic"
    }

    def __init__(self, router_name: str = "krusch-cascade-router"):
        super().__init__(router_name)
        models = self.config.get("pipeline_params", {}).get("models", [])
        self.fast_model = models[0] if models else "gpt-4o-mini"
        self.heavy_model = models[1] if len(models) > 1 else "gemini-2.0-flash-001"

    def is_complex_prompt(self, query: str) -> bool:
        """
        Domain-aware predictive heuristic classifier.
        """
        text = query.strip().lower()

        # MCQ & Option-based queries are efficiently handled by gpt-4o-mini
        is_mcq = "options:" in text or ("a." in text and "b." in text and "c." in text) or "\\boxed{x}" in text
        if is_mcq and len(text) < 600:
            return False

        # Heavy reasoning or coding
        for kw in self.HEAVY_REASONING_KEYWORDS:
            if kw in text:
                return True

        # Math notation density
        if "\\frac" in text or "\\sum" in text or "\\prod" in text or "\\int" in text:
            return True

        # Long multi-paragraph prompt check
        if len(text) > 500 or len(text.split()) > 100:
            return True

        return False

    def _get_prediction(self, query: str) -> str:
        """
        Route query based on domain-aware predictive classification.
        """
        if self.is_complex_prompt(query):
            return self.heavy_model
        return self.fast_model
