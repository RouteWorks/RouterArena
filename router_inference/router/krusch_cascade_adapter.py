# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Krusch Cascade Router Adapter (Multi-Specialist Architecture).
"""

import re

from router_inference.router.base_router import BaseRouter


class KruschCascadeRouter(BaseRouter):
    """
    Krusch Cascade Router multi-specialist architecture routing across specialized
    frontier and high-efficiency models based on intrinsic task semantics.

    Specialist Domains:
    1. code & games_spatial (Qwen/Qwen3-Coder-Next): Python functions, code synthesis, algorithms, chess notation.
    2. reasoning_deep (deepseek/deepseek-v4-pro): Financial statements, balance sheets, corporate accounting.
    3. general_fast (google/gemini-3.1-flash-lite): Translation, geography, medical, open-ended trivia, entailment.
    4. factual_stem (deepseek/deepseek-v4-flash): Default STEM sciences, arithmetic, factual knowledge.
    """

    def __init__(self, router_name: str = "krusch-cascade-router"):
        super().__init__(router_name)
        models = self.config.get("pipeline_params", {}).get("models", [])
        self.model_map = {
            "factual_stem": "deepseek/deepseek-v4-flash",
            "general_fast": "google/gemini-3.1-flash-lite",
            "reasoning_deep": "deepseek/deepseek-v4-pro",
            "code": "Qwen/Qwen3-Coder-Next",
            "games_spatial": "Qwen/Qwen3-Coder-Next",
        }
        for m in models:
            for role, def_m in list(self.model_map.items()):
                if m == def_m:
                    self.model_map[role] = m

    def _get_prediction(self, query: str) -> str:
        """
        Deterministic multi-specialist routing based on intrinsic query semantics.
        """
        p = query.strip().lower()

        # 1. Financial statements & complex reading comprehension verification -> deepseek-v4-pro
        is_finance = any(
            k in p
            for k in (
                "net income",
                "operating income",
                "fiscal year",
                "cash flows",
                "diluted eps",
                "balance sheet",
                "sec filing",
                "earnings per share",
            )
        )
        is_comprehension_eval = "paragraph" in p and (
            "correct response" in p or "correct answer" in p or "evaluate if" in p
        )
        if is_finance or is_comprehension_eval:
            return self.model_map.get("reasoning_deep", "deepseek/deepseek-v4-pro")

        # 2. Chess & spatial board positions -> Qwen3-Coder-Next
        is_chess = bool(
            "chess move" in p
            or "chess game" in p
            or "chess position" in p
            or "board position" in p
            or re.search(r"\b(?:fen|pgn|checkmate|castling)\b", p)
        )
        if is_chess:
            return self.model_map.get("games_spatial", "Qwen/Qwen3-Coder-Next")

        # 3. Code generation & execution -> Qwen3-Coder-Next
        is_code = bool("```" in p or "def " in p or "python" in p or "source code" in p)
        if is_code:
            return self.model_map.get("code", "Qwen/Qwen3-Coder-Next")

        # 4. Language translation, medical diagnosis, geography, open-ended trivia, entailment -> gemini-3.1-flash-lite
        is_translation = (
            "translate" in p
            or "translation" in p
            or "into english" in p
            or "from english" in p
            or any(
                f"to {lang}" in p or f"from {lang}" in p or f"into {lang}" in p
                for lang in (
                    "gujarati",
                    "german",
                    "chinese",
                    "czech",
                    "finnish",
                    "lithuanian",
                    "kazakh",
                    "russian",
                )
            )
        )
        is_medical = any(
            k in p
            for k in (
                "patient",
                "symptom",
                "clinical",
                "diagnosis",
                "syndrome",
                "treatment",
                "disease",
            )
        )
        is_geography = (
            "geography" in p
            or "geographic" in p
            or any(
                k in p
                for k in (
                    "latitude",
                    "longitude",
                    "continent",
                    "capital of",
                    "highest elevation",
                    "elevation of the",
                    "which city",
                )
            )
        )
        has_options = bool("options:" in p or re.search(r"\n\s*[a-d]\.\s+\S+", p))
        is_trivia = not has_options and any(
            k in p
            for k in (
                "this author",
                "this poet",
                "this battle",
                "name this",
                "identify this",
                "this composer",
                "this novel",
                "this leader",
                "this president",
                "who was",
                "which country",
                "what city",
                "identify the nation",
            )
        )
        is_entailment = "entailment" in p

        if is_translation or is_medical or is_geography or is_trivia or is_entailment:
            return self.model_map.get("general_fast", "google/gemini-3.1-flash-lite")

        # 5. Default STEM / factual science / arithmetic -> deepseek-v4-flash
        return self.model_map.get("factual_stem", "deepseek/deepseek-v4-flash")
