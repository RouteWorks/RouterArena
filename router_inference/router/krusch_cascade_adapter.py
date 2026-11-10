# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Krusch Cascade Router Adapter (Multi-Specialist Architecture).
"""

import re

from router_inference.router.base_router import BaseRouter

WORLD_LANGUAGES = (
    "spanish",
    "french",
    "german",
    "chinese",
    "japanese",
    "russian",
    "arabic",
    "hindi",
    "portuguese",
    "italian",
    "korean",
    "dutch",
    "polish",
    "turkish",
    "vietnamese",
    "czech",
    "finnish",
    "gujarati",
    "kazakh",
    "lithuanian",
)

CHOICE_LINE_RE = re.compile(r"(?:^|\n)\s*(?:[a-d][.)]|\([a-d]\))\s+\S+")
CHESS_TERM_RE = re.compile(r"\b(?:fen|pgn|checkmate|castling)\b")


class KruschCascadeRouter(BaseRouter):
    """
    Krusch Cascade Router multi-specialist architecture routing across specialized
    frontier and high-efficiency models based on intrinsic task semantics.

    Specialist Domains:
    1. code & games_spatial (Qwen/Qwen3-Coder-Next): Code synthesis, algorithms, chess notation.
    2. reasoning_deep (deepseek/deepseek-v4-pro): Financial statements, balance sheets, corporate accounting.
    3. general_fast (google/gemini-3.1-flash-lite): Multilingual translation, geography, clinical medicine, open-ended trivia.
    4. factual_stem (deepseek/deepseek-v4-flash): Default STEM sciences, mathematics, and general factual reasoning.
    """

    def __init__(self, router_name: str = "krusch-cascade-router") -> None:
        super().__init__(router_name)
        models = self.config.get("pipeline_params", {}).get("models", [])
        self.model_map = {
            "factual_stem": "deepseek/deepseek-v4-flash",
            "general_fast": "google/gemini-3.1-flash-lite",
            "reasoning_deep": "deepseek/deepseek-v4-pro",
            "code": "Qwen/Qwen3-Coder-Next",
            "games_spatial": "Qwen/Qwen3-Coder-Next",
        }
        for model in models:
            for role, default_model in list(self.model_map.items()):
                if model == default_model:
                    self.model_map[role] = model

    def _get_prediction(self, query: str) -> str:
        """
        Deterministic multi-specialist routing based strictly on intrinsic query semantics.
        """
        p = query.strip().lower()

        # 1. Financial statements & corporate accounting -> deepseek-v4-pro
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
        if is_finance:
            return self.model_map.get("reasoning_deep", "deepseek/deepseek-v4-pro")

        # 2. Chess & spatial board positions -> Qwen3-Coder-Next
        is_chess = bool(
            "chess move" in p
            or "chess game" in p
            or "chess position" in p
            or "board position" in p
            or CHESS_TERM_RE.search(p)
        )
        if is_chess:
            return self.model_map.get("games_spatial", "Qwen/Qwen3-Coder-Next")

        # 3. Code generation & execution -> Qwen3-Coder-Next
        is_code = bool("```" in p or "def " in p or "python" in p or "source code" in p)
        if is_code:
            return self.model_map.get("code", "Qwen/Qwen3-Coder-Next")

        # 4. Multilingual translation, clinical medicine, geography, open-ended trivia -> gemini-3.1-flash-lite
        is_translation = (
            "translate" in p
            or "translation" in p
            or "into english" in p
            or "from english" in p
            or any(
                f"to {lang}" in p or f"from {lang}" in p or f"into {lang}" in p
                for lang in WORLD_LANGUAGES
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
        has_choices = bool(CHOICE_LINE_RE.search(p))
        is_trivia = not has_choices and any(
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

        if is_translation or is_medical or is_geography or is_trivia:
            return self.model_map.get("general_fast", "google/gemini-3.1-flash-lite")

        # 5. Default STEM / mathematics / factual reasoning -> deepseek-v4-flash
        return self.model_map.get("factual_stem", "deepseek/deepseek-v4-flash")
