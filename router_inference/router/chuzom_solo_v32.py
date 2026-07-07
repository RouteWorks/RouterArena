# SPDX-License-Identifier: MIT
"""Chuzom single-model baseline router: routes every query to deepseek-v3.2.

Chosen for best accuracy-per-dollar from PUBLISHED benchmarks. Uses only the
query (never RA templates), no RA-derived supervision, evaluator unmodified.
"""

from router_inference.router.base_router import BaseRouter


class ChuzomSoloV32Router(BaseRouter):
    def __init__(self, router_name: str) -> None:
        super().__init__(router_name)
        self._pick = (
            "deepseek/deepseek-v3.2"
            if "deepseek/deepseek-v3.2" in self.models
            else self.models[0]
        )

    def _get_prediction(self, query: str) -> str:
        return self._pick
