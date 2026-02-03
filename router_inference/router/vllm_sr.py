# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
vLLM Semantic Router implementation for RouterArena.

This router calls the vllm-project/semantic-router classification API
to determine the category of a query, then maps it to one of the
configured models.
"""

import json
import urllib.request
from typing import Dict, Optional

from router_inference.router.base_router import BaseRouter


class VLLMSR(BaseRouter):
    """
    vLLM Semantic Router implementation.

    Uses the vllm-sr classification API to categorize queries and route
    them to the most appropriate model based on the detected category.
    """

    def __init__(self, router_name: str):
        super().__init__(router_name)

        # Load configuration from config file
        pipeline_params = self.config["pipeline_params"]
        self.category_model_mapping = pipeline_params.get("category_model_mapping", {})
        self.default_model = pipeline_params.get("default_model", "gpt-4o-mini")
        self.base_url = pipeline_params.get("base_url", "http://localhost:8080")

    def _call_classify_api(self, query: str) -> Optional[Dict]:
        """Call the vllm-sr classification API."""
        try:
            data = json.dumps({"text": query}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/v1/classify/intent",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"  Warning: Classification API failed: {e}")
            return None

    def _get_prediction(self, query: str) -> str:
        """Get the model prediction using vllm-sr classification."""
        result = self._call_classify_api(query)

        if result and "classification" in result:
            category = result["classification"].get("category", "other")
            model = self.category_model_mapping.get(category, self.default_model)
            if model in self.models:
                return model

        return (
            self.default_model if self.default_model in self.models else self.models[0]
        )
