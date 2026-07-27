# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Lynkr router adapter.

Queries a running Lynkr gateway (github.com/Fast-Editor/Lynkr) for its
routing decision on each prompt and maps Lynkr's complexity tier
(SIMPLE / MEDIUM / COMPLEX / REASONING) to a model from the config's
tier_map. Lynkr's /routing/analyze?mode=intent endpoint runs the same
WS7 anchor intent scorer used by its live proxy routing, without
performing any LLM inference.

Requires a Lynkr instance (default http://localhost:8081, override with
LYNKR_URL). Lynkr's embedding backend (Ollama nomic-embed-text) should be
up so scoring runs in anchor mode; if embeddings are unavailable Lynkr
falls back to lexical scoring, which is also its live-serving behavior.
"""

import json
import os
import time
import urllib.request

from router_inference.router.base_router import BaseRouter


class LynkrRouter(BaseRouter):
    """Adapter that proxies routing decisions to a live Lynkr gateway."""

    def __init__(self, router_name: str):
        super().__init__(router_name)
        params = self.config["pipeline_params"]
        self.tier_map = params["tier_map"]
        self.default_model = params.get("default_model", self.models[0])
        self.base_url = os.environ.get("LYNKR_URL", "http://localhost:8081")
        self.endpoint = f"{self.base_url}/routing/analyze?mode=intent"
        for tier, model in self.tier_map.items():
            if model not in self.models:
                raise ValueError(
                    f"tier_map[{tier}] = '{model}' is not in the config model pool"
                )

    def _get_prediction(self, query: str) -> str:
        payload = json.dumps({"messages": [{"role": "user", "content": query}]}).encode(
            "utf-8"
        )
        last_err = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    self.endpoint,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                tier = body.get("tier")
                if tier in self.tier_map:
                    return self.tier_map[tier]
                raise ValueError(f"Lynkr returned unknown tier: {tier!r}")
            except (urllib.error.URLError, TimeoutError, ValueError) as err:
                last_err = err
                time.sleep(1 + attempt)
        raise RuntimeError(
            f"Lynkr routing failed after 3 attempts ({last_err}). "
            f"Is Lynkr running at {self.base_url}?"
        )
