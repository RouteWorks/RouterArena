# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""KDB Router: content-only, cost-first routing.

The question body (first/last paragraph dropped; line labels, option letters and "None" placeholders stripped) is
embedded with all-MiniLM-L6-v2 and classified into one of 11 content categories by a logistic regression trained on
an external calibration set (no RouterArena data). Each category maps to a model, optionally with a reasoning setting
(`model@low` = reasoning effort low, `model@off` = reasoning off); the prediction is the base model name.
Weights and table: config/kdb-router-model.json.
"""

import json
import os
import re

import numpy as np

from router_inference.router.base_router import BaseRouter

_CFG = os.path.join(os.path.dirname(__file__), "..", "config", "kdb-router-model.json")
_LABEL = re.compile(r"(?m)^[ \t]*[A-Za-z][A-Za-z0-9 _\-\"']{0,30}:[ \t]*")
_OPTION = re.compile(r"(?m)^[ \t]*\(?[A-Ja-j][\).:\]][ \t]+")
_PLACEHOLDER = re.compile(r"(?mi)^[ \t]*(none|null|n/a)[ \t]*$")


def routing_text(prompt: str) -> str:
    paras = [p for p in prompt.strip().split("\n\n") if p.strip()]
    body = prompt.strip() if len(paras) < 3 else "\n\n".join(paras[1:-1]).strip()
    text = _PLACEHOLDER.sub("", _OPTION.sub("", _LABEL.sub("", body)))
    return re.sub(r"\n{2,}", "\n", text).strip()


class KDBRouter(BaseRouter):
    def __init__(self, router_name: str):
        super().__init__(router_name)
        from sentence_transformers import SentenceTransformer

        with open(_CFG) as f:
            cfg = json.load(f)
        self._embed = SentenceTransformer(
            cfg["embed_model"], revision=cfg["embed_revision"]
        )
        self._classes = cfg["classes"]
        self._coef = np.array(cfg["coef"])
        self._intercept = np.array(cfg["intercept"])
        self._policy = cfg["policy"]
        self._default = cfg["default"]

    def route_option(self, query: str) -> str:
        x = self._embed.encode(
            [routing_text(query)], normalize_embeddings=True, show_progress_bar=False
        )[0]
        cat = self._classes[int(np.argmax(self._coef @ x + self._intercept))]
        return self._policy.get(cat, self._default)

    def _get_prediction(self, query: str) -> str:
        return self.route_option(query).split("@")[0]
