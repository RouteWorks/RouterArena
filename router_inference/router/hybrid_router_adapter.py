# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project.
# SPDX-License-Identifier: Apache-2.0

"""
Hybrid Router adapter for RouterArena.

Loads pre-trained MLP heads, temperature calibration, PCHIP budget curves,
and cost model. Then, wraps them in a HybridRouter instance.

Expected checkpoint layout (relative to project root):
    checkpoints/hybrid-router/
        heads/
            qwen_qwen3-235b-a22b-2507.pt
            qwen_qwen3-30b-a3b-instruct-2507.pt
            mistralai_ministral-3b-2512.pt
        curves.npz
        temperatures.json
"""

import os
import torch

from router_inference.router.base_router import BaseRouter
from hybrid_router.encoder import HybridRouterEncoder
from hybrid_router.model_heads import ModelHeadCollection
from hybrid_router.calibration import TemperatureScaler
from hybrid_router.budget_curves import BudgetCurves
from hybrid_router.cost_model import CostModel
from hybrid_router.router import HybridRouter


def _model_name_to_filename(model_name: str) -> str:
    """
    Converts a canonical model name to a safe filename stem.
    e.g. 'qwen/qwen3-235b-a22b-2507' -> 'qwen_qwen3-235b-a22b-2507'
    """
    return model_name.replace("/", "_")


class HybridRouterAdapter(BaseRouter):
    """
    RouterArena adapter for HybridRouter.

    Loads all trained artifacts on construction and delegates routing
    to HybridRouter.route(), returning only the model name to satisfy
    the BaseRouter interface.
    """

    def __init__(self, router_name: str):
        super().__init__(router_name)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        ckpt_dir = os.path.join(project_root, "checkpoints", "hybrid-router")

        # Load encoder (downloads all-mpnet-base-b2 on first run, cached after)
        encoder = HybridRouterEncoder()

        # Load MLP heads
        model_names = self.models
        collection = ModelHeadCollection(model_names)
        heads_dir = os.path.join(ckpt_dir, "heads")
        for name in model_names:
            filename = _model_name_to_filename(name) + ".pt"
            path = os.path.join(heads_dir, filename)
            state = torch.load(path, map_location="cpu", weights_only=True)
            collection.heads[name].load_state_dict(state)
            collection.heads[name].eval()

        # Load temperature calibration
        scaler = TemperatureScaler.load(os.path.join(ckpt_dir, "temperatures.json"))

        # Load PCHIP budget curves
        curves = BudgetCurves.load(os.path.join(ckpt_dir, "curves.npz"))

        # Load cost model from project-level model_cost.json
        cost_model_path = os.path.join(project_root, "model_cost", "model_cost.json")
        cost_model = CostModel.from_json(cost_model_path)

        self.router = HybridRouter(
            encoder=encoder,
            heads=collection,
            scaler=scaler,
            curves=curves,
            cost_model=cost_model,
        )

    def _get_prediction(self, query: str) -> str:
        model_name, _budget = self.router.route(query)
        return model_name
