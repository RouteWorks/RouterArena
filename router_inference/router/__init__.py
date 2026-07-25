# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Router inference module for RouterArena."""

from router_inference.router.auto_router import auto_router
from router_inference.router.base_router import BaseRouter
from router_inference.router.chuzom_solo_v32 import ChuzomSoloV32Router
from router_inference.router.example_router import ExampleRouter
from router_inference.router.krusch_cascade_adapter import KruschCascadeRouter
from router_inference.router.llm_router import LLMRouter
from router_inference.router.lynkr_router import LynkrRouter
from router_inference.router.vllm_sr import VLLMSR

__all__ = [
    "VLLMSR",
    "BaseRouter",
    "ChuzomSoloV32Router",
    "ExampleRouter",
    "KruschCascadeRouter",
    "LLMRouter",
    "LynkrRouter",
    "auto_router",
]

