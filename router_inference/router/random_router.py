# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Random router implementation.

This router randomly selects a model from the configured model pool
for each query. Useful as a baseline for testing and benchmarking.
"""

import random

from router_inference.router.base_router import BaseRouter


class RandomRouter(BaseRouter):
    """
    Random router implementation.

    This router randomly selects a model from the configured model pool
    for each query, using a uniform distribution. Supports an optional
    seed for reproducibility.
    """

    def __init__(self, router_name: str):
        super().__init__(router_name)
        seed = self.config.get("pipeline_params", {}).get("seed", None)
        self.rng = random.Random(seed)

    def _get_prediction(self, query: str) -> str:
        """
        Randomly select a model from the configured model pool.

        Args:
            query: The input query string (unused for random selection)

        Returns:
            Name of a randomly selected model
        """
        return self.rng.choice(self.models)
