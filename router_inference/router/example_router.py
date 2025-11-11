# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Example router implementation.

This is a simple example router that demonstrates how to implement
the BaseRouter abstract class. It selects the first model in the config
for all queries.
"""

from router_inference.router.base_router import BaseRouter


class ExampleRouter(BaseRouter):
    """
    Example router implementation.

    This router simply cycles through the models from the config for all queries.
    This is intended as a demonstration and should be replaced with actual
    routing logic in production implementations.
    """

    def __init__(self, router_name: str):
        super().__init__(router_name)
        self.counter = 0
        self.length = len(self.models)

    def _get_prediction(self, query: str) -> str:
        """
        Get the model prediction for a given query (internal implementation).

        This example implementation cycles through models in the config.
        In a real implementation, you would analyze the query and select
        the most appropriate model.

        Args:
            query: The input query string

        Returns:
            Name of the selected model
        """
        # Simple example: cycle through models
        model_name = self.models[self.counter % self.length]
        self.counter += 1
        return model_name
