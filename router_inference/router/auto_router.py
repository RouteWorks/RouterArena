# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Example router implementation.

This is a simple example router that demonstrates how to implement
the BaseRouter abstract class. It selects the first model in the config
for all queries.
"""

from router_inference.router.base_router import BaseRouter
import httpx


class auto_router(BaseRouter):
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

        self.id_list = [865, 866, 287, 305]
        self.id_to_modelname = {
            865: "deepseek/deepseek-v3.2",
            866: "qwen/qwen3-235b-a22b-2507",
            287: "qwen/qwen3.5-9b",
            305: "openai/gpt-4o",
        }

    def get_routing_result(self, query):
        """Test the routing API with the specified request"""
        url = "http://10.109.4.26:8501/api/v1/routing"

        headers = {"Content-Type": "application/json"}

        data = {
            "default": 287,
            "models": self.id_list,
            "messages": [{"role": "user", "content": query}],
            "strategy": 1,
            "qualityPercentage": 60,
            "option": {
                "multiRoundJudgeMode": 0,
                "enableRagWebJudge": False,
                "enableDomainJudge": False,
            },
        }

        # Send POST request
        response = httpx.post(url, headers=headers, json=data)

        model_name = self.id_to_modelname[response.json()["data"]["id"]]

        return model_name

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
        model_name = self.get_routing_result(query)
        print("----------------model_name:", model_name)
        self.counter += 1
        return model_name
