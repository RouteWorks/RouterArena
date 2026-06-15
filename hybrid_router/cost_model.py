import json


class CostModel:
    def __init__(self, pricing: dict):
        """
        Takes a pricing dict which has model names as keys and sub-dicts with "input_token_price_per_million"
        and "output_token_price_per_million" as values. Stores it as self.pricing.
        """
        self.pricing = pricing

    def estimate(self, model_name: str, input_tokens: int, output_budget: int) -> float:
        """
        Returns the estimated cost in USD using cost = (input_tokens * input_price_per_million / 1,000,000) +
        (output_budget * output_price_per_million / 1,000,000).
        """
        input_price_per_million = self.pricing[model_name][
            "input_token_price_per_million"
        ]
        output_price_per_million = self.pricing[model_name][
            "output_token_price_per_million"
        ]

        cost = (input_tokens * input_price_per_million / 1000000) + (
            output_budget * output_price_per_million / 1000000
        )

        return cost

    @classmethod
    def from_json(cls, path: str):
        """
        A class method that loads a JSON file and returns a CostModel instance.
        This is what the adapter will use to load model_cost or model_cost.json at startup.
        """
        with open(path, "r") as file:
            data = json.load(file)

        return cls(data)
