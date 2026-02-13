# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Universal model names for ICLR router evaluation.

This module contains the list of universal model names that correspond to
files in ./router_evaluation/llm_inference/outputs/
"""

universal_names = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-1106-preview",
    "o4-mini",
    "gpt-5-chat-latest",
    "gpt-5-mini",
    "gpt-5-nano",
    # Anthropic models
    "claude-3-haiku-20240307",
    "claude-3-7-sonnet-20250219",
    # Google models
    "gemini-2.0-flash-001",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    # Mistral models
    "mistral-medium",
    "codestral-latest",
    "open-mixtral-8x7b",
    "mistral-large-latest",
    "mistral-medium-latest",
    "mistral-small-latest",
    "open-mistral-7b",
    "open-mistral-nemo",
    # DeepSeek models
    "deepseek-coder",
    # Together AI models
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
    "meta-llama/Llama-3-70b-chat-hf",
    # OpenRouter
    "mistralai/mixtral-8x7b-instruct",
    "mistralai/mistral-7b-instruct",
    "meta-llama/llama-3-8b-instruct",
    "anthropic/claude-3.5-sonnet",
    "Qwen/QwQ-32B",
    "xiaomi/mimo-v2-flash",
    "mistralai/devstral-2512:free",
    # Replicate
    "meta/codellama-34b-instruct",
    # AWS Bedrock
    "llama-3-1-8b-instruct",
    "llama-3-2-1b-instruct",
    "llama-3-2-3b-instruct",
    "llama-3-3-70b-instruct",
    "llama-3-1-405b-instruct",
    # Zhipu
    "glm-4-air",
    "glm-4-flash",
    "glm-4-plus",
    # R2-Router
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
    "qwen/qwen3-30b-a3b-instruct-2507",
    "Qwen/Qwen3-Coder-Next",
    "qwen/qwen3-coder-30b-a3b-instruct",
    "mistralai/ministral-3-3b-2512",
    "mistralai/ministral-3-8b-2512",
    "mistralai/ministral-3-14b-2512",
    "google/gemma-3n-e4b-it",
    "claude-haiku-4.5",
]


mapping: dict[str, str] = {
    # this mapping is for the model names in your config file to be converted to universal model names that is supported in our pipeline.
}


class ModelNameManager:
    """
    Manager for model names.
    """

    def __init__(self):
        self.universal_names = universal_names
        # Basic mapping for common variations

        self.missing_models = set()

    def get_universal_name_non_static(self, model_name: str) -> str:
        """Convert a model name to its universal equivalent."""

        if model_name in universal_names:
            return model_name
        elif model_name in mapping:
            return mapping[model_name]
        else:
            self.missing_models.add(model_name)
            # raise ValueError(f"Model name {model_name} not found in universal_names or mapping")

        return model_name

    @staticmethod
    def get_universal_name(model_name: str) -> str:
        """Convert a model name to its universal equivalent."""

        if model_name in universal_names:
            return model_name
        elif model_name in mapping:
            return mapping[model_name]
        else:
            # self.missing_models.add(model_name)
            raise ValueError(
                f"Model name {model_name} not found in universal_names or mapping"
            )
