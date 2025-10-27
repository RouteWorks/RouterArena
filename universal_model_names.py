"""
Universal model names for ICLR router evaluation.

This module contains the list of universal model names that correspond to
files in ./router_evaluation/llm_inference/outputs/
"""

universal_names = [
    "01-ai_Yi-34B-Chat",
    "claude-3-5-haiku",
    "claude-3-5-sonnet",
    "claude-3-7-sonnet-20250219",
    "claude-3-haiku-20240307",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "codellama_CodeLlama-34b-Instruct-hf",
    "codestral-latest",
    "cognitivecomputations_dolphin-2.6-mistral-7b",
    "cognitivecomputations_dolphin-2.9-llama3-8b",
    "deepseek_deepseek-chat",
    "gemini-1.5-pro-latest",
    "gemini-2.0-flash-001",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "glm-4-air",
    "glm-4-flash",
    "glm-4-plus",
    "google_gemini-flash-1.5",
    "gpt-3.5-turbo-1106",
    "gpt-4.1",
    "gpt-4-1106-preview",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gpt-5-chat-latest",
    "gpt-5-mini",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano",
    "gpt-5-nano-2025-08-07",
    "HuggingFaceH4_zephyr-7b-beta",
    "ibm-granite_granite-3.0-2b-instruct",
    "ibm-granite_granite-3.0-8b-instruct",
    "itpossible_Chinese-Mistral-7B-v0.1",
    "llama-3-1-405b-instruct",
    "llama-3-1-8b-instruct",
    "llama-3-2-1b-instruct",
    "llama-3-2-3b-instruct",
    "llama-3-3-70b-instruct",
    "meta_codellama-34b-instruct",
    "meta_llama-2-70b-chat",
    "meta-llama_Llama-2-70b-chat-hf",
    "meta-llama_Llama-2-7b-chat-hf",
    "meta-llama_Llama-3.1-70B-Instruct",
    "meta-llama_Llama-3.1-8B-Instruct",
    "meta-llama_Llama-3-70b-chat-hf",
    "meta-llama_llama-3-8b-instruct",
    "meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama_Meta-Llama-3-70B",
    "meta-llama_Meta-Llama-3-70B-Instruct",
    "meta-llama_Meta-Llama-3-70B-Instruct-Turbo",
    "meta-llama_Meta-Llama-3-8B-Instruct",
    "meta-math_MetaMath-Mistral-7B",
    "mistralai_Ministral-8B-Instruct-2410",
    "mistralai_mistral-7b-instruct",
    "mistralai_Mistral-7B-Instruct-v0.2",
    "mistralai_Mistral-7B-Instruct-v0.3",
    "mistralai_mixtral-8x7b-instruct",
    "mistralai_Mixtral-8x7B-Instruct-v0.1",
    "mistral-large-latest",
    "mistral-medium-latest",
    "mistral-small-latest",
    "NousResearch_Nous-Hermes-2-Yi-34B",
    "o1-mini",
    "o4-mini",
    "o4-mini-2025-04-16",
    "open-mistral-7b",
    "open-mistral-nemo",
    "open-mixtral-8x7b",
    "Qwen_Qwen1.5-72B-Chat",
    "Qwen_Qwen2.5-32B",
    "Qwen_Qwen2.5-32B-Instruct-GPTQ-Int4",
    "qwen_qwen-2.5-72b-instruct",
    "Qwen_Qwen2.5-72B-Instruct",
    "qwen_qwen-2.5-7b-instruct",
    "Qwen_Qwen2.5-7B-Instruct",
    "Qwen_Qwen2.5-Math-7B-Instruct",
    "Qwen_QwQ-32B",
    "togetherai_Meta-Llama-3.1-70B-Instruct-Turbo",
    "WizardLM_WizardLM-13B-V1.2",
]


mapping = {
    "anthropic_claude-3.5-sonnet": "claude-3-5-sonnet",
    "ibm-granite_granite-3-8b-instruct": "ibm-granite_granite-3.0-8b-instruct",
    "ibm-granite_granite-3-2b-instruct": "ibm-granite_granite-3.0-2b-instruct",
    "HuggingFaceH4/zephyr-7b-beta": "HuggingFaceH4_zephyr-7b-beta",
    "itpossible/Chinese-Mistral-7B-v0.1": "itpossible_Chinese-Mistral-7B-v0.1",
    "cognitivecomputations/dolphin-2.6-mistral-7b": "cognitivecomputations_dolphin-2.6-mistral-7b",
    "meta-math/MetaMath-Mistral-7B": "meta-math_MetaMath-Mistral-7B",
    "mistralai/mistral-7b-instruct": "mistralai_mistral-7b-instruct",
    "meta-llama/llama-3-8b-instruct": "meta-llama_llama-3-8b-instruct",
    "cognitivecomputations/dolphin-2.9-llama3-8b": "cognitivecomputations_dolphin-2.9-llama3-8b",
    "mistralai/mistral-7b-chat": "mistralai_mistral-7b-instruct",
    "meta/llama-2-7b-chat": "meta-llama_Llama-2-7b-chat-hf",
    "meta/llama-3.1-turbo-70b-chat": "meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta/llama-3-70b-chat": "meta-llama_Llama-3-70b-chat-hf",
    "mistralui/mixtral-8x7b-chat": "mistralai_Mixtral-8x7B-Instruct-v0.1",
    "nousresearch/nous-34b-chat": "NousResearch_Nous-Hermes-2-Yi-34B",
    "meta/llama-3-turbo-70b-chat": "meta-llama_Meta-Llama-3-70B-Instruct-Turbo",
    "qwen/qwen-1.5-72b-chat": "Qwen_Qwen1.5-72B-Chat",
    "mixtral-8x7b-instruct-v0.1": "mistralai_Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/llama-3.1-405b-instruct": "llama-3-1-405b-instruct",
    "mistralai/mixtral-8x7b-chat": "mistralai_Mixtral-8x7B-Instruct-v0.1",
    "anthropic/claude-3.5-sonnet": "claude-3-5-sonnet",
    "openai/gpt-4o": "gpt-4o",
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
