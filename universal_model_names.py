# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Universal model names for ICLR router evaluation.

This module contains the list of universal model names that correspond to
files in ./router_evaluation/llm_inference/outputs/
"""

from __future__ import annotations

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
    "gpt-5",
    "gpt-5-chat",
    "gpt-5.2",
    "gpt-5.2-chat",
    "gpt-5.3-chat",
    "gpt-5.4-nano",
    "gpt-5.4",
    "gpt-oss-120b",
    # Anthropic models
    "claude-3-haiku-20240307",
    "claude-3-7-sonnet-20250219",
    "claude-opus-4-1",
    "claude-opus-4-6",
    # Google models
    "gemini-2.0-flash-001",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "google/gemini-3.1-flash-lite",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-pro",
    "anthropic/claude-sonnet-4.5",
    "gemini-3-flash-preview",
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
    "deepseek/deepseek-v4-flash",
    "deepseek-v3.1",
    "deepseek-v3.2",
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
    "qwen/qwen3.5-9b",
    "deepseek/deepseek-v3.2",
    "openai/gpt-4o",
    "qwen/qwen3-235b-a22b-2507",
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
    # meta models
    "llama-4-maverick-17b-128e-instruct-fp8",
    # xAI models
    "grok-4",
    "grok-4-1-fast-reasoning",
    "grok-4.3",
    # R2-Router
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-next-80b-a3b-instruct",
    "qwen/qwen3-30b-a3b-instruct-2507",
    "Qwen/Qwen3-Coder-Next",
    "qwen/qwen3-coder-30b-a3b-instruct",
    "mistralai/ministral-3b-2512",
    "mistralai/ministral-3-3b-2512",
    "mistralai/ministral-3-8b-2512",
    "mistralai/ministral-3-14b-2512",
    "google/gemma-3n-e4b-it",
    "claude-haiku-4.5",
    # Weave Router (v0.27)
    "claude-opus-4-7",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "gpt-5.5",
    "gpt-5.4-mini",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
    "deepseek/deepseek-v4-pro",
    "qwen/qwen3.5-flash-02-23",
    "deepseek/deepseek-v4-flash",
    "moonshotai/kimi-k2.5",
    # OrcaRouter pool additions
    "claude-sonnet-4",
    "claude-haiku-4-5-20251001",
    "deepseek-chat",
    "deepseek-reasoner",
    "gemini-2.5-flash-lite",
    "qwen3-235b-a22b-instruct-2507",
    "qwen3-30b-a3b-instruct-2507",
    # Lynkr pool additions (all served via OpenRouter)
    "openai/gpt-oss-120b",
    "z-ai/glm-4.7",
    # Sqwish Router pool additions (OpenRouter-served)
    "gpt-oss-20b",
    "xiaomi/mimo-v2.5",
    # Paix2-router pool additions
    "MiniMax-M3",
    "agnes-2.0-flash",
    "THUDM/GLM-4-9B-0414",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    # KT-ModelRouter pool additions
    "google/gemma-4-31b-it",
]


mapping: dict[str, str] = {
    # this mapping is for the model names in your config file to be converted to universal model names that is supported in our pipeline.
    # OrcaRouter provider-prefixed → bare forms (used by arena-eval pipeline)
    "anthropic/claude-sonnet-4": "claude-sonnet-4",
    "anthropic/claude-sonnet-4-5": "claude-sonnet-4-5",
    "anthropic/claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
    "deepseek/deepseek-chat": "deepseek-chat",
    "deepseek/deepseek-reasoner": "deepseek-reasoner",
    "google/gemini-2.5-flash": "gemini-2.5-flash",
    "google/gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "openai/gpt-5-mini": "gpt-5-mini",
    "alibaba/qwen3-235b-a22b-instruct-2507": "qwen3-235b-a22b-instruct-2507",
    "alibaba/qwen3-30b-a3b-instruct-2507": "qwen3-30b-a3b-instruct-2507",
    # KT-ModelRouter: provider-prefixed form used in our predictions file
    "google/gemini-3-flash-preview": "gemini-3-flash-preview",
}


def canonical_key(model_name: str) -> str:
    """Fold a model name to a key that ignores cosmetic spelling differences.

    The registries in this repo were written by many hands and disagree on two
    purely cosmetic points: letter case (``Qwen/Qwen3-Coder-Next`` vs
    ``qwen/qwen3-coder-next``) and the vendor separator (``qwen_qwen3-coder``
    vs ``qwen/qwen3-coder``). Both spellings name the same model, but an exact
    dict lookup treats them as different models -- and a model that fails to
    resolve cannot be priced. See issue #193.

    Folding is deliberately conservative: it normalises case and the separator
    and nothing else. It does NOT strip vendor prefixes, because ``vendor/foo``
    and a bare ``foo`` are not reliably the same model and the explicit
    ``mapping`` table below already records the cases where they are.
    """

    return model_name.strip().lower().replace("_", "/")


def _build_canonical_index() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Map canonical_key -> universal name, plus any keys that are ambiguous.

    Two different universal names that fold to the same key, or an alias whose
    key shadows a universal name and points somewhere else, make the fold
    ambiguous. That is a registry bug rather than a submission bug, so it is
    reported (see ``tools/check_model_names.py``) rather than raised at import
    time -- importing this module must never be the thing that breaks a run.
    """

    index: dict[str, str] = {}
    collisions: dict[str, set[str]] = {}

    def add(key: str, target: str) -> None:
        if key in index and index[key] != target:
            collisions.setdefault(key, {index[key]}).add(target)
            return  # first writer wins, matching lookup precedence below
        index[key] = target

    for name in universal_names:
        add(canonical_key(name), name)
    for alias, target in mapping.items():
        add(canonical_key(alias), target)

    return index, {key: sorted(names) for key, names in collisions.items()}


canonical_index: dict[str, str]
canonical_collisions: dict[str, list[str]]
canonical_index, canonical_collisions = _build_canonical_index()


def resolve_universal_name(model_name: str) -> str | None:
    """Resolve a model name to its universal name, or None if unknown.

    Tries, in order: the name as written, the explicit ``mapping`` table, then
    the canonical fold. Returning None (rather than the input unchanged) lets
    callers decide whether an unresolvable name is fatal.
    """

    if model_name in universal_names:
        return model_name
    if model_name in mapping:
        return mapping[model_name]
    return canonical_index.get(canonical_key(model_name))


class ModelNameManager:
    """
    Manager for model names.
    """

    def __init__(self):
        self.universal_names = universal_names
        # Basic mapping for common variations

        self.missing_models = set()

    def get_universal_name_non_static(self, model_name: str) -> str:
        """Convert a model name to its universal equivalent.

        Non-fatal by design: unknown names are recorded in ``missing_models``
        and returned unchanged, so a caller can survey a whole run before
        deciding. Callers that must not proceed on an unknown model should use
        the static :meth:`get_universal_name`, or check ``missing_models``
        afterwards. See ``tools/check_model_names.py``.
        """

        resolved = resolve_universal_name(model_name)
        if resolved is None:
            self.missing_models.add(model_name)
            return model_name

        return resolved

    @staticmethod
    def get_universal_name(model_name: str) -> str:
        """Convert a model name to its universal equivalent.

        Raises ``ValueError`` if the name cannot be resolved, so that an
        unpriceable model fails loudly rather than being scored around.
        """

        resolved = resolve_universal_name(model_name)
        if resolved is None:
            raise ValueError(
                f"Model name {model_name} not found in universal_names or mapping"
            )

        return resolved
