# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities for LLM inference pipelines.
Contains common initialization and utility functions.
"""

import sys
import os
import warnings
from zoneinfo import ZoneInfo
import datetime
import logging
from typing import Any, Dict, List, Optional
from main_utils import set_logger, lock_seed

# Suppress the specific langchain warning
warnings.filterwarnings(
    "ignore", message="Convert_system_message_to_human will be deprecated!"
)


def setup_environment() -> tuple[str, str]:
    """Set up the environment for LLM inference."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "../"))
    sys.path.append(base_dir)
    os.chdir(base_dir)

    # Initialize API keys and HuggingFace login
    from huggingface_hub import login
    from dotenv import load_dotenv

    # initialize_all_api_keys()
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)

    return current_dir, base_dir


def setup_logging_and_seed(
    current_dir: str, model_name: str
) -> tuple[logging.Logger, int]:
    """Set up logging and seed for reproducible results."""
    SEED = 42
    lock_seed(SEED)

    logger = logging.getLogger(__name__)
    logger = set_logger(current_dir, model_name=model_name)

    return logger, SEED


def get_timezone_info() -> tuple[ZoneInfo, datetime.datetime]:
    """Get timezone information for logging."""
    ct_timezone = ZoneInfo("America/Chicago")
    start_time = datetime.datetime.now(ct_timezone)
    return ct_timezone, start_time


def log_pipeline_start(
    logger: logging.Logger,
    pipeline_type: str,
    model_name: str,
    start_time: datetime.datetime,
    seed: int,
):
    """Log pipeline start information."""
    logger.info("=" * 60)
    logger.info(f"{pipeline_type.upper()} PIPELINE STARTED")
    logger.info(f"Model: {model_name}")
    logger.info(f"Start time: {start_time}")
    logger.info(f"Seed: {seed}")
    logger.info("=" * 60)


def log_pipeline_completion(
    logger: logging.Logger,
    pipeline_type: str,
    model_name: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    results: Optional[List[Dict[str, Any]]] = None,
):
    """Log pipeline completion information."""
    duration_minutes = (end_time - start_time).total_seconds() / 60

    logger.info("=" * 60)
    logger.info(f"{pipeline_type.upper()} PIPELINE COMPLETED")
    logger.info(f"Model: {model_name}")
    logger.info(f"End time: {end_time}")
    logger.info(f"Duration: {duration_minutes:.2f} minutes")

    # Summary statistics
    if results:
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]
        total_tokens = sum(
            r.get("token_usage", {}).get("total_tokens", 0) for r in successful_results
        )

        logger.info(
            f"Results: {len(successful_results)} successful, {len(failed_results)} failed"
        )
        logger.info(
            f"Total tokens: {total_tokens:,} | Avg per success: {total_tokens / len(successful_results):.1f}"
            if successful_results
            else "No successful results"
        )

    logger.info("=" * 60)


def create_result_entry(
    global_index: int,
    question: str,
    model_name: str,
    inference_result: dict,
    additional_fields: Optional[Dict[str, Any]] = None,
) -> dict:
    """Create a standardized result entry."""
    result_entry = {
        "global_index": global_index,
        "question": question,
        "llm_selected": model_name,
        "generated_answer": inference_result.get("response", ""),
        "token_usage": inference_result.get("token_usage", {}),
        "success": inference_result.get("success", False),
        "provider": inference_result.get("provider", "unknown"),
        "error": inference_result.get("error", None)
        if not inference_result.get("success", False)
        else None,
    }

    if additional_fields:
        result_entry.update(additional_fields)

    return result_entry


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in filenames."""
    return model_name.replace("/", "_").replace("\\", "_")


def build_chat(tokenizer, prompt, chat_template):
    """
    Build chat-formatted prompt using the specified chat template.

    Args:
        tokenizer: HuggingFace tokenizer object
        prompt: Input prompt string
        chat_template: Chat template name (e.g., 'llama2', 'llama3', 'mistral', etc.)

    Returns:
        Formatted prompt string
    """
    if chat_template is None:
        return prompt

    if "llama3" in chat_template.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif "llama2" in chat_template.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif "longchat" in chat_template or "vicuna" in chat_template:
        from fastchat.model import get_conversation_template  # type: ignore[import-not-found]

        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "mistral" in chat_template.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif "recurrentgemma" in chat_template:
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif "mamba-chat" in chat_template:
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif "granite" in chat_template.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif "zephyr" in chat_template.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif "rwkv" in chat_template:
        prompt = f"""User: hi

                Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

                User: {prompt}

                Assistant:"""
    elif "qwen" in chat_template.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    elif "wizardlm" in chat_template.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    return prompt
