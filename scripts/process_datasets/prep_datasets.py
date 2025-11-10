# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import base64
import json
import pickle
import zlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from datasets import load_dataset, load_from_disk
from typing import Dict, Any, List

save_dir = "./dataset/"

# Load public dataset (without answers for full split)
router_benchmark = load_dataset("RouteWorks/RouterArena", split="sub_10")
router_benchmark.save_to_disk(os.path.join(save_dir, "routerarena_10"))

router_benchmark = load_dataset("RouteWorks/RouterArena", split="full")
router_benchmark.save_to_disk(os.path.join(save_dir, "routerarena"))

# If HF_TOKEN is available, also load from private repo with answers for full split
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if hf_token:
    try:
        print(
            "[prep] HF_TOKEN detected. Loading full dataset with answers from private repo..."
        )
        router_benchmark_with_answers = load_dataset(
            "RouteWorks/RouterEvalBenchmark",
            split="full",
            token=hf_token,
        )
        # Overwrite the full dataset with the one containing answers
        router_benchmark_with_answers.save_to_disk(
            os.path.join(save_dir, "routerarena")
        )
        print("[prep] Successfully loaded full dataset with answers from private repo.")
    except Exception as e:
        print(
            f"[prep] Warning: Could not load from private repo (this is OK for public users): {e}"
        )
        print(
            "[prep] Continuing with public dataset (answers will be missing for full split)."
        )
else:
    print(
        "[prep] No HF_TOKEN found. Using public dataset (answers missing for full split)."
    )


def escape_format_braces(text):
    """
    Escape curly braces in input text to prevent them from being interpreted
    as format variables when using str.format().
    """
    if not isinstance(text, str):
        return text
    result = ""
    i = 0
    while i < len(text):
        if text[i] == "{":
            if i + 1 < len(text) and text[i + 1] == "{":
                result += "{{"
                i += 2
            else:
                result += "{{"
                i += 1
        elif text[i] == "}":
            if i + 1 < len(text) and text[i + 1] == "}":
                result += "}}"
                i += 2
            else:
                result += "}}"
                i += 1
        else:
            result += text[i]
            i += 1
    return result


def safe_format_prompt(prompt_template, **kwargs):
    """
    Safely format a prompt template by escaping curly braces in the input data.
    """
    escaped_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            escaped_kwargs[key] = escape_format_braces(value)
        else:
            escaped_kwargs[key] = value
    return prompt_template.format(**escaped_kwargs)


def build_formatted_prompts_from_router_eval_benchmark(
    split_name: str,
) -> List[Dict[str, Any]]:
    # Load benchmark split
    ds = load_dataset("RouteWorks/RouterArena")[split_name]

    # Load LiveCodeBench dataset saved to disk (if available)
    lcd_dataset_list: List[Dict[str, Any]] = []
    lcd_global_idx_map: Dict[str, Dict[str, Any]] = {}
    lcd_index_map: Dict[str, Dict[str, Any]] = {}
    lcd_prompt_prefix_map: Dict[str, Dict[str, Any]] = {}
    try:
        lcd = load_from_disk("./dataset/livecodebench")
        # Convert to list for index-based access
        lcd_dataset_list = lcd.to_list()
        # Build fast lookup maps
        for item in lcd_dataset_list:
            gidx_key = (
                str(item.get("global_idx"))
                if item.get("global_idx") is not None
                else None
            )
            if gidx_key is not None:
                lcd_global_idx_map[gidx_key] = item
            idx_key = (
                str(item.get("_index")) if item.get("_index") is not None else None
            )
            if idx_key is not None:
                lcd_index_map[idx_key] = item
            # Use prompt prefix (from processed dataset) for fuzzy matching fallback
            prompt_text = item.get("prompt") or ""
            if isinstance(prompt_text, str) and prompt_text:
                lcd_prompt_prefix_map[prompt_text[:120]] = item
    except Exception as e:
        print(
            f"[prep] Warning: could not load ./dataset/livecodebench ({e}). LiveCodeBench prompts may fail."
        )

    # Load configs for all known datasets
    config_dir = "./config/eval_config/zero-shot"
    dataset_names = [
        "AIME",
        "ArcMMLU",
        "AsDiv",
        "ChessInstruct_mcq",
        "ChessInstruct",
        "Ethics_commonsense",
        "Ethics_deontology",
        "Ethics_justice",
        "Ethics_virtue",
        "FinQA",
        "GeoBench",
        "GeoGraphyData",
        "GSM8K",
        "LiveCodeBench",
        "MATH",
        "MathQA",
        "MedMCQA",
        "MMLUPro",
        "MMLU",
        "MusicTheoryBench",
        "NarrativeQA",
        "OpenTDB",
        "PubMedQA",
        "QANTA",
        "SocialiQA",
        "SuperGLUE-CausalReasoning",
        "SuperGLUE-ClozeTest",
        "SuperGLUE-Entailment",
        "SuperGLUE-QA",
        "SuperGLUE-RC",
        "SuperGLUE-Wic",
        "SuperGLUE-Wsc",
        "WMT19-cs-en",
        "WMT19-de-en",
        "WMT19-fi-en",
        "WMT19-gu-en",
        "WMT19-kk-en",
        "WMT19-lt-en",
        "WMT19-ru-en",
        "WMT19-zh-en",
    ]
    dataset_configs: Dict[str, Dict[str, Any]] = {}
    for dataset_name in dataset_names:
        cfg_path = os.path.join(config_dir, f"{dataset_name}.json")
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                dataset_configs[dataset_name] = cfg.get("eval_params", {})
        except FileNotFoundError:
            print(f"[prep] Warning: Config file not found for {dataset_name}")
            continue

    formatted: List[Dict[str, Any]] = []
    for row in ds:
        global_index = (
            row.get("Global Index")
            or row.get("global_index")
            or row.get("global index")
        )
        # Special handling: if Global Index starts with "Ethics", include up to next "_"
        if not row.get("Dataset name"):
            global_index_parts = row["Global Index"].split("_")
            if global_index_parts[0] == "Ethics" and len(global_index_parts) >= 2:
                dataset_name_full = f"{global_index_parts[0]}_{global_index_parts[1]}"
            else:
                dataset_name_full = global_index_parts[0]
        else:
            dataset_name_full = row.get("Dataset name")

        if "Ethics" in dataset_name_full:
            base_dataset_name = dataset_name_full
        elif "ChessInstruct_mcq" in dataset_name_full:
            base_dataset_name = dataset_name_full
        else:
            base_dataset_name = str(dataset_name_full).split("_", 1)[0]

        assert base_dataset_name in dataset_configs, (
            f"No config for {base_dataset_name}"
        )
        eval_params = dataset_configs[base_dataset_name]

        # Build options string if present
        options_list = row.get("Options")
        options_str = ""
        if options_list:
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for i, opt in enumerate(options_list):
                letter = letters[i] if i < len(letters) else "-"
                options_str += f"{letter}. {opt}\n"

        question = row.get("Question", "")
        context_val = row.get("Context", "")
        context_for_prompt = context_val if context_val != "" else "None"

        if base_dataset_name == "LiveCodeBench":
            # Primary: use global index to match the processed lcb dataset's global_idx
            lcd_row = lcd_global_idx_map.get(str(global_index))
            if lcd_row is None:
                # Fallback: try _index suffix
                idx = str(global_index).split("_")[-1]
                lcd_row = lcd_index_map.get(idx)
            if lcd_row is None:
                # Fallback: fuzzy match by question/prompt prefix
                q_prefix = (question or "")[:120]
                lcd_row = lcd_prompt_prefix_map.get(q_prefix)
            if lcd_row is None:
                raise AssertionError(
                    f"LiveCodeBench row not found for Global Index {global_index} by global_idx, index, or prompt match."
                )
            prompt = (
                eval_params.get("is_stdin_prompt")
                if lcd_row.get("is_stdin")
                else eval_params.get("not_is_stdin_prompt")
            )
            prompt_formatted = safe_format_prompt(
                prompt or "{Question}", Question=question
            )
        elif base_dataset_name == "SuperGLUE-RC":
            prompt_formatted = safe_format_prompt(
                eval_params.get("prompt", "{Question}"),
                Question=question,
                Answer=row.get("Answer", ""),
            )
        elif base_dataset_name == "SuperGLUE-Wic":
            prompt_formatted = safe_format_prompt(
                eval_params.get("prompt", "{Question}"),
                Question=question,
                Context=context_val,
            )
        elif not options_list:
            prompt_formatted = safe_format_prompt(
                eval_params.get("prompt", "{Question}"),
                Context=context_for_prompt,
                Question=question,
            )
        else:
            prompt_formatted = safe_format_prompt(
                eval_params.get("prompt", "{Question}"),
                Context=context_for_prompt,
                Question=question,
                Options=options_str,
            )

        if len(prompt_formatted) > 10000:
            prompt_formatted = f"{prompt_formatted[:5000]}...{prompt_formatted[-5000:]}"

        assert len(prompt_formatted) > 0, (
            f"Prompt formatted is empty for {dataset_name_full}"
        )

        formatted.append(
            {
                "prompt_formatted": prompt_formatted,
                "global index": global_index,
            }
        )
    return formatted


def write_pipeline_datasets() -> None:
    out_dir = "./dataset"
    os.makedirs(out_dir, exist_ok=True)

    # full split
    full_data = build_formatted_prompts_from_router_eval_benchmark("full")
    with open(os.path.join(out_dir, "router_data.json"), "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)
    print(
        f"[prep] Wrote {len(full_data)} items to {os.path.join(out_dir, 'router_data.json')}"
    )

    # sub_10 split (if available)
    try:
        sub10_data = build_formatted_prompts_from_router_eval_benchmark("sub_10")
        with open(
            os.path.join(out_dir, "router_data_10.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(sub10_data, f, ensure_ascii=False, indent=2)
        print(
            f"[prep] Wrote {len(sub10_data)} items to {os.path.join(out_dir, 'router_data_10.json')}"
        )
    except Exception as e:
        print(f"[prep] Warning: could not build sub_10 split: {e}")


def add_idx_map(x: dict, idx: int) -> dict:
    # We convert to string for consistency
    x["_index"] = str(idx)
    return x


def translate_private_test_cases(encoded_data):
    try:
        decoded_data = base64.b64decode(encoded_data)
        decompressed_data = zlib.decompress(decoded_data)
        original_data = pickle.loads(decompressed_data)
        return json.loads(original_data)
    except Exception as e:
        print(f"Error processing private_test_cases: {e}")
        return None


def has_test_type(tests, type):  ## helper to select specific type of problems
    """
    Check if any test in the test list has 'testtype' set to 'type'.
    """
    try:
        test_list = json.loads(tests)
        for test in test_list:
            if test.get("testtype") == type:
                return True
        return False
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        print(f"Error parsing tests in has_test_type: {e}")
        return False


def map_to_example(row):
    return {
        "prompt": row["question_content"],
        "test": row["private_test_cases"],
        "entry_point": row["starter_code"],
        "canonical_solution": "",  # seems like live code bench lite does not have this field
        "task_id": row["question_id"],
        "is_stdin": has_test_type(row["public_test_cases"], "stdin"),
        "public_test_cases": row["public_test_cases"],
        "difficulty": row["difficulty"],
        "global_idx": row.get("global_idx", None),  # Add global_idx field
        # "_index": row["_index"],
    }


router_benchmark_df = router_benchmark.to_pandas()
router_benchmark_lcb = router_benchmark_df[
    router_benchmark_df["Global Index"].str.contains("LiveCodeBench")
]
router_benchmark_lcb["idx"] = (
    router_benchmark_lcb["Global Index"].str.split("_").str[-1]
)

# Load LiveCodeBench datasets with error handling
lcd_v2 = load_dataset("lighteval/code_generation_lite", "release_v2", split="test")


# Filter lcb_codegen to keep only rows whose question_content is in router_benchmark_lcb
router_benchmark_lcb_questions = set(router_benchmark_lcb["idx"])
print(
    f"Number of unique questions in router_benchmark_lcb: {len(router_benchmark_lcb_questions)}"
)

# Add index column to the dataset using map
lcb_codegen = lcd_v2.map(add_idx_map, with_indices=True)

# Create a mapping from question content to global index
# We'll match questions by comparing their content
question_to_global_idx = {}
for _, row in router_benchmark_lcb.iterrows():
    question_to_global_idx[row["Question"]] = row["Global Index"]

print(f"Created mapping for {len(question_to_global_idx)} questions")


# Add global_idx field to each example by matching question content
def add_global_idx(example):
    # Try to find matching question in router_benchmark_lcb
    prompt = example["question_content"]

    # Look for exact match first
    if prompt in question_to_global_idx:
        example["global_idx"] = question_to_global_idx[prompt]
    else:
        # Try partial matching (first 100 characters)
        prompt_start = prompt[:100]
        for question, global_idx in question_to_global_idx.items():
            if question.startswith(prompt_start) or prompt_start in question:
                example["global_idx"] = global_idx
                break
        else:
            example["global_idx"] = None

    return example


lcb_codegen = lcb_codegen.map(add_global_idx)
print(f"Added global_idx field to {len(lcb_codegen)} examples")

# Filter to keep only examples that have a global_idx (i.e., are in our benchmark)
lcb_codegen = lcb_codegen.filter(lambda example: example["global_idx"] is not None)
print(f"Filtered to {len(lcb_codegen)} examples with matching global_idx")

lcb_codegen = lcb_codegen.map(
    lambda example: {
        "private_test_cases": translate_private_test_cases(
            example["private_test_cases"]
        )
    },
    writer_batch_size=100,
)

# Filter out examples where private_test_cases translation failed
lcb_codegen = lcb_codegen.filter(
    lambda example: example["private_test_cases"] is not None
)
print(f"Number of rows after filtering out failed translations: {len(lcb_codegen)}")

# Fix the remove_columns issue by creating a new list without "_index"
columns_to_remove = [col for col in lcb_codegen.column_names if col != "_index"]
lcb_codegen = lcb_codegen.map(
    map_to_example,
    remove_columns=columns_to_remove,
    writer_batch_size=100,
)

lcb_codegen.save_to_disk(os.path.join(save_dir, "livecodebench"))

write_pipeline_datasets()
