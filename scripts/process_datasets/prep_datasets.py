# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import base64
import json
import pickle
import zlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from datasets import load_dataset

save_dir = "./dataset/"

router_benchmark = load_dataset("louielu02/RouterEvalBenchmark", split="full")
router_benchmark.save_to_disk(os.path.join(save_dir, "routerevalbench"))


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
    router_benchmark_df["Dataset name"] == "LiveCodeBench"
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
