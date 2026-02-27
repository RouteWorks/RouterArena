# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Prepare HLE (Humanity's Last Exam) dataset for RouterArena pipeline.

This script:
1. Loads the cais/hle dataset from HuggingFace (test split only)
2. Filters to TEXT-ONLY questions (skips entries with non-empty image fields
   since RouterArena currently only supports text-based LLMs)
3. Formats prompts based on answer_type:
   - "exactMatch": direct open-QA prompt
   - "multipleChoice": MCQ prompt
4. Creates dataset/hle_data.json (for router inference)
5. Creates dataset/hle_ground_truth.json (for evaluation)

How to run:
    uv run python scripts/prepare_hle_data.py
"""

from datasets import load_dataset
import json
import os

# Ensure dataset directory exists
os.makedirs("dataset", exist_ok=True)

# Prompt templates (must match eval config!)
EXACT_MATCH_PROMPT = """Please read the following question and provide the correct answer.

Question: {question}

Provide the correct answer in \\boxed{{X}}, where X is the correct answer. Keep the explanation or feedback within 3 sentences."""

MCQ_PROMPT = """Please read the following multiple-choice question and provide the most likely correct answer based on the options given.

Question: {question}

Provide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences."""


def has_image(item):
    """Return True if the item requires image context."""
    image_val = item.get("image", "")
    if image_val and str(image_val).strip():
        return True
    return False


def process_hle():
    """
    Load and process the HLE dataset.

    Only processes the test split (the only split available).
    Filters to text-only entries.

    Returns:
        Tuple of (formatted_data, ground_truth) lists
    """
    print("Loading cais/hle from HuggingFace...")
    ds = load_dataset("cais/hle")

    formatted_data = []
    ground_truth = []
    entry_index = 0
    skipped_image = 0
    skipped_no_answer = 0
    type_counts: dict[str, int] = {}

    for split_name in ds.keys():
        split_ds = ds[split_name]
        print(f"  Processing {split_name} split: {len(split_ds)} entries")

        for item in split_ds:
            question_id = item.get("id", str(entry_index))
            question = item.get("question", "")
            answer = item.get("answer", "")
            answer_type = item.get("answer_type", "exactMatch")
            category = item.get("category", "")
            raw_subject = item.get("raw_subject", "")

            # Skip image-based questions
            if has_image(item):
                skipped_image += 1
                entry_index += 1
                continue

            # Skip entries with no answer
            if not answer or not answer.strip():
                skipped_no_answer += 1
                entry_index += 1
                continue

            # Track answer types
            type_counts[answer_type] = type_counts.get(answer_type, 0) + 1

            # Format prompt based on answer type
            if answer_type == "multipleChoice":
                prompt = MCQ_PROMPT.format(question=question)
            else:
                # exactMatch and any other types
                prompt = EXACT_MATCH_PROMPT.format(question=question)

            # Create global index
            global_index = f"HLE_{entry_index}"

            # Append formatted prompt
            formatted_data.append(
                {
                    "prompt_formatted": prompt,
                    "global index": global_index,
                }
            )

            # Append ground truth
            ground_truth.append(
                {
                    "global_index": global_index,
                    "question": question,
                    "answer": answer,
                    "metadata": {
                        "id": question_id,
                        "answer_type": answer_type,
                        "category": category,
                        "raw_subject": raw_subject,
                        "split": split_name,
                    },
                }
            )

            entry_index += 1

    print(
        f"  Skipped {skipped_image} image-based questions (not supported by text LLMs)"
    )
    if skipped_no_answer > 0:
        print(f"  Skipped {skipped_no_answer} entries with no answer")
    print(f"  Answer type distribution: {type_counts}")
    print(f"  Total HLE text-only entries: {len(formatted_data)}")
    return formatted_data, ground_truth


# Process dataset
print("=" * 60)
print("Preparing HLE dataset for RouterArena")
print("=" * 60)

all_data, all_gt = process_hle()

print(f"\nTotal: {len(all_data)} text-only entries")

# Save dataset file
dataset_path = "dataset/hle_data.json"
with open(dataset_path, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)
print(f"\n✓ Created {len(all_data)} entries in {dataset_path}")

# Save ground truth file
gt_path = "dataset/hle_ground_truth.json"
with open(gt_path, "w", encoding="utf-8") as f:
    json.dump(all_gt, f, indent=2, ensure_ascii=False)
print(f"✓ Created {len(all_gt)} ground truth entries in {gt_path}")

# Verify files
print("\n[Verification] Checking files...")
try:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✓ Dataset file: {len(data)} entries")
    if data:
        print(f"  First entry keys: {list(data[0].keys())}")
        print(f"  First global_index: {data[0].get('global index')}")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    print(f"✓ Ground truth file: {len(gt)} entries")
    if gt:
        print(f"  First entry keys: {list(gt[0].keys())}")
        print(f"  First answer: {gt[0].get('answer')}")
        # Show answer type distribution
        type_dist: dict[str, int] = {}
        for e in gt:
            at = e.get("metadata", {}).get("answer_type", "unknown")
            type_dist[at] = type_dist.get(at, 0) + 1
        print(f"  Answer type distribution: {type_dist}")

    # Verify matching indices
    data_indices = {e.get("global index") for e in data}
    gt_indices = {e.get("global_index") for e in gt}
    if data_indices == gt_indices:
        print(
            f"✓ All {len(data_indices)} indices match between dataset and ground truth"
        )
    else:
        print("⚠ Warning: Indices mismatch detected!")

    print("\n✓ All files created and verified successfully!")

except Exception as e:
    print(f"✗ Verification failed: {e}")
    raise
