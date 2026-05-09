# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Prepare HellaSwag dataset for RouterArena pipeline.

This script:
1. Loads HellaSwag dataset from HuggingFace (Rowan/hellaswag)
2. Formats prompts according to RouterArena requirements
3. Creates dataset/hellaswag_data.json (for router inference)
4. Creates dataset/hellaswag_ground_truth.json (for evaluation)

How to run:
    uv run python scripts/prepare_hellaswag_data.py

Prerequisites:
    - Install packages: uv sync (or pip install datasets)
"""

from datasets import load_dataset
import json
import os

# Ensure dataset directory exists
os.makedirs("dataset", exist_ok=True)

# Define prompt template (must match eval config!)
PROMPT_TEMPLATE = """Please read the following multiple-choice questions and provide the most likely correct answer based on the options given.

Context: {context}

Question: {question}

Options:
{options}

Provide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences."""

# Mapping from label index to letter
INDEX_TO_LETTER = {"0": "A", "1": "B", "2": "C", "3": "D"}


def process_hellaswag():
    """
    Load and process the HellaSwag dataset.

    Combines train and validation splits (test split has no labels).

    Returns:
        Tuple of (formatted_data, ground_truth) lists
    """
    print("Loading HellaSwag from HuggingFace...")
    ds = load_dataset("Rowan/hellaswag")

    formatted_data = []
    ground_truth = []
    entry_index = 0
    skipped_no_label = 0

    # Process train and validation splits (test has no labels)
    for split_name in ds.keys():
        split_ds = ds[split_name]
        print(f"  Processing {split_name} split: {len(split_ds)} entries")

        for item in split_ds:
            activity_label = item.get("activity_label", "")
            ctx = item.get("ctx", "")
            endings = item.get("endings", [])
            label = str(item.get("label", ""))

            # Skip entries with no label (test split)
            if label == "" or label not in INDEX_TO_LETTER:
                skipped_no_label += 1
                entry_index += 1
                continue

            if not endings:
                entry_index += 1
                continue

            # Convert label index to letter (0->A, 1->B, 2->C, 3->D)
            answer_letter = INDEX_TO_LETTER[label]

            # Build question from activity label
            question = (
                f"What is the most likely continuation of the following scenario "
                f"about '{activity_label}'?"
            )

            # Format options as "A. ending1\nB. ending2\n..."
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            options_str = ""
            for i, ending in enumerate(endings):
                options_str += f"{letters[i]}. {ending}\n"

            # Build the complete prompt
            prompt = PROMPT_TEMPLATE.format(
                context=ctx or "None",
                question=question,
                options=options_str.strip(),
            )

            # Create global index
            global_index = f"HellaSwag_{entry_index}"

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
                    "answer": answer_letter,
                    "options": endings,
                    "context": ctx,
                    "metadata": {
                        "activity_label": activity_label,
                        "source_id": item.get("source_id", ""),
                        "split": split_name,
                        "ind": item.get("ind", -1),
                    },
                }
            )

            entry_index += 1

    if skipped_no_label > 0:
        print(f"  Skipped {skipped_no_label} entries with no label (test split)")
    print(f"  Total HellaSwag entries with labels: {len(formatted_data)}")
    return formatted_data, ground_truth


# Process dataset
print("=" * 60)
print("Preparing HellaSwag dataset for RouterArena")
print("=" * 60)

all_data, all_gt = process_hellaswag()

print(f"\nTotal: {len(all_data)} entries")

# Save dataset file
dataset_path = "dataset/hellaswag_data.json"
with open(dataset_path, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)
print(f"\n✓ Created {len(all_data)} entries in {dataset_path}")

# Save ground truth file
gt_path = "dataset/hellaswag_ground_truth.json"
with open(gt_path, "w", encoding="utf-8") as f:
    json.dump(all_gt, f, indent=2, ensure_ascii=False)
print(f"✓ Created {len(all_gt)} ground truth entries in {gt_path}")

# Verify files
print("\n[Verification] Checking files...")
try:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✓ Dataset file: {len(data)} entries")
    print(f"  First entry keys: {list(data[0].keys())}")
    print(f"  First global_index: {data[0].get('global index')}")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    print(f"✓ Ground truth file: {len(gt)} entries")
    print(f"  First entry keys: {list(gt[0].keys())}")
    print(f"  First answer: {gt[0].get('answer')}")

    # Verify matching indices
    data_indices = {e.get("global index") for e in data}
    gt_indices = {e.get("global_index") for e in gt}
    if data_indices == gt_indices:
        print(
            f"✓ All {len(data_indices)} indices match between dataset and ground truth"
        )
    else:
        missing_in_data = gt_indices - data_indices
        missing_in_gt = data_indices - gt_indices
        if missing_in_data:
            print(
                f"⚠ Warning: {len(missing_in_data)} indices in "
                f"ground truth not in dataset"
            )
        if missing_in_gt:
            print(
                f"⚠ Warning: {len(missing_in_gt)} indices in "
                f"dataset not in ground truth"
            )

    # Verify answer distribution
    answer_letters = {e.get("answer") for e in gt}
    print(f"✓ Unique answer letters: {sorted(answer_letters)}")

    print("\n✓ All files created and verified successfully!")
    print("\nNext steps:")
    print(
        "1. Review dataset/hellaswag_data.json to ensure prompts are "
        "formatted correctly"
    )
    print("2. Review dataset/hellaswag_ground_truth.json to ensure answers are correct")
    print("3. Proceed to Step 3: Router Inference Setup")

except Exception as e:
    print(f"✗ Verification failed: {e}")
    raise
