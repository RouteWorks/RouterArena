# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Prepare WinoGrande dataset for RouterArena pipeline.

This script:
1. Loads all WinoGrande subsets from HuggingFace (allenai/winogrande)
2. Formats prompts according to RouterArena requirements
3. Creates dataset/winogrande_data.json (for router inference)
4. Creates dataset/winogrande_ground_truth.json (for evaluation)

Subsets: winogrande_debiased, winogrande_l, winogrande_m,
         winogrande_s, winogrande_xl, winogrande_xs

How to run:
    uv run python scripts/prepare_winogrande_data.py

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

# Mapping from answer index to letter
ANSWER_TO_LETTER = {"1": "A", "2": "B"}

# All WinoGrande subsets
SUBSETS = [
    "winogrande_debiased",
    "winogrande_l",
    "winogrande_m",
    "winogrande_s",
    "winogrande_xl",
    "winogrande_xs",
]


def process_subset(subset_name: str):
    """
    Load and process a single WinoGrande subset.

    Combines all available splits with valid labels.

    Args:
        subset_name: e.g. "winogrande_xl"

    Returns:
        Tuple of (formatted_data, ground_truth) lists
    """
    print(f"\nLoading {subset_name} from HuggingFace...")
    ds = load_dataset("allenai/winogrande", subset_name)

    formatted_data = []
    ground_truth = []
    entry_index = 0
    skipped_no_label = 0

    # Friendly prefix for global index (e.g. "WinoGrande-xl")
    # Strip "winogrande_" prefix and convert to a clean label
    suffix = subset_name.replace("winogrande_", "")
    index_prefix = f"WinoGrande-{suffix}"

    for split_name in ds.keys():
        split_ds = ds[split_name]
        print(f"  Processing {split_name} split: {len(split_ds)} entries")

        for item in split_ds:
            sentence = item.get("sentence", "")
            option1 = item.get("option1", "")
            option2 = item.get("option2", "")
            answer = str(item.get("answer", ""))

            # Skip entries with no valid label (test split uses "" or "0")
            if answer not in ANSWER_TO_LETTER:
                skipped_no_label += 1
                entry_index += 1
                continue

            # Convert answer index to letter (1->A, 2->B)
            answer_letter = ANSWER_TO_LETTER[answer]

            # Build question from the sentence with blank
            question = (
                "Which option best fills the blank '_' in the following sentence?"
            )

            # Format options
            options_str = f"A. {option1}\nB. {option2}"

            # Build the complete prompt
            prompt = PROMPT_TEMPLATE.format(
                context=sentence,
                question=question,
                options=options_str,
            )

            # Create global index
            global_index = f"{index_prefix}_{entry_index}"

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
                    "options": [option1, option2],
                    "context": sentence,
                    "metadata": {
                        "subset": subset_name,
                        "split": split_name,
                    },
                }
            )

            entry_index += 1

    if skipped_no_label > 0:
        print(f"  Skipped {skipped_no_label} entries with no label (test split)")
    print(f"  Total {subset_name} entries with labels: {len(formatted_data)}")
    return formatted_data, ground_truth


# Process all subsets
print("=" * 60)
print("Preparing WinoGrande dataset for RouterArena")
print("=" * 60)

all_data = []
all_gt = []
subset_counts = {}

for subset in SUBSETS:
    data, gt = process_subset(subset)
    all_data.extend(data)
    all_gt.extend(gt)
    subset_counts[subset] = len(data)

print(f"\nCombined: {len(all_data)} total entries")
for subset, count in subset_counts.items():
    print(f"  {subset}: {count}")

# Save dataset file
dataset_path = "dataset/winogrande_data.json"
with open(dataset_path, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)
print(f"\n✓ Created {len(all_data)} entries in {dataset_path}")

# Save ground truth file
gt_path = "dataset/winogrande_ground_truth.json"
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
            f"✓ All {len(data_indices)} indices match between "
            f"dataset and ground truth"
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

except Exception as e:
    print(f"✗ Verification failed: {e}")
    raise
