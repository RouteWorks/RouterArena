# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Prepare ARC dataset for RouterArena pipeline.

This script:
1. Loads ARC-Challenge and ARC-Easy datasets from HuggingFace
2. Formats prompts according to RouterArena requirements
3. Creates dataset/arc_data.json (for router inference)
4. Creates dataset/arc_ground_truth.json (for evaluation)

How to run:
    uv run python scripts/prepare_arc_data.py

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

# Mapping from numeric answer keys to letters
NUMERIC_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D"}


def normalize_answer_key(answer_key: str) -> str:
    """
    Normalize answerKey to a letter (A, B, C, D).

    Some entries use numeric keys (1, 2, 3, 4) instead of letters.
    """
    answer_key = answer_key.strip()
    if answer_key in NUMERIC_TO_LETTER:
        return NUMERIC_TO_LETTER[answer_key]
    return answer_key.upper()


def process_subset(subset_name: str):
    """
    Load and process a single ARC subset (ARC-Challenge or ARC-Easy).

    Combines all splits (train, validation, test) for maximum coverage.

    Args:
        subset_name: "ARC-Challenge" or "ARC-Easy"

    Returns:
        Tuple of (formatted_data, ground_truth) lists
    """
    print(f"\nLoading {subset_name} from HuggingFace...")
    ds = load_dataset("allenai/ai2_arc", subset_name)

    formatted_data = []
    ground_truth = []
    entry_index = 0

    # Process all available splits
    for split_name in ds.keys():
        split_ds = ds[split_name]
        print(f"  Processing {split_name} split: {len(split_ds)} entries")

        for item in split_ds:
            question = item.get("question", "")
            choices = item.get("choices", {})
            answer_key = item.get("answerKey", "")

            # Extract options text and labels
            choice_texts = choices.get("text", [])
            choice_labels = choices.get("label", [])

            if not choice_texts or not choice_labels:
                print(
                    f"  Warning: Skipping {subset_name}_{entry_index}: missing choices"
                )
                continue

            # Normalize labels: convert numeric labels ('1','2','3','4')
            # to letters ('A','B','C','D')
            normalized_labels = []
            for label in choice_labels:
                if label.strip() in NUMERIC_TO_LETTER:
                    normalized_labels.append(NUMERIC_TO_LETTER[label.strip()])
                else:
                    normalized_labels.append(label.upper())

            # Normalize answer key (handle numeric 1,2,3,4 -> A,B,C,D)
            answer_letter = normalize_answer_key(answer_key)

            # Validate that answer_letter is one of the normalized labels
            if answer_letter not in normalized_labels:
                print(
                    f"  Warning: answer '{answer_letter}' not in labels "
                    f"{normalized_labels} for {subset_name}_{entry_index}, "
                    f"skipping"
                )
                continue

            # Format options using normalized letter labels
            options_str = ""
            for label, text in zip(normalized_labels, choice_texts):
                options_str += f"{label}. {text}\n"

            # Build the complete prompt
            prompt = PROMPT_TEMPLATE.format(
                context="None", question=question, options=options_str.strip()
            )

            # Create global index with subset prefix
            global_index = f"{subset_name}_{entry_index}"

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
                    "options": choice_texts,
                    "context": "",
                    "metadata": {
                        "id": item.get("id", ""),
                        "subset": subset_name,
                        "split": split_name,
                    },
                }
            )

            entry_index += 1

    print(f"  Total {subset_name} entries: {entry_index}")
    return formatted_data, ground_truth


# Process both subsets
print("=" * 60)
print("Preparing ARC dataset for RouterArena")
print("=" * 60)

challenge_data, challenge_gt = process_subset("ARC-Challenge")
easy_data, easy_gt = process_subset("ARC-Easy")

# Combine both subsets
all_data = challenge_data + easy_data
all_gt = challenge_gt + easy_gt

print(f"\nCombined: {len(all_data)} total entries")
print(f"  ARC-Challenge: {len(challenge_data)}")
print(f"  ARC-Easy: {len(easy_data)}")

# Save dataset file
dataset_path = "dataset/arc_data.json"
with open(dataset_path, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)
print(f"\n✓ Created {len(all_data)} entries in {dataset_path}")

# Save ground truth file
gt_path = "dataset/arc_ground_truth.json"
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

    # Verify no numeric answer keys remain
    answer_letters = {e.get("answer") for e in gt}
    print(f"✓ Unique answer letters: {sorted(answer_letters)}")
    numeric_answers = {a for a in answer_letters if a and a.isdigit()}
    if numeric_answers:
        print(f"⚠ Warning: Found numeric answer keys: {numeric_answers}")
    else:
        print("✓ No numeric answer keys (all converted to letters)")

    # Check subset distribution
    challenge_count = sum(
        1 for e in gt if e.get("metadata", {}).get("subset") == "ARC-Challenge"
    )
    easy_count = sum(1 for e in gt if e.get("metadata", {}).get("subset") == "ARC-Easy")
    print(
        f"✓ Subset distribution: ARC-Challenge={challenge_count}, ARC-Easy={easy_count}"
    )

    print("\n✓ All files created and verified successfully!")
    print("\nNext steps:")
    print("1. Review dataset/arc_data.json to ensure prompts are formatted correctly")
    print("2. Review dataset/arc_ground_truth.json to ensure answers are correct")
    print("3. Proceed to Step 3: Router Inference Setup")

except Exception as e:
    print(f"✗ Verification failed: {e}")
    raise
