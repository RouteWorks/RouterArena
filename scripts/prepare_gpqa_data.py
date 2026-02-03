# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Prepare GPQA dataset for RouterArena pipeline.

This script:
1. Loads GPQA dataset from HuggingFace
2. Formats prompts according to RouterArena requirements
3. Creates dataset/gpqa_data.json (for router inference)
4. Creates dataset/gpqa_ground_truth.json (for evaluation)

How to run:
    uv run python scripts/prepare_gpqa_data.py

Prerequisites:
    - Install packages: uv sync (or pip install datasets)
    - (Optional) If authentication needed: huggingface-cli login
"""

from datasets import load_dataset, DatasetDict
import json
import os
import random

# Ensure dataset directory exists
os.makedirs("dataset", exist_ok=True)

# Load the dataset. If authentication is needed, ensure you are logged in.
print("Loading GPQA dataset from HuggingFace...")
raw_gpqa = load_dataset("Idavidrein/gpqa", "gpqa_diamond")

# Extract the actual dataset from DatasetDict if needed
if isinstance(raw_gpqa, DatasetDict):
    split_names = list(raw_gpqa.keys())
    print(f"Found splits: {split_names}")
    # Get the first split (usually 'train')
    split_key = "train" if "train" in raw_gpqa else split_names[0]
    print(f"Using split: {split_key}")
    gpqa_dataset = raw_gpqa[split_key]
else:
    # If it's already a single Dataset, use it directly
    gpqa_dataset = raw_gpqa

print(f"Loaded {len(gpqa_dataset)} GPQA entries")
print(f"Dataset features: {gpqa_dataset.features}")

# Inspect first entry to understand structure
if len(gpqa_dataset) > 0:
    print("\nFirst entry sample:")
    print(gpqa_dataset[0])
    print()

# Define prompt template (must match eval config!)
PROMPT_TEMPLATE = """Please read the following multiple-choice questions and provide the most likely correct answer based on the options given.

Context: {context}

Question: {question}

Options:
{options}

Provide the correct letter choice in \\boxed{{X}}, where X is the correct letter choice. Keep the explanation or feedback within 3 sentences."""

# Step 2.2: Create dataset file with formatted prompts
print("\n[Step 2.2] Formatting prompts and creating dataset file...")
formatted_data = []
# Store shuffled options and answer letters for use in ground truth generation
shuffled_data = {}  # {index: (shuffled_options, answer_letter)}

for i, item in enumerate(gpqa_dataset):
    # Extract fields (adjust field names based on actual dataset structure)
    question = item.get("question", item.get("Question", ""))
    # GPQA dataset has separate fields for correct and incorrect answers
    # Construct options list from these fields
    correct_answer = item.get(
        "Correct Answer",
        item.get("correct_answer", item.get("answer", item.get("Answer", ""))),
    )
    incorrect_1 = item.get("Incorrect Answer 1", "")
    incorrect_2 = item.get("Incorrect Answer 2", "")
    incorrect_3 = item.get("Incorrect Answer 3", "")

    # Build options list with all answers
    all_options = [
        opt for opt in [correct_answer, incorrect_1, incorrect_2, incorrect_3] if opt
    ]

    # If we have a pre-formatted options list, use that instead
    if item.get("options") or item.get("Options"):
        all_options = item.get("options", item.get("Options", []))
        # Still need to find correct answer in the list
        if not correct_answer:
            correct_answer = item.get(
                "Correct Answer",
                item.get("correct_answer", item.get("answer", item.get("Answer", ""))),
            )

    # Shuffle options using deterministic seed based on index for reproducibility
    # This ensures the same question always gets the same shuffle
    random.seed(i)
    shuffled_options = all_options.copy()
    random.shuffle(shuffled_options)

    # Find the index of the correct answer in the shuffled list
    answer_index = -1
    if correct_answer:
        try:
            answer_index = shuffled_options.index(correct_answer)
        except ValueError:
            # Try case-insensitive matching
            for idx, opt in enumerate(shuffled_options):
                if (
                    opt
                    and correct_answer
                    and opt.strip().lower() == correct_answer.strip().lower()
                ):
                    answer_index = idx
                    break

    # Convert index to letter (A, B, C, D, etc.)
    if answer_index >= 0:
        answer_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[answer_index]
    else:
        answer_letter = "A"  # Default fallback
        print(
            f"Warning: Could not find correct answer in options for GPQA_{i}, defaulting to A"
        )

    # Store shuffled data for ground truth generation
    shuffled_data[i] = (shuffled_options, answer_letter)

    context = item.get("context", item.get("Context", ""))

    # Format shuffled options as "A. option1\nB. option2\n..."
    options_str = ""
    for j, opt in enumerate(shuffled_options):
        letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[j]
        options_str += f"{letter}. {opt}\n"

    # Build the complete prompt
    prompt = PROMPT_TEMPLATE.format(
        context=context or "None", question=question, options=options_str.strip()
    )

    # Create the dataset entry
    formatted_data.append(
        {
            "prompt_formatted": prompt,
            "global index": f"GPQA_{i}",  # CRITICAL: Prefix must match dataset name
        }
    )

# Save dataset file
dataset_path = "dataset/gpqa_data.json"
with open(dataset_path, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, indent=2, ensure_ascii=False)

print(f"✓ Created {len(formatted_data)} GPQA entries in {dataset_path}")

# Step 2.3: Create ground truth file
print("\n[Step 2.3] Creating ground truth file...")
ground_truth = []
for i, item in enumerate(gpqa_dataset):
    # Extract fields (same as above)
    question = item.get("question", item.get("Question", ""))
    context = item.get("context", item.get("Context", ""))

    # Use the same shuffled options and answer letter from step 2.2
    if i in shuffled_data:
        shuffled_options, answer_letter = shuffled_data[i]
    else:
        # Fallback: regenerate shuffle if somehow missing (shouldn't happen)
        correct_answer = item.get(
            "Correct Answer",
            item.get("correct_answer", item.get("answer", item.get("Answer", ""))),
        )
        incorrect_1 = item.get("Incorrect Answer 1", "")
        incorrect_2 = item.get("Incorrect Answer 2", "")
        incorrect_3 = item.get("Incorrect Answer 3", "")
        all_options = [
            opt
            for opt in [correct_answer, incorrect_1, incorrect_2, incorrect_3]
            if opt
        ]

        if item.get("options") or item.get("Options"):
            all_options = item.get("options", item.get("Options", []))

        random.seed(i)
        shuffled_options = all_options.copy()
        random.shuffle(shuffled_options)

        answer_index = -1
        if correct_answer:
            try:
                answer_index = shuffled_options.index(correct_answer)
            except ValueError:
                for idx, opt in enumerate(shuffled_options):
                    if (
                        opt
                        and correct_answer
                        and opt.strip().lower() == correct_answer.strip().lower()
                    ):
                        answer_index = idx
                        break

        if answer_index >= 0:
            answer_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[answer_index]
        else:
            answer_letter = "A"
            print(
                f"Warning: Could not find correct answer in options for GPQA_{i}, defaulting to A"
            )

    ground_truth.append(
        {
            "global_index": f"GPQA_{i}",  # MUST match dataset file
            "question": question,
            "answer": answer_letter,  # Store as letter (A, B, C, D)
            "options": shuffled_options,  # Use shuffled options to match the prompt
            "context": context or "",
            "metadata": item.get("metadata", {}),
        }
    )

# Save ground truth file
gt_path = "dataset/gpqa_ground_truth.json"
with open(gt_path, "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, indent=2, ensure_ascii=False)

print(f"✓ Created {len(ground_truth)} GPQA ground truth entries in {gt_path}")

# Step 2.4: Verify files
print("\n[Step 2.4] Verifying files...")
try:
    # Check dataset file structure
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✓ Dataset file: {len(data)} entries")
    print(f"  First entry keys: {list(data[0].keys())}")
    print(f"  First global_index: {data[0].get('global index')}")

    # Check ground truth file structure
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
                f"⚠ Warning: {len(missing_in_data)} indices in ground truth not in dataset"
            )
        if missing_in_gt:
            print(
                f"⚠ Warning: {len(missing_in_gt)} indices in dataset not in ground truth"
            )

    print("\n✓ All files created and verified successfully!")
    print("\nNext steps:")
    print("1. Review dataset/gpqa_data.json to ensure prompts are formatted correctly")
    print("2. Review dataset/gpqa_ground_truth.json to ensure answers are correct")
    print("3. Proceed to Step 3: Router Inference Setup")

except Exception as e:
    print(f"✗ Verification failed: {e}")
    raise
