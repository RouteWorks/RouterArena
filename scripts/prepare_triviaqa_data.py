# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Prepare TriviaQA dataset for RouterArena pipeline.

This script:
1. Loads TriviaQA from HuggingFace (mandarjoshi/trivia_qa)
   Subsets: rc, rc.nocontext, rc.web, rc.web.nocontext,
            rc.wikipedia, rc.wikipedia.nocontext,
            unfiltered, unfiltered.nocontext
2. Formats prompts according to RouterArena Open-QA requirements
3. Creates dataset/triviaqa_data.json (for router inference)
4. Creates dataset/triviaqa_ground_truth.json (for evaluation), saving aliases

How to run:
    uv run python scripts/prepare_triviaqa_data.py
"""

from datasets import load_dataset
import json
import os

# Ensure dataset directory exists
os.makedirs("dataset", exist_ok=True)

# Define prompt template (must match eval config!)
PROMPT_TEMPLATE = """Please read the following question and provide the correct answer.

Question: {question}

Provide the correct answer in \\boxed{{X}}, where X is the correct and the most common answer to the question. Keep the explanation or feedback within 3 sentences."""

# All TriviaQA subsets
SUBSETS = [
    "rc",
    "rc.nocontext",
    "rc.web",
    "rc.web.nocontext",
    "rc.wikipedia",
    "rc.wikipedia.nocontext",
    "unfiltered",
    "unfiltered.nocontext",
]


def process_subset(subset_name: str):
    """
    Load and process a single TriviaQA subset.

    Combines all available splits with valid answers.

    Args:
        subset_name: e.g. "rc.web"

    Returns:
        Tuple of (formatted_data, ground_truth) lists
    """
    print(f"\nLoading {subset_name} from HuggingFace...")
    ds = load_dataset("mandarjoshi/trivia_qa", name=subset_name)

    formatted_data = []
    ground_truth = []
    entry_index = 0
    skipped_no_label = 0

    index_prefix = f"TriviaQA-{subset_name}"

    for split_name in ds.keys():
        split_ds = ds[split_name]
        print(f"  Processing {split_name} split: {len(split_ds)} entries")

        for item in split_ds:
            question = item.get("question", "")
            answer_dict = item.get("answer", {})
            aliases = answer_dict.get("aliases", [])

            # Skip entries with no valid aliases (e.g. test split)
            if not aliases or len(aliases) == 0:
                skipped_no_label += 1
                entry_index += 1
                continue

            # Ensure all aliases are strings
            aliases = [str(a) for a in aliases]

            # Build the complete prompt
            prompt = PROMPT_TEMPLATE.format(question=question)

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
                    "answer": aliases,  # Save list of aliases
                    "metadata": {
                        "subset": subset_name,
                        "split": split_name,
                        "question_id": item.get("question_id", ""),
                        "question_source": item.get("question_source", ""),
                    },
                }
            )

            entry_index += 1

    if skipped_no_label > 0:
        print(f"  Skipped {skipped_no_label} entries with no answer (test split)")
    print(f"  Total {subset_name} entries with answers: {len(formatted_data)}")
    return formatted_data, ground_truth


# Process all subsets
print("=" * 60)
print("Preparing TriviaQA dataset for RouterArena")
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
dataset_path = "dataset/triviaqa_data.json"
with open(dataset_path, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)
print(f"\n✓ Created {len(all_data)} entries in {dataset_path}")

# Save ground truth file
gt_path = "dataset/triviaqa_ground_truth.json"
with open(gt_path, "w", encoding="utf-8") as f:
    json.dump(all_gt, f, indent=2, ensure_ascii=False)
print(f"✓ Created {len(all_gt)} ground truth entries in {gt_path}")

# Verify files
print("\n[Verification] Checking files...")
try:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✓ Dataset file: {len(data)} entries")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    print(f"✓ Ground truth file: {len(gt)} entries")

    if len(gt) > 0:
        print(f"  First answer aliases format: {gt[0].get('answer')}")

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
