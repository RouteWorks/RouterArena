# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from datasets import load_from_disk


def escape_format_braces(text):
    """
    Escape curly braces in input text to prevent them from being interpreted
    as format variables when using str.format().

    Args:
        text (str): Input text that may contain curly braces

    Returns:
        str: Text with curly braces properly escaped for format strings

    Example:
        escape_format_braces("Find {x-2} and {k^3}") -> "Find {{x-2}} and {{k^3}}"
    """
    if not isinstance(text, str):
        return text

    # Replace single { with {{ and single } with }}
    # But preserve already escaped braces ({{ and }})
    result = ""
    i = 0
    while i < len(text):
        if text[i] == "{":
            if i + 1 < len(text) and text[i + 1] == "{":
                # Already escaped, keep as is
                result += "{{"
                i += 2
            else:
                # Single brace, escape it
                result += "{{"
                i += 1
        elif text[i] == "}":
            if i + 1 < len(text) and text[i + 1] == "}":
                # Already escaped, keep as is
                result += "}}"
                i += 2
            else:
                # Single brace, escape it
                result += "}}"
                i += 1
        else:
            result += text[i]
            i += 1

    return result


def safe_format_prompt(prompt_template, **kwargs):
    """
    Safely format a prompt template by escaping curly braces in the input data.

    Args:
        prompt_template (str): The prompt template with placeholders like {Context}, {Question}
        **kwargs: Keyword arguments for the template placeholders

    Returns:
        str: Formatted prompt with input data properly escaped
    """
    # Escape curly braces in all input values
    escaped_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            escaped_kwargs[key] = escape_format_braces(value)
        else:
            escaped_kwargs[key] = value

    return prompt_template.format(**escaped_kwargs)


def load_data(config):
    eval_config = config["eval_params"]
    dataset_name = eval_config["dataset"]
    if dataset_name == "LiveCodeBench":
        is_stdin_prompt = eval_config["is_stdin_prompt"]
        not_is_stdin_prompt = eval_config["not_is_stdin_prompt"]
        lcd_dataset = load_from_disk("./dataset/livecodebench").to_list()
    else:
        prompt = eval_config["prompt"]

    router_eval_bench = load_from_disk("./dataset/routerevalbench")
    router_eval_bench_df = pd.DataFrame(router_eval_bench)

    router_eval_bench_df["dataset_name_parsed"] = router_eval_bench_df[
        "Dataset name"
    ].apply(lambda x: x.split("_")[0])
    if "Ethics" in eval_config["dataset"]:
        router_eval_bench_df_partial = router_eval_bench_df[
            router_eval_bench_df["Dataset name"].str.contains(eval_config["dataset"])
        ]
    elif "MMLUPro" in eval_config["dataset"]:
        router_eval_bench_df_partial = router_eval_bench_df[
            router_eval_bench_df["Dataset name"].str.contains(eval_config["dataset"])
        ]
    elif "OpenTDB" in eval_config["dataset"]:
        router_eval_bench_df_partial = router_eval_bench_df[
            router_eval_bench_df["Dataset name"].str.contains(eval_config["dataset"])
        ]
    elif "QANTA" in eval_config["dataset"]:
        router_eval_bench_df_partial = router_eval_bench_df[
            router_eval_bench_df["Dataset name"].str.contains(eval_config["dataset"])
        ]
    elif "MMLU" in eval_config["dataset"]:
        router_eval_bench_df_partial = router_eval_bench_df[
            router_eval_bench_df["Dataset name"].str.contains("MMLU_")
        ]
    else:
        router_eval_bench_df_partial = router_eval_bench_df[
            router_eval_bench_df["Dataset name"] == eval_config["dataset"]
        ]

    assert len(router_eval_bench_df_partial) > 0, (
        f"No data found for dataset: {eval_config['dataset']}"
    )

    data = []
    all_data = []

    for index, (_, row) in enumerate(router_eval_bench_df_partial.iterrows()):
        options_str = ""
        for i, option in enumerate(row["Options"]):
            letter = chr(65 + i)  # 65 is ASCII for 'A'
            options_str += f"{letter}. {option}\n"
        if eval_config["dataset"] == "LiveCodeBench":
            idx = row["Global Index"].split("_")[-1]
            lcd_dataset_row = lcd_dataset[int(idx)]
            assert idx == lcd_dataset_row["_index"], (
                f"Index mismatch: {idx} != {lcd_dataset_row['_index']}"
            )
            prompt = (
                is_stdin_prompt if lcd_dataset_row["is_stdin"] else not_is_stdin_prompt
            )
            prompt_formatted = safe_format_prompt(prompt, Question=row["Question"])
        elif eval_config["dataset"] == "SuperGLUE-RC":
            prompt_formatted = safe_format_prompt(
                prompt, Question=row["Question"], Answer=row["Answer"]
            )
        elif eval_config["dataset"] == "SuperGLUE-Wic":
            prompt_formatted = safe_format_prompt(
                prompt, Question=row["Question"], Context=row["Context"]
            )
        elif not row["Options"]:
            prompt_formatted = safe_format_prompt(
                prompt,
                Context=row["Context"] if row["Context"] != "" else "None",
                Question=row["Question"],
            )
        else:
            prompt_formatted = safe_format_prompt(
                prompt,
                Context=row["Context"] if row["Context"] != "" else "None",
                Question=row["Question"],
                Options=options_str,
            )

        if len(prompt_formatted) > 10000:
            prompt_formatted = (
                prompt_formatted[:5000] + "..." + prompt_formatted[-5000:]
            )

        data.append(
            {
                "question": prompt_formatted,
                "global index": row["Global Index"],
            }
        )
        all_data.append(
            {
                "question": row["Question"],
                "global index": row["Global Index"],
                "context": row["Context"],
                "answer": row["Answer"],
                "options": row["Options"],
                "metadata": row["Metadata"],
            }
        )

    return all_data, data
