# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

import re
import string
import json
import copy

import jieba
from fuzzywuzzy import fuzz
import difflib

from collections import Counter
from rouge import Rouge
import regex

from metric_utils import (
    choice_answer_clean,
    is_digit,
    parse_digits,
    numeric_equal,
    str_to_pmatrix,
    symbolic_equal,
)

from enhanced_extractor import (
    extract_boxed_answer as enhanced_extract_boxed_answer,
    extract_math_answer as enhanced_extract_math_answer,
    extract_chess_move as enhanced_extract_chess_move,
)

ENHANCED_EXTRACTION_AVAILABLE = True


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r"段落(\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip("\n").split("\n")
    prediction = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction = line
            break
    return fuzz.ratio(prediction, ground_truth) / 100


def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if em_match_list != 0:
        if ground_truth in em_match_list:
            score = 1.0 / len(em_match_list)
        else:
            score = 0.0
    else:
        best_match = None
        highest_similarity = 0.0
        for string in all_classes:
            similarity = difflib.SequenceMatcher(None, string, prediction).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = string
        score = float(best_match == ground_truth)
    return score


def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    score = rouge_score(prediction, ground_truth)
    return score


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)


def mcq_exact_match(predictions, ground_truths, **kwargs):
    """
    Calculate exact match for MCQ by extracting answers from boxed format and comparing with ground truth.
    Handles both single prediction-answer pairs and lists.
    """
    if not isinstance(predictions, list):
        predictions = [predictions]
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]

    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) must match number of ground truths ({len(ground_truths)})"
        )

    correct = 0
    total = len(predictions)
    raw_results = []

    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        try:
            # Extract answer from boxed format (e.g., \boxed{A} -> A)
            # Use enhanced extraction if available, with model information
            if ENHANCED_EXTRACTION_AVAILABLE:
                extracted_pred = enhanced_extract_boxed_answer(pred, "MCQ")
            else:
                extracted_pred = extract_boxed_answer(pred)

            # Compare with ground truth
            is_correct = extracted_pred == gt
            correct += int(is_correct)

            # Create base result dict
            result_dict = {
                "prediction": pred,
                "extracted_answer": extracted_pred,
                "ground_truth": gt,
                "correct": is_correct,
            }

            raw_results.append(result_dict)

        except Exception as e:
            # Handle individual question errors gracefully
            print(f"    Warning: Error processing mcq_exact_match question {i}: {e}")
            print(f"    Ground truth: {gt}, Prediction: {pred[:100]}...")

            # Add error result but continue processing
            result_dict = {
                "prediction": pred,
                "extracted_answer": "ERROR",
                "ground_truth": gt,
                "correct": False,
                "error": str(e),
            }

            raw_results.append(result_dict)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, raw_results


def mcq_accuracy(predictions, ground_truths, **kwargs):
    """
    Calculate MCQ accuracy by extracting answers from boxed format and comparing with ground truth.
    Handles both single prediction-answer pairs and lists.
    """
    if not isinstance(predictions, list):
        predictions = [predictions]
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]

    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) must match number of ground truths ({len(ground_truths)})"
        )

    correct = 0
    total = len(predictions)
    raw_results = []

    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        try:
            # Handle ground truth parsing with error tolerance
            if gt in [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
                "L",
                "M",
                "N",
                "O",
                "P",
                "Q",
                "R",
                "S",
                "T",
                "U",
                "V",
                "W",
                "X",
                "Y",
                "Z",
            ]:
                gt_letter = gt
            elif gt in [
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "t",
                "u",
                "v",
                "w",
                "x",
                "y",
                "z",
            ]:
                gt_letter = gt.upper()
            else:
                # Try to convert to integer, handle float strings like "1.0"
                try:
                    gt_int = int(float(gt))  # Convert "1.0" -> 1.0 -> 1
                    if gt_int in range(0, 26):
                        gt_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[gt_int]
                    else:
                        raise ValueError(f"Ground truth index out of range: {gt}")
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid ground truth format: {gt}")

            # Extract answer from boxed format (e.g., \boxed{A} -> A)
            # Use enhanced extraction if available, with model information
            if ENHANCED_EXTRACTION_AVAILABLE:
                extracted_pred = enhanced_extract_boxed_answer(pred, "MCQ")
            else:
                extracted_pred = extract_boxed_answer(pred)

            # Compare with ground truth
            is_correct = extracted_pred == gt_letter
            correct += int(is_correct)

            # Create base result dict
            result_dict = {
                "prediction": pred,
                "extracted_answer": extracted_pred,
                "ground_truth": gt,
                "correct": is_correct,
            }

            raw_results.append(result_dict)

        except Exception as e:
            # Handle individual question errors gracefully
            print(f"    Warning: Error processing question {i}: {e}")
            print(f"    Ground truth: {gt}, Prediction: {pred[:100]}...")

            # Add error result but continue processing
            result_dict = {
                "prediction": pred,
                "extracted_answer": "ERROR",
                "ground_truth": gt,
                "correct": False,
                "error": str(e),
            }

            raw_results.append(result_dict)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, raw_results


def extract_boxed_answer(text):
    """
    Extract the answer from boxed format like \\boxed{A} or \boxed{A}
    """
    import re

    text = text.replace("\\text{", "")

    # Look for \boxed{X} pattern
    pattern = r"\\boxed\{([^}]+)\}"
    matches = re.findall(pattern, text)

    if matches:
        # Return the first match, stripped of whitespace
        return matches[0].strip()

    # If no boxed format found, try to extract single letter answers
    # Look for standalone letters A-Z at the end or as the main content
    pattern = r"\b([A-Z])\b"
    matches = re.findall(pattern, text)

    if matches:
        # Return the last letter found (often the final answer)
        return matches[-1]

    # If still no match, return the original text stripped
    return text.strip()


def math_metric(predictions, ground_truths, **kwargs):
    """
    Calculate math accuracy for MATH dataset by extracting answers from boxed format
    and using math_equal for comparison.
    Handles both single prediction-answer pairs and lists.
    """
    if not isinstance(predictions, list):
        predictions = [predictions]
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]

    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) must match number of ground truths ({len(ground_truths)})"
        )

    correct = 0
    total = len(predictions)
    raw_results = []

    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        try:
            # Extract answer from boxed format if present
            # Use enhanced extraction if available, with model information
            if ENHANCED_EXTRACTION_AVAILABLE:
                extracted_pred = enhanced_extract_math_answer(pred)
            else:
                extracted_pred = extract_math_answer(pred)

            if len(extracted_pred.split("=")) > 1:
                extracted_pred = extracted_pred.split("=")[-1].strip()

            # Use math_equal for comparison
            is_correct = math_equal(extracted_pred, gt)
            correct += int(is_correct)

            # Create base result dict
            result_dict = {
                "prediction": pred,
                "extracted_answer": extracted_pred,
                "ground_truth": gt,
                "correct": is_correct,
            }

            raw_results.append(result_dict)

        except Exception as e:
            # Handle individual question errors gracefully
            print(f"    Warning: Error processing math question {i}: {e}")
            print(f"    Ground truth: {gt}, Prediction: {pred[:100]}...")

            # Add error result but continue processing
            result_dict = {
                "prediction": pred,
                "extracted_answer": "ERROR",
                "ground_truth": gt,
                "correct": False,
                "error": str(e),
            }

            raw_results.append(result_dict)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, raw_results


def extract_math_answer(text):
    """
    Extract the mathematical answer from text, handling boxed format like \\boxed{answer}
    Handles nested braces properly for LaTeX expressions.
    """
    import re

    # Look for \boxed{ pattern and extract content with balanced braces
    boxed_pattern = r"\\boxed\{"

    # Find the start of \boxed{
    start_match = re.search(boxed_pattern, text)
    if start_match:
        start_pos = start_match.end() - 1  # Position of the opening brace

        # Find the matching closing brace
        brace_count = 0
        for i, char in enumerate(text[start_pos:], start_pos):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace
                    content = text[start_pos + 1 : i]
                    return content.strip()

    # If no boxed format found, return the original text stripped
    return text.strip()


def math_equal(
    prediction, reference, include_percentage: bool = True, is_close: bool = True
):
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal

    copied from https://github.com/NovaSky-AI/SkyThought/blob/main/skythought/evals/util/math_parsing_util.py#L393
    """
    if prediction is None or reference is None:
        return False
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(prediction) == reference
    ):
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            pred_val = parse_digits(prediction)
            ref_val = parse_digits(reference)
            if pred_val is not None and ref_val is not None:
                # number questions
                if include_percentage:
                    gt_result: list[float] = [ref_val / 100, ref_val, ref_val * 100]
                else:
                    gt_result = [ref_val]
                for item in gt_result:
                    try:
                        if is_close:
                            if numeric_equal(pred_val, item):
                                return True
                        else:
                            if item == pred_val:
                                return True
                    except Exception:
                        continue
                return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## pmatrix (amps)
    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = str_to_pmatrix(reference)

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i], ref_parts[i], include_percentage, is_close
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True
    if (
        (
            prediction.startswith("\\begin{pmatrix}")
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (
            prediction.endswith("\\end{pmatrix}")
            or prediction.endswith("\\end{bmatrix}")
        )
        and (
            reference.startswith("\\begin{pmatrix}")
            or reference.startswith("\\begin{bmatrix}")
        )
        and (
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):
        pred_lines = [
            line.strip()
            for line in prediction[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                            )
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1], reference, include_percentage, is_close
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction, reference.split("=")[1], include_percentage, is_close
        ):
            return True

    if symbolic_equal(prediction, reference):
        return True

    return False


def exact_match(predictions, ground_truths, **kwargs):
    """
    Calculate exact match accuracy for QANTA dataset by extracting answers from boxed format
    and normalizing strings for comparison.
    Handles both single prediction-answer pairs and lists.
    """
    if not isinstance(predictions, list):
        predictions = [predictions]
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]

    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) must match number of ground truths ({len(ground_truths)})"
        )

    correct = 0
    total = len(predictions)
    raw_results = []

    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        # Extract answer from boxed format (e.g., \boxed{Os_Lusíadas} -> Os_Lusíadas)
        # Use enhanced extraction if available, with model information
        if ENHANCED_EXTRACTION_AVAILABLE:
            extracted_pred = enhanced_extract_boxed_answer(pred, "QANTA")
        else:
            extracted_pred = extract_boxed_answer(pred)

        # Normalize both prediction and ground truth for comparison
        normalized_pred = normalize_qanta_answer(extracted_pred)
        normalized_gt = normalize_qanta_answer(gt)

        # Compare normalized strings
        is_correct = normalized_pred == normalized_gt
        correct += int(is_correct)

        # Create base result dict
        result_dict = {
            "prediction": pred,
            "extracted_answer": normalized_pred,
            "ground_truth": normalized_gt,
            "correct": is_correct,
        }

        raw_results.append(result_dict)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, raw_results


def normalize_qanta_answer(answer):
    """
    Normalize QANTA answer for exact match comparison.
    Handles underscores, case sensitivity, and trailing spaces.

    Args:
        answer: Raw answer string (e.g., "Os_Lusíadas", "Carl_Nielsen")

    Returns:
        Normalized answer string for comparison
    """
    if not answer:
        return ""

    # Convert to string and strip whitespace
    answer = str(answer).strip()

    # Convert underscores to spaces
    answer = answer.replace("_", " ")

    # Convert to lowercase for case-insensitive comparison
    answer = answer.lower()

    # Remove extra whitespace
    answer = " ".join(answer.split())

    return answer


def chess_accuracy(predictions, ground_truths, **kwargs):
    """
    Calculate chess accuracy for ChessInstruct dataset by extracting chess moves from boxed format
    and comparing with ground truth moves in standard coordinate notation.
    Handles both single prediction-answer pairs and lists.
    """
    if not isinstance(predictions, list):
        predictions = [predictions]
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]

    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) must match number of ground truths ({len(ground_truths)})"
        )

    correct = 0
    total = len(predictions)
    raw_results = []

    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        # Extract chess move from boxed format (e.g., \boxed{a2a3} -> a2a3)
        # Use enhanced extraction if available, with model information
        if ENHANCED_EXTRACTION_AVAILABLE:
            extracted_pred = enhanced_extract_chess_move(pred)
        else:
            extracted_pred = extract_boxed_answer(pred)

        # Normalize both prediction and ground truth for comparison
        normalized_pred = normalize_chess_move(extracted_pred)
        normalized_gt = normalize_chess_move(gt)

        # Compare normalized chess moves
        is_correct = normalized_pred == normalized_gt
        correct += int(is_correct)

        # Create base result dict
        result_dict = {
            "prediction": pred,
            "extracted_answer": normalized_pred,
            "ground_truth": normalized_gt,
            "correct": is_correct,
        }

        raw_results.append(result_dict)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, raw_results


def normalize_chess_move(move):
    """
    Normalize chess move for comparison.
    Handles different formats and standardizes to lowercase coordinate notation.
    Also handles multiple choice answers (a, b, c, etc.) and numeric answers (0, 1, 2, etc.).

    Args:
        move: Raw chess move string (e.g., "a2a3", "A2A3", "a2-a3", "a2 a3")
              or multiple choice answer (e.g., "a", "b", "c") or numeric answer (e.g., "0", "1", "2")

    Returns:
        Normalized string for comparison
    """
    if not move:
        return ""

    # Convert to string and strip whitespace
    move = str(move).strip()

    # Convert to lowercase
    move = move.lower()

    # Handle multiple choice answers (single letters a-z)
    if len(move) == 1 and move.isalpha():
        return move

    # Handle numeric answers (0, 1, 2, etc.) - convert to letters for consistency
    if move.isdigit():
        numeric_val = int(move)
        if 0 <= numeric_val <= 25:  # Support up to 26 options (a-z)
            return chr(ord("a") + numeric_val)
        return move

    # Handle chess moves - remove common separators and spaces
    move = move.replace("-", "").replace(" ", "").replace("_", "")

    # Remove any non-alphanumeric characters except for the move notation
    # Chess moves should be 4 characters: file-rank-file-rank (e.g., a2a3)
    import re

    # Keep only letters and numbers, remove everything else
    move = re.sub(r"[^a-z0-9]", "", move)

    return move


def code_accuracy(predictions, ground_truths, **kwargs):
    """
    Calculate code accuracy for livecodebench dataset by extracting code from predictions
    and running test cases to check correctness.
    Handles both single prediction-answer pairs and lists.
    """
    if not isinstance(predictions, list):
        predictions = [predictions]
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]

    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) must match number of ground truths ({len(ground_truths)})"
        )

    # Import livecodebench_util functions
    from livecodebench_util import has_code, check_correctness, post_process_code

    total_score = 0.0
    total = len(predictions)
    raw_results = []

    for i, (prediction, problem) in enumerate(zip(predictions, ground_truths)):
        score = 0.0
        processed_code = None

        # Extract code from prediction using has_code
        code_filter_result = has_code(prediction)

        if not code_filter_result:
            # No code found in prediction
            score = 0.0
        else:
            # Get the last code block (most likely the final answer)
            last_code = code_filter_result[-1]

            # Handle problem format - could be string or dict
            if isinstance(problem, str):
                try:
                    problem_to_check = json.loads(problem)
                except json.JSONDecodeError:
                    # If it's not JSON, treat as raw problem data
                    problem_to_check = {"test": problem, "is_stdin": False}
            else:
                problem_to_check = copy.deepcopy(problem)

            # Post-process the extracted code
            processed_code = post_process_code(last_code)

            # Check correctness using livecodebench_util
            score = check_correctness(
                problem=problem_to_check,
                completion=processed_code,
                timeout=6,
                is_extracted=not problem_to_check.get("is_stdin", False),
            )

        total_score += score

        # Create result dict
        result_dict = {
            "prediction": prediction,
            "extracted_code": code_filter_result[-1] if code_filter_result else None,
            "processed_code": processed_code if code_filter_result else None,
            "score": score,
            "has_code": bool(code_filter_result),
        }

        raw_results.append(result_dict)

    # Calculate final accuracy
    accuracy = total_score / total if total > 0 else 0.0
    return accuracy, raw_results


def meteor_score(predictions, ground_truths, **kwargs):
    """
    计算METEOR分数，用于评估文本生成质量。
    METEOR考虑了精确度、召回率、词序匹配和同义词匹配。

    Args:
        predictions: 预测文本列表或单个预测文本
        ground_truths: 真实文本列表或单个真实文本
        **kwargs: 其他参数

    Returns:
        float: METEOR分数 (0-1之间)
    """
    if not isinstance(predictions, list):
        predictions = [predictions]
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]

    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"预测数量 ({len(predictions)}) 必须与真实值数量 ({len(ground_truths)}) 匹配"
        )

    total_meteor = 0.0
    valid_count = 0
    raw_results = []

    for pred, gt in zip(predictions, ground_truths):
        if not pred or not gt:
            continue
        # Use enhanced extraction if available
        if ENHANCED_EXTRACTION_AVAILABLE:
            extracted_pred = enhanced_extract_boxed_answer(pred)
        else:
            extracted_pred = extract_boxed_answer(pred)
        # 计算单个预测的METEOR分数
        score = _calculate_single_meteor(extracted_pred, gt)
        total_meteor += score
        valid_count += 1

        result_dict = {
            "prediction": pred,
            "extracted_answer": extracted_pred,
            "score": score,
        }
        raw_results.append(result_dict)

    return total_meteor / valid_count if valid_count > 0 else 0.0, raw_results


def _calculate_single_meteor(prediction, ground_truth):
    """
    计算单个预测和真实值之间的METEOR分数。

    Args:
        prediction: 预测文本
        ground_truth: 真实文本

    Returns:
        float: METEOR分数
    """
    # 分词和预处理
    pred_tokens = _tokenize_text(prediction)
    gt_tokens = _tokenize_text(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    # 计算精确度和召回率
    precision, recall = _calculate_precision_recall(pred_tokens, gt_tokens)

    if precision == 0 and recall == 0:
        return 0.0

    # 计算F分数
    if precision + recall == 0:
        f_score = 0.0
    else:
        f_score = 2 * precision * recall / (precision + recall)

    # 计算词序惩罚
    penalty = _calculate_fragmentation_penalty(pred_tokens, gt_tokens)

    # 计算最终METEOR分数
    meteor_score = f_score * (1 - penalty)

    return meteor_score


def _tokenize_text(text):
    """
    对文本进行分词和预处理。

    Args:
        text: 输入文本

    Returns:
        list: 分词后的词列表
    """
    if not text:
        return []

    # 转换为小写
    text = text.lower()

    # 移除标点符号
    text = re.sub(r"[^\w\s]", " ", text)

    # 检查是否包含中文字符
    if re.search(r"[\u4e00-\u9fff]", text):
        # 中文文本：按字符分割（简单处理）
        tokens = [char for char in text if char.strip() and char != " "]
    else:
        # 英文文本：按空格分割
        tokens = text.split()

    # 移除空字符串
    tokens = [token for token in tokens if token.strip()]

    return tokens


def _calculate_precision_recall(pred_tokens, gt_tokens):
    """
    计算精确度和召回率。

    Args:
        pred_tokens: 预测文本的词列表
        gt_tokens: 真实文本的词列表

    Returns:
        tuple: (precision, recall)
    """
    # 计算匹配的词数
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    # 计算交集（匹配的词）
    matches = 0
    for word in pred_counter:
        if word in gt_counter:
            matches += min(pred_counter[word], gt_counter[word])

    # 计算精确度和召回率
    precision = matches / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = matches / len(gt_tokens) if len(gt_tokens) > 0 else 0.0

    return precision, recall


def _calculate_fragmentation_penalty(pred_tokens, gt_tokens):
    """
    计算词序惩罚，基于连续匹配的片段数量。

    Args:
        pred_tokens: 预测文本的词列表
        gt_tokens: 真实文本的词列表

    Returns:
        float: 惩罚值 (0-1之间)
    """
    # 找到匹配的词
    matched_words = set(pred_tokens) & set(gt_tokens)

    if not matched_words:
        return 1.0

    # 计算预测文本中的连续匹配片段数
    pred_chunks = _count_matching_chunks(pred_tokens, matched_words)

    # 计算真实文本中的连续匹配片段数
    gt_chunks = _count_matching_chunks(gt_tokens, matched_words)

    # 计算总匹配词数
    total_matches = len(matched_words)

    # 计算惩罚
    if total_matches == 0:
        return 1.0

    # 使用较小的片段数来计算惩罚
    min_chunks = min(pred_chunks, gt_chunks)

    # 改进的惩罚计算：当片段数接近总匹配数时，惩罚应该很小
    # 当片段数远小于总匹配数时，惩罚应该较大
    if min_chunks == total_matches:
        penalty = 0.0  # 完全连续匹配，无惩罚
    else:
        penalty = 0.5 * (min_chunks / total_matches) ** 2

    return penalty


def _count_matching_chunks(tokens, matched_words):
    """
    计算文本中连续匹配词的片段数量。

    Args:
        tokens: 词列表
        matched_words: 匹配的词集合

    Returns:
        int: 连续匹配片段的数量
    """
    chunks = 0
    in_chunk = False

    for token in tokens:
        if token in matched_words:
            if not in_chunk:
                chunks += 1
                in_chunk = True
        else:
            in_chunk = False

    return chunks


def _convert_superglue_answer(extracted_answer, ground_truth):
    """
    Convert extracted answer to match ground truth format for SuperGLUE datasets.

    Args:
        extracted_answer: The extracted answer (usually a letter like "A", "B", etc.)
        ground_truth: The ground truth value (e.g., "0.0", "1.0", "Yes", "No")

    Returns:
        Converted answer that matches the ground truth format
    """
    # If ground truth is numeric (0.0, 1.0), convert letters to numbers
    if ground_truth in ["0.0", "1.0", "0", "1"]:
        # Common mappings for numeric SuperGLUE tasks
        letter_to_number = {
            "A": "0.0",
            "0": "0.0",
            "FALSE": "0.0",
            "False": "0.0",
            "false": "0.0",
            "B": "1.0",
            "1": "1.0",
            "TRUE": "1.0",
            "True": "1.0",
            "true": "1.0",
        }
        return letter_to_number.get(extracted_answer.upper(), extracted_answer)

    # If ground truth is Yes/No, convert letters to Yes/No
    elif ground_truth in ["Yes", "No", "yes", "no", "YES", "NO"]:
        # Common mappings for Yes/No SuperGLUE tasks
        letter_to_yesno = {
            "A": "Yes",
            "X": "Yes",
            "1": "Yes",
            "TRUE": "Yes",
            "True": "Yes",
            "true": "Yes",
            "B": "No",
            "Y": "No",
            "0": "No",
            "FALSE": "No",
            "False": "No",
            "false": "No",
        }
        return letter_to_yesno.get(extracted_answer.upper(), extracted_answer)

    # If no specific mapping found, return original extracted answer
    return extracted_answer


def superglue_exact_match(predictions, ground_truths, **kwargs):
    """
    Calculate exact match accuracy for SuperGLUE-Entailment dataset by extracting answers from boxed format
    and normalizing strings for comparison.
    Handles both single prediction-answer pairs and lists.
    """
    if not isinstance(predictions, list):
        predictions = [predictions]
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]

    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) must match number of ground truths ({len(ground_truths)})"
        )

    total_score = 0.0
    total = len(predictions)
    raw_results = []

    for i, (prediction, ground_truth) in enumerate(zip(predictions, ground_truths)):
        if not prediction or not ground_truth:
            continue

        # Extract answer from boxed format
        # Use enhanced extraction if available, with model information
        if ENHANCED_EXTRACTION_AVAILABLE:
            extracted_answer = enhanced_extract_boxed_answer(prediction, "SuperGLUE")
        else:
            extracted_answer = extract_boxed_answer(prediction)

        # Convert extracted answer to match ground truth format
        converted_answer = _convert_superglue_answer(extracted_answer, ground_truth)

        # Calculate exact match
        is_correct = converted_answer == ground_truth
        total_score += is_correct

        # Create result dict
        raw_results.append(
            {
                "prediction": prediction,
                "extracted_answer": extracted_answer,  # Store original extracted answer, not normalized
                "ground_truth": ground_truth,  # Store original ground truth, not normalized
                "correct": is_correct,
            }
        )

    # Calculate final accuracy
    accuracy = total_score / total if total > 0 else 0.0
    return accuracy, raw_results


def superglue_clozetest(predictions, ground_truths, **kwargs):
    """
    SuperGLUE ClozeTest evaluation metric using MCQ accuracy.
    Treats ClozeTest as multiple choice questions and calculates accuracy.

    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers (can be multiple answers per prediction)

    Returns:
        Accuracy score and detailed results
    """
    if not isinstance(predictions, list):
        predictions = [predictions]
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]

    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) must match number of ground truths ({len(ground_truths)})"
        )

    correct = 0
    total = len(predictions)
    raw_results = []

    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        try:
            # Handle different ground truth formats for SuperGLUE-ClozeTest
            if isinstance(gt, str):
                # Check if it's already a letter
                if gt in [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                    "F",
                    "G",
                    "H",
                    "I",
                    "J",
                    "K",
                    "L",
                    "M",
                    "N",
                    "O",
                    "P",
                    "Q",
                    "R",
                    "S",
                    "T",
                    "U",
                    "V",
                    "W",
                    "X",
                    "Y",
                    "Z",
                ]:
                    gt_letter = gt
                elif gt in [
                    "a",
                    "b",
                    "c",
                    "d",
                    "e",
                    "f",
                    "g",
                    "h",
                    "i",
                    "j",
                    "k",
                    "l",
                    "m",
                    "n",
                    "o",
                    "p",
                    "q",
                    "r",
                    "s",
                    "t",
                    "u",
                    "v",
                    "w",
                    "x",
                    "y",
                    "z",
                ]:
                    gt_letter = gt.upper()
                else:
                    # Try to convert to integer, handle float strings like "1.0"
                    try:
                        gt_int = int(float(gt))  # Convert "1.0" -> 1.0 -> 1
                        if gt_int in range(0, 26):
                            gt_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[gt_int]
                        else:
                            raise ValueError(f"Ground truth index out of range: {gt}")
                    except (ValueError, TypeError):
                        # If it's not a number, it might be the actual answer text
                        # In this case, we'll compare the extracted prediction directly with the ground truth text
                        gt_letter = gt  # Use the text as-is for comparison
            elif isinstance(gt, int):
                # Handle integer indices
                if gt in range(0, 26):
                    gt_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[gt]
                else:
                    raise ValueError(f"Ground truth index out of range: {gt}")
            else:
                raise ValueError(f"Invalid ground truth format: {gt}")

            # Extract answer from boxed format (e.g., \boxed{A} -> A)
            # Use enhanced extraction if available, with model information
            if ENHANCED_EXTRACTION_AVAILABLE:
                extracted_pred = enhanced_extract_boxed_answer(
                    pred, "SuperGLUE-ClozeTest"
                )
            else:
                extracted_pred = extract_boxed_answer(pred)

            # Compare with ground truth
            is_correct = extracted_pred == gt_letter
            correct += int(is_correct)

            # Create base result dict
            result_dict = {
                "prediction": pred,
                "extracted_answer": extracted_pred,
                "ground_truth": gt,
                "correct": is_correct,
            }

            raw_results.append(result_dict)

        except Exception as e:
            # Handle individual question errors gracefully
            print(
                f"    Warning: Error processing superglue_clozetest question {i}: {e}"
            )
            print(f"    Ground truth: {gt}, Prediction: {pred[:100]}...")

            # Add error result but continue processing
            result_dict = {
                "prediction": pred,
                "extracted_answer": "ERROR",
                "ground_truth": gt,
                "correct": False,
                "error": str(e),
            }

            raw_results.append(result_dict)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, raw_results
