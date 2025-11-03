# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

import re
import regex
from math import isclose

from typing import Any, Optional, List

from latex2sympy2 import latex2sympy
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


def choice_answer_clean(pred: str) -> str:
    """Helper function for standardizing multiple choice answers"""
    cleaned = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    matches: List[str] = re.findall(r"\b(A|B|C|D|E)\b", cleaned.upper())
    if matches:
        result = matches[-1]
    else:
        result = cleaned.strip().strip(".")
    result = result.rstrip(".").rstrip("/")
    return result


def parse_digits(num: Any) -> Optional[float]:
    normalized = regex.sub(",", "", str(num))
    try:
        return float(normalized)
    except Exception:
        if normalized.endswith("%"):
            normalized = normalized[:-1]
            if normalized.endswith("\\"):
                normalized = normalized[:-1]
            try:
                return float(normalized) / 100
            except Exception:
                pass
    return None


def is_digit(num: Any) -> bool:
    # paired with parse_digits
    return parse_digits(num) is not None


def numeric_equal(prediction: float, reference: float) -> bool:
    return isclose(reference, prediction, rel_tol=1e-4)


def str_to_pmatrix(input_str: str) -> str:
    stripped = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", stripped)
    pmatrix_list: List[str] = []

    for m in matrix_str:
        content = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + content.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


def symbolic_equal(a: Any, b: Any) -> bool:
    def _parse(s: Any) -> Any:
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except Exception:
                try:
                    return f(s)
                except Exception:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except Exception:
        pass

    return False
