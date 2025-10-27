# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""
Enhanced Answer Extractor for LLM Evaluation

This module provides the three enhanced answer extraction methods:
1. extract_boxed_answer - Enhanced version with multiple pattern support
2. extract_math_answer - Enhanced mathematical answer extraction
3. extract_chess_move - Enhanced chess move extraction

These methods maintain compatibility with the existing evaluation pipeline while
providing enhanced extraction for models with low \boxed{} usage.
"""

import re
from typing import Optional


class EnhancedAnswerExtractor:
    """
    Enhanced answer extractor that handles multiple answer patterns
    while maintaining compatibility with existing evaluation pipeline.
    """

    def __init__(self):
        """
        Initialize the enhanced extractor.
        """

        # Define extraction patterns
        self.patterns = {
            # Standard \boxed{} format
            "boxed": r"\\boxed\{([^}]+)\}",
            "boxed_balanced": r"\\boxed\{",  # For balanced brace extraction
            # Multiple choice patterns
            "mcq_letter": r"\b([A-Z])\b(?!\w)",  # Standalone letters A-Z
            "mcq_option": r"(?:option|choice|select)[:\s]*([A-Z])",
            "mcq_letter_alt": r"(?:the\s+)?(?:answer|choice)\s*[:\-]\s*([A-Z])",
            "mcq_answer_colon": r"answer\s*:\s*([A-Z])",  # "Answer: C" pattern
            # Mathematical answer patterns
            "math_answer": r"(?:the\s+)?(?:final\s+)?answer\s*[:\-]\s*([^\n]+)",
            "math_result": r"(?:result|solution)\s*[:\-]\s*([^\n]+)",
            "math_number": r"(?:answer|result)\s*[:\-]\s*(\d+(?:\.\d+)?)",
            "math_is": r"(?:the\s+)?(?:answer|result)\s+is\s+([^\n]+)",
            # Code patterns for LiveCodeBench
            "python_code": r"```python\s*\n(.*?)\n```",
            "code_block": r"```(?:python)?\s*\n(.*?)\n```",
            "code_function": r"def\s+\w+\([^)]*\):\s*\n(.*?)(?=\n\w|\n```|\Z)",
            # General answer patterns
            "quoted": r'["\']([^"\']+)["\']',
            "parentheses": r"\(([^)]+)\)",
            "brackets": r"\[([^\]]+)\]",
            "colon_answer": r"(?:answer|result|solution)[:\s]+([^\n]+)",
            # Chess move patterns
            "chess_move": r"\b([a-h][1-8][a-h][1-8])\b",
            "chess_move_alt": r"(?:move|play)[:\s]*([a-h][1-8][a-h][1-8])",
        }

        # Cleanup patterns for answer normalization
        self.cleanup_patterns = [
            r"^(?:the\s+)?(?:answer|result|solution)\s*[:\-]\s*",
            r"^(?:option|choice)\s*[:\-]\s*",
            r"^(?:letter|choice)\s*[:\-]\s*",
            r"\s*[:\-]\s*(?:is\s+)?the\s+(?:answer|result|solution)\.?$",
            r"\s*[:\-]\s*(?:is\s+)?correct\.?$",
            r"^[:\-\s]+",
            r"[:\-\s]+$",
            r"\.$",  # Remove trailing period
            r'^["\']|["\']$',  # Remove surrounding quotes
            r"^(?:the\s+)?(?:final\s+)?answer\s+is\s+",  # Remove "the final answer is"
            r"^(?:the\s+)?answer\s+is\s+",  # Remove "the answer is"
            r"^(?:the\s+)?result\s+is\s+",  # Remove "the result is"
        ]

    def has_boxed_pattern(self, text: str) -> bool:
        """Check if text contains \boxed{} pattern"""
        if not text:
            return False
        return bool(re.search(r"\\boxed\{", text))

    def extract_boxed_answer(self, text: str, dataset: str = None) -> str:
        """
        Enhanced version of extract_boxed_answer that handles multiple patterns.

        This function automatically detects if \boxed{} pattern exists and uses
        appropriate extraction method.

        Args:
            text: The generated answer text
            model_name: Name of the model (for context)
            dataset: Dataset name (for context)

        Returns:
            Extracted answer string
        """
        if not text or not text.strip():
            return ""

        text = text.strip()

        # Check if text contains \boxed{} pattern
        if self.has_boxed_pattern(text):
            # Use standard \boxed{} extraction
            standard_result = self._extract_standard_boxed(text)
            if standard_result:
                return standard_result

        # If no \boxed{} pattern found, use enhanced extraction
        enhanced_result = self._extract_enhanced_answer(text, dataset)
        if enhanced_result:
            return enhanced_result

        # Fallback to original logic
        return self._extract_fallback_answer(text)

    def extract_math_answer(self, text: str) -> str:
        """
        Enhanced version of extract_math_answer that handles multiple patterns.

        This function automatically detects if \boxed{} pattern exists and uses
        appropriate extraction method.

        Args:
            text: The generated answer text
            model_name: Name of the model (for context)

        Returns:
            Extracted mathematical answer string
        """
        if not text or not text.strip():
            return ""

        text = text.strip()

        # Check if text contains \boxed{} pattern
        if self.has_boxed_pattern(text):
            # Use standard \boxed{} extraction
            standard_result = self._extract_standard_math_boxed(text)
            if standard_result:
                return standard_result

        # If no \boxed{} pattern found, use enhanced extraction
        enhanced_result = self._extract_enhanced_math_answer(text)
        if enhanced_result:
            return enhanced_result

        # Fallback to original logic
        return text.strip()

    def extract_chess_move(self, text: str) -> str:
        """
        Extract chess move from text, automatically detecting \boxed{} pattern.

        Args:
            text: The generated answer text
            model_name: Name of the model (for context)

        Returns:
            Extracted chess move string
        """
        if not text or not text.strip():
            return ""

        text = text.strip()

        # Check if text contains \boxed{} pattern
        if self.has_boxed_pattern(text):
            # Use standard \boxed{} extraction
            standard_result = self._extract_standard_boxed(text)
            if standard_result:
                return self._normalize_chess_move(standard_result)

        # If no \boxed{} pattern found, use enhanced extraction
        # Try chess move patterns
        matches = re.findall(self.patterns["chess_move"], text, re.IGNORECASE)
        if matches:
            return self._normalize_chess_move(matches[-1])

        matches = re.findall(self.patterns["chess_move_alt"], text, re.IGNORECASE)
        if matches:
            return self._normalize_chess_move(matches[-1])

        # Fallback to original logic
        return self._normalize_chess_move(self._extract_fallback_answer(text))

    def _extract_standard_boxed(self, text: str) -> Optional[str]:
        """Extract answer using standard \boxed{} format (original logic)"""
        text = text.replace("\\text{", "")

        # Look for \boxed{X} pattern
        pattern = r"\\boxed\{([^}]+)\}"
        matches = re.findall(pattern, text)

        if matches:
            return matches[0].strip()

        return None

    def _extract_standard_math_boxed(self, text: str) -> Optional[str]:
        """Extract mathematical answer using standard \boxed{} format (original logic)"""
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

        return None

    def _extract_enhanced_answer(self, text: str, dataset: str = None) -> Optional[str]:
        """Extract answer using enhanced patterns"""
        # Special handling for LiveCodeBench
        if dataset == "LiveCodeBench":
            return self._extract_code_answer(text)

        if dataset == "SuperGLUE-ClozeTest":
            return text[0].upper()

        # Try multiple choice extraction
        mcq_result = self._extract_mcq_answer(text)
        if mcq_result:
            return mcq_result

        # Try mathematical answer extraction
        math_result = self._extract_math_answer(text)
        if math_result:
            return math_result

        # Try general answer extraction
        general_result = self._extract_general_answer(text)
        if general_result:
            return general_result

        return None

    def _extract_enhanced_math_answer(self, text: str) -> Optional[str]:
        """Extract mathematical answer using enhanced patterns"""
        # Try mathematical answer patterns
        math_patterns = [
            self.patterns["math_answer"],
            self.patterns["math_result"],
            self.patterns["math_number"],
            self.patterns["math_is"],
        ]

        for pattern in math_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                answer = matches[0].strip()
                cleaned = self._clean_answer(answer)
                if cleaned:
                    return cleaned

        # If no pattern matches, try to extract numbers from the text
        # Look for numbers that might be answers
        number_patterns = [
            r"\b(\d+(?:\.\d+)?)\b",  # Simple numbers
            r"\b(\d+(?:\.\d+)?)\s*$",  # Numbers at end of text
        ]

        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Take the last number found (often the final answer)
                return matches[-1]

        return None

    def _extract_mcq_answer(self, text: str) -> Optional[str]:
        """Extract multiple choice answer"""
        # Try "Answer: X" pattern first (most specific)
        matches = re.findall(self.patterns["mcq_answer_colon"], text, re.IGNORECASE)
        if matches:
            return matches[0].upper()

        # Try "Option X" patterns
        matches = re.findall(self.patterns["mcq_option"], text, re.IGNORECASE)
        if matches:
            return matches[0].upper()

        # Try "The answer is X" patterns
        matches = re.findall(self.patterns["mcq_letter_alt"], text, re.IGNORECASE)
        if matches:
            return matches[0].upper()

        # Try standalone letters A-Z (least specific, do last)
        matches = re.findall(self.patterns["mcq_letter"], text)
        if matches:
            return matches[-1]  # Take the last letter found

        return None

    def _extract_math_answer(self, text: str) -> Optional[str]:
        """Extract mathematical answer using various patterns"""
        # Try mathematical answer patterns
        math_patterns = [
            self.patterns["math_answer"],
            self.patterns["math_result"],
            self.patterns["math_number"],
            self.patterns["math_is"],
        ]

        for pattern in math_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                answer = matches[0].strip()
                cleaned = self._clean_answer(answer)
                if cleaned:
                    return cleaned

        return None

    def _extract_code_answer(self, text: str) -> Optional[str]:
        """Extract code answer for LiveCodeBench"""
        # Try Python code blocks
        matches = re.findall(
            self.patterns["python_code"], text, re.DOTALL | re.IGNORECASE
        )
        if matches:
            return matches[-1].strip()  # Take the last code block

        # Try general code blocks
        matches = re.findall(
            self.patterns["code_block"], text, re.DOTALL | re.IGNORECASE
        )
        if matches:
            return matches[-1].strip()

        # Try function definitions
        matches = re.findall(
            self.patterns["code_function"], text, re.DOTALL | re.IGNORECASE
        )
        if matches:
            return matches[-1].strip()

        return None

    def _extract_general_answer(self, text: str) -> Optional[str]:
        """Extract general answer using various patterns"""
        # Try quoted answers
        matches = re.findall(self.patterns["quoted"], text)
        if matches:
            return matches[-1].strip()

        # Try parentheses answers
        matches = re.findall(self.patterns["parentheses"], text)
        if matches:
            return matches[-1].strip()

        # Try bracket answers
        matches = re.findall(self.patterns["brackets"], text)
        if matches:
            return matches[-1].strip()

        # Try colon-separated answers
        matches = re.findall(self.patterns["colon_answer"], text, re.IGNORECASE)
        if matches:
            answer = matches[-1].strip()
            cleaned = self._clean_answer(answer)
            if cleaned:
                return cleaned

        return None

    def _extract_fallback_answer(self, text: str) -> str:
        """Fallback extraction using original logic"""
        # Look for standalone letters A-Z at the end or as the main content
        pattern = r"\b([A-Z])\b"
        matches = re.findall(pattern, text)

        if matches:
            # Return the last letter found (often the final answer)
            return matches[-1]

        # If still no match, return the original text stripped
        return text.strip()

    def _clean_answer(self, answer: str) -> str:
        """Clean extracted answer by removing common prefixes/suffixes"""
        if not answer:
            return answer

        cleaned = answer.strip()

        # Apply cleanup patterns
        for pattern in self.cleanup_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Remove extra whitespace
        cleaned = " ".join(cleaned.split())

        return cleaned.strip()

    def _normalize_chess_move(self, move: str) -> str:
        """Normalize chess move for comparison"""
        if not move:
            return ""

        # Convert to string and strip whitespace
        move = str(move).strip()

        # Convert to lowercase
        move = move.lower()

        # Remove common separators and spaces
        move = move.replace("-", "").replace(" ", "").replace("_", "")

        # Keep only letters and numbers, remove everything else
        move = re.sub(r"[^a-z0-9]", "", move)

        return move


# Global instance for easy import
enhanced_extractor = EnhancedAnswerExtractor()


# Drop-in replacement functions that maintain compatibility
def extract_boxed_answer(text: str, dataset: str = None) -> str:
    """
    Enhanced extract_boxed_answer function that automatically uses improved extraction
    for models with low \boxed{} usage.

    This function maintains full compatibility with the original extract_boxed_answer
    while providing enhanced extraction capabilities.
    """
    return enhanced_extractor.extract_boxed_answer(text, dataset)


def extract_math_answer(text: str) -> str:
    """
    Enhanced extract_math_answer function that automatically uses improved extraction
    for models with low \boxed{} usage.

    This function maintains full compatibility with the original extract_math_answer
    while providing enhanced extraction capabilities.
    """
    return enhanced_extractor.extract_math_answer(text)


def extract_chess_move(text: str) -> str:
    """
    Enhanced chess move extraction function.
    """
    return enhanced_extractor.extract_chess_move(text)
