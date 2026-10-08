# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

import argparse
import contextlib
import importlib.util
import io
import json
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "scripts/submit_kilo_auto_efficient.py"
SPEC = importlib.util.spec_from_file_location("kilo_submission", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load submission driver: {SOURCE}")
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class CapacityRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.clock = patch.object(DRIVER.time, "monotonic", lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.adaptive = DRIVER.AdaptiveConcurrency(16, 8400, "z-ai/glm-5.3-flash")

    def window(self, count=20, fallbacks=0, errors=0):
        for index in range(count):
            self.adaptive.record(
                {
                    "prediction": "z-ai/glm-5.3-flash"
                    if index < fallbacks
                    else "xiaomi/mimo-v2.6-pro",
                    "generated_result": {
                        "success": index >= errors,
                        **(
                            {"generation_error": {"code": 503}}
                            if index < errors
                            else {}
                        ),
                    },
                }
            )
        self.now += 45
        self.adaptive.evaluate(count)

    def test_two_healthy_windows_resume_after_classifier_fallbacks(self):
        self.window(100, fallbacks=3)
        self.assertEqual(self.adaptive.concurrency, 8)
        self.window()
        self.assertEqual(self.adaptive.concurrency, 8)
        self.assertFalse(self.adaptive.ramping)
        self.window()
        self.assertEqual(self.adaptive.concurrency, 16)
        self.assertTrue(self.adaptive.ramping)
        self.assertIsNone(self.adaptive.limit_reason)

    def test_provider_failure_resets_the_healthy_window_streak(self):
        self.window(100, fallbacks=3)
        self.window()
        self.window(errors=1)
        self.assertEqual(self.adaptive.concurrency, 4)
        self.window()
        self.assertEqual(self.adaptive.concurrency, 4)
        self.assertFalse(self.adaptive.ramping)
        self.window()
        self.assertEqual(self.adaptive.concurrency, 8)
        self.assertTrue(self.adaptive.ramping)

    def test_healthy_windows_do_not_reopen_a_throughput_plateau(self):
        self.window()
        self.window()
        self.window()
        self.assertEqual(self.adaptive.limit_reason, "throughput_plateau")
        self.window()
        self.window()
        self.assertEqual(self.adaptive.concurrency, 32)
        self.assertFalse(self.adaptive.ramping)


class ProviderRetryPolicyTest(unittest.TestCase):
    def test_numeric_and_string_provider_codes_are_retryable(self):
        for code in (429, "429", 500, "500", 502, "502", 503, "503", 504, "504"):
            with self.subTest(code=code):
                result = {
                    "generated_result": {
                        "success": False,
                        "generation_error": {"code": code},
                    }
                }
                self.assertTrue(DRIVER.provider_error_retryable(result))

    def test_successful_answers_remain_locked_despite_error_metadata(self):
        result = {
            "generated_result": {
                "success": True,
                "generation_error": {"code": "503"},
            }
        }
        self.assertFalse(DRIVER.provider_error_retryable(result))

    def test_empty_answers_missing_usage_and_refusals_do_not_qualify(self):
        for error in (
            {"type": "empty_answer"},
            {"type": "missing_usage"},
            {"type": "content_filter"},
            {"code": "400"},
            {"code": "503 unrelated text"},
        ):
            with self.subTest(error=error):
                result = {
                    "generated_result": {
                        "success": False,
                        "generation_error": error,
                    }
                }
                self.assertFalse(DRIVER.provider_error_retryable(result))

    def test_incomplete_retry_requires_the_exact_approved_index(self):
        for failure_type in ("empty_answer", "missing_usage"):
            with self.subTest(failure_type=failure_type):
                result = {
                    "global_index": "approved-row",
                    "generated_result": {
                        "success": False,
                        "generation_error": {"type": failure_type},
                    },
                }
                self.assertFalse(DRIVER.retryable_result(result, set()))
                self.assertFalse(DRIVER.retryable_result(result, {"another-row"}))
                self.assertTrue(DRIVER.retryable_result(result, {"approved-row"}))

    def test_approval_does_not_unlock_a_success_or_a_different_failure(self):
        for generation in (
            {"success": True, "generation_error": {"type": "missing_usage"}},
            {"success": False, "generation_error": {"type": "content_filter"}},
            {"success": False, "generation_error": {"code": 400}},
        ):
            with self.subTest(generation=generation):
                result = {
                    "global_index": "approved-row",
                    "generated_result": generation,
                }
                self.assertFalse(DRIVER.retryable_result(result, {"approved-row"}))


class ApprovedReplayTest(unittest.TestCase):
    def test_approved_retries_preserve_successes_and_require_approval_on_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            (root / "router_inference/config").mkdir(parents=True)
            inputs = [
                {"global index": index, "prompt_formatted": "unchanged " + index}
                for index in ("stable", "empty", "usage")
            ]
            (root / "dataset/router_data.json").write_text(json.dumps(inputs))
            (root / "router_inference/config/kilo-auto-efficient.json").write_text(
                json.dumps({"pipeline_params": {"models": ["xiaomi/mimo-v2.6-pro"]}})
            )
            calls = []

            def fetch(endpoint, key, row, timeout, split):
                index = row["global index"]
                calls.append(index)
                initial = calls.count(index) == 1
                answer = "" if index == "empty" and initial else "captured answer"
                tokens = 0 if index == "usage" and initial else 3
                return {
                    "global_index": index,
                    "completion_duration_seconds": 1,
                    "response_body": json.dumps(
                        {
                            "model": "xiaomi/mimo-v2.6-pro",
                            "choices": [{"message": {"content": answer}}],
                            "usage": {
                                "prompt_tokens": 2,
                                "completion_tokens": tokens,
                                "total_tokens": 2 + tokens,
                            },
                        }
                    ),
                }

            args = argparse.Namespace(
                split="full",
                concurrency=1,
                timeout=30,
                limit=None,
                retry_provider_errors=False,
                order_seed=None,
                adaptive_concurrency=False,
                retry_approved_indices=[],
            )
            with (
                patch.object(DRIVER, "ROOT", root),
                patch.dict(DRIVER.SPLITS, {"full": ("router_data.json", 3)}),
                patch.object(DRIVER, "fetch", fetch),
                patch.object(DRIVER.signal, "signal"),
                patch.dict(
                    os.environ,
                    {
                        "KILO_API_KEY": "test-only",
                        "KILO_API_URL": "https://example.test/chat/completions",
                        "KILO_RUN_ID": "approved-replay-test",
                    },
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(DRIVER.run(args), 0)
                output = root / "router_inference/predictions/kilo-auto-efficient.json"
                original = json.loads(output.read_text())[0]
                args.retry_provider_errors = True
                args.retry_approved_indices = ["empty", "usage"]
                self.assertEqual(DRIVER.run(args), 0)
                completed = json.loads(output.read_text())
                self.assertEqual(completed[0], original)
                self.assertEqual(calls, ["stable", "empty", "usage", "empty", "usage"])
                self.assertTrue(
                    all(row["generated_result"]["success"] for row in completed)
                )
                args.retry_approved_indices = []
                with self.assertRaisesRegex(ValueError, "unapproved failed result"):
                    DRIVER.run(args)
                args.retry_approved_indices = ["empty", "usage"]
                self.assertEqual(DRIVER.run(args), 0)
                self.assertEqual(calls.count("stable"), 1)


if __name__ == "__main__":
    unittest.main()
