# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "scripts/submit_kilo_auto_efficient.py"
SPEC = importlib.util.spec_from_file_location("kilo_submission", SOURCE)
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


if __name__ == "__main__":
    unittest.main()
