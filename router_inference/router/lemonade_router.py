# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Lemonade local-first tri-ensemble router (v5.2).

Two generator models served on-device by Lemonade
(https://github.com/lemonade-sdk/lemonade) — DeepSeek-V4-Flash (IQ2XXS, ds4
recipe) and Qwen3.8-27B (UD-Q4, llamacpp recipe) — vote alongside one cloud
model (gemini-3-flash-preview) on every query, with a third on-device model,
LFM2.5-Embedding-350M (F16 GGUF, llamacpp recipe), acting as a semantic
referee on degraded free-answer comparisons. ~80% of queries are answered
entirely on-device.

Per-query policy (all thresholds fixed a priori; calibrated only on external
public benchmarks and label-free agreement statistics — no RouterArena labels
were used to tune anything):

  code-class prompt (no options, no boxed request) -> gemini-3-flash-preview
  MCQ: majority letter of the three votes; the submitted response comes from a
       majority member, preferring the local models; no majority ->
       gemini-3-flash-preview
  free-answer: the local DeepSeek answer is kept when the local Qwen answer
       corroborates it (exact match or token-F1 >= 0.5 on the extracted final
       answers). When either side produced no boxed answer, the comparison ran
       on degraded last-200-chars extracts; keeping the local answer then
       additionally requires LFM2.5-Embedding-350M cosine >= 0.7 between the
       two extracts (embedding veto). Otherwise -> gemini-3-flash-preview.

The local models' votes cost $0: they run on the submitter's own hardware
(2x AMD Ryzen AI Max+ 395). Declared pricing for the lemonade/* models is an
honest self-hosting estimate (power + amortization); the cloud models use
their public list prices. Full methodology, calibration data, and runtime
telemetry: https://github.com/ramkrishna2910/lemonade-router-routerarena
"""

import json
import math
import os
import re
import urllib.request
from collections import Counter
from typing import TYPE_CHECKING, Optional

from router_inference.router.base_router import BaseRouter

if TYPE_CHECKING:
    from llm_inference.model_inference import ModelInference

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def _letter(text):
    m = BOXED.findall(text or "")
    return m[-1].strip().upper()[:1] if m else None


def _extract_free(text):
    """Returns (extracted_answer, had_boxed)."""
    m = BOXED.findall(text or "")
    if m:
        return m[-1].strip(), True
    return (text or "").strip()[-200:], False


def _token_f1(a, b):
    ta = set(re.findall(r"\w+", (a or "").lower()))
    tb = set(re.findall(r"\w+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return 2 * inter / (len(ta) + len(tb)) if inter else 0.0


class LemonadeRouter(BaseRouter):
    """Live implementation: queries the two lemonade-served local generators,
    then applies the frozen v5.2 vote/corroboration policy above (including
    the lemonade-served embedding veto). Requires a running lemonade server
    (LEMONADE_BASE_URL) and an OPENROUTER_API_KEY for the cloud voter."""

    LOCAL_DS4 = "lemonade/deepseek-v4-flash"
    LOCAL_QWEN = "lemonade/Qwen3.8-27B-GGUF-UD-Q4_K_XL"
    LOCAL_JUDGE = "lemonade/LFM2.5-Embedding-350M"
    CLOUD_G3FP = "gemini-3-flash-preview"
    EMB_THRESHOLD = 0.7

    def __init__(self, router_name: str):
        super().__init__(router_name)
        self._inference: Optional["ModelInference"] = None

    def _infer(self, model, query):
        if self._inference is None:
            from llm_inference.model_inference import ModelInference

            self._inference = ModelInference()
        return self._inference.infer(model, query)

    def _embed_cosine(self, a: str, b: str) -> float:
        """Cosine similarity via the lemonade-served embedding judge."""
        base = os.getenv("LEMONADE_BASE_URL", "http://localhost:8000/api/v1")
        model = os.getenv("LEMONADE_EMB_MODEL", "user.LFM2.5-Embedding-350M")
        req = urllib.request.Request(
            f"{base}/embeddings",
            data=json.dumps({"model": model, "input": [a, b]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as f:
            va, vb = (e["embedding"] for e in json.load(f)["data"])
        num = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x * x for x in va))
        nb = math.sqrt(sum(x * x for x in vb))
        return num / (na * nb) if na and nb else 0.0

    def _get_prediction(self, query: str) -> str:
        is_code = "Options:" not in query and "boxed" not in query
        if is_code:
            return self.CLOUD_G3FP

        r_ds4 = self._infer(self.LOCAL_DS4, query)
        r_qwen = self._infer(self.LOCAL_QWEN, query)

        if "Options:" in query:  # MCQ: tri-vote
            r_g3 = self._infer(self.CLOUD_G3FP, query)
            letters = {
                self.LOCAL_DS4: _letter(r_ds4.get("response")),
                self.LOCAL_QWEN: _letter(r_qwen.get("response")),
                self.CLOUD_G3FP: _letter(r_g3.get("response")),
            }
            votes = Counter(v for v in letters.values() if v)
            if votes:
                top, n = votes.most_common(1)[0]
                if n >= 2:
                    for model in (self.LOCAL_DS4, self.LOCAL_QWEN, self.CLOUD_G3FP):
                        if letters[model] == top:
                            return model
            return self.CLOUD_G3FP

        # free-answer: corroboration is judged on the extracted final answers
        # (boxed content when present), never on whole responses — otherwise
        # contradictory answers with similar phrasing would false-agree.
        a1, boxed1 = _extract_free(r_ds4.get("response"))
        a2, boxed2 = _extract_free(r_qwen.get("response"))
        if a1 and a2 and (a1 == a2 or _token_f1(a1, a2) >= 0.5):
            if boxed1 and boxed2:
                return self.LOCAL_DS4
            # fallback comparison (>=1 side unboxed): token-F1 alone is weak on
            # long-form extracts, so keeping local additionally requires the
            # on-device embedding judge to corroborate semantically.
            if self._embed_cosine(a1, a2) >= self.EMB_THRESHOLD:
                return self.LOCAL_DS4
        return self.CLOUD_G3FP
