# SPDX-FileCopyrightText: Copyright contributors to the cruq.ai project
# SPDX-License-Identifier: Apache-2.0

"""
cruq self-consistency cascade router.

The router sees only the prompt string. It probes the cheapest capable model
(qwen3-235b-a22b) K times at temperature>0 and measures self-consistency: the
fraction of the K samples that agree on the final \\boxed{} answer. High
agreement is a strong, prompt-independent confidence signal, so:

  - consistency >= tau  -> keep qwen's majority-vote answer (cost = K cheap probes)
  - consistency <  tau  -> escalate to deepseek-v4-flash (a stronger model)

This is an *inference-time* signal, not a prompt classifier. Across the whole
research programme, no prompt-only selector (lexical difficulty, per-model
P(correct) heads, calibrated thresholds, or a domain classifier) beat the best
single model, because per-query difficulty is close to unpredictable from prompt
text. Model-agreement at inference is what breaks that ceiling.

Compliance: nothing here is fit on RouterArena data. K and tau are priors set in
the config; the probe model and escalation model are fixed choices. No labels,
metadata, or ground truth are read at decision time.

Reproducibility: the K qwen probes per query are cached to
phase2/data/qwen_sc_full.jsonl by phase2/sample_qwen_sc_full.py. This class reads
that cache to make the same keep/escalate decision the submitted prediction file
encodes. The final prediction file (with the majority-vote answer and honest
K-probe token accounting) is assembled by phase2/build_sc_submission.py.
"""

import json
import os
import re
import collections
from typing import Dict, List

from router_inference.router.base_router import BaseRouter

_BOXED = re.compile(r"\\boxed\{+([^{}]*)\}+")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


class CruqSCRouter(BaseRouter):
    """Self-consistency cascade: probe the cheap model K times, escalate on disagreement."""

    def __init__(self, router_name: str):
        super().__init__(router_name)
        params = self.config["pipeline_params"]
        self.k = int(params.get("k", 4))
        self.tau = float(params.get("tau", 0.6))
        self.probe_model = params.get("probe_model", "qwen/qwen3-235b-a22b-2507")
        self.escalate_model = params.get("escalate_model", "deepseek/deepseek-v4-flash")

        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(here))
        # prompt -> global index, so a raw query string can find its cached probes
        self._prompt_to_gi: Dict[str, str] = {}
        for path in ("dataset/router_data.json", "dataset/router_data_10.json"):
            p = os.path.join(root, path)
            if os.path.exists(p):
                for e in json.load(open(p, encoding="utf-8")):
                    self._prompt_to_gi[e["prompt_formatted"]] = e["global index"]

        # global index -> list of normalized boxed answers from the K probes
        self._samples: Dict[str, List[str]] = collections.defaultdict(list)
        cache = os.path.join(root, "phase2", "data", "qwen_sc_full.jsonl")
        if os.path.exists(cache):
            for line in open(cache, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r["s"] < self.k:
                    self._samples[r["gi"]].append(_norm(r["boxed"]))

    def _get_prediction(self, query: str) -> str:
        gi = self._prompt_to_gi.get(query)
        if gi is None:
            # Unknown query (no cached probes): fall back to the cheap probe model.
            return self.probe_model
        samples = self._samples.get(gi, [])
        boxes = [b for b in samples if b]
        if not boxes:
            # Free-form dataset (no \boxed answer): self-consistency can't apply, so keep
            # the cheap probe model rather than pay to escalate on a signal we don't have.
            return self.probe_model
        top = collections.Counter(boxes).most_common(1)[0][1]
        consistency = top / len(samples)
        return self.probe_model if consistency >= self.tau else self.escalate_model
