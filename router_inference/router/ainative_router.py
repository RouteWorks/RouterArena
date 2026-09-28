# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""AINative Router adapter.

Task-group routing policy fit only on an external calibration set (public source benchmarks with every
RouterArena item removed; leakage-checked). The policy, signal code and kNN index are hash-frozen
(artifacts/FREEZE.json) before any RouterArena query is routed; the router verifies the hashes at load.

Signals: task group from the public eval-config template head (kNN over calibration prompts when the head
is paraphrased). No RouterArena query, answer, label, or other submission's predictions are used.
"""

import os
import sys

from router_inference.router.base_router import BaseRouter

_PROJECT_ROOT = os.environ.get(
    "ARENA_ROUTER_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")),
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class AINativeRouter(BaseRouter):
    def __init__(self, router_name: str):
        super().__init__(router_name)
        from arena_router.router import ArenaRouter as _Impl

        self._impl = _Impl()
        unknown = set(self._impl.policy.models) - set(self.models)
        if unknown:
            raise ValueError(f"policy models not in config: {unknown}")

    def _get_prediction(self, query: str) -> str:
        return self._impl.route(query)
