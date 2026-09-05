# SPDX-License-Identifier: Apache-2.0
"""Shunt's real router engine as a RouterArena ``BaseRouter`` (``session_cascade``).

Thin adapter over the shipped Shunt package (https://github.com/KookaS/shunt,
affiliation @KookaS, open-source): it builds the engine with Shunt's shipped
``session_cascade`` strategy over Shunt's live menu and routes each prompt through
``engine.decide``. RouterArena's stateless single-turn / no-verification regime
gives Shunt no verified-outcome channel, so the verified-failure escalation
ladder never fires and the router stays on its cheapest live model
(``deepseek-v4-flash``).

Reproducibility: generated against Shunt pinned at ``SHUNT_PIN``. Choices are
deterministic (the cheapest live model per prompt); recorded answers are the
model's own outputs. Install the pinned commit, or set ``SHUNT_SRC`` to a Shunt
``src/`` checkout, before importing.
"""

import itertools
import os
import sys

from router_inference.router.base_router import BaseRouter

# The Shunt commit this submission was generated with (Shunt is not on PyPI):
#   pip install "git+https://github.com/KookaS/shunt@<SHUNT_PIN>"
SHUNT_PIN = "7ee66af5ddcbd78886a58e5336b21ffc5291da56"

# The live menu (Shunt `router.yaml` pool ids, cheapest-first) and the map from
# the engine's shunt id back to the arena model name predictions must carry.
_MENU_SHUNT_IDS = ["deepseek-v4-flash", "deepseek-v4-pro"]
_MODEL_TO_ARENA = {
    "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
}


def _import_shunt() -> None:
    """Import-for-effect so a missing Shunt fails fast with a pinned hint."""
    src = os.environ.get("SHUNT_SRC")
    if src:
        sys.path.insert(0, src)
    try:
        import shunt  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - re-raised with the owner-facing hint
        raise RuntimeError(
            "could not import the Shunt package (SHUNT_SRC="
            + (repr(src) if src else "unset")
            + "). Point SHUNT_SRC at a Shunt src/ checkout, or install the pinned "
            + "commit: pip install 'git+https://github.com/KookaS/shunt@"
            + SHUNT_PIN
            + "'."
        ) from exc


class ShuntSessionCascade(BaseRouter):
    """Route every prompt through the real Shunt engine (``session_cascade``)."""

    def __init__(self, router_name: str) -> None:
        super().__init__(router_name)
        self._engine = None
        self._counter = itertools.count(1)

    def _get_prediction(self, query: str) -> str:
        engine = self._engine
        if engine is None:
            engine = self._build_engine()
            self._engine = engine
        session_id = "shunt-arena-%06d" % next(self._counter)
        chosen, _reason, _provenance = engine.decide(
            session_id=session_id, prompt_text=query
        )
        if chosen not in _MODEL_TO_ARENA:
            raise ValueError("Shunt chose model %r which has no arena mapping" % chosen)
        return _MODEL_TO_ARENA[chosen]

    def _build_engine(self):
        _import_shunt()
        from shunt.models import ModelPool
        from shunt.router.engine import RouterEngine
        from shunt.router.policy import RouterPolicy
        from shunt.router.selection import SelectionRule
        from shunt.router.strategies import build_strategy

        class _NullSessionManager:
            def get_session(self, session_id):
                return None

        class _NullOutcomeIndex:
            def count_labeled(self):
                raise NotImplementedError

            def count_total_labeled(self):
                raise NotImplementedError

            def effective_labeled(self):
                raise NotImplementedError

            def effective_tier2(self):
                raise NotImplementedError

            def model_priors(self):
                raise NotImplementedError

            def query(self, embedding, k=20):
                raise NotImplementedError

        pool = ModelPool()
        live = [name for name in _MENU_SHUNT_IDS if pool.get_model(name) is not None]
        if not live:
            raise RuntimeError(
                "none of the menu shunt ids %r are present in the Shunt registry (%r)"
                % (_MENU_SHUNT_IDS, pool.model_names())
            )
        pool.restrict_to_live(live)
        policy = RouterPolicy(strategy="session_cascade", models=live)
        selection_rule = SelectionRule(
            min_success_rate=policy.policy.success_rate_threshold,
            min_samples=policy.policy.min_samples,
        )
        strategy = build_strategy(policy.strategy, selection_rule)
        return RouterEngine(
            model_pool=pool,
            session_manager=_NullSessionManager(),
            outcome_index=_NullOutcomeIndex(),
            selection_rule=selection_rule,
            strategy=strategy,
            strategy_id=policy.strategy,
            neighbor_k=policy.policy.k,
            exploration=None,
            escalation=policy.escalation.to_config(),
            task_key_resolver=None,
        )
