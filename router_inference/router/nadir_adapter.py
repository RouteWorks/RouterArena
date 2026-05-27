"""RouterArena adapter for the Nadir /v1/route_only endpoint.

This module is the Nadir side of the RouterArena integration. RouterArena's
evaluation harness imports `NadirRouter` and calls `_get_prediction(query)`
per prompt; we translate that into an HTTPS call to /v1/route_only, parse
the trained classifier's tier decision, and map it back to a model name.

Two API layers:
  - `NadirRouter.route(prompt)` — strict. Returns a `RouteDecision` and
    raises `NadirRouterError` on transport, HTTP, or schema problems. Used
    by tests and by callers that want failures to be visible.
  - `NadirRouter._get_prediction(query)` — RouterArena's contract. Never
    raises; on any error, logs to stderr and returns mid-tier
    (`claude-sonnet-4-6`) so a leaderboard run completes deterministically.

Environment variables:
  NADIR_BACKEND_URL — base URL of the Nadir backend, e.g.
                       https://cgmuqcg2di.us-east-1.awsapprunner.com
  NADIR_API_KEY     — eval-only API key. MUST have no clusters and no
                      expert models attached, else /v1/route_only returns 503.

Schema fingerprint is locked at adapter build time. The backend emits the
same constant in every response; a mismatch is treated as untrusted data.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


# ──────────────────────────────────────────────────────────────────────────
# Schema fingerprint — must match _ROUTE_ONLY_SCHEMA_FINGERPRINT in
# backend/app/api/route_only.py. Recompute with:
#
#   python3 -c "import hashlib; fs=sorted(['schema_fingerprint','tier',\
#     'model','complexity_score','classifier_confidence','latency_ms',\
#     'classifier_version']); \
#     print(hashlib.sha256(','.join(fs).encode()).hexdigest())"
# ──────────────────────────────────────────────────────────────────────────

EXPECTED_SCHEMA_FINGERPRINT: str = (
    "7a1538f6cc8bf7960d564dc00b58f2e336b685af50bd123a01e2dc569731efb4"
)

# Confidence-histogram bucket boundaries.
_BUCKETS = ["<0.4", "0.4-0.6", "0.6-0.8", ">0.8"]

LOW_CONFIDENCE_THRESHOLD: float = 0.4
LOW_CONFIDENCE_FLAG_RATIO: float = 0.15  # >15% in <0.4 → NEEDS_FOUNDER_REVIEW


def _bucket_for(confidence: float) -> str:
    if confidence < 0.4:
        return "<0.4"
    if confidence < 0.6:
        return "0.4-0.6"
    if confidence < 0.8:
        return "0.6-0.8"
    return ">0.8"


def confidence_histogram() -> Dict[str, int]:
    """Return a fresh empty histogram."""
    return {b: 0 for b in _BUCKETS}


def flag_smoke_run(histogram: Dict[str, int]) -> Dict[str, Any]:
    """Compute the smoke-run verdict from a confidence histogram.

    Returns a dict with `verdict` (PASS / NEEDS_FOUNDER_REVIEW) and the
    fraction of low-confidence calls. The CLI uses exit code 0 in either
    case; CI greps the `verdict` field instead.
    """
    total = sum(histogram.values()) or 1
    low = histogram.get("<0.4", 0)
    ratio = low / total
    verdict = (
        "NEEDS_FOUNDER_REVIEW" if ratio > LOW_CONFIDENCE_FLAG_RATIO else "PASS"
    )
    return {
        "verdict": verdict,
        "low_confidence_ratio": ratio,
        "low_confidence_count": low,
        "total": total,
        "histogram": histogram,
    }


# ──────────────────────────────────────────────────────────────────────────
# RouterArena BaseRouter — try to import; fall back to a local stub so the
# package is importable outside the RouterArena fork (e.g. for tests in
# this repo).
# ──────────────────────────────────────────────────────────────────────────

try:  # pragma: no cover - depends on RouterArena being on sys.path
    from router_arena.routers.base_router import BaseRouter  # type: ignore
except Exception:  # pragma: no cover

    class BaseRouter:  # type: ignore[no-redef]
        """Local stand-in for RouterArena's BaseRouter.

        The real class has a richer interface; the only method downstream
        eval code calls is `_get_prediction(query) -> str`.
        """

        def __init__(self, config_path: str = "") -> None:
            self.config_path = config_path
            self.config: Dict[str, Any] = {}
            if config_path and os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        self.config = json.load(f)
                except (OSError, json.JSONDecodeError):
                    self.config = {}


# ──────────────────────────────────────────────────────────────────────────
# Public types.
# ──────────────────────────────────────────────────────────────────────────


class NadirRouterError(RuntimeError):
    """Raised by `NadirRouter.route()` on any transport, HTTP, or schema
    problem. `_get_prediction` catches this internally and falls back to
    mid-tier, but tests and strict callers see the raise."""


@dataclass(frozen=True)
class RouteDecision:
    """One routing decision plus the metadata the smoke script needs."""

    tier: str
    model: str
    complexity_score: float
    classifier_confidence: float
    latency_ms: int
    classifier_version: str
    schema_fingerprint: str
    classifier_sha: str  # x-nadir-classifier-sha response header


# ──────────────────────────────────────────────────────────────────────────
# Adapter.
# ──────────────────────────────────────────────────────────────────────────


class NadirRouter(BaseRouter):
    """HTTP adapter for Nadir's /v1/route_only.

    Two ways to use:
      - `route(prompt)` → `RouteDecision`. Raises on any error.
      - `_get_prediction(query)` → str (model name). Never raises;
        falls back to `claude-sonnet-4-6` on any error.
    """

    EXPECTED_SCHEMA_FINGERPRINT: str = EXPECTED_SCHEMA_FINGERPRINT

    _TIER_TO_MODEL: Dict[str, str] = {
        "simple": "claude-haiku-4-5",
        "medium": "claude-sonnet-4-6",
        "complex": "claude-opus-4-6",
    }
    _FALLBACK_MODEL: str = "claude-sonnet-4-6"  # mid-tier on any error

    def __init__(
        self,
        config_path: str = "",
        *,
        client: Optional[httpx.Client] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        super().__init__(config_path)
        self._base_url = (
            base_url
            if base_url is not None
            else os.environ.get("NADIR_BACKEND_URL", "")
        )
        self._api_key = (
            api_key if api_key is not None else os.environ.get("NADIR_API_KEY", "")
        )
        self._timeout = timeout
        self._client = (
            client if client is not None else httpx.Client(timeout=timeout)
        )
        self._owns_client = client is None
        self._confidence_histogram: Dict[str, int] = confidence_histogram()
        # First-seen header values, for cross-call consistency checks in the
        # smoke script. Set on the first successful response.
        self._first_classifier_sha: Optional[str] = None
        self._first_schema_fingerprint: Optional[str] = None

    # ── Inspectors ────────────────────────────────────────────────────────

    @property
    def histogram(self) -> Dict[str, int]:
        """Return a copy of the in-flight confidence histogram."""
        return dict(self._confidence_histogram)

    @property
    def first_classifier_sha(self) -> Optional[str]:
        return self._first_classifier_sha

    @property
    def first_schema_fingerprint(self) -> Optional[str]:
        return self._first_schema_fingerprint

    def smoke_verdict(self) -> Dict[str, Any]:
        """Apply the >15% <0.4 rule to the in-flight histogram."""
        return flag_smoke_run(self.histogram)

    # ── Strict core: raises on any error. ─────────────────────────────────

    def route(self, prompt: str) -> RouteDecision:
        """Route a single prompt. Raises `NadirRouterError` on any failure.

        Use this in tests and in callers that want to see errors. The
        RouterArena harness calls `_get_prediction` instead, which wraps
        this with a never-raise fallback.
        """
        if not self._base_url:
            raise NadirRouterError(
                "NADIR_BACKEND_URL is not set (pass base_url= or set env var)"
            )

        url = f"{self._base_url.rstrip('/')}/v1/route_only"
        payload = {"messages": [{"role": "user", "content": prompt}]}
        headers = {"X-API-Key": self._api_key, "Content-Type": "application/json"}

        try:
            resp = self._client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise NadirRouterError(f"timeout after {self._timeout}s: {exc}") from exc
        except httpx.RequestError as exc:
            raise NadirRouterError(f"request error: {exc}") from exc

        if resp.status_code < 200 or resp.status_code >= 300:
            raise NadirRouterError(
                f"HTTP {resp.status_code} from /v1/route_only: "
                f"{resp.text[:300]}"
            )

        try:
            body = resp.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise NadirRouterError(f"non-JSON response body: {exc}") from exc

        fp = body.get("schema_fingerprint")
        if fp != self.EXPECTED_SCHEMA_FINGERPRINT:
            raise NadirRouterError(
                f"schema fingerprint mismatch: expected "
                f"{self.EXPECTED_SCHEMA_FINGERPRINT!r}, got {fp!r}"
            )

        tier = body.get("tier")
        if tier not in self._TIER_TO_MODEL:
            raise NadirRouterError(f"unknown tier {tier!r}")

        # Confidence histogram bookkeeping.
        try:
            confidence = float(body.get("classifier_confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        self._confidence_histogram[_bucket_for(confidence)] += 1

        # Cross-call consistency: cache the first observed header values.
        classifier_sha = resp.headers.get("x-nadir-classifier-sha", "")
        if self._first_classifier_sha is None:
            self._first_classifier_sha = classifier_sha
        if self._first_schema_fingerprint is None:
            self._first_schema_fingerprint = fp

        try:
            latency_ms = int(body.get("latency_ms", 0))
        except (TypeError, ValueError):
            latency_ms = 0
        try:
            complexity_score = float(body.get("complexity_score", 0.0))
        except (TypeError, ValueError):
            complexity_score = 0.0

        return RouteDecision(
            tier=tier,
            model=self._TIER_TO_MODEL[tier],
            complexity_score=complexity_score,
            classifier_confidence=confidence,
            latency_ms=latency_ms,
            classifier_version=str(body.get("classifier_version", "")),
            schema_fingerprint=fp,
            classifier_sha=classifier_sha,
        )

    # ── RouterArena contract: never raises. ───────────────────────────────

    def _get_prediction(self, query: str) -> str:
        """Return a model name for `query`. Never raises.

        On any error, logs to stderr and returns mid-tier so the
        leaderboard run completes deterministically. Tests assert raises
        via `route()`; this is the production callable.
        """
        try:
            decision = self.route(query)
        except NadirRouterError as exc:
            print(
                f"[NadirRouter] error, falling back to mid-tier: {exc}",
                file=sys.stderr,
            )
            return self._FALLBACK_MODEL
        return decision.model

    def close(self) -> None:
        if self._owns_client:
            try:
                self._client.close()
            except Exception:  # pragma: no cover
                pass
