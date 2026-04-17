"""Manifold Markets read-only adapter (AGT-03 Phase 1).

Phase 1: read-only — discover markets, fetch a market, fetch its
resolution. No write access (no prediction submission, no betting).
The downstream integration into the AGT-05 reputation flow consumes
the resolution events this adapter produces.

API: https://docs.manifold.markets/api

This adapter:
- Uses Manifold's public REST API; no auth required for reads.
- Has no implicit network calls — every method takes an explicit
  ``http_client`` callable that returns ``(status_code, body_text)``.
  Callers wire in ``httpx``, ``urllib``, or a stub for tests.
- Maps Manifold's market shape to a normalized
  :class:`ManifoldMarket` and Manifold's resolution to a
  :class:`ManifoldResolution`. The :func:`manifold_to_market_resolution`
  helper bridges into the AGT-04 :class:`aragora.markets.types.
  ResolutionEvent` shape so the same downstream pipeline (settle_claim,
  Brier scoring, reputation deltas) works for both internal synthetic
  GitHub markets and external Manifold markets.

Out of scope for this PR (deferred to AGT-03 Phase 2 / Phase 3):
- Prediction submission (bot-policy compliance + API-key handling)
- Per-agent Brier score wiring (lives in AGT-05)
- Real-money venues (Kalshi, Polymarket) — see venue-stack plan
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)

MANIFOLD_API_BASE = "https://api.manifold.markets/v0"

# Per AGT-03 plan: avoid markets with <30-day resolution windows in
# initial calibration phase (noise reduction)
DEFAULT_MIN_WINDOW_DAYS = 30


class ManifoldError(RuntimeError):
    """Raised when a Manifold API call fails or returns an unexpected shape."""


# An HTTP client is any callable: (method, url, headers) -> (status_code, body_text).
# Keeping this thin lets callers inject httpx, urllib, or a deterministic test stub.
HttpClient = Callable[..., tuple[int, str]]


@dataclass(frozen=True)
class ManifoldMarket:
    """Normalized Manifold market view.

    Mirrors the Manifold "FullMarket" shape selectively. Only the fields
    we need for AGT-03 read-only flows are surfaced; the rest stays in
    ``raw`` for downstream consumers that need it.
    """

    market_id: str
    slug: str
    question: str
    creator_username: str
    created_time_ms: int
    close_time_ms: int | None
    resolution: str | None
    is_resolved: bool
    outcome_type: str
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api_payload(cls, payload: dict[str, Any]) -> "ManifoldMarket":
        market_id = str(payload.get("id") or "").strip()
        if not market_id:
            raise ManifoldError("Manifold market payload missing 'id'")
        return cls(
            market_id=market_id,
            slug=str(payload.get("slug") or ""),
            question=str(payload.get("question") or ""),
            creator_username=str(payload.get("creatorUsername") or ""),
            created_time_ms=int(payload.get("createdTime") or 0),
            close_time_ms=(
                int(payload["closeTime"]) if payload.get("closeTime") is not None else None
            ),
            resolution=(
                str(payload["resolution"]) if payload.get("resolution") is not None else None
            ),
            is_resolved=bool(payload.get("isResolved")),
            outcome_type=str(payload.get("outcomeType") or "BINARY"),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class ManifoldResolution:
    """Normalized Manifold resolution event.

    ``outcome`` follows the same ternary shape as
    :class:`aragora.markets.types.ResolutionEvent` so the AGT-05 settlement
    flow can consume it without further normalization.
    """

    market_id: str
    outcome: str  # "yes" | "no" | "inconclusive"
    resolved_at_ms: int | None
    raw: dict[str, Any] = field(default_factory=dict)


def _normalize_outcome(resolution_field: str | None) -> str:
    """Map Manifold's resolution string to our ternary outcome shape."""
    if resolution_field is None:
        return "inconclusive"
    lowered = str(resolution_field).strip().upper()
    if lowered in {"YES"}:
        return "yes"
    if lowered in {"NO"}:
        return "no"
    if lowered in {"CANCEL", "MKT", "CHOOSE_ONE", "CHOOSE_MULTIPLE"}:
        # MKT (probability-weighted) and multi-choice resolutions are
        # not binary YES/NO outcomes; the AGT-05 layer treats them as
        # inconclusive for binary-Brier purposes.
        return "inconclusive"
    return "inconclusive"


@dataclass
class ManifoldAdapter:
    """Read-only adapter to the Manifold Markets API.

    Parameters:
        http_client: callable that performs HTTP calls. Signature:
            ``http_client(method: str, url: str, headers: dict) -> (int, str)``.
            Required — no default network access.
        api_base: override the base URL for testing.
        min_window_days: filter out markets with very short resolution
            windows from :meth:`discover_unresolved_markets` (default 30).
    """

    http_client: HttpClient
    api_base: str = MANIFOLD_API_BASE
    min_window_days: int = DEFAULT_MIN_WINDOW_DAYS

    def _get(self, path: str) -> Any:
        url = f"{self.api_base.rstrip('/')}/{path.lstrip('/')}"
        try:
            status, body = self.http_client("GET", url, {"Accept": "application/json"})
        except Exception as exc:  # noqa: BLE001 - adapter wraps all transport errors
            raise ManifoldError(f"manifold transport error for {path}: {exc}") from exc
        if status >= 400:
            raise ManifoldError(f"manifold {path} returned HTTP {status}: {body[:200]}")
        try:
            return json.loads(body or "null")
        except json.JSONDecodeError as exc:
            raise ManifoldError(f"manifold {path} returned non-JSON: {exc}") from exc

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    def fetch_market(self, market_id: str) -> ManifoldMarket:
        """Fetch a single market by id."""
        if not market_id:
            raise ManifoldError("market_id is required")
        payload = self._get(f"market/{market_id}")
        if not isinstance(payload, dict):
            raise ManifoldError(f"manifold market/{market_id} returned non-object payload")
        return ManifoldMarket.from_api_payload(payload)

    def fetch_market_by_slug(self, slug: str) -> ManifoldMarket:
        """Fetch a single market by slug."""
        if not slug:
            raise ManifoldError("slug is required")
        payload = self._get(f"slug/{slug}")
        if not isinstance(payload, dict):
            raise ManifoldError(f"manifold slug/{slug} returned non-object payload")
        return ManifoldMarket.from_api_payload(payload)

    def list_markets(self, *, limit: int = 50, before: str | None = None) -> list[ManifoldMarket]:
        """List recent markets (Manifold's pagination is `before=<market_id>`)."""
        if limit < 1 or limit > 1000:
            raise ManifoldError("limit must be in [1, 1000]")
        path = f"markets?limit={int(limit)}"
        if before:
            path += f"&before={before}"
        payload = self._get(path)
        if not isinstance(payload, list):
            raise ManifoldError("manifold markets endpoint returned non-list payload")
        out: list[ManifoldMarket] = []
        for entry in payload:
            if isinstance(entry, dict):
                try:
                    out.append(ManifoldMarket.from_api_payload(entry))
                except ManifoldError:
                    logger.debug("skipping malformed manifold market entry")
        return out

    def discover_unresolved_markets(
        self,
        *,
        limit: int = 50,
        now_ms: int | None = None,
    ) -> list[ManifoldMarket]:
        """List unresolved markets whose close time is sufficiently far out.

        Filters per the AGT-03 plan: BINARY only, not yet resolved,
        close time at least :attr:`min_window_days` away.
        """
        markets = self.list_markets(limit=limit)
        reference_ms = (
            now_ms if now_ms is not None else int(datetime.now(tz=UTC).timestamp() * 1000)
        )
        threshold_ms = self.min_window_days * 24 * 3600 * 1000
        out: list[ManifoldMarket] = []
        for market in markets:
            if market.is_resolved:
                continue
            if market.outcome_type.upper() != "BINARY":
                continue
            if market.close_time_ms is None:
                continue
            if market.close_time_ms - reference_ms < threshold_ms:
                continue
            out.append(market)
        return out

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def fetch_resolution(self, market_id: str) -> ManifoldResolution | None:
        """Return the resolution event for a market, or None if unresolved."""
        market = self.fetch_market(market_id)
        if not market.is_resolved:
            return None
        return ManifoldResolution(
            market_id=market.market_id,
            outcome=_normalize_outcome(market.resolution),
            resolved_at_ms=market.raw.get("resolutionTime"),
            raw={"resolution": market.resolution, "outcomeType": market.outcome_type},
        )


def manifold_to_market_resolution(
    resolution: ManifoldResolution,
    *,
    resolved_at: datetime | None = None,
):
    """Bridge from a Manifold resolution to the AGT-04 ResolutionEvent shape.

    Lazily imports :mod:`aragora.markets.types` so this connector can
    be used without requiring the markets module to be loaded.
    """
    from aragora.markets.types import ResolutionEvent as MarketResolution

    when = resolved_at
    if when is None and resolution.resolved_at_ms:
        when = datetime.fromtimestamp(resolution.resolved_at_ms / 1000.0, tz=UTC)

    if resolution.outcome == "yes":
        return MarketResolution.yes(
            market_id=resolution.market_id,
            resolution_source="manifold",
            evidence=dict(resolution.raw),
            resolved_at=when,
        )
    if resolution.outcome == "no":
        return MarketResolution.no(
            market_id=resolution.market_id,
            resolution_source="manifold",
            evidence=dict(resolution.raw),
            resolved_at=when,
        )
    return MarketResolution.inconclusive(
        market_id=resolution.market_id,
        resolution_source="manifold",
        evidence=dict(resolution.raw),
        resolved_at=when,
    )


__all__ = [
    "DEFAULT_MIN_WINDOW_DAYS",
    "MANIFOLD_API_BASE",
    "ManifoldAdapter",
    "ManifoldError",
    "ManifoldMarket",
    "ManifoldResolution",
    "manifold_to_market_resolution",
]
