"""Tests for the AGT-03 Manifold Markets read-only adapter."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from aragora.connectors.prediction_markets.manifold import (
    ManifoldAdapter,
    ManifoldError,
    ManifoldMarket,
    ManifoldResolution,
    _normalize_outcome,
    manifold_to_market_resolution,
)


def _make_payload(
    *,
    market_id: str = "mkt_xyz",
    is_resolved: bool = False,
    resolution: str | None = None,
    close_time_ms: int | None = None,
    outcome_type: str = "BINARY",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "id": market_id,
        "slug": f"slug-{market_id}",
        "question": f"Will {market_id} happen?",
        "creatorUsername": "alice",
        "createdTime": 1_700_000_000_000,
        "closeTime": close_time_ms,
        "resolution": resolution,
        "isResolved": is_resolved,
        "outcomeType": outcome_type,
    }
    if extra:
        payload.update(extra)
    return payload


def _stub_client(responses: dict[str, tuple[int, str]]):
    """Build a deterministic HTTP stub keyed by URL suffix."""

    def client(method: str, url: str, headers: dict) -> tuple[int, str]:
        for suffix, (status, body) in responses.items():
            if url.endswith(suffix):
                return (status, body)
        return (404, json.dumps({"error": f"no stub for {url}"}))

    return client


class TestManifoldMarketParse:
    def test_from_api_payload_basic(self) -> None:
        m = ManifoldMarket.from_api_payload(_make_payload(market_id="abc"))
        assert m.market_id == "abc"
        assert m.outcome_type == "BINARY"
        assert m.is_resolved is False

    def test_from_api_payload_missing_id_raises(self) -> None:
        with pytest.raises(ManifoldError):
            ManifoldMarket.from_api_payload({"slug": "x"})

    def test_from_api_payload_resolved_carries_resolution(self) -> None:
        m = ManifoldMarket.from_api_payload(
            _make_payload(market_id="r1", is_resolved=True, resolution="YES")
        )
        assert m.is_resolved is True
        assert m.resolution == "YES"


class TestNormalizeOutcome:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("YES", "yes"),
            ("yes", "yes"),
            ("NO", "no"),
            ("no", "no"),
            ("CANCEL", "inconclusive"),
            ("MKT", "inconclusive"),
            ("CHOOSE_ONE", "inconclusive"),
            (None, "inconclusive"),
            ("WHATEVER", "inconclusive"),
        ],
    )
    def test_normalization(self, raw: str | None, expected: str) -> None:
        assert _normalize_outcome(raw) == expected


class TestAdapterFetch:
    def test_fetch_market_success(self) -> None:
        client = _stub_client({"market/abc": (200, json.dumps(_make_payload(market_id="abc")))})
        adapter = ManifoldAdapter(http_client=client)
        market = adapter.fetch_market("abc")
        assert market.market_id == "abc"
        assert market.outcome_type == "BINARY"

    def test_fetch_market_requires_id(self) -> None:
        adapter = ManifoldAdapter(http_client=_stub_client({}))
        with pytest.raises(ManifoldError):
            adapter.fetch_market("")

    def test_fetch_market_http_error(self) -> None:
        client = _stub_client({"market/oops": (500, "internal error")})
        adapter = ManifoldAdapter(http_client=client)
        with pytest.raises(ManifoldError):
            adapter.fetch_market("oops")

    def test_fetch_market_non_json(self) -> None:
        client = _stub_client({"market/notjson": (200, "<html>")})
        adapter = ManifoldAdapter(http_client=client)
        with pytest.raises(ManifoldError):
            adapter.fetch_market("notjson")

    def test_fetch_market_transport_error_wrapped(self) -> None:
        def boom(method, url, headers):
            raise OSError("network down")

        adapter = ManifoldAdapter(http_client=boom)
        with pytest.raises(ManifoldError) as info:
            adapter.fetch_market("x")
        assert "transport error" in str(info.value)

    def test_list_markets(self) -> None:
        client = _stub_client(
            {
                "markets?limit=2": (
                    200,
                    json.dumps(
                        [
                            _make_payload(market_id="a"),
                            _make_payload(market_id="b"),
                        ]
                    ),
                )
            }
        )
        adapter = ManifoldAdapter(http_client=client)
        markets = adapter.list_markets(limit=2)
        assert {m.market_id for m in markets} == {"a", "b"}

    def test_list_markets_skips_malformed_entries(self) -> None:
        client = _stub_client(
            {
                "markets?limit=3": (
                    200,
                    json.dumps(
                        [
                            _make_payload(market_id="a"),
                            "not_a_dict",
                            {"slug": "no-id"},  # missing id
                        ]
                    ),
                )
            }
        )
        adapter = ManifoldAdapter(http_client=client)
        markets = adapter.list_markets(limit=3)
        assert [m.market_id for m in markets] == ["a"]

    def test_list_markets_invalid_limit(self) -> None:
        adapter = ManifoldAdapter(http_client=_stub_client({}))
        with pytest.raises(ManifoldError):
            adapter.list_markets(limit=0)
        with pytest.raises(ManifoldError):
            adapter.list_markets(limit=10_000)


class TestDiscoverUnresolvedMarkets:
    def test_filters_short_window_and_resolved(self) -> None:
        now = datetime(2026, 4, 17, tzinfo=UTC)
        now_ms = int(now.timestamp() * 1000)
        far = int((now + timedelta(days=45)).timestamp() * 1000)
        near = int((now + timedelta(days=10)).timestamp() * 1000)

        client = _stub_client(
            {
                "markets?limit=4": (
                    200,
                    json.dumps(
                        [
                            _make_payload(
                                market_id="far_open",
                                is_resolved=False,
                                close_time_ms=far,
                            ),
                            _make_payload(
                                market_id="near_open",
                                is_resolved=False,
                                close_time_ms=near,
                            ),
                            _make_payload(
                                market_id="far_resolved",
                                is_resolved=True,
                                resolution="YES",
                                close_time_ms=far,
                            ),
                            _make_payload(
                                market_id="multi",
                                is_resolved=False,
                                close_time_ms=far,
                                outcome_type="MULTIPLE_CHOICE",
                            ),
                        ]
                    ),
                )
            }
        )
        adapter = ManifoldAdapter(http_client=client, min_window_days=30)
        markets = adapter.discover_unresolved_markets(limit=4, now_ms=now_ms)
        assert [m.market_id for m in markets] == ["far_open"]


class TestFetchResolution:
    def test_unresolved_returns_none(self) -> None:
        client = _stub_client({"market/open": (200, json.dumps(_make_payload(market_id="open")))})
        adapter = ManifoldAdapter(http_client=client)
        assert adapter.fetch_resolution("open") is None

    def test_resolved_yes(self) -> None:
        client = _stub_client(
            {
                "market/r1": (
                    200,
                    json.dumps(
                        _make_payload(
                            market_id="r1",
                            is_resolved=True,
                            resolution="YES",
                            extra={"resolutionTime": 1_700_000_500_000},
                        )
                    ),
                )
            }
        )
        adapter = ManifoldAdapter(http_client=client)
        resolution = adapter.fetch_resolution("r1")
        assert isinstance(resolution, ManifoldResolution)
        assert resolution.outcome == "yes"
        assert resolution.resolved_at_ms == 1_700_000_500_000

    def test_resolved_cancel_maps_to_inconclusive(self) -> None:
        client = _stub_client(
            {
                "market/c1": (
                    200,
                    json.dumps(
                        _make_payload(
                            market_id="c1",
                            is_resolved=True,
                            resolution="CANCEL",
                        )
                    ),
                )
            }
        )
        adapter = ManifoldAdapter(http_client=client)
        resolution = adapter.fetch_resolution("c1")
        assert resolution is not None
        assert resolution.outcome == "inconclusive"


class TestBridgeToMarketResolution:
    def test_yes_bridge(self) -> None:
        from aragora.markets.types import ResolutionEvent as MarketResolution

        m_res = ManifoldResolution(
            market_id="mkt_x",
            outcome="yes",
            resolved_at_ms=1_700_000_500_000,
            raw={"resolution": "YES"},
        )
        bridged = manifold_to_market_resolution(m_res)
        assert isinstance(bridged, MarketResolution)
        assert bridged.outcome == "yes"
        assert bridged.resolution_source == "manifold"
        assert bridged.evidence["resolution"] == "YES"

    def test_no_bridge(self) -> None:
        m_res = ManifoldResolution(market_id="mkt_y", outcome="no", resolved_at_ms=None, raw={})
        bridged = manifold_to_market_resolution(m_res)
        assert bridged.outcome == "no"

    def test_inconclusive_bridge(self) -> None:
        m_res = ManifoldResolution(
            market_id="mkt_z", outcome="inconclusive", resolved_at_ms=None, raw={}
        )
        bridged = manifold_to_market_resolution(m_res)
        assert bridged.outcome == "inconclusive"

    def test_explicit_resolved_at_used(self) -> None:
        m_res = ManifoldResolution(market_id="mkt_t", outcome="yes", resolved_at_ms=None, raw={})
        when = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        bridged = manifold_to_market_resolution(m_res, resolved_at=when)
        assert bridged.resolved_at.startswith("2026-04-17T12:00:00")
