"""Comprehensive tests for the cross-pollination handler module.

Covers all 9 handler classes:
- CrossPollinationStatsHandler (GET /api/v1/cross-pollination/stats)
- CrossPollinationSubscribersHandler (GET /api/v1/cross-pollination/subscribers)
- CrossPollinationBridgeHandler (GET /api/v1/cross-pollination/bridge)
- CrossPollinationMetricsHandler (GET /api/v1/cross-pollination/metrics)
- CrossPollinationResetHandler (POST /api/v1/cross-pollination/reset)
- CrossPollinationKMHandler (GET /api/v1/cross-pollination/km)
- CrossPollinationKMSyncHandler (POST /api/v1/cross-pollination/km/sync)
- CrossPollinationKMStalenessHandler (POST /api/v1/cross-pollination/km/staleness-check)
- CrossPollinationKMCultureHandler (GET /api/v1/cross-pollination/km/culture)

Also covers:
- register_routes helper
- Rate limiter configuration
- Error handling paths (ImportError, KeyError, ValueError, etc.)
- Edge cases and input validation
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.cross_pollination import (
    CrossPollinationBridgeHandler,
    CrossPollinationKMCultureHandler,
    CrossPollinationKMHandler,
    CrossPollinationKMStalenessHandler,
    CrossPollinationKMSyncHandler,
    CrossPollinationMetricsHandler,
    CrossPollinationResetHandler,
    CrossPollinationStatsHandler,
    CrossPollinationSubscribersHandler,
    _cross_pollination_limiter,
    register_routes,
)

# Patch targets for local imports inside handler methods
_PATCH_MGR = "aragora.events.cross_subscribers.get_cross_subscriber_manager"
_PATCH_BRIDGE = "aragora.events.arena_bridge.EVENT_TYPE_MAP"
_PATCH_METRICS = "aragora.server.prometheus_cross_pollination.get_cross_pollination_metrics_text"
_PATCH_RANKING = "aragora.knowledge.mound.adapters.RankingAdapter"
_PATCH_RLM = "aragora.knowledge.mound.adapters.rlm_adapter.RlmAdapter"
_PATCH_MOUND = "aragora.knowledge.mound.get_knowledge_mound"
_PATCH_STALENESS = "aragora.knowledge.mound.staleness.StalenessDetector"
_PATCH_STALENESS_CFG = "aragora.knowledge.mound.staleness.StalenessConfig"
_PATCH_RECORD_SYNC = "aragora.server.prometheus_cross_pollination.record_km_adapter_sync"
_PATCH_RECORD_STALE = "aragora.server.prometheus_cross_pollination.record_km_staleness_check"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the module-level rate limiter between tests."""
    _cross_pollination_limiter._buckets = defaultdict(list)
    _cross_pollination_limiter._requests = _cross_pollination_limiter._buckets
    yield
    _cross_pollination_limiter._buckets = defaultdict(list)
    _cross_pollination_limiter._requests = _cross_pollination_limiter._buckets


def _mock_manager(**overrides):
    """Create a mock cross-subscriber manager."""
    mgr = MagicMock()
    mgr.get_stats.return_value = overrides.get("stats", {})
    mgr._subscribers = overrides.get("subscribers", {})
    mgr.reset_stats.return_value = None
    mgr.get_batch_stats.return_value = overrides.get("batch_stats", {"pending": 0})
    mgr.flush_all_batches.return_value = overrides.get("flush_count", 3)
    # Prevent MagicMock auto-attribute creation for adapter caches.
    # The handler uses getattr(manager, "_ranking_adapter", None) which
    # on MagicMock would return a new MagicMock instead of None.
    mgr._ranking_adapter = None
    mgr._rlm_adapter = None
    return mgr


# ============================================================================
# CrossPollinationStatsHandler
# ============================================================================


class TestCrossPollinationStatsHandler:
    """Tests for GET /api/v1/cross-pollination/stats."""

    def _make(self, ctx=None):
        return CrossPollinationStatsHandler(ctx=ctx)

    # --- Routes ---

    def test_routes_include_stats(self):
        h = self._make()
        assert "/api/v1/cross-pollination/stats" in h.ROUTES

    def test_routes_include_conflicts(self):
        h = self._make()
        assert "/api/v1/cross-pollination/conflicts" in h.ROUTES

    def test_routes_include_federation(self):
        h = self._make()
        assert "/api/v1/cross-pollination/federation" in h.ROUTES

    def test_routes_include_federation_sync(self):
        h = self._make()
        assert "/api/v1/cross-pollination/federation/sync" in h.ROUTES

    def test_routes_include_subscribe(self):
        h = self._make()
        assert "/api/v1/cross-pollination/subscribe" in h.ROUTES

    def test_routes_include_sync_status(self):
        h = self._make()
        assert "/api/v1/cross-pollination/sync/status" in h.ROUTES

    def test_routes_include_sync_trigger(self):
        h = self._make()
        assert "/api/v1/cross-pollination/sync/trigger" in h.ROUTES

    def test_routes_count(self):
        h = self._make()
        assert len(h.ROUTES) == 7

    # --- Init ---

    def test_init_default_ctx(self):
        h = CrossPollinationStatsHandler()
        assert h.ctx == {}

    def test_init_custom_ctx(self):
        ctx = {"server": "test"}
        h = CrossPollinationStatsHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_init_none_ctx(self):
        h = CrossPollinationStatsHandler(ctx=None)
        assert h.ctx == {}

    # --- GET success ---

    @pytest.mark.asyncio
    async def test_get_success_basic(self):
        h = self._make()
        mgr = _mock_manager(
            stats={
                "s1": {"events_processed": 5, "events_failed": 0, "enabled": True},
            }
        )
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "ok"

    @pytest.mark.asyncio
    async def test_get_summary_totals(self):
        h = self._make()
        mgr = _mock_manager(
            stats={
                "s1": {"events_processed": 10, "events_failed": 1, "enabled": True},
                "s2": {"events_processed": 20, "events_failed": 2, "enabled": False},
                "s3": {"events_processed": 5, "events_failed": 0, "enabled": True},
            }
        )
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["summary"]["total_subscribers"] == 3
        assert body["summary"]["enabled_subscribers"] == 2
        assert body["summary"]["total_events_processed"] == 35
        assert body["summary"]["total_events_failed"] == 3

    @pytest.mark.asyncio
    async def test_get_empty_stats(self):
        h = self._make()
        mgr = _mock_manager(stats={})
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["summary"]["total_subscribers"] == 0
        assert body["summary"]["total_events_processed"] == 0

    @pytest.mark.asyncio
    async def test_get_subscribers_without_enabled_key(self):
        h = self._make()
        mgr = _mock_manager(
            stats={
                "s1": {"events_processed": 1, "events_failed": 0},
            }
        )
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        # When enabled key is missing, default is True
        assert body["summary"]["enabled_subscribers"] == 1

    @pytest.mark.asyncio
    async def test_get_includes_raw_subscribers(self):
        h = self._make()
        stats = {"sub_a": {"events_processed": 7, "events_failed": 1, "enabled": True}}
        mgr = _mock_manager(stats=stats)
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["subscribers"] == stats

    # --- GET errors ---

    @pytest.mark.asyncio
    async def test_get_import_error_returns_503(self):
        h = self._make()
        with patch.dict("sys.modules", {"aragora.events.cross_subscribers": None}):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 503
        assert "not available" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_get_key_error_returns_500(self):
        h = self._make()
        mgr = MagicMock()
        mgr.get_stats.side_effect = KeyError("bad key")
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_value_error_returns_500(self):
        h = self._make()
        mgr = MagicMock()
        mgr.get_stats.side_effect = ValueError("bad value")
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_type_error_returns_500(self):
        h = self._make()
        mgr = MagicMock()
        mgr.get_stats.side_effect = TypeError("bad type")
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_attribute_error_returns_500(self):
        h = self._make()
        with patch(_PATCH_MGR, side_effect=AttributeError("no attr")):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_os_error_returns_500(self):
        h = self._make()
        with patch(_PATCH_MGR, side_effect=OSError("disk full")):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 500


# ============================================================================
# CrossPollinationSubscribersHandler
# ============================================================================


class TestCrossPollinationSubscribersHandler:
    """Tests for GET /api/v1/cross-pollination/subscribers."""

    def _make(self):
        return CrossPollinationSubscribersHandler(server_context={})

    def test_routes(self):
        h = self._make()
        assert "/api/v1/cross-pollination/subscribers" in h.ROUTES

    @pytest.mark.asyncio
    async def test_get_with_subscribers(self):
        h = self._make()
        evt = MagicMock()
        evt.value = "debate_complete"
        handler_fn = MagicMock(__name__="on_complete")
        mgr = MagicMock()
        mgr._subscribers = {evt: [("sub1", handler_fn)]}

        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["count"] == 1
        assert body["subscribers"][0]["name"] == "sub1"
        assert body["subscribers"][0]["event_type"] == "debate_complete"
        assert body["subscribers"][0]["handler"] == "on_complete"

    @pytest.mark.asyncio
    async def test_get_multiple_event_types(self):
        h = self._make()
        evt1 = MagicMock()
        evt1.value = "type_a"
        evt2 = MagicMock()
        evt2.value = "type_b"
        fn1 = MagicMock(__name__="handler_a")
        fn2 = MagicMock(__name__="handler_b")
        fn3 = MagicMock(__name__="handler_c")
        mgr = MagicMock()
        mgr._subscribers = {
            evt1: [("s1", fn1), ("s2", fn2)],
            evt2: [("s3", fn3)],
        }

        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["count"] == 3

    @pytest.mark.asyncio
    async def test_get_handler_without_name(self):
        h = self._make()
        evt = MagicMock()
        evt.value = "ev"
        handler_obj = 42  # int has no __name__
        mgr = MagicMock()
        mgr._subscribers = {evt: [("anon", handler_obj)]}

        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["subscribers"][0]["handler"] == "42"

    @pytest.mark.asyncio
    async def test_get_empty_subscribers(self):
        h = self._make()
        mgr = MagicMock()
        mgr._subscribers = {}

        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["count"] == 0
        assert body["subscribers"] == []

    @pytest.mark.asyncio
    async def test_get_import_error(self):
        h = self._make()
        with patch.dict("sys.modules", {"aragora.events.cross_subscribers": None}):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_attribute_error(self):
        h = self._make()
        with patch(_PATCH_MGR, side_effect=AttributeError("no attr")):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_key_error(self):
        h = self._make()
        mgr = MagicMock()
        mgr._subscribers = MagicMock()
        mgr._subscribers.items.side_effect = KeyError("k")
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 500


# ============================================================================
# CrossPollinationBridgeHandler
# ============================================================================


class TestCrossPollinationBridgeHandler:
    """Tests for GET /api/v1/cross-pollination/bridge."""

    def _make(self, ctx=None):
        return CrossPollinationBridgeHandler(ctx=ctx)

    def test_routes(self):
        h = self._make()
        assert "/api/v1/cross-pollination/bridge" in h.ROUTES

    def test_init_default(self):
        h = CrossPollinationBridgeHandler()
        assert h.ctx == {}

    def test_init_with_ctx(self):
        ctx = {"k": "v"}
        h = CrossPollinationBridgeHandler(ctx=ctx)
        assert h.ctx is ctx

    @pytest.mark.asyncio
    async def test_get_success(self):
        h = self._make()
        mock_map = {
            "debate_start": MagicMock(value="stream_start"),
            "debate_end": MagicMock(value="stream_end"),
        }
        with patch(_PATCH_BRIDGE, mock_map):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["status"] == "ok"
        assert body["mapped_event_count"] == 2
        assert body["event_mappings"]["debate_start"] == "stream_start"

    @pytest.mark.asyncio
    async def test_get_empty_map(self):
        h = self._make()
        with patch(_PATCH_BRIDGE, {}):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["mapped_event_count"] == 0
        assert body["event_mappings"] == {}

    @pytest.mark.asyncio
    async def test_get_import_error(self):
        h = self._make()
        with patch.dict("sys.modules", {"aragora.events.arena_bridge": None}):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_key_error(self):
        h = self._make()
        mock_map = MagicMock()
        mock_map.items.side_effect = KeyError("bad")
        with patch(_PATCH_BRIDGE, mock_map):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_type_error(self):
        h = self._make()
        mock_map = MagicMock()
        mock_map.items.side_effect = TypeError("bad")
        with patch(_PATCH_BRIDGE, mock_map):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_value_error(self):
        h = self._make()
        mock_map = MagicMock()
        mock_map.items.side_effect = ValueError("bad")
        with patch(_PATCH_BRIDGE, mock_map):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 500


# ============================================================================
# CrossPollinationMetricsHandler
# ============================================================================


class TestCrossPollinationMetricsHandler:
    """Tests for GET /api/v1/cross-pollination/metrics."""

    def _make(self):
        return CrossPollinationMetricsHandler(server_context={})

    def test_routes(self):
        h = self._make()
        assert "/api/v1/cross-pollination/metrics" in h.ROUTES

    @pytest.mark.asyncio
    async def test_get_success_str_body(self):
        h = self._make()
        text = "# HELP cross_pollination_events Total events\ncross_pollination_events 42\n"
        with patch(_PATCH_METRICS, return_value=text):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 200
        assert result.content_type == "text/plain; version=0.0.4; charset=utf-8"
        assert result.body == text.encode("utf-8")

    @pytest.mark.asyncio
    async def test_get_success_bytes_body(self):
        h = self._make()
        raw = b"# metrics\nfoo 1\n"
        with patch(_PATCH_METRICS, return_value=raw):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 200
        assert result.body is raw

    @pytest.mark.asyncio
    async def test_get_import_error(self):
        h = self._make()
        with patch.dict(
            "sys.modules",
            {"aragora.server.prometheus_cross_pollination": None},
        ):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_key_error(self):
        h = self._make()
        with patch(_PATCH_METRICS, side_effect=KeyError("bad")):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_os_error(self):
        h = self._make()
        with patch(_PATCH_METRICS, side_effect=OSError("io fail")):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_value_error(self):
        h = self._make()
        with patch(_PATCH_METRICS, side_effect=ValueError("bad")):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_headers(self):
        h = self._make()
        with patch(_PATCH_METRICS, return_value="data"):
            result = await h.get.__wrapped__(h)
        assert result.headers["Content-Type"] == "text/plain; version=0.0.4; charset=utf-8"


# ============================================================================
# CrossPollinationResetHandler
# ============================================================================


class TestCrossPollinationResetHandler:
    """Tests for POST /api/v1/cross-pollination/reset."""

    def _make(self):
        return CrossPollinationResetHandler(server_context={})

    def test_routes(self):
        h = self._make()
        assert "/api/v1/cross-pollination/reset" in h.ROUTES

    @pytest.mark.asyncio
    async def test_post_success(self):
        h = self._make()
        mgr = _mock_manager()
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.post.__wrapped__(h)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "ok"
        assert "reset" in body["message"].lower()
        mgr.reset_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_import_error(self):
        h = self._make()
        with patch.dict("sys.modules", {"aragora.events.cross_subscribers": None}):
            result = await h.post.__wrapped__(h)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_post_key_error(self):
        h = self._make()
        mgr = MagicMock()
        mgr.reset_stats.side_effect = KeyError("k")
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.post.__wrapped__(h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_post_value_error(self):
        h = self._make()
        mgr = MagicMock()
        mgr.reset_stats.side_effect = ValueError("fail")
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.post.__wrapped__(h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_post_type_error(self):
        h = self._make()
        mgr = MagicMock()
        mgr.reset_stats.side_effect = TypeError("fail")
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.post.__wrapped__(h)
        assert _status(result) == 500


# ============================================================================
# CrossPollinationKMHandler
# ============================================================================


class TestCrossPollinationKMHandler:
    """Tests for GET /api/v1/cross-pollination/km."""

    def _make(self):
        return CrossPollinationKMHandler(server_context={})

    def test_routes(self):
        h = self._make()
        assert "/api/v1/cross-pollination/km" in h.ROUTES

    @pytest.mark.asyncio
    async def test_get_success(self):
        h = self._make()
        mgr = _mock_manager(
            stats={
                "memory_to_mound": {"events_processed": 10, "events_failed": 0},
                "mound_to_memory_retrieval": {"events_processed": 5, "events_failed": 1},
            },
            batch_stats={"pending": 2, "flushed": 10},
        )
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["status"] == "ok"
        assert body["summary"]["total_km_handlers"] == 15
        assert body["summary"]["inbound_handlers"] > 0
        assert body["summary"]["outbound_handlers"] > 0
        assert "adapters" in body
        assert "ranking" in body["adapters"]

    @pytest.mark.asyncio
    async def test_get_inbound_outbound_split(self):
        h = self._make()
        mgr = _mock_manager(
            stats={
                "memory_to_mound": {"events_processed": 10, "events_failed": 0},
                "mound_to_memory_retrieval": {"events_processed": 5, "events_failed": 0},
            },
        )
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["summary"]["inbound_events_processed"] == 10
        assert body["summary"]["outbound_events_processed"] == 5

    @pytest.mark.asyncio
    async def test_get_missing_handler_stats_default_to_zero(self):
        h = self._make()
        mgr = _mock_manager(stats={})
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        for name, stat in body["handlers"].items():
            assert stat["events_processed"] == 0
            assert stat["events_failed"] == 0

    @pytest.mark.asyncio
    async def test_get_batch_queue_included(self):
        h = self._make()
        batch = {"pending": 5, "total_flushed": 100}
        mgr = _mock_manager(batch_stats=batch)
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["batch_queue"] == batch

    @pytest.mark.asyncio
    async def test_get_adapters_dict(self):
        h = self._make()
        mgr = _mock_manager()
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        expected_adapters = [
            "ranking",
            "rlm",
            "continuum",
            "belief",
            "insights",
            "evidence",
            "consensus",
            "critique",
        ]
        for adapter in expected_adapters:
            assert adapter in body["adapters"]

    @pytest.mark.asyncio
    async def test_get_import_error(self):
        h = self._make()
        with patch.dict("sys.modules", {"aragora.events.cross_subscribers": None}):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_os_error(self):
        h = self._make()
        with patch(_PATCH_MGR, side_effect=OSError("disk fail")):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_key_error(self):
        h = self._make()
        with patch(_PATCH_MGR, side_effect=KeyError("k")):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 500


# ============================================================================
# CrossPollinationKMSyncHandler
# ============================================================================


class TestCrossPollinationKMSyncHandler:
    """Tests for POST /api/v1/cross-pollination/km/sync."""

    def _make(self):
        return CrossPollinationKMSyncHandler(server_context={})

    def test_routes(self):
        h = self._make()
        assert "/api/v1/cross-pollination/km/sync" in h.ROUTES

    @pytest.mark.asyncio
    async def test_post_both_adapters_with_data(self):
        h = self._make()
        mgr = _mock_manager(flush_count=5)

        mock_ranking = MagicMock()
        mock_ranking.get_stats.return_value = {
            "total_expertise_records": 10,
            "domains": {"ml": 5, "nlp": 5},
        }

        mock_rlm = MagicMock()
        mock_rlm.get_stats.return_value = {"total_patterns": 8}

        with (
            patch(_PATCH_MGR, return_value=mgr),
            patch(_PATCH_RANKING, return_value=mock_ranking),
            patch(_PATCH_RLM, return_value=mock_rlm),
            patch(_PATCH_RECORD_SYNC),
        ):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert body["status"] == "ok"
        assert body["results"]["ranking"]["status"] == "synced"
        assert body["results"]["ranking"]["records"] == 10
        assert body["results"]["rlm"]["status"] == "synced"
        assert body["results"]["rlm"]["patterns"] == 8
        assert body["batches_flushed"] == 5
        assert "duration_ms" in body

    @pytest.mark.asyncio
    async def test_post_empty_adapters(self):
        h = self._make()
        mgr = _mock_manager()

        mock_ranking = MagicMock()
        mock_ranking.get_stats.return_value = {"total_expertise_records": 0}

        mock_rlm = MagicMock()
        mock_rlm.get_stats.return_value = {"total_patterns": 0}

        with (
            patch(_PATCH_MGR, return_value=mgr),
            patch(_PATCH_RANKING, return_value=mock_ranking),
            patch(_PATCH_RLM, return_value=mock_rlm),
            patch(_PATCH_RECORD_SYNC),
        ):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert body["results"]["ranking"]["status"] == "empty"
        assert body["results"]["rlm"]["status"] == "empty"

    @pytest.mark.asyncio
    async def test_post_ranking_import_error(self):
        h = self._make()
        mgr = _mock_manager()

        mock_rlm = MagicMock()
        mock_rlm.get_stats.return_value = {"total_patterns": 0}

        import aragora.knowledge.mound.adapters as _adapters_pkg

        _orig = _adapters_pkg.RankingAdapter
        delattr(_adapters_pkg, "RankingAdapter")
        try:
            with (
                patch(_PATCH_MGR, return_value=mgr),
                patch(_PATCH_RLM, return_value=mock_rlm),
                patch(_PATCH_RECORD_SYNC),
            ):
                result = await h.post.__wrapped__(h)
        finally:
            _adapters_pkg.RankingAdapter = _orig
        body = _body(result)
        assert body["results"]["ranking"]["status"] == "unavailable"
        assert body["status"] == "ok"

    @pytest.mark.asyncio
    async def test_post_rlm_import_error(self):
        h = self._make()
        mgr = _mock_manager()

        mock_ranking = MagicMock()
        mock_ranking.get_stats.return_value = {"total_expertise_records": 0}

        with (
            patch(_PATCH_MGR, return_value=mgr),
            patch(_PATCH_RANKING, return_value=mock_ranking),
            patch.dict("sys.modules", {"aragora.knowledge.mound.adapters.rlm_adapter": None}),
            patch(_PATCH_RECORD_SYNC),
        ):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert body["results"]["rlm"]["status"] == "unavailable"

    @pytest.mark.asyncio
    async def test_post_ranking_runtime_error(self):
        h = self._make()
        mgr = _mock_manager()

        mock_ranking = MagicMock()
        mock_ranking.get_stats.side_effect = ValueError("bad data")

        mock_rlm = MagicMock()
        mock_rlm.get_stats.return_value = {"total_patterns": 0}

        with (
            patch(_PATCH_MGR, return_value=mgr),
            patch(_PATCH_RANKING, return_value=mock_ranking),
            patch(_PATCH_RLM, return_value=mock_rlm),
            patch(_PATCH_RECORD_SYNC),
        ):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert body["results"]["ranking"]["status"] == "error"

    @pytest.mark.asyncio
    async def test_post_rlm_runtime_error(self):
        h = self._make()
        mgr = _mock_manager()

        mock_ranking = MagicMock()
        mock_ranking.get_stats.return_value = {"total_expertise_records": 0}

        mock_rlm = MagicMock()
        mock_rlm.get_stats.side_effect = OSError("disk")

        with (
            patch(_PATCH_MGR, return_value=mgr),
            patch(_PATCH_RANKING, return_value=mock_ranking),
            patch(_PATCH_RLM, return_value=mock_rlm),
            patch(_PATCH_RECORD_SYNC),
        ):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert body["results"]["rlm"]["status"] == "error"

    @pytest.mark.asyncio
    async def test_post_uses_cached_ranking_adapter(self):
        h = self._make()
        mgr = _mock_manager()
        cached_adapter = MagicMock()
        cached_adapter.get_stats.return_value = {"total_expertise_records": 3, "domains": {}}
        mgr._ranking_adapter = cached_adapter

        mock_rlm = MagicMock()
        mock_rlm.get_stats.return_value = {"total_patterns": 0}

        with (
            patch(_PATCH_MGR, return_value=mgr),
            patch(_PATCH_RLM, return_value=mock_rlm),
            patch(_PATCH_RECORD_SYNC),
        ):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert body["results"]["ranking"]["status"] == "synced"
        cached_adapter.get_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_uses_cached_rlm_adapter(self):
        h = self._make()
        mgr = _mock_manager()
        cached_rlm = MagicMock()
        cached_rlm.get_stats.return_value = {"total_patterns": 2}
        mgr._rlm_adapter = cached_rlm

        mock_ranking = MagicMock()
        mock_ranking.get_stats.return_value = {"total_expertise_records": 0}

        with (
            patch(_PATCH_MGR, return_value=mgr),
            patch(_PATCH_RANKING, return_value=mock_ranking),
            patch(_PATCH_RECORD_SYNC),
        ):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert body["results"]["rlm"]["status"] == "synced"
        cached_rlm.get_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_metrics_import_error_silent(self):
        """Metrics recording ImportError is silently ignored."""
        h = self._make()
        mgr = _mock_manager()

        mock_ranking = MagicMock()
        mock_ranking.get_stats.return_value = {"total_expertise_records": 0}
        mock_rlm = MagicMock()
        mock_rlm.get_stats.return_value = {"total_patterns": 0}

        with (
            patch(_PATCH_MGR, return_value=mgr),
            patch(_PATCH_RANKING, return_value=mock_ranking),
            patch(_PATCH_RLM, return_value=mock_rlm),
            patch.dict("sys.modules", {"aragora.server.prometheus_cross_pollination": None}),
        ):
            result = await h.post.__wrapped__(h)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_post_manager_import_error(self):
        h = self._make()
        with patch.dict("sys.modules", {"aragora.events.cross_subscribers": None}):
            result = await h.post.__wrapped__(h)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_post_manager_os_error(self):
        h = self._make()
        with patch(_PATCH_MGR, side_effect=OSError("fail")):
            result = await h.post.__wrapped__(h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_post_duration_ms_is_number(self):
        h = self._make()
        mgr = _mock_manager()
        mock_ranking = MagicMock()
        mock_ranking.get_stats.return_value = {"total_expertise_records": 0}
        mock_rlm = MagicMock()
        mock_rlm.get_stats.return_value = {"total_patterns": 0}

        with (
            patch(_PATCH_MGR, return_value=mgr),
            patch(_PATCH_RANKING, return_value=mock_ranking),
            patch(_PATCH_RLM, return_value=mock_rlm),
            patch(_PATCH_RECORD_SYNC),
        ):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert isinstance(body["duration_ms"], (int, float))

    @pytest.mark.asyncio
    async def test_post_ranking_domains_included(self):
        h = self._make()
        mgr = _mock_manager()
        mock_ranking = MagicMock()
        mock_ranking.get_stats.return_value = {
            "total_expertise_records": 5,
            "domains": {"security": 2, "ops": 3},
        }
        mock_rlm = MagicMock()
        mock_rlm.get_stats.return_value = {"total_patterns": 0}

        with (
            patch(_PATCH_MGR, return_value=mgr),
            patch(_PATCH_RANKING, return_value=mock_ranking),
            patch(_PATCH_RLM, return_value=mock_rlm),
            patch(_PATCH_RECORD_SYNC),
        ):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert "domains" in body["results"]["ranking"]
        assert set(body["results"]["ranking"]["domains"]) == {"security", "ops"}

    @pytest.mark.asyncio
    async def test_post_ranking_attribute_error(self):
        h = self._make()
        mgr = _mock_manager()

        mock_ranking = MagicMock()
        mock_ranking.get_stats.side_effect = AttributeError("no attr")

        mock_rlm = MagicMock()
        mock_rlm.get_stats.return_value = {"total_patterns": 0}

        with (
            patch(_PATCH_MGR, return_value=mgr),
            patch(_PATCH_RANKING, return_value=mock_ranking),
            patch(_PATCH_RLM, return_value=mock_rlm),
            patch(_PATCH_RECORD_SYNC),
        ):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert body["results"]["ranking"]["status"] == "error"

    @pytest.mark.asyncio
    async def test_post_rlm_key_error(self):
        h = self._make()
        mgr = _mock_manager()

        mock_ranking = MagicMock()
        mock_ranking.get_stats.return_value = {"total_expertise_records": 0}

        mock_rlm = MagicMock()
        mock_rlm.get_stats.side_effect = KeyError("k")

        with (
            patch(_PATCH_MGR, return_value=mgr),
            patch(_PATCH_RANKING, return_value=mock_ranking),
            patch(_PATCH_RLM, return_value=mock_rlm),
            patch(_PATCH_RECORD_SYNC),
        ):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert body["results"]["rlm"]["status"] == "error"


# ============================================================================
# CrossPollinationKMStalenessHandler
# ============================================================================


class TestCrossPollinationKMStalenessHandler:
    """Tests for POST /api/v1/cross-pollination/km/staleness-check."""

    def _make(self):
        return CrossPollinationKMStalenessHandler(server_context={})

    def test_routes(self):
        h = self._make()
        assert "/api/v1/cross-pollination/km/staleness-check" in h.ROUTES

    @pytest.mark.asyncio
    async def test_post_with_stale_nodes(self):
        h = self._make()
        mock_mound = MagicMock()
        mock_detector = MagicMock()
        mock_detector.get_stale_nodes = AsyncMock(return_value=["node1", "node2", "node3"])

        with (
            patch(_PATCH_MOUND, return_value=mock_mound),
            patch(_PATCH_STALENESS, return_value=mock_detector),
            patch(_PATCH_STALENESS_CFG),
            patch(_PATCH_RECORD_STALE),
        ):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert body["status"] == "ok"
        assert body["stale_nodes"] == 3
        assert body["threshold"] == 0.7
        assert "duration_ms" in body
        assert body["workspace_id"] == "default"

    @pytest.mark.asyncio
    async def test_post_no_stale_nodes(self):
        h = self._make()
        mock_mound = MagicMock()
        mock_detector = MagicMock()
        mock_detector.get_stale_nodes = AsyncMock(return_value=[])

        with (
            patch(_PATCH_MOUND, return_value=mock_mound),
            patch(_PATCH_STALENESS, return_value=mock_detector),
            patch(_PATCH_STALENESS_CFG),
            patch(_PATCH_RECORD_STALE),
        ):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert body["stale_nodes"] == 0

    @pytest.mark.asyncio
    async def test_post_null_stale_nodes(self):
        h = self._make()
        mock_mound = MagicMock()
        mock_detector = MagicMock()
        mock_detector.get_stale_nodes = AsyncMock(return_value=None)

        with (
            patch(_PATCH_MOUND, return_value=mock_mound),
            patch(_PATCH_STALENESS, return_value=mock_detector),
            patch(_PATCH_STALENESS_CFG),
            patch(_PATCH_RECORD_STALE),
        ):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert body["stale_nodes"] == 0

    @pytest.mark.asyncio
    async def test_post_mound_not_available(self):
        h = self._make()
        with (
            patch(_PATCH_MOUND, return_value=None),
            patch(_PATCH_STALENESS),
            patch(_PATCH_STALENESS_CFG),
        ):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert body["status"] == "ok"
        assert body["stale_nodes"] == 0
        assert "not available" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_post_import_error_staleness_module(self):
        h = self._make()
        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": None,
                "aragora.knowledge.mound.staleness": None,
            },
        ):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert body["status"] == "ok"
        assert body["stale_nodes"] == 0
        assert "not available" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_post_metrics_import_error_silent(self):
        h = self._make()
        mock_mound = MagicMock()
        mock_detector = MagicMock()
        mock_detector.get_stale_nodes = AsyncMock(return_value=[])

        with (
            patch(_PATCH_MOUND, return_value=mock_mound),
            patch(_PATCH_STALENESS, return_value=mock_detector),
            patch(_PATCH_STALENESS_CFG),
            patch.dict("sys.modules", {"aragora.server.prometheus_cross_pollination": None}),
        ):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert body["status"] == "ok"

    @pytest.mark.asyncio
    async def test_post_value_error(self):
        h = self._make()
        # The outer try catches ValueError
        with (
            patch(_PATCH_MOUND, side_effect=ValueError("bad")),
            patch(_PATCH_STALENESS),
            patch(_PATCH_STALENESS_CFG),
        ):
            result = await h.post.__wrapped__(h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_post_os_error(self):
        h = self._make()
        with (
            patch(_PATCH_MOUND, side_effect=OSError("disk")),
            patch(_PATCH_STALENESS),
            patch(_PATCH_STALENESS_CFG),
        ):
            result = await h.post.__wrapped__(h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_post_key_error(self):
        h = self._make()
        with (
            patch(_PATCH_MOUND, side_effect=KeyError("k")),
            patch(_PATCH_STALENESS),
            patch(_PATCH_STALENESS_CFG),
        ):
            result = await h.post.__wrapped__(h)
        assert _status(result) == 500


# ============================================================================
# CrossPollinationKMCultureHandler
# ============================================================================


class TestCrossPollinationKMCultureHandler:
    """Tests for GET /api/v1/cross-pollination/km/culture."""

    def _make(self):
        return CrossPollinationKMCultureHandler(server_context={})

    def test_routes(self):
        h = self._make()
        assert "/api/v1/cross-pollination/km/culture" in h.ROUTES

    @pytest.mark.asyncio
    async def test_get_with_accumulator(self):
        h = self._make()
        mock_mound = MagicMock()
        mock_accumulator = MagicMock()
        mock_accumulator.get_patterns_summary.return_value = {
            "patterns": [{"type": "consensus_style", "count": 5}],
            "total": 5,
        }
        mock_mound._culture_accumulator = mock_accumulator

        with patch(_PATCH_MOUND, return_value=mock_mound):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["status"] == "ok"
        assert body["workspace_id"] == "default"
        assert body["total"] == 5

    @pytest.mark.asyncio
    async def test_get_mound_none(self):
        h = self._make()
        with patch(_PATCH_MOUND, return_value=None):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["status"] == "ok"
        assert body["patterns"] == []
        assert "not available" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_get_mound_without_accumulator_attr(self):
        h = self._make()
        mock_mound = MagicMock(spec=[])  # Empty spec means no attributes
        with patch(_PATCH_MOUND, return_value=mock_mound):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["status"] == "ok"
        assert body["patterns"] == []

    @pytest.mark.asyncio
    async def test_get_accumulator_is_none(self):
        h = self._make()
        mock_mound = MagicMock()
        mock_mound._culture_accumulator = None

        with patch(_PATCH_MOUND, return_value=mock_mound):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["status"] == "ok"
        assert body["patterns"] == []
        assert "not initialized" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_get_import_error(self):
        h = self._make()
        with patch.dict("sys.modules", {"aragora.knowledge.mound": None}):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["status"] == "ok"
        assert body["patterns"] == []
        assert "not available" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_get_workspace_from_request(self):
        """When handler has a request attribute with query, it uses workspace_id."""
        h = self._make()
        mock_req = MagicMock()
        mock_req.query.get.return_value = "custom-ws"
        h.request = mock_req

        mock_mound = MagicMock()
        mock_mound._culture_accumulator = None

        with patch(_PATCH_MOUND, return_value=mock_mound):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["workspace_id"] == "custom-ws"

    @pytest.mark.asyncio
    async def test_get_workspace_default_no_request(self):
        h = self._make()
        # Ensure no .request attribute
        if hasattr(h, "request"):
            delattr(h, "request")

        mock_mound = MagicMock()
        mock_mound._culture_accumulator = None

        with patch(_PATCH_MOUND, return_value=mock_mound):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["workspace_id"] == "default"

    @pytest.mark.asyncio
    async def test_get_key_error(self):
        h = self._make()
        with patch(_PATCH_MOUND, side_effect=KeyError("k")):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_os_error(self):
        h = self._make()
        with patch(_PATCH_MOUND, side_effect=OSError("io")):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_value_error(self):
        h = self._make()
        with patch(_PATCH_MOUND, side_effect=ValueError("v")):
            result = await h.get.__wrapped__(h)
        assert _status(result) == 500


# ============================================================================
# register_routes
# ============================================================================


class TestRegisterRoutes:
    """Tests for the register_routes helper function."""

    def test_register_with_add_route(self):
        router = MagicMock()
        router.add_route = MagicMock()
        del router.add_api_route

        register_routes(router, server_context={})

        assert router.add_route.call_count == 9

    def test_register_with_add_api_route(self):
        router = MagicMock(spec=[])
        router.add_api_route = MagicMock()

        register_routes(router, server_context={})

        assert router.add_api_route.call_count == 9

    def test_register_with_add_route_prefers_add_route(self):
        router = MagicMock()
        router.add_route = MagicMock()
        router.add_api_route = MagicMock()

        register_routes(router, server_context={})

        assert router.add_route.call_count == 9
        assert router.add_api_route.call_count == 0

    def test_register_with_none_context(self):
        router = MagicMock()
        router.add_route = MagicMock()
        del router.add_api_route

        register_routes(router, server_context=None)
        assert router.add_route.call_count == 9

    def test_register_routes_include_all_paths(self):
        router = MagicMock()
        router.add_route = MagicMock()
        del router.add_api_route

        register_routes(router, server_context={})

        calls = router.add_route.call_args_list
        paths = [c[0][1] for c in calls]

        assert "/api/v1/cross-pollination/stats" in paths
        assert "/api/v1/cross-pollination/subscribers" in paths
        assert "/api/v1/cross-pollination/bridge" in paths
        assert "/api/v1/cross-pollination/metrics" in paths
        assert "/api/v1/cross-pollination/km" in paths
        assert "/api/v1/cross-pollination/reset" in paths
        assert "/api/v1/cross-pollination/km/sync" in paths
        assert "/api/v1/cross-pollination/km/staleness-check" in paths
        assert "/api/v1/cross-pollination/km/culture" in paths

    def test_register_correct_methods(self):
        router = MagicMock()
        router.add_route = MagicMock()
        del router.add_api_route

        register_routes(router, server_context={})

        calls = router.add_route.call_args_list
        method_path = {c[0][1]: c[0][0] for c in calls}

        assert method_path["/api/v1/cross-pollination/stats"] == "GET"
        assert method_path["/api/v1/cross-pollination/subscribers"] == "GET"
        assert method_path["/api/v1/cross-pollination/bridge"] == "GET"
        assert method_path["/api/v1/cross-pollination/metrics"] == "GET"
        assert method_path["/api/v1/cross-pollination/km"] == "GET"
        assert method_path["/api/v1/cross-pollination/km/culture"] == "GET"
        assert method_path["/api/v1/cross-pollination/reset"] == "POST"
        assert method_path["/api/v1/cross-pollination/km/sync"] == "POST"
        assert method_path["/api/v1/cross-pollination/km/staleness-check"] == "POST"

    def test_register_handles_route_error(self):
        """register_routes should not raise even if add_route fails."""
        router = MagicMock()
        router.add_route = MagicMock(side_effect=ValueError("duplicate route"))
        del router.add_api_route

        register_routes(router, server_context={})

    def test_register_handles_type_error(self):
        router = MagicMock()
        router.add_route = MagicMock(side_effect=TypeError("bad"))
        del router.add_api_route

        register_routes(router, server_context={})

    def test_register_handles_os_error(self):
        router = MagicMock()
        router.add_route = MagicMock(side_effect=OSError("fail"))
        del router.add_api_route

        register_routes(router, server_context={})

    def test_register_handles_attribute_error(self):
        router = MagicMock()
        router.add_route = MagicMock(side_effect=AttributeError("fail"))
        del router.add_api_route

        register_routes(router, server_context={})


# ============================================================================
# Rate Limiter
# ============================================================================


class TestRateLimiter:
    """Tests for the module-level rate limiter."""

    def test_limiter_exists(self):
        assert _cross_pollination_limiter is not None

    def test_limiter_rpm(self):
        assert _cross_pollination_limiter.rpm == 60

    def test_limiter_has_is_allowed(self):
        assert hasattr(_cross_pollination_limiter, "is_allowed")

    def test_limiter_allows_requests(self):
        assert _cross_pollination_limiter.is_allowed("test-ip")

    def test_limiter_tracks_requests(self):
        for _ in range(5):
            _cross_pollination_limiter.is_allowed("track-ip")
        assert len(_cross_pollination_limiter._buckets["track-ip"]) > 0


# ============================================================================
# Module exports
# ============================================================================


class TestModuleExports:
    """Tests for __all__ exports."""

    def test_all_handlers_exported(self):
        from aragora.server.handlers import cross_pollination

        expected = [
            "CrossPollinationStatsHandler",
            "CrossPollinationSubscribersHandler",
            "CrossPollinationBridgeHandler",
            "CrossPollinationMetricsHandler",
            "CrossPollinationResetHandler",
            "CrossPollinationKMHandler",
            "CrossPollinationKMSyncHandler",
            "CrossPollinationKMStalenessHandler",
            "CrossPollinationKMCultureHandler",
            "register_routes",
        ]
        for name in expected:
            assert name in cross_pollination.__all__

    def test_all_count(self):
        from aragora.server.handlers import cross_pollination

        assert len(cross_pollination.__all__) == 10


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Miscellaneous edge case tests."""

    @pytest.mark.asyncio
    async def test_stats_handler_large_subscriber_set(self):
        h = CrossPollinationStatsHandler()
        stats = {
            f"sub_{i}": {"events_processed": i * 10, "events_failed": i, "enabled": i % 2 == 0}
            for i in range(100)
        }
        mgr = _mock_manager(stats=stats)
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["summary"]["total_subscribers"] == 100
        assert body["summary"]["total_events_processed"] == sum(i * 10 for i in range(100))

    @pytest.mark.asyncio
    async def test_km_handler_all_15_handlers_present(self):
        h = CrossPollinationKMHandler(server_context={})
        mgr = _mock_manager()
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert len(body["handlers"]) == 15

    @pytest.mark.asyncio
    async def test_km_handler_specific_handler_names(self):
        h = CrossPollinationKMHandler(server_context={})
        mgr = _mock_manager()
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        expected_handlers = [
            "memory_to_mound",
            "mound_to_memory_retrieval",
            "belief_to_mound",
            "mound_to_belief",
            "rlm_to_mound",
            "mound_to_rlm",
            "elo_to_mound",
            "mound_to_team_selection",
            "insight_to_mound",
            "flip_to_mound",
            "mound_to_trickster",
            "culture_to_debate",
            "staleness_to_debate",
            "provenance_to_mound",
            "mound_to_provenance",
        ]
        for name in expected_handlers:
            assert name in body["handlers"], f"Missing handler: {name}"

    @pytest.mark.asyncio
    async def test_km_sync_sets_adapter_on_manager(self):
        """When no cached adapter, the sync handler creates and caches it."""
        h = CrossPollinationKMSyncHandler(server_context={})
        mgr = _mock_manager()

        mock_ranking = MagicMock()
        mock_ranking.get_stats.return_value = {"total_expertise_records": 0}
        mock_rlm = MagicMock()
        mock_rlm.get_stats.return_value = {"total_patterns": 0}

        with (
            patch(_PATCH_MGR, return_value=mgr),
            patch(_PATCH_RANKING, return_value=mock_ranking),
            patch(_PATCH_RLM, return_value=mock_rlm),
            patch(_PATCH_RECORD_SYNC),
        ):
            result = await h.post.__wrapped__(h)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_culture_handler_summary_spread(self):
        """Culture handler spreads summary dict into response."""
        h = CrossPollinationKMCultureHandler(server_context={})
        mock_mound = MagicMock()
        mock_accumulator = MagicMock()
        mock_accumulator.get_patterns_summary.return_value = {
            "patterns": [{"name": "p1"}],
            "total": 1,
            "categories": {"debate": 1},
        }
        mock_mound._culture_accumulator = mock_accumulator

        with patch(_PATCH_MOUND, return_value=mock_mound):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["categories"] == {"debate": 1}
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_staleness_message_includes_count(self):
        h = CrossPollinationKMStalenessHandler(server_context={})
        mock_mound = MagicMock()
        mock_detector = MagicMock()
        mock_detector.get_stale_nodes = AsyncMock(return_value=["a", "b"])

        with (
            patch(_PATCH_MOUND, return_value=mock_mound),
            patch(_PATCH_STALENESS, return_value=mock_detector),
            patch(_PATCH_STALENESS_CFG),
            patch(_PATCH_RECORD_STALE),
        ):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert "2" in body["message"]

    @pytest.mark.asyncio
    async def test_subscribers_handler_with_named_function(self):
        h = CrossPollinationSubscribersHandler(server_context={})
        evt = MagicMock()
        evt.value = "test_event"

        def my_handler(event):
            pass

        mgr = MagicMock()
        mgr._subscribers = {evt: [("named_sub", my_handler)]}

        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["subscribers"][0]["handler"] == "my_handler"

    @pytest.mark.asyncio
    async def test_stats_all_disabled_subscribers(self):
        h = CrossPollinationStatsHandler()
        stats = {
            "s1": {"events_processed": 5, "events_failed": 0, "enabled": False},
            "s2": {"events_processed": 3, "events_failed": 0, "enabled": False},
        }
        mgr = _mock_manager(stats=stats)
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["summary"]["enabled_subscribers"] == 0
        assert body["summary"]["total_subscribers"] == 2

    @pytest.mark.asyncio
    async def test_stats_all_enabled_subscribers(self):
        h = CrossPollinationStatsHandler()
        stats = {
            "s1": {"events_processed": 5, "events_failed": 0, "enabled": True},
            "s2": {"events_processed": 3, "events_failed": 0, "enabled": True},
        }
        mgr = _mock_manager(stats=stats)
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["summary"]["enabled_subscribers"] == 2

    @pytest.mark.asyncio
    async def test_bridge_single_mapping(self):
        h = CrossPollinationBridgeHandler()
        mock_map = {"only_event": MagicMock(value="only_stream")}
        with patch(_PATCH_BRIDGE, mock_map):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        assert body["mapped_event_count"] == 1
        assert "only_event" in body["event_mappings"]

    @pytest.mark.asyncio
    async def test_reset_handler_body_message(self):
        h = CrossPollinationResetHandler(server_context={})
        mgr = _mock_manager()
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.post.__wrapped__(h)
        body = _body(result)
        assert body["message"] == "Cross-subscriber statistics reset"

    @pytest.mark.asyncio
    async def test_km_handler_inbound_outbound_handler_counts(self):
        h = CrossPollinationKMHandler(server_context={})
        mgr = _mock_manager()
        with patch(_PATCH_MGR, return_value=mgr):
            result = await h.get.__wrapped__(h)
        body = _body(result)
        # 6 outbound (mound_to_*), 9 inbound
        assert body["summary"]["outbound_handlers"] == 6
        assert body["summary"]["inbound_handlers"] == 9
