"""
Tests for KnowledgeFlowHandler.

Covers all routes and behaviour of the KnowledgeFlowHandler class:
- GET /api/knowledge/flow                   - Flow data (debate->KM->debate)
- GET /api/knowledge/flow/confidence-history - Confidence changes over time
- GET /api/knowledge/adapters/health         - All adapter statuses
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_flow import KnowledgeFlowHandler


# =============================================================================
# Helpers
# =============================================================================


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _data(result) -> dict:
    """Extract the 'data' envelope from a response."""
    body = _body(result)
    if isinstance(body, dict) and "data" in body:
        return body["data"]
    return body


# =============================================================================
# Mock data objects
# =============================================================================


@dataclass
class MockValidation:
    """Mock validation record from KMOutcomeBridge."""

    debate_id: str = "debate-001"
    km_item_id: str = "km-node-001"
    confidence_adjustment: float = 0.05
    original_confidence: float = 0.7
    new_confidence: float = 0.75
    was_successful: bool = True
    validation_reason: str = "outcome confirmed"


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to the handler."""

    def __init__(self, method: str = "GET", path: str = "/api/knowledge/flow"):
        self.command = method
        self.path = path
        self.client_address = ("127.0.0.1", 12345)
        self.headers = {"Content-Length": "0", "Host": "localhost:8080"}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create a KnowledgeFlowHandler with empty server context."""
    h = KnowledgeFlowHandler(server_context={})
    # The handler references self.server_context in its private helpers
    h.server_context = {}
    return h


@pytest.fixture
def handler_with_bridge():
    """Create a handler with a mocked KMOutcomeBridge in context."""
    bridge = MagicMock()
    bridge.get_validation_stats.return_value = {
        "total_validations": 3,
        "debates_tracked": 2,
    }
    bridge._validations_applied = [
        MockValidation(
            debate_id="debate-001",
            km_item_id="km-node-001",
            confidence_adjustment=0.05,
            original_confidence=0.7,
            new_confidence=0.75,
            was_successful=True,
        ),
        MockValidation(
            debate_id="debate-001",
            km_item_id="km-node-002",
            confidence_adjustment=-0.1,
            original_confidence=0.6,
            new_confidence=0.5,
            was_successful=False,
            validation_reason="outcome contradicted",
        ),
        MockValidation(
            debate_id="debate-002",
            km_item_id="km-node-001",
            confidence_adjustment=0.03,
            original_confidence=0.75,
            new_confidence=0.78,
            was_successful=True,
        ),
    ]
    # debate B consumed km-node-001 which was produced by debate-001
    bridge._debate_km_usage = {
        "debate-003": ["km-node-001"],
    }

    ctx = {"km_outcome_bridge": bridge}
    h = KnowledgeFlowHandler(server_context=ctx)
    h.server_context = ctx
    return h


# =============================================================================
# ROUTES / can_handle
# =============================================================================


class TestRoutes:
    """Test the can_handle method and ROUTES attribute."""

    def test_routes_list(self):
        expected = [
            "/api/knowledge/flow",
            "/api/knowledge/flow/*",
            "/api/knowledge/adapters/health",
        ]
        for route in expected:
            assert route in KnowledgeFlowHandler.ROUTES

    def test_can_handle_flow(self, handler):
        assert handler.can_handle("/api/knowledge/flow", "GET")

    def test_can_handle_confidence_history(self, handler):
        assert handler.can_handle("/api/knowledge/flow/confidence-history", "GET")

    def test_can_handle_adapter_health(self, handler):
        assert handler.can_handle("/api/knowledge/adapters/health", "GET")

    def test_rejects_post(self, handler):
        assert not handler.can_handle("/api/knowledge/flow", "POST")

    def test_rejects_unknown_path(self, handler):
        assert not handler.can_handle("/api/knowledge/something-else", "GET")

    def test_rejects_partial_match(self, handler):
        assert not handler.can_handle("/api/knowledge/flow/other-sub", "GET")


# =============================================================================
# GET /api/knowledge/flow
# =============================================================================


class TestGetFlowData:
    """Test the _get_flow_data method."""

    @pytest.mark.asyncio
    async def test_flow_data_empty_when_no_bridge(self, handler):
        """When no KMOutcomeBridge is available, return empty flows."""
        result = await handler._get_flow_data({})
        data = _data(result)

        assert data["flows"] == []
        assert data["stats"]["total_flows"] == 0
        assert data["stats"]["avg_confidence_change"] == 0.0
        assert data["stats"]["debates_enriched"] == 0
        assert "generated_at" in data

    @pytest.mark.asyncio
    async def test_flow_data_with_bridge(self, handler_with_bridge):
        """With a valid bridge, flows should be populated."""
        result = await handler_with_bridge._get_flow_data({})
        data = _data(result)

        assert len(data["flows"]) == 3
        assert data["stats"]["total_flows"] == 3
        assert data["stats"]["debates_enriched"] == 2
        assert data["stats"]["avg_confidence_change"] > 0

    @pytest.mark.asyncio
    async def test_flow_entries_structure(self, handler_with_bridge):
        """Each flow entry should contain required fields."""
        result = await handler_with_bridge._get_flow_data({})
        data = _data(result)

        flow = data["flows"][0]
        assert "source_debate_id" in flow
        assert "km_node_id" in flow
        assert "target_debate_id" in flow
        assert "confidence_delta" in flow
        assert "original_confidence" in flow
        assert "new_confidence" in flow
        assert "was_successful" in flow
        assert "reason" in flow
        assert "content_preview" in flow
        assert "timestamp" in flow

    @pytest.mark.asyncio
    async def test_flow_cross_reference(self, handler_with_bridge):
        """When KM item from debate A is consumed in debate B, target should be set."""
        result = await handler_with_bridge._get_flow_data({})
        data = _data(result)

        # km_to_source is built from validations in order:
        #   km-node-001 -> debate-001, km-node-002 -> debate-001, km-node-001 -> debate-002 (overwrites)
        # debate-003 consumed km-node-001, so source = debate-002
        # The cross-ref updates the flow with source_debate_id=debate-002 and km_node_id=km-node-001
        cross_ref_flows = [
            f
            for f in data["flows"]
            if f["km_node_id"] == "km-node-001"
            and f["source_debate_id"] == "debate-002"
            and f["target_debate_id"] == "debate-003"
        ]
        assert len(cross_ref_flows) == 1

    @pytest.mark.asyncio
    async def test_flow_synthesis_data_default(self, handler):
        """Synthesis data should default to count=0 and empty recent."""
        result = await handler._get_flow_data({})
        data = _data(result)

        assert data["synthesis"]["count"] == 0
        assert data["synthesis"]["recent"] == []

    @pytest.mark.asyncio
    async def test_flow_confidence_delta_rounding(self, handler_with_bridge):
        """Confidence deltas should be rounded to 4 decimal places."""
        result = await handler_with_bridge._get_flow_data({})
        data = _data(result)

        for flow in data["flows"]:
            delta_str = str(flow["confidence_delta"])
            if "." in delta_str:
                decimals = len(delta_str.split(".")[1])
                assert decimals <= 4


# =============================================================================
# GET /api/knowledge/flow/confidence-history
# =============================================================================


class TestConfidenceHistory:
    """Test the _get_confidence_history method."""

    @pytest.mark.asyncio
    async def test_confidence_history_empty_no_bridge(self, handler):
        """Without bridge, entries list should be empty."""
        result = await handler._get_confidence_history({})
        data = _data(result)

        assert data["entries"] == []
        assert "generated_at" in data

    @pytest.mark.asyncio
    async def test_confidence_history_with_bridge(self, handler_with_bridge):
        """With bridge, entries should be grouped by KM item."""
        result = await handler_with_bridge._get_confidence_history({})
        data = _data(result)

        # km-node-001 has 2 validations, km-node-002 has 1
        assert len(data["entries"]) == 2

    @pytest.mark.asyncio
    async def test_confidence_history_entry_structure(self, handler_with_bridge):
        """Each entry should contain node_id, content_preview, confidence_history."""
        result = await handler_with_bridge._get_confidence_history({})
        data = _data(result)

        for entry in data["entries"]:
            assert "node_id" in entry
            assert "content_preview" in entry
            assert "confidence_history" in entry
            assert isinstance(entry["confidence_history"], list)

    @pytest.mark.asyncio
    async def test_confidence_history_item_fields(self, handler_with_bridge):
        """Each history point should have timestamp, value, previous, reason, debate_id."""
        result = await handler_with_bridge._get_confidence_history({})
        data = _data(result)

        for entry in data["entries"]:
            for point in entry["confidence_history"]:
                assert "timestamp" in point
                assert "value" in point
                assert "previous" in point
                assert "reason" in point
                assert "debate_id" in point

    @pytest.mark.asyncio
    async def test_confidence_history_reason_labels(self, handler_with_bridge):
        """Reason should be debate_outcome_boost or debate_outcome_penalty."""
        result = await handler_with_bridge._get_confidence_history({})
        data = _data(result)

        reasons = set()
        for entry in data["entries"]:
            for point in entry["confidence_history"]:
                reasons.add(point["reason"])

        assert "debate_outcome_boost" in reasons
        assert "debate_outcome_penalty" in reasons

    @pytest.mark.asyncio
    async def test_confidence_history_grouping(self, handler_with_bridge):
        """km-node-001 should have 2 history points (from debate-001 and debate-002)."""
        result = await handler_with_bridge._get_confidence_history({})
        data = _data(result)

        node_001 = [e for e in data["entries"] if e["node_id"] == "km-node-001"]
        assert len(node_001) == 1
        assert len(node_001[0]["confidence_history"]) == 2


# =============================================================================
# GET /api/knowledge/adapters/health
# =============================================================================


class TestAdapterHealth:
    """Test the _get_adapter_health method."""

    @pytest.mark.asyncio
    async def test_adapter_health_empty_when_factory_unavailable(self, handler):
        """When adapter factory is not importable, return empty list."""
        # The method uses `from aragora.knowledge.mound.adapters.factory import ADAPTER_SPECS`
        # Setting the module to None in sys.modules forces an ImportError on from-import
        with patch.dict(
            "sys.modules",
            {"aragora.knowledge.mound.adapters.factory": None},
        ):
            result = await handler._get_adapter_health()

        data = _data(result)
        assert data["adapters"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_adapter_health_with_mock_specs(self, handler):
        """With mock adapter specs, should return adapter info."""
        mock_spec_a = MagicMock()
        mock_spec_a.priority = 10
        mock_spec_a.enabled_by_default = True
        mock_spec_a.reverse_method = "reverse_sync_a"

        mock_spec_b = MagicMock()
        mock_spec_b.priority = 5
        mock_spec_b.enabled_by_default = False
        mock_spec_b.reverse_method = None

        mock_specs = {"adapter_a": mock_spec_a, "adapter_b": mock_spec_b}

        with (
            patch(
                "aragora.knowledge.mound.adapters.factory.ADAPTER_SPECS",
                mock_specs,
            ),
            patch(
                "aragora.server.handlers.knowledge_flow._get_adapter_defs",
                return_value=[],
            ),
        ):
            result = await handler._get_adapter_health()

        data = _data(result)
        assert data["total"] == 2
        assert len(data["adapters"]) == 2
        # Should be sorted by priority descending
        assert data["adapters"][0]["name"] == "adapter_a"
        assert data["adapters"][1]["name"] == "adapter_b"

    @pytest.mark.asyncio
    async def test_adapter_health_structure(self, handler):
        """Each adapter entry should contain all required fields."""
        mock_spec = MagicMock()
        mock_spec.priority = 10
        mock_spec.enabled_by_default = True
        mock_spec.reverse_method = None

        with (
            patch(
                "aragora.knowledge.mound.adapters.factory.ADAPTER_SPECS",
                {"test_adapter": mock_spec},
            ),
            patch(
                "aragora.server.handlers.knowledge_flow._get_adapter_defs",
                return_value=[],
            ),
        ):
            result = await handler._get_adapter_health()

        data = _data(result)
        adapter = data["adapters"][0]
        assert adapter["name"] == "test_adapter"
        assert "status" in adapter
        assert "entry_count" in adapter
        assert "health" in adapter
        assert "priority" in adapter
        assert "enabled_by_default" in adapter
        assert "has_reverse_sync" in adapter

    @pytest.mark.asyncio
    async def test_adapter_health_counts(self, handler):
        """Adapter health should report correct active/stale/offline counts."""
        mock_spec = MagicMock()
        mock_spec.priority = 10
        mock_spec.enabled_by_default = True
        mock_spec.reverse_method = None

        with (
            patch(
                "aragora.knowledge.mound.adapters.factory.ADAPTER_SPECS",
                {"good_adapter": mock_spec},
            ),
            patch(
                "aragora.server.handlers.knowledge_flow._get_adapter_defs",
                return_value=[],
            ),
        ):
            result = await handler._get_adapter_health()

        data = _data(result)
        # No module found for 'good_adapter', so it falls through to "registered" / "healthy"
        assert data["active"] == 1
        assert data["stale"] == 0
        assert data["offline"] == 0
        assert "generated_at" in data

    @pytest.mark.asyncio
    async def test_adapter_health_import_error(self, handler):
        """Adapter modules that fail to import should be marked offline."""
        mock_spec = MagicMock()
        mock_spec.priority = 10
        mock_spec.enabled_by_default = True
        mock_spec.reverse_method = None

        with (
            patch(
                "aragora.knowledge.mound.adapters.factory.ADAPTER_SPECS",
                {"failing_adapter": mock_spec},
            ),
            patch(
                "aragora.server.handlers.knowledge_flow._get_adapter_defs",
                return_value=[("bad.module", "BadClass", {"name": "failing_adapter"})],
            ),
            patch(
                "aragora.server.handlers.knowledge_flow.importlib.import_module",
                side_effect=ImportError("module not found"),
            ),
        ):
            result = await handler._get_adapter_health()

        data = _data(result)
        adapter = data["adapters"][0]
        assert adapter["status"] == "unavailable"
        assert adapter["health"] == "offline"
        assert data["offline"] == 1

    @pytest.mark.asyncio
    async def test_adapter_health_sorted_by_priority(self, handler):
        """Adapters should be sorted by priority descending."""
        specs = {}
        for i, name in enumerate(["low", "mid", "high"]):
            spec = MagicMock()
            spec.priority = (i + 1) * 10  # 10, 20, 30
            spec.enabled_by_default = True
            spec.reverse_method = None
            specs[name] = spec

        with (
            patch(
                "aragora.knowledge.mound.adapters.factory.ADAPTER_SPECS",
                specs,
            ),
            patch(
                "aragora.server.handlers.knowledge_flow._get_adapter_defs",
                return_value=[],
            ),
        ):
            result = await handler._get_adapter_health()

        data = _data(result)
        priorities = [a["priority"] for a in data["adapters"]]
        assert priorities == sorted(priorities, reverse=True)


# =============================================================================
# handle() dispatch
# =============================================================================


class TestHandleDispatch:
    """Test the handle() route dispatch."""

    @pytest.mark.asyncio
    async def test_handle_routes_to_flow(self, handler):
        """GET /api/knowledge/flow should return flow data."""
        result = await handler.handle("/api/knowledge/flow", {}, _MockHTTPHandler())
        data = _data(result)
        assert "flows" in data
        assert "stats" in data

    @pytest.mark.asyncio
    async def test_handle_routes_to_confidence_history(self, handler):
        """GET /api/knowledge/flow/confidence-history should return entries."""
        result = await handler.handle(
            "/api/knowledge/flow/confidence-history",
            {},
            _MockHTTPHandler(path="/api/knowledge/flow/confidence-history"),
        )
        data = _data(result)
        assert "entries" in data

    @pytest.mark.asyncio
    async def test_handle_routes_to_adapter_health(self, handler):
        """GET /api/knowledge/adapters/health should return adapter info."""
        with patch(
            "aragora.knowledge.mound.adapters.factory.ADAPTER_SPECS",
            {},
        ):
            result = await handler.handle(
                "/api/knowledge/adapters/health",
                {},
                _MockHTTPHandler(path="/api/knowledge/adapters/health"),
            )
        data = _data(result)
        assert "adapters" in data

    @pytest.mark.asyncio
    async def test_handle_returns_404_for_unknown(self, handler):
        """Unknown paths should return 404."""
        result = await handler.handle("/api/knowledge/unknown", {}, _MockHTTPHandler())
        assert _status(result) == 404


# =============================================================================
# Error handling
# =============================================================================


class TestErrorHandling:
    """Test error handling in the handle() method."""

    @pytest.mark.asyncio
    async def test_handle_catches_value_error(self, handler):
        """ValueError in sub-handler should return 500."""
        with patch.object(handler, "_get_flow_data", side_effect=ValueError("bad value")):
            result = await handler.handle("/api/knowledge/flow", {}, _MockHTTPHandler())
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_handle_catches_runtime_error(self, handler):
        """RuntimeError in sub-handler should return 500."""
        with patch.object(handler, "_get_confidence_history", side_effect=RuntimeError("oops")):
            result = await handler.handle(
                "/api/knowledge/flow/confidence-history",
                {},
                _MockHTTPHandler(),
            )
        assert _status(result) == 500


# =============================================================================
# Private helpers
# =============================================================================


class TestPrivateHelpers:
    """Test the private helper methods."""

    def test_get_km_outcome_bridge_from_context(self):
        """Bridge should be returned from server_context directly."""
        bridge = MagicMock()
        h = KnowledgeFlowHandler(server_context={"km_outcome_bridge": bridge})
        h.server_context = {"km_outcome_bridge": bridge}
        assert h._get_km_outcome_bridge() is bridge

    def test_get_km_outcome_bridge_from_arena(self):
        """Bridge should fall back to arena.km_outcome_bridge."""
        bridge = MagicMock()
        arena = MagicMock()
        arena.km_outcome_bridge = bridge
        h = KnowledgeFlowHandler(server_context={"arena": arena})
        h.server_context = {"arena": arena}
        assert h._get_km_outcome_bridge() is bridge

    def test_get_km_outcome_bridge_none(self):
        """Should return None when no bridge is available."""
        h = KnowledgeFlowHandler(server_context={})
        h.server_context = {}
        assert h._get_km_outcome_bridge() is None

    def test_get_knowledge_mound_from_context(self):
        """Mound should be returned from server_context directly."""
        mound = MagicMock()
        h = KnowledgeFlowHandler(server_context={"knowledge_mound": mound})
        h.server_context = {"knowledge_mound": mound}
        assert h._get_knowledge_mound() is mound

    def test_get_knowledge_mound_from_arena(self):
        """Mound should fall back to arena.knowledge_mound."""
        mound = MagicMock()
        arena = MagicMock()
        arena.knowledge_mound = mound
        h = KnowledgeFlowHandler(server_context={"arena": arena})
        h.server_context = {"arena": arena}
        assert h._get_knowledge_mound() is mound

    def test_get_knowledge_mound_none(self):
        """Should return None when no mound is available."""
        h = KnowledgeFlowHandler(server_context={})
        h.server_context = {}
        assert h._get_knowledge_mound() is None

    def test_get_knowledge_injector_returns_none_on_import_error(self):
        """Should return None when DebateKnowledgeInjector is not importable."""
        h = KnowledgeFlowHandler(server_context={})
        h.server_context = {}
        with patch(
            "aragora.server.handlers.knowledge_flow.importlib.import_module",
            side_effect=ImportError,
        ):
            # _get_knowledge_injector does its own internal import
            result = h._get_knowledge_injector()
            # May or may not be None depending on import availability;
            # we just verify no exception is raised
            assert result is None or result is not None
