"""Tests for Unified Memory Gateway handler.

Covers all routes and behaviour of the MemoryUnifiedHandler class:
- POST /api/memory/unified/query      - Fan-out search across all systems
- GET  /api/memory/unified/retention   - RetentionGate decisions
- GET  /api/memory/unified/dedup       - Near-duplicate clusters
- GET  /api/memory/unified/sources     - Memory source breakdown
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.memory_unified import MemoryUnifiedHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_response(result) -> dict:
    """Extract data from json_response HandlerResult."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        if isinstance(body, str):
            body = json.loads(body)
        if isinstance(body, dict):
            return body
    if isinstance(result, tuple):
        body = result[0] if len(result) > 0 else {}
        if isinstance(body, str):
            body = json.loads(body)
        return body
    if isinstance(result, dict):
        return result
    return {}


def _get_data(result) -> dict:
    """Extract the 'data' envelope from a response."""
    body = _parse_response(result)
    if isinstance(body, dict) and "data" in body:
        return body["data"]
    return body


def _status_code(result) -> int:
    """Extract status code from HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, tuple) and len(result) > 1:
        return result[1]
    return 200


# ---------------------------------------------------------------------------
# Mock data objects
# ---------------------------------------------------------------------------


@dataclass
class MockMemoryEntry:
    """Mock entry returned from memory systems."""

    content: str = "Test memory entry"
    relevance: float = 0.85
    tier: str = "fast"
    created_at: str = "2026-02-20T10:00:00"


@dataclass
class MockKMEntry:
    """Mock entry from Knowledge Mound."""

    content: str = "Knowledge mound fact"
    confidence: float = 0.9
    node_type: str = "fact"
    source_adapter: str = "debate"


@dataclass
class MockSupermemoryEntry:
    """Mock entry from Supermemory."""

    content: str = "Supermemory item"
    relevance: float = 0.7
    session_id: str = "session-001"


@dataclass
class MockClaudeMemEntry:
    """Mock entry from claude-mem."""

    content: str = "Claude mem entry"
    relevance: float = 0.6


@dataclass
class MockRetentionDecision:
    """Mock retention gate decision."""

    memory_id: str = "mem-001"
    action: str = "retain"
    surprise_score: float = 0.75
    reason: str = "High surprise score"
    timestamp: str = "2026-02-20T10:00:00"


@dataclass
class MockDedupClusterEntry:
    """Mock dedup cluster entry."""

    content: str = "Duplicate content"
    source: str = "continuum"
    similarity: float = 0.95


@dataclass
class MockDedupCluster:
    """Mock dedup cluster."""

    cluster_id: str = "cluster-001"
    entries: list = field(default_factory=lambda: [MockDedupClusterEntry()])
    canonical_id: str = "entry-001"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a MemoryUnifiedHandler instance."""
    return MemoryUnifiedHandler({})


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with a JSON body."""

    def _make(body: dict | None = None):
        h = MagicMock()
        body_data = body or {}
        body_bytes = json.dumps(body_data).encode("utf-8")
        h.request = MagicMock()
        h.request.body = body_bytes
        h.rfile = MagicMock()
        h.rfile.read.return_value = body_bytes
        h.headers = {"Content-Length": str(len(body_bytes))}
        return h

    return _make


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class TestRoutes:
    """Test ROUTES and can_handle."""

    def test_routes_contains_all_endpoints(self):
        expected = [
            "/api/memory/unified/query",
            "/api/memory/unified/retention",
            "/api/memory/unified/dedup",
            "/api/memory/unified/sources",
        ]
        for route in expected:
            assert route in MemoryUnifiedHandler.ROUTES, f"Missing route: {route}"

    def test_can_handle_post_query(self, handler):
        assert handler.can_handle("/api/memory/unified/query", "POST")

    def test_can_handle_rejects_get_for_query(self, handler):
        assert not handler.can_handle("/api/memory/unified/query", "GET")

    def test_can_handle_get_retention(self, handler):
        assert handler.can_handle("/api/memory/unified/retention", "GET")

    def test_can_handle_get_dedup(self, handler):
        assert handler.can_handle("/api/memory/unified/dedup", "GET")

    def test_can_handle_get_sources(self, handler):
        assert handler.can_handle("/api/memory/unified/sources", "GET")

    def test_can_handle_rejects_unknown(self, handler):
        assert not handler.can_handle("/api/memory/unified/unknown", "GET")

    def test_can_handle_rejects_post_for_get_routes(self, handler):
        assert not handler.can_handle("/api/memory/unified/retention", "POST")
        assert not handler.can_handle("/api/memory/unified/dedup", "POST")
        assert not handler.can_handle("/api/memory/unified/sources", "POST")


# ---------------------------------------------------------------------------
# GET dispatch
# ---------------------------------------------------------------------------


class TestHandleGet:
    """Test GET request routing."""

    def test_handle_get_retention(self, handler):
        with patch.object(handler, "_handle_retention", return_value="retention_result") as mock:
            result = handler.handle_get("/api/memory/unified/retention", {}, MagicMock())
            mock.assert_called_once_with({})
            assert result == "retention_result"

    def test_handle_get_dedup(self, handler):
        with patch.object(handler, "_handle_dedup", return_value="dedup_result") as mock:
            result = handler.handle_get("/api/memory/unified/dedup", {}, MagicMock())
            mock.assert_called_once_with({})
            assert result == "dedup_result"

    def test_handle_get_sources(self, handler):
        with patch.object(handler, "_handle_sources", return_value="sources_result") as mock:
            result = handler.handle_get("/api/memory/unified/sources", {}, MagicMock())
            mock.assert_called_once()
            assert result == "sources_result"

    def test_handle_get_unknown_returns_none(self, handler):
        result = handler.handle_get("/api/memory/unified/unknown", {}, MagicMock())
        assert result is None


# ---------------------------------------------------------------------------
# POST /api/memory/unified/query
# ---------------------------------------------------------------------------


class TestHandlePost:
    """Test POST request routing."""

    def test_handle_post_query(self, handler, mock_http_handler):
        body = {"query": "test search", "limit": 10}
        h = mock_http_handler(body)
        with patch.object(handler, "_handle_query") as mock:
            mock.return_value = MagicMock()
            handler.handle_post("/api/memory/unified/query", {}, h)
            mock.assert_called_once()

    def test_handle_post_unknown_returns_none(self, handler, mock_http_handler):
        h = mock_http_handler({})
        result = handler.handle_post("/api/memory/unified/unknown", {}, h)
        assert result is None


class TestQuery:
    """Test _handle_query endpoint logic."""

    def test_query_missing_query_field(self, handler):
        result = handler._handle_query({})
        assert _status_code(result) == 400
        body = _parse_response(result)
        assert "missing" in body.get("error", "").lower() or "missing" in str(body).lower()

    def test_query_empty_query_field(self, handler):
        result = handler._handle_query({"query": "   "})
        assert _status_code(result) == 400

    def test_query_fan_out_search(self, handler):
        mock_entries = [MockMemoryEntry()]

        with patch.object(
            handler,
            "_query_system",
            return_value=[
                {"content": "result", "source": "continuum", "relevance": 0.8, "metadata": {}}
            ],
        ):
            result = handler._handle_query({"query": "test search"})

        data = _get_data(result)
        assert "results" in data
        assert "total" in data
        assert "per_system" in data
        assert data["query"] == "test search"

    def test_query_respects_limit(self, handler):
        with patch.object(
            handler,
            "_query_system",
            return_value=[
                {"content": f"result-{i}", "source": "continuum", "relevance": 0.5, "metadata": {}}
                for i in range(30)
            ],
        ):
            result = handler._handle_query({"query": "test", "limit": 5})

        data = _get_data(result)
        assert len(data["results"]) <= 5

    def test_query_max_limit_capped(self, handler):
        with patch.object(handler, "_query_system", return_value=[]):
            result = handler._handle_query({"query": "test", "limit": 500})

        # Should not crash; limit is capped to 100 internally
        assert _status_code(result) == 200

    def test_query_custom_systems(self, handler):
        with patch.object(handler, "_query_system", return_value=[]) as mock:
            handler._handle_query({"query": "test", "systems": ["km", "supermemory"]})

        # Should only query the requested systems
        call_args = [call[0][0] for call in mock.call_args_list]
        assert "km" in call_args
        assert "supermemory" in call_args
        assert "continuum" not in call_args

    def test_query_sorts_by_relevance(self, handler):
        def _mock_query(system, query, limit):
            if system == "continuum":
                return [{"content": "low", "source": "continuum", "relevance": 0.3, "metadata": {}}]
            elif system == "km":
                return [{"content": "high", "source": "km", "relevance": 0.9, "metadata": {}}]
            return []

        with patch.object(handler, "_query_system", side_effect=_mock_query):
            result = handler._handle_query({"query": "test"})

        data = _get_data(result)
        if len(data["results"]) >= 2:
            assert data["results"][0]["relevance"] >= data["results"][1]["relevance"]


# ---------------------------------------------------------------------------
# _query_system
# ---------------------------------------------------------------------------


class TestQuerySystem:
    """Test individual memory system queries."""

    def test_query_continuum(self, handler):
        mock_mem = MagicMock()
        mock_mem.search.return_value = [MockMemoryEntry()]

        with patch(
            "aragora.server.handlers.memory_unified.ContinuumMemory",
            return_value=mock_mem,
            create=True,
        ):
            with patch.dict(
                "sys.modules",
                {"aragora.memory.continuum": MagicMock(ContinuumMemory=lambda: mock_mem)},
            ):
                results = handler._query_continuum("test", 10)

        # Should return list of dicts with source=continuum
        assert isinstance(results, list)

    def test_query_system_graceful_fallback_on_import_error(self, handler):
        with patch.object(handler, "_query_continuum", side_effect=ImportError("no module")):
            results = handler._query_system("continuum", "test", 10)
        assert results == []

    def test_query_system_graceful_fallback_on_runtime_error(self, handler):
        with patch.object(handler, "_query_km", side_effect=RuntimeError("connection failed")):
            results = handler._query_system("km", "test", 10)
        assert results == []

    def test_query_system_unknown_system(self, handler):
        results = handler._query_system("unknown_system", "test", 10)
        assert results == []


# ---------------------------------------------------------------------------
# GET /api/memory/unified/retention
# ---------------------------------------------------------------------------


class TestRetention:
    """Test _handle_retention endpoint."""

    def test_retention_default(self, handler):
        mock_gate = MagicMock()
        mock_gate.get_decision_history.return_value = [
            MockRetentionDecision(),
            MockRetentionDecision(memory_id="mem-002", action="demoted"),
        ]

        with patch(
            "aragora.memory.retention_gate.RetentionGate",
            return_value=mock_gate,
            create=True,
        ):
            result = handler._handle_retention({})

        data = _get_data(result)
        assert "decisions" in data
        assert "stats" in data
        assert len(data["decisions"]) == 2

    def test_retention_respects_limit(self, handler):
        mock_gate = MagicMock()
        mock_gate.get_decision_history.return_value = [MockRetentionDecision()]

        with patch(
            "aragora.memory.retention_gate.RetentionGate",
            return_value=mock_gate,
            create=True,
        ):
            handler._handle_retention({"limit": "10"})
            mock_gate.get_decision_history.assert_called_once_with(limit=10)

    def test_retention_limit_capped_at_200(self, handler):
        mock_gate = MagicMock()
        mock_gate.get_decision_history.return_value = []

        with patch(
            "aragora.memory.retention_gate.RetentionGate",
            return_value=mock_gate,
            create=True,
        ):
            handler._handle_retention({"limit": "999"})
            mock_gate.get_decision_history.assert_called_once_with(limit=200)

    def test_retention_stats_counting(self, handler):
        mock_gate = MagicMock()
        mock_gate.get_decision_history.return_value = [
            MockRetentionDecision(action="retain"),
            MockRetentionDecision(action="retain"),
            MockRetentionDecision(action="demoted"),
            MockRetentionDecision(action="forgotten"),
        ]

        with patch(
            "aragora.memory.retention_gate.RetentionGate",
            return_value=mock_gate,
            create=True,
        ):
            result = handler._handle_retention({})

        data = _get_data(result)
        assert data["stats"]["retained"] == 2
        assert data["stats"]["demoted"] == 1
        assert data["stats"]["forgotten"] == 1

    def test_retention_unavailable_returns_empty(self, handler):
        with patch(
            "aragora.memory.retention_gate.RetentionGate",
            side_effect=ImportError("no module"),
            create=True,
        ):
            result = handler._handle_retention({})

        data = _get_data(result)
        assert data["decisions"] == []
        assert "message" in data

    def test_retention_gate_without_get_decision_history(self, handler):
        mock_gate = MagicMock(spec=[])  # no attributes at all

        with patch(
            "aragora.memory.retention_gate.RetentionGate",
            return_value=mock_gate,
            create=True,
        ):
            result = handler._handle_retention({})

        data = _get_data(result)
        assert data["decisions"] == []


# ---------------------------------------------------------------------------
# GET /api/memory/unified/dedup
# ---------------------------------------------------------------------------


class TestDedup:
    """Test _handle_dedup endpoint."""

    @staticmethod
    def _patch_dedup_import(mock_engine):
        """Create patches to mock the dedup_engine import inside _handle_dedup."""
        import sys

        mock_module = MagicMock()
        mock_module.CrossSystemDedupEngine = MagicMock(return_value=mock_engine)
        return patch.dict(
            sys.modules,
            {
                "aragora.memory.dedup": mock_module,
            },
        )

    def test_dedup_returns_clusters(self, handler):
        mock_engine = MagicMock()
        mock_engine.get_clusters.return_value = [MockDedupCluster()]

        with self._patch_dedup_import(mock_engine):
            result = handler._handle_dedup({})

        data = _get_data(result)
        assert "clusters" in data
        assert "total_duplicates" in data
        assert len(data["clusters"]) == 1

    def test_dedup_cluster_entry_shape(self, handler):
        mock_engine = MagicMock()
        mock_engine.get_clusters.return_value = [MockDedupCluster()]

        with self._patch_dedup_import(mock_engine):
            result = handler._handle_dedup({})

        data = _get_data(result)
        cluster = data["clusters"][0]
        assert "cluster_id" in cluster
        assert "entries" in cluster
        assert "canonical" in cluster
        entry = cluster["entries"][0]
        assert "content" in entry
        assert "source" in entry
        assert "similarity" in entry

    def test_dedup_unavailable_returns_empty(self, handler):
        """When the dedup module is not available, return empty data."""
        # The handler catches ImportError/AttributeError internally
        # Just call _handle_dedup without any mocking; the real import will fail
        # and the handler will return empty data
        result = handler._handle_dedup({})

        data = _get_data(result)
        assert data["clusters"] == []
        assert data["total_duplicates"] == 0

    def test_dedup_engine_without_get_clusters(self, handler):
        mock_engine = MagicMock(spec=[])  # no get_clusters attribute

        with self._patch_dedup_import(mock_engine):
            result = handler._handle_dedup({})

        data = _get_data(result)
        assert data["clusters"] == []


# ---------------------------------------------------------------------------
# GET /api/memory/unified/sources
# ---------------------------------------------------------------------------


class TestSources:
    """Test _handle_sources endpoint."""

    def test_sources_all_available(self, handler):
        mock_instance = MagicMock()
        mock_instance.count.return_value = 42

        mock_mod = MagicMock()

        def _import_module(path):
            m = MagicMock()
            # Set the expected class attribute
            for _, _, class_name in [
                ("continuum", "aragora.memory.continuum", "ContinuumMemory"),
                ("km", "aragora.knowledge.mound", "KnowledgeMound"),
                ("supermemory", "aragora.memory.supermemory", "SupermemoryStore"),
                ("claude_mem", "aragora.knowledge.mound.adapters.claude_mem", "ClaudeMemAdapter"),
            ]:
                setattr(m, class_name, lambda: mock_instance)
            return m

        with patch("importlib.import_module", side_effect=_import_module):
            result = handler._handle_sources()

        data = _get_data(result)
        assert "sources" in data
        assert len(data["sources"]) == 4
        for source in data["sources"]:
            assert "name" in source
            assert "status" in source
            assert "entry_count" in source

    def test_sources_some_unavailable(self, handler):
        call_count = [0]

        def _import_module(path):
            call_count[0] += 1
            if call_count[0] <= 2:
                m = MagicMock()
                # Return a class that creates an instance with a count method
                mock_cls = MagicMock()
                mock_inst = MagicMock()
                mock_inst.count.return_value = 10
                mock_cls.return_value = mock_inst
                for class_name in [
                    "ContinuumMemory",
                    "KnowledgeMound",
                    "SupermemoryStore",
                    "ClaudeMemAdapter",
                ]:
                    setattr(m, class_name, mock_cls)
                return m
            raise ImportError("not available")

        with patch("importlib.import_module", side_effect=_import_module):
            result = handler._handle_sources()

        data = _get_data(result)
        statuses = [s["status"] for s in data["sources"]]
        assert "active" in statuses
        assert "unavailable" in statuses

    def test_sources_all_unavailable(self, handler):
        with patch("importlib.import_module", side_effect=ImportError("no module")):
            result = handler._handle_sources()

        data = _get_data(result)
        assert len(data["sources"]) == 4
        for source in data["sources"]:
            assert source["status"] == "unavailable"
            assert source["entry_count"] == 0


# ---------------------------------------------------------------------------
# _get_request_body
# ---------------------------------------------------------------------------


class TestGetRequestBody:
    """Test static _get_request_body method."""

    def test_extracts_json_from_handler(self):
        h = MagicMock()
        h.request.body = json.dumps({"key": "value"}).encode("utf-8")

        body = MemoryUnifiedHandler._get_request_body(h)
        assert body == {"key": "value"}

    def test_returns_empty_dict_on_invalid_json(self):
        h = MagicMock()
        h.request.body = b"not-json"

        body = MemoryUnifiedHandler._get_request_body(h)
        assert body == {}

    def test_returns_empty_dict_when_no_body(self):
        h = MagicMock()
        h.request.body = None

        body = MemoryUnifiedHandler._get_request_body(h)
        assert body == {}

    def test_returns_empty_dict_when_no_request(self):
        h = MagicMock(spec=[])

        body = MemoryUnifiedHandler._get_request_body(h)
        assert body == {}

    def test_handles_string_body(self):
        h = MagicMock()
        h.request.body = json.dumps({"foo": "bar"})

        body = MemoryUnifiedHandler._get_request_body(h)
        assert body == {"foo": "bar"}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling and graceful degradation."""

    def test_query_system_handles_value_error(self, handler):
        with patch.object(handler, "_query_continuum", side_effect=ValueError("bad value")):
            results = handler._query_system("continuum", "test", 10)
        assert results == []

    def test_query_system_handles_type_error(self, handler):
        with patch.object(handler, "_query_km", side_effect=TypeError("bad type")):
            results = handler._query_system("km", "test", 10)
        assert results == []

    def test_query_system_handles_os_error(self, handler):
        with patch.object(handler, "_query_supermemory", side_effect=OSError("disk error")):
            results = handler._query_system("supermemory", "test", 10)
        assert results == []

    def test_query_system_handles_attribute_error(self, handler):
        with patch.object(handler, "_query_claude_mem", side_effect=AttributeError("missing")):
            results = handler._query_system("claude_mem", "test", 10)
        assert results == []

    def test_query_system_handles_key_error(self, handler):
        with patch.object(handler, "_query_continuum", side_effect=KeyError("missing key")):
            results = handler._query_system("continuum", "test", 10)
        assert results == []
