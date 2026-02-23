"""Tests for ExportOperationsMixin (aragora/server/handlers/knowledge_base/mound/export.py).

Covers all routes and behavior of the export operations mixin:
- GET  /api/v1/knowledge/mound/export/d3       - Export graph as D3 JSON
- GET  /api/v1/knowledge/mound/export/graphml   - Export graph as GraphML XML
- POST /api/v1/knowledge/mound/index/repository - Index a repository

Error cases: missing mound, export failures, invalid bodies, missing fields,
repository not found, import errors, and parameter clamping.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.export import (
    ExportOperationsMixin,
)
from aragora.server.handlers.utils.responses import HandlerResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_TOKEN = "test-token-123"

_RUN_ASYNC_PATCH = (
    "aragora.server.handlers.knowledge_base.mound.export._run_async"
)


def _body(result) -> dict | str:
    """Extract JSON body dict (or raw string for XML) from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return -1
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Autouse fixture: bypass @require_auth by making auth_config accept our token
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _bypass_require_auth(monkeypatch):
    """Patch auth_config so the @require_auth decorator accepts our test token."""
    from aragora.server import auth as auth_module

    monkeypatch.setattr(auth_module.auth_config, "api_token", _TEST_TOKEN)
    monkeypatch.setattr(
        auth_module.auth_config, "validate_token", lambda token: token == _TEST_TOKEN
    )


# ---------------------------------------------------------------------------
# Mock HTTP handler
# ---------------------------------------------------------------------------


@dataclass
class MockHTTPHandler:
    """Lightweight mock HTTP handler for export tests."""

    command: str = "GET"
    path: str = ""
    headers: dict[str, str] = field(
        default_factory=lambda: {
            "User-Agent": "test-agent",
            "Authorization": f"Bearer {_TEST_TOKEN}",
            "Content-Length": "0",
        }
    )
    client_address: tuple = ("127.0.0.1", 12345)
    rfile: Any = field(default_factory=lambda: io.BytesIO(b""))

    @classmethod
    def get(cls) -> MockHTTPHandler:
        return cls(command="GET")

    @classmethod
    def post(cls, body: dict | None = None) -> MockHTTPHandler:
        if body is not None:
            raw = json.dumps(body).encode("utf-8")
            return cls(
                command="POST",
                headers={
                    "User-Agent": "test-agent",
                    "Authorization": f"Bearer {_TEST_TOKEN}",
                    "Content-Length": str(len(raw)),
                },
                rfile=io.BytesIO(raw),
            )
        return cls(command="POST")


# ---------------------------------------------------------------------------
# Concrete test class combining the mixin with stubs
# ---------------------------------------------------------------------------


class ExportTestHandler(ExportOperationsMixin):
    """Concrete handler for testing the export mixin."""

    def __init__(self, mound=None):
        self._mound_instance = mound

    def _get_mound(self):
        return self._mound_instance

    def require_auth_or_error(self, handler):
        """Mock auth that always succeeds."""
        user = MagicMock()
        user.authenticated = True
        user.user_id = "test-user-001"
        return user, None


class ExportTestHandlerAuthFail(ExportOperationsMixin):
    """Concrete handler where auth always fails."""

    def __init__(self, mound=None):
        self._mound_instance = mound

    def _get_mound(self):
        return self._mound_instance

    def require_auth_or_error(self, handler):
        """Mock auth that always fails."""
        from aragora.server.handlers.utils.responses import error_response

        return None, error_response("Authentication required", 401)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mound():
    """Create a MagicMock KnowledgeMound."""
    return MagicMock()


@pytest.fixture
def handler(mock_mound):
    """Handler with a live mound."""
    return ExportTestHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Handler with no mound (returns None)."""
    return ExportTestHandler(mound=None)


@pytest.fixture
def handler_auth_fail(mock_mound):
    """Handler where auth always fails."""
    return ExportTestHandlerAuthFail(mound=mock_mound)


# ===========================================================================
# D3 Export Tests
# ===========================================================================


class TestHandleExportD3:
    """Tests for _handle_export_d3."""

    def test_d3_export_success(self, handler, mock_mound):
        """Basic successful D3 export returns nodes and links."""
        d3_result = {
            "nodes": [{"id": "n1", "label": "Node 1"}],
            "links": [{"source": "n1", "target": "n2"}],
        }
        with patch(_RUN_ASYNC_PATCH, return_value=d3_result):
            result = handler._handle_export_d3({})
        assert _status(result) == 200
        body = _body(result)
        assert body["format"] == "d3"
        assert body["nodes"] == d3_result["nodes"]
        assert body["links"] == d3_result["links"]
        assert body["total_nodes"] == 1
        assert body["total_links"] == 1

    def test_d3_export_empty_graph(self, handler, mock_mound):
        """D3 export with empty graph returns zero counts."""
        d3_result = {"nodes": [], "links": []}
        with patch(_RUN_ASYNC_PATCH, return_value=d3_result):
            result = handler._handle_export_d3({})
        assert _status(result) == 200
        body = _body(result)
        assert body["total_nodes"] == 0
        assert body["total_links"] == 0

    def test_d3_export_with_start_node_id(self, handler, mock_mound):
        """D3 export passes start_node_id to mound."""
        d3_result = {"nodes": [{"id": "n5"}], "links": []}
        with patch(_RUN_ASYNC_PATCH, return_value=d3_result) as mock_run:
            result = handler._handle_export_d3({"start_node_id": "n5"})
        assert _status(result) == 200
        # Verify export_graph_d3 was called with start_node_id
        mock_mound.export_graph_d3.assert_called_once()
        call_kwargs = mock_mound.export_graph_d3.call_args
        assert call_kwargs[1]["start_node_id"] == "n5" or call_kwargs.kwargs.get("start_node_id") == "n5"

    def test_d3_export_custom_depth(self, handler, mock_mound):
        """D3 export respects custom depth parameter."""
        d3_result = {"nodes": [], "links": []}
        with patch(_RUN_ASYNC_PATCH, return_value=d3_result):
            result = handler._handle_export_d3({"depth": "7"})
        assert _status(result) == 200
        call_kwargs = mock_mound.export_graph_d3.call_args
        # depth should be 7 (within 1-10 range)
        assert call_kwargs.kwargs.get("depth") == 7 or call_kwargs[1].get("depth") == 7

    def test_d3_export_custom_limit(self, handler, mock_mound):
        """D3 export respects custom limit parameter."""
        d3_result = {"nodes": [], "links": []}
        with patch(_RUN_ASYNC_PATCH, return_value=d3_result):
            result = handler._handle_export_d3({"limit": "250"})
        assert _status(result) == 200
        call_kwargs = mock_mound.export_graph_d3.call_args
        assert call_kwargs.kwargs.get("limit") == 250 or call_kwargs[1].get("limit") == 250

    def test_d3_export_depth_clamped_min(self, handler, mock_mound):
        """D3 export clamps depth to minimum of 1."""
        d3_result = {"nodes": [], "links": []}
        with patch(_RUN_ASYNC_PATCH, return_value=d3_result):
            result = handler._handle_export_d3({"depth": "0"})
        assert _status(result) == 200
        call_kwargs = mock_mound.export_graph_d3.call_args
        assert call_kwargs.kwargs.get("depth") == 1 or call_kwargs[1].get("depth") == 1

    def test_d3_export_depth_clamped_max(self, handler, mock_mound):
        """D3 export clamps depth to maximum of 10."""
        d3_result = {"nodes": [], "links": []}
        with patch(_RUN_ASYNC_PATCH, return_value=d3_result):
            result = handler._handle_export_d3({"depth": "999"})
        assert _status(result) == 200
        call_kwargs = mock_mound.export_graph_d3.call_args
        assert call_kwargs.kwargs.get("depth") == 10 or call_kwargs[1].get("depth") == 10

    def test_d3_export_limit_clamped_min(self, handler, mock_mound):
        """D3 export clamps limit to minimum of 1."""
        d3_result = {"nodes": [], "links": []}
        with patch(_RUN_ASYNC_PATCH, return_value=d3_result):
            result = handler._handle_export_d3({"limit": "-5"})
        assert _status(result) == 200
        call_kwargs = mock_mound.export_graph_d3.call_args
        assert call_kwargs.kwargs.get("limit") == 1 or call_kwargs[1].get("limit") == 1

    def test_d3_export_limit_clamped_max(self, handler, mock_mound):
        """D3 export clamps limit to maximum of 500."""
        d3_result = {"nodes": [], "links": []}
        with patch(_RUN_ASYNC_PATCH, return_value=d3_result):
            result = handler._handle_export_d3({"limit": "9999"})
        assert _status(result) == 200
        call_kwargs = mock_mound.export_graph_d3.call_args
        assert call_kwargs.kwargs.get("limit") == 500 or call_kwargs[1].get("limit") == 500

    def test_d3_export_default_params(self, handler, mock_mound):
        """D3 export uses default depth=3 and limit=100 when not provided."""
        d3_result = {"nodes": [], "links": []}
        with patch(_RUN_ASYNC_PATCH, return_value=d3_result):
            result = handler._handle_export_d3({})
        assert _status(result) == 200
        call_kwargs = mock_mound.export_graph_d3.call_args
        assert call_kwargs.kwargs.get("depth") == 3 or call_kwargs[1].get("depth") == 3
        assert call_kwargs.kwargs.get("limit") == 100 or call_kwargs[1].get("limit") == 100

    def test_d3_export_no_mound_returns_503(self, handler_no_mound):
        """D3 export returns 503 when Knowledge Mound is not available."""
        result = handler_no_mound._handle_export_d3({})
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body.get("error", "").lower()

    def test_d3_export_key_error(self, handler, mock_mound):
        """D3 export returns 500 on KeyError."""
        with patch(_RUN_ASYNC_PATCH, side_effect=KeyError("missing_key")):
            result = handler._handle_export_d3({})
        assert _status(result) == 500

    def test_d3_export_value_error(self, handler, mock_mound):
        """D3 export returns 500 on ValueError."""
        with patch(_RUN_ASYNC_PATCH, side_effect=ValueError("bad value")):
            result = handler._handle_export_d3({})
        assert _status(result) == 500

    def test_d3_export_os_error(self, handler, mock_mound):
        """D3 export returns 500 on OSError."""
        with patch(_RUN_ASYNC_PATCH, side_effect=OSError("disk error")):
            result = handler._handle_export_d3({})
        assert _status(result) == 500

    def test_d3_export_type_error(self, handler, mock_mound):
        """D3 export returns 500 on TypeError."""
        with patch(_RUN_ASYNC_PATCH, side_effect=TypeError("type mismatch")):
            result = handler._handle_export_d3({})
        assert _status(result) == 500

    def test_d3_export_runtime_error(self, handler, mock_mound):
        """D3 export returns 500 on RuntimeError."""
        with patch(_RUN_ASYNC_PATCH, side_effect=RuntimeError("unexpected")):
            result = handler._handle_export_d3({})
        assert _status(result) == 500

    def test_d3_export_multiple_nodes_and_links(self, handler, mock_mound):
        """D3 export correctly counts multiple nodes and links."""
        d3_result = {
            "nodes": [{"id": f"n{i}"} for i in range(5)],
            "links": [{"source": f"n{i}", "target": f"n{i+1}"} for i in range(4)],
        }
        with patch(_RUN_ASYNC_PATCH, return_value=d3_result):
            result = handler._handle_export_d3({})
        assert _status(result) == 200
        body = _body(result)
        assert body["total_nodes"] == 5
        assert body["total_links"] == 4

    def test_d3_export_start_node_id_none(self, handler, mock_mound):
        """D3 export passes None for start_node_id when not provided."""
        d3_result = {"nodes": [], "links": []}
        with patch(_RUN_ASYNC_PATCH, return_value=d3_result):
            result = handler._handle_export_d3({})
        assert _status(result) == 200
        call_kwargs = mock_mound.export_graph_d3.call_args
        assert call_kwargs.kwargs.get("start_node_id") is None or call_kwargs[1].get("start_node_id") is None

    def test_d3_export_content_type_json(self, handler, mock_mound):
        """D3 export returns application/json content type."""
        d3_result = {"nodes": [], "links": []}
        with patch(_RUN_ASYNC_PATCH, return_value=d3_result):
            result = handler._handle_export_d3({})
        assert result.content_type == "application/json"


# ===========================================================================
# GraphML Export Tests
# ===========================================================================


class TestHandleExportGraphML:
    """Tests for _handle_export_graphml."""

    def test_graphml_export_success(self, handler, mock_mound):
        """Basic successful GraphML export returns XML content."""
        graphml_xml = '<?xml version="1.0"?><graphml><graph></graph></graphml>'
        with patch(_RUN_ASYNC_PATCH, return_value=graphml_xml):
            result = handler._handle_export_graphml({})
        assert _status(result) == 200
        assert result.content_type == "application/xml"
        raw_body = result.body.decode("utf-8") if isinstance(result.body, bytes) else result.body
        assert "<graphml>" in raw_body

    def test_graphml_export_content_is_bytes(self, handler, mock_mound):
        """GraphML export body is encoded as UTF-8 bytes."""
        graphml_xml = "<graphml></graphml>"
        with patch(_RUN_ASYNC_PATCH, return_value=graphml_xml):
            result = handler._handle_export_graphml({})
        assert isinstance(result.body, bytes)
        assert result.body == graphml_xml.encode("utf-8")

    def test_graphml_export_with_start_node_id(self, handler, mock_mound):
        """GraphML export passes start_node_id to mound."""
        graphml_xml = "<graphml></graphml>"
        with patch(_RUN_ASYNC_PATCH, return_value=graphml_xml):
            result = handler._handle_export_graphml({"start_node_id": "node-42"})
        assert _status(result) == 200
        mock_mound.export_graph_graphml.assert_called_once()
        call_kwargs = mock_mound.export_graph_graphml.call_args
        assert call_kwargs.kwargs.get("start_node_id") == "node-42" or call_kwargs[1].get("start_node_id") == "node-42"

    def test_graphml_export_custom_depth(self, handler, mock_mound):
        """GraphML export respects custom depth parameter."""
        graphml_xml = "<graphml></graphml>"
        with patch(_RUN_ASYNC_PATCH, return_value=graphml_xml):
            result = handler._handle_export_graphml({"depth": "5"})
        assert _status(result) == 200
        call_kwargs = mock_mound.export_graph_graphml.call_args
        assert call_kwargs.kwargs.get("depth") == 5 or call_kwargs[1].get("depth") == 5

    def test_graphml_export_custom_limit(self, handler, mock_mound):
        """GraphML export respects custom limit parameter."""
        graphml_xml = "<graphml></graphml>"
        with patch(_RUN_ASYNC_PATCH, return_value=graphml_xml):
            result = handler._handle_export_graphml({"limit": "300"})
        assert _status(result) == 200
        call_kwargs = mock_mound.export_graph_graphml.call_args
        assert call_kwargs.kwargs.get("limit") == 300 or call_kwargs[1].get("limit") == 300

    def test_graphml_export_depth_clamped_min(self, handler, mock_mound):
        """GraphML export clamps depth to minimum of 1."""
        graphml_xml = "<graphml></graphml>"
        with patch(_RUN_ASYNC_PATCH, return_value=graphml_xml):
            result = handler._handle_export_graphml({"depth": "-1"})
        assert _status(result) == 200
        call_kwargs = mock_mound.export_graph_graphml.call_args
        assert call_kwargs.kwargs.get("depth") == 1 or call_kwargs[1].get("depth") == 1

    def test_graphml_export_depth_clamped_max(self, handler, mock_mound):
        """GraphML export clamps depth to maximum of 10."""
        graphml_xml = "<graphml></graphml>"
        with patch(_RUN_ASYNC_PATCH, return_value=graphml_xml):
            result = handler._handle_export_graphml({"depth": "50"})
        assert _status(result) == 200
        call_kwargs = mock_mound.export_graph_graphml.call_args
        assert call_kwargs.kwargs.get("depth") == 10 or call_kwargs[1].get("depth") == 10

    def test_graphml_export_limit_clamped_max(self, handler, mock_mound):
        """GraphML export clamps limit to maximum of 500."""
        graphml_xml = "<graphml></graphml>"
        with patch(_RUN_ASYNC_PATCH, return_value=graphml_xml):
            result = handler._handle_export_graphml({"limit": "1000"})
        assert _status(result) == 200
        call_kwargs = mock_mound.export_graph_graphml.call_args
        assert call_kwargs.kwargs.get("limit") == 500 or call_kwargs[1].get("limit") == 500

    def test_graphml_export_no_mound_returns_503(self, handler_no_mound):
        """GraphML export returns 503 when Knowledge Mound is not available."""
        result = handler_no_mound._handle_export_graphml({})
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body.get("error", "").lower()

    def test_graphml_export_key_error(self, handler, mock_mound):
        """GraphML export returns 500 on KeyError."""
        with patch(_RUN_ASYNC_PATCH, side_effect=KeyError("missing")):
            result = handler._handle_export_graphml({})
        assert _status(result) == 500

    def test_graphml_export_value_error(self, handler, mock_mound):
        """GraphML export returns 500 on ValueError."""
        with patch(_RUN_ASYNC_PATCH, side_effect=ValueError("invalid")):
            result = handler._handle_export_graphml({})
        assert _status(result) == 500

    def test_graphml_export_os_error(self, handler, mock_mound):
        """GraphML export returns 500 on OSError."""
        with patch(_RUN_ASYNC_PATCH, side_effect=OSError("io fail")):
            result = handler._handle_export_graphml({})
        assert _status(result) == 500

    def test_graphml_export_type_error(self, handler, mock_mound):
        """GraphML export returns 500 on TypeError."""
        with patch(_RUN_ASYNC_PATCH, side_effect=TypeError("nope")):
            result = handler._handle_export_graphml({})
        assert _status(result) == 500

    def test_graphml_export_runtime_error(self, handler, mock_mound):
        """GraphML export returns 500 on RuntimeError."""
        with patch(_RUN_ASYNC_PATCH, side_effect=RuntimeError("crash")):
            result = handler._handle_export_graphml({})
        assert _status(result) == 500

    def test_graphml_export_unicode_content(self, handler, mock_mound):
        """GraphML export handles Unicode content correctly."""
        graphml_xml = '<graphml><node label="\u00e9\u00e8\u00ea"/></graphml>'
        with patch(_RUN_ASYNC_PATCH, return_value=graphml_xml):
            result = handler._handle_export_graphml({})
        assert _status(result) == 200
        decoded = result.body.decode("utf-8")
        assert "\u00e9\u00e8\u00ea" in decoded

    def test_graphml_export_default_params(self, handler, mock_mound):
        """GraphML export uses default depth=3 and limit=100."""
        graphml_xml = "<graphml></graphml>"
        with patch(_RUN_ASYNC_PATCH, return_value=graphml_xml):
            result = handler._handle_export_graphml({})
        assert _status(result) == 200
        call_kwargs = mock_mound.export_graph_graphml.call_args
        assert call_kwargs.kwargs.get("depth") == 3 or call_kwargs[1].get("depth") == 3
        assert call_kwargs.kwargs.get("limit") == 100 or call_kwargs[1].get("limit") == 100


# ===========================================================================
# Index Repository Tests
# ===========================================================================


class TestHandleIndexRepository:
    """Tests for _handle_index_repository."""

    _CRAWLER_PATCH = "aragora.connectors.repository_crawler.RepositoryCrawler"
    _CONFIG_PATCH = "aragora.connectors.repository_crawler.CrawlConfig"

    def test_index_repo_success(self, handler, mock_mound):
        """Successful repository indexing returns expected fields."""
        body = {"repo_path": "/tmp/my-repo"}
        http_handler = MockHTTPHandler.post(body)

        mock_crawl_result = MagicMock()
        mock_crawl_result.repository_name = "my-repo"
        mock_crawl_result.repository_path = "/tmp/my-repo"
        mock_crawl_result.total_files = 42
        mock_crawl_result.total_lines = 1000
        mock_crawl_result.total_bytes = 50000
        mock_crawl_result.file_type_counts = {".py": 30, ".md": 12}
        mock_crawl_result.symbol_counts = {"functions": 50, "classes": 10}
        mock_crawl_result.crawl_duration_ms = 123.4
        mock_crawl_result.errors = []
        mock_crawl_result.warnings = []
        mock_crawl_result.git_info = {"branch": "main"}

        mock_crawler_instance = MagicMock()

        def run_async_side_effect(coro):
            # First call: crawl, second call: index_to_mound
            return coro

        call_count = 0

        def run_async_dispatch(coro):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_crawl_result
            return 42  # nodes_created

        with patch(_RUN_ASYNC_PATCH, side_effect=run_async_dispatch), \
             patch(
                 "aragora.connectors.repository_crawler.RepositoryCrawler",
                 return_value=mock_crawler_instance,
             ), \
             patch(
                 "aragora.connectors.repository_crawler.CrawlConfig",
             ):
            result = handler._handle_index_repository(http_handler)

        assert _status(result) == 200
        resp = _body(result)
        assert resp["status"] == "completed"
        assert resp["repository"] == "my-repo"
        assert resp["repository_path"] == "/tmp/my-repo"
        assert resp["workspace_id"] == "default"
        assert resp["total_files"] == 42
        assert resp["total_lines"] == 1000
        assert resp["total_bytes"] == 50000
        assert resp["nodes_created"] == 42
        assert resp["file_type_counts"] == {".py": 30, ".md": 12}
        assert resp["crawl_duration_ms"] == 123.4
        assert resp["git_info"] == {"branch": "main"}

    def test_index_repo_custom_workspace_id(self, handler, mock_mound):
        """Repository indexing uses provided workspace_id."""
        body = {"repo_path": "/tmp/repo", "workspace_id": "ws-custom"}
        http_handler = MockHTTPHandler.post(body)

        mock_crawl_result = MagicMock()
        mock_crawl_result.repository_name = "repo"
        mock_crawl_result.repository_path = "/tmp/repo"
        mock_crawl_result.total_files = 1
        mock_crawl_result.total_lines = 10
        mock_crawl_result.total_bytes = 100
        mock_crawl_result.file_type_counts = {}
        mock_crawl_result.symbol_counts = {}
        mock_crawl_result.crawl_duration_ms = 10.0
        mock_crawl_result.errors = []
        mock_crawl_result.warnings = []
        mock_crawl_result.git_info = None

        call_count = 0

        def run_async_dispatch(coro):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_crawl_result
            return 1

        with patch(_RUN_ASYNC_PATCH, side_effect=run_async_dispatch), \
             patch(
                 "aragora.connectors.repository_crawler.RepositoryCrawler",
             ) as mock_crawler_cls, \
             patch(
                 "aragora.connectors.repository_crawler.CrawlConfig",
             ):
            result = handler._handle_index_repository(http_handler)

        assert _status(result) == 200
        resp = _body(result)
        assert resp["workspace_id"] == "ws-custom"

    def test_index_repo_auth_failure(self, handler_auth_fail, mock_mound):
        """Repository indexing returns 401 when auth fails."""
        body = {"repo_path": "/tmp/repo"}
        http_handler = MockHTTPHandler.post(body)
        result = handler_auth_fail._handle_index_repository(http_handler)
        assert _status(result) == 401

    def test_index_repo_empty_body(self, handler):
        """Repository indexing returns 400 when body is empty (Content-Length: 0)."""
        http_handler = MockHTTPHandler(
            command="POST",
            headers={
                "User-Agent": "test-agent",
                "Authorization": f"Bearer {_TEST_TOKEN}",
                "Content-Length": "0",
            },
            rfile=io.BytesIO(b""),
        )
        result = handler._handle_index_repository(http_handler)
        assert _status(result) == 400
        body = _body(result)
        assert "body required" in body.get("error", "").lower()

    def test_index_repo_invalid_json(self, handler):
        """Repository indexing returns 400 when body is invalid JSON."""
        raw = b"not-json-at-all"
        http_handler = MockHTTPHandler(
            command="POST",
            headers={
                "User-Agent": "test-agent",
                "Authorization": f"Bearer {_TEST_TOKEN}",
                "Content-Length": str(len(raw)),
            },
            rfile=io.BytesIO(raw),
        )
        result = handler._handle_index_repository(http_handler)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid" in body.get("error", "").lower()

    def test_index_repo_missing_repo_path(self, handler):
        """Repository indexing returns 400 when repo_path is missing."""
        body = {"workspace_id": "ws-1"}
        http_handler = MockHTTPHandler.post(body)
        result = handler._handle_index_repository(http_handler)
        assert _status(result) == 400
        body_resp = _body(result)
        assert "repo_path" in body_resp.get("error", "").lower()

    def test_index_repo_empty_repo_path(self, handler):
        """Repository indexing returns 400 when repo_path is empty string."""
        body = {"repo_path": ""}
        http_handler = MockHTTPHandler.post(body)
        result = handler._handle_index_repository(http_handler)
        assert _status(result) == 400
        body_resp = _body(result)
        assert "repo_path" in body_resp.get("error", "").lower()

    def test_index_repo_no_mound_returns_503(self, handler_no_mound):
        """Repository indexing returns 503 when mound is not available."""
        body = {"repo_path": "/tmp/repo"}
        http_handler = MockHTTPHandler.post(body)
        result = handler_no_mound._handle_index_repository(http_handler)
        assert _status(result) == 503
        resp = _body(result)
        assert "not available" in resp.get("error", "").lower()

    def test_index_repo_file_not_found(self, handler, mock_mound):
        """Repository indexing returns 404 when repo path doesn't exist."""
        body = {"repo_path": "/nonexistent/path"}
        http_handler = MockHTTPHandler.post(body)

        with patch(_RUN_ASYNC_PATCH, side_effect=FileNotFoundError("no such directory")), \
             patch(
                 "aragora.connectors.repository_crawler.RepositoryCrawler",
             ), \
             patch(
                 "aragora.connectors.repository_crawler.CrawlConfig",
             ):
            result = handler._handle_index_repository(http_handler)
        assert _status(result) == 404
        resp = _body(result)
        assert "not found" in resp.get("error", "").lower()

    def test_index_repo_import_error(self, handler, mock_mound):
        """Repository indexing returns 500 on ImportError (crawler not available)."""
        body = {"repo_path": "/tmp/repo"}
        http_handler = MockHTTPHandler.post(body)

        with patch(
            "aragora.connectors.repository_crawler.RepositoryCrawler",
            side_effect=ImportError("no module"),
        ), patch(
            "aragora.connectors.repository_crawler.CrawlConfig",
        ):
            result = handler._handle_index_repository(http_handler)
        # ImportError is caught by the except block or the handle_errors decorator
        assert _status(result) == 500

    def test_index_repo_runtime_error(self, handler, mock_mound):
        """Repository indexing returns 500 on RuntimeError during crawl."""
        body = {"repo_path": "/tmp/repo"}
        http_handler = MockHTTPHandler.post(body)

        with patch(_RUN_ASYNC_PATCH, side_effect=RuntimeError("crawl failed")), \
             patch(
                 "aragora.connectors.repository_crawler.RepositoryCrawler",
             ), \
             patch(
                 "aragora.connectors.repository_crawler.CrawlConfig",
             ):
            result = handler._handle_index_repository(http_handler)
        assert _status(result) == 500

    def test_index_repo_os_error(self, handler, mock_mound):
        """Repository indexing returns 500 on OSError during crawl."""
        body = {"repo_path": "/tmp/repo"}
        http_handler = MockHTTPHandler.post(body)

        with patch(_RUN_ASYNC_PATCH, side_effect=OSError("permission denied")), \
             patch(
                 "aragora.connectors.repository_crawler.RepositoryCrawler",
             ), \
             patch(
                 "aragora.connectors.repository_crawler.CrawlConfig",
             ):
            result = handler._handle_index_repository(http_handler)
        assert _status(result) == 500

    def test_index_repo_errors_truncated(self, handler, mock_mound):
        """Repository indexing truncates errors to first 10."""
        body = {"repo_path": "/tmp/repo"}
        http_handler = MockHTTPHandler.post(body)

        mock_crawl_result = MagicMock()
        mock_crawl_result.repository_name = "repo"
        mock_crawl_result.repository_path = "/tmp/repo"
        mock_crawl_result.total_files = 100
        mock_crawl_result.total_lines = 5000
        mock_crawl_result.total_bytes = 250000
        mock_crawl_result.file_type_counts = {}
        mock_crawl_result.symbol_counts = {}
        mock_crawl_result.crawl_duration_ms = 500.0
        mock_crawl_result.errors = [f"error-{i}" for i in range(20)]
        mock_crawl_result.warnings = [f"warn-{i}" for i in range(15)]
        mock_crawl_result.git_info = None

        call_count = 0

        def run_async_dispatch(coro):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_crawl_result
            return 50

        with patch(_RUN_ASYNC_PATCH, side_effect=run_async_dispatch), \
             patch(
                 "aragora.connectors.repository_crawler.RepositoryCrawler",
             ), \
             patch(
                 "aragora.connectors.repository_crawler.CrawlConfig",
             ):
            result = handler._handle_index_repository(http_handler)

        assert _status(result) == 200
        resp = _body(result)
        assert len(resp["errors"]) == 10
        assert len(resp["warnings"]) == 10

    def test_index_repo_no_errors_empty_list(self, handler, mock_mound):
        """Repository indexing returns empty list when no errors/warnings."""
        body = {"repo_path": "/tmp/repo"}
        http_handler = MockHTTPHandler.post(body)

        mock_crawl_result = MagicMock()
        mock_crawl_result.repository_name = "repo"
        mock_crawl_result.repository_path = "/tmp/repo"
        mock_crawl_result.total_files = 5
        mock_crawl_result.total_lines = 100
        mock_crawl_result.total_bytes = 5000
        mock_crawl_result.file_type_counts = {}
        mock_crawl_result.symbol_counts = {}
        mock_crawl_result.crawl_duration_ms = 50.0
        mock_crawl_result.errors = None
        mock_crawl_result.warnings = None
        mock_crawl_result.git_info = None

        call_count = 0

        def run_async_dispatch(coro):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_crawl_result
            return 5

        with patch(_RUN_ASYNC_PATCH, side_effect=run_async_dispatch), \
             patch(
                 "aragora.connectors.repository_crawler.RepositoryCrawler",
             ), \
             patch(
                 "aragora.connectors.repository_crawler.CrawlConfig",
             ):
            result = handler._handle_index_repository(http_handler)

        assert _status(result) == 200
        resp = _body(result)
        assert resp["errors"] == []
        assert resp["warnings"] == []

    def test_index_repo_custom_crawl_config(self, handler, mock_mound):
        """Repository indexing passes custom config parameters."""
        body = {
            "repo_path": "/tmp/repo",
            "include_patterns": ["*.py"],
            "exclude_patterns": ["**/test/**"],
            "max_file_size_bytes": 500_000,
            "max_files": 5_000,
            "extract_symbols": False,
            "extract_dependencies": False,
            "extract_docstrings": False,
            "incremental": False,
        }
        http_handler = MockHTTPHandler.post(body)

        mock_crawl_result = MagicMock()
        mock_crawl_result.repository_name = "repo"
        mock_crawl_result.repository_path = "/tmp/repo"
        mock_crawl_result.total_files = 10
        mock_crawl_result.total_lines = 200
        mock_crawl_result.total_bytes = 10000
        mock_crawl_result.file_type_counts = {".py": 10}
        mock_crawl_result.symbol_counts = {}
        mock_crawl_result.crawl_duration_ms = 80.0
        mock_crawl_result.errors = []
        mock_crawl_result.warnings = []
        mock_crawl_result.git_info = None

        call_count = 0

        def run_async_dispatch(coro):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_crawl_result
            return 10

        with patch(_RUN_ASYNC_PATCH, side_effect=run_async_dispatch), \
             patch(
                 "aragora.connectors.repository_crawler.RepositoryCrawler",
             ), \
             patch(
                 "aragora.connectors.repository_crawler.CrawlConfig",
             ) as mock_config_cls:
            result = handler._handle_index_repository(http_handler)

        assert _status(result) == 200
        # Verify CrawlConfig was called with custom parameters
        mock_config_cls.assert_called_once()
        config_kwargs = mock_config_cls.call_args
        assert config_kwargs.kwargs.get("include_patterns") == ["*.py"] or \
            (config_kwargs[1] and config_kwargs[1].get("include_patterns") == ["*.py"])

    def test_index_repo_default_exclude_patterns(self, handler, mock_mound):
        """Repository indexing uses default exclude patterns when not specified."""
        body = {"repo_path": "/tmp/repo"}
        http_handler = MockHTTPHandler.post(body)

        mock_crawl_result = MagicMock()
        mock_crawl_result.repository_name = "repo"
        mock_crawl_result.repository_path = "/tmp/repo"
        mock_crawl_result.total_files = 1
        mock_crawl_result.total_lines = 10
        mock_crawl_result.total_bytes = 100
        mock_crawl_result.file_type_counts = {}
        mock_crawl_result.symbol_counts = {}
        mock_crawl_result.crawl_duration_ms = 5.0
        mock_crawl_result.errors = []
        mock_crawl_result.warnings = []
        mock_crawl_result.git_info = None

        call_count = 0

        def run_async_dispatch(coro):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_crawl_result
            return 1

        with patch(_RUN_ASYNC_PATCH, side_effect=run_async_dispatch), \
             patch(
                 "aragora.connectors.repository_crawler.RepositoryCrawler",
             ), \
             patch(
                 "aragora.connectors.repository_crawler.CrawlConfig",
             ) as mock_config_cls:
            result = handler._handle_index_repository(http_handler)

        assert _status(result) == 200
        config_call = mock_config_cls.call_args
        exclude = config_call.kwargs.get("exclude_patterns") or config_call[1].get("exclude_patterns", [])
        # Should contain the default patterns
        assert "**/node_modules/**" in exclude
        assert "**/.git/**" in exclude
        assert "**/venv/**" in exclude

    def test_index_repo_value_error(self, handler, mock_mound):
        """Repository indexing returns 500 on ValueError during crawl."""
        body = {"repo_path": "/tmp/repo"}
        http_handler = MockHTTPHandler.post(body)

        with patch(_RUN_ASYNC_PATCH, side_effect=ValueError("bad value")), \
             patch(
                 "aragora.connectors.repository_crawler.RepositoryCrawler",
             ), \
             patch(
                 "aragora.connectors.repository_crawler.CrawlConfig",
             ):
            result = handler._handle_index_repository(http_handler)
        assert _status(result) == 500

    def test_index_repo_key_error(self, handler, mock_mound):
        """Repository indexing returns 500 on KeyError during crawl."""
        body = {"repo_path": "/tmp/repo"}
        http_handler = MockHTTPHandler.post(body)

        with patch(_RUN_ASYNC_PATCH, side_effect=KeyError("missing")), \
             patch(
                 "aragora.connectors.repository_crawler.RepositoryCrawler",
             ), \
             patch(
                 "aragora.connectors.repository_crawler.CrawlConfig",
             ):
            result = handler._handle_index_repository(http_handler)
        assert _status(result) == 500
