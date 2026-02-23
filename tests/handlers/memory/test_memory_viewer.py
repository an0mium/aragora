"""Tests for MemoryViewerMixin and its integration in MemoryHandler.

Tests cover:
- MemoryViewerMixin._html_response creation
- MemoryViewerMixin._render_viewer HTML rendering
- MemoryHandler routing to /api/v1/memory/viewer
- MemoryHandler routing to /api/v1/memory/search-index
- MemoryHandler routing to /api/v1/memory/search-timeline
- MemoryHandler routing to /api/v1/memory/entries
- MemoryHandler.can_handle path recognition
- Path normalization for legacy /api/memory/ routes
- Rate limiting on viewer endpoint
- Error responses for missing continuum memory
- Error responses for missing parameters
- Search index with hybrid search
- Search index with external sources
- Timeline retrieval with anchor
- Timeline not supported by backend
- Entries retrieval by ID
- Entries retrieval unsupported backend
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Lightweight stubs to avoid real memory system imports
# ---------------------------------------------------------------------------


class _MemoryTier(Enum):
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    GLACIAL = "glacial"


class _MockMemoryEntry:
    """Mock memory entry that behaves like ContinuumMemory entries."""

    def __init__(
        self,
        id: str = "mem-001",
        content: str = "Test memory content",
        tier: _MemoryTier = _MemoryTier.FAST,
        importance: float = 0.75,
        surprise_score: float = 0.3,
        created_at: str = "2026-01-15T12:00:00Z",
        updated_at: str = "2026-01-15T12:00:00Z",
        red_line: bool = False,
        red_line_reason: str = "",
        metadata: dict | None = None,
        memory_id: str | None = None,
    ):
        self.id = id
        self.memory_id = memory_id or id
        self.content = content
        self.tier = tier
        self.importance = importance
        self.surprise_score = surprise_score
        self.created_at = created_at
        self.updated_at = updated_at
        self.red_line = red_line
        self.red_line_reason = red_line_reason
        self.metadata = metadata or {}


class _MockHybridResult:
    """Mock hybrid search result."""

    def __init__(self, memory_id: str, combined_score: float = 0.9):
        self.memory_id = memory_id
        self.combined_score = combined_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _http() -> MagicMock:
    """Return a mock HTTP handler object."""
    mock = MagicMock()
    mock.command = "GET"
    mock.client_address = ("127.0.0.1", 54321)
    mock.headers = {"Content-Length": "2"}
    mock.rfile.read.return_value = b"{}"
    return mock


def _body(result) -> dict[str, Any]:
    """Parse the JSON body out of a HandlerResult."""
    return json.loads(result.body)


def _make_continuum(
    entries: list | None = None,
    has_timeline: bool = True,
    has_get_many: bool = True,
    has_hybrid: bool = False,
) -> MagicMock:
    """Create a mock continuum memory with configurable capabilities."""
    mock = MagicMock()
    mock.retrieve.return_value = entries or []

    if has_timeline:
        mock.get_timeline_entries = MagicMock()
    else:
        # Remove the method entirely so hasattr check fails
        if hasattr(mock, "get_timeline_entries"):
            del mock.get_timeline_entries

    if has_get_many:
        mock.get_many = MagicMock(return_value=entries or [])
    else:
        if hasattr(mock, "get_many"):
            del mock.get_many

    if has_hybrid:
        mock.hybrid_search = MagicMock(return_value=[])
    else:
        if hasattr(mock, "hybrid_search"):
            del mock.hybrid_search

    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a MemoryHandler with mocked dependencies."""
    with patch(
        "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
    ), patch(
        "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
    ), patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    ):
        from aragora.server.handlers.memory.memory import MemoryHandler

        h = MemoryHandler(server_context={})
        return h


@pytest.fixture
def handler_with_continuum():
    """Create a MemoryHandler with a mock continuum memory."""
    entries = [
        _MockMemoryEntry(id="mem-001", content="First memory"),
        _MockMemoryEntry(id="mem-002", content="Second memory"),
    ]
    continuum = _make_continuum(entries=entries)

    with patch(
        "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
    ), patch(
        "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
    ), patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    ):
        from aragora.server.handlers.memory.memory import MemoryHandler

        h = MemoryHandler(server_context={"continuum_memory": continuum})
        return h, continuum


# ============================================================================
# MemoryViewerMixin._html_response Tests
# ============================================================================


class TestHtmlResponse:
    """Tests for MemoryViewerMixin._html_response."""

    def test_html_response_default_status(self, handler):
        """HTML response defaults to 200 status."""
        result = handler._html_response("<h1>Hello</h1>")
        assert result.status_code == 200
        assert result.content_type == "text/html"
        assert result.body == b"<h1>Hello</h1>"

    def test_html_response_custom_status(self, handler):
        """HTML response respects custom status code."""
        result = handler._html_response("<p>Error</p>", status=404)
        assert result.status_code == 404
        assert result.content_type == "text/html"

    def test_html_response_encodes_utf8(self, handler):
        """HTML response encodes unicode content to UTF-8 bytes."""
        result = handler._html_response("<p>Caf\u00e9 \u2603</p>")
        assert result.body == "<p>Caf\u00e9 \u2603</p>".encode("utf-8")

    def test_html_response_empty_string(self, handler):
        """HTML response handles empty string."""
        result = handler._html_response("")
        assert result.status_code == 200
        assert result.body == b""


# ============================================================================
# MemoryViewerMixin._render_viewer Tests
# ============================================================================


class TestRenderViewer:
    """Tests for MemoryViewerMixin._render_viewer."""

    def test_render_viewer_returns_html(self, handler):
        """Viewer renders the full HTML UI."""
        result = handler._render_viewer()
        assert result.status_code == 200
        assert result.content_type == "text/html"

    def test_render_viewer_contains_title(self, handler):
        """Viewer HTML contains the expected title."""
        result = handler._render_viewer()
        html = result.body.decode("utf-8")
        assert "<title>Aragora Memory Viewer</title>" in html

    def test_render_viewer_contains_search_button(self, handler):
        """Viewer HTML contains the search button."""
        result = handler._render_viewer()
        html = result.body.decode("utf-8")
        assert 'id="searchBtn"' in html

    def test_render_viewer_contains_tier_checkboxes(self, handler):
        """Viewer HTML contains tier filter checkboxes."""
        result = handler._render_viewer()
        html = result.body.decode("utf-8")
        for tier in ("fast", "medium", "slow", "glacial"):
            assert f'value="{tier}"' in html

    def test_render_viewer_contains_api_base(self, handler):
        """Viewer HTML references the correct API base path."""
        result = handler._render_viewer()
        html = result.body.decode("utf-8")
        assert '"/api/v1/memory"' in html

    def test_render_viewer_contains_external_options(self, handler):
        """Viewer HTML contains external memory source checkboxes."""
        result = handler._render_viewer()
        html = result.body.decode("utf-8")
        assert 'id="extSupermemory"' in html
        assert 'id="extClaudeMem"' in html

    def test_render_viewer_contains_grid_layout(self, handler):
        """Viewer HTML has the three-panel grid layout."""
        result = handler._render_viewer()
        html = result.body.decode("utf-8")
        assert "Search" in html
        assert "Index" in html
        assert "Timeline" in html

    def test_render_viewer_contains_javascript(self, handler):
        """Viewer HTML includes the interactive JavaScript."""
        result = handler._render_viewer()
        html = result.body.decode("utf-8")
        assert "async function searchIndex()" in html
        assert "async function loadTimeline(" in html
        assert "async function loadEntry(" in html

    def test_render_viewer_contains_entry_panel(self, handler):
        """Viewer HTML contains the entry display panel."""
        result = handler._render_viewer()
        html = result.body.decode("utf-8")
        assert 'id="entry"' in html
        assert "Select a memory to view full content." in html


# ============================================================================
# MemoryHandler.can_handle Tests
# ============================================================================


class TestCanHandle:
    """Tests for MemoryHandler.can_handle route matching."""

    def test_can_handle_viewer_path(self, handler):
        """Handler recognizes the viewer path."""
        assert handler.can_handle("/api/v1/memory/viewer") is True

    def test_can_handle_search_index(self, handler):
        """Handler recognizes search-index path."""
        assert handler.can_handle("/api/v1/memory/search-index") is True

    def test_can_handle_search_timeline(self, handler):
        """Handler recognizes search-timeline path."""
        assert handler.can_handle("/api/v1/memory/search-timeline") is True

    def test_can_handle_entries(self, handler):
        """Handler recognizes entries path."""
        assert handler.can_handle("/api/v1/memory/entries") is True

    def test_can_handle_search(self, handler):
        """Handler recognizes search path."""
        assert handler.can_handle("/api/v1/memory/search") is True

    def test_can_handle_critiques(self, handler):
        """Handler recognizes critiques path."""
        assert handler.can_handle("/api/v1/memory/critiques") is True

    def test_can_handle_continuum_retrieve(self, handler):
        """Handler recognizes continuum retrieve path."""
        assert handler.can_handle("/api/v1/memory/continuum/retrieve") is True

    def test_can_handle_tier_stats(self, handler):
        """Handler recognizes tier-stats path."""
        assert handler.can_handle("/api/v1/memory/tier-stats") is True

    def test_can_handle_pressure(self, handler):
        """Handler recognizes pressure path."""
        assert handler.can_handle("/api/v1/memory/pressure") is True

    def test_can_handle_tiers(self, handler):
        """Handler recognizes tiers path."""
        assert handler.can_handle("/api/v1/memory/tiers") is True

    def test_can_handle_wildcard_memory_subpaths(self, handler):
        """Handler handles arbitrary memory subpaths via wildcard."""
        assert handler.can_handle("/api/v1/memory/some-custom-path") is True

    def test_cannot_handle_non_memory_path(self, handler):
        """Handler rejects non-memory paths."""
        assert handler.can_handle("/api/v1/debates/list") is False
        assert handler.can_handle("/api/v1/agents") is False


# ============================================================================
# Path Normalization Tests
# ============================================================================


class TestPathNormalization:
    """Tests for MemoryHandler._normalize_path."""

    def test_normalize_legacy_path(self, handler):
        """Legacy /api/memory/ paths are normalized to /api/v1/memory/."""
        result = handler._normalize_path("/api/memory/viewer")
        assert result == "/api/v1/memory/viewer"

    def test_normalize_already_versioned_path(self, handler):
        """Already versioned paths remain unchanged."""
        result = handler._normalize_path("/api/v1/memory/viewer")
        assert result == "/api/v1/memory/viewer"

    def test_normalize_legacy_search_index(self, handler):
        """Legacy search-index path is normalized."""
        result = handler._normalize_path("/api/memory/search-index")
        assert result == "/api/v1/memory/search-index"

    def test_normalize_non_memory_path(self, handler):
        """Non-memory paths pass through unchanged."""
        result = handler._normalize_path("/api/v1/debates/list")
        assert result == "/api/v1/debates/list"


# ============================================================================
# Viewer Route Tests (via handler.handle)
# ============================================================================


class TestViewerRoute:
    """Tests for the /api/v1/memory/viewer route through MemoryHandler.handle."""

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    def test_viewer_route_returns_html(self, mock_limiter, handler):
        """GET /api/v1/memory/viewer returns the HTML viewer."""
        mock_limiter.is_allowed.return_value = True
        mock_http = _http()

        result = handler.handle("/api/v1/memory/viewer", {}, mock_http)

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "text/html"
        html = result.body.decode("utf-8")
        assert "Memory Viewer" in html

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    def test_viewer_route_rate_limited(self, mock_limiter, handler):
        """GET /api/v1/memory/viewer returns 429 when rate limited."""
        mock_limiter.is_allowed.return_value = False
        mock_http = _http()

        result = handler.handle("/api/v1/memory/viewer", {}, mock_http)

        assert result is not None
        assert result.status_code == 429

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    def test_viewer_via_legacy_path(self, mock_limiter, handler):
        """GET /api/memory/viewer also returns the HTML viewer after normalization."""
        mock_limiter.is_allowed.return_value = True
        mock_http = _http()

        result = handler.handle("/api/memory/viewer", {}, mock_http)

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "text/html"


# ============================================================================
# Search Index Route Tests
# ============================================================================


class TestSearchIndexRoute:
    """Tests for /api/v1/memory/search-index route."""

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    @patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    )
    def test_search_index_missing_query(self, mock_run, mock_limiter):
        """Search index returns 400 when query param is missing."""
        mock_limiter.is_allowed.return_value = True

        continuum = _make_continuum(entries=[])

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": continuum})
            mock_http = _http()

            result = h.handle("/api/v1/memory/search-index", {}, mock_http)

            assert result is not None
            assert result.status_code == 400
            body = _body(result)
            assert "q" in body.get("error", "").lower() or "query" in body.get("error", "").lower()

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    def test_search_index_no_continuum(self, mock_limiter):
        """Search index returns 503 when continuum not available."""
        mock_limiter.is_allowed.return_value = True

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={})
            mock_http = _http()

            result = h.handle(
                "/api/v1/memory/search-index", {"q": "test"}, mock_http
            )

            assert result is not None
            assert result.status_code == 503

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    @patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    )
    def test_search_index_no_continuum_in_ctx(self, mock_run, mock_limiter):
        """Search index returns 503 when continuum not in context."""
        mock_limiter.is_allowed.return_value = True

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={})
            mock_http = _http()

            result = h.handle(
                "/api/v1/memory/search-index", {"q": "test"}, mock_http
            )

            assert result is not None
            assert result.status_code == 503

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    @patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    )
    def test_search_index_success(self, mock_run, mock_limiter):
        """Search index returns results from continuum."""
        mock_limiter.is_allowed.return_value = True

        entries = [
            _MockMemoryEntry(id="mem-001", content="Test memory about AI"),
            _MockMemoryEntry(id="mem-002", content="Test memory about ML"),
        ]
        continuum = _make_continuum(entries=entries)

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": continuum})
            mock_http = _http()

            result = h.handle(
                "/api/v1/memory/search-index",
                {"q": "AI", "limit": "10"},
                mock_http,
            )

            assert result is not None
            assert result.status_code == 200
            body = _body(result)
            assert body["query"] == "AI"
            assert body["count"] == 2
            assert len(body["results"]) == 2
            assert body["results"][0]["source"] == "continuum"
            continuum.retrieve.assert_called_once()

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    @patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    )
    def test_search_index_with_external_supermemory(self, mock_run, mock_limiter):
        """Search index includes external supermemory results."""
        mock_limiter.is_allowed.return_value = True

        entries = [_MockMemoryEntry(id="mem-001", content="Memory")]
        continuum = _make_continuum(entries=entries)

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": continuum})
            # Mock the external search methods
            h._search_supermemory = MagicMock(
                return_value=[{"source": "supermemory", "preview": "ext result"}]
            )
            h._search_claude_mem = MagicMock(return_value=[])

            mock_http = _http()
            result = h.handle(
                "/api/v1/memory/search-index",
                {"q": "test", "include_external": "true", "external": "supermemory"},
                mock_http,
            )

            assert result is not None
            assert result.status_code == 200
            body = _body(result)
            assert "supermemory" in body["external_sources"]
            assert len(body["external_results"]) == 1
            h._search_supermemory.assert_called_once()

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    @patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    )
    def test_search_index_with_external_claude_mem(self, mock_run, mock_limiter):
        """Search index includes external claude-mem results."""
        mock_limiter.is_allowed.return_value = True

        entries = []
        continuum = _make_continuum(entries=entries)

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": continuum})
            h._search_supermemory = MagicMock(return_value=[])
            h._search_claude_mem = MagicMock(
                return_value=[{"source": "claude-mem", "preview": "claude result"}]
            )

            mock_http = _http()
            result = h.handle(
                "/api/v1/memory/search-index",
                {"q": "test", "include_external": "true", "external": "claude-mem"},
                mock_http,
            )

            assert result is not None
            assert result.status_code == 200
            body = _body(result)
            assert "claude-mem" in body["external_sources"]
            assert len(body["external_results"]) == 1
            h._search_claude_mem.assert_called_once()

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    @patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    )
    def test_search_index_rate_limited(self, mock_run, mock_limiter, handler):
        """Search index returns 429 when rate limited."""
        mock_limiter.is_allowed.return_value = False
        mock_http = _http()

        result = handler.handle(
            "/api/v1/memory/search-index", {"q": "test"}, mock_http
        )

        assert result is not None
        assert result.status_code == 429

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    @patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    )
    def test_search_index_hybrid_mode(self, mock_run, mock_limiter):
        """Search index uses hybrid search when use_hybrid=true."""
        mock_limiter.is_allowed.return_value = True

        entry = _MockMemoryEntry(id="mem-001", content="Hybrid result")
        hybrid_result = _MockHybridResult(memory_id="mem-001", combined_score=0.95)
        continuum = _make_continuum(entries=[entry], has_hybrid=True)
        continuum.hybrid_search = MagicMock(return_value=[hybrid_result])
        continuum.get_many.return_value = [entry]

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": continuum})
            mock_http = _http()

            result = h.handle(
                "/api/v1/memory/search-index",
                {"q": "AI", "use_hybrid": "true"},
                mock_http,
            )

            assert result is not None
            assert result.status_code == 200
            body = _body(result)
            assert body["use_hybrid"] is True
            assert body["count"] == 1
            assert body["results"][0]["id"] == "mem-001"

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    @patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    )
    def test_search_index_empty_results(self, mock_run, mock_limiter):
        """Search index returns empty results when continuum has no matches."""
        mock_limiter.is_allowed.return_value = True

        continuum = _make_continuum(entries=[])

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": continuum})
            mock_http = _http()

            result = h.handle(
                "/api/v1/memory/search-index", {"q": "nonexistent"}, mock_http
            )

            assert result is not None
            assert result.status_code == 200
            body = _body(result)
            assert body["count"] == 0
            assert body["results"] == []


# ============================================================================
# Search Timeline Route Tests
# ============================================================================


class TestSearchTimelineRoute:
    """Tests for /api/v1/memory/search-timeline route."""

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    @patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    )
    def test_timeline_missing_anchor_id(self, mock_run, mock_limiter):
        """Timeline returns 400 when anchor_id is missing."""
        mock_limiter.is_allowed.return_value = True

        continuum = _make_continuum()

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": continuum})
            mock_http = _http()

            result = h.handle("/api/v1/memory/search-timeline", {}, mock_http)

            assert result is not None
            assert result.status_code == 400
            body = _body(result)
            assert "anchor_id" in body.get("error", "").lower()

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    def test_timeline_no_continuum(self, mock_limiter):
        """Timeline returns 503 when continuum not available."""
        mock_limiter.is_allowed.return_value = True

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={})
            mock_http = _http()

            result = h.handle(
                "/api/v1/memory/search-timeline",
                {"anchor_id": "mem-001"},
                mock_http,
            )

            assert result is not None
            assert result.status_code == 503

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    @patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    )
    def test_timeline_not_supported(self, mock_run, mock_limiter):
        """Timeline returns 501 when backend lacks get_timeline_entries."""
        mock_limiter.is_allowed.return_value = True

        continuum = _make_continuum(has_timeline=False)

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": continuum})
            mock_http = _http()

            result = h.handle(
                "/api/v1/memory/search-timeline",
                {"anchor_id": "mem-001"},
                mock_http,
            )

            assert result is not None
            assert result.status_code == 501

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    @patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    )
    def test_timeline_anchor_not_found(self, mock_run, mock_limiter):
        """Timeline returns 404 when anchor is not found."""
        mock_limiter.is_allowed.return_value = True

        continuum = _make_continuum()
        continuum.get_timeline_entries.return_value = None

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": continuum})
            mock_http = _http()

            result = h.handle(
                "/api/v1/memory/search-timeline",
                {"anchor_id": "nonexistent"},
                mock_http,
            )

            assert result is not None
            assert result.status_code == 404

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    @patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    )
    def test_timeline_success(self, mock_run, mock_limiter):
        """Timeline returns anchor with before/after entries."""
        mock_limiter.is_allowed.return_value = True

        anchor = _MockMemoryEntry(id="mem-002", content="Anchor memory")
        before_entry = _MockMemoryEntry(id="mem-001", content="Before memory")
        after_entry = _MockMemoryEntry(id="mem-003", content="After memory")

        continuum = _make_continuum()
        continuum.get_timeline_entries.return_value = {
            "anchor": anchor,
            "before": [before_entry],
            "after": [after_entry],
        }

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": continuum})
            mock_http = _http()

            result = h.handle(
                "/api/v1/memory/search-timeline",
                {"anchor_id": "mem-002", "before": "1", "after": "1"},
                mock_http,
            )

            assert result is not None
            assert result.status_code == 200
            body = _body(result)
            assert body["anchor_id"] == "mem-002"
            assert body["anchor"]["id"] == "mem-002"
            assert len(body["before"]) == 1
            assert len(body["after"]) == 1

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    def test_timeline_rate_limited(self, mock_limiter, handler):
        """Timeline returns 429 when rate limited."""
        mock_limiter.is_allowed.return_value = False
        mock_http = _http()

        result = handler.handle(
            "/api/v1/memory/search-timeline",
            {"anchor_id": "mem-001"},
            mock_http,
        )

        assert result is not None
        assert result.status_code == 429


# ============================================================================
# Entries Route Tests
# ============================================================================


class TestEntriesRoute:
    """Tests for /api/v1/memory/entries route."""

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    @patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    )
    def test_entries_missing_ids(self, mock_run, mock_limiter):
        """Entries returns 400 when ids param is missing."""
        mock_limiter.is_allowed.return_value = True

        continuum = _make_continuum()

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": continuum})
            mock_http = _http()

            result = h.handle("/api/v1/memory/entries", {}, mock_http)

            assert result is not None
            assert result.status_code == 400
            body = _body(result)
            assert "ids" in body.get("error", "").lower()

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    @patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    )
    def test_entries_empty_ids(self, mock_run, mock_limiter):
        """Entries returns 400 when ids param is empty string."""
        mock_limiter.is_allowed.return_value = True

        continuum = _make_continuum()

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": continuum})
            mock_http = _http()

            result = h.handle(
                "/api/v1/memory/entries", {"ids": ""}, mock_http
            )

            assert result is not None
            assert result.status_code == 400

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    def test_entries_no_continuum(self, mock_limiter):
        """Entries returns 503 when continuum not available."""
        mock_limiter.is_allowed.return_value = True

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={})
            mock_http = _http()

            result = h.handle(
                "/api/v1/memory/entries", {"ids": "mem-001"}, mock_http
            )

            assert result is not None
            assert result.status_code == 503

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    @patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    )
    def test_entries_no_get_many(self, mock_run, mock_limiter):
        """Entries returns 501 when backend lacks get_many."""
        mock_limiter.is_allowed.return_value = True

        continuum = _make_continuum(has_get_many=False)

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": continuum})
            mock_http = _http()

            result = h.handle(
                "/api/v1/memory/entries", {"ids": "mem-001"}, mock_http
            )

            assert result is not None
            assert result.status_code == 501

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    @patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    )
    def test_entries_success_single(self, mock_run, mock_limiter):
        """Entries returns full entry data for a single ID."""
        mock_limiter.is_allowed.return_value = True

        entry = _MockMemoryEntry(
            id="mem-001",
            content="Full content of memory entry",
            importance=0.8,
            tier=_MemoryTier.MEDIUM,
        )
        continuum = _make_continuum(entries=[entry])

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": continuum})
            mock_http = _http()

            result = h.handle(
                "/api/v1/memory/entries", {"ids": "mem-001"}, mock_http
            )

            assert result is not None
            assert result.status_code == 200
            body = _body(result)
            assert body["count"] == 1
            assert body["ids"] == ["mem-001"]
            assert len(body["entries"]) == 1
            assert body["entries"][0]["id"] == "mem-001"
            assert "content" in body["entries"][0]

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    @patch(
        "aragora.server.handlers.memory.memory_progressive.run_async",
        side_effect=lambda coro: coro,
    )
    def test_entries_success_multiple(self, mock_run, mock_limiter):
        """Entries returns full data for multiple IDs."""
        mock_limiter.is_allowed.return_value = True

        entries = [
            _MockMemoryEntry(id="mem-001", content="First"),
            _MockMemoryEntry(id="mem-002", content="Second"),
        ]
        continuum = _make_continuum(entries=entries)

        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": continuum})
            mock_http = _http()

            result = h.handle(
                "/api/v1/memory/entries",
                {"ids": "mem-001,mem-002"},
                mock_http,
            )

            assert result is not None
            assert result.status_code == 200
            body = _body(result)
            assert body["count"] == 2
            assert body["ids"] == ["mem-001", "mem-002"]

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    def test_entries_rate_limited(self, mock_limiter, handler):
        """Entries returns 429 when rate limited."""
        mock_limiter.is_allowed.return_value = False
        mock_http = _http()

        result = handler.handle(
            "/api/v1/memory/entries", {"ids": "mem-001"}, mock_http
        )

        assert result is not None
        assert result.status_code == 429


# ============================================================================
# Utility Method Tests
# ============================================================================


class TestUtilityMethods:
    """Tests for MemoryHandler utility methods used by viewer routes."""

    def test_estimate_tokens_empty(self, handler):
        """Token estimation returns 0 for empty text."""
        assert handler._estimate_tokens("") == 0

    def test_estimate_tokens_short(self, handler):
        """Token estimation for short text returns minimum 1."""
        assert handler._estimate_tokens("hi") >= 1

    def test_estimate_tokens_long(self, handler):
        """Token estimation scales with text length (approx len/4)."""
        text = "a" * 400
        tokens = handler._estimate_tokens(text)
        assert tokens == 100

    def test_parse_bool_param_true_values(self, handler):
        """Boolean parser recognizes various true values."""
        for val in ("1", "true", "yes", "y", "on", "True", "TRUE"):
            assert handler._parse_bool_param({"flag": val}, "flag") is True

    def test_parse_bool_param_false_values(self, handler):
        """Boolean parser recognizes false values."""
        for val in ("0", "false", "no", "n", "off", "False"):
            assert handler._parse_bool_param({"flag": val}, "flag") is False

    def test_parse_bool_param_missing(self, handler):
        """Boolean parser returns default when param missing."""
        assert handler._parse_bool_param({}, "flag") is False
        assert handler._parse_bool_param({}, "flag", default=True) is True

    def test_format_entry_summary_dict(self, handler):
        """Format entry summary from dict input."""
        entry = {
            "id": "mem-001",
            "content": "Short content",
            "tier": "fast",
            "importance": 0.75,
        }
        result = handler._format_entry_summary(entry)
        assert result["id"] == "mem-001"
        assert result["preview"] == "Short content"
        assert result["importance"] == 0.75
        assert result["tier"] == "fast"
        assert result["token_estimate"] >= 1

    def test_format_entry_summary_object(self, handler):
        """Format entry summary from mock entry object."""
        entry = _MockMemoryEntry(
            id="mem-002",
            content="Object content here",
            tier=_MemoryTier.MEDIUM,
            importance=0.9,
        )
        result = handler._format_entry_summary(entry)
        assert result["id"] == "mem-002"
        assert result["preview"] == "Object content here"
        assert result["importance"] == 0.9

    def test_format_entry_summary_long_content_truncated(self, handler):
        """Format entry truncates long content preview."""
        entry = _MockMemoryEntry(content="x" * 500)
        result = handler._format_entry_summary(entry, preview_chars=100)
        assert result["preview"].endswith("...")
        assert len(result["preview"]) <= 104  # 100 chars + "..."

    def test_format_entry_summary_with_red_line(self, handler):
        """Format entry includes red_line fields when present."""
        entry = _MockMemoryEntry(red_line=True, red_line_reason="Safety concern")
        result = handler._format_entry_summary(entry)
        assert result["red_line"] is True
        assert result["red_line_reason"] == "Safety concern"

    def test_format_entry_full_includes_content(self, handler):
        """Format entry full includes the full content field."""
        entry = _MockMemoryEntry(id="mem-001", content="Full content here")
        result = handler._format_entry_full(entry)
        assert result["content"] == "Full content here"
        assert result["metadata"] == {}

    def test_format_entry_full_includes_metadata(self, handler):
        """Format entry full includes metadata."""
        entry = _MockMemoryEntry(
            id="mem-001",
            content="Content",
            metadata={"source": "test", "tags": ["ai"]},
        )
        result = handler._format_entry_full(entry)
        assert result["metadata"] == {"source": "test", "tags": ["ai"]}


# ============================================================================
# POST-only Endpoint Tests
# ============================================================================


class TestPostOnlyEndpoints:
    """Tests for endpoints that reject GET requests."""

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    def test_consolidate_rejects_get(self, mock_limiter, handler):
        """GET on consolidate returns 405 Method Not Allowed."""
        mock_limiter.is_allowed.return_value = True
        mock_http = _http()

        result = handler.handle(
            "/api/v1/memory/continuum/consolidate", {}, mock_http
        )

        assert result is not None
        assert result.status_code == 405
        body = _body(result)
        assert "POST" in body.get("error", "")

    @patch(
        "aragora.server.handlers.memory.memory._retrieve_limiter"
    )
    def test_cleanup_rejects_get(self, mock_limiter, handler):
        """GET on cleanup returns 405 Method Not Allowed."""
        mock_limiter.is_allowed.return_value = True
        mock_http = _http()

        result = handler.handle(
            "/api/v1/memory/continuum/cleanup", {}, mock_http
        )

        assert result is not None
        assert result.status_code == 405
        body = _body(result)
        assert "POST" in body.get("error", "")


# ============================================================================
# Handler Initialization Tests
# ============================================================================


class TestHandlerInit:
    """Tests for MemoryHandler initialization."""

    def test_init_with_empty_context(self, handler):
        """Handler initializes with empty context."""
        assert handler.ctx == {}

    def test_init_with_server_context(self):
        """Handler initializes with provided server context."""
        with patch(
            "aragora.server.handlers.memory.memory.MemoryTier", _MemoryTier
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
        ):
            from aragora.server.handlers.memory.memory import MemoryHandler

            ctx = {"continuum_memory": MagicMock(), "user_store": MagicMock()}
            h = MemoryHandler(server_context=ctx)
            assert h.ctx is ctx

    def test_handle_unmatched_path_returns_none(self, handler):
        """Handler returns None for paths it doesn't specifically route."""
        mock_http = _http()
        # A path under /api/v1/memory/ that isn't explicitly routed
        # but falls through all if-checks to return None
        result = handler.handle("/api/v1/memory/nonexistent-endpoint", {}, mock_http)
        # The handler returns None for unmatched paths within its domain
        assert result is None


# ============================================================================
# ROUTES Constant Tests
# ============================================================================


class TestRoutes:
    """Tests for the handler ROUTES constant."""

    def test_routes_includes_viewer(self, handler):
        """ROUTES includes the viewer endpoint."""
        assert "/api/v1/memory/viewer" in handler.ROUTES

    def test_routes_includes_search_index(self, handler):
        """ROUTES includes search-index endpoint."""
        assert "/api/v1/memory/search-index" in handler.ROUTES

    def test_routes_includes_search_timeline(self, handler):
        """ROUTES includes search-timeline endpoint."""
        assert "/api/v1/memory/search-timeline" in handler.ROUTES

    def test_routes_includes_entries(self, handler):
        """ROUTES includes entries endpoint."""
        assert "/api/v1/memory/entries" in handler.ROUTES

    def test_routes_includes_critiques(self, handler):
        """ROUTES includes critiques endpoint."""
        assert "/api/v1/memory/critiques" in handler.ROUTES

    def test_routes_includes_advanced_operations(self, handler):
        """ROUTES includes advanced SDK parity operations."""
        advanced = [
            "/api/v1/memory/compact",
            "/api/v1/memory/export",
            "/api/v1/memory/import",
            "/api/v1/memory/prune",
            "/api/v1/memory/vacuum",
        ]
        for route in advanced:
            assert route in handler.ROUTES, f"Missing route: {route}"
