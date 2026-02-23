"""Tests for MemoryCritiquesMixin handler methods.

Tests cover all code paths in aragora/server/handlers/memory/memory_critiques.py:
- _get_critiques: Browse critique store entries with filtering and pagination
  - Critique store not available (CRITIQUE_STORE_AVAILABLE=False)
  - Nomic dir not configured
  - Default params (no agent, default limit/offset)
  - Agent filtering
  - Custom limit and offset
  - Limit clamping (min/max boundaries)
  - Offset clamping (min/max boundaries)
  - Content building from issues and suggestions
  - Content truncation at 300 chars
  - Empty issues and suggestions
  - get_critique_store returns None, fallback to CritiqueStore()
  - Both get_critique_store and CritiqueStore return None
  - Exception handling (KeyError, ValueError, OSError, TypeError, etc.)
  - Pagination total count (with and without agent filter)
  - Rate limiting on the endpoint
  - Routing through MemoryHandler.handle()
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock Critique dataclass mirroring aragora.core_types.Critique
# ---------------------------------------------------------------------------


@dataclass
class _MockCritique:
    """Mock Critique matching the fields accessed by the handler."""

    agent: str = "claude"
    target_agent: str = "gpt4"
    target_content: str = "test proposal"
    issues: list[str] | None = None
    suggestions: list[str] | None = None
    severity: float = 5.0
    reasoning: str = "test reasoning"

    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.suggestions is None:
            self.suggestions = []


def _make_critique(
    *,
    agent: str = "claude",
    target_agent: str = "gpt4",
    issues: list[str] | None = None,
    suggestions: list[str] | None = None,
    severity: float = 5.0,
) -> _MockCritique:
    """Helper to create mock critiques."""
    return _MockCritique(
        agent=agent,
        target_agent=target_agent,
        issues=issues if issues is not None else [],
        suggestions=suggestions if suggestions is not None else [],
        severity=severity,
    )


# ---------------------------------------------------------------------------
# Patch target constants -- the mixin imports from these locations at call time
# ---------------------------------------------------------------------------

_CRITIQUE_AVAILABLE = "aragora.server.handlers.memory.memory.CRITIQUE_STORE_AVAILABLE"
_CRITIQUE_STORE_CLS = "aragora.server.handlers.memory.memory.CritiqueStore"
_GET_CRITIQUE_STORE = "aragora.stores.canonical.get_critique_store"
_CONTINUUM_AVAILABLE = "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict[str, Any]:
    """Parse the JSON body out of a HandlerResult."""
    return json.loads(result.body)


def _status(result) -> int:
    """Extract status code from a HandlerResult."""
    return result.status_code


def _http() -> MagicMock:
    """Return a mock HTTP handler object."""
    mock = MagicMock()
    mock.command = "GET"
    mock.client_address = ("127.0.0.1", 54321)
    mock.headers = {"Content-Length": "2"}
    mock.rfile.read.return_value = b"{}"
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_critique_store():
    """Create a mock critique store."""
    store = MagicMock()
    store.get_recent.return_value = []
    return store


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset module-level rate limiters between tests."""
    try:
        from aragora.server.handlers.memory import memory as mem_mod

        for attr in ("_retrieve_limiter", "_stats_limiter", "_mutation_limiter"):
            limiter = getattr(mem_mod, attr, None)
            if limiter and hasattr(limiter, "_requests"):
                limiter._requests.clear()
            if limiter and hasattr(limiter, "reset"):
                limiter.reset()
    except Exception:
        pass

    # Also reset the decorator-level rate limiter for memory_read
    try:
        from aragora.server.handlers.utils.rate_limit import _named_limiters

        if "memory_read" in _named_limiters:
            limiter = _named_limiters["memory_read"]
            if hasattr(limiter, "_requests"):
                limiter._requests.clear()
            if hasattr(limiter, "reset"):
                limiter.reset()
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True)
def _patch_modules_available():
    """Ensure CRITIQUE_STORE_AVAILABLE and CONTINUUM_AVAILABLE are True."""
    with patch(_CRITIQUE_AVAILABLE, True), patch(_CONTINUUM_AVAILABLE, True):
        yield


@pytest.fixture
def handler(mock_critique_store):
    """Create a MemoryHandler backed by mock critique store."""
    with patch(_CRITIQUE_AVAILABLE, True), patch(_CONTINUUM_AVAILABLE, True):
        from aragora.server.handlers.memory.memory import MemoryHandler

        h = MemoryHandler(server_context={"nomic_dir": "/tmp/test_nomic"})
        h._auth_context = None
        return h


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    return _http()


# ===========================================================================
# TestGetCritiquesStoreUnavailable
# ===========================================================================


class TestGetCritiquesStoreUnavailable:
    """Tests for when critique store is not available."""

    def test_critique_store_not_available_returns_503(
        self, handler, mock_http_handler
    ):
        """Returns 503 when CRITIQUE_STORE_AVAILABLE is False."""
        with patch(_CRITIQUE_AVAILABLE, False):
            result = handler._get_critiques({})
            assert _status(result) == 503
            assert "not available" in _body(result)["error"]

    def test_nomic_dir_not_configured_returns_503(self, mock_http_handler):
        """Returns 503 when nomic_dir is not in context."""
        with patch(_CRITIQUE_AVAILABLE, True), patch(_CONTINUUM_AVAILABLE, True):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={})
            h._auth_context = None
            result = h._get_critiques({})
            assert _status(result) == 503
            assert "not configured" in _body(result)["error"]

    def test_nomic_dir_empty_string_returns_503(self, mock_http_handler):
        """Returns 503 when nomic_dir is an empty string (falsy)."""
        with patch(_CRITIQUE_AVAILABLE, True), patch(_CONTINUUM_AVAILABLE, True):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"nomic_dir": ""})
            h._auth_context = None
            result = h._get_critiques({})
            assert _status(result) == 503

    def test_nomic_dir_none_returns_503(self, mock_http_handler):
        """Returns 503 when nomic_dir is None."""
        with patch(_CRITIQUE_AVAILABLE, True), patch(_CONTINUUM_AVAILABLE, True):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"nomic_dir": None})
            h._auth_context = None
            result = h._get_critiques({})
            assert _status(result) == 503

    def test_get_critique_store_and_class_both_none_returns_503(
        self, handler, mock_http_handler
    ):
        """Returns 503 when both get_critique_store and CritiqueStore are None."""
        with patch(
            _GET_CRITIQUE_STORE, return_value=None
        ), patch(
            _CRITIQUE_STORE_CLS, None
        ):
            result = handler._get_critiques({})
            assert _status(result) == 503
            assert "not available" in _body(result)["error"]


# ===========================================================================
# TestGetCritiquesBasic
# ===========================================================================


class TestGetCritiquesBasic:
    """Tests for basic _get_critiques functionality."""

    def test_empty_store_returns_empty_list(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Returns empty results when store has no critiques."""
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            assert _status(result) == 200
            body = _body(result)
            assert body["critiques"] == []
            assert body["count"] == 0
            assert body["total"] == 0

    def test_single_critique_returned(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Returns properly formatted single critique."""
        crit = _make_critique(
            agent="claude",
            target_agent="gpt4",
            issues=["Issue 1"],
            suggestions=["Suggestion 1"],
            severity=7.5,
        )
        mock_critique_store.get_recent.return_value = [crit]
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            assert _status(result) == 200
            body = _body(result)
            assert body["count"] == 1
            entry = body["critiques"][0]
            assert entry["agent"] == "claude"
            assert entry["target_agent"] == "gpt4"
            assert entry["severity"] == 7.5
            assert "Issue 1" in entry["content"]
            assert "Suggestion 1" in entry["content"]

    def test_critique_fields_structure(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Each critique has all expected fields."""
        crit = _make_critique()
        mock_critique_store.get_recent.return_value = [crit]
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            entry = _body(result)["critiques"][0]
            expected_keys = {
                "id",
                "debate_id",
                "agent",
                "target_agent",
                "critique_type",
                "content",
                "severity",
                "accepted",
                "created_at",
            }
            assert set(entry.keys()) == expected_keys

    def test_null_fields_in_critique(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Fields not available from Critique dataclass are None."""
        crit = _make_critique()
        mock_critique_store.get_recent.return_value = [crit]
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            entry = _body(result)["critiques"][0]
            assert entry["id"] is None
            assert entry["debate_id"] is None
            assert entry["critique_type"] is None
            assert entry["accepted"] is None
            assert entry["created_at"] is None

    def test_multiple_critiques_returned(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Returns multiple critiques from the store."""
        crits = [
            _make_critique(agent="claude", severity=3.0),
            _make_critique(agent="gpt4", severity=7.0),
            _make_critique(agent="gemini", severity=9.0),
        ]
        mock_critique_store.get_recent.return_value = crits
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            body = _body(result)
            assert body["count"] == 3
            agents = [c["agent"] for c in body["critiques"]]
            assert agents == ["claude", "gpt4", "gemini"]


# ===========================================================================
# TestGetCritiquesContent
# ===========================================================================


class TestGetCritiquesContent:
    """Tests for content building from issues and suggestions."""

    def test_content_from_issues_only(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Content is built from issues when no suggestions present."""
        crit = _make_critique(issues=["Bad logic", "Wrong format"], suggestions=[])
        mock_critique_store.get_recent.return_value = [crit]
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            content = _body(result)["critiques"][0]["content"]
            assert "Bad logic" in content
            assert "Wrong format" in content

    def test_content_from_suggestions_only(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Content is built from suggestions when no issues present."""
        crit = _make_critique(issues=[], suggestions=["Use caching", "Add logging"])
        mock_critique_store.get_recent.return_value = [crit]
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            content = _body(result)["critiques"][0]["content"]
            assert "Use caching" in content
            assert "Add logging" in content

    def test_content_from_both_issues_and_suggestions(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Content includes both issues and suggestions joined by semicolons."""
        crit = _make_critique(
            issues=["Issue A"],
            suggestions=["Suggestion B"],
        )
        mock_critique_store.get_recent.return_value = [crit]
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            content = _body(result)["critiques"][0]["content"]
            assert "Issue A" in content
            assert "Suggestion B" in content
            assert "; " in content

    def test_content_empty_when_no_issues_or_suggestions(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Content is empty string when both issues and suggestions are empty."""
        crit = _make_critique(issues=[], suggestions=[])
        mock_critique_store.get_recent.return_value = [crit]
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            content = _body(result)["critiques"][0]["content"]
            assert content == ""

    def test_content_truncated_at_300_chars(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Content longer than 300 characters is truncated."""
        long_issue = "x" * 350
        crit = _make_critique(issues=[long_issue], suggestions=[])
        mock_critique_store.get_recent.return_value = [crit]
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            content = _body(result)["critiques"][0]["content"]
            assert len(content) <= 300

    def test_issues_limited_to_first_two(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Only the first 2 issues are included in content."""
        crit = _make_critique(
            issues=["Issue 1", "Issue 2", "Issue 3", "Issue 4"],
            suggestions=[],
        )
        mock_critique_store.get_recent.return_value = [crit]
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            content = _body(result)["critiques"][0]["content"]
            assert "Issue 1" in content
            assert "Issue 2" in content
            assert "Issue 3" not in content

    def test_suggestions_limited_to_first_two(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Only the first 2 suggestions are included in content."""
        crit = _make_critique(
            issues=[],
            suggestions=["Sug 1", "Sug 2", "Sug 3", "Sug 4"],
        )
        mock_critique_store.get_recent.return_value = [crit]
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            content = _body(result)["critiques"][0]["content"]
            assert "Sug 1" in content
            assert "Sug 2" in content
            assert "Sug 3" not in content

    def test_content_from_two_issues_and_two_suggestions(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Content is built from 2 issues + 2 suggestions (max 4 parts)."""
        crit = _make_critique(
            issues=["I1", "I2", "I3"],
            suggestions=["S1", "S2", "S3"],
        )
        mock_critique_store.get_recent.return_value = [crit]
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            content = _body(result)["critiques"][0]["content"]
            assert "I1" in content
            assert "I2" in content
            assert "I3" not in content
            assert "S1" in content
            assert "S2" in content
            assert "S3" not in content


# ===========================================================================
# TestGetCritiquesAgentFilter
# ===========================================================================


class TestGetCritiquesAgentFilter:
    """Tests for agent-based filtering."""

    def test_filter_by_agent(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Filters critiques by agent name."""
        crits = [
            _make_critique(agent="claude"),
            _make_critique(agent="gpt4"),
            _make_critique(agent="claude"),
        ]
        mock_critique_store.get_recent.return_value = crits
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({"agent": "claude"})
            body = _body(result)
            assert body["count"] == 2
            assert all(c["agent"] == "claude" for c in body["critiques"])

    def test_filter_by_agent_no_match(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Returns empty when no critiques match agent filter."""
        crits = [_make_critique(agent="claude"), _make_critique(agent="gpt4")]
        mock_critique_store.get_recent.return_value = crits
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({"agent": "nonexistent"})
            body = _body(result)
            assert body["count"] == 0
            assert body["critiques"] == []

    def test_agent_filter_in_response_metadata(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Filters metadata includes agent name."""
        mock_critique_store.get_recent.return_value = []
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({"agent": "claude"})
            body = _body(result)
            assert body["filters"]["agent"] == "claude"

    def test_no_agent_filter_in_response_metadata(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Filters metadata has None when no agent filter."""
        mock_critique_store.get_recent.return_value = []
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            body = _body(result)
            assert body["filters"]["agent"] is None

    def test_agent_filter_fetch_limit_extra(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """When agent filter is set, fetch_limit includes +100 extra."""
        crits = [_make_critique(agent="claude") for _ in range(5)]
        mock_critique_store.get_recent.return_value = crits
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            handler._get_critiques({"agent": "claude", "limit": "10", "offset": "0"})
            # fetch_limit = 10 + 0 + 100 = 110
            # First call is for fetching; check its limit kwarg
            first_call = mock_critique_store.get_recent.call_args_list[0]
            assert first_call.kwargs.get("limit") == 110

    def test_no_agent_filter_fetch_limit_exact(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Without agent filter, fetch_limit is limit + offset."""
        mock_critique_store.get_recent.return_value = []
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            handler._get_critiques({"limit": "15", "offset": "5"})
            # fetch_limit = 15 + 5 = 20
            first_call = mock_critique_store.get_recent.call_args_list[0]
            assert first_call.kwargs.get("limit") == 20


# ===========================================================================
# TestGetCritiquesPagination
# ===========================================================================


class TestGetCritiquesPagination:
    """Tests for limit and offset pagination."""

    def test_default_limit_is_20(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Default limit is 20."""
        mock_critique_store.get_recent.return_value = []
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            body = _body(result)
            assert body["limit"] == 20

    def test_default_offset_is_0(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Default offset is 0."""
        mock_critique_store.get_recent.return_value = []
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            body = _body(result)
            assert body["offset"] == 0

    def test_custom_limit(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Custom limit is respected."""
        crits = [_make_critique(agent=f"agent_{i}") for i in range(50)]
        mock_critique_store.get_recent.return_value = crits
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({"limit": "5"})
            body = _body(result)
            assert body["count"] == 5
            assert body["limit"] == 5

    def test_custom_offset(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Custom offset skips first N results."""
        crits = [_make_critique(agent=f"agent_{i}") for i in range(10)]
        mock_critique_store.get_recent.return_value = crits
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({"offset": "3", "limit": "2"})
            body = _body(result)
            assert body["count"] == 2
            assert body["offset"] == 3
            # Should get agents 3 and 4
            assert body["critiques"][0]["agent"] == "agent_3"
            assert body["critiques"][1]["agent"] == "agent_4"

    def test_limit_clamped_to_max_100(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Limit is clamped to maximum 100."""
        mock_critique_store.get_recent.return_value = []
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({"limit": "500"})
            body = _body(result)
            assert body["limit"] == 100

    def test_limit_clamped_to_min_1(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Limit is clamped to minimum 1."""
        mock_critique_store.get_recent.return_value = []
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({"limit": "0"})
            body = _body(result)
            assert body["limit"] == 1

    def test_offset_clamped_to_min_0(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Offset is clamped to minimum 0."""
        mock_critique_store.get_recent.return_value = []
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({"offset": "-5"})
            body = _body(result)
            assert body["offset"] == 0

    def test_offset_clamped_to_max_10000(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Offset is clamped to maximum 10000."""
        mock_critique_store.get_recent.return_value = []
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({"offset": "99999"})
            body = _body(result)
            assert body["offset"] == 10000

    def test_total_count_without_agent_filter(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Total count queries store with limit=10000 when no agent filter."""
        crits = [_make_critique() for _ in range(5)]
        all_crits = [_make_critique() for _ in range(25)]
        # First call for paginated results, second for total count
        mock_critique_store.get_recent.side_effect = [crits, all_crits]
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({"limit": "5"})
            body = _body(result)
            assert body["total"] == 25

    def test_total_count_with_agent_filter(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Total count is based on filtered results when agent filter is set."""
        crits = [
            _make_critique(agent="claude"),
            _make_critique(agent="gpt4"),
            _make_critique(agent="claude"),
            _make_critique(agent="claude"),
        ]
        mock_critique_store.get_recent.return_value = crits
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({"agent": "claude", "limit": "2"})
            body = _body(result)
            assert body["total"] == 3  # 3 claude critiques total
            assert body["count"] == 2  # limited to 2

    def test_offset_beyond_available_returns_empty(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Offset beyond available critiques returns empty list."""
        crits = [_make_critique() for _ in range(3)]
        mock_critique_store.get_recent.return_value = crits
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({"offset": "100"})
            body = _body(result)
            assert body["count"] == 0
            assert body["critiques"] == []


# ===========================================================================
# TestGetCritiquesStoreFallback
# ===========================================================================


class TestGetCritiquesStoreFallback:
    """Tests for critique store fallback logic."""

    def test_fallback_to_critique_store_class(
        self, handler, mock_http_handler
    ):
        """Falls back to CritiqueStore(nomic_dir) when get_critique_store returns None."""
        mock_store = MagicMock()
        mock_store.get_recent.return_value = []
        mock_cs_class = MagicMock(return_value=mock_store)

        with patch(
            _GET_CRITIQUE_STORE, return_value=None
        ), patch(
            _CRITIQUE_STORE_CLS, mock_cs_class
        ):
            result = handler._get_critiques({})
            assert _status(result) == 200
            mock_cs_class.assert_called_once_with("/tmp/test_nomic")

    def test_get_critique_store_used_when_available(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Uses get_critique_store result directly when it returns a store."""
        mock_critique_store.get_recent.return_value = []
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            assert _status(result) == 200

    def test_get_critique_store_none_and_class_none_returns_503(
        self, handler, mock_http_handler
    ):
        """Returns 503 when get_critique_store returns None and CritiqueStore is None."""
        with patch(_GET_CRITIQUE_STORE, return_value=None), patch(
            _CRITIQUE_STORE_CLS, None
        ):
            result = handler._get_critiques({})
            assert _status(result) == 503


# ===========================================================================
# TestGetCritiquesErrorHandling
# ===========================================================================


class TestGetCritiquesErrorHandling:
    """Tests for exception handling in _get_critiques."""

    def test_key_error_returns_500(self, handler, mock_http_handler):
        """KeyError in store operations returns 500."""
        mock_store = MagicMock()
        mock_store.get_recent.side_effect = KeyError("missing_key")
        with patch(_GET_CRITIQUE_STORE, return_value=mock_store):
            result = handler._get_critiques({})
            assert _status(result) == 500

    def test_value_error_returns_500(self, handler, mock_http_handler):
        """ValueError in store operations returns 500."""
        mock_store = MagicMock()
        mock_store.get_recent.side_effect = ValueError("bad value")
        with patch(_GET_CRITIQUE_STORE, return_value=mock_store):
            result = handler._get_critiques({})
            assert _status(result) == 500

    def test_os_error_returns_500(self, handler, mock_http_handler):
        """OSError in store operations returns 500."""
        mock_store = MagicMock()
        mock_store.get_recent.side_effect = OSError("disk full")
        with patch(_GET_CRITIQUE_STORE, return_value=mock_store):
            result = handler._get_critiques({})
            assert _status(result) == 500

    def test_type_error_returns_500(self, handler, mock_http_handler):
        """TypeError in store operations returns 500."""
        mock_store = MagicMock()
        mock_store.get_recent.side_effect = TypeError("wrong type")
        with patch(_GET_CRITIQUE_STORE, return_value=mock_store):
            result = handler._get_critiques({})
            assert _status(result) == 500

    def test_attribute_error_returns_500(self, handler, mock_http_handler):
        """AttributeError in store operations returns 500."""
        mock_store = MagicMock()
        mock_store.get_recent.side_effect = AttributeError("no attribute")
        with patch(_GET_CRITIQUE_STORE, return_value=mock_store):
            result = handler._get_critiques({})
            assert _status(result) == 500

    def test_runtime_error_returns_500(self, handler, mock_http_handler):
        """RuntimeError in store operations returns 500."""
        mock_store = MagicMock()
        mock_store.get_recent.side_effect = RuntimeError("failed")
        with patch(_GET_CRITIQUE_STORE, return_value=mock_store):
            result = handler._get_critiques({})
            assert _status(result) == 500

    def test_error_response_uses_safe_error_message(
        self, handler, mock_http_handler
    ):
        """Error responses use safe_error_message (no internal details)."""
        mock_store = MagicMock()
        mock_store.get_recent.side_effect = OSError("/secret/path/leaked")
        with patch(_GET_CRITIQUE_STORE, return_value=mock_store):
            result = handler._get_critiques({})
            body = _body(result)
            # safe_error_message should sanitize the path
            assert "/secret/path/leaked" not in body.get("error", "")


# ===========================================================================
# TestGetCritiquesRouting
# ===========================================================================


class TestGetCritiquesRouting:
    """Tests for routing through the main MemoryHandler."""

    def test_route_via_handler_handle(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Routing through handle() for /api/v1/memory/critiques works."""
        mock_critique_store.get_recent.return_value = []
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler.handle(
                "/api/v1/memory/critiques", {}, mock_http_handler
            )
            assert result is not None
            assert _status(result) == 200

    def test_legacy_route_normalization(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Legacy /api/memory/critiques normalizes to /api/v1/memory/critiques."""
        mock_critique_store.get_recent.return_value = []
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler.handle(
                "/api/memory/critiques", {}, mock_http_handler
            )
            assert result is not None
            assert _status(result) == 200

    def test_can_handle_critiques_path(self, handler):
        """can_handle returns True for /api/v1/memory/critiques."""
        assert handler.can_handle("/api/v1/memory/critiques") is True

    def test_can_handle_legacy_critiques_path(self, handler):
        """can_handle returns True for /api/memory/critiques (legacy)."""
        assert handler.can_handle("/api/memory/critiques") is True


# ===========================================================================
# TestGetCritiquesParamBoundaries
# ===========================================================================


class TestGetCritiquesParamBoundaries:
    """Tests for parameter boundary handling."""

    def test_agent_param_truncated_at_100_chars(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Agent name is truncated at 100 characters."""
        long_agent = "a" * 200
        crits = [_make_critique(agent=long_agent[:100])]
        mock_critique_store.get_recent.return_value = crits
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({"agent": long_agent})
            body = _body(result)
            # The filter value should be truncated
            assert len(body["filters"]["agent"]) <= 100

    def test_non_numeric_limit_uses_default(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Non-numeric limit parameter falls back to default 20."""
        mock_critique_store.get_recent.return_value = []
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({"limit": "abc"})
            body = _body(result)
            assert body["limit"] == 20

    def test_non_numeric_offset_uses_default(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Non-numeric offset parameter falls back to default 0."""
        mock_critique_store.get_recent.return_value = []
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({"offset": "xyz"})
            body = _body(result)
            assert body["offset"] == 0

    def test_negative_limit_clamped_to_1(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Negative limit is clamped to minimum of 1."""
        mock_critique_store.get_recent.return_value = []
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({"limit": "-10"})
            body = _body(result)
            assert body["limit"] == 1


# ===========================================================================
# TestGetCritiquesResponseStructure
# ===========================================================================


class TestGetCritiquesResponseStructure:
    """Tests for the overall response structure."""

    def test_response_has_all_top_level_keys(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Response contains all expected top-level keys."""
        mock_critique_store.get_recent.return_value = []
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            body = _body(result)
            assert "critiques" in body
            assert "count" in body
            assert "total" in body
            assert "offset" in body
            assert "limit" in body
            assert "filters" in body

    def test_response_content_type_is_json(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Response content type is application/json."""
        mock_critique_store.get_recent.return_value = []
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({})
            assert "json" in result.content_type

    def test_count_matches_critiques_length(
        self, handler, mock_http_handler, mock_critique_store
    ):
        """Count field matches actual number of critiques returned."""
        crits = [_make_critique() for _ in range(7)]
        mock_critique_store.get_recent.return_value = crits
        with patch(_GET_CRITIQUE_STORE, return_value=mock_critique_store):
            result = handler._get_critiques({"limit": "5"})
            body = _body(result)
            assert body["count"] == len(body["critiques"])
            assert body["count"] == 5
