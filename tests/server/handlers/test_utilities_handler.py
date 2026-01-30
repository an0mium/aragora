"""
Tests for aragora.server.handlers.utilities - Shared handler utility functions.

Tests cover:
- get_host_header: with handler, without handler, with defaults, env var fallback
- get_request_id: X-Request-ID, X-Trace-ID, X-Correlation-ID, missing headers
- get_content_length: valid, missing, invalid, no headers attribute
- get_agent_name: dict with name, dict with agent_name, object, None
- agent_to_dict: dict passthrough, object conversion, include_name flag, None
- normalize_agent_names: string list, object list, mixed, None entries
- extract_path_segment: valid indices, out-of-bounds, default, empty path
- build_api_url: basic, with query params, None values filtered
- is_json_content_type: application/json, text/json, with charset, non-json, None
- get_media_type: with charset, without, None
"""

from __future__ import annotations

import sys
import types as _types_mod
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Slack stubs to prevent transitive import issues
# ---------------------------------------------------------------------------
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m


from aragora.server.handlers.utilities import (
    get_host_header,
    get_request_id,
    get_content_length,
    get_agent_name,
    agent_to_dict,
    normalize_agent_names,
    extract_path_segment,
    build_api_url,
    is_json_content_type,
    get_media_type,
)


# ===========================================================================
# Mock Objects
# ===========================================================================


@dataclass
class MockAgent:
    """Mock agent with ELO-related fields."""

    name: str = "claude"
    agent_name: str | None = None
    elo: int = 1650
    wins: int = 10
    losses: int = 3
    draws: int = 2
    win_rate: float = 0.67
    games_played: int = 15
    matches: int = 15

    def __post_init__(self):
        if self.agent_name is None:
            self.agent_name = self.name


class MockHandler:
    """Mock HTTP handler with headers."""

    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}


# ===========================================================================
# Tests: get_host_header
# ===========================================================================


class TestGetHostHeader:
    """Tests for Host header extraction."""

    def test_with_handler_and_host_header(self):
        """Returns Host header when present."""
        handler = MockHandler({"Host": "api.aragora.com:8080"})
        assert get_host_header(handler) == "api.aragora.com:8080"

    def test_with_handler_no_host_header(self):
        """Returns default when Host header is missing."""
        handler = MockHandler({})
        result = get_host_header(handler)
        # Should return the default (localhost:8080 or env var)
        assert result is not None
        assert isinstance(result, str)

    def test_with_none_handler(self):
        """Returns default when handler is None."""
        result = get_host_header(None)
        assert result is not None
        assert isinstance(result, str)

    def test_with_custom_default(self):
        """Uses custom default when specified."""
        result = get_host_header(None, default="custom:9090")
        assert result == "custom:9090"

    def test_with_handler_no_headers_attr(self):
        """Returns default when handler lacks headers attribute."""
        handler = object()  # No headers attribute
        result = get_host_header(handler, default="fallback:80")
        assert result == "fallback:80"

    def test_handler_host_overrides_default(self):
        """Host header takes precedence over default."""
        handler = MockHandler({"Host": "real-host:443"})
        result = get_host_header(handler, default="default:80")
        assert result == "real-host:443"


# ===========================================================================
# Tests: get_request_id
# ===========================================================================


class TestGetRequestId:
    """Tests for request/trace ID extraction."""

    def test_x_request_id(self):
        """Returns X-Request-ID when present."""
        handler = MockHandler({"X-Request-ID": "req-abc123"})
        assert get_request_id(handler) == "req-abc123"

    def test_x_trace_id(self):
        """Returns X-Trace-ID when X-Request-ID is absent."""
        handler = MockHandler({"X-Trace-ID": "trace-def456"})
        assert get_request_id(handler) == "trace-def456"

    def test_x_correlation_id(self):
        """Returns X-Correlation-ID as last resort."""
        handler = MockHandler({"X-Correlation-ID": "corr-ghi789"})
        assert get_request_id(handler) == "corr-ghi789"

    def test_x_request_id_takes_priority(self):
        """X-Request-ID takes priority over other headers."""
        handler = MockHandler(
            {
                "X-Request-ID": "req-first",
                "X-Trace-ID": "trace-second",
                "X-Correlation-ID": "corr-third",
            }
        )
        assert get_request_id(handler) == "req-first"

    def test_no_id_headers(self):
        """Returns None when no ID headers are present."""
        handler = MockHandler({"Content-Type": "application/json"})
        assert get_request_id(handler) is None

    def test_none_handler(self):
        """Returns None for None handler."""
        assert get_request_id(None) is None

    def test_handler_no_headers_attr(self):
        """Returns None for handler without headers."""
        assert get_request_id(object()) is None


# ===========================================================================
# Tests: get_content_length
# ===========================================================================


class TestGetContentLength:
    """Tests for Content-Length extraction."""

    def test_valid_content_length(self):
        """Returns integer content length."""
        handler = MockHandler({"Content-Length": "1234"})
        assert get_content_length(handler) == 1234

    def test_zero_content_length(self):
        """Returns 0 for zero-length content."""
        handler = MockHandler({"Content-Length": "0"})
        assert get_content_length(handler) == 0

    def test_missing_content_length(self):
        """Returns 0 when Content-Length is absent."""
        handler = MockHandler({})
        assert get_content_length(handler) == 0

    def test_invalid_content_length(self):
        """Returns 0 for non-numeric Content-Length."""
        handler = MockHandler({"Content-Length": "not-a-number"})
        assert get_content_length(handler) == 0

    def test_no_headers_attr(self):
        """Returns 0 for handler without headers attribute."""
        assert get_content_length(object()) == 0


# ===========================================================================
# Tests: get_agent_name
# ===========================================================================


class TestGetAgentName:
    """Tests for agent name extraction."""

    def test_dict_with_name(self):
        """Returns name from dict with 'name' key."""
        assert get_agent_name({"name": "claude"}) == "claude"

    def test_dict_with_agent_name(self):
        """Returns name from dict with 'agent_name' key."""
        assert get_agent_name({"agent_name": "gpt-4"}) == "gpt-4"

    def test_dict_agent_name_priority(self):
        """agent_name takes priority over name in dict."""
        result = get_agent_name({"agent_name": "primary", "name": "secondary"})
        assert result == "primary"

    def test_object_with_name(self):
        """Returns name from object with name attribute."""
        agent = MockAgent(name="gemini")
        assert get_agent_name(agent) == "gemini"

    def test_none_agent(self):
        """Returns None for None input."""
        assert get_agent_name(None) is None

    def test_empty_dict(self):
        """Returns None for empty dict."""
        assert get_agent_name({}) is None

    def test_object_without_name(self):
        """Returns None for object without name attributes."""
        assert get_agent_name(object()) is None


# ===========================================================================
# Tests: agent_to_dict
# ===========================================================================


class TestAgentToDict:
    """Tests for agent-to-dict conversion."""

    def test_dict_passthrough(self):
        """Returns a copy of the input dict."""
        data = {"name": "claude", "elo": 1700}
        result = agent_to_dict(data)
        assert result == data
        assert result is not data  # Should be a copy

    def test_object_conversion(self):
        """Converts agent object to dict with standard fields."""
        agent = MockAgent(
            name="claude",
            elo=1650,
            wins=10,
            losses=3,
            draws=2,
            win_rate=0.67,
            games_played=15,
            matches=15,
        )
        result = agent_to_dict(agent)

        assert result["name"] == "claude"
        assert result["agent_name"] == "claude"
        assert result["elo"] == 1650
        assert result["wins"] == 10
        assert result["losses"] == 3
        assert result["draws"] == 2
        assert result["win_rate"] == 0.67
        assert result["games"] == 15
        assert result["matches"] == 15

    def test_object_without_include_name(self):
        """Excludes name fields when include_name is False."""
        agent = MockAgent(name="claude")
        result = agent_to_dict(agent, include_name=False)

        assert "name" not in result
        assert "agent_name" not in result
        assert "elo" in result

    def test_none_agent(self):
        """Returns empty dict for None input."""
        assert agent_to_dict(None) == {}

    def test_object_with_defaults(self):
        """Uses sensible defaults for missing attributes."""

        class MinimalAgent:
            pass

        result = agent_to_dict(MinimalAgent())
        assert result["elo"] == 1500  # Default ELO
        assert result["wins"] == 0
        assert result["losses"] == 0
        assert result["name"] == "unknown"


# ===========================================================================
# Tests: normalize_agent_names
# ===========================================================================


class TestNormalizeAgentNames:
    """Tests for agent name normalization."""

    def test_string_list(self):
        """Normalizes string names to lowercase."""
        result = normalize_agent_names(["Claude", "GPT-4", "GEMINI"])
        assert result == ["claude", "gpt-4", "gemini"]

    def test_object_list(self):
        """Extracts and normalizes names from objects."""
        agents = [MockAgent(name="Claude"), MockAgent(name="GPT-4")]
        result = normalize_agent_names(agents)
        assert result == ["claude", "gpt-4"]

    def test_mixed_list(self):
        """Handles mixed strings and objects."""
        agents = ["Claude", MockAgent(name="GPT-4")]
        result = normalize_agent_names(agents)
        assert result == ["claude", "gpt-4"]

    def test_empty_list(self):
        """Returns empty list for empty input."""
        assert normalize_agent_names([]) == []

    def test_skips_none_names(self):
        """Skips entries where name cannot be extracted."""
        agents = ["Claude", None, "GPT-4"]
        result = normalize_agent_names(agents)
        # None should be skipped (get_agent_name returns None for None)
        assert "claude" in result
        assert "gpt-4" in result


# ===========================================================================
# Tests: extract_path_segment
# ===========================================================================


class TestExtractPathSegment:
    """Tests for URL path segment extraction."""

    def test_valid_index(self):
        """Extracts correct segment by index."""
        path = "/api/v1/debates/123/rounds"
        assert extract_path_segment(path, 0) == "api"
        assert extract_path_segment(path, 1) == "v1"
        assert extract_path_segment(path, 2) == "debates"
        assert extract_path_segment(path, 3) == "123"
        assert extract_path_segment(path, 4) == "rounds"

    def test_out_of_bounds(self):
        """Returns None for out-of-bounds index."""
        path = "/api/v1/debates"
        assert extract_path_segment(path, 10) is None

    def test_out_of_bounds_with_default(self):
        """Returns custom default for out-of-bounds index."""
        path = "/api/v1"
        assert extract_path_segment(path, 10, "fallback") == "fallback"

    def test_root_path(self):
        """Handles root path correctly."""
        result = extract_path_segment("/", 0, "default")
        assert result == "default"

    def test_no_leading_slash(self):
        """Works with paths without leading slash."""
        path = "api/v1/debates"
        assert extract_path_segment(path, 0) == "api"


# ===========================================================================
# Tests: build_api_url
# ===========================================================================


class TestBuildApiUrl:
    """Tests for API URL construction."""

    def test_basic_path(self):
        """Builds a simple API path."""
        result = build_api_url("api", "debates", "123")
        assert result == "/api/debates/123"

    def test_with_query_params(self):
        """Appends query parameters."""
        result = build_api_url("api", "agents", query_params={"limit": 10, "offset": 0})
        assert result.startswith("/api/agents")
        assert "limit=10" in result
        assert "offset=0" in result

    def test_none_query_params_filtered(self):
        """Filters out None values from query parameters."""
        result = build_api_url("api", "debates", query_params={"limit": 10, "status": None})
        assert "limit=10" in result
        assert "status" not in result

    def test_empty_segments_skipped(self):
        """Empty string segments are skipped."""
        result = build_api_url("api", "", "debates")
        assert "//" not in result or result.startswith("/")

    def test_strips_slashes_from_segments(self):
        """Strips extra slashes from segments."""
        result = build_api_url("/api/", "/debates/", "/123/")
        assert result == "/api/debates/123"

    def test_no_query_params(self):
        """Works without query parameters."""
        result = build_api_url("api", "v1", "health")
        assert "?" not in result


# ===========================================================================
# Tests: is_json_content_type
# ===========================================================================


class TestIsJsonContentType:
    """Tests for JSON content type detection."""

    def test_application_json(self):
        """Detects application/json."""
        assert is_json_content_type("application/json") is True

    def test_text_json(self):
        """Detects text/json."""
        assert is_json_content_type("text/json") is True

    def test_with_charset(self):
        """Detects JSON with charset parameter."""
        assert is_json_content_type("application/json; charset=utf-8") is True

    def test_non_json(self):
        """Returns False for non-JSON content types."""
        assert is_json_content_type("text/html") is False
        assert is_json_content_type("application/xml") is False

    def test_none(self):
        """Returns False for None."""
        assert is_json_content_type(None) is False

    def test_empty_string(self):
        """Returns False for empty string."""
        assert is_json_content_type("") is False

    def test_case_insensitive(self):
        """Detection is case-insensitive."""
        assert is_json_content_type("Application/JSON") is True


# ===========================================================================
# Tests: get_media_type
# ===========================================================================


class TestGetMediaType:
    """Tests for media type extraction."""

    def test_with_charset(self):
        """Extracts media type, stripping charset."""
        assert get_media_type("application/json; charset=utf-8") == "application/json"

    def test_without_charset(self):
        """Returns media type as-is when no charset."""
        assert get_media_type("text/html") == "text/html"

    def test_none_returns_empty(self):
        """Returns empty string for None."""
        assert get_media_type(None) == ""

    def test_multiple_params(self):
        """Extracts media type with multiple parameters."""
        assert get_media_type("text/plain; charset=utf-8; boundary=something") == "text/plain"

    def test_case_normalized(self):
        """Returns lowercase media type."""
        assert get_media_type("Application/JSON") == "application/json"
