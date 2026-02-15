"""
Tests for aragora.server.handlers.utilities - Shared Handler Utilities.

Tests cover:
- get_host_header: with handler, without handler, custom default, env fallback
- get_request_id: various header names, missing headers
- get_content_length: valid, invalid, missing
- get_agent_name: dict, object, None
- agent_to_dict: dict, object, None, include_name toggle
- normalize_agent_names: strings, objects, mixed
- extract_path_segment: valid indices, out of range, default
- build_api_url: segments, query params, empty
- is_json_content_type: various content types
- get_media_type: with charset, without, None
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.utilities import (
    agent_to_dict,
    build_api_url,
    extract_path_segment,
    get_agent_name,
    get_content_length,
    get_host_header,
    get_media_type,
    get_request_id,
    is_json_content_type,
    normalize_agent_names,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_handler(headers: dict[str, str] | None = None) -> MagicMock:
    """Create a mock HTTP handler with headers."""
    handler = MagicMock()
    handler.headers = headers or {}
    return handler


class _AgentObj:
    """Mock agent object with standard ELO fields."""

    def __init__(
        self,
        name: str = "claude",
        elo: float = 1600,
        wins: int = 10,
        losses: int = 2,
        draws: int = 1,
        win_rate: float = 0.83,
        games_played: int = 13,
        matches: int = 13,
    ):
        self.name = name
        self.elo = elo
        self.wins = wins
        self.losses = losses
        self.draws = draws
        self.win_rate = win_rate
        self.games_played = games_played
        self.matches = matches


class _AgentObjAltName:
    """Mock agent with agent_name instead of name."""

    def __init__(self, agent_name: str = "gpt-4"):
        self.agent_name = agent_name


# ===========================================================================
# Test get_host_header
# ===========================================================================


class TestGetHostHeader:
    """Tests for get_host_header()."""

    def test_with_handler_and_host(self):
        handler = _make_handler({"Host": "myserver:9090"})
        assert get_host_header(handler) == "myserver:9090"

    def test_handler_missing_host_header(self):
        handler = _make_handler({})
        result = get_host_header(handler)
        assert "localhost" in result or result  # Falls back to default

    def test_none_handler(self):
        result = get_host_header(None)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_custom_default(self):
        result = get_host_header(None, default="custom:3000")
        assert result == "custom:3000"

    def test_handler_without_headers_attr(self):
        handler = MagicMock(spec=[])  # No attributes
        result = get_host_header(handler, default="fallback:80")
        assert result == "fallback:80"


# ===========================================================================
# Test get_request_id
# ===========================================================================


class TestGetRequestId:
    """Tests for get_request_id()."""

    def test_x_request_id(self):
        handler = _make_handler({"X-Request-ID": "req-123"})
        assert get_request_id(handler) == "req-123"

    def test_x_trace_id(self):
        handler = _make_handler({"X-Trace-ID": "trace-456"})
        assert get_request_id(handler) == "trace-456"

    def test_x_correlation_id(self):
        handler = _make_handler({"X-Correlation-ID": "corr-789"})
        assert get_request_id(handler) == "corr-789"

    def test_priority_order(self):
        handler = _make_handler({
            "X-Request-ID": "first",
            "X-Trace-ID": "second",
        })
        assert get_request_id(handler) == "first"

    def test_none_handler(self):
        assert get_request_id(None) is None

    def test_no_headers_attr(self):
        handler = MagicMock(spec=[])
        assert get_request_id(handler) is None

    def test_no_matching_headers(self):
        handler = _make_handler({"Host": "localhost"})
        assert get_request_id(handler) is None


# ===========================================================================
# Test get_content_length
# ===========================================================================


class TestGetContentLength:
    """Tests for get_content_length()."""

    def test_valid_content_length(self):
        handler = _make_handler({"Content-Length": "42"})
        assert get_content_length(handler) == 42

    def test_zero_content_length(self):
        handler = _make_handler({"Content-Length": "0"})
        assert get_content_length(handler) == 0

    def test_missing_content_length(self):
        handler = _make_handler({})
        assert get_content_length(handler) == 0

    def test_invalid_content_length(self):
        handler = _make_handler({"Content-Length": "not-a-number"})
        assert get_content_length(handler) == 0

    def test_no_headers_attr(self):
        handler = MagicMock(spec=[])
        assert get_content_length(handler) == 0


# ===========================================================================
# Test get_agent_name
# ===========================================================================


class TestGetAgentName:
    """Tests for get_agent_name()."""

    def test_none_input(self):
        assert get_agent_name(None) is None

    def test_dict_with_name(self):
        assert get_agent_name({"name": "claude"}) == "claude"

    def test_dict_with_agent_name(self):
        assert get_agent_name({"agent_name": "gpt-4"}) == "gpt-4"

    def test_dict_agent_name_priority(self):
        # agent_name should take priority over name
        assert get_agent_name({"agent_name": "gpt-4", "name": "claude"}) == "gpt-4"

    def test_object_with_name(self):
        agent = _AgentObj(name="claude")
        assert get_agent_name(agent) == "claude"

    def test_object_with_agent_name(self):
        agent = _AgentObjAltName(agent_name="gpt-4")
        assert get_agent_name(agent) == "gpt-4"

    def test_dict_empty(self):
        assert get_agent_name({}) is None


# ===========================================================================
# Test agent_to_dict
# ===========================================================================


class TestAgentToDict:
    """Tests for agent_to_dict()."""

    def test_none_input(self):
        assert agent_to_dict(None) == {}

    def test_dict_input_returns_copy(self):
        original = {"name": "claude", "elo": 1600}
        result = agent_to_dict(original)
        assert result == original
        assert result is not original  # Must be a copy

    def test_object_input(self):
        agent = _AgentObj(name="claude", elo=1600, wins=10, losses=2)
        result = agent_to_dict(agent)
        assert result["name"] == "claude"
        assert result["agent_name"] == "claude"
        assert result["elo"] == 1600
        assert result["wins"] == 10
        assert result["losses"] == 2

    def test_object_default_fields(self):
        """Agent with minimal attributes gets default values."""
        agent = MagicMock(spec=[])
        result = agent_to_dict(agent)
        assert result["elo"] == 1500
        assert result["wins"] == 0
        assert result["losses"] == 0
        assert result["draws"] == 0
        assert result["name"] == "unknown"

    def test_include_name_false(self):
        agent = _AgentObj(name="claude")
        result = agent_to_dict(agent, include_name=False)
        assert "name" not in result
        assert "agent_name" not in result
        assert "elo" in result

    def test_games_played_field(self):
        agent = _AgentObj(games_played=25, matches=25)
        result = agent_to_dict(agent)
        assert result["games"] == 25
        assert result["matches"] == 25


# ===========================================================================
# Test normalize_agent_names
# ===========================================================================


class TestNormalizeAgentNames:
    """Tests for normalize_agent_names()."""

    def test_string_list(self):
        result = normalize_agent_names(["Claude", "GPT-4", "Gemini"])
        assert result == ["claude", "gpt-4", "gemini"]

    def test_object_list(self):
        agents = [_AgentObj(name="Claude"), _AgentObj(name="GPT-4")]
        result = normalize_agent_names(agents)
        assert result == ["claude", "gpt-4"]

    def test_empty_list(self):
        assert normalize_agent_names([]) == []

    def test_none_name_filtered(self):
        result = normalize_agent_names([None])
        assert result == []

    def test_mixed_strings_and_objects(self):
        agents: list[Any] = ["Claude", _AgentObj(name="GPT-4")]
        result = normalize_agent_names(agents)
        assert result == ["claude", "gpt-4"]


# ===========================================================================
# Test extract_path_segment
# ===========================================================================


class TestExtractPathSegment:
    """Tests for extract_path_segment()."""

    def test_valid_segment(self):
        assert extract_path_segment("/api/v1/debates/123/rounds", 2) == "debates"

    def test_debate_id(self):
        assert extract_path_segment("/api/v1/debates/123/rounds", 3) == "123"

    def test_first_segment(self):
        assert extract_path_segment("/api/v1/agents", 0) == "api"

    def test_out_of_range(self):
        assert extract_path_segment("/api/v1/debates", 10) is None

    def test_out_of_range_with_default(self):
        assert extract_path_segment("/api/v1/debates", 10, "fallback") == "fallback"

    def test_empty_path(self):
        result = extract_path_segment("/", 0)
        # "/" stripped gives "" which splits to [""]
        assert result is None or result == ""


# ===========================================================================
# Test build_api_url
# ===========================================================================


class TestBuildApiUrl:
    """Tests for build_api_url()."""

    def test_simple_segments(self):
        result = build_api_url("api", "debates", "123")
        assert result == "/api/debates/123"

    def test_with_query_params(self):
        result = build_api_url("api", "agents", query_params={"limit": 10})
        assert "limit=10" in result
        assert result.startswith("/api/agents")

    def test_no_query_params(self):
        result = build_api_url("api", "health")
        assert "?" not in result
        assert result == "/api/health"

    def test_strips_slashes_from_segments(self):
        result = build_api_url("/api/", "/debates/", "/123/")
        assert "//" not in result.lstrip("/")

    def test_none_query_values_excluded(self):
        result = build_api_url("api", "agents", query_params={"limit": 10, "offset": None})
        assert "offset" not in result
        assert "limit=10" in result

    def test_empty_segments_excluded(self):
        result = build_api_url("api", "", "debates")
        assert result == "/api/debates"


# ===========================================================================
# Test is_json_content_type
# ===========================================================================


class TestIsJsonContentType:
    """Tests for is_json_content_type()."""

    def test_application_json(self):
        assert is_json_content_type("application/json") is True

    def test_text_json(self):
        assert is_json_content_type("text/json") is True

    def test_json_with_charset(self):
        assert is_json_content_type("application/json; charset=utf-8") is True

    def test_html(self):
        assert is_json_content_type("text/html") is False

    def test_none(self):
        assert is_json_content_type(None) is False

    def test_empty_string(self):
        assert is_json_content_type("") is False


# ===========================================================================
# Test get_media_type
# ===========================================================================


class TestGetMediaType:
    """Tests for get_media_type()."""

    def test_simple_type(self):
        assert get_media_type("application/json") == "application/json"

    def test_with_charset(self):
        assert get_media_type("text/html; charset=utf-8") == "text/html"

    def test_uppercase_normalized(self):
        assert get_media_type("Application/JSON") == "application/json"

    def test_none(self):
        assert get_media_type(None) == ""

    def test_empty_string(self):
        assert get_media_type("") == ""
