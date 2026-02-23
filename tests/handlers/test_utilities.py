"""Tests for shared handler utilities module.

Covers all exported utility functions:
- get_host_header: Host header extraction with fallback handling
- get_request_id: Request/trace ID extraction from multiple header variants
- get_content_length: Content-Length parsing with error handling
- get_agent_name: Agent name extraction from dicts, objects, and None
- agent_to_dict: Agent-to-dict conversion with ELO fields
- normalize_agent_names: Lowercase normalization of agent names
- extract_path_segment: URL path segment extraction by index
- build_api_url: API URL construction from segments and query params
- is_json_content_type: JSON content type detection
- get_media_type: Media type extraction from Content-Type header
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.utilities import (
    _DEFAULT_HOST,
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


# =============================================================================
# Test Helpers / Fixtures
# =============================================================================


@dataclass
class MockAgentRating:
    """Mimics AgentRating for testing agent_to_dict and get_agent_name."""

    agent_name: str = "claude"
    elo: float = 1600.0
    wins: int = 10
    losses: int = 3
    draws: int = 2
    win_rate: float = 0.67
    games_played: int = 15
    matches: int = 15


@dataclass
class MinimalAgent:
    """Agent with only a name attribute."""

    name: str = "gpt-4"


@dataclass
class EmptyAgent:
    """Agent with no name-related attributes."""

    model: str = "some-model"


class MockHandlerWithHeaders:
    """Mock HTTP handler with a headers dict."""

    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}


class MockHandlerNoHeaders:
    """Mock HTTP handler without a headers attribute."""

    pass


# =============================================================================
# get_host_header Tests
# =============================================================================


class TestGetHostHeader:
    """Tests for get_host_header utility."""

    def test_returns_host_from_handler(self):
        handler = MockHandlerWithHeaders({"Host": "example.com:9090"})
        assert get_host_header(handler) == "example.com:9090"

    def test_returns_default_when_handler_is_none(self):
        result = get_host_header(None)
        assert result == _DEFAULT_HOST

    def test_returns_explicit_default_when_handler_is_none(self):
        result = get_host_header(None, default="custom:8080")
        assert result == "custom:8080"

    def test_returns_default_when_host_header_missing(self):
        handler = MockHandlerWithHeaders({})
        result = get_host_header(handler)
        assert result == _DEFAULT_HOST

    def test_returns_explicit_default_when_host_missing(self):
        handler = MockHandlerWithHeaders({})
        result = get_host_header(handler, default="fallback:3000")
        assert result == "fallback:3000"

    def test_returns_default_when_handler_has_no_headers_attr(self):
        handler = MockHandlerNoHeaders()
        result = get_host_header(handler)
        assert result == _DEFAULT_HOST

    def test_returns_explicit_default_when_no_headers_attr(self):
        handler = MockHandlerNoHeaders()
        result = get_host_header(handler, default="my-default")
        assert result == "my-default"

    def test_default_host_env_var(self):
        """Verify _DEFAULT_HOST uses ARAGORA_DEFAULT_HOST or localhost:8080."""
        # _DEFAULT_HOST is set at module import time from env
        expected = os.environ.get("ARAGORA_DEFAULT_HOST", "localhost:8080")
        assert _DEFAULT_HOST == expected

    def test_host_with_port(self):
        handler = MockHandlerWithHeaders({"Host": "api.aragora.io:443"})
        assert get_host_header(handler) == "api.aragora.io:443"

    def test_host_without_port(self):
        handler = MockHandlerWithHeaders({"Host": "api.aragora.io"})
        assert get_host_header(handler) == "api.aragora.io"

    def test_empty_host_value(self):
        """Empty Host header is still returned (truthy check not applied)."""
        handler = MockHandlerWithHeaders({"Host": ""})
        assert get_host_header(handler) == ""

    def test_handler_with_dict_like_headers(self):
        """Headers accessed via .get() -- any dict-like object works."""
        handler = MagicMock()
        handler.headers = {"Host": "mock-host:1234"}
        assert get_host_header(handler) == "mock-host:1234"


# =============================================================================
# get_request_id Tests
# =============================================================================


class TestGetRequestId:
    """Tests for get_request_id utility."""

    def test_returns_none_when_handler_is_none(self):
        assert get_request_id(None) is None

    def test_returns_none_when_no_headers_attr(self):
        handler = MockHandlerNoHeaders()
        assert get_request_id(handler) is None

    def test_extracts_x_request_id(self):
        handler = MockHandlerWithHeaders({"X-Request-ID": "req-123"})
        assert get_request_id(handler) == "req-123"

    def test_extracts_x_trace_id(self):
        handler = MockHandlerWithHeaders({"X-Trace-ID": "trace-456"})
        assert get_request_id(handler) == "trace-456"

    def test_extracts_x_correlation_id(self):
        handler = MockHandlerWithHeaders({"X-Correlation-ID": "corr-789"})
        assert get_request_id(handler) == "corr-789"

    def test_priority_order_request_id_first(self):
        """X-Request-ID takes priority over X-Trace-ID and X-Correlation-ID."""
        handler = MockHandlerWithHeaders(
            {
                "X-Request-ID": "req-1",
                "X-Trace-ID": "trace-2",
                "X-Correlation-ID": "corr-3",
            }
        )
        assert get_request_id(handler) == "req-1"

    def test_priority_order_trace_id_second(self):
        """X-Trace-ID takes priority over X-Correlation-ID when no X-Request-ID."""
        handler = MockHandlerWithHeaders(
            {
                "X-Trace-ID": "trace-2",
                "X-Correlation-ID": "corr-3",
            }
        )
        assert get_request_id(handler) == "trace-2"

    def test_returns_none_when_no_id_headers(self):
        handler = MockHandlerWithHeaders({"Content-Type": "text/html"})
        assert get_request_id(handler) is None

    def test_empty_headers(self):
        handler = MockHandlerWithHeaders({})
        assert get_request_id(handler) is None

    def test_empty_request_id_falls_through(self):
        """Empty string is falsy, so it should fall through to next header."""
        handler = MockHandlerWithHeaders(
            {
                "X-Request-ID": "",
                "X-Trace-ID": "trace-ok",
            }
        )
        assert get_request_id(handler) == "trace-ok"

    def test_all_empty_returns_none_or_empty(self):
        """All headers empty should return empty string (falsy)."""
        handler = MockHandlerWithHeaders(
            {
                "X-Request-ID": "",
                "X-Trace-ID": "",
                "X-Correlation-ID": "",
            }
        )
        # The or-chain returns the last empty string
        result = get_request_id(handler)
        assert result == "" or result is None


# =============================================================================
# get_content_length Tests
# =============================================================================


class TestGetContentLength:
    """Tests for get_content_length utility."""

    def test_returns_zero_when_no_headers_attr(self):
        handler = MockHandlerNoHeaders()
        assert get_content_length(handler) == 0

    def test_returns_content_length_as_int(self):
        handler = MockHandlerWithHeaders({"Content-Length": "1024"})
        assert get_content_length(handler) == 1024

    def test_returns_zero_when_header_missing(self):
        handler = MockHandlerWithHeaders({})
        assert get_content_length(handler) == 0

    def test_returns_zero_for_non_numeric_value(self):
        handler = MockHandlerWithHeaders({"Content-Length": "not-a-number"})
        assert get_content_length(handler) == 0

    def test_returns_zero_for_empty_string(self):
        handler = MockHandlerWithHeaders({"Content-Length": ""})
        assert get_content_length(handler) == 0

    def test_handles_zero_content_length(self):
        handler = MockHandlerWithHeaders({"Content-Length": "0"})
        assert get_content_length(handler) == 0

    def test_large_content_length(self):
        handler = MockHandlerWithHeaders({"Content-Length": "999999999"})
        assert get_content_length(handler) == 999999999

    def test_negative_content_length(self):
        """Negative values are valid ints, function passes them through."""
        handler = MockHandlerWithHeaders({"Content-Length": "-1"})
        assert get_content_length(handler) == -1

    def test_float_content_length_raises(self):
        """Float strings are invalid for int(), returns 0."""
        handler = MockHandlerWithHeaders({"Content-Length": "10.5"})
        assert get_content_length(handler) == 0

    def test_none_handler_attribute_content_length(self):
        """Handler with headers=None gets handled by hasattr check."""
        handler = MagicMock(spec=[])  # No headers attribute
        assert get_content_length(handler) == 0


# =============================================================================
# get_agent_name Tests
# =============================================================================


class TestGetAgentName:
    """Tests for get_agent_name utility."""

    def test_returns_none_for_none(self):
        assert get_agent_name(None) is None

    def test_dict_with_agent_name_key(self):
        assert get_agent_name({"agent_name": "claude"}) == "claude"

    def test_dict_with_name_key(self):
        assert get_agent_name({"name": "gpt-4"}) == "gpt-4"

    def test_dict_agent_name_takes_priority(self):
        """agent_name is checked before name in dict."""
        agent = {"agent_name": "claude", "name": "gpt-4"}
        assert get_agent_name(agent) == "claude"

    def test_dict_with_neither_key(self):
        assert get_agent_name({"elo": 1500}) is None

    def test_empty_dict(self):
        assert get_agent_name({}) is None

    def test_object_with_agent_name(self):
        agent = MockAgentRating(agent_name="gemini")
        assert get_agent_name(agent) == "gemini"

    def test_object_with_name_only(self):
        agent = MinimalAgent(name="gpt-4")
        assert get_agent_name(agent) == "gpt-4"

    def test_object_with_both_prefers_agent_name(self):
        agent = MagicMock()
        agent.agent_name = "claude"
        agent.name = "alt-name"
        assert get_agent_name(agent) == "claude"

    def test_object_with_no_name_attrs(self):
        agent = EmptyAgent()
        assert get_agent_name(agent) is None

    def test_dict_with_empty_agent_name_falls_to_name(self):
        """Empty string agent_name is falsy, should fall to name."""
        agent = {"agent_name": "", "name": "gpt-4"}
        assert get_agent_name(agent) == "gpt-4"

    def test_dict_with_none_agent_name_falls_to_name(self):
        agent = {"agent_name": None, "name": "gpt-4"}
        assert get_agent_name(agent) == "gpt-4"

    def test_string_agent(self):
        """String is not a dict; getattr will be used. Strings have no agent_name."""
        result = get_agent_name("claude")
        # str has no agent_name/name attrs, but str chars are not relevant
        assert result is None

    def test_integer_agent(self):
        """Numeric values have no name attributes."""
        assert get_agent_name(42) is None


# =============================================================================
# agent_to_dict Tests
# =============================================================================


class TestAgentToDict:
    """Tests for agent_to_dict utility."""

    def test_returns_empty_dict_for_none(self):
        assert agent_to_dict(None) == {}

    def test_dict_returns_copy(self):
        original = {"name": "claude", "elo": 1600}
        result = agent_to_dict(original)
        assert result == original
        # Verify it's a copy, not the same object
        assert result is not original

    def test_dict_copy_is_independent(self):
        original = {"name": "claude", "elo": 1600}
        result = agent_to_dict(original)
        result["elo"] = 9999
        assert original["elo"] == 1600

    def test_object_with_all_fields(self):
        agent = MockAgentRating(
            agent_name="claude",
            elo=1650.0,
            wins=20,
            losses=5,
            draws=3,
            win_rate=0.714,
            games_played=28,
            matches=28,
        )
        result = agent_to_dict(agent)
        assert result["name"] == "claude"
        assert result["agent_name"] == "claude"
        assert result["elo"] == 1650.0
        assert result["wins"] == 20
        assert result["losses"] == 5
        assert result["draws"] == 3
        assert result["win_rate"] == 0.714
        assert result["games"] == 28
        assert result["matches"] == 28

    def test_object_with_default_fields(self):
        agent = MockAgentRating()
        result = agent_to_dict(agent)
        assert result["name"] == "claude"
        assert result["elo"] == 1600.0
        assert result["wins"] == 10
        assert result["losses"] == 3

    def test_object_with_minimal_attrs(self):
        """Agent with only name; ELO fields should use defaults."""
        agent = MinimalAgent(name="gpt-4")
        result = agent_to_dict(agent)
        assert result["name"] == "gpt-4"
        assert result["agent_name"] == "gpt-4"
        assert result["elo"] == 1500  # default
        assert result["wins"] == 0  # default
        assert result["losses"] == 0
        assert result["draws"] == 0
        assert result["win_rate"] == 0.0
        assert result["games"] == 0
        assert result["matches"] == 0

    def test_include_name_true_by_default(self):
        agent = MockAgentRating(agent_name="claude")
        result = agent_to_dict(agent)
        assert "name" in result
        assert "agent_name" in result

    def test_include_name_false(self):
        agent = MockAgentRating(agent_name="claude")
        result = agent_to_dict(agent, include_name=False)
        assert "name" not in result
        assert "agent_name" not in result
        assert "elo" in result

    def test_object_with_no_name_uses_unknown(self):
        agent = EmptyAgent()
        result = agent_to_dict(agent)
        assert result["name"] == "unknown"
        assert result["agent_name"] == "unknown"

    def test_object_games_prefers_games_played(self):
        """games_played attribute is preferred over games."""
        agent = MagicMock()
        agent.agent_name = "test"
        agent.name = "test"
        agent.elo = 1500
        agent.wins = 0
        agent.losses = 0
        agent.draws = 0
        agent.win_rate = 0.0
        agent.games_played = 42
        agent.games = 99  # Should NOT be used since games_played exists
        agent.matches = 10
        result = agent_to_dict(agent)
        assert result["games"] == 42

    def test_object_games_falls_back_to_games(self):
        """If games_played is missing, falls back to games attribute."""

        class AgentWithGames:
            agent_name = "fallback-agent"
            elo = 1500
            wins = 0
            losses = 0
            draws = 0
            win_rate = 0.0
            games = 77
            matches = 5

        agent = AgentWithGames()
        result = agent_to_dict(agent)
        assert result["games"] == 77

    def test_dict_include_name_ignored(self):
        """For dict input, include_name param is irrelevant; copy is returned."""
        original = {"name": "test", "elo": 1500}
        result = agent_to_dict(original, include_name=False)
        # Dict returns a copy regardless of include_name
        assert result == original

    def test_none_with_include_name_false(self):
        assert agent_to_dict(None, include_name=False) == {}


# =============================================================================
# normalize_agent_names Tests
# =============================================================================


class TestNormalizeAgentNames:
    """Tests for normalize_agent_names utility."""

    def test_empty_list(self):
        assert normalize_agent_names([]) == []

    def test_string_names_lowered(self):
        result = normalize_agent_names(["Claude", "GPT-4", "GEMINI"])
        assert result == ["claude", "gpt-4", "gemini"]

    def test_already_lowercase(self):
        result = normalize_agent_names(["claude", "gpt-4"])
        assert result == ["claude", "gpt-4"]

    def test_mixed_case(self):
        result = normalize_agent_names(["ClAuDe", "Gpt-4-Turbo"])
        assert result == ["claude", "gpt-4-turbo"]

    def test_object_agents(self):
        agents = [MockAgentRating(agent_name="Claude"), MinimalAgent(name="GPT-4")]
        result = normalize_agent_names(agents)
        assert result == ["claude", "gpt-4"]

    def test_dict_agents(self):
        agents = [{"name": "Claude"}, {"agent_name": "GPT-4"}]
        result = normalize_agent_names(agents)
        assert result == ["claude", "gpt-4"]

    def test_none_names_skipped(self):
        """Agents with no extractable name are excluded."""
        agents = [EmptyAgent(), "Claude", {"elo": 1500}]
        result = normalize_agent_names(agents)
        assert result == ["claude"]

    def test_empty_string_name_skipped(self):
        """Empty string names are falsy and should be skipped."""
        agents = ["", "Claude", ""]
        result = normalize_agent_names(agents)
        assert result == ["claude"]

    def test_mixed_types(self):
        """Mix of strings, dicts, and objects."""
        agents = [
            "Claude",
            {"name": "GPT-4"},
            MockAgentRating(agent_name="Gemini"),
        ]
        result = normalize_agent_names(agents)
        assert result == ["claude", "gpt-4", "gemini"]

    def test_preserves_order(self):
        agents = ["Zebra", "Apple", "Mango"]
        result = normalize_agent_names(agents)
        assert result == ["zebra", "apple", "mango"]


# =============================================================================
# extract_path_segment Tests
# =============================================================================


class TestExtractPathSegment:
    """Tests for extract_path_segment utility."""

    def test_basic_extraction(self):
        path = "/api/v1/debates/123/rounds"
        assert extract_path_segment(path, 0) == "api"
        assert extract_path_segment(path, 1) == "v1"
        assert extract_path_segment(path, 2) == "debates"
        assert extract_path_segment(path, 3) == "123"
        assert extract_path_segment(path, 4) == "rounds"

    def test_index_out_of_range(self):
        path = "/api/v1"
        assert extract_path_segment(path, 5) is None

    def test_index_out_of_range_with_default(self):
        path = "/api/v1"
        assert extract_path_segment(path, 5, "fallback") == "fallback"

    def test_root_path(self):
        assert extract_path_segment("/", 0) is None

    def test_root_path_with_default(self):
        assert extract_path_segment("/", 0, "default") == "default"

    def test_no_leading_slash(self):
        path = "api/v1/debates"
        assert extract_path_segment(path, 0) == "api"
        assert extract_path_segment(path, 2) == "debates"

    def test_trailing_slash(self):
        path = "/api/v1/debates/"
        assert extract_path_segment(path, 0) == "api"
        assert extract_path_segment(path, 2) == "debates"

    def test_single_segment(self):
        assert extract_path_segment("/health", 0) == "health"

    def test_empty_path(self):
        assert extract_path_segment("", 0) is None

    def test_empty_path_with_default(self):
        assert extract_path_segment("", 0, "def") == "def"

    def test_index_zero(self):
        assert extract_path_segment("/first/second", 0) == "first"

    def test_default_is_none_by_default(self):
        assert extract_path_segment("/short", 10) is None

    def test_path_with_special_characters(self):
        path = "/api/v1/agents/claude-3.5-sonnet"
        assert extract_path_segment(path, 3) == "claude-3.5-sonnet"

    def test_path_with_uuid(self):
        path = "/api/v1/debates/550e8400-e29b-41d4-a716-446655440000"
        assert extract_path_segment(path, 3) == "550e8400-e29b-41d4-a716-446655440000"


# =============================================================================
# build_api_url Tests
# =============================================================================


class TestBuildApiUrl:
    """Tests for build_api_url utility."""

    def test_basic_url(self):
        result = build_api_url("api", "v1", "debates")
        assert result == "/api/v1/debates"

    def test_single_segment(self):
        result = build_api_url("health")
        assert result == "/health"

    def test_with_query_params(self):
        result = build_api_url("api", "agents", query_params={"limit": 10})
        assert result == "/api/agents?limit=10"

    def test_with_multiple_query_params(self):
        result = build_api_url(
            "api",
            "debates",
            query_params={"limit": 10, "offset": 20},
        )
        assert "limit=10" in result
        assert "offset=20" in result
        assert result.startswith("/api/debates?")

    def test_query_params_none_values_excluded(self):
        result = build_api_url(
            "api",
            "agents",
            query_params={"limit": 10, "cursor": None},
        )
        assert "cursor" not in result
        assert "limit=10" in result

    def test_query_params_all_none(self):
        """All None values means no query string."""
        result = build_api_url("api", query_params={"a": None, "b": None})
        assert "?" not in result
        assert result == "/api"

    def test_empty_query_params(self):
        result = build_api_url("api", query_params={})
        assert result == "/api"

    def test_no_query_params(self):
        result = build_api_url("api", "v1")
        assert result == "/api/v1"

    def test_strips_slashes_from_segments(self):
        result = build_api_url("/api/", "/v1/", "/debates/")
        assert result == "/api/v1/debates"

    def test_empty_segments_filtered(self):
        result = build_api_url("api", "", "v1")
        assert result == "/api/v1"

    def test_no_segments(self):
        result = build_api_url()
        assert result == "/"

    def test_numeric_segment(self):
        result = build_api_url("api", "debates", 123)
        assert result == "/api/debates/123"

    def test_query_params_with_string_values(self):
        result = build_api_url("api", query_params={"status": "active"})
        assert result == "/api?status=active"

    def test_query_params_with_boolean(self):
        result = build_api_url("api", query_params={"verbose": True})
        assert result == "/api?verbose=True"


# =============================================================================
# is_json_content_type Tests
# =============================================================================


class TestIsJsonContentType:
    """Tests for is_json_content_type utility."""

    def test_application_json(self):
        assert is_json_content_type("application/json") is True

    def test_text_json(self):
        assert is_json_content_type("text/json") is True

    def test_application_json_with_charset(self):
        assert is_json_content_type("application/json; charset=utf-8") is True

    def test_text_json_with_charset(self):
        assert is_json_content_type("text/json; charset=utf-8") is True

    def test_application_json_uppercase(self):
        assert is_json_content_type("Application/JSON") is True

    def test_text_html_is_not_json(self):
        assert is_json_content_type("text/html") is False

    def test_text_plain_is_not_json(self):
        assert is_json_content_type("text/plain") is False

    def test_multipart_form_is_not_json(self):
        assert is_json_content_type("multipart/form-data") is False

    def test_none_returns_false(self):
        assert is_json_content_type(None) is False

    def test_empty_string_returns_false(self):
        assert is_json_content_type("") is False

    def test_application_json_patch(self):
        """application/json-patch+json is NOT recognized (only exact matches)."""
        assert is_json_content_type("application/json-patch+json") is False

    def test_with_extra_params(self):
        assert is_json_content_type("application/json; charset=utf-8; boundary=foo") is True

    def test_whitespace_around_media_type(self):
        assert is_json_content_type("  application/json  ") is True


# =============================================================================
# get_media_type Tests
# =============================================================================


class TestGetMediaType:
    """Tests for get_media_type utility."""

    def test_simple_content_type(self):
        assert get_media_type("application/json") == "application/json"

    def test_strips_charset(self):
        assert get_media_type("application/json; charset=utf-8") == "application/json"

    def test_strips_multiple_params(self):
        assert get_media_type("text/html; charset=utf-8; boundary=foo") == "text/html"

    def test_lowercases_result(self):
        assert get_media_type("Application/JSON") == "application/json"

    def test_strips_whitespace(self):
        assert get_media_type("  text/plain  ") == "text/plain"

    def test_none_returns_empty(self):
        assert get_media_type(None) == ""

    def test_empty_string_returns_empty(self):
        assert get_media_type("") == ""

    def test_multipart_form_data(self):
        assert get_media_type("multipart/form-data; boundary=----") == "multipart/form-data"

    def test_text_xml(self):
        assert get_media_type("text/xml") == "text/xml"

    def test_octet_stream(self):
        assert get_media_type("application/octet-stream") == "application/octet-stream"


# =============================================================================
# Integration / Cross-Function Tests
# =============================================================================


class TestCrossFunctionIntegration:
    """Tests that verify multiple utilities work together correctly."""

    def test_get_agent_name_then_normalize(self):
        """get_agent_name output feeds into normalize_agent_names."""
        agents = [
            MockAgentRating(agent_name="Claude"),
            {"name": "GPT-4"},
            MinimalAgent(name="Gemini"),
        ]
        names = normalize_agent_names(agents)
        assert names == ["claude", "gpt-4", "gemini"]

    def test_agent_to_dict_preserves_name_from_get_agent_name(self):
        """agent_to_dict uses get_agent_name internally."""
        agent = MockAgentRating(agent_name="Claude-3.5")
        result = agent_to_dict(agent)
        assert result["name"] == "Claude-3.5"
        assert result["agent_name"] == "Claude-3.5"

    def test_extract_path_segment_with_build_api_url(self):
        """URL built with build_api_url can be parsed with extract_path_segment."""
        url = build_api_url("api", "v1", "debates", "abc-123")
        assert extract_path_segment(url, 0) == "api"
        assert extract_path_segment(url, 1) == "v1"
        assert extract_path_segment(url, 2) == "debates"
        assert extract_path_segment(url, 3) == "abc-123"

    def test_content_type_and_media_type_consistency(self):
        """is_json_content_type and get_media_type agree on JSON detection."""
        ct = "application/json; charset=utf-8"
        assert is_json_content_type(ct) is True
        assert get_media_type(ct) == "application/json"

    def test_request_header_chain(self):
        """Test extracting multiple headers from the same handler."""
        handler = MockHandlerWithHeaders(
            {
                "Host": "api.aragora.io",
                "X-Request-ID": "req-abc",
                "Content-Length": "512",
            }
        )
        assert get_host_header(handler) == "api.aragora.io"
        assert get_request_id(handler) == "req-abc"
        assert get_content_length(handler) == 512


# =============================================================================
# Edge Cases & Boundary Tests
# =============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_get_agent_name_with_zero_value(self):
        """Zero is falsy but shouldn't appear as a name; dict returns it."""
        assert get_agent_name({"name": 0}) == 0

    def test_agent_to_dict_with_custom_object(self):
        """Arbitrary objects get default ELO values."""

        class CustomObj:
            agent_name = "custom"

        result = agent_to_dict(CustomObj())
        assert result["name"] == "custom"
        assert result["elo"] == 1500

    def test_normalize_agent_names_single(self):
        assert normalize_agent_names(["CLAUDE"]) == ["claude"]

    def test_build_api_url_with_none_query_params(self):
        result = build_api_url("api", query_params=None)
        assert result == "/api"

    def test_extract_path_segment_double_slash(self):
        """Double slashes produce empty segments that get stripped."""
        path = "/api//v1"
        # strip("/") -> "api//v1", split("/") -> ["api", "", "v1"]
        assert extract_path_segment(path, 0) == "api"
        # Index 1 is empty string, which should return default
        assert extract_path_segment(path, 1) is None
        assert extract_path_segment(path, 2) == "v1"

    def test_get_content_length_with_whitespace(self):
        """int() handles leading/trailing whitespace."""
        handler = MockHandlerWithHeaders({"Content-Length": " 256 "})
        assert get_content_length(handler) == 256

    def test_is_json_with_vendor_prefix(self):
        """Vendor JSON types are not recognized as JSON."""
        assert is_json_content_type("application/vnd.api+json") is False

    def test_get_media_type_semicolon_only(self):
        assert get_media_type(";charset=utf-8") == ""

    def test_get_host_header_handler_with_none_default(self):
        """Passing default=None explicitly should still use _DEFAULT_HOST."""
        handler = MockHandlerWithHeaders({})
        result = get_host_header(handler, default=None)
        assert result == _DEFAULT_HOST

    def test_agent_to_dict_dict_with_extra_fields(self):
        """Dict input preserves all fields including non-standard ones."""
        original = {"name": "claude", "elo": 1600, "custom_field": "value"}
        result = agent_to_dict(original)
        assert result["custom_field"] == "value"

    def test_normalize_agent_names_with_none_in_list(self):
        """None values in the list get None from get_agent_name and are skipped."""
        agents = ["Claude", None, "GPT-4"]
        result = normalize_agent_names(agents)
        # None -> get_agent_name(None) -> None -> skipped? No, isinstance check:
        # None is not str, so get_agent_name(None) is called -> returns None -> skipped
        assert result == ["claude", "gpt-4"]


# =============================================================================
# Module-Level Constants Tests
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports and constants."""

    def test_all_exports_importable(self):
        """Verify all names in __all__ are importable."""
        from aragora.server.handlers import utilities

        for name in utilities.__all__:
            assert hasattr(utilities, name), f"{name} listed in __all__ but not defined"

    def test_default_host_is_string(self):
        assert isinstance(_DEFAULT_HOST, str)

    def test_default_host_not_empty(self):
        assert len(_DEFAULT_HOST) > 0
