"""
Tests for handler utilities module.

Tests cover:
- Request utilities (get_host_header, get_request_id, get_content_length)
- Agent utilities (get_agent_name, agent_to_dict, normalize_agent_names)
- Path utilities (extract_path_segment, build_api_url)
- Content type utilities (is_json_content_type, get_media_type)
"""

from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

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


class TestGetHostHeader:
    """Tests for get_host_header function."""

    def test_returns_header_value(self):
        """Test extracting Host header from handler."""
        handler = MagicMock()
        handler.headers = {"Host": "example.com"}

        result = get_host_header(handler)
        assert result == "example.com"

    def test_returns_default_when_handler_is_none(self):
        """Test default when handler is None."""
        result = get_host_header(None)
        assert result == "localhost:8080"

    def test_returns_custom_default_when_handler_is_none(self):
        """Test custom default when handler is None."""
        result = get_host_header(None, default="custom.host:9000")
        assert result == "custom.host:9000"

    def test_returns_default_when_no_host_header(self):
        """Test default when Host header is missing."""
        handler = MagicMock()
        handler.headers = {}

        result = get_host_header(handler)
        assert result == "localhost:8080"

    def test_returns_default_when_handler_has_no_headers(self):
        """Test default when handler has no headers attribute."""
        handler = MagicMock(spec=[])  # Empty spec, no headers
        del handler.headers

        result = get_host_header(handler)
        assert result == "localhost:8080"

    def test_returns_custom_default_when_header_missing(self):
        """Test custom default with explicit value."""
        handler = MagicMock()
        handler.headers = {}

        result = get_host_header(handler, default="fallback.host")
        assert result == "fallback.host"

    @patch.dict("os.environ", {"ARAGORA_DEFAULT_HOST": "env.host:5000"})
    def test_uses_environment_default(self):
        """Test that environment variable is used for default."""
        # Need to reimport to pick up env change
        from aragora.server.handlers import utilities

        # Force re-read of _DEFAULT_HOST
        original = utilities._DEFAULT_HOST
        utilities._DEFAULT_HOST = "env.host:5000"
        try:
            result = get_host_header(None)
            assert result == "env.host:5000"
        finally:
            utilities._DEFAULT_HOST = original


class TestGetRequestId:
    """Tests for get_request_id function."""

    def test_returns_none_when_handler_is_none(self):
        """Test None handler."""
        result = get_request_id(None)
        assert result is None

    def test_returns_none_when_handler_has_no_headers(self):
        """Test handler without headers attribute."""
        handler = MagicMock(spec=[])
        del handler.headers

        result = get_request_id(handler)
        assert result is None

    def test_extracts_x_request_id(self):
        """Test extracting X-Request-ID header."""
        handler = MagicMock()
        handler.headers = {"X-Request-ID": "req-123"}

        result = get_request_id(handler)
        assert result == "req-123"

    def test_extracts_x_trace_id(self):
        """Test extracting X-Trace-ID header."""
        handler = MagicMock()
        handler.headers = {"X-Trace-ID": "trace-456"}

        result = get_request_id(handler)
        assert result == "trace-456"

    def test_extracts_x_correlation_id(self):
        """Test extracting X-Correlation-ID header."""
        handler = MagicMock()
        handler.headers = {"X-Correlation-ID": "corr-789"}

        result = get_request_id(handler)
        assert result == "corr-789"

    def test_prefers_x_request_id_over_others(self):
        """Test header priority order."""
        handler = MagicMock()
        handler.headers = {
            "X-Request-ID": "req-123",
            "X-Trace-ID": "trace-456",
            "X-Correlation-ID": "corr-789",
        }

        result = get_request_id(handler)
        assert result == "req-123"

    def test_returns_none_when_no_id_headers(self):
        """Test when no ID headers present."""
        handler = MagicMock()
        handler.headers = {"Content-Type": "application/json"}

        result = get_request_id(handler)
        assert result is None


class TestGetContentLength:
    """Tests for get_content_length function."""

    def test_returns_zero_when_no_headers(self):
        """Test handler without headers."""
        handler = MagicMock(spec=[])
        del handler.headers

        result = get_content_length(handler)
        assert result == 0

    def test_returns_content_length_value(self):
        """Test extracting valid Content-Length."""
        handler = MagicMock()
        handler.headers = {"Content-Length": "1024"}

        result = get_content_length(handler)
        assert result == 1024

    def test_returns_zero_when_header_missing(self):
        """Test when Content-Length is missing."""
        handler = MagicMock()
        handler.headers = {}

        result = get_content_length(handler)
        assert result == 0

    def test_returns_zero_when_header_invalid(self):
        """Test invalid Content-Length value."""
        handler = MagicMock()
        handler.headers = {"Content-Length": "not-a-number"}

        result = get_content_length(handler)
        assert result == 0

    def test_returns_zero_when_header_is_none(self):
        """Test None Content-Length value."""
        handler = MagicMock()
        handler.headers = {"Content-Length": None}

        result = get_content_length(handler)
        assert result == 0


class TestGetAgentName:
    """Tests for get_agent_name function."""

    def test_returns_none_for_none_input(self):
        """Test None input."""
        result = get_agent_name(None)
        assert result is None

    def test_extracts_name_from_dict_with_name(self):
        """Test dict with 'name' key."""
        result = get_agent_name({"name": "claude"})
        assert result == "claude"

    def test_extracts_name_from_dict_with_agent_name(self):
        """Test dict with 'agent_name' key."""
        result = get_agent_name({"agent_name": "gpt-4"})
        assert result == "gpt-4"

    def test_prefers_agent_name_over_name_in_dict(self):
        """Test priority of agent_name over name."""
        result = get_agent_name({"agent_name": "preferred", "name": "fallback"})
        assert result == "preferred"

    def test_extracts_name_from_object_with_name(self):
        """Test object with name attribute."""
        @dataclass
        class Agent:
            name: str

        result = get_agent_name(Agent(name="gemini"))
        assert result == "gemini"

    def test_extracts_name_from_object_with_agent_name(self):
        """Test object with agent_name attribute."""
        @dataclass
        class Agent:
            agent_name: str

        result = get_agent_name(Agent(agent_name="llama"))
        assert result == "llama"

    def test_returns_none_for_dict_without_name(self):
        """Test dict without name keys."""
        result = get_agent_name({"other": "value"})
        assert result is None

    def test_returns_none_for_object_without_name(self):
        """Test object without name attributes."""
        @dataclass
        class NotAnAgent:
            value: str

        result = get_agent_name(NotAnAgent(value="test"))
        assert result is None


class TestAgentToDict:
    """Tests for agent_to_dict function."""

    def test_returns_empty_dict_for_none(self):
        """Test None input."""
        result = agent_to_dict(None)
        assert result == {}

    def test_returns_copy_for_dict_input(self):
        """Test dict input is copied."""
        original = {"name": "claude", "elo": 1600}
        result = agent_to_dict(original)

        assert result == original
        assert result is not original  # Should be a copy

    def test_extracts_standard_fields_from_object(self):
        """Test extracting ELO fields from object."""
        @dataclass
        class Agent:
            name: str
            elo: int = 1500
            wins: int = 0
            losses: int = 0
            draws: int = 0
            win_rate: float = 0.0
            games_played: int = 0
            matches: int = 0

        agent = Agent(name="claude", elo=1650, wins=10, losses=5)
        result = agent_to_dict(agent)

        assert result["name"] == "claude"
        assert result["agent_name"] == "claude"
        assert result["elo"] == 1650
        assert result["wins"] == 10
        assert result["losses"] == 5
        assert result["draws"] == 0

    def test_uses_defaults_for_missing_attributes(self):
        """Test default values for missing attributes."""
        @dataclass
        class MinimalAgent:
            name: str

        result = agent_to_dict(MinimalAgent(name="minimal"))

        assert result["elo"] == 1500
        assert result["wins"] == 0
        assert result["losses"] == 0

    def test_exclude_name_when_include_name_false(self):
        """Test excluding name fields."""
        @dataclass
        class Agent:
            name: str

        result = agent_to_dict(Agent(name="claude"), include_name=False)

        assert "name" not in result
        assert "agent_name" not in result

    def test_uses_games_attribute_fallback(self):
        """Test games_played vs games attribute fallback."""
        @dataclass
        class AgentWithGames:
            name: str
            games: int

        result = agent_to_dict(AgentWithGames(name="test", games=50))
        assert result["games"] == 50


class TestNormalizeAgentNames:
    """Tests for normalize_agent_names function."""

    def test_returns_empty_list_for_empty_input(self):
        """Test empty input."""
        result = normalize_agent_names([])
        assert result == []

    def test_normalizes_string_names(self):
        """Test normalizing string names to lowercase."""
        result = normalize_agent_names(["Claude", "GPT-4", "GEMINI"])
        assert result == ["claude", "gpt-4", "gemini"]

    def test_extracts_names_from_objects(self):
        """Test extracting names from objects."""
        @dataclass
        class Agent:
            name: str

        agents = [Agent(name="Claude"), Agent(name="GPT-4")]
        result = normalize_agent_names(agents)

        assert result == ["claude", "gpt-4"]

    def test_extracts_names_from_dicts(self):
        """Test extracting names from dicts."""
        agents = [{"name": "Claude"}, {"agent_name": "GPT-4"}]
        result = normalize_agent_names(agents)

        assert result == ["claude", "gpt-4"]

    def test_skips_none_names(self):
        """Test skipping entries with no name."""
        result = normalize_agent_names(["Claude", {"other": "val"}, "GPT-4"])
        assert result == ["claude", "gpt-4"]

    def test_handles_mixed_input_types(self):
        """Test mixed strings, dicts, and objects."""
        @dataclass
        class Agent:
            name: str

        agents = ["String", {"name": "Dict"}, Agent(name="Object")]
        result = normalize_agent_names(agents)

        assert result == ["string", "dict", "object"]


class TestExtractPathSegment:
    """Tests for extract_path_segment function."""

    def test_extracts_segment_at_index(self):
        """Test extracting segment by index."""
        path = "/api/debates/123/rounds"

        assert extract_path_segment(path, 0) == "api"
        assert extract_path_segment(path, 1) == "debates"
        assert extract_path_segment(path, 2) == "123"
        assert extract_path_segment(path, 3) == "rounds"

    def test_returns_none_for_out_of_bounds(self):
        """Test out of bounds index."""
        path = "/api/debates"

        result = extract_path_segment(path, 5)
        assert result is None

    def test_returns_custom_default_for_out_of_bounds(self):
        """Test custom default for out of bounds."""
        path = "/api/debates"

        result = extract_path_segment(path, 5, default="fallback")
        assert result == "fallback"

    def test_handles_path_with_trailing_slash(self):
        """Test path with trailing slash."""
        path = "/api/debates/"

        assert extract_path_segment(path, 0) == "api"
        assert extract_path_segment(path, 1) == "debates"

    def test_handles_path_without_leading_slash(self):
        """Test path without leading slash."""
        path = "api/debates/123"

        assert extract_path_segment(path, 0) == "api"
        assert extract_path_segment(path, 2) == "123"

    def test_returns_default_for_empty_segment(self):
        """Test empty segment returns default."""
        path = "/api//debates"

        # Empty segment at index 1 should return default
        result = extract_path_segment(path, 1, default="default")
        assert result == "default"


class TestBuildApiUrl:
    """Tests for build_api_url function."""

    def test_builds_simple_url(self):
        """Test building simple URL."""
        result = build_api_url("api", "debates", "123")
        assert result == "/api/debates/123"

    def test_handles_single_segment(self):
        """Test single segment."""
        result = build_api_url("api")
        assert result == "/api"

    def test_handles_empty_segments(self):
        """Test empty/None segments are skipped."""
        result = build_api_url("api", "", "debates", None, "123")
        assert result == "/api/debates/123"

    def test_strips_slashes_from_segments(self):
        """Test stripping slashes from segments."""
        result = build_api_url("/api/", "/debates/", "/123/")
        assert result == "/api/debates/123"

    def test_adds_query_params(self):
        """Test adding query parameters."""
        result = build_api_url("api", "agents", query_params={"limit": 10})
        assert result == "/api/agents?limit=10"

    def test_adds_multiple_query_params(self):
        """Test multiple query parameters."""
        result = build_api_url("api", "agents", query_params={"limit": 10, "offset": 20})
        assert "limit=10" in result
        assert "offset=20" in result

    def test_skips_none_query_params(self):
        """Test None query params are skipped."""
        result = build_api_url("api", "agents", query_params={"limit": 10, "offset": None})
        assert result == "/api/agents?limit=10"

    def test_handles_empty_query_params(self):
        """Test empty query params dict."""
        result = build_api_url("api", "agents", query_params={})
        assert result == "/api/agents"


class TestIsJsonContentType:
    """Tests for is_json_content_type function."""

    def test_returns_false_for_none(self):
        """Test None input."""
        assert is_json_content_type(None) is False

    def test_returns_false_for_empty_string(self):
        """Test empty string."""
        assert is_json_content_type("") is False

    def test_recognizes_application_json(self):
        """Test application/json."""
        assert is_json_content_type("application/json") is True

    def test_recognizes_text_json(self):
        """Test text/json."""
        assert is_json_content_type("text/json") is True

    def test_handles_charset_parameter(self):
        """Test content type with charset."""
        assert is_json_content_type("application/json; charset=utf-8") is True

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert is_json_content_type("APPLICATION/JSON") is True
        assert is_json_content_type("Application/Json") is True

    def test_returns_false_for_non_json(self):
        """Test non-JSON content types."""
        assert is_json_content_type("text/html") is False
        assert is_json_content_type("application/xml") is False
        assert is_json_content_type("text/plain") is False


class TestGetMediaType:
    """Tests for get_media_type function."""

    def test_returns_empty_for_none(self):
        """Test None input."""
        assert get_media_type(None) == ""

    def test_returns_empty_for_empty_string(self):
        """Test empty string."""
        assert get_media_type("") == ""

    def test_extracts_media_type(self):
        """Test extracting media type."""
        assert get_media_type("application/json") == "application/json"

    def test_strips_charset(self):
        """Test stripping charset parameter."""
        assert get_media_type("application/json; charset=utf-8") == "application/json"

    def test_strips_multiple_parameters(self):
        """Test stripping multiple parameters."""
        result = get_media_type("text/html; charset=utf-8; boundary=something")
        assert result == "text/html"

    def test_lowercases_result(self):
        """Test result is lowercase."""
        assert get_media_type("Application/JSON") == "application/json"

    def test_handles_whitespace(self):
        """Test handling whitespace."""
        assert get_media_type("  application/json  ") == "application/json"
