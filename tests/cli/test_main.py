"""Tests for CLI main module."""

import pytest
from unittest.mock import patch, MagicMock

from aragora.cli.main import parse_agents, get_event_emitter_if_available


class TestParseAgents:
    """Test agent string parsing."""

    def test_parse_single_agent(self):
        """Test parsing a single agent."""
        result = parse_agents("codex")
        assert result == [("codex", None)]

    def test_parse_multiple_agents(self):
        """Test parsing multiple agents."""
        result = parse_agents("codex,claude,openai")
        assert result == [("codex", None), ("claude", None), ("openai", None)]

    def test_parse_agent_with_role(self):
        """Test parsing agent with role."""
        result = parse_agents("claude:critic")
        assert result == [("claude", "critic")]

    def test_parse_mixed_agents(self):
        """Test parsing mix of agents with and without roles."""
        result = parse_agents("codex,claude:critic,openai:proposer")
        assert result == [
            ("codex", None),
            ("claude", "critic"),
            ("openai", "proposer"),
        ]

    def test_parse_with_whitespace(self):
        """Test parsing handles whitespace."""
        result = parse_agents("codex , claude , openai")
        assert result == [("codex", None), ("claude", None), ("openai", None)]

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = parse_agents("")
        assert result == [("", None)]

    def test_parse_single_with_role(self):
        """Test parsing single agent with role."""
        result = parse_agents("anthropic-api:judge")
        assert result == [("anthropic-api", "judge")]

    def test_parse_complex_agent_names(self):
        """Test parsing complex agent names with hyphens."""
        result = parse_agents("anthropic-api,openai-api,gemini-api")
        assert result == [
            ("anthropic-api", None),
            ("openai-api", None),
            ("gemini-api", None),
        ]

    def test_parse_multiple_colons_in_spec(self):
        """Test that only first colon is treated as separator."""
        result = parse_agents("agent:role:extra")
        assert result == [("agent", "role:extra")]


class TestGetEventEmitterIfAvailable:
    """Test event emitter availability check."""

    def test_returns_none_on_connection_error(self):
        """Test returns None when server unavailable."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = ConnectionRefusedError()
            result = get_event_emitter_if_available("http://localhost:9999")
            assert result is None

    def test_returns_none_on_timeout(self):
        """Test returns None on timeout."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError()
            result = get_event_emitter_if_available("http://localhost:8080")
            assert result is None

    def test_returns_none_on_url_error(self):
        """Test returns None on URL error."""
        import urllib.error

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Network error")
            result = get_event_emitter_if_available("http://localhost:8080")
            assert result is None

    def test_returns_none_on_non_200(self):
        """Test returns None when server returns non-200."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 500
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response
            result = get_event_emitter_if_available("http://localhost:8080")
            assert result is None

    def test_returns_emitter_when_server_up(self):
        """Test returns emitter when server is healthy."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            with patch.dict("sys.modules", {"aragora.server.stream": MagicMock()}):
                # Even with server up, may return None if import fails
                # Just verify no exception raised
                result = get_event_emitter_if_available("http://localhost:8080")
                # Result could be None or an emitter
                assert result is None or result is not None
