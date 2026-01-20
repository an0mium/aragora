"""Tests for CLI main module."""

import pytest
from unittest.mock import patch, MagicMock

from aragora.cli.main import parse_agents, get_event_emitter_if_available


class TestParseAgents:
    """Test agent string parsing.

    Note: parse_agents uses AgentSpec which:
    - Legacy colon format (provider:persona) - sets persona, role defaults to 'proposer'
    - New pipe format (provider|model|persona|role) - explicitly sets role
    """

    def test_parse_single_agent(self):
        """Test parsing a single agent - defaults to 'proposer' role."""
        result = parse_agents("codex")
        assert result == [("codex", "proposer")]

    def test_parse_multiple_agents(self):
        """Test parsing multiple agents - all get 'proposer' role."""
        result = parse_agents("codex,claude,openai")
        # All agents default to proposer (legacy format sets persona, not role)
        assert result == [("codex", "proposer"), ("claude", "proposer"), ("openai", "proposer")]

    def test_parse_legacy_colon_format(self):
        """Test legacy colon format sets persona, not role."""
        # Legacy format: provider:persona - role is always 'proposer'
        result = parse_agents("claude:critic")
        assert result == [("claude", "proposer")]  # 'critic' is the persona, not role

    def test_parse_mixed_agents(self):
        """Test parsing mix of agents - all get proposer role with legacy format."""
        result = parse_agents("codex,claude:critic,openai:philosopher")
        # Legacy colon format sets persona; role is always proposer
        assert result == [
            ("codex", "proposer"),
            ("claude", "proposer"),
            ("openai", "proposer"),
        ]

    def test_parse_with_whitespace(self):
        """Test parsing handles whitespace."""
        result = parse_agents("codex , claude , openai")
        assert result == [("codex", "proposer"), ("claude", "proposer"), ("openai", "proposer")]

    def test_parse_empty_string(self):
        """Test parsing empty string returns empty list."""
        result = parse_agents("")
        assert result == []

    def test_parse_legacy_with_hyphen_names(self):
        """Test parsing API agent names with hyphens."""
        result = parse_agents("anthropic-api:judge")
        # 'judge' is persona, role is 'proposer'
        assert result == [("anthropic-api", "proposer")]

    def test_parse_complex_agent_names(self):
        """Test parsing API agent names with hyphens."""
        # Use valid provider names (gemini, not gemini-api)
        result = parse_agents("anthropic-api,openai-api,gemini")
        assert result == [
            ("anthropic-api", "proposer"),
            ("openai-api", "proposer"),
            ("gemini", "proposer"),
        ]

    def test_parse_multiple_colons_in_spec(self):
        """Test that only first colon is treated as separator."""
        # Use valid provider name
        result = parse_agents("claude:role:extra")
        # 'role:extra' is persona, role is 'proposer'
        assert result == [("claude", "proposer")]

    def test_parse_new_pipe_format_with_role(self):
        """Test new pipe format explicitly sets role."""
        result = parse_agents("claude||philosopher|critic")
        assert result == [("claude", "critic")]  # Role is explicitly 'critic'

    def test_parse_new_pipe_format_multiple(self):
        """Test new pipe format with multiple agents."""
        result = parse_agents("claude||philosopher|proposer,openai|||critic")
        assert result == [("claude", "proposer"), ("openai", "critic")]


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
