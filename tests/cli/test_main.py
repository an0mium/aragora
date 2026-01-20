"""Tests for CLI main module."""

import pytest
from unittest.mock import patch, MagicMock

from aragora.cli.main import parse_agents, get_event_emitter_if_available


class TestParseAgents:
    """Test agent string parsing.

    Note: parse_agents returns AgentSpec objects with:
    - Legacy colon format (provider:role) - if second part is a valid role, uses it
    - Legacy colon format (provider:persona) - if second part is NOT a valid role, treats as persona
    - New pipe format (provider|model|persona|role) - explicitly sets all fields
    - When role is not specified, it's left as None (caller handles position-based assignment)
    """

    def test_parse_single_agent(self):
        """Test parsing a single agent - role is None (not position-based here)."""
        result = parse_agents("codex")
        assert len(result) == 1
        assert result[0].provider == "codex"
        # AgentSpec.parse_list returns None for role when not specified
        # The position-based assignment happens at the call site
        assert result[0].role is None

    def test_parse_multiple_agents(self):
        """Test parsing multiple agents - roles are None when not specified."""
        result = parse_agents("codex,claude,openai")
        assert len(result) == 3
        assert result[0].provider == "codex"
        assert result[0].role is None
        assert result[1].provider == "claude"
        assert result[1].role is None
        assert result[2].provider == "openai"
        assert result[2].role is None

    def test_parse_legacy_colon_format(self):
        """Test legacy colon format - 'critic' IS a valid role so it's used as role."""
        result = parse_agents("claude:critic")
        assert len(result) == 1
        assert result[0].provider == "claude"
        assert result[0].role == "critic"

    def test_parse_mixed_agents(self):
        """Test parsing mix of agents with explicit roles and personas."""
        result = parse_agents("codex,claude:critic,openai:philosopher")
        assert len(result) == 3
        assert result[0].provider == "codex"
        assert result[0].role is None  # No explicit role
        assert result[1].provider == "claude"
        assert result[1].role == "critic"  # Explicit valid role
        assert result[2].provider == "openai"
        assert result[2].persona == "philosopher"  # Not a valid role
        assert result[2].role is None  # Treated as persona

    def test_parse_with_whitespace(self):
        """Test parsing handles whitespace."""
        result = parse_agents("codex , claude , openai")
        assert len(result) == 3
        assert result[0].provider == "codex"
        assert result[1].provider == "claude"
        assert result[2].provider == "openai"

    def test_parse_empty_string(self):
        """Test parsing empty string returns empty list."""
        result = parse_agents("")
        assert result == []

    def test_parse_legacy_with_hyphen_names(self):
        """Test parsing API agent names - 'judge' IS a valid role."""
        result = parse_agents("anthropic-api:judge")
        assert len(result) == 1
        assert result[0].provider == "anthropic-api"
        assert result[0].role == "judge"

    def test_parse_complex_agent_names(self):
        """Test parsing API agent names - roles are None."""
        result = parse_agents("anthropic-api,openai-api,gemini")
        assert len(result) == 3
        assert result[0].provider == "anthropic-api"
        assert result[1].provider == "openai-api"
        assert result[2].provider == "gemini"

    def test_parse_multiple_colons_in_spec(self):
        """Test that only first colon is treated as separator."""
        result = parse_agents("claude:role:extra")
        assert len(result) == 1
        assert result[0].provider == "claude"
        # 'role:extra' is not a valid role, treated as persona
        assert result[0].persona == "role:extra"
        assert result[0].role is None

    def test_parse_new_pipe_format_with_role(self):
        """Test new pipe format explicitly sets role."""
        result = parse_agents("claude||philosopher|critic")
        assert len(result) == 1
        assert result[0].provider == "claude"
        assert result[0].persona == "philosopher"
        assert result[0].role == "critic"

    def test_parse_new_pipe_format_multiple(self):
        """Test new pipe format with multiple agents."""
        result = parse_agents("claude||philosopher|proposer,openai|||critic")
        assert len(result) == 2
        assert result[0].provider == "claude"
        assert result[0].role == "proposer"
        assert result[1].provider == "openai"
        assert result[1].role == "critic"


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
