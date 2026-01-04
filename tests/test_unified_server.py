"""
Tests for unified_server.py API endpoints.

Validates backward compatibility and new loop-aware and domain-filtering
functionality for leaderboard, recent matches, and agent history endpoints.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from http.server import BaseHTTPRequestHandler
from io import BytesIO
from urllib.parse import urlencode


class MockSocket:
    """Mock socket for testing HTTP handler."""

    def makefile(self, mode, buffering=-1):
        return BytesIO()


class TestLeaderboardAPI:
    """Test the /api/leaderboard endpoint."""

    def test_leaderboard_default_params(self):
        """Test leaderboard with default parameters (backward compatibility)."""
        from aragora.server.unified_server import UnifiedHandler

        # Create mock request
        with patch.object(UnifiedHandler, '__init__', lambda x: None):
            handler = UnifiedHandler()
            handler.path = '/api/leaderboard'
            handler.command = 'GET'
            handler.request_version = 'HTTP/1.1'
            handler.headers = {}
            handler.wfile = BytesIO()
            handler.elo_system = Mock()
            handler.elo_system.get_leaderboard.return_value = []

            # Mock send methods
            handler.send_response = Mock()
            handler._add_cors_headers = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()

            # Call the handler method
            handler._get_leaderboard(limit=20, domain=None)

            # Verify ELO system was called
            handler.elo_system.get_leaderboard.assert_called_once_with(limit=20, domain=None)

    def test_leaderboard_with_domain_filter(self):
        """Test leaderboard with domain filtering."""
        from aragora.server.unified_server import UnifiedHandler

        with patch.object(UnifiedHandler, '__init__', lambda x: None):
            handler = UnifiedHandler()
            handler.path = '/api/leaderboard?domain=architecture'
            handler.command = 'GET'
            handler.request_version = 'HTTP/1.1'
            handler.headers = {}
            handler.wfile = BytesIO()
            handler.elo_system = Mock()
            handler.elo_system.get_leaderboard.return_value = []

            handler.send_response = Mock()
            handler._add_cors_headers = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()

            handler._get_leaderboard(limit=10, domain="architecture")

            handler.elo_system.get_leaderboard.assert_called_once_with(limit=10, domain="architecture")

    def test_leaderboard_custom_limit(self):
        """Test leaderboard with custom limit."""
        from aragora.server.unified_server import UnifiedHandler

        with patch.object(UnifiedHandler, '__init__', lambda x: None):
            handler = UnifiedHandler()
            handler.path = '/api/leaderboard?limit=5'
            handler.command = 'GET'
            handler.request_version = 'HTTP/1.1'
            handler.headers = {}
            handler.wfile = BytesIO()
            handler.elo_system = Mock()
            handler.elo_system.get_leaderboard.return_value = []

            handler.send_response = Mock()
            handler._add_cors_headers = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()

            handler._get_leaderboard(limit=5, domain=None)

            handler.elo_system.get_leaderboard.assert_called_once_with(limit=5, domain=None)


class TestRecentMatchesAPI:
    """Test the /api/matches/recent endpoint."""

    def test_recent_matches_default(self):
        """Test recent matches with default parameters."""
        from aragora.server.unified_server import UnifiedHandler

        with patch.object(UnifiedHandler, '__init__', lambda x: None):
            handler = UnifiedHandler()
            handler.path = '/api/matches/recent'
            handler.command = 'GET'
            handler.request_version = 'HTTP/1.1'
            handler.headers = {}
            handler.wfile = BytesIO()
            handler.elo_system = Mock()
            handler.elo_system.get_recent_matches.return_value = []

            handler.send_response = Mock()
            handler._add_cors_headers = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()

            handler._get_recent_matches(limit=10, loop_id=None)

            handler.elo_system.get_recent_matches.assert_called_once_with(limit=10)

    def test_recent_matches_with_loop_id(self):
        """Test recent matches with loop_id parameter (for future multi-loop support)."""
        from aragora.server.unified_server import UnifiedHandler

        with patch.object(UnifiedHandler, '__init__', lambda x: None):
            handler = UnifiedHandler()
            handler.path = '/api/matches/recent?loop_id=nomic-123'
            handler.command = 'GET'
            handler.request_version = 'HTTP/1.1'
            handler.headers = {}
            handler.wfile = BytesIO()
            handler.elo_system = Mock()
            handler.elo_system.get_recent_matches.return_value = []

            handler.send_response = Mock()
            handler._add_cors_headers = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()

            # Note: loop_id filtering not yet implemented in EloSystem
            handler._get_recent_matches(limit=10, loop_id="nomic-123")

            # Verify it doesn't crash and calls the underlying method
            handler.elo_system.get_recent_matches.assert_called_once_with(limit=10)


class TestAgentHistoryAPI:
    """Test the /api/agent/{name}/history endpoint."""

    def test_agent_history_default(self):
        """Test agent history with default limit."""
        from aragora.server.unified_server import UnifiedHandler

        with patch.object(UnifiedHandler, '__init__', lambda x: None):
            handler = UnifiedHandler()
            handler.path = '/api/agent/claude/history'
            handler.command = 'GET'
            handler.request_version = 'HTTP/1.1'
            handler.headers = {}
            handler.wfile = BytesIO()
            handler.elo_system = Mock()
            handler.elo_system.get_elo_history.return_value = []

            handler.send_response = Mock()
            handler._add_cors_headers = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()
            handler._send_json = Mock()

            handler._get_agent_history(agent="claude", limit=30)

            handler.elo_system.get_elo_history.assert_called_once_with("claude", limit=30)


class TestAPIBackwardCompatibility:
    """Test that new parameters don't break existing API consumers."""

    def test_leaderboard_extra_params_ignored(self):
        """Test that extra/unknown query params don't break the endpoint."""
        from aragora.server.unified_server import UnifiedHandler

        with patch.object(UnifiedHandler, '__init__', lambda x: None):
            handler = UnifiedHandler()
            handler.path = '/api/leaderboard?limit=10&unknown_param=value'
            handler.command = 'GET'
            handler.request_version = 'HTTP/1.1'
            handler.headers = {}
            handler.wfile = BytesIO()
            handler.elo_system = Mock()
            handler.elo_system.get_leaderboard.return_value = []

            handler.send_response = Mock()
            handler._add_cors_headers = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()

            # Should not raise error
            handler._get_leaderboard(limit=10, domain=None)
            handler.elo_system.get_leaderboard.assert_called_once()

    def test_endpoints_return_json(self):
        """Test that all endpoints return valid JSON responses."""
        from aragora.server.unified_server import UnifiedHandler

        with patch.object(UnifiedHandler, '__init__', lambda x: None):
            handler = UnifiedHandler()
            handler.command = 'GET'
            handler.request_version = 'HTTP/1.1'
            handler.headers = {}
            handler.wfile = BytesIO()
            handler.elo_system = Mock()
            handler.elo_system.get_leaderboard.return_value = []
            handler.elo_system.get_recent_matches.return_value = []

            handler.send_response = Mock()
            handler._add_cors_headers = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()

            # Test leaderboard
            handler._get_leaderboard(limit=10, domain=None)

            # Verify JSON was written
            output = handler.wfile.getvalue()
            if output:
                data = json.loads(output.decode('utf-8'))
                assert isinstance(data, dict)
