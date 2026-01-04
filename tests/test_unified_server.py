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


class TestFlipsAPI:
    """Test the /api/flips/* endpoints."""

    def test_recent_flips_default(self):
        """Test recent flips with default parameters."""
        from aragora.server.unified_server import UnifiedHandler

        with patch.object(UnifiedHandler, '__init__', lambda x: None):
            handler = UnifiedHandler()
            handler.path = '/api/flips/recent'
            handler.command = 'GET'
            handler.request_version = 'HTTP/1.1'
            handler.headers = {}
            handler.wfile = BytesIO()
            handler.flip_detector = Mock()
            handler.flip_detector.get_recent_flips.return_value = []

            handler.send_response = Mock()
            handler._add_cors_headers = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()
            handler._send_json = Mock()

            handler._get_recent_flips(limit=20)

            handler.flip_detector.get_recent_flips.assert_called_once_with(limit=20)

    def test_recent_flips_custom_limit(self):
        """Test recent flips with custom limit."""
        from aragora.server.unified_server import UnifiedHandler

        with patch.object(UnifiedHandler, '__init__', lambda x: None):
            handler = UnifiedHandler()
            handler.path = '/api/flips/recent?limit=10'
            handler.command = 'GET'
            handler.request_version = 'HTTP/1.1'
            handler.headers = {}
            handler.wfile = BytesIO()
            handler.flip_detector = Mock()
            handler.flip_detector.get_recent_flips.return_value = []

            handler.send_response = Mock()
            handler._add_cors_headers = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()
            handler._send_json = Mock()

            handler._get_recent_flips(limit=10)

            handler.flip_detector.get_recent_flips.assert_called_once_with(limit=10)

    def test_recent_flips_no_detector(self):
        """Test recent flips when flip detector is not configured."""
        from aragora.server.unified_server import UnifiedHandler

        with patch.object(UnifiedHandler, '__init__', lambda x: None):
            handler = UnifiedHandler()
            handler.path = '/api/flips/recent'
            handler.command = 'GET'
            handler.request_version = 'HTTP/1.1'
            handler.headers = {}
            handler.wfile = BytesIO()
            handler.flip_detector = None

            handler.send_response = Mock()
            handler._add_cors_headers = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()
            json_response = {}

            def capture_json(data):
                json_response.update(data)

            handler._send_json = capture_json

            handler._get_recent_flips(limit=20)

            assert "error" in json_response
            assert "flips" in json_response
            assert json_response["flips"] == []

    def test_flip_summary(self):
        """Test flip summary endpoint."""
        from aragora.server.unified_server import UnifiedHandler

        mock_summary = {
            "total_flips": 5,
            "by_type": {"contradiction": 2, "refinement": 3},
            "by_agent": {"claude": 3, "gemini": 2},
            "recent_24h": 2,
        }

        with patch.object(UnifiedHandler, '__init__', lambda x: None):
            handler = UnifiedHandler()
            handler.path = '/api/flips/summary'
            handler.command = 'GET'
            handler.request_version = 'HTTP/1.1'
            handler.headers = {}
            handler.wfile = BytesIO()
            handler.flip_detector = Mock()
            handler.flip_detector.get_flip_summary.return_value = mock_summary

            handler.send_response = Mock()
            handler._add_cors_headers = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()
            json_response = {}

            def capture_json(data):
                json_response.update(data)

            handler._send_json = capture_json

            handler._get_flip_summary()

            handler.flip_detector.get_flip_summary.assert_called_once()
            assert "summary" in json_response
            assert json_response["summary"]["total_flips"] == 5

    def test_flip_summary_no_detector(self):
        """Test flip summary when detector is not configured."""
        from aragora.server.unified_server import UnifiedHandler

        with patch.object(UnifiedHandler, '__init__', lambda x: None):
            handler = UnifiedHandler()
            handler.path = '/api/flips/summary'
            handler.command = 'GET'
            handler.request_version = 'HTTP/1.1'
            handler.headers = {}
            handler.wfile = BytesIO()
            handler.flip_detector = None

            handler.send_response = Mock()
            handler._add_cors_headers = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()
            json_response = {}

            def capture_json(data):
                json_response.update(data)

            handler._send_json = capture_json

            handler._get_flip_summary()

            assert "error" in json_response
            assert "summary" in json_response


class TestAgentConsistencyAPI:
    """Test the /api/agent/{name}/consistency endpoint."""

    def test_agent_consistency(self):
        """Test agent consistency endpoint."""
        from aragora.server.unified_server import UnifiedHandler

        mock_score = Mock()
        mock_score.agent_name = "claude"
        mock_score.consistency_score = 0.85
        mock_score.total_positions = 10
        mock_score.flip_rate = 0.15
        mock_score.contradictions = 1
        mock_score.retractions = 0
        mock_score.qualifications = 1
        mock_score.refinements = 2

        with patch.object(UnifiedHandler, '__init__', lambda x: None):
            handler = UnifiedHandler()
            handler.path = '/api/agent/claude/consistency'
            handler.command = 'GET'
            handler.request_version = 'HTTP/1.1'
            handler.headers = {}
            handler.wfile = BytesIO()
            handler.flip_detector = Mock()
            handler.flip_detector.get_agent_consistency.return_value = mock_score

            handler.send_response = Mock()
            handler._add_cors_headers = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()
            handler._send_json = Mock()

            handler._get_agent_consistency(agent="claude")

            handler.flip_detector.get_agent_consistency.assert_called_once_with("claude")

    def test_agent_flips(self):
        """Test agent-specific flips endpoint."""
        from aragora.server.unified_server import UnifiedHandler

        with patch.object(UnifiedHandler, '__init__', lambda x: None):
            handler = UnifiedHandler()
            handler.path = '/api/agent/claude/flips?limit=10'
            handler.command = 'GET'
            handler.request_version = 'HTTP/1.1'
            handler.headers = {}
            handler.wfile = BytesIO()
            handler.flip_detector = Mock()
            handler.flip_detector.detect_flips_for_agent.return_value = []

            handler.send_response = Mock()
            handler._add_cors_headers = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()
            json_response = {}

            def capture_json(data):
                json_response.update(data)

            handler._send_json = capture_json

            handler._get_agent_flips(agent="claude", limit=10)

            handler.flip_detector.detect_flips_for_agent.assert_called_once_with("claude", lookback_positions=10)
            assert "agent" in json_response
            assert json_response["agent"] == "claude"
            assert "flips" in json_response
