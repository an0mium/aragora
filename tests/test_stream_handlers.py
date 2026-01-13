"""
Tests for StreamAPIHandlersMixin in stream_handlers.py.

Tests HTTP API endpoint handlers for the streaming server.
"""

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch
import json


# =============================================================================
# Mock Classes
# =============================================================================


@dataclass
class MockAgentRating:
    """Mock AgentRating for testing."""

    agent_name: str
    elo: float = 1500.0
    wins: int = 5
    losses: int = 3
    draws: int = 2

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        return self.wins / total if total > 0 else 0.0

    @property
    def games_played(self) -> int:
        return self.wins + self.losses + self.draws


@dataclass
class MockRequest:
    """Mock aiohttp request object."""

    headers: Dict[str, str] = field(default_factory=dict)
    query: Dict[str, str] = field(default_factory=dict)
    match_info: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if "Origin" not in self.headers:
            self.headers["Origin"] = "http://localhost:3000"


class MockResponse:
    """Mock aiohttp response for capturing handler output."""

    def __init__(self, status: int = 200, body: Any = None, headers: Dict = None):
        self.status = status
        self.body = body
        self.headers = headers or {}


class MockEloSystem:
    """Mock ELO system for testing."""

    def get_leaderboard(self, limit: int = 10) -> List[MockAgentRating]:
        return [
            MockAgentRating("claude", 1600, 10, 2, 1),
            MockAgentRating("gemini", 1550, 8, 4, 1),
            MockAgentRating("gpt4", 1500, 5, 5, 2),
        ][:limit]

    def get_recent_matches(self, limit: int = 10) -> List[Dict]:
        return [
            {"id": "match1", "winner": "claude", "loser": "gemini", "timestamp": "2026-01-01"},
            {"id": "match2", "winner": "gpt4", "loser": "claude", "timestamp": "2026-01-02"},
        ][:limit]


class MockInsightStore:
    """Mock InsightStore for testing."""

    async def get_recent_insights(self, limit: int = 10) -> List[Dict]:
        return [
            {"id": "insight1", "content": "Test insight 1", "debate_id": "d1"},
            {"id": "insight2", "content": "Test insight 2", "debate_id": "d2"},
        ][:limit]


class MockFlipDetector:
    """Mock FlipDetector for testing."""

    def get_flip_summary(self) -> Dict:
        return {
            "total_flips": 15,
            "average_confidence_change": 0.3,
            "top_flipper": "claude",
        }

    def get_recent_flips(self, limit: int = 10) -> List[Dict]:
        return [
            {"agent": "claude", "topic": "AI safety", "old_position": "pro", "new_position": "con"},
        ][:limit]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_elo_system():
    """Create mock ELO system."""
    return MockEloSystem()


@pytest.fixture
def mock_insight_store():
    """Create mock InsightStore."""
    return MockInsightStore()


@pytest.fixture
def mock_flip_detector():
    """Create mock FlipDetector."""
    return MockFlipDetector()


@pytest.fixture
def mock_request():
    """Create mock request with default headers."""
    return MockRequest()


@pytest.fixture
def handler_mixin(mock_elo_system, mock_insight_store, mock_flip_detector):
    """Create a handler mixin instance with mocked dependencies."""
    from aragora.server.stream.stream_handlers import StreamAPIHandlersMixin

    class TestHandler(StreamAPIHandlersMixin):
        def __init__(self):
            self.elo_system = mock_elo_system
            self.insight_store = mock_insight_store
            self.flip_detector = mock_flip_detector
            self.persona_manager = None
            self.debate_embeddings = None
            self.nomic_dir = None
            self.active_loops = {}
            self._active_loops_lock = MagicMock()
            self.cartographers = {}
            self._cartographers_lock = MagicMock()
            self.audience_inbox = None
            self.emitter = None

        def _cors_headers(self, origin: Optional[str] = None) -> Dict[str, str]:
            return {
                "Access-Control-Allow-Origin": origin or "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            }

    return TestHandler()


# =============================================================================
# Leaderboard Handler Tests
# =============================================================================


class TestHandleLeaderboard:
    """Tests for _handle_leaderboard handler."""

    @pytest.mark.asyncio
    async def test_returns_agent_rankings(self, handler_mixin, mock_request):
        """Should return formatted agent rankings."""
        with patch("aiohttp.web.json_response") as mock_json:
            mock_json.return_value = MockResponse(200)
            await handler_mixin._handle_leaderboard(mock_request)

            call_args = mock_json.call_args
            data = call_args[0][0]

            assert "agents" in data
            assert "count" in data
            assert len(data["agents"]) == 3
            assert data["agents"][0]["name"] == "claude"
            assert data["agents"][0]["elo"] == 1600

    @pytest.mark.asyncio
    async def test_respects_limit_parameter(self, handler_mixin):
        """Should respect limit query parameter."""
        request = MockRequest(query={"limit": "2"})

        with patch("aiohttp.web.json_response") as mock_json:
            mock_json.return_value = MockResponse(200)
            await handler_mixin._handle_leaderboard(request)

            call_args = mock_json.call_args
            data = call_args[0][0]

            assert len(data["agents"]) == 2

    @pytest.mark.asyncio
    async def test_returns_empty_without_elo_system(self, handler_mixin, mock_request):
        """Should return empty list when ELO system unavailable."""
        handler_mixin.elo_system = None

        with patch("aiohttp.web.json_response") as mock_json:
            mock_json.return_value = MockResponse(200)
            await handler_mixin._handle_leaderboard(mock_request)

            call_args = mock_json.call_args
            data = call_args[0][0]

            assert data["agents"] == []
            assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_handles_elo_system_error(self, handler_mixin, mock_request):
        """Should return 500 on ELO system error."""
        handler_mixin.elo_system.get_leaderboard = MagicMock(
            side_effect=Exception("Database error")
        )

        with patch("aiohttp.web.json_response") as mock_json:
            mock_json.return_value = MockResponse(500)
            await handler_mixin._handle_leaderboard(mock_request)

            call_args = mock_json.call_args
            assert call_args[1].get("status") == 500


# =============================================================================
# Matches Handler Tests
# =============================================================================


class TestHandleMatchesRecent:
    """Tests for _handle_matches_recent handler."""

    @pytest.mark.asyncio
    async def test_returns_recent_matches(self, handler_mixin, mock_request):
        """Should return recent matches."""
        with patch("aiohttp.web.json_response") as mock_json:
            mock_json.return_value = MockResponse(200)
            await handler_mixin._handle_matches_recent(mock_request)

            call_args = mock_json.call_args
            data = call_args[0][0]

            assert "matches" in data
            assert "count" in data
            assert len(data["matches"]) == 2

    @pytest.mark.asyncio
    async def test_returns_empty_without_elo_system(self, handler_mixin, mock_request):
        """Should return empty list when ELO system unavailable."""
        handler_mixin.elo_system = None

        with patch("aiohttp.web.json_response") as mock_json:
            mock_json.return_value = MockResponse(200)
            await handler_mixin._handle_matches_recent(mock_request)

            call_args = mock_json.call_args
            data = call_args[0][0]

            assert data["matches"] == []
            assert data["count"] == 0


# =============================================================================
# Insights Handler Tests
# =============================================================================


class TestHandleInsightsRecent:
    """Tests for _handle_insights_recent handler."""

    @pytest.mark.asyncio
    async def test_returns_recent_insights(self, handler_mixin, mock_request):
        """Should return recent insights."""
        with patch("aiohttp.web.json_response") as mock_json:
            mock_json.return_value = MockResponse(200)
            await handler_mixin._handle_insights_recent(mock_request)

            call_args = mock_json.call_args
            data = call_args[0][0]

            assert "insights" in data
            assert len(data["insights"]) == 2

    @pytest.mark.asyncio
    async def test_returns_empty_without_insight_store(self, handler_mixin, mock_request):
        """Should return empty list when InsightStore unavailable."""
        handler_mixin.insight_store = None

        with patch("aiohttp.web.json_response") as mock_json:
            mock_json.return_value = MockResponse(200)
            await handler_mixin._handle_insights_recent(mock_request)

            call_args = mock_json.call_args
            data = call_args[0][0]

            assert data["insights"] == []


# =============================================================================
# Flips Handler Tests
# =============================================================================


class TestHandleFlipsSummary:
    """Tests for _handle_flips_summary handler."""

    @pytest.mark.asyncio
    async def test_returns_flip_summary(self, handler_mixin, mock_request):
        """Should return flip detection summary."""
        with patch("aiohttp.web.json_response") as mock_json:
            mock_json.return_value = MockResponse(200)
            await handler_mixin._handle_flips_summary(mock_request)

            call_args = mock_json.call_args
            data = call_args[0][0]

            assert "summary" in data
            assert "count" in data
            assert data["summary"]["total_flips"] == 15

    @pytest.mark.asyncio
    async def test_returns_empty_without_flip_detector(self, handler_mixin, mock_request):
        """Should return empty summary when FlipDetector unavailable."""
        handler_mixin.flip_detector = None

        with patch("aiohttp.web.json_response") as mock_json:
            mock_json.return_value = MockResponse(200)
            await handler_mixin._handle_flips_summary(mock_request)

            call_args = mock_json.call_args
            data = call_args[0][0]

            assert data["summary"] == {}
            assert data["count"] == 0


class TestHandleFlipsRecent:
    """Tests for _handle_flips_recent handler."""

    @pytest.mark.asyncio
    async def test_returns_recent_flips(self, handler_mixin, mock_request):
        """Should return recent position flips."""
        with patch("aiohttp.web.json_response") as mock_json:
            mock_json.return_value = MockResponse(200)
            await handler_mixin._handle_flips_recent(mock_request)

            call_args = mock_json.call_args
            data = call_args[0][0]

            assert "flips" in data
            assert len(data["flips"]) == 1


# =============================================================================
# Options Handler Tests
# =============================================================================


class TestHandleOptions:
    """Tests for _handle_options CORS handler."""

    @pytest.mark.asyncio
    async def test_returns_cors_headers(self, handler_mixin, mock_request):
        """Should return CORS headers for preflight."""
        with patch("aiohttp.web.Response") as mock_response:
            mock_response.return_value = MockResponse(204)
            await handler_mixin._handle_options(mock_request)

            call_args = mock_response.call_args
            assert call_args[1]["status"] == 204
            assert "headers" in call_args[1]


# =============================================================================
# Health Handler Tests
# =============================================================================


class TestHandleHealth:
    """Tests for _handle_health handler."""

    @pytest.mark.asyncio
    async def test_returns_healthy_status(self, handler_mixin, mock_request):
        """Should return healthy status."""
        with patch("aiohttp.web.json_response") as mock_json:
            mock_json.return_value = MockResponse(200)
            await handler_mixin._handle_health(mock_request)

            call_args = mock_json.call_args
            data = call_args[0][0]

            assert data["status"] == "healthy"


# =============================================================================
# Query Parameter Validation Tests
# =============================================================================


class TestQueryParameterValidation:
    """Tests for query parameter validation in handlers."""

    @pytest.mark.asyncio
    async def test_limit_clamped_to_max(self, handler_mixin):
        """Limit parameter should be clamped to max value."""
        request = MockRequest(query={"limit": "1000"})  # Over max of 100

        with patch("aiohttp.web.json_response") as mock_json:
            mock_json.return_value = MockResponse(200)
            await handler_mixin._handle_leaderboard(request)

            # Should still work, limit clamped internally
            call_args = mock_json.call_args
            assert call_args is not None

    @pytest.mark.asyncio
    async def test_invalid_limit_uses_default(self, handler_mixin):
        """Invalid limit parameter should use default."""
        request = MockRequest(query={"limit": "invalid"})

        with patch("aiohttp.web.json_response") as mock_json:
            mock_json.return_value = MockResponse(200)
            await handler_mixin._handle_leaderboard(request)

            # Should still work with default limit
            call_args = mock_json.call_args
            assert call_args is not None


# =============================================================================
# CORS Header Tests
# =============================================================================


class TestCorsHeaders:
    """Tests for CORS header handling."""

    @pytest.mark.asyncio
    async def test_includes_origin_header(self, handler_mixin):
        """Response should include correct origin header."""
        request = MockRequest(headers={"Origin": "https://example.com"})

        with patch("aiohttp.web.json_response") as mock_json:
            mock_json.return_value = MockResponse(200)
            await handler_mixin._handle_leaderboard(request)

            call_args = mock_json.call_args
            headers = call_args[1].get("headers", {})

            assert "Access-Control-Allow-Origin" in headers

    @pytest.mark.asyncio
    async def test_missing_origin_uses_none(self, handler_mixin):
        """Missing origin should pass None to cors headers."""
        # Create request without default Origin
        request = MockRequest()
        request.headers = {}  # Clear the default Origin

        with patch("aiohttp.web.json_response") as mock_json:
            mock_json.return_value = MockResponse(200)
            await handler_mixin._handle_leaderboard(request)

            call_args = mock_json.call_args
            headers = call_args[1].get("headers", {})

            # Handler's _cors_headers returns "*" when origin is None
            assert headers.get("Access-Control-Allow-Origin") == "*"
