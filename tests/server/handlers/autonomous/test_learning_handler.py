"""Tests for autonomous learning handler."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from aragora.server.handlers.autonomous import learning


# =============================================================================
# Mock Classes
# =============================================================================


class MockEloUpdater:
    """Mock ELO updater."""

    def __init__(self):
        self._ratings = {}

    def get_all_ratings(self):
        return self._ratings

    def get_rating(self, agent_id):
        return self._ratings.get(agent_id, 1000)


class MockPatternExtractor:
    """Mock pattern extractor."""

    def __init__(self):
        self._patterns = []

    def get_patterns(self, pattern_type=None):
        if pattern_type:
            return [p for p in self._patterns if p.pattern_type == pattern_type]
        return self._patterns


class MockPattern:
    """Mock pattern for testing."""

    def __init__(
        self,
        id: str = "pattern-001",
        pattern_type: str = "consensus",
        description: str = "Test pattern",
        confidence: float = 0.8,
        evidence_count: int = 5,
        agents_involved: list = None,
        topics: list = None,
    ):
        self.id = id
        self.pattern_type = pattern_type
        self.description = description
        self.confidence = confidence
        self.evidence_count = evidence_count
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.agents_involved = agents_involved or []
        self.topics = topics or []


class MockCalibration:
    """Mock calibration data."""

    def __init__(
        self,
        agent_id: str = "agent-001",
        elo_rating: float = 1000,
        confidence_accuracy: float = 0.8,
        topic_strengths: list = None,
        topic_weaknesses: list = None,
        total_debates: int = 10,
        win_rate: float = 0.5,
    ):
        self.agent_id = agent_id
        self.elo_rating = elo_rating
        self.confidence_accuracy = confidence_accuracy
        self.topic_strengths = topic_strengths or []
        self.topic_weaknesses = topic_weaknesses or []
        self.last_updated = datetime.now()
        self.total_debates = total_debates
        self.win_rate = win_rate


class MockLearningEvent:
    """Mock learning event."""

    def __init__(self, id: str = "event-001", event_type_value: str = "debate_outcome"):
        self.id = id
        self.event_type = MagicMock(value=event_type_value)
        self.applied = True


class MockContinuousLearner:
    """Mock ContinuousLearner for testing."""

    def __init__(self):
        self.elo_updater = MockEloUpdater()
        self.pattern_extractor = MockPatternExtractor()
        self._calibrations = {}

    def get_calibration(self, agent_id):
        return self._calibrations.get(agent_id)

    def get_all_calibrations(self):
        return self._calibrations

    async def on_debate_completed(self, **kwargs):
        return MockLearningEvent()

    async def on_user_feedback(self, **kwargs):
        return MockLearningEvent(event_type_value="user_feedback")

    async def run_periodic_learning(self):
        return {"events_processed": 5, "patterns_extracted": 2}


class MockAuthContext:
    """Mock authorization context."""

    def __init__(self, user_id="test-user"):
        self.user_id = user_id


class MockPermissionDecision:
    """Mock permission decision."""

    def __init__(self, allowed=True, reason=None):
        self.allowed = allowed
        self.reason = reason or ""


class MockPermissionChecker:
    """Mock permission checker."""

    def check_permission(self, ctx, permission):
        return MockPermissionDecision(allowed=True)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_learner():
    """Create mock continuous learner."""
    return MockContinuousLearner()


@pytest.fixture
def mock_auth_context():
    """Create mock auth context."""
    return MockAuthContext()


@pytest.fixture
def mock_permission_checker():
    """Create mock permission checker."""
    return MockPermissionChecker()


# =============================================================================
# Test LearningHandler.get_agent_ratings
# =============================================================================


class TestLearningHandlerGetRatings:
    """Tests for GET /api/autonomous/learning/ratings endpoint."""

    @pytest.mark.asyncio
    async def test_get_ratings_empty(
        self, mock_learner, mock_auth_context, mock_permission_checker
    ):
        """Should return empty ratings dict."""
        with (
            patch.object(learning, "get_continuous_learner", return_value=mock_learner),
            patch.object(
                learning,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                learning,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            response = await learning.LearningHandler.get_agent_ratings(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["ratings"] == {}
            assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_get_ratings_with_data(
        self, mock_learner, mock_auth_context, mock_permission_checker
    ):
        """Should return agent ratings."""
        mock_learner.elo_updater._ratings = {"agent-1": 1200, "agent-2": 1100}

        with (
            patch.object(learning, "get_continuous_learner", return_value=mock_learner),
            patch.object(
                learning,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                learning,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            response = await learning.LearningHandler.get_agent_ratings(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["count"] == 2
            assert body["ratings"]["agent-1"] == 1200


# =============================================================================
# Test LearningHandler.get_agent_calibration
# =============================================================================


class TestLearningHandlerGetCalibration:
    """Tests for GET /api/autonomous/learning/calibration/{agent_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_calibration_not_found(
        self, mock_learner, mock_auth_context, mock_permission_checker
    ):
        """Should return null calibration when not found."""
        with (
            patch.object(learning, "get_continuous_learner", return_value=mock_learner),
            patch.object(
                learning,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                learning,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "unknown-agent"

            response = await learning.LearningHandler.get_agent_calibration(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["calibration"] is None

    @pytest.mark.asyncio
    async def test_get_calibration_success(
        self, mock_learner, mock_auth_context, mock_permission_checker
    ):
        """Should return calibration data."""
        mock_learner._calibrations = {"agent-1": MockCalibration(agent_id="agent-1")}

        with (
            patch.object(learning, "get_continuous_learner", return_value=mock_learner),
            patch.object(
                learning,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                learning,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "agent-1"

            response = await learning.LearningHandler.get_agent_calibration(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["calibration"]["agent_id"] == "agent-1"


# =============================================================================
# Test LearningHandler.record_debate_outcome
# =============================================================================


class TestLearningHandlerRecordDebate:
    """Tests for POST /api/autonomous/learning/debate endpoint."""

    @pytest.mark.asyncio
    async def test_record_debate_success(
        self, mock_learner, mock_auth_context, mock_permission_checker
    ):
        """Should record debate outcome."""
        with (
            patch.object(learning, "get_continuous_learner", return_value=mock_learner),
            patch.object(
                learning,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                learning,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.json = AsyncMock(
                return_value={
                    "debate_id": "debate-1",
                    "agents": ["agent-1", "agent-2"],
                    "winner": "agent-1",
                    "consensus_reached": True,
                }
            )

            response = await learning.LearningHandler.record_debate_outcome(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["event"]["applied"] is True

    @pytest.mark.asyncio
    async def test_record_debate_missing_fields(
        self, mock_learner, mock_auth_context, mock_permission_checker
    ):
        """Should return 400 for missing required fields."""
        with (
            patch.object(learning, "get_continuous_learner", return_value=mock_learner),
            patch.object(
                learning,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                learning,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.json = AsyncMock(return_value={})

            response = await learning.LearningHandler.record_debate_outcome(request)

            assert response.status == 400


# =============================================================================
# Test LearningHandler.record_user_feedback
# =============================================================================


class TestLearningHandlerRecordFeedback:
    """Tests for POST /api/autonomous/learning/feedback endpoint."""

    @pytest.mark.asyncio
    async def test_record_feedback_success(
        self, mock_learner, mock_auth_context, mock_permission_checker
    ):
        """Should record user feedback."""
        with (
            patch.object(learning, "get_continuous_learner", return_value=mock_learner),
            patch.object(
                learning,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                learning,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.json = AsyncMock(
                return_value={
                    "debate_id": "debate-1",
                    "agent_id": "agent-1",
                    "feedback_type": "helpful",
                    "score": 0.8,
                }
            )

            response = await learning.LearningHandler.record_user_feedback(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True

    @pytest.mark.asyncio
    async def test_record_feedback_missing_fields(
        self, mock_learner, mock_auth_context, mock_permission_checker
    ):
        """Should return 400 for missing required fields."""
        with (
            patch.object(learning, "get_continuous_learner", return_value=mock_learner),
            patch.object(
                learning,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                learning,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.json = AsyncMock(return_value={"debate_id": "debate-1"})

            response = await learning.LearningHandler.record_user_feedback(request)

            assert response.status == 400


# =============================================================================
# Test LearningHandler.get_patterns
# =============================================================================


class TestLearningHandlerGetPatterns:
    """Tests for GET /api/autonomous/learning/patterns endpoint."""

    @pytest.mark.asyncio
    async def test_get_patterns_empty(
        self, mock_learner, mock_auth_context, mock_permission_checker
    ):
        """Should return empty patterns list."""
        with (
            patch.object(learning, "get_continuous_learner", return_value=mock_learner),
            patch.object(
                learning,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                learning,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.query.get.return_value = None

            response = await learning.LearningHandler.get_patterns(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["patterns"] == []

    @pytest.mark.asyncio
    async def test_get_patterns_with_data(
        self, mock_learner, mock_auth_context, mock_permission_checker
    ):
        """Should return patterns list."""
        mock_learner.pattern_extractor._patterns = [
            MockPattern(id="p1"),
            MockPattern(id="p2"),
        ]

        with (
            patch.object(learning, "get_continuous_learner", return_value=mock_learner),
            patch.object(
                learning,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                learning,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.query.get.return_value = None

            response = await learning.LearningHandler.get_patterns(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["count"] == 2


# =============================================================================
# Test Route Registration
# =============================================================================


class TestLearningHandlerRoutes:
    """Tests for route registration."""

    def test_register_routes(self):
        """Should register all learning routes."""
        app = web.Application()
        learning.LearningHandler.register_routes(app)

        routes = [r.resource.canonical for r in app.router.routes()]
        assert "/api/v1/autonomous/learning/ratings" in routes
        assert "/api/v1/autonomous/learning/calibration/{agent_id}" in routes
        assert "/api/v1/autonomous/learning/calibrations" in routes
        assert "/api/v1/autonomous/learning/debate" in routes
        assert "/api/v1/autonomous/learning/feedback" in routes
        assert "/api/v1/autonomous/learning/patterns" in routes
        assert "/api/v1/autonomous/learning/run" in routes


# =============================================================================
# Test Global Functions
# =============================================================================


class TestContinuousLearnerSingleton:
    """Tests for continuous learner singleton functions."""

    def test_get_continuous_learner_creates_singleton(self):
        """get_continuous_learner should return same instance."""
        learning._continuous_learner = None

        learner1 = learning.get_continuous_learner()
        learner2 = learning.get_continuous_learner()

        assert learner1 is learner2

        # Clean up
        learning._continuous_learner = None

    def test_set_continuous_learner(self):
        """set_continuous_learner should update the global instance."""
        mock = MockContinuousLearner()
        learning.set_continuous_learner(mock)

        assert learning.get_continuous_learner() is mock

        # Clean up
        learning._continuous_learner = None
