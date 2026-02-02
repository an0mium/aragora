"""
Tests for AutonomousLearningHandler - Autonomous learning HTTP endpoints.

Tests cover:
- Training session management (list, create, get, stop)
- Learning metrics (list, get by type, aggregation)
- Pattern detection (list, get, validate)
- Knowledge extraction (list, get, extract)
- Feedback submission
- Recommendations generation
- Performance statistics
- Calibration
- Circuit breaker behavior
- Rate limiting
- Error handling

Stability: STABLE
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.autonomous_learning import (
    AutonomousLearningHandler,
    TrainingSession,
    SessionStatus,
    LearningMode,
    LearningMetric,
    MetricType,
    DetectedPattern,
    PatternType,
    ExtractedKnowledge,
    LearningFeedback,
    FeedbackType,
    LearningRecommendation,
    PerformanceStats,
    create_autonomous_learning_handler,
    MAX_ACTIVE_SESSIONS,
    MIN_PATTERN_CONFIDENCE,
)


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


class MockHTTPHandler:
    """Mock HTTP request handler for testing."""

    def __init__(
        self,
        method: str = "GET",
        body: dict | None = None,
        user_id: str | None = "test-user",
    ):
        self.command = method
        self._body = json.dumps(body or {}).encode() if body else b""
        self.headers = {
            "Content-Length": str(len(self._body)) if self._body else "0",
            "Content-Type": "application/json" if body else "",
            "Authorization": f"Bearer test-token-{user_id}" if user_id else "",
        }
        self.rfile = io.BytesIO(self._body)
        self._user_id = user_id

    @property
    def user_id(self) -> str:
        return self._user_id or "anonymous"


class MockUserAuthContext:
    """Mock user authentication context."""

    def __init__(self, user_id: str = "test-user", is_admin: bool = False):
        self.user_id = user_id
        self.is_authenticated = True
        self.is_admin = is_admin
        self.roles = ["admin"] if is_admin else ["user"]
        self.permissions = ["learning:read", "learning:write"]


@pytest.fixture
def server_context():
    """Create server context."""
    return {}


@pytest.fixture
def handler(server_context):
    """Create AutonomousLearningHandler instance."""
    return AutonomousLearningHandler(server_context)


@pytest.fixture
def sample_session():
    """Create a sample training session."""
    now = datetime.now(timezone.utc)
    return TrainingSession(
        id="session_001",
        name="Test Session",
        mode=LearningMode.SUPERVISED,
        status=SessionStatus.RUNNING,
        created_at=now,
        started_at=now,
        owner_id="test-user",
        epochs_completed=50,
        total_epochs=100,
        current_loss=0.15,
        best_loss=0.12,
    )


@pytest.fixture
def sample_pattern():
    """Create a sample detected pattern."""
    now = datetime.now(timezone.utc)
    return DetectedPattern(
        id="pattern_001",
        pattern_type=PatternType.CONSENSUS,
        confidence=0.85,
        description="Strong consensus pattern detected",
        detected_at=now,
        source_debates=["debate_1", "debate_2"],
        agents_involved=["claude", "gpt-4"],
        frequency=5,
    )


@pytest.fixture
def sample_knowledge():
    """Create a sample extracted knowledge item."""
    now = datetime.now(timezone.utc)
    return ExtractedKnowledge(
        id="knowledge_001",
        title="Test Knowledge",
        content="This is extracted knowledge content.",
        source_type="debate_analysis",
        source_debates=["debate_1"],
        confidence=0.9,
        extracted_at=now,
        topics=["testing", "ai"],
    )


# ===========================================================================
# TrainingSession Tests
# ===========================================================================


class TestTrainingSession:
    """Tests for TrainingSession dataclass."""

    def test_training_session_creation(self):
        """Should create training session with all fields."""
        now = datetime.now(timezone.utc)
        session = TrainingSession(
            id="session_001",
            name="My Training",
            mode=LearningMode.REINFORCEMENT,
            status=SessionStatus.RUNNING,
            created_at=now,
            owner_id="user-001",
        )

        assert session.id == "session_001"
        assert session.name == "My Training"
        assert session.mode == LearningMode.REINFORCEMENT
        assert session.status == SessionStatus.RUNNING

    def test_training_session_progress_percent(self):
        """Should calculate progress percentage correctly."""
        session = TrainingSession(
            id="session_001",
            name="Test",
            mode=LearningMode.SUPERVISED,
            status=SessionStatus.RUNNING,
            created_at=datetime.now(timezone.utc),
            owner_id="user-001",
            epochs_completed=25,
            total_epochs=100,
        )

        assert session.progress_percent == 25.0

    def test_training_session_progress_zero_epochs(self):
        """Should handle zero total epochs."""
        session = TrainingSession(
            id="session_001",
            name="Test",
            mode=LearningMode.SUPERVISED,
            status=SessionStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            owner_id="user-001",
            epochs_completed=0,
            total_epochs=0,
        )

        assert session.progress_percent == 0.0

    def test_training_session_to_dict(self):
        """Should convert session to dictionary."""
        now = datetime.now(timezone.utc)
        session = TrainingSession(
            id="session_001",
            name="Test",
            mode=LearningMode.SUPERVISED,
            status=SessionStatus.COMPLETED,
            created_at=now,
            owner_id="user-001",
            metrics={"accuracy": 0.95},
        )

        result = session.to_dict()

        assert result["id"] == "session_001"
        assert result["status"] == "completed"
        assert result["mode"] == "supervised"
        assert result["metrics"] == {"accuracy": 0.95}


class TestLearningMetric:
    """Tests for LearningMetric dataclass."""

    def test_learning_metric_creation(self):
        """Should create learning metric."""
        now = datetime.now(timezone.utc)
        metric = LearningMetric(
            metric_type=MetricType.ACCURACY,
            value=0.92,
            timestamp=now,
            session_id="session_001",
        )

        assert metric.metric_type == MetricType.ACCURACY
        assert metric.value == 0.92

    def test_learning_metric_to_dict(self):
        """Should convert metric to dictionary."""
        now = datetime.now(timezone.utc)
        metric = LearningMetric(
            metric_type=MetricType.LOSS,
            value=0.1234567,
            timestamp=now,
        )

        result = metric.to_dict()

        assert result["metric_type"] == "loss"
        assert result["value"] == 0.1235  # Rounded to 4 decimal places


class TestDetectedPattern:
    """Tests for DetectedPattern dataclass."""

    def test_detected_pattern_creation(self):
        """Should create detected pattern."""
        now = datetime.now(timezone.utc)
        pattern = DetectedPattern(
            id="pattern_001",
            pattern_type=PatternType.CONSENSUS,
            confidence=0.85,
            description="Test pattern",
            detected_at=now,
        )

        assert pattern.id == "pattern_001"
        assert pattern.confidence == 0.85
        assert pattern.is_validated is False

    def test_detected_pattern_to_dict(self):
        """Should convert pattern to dictionary."""
        now = datetime.now(timezone.utc)
        pattern = DetectedPattern(
            id="pattern_001",
            pattern_type=PatternType.AGENT_PREFERENCE,
            confidence=0.75,
            description="Agent preference pattern",
            detected_at=now,
            agents_involved=["claude"],
            is_validated=True,
            validated_by="admin",
            validated_at=now,
        )

        result = pattern.to_dict()

        assert result["pattern_type"] == "agent_preference"
        assert result["is_validated"] is True
        assert result["validated_by"] == "admin"


class TestExtractedKnowledge:
    """Tests for ExtractedKnowledge dataclass."""

    def test_extracted_knowledge_creation(self):
        """Should create extracted knowledge."""
        now = datetime.now(timezone.utc)
        knowledge = ExtractedKnowledge(
            id="knowledge_001",
            title="Test Knowledge",
            content="Content here",
            source_type="debate",
            source_debates=["debate_1"],
            confidence=0.9,
            extracted_at=now,
        )

        assert knowledge.id == "knowledge_001"
        assert knowledge.confidence == 0.9

    def test_extracted_knowledge_to_dict(self):
        """Should convert knowledge to dictionary."""
        now = datetime.now(timezone.utc)
        knowledge = ExtractedKnowledge(
            id="knowledge_001",
            title="Test",
            content="Content",
            source_type="analysis",
            source_debates=["d1", "d2"],
            confidence=0.8,
            extracted_at=now,
            topics=["ai", "ml"],
        )

        result = knowledge.to_dict()

        assert result["source_type"] == "analysis"
        assert result["topics"] == ["ai", "ml"]


class TestPerformanceStats:
    """Tests for PerformanceStats dataclass."""

    def test_performance_stats_success_rate(self):
        """Should calculate success rate correctly."""
        stats = PerformanceStats(
            total_sessions=10,
            successful_sessions=8,
            failed_sessions=2,
            average_accuracy=0.9,
            average_loss=0.1,
            total_epochs_trained=1000,
            total_training_time_hours=10.5,
            patterns_detected=50,
            knowledge_items_extracted=20,
            feedback_received=100,
            last_updated=datetime.now(timezone.utc),
        )

        result = stats.to_dict()

        assert result["success_rate"] == 80.0

    def test_performance_stats_zero_sessions(self):
        """Should handle zero sessions."""
        stats = PerformanceStats(
            total_sessions=0,
            successful_sessions=0,
            failed_sessions=0,
            average_accuracy=0.0,
            average_loss=0.0,
            total_epochs_trained=0,
            total_training_time_hours=0.0,
            patterns_detected=0,
            knowledge_items_extracted=0,
            feedback_received=0,
            last_updated=datetime.now(timezone.utc),
        )

        result = stats.to_dict()

        assert result["success_rate"] == 0.0


# ===========================================================================
# AutonomousLearningHandler Tests - Sessions
# ===========================================================================


class TestLearningHandlerListSessions:
    """Tests for AutonomousLearningHandler.handle (list sessions)."""

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, handler):
        """Should return empty list when no sessions."""
        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/sessions", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert data["sessions"] == []
        assert data["pagination"]["total"] == 0

    @pytest.mark.asyncio
    async def test_list_sessions_with_results(self, handler, sample_session):
        """Should return sessions when available."""
        handler._sessions[sample_session.id] = sample_session

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/sessions", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["id"] == "session_001"

    @pytest.mark.asyncio
    async def test_list_sessions_filter_by_status(self, handler):
        """Should filter sessions by status."""
        now = datetime.now(timezone.utc)
        handler._sessions["s1"] = TrainingSession(
            id="s1",
            name="S1",
            mode=LearningMode.SUPERVISED,
            status=SessionStatus.RUNNING,
            created_at=now,
            owner_id="user",
        )
        handler._sessions["s2"] = TrainingSession(
            id="s2",
            name="S2",
            mode=LearningMode.SUPERVISED,
            status=SessionStatus.COMPLETED,
            created_at=now,
            owner_id="user",
        )

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle(
                "/api/v2/learning/sessions",
                {"status": "running"},
                mock_request,
            )

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["id"] == "s1"

    @pytest.mark.asyncio
    async def test_list_sessions_filter_by_mode(self, handler):
        """Should filter sessions by mode."""
        now = datetime.now(timezone.utc)
        handler._sessions["s1"] = TrainingSession(
            id="s1",
            name="S1",
            mode=LearningMode.SUPERVISED,
            status=SessionStatus.RUNNING,
            created_at=now,
            owner_id="user",
        )
        handler._sessions["s2"] = TrainingSession(
            id="s2",
            name="S2",
            mode=LearningMode.REINFORCEMENT,
            status=SessionStatus.RUNNING,
            created_at=now,
            owner_id="user",
        )

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle(
                "/api/v2/learning/sessions",
                {"mode": "reinforcement"},
                mock_request,
            )

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["id"] == "s2"

    @pytest.mark.asyncio
    async def test_list_sessions_invalid_status(self, handler):
        """Should reject invalid status filter."""
        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle(
                "/api/v2/learning/sessions",
                {"status": "invalid_status"},
                mock_request,
            )

        assert result is not None
        assert result.status == 400


class TestLearningHandlerCreateSession:
    """Tests for AutonomousLearningHandler.handle_post (create session)."""

    @pytest.mark.asyncio
    async def test_create_session_success(self, handler):
        """Should create session successfully."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={"name": "New Training", "mode": "supervised"},
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/learning/sessions", {}, mock_request)

        assert result is not None
        assert result.status == 201
        data = json.loads(result.body)
        assert "session" in data
        assert data["session"]["name"] == "New Training"
        assert data["session"]["status"] == "running"

    @pytest.mark.asyncio
    async def test_create_session_missing_name(self, handler):
        """Should reject session without name."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={"mode": "supervised"},
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/learning/sessions", {}, mock_request)

        assert result is not None
        assert result.status == 400

    @pytest.mark.asyncio
    async def test_create_session_invalid_mode(self, handler):
        """Should reject invalid learning mode."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={"name": "Test", "mode": "invalid_mode"},
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/learning/sessions", {}, mock_request)

        assert result is not None
        assert result.status == 400

    @pytest.mark.asyncio
    async def test_create_session_max_active_reached(self, handler):
        """Should reject when max active sessions reached."""
        now = datetime.now(timezone.utc)
        for i in range(MAX_ACTIVE_SESSIONS):
            handler._sessions[f"s{i}"] = TrainingSession(
                id=f"s{i}",
                name=f"Session {i}",
                mode=LearningMode.SUPERVISED,
                status=SessionStatus.RUNNING,
                created_at=now,
                owner_id="user",
            )

        mock_request = MockHTTPHandler(
            method="POST",
            body={"name": "One More", "mode": "supervised"},
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/learning/sessions", {}, mock_request)

        assert result is not None
        assert result.status == 400
        assert "maximum" in json.loads(result.body).get("error", "").lower()


class TestLearningHandlerGetSession:
    """Tests for AutonomousLearningHandler.handle (get session)."""

    @pytest.mark.asyncio
    async def test_get_session_success(self, handler, sample_session):
        """Should return session details."""
        handler._sessions[sample_session.id] = sample_session

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle(
                f"/api/v2/learning/sessions/{sample_session.id}",
                {},
                mock_request,
            )

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert data["id"] == sample_session.id

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, handler):
        """Should return 404 for non-existent session."""
        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle(
                "/api/v2/learning/sessions/nonexistent",
                {},
                mock_request,
            )

        assert result is not None
        assert result.status == 404


class TestLearningHandlerStopSession:
    """Tests for AutonomousLearningHandler.handle_post (stop session)."""

    @pytest.mark.asyncio
    async def test_stop_session_success(self, handler, sample_session):
        """Should stop running session."""
        handler._sessions[sample_session.id] = sample_session

        mock_request = MockHTTPHandler(method="POST")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post(
                f"/api/v2/learning/sessions/{sample_session.id}/stop",
                {},
                mock_request,
            )

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert data["session"]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_session_not_found(self, handler):
        """Should return 404 for non-existent session."""
        mock_request = MockHTTPHandler(method="POST")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post(
                "/api/v2/learning/sessions/nonexistent/stop",
                {},
                mock_request,
            )

        assert result is not None
        assert result.status == 404

    @pytest.mark.asyncio
    async def test_stop_session_already_completed(self, handler):
        """Should reject stopping completed session."""
        now = datetime.now(timezone.utc)
        session = TrainingSession(
            id="s1",
            name="Test",
            mode=LearningMode.SUPERVISED,
            status=SessionStatus.COMPLETED,
            created_at=now,
            owner_id="user",
        )
        handler._sessions[session.id] = session

        mock_request = MockHTTPHandler(method="POST")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post(
                f"/api/v2/learning/sessions/{session.id}/stop",
                {},
                mock_request,
            )

        assert result is not None
        assert result.status == 400


# ===========================================================================
# AutonomousLearningHandler Tests - Metrics
# ===========================================================================


class TestLearningHandlerMetrics:
    """Tests for AutonomousLearningHandler metrics operations."""

    @pytest.mark.asyncio
    async def test_get_metrics_empty(self, handler):
        """Should return empty metrics list."""
        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/metrics", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert data["metrics"] == []

    @pytest.mark.asyncio
    async def test_get_metrics_with_results(self, handler):
        """Should return metrics."""
        now = datetime.now(timezone.utc)
        handler._metrics.append(
            LearningMetric(
                metric_type=MetricType.ACCURACY,
                value=0.92,
                timestamp=now,
            )
        )

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/metrics", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert len(data["metrics"]) == 1

    @pytest.mark.asyncio
    async def test_get_metric_by_type(self, handler):
        """Should return metrics of specific type."""
        now = datetime.now(timezone.utc)
        handler._metrics.append(
            LearningMetric(
                metric_type=MetricType.ACCURACY,
                value=0.92,
                timestamp=now,
            )
        )
        handler._metrics.append(
            LearningMetric(
                metric_type=MetricType.LOSS,
                value=0.1,
                timestamp=now,
            )
        )

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/metrics/accuracy", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert data["metric_type"] == "accuracy"
        assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_get_metric_invalid_type(self, handler):
        """Should reject invalid metric type."""
        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle(
                "/api/v2/learning/metrics/invalid_type",
                {},
                mock_request,
            )

        assert result is not None
        assert result.status == 400


# ===========================================================================
# AutonomousLearningHandler Tests - Patterns
# ===========================================================================


class TestLearningHandlerPatterns:
    """Tests for AutonomousLearningHandler pattern operations."""

    @pytest.mark.asyncio
    async def test_list_patterns_empty(self, handler):
        """Should return empty patterns list."""
        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/patterns", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert data["patterns"] == []

    @pytest.mark.asyncio
    async def test_list_patterns_with_results(self, handler, sample_pattern):
        """Should return patterns."""
        handler._patterns[sample_pattern.id] = sample_pattern

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/patterns", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert len(data["patterns"]) == 1

    @pytest.mark.asyncio
    async def test_list_patterns_filter_by_type(self, handler, sample_pattern):
        """Should filter patterns by type."""
        handler._patterns[sample_pattern.id] = sample_pattern

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle(
                "/api/v2/learning/patterns",
                {"pattern_type": "consensus"},
                mock_request,
            )

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert len(data["patterns"]) == 1

    @pytest.mark.asyncio
    async def test_list_patterns_validated_only(self, handler):
        """Should filter to validated patterns only."""
        now = datetime.now(timezone.utc)
        handler._patterns["p1"] = DetectedPattern(
            id="p1",
            pattern_type=PatternType.CONSENSUS,
            confidence=0.9,
            description="Pattern 1",
            detected_at=now,
            is_validated=True,
        )
        handler._patterns["p2"] = DetectedPattern(
            id="p2",
            pattern_type=PatternType.CONSENSUS,
            confidence=0.8,
            description="Pattern 2",
            detected_at=now,
            is_validated=False,
        )

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle(
                "/api/v2/learning/patterns",
                {"validated": "true"},
                mock_request,
            )

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert len(data["patterns"]) == 1
        assert data["patterns"][0]["id"] == "p1"

    @pytest.mark.asyncio
    async def test_get_pattern_success(self, handler, sample_pattern):
        """Should return pattern details."""
        handler._patterns[sample_pattern.id] = sample_pattern

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle(
                f"/api/v2/learning/patterns/{sample_pattern.id}",
                {},
                mock_request,
            )

        assert result is not None
        assert result.status == 200

    @pytest.mark.asyncio
    async def test_validate_pattern_success(self, handler, sample_pattern):
        """Should validate pattern successfully."""
        handler._patterns[sample_pattern.id] = sample_pattern

        mock_request = MockHTTPHandler(method="POST")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post(
                f"/api/v2/learning/patterns/{sample_pattern.id}/validate",
                {},
                mock_request,
            )

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert data["pattern"]["is_validated"] is True


# ===========================================================================
# AutonomousLearningHandler Tests - Knowledge
# ===========================================================================


class TestLearningHandlerKnowledge:
    """Tests for AutonomousLearningHandler knowledge operations."""

    @pytest.mark.asyncio
    async def test_list_knowledge_empty(self, handler):
        """Should return empty knowledge list."""
        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/knowledge", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert data["knowledge"] == []

    @pytest.mark.asyncio
    async def test_list_knowledge_with_results(self, handler, sample_knowledge):
        """Should return knowledge items."""
        handler._knowledge[sample_knowledge.id] = sample_knowledge

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/knowledge", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert len(data["knowledge"]) == 1

    @pytest.mark.asyncio
    async def test_extract_knowledge_success(self, handler):
        """Should extract knowledge successfully."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={"debate_ids": ["debate_1", "debate_2"], "title": "New Knowledge"},
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post(
                "/api/v2/learning/knowledge/extract",
                {},
                mock_request,
            )

        assert result is not None
        assert result.status == 201
        data = json.loads(result.body)
        assert "knowledge" in data

    @pytest.mark.asyncio
    async def test_extract_knowledge_missing_debates(self, handler):
        """Should reject extraction without debate_ids."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={"title": "Test"},
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post(
                "/api/v2/learning/knowledge/extract",
                {},
                mock_request,
            )

        assert result is not None
        assert result.status == 400


# ===========================================================================
# AutonomousLearningHandler Tests - Feedback
# ===========================================================================


class TestLearningHandlerFeedback:
    """Tests for AutonomousLearningHandler feedback operations."""

    @pytest.mark.asyncio
    async def test_submit_feedback_success(self, handler):
        """Should submit feedback successfully."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={
                "feedback_type": "positive",
                "target_type": "session",
                "target_id": "session_001",
                "comment": "Great results!",
            },
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/learning/feedback", {}, mock_request)

        assert result is not None
        assert result.status == 201
        data = json.loads(result.body)
        assert "feedback" in data
        assert data["feedback"]["feedback_type"] == "positive"

    @pytest.mark.asyncio
    async def test_submit_feedback_invalid_type(self, handler):
        """Should reject invalid feedback type."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={
                "feedback_type": "invalid",
                "target_type": "session",
                "target_id": "session_001",
            },
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/learning/feedback", {}, mock_request)

        assert result is not None
        assert result.status == 400

    @pytest.mark.asyncio
    async def test_submit_feedback_missing_target(self, handler):
        """Should reject feedback without target."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={"feedback_type": "positive"},
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/learning/feedback", {}, mock_request)

        assert result is not None
        assert result.status == 400


# ===========================================================================
# AutonomousLearningHandler Tests - Recommendations
# ===========================================================================


class TestLearningHandlerRecommendations:
    """Tests for AutonomousLearningHandler recommendations operations."""

    @pytest.mark.asyncio
    async def test_get_recommendations_empty_state(self, handler):
        """Should return recommendations for empty state."""
        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/recommendations", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert "recommendations" in data
        # Should recommend starting a session when none exist
        assert len(data["recommendations"]) >= 1

    @pytest.mark.asyncio
    async def test_get_recommendations_with_unvalidated_patterns(self, handler):
        """Should recommend validating patterns."""
        now = datetime.now(timezone.utc)
        handler._patterns["p1"] = DetectedPattern(
            id="p1",
            pattern_type=PatternType.CONSENSUS,
            confidence=0.85,
            description="High confidence pattern",
            detected_at=now,
            is_validated=False,
        )

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/recommendations", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        # Should include recommendation to validate patterns
        titles = [r["title"] for r in data["recommendations"]]
        assert any("pattern" in t.lower() for t in titles)


# ===========================================================================
# AutonomousLearningHandler Tests - Performance
# ===========================================================================


class TestLearningHandlerPerformance:
    """Tests for AutonomousLearningHandler performance operations."""

    @pytest.mark.asyncio
    async def test_get_performance_empty(self, handler):
        """Should return performance stats for empty state."""
        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/performance", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert "performance" in data
        assert data["performance"]["total_sessions"] == 0

    @pytest.mark.asyncio
    async def test_get_performance_with_sessions(self, handler, sample_session):
        """Should calculate performance from sessions."""
        handler._sessions[sample_session.id] = sample_session

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/performance", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert data["performance"]["total_sessions"] == 1


# ===========================================================================
# AutonomousLearningHandler Tests - Calibration
# ===========================================================================


class TestLearningHandlerCalibration:
    """Tests for AutonomousLearningHandler calibration operations."""

    @pytest.mark.asyncio
    async def test_calibrate_success(self, handler):
        """Should trigger calibration successfully."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={"agent_ids": ["claude", "gpt-4"]},
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/learning/calibrate", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert "calibration_id" in data
        assert "metric" in data

    @pytest.mark.asyncio
    async def test_calibrate_empty_agents(self, handler):
        """Should calibrate with empty agents list."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={},
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/learning/calibrate", {}, mock_request)

        assert result is not None
        assert result.status == 200


# ===========================================================================
# AutonomousLearningHandler Tests - Circuit Breaker
# ===========================================================================


class TestLearningHandlerCircuitBreaker:
    """Tests for circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_success(self, handler):
        """Should record success and keep circuit closed."""
        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/sessions", {}, mock_request)

        assert result is not None
        assert result.status == 200

        cb = handler._get_circuit_breaker()
        assert cb.can_proceed() is True


# ===========================================================================
# Factory Function Tests
# ===========================================================================


class TestCreateAutonomousLearningHandler:
    """Tests for handler factory function."""

    def test_create_handler(self):
        """Should create handler instance."""
        ctx = {}
        handler = create_autonomous_learning_handler(ctx)

        assert isinstance(handler, AutonomousLearningHandler)

    def test_handler_has_routes(self):
        """Handler should define routes."""
        assert hasattr(AutonomousLearningHandler, "ROUTES")
        assert len(AutonomousLearningHandler.ROUTES) > 0

    def test_handler_can_handle(self):
        """Handler should respond to can_handle."""
        handler = AutonomousLearningHandler({})

        assert handler.can_handle("/api/v2/learning/sessions", "GET") is True
        assert handler.can_handle("/api/v2/learning/metrics", "GET") is True
        assert handler.can_handle("/api/v2/other/path", "GET") is False


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestLearningHandlerErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_path_returns_none(self, handler):
        """Should return None for unhandled paths."""
        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/unknown", {}, mock_request)

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_session_path(self, handler):
        """Should return error for invalid session path."""
        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/sessions/", {}, mock_request)

        # Empty session_id results in 404 (session "" not found)
        # This is acceptable behavior - the session with empty ID doesn't exist
        assert result is None or result.status in (400, 404)


# ===========================================================================
# RBAC Permission Tests
# ===========================================================================


class TestLearningHandlerRBACPermissions:
    """Tests for RBAC permission enforcement."""

    def test_list_sessions_has_permission_decorator(self, handler):
        """_list_sessions should be decorated with require_permission."""
        # Verify the method has been wrapped by the decorator
        method = handler._list_sessions
        assert hasattr(method, "__wrapped__")

    def test_create_session_has_permission_decorator(self, handler):
        """_create_session should be decorated with require_permission."""
        method = handler._create_session
        assert hasattr(method, "__wrapped__")

    def test_stop_session_has_permission_decorator(self, handler):
        """_stop_session should be decorated with require_permission."""
        method = handler._stop_session
        assert hasattr(method, "__wrapped__")

    def test_get_metrics_has_permission_decorator(self, handler):
        """_get_metrics should be decorated with require_permission."""
        method = handler._get_metrics
        assert hasattr(method, "__wrapped__")

    def test_submit_feedback_has_permission_decorator(self, handler):
        """_submit_feedback should be decorated with require_permission."""
        method = handler._submit_feedback
        assert hasattr(method, "__wrapped__")

    def test_calibrate_has_permission_decorator(self, handler):
        """_calibrate should be decorated with require_permission."""
        method = handler._calibrate
        assert hasattr(method, "__wrapped__")


# ===========================================================================
# SelectionFeedbackLoop Integration Tests
# ===========================================================================


class TestSelectionFeedbackLoopIntegration:
    """Tests for SelectionFeedbackLoop integration with autonomous learning."""

    @pytest.mark.asyncio
    async def test_calibration_stores_metric(self, handler):
        """Calibration should store a metric for tracking."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={"agent_ids": ["claude", "gpt-4"], "force": True},
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/learning/calibrate", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert data["metric"]["metadata"]["agent_ids"] == ["claude", "gpt-4"]
        assert data["metric"]["metadata"]["forced"] is True

        # Verify metric was stored
        assert len(handler._metrics) == 1
        assert handler._metrics[0].metric_type == MetricType.CALIBRATION

    @pytest.mark.asyncio
    async def test_debate_outcome_feedback_loop(self, handler):
        """Feedback on debate outcomes should be stored for learning."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={
                "feedback_type": "positive",
                "target_type": "session",
                "target_id": "session_001",
                "comment": "Agent calibration improved accuracy",
                "rating": 5,
            },
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/learning/feedback", {}, mock_request)

        assert result is not None
        assert result.status == 201

        # Verify feedback was stored
        assert len(handler._feedback) == 1
        feedback = handler._feedback[0]
        assert feedback.feedback_type == FeedbackType.POSITIVE
        assert feedback.rating == 5


# ===========================================================================
# Agent Calibration Update Tests
# ===========================================================================


class TestAgentCalibrationUpdates:
    """Tests for agent calibration updates."""

    @pytest.mark.asyncio
    async def test_calibration_includes_agent_ids(self, handler):
        """Calibration should include specified agent IDs."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={"agent_ids": ["claude", "gemini", "gpt-4"]},
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/learning/calibrate", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)

        assert "claude" in data["metric"]["metadata"]["agent_ids"]
        assert "gemini" in data["metric"]["metadata"]["agent_ids"]
        assert "gpt-4" in data["metric"]["metadata"]["agent_ids"]

    @pytest.mark.asyncio
    async def test_calibration_with_force_flag(self, handler):
        """Force calibration should be recorded in metadata."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={"force": True},
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/learning/calibrate", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)

        assert data["metric"]["metadata"]["forced"] is True


# ===========================================================================
# Learning From Debate Outcomes Tests
# ===========================================================================


class TestLearningFromDebateOutcomes:
    """Tests for learning from debate outcomes."""

    @pytest.mark.asyncio
    async def test_knowledge_extraction_from_debates(self, handler):
        """Should extract knowledge from debate IDs."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={
                "debate_ids": ["debate_001", "debate_002", "debate_003"],
                "title": "Multi-agent consensus patterns",
                "topics": ["ai", "consensus", "debate"],
            },
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post(
                "/api/v2/learning/knowledge/extract",
                {},
                mock_request,
            )

        assert result is not None
        assert result.status == 201
        data = json.loads(result.body)

        assert len(data["knowledge"]["source_debates"]) == 3
        assert "debate_001" in data["knowledge"]["source_debates"]

    @pytest.mark.asyncio
    async def test_pattern_detection_stores_debate_sources(self, handler, sample_pattern):
        """Detected patterns should track source debates."""
        handler._patterns[sample_pattern.id] = sample_pattern

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle(
                f"/api/v2/learning/patterns/{sample_pattern.id}",
                {},
                mock_request,
            )

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)

        assert "source_debates" in data
        assert len(data["source_debates"]) == 2


# ===========================================================================
# Model Improvement Mechanism Tests
# ===========================================================================


class TestModelImprovementMechanisms:
    """Tests for model improvement mechanisms."""

    @pytest.mark.asyncio
    async def test_training_session_tracks_loss_improvement(self, handler):
        """Training sessions should track loss improvement."""
        now = datetime.now(timezone.utc)
        session = TrainingSession(
            id="training_001",
            name="Improvement Tracking Session",
            mode=LearningMode.SUPERVISED,
            status=SessionStatus.RUNNING,
            created_at=now,
            started_at=now,
            owner_id="test-user",
            epochs_completed=50,
            total_epochs=100,
            current_loss=0.15,
            best_loss=0.10,
        )
        handler._sessions[session.id] = session

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle(
                f"/api/v2/learning/sessions/{session.id}",
                {},
                mock_request,
            )

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)

        assert data["current_loss"] == 0.15
        assert data["best_loss"] == 0.10

    @pytest.mark.asyncio
    async def test_performance_stats_include_learning_metrics(self, handler, sample_session):
        """Performance stats should include aggregated learning metrics."""
        # Add a completed session with metrics
        completed_session = TrainingSession(
            id="completed_001",
            name="Completed Session",
            mode=LearningMode.SUPERVISED,
            status=SessionStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
            owner_id="test-user",
            epochs_completed=100,
            total_epochs=100,
            current_loss=0.05,
            metrics={"accuracy": 0.95},
        )
        handler._sessions[completed_session.id] = completed_session

        # Add a pattern
        handler._patterns["p1"] = DetectedPattern(
            id="p1",
            pattern_type=PatternType.CONSENSUS,
            confidence=0.9,
            description="Test pattern",
            detected_at=datetime.now(timezone.utc),
        )

        # Add knowledge
        handler._knowledge["k1"] = ExtractedKnowledge(
            id="k1",
            title="Test Knowledge",
            content="Content",
            source_type="debate",
            source_debates=["d1"],
            confidence=0.8,
            extracted_at=datetime.now(timezone.utc),
        )

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/performance", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)

        assert data["performance"]["total_sessions"] == 1
        assert data["performance"]["successful_sessions"] == 1
        assert data["performance"]["patterns_detected"] == 1
        assert data["performance"]["knowledge_items_extracted"] == 1


# ===========================================================================
# Feedback Loop Endpoint Tests
# ===========================================================================


class TestFeedbackLoopEndpoints:
    """Tests for feedback loop endpoints."""

    @pytest.mark.asyncio
    async def test_positive_feedback_affects_recommendations(self, handler):
        """Positive feedback should not trigger negative feedback recommendations."""
        now = datetime.now(timezone.utc)
        # Add some positive feedback
        for i in range(5):
            handler._feedback.append(
                LearningFeedback(
                    id=f"feedback_{i}",
                    feedback_type=FeedbackType.POSITIVE,
                    target_type="session",
                    target_id=f"session_{i}",
                    comment="Good!",
                    submitted_by="test-user",
                    submitted_at=now,
                    rating=5,
                )
            )

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/recommendations", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)

        # Should not recommend addressing negative feedback since there isn't any
        titles = [r["title"] for r in data["recommendations"]]
        assert not any("negative" in t.lower() for t in titles)

    @pytest.mark.asyncio
    async def test_negative_feedback_triggers_recommendations(self, handler):
        """Excessive negative feedback should trigger recommendations."""
        now = datetime.now(timezone.utc)
        # Add negative feedback
        for i in range(5):
            handler._feedback.append(
                LearningFeedback(
                    id=f"feedback_{i}",
                    feedback_type=FeedbackType.NEGATIVE,
                    target_type="session",
                    target_id=f"session_{i}",
                    comment="Not good",
                    submitted_by="test-user",
                    submitted_at=now,
                    rating=1,
                )
            )

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/learning/recommendations", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)

        # Should recommend addressing negative feedback
        titles = [r["title"] for r in data["recommendations"]]
        assert any("negative" in t.lower() for t in titles)

    @pytest.mark.asyncio
    async def test_correction_feedback_type(self, handler):
        """Correction feedback should be stored correctly."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={
                "feedback_type": "correction",
                "target_type": "knowledge",
                "target_id": "knowledge_001",
                "comment": "The extracted knowledge needs correction",
            },
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/learning/feedback", {}, mock_request)

        assert result is not None
        assert result.status == 201
        data = json.loads(result.body)

        assert data["feedback"]["feedback_type"] == "correction"
        assert data["feedback"]["target_type"] == "knowledge"


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestAutonomousLearningIntegration:
    """Integration tests for autonomous learning handler."""

    @pytest.mark.asyncio
    async def test_full_session_lifecycle(self, handler):
        """Test complete session lifecycle: create -> monitor -> stop."""
        # 1. Create session
        create_request = MockHTTPHandler(
            method="POST",
            body={"name": "Integration Test Session", "mode": "reinforcement"},
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            create_result = await handler.handle_post(
                "/api/v2/learning/sessions",
                {},
                create_request,
            )

        assert create_result is not None
        assert create_result.status == 201
        session_id = json.loads(create_result.body)["session"]["id"]

        # 2. Get session details
        get_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            get_result = await handler.handle(
                f"/api/v2/learning/sessions/{session_id}",
                {},
                get_request,
            )

        assert get_result is not None
        assert get_result.status == 200
        assert json.loads(get_result.body)["status"] == "running"

        # 3. Stop session
        stop_request = MockHTTPHandler(method="POST")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            stop_result = await handler.handle_post(
                f"/api/v2/learning/sessions/{session_id}/stop",
                {},
                stop_request,
            )

        assert stop_result is not None
        assert stop_result.status == 200
        assert json.loads(stop_result.body)["session"]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_pattern_lifecycle(self, handler, sample_pattern):
        """Test pattern lifecycle: detect -> list -> validate."""
        handler._patterns[sample_pattern.id] = sample_pattern

        # 1. List patterns
        list_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            list_result = await handler.handle(
                "/api/v2/learning/patterns",
                {},
                list_request,
            )

        assert list_result is not None
        assert list_result.status == 200
        assert len(json.loads(list_result.body)["patterns"]) == 1

        # 2. Validate pattern
        validate_request = MockHTTPHandler(method="POST")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            validate_result = await handler.handle_post(
                f"/api/v2/learning/patterns/{sample_pattern.id}/validate",
                {},
                validate_request,
            )

        assert validate_result is not None
        assert validate_result.status == 200
        assert json.loads(validate_result.body)["pattern"]["is_validated"] is True

        # 3. Filter validated patterns
        validated_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            validated_result = await handler.handle(
                "/api/v2/learning/patterns",
                {"validated": "true"},
                validated_request,
            )

        assert validated_result is not None
        assert validated_result.status == 200
        validated_data = json.loads(validated_result.body)
        assert len(validated_data["patterns"]) == 1

    @pytest.mark.asyncio
    async def test_knowledge_extraction_lifecycle(self, handler):
        """Test knowledge extraction lifecycle: extract -> list -> get."""
        # 1. Extract knowledge
        extract_request = MockHTTPHandler(
            method="POST",
            body={
                "debate_ids": ["debate_1", "debate_2"],
                "title": "Test Knowledge",
                "topics": ["testing"],
            },
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            extract_result = await handler.handle_post(
                "/api/v2/learning/knowledge/extract",
                {},
                extract_request,
            )

        assert extract_result is not None
        assert extract_result.status == 201
        knowledge_id = json.loads(extract_result.body)["knowledge"]["id"]

        # 2. List knowledge
        list_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            list_result = await handler.handle(
                "/api/v2/learning/knowledge",
                {},
                list_request,
            )

        assert list_result is not None
        assert list_result.status == 200
        assert len(json.loads(list_result.body)["knowledge"]) == 1

        # 3. Get specific knowledge item
        get_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            get_result = await handler.handle(
                f"/api/v2/learning/knowledge/{knowledge_id}",
                {},
                get_request,
            )

        assert get_result is not None
        assert get_result.status == 200
        assert json.loads(get_result.body)["id"] == knowledge_id
