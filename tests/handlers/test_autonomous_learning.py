"""Tests for the AutonomousLearningHandler REST endpoints.

Covers all routes and behavior of the AutonomousLearningHandler class:
- can_handle() route matching (all routes + rejection)
- GET  /api/v2/learning/sessions          - List training sessions
- POST /api/v2/learning/sessions          - Start new training session
- GET  /api/v2/learning/sessions/:id      - Get session details
- POST /api/v2/learning/sessions/:id/stop - Stop training session
- GET  /api/v2/learning/metrics           - Get learning metrics
- GET  /api/v2/learning/metrics/:type     - Get specific metric
- POST /api/v2/learning/feedback          - Submit learning feedback
- GET  /api/v2/learning/patterns          - List detected patterns
- POST /api/v2/learning/patterns/:id/validate - Validate a pattern
- GET  /api/v2/learning/knowledge         - Get extracted knowledge
- POST /api/v2/learning/knowledge/extract - Trigger knowledge extraction
- GET  /api/v2/learning/recommendations   - Get learning recommendations
- GET  /api/v2/learning/performance       - Get model performance stats
- POST /api/v2/learning/calibrate         - Trigger calibration
- Error handling (400, 404, 500, 503)
- Circuit breaker integration
- Edge cases (missing params, invalid JSON, empty body)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.autonomous_learning import (
    AutonomousLearningHandler,
    DetectedPattern,
    ExtractedKnowledge,
    FeedbackType,
    LearningFeedback,
    LearningMetric,
    LearningMode,
    MAX_ACTIVE_SESSIONS,
    MetricType,
    PatternType,
    SessionStatus,
    TrainingSession,
    create_autonomous_learning_handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to handler methods."""

    def __init__(
        self,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        client_address: tuple[str, int] = ("127.0.0.1", 12345),
    ):
        self.command = method
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self.client_address = client_address
        self.path = ""

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers["Content-Length"] = str(len(raw))
            self.headers["Content-Type"] = "application/json"
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an AutonomousLearningHandler with empty server context."""
    return AutonomousLearningHandler({})


@pytest.fixture(autouse=True)
def _patch_rate_limit(monkeypatch):
    """Bypass rate limiting for tests."""
    monkeypatch.setenv("ARAGORA_USE_DISTRIBUTED_RATE_LIMIT", "false")


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset circuit breaker between tests to avoid cross-test contamination."""
    yield
    # Reset the global CB registry so each test starts fresh
    try:
        from aragora.resilience.circuit_breaker import _circuit_breakers

        _circuit_breakers.pop("autonomous_learning", None)
    except (ImportError, AttributeError):
        pass


def _add_session(
    handler: AutonomousLearningHandler,
    session_id: str = "session_abc123",
    name: str = "Test Session",
    mode: LearningMode = LearningMode.SUPERVISED,
    status: SessionStatus = SessionStatus.RUNNING,
    **kwargs,
) -> TrainingSession:
    """Helper to add a session directly to handler state."""
    now = datetime.now(timezone.utc)
    session = TrainingSession(
        id=session_id,
        name=name,
        mode=mode,
        status=status,
        created_at=kwargs.pop("created_at", now),
        started_at=kwargs.pop("started_at", now),
        owner_id=kwargs.pop("owner_id", "test-user"),
        **kwargs,
    )
    handler._sessions[session_id] = session
    return session


def _add_pattern(
    handler: AutonomousLearningHandler,
    pattern_id: str = "pattern_abc123",
    pattern_type: PatternType = PatternType.CONSENSUS,
    confidence: float = 0.85,
    **kwargs,
) -> DetectedPattern:
    """Helper to add a pattern directly to handler state."""
    now = datetime.now(timezone.utc)
    pattern = DetectedPattern(
        id=pattern_id,
        pattern_type=pattern_type,
        confidence=confidence,
        description=kwargs.pop("description", "Test pattern"),
        detected_at=kwargs.pop("detected_at", now),
        **kwargs,
    )
    handler._patterns[pattern_id] = pattern
    return pattern


def _add_knowledge(
    handler: AutonomousLearningHandler,
    knowledge_id: str = "knowledge_abc123",
    **kwargs,
) -> ExtractedKnowledge:
    """Helper to add a knowledge item directly to handler state."""
    now = datetime.now(timezone.utc)
    knowledge = ExtractedKnowledge(
        id=knowledge_id,
        title=kwargs.pop("title", "Test Knowledge"),
        content=kwargs.pop("content", "Some knowledge content"),
        source_type=kwargs.pop("source_type", "debate_analysis"),
        source_debates=kwargs.pop("source_debates", ["debate_1"]),
        confidence=kwargs.pop("confidence", 0.9),
        extracted_at=kwargs.pop("extracted_at", now),
        **kwargs,
    )
    handler._knowledge[knowledge_id] = knowledge
    return knowledge


def _add_metric(
    handler: AutonomousLearningHandler,
    metric_type: MetricType = MetricType.ACCURACY,
    value: float = 0.95,
    **kwargs,
) -> LearningMetric:
    """Helper to add a metric directly to handler state."""
    now = datetime.now(timezone.utc)
    metric = LearningMetric(
        metric_type=metric_type,
        value=value,
        timestamp=kwargs.pop("timestamp", now),
        **kwargs,
    )
    handler._metrics.append(metric)
    return metric


# ---------------------------------------------------------------------------
# Initialization and Factory
# ---------------------------------------------------------------------------


class TestInitialization:
    """Tests for handler initialization."""

    def test_init_creates_empty_state(self, handler):
        assert handler._sessions == {}
        assert handler._metrics == []
        assert handler._patterns == {}
        assert handler._knowledge == {}
        assert handler._feedback == []
        assert handler._circuit_breaker is None

    def test_factory_creates_handler(self):
        h = create_autonomous_learning_handler({})
        assert isinstance(h, AutonomousLearningHandler)

    def test_factory_passes_context(self):
        ctx = {"key": "value"}
        h = create_autonomous_learning_handler(ctx)
        assert h.ctx == ctx


# ---------------------------------------------------------------------------
# can_handle() routing
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for route matching via can_handle()."""

    def test_sessions_get(self, handler):
        assert handler.can_handle("/api/v2/learning/sessions", "GET")

    def test_sessions_post(self, handler):
        assert handler.can_handle("/api/v2/learning/sessions", "POST")

    def test_sessions_with_id(self, handler):
        assert handler.can_handle("/api/v2/learning/sessions/session_abc", "GET")

    def test_sessions_stop(self, handler):
        assert handler.can_handle("/api/v2/learning/sessions/session_abc/stop", "POST")

    def test_metrics(self, handler):
        assert handler.can_handle("/api/v2/learning/metrics", "GET")

    def test_metrics_specific(self, handler):
        assert handler.can_handle("/api/v2/learning/metrics/accuracy", "GET")

    def test_feedback(self, handler):
        assert handler.can_handle("/api/v2/learning/feedback", "POST")

    def test_patterns(self, handler):
        assert handler.can_handle("/api/v2/learning/patterns", "GET")

    def test_patterns_validate(self, handler):
        assert handler.can_handle("/api/v2/learning/patterns/p1/validate", "POST")

    def test_knowledge(self, handler):
        assert handler.can_handle("/api/v2/learning/knowledge", "GET")

    def test_knowledge_extract(self, handler):
        assert handler.can_handle("/api/v2/learning/knowledge/extract", "POST")

    def test_recommendations(self, handler):
        assert handler.can_handle("/api/v2/learning/recommendations", "GET")

    def test_performance(self, handler):
        assert handler.can_handle("/api/v2/learning/performance", "GET")

    def test_calibrate(self, handler):
        assert handler.can_handle("/api/v2/learning/calibrate", "POST")

    def test_delete_accepted(self, handler):
        assert handler.can_handle("/api/v2/learning/sessions", "DELETE")

    def test_unrelated_path_rejected(self, handler):
        assert not handler.can_handle("/api/v2/debates", "GET")

    def test_v1_learning_rejected(self, handler):
        """v1 learning paths are not in can_handle prefix check."""
        assert not handler.can_handle("/api/v1/learning/sessions", "GET")

    def test_put_method_rejected(self, handler):
        assert not handler.can_handle("/api/v2/learning/sessions", "PUT")

    def test_patch_method_rejected(self, handler):
        assert not handler.can_handle("/api/v2/learning/sessions", "PATCH")

    def test_empty_path_rejected(self, handler):
        assert not handler.can_handle("", "GET")

    def test_bare_learning_rejected(self, handler):
        """Path /api/v2/learning without trailing slash does not match."""
        assert not handler.can_handle("/api/v2/learning", "GET")


# ---------------------------------------------------------------------------
# GET /api/v2/learning/sessions - List Sessions
# ---------------------------------------------------------------------------


class TestListSessions:
    """Tests for session listing."""

    @pytest.mark.asyncio
    async def test_empty_sessions(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/sessions", {}, http)
        body = _body(result)
        assert body["sessions"] == []
        assert body["pagination"]["total"] == 0
        assert body["pagination"]["has_more"] is False

    @pytest.mark.asyncio
    async def test_list_with_sessions(self, handler):
        _add_session(handler, "s1", name="Session 1")
        _add_session(handler, "s2", name="Session 2")
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/sessions", {}, http)
        body = _body(result)
        assert len(body["sessions"]) == 2
        assert body["pagination"]["total"] == 2

    @pytest.mark.asyncio
    async def test_filter_by_status(self, handler):
        _add_session(handler, "s1", status=SessionStatus.RUNNING)
        _add_session(handler, "s2", status=SessionStatus.COMPLETED)
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/sessions", {"status": "running"}, http
        )
        body = _body(result)
        assert len(body["sessions"]) == 1
        assert body["sessions"][0]["id"] == "s1"

    @pytest.mark.asyncio
    async def test_filter_by_invalid_status(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/sessions", {"status": "bogus"}, http
        )
        assert _status(result) == 400
        assert "Invalid status" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_filter_by_mode(self, handler):
        _add_session(handler, "s1", mode=LearningMode.SUPERVISED)
        _add_session(handler, "s2", mode=LearningMode.FEDERATED)
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/sessions", {"mode": "federated"}, http
        )
        body = _body(result)
        assert len(body["sessions"]) == 1
        assert body["sessions"][0]["id"] == "s2"

    @pytest.mark.asyncio
    async def test_filter_by_invalid_mode(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/sessions", {"mode": "bogus"}, http
        )
        assert _status(result) == 400
        assert "Invalid mode" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_pagination_limit_offset(self, handler):
        for i in range(5):
            _add_session(
                handler,
                f"s{i}",
                name=f"Session {i}",
                created_at=datetime(2026, 1, 1 + i, tzinfo=timezone.utc),
            )
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/sessions", {"limit": "2", "offset": "0"}, http
        )
        body = _body(result)
        assert len(body["sessions"]) == 2
        assert body["pagination"]["total"] == 5
        assert body["pagination"]["has_more"] is True

    @pytest.mark.asyncio
    async def test_pagination_offset_beyond_total(self, handler):
        _add_session(handler, "s1")
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/sessions", {"offset": "100"}, http
        )
        body = _body(result)
        assert body["sessions"] == []
        assert body["pagination"]["total"] == 1

    @pytest.mark.asyncio
    async def test_sessions_sorted_by_created_at_descending(self, handler):
        _add_session(
            handler, "old", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )
        _add_session(
            handler, "new", created_at=datetime(2026, 2, 1, tzinfo=timezone.utc)
        )
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/sessions", {}, http)
        body = _body(result)
        assert body["sessions"][0]["id"] == "new"
        assert body["sessions"][1]["id"] == "old"


# ---------------------------------------------------------------------------
# POST /api/v2/learning/sessions - Create Session
# ---------------------------------------------------------------------------


class TestCreateSession:
    """Tests for session creation."""

    @pytest.mark.asyncio
    async def test_create_session_success(self, handler):
        http = MockHTTPHandler(method="POST", body={"name": "My Session"})
        result = await handler.handle_post("/api/v2/learning/sessions", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert "session" in body
        assert body["session"]["name"] == "My Session"
        assert body["session"]["status"] == "running"
        assert body["session"]["mode"] == "supervised"
        assert len(handler._sessions) == 1

    @pytest.mark.asyncio
    async def test_create_session_with_mode(self, handler):
        http = MockHTTPHandler(
            method="POST",
            body={"name": "Transfer Session", "mode": "transfer"},
        )
        result = await handler.handle_post("/api/v2/learning/sessions", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["session"]["mode"] == "transfer"

    @pytest.mark.asyncio
    async def test_create_session_with_config(self, handler):
        http = MockHTTPHandler(
            method="POST",
            body={
                "name": "Configured Session",
                "config": {"learning_rate": 0.01},
                "total_epochs": 50,
            },
        )
        result = await handler.handle_post("/api/v2/learning/sessions", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["session"]["config"] == {"learning_rate": 0.01}
        assert body["session"]["total_epochs"] == 50

    @pytest.mark.asyncio
    async def test_create_session_missing_name(self, handler):
        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post("/api/v2/learning/sessions", {}, http)
        assert _status(result) == 400
        assert "name is required" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_create_session_blank_name(self, handler):
        http = MockHTTPHandler(method="POST", body={"name": "   "})
        result = await handler.handle_post("/api/v2/learning/sessions", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_session_invalid_mode(self, handler):
        http = MockHTTPHandler(
            method="POST", body={"name": "Bad Mode", "mode": "invalid"}
        )
        result = await handler.handle_post("/api/v2/learning/sessions", {}, http)
        assert _status(result) == 400
        assert "Invalid learning mode" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_create_session_max_active_reached(self, handler):
        # Fill up to MAX_ACTIVE_SESSIONS
        for i in range(MAX_ACTIVE_SESSIONS):
            _add_session(handler, f"active_{i}", status=SessionStatus.RUNNING)
        http = MockHTTPHandler(method="POST", body={"name": "One Too Many"})
        result = await handler.handle_post("/api/v2/learning/sessions", {}, http)
        assert _status(result) == 400
        assert "Maximum active sessions" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_create_session_completed_dont_count(self, handler):
        """Completed sessions should not count toward the active limit."""
        for i in range(MAX_ACTIVE_SESSIONS):
            _add_session(handler, f"done_{i}", status=SessionStatus.COMPLETED)
        http = MockHTTPHandler(method="POST", body={"name": "Still Room"})
        result = await handler.handle_post("/api/v2/learning/sessions", {}, http)
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_create_session_assigns_owner(self, handler):
        http = MockHTTPHandler(method="POST", body={"name": "Owned Session"})
        result = await handler.handle_post("/api/v2/learning/sessions", {}, http)
        body = _body(result)
        # With the conftest mock, get_current_user returns user_id="test-user-001"
        assert body["session"]["owner_id"] == "test-user-001"


# ---------------------------------------------------------------------------
# GET /api/v2/learning/sessions/:id - Get Session
# ---------------------------------------------------------------------------


class TestGetSession:
    """Tests for getting a specific session."""

    @pytest.mark.asyncio
    async def test_get_existing_session(self, handler):
        _add_session(handler, "session_123", name="My Session")
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/sessions/session_123", {}, http
        )
        body = _body(result)
        assert body["id"] == "session_123"
        assert body["name"] == "My Session"

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/sessions/does_not_exist", {}, http
        )
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_session_to_dict_fields(self, handler):
        _add_session(
            handler,
            "session_123",
            name="Full Session",
            mode=LearningMode.REINFORCEMENT,
            status=SessionStatus.RUNNING,
            epochs_completed=25,
            total_epochs=100,
        )
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/sessions/session_123", {}, http
        )
        body = _body(result)
        assert body["mode"] == "reinforcement"
        assert body["status"] == "running"
        assert body["epochs_completed"] == 25
        assert body["total_epochs"] == 100
        assert body["progress_percent"] == 25.0


# ---------------------------------------------------------------------------
# POST /api/v2/learning/sessions/:id/stop - Stop Session
# ---------------------------------------------------------------------------


class TestStopSession:
    """Tests for stopping a training session."""

    @pytest.mark.asyncio
    async def test_stop_running_session(self, handler):
        _add_session(handler, "session_123", status=SessionStatus.RUNNING)
        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post(
            "/api/v2/learning/sessions/session_123/stop", {}, http
        )
        body = _body(result)
        assert body["session"]["status"] == "cancelled"
        assert body["session"]["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_stop_pending_session(self, handler):
        _add_session(handler, "session_123", status=SessionStatus.PENDING)
        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post(
            "/api/v2/learning/sessions/session_123/stop", {}, http
        )
        body = _body(result)
        assert body["session"]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_already_completed(self, handler):
        _add_session(handler, "session_123", status=SessionStatus.COMPLETED)
        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post(
            "/api/v2/learning/sessions/session_123/stop", {}, http
        )
        assert _status(result) == 400
        assert "Cannot stop session" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_stop_already_failed(self, handler):
        _add_session(handler, "session_123", status=SessionStatus.FAILED)
        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post(
            "/api/v2/learning/sessions/session_123/stop", {}, http
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_stop_nonexistent_session(self, handler):
        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post(
            "/api/v2/learning/sessions/doesnt_exist/stop", {}, http
        )
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# GET /api/v2/learning/metrics - Get Metrics
# ---------------------------------------------------------------------------


class TestGetMetrics:
    """Tests for metrics listing."""

    @pytest.mark.asyncio
    async def test_empty_metrics(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/metrics", {}, http)
        body = _body(result)
        assert body["metrics"] == []
        assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_list_metrics(self, handler):
        _add_metric(handler, MetricType.ACCURACY, 0.95)
        _add_metric(handler, MetricType.LOSS, 0.05)
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/metrics", {}, http)
        body = _body(result)
        assert body["count"] == 2

    @pytest.mark.asyncio
    async def test_filter_by_session_id(self, handler):
        _add_metric(handler, session_id="s1")
        _add_metric(handler, session_id="s2")
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/metrics", {"session_id": "s1"}, http
        )
        body = _body(result)
        assert body["count"] == 1

    @pytest.mark.asyncio
    async def test_filter_by_agent_id(self, handler):
        _add_metric(handler, agent_id="agent_1")
        _add_metric(handler, agent_id="agent_2")
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/metrics", {"agent_id": "agent_1"}, http
        )
        body = _body(result)
        assert body["count"] == 1

    @pytest.mark.asyncio
    async def test_metrics_limit(self, handler):
        for i in range(10):
            _add_metric(handler, value=float(i))
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/metrics", {"limit": "3"}, http
        )
        body = _body(result)
        assert body["count"] == 3

    @pytest.mark.asyncio
    async def test_metrics_sorted_by_timestamp(self, handler):
        _add_metric(
            handler, value=0.1, timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )
        _add_metric(
            handler, value=0.9, timestamp=datetime(2026, 2, 1, tzinfo=timezone.utc)
        )
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/metrics", {}, http)
        body = _body(result)
        # Most recent first
        assert body["metrics"][0]["value"] == 0.9


# ---------------------------------------------------------------------------
# GET /api/v2/learning/metrics/:type - Get Metric by Type
# ---------------------------------------------------------------------------


class TestGetMetricByType:
    """Tests for getting metrics of a specific type."""

    @pytest.mark.asyncio
    async def test_get_accuracy_metrics(self, handler):
        _add_metric(handler, MetricType.ACCURACY, 0.9)
        _add_metric(handler, MetricType.ACCURACY, 0.8)
        _add_metric(handler, MetricType.LOSS, 0.1)
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/metrics/accuracy", {}, http
        )
        body = _body(result)
        assert body["metric_type"] == "accuracy"
        assert body["count"] == 2
        assert body["average"] == 0.85
        assert body["min"] == 0.8
        assert body["max"] == 0.9

    @pytest.mark.asyncio
    async def test_get_empty_metric_type(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/metrics/accuracy", {}, http
        )
        body = _body(result)
        assert body["count"] == 0
        assert body["average"] == 0.0
        assert body["min"] == 0.0
        assert body["max"] == 0.0

    @pytest.mark.asyncio
    async def test_get_invalid_metric_type(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/metrics/bogus_type", {}, http
        )
        assert _status(result) == 400
        assert "Invalid metric type" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_recent_limited_to_10(self, handler):
        for i in range(15):
            _add_metric(
                handler,
                MetricType.LOSS,
                float(i) / 100,
                timestamp=datetime(2026, 1, 1, hour=i, tzinfo=timezone.utc),
            )
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/metrics/loss", {}, http)
        body = _body(result)
        assert body["count"] == 15
        assert len(body["recent"]) == 10


# ---------------------------------------------------------------------------
# POST /api/v2/learning/feedback - Submit Feedback
# ---------------------------------------------------------------------------


class TestSubmitFeedback:
    """Tests for submitting learning feedback."""

    @pytest.mark.asyncio
    async def test_submit_feedback_success(self, handler):
        http = MockHTTPHandler(
            method="POST",
            body={
                "feedback_type": "positive",
                "target_type": "session",
                "target_id": "session_123",
                "comment": "Great results!",
            },
        )
        result = await handler.handle_post("/api/v2/learning/feedback", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["feedback"]["feedback_type"] == "positive"
        assert body["feedback"]["target_type"] == "session"
        assert body["feedback"]["target_id"] == "session_123"
        assert body["feedback"]["comment"] == "Great results!"
        assert len(handler._feedback) == 1

    @pytest.mark.asyncio
    async def test_submit_feedback_with_rating(self, handler):
        http = MockHTTPHandler(
            method="POST",
            body={
                "feedback_type": "positive",
                "target_type": "pattern",
                "target_id": "pattern_123",
                "comment": "Good",
                "rating": 5,
            },
        )
        result = await handler.handle_post("/api/v2/learning/feedback", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["feedback"]["rating"] == 5

    @pytest.mark.asyncio
    async def test_submit_feedback_invalid_type(self, handler):
        http = MockHTTPHandler(
            method="POST",
            body={
                "feedback_type": "bogus",
                "target_type": "session",
                "target_id": "s1",
            },
        )
        result = await handler.handle_post("/api/v2/learning/feedback", {}, http)
        assert _status(result) == 400
        assert "Invalid feedback type" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_submit_feedback_missing_target_type(self, handler):
        http = MockHTTPHandler(
            method="POST",
            body={"feedback_type": "positive", "target_id": "s1"},
        )
        result = await handler.handle_post("/api/v2/learning/feedback", {}, http)
        assert _status(result) == 400
        assert "target_type and target_id are required" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_submit_feedback_missing_target_id(self, handler):
        http = MockHTTPHandler(
            method="POST",
            body={"feedback_type": "positive", "target_type": "session"},
        )
        result = await handler.handle_post("/api/v2/learning/feedback", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_submit_feedback_invalid_target_type(self, handler):
        http = MockHTTPHandler(
            method="POST",
            body={
                "feedback_type": "positive",
                "target_type": "invalid_target",
                "target_id": "x",
            },
        )
        result = await handler.handle_post("/api/v2/learning/feedback", {}, http)
        assert _status(result) == 400
        assert "Invalid target_type" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_submit_feedback_default_neutral(self, handler):
        http = MockHTTPHandler(
            method="POST",
            body={
                "target_type": "knowledge",
                "target_id": "k1",
            },
        )
        result = await handler.handle_post("/api/v2/learning/feedback", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["feedback"]["feedback_type"] == "neutral"

    @pytest.mark.asyncio
    async def test_submit_feedback_assigns_submitter(self, handler):
        http = MockHTTPHandler(
            method="POST",
            body={
                "target_type": "session",
                "target_id": "s1",
            },
        )
        result = await handler.handle_post("/api/v2/learning/feedback", {}, http)
        body = _body(result)
        assert body["feedback"]["submitted_by"] == "test-user-001"


# ---------------------------------------------------------------------------
# GET /api/v2/learning/patterns - List Patterns
# ---------------------------------------------------------------------------


class TestListPatterns:
    """Tests for pattern listing."""

    @pytest.mark.asyncio
    async def test_empty_patterns(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/patterns", {}, http)
        body = _body(result)
        assert body["patterns"] == []
        assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_list_patterns(self, handler):
        _add_pattern(handler, "p1", confidence=0.9)
        _add_pattern(handler, "p2", confidence=0.8)
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/patterns", {}, http)
        body = _body(result)
        assert body["count"] == 2

    @pytest.mark.asyncio
    async def test_filter_by_pattern_type(self, handler):
        _add_pattern(handler, "p1", pattern_type=PatternType.CONSENSUS)
        _add_pattern(handler, "p2", pattern_type=PatternType.TEMPORAL)
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/patterns", {"pattern_type": "temporal"}, http
        )
        body = _body(result)
        assert body["count"] == 1
        assert body["patterns"][0]["pattern_type"] == "temporal"

    @pytest.mark.asyncio
    async def test_filter_by_invalid_pattern_type(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/patterns", {"pattern_type": "bogus"}, http
        )
        assert _status(result) == 400
        assert "Invalid pattern type" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_filter_validated_only(self, handler):
        _add_pattern(handler, "p1", is_validated=True)
        _add_pattern(handler, "p2", is_validated=False)
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/patterns", {"validated": "true"}, http
        )
        body = _body(result)
        assert body["count"] == 1
        assert body["patterns"][0]["id"] == "p1"

    @pytest.mark.asyncio
    async def test_filter_min_confidence(self, handler):
        _add_pattern(handler, "p_low", confidence=0.3)
        _add_pattern(handler, "p_high", confidence=0.9)
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/patterns", {"min_confidence": "0.8"}, http
        )
        body = _body(result)
        assert body["count"] == 1
        assert body["patterns"][0]["id"] == "p_high"

    @pytest.mark.asyncio
    async def test_default_min_confidence_filters_low(self, handler):
        _add_pattern(handler, "p_low", confidence=0.3)
        _add_pattern(handler, "p_above", confidence=0.6)
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/patterns", {}, http)
        body = _body(result)
        # Default MIN_PATTERN_CONFIDENCE = 0.5
        assert body["count"] == 1
        assert body["patterns"][0]["id"] == "p_above"

    @pytest.mark.asyncio
    async def test_invalid_min_confidence_uses_default(self, handler):
        _add_pattern(handler, "p1", confidence=0.6)
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/patterns", {"min_confidence": "not_a_number"}, http
        )
        body = _body(result)
        # Falls back to MIN_PATTERN_CONFIDENCE = 0.5, so p1 (0.6) is included
        assert body["count"] == 1

    @pytest.mark.asyncio
    async def test_patterns_sorted_by_confidence_desc(self, handler):
        _add_pattern(handler, "low", confidence=0.6)
        _add_pattern(handler, "high", confidence=0.95)
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/patterns", {}, http)
        body = _body(result)
        assert body["patterns"][0]["id"] == "high"

    @pytest.mark.asyncio
    async def test_patterns_limit(self, handler):
        for i in range(10):
            _add_pattern(handler, f"p{i}", confidence=0.5 + i * 0.04)
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/patterns", {"limit": "3"}, http
        )
        body = _body(result)
        assert body["count"] == 3


# ---------------------------------------------------------------------------
# GET /api/v2/learning/patterns/:id - Get Pattern
# ---------------------------------------------------------------------------


class TestGetPattern:
    """Tests for getting a specific pattern."""

    @pytest.mark.asyncio
    async def test_get_existing_pattern(self, handler):
        _add_pattern(handler, "pattern_xyz", description="A consensus pattern")
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/patterns/pattern_xyz", {}, http
        )
        body = _body(result)
        assert body["id"] == "pattern_xyz"
        assert body["description"] == "A consensus pattern"

    @pytest.mark.asyncio
    async def test_get_nonexistent_pattern(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/patterns/nonexistent", {}, http
        )
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# POST /api/v2/learning/patterns/:id/validate - Validate Pattern
# ---------------------------------------------------------------------------


class TestValidatePattern:
    """Tests for pattern validation."""

    @pytest.mark.asyncio
    async def test_validate_pattern_success(self, handler):
        _add_pattern(handler, "pattern_123")
        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post(
            "/api/v2/learning/patterns/pattern_123/validate", {}, http
        )
        body = _body(result)
        assert body["pattern"]["is_validated"] is True
        assert body["pattern"]["validated_by"] == "test-user-001"
        assert body["pattern"]["validated_at"] is not None
        assert "validated" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_validate_nonexistent_pattern(self, handler):
        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post(
            "/api/v2/learning/patterns/nope/validate", {}, http
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_validate_pattern_short_path_returns_400(self, handler):
        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post(
            "/api/v2/learning/patterns/", {}, http
        )
        # Pattern path with empty pattern_id but missing "validate" segment
        # len(parts) < 7 -> 400
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# GET /api/v2/learning/knowledge - List Knowledge
# ---------------------------------------------------------------------------


class TestListKnowledge:
    """Tests for knowledge listing."""

    @pytest.mark.asyncio
    async def test_empty_knowledge(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/knowledge", {}, http)
        body = _body(result)
        assert body["knowledge"] == []
        assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_list_knowledge(self, handler):
        _add_knowledge(handler, "k1")
        _add_knowledge(handler, "k2")
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/knowledge", {}, http)
        body = _body(result)
        assert body["count"] == 2

    @pytest.mark.asyncio
    async def test_filter_verified_only(self, handler):
        _add_knowledge(handler, "k1", is_verified=True)
        _add_knowledge(handler, "k2", is_verified=False)
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/knowledge", {"verified": "true"}, http
        )
        body = _body(result)
        assert body["count"] == 1
        assert body["knowledge"][0]["id"] == "k1"

    @pytest.mark.asyncio
    async def test_filter_by_source_type(self, handler):
        _add_knowledge(handler, "k1", source_type="debate_analysis")
        _add_knowledge(handler, "k2", source_type="manual")
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/knowledge", {"source_type": "manual"}, http
        )
        body = _body(result)
        assert body["count"] == 1
        assert body["knowledge"][0]["id"] == "k2"

    @pytest.mark.asyncio
    async def test_knowledge_sorted_by_confidence(self, handler):
        _add_knowledge(handler, "low", confidence=0.5)
        _add_knowledge(handler, "high", confidence=0.99)
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/knowledge", {}, http)
        body = _body(result)
        assert body["knowledge"][0]["id"] == "high"


# ---------------------------------------------------------------------------
# GET /api/v2/learning/knowledge/:id - Get Knowledge Item
# ---------------------------------------------------------------------------


class TestGetKnowledgeItem:
    """Tests for getting a specific knowledge item."""

    @pytest.mark.asyncio
    async def test_get_existing(self, handler):
        _add_knowledge(handler, "knowledge_xyz", title="My Knowledge")
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/knowledge/knowledge_xyz", {}, http
        )
        body = _body(result)
        assert body["id"] == "knowledge_xyz"
        assert body["title"] == "My Knowledge"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/knowledge/nonexistent", {}, http
        )
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# POST /api/v2/learning/knowledge/extract - Extract Knowledge
# ---------------------------------------------------------------------------


class TestExtractKnowledge:
    """Tests for knowledge extraction."""

    @pytest.mark.asyncio
    async def test_extract_success(self, handler):
        http = MockHTTPHandler(
            method="POST",
            body={"debate_ids": ["d1", "d2"], "title": "Extracted"},
        )
        result = await handler.handle_post(
            "/api/v2/learning/knowledge/extract", {}, http
        )
        assert _status(result) == 201
        body = _body(result)
        assert "knowledge" in body
        assert body["knowledge"]["source_debates"] == ["d1", "d2"]
        assert body["knowledge"]["title"] == "Extracted"
        assert len(handler._knowledge) == 1

    @pytest.mark.asyncio
    async def test_extract_missing_debate_ids(self, handler):
        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post(
            "/api/v2/learning/knowledge/extract", {}, http
        )
        assert _status(result) == 400
        assert "debate_ids is required" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_extract_empty_debate_ids(self, handler):
        http = MockHTTPHandler(method="POST", body={"debate_ids": []})
        result = await handler.handle_post(
            "/api/v2/learning/knowledge/extract", {}, http
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_extract_default_fields(self, handler):
        http = MockHTTPHandler(
            method="POST", body={"debate_ids": ["d1"]}
        )
        result = await handler.handle_post(
            "/api/v2/learning/knowledge/extract", {}, http
        )
        body = _body(result)
        assert body["knowledge"]["title"] == "Extracted Knowledge"
        assert body["knowledge"]["source_type"] == "debate_analysis"
        assert body["knowledge"]["topics"] == ["general"]

    @pytest.mark.asyncio
    async def test_extract_custom_topics(self, handler):
        http = MockHTTPHandler(
            method="POST",
            body={"debate_ids": ["d1"], "topics": ["ai", "safety"]},
        )
        result = await handler.handle_post(
            "/api/v2/learning/knowledge/extract", {}, http
        )
        body = _body(result)
        assert body["knowledge"]["topics"] == ["ai", "safety"]


# ---------------------------------------------------------------------------
# GET /api/v2/learning/recommendations - Get Recommendations
# ---------------------------------------------------------------------------


class TestGetRecommendations:
    """Tests for recommendations."""

    @pytest.mark.asyncio
    async def test_recommendations_no_sessions(self, handler):
        """With no running sessions, suggests starting one."""
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/recommendations", {}, http)
        body = _body(result)
        assert body["count"] >= 1
        titles = [r["title"] for r in body["recommendations"]]
        assert any("Start a training session" in t for t in titles)

    @pytest.mark.asyncio
    async def test_recommendations_with_running_session(self, handler):
        """With a running session, no 'start' recommendation."""
        _add_session(handler, "s1", status=SessionStatus.RUNNING)
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/recommendations", {}, http)
        body = _body(result)
        titles = [r["title"] for r in body["recommendations"]]
        assert not any("Start a training session" in t for t in titles)

    @pytest.mark.asyncio
    async def test_recommendations_unvalidated_patterns(self, handler):
        """High-confidence unvalidated patterns trigger validation recommendation."""
        _add_pattern(handler, "p1", confidence=0.8, is_validated=False)
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/recommendations", {}, http)
        body = _body(result)
        titles = [r["title"] for r in body["recommendations"]]
        assert any("Validate" in t for t in titles)

    @pytest.mark.asyncio
    async def test_recommendations_negative_feedback(self, handler):
        """Many negative feedback items trigger address recommendation."""
        now = datetime.now(timezone.utc)
        for i in range(5):
            handler._feedback.append(
                LearningFeedback(
                    id=f"f{i}",
                    feedback_type=FeedbackType.NEGATIVE,
                    target_type="session",
                    target_id="s1",
                    comment="Bad",
                    submitted_by="user",
                    submitted_at=now - timedelta(days=1),
                )
            )
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/recommendations", {}, http)
        body = _body(result)
        titles = [r["title"] for r in body["recommendations"]]
        assert any("negative feedback" in t.lower() for t in titles)

    @pytest.mark.asyncio
    async def test_recommendations_limit(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle(
            "/api/v2/learning/recommendations", {"limit": "1"}, http
        )
        body = _body(result)
        assert body["count"] <= 1

    @pytest.mark.asyncio
    async def test_recommendations_sorted_by_priority(self, handler):
        """Recommendations should be sorted by priority (low number = high priority)."""
        _add_pattern(handler, "p1", confidence=0.9, is_validated=False)
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/recommendations", {}, http)
        body = _body(result)
        if len(body["recommendations"]) >= 2:
            priorities = [r["priority"] for r in body["recommendations"]]
            assert priorities == sorted(priorities)


# ---------------------------------------------------------------------------
# GET /api/v2/learning/performance - Get Performance Stats
# ---------------------------------------------------------------------------


class TestGetPerformance:
    """Tests for performance statistics."""

    @pytest.mark.asyncio
    async def test_performance_empty_state(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/performance", {}, http)
        body = _body(result)
        perf = body["performance"]
        assert perf["total_sessions"] == 0
        assert perf["successful_sessions"] == 0
        assert perf["failed_sessions"] == 0
        assert perf["average_accuracy"] == 0.0
        assert perf["average_loss"] == 0.0
        assert perf["total_epochs_trained"] == 0
        assert perf["patterns_detected"] == 0
        assert perf["knowledge_items_extracted"] == 0
        assert perf["feedback_received"] == 0

    @pytest.mark.asyncio
    async def test_performance_with_sessions(self, handler):
        s1 = _add_session(handler, "s1", status=SessionStatus.COMPLETED, epochs_completed=50)
        s1.metrics["accuracy"] = 0.9
        s1.current_loss = 0.1
        _add_session(handler, "s2", status=SessionStatus.FAILED, epochs_completed=10)
        _add_session(handler, "s3", status=SessionStatus.RUNNING, epochs_completed=20)
        _add_pattern(handler, "p1")
        _add_knowledge(handler, "k1")

        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/performance", {}, http)
        body = _body(result)
        perf = body["performance"]
        assert perf["total_sessions"] == 3
        assert perf["successful_sessions"] == 1
        assert perf["failed_sessions"] == 1
        assert perf["average_accuracy"] == 0.9
        assert perf["average_loss"] == 0.1
        assert perf["total_epochs_trained"] == 80
        assert perf["patterns_detected"] == 1
        assert perf["knowledge_items_extracted"] == 1

    @pytest.mark.asyncio
    async def test_performance_success_rate_calculation(self, handler):
        _add_session(handler, "s1", status=SessionStatus.COMPLETED)
        _add_session(handler, "s2", status=SessionStatus.COMPLETED)
        _add_session(handler, "s3", status=SessionStatus.FAILED)
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/performance", {}, http)
        body = _body(result)
        # 2 successful out of 3 total = 66.67%
        assert body["performance"]["success_rate"] == 66.67


# ---------------------------------------------------------------------------
# POST /api/v2/learning/calibrate - Calibrate
# ---------------------------------------------------------------------------


class TestCalibrate:
    """Tests for triggering calibration."""

    @pytest.mark.asyncio
    async def test_calibrate_success(self, handler):
        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post("/api/v2/learning/calibrate", {}, http)
        body = _body(result)
        assert "calibration_id" in body
        assert body["calibration_id"].startswith("calibration_")
        assert "metric" in body
        assert body["metric"]["metric_type"] == "calibration"
        assert body["message"] == "Calibration completed successfully"
        # Metric was added to internal state
        assert len(handler._metrics) == 1

    @pytest.mark.asyncio
    async def test_calibrate_with_agent_ids(self, handler):
        http = MockHTTPHandler(
            method="POST", body={"agent_ids": ["a1", "a2"], "force": True}
        )
        result = await handler.handle_post("/api/v2/learning/calibrate", {}, http)
        body = _body(result)
        assert body["metric"]["metadata"]["agent_ids"] == ["a1", "a2"]
        assert body["metric"]["metadata"]["forced"] is True

    @pytest.mark.asyncio
    async def test_calibrate_default_no_force(self, handler):
        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post("/api/v2/learning/calibrate", {}, http)
        body = _body(result)
        assert body["metric"]["metadata"]["forced"] is False


# ---------------------------------------------------------------------------
# Circuit Breaker Integration
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """Tests for circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_returns_503_on_get(self, handler):
        cb = handler._get_circuit_breaker()
        # Force the circuit open
        for _ in range(10):
            cb.record_failure()
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/sessions", {}, http)
        assert _status(result) == 503
        assert "temporarily unavailable" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_returns_503_on_post(self, handler):
        cb = handler._get_circuit_breaker()
        for _ in range(10):
            cb.record_failure()
        http = MockHTTPHandler(method="POST", body={"name": "Test"})
        result = await handler.handle_post("/api/v2/learning/sessions", {}, http)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_circuit_breaker_created_lazily(self, handler):
        assert handler._circuit_breaker is None
        handler._get_circuit_breaker()
        assert handler._circuit_breaker is not None

    @pytest.mark.asyncio
    async def test_successful_request_records_success(self, handler):
        http = MockHTTPHandler()
        await handler.handle("/api/v2/learning/sessions", {}, http)
        cb = handler._get_circuit_breaker()
        # Just verify no error - the CB should have success recorded internally


# ---------------------------------------------------------------------------
# handle() method routing
# ---------------------------------------------------------------------------


class TestHandleRouting:
    """Tests for the handle() GET routing."""

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_non_get(self, handler):
        """handle() only processes GET requests; POST should return None."""
        http = MockHTTPHandler(method="POST")
        result = await handler.handle("/api/v2/learning/sessions", {}, http)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_unknown_path(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/unknown_endpoint", {}, http)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_null_query_params(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v2/learning/sessions", None, http)
        body = _body(result)
        assert "sessions" in body


# ---------------------------------------------------------------------------
# handle_post() method routing
# ---------------------------------------------------------------------------


class TestHandlePostRouting:
    """Tests for the handle_post() POST routing."""

    @pytest.mark.asyncio
    async def test_post_returns_none_for_unknown_path(self, handler):
        http = MockHTTPHandler(method="POST", body={})
        result = await handler.handle_post("/api/v2/learning/unknown", {}, http)
        assert result is None

    @pytest.mark.asyncio
    async def test_post_session_path_too_short(self, handler):
        """A session path without session_id should return 400 or None."""
        http = MockHTTPHandler(method="POST", body={})
        # /api/v2/learning/sessions/ with empty session_id part -> parts[5] = ""
        # Then it looks for parts[6] == "stop" but parts has only 6 elements
        # so it falls through to pattern validation check
        result = await handler.handle_post("/api/v2/learning/sessions/", {}, http)
        # The path splits to ["", "api", "v2", "learning", "sessions", ""]
        # len(parts) >= 6, session_id = "", len(parts) == 6 so no parts[6]
        # Falls through -> returns None (no match for stop action)
        assert result is None

    @pytest.mark.asyncio
    async def test_post_null_query_params(self, handler):
        http = MockHTTPHandler(method="POST", body={"name": "Test"})
        result = await handler.handle_post("/api/v2/learning/sessions", None, http)
        assert _status(result) == 201


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error paths."""

    @pytest.mark.asyncio
    async def test_get_catches_value_error(self, handler):
        """ValueError during GET handling returns 500."""
        with patch.object(handler, "_list_sessions", side_effect=ValueError("test")):
            http = MockHTTPHandler()
            result = await handler.handle("/api/v2/learning/sessions", {}, http)
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_catches_runtime_error(self, handler):
        """RuntimeError during GET handling returns 500."""
        with patch.object(handler, "_get_performance", side_effect=RuntimeError("boom")):
            http = MockHTTPHandler()
            result = await handler.handle("/api/v2/learning/performance", {}, http)
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_post_catches_value_error(self, handler):
        """ValueError during POST handling returns 500."""
        with patch.object(handler, "_create_session", side_effect=ValueError("test")):
            http = MockHTTPHandler(method="POST", body={"name": "Test"})
            result = await handler.handle_post("/api/v2/learning/sessions", {}, http)
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_post_catches_type_error(self, handler):
        """TypeError during POST handling returns 500."""
        with patch.object(handler, "_submit_feedback", side_effect=TypeError("bad")):
            http = MockHTTPHandler(
                method="POST",
                body={"target_type": "session", "target_id": "s1"},
            )
            result = await handler.handle_post("/api/v2/learning/feedback", {}, http)
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# Dataclass Serialization
# ---------------------------------------------------------------------------


class TestDataclassSerialization:
    """Tests for dataclass to_dict() methods."""

    def test_training_session_to_dict(self):
        now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        session = TrainingSession(
            id="s1",
            name="Test",
            mode=LearningMode.SUPERVISED,
            status=SessionStatus.RUNNING,
            created_at=now,
            started_at=now,
            total_epochs=200,
            epochs_completed=50,
        )
        d = session.to_dict()
        assert d["id"] == "s1"
        assert d["mode"] == "supervised"
        assert d["status"] == "running"
        assert d["progress_percent"] == 25.0
        assert d["best_loss"] is None  # inf serialized as None

    def test_training_session_progress_zero_epochs(self):
        session = TrainingSession(
            id="s1",
            name="Test",
            mode=LearningMode.SUPERVISED,
            status=SessionStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            total_epochs=0,
        )
        assert session.progress_percent == 0.0

    def test_detected_pattern_to_dict(self):
        now = datetime(2026, 1, 15, tzinfo=timezone.utc)
        pattern = DetectedPattern(
            id="p1",
            pattern_type=PatternType.CROSS_DEBATE,
            confidence=0.87654,
            description="Cross-debate consensus",
            detected_at=now,
            is_validated=True,
            validated_by="admin",
            validated_at=now,
        )
        d = pattern.to_dict()
        assert d["pattern_type"] == "cross_debate"
        assert d["confidence"] == 0.8765
        assert d["is_validated"] is True
        assert d["validated_at"] is not None

    def test_learning_metric_to_dict(self):
        now = datetime(2026, 1, 15, tzinfo=timezone.utc)
        metric = LearningMetric(
            metric_type=MetricType.F1_SCORE,
            value=0.92345,
            timestamp=now,
            session_id="s1",
        )
        d = metric.to_dict()
        assert d["metric_type"] == "f1_score"
        assert d["value"] == 0.9234
        assert d["session_id"] == "s1"

    def test_extracted_knowledge_to_dict(self):
        now = datetime(2026, 1, 15, tzinfo=timezone.utc)
        knowledge = ExtractedKnowledge(
            id="k1",
            title="Test Knowledge",
            content="Content here",
            source_type="debate_analysis",
            source_debates=["d1"],
            confidence=0.87654,
            extracted_at=now,
            topics=["ai"],
        )
        d = knowledge.to_dict()
        assert d["confidence"] == 0.8765
        assert d["topics"] == ["ai"]

    def test_learning_feedback_to_dict(self):
        now = datetime(2026, 1, 15, tzinfo=timezone.utc)
        feedback = LearningFeedback(
            id="f1",
            feedback_type=FeedbackType.CORRECTION,
            target_type="session",
            target_id="s1",
            comment="Needs fixing",
            submitted_by="user1",
            submitted_at=now,
            rating=3,
        )
        d = feedback.to_dict()
        assert d["feedback_type"] == "correction"
        assert d["rating"] == 3


# ---------------------------------------------------------------------------
# Enum Coverage
# ---------------------------------------------------------------------------


class TestEnums:
    """Tests for enum value coverage."""

    def test_session_status_values(self):
        assert set(s.value for s in SessionStatus) == {
            "pending", "running", "paused", "completed", "failed", "cancelled"
        }

    def test_learning_mode_values(self):
        assert set(m.value for m in LearningMode) == {
            "supervised", "reinforcement", "self_supervised", "transfer", "federated"
        }

    def test_metric_type_values(self):
        assert set(t.value for t in MetricType) == {
            "accuracy", "loss", "precision", "recall", "f1_score",
            "calibration", "convergence"
        }

    def test_pattern_type_values(self):
        assert set(t.value for t in PatternType) == {
            "consensus", "disagreement", "agent_preference",
            "topic_cluster", "temporal", "cross_debate"
        }

    def test_feedback_type_values(self):
        assert set(t.value for t in FeedbackType) == {
            "positive", "negative", "neutral", "correction"
        }
