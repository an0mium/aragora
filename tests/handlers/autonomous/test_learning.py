"""Comprehensive tests for LearningHandler.

Tests cover:
- get_agent_ratings (GET /api/v1/autonomous/learning/ratings)
- get_agent_calibration (GET /api/v1/autonomous/learning/calibration/{agent_id})
- get_all_calibrations (GET /api/v1/autonomous/learning/calibrations)
- record_debate_outcome (POST /api/v1/autonomous/learning/debate)
- record_user_feedback (POST /api/v1/autonomous/learning/feedback)
- get_patterns (GET /api/v1/autonomous/learning/patterns)
- run_periodic_learning (POST /api/v1/autonomous/learning/run)
- Auth / permission checks (unauthorized, forbidden)
- Error-handling paths (KeyError, ValueError, TypeError, AttributeError, RuntimeError)
- Input validation (missing fields, empty values)
- Global accessors (get/set_continuous_learner)
- register_routes
- Handler init
- Edge cases (unicode, special chars, empty collections)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from aragora.server.handlers.autonomous.learning import (
    LearningHandler,
    get_continuous_learner,
    set_continuous_learner,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _parse(response: web.Response) -> dict:
    """Extract JSON dict from an aiohttp json_response."""
    return json.loads(response.body)


def _make_request(
    method: str = "GET",
    query: dict | None = None,
    match_info: dict | None = None,
    body: dict | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics an aiohttp web.Request."""
    req = MagicMock()
    req.method = method
    req.query = query or {}

    mi_data = match_info or {}
    mi_mock = MagicMock()
    mi_mock.get = MagicMock(side_effect=lambda k, default=None: mi_data.get(k, default))
    req.match_info = mi_mock

    if body is not None:
        req.json = AsyncMock(return_value=body)
        raw = json.dumps(body).encode()
        req.read = AsyncMock(return_value=raw)
        req.text = AsyncMock(return_value=json.dumps(body))
        req.content_type = "application/json"
        req.content_length = len(raw)
        req.can_read_body = True
    else:
        req.json = AsyncMock(return_value={})
        req.read = AsyncMock(return_value=b"{}")
        req.text = AsyncMock(return_value="{}")
        req.content_type = "application/json"
        req.content_length = 2
        req.can_read_body = True

    req.remote = "127.0.0.1"
    req.transport = MagicMock()
    req.transport.get_extra_info.return_value = ("127.0.0.1", 12345)

    return req


def _make_calibration(
    agent_id: str = "agent-1",
    elo_rating: float = 1600.0,
    confidence_accuracy: float = 0.75,
    topic_strengths: dict | None = None,
    topic_weaknesses: dict | None = None,
    last_updated: datetime | None = None,
    total_debates: int = 10,
    win_rate: float = 0.6,
) -> MagicMock:
    """Build a mock AgentCalibration object."""
    obj = MagicMock()
    obj.agent_id = agent_id
    obj.elo_rating = elo_rating
    obj.confidence_accuracy = confidence_accuracy
    obj.topic_strengths = topic_strengths if topic_strengths is not None else {"python": 0.9}
    obj.topic_weaknesses = topic_weaknesses if topic_weaknesses is not None else {"rust": 0.3}
    obj.last_updated = last_updated or datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    obj.total_debates = total_debates
    obj.win_rate = win_rate
    return obj


def _make_pattern(
    pattern_id: str = "pat-1",
    pattern_type: str = "consensus_strategy",
    description: str = "Agents converge faster with structured prompts",
    confidence: float = 0.85,
    evidence_count: int = 5,
    first_seen: datetime | None = None,
    last_seen: datetime | None = None,
    agents_involved: list | None = None,
    topics: list | None = None,
) -> MagicMock:
    """Build a mock ExtractedPattern object."""
    obj = MagicMock()
    obj.id = pattern_id
    obj.pattern_type = pattern_type
    obj.description = description
    obj.confidence = confidence
    obj.evidence_count = evidence_count
    obj.first_seen = first_seen or datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    obj.last_seen = last_seen or datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    obj.agents_involved = agents_involved if agents_involved is not None else ["claude", "gpt-4"]
    obj.topics = topics if topics is not None else ["architecture", "testing"]
    return obj


def _make_learning_event(
    event_id: str = "evt-1",
    event_type_value: str = "debate_completed",
    applied: bool = True,
) -> MagicMock:
    """Build a mock LearningEvent object."""
    obj = MagicMock()
    obj.id = event_id
    obj.event_type = MagicMock()
    obj.event_type.value = event_type_value
    obj.applied = applied
    return obj


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_learner_globals():
    """Reset the global continuous learner between tests."""
    import aragora.server.handlers.autonomous.learning as mod

    old_learner = mod._continuous_learner
    mod._continuous_learner = None
    yield
    mod._continuous_learner = old_learner


@pytest.fixture
def mock_learner():
    """Create a mock ContinuousLearner instance."""
    learner = MagicMock()
    # ELO updater
    learner.elo_updater = MagicMock()
    learner.elo_updater.get_all_ratings.return_value = {}
    learner.elo_updater.get_rating.return_value = 1500.0
    # Pattern extractor
    learner.pattern_extractor = MagicMock()
    learner.pattern_extractor.get_patterns.return_value = []
    # Calibration methods
    learner.get_calibration = MagicMock(return_value=None)
    learner.get_all_calibrations = MagicMock(return_value={})
    # Async methods
    learner.on_debate_completed = AsyncMock(return_value=_make_learning_event())
    learner.on_user_feedback = AsyncMock(return_value=_make_learning_event(event_type_value="user_feedback"))
    learner.run_periodic_learning = AsyncMock(return_value={"patterns_extracted": 0, "knowledge_decayed": 0})
    return learner


@pytest.fixture
def install_learner(mock_learner):
    """Set mock learner as the global singleton."""
    set_continuous_learner(mock_learner)
    return mock_learner


# ---------------------------------------------------------------------------
# get_agent_ratings endpoint
# ---------------------------------------------------------------------------


class TestGetAgentRatings:
    @pytest.mark.asyncio
    async def test_ratings_empty(self, install_learner):
        install_learner.elo_updater.get_all_ratings.return_value = {}
        req = _make_request()
        resp = await LearningHandler.get_agent_ratings(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["ratings"] == {}
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_ratings_single_agent(self, install_learner):
        install_learner.elo_updater.get_all_ratings.return_value = {"claude": 1650.0}
        req = _make_request()
        resp = await LearningHandler.get_agent_ratings(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["ratings"]["claude"] == 1650.0
        assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_ratings_multiple_agents(self, install_learner):
        ratings = {"claude": 1700.0, "gpt-4": 1550.0, "gemini": 1480.0}
        install_learner.elo_updater.get_all_ratings.return_value = ratings
        req = _make_request()
        resp = await LearningHandler.get_agent_ratings(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["count"] == 3
        assert data["ratings"]["claude"] == 1700.0
        assert data["ratings"]["gpt-4"] == 1550.0
        assert data["ratings"]["gemini"] == 1480.0

    @pytest.mark.asyncio
    async def test_ratings_response_keys(self, install_learner):
        install_learner.elo_updater.get_all_ratings.return_value = {"a": 1500.0}
        req = _make_request()
        resp = await LearningHandler.get_agent_ratings(req)

        data = await _parse(resp)
        assert set(data.keys()) == {"success", "ratings", "count"}

    @pytest.mark.asyncio
    async def test_ratings_runtime_error(self, install_learner):
        install_learner.elo_updater.get_all_ratings.side_effect = RuntimeError("db down")
        req = _make_request()
        resp = await LearningHandler.get_agent_ratings(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert data["success"] is False
        assert "Failed to retrieve ratings" in data["error"]

    @pytest.mark.asyncio
    async def test_ratings_key_error(self, install_learner):
        install_learner.elo_updater.get_all_ratings.side_effect = KeyError("missing")
        req = _make_request()
        resp = await LearningHandler.get_agent_ratings(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_ratings_value_error(self, install_learner):
        install_learner.elo_updater.get_all_ratings.side_effect = ValueError("invalid")
        req = _make_request()
        resp = await LearningHandler.get_agent_ratings(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_ratings_type_error(self, install_learner):
        install_learner.elo_updater.get_all_ratings.side_effect = TypeError("bad type")
        req = _make_request()
        resp = await LearningHandler.get_agent_ratings(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_ratings_attribute_error(self, install_learner):
        install_learner.elo_updater.get_all_ratings.side_effect = AttributeError("no attr")
        req = _make_request()
        resp = await LearningHandler.get_agent_ratings(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_ratings_unauthorized(self, install_learner):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.learning.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request()
            resp = await LearningHandler.get_agent_ratings(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_ratings_forbidden(self, install_learner):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.learning.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.learning.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request()
            resp = await LearningHandler.get_agent_ratings(req)

        assert resp.status == 403
        data = await _parse(resp)
        assert "Permission denied" in data["error"]


# ---------------------------------------------------------------------------
# get_agent_calibration endpoint
# ---------------------------------------------------------------------------


class TestGetAgentCalibration:
    @pytest.mark.asyncio
    async def test_calibration_found(self, install_learner):
        cal = _make_calibration(agent_id="claude")
        install_learner.get_calibration.return_value = cal

        req = _make_request(match_info={"agent_id": "claude"})
        resp = await LearningHandler.get_agent_calibration(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["calibration"]["agent_id"] == "claude"
        assert data["calibration"]["elo_rating"] == 1600.0
        assert data["calibration"]["confidence_accuracy"] == 0.75
        assert data["calibration"]["topic_strengths"] == {"python": 0.9}
        assert data["calibration"]["topic_weaknesses"] == {"rust": 0.3}
        assert data["calibration"]["total_debates"] == 10
        assert data["calibration"]["win_rate"] == 0.6
        assert data["calibration"]["last_updated"] == "2026-02-01T12:00:00+00:00"

    @pytest.mark.asyncio
    async def test_calibration_not_found(self, install_learner):
        install_learner.get_calibration.return_value = None

        req = _make_request(match_info={"agent_id": "unknown-agent"})
        resp = await LearningHandler.get_agent_calibration(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["calibration"] is None
        assert "No calibration data" in data["message"]

    @pytest.mark.asyncio
    async def test_calibration_null_last_updated(self, install_learner):
        cal = _make_calibration(agent_id="new-agent")
        cal.last_updated = None
        install_learner.get_calibration.return_value = cal

        req = _make_request(match_info={"agent_id": "new-agent"})
        resp = await LearningHandler.get_agent_calibration(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["calibration"]["last_updated"] is None

    @pytest.mark.asyncio
    async def test_calibration_response_keys(self, install_learner):
        cal = _make_calibration()
        install_learner.get_calibration.return_value = cal

        req = _make_request(match_info={"agent_id": "agent-1"})
        resp = await LearningHandler.get_agent_calibration(req)

        data = await _parse(resp)
        expected_keys = {
            "agent_id", "elo_rating", "confidence_accuracy",
            "topic_strengths", "topic_weaknesses", "last_updated",
            "total_debates", "win_rate",
        }
        assert expected_keys == set(data["calibration"].keys())

    @pytest.mark.asyncio
    async def test_calibration_empty_topic_strengths(self, install_learner):
        cal = _make_calibration(topic_strengths={}, topic_weaknesses={})
        install_learner.get_calibration.return_value = cal

        req = _make_request(match_info={"agent_id": "agent-1"})
        resp = await LearningHandler.get_agent_calibration(req)

        data = await _parse(resp)
        assert data["calibration"]["topic_strengths"] == {}
        assert data["calibration"]["topic_weaknesses"] == {}

    @pytest.mark.asyncio
    async def test_calibration_runtime_error(self, install_learner):
        install_learner.get_calibration.side_effect = RuntimeError("db down")

        req = _make_request(match_info={"agent_id": "agent-1"})
        resp = await LearningHandler.get_agent_calibration(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert data["success"] is False
        assert "Failed to retrieve calibration data" in data["error"]

    @pytest.mark.asyncio
    async def test_calibration_key_error(self, install_learner):
        install_learner.get_calibration.side_effect = KeyError("missing")

        req = _make_request(match_info={"agent_id": "agent-1"})
        resp = await LearningHandler.get_agent_calibration(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_calibration_value_error(self, install_learner):
        install_learner.get_calibration.side_effect = ValueError("invalid")

        req = _make_request(match_info={"agent_id": "agent-1"})
        resp = await LearningHandler.get_agent_calibration(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_calibration_type_error(self, install_learner):
        install_learner.get_calibration.side_effect = TypeError("bad type")

        req = _make_request(match_info={"agent_id": "agent-1"})
        resp = await LearningHandler.get_agent_calibration(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_calibration_attribute_error(self, install_learner):
        install_learner.get_calibration.side_effect = AttributeError("no attr")

        req = _make_request(match_info={"agent_id": "agent-1"})
        resp = await LearningHandler.get_agent_calibration(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_calibration_unauthorized(self, install_learner):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.learning.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(match_info={"agent_id": "agent-1"})
            resp = await LearningHandler.get_agent_calibration(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_calibration_forbidden(self, install_learner):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.learning.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.learning.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(match_info={"agent_id": "agent-1"})
            resp = await LearningHandler.get_agent_calibration(req)

        assert resp.status == 403
        data = await _parse(resp)
        assert "Permission denied" in data["error"]


# ---------------------------------------------------------------------------
# get_all_calibrations endpoint
# ---------------------------------------------------------------------------


class TestGetAllCalibrations:
    @pytest.mark.asyncio
    async def test_all_calibrations_empty(self, install_learner):
        install_learner.get_all_calibrations.return_value = {}

        req = _make_request()
        resp = await LearningHandler.get_all_calibrations(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["calibrations"] == {}
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_all_calibrations_single(self, install_learner):
        cal = _make_calibration(agent_id="claude", elo_rating=1700.0, total_debates=20, win_rate=0.7)
        install_learner.get_all_calibrations.return_value = {"claude": cal}

        req = _make_request()
        resp = await LearningHandler.get_all_calibrations(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["count"] == 1
        assert data["calibrations"]["claude"]["elo_rating"] == 1700.0
        assert data["calibrations"]["claude"]["total_debates"] == 20
        assert data["calibrations"]["claude"]["win_rate"] == 0.7
        assert data["calibrations"]["claude"]["last_updated"] == "2026-02-01T12:00:00+00:00"

    @pytest.mark.asyncio
    async def test_all_calibrations_multiple(self, install_learner):
        cal1 = _make_calibration(agent_id="claude", elo_rating=1700.0)
        cal2 = _make_calibration(agent_id="gpt-4", elo_rating=1550.0)
        install_learner.get_all_calibrations.return_value = {"claude": cal1, "gpt-4": cal2}

        req = _make_request()
        resp = await LearningHandler.get_all_calibrations(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["count"] == 2
        assert "claude" in data["calibrations"]
        assert "gpt-4" in data["calibrations"]

    @pytest.mark.asyncio
    async def test_all_calibrations_null_last_updated(self, install_learner):
        cal = _make_calibration(agent_id="new-agent")
        cal.last_updated = None
        install_learner.get_all_calibrations.return_value = {"new-agent": cal}

        req = _make_request()
        resp = await LearningHandler.get_all_calibrations(req)

        data = await _parse(resp)
        assert data["calibrations"]["new-agent"]["last_updated"] is None

    @pytest.mark.asyncio
    async def test_all_calibrations_response_keys_per_agent(self, install_learner):
        cal = _make_calibration(agent_id="agent-1")
        install_learner.get_all_calibrations.return_value = {"agent-1": cal}

        req = _make_request()
        resp = await LearningHandler.get_all_calibrations(req)

        data = await _parse(resp)
        expected_keys = {"elo_rating", "total_debates", "win_rate", "last_updated"}
        assert expected_keys == set(data["calibrations"]["agent-1"].keys())

    @pytest.mark.asyncio
    async def test_all_calibrations_runtime_error(self, install_learner):
        install_learner.get_all_calibrations.side_effect = RuntimeError("db down")

        req = _make_request()
        resp = await LearningHandler.get_all_calibrations(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert data["success"] is False
        assert "Failed to retrieve calibrations" in data["error"]

    @pytest.mark.asyncio
    async def test_all_calibrations_key_error(self, install_learner):
        install_learner.get_all_calibrations.side_effect = KeyError("missing")

        req = _make_request()
        resp = await LearningHandler.get_all_calibrations(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_all_calibrations_value_error(self, install_learner):
        install_learner.get_all_calibrations.side_effect = ValueError("invalid")

        req = _make_request()
        resp = await LearningHandler.get_all_calibrations(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_all_calibrations_type_error(self, install_learner):
        install_learner.get_all_calibrations.side_effect = TypeError("bad type")

        req = _make_request()
        resp = await LearningHandler.get_all_calibrations(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_all_calibrations_attribute_error(self, install_learner):
        install_learner.get_all_calibrations.side_effect = AttributeError("no attr")

        req = _make_request()
        resp = await LearningHandler.get_all_calibrations(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_all_calibrations_unauthorized(self, install_learner):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.learning.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request()
            resp = await LearningHandler.get_all_calibrations(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_all_calibrations_forbidden(self, install_learner):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.learning.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.learning.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request()
            resp = await LearningHandler.get_all_calibrations(req)

        assert resp.status == 403
        data = await _parse(resp)
        assert "Permission denied" in data["error"]


# ---------------------------------------------------------------------------
# record_debate_outcome endpoint
# ---------------------------------------------------------------------------


class TestRecordDebateOutcome:
    @pytest.mark.asyncio
    async def test_debate_outcome_success(self, install_learner):
        event = _make_learning_event(event_id="evt-debate-1", event_type_value="debate_completed")
        install_learner.on_debate_completed.return_value = event
        install_learner.elo_updater.get_rating.side_effect = lambda a: {"claude": 1650.0, "gpt-4": 1550.0}.get(a, 1500.0)

        req = _make_request(
            method="POST",
            body={
                "debate_id": "debate-42",
                "agents": ["claude", "gpt-4"],
                "winner": "claude",
                "votes": {"claude": 5, "gpt-4": 3},
                "consensus_reached": True,
                "topics": ["architecture"],
            },
        )
        resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["event"]["id"] == "evt-debate-1"
        assert data["event"]["event_type"] == "debate_completed"
        assert data["event"]["applied"] is True
        assert data["updated_ratings"]["claude"] == 1650.0
        assert data["updated_ratings"]["gpt-4"] == 1550.0

    @pytest.mark.asyncio
    async def test_debate_outcome_no_winner(self, install_learner):
        event = _make_learning_event()
        install_learner.on_debate_completed.return_value = event

        req = _make_request(
            method="POST",
            body={
                "debate_id": "debate-43",
                "agents": ["claude", "gpt-4"],
                "consensus_reached": False,
                "topics": ["testing"],
            },
        )
        resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        # Verify winner was passed as None
        call_kwargs = install_learner.on_debate_completed.call_args[1]
        assert call_kwargs["winner"] is None

    @pytest.mark.asyncio
    async def test_debate_outcome_with_metadata(self, install_learner):
        event = _make_learning_event()
        install_learner.on_debate_completed.return_value = event

        req = _make_request(
            method="POST",
            body={
                "debate_id": "debate-44",
                "agents": ["claude"],
                "topics": [],
                "metadata": {"source": "api", "round_count": 3},
            },
        )
        resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 200
        call_kwargs = install_learner.on_debate_completed.call_args[1]
        assert call_kwargs["metadata"] == {"source": "api", "round_count": 3}

    @pytest.mark.asyncio
    async def test_debate_outcome_missing_debate_id(self, install_learner):
        req = _make_request(
            method="POST",
            body={
                "agents": ["claude", "gpt-4"],
            },
        )
        resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False
        assert "debate_id" in data["error"]
        assert "agents" in data["error"]

    @pytest.mark.asyncio
    async def test_debate_outcome_missing_agents(self, install_learner):
        req = _make_request(
            method="POST",
            body={
                "debate_id": "debate-45",
            },
        )
        resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_debate_outcome_empty_agents(self, install_learner):
        req = _make_request(
            method="POST",
            body={
                "debate_id": "debate-46",
                "agents": [],
            },
        )
        resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_debate_outcome_empty_debate_id(self, install_learner):
        req = _make_request(
            method="POST",
            body={
                "debate_id": "",
                "agents": ["claude"],
            },
        )
        resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_debate_outcome_null_debate_id(self, install_learner):
        req = _make_request(
            method="POST",
            body={
                "debate_id": None,
                "agents": ["claude"],
            },
        )
        resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_debate_outcome_defaults(self, install_learner):
        """Verify default values for optional fields."""
        event = _make_learning_event()
        install_learner.on_debate_completed.return_value = event

        req = _make_request(
            method="POST",
            body={
                "debate_id": "debate-47",
                "agents": ["claude"],
            },
        )
        resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 200
        call_kwargs = install_learner.on_debate_completed.call_args[1]
        assert call_kwargs["winner"] is None
        assert call_kwargs["votes"] == {}
        assert call_kwargs["consensus_reached"] is False
        assert call_kwargs["topics"] == []
        assert call_kwargs["metadata"] is None

    @pytest.mark.asyncio
    async def test_debate_outcome_runtime_error(self, install_learner):
        install_learner.on_debate_completed.side_effect = RuntimeError("db down")

        req = _make_request(
            method="POST",
            body={
                "debate_id": "debate-48",
                "agents": ["claude"],
            },
        )
        resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert data["success"] is False
        assert "Failed to record debate outcome" in data["error"]

    @pytest.mark.asyncio
    async def test_debate_outcome_key_error(self, install_learner):
        install_learner.on_debate_completed.side_effect = KeyError("missing")

        req = _make_request(
            method="POST",
            body={"debate_id": "d", "agents": ["a"]},
        )
        resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_debate_outcome_value_error(self, install_learner):
        install_learner.on_debate_completed.side_effect = ValueError("invalid")

        req = _make_request(
            method="POST",
            body={"debate_id": "d", "agents": ["a"]},
        )
        resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_debate_outcome_type_error(self, install_learner):
        install_learner.on_debate_completed.side_effect = TypeError("bad type")

        req = _make_request(
            method="POST",
            body={"debate_id": "d", "agents": ["a"]},
        )
        resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_debate_outcome_attribute_error(self, install_learner):
        install_learner.on_debate_completed.side_effect = AttributeError("no attr")

        req = _make_request(
            method="POST",
            body={"debate_id": "d", "agents": ["a"]},
        )
        resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_debate_outcome_invalid_json(self, install_learner):
        req = MagicMock()
        req.method = "POST"
        req.json = AsyncMock(side_effect=ValueError("bad json"))
        req.read = AsyncMock(return_value=b"not json")
        req.text = AsyncMock(return_value="not json")
        req.content_type = "application/json"
        req.content_length = 8
        req.can_read_body = True
        req.remote = "127.0.0.1"
        req.transport = MagicMock()
        req.transport.get_extra_info.return_value = ("127.0.0.1", 12345)

        resp = await LearningHandler.record_debate_outcome(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_debate_outcome_unauthorized(self, install_learner):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.learning.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(
                method="POST",
                body={"debate_id": "d", "agents": ["a"]},
            )
            resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_debate_outcome_forbidden(self, install_learner):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.learning.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.learning.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(
                method="POST",
                body={"debate_id": "d", "agents": ["a"]},
            )
            resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 403
        data = await _parse(resp)
        assert "Permission denied" in data["error"]


# ---------------------------------------------------------------------------
# record_user_feedback endpoint
# ---------------------------------------------------------------------------


class TestRecordUserFeedback:
    @pytest.mark.asyncio
    async def test_feedback_success(self, install_learner):
        event = _make_learning_event(event_id="evt-fb-1", event_type_value="user_feedback")
        install_learner.on_user_feedback.return_value = event

        req = _make_request(
            method="POST",
            body={
                "debate_id": "debate-50",
                "agent_id": "claude",
                "feedback_type": "helpful",
                "score": 0.8,
            },
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["event"]["id"] == "evt-fb-1"
        assert data["event"]["event_type"] == "user_feedback"
        assert data["event"]["applied"] is True

    @pytest.mark.asyncio
    async def test_feedback_with_metadata(self, install_learner):
        event = _make_learning_event()
        install_learner.on_user_feedback.return_value = event

        req = _make_request(
            method="POST",
            body={
                "debate_id": "debate-51",
                "agent_id": "claude",
                "feedback_type": "accurate",
                "score": 1.0,
                "metadata": {"user": "admin"},
            },
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 200
        call_kwargs = install_learner.on_user_feedback.call_args[1]
        assert call_kwargs["metadata"] == {"user": "admin"}

    @pytest.mark.asyncio
    async def test_feedback_default_score(self, install_learner):
        event = _make_learning_event()
        install_learner.on_user_feedback.return_value = event

        req = _make_request(
            method="POST",
            body={
                "debate_id": "debate-52",
                "agent_id": "claude",
                "feedback_type": "helpful",
            },
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 200
        call_kwargs = install_learner.on_user_feedback.call_args[1]
        assert call_kwargs["score"] == 0.0

    @pytest.mark.asyncio
    async def test_feedback_negative_score(self, install_learner):
        event = _make_learning_event()
        install_learner.on_user_feedback.return_value = event

        req = _make_request(
            method="POST",
            body={
                "debate_id": "debate-53",
                "agent_id": "claude",
                "feedback_type": "unhelpful",
                "score": -0.5,
            },
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 200
        call_kwargs = install_learner.on_user_feedback.call_args[1]
        assert call_kwargs["score"] == -0.5

    @pytest.mark.asyncio
    async def test_feedback_score_as_int(self, install_learner):
        """Integer score should be converted to float."""
        event = _make_learning_event()
        install_learner.on_user_feedback.return_value = event

        req = _make_request(
            method="POST",
            body={
                "debate_id": "debate-54",
                "agent_id": "claude",
                "feedback_type": "helpful",
                "score": 1,
            },
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 200
        call_kwargs = install_learner.on_user_feedback.call_args[1]
        assert isinstance(call_kwargs["score"], float)
        assert call_kwargs["score"] == 1.0

    @pytest.mark.asyncio
    async def test_feedback_missing_debate_id(self, install_learner):
        req = _make_request(
            method="POST",
            body={
                "agent_id": "claude",
                "feedback_type": "helpful",
            },
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False
        assert "debate_id" in data["error"]

    @pytest.mark.asyncio
    async def test_feedback_missing_agent_id(self, install_learner):
        req = _make_request(
            method="POST",
            body={
                "debate_id": "debate-55",
                "feedback_type": "helpful",
            },
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False
        assert "agent_id" in data["error"]

    @pytest.mark.asyncio
    async def test_feedback_missing_feedback_type(self, install_learner):
        req = _make_request(
            method="POST",
            body={
                "debate_id": "debate-56",
                "agent_id": "claude",
            },
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False
        assert "feedback_type" in data["error"]

    @pytest.mark.asyncio
    async def test_feedback_missing_all_required(self, install_learner):
        req = _make_request(
            method="POST",
            body={},
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_feedback_empty_debate_id(self, install_learner):
        req = _make_request(
            method="POST",
            body={
                "debate_id": "",
                "agent_id": "claude",
                "feedback_type": "helpful",
            },
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_feedback_empty_agent_id(self, install_learner):
        req = _make_request(
            method="POST",
            body={
                "debate_id": "debate-57",
                "agent_id": "",
                "feedback_type": "helpful",
            },
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_feedback_empty_feedback_type(self, install_learner):
        req = _make_request(
            method="POST",
            body={
                "debate_id": "debate-58",
                "agent_id": "claude",
                "feedback_type": "",
            },
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_feedback_runtime_error(self, install_learner):
        install_learner.on_user_feedback.side_effect = RuntimeError("db down")

        req = _make_request(
            method="POST",
            body={
                "debate_id": "d",
                "agent_id": "a",
                "feedback_type": "helpful",
            },
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert data["success"] is False
        assert "Failed to record feedback" in data["error"]

    @pytest.mark.asyncio
    async def test_feedback_key_error(self, install_learner):
        install_learner.on_user_feedback.side_effect = KeyError("missing")

        req = _make_request(
            method="POST",
            body={"debate_id": "d", "agent_id": "a", "feedback_type": "helpful"},
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_feedback_value_error(self, install_learner):
        install_learner.on_user_feedback.side_effect = ValueError("invalid")

        req = _make_request(
            method="POST",
            body={"debate_id": "d", "agent_id": "a", "feedback_type": "helpful"},
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_feedback_type_error(self, install_learner):
        install_learner.on_user_feedback.side_effect = TypeError("bad type")

        req = _make_request(
            method="POST",
            body={"debate_id": "d", "agent_id": "a", "feedback_type": "helpful"},
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_feedback_attribute_error(self, install_learner):
        install_learner.on_user_feedback.side_effect = AttributeError("no attr")

        req = _make_request(
            method="POST",
            body={"debate_id": "d", "agent_id": "a", "feedback_type": "helpful"},
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_feedback_invalid_json(self, install_learner):
        req = MagicMock()
        req.method = "POST"
        req.json = AsyncMock(side_effect=ValueError("bad json"))
        req.read = AsyncMock(return_value=b"not json")
        req.text = AsyncMock(return_value="not json")
        req.content_type = "application/json"
        req.content_length = 8
        req.can_read_body = True
        req.remote = "127.0.0.1"
        req.transport = MagicMock()
        req.transport.get_extra_info.return_value = ("127.0.0.1", 12345)

        resp = await LearningHandler.record_user_feedback(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_feedback_unauthorized(self, install_learner):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.learning.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(
                method="POST",
                body={"debate_id": "d", "agent_id": "a", "feedback_type": "helpful"},
            )
            resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_feedback_forbidden(self, install_learner):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.learning.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.learning.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(
                method="POST",
                body={"debate_id": "d", "agent_id": "a", "feedback_type": "helpful"},
            )
            resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 403
        data = await _parse(resp)
        assert "Permission denied" in data["error"]


# ---------------------------------------------------------------------------
# get_patterns endpoint
# ---------------------------------------------------------------------------


class TestGetPatterns:
    @pytest.mark.asyncio
    async def test_patterns_empty(self, install_learner):
        install_learner.pattern_extractor.get_patterns.return_value = []

        req = _make_request()
        resp = await LearningHandler.get_patterns(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["patterns"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_patterns_single(self, install_learner):
        pat = _make_pattern()
        install_learner.pattern_extractor.get_patterns.return_value = [pat]

        req = _make_request()
        resp = await LearningHandler.get_patterns(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["count"] == 1
        p = data["patterns"][0]
        assert p["id"] == "pat-1"
        assert p["pattern_type"] == "consensus_strategy"
        assert p["description"] == "Agents converge faster with structured prompts"
        assert p["confidence"] == 0.85
        assert p["evidence_count"] == 5
        assert p["first_seen"] == "2026-01-15T10:00:00+00:00"
        assert p["last_seen"] == "2026-02-01T12:00:00+00:00"
        assert p["agents_involved"] == ["claude", "gpt-4"]
        assert p["topics"] == ["architecture", "testing"]

    @pytest.mark.asyncio
    async def test_patterns_multiple(self, install_learner):
        pats = [
            _make_pattern(pattern_id="p-1", pattern_type="consensus_strategy"),
            _make_pattern(pattern_id="p-2", pattern_type="failure_mode"),
            _make_pattern(pattern_id="p-3", pattern_type="topic_expertise"),
        ]
        install_learner.pattern_extractor.get_patterns.return_value = pats

        req = _make_request()
        resp = await LearningHandler.get_patterns(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["count"] == 3
        ids = [p["id"] for p in data["patterns"]]
        assert ids == ["p-1", "p-2", "p-3"]

    @pytest.mark.asyncio
    async def test_patterns_filter_by_type(self, install_learner):
        pat = _make_pattern(pattern_type="failure_mode")
        install_learner.pattern_extractor.get_patterns.return_value = [pat]

        req = _make_request(query={"pattern_type": "failure_mode"})
        resp = await LearningHandler.get_patterns(req)

        assert resp.status == 200
        install_learner.pattern_extractor.get_patterns.assert_called_once_with("failure_mode")

    @pytest.mark.asyncio
    async def test_patterns_no_filter(self, install_learner):
        install_learner.pattern_extractor.get_patterns.return_value = []

        req = _make_request()
        resp = await LearningHandler.get_patterns(req)

        assert resp.status == 200
        install_learner.pattern_extractor.get_patterns.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_patterns_response_keys(self, install_learner):
        pat = _make_pattern()
        install_learner.pattern_extractor.get_patterns.return_value = [pat]

        req = _make_request()
        resp = await LearningHandler.get_patterns(req)

        data = await _parse(resp)
        expected_keys = {
            "id", "pattern_type", "description", "confidence",
            "evidence_count", "first_seen", "last_seen",
            "agents_involved", "topics",
        }
        assert expected_keys == set(data["patterns"][0].keys())

    @pytest.mark.asyncio
    async def test_patterns_empty_agents_and_topics(self, install_learner):
        pat = _make_pattern(agents_involved=[], topics=[])
        install_learner.pattern_extractor.get_patterns.return_value = [pat]

        req = _make_request()
        resp = await LearningHandler.get_patterns(req)

        data = await _parse(resp)
        assert data["patterns"][0]["agents_involved"] == []
        assert data["patterns"][0]["topics"] == []

    @pytest.mark.asyncio
    async def test_patterns_runtime_error(self, install_learner):
        install_learner.pattern_extractor.get_patterns.side_effect = RuntimeError("db down")

        req = _make_request()
        resp = await LearningHandler.get_patterns(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert data["success"] is False
        assert "Failed to retrieve patterns" in data["error"]

    @pytest.mark.asyncio
    async def test_patterns_key_error(self, install_learner):
        install_learner.pattern_extractor.get_patterns.side_effect = KeyError("missing")

        req = _make_request()
        resp = await LearningHandler.get_patterns(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_patterns_value_error(self, install_learner):
        install_learner.pattern_extractor.get_patterns.side_effect = ValueError("invalid")

        req = _make_request()
        resp = await LearningHandler.get_patterns(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_patterns_type_error(self, install_learner):
        install_learner.pattern_extractor.get_patterns.side_effect = TypeError("bad type")

        req = _make_request()
        resp = await LearningHandler.get_patterns(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_patterns_attribute_error(self, install_learner):
        install_learner.pattern_extractor.get_patterns.side_effect = AttributeError("no attr")

        req = _make_request()
        resp = await LearningHandler.get_patterns(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_patterns_unauthorized(self, install_learner):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.learning.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request()
            resp = await LearningHandler.get_patterns(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_patterns_forbidden(self, install_learner):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.learning.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.learning.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request()
            resp = await LearningHandler.get_patterns(req)

        assert resp.status == 403
        data = await _parse(resp)
        assert "Permission denied" in data["error"]


# ---------------------------------------------------------------------------
# run_periodic_learning endpoint
# ---------------------------------------------------------------------------


class TestRunPeriodicLearning:
    @pytest.mark.asyncio
    async def test_periodic_learning_success(self, install_learner):
        summary = {
            "patterns_extracted": 3,
            "knowledge_decayed": 1,
            "ratings_decayed": 2,
        }
        install_learner.run_periodic_learning.return_value = summary

        req = _make_request(method="POST")
        resp = await LearningHandler.run_periodic_learning(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["summary"]["patterns_extracted"] == 3
        assert data["summary"]["knowledge_decayed"] == 1
        assert data["summary"]["ratings_decayed"] == 2

    @pytest.mark.asyncio
    async def test_periodic_learning_empty_summary(self, install_learner):
        install_learner.run_periodic_learning.return_value = {}

        req = _make_request(method="POST")
        resp = await LearningHandler.run_periodic_learning(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["summary"] == {}

    @pytest.mark.asyncio
    async def test_periodic_learning_runtime_error(self, install_learner):
        install_learner.run_periodic_learning.side_effect = RuntimeError("task failed")

        req = _make_request(method="POST")
        resp = await LearningHandler.run_periodic_learning(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert data["success"] is False
        assert "Periodic learning run failed" in data["error"]

    @pytest.mark.asyncio
    async def test_periodic_learning_key_error(self, install_learner):
        install_learner.run_periodic_learning.side_effect = KeyError("missing")

        req = _make_request(method="POST")
        resp = await LearningHandler.run_periodic_learning(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_periodic_learning_value_error(self, install_learner):
        install_learner.run_periodic_learning.side_effect = ValueError("invalid")

        req = _make_request(method="POST")
        resp = await LearningHandler.run_periodic_learning(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_periodic_learning_type_error(self, install_learner):
        install_learner.run_periodic_learning.side_effect = TypeError("bad type")

        req = _make_request(method="POST")
        resp = await LearningHandler.run_periodic_learning(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_periodic_learning_attribute_error(self, install_learner):
        install_learner.run_periodic_learning.side_effect = AttributeError("no attr")

        req = _make_request(method="POST")
        resp = await LearningHandler.run_periodic_learning(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_periodic_learning_unauthorized(self, install_learner):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.learning.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(method="POST")
            resp = await LearningHandler.run_periodic_learning(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_periodic_learning_forbidden(self, install_learner):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.learning.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.learning.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(method="POST")
            resp = await LearningHandler.run_periodic_learning(req)

        assert resp.status == 403
        data = await _parse(resp)
        assert "Permission denied" in data["error"]


# ---------------------------------------------------------------------------
# register_routes
# ---------------------------------------------------------------------------


class TestRegisterRoutes:
    def test_register_routes_default_prefix(self):
        app = web.Application()
        LearningHandler.register_routes(app)

        route_paths = [
            r.resource.canonical
            for r in app.router.routes()
            if hasattr(r, "resource") and r.resource
        ]

        assert any("/learning/ratings" in p for p in route_paths)
        assert any("agent_id" in p and "/learning/calibration/" in p for p in route_paths)
        assert any("/learning/calibrations" in p for p in route_paths)
        assert any("/learning/debate" in p for p in route_paths)
        assert any("/learning/feedback" in p for p in route_paths)
        assert any("/learning/patterns" in p for p in route_paths)
        assert any("/learning/run" in p for p in route_paths)

    def test_register_routes_custom_prefix(self):
        app = web.Application()
        LearningHandler.register_routes(app, prefix="/custom/api")

        route_paths = [
            r.resource.canonical
            for r in app.router.routes()
            if hasattr(r, "resource") and r.resource
        ]

        assert any("/custom/api/learning/ratings" in p for p in route_paths)
        assert any("/custom/api/learning/debate" in p for p in route_paths)
        assert any("/custom/api/learning/feedback" in p for p in route_paths)
        assert any("/custom/api/learning/run" in p for p in route_paths)

    def test_register_routes_count(self):
        app = web.Application()
        LearningHandler.register_routes(app)

        route_count = sum(
            1
            for r in app.router.routes()
            if hasattr(r, "resource") and r.resource
        )
        # 4 GET (ratings, calibration/{id}, calibrations, patterns)
        # 3 POST (debate, feedback, run)
        # 4 HEAD auto-added for GETs
        # Total: 4 + 3 + 4 = 11
        assert route_count == 11

    def test_register_routes_methods(self):
        app = web.Application()
        LearningHandler.register_routes(app)

        method_map: dict[str, list[str]] = {}
        for r in app.router.routes():
            if hasattr(r, "resource") and r.resource:
                key = r.resource.canonical
                if key not in method_map:
                    method_map[key] = []
                method_map[key].append(r.method)

        for canonical, methods in method_map.items():
            if canonical.endswith("/learning/ratings"):
                assert "GET" in methods
            elif canonical.endswith("/learning/debate"):
                assert "POST" in methods
            elif canonical.endswith("/learning/feedback"):
                assert "POST" in methods
            elif canonical.endswith("/learning/run"):
                assert "POST" in methods
            elif canonical.endswith("/learning/patterns"):
                assert "GET" in methods
            elif canonical.endswith("/learning/calibrations"):
                assert "GET" in methods


# ---------------------------------------------------------------------------
# Handler init and global accessors
# ---------------------------------------------------------------------------


class TestHandlerInit:
    def test_handler_init_default(self):
        handler = LearningHandler()
        assert handler.ctx == {}

    def test_handler_init_custom_ctx(self):
        handler = LearningHandler(ctx={"env": "test"})
        assert handler.ctx == {"env": "test"}

    def test_handler_init_none_ctx(self):
        handler = LearningHandler(ctx=None)
        assert handler.ctx == {}


class TestGlobalAccessors:
    def test_get_continuous_learner_creates_instance(self):
        learner = get_continuous_learner()
        assert learner is not None

    def test_get_continuous_learner_singleton(self):
        l1 = get_continuous_learner()
        l2 = get_continuous_learner()
        assert l1 is l2

    def test_set_and_get_continuous_learner(self):
        custom = MagicMock()
        set_continuous_learner(custom)
        assert get_continuous_learner() is custom

    def test_set_continuous_learner_replaces(self):
        first = MagicMock()
        second = MagicMock()
        set_continuous_learner(first)
        set_continuous_learner(second)
        assert get_continuous_learner() is second


# ---------------------------------------------------------------------------
# Security edge cases
# ---------------------------------------------------------------------------


class TestSecurityEdgeCases:
    @pytest.mark.asyncio
    async def test_path_traversal_in_agent_id(self, install_learner):
        install_learner.get_calibration.return_value = None

        req = _make_request(match_info={"agent_id": "../../../etc/passwd"})
        resp = await LearningHandler.get_agent_calibration(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["calibration"] is None

    @pytest.mark.asyncio
    async def test_sql_injection_in_agent_id(self, install_learner):
        install_learner.get_calibration.return_value = None

        req = _make_request(match_info={"agent_id": "'; DROP TABLE agents; --"})
        resp = await LearningHandler.get_agent_calibration(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["calibration"] is None

    @pytest.mark.asyncio
    async def test_xss_in_debate_id(self, install_learner):
        """XSS payload in debate_id should be passed through safely."""
        event = _make_learning_event()
        install_learner.on_debate_completed.return_value = event

        req = _make_request(
            method="POST",
            body={
                "debate_id": "<script>alert('xss')</script>",
                "agents": ["claude"],
            },
        )
        resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 200
        call_kwargs = install_learner.on_debate_completed.call_args[1]
        assert call_kwargs["debate_id"] == "<script>alert('xss')</script>"

    @pytest.mark.asyncio
    async def test_very_long_agent_id(self, install_learner):
        install_learner.get_calibration.return_value = None
        long_id = "a" * 10000

        req = _make_request(match_info={"agent_id": long_id})
        resp = await LearningHandler.get_agent_calibration(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["calibration"] is None

    @pytest.mark.asyncio
    async def test_unicode_in_feedback_type(self, install_learner):
        event = _make_learning_event()
        install_learner.on_user_feedback.return_value = event

        req = _make_request(
            method="POST",
            body={
                "debate_id": "debate-99",
                "agent_id": "claude",
                "feedback_type": "hilfreich",
            },
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 200
        call_kwargs = install_learner.on_user_feedback.call_args[1]
        assert call_kwargs["feedback_type"] == "hilfreich"

    @pytest.mark.asyncio
    async def test_special_chars_in_topics(self, install_learner):
        event = _make_learning_event()
        install_learner.on_debate_completed.return_value = event

        req = _make_request(
            method="POST",
            body={
                "debate_id": "debate-100",
                "agents": ["claude"],
                "topics": ["C++/C#", "node.js@18", "rust-lang/2024"],
            },
        )
        resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 200
        call_kwargs = install_learner.on_debate_completed.call_args[1]
        assert call_kwargs["topics"] == ["C++/C#", "node.js@18", "rust-lang/2024"]


# ---------------------------------------------------------------------------
# Integration / cross-endpoint edge cases
# ---------------------------------------------------------------------------


class TestIntegrationEdgeCases:
    @pytest.mark.asyncio
    async def test_ratings_uses_get_continuous_learner(self):
        mock_l = MagicMock()
        mock_l.elo_updater = MagicMock()
        mock_l.elo_updater.get_all_ratings.return_value = {"x": 1500.0}
        set_continuous_learner(mock_l)

        req = _make_request()
        resp = await LearningHandler.get_agent_ratings(req)

        assert resp.status == 200
        mock_l.elo_updater.get_all_ratings.assert_called_once()

    @pytest.mark.asyncio
    async def test_calibration_uses_get_continuous_learner(self):
        mock_l = MagicMock()
        mock_l.get_calibration.return_value = None
        set_continuous_learner(mock_l)

        req = _make_request(match_info={"agent_id": "agent-1"})
        resp = await LearningHandler.get_agent_calibration(req)

        assert resp.status == 200
        mock_l.get_calibration.assert_called_once_with("agent-1")

    @pytest.mark.asyncio
    async def test_all_calibrations_uses_get_continuous_learner(self):
        mock_l = MagicMock()
        mock_l.get_all_calibrations.return_value = {}
        set_continuous_learner(mock_l)

        req = _make_request()
        resp = await LearningHandler.get_all_calibrations(req)

        assert resp.status == 200
        mock_l.get_all_calibrations.assert_called_once()

    @pytest.mark.asyncio
    async def test_debate_outcome_uses_get_continuous_learner(self):
        event = _make_learning_event()
        mock_l = MagicMock()
        mock_l.on_debate_completed = AsyncMock(return_value=event)
        mock_l.elo_updater = MagicMock()
        mock_l.elo_updater.get_rating.return_value = 1500.0
        set_continuous_learner(mock_l)

        req = _make_request(
            method="POST",
            body={"debate_id": "d-1", "agents": ["a"]},
        )
        resp = await LearningHandler.record_debate_outcome(req)

        assert resp.status == 200
        mock_l.on_debate_completed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_feedback_uses_get_continuous_learner(self):
        event = _make_learning_event()
        mock_l = MagicMock()
        mock_l.on_user_feedback = AsyncMock(return_value=event)
        set_continuous_learner(mock_l)

        req = _make_request(
            method="POST",
            body={"debate_id": "d", "agent_id": "a", "feedback_type": "helpful"},
        )
        resp = await LearningHandler.record_user_feedback(req)

        assert resp.status == 200
        mock_l.on_user_feedback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_patterns_uses_get_continuous_learner(self):
        mock_l = MagicMock()
        mock_l.pattern_extractor = MagicMock()
        mock_l.pattern_extractor.get_patterns.return_value = []
        set_continuous_learner(mock_l)

        req = _make_request()
        resp = await LearningHandler.get_patterns(req)

        assert resp.status == 200
        mock_l.pattern_extractor.get_patterns.assert_called_once()

    @pytest.mark.asyncio
    async def test_periodic_learning_uses_get_continuous_learner(self):
        mock_l = MagicMock()
        mock_l.run_periodic_learning = AsyncMock(return_value={"ok": True})
        set_continuous_learner(mock_l)

        req = _make_request(method="POST")
        resp = await LearningHandler.run_periodic_learning(req)

        assert resp.status == 200
        mock_l.run_periodic_learning.assert_awaited_once()
