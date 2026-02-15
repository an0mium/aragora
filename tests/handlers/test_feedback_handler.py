"""Tests for agent feedback (SelectionFeedbackLoop) handler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.selection_feedback import (
    AgentFeedbackState,
    SelectionFeedbackLoop,
)
from aragora.server.handlers.agents.feedback import FeedbackHandler


@pytest.fixture
def handler():
    return FeedbackHandler(ctx={})


@pytest.fixture
def mock_loop():
    """Create a SelectionFeedbackLoop with some test data."""
    loop = SelectionFeedbackLoop()
    loop.process_debate_outcome(
        debate_id="d1",
        participants=["claude", "gpt4"],
        winner="claude",
        domain="security",
    )
    loop.process_debate_outcome(
        debate_id="d2",
        participants=["claude", "gpt4"],
        winner="gpt4",
        domain="finance",
    )
    loop.process_debate_outcome(
        debate_id="d3",
        participants=["claude", "gpt4"],
        winner="claude",
        domain="security",
    )
    return loop


class TestFeedbackMetrics:
    """Tests for GET /api/v1/agents/feedback/metrics."""

    def test_metrics_returns_correct_format(self, handler, mock_loop):
        with patch(
            "aragora.server.handlers.agents.feedback._get_feedback_loop",
            return_value=mock_loop,
        ):
            result = handler.handle(
                "/api/v1/agents/feedback/metrics", {}, MagicMock()
            )
        assert result is not None
        body = result[0]
        assert body["debates_processed"] == 3
        assert body["agents_tracked"] == 2
        assert "average_adjustment" in body

    def test_metrics_empty_loop(self, handler):
        with patch(
            "aragora.server.handlers.agents.feedback._get_feedback_loop",
            return_value=None,
        ):
            result = handler.handle(
                "/api/v1/agents/feedback/metrics", {}, MagicMock()
            )
        assert result is not None
        body = result[0]
        assert body["debates_processed"] == 0
        assert body["agents_tracked"] == 0


class TestFeedbackStates:
    """Tests for GET /api/v1/agents/feedback/states."""

    def test_states_returns_all_agents(self, handler, mock_loop):
        with patch(
            "aragora.server.handlers.agents.feedback._get_feedback_loop",
            return_value=mock_loop,
        ):
            result = handler.handle(
                "/api/v1/agents/feedback/states", {}, MagicMock()
            )
        assert result is not None
        body = result[0]
        assert body["count"] == 2
        assert "claude" in body["agents"]
        assert "gpt4" in body["agents"]

    def test_states_agent_fields(self, handler, mock_loop):
        with patch(
            "aragora.server.handlers.agents.feedback._get_feedback_loop",
            return_value=mock_loop,
        ):
            result = handler.handle(
                "/api/v1/agents/feedback/states", {}, MagicMock()
            )
        body = result[0]
        claude = body["agents"]["claude"]
        assert claude["total_debates"] == 3
        assert claude["wins"] == 2
        assert claude["losses"] == 1
        assert "win_rate" in claude
        assert "timeout_rate" in claude
        assert "calibration_score" in claude

    def test_states_empty_returns_defaults(self, handler):
        with patch(
            "aragora.server.handlers.agents.feedback._get_feedback_loop",
            return_value=None,
        ):
            result = handler.handle(
                "/api/v1/agents/feedback/states", {}, MagicMock()
            )
        body = result[0]
        assert body["count"] == 0
        assert body["agents"] == {}


class TestFeedbackDomains:
    """Tests for GET /api/v1/agents/{name}/feedback/domains."""

    def test_domain_weights_for_agent(self, handler, mock_loop):
        with patch(
            "aragora.server.handlers.agents.feedback._get_feedback_loop",
            return_value=mock_loop,
        ):
            result = handler.handle(
                "/api/v1/agents/claude/feedback/domains", {}, MagicMock()
            )
        assert result is not None
        body = result[0]
        assert body["agent"] == "claude"
        assert "security" in body["domains"]

    def test_domain_filter_works(self, handler, mock_loop):
        with patch(
            "aragora.server.handlers.agents.feedback._get_feedback_loop",
            return_value=mock_loop,
        ):
            result = handler.handle(
                "/api/v1/agents/claude/feedback/domains",
                {"domain": ["security"]},
                MagicMock(),
            )
        body = result[0]
        assert "security" in body["domains"]
        # Only one domain returned since filter was applied
        assert len(body["domains"]) == 1

    def test_unknown_agent_returns_empty(self, handler, mock_loop):
        with patch(
            "aragora.server.handlers.agents.feedback._get_feedback_loop",
            return_value=mock_loop,
        ):
            result = handler.handle(
                "/api/v1/agents/nonexistent/feedback/domains", {}, MagicMock()
            )
        body = result[0]
        assert body["agent"] == "nonexistent"
        assert body["domains"] == {}

    def test_no_loop_returns_empty(self, handler):
        with patch(
            "aragora.server.handlers.agents.feedback._get_feedback_loop",
            return_value=None,
        ):
            result = handler.handle(
                "/api/v1/agents/claude/feedback/domains", {}, MagicMock()
            )
        body = result[0]
        assert body["domains"] == {}


class TestCanHandle:
    """Tests for route matching."""

    def test_can_handle_metrics(self, handler):
        assert handler.can_handle("/api/v1/agents/feedback/metrics")

    def test_can_handle_states(self, handler):
        assert handler.can_handle("/api/v1/agents/feedback/states")

    def test_can_handle_domains(self, handler):
        assert handler.can_handle("/api/v1/agents/claude/feedback/domains")

    def test_cannot_handle_unrelated(self, handler):
        assert not handler.can_handle("/api/v1/debates")


@pytest.mark.no_auto_auth
class TestFeedbackPermission:
    """Tests for permission guard on feedback endpoints."""

    def test_metrics_requires_permission(self, handler):
        """Without auth bypass, ensure the decorator is present."""
        # The @require_permission decorator should be on handle()
        assert hasattr(handler.handle, "__wrapped__") or hasattr(
            handler.handle, "_permission"
        )
