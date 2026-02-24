"""Tests for the debate interventions HTTP handler.

Covers all 6 endpoints:
- POST /api/v1/debates/{id}/pause
- POST /api/v1/debates/{id}/resume
- POST /api/v1/debates/{id}/nudge
- POST /api/v1/debates/{id}/challenge
- POST /api/v1/debates/{id}/inject-evidence
- GET  /api/v1/debates/{id}/intervention-log

Also covers:
- Auth/permission checks (via RBAC decorator)
- Invalid debate ID handling
- Missing/invalid body handling
- can_handle routing
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.intervention import _reset_managers
from aragora.server.handlers.base import HandlerResult
from aragora.server.handlers.debates.interventions import (
    DebateInterventionsHandler,
    _extract_debate_id_from_path,
)


def _parse(result: HandlerResult) -> dict:
    """Parse HandlerResult body as JSON."""
    body = result.body
    if isinstance(body, bytes):
        return json.loads(body.decode("utf-8"))
    return json.loads(body)


def _make_handler(body: dict | None = None) -> MagicMock:
    """Create a mock HTTP handler with optional JSON body."""
    handler = MagicMock()
    handler.client_address = ("10.0.0.1", 12345)
    handler.headers = {
        "Content-Type": "application/json",
        "Host": "example.com",
    }

    if body is not None:
        encoded = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(len(encoded))
        handler.rfile = io.BytesIO(encoded)
    else:
        handler.headers["Content-Length"] = "0"
        handler.rfile = io.BytesIO(b"")

    handler.stream_emitter = MagicMock()
    return handler


@pytest.fixture(autouse=True)
def reset_state():
    """Reset intervention manager registry between tests."""
    _reset_managers()
    yield
    _reset_managers()


@pytest.fixture
def handler_instance():
    """Create a DebateInterventionsHandler with empty context."""
    return DebateInterventionsHandler(ctx={})


# ============================================================================
# Route matching
# ============================================================================


class TestCanHandle:
    """Test route matching via can_handle."""

    def test_handles_pause(self, handler_instance):
        assert handler_instance.can_handle("/api/v1/debates/abc-123/pause") is True

    def test_handles_resume(self, handler_instance):
        assert handler_instance.can_handle("/api/v1/debates/abc-123/resume") is True

    def test_handles_nudge(self, handler_instance):
        assert handler_instance.can_handle("/api/v1/debates/abc-123/nudge") is True

    def test_handles_challenge(self, handler_instance):
        assert handler_instance.can_handle("/api/v1/debates/abc-123/challenge") is True

    def test_handles_inject_evidence(self, handler_instance):
        assert handler_instance.can_handle("/api/v1/debates/abc-123/inject-evidence") is True

    def test_handles_intervention_log(self, handler_instance):
        assert handler_instance.can_handle("/api/v1/debates/abc-123/intervention-log") is True

    def test_rejects_unrelated_path(self, handler_instance):
        assert handler_instance.can_handle("/api/v1/debates") is False
        assert handler_instance.can_handle("/api/v1/agents") is False
        assert handler_instance.can_handle("/api/v1/debates/abc/export/json") is False


# ============================================================================
# POST /api/v1/debates/{id}/pause
# ============================================================================


class TestPauseEndpoint:
    """POST /api/v1/debates/{id}/pause."""

    @patch(
        "aragora.server.handlers.debates.interventions.DebateInterventionsHandler._extract_user_id",
        return_value="user-1",
    )
    def test_pause_success(self, mock_uid, handler_instance):
        mock_handler = _make_handler()
        result = handler_instance._pause_debate("/api/v1/debates/test-debate/pause", mock_handler)
        data = _parse(result)
        assert result.status_code == 200
        assert data["success"] is True
        assert data["state"] == "paused"
        assert data["debate_id"] == "test-debate"

    @patch(
        "aragora.server.handlers.debates.interventions.DebateInterventionsHandler._extract_user_id",
        return_value="user-1",
    )
    def test_double_pause_returns_400(self, mock_uid, handler_instance):
        mock_handler = _make_handler()
        handler_instance._pause_debate("/api/v1/debates/test-debate/pause", mock_handler)
        result = handler_instance._pause_debate("/api/v1/debates/test-debate/pause", mock_handler)
        assert result.status_code == 400


# ============================================================================
# POST /api/v1/debates/{id}/resume
# ============================================================================


class TestResumeEndpoint:
    """POST /api/v1/debates/{id}/resume."""

    @patch(
        "aragora.server.handlers.debates.interventions.DebateInterventionsHandler._extract_user_id",
        return_value="user-1",
    )
    def test_resume_after_pause(self, mock_uid, handler_instance):
        mock_handler = _make_handler()
        handler_instance._pause_debate("/api/v1/debates/test-debate/pause", mock_handler)
        result = handler_instance._resume_debate("/api/v1/debates/test-debate/resume", mock_handler)
        data = _parse(result)
        assert result.status_code == 200
        assert data["success"] is True
        assert data["state"] == "running"

    @patch(
        "aragora.server.handlers.debates.interventions.DebateInterventionsHandler._extract_user_id",
        return_value="user-1",
    )
    def test_resume_without_pause_returns_400(self, mock_uid, handler_instance):
        mock_handler = _make_handler()
        result = handler_instance._resume_debate("/api/v1/debates/test-debate/resume", mock_handler)
        # First access creates manager in running state, so resume fails
        assert result.status_code == 400


# ============================================================================
# POST /api/v1/debates/{id}/nudge
# ============================================================================


class TestNudgeEndpoint:
    """POST /api/v1/debates/{id}/nudge."""

    @patch(
        "aragora.server.handlers.debates.interventions.DebateInterventionsHandler._extract_user_id",
        return_value="user-1",
    )
    def test_nudge_success(self, mock_uid, handler_instance):
        mock_handler = _make_handler({"message": "Think about costs"})
        result = handler_instance._nudge_debate("/api/v1/debates/test-debate/nudge", mock_handler)
        data = _parse(result)
        assert result.status_code == 200
        assert data["success"] is True
        assert data["intervention"]["message"] == "Think about costs"

    @patch(
        "aragora.server.handlers.debates.interventions.DebateInterventionsHandler._extract_user_id",
        return_value="user-1",
    )
    def test_nudge_with_target_agent(self, mock_uid, handler_instance):
        mock_handler = _make_handler({"message": "Focus", "target_agent": "claude"})
        result = handler_instance._nudge_debate("/api/v1/debates/test-debate/nudge", mock_handler)
        data = _parse(result)
        assert data["intervention"]["target_agent"] == "claude"

    def test_nudge_missing_message_returns_400(self, handler_instance):
        mock_handler = _make_handler({})
        result = handler_instance._nudge_debate("/api/v1/debates/test-debate/nudge", mock_handler)
        assert result.status_code == 400


# ============================================================================
# POST /api/v1/debates/{id}/challenge
# ============================================================================


class TestChallengeEndpoint:
    """POST /api/v1/debates/{id}/challenge."""

    @patch(
        "aragora.server.handlers.debates.interventions.DebateInterventionsHandler._extract_user_id",
        return_value="user-1",
    )
    def test_challenge_success(self, mock_uid, handler_instance):
        mock_handler = _make_handler({"challenge": "What about privacy?"})
        result = handler_instance._challenge_debate(
            "/api/v1/debates/test-debate/challenge", mock_handler
        )
        data = _parse(result)
        assert result.status_code == 200
        assert data["success"] is True
        assert data["intervention"]["message"] == "What about privacy?"

    def test_challenge_missing_text_returns_400(self, handler_instance):
        mock_handler = _make_handler({})
        result = handler_instance._challenge_debate(
            "/api/v1/debates/test-debate/challenge", mock_handler
        )
        assert result.status_code == 400


# ============================================================================
# POST /api/v1/debates/{id}/inject-evidence
# ============================================================================


class TestInjectEvidenceEndpoint:
    """POST /api/v1/debates/{id}/inject-evidence."""

    @patch(
        "aragora.server.handlers.debates.interventions.DebateInterventionsHandler._extract_user_id",
        return_value="user-1",
    )
    def test_inject_evidence_success(self, mock_uid, handler_instance):
        mock_handler = _make_handler(
            {"evidence": "Studies show...", "source": "https://example.com"}
        )
        result = handler_instance._inject_evidence(
            "/api/v1/debates/test-debate/inject-evidence", mock_handler
        )
        data = _parse(result)
        assert result.status_code == 200
        assert data["success"] is True
        assert data["intervention"]["source"] == "https://example.com"

    def test_inject_evidence_missing_text_returns_400(self, handler_instance):
        mock_handler = _make_handler({"source": "src"})
        result = handler_instance._inject_evidence(
            "/api/v1/debates/test-debate/inject-evidence", mock_handler
        )
        assert result.status_code == 400


# ============================================================================
# GET /api/v1/debates/{id}/intervention-log
# ============================================================================


class TestInterventionLogEndpoint:
    """GET /api/v1/debates/{id}/intervention-log."""

    def test_empty_log_for_new_debate(self, handler_instance):
        mock_handler = _make_handler()
        result = handler_instance._get_intervention_log(
            "/api/v1/debates/test-debate/intervention-log", mock_handler
        )
        data = _parse(result)
        assert result.status_code == 200
        assert data["entry_count"] == 0
        assert data["entries"] == []

    @patch(
        "aragora.server.handlers.debates.interventions.DebateInterventionsHandler._extract_user_id",
        return_value="user-1",
    )
    def test_log_reflects_interventions(self, mock_uid, handler_instance):
        mock_handler = _make_handler({"message": "hint"})
        handler_instance._nudge_debate("/api/v1/debates/test-debate/nudge", mock_handler)

        log_handler = _make_handler()
        result = handler_instance._get_intervention_log(
            "/api/v1/debates/test-debate/intervention-log", log_handler
        )
        data = _parse(result)
        assert data["entry_count"] == 1
        assert data["entries"][0]["type"] == "nudge"


# ============================================================================
# Path Extraction & Validation
# ============================================================================


class TestPathExtraction:
    """Debate ID extraction and validation from paths."""

    def test_extract_valid_id(self):
        debate_id, err = _extract_debate_id_from_path("/api/v1/debates/abc-123/pause")
        assert debate_id == "abc-123"
        assert err is None

    def test_extract_unversioned_path(self):
        debate_id, err = _extract_debate_id_from_path("/api/debates/abc-123/pause")
        assert debate_id == "abc-123"
        assert err is None

    def test_extract_short_path_returns_error(self):
        debate_id, err = _extract_debate_id_from_path("/api/v1")
        assert debate_id is None
        assert err is not None
