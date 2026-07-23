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


class _FakeStorage:
    """Minimal storage stub: knows a fixed set of debate IDs."""

    def __init__(self, known_ids: set[str]):
        self.known_ids = set(known_ids)

    def get_debate(self, debate_id: str) -> dict | None:
        if debate_id in self.known_ids:
            return {"id": debate_id, "status": "running"}
        return None


# Debate IDs the endpoint tests exercise; the existence gate (added with the
# CD-098 route wiring) requires the debate to be resolvable via active state
# or storage before any intervention manager is created.
KNOWN_DEBATE_IDS = {"abc-123", "test-debate"}


@pytest.fixture
def handler_instance():
    """Create a DebateInterventionsHandler whose storage knows the test debates."""
    return DebateInterventionsHandler(ctx={"storage": _FakeStorage(KNOWN_DEBATE_IDS)})


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

    def test_extract_extra_segments_returns_error(self):
        """Exact shape /api/debates/{id}/{action}: extra segments rejected,
        never treated as an action on the first segment."""
        debate_id, err = _extract_debate_id_from_path("/api/v1/debates/victim/anything/pause")
        assert debate_id is None
        assert err is not None


# ============================================================================
# Path-shape enforcement (round-3 P2): malformed paths with extra segments
# must not dispatch here nor act on the leading ID segment.
# ============================================================================


class TestMalformedPathShape:
    """/api/v1/debates/{id}/extra/{action} is rejected at every layer."""

    MALFORMED_V1 = "/api/v1/debates/victim/anything/pause"
    MALFORMED_UNVERSIONED = "/api/debates/victim/anything/pause"

    def test_can_handle_rejects_extra_segments(self, handler_instance):
        assert handler_instance.can_handle(self.MALFORMED_V1) is False
        assert handler_instance.can_handle(self.MALFORMED_UNVERSIONED) is False

    def test_can_handle_accepts_exact_shape(self, handler_instance):
        assert handler_instance.can_handle("/api/v1/debates/victim/pause") is True
        assert handler_instance.can_handle("/api/debates/victim/pause") is True

    @patch(
        "aragora.server.handlers.debates.interventions.DebateInterventionsHandler._extract_user_id",
        return_value="user-1",
    )
    def test_malformed_pause_is_rejected_and_no_action_taken(self, mock_uid, handler_instance):
        """Even if a malformed path reaches the endpoint (defense in depth),
        it is rejected and no intervention state is created for 'victim'."""
        from aragora.debate.intervention import get_intervention_manager

        result = handler_instance._pause_debate(self.MALFORMED_V1, _make_handler())
        assert result.status_code in (400, 404)
        assert get_intervention_manager("victim", create=False) is None

    @patch(
        "aragora.server.handlers.debates.interventions.DebateInterventionsHandler._extract_user_id",
        return_value="user-1",
    )
    def test_malformed_path_does_not_pause_existing_debate(self, mock_uid):
        """A real debate must not be paused through a malformed path."""
        storage = _FakeStorage({"victim"})
        handler_instance = DebateInterventionsHandler(ctx={"storage": storage})
        result = handler_instance._pause_debate(self.MALFORMED_V1, _make_handler())
        assert result.status_code in (400, 404)
        # The exact-shape path still works and shows the debate unpaused.
        ok = handler_instance._pause_debate("/api/v1/debates/victim/pause", _make_handler())
        assert ok.status_code == 200  # first pause succeeds => was never paused

    def test_malformed_intervention_log_rejected(self, handler_instance):
        result = handler_instance._get_intervention_log(
            "/api/v1/debates/victim/anything/intervention-log", _make_handler()
        )
        assert result.status_code in (400, 404)


# ============================================================================
# Debate existence gate (CD-098 round-2 P2): interventions on nonexistent
# debates must 404 and must NOT create global intervention manager state.
# ============================================================================


class TestNonexistentDebateReturns404:
    """Every intervention action 404s for unknown debate IDs, side-effect free."""

    GHOST = "ghost-debate"

    def _assert_no_manager_state(self):
        from aragora.debate.intervention import get_intervention_manager

        assert get_intervention_manager(self.GHOST, create=False) is None

    @patch(
        "aragora.server.handlers.debates.interventions.DebateInterventionsHandler._extract_user_id",
        return_value="user-1",
    )
    def test_pause_nonexistent_404_no_state(self, mock_uid, handler_instance):
        result = handler_instance._pause_debate(
            f"/api/v1/debates/{self.GHOST}/pause", _make_handler()
        )
        assert result.status_code == 404
        self._assert_no_manager_state()

    @patch(
        "aragora.server.handlers.debates.interventions.DebateInterventionsHandler._extract_user_id",
        return_value="user-1",
    )
    def test_resume_nonexistent_404_no_state(self, mock_uid, handler_instance):
        result = handler_instance._resume_debate(
            f"/api/v1/debates/{self.GHOST}/resume", _make_handler()
        )
        assert result.status_code == 404
        self._assert_no_manager_state()

    @patch(
        "aragora.server.handlers.debates.interventions.DebateInterventionsHandler._extract_user_id",
        return_value="user-1",
    )
    def test_nudge_nonexistent_404_no_state(self, mock_uid, handler_instance):
        result = handler_instance._nudge_debate(
            f"/api/v1/debates/{self.GHOST}/nudge",
            _make_handler({"message": "hint"}),
        )
        assert result.status_code == 404
        self._assert_no_manager_state()

    @patch(
        "aragora.server.handlers.debates.interventions.DebateInterventionsHandler._extract_user_id",
        return_value="user-1",
    )
    def test_challenge_nonexistent_404_no_state(self, mock_uid, handler_instance):
        result = handler_instance._challenge_debate(
            f"/api/v1/debates/{self.GHOST}/challenge",
            _make_handler({"challenge": "counterpoint"}),
        )
        assert result.status_code == 404
        self._assert_no_manager_state()

    @patch(
        "aragora.server.handlers.debates.interventions.DebateInterventionsHandler._extract_user_id",
        return_value="user-1",
    )
    def test_inject_evidence_nonexistent_404_no_state(self, mock_uid, handler_instance):
        result = handler_instance._inject_evidence(
            f"/api/v1/debates/{self.GHOST}/inject-evidence",
            _make_handler({"evidence": "some evidence"}),
        )
        assert result.status_code == 404
        self._assert_no_manager_state()

    def test_intervention_log_nonexistent_404(self, handler_instance):
        result = handler_instance._get_intervention_log(
            f"/api/v1/debates/{self.GHOST}/intervention-log", _make_handler()
        )
        assert result.status_code == 404
        self._assert_no_manager_state()

    def test_intervention_log_existing_debate_empty_log_200(self, handler_instance):
        """A real debate with no interventions still gets the empty log, not 404."""
        result = handler_instance._get_intervention_log(
            "/api/v1/debates/test-debate/intervention-log", _make_handler()
        )
        data = _parse(result)
        assert result.status_code == 200
        assert data["entry_count"] == 0

    @patch(
        "aragora.server.handlers.debates.interventions.DebateInterventionsHandler._extract_user_id",
        return_value="user-1",
    )
    def test_active_state_only_debate_passes_gate(self, mock_uid):
        """A debate known only to the active state manager (no storage) is real."""
        handler_instance = DebateInterventionsHandler(ctx={})
        state_mgr = MagicMock()
        state_mgr.get_debate.return_value = object()
        with patch("aragora.server.state.get_state_manager", return_value=state_mgr):
            result = handler_instance._pause_debate(
                "/api/v1/debates/live-only/pause", _make_handler()
            )
        assert result.status_code == 200

    @patch(
        "aragora.server.handlers.debates.interventions.DebateInterventionsHandler._extract_user_id",
        return_value="user-1",
    )
    def test_existing_manager_passes_gate_without_state_or_storage(self, mock_uid):
        """A previously created manager keeps resume working after state/storage move on."""
        from aragora.debate.intervention import get_intervention_manager

        # Simulate a manager created while the debate was live and validated.
        get_intervention_manager("was-live", create=True)
        handler_instance = DebateInterventionsHandler(ctx={})
        result = handler_instance._pause_debate("/api/v1/debates/was-live/pause", _make_handler())
        assert result.status_code == 200
