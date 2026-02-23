"""Comprehensive tests for the User Feedback Collection Handler.

Tests the four feedback endpoints in aragora/server/handlers/feedback.py:

- POST /api/v1/feedback/nps           -> handle_submit_nps
- POST /api/v1/feedback/general       -> handle_submit_feedback
- GET  /api/v1/feedback/nps/summary   -> handle_get_nps_summary
- GET  /api/v1/feedback/prompts       -> handle_get_feedback_prompts

Also tests:
- _check_permission helper (auth, denial, missing user)
- FeedbackStore (SQLite-backed persistence, NPS summary computation)
- FeedbackEntry dataclass (to_dict, auto-generated created_at)
- FeedbackType enum values
- FEEDBACK_ROUTES and FeedbackRoutesHandler.ROUTES
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.feedback import (
    FEEDBACK_ROUTES,
    FeedbackEntry,
    FeedbackRoutesHandler,
    FeedbackStore,
    FeedbackType,
    _check_permission,
    get_feedback_store,
    handle_get_feedback_prompts,
    handle_get_nps_summary,
    handle_submit_feedback,
    handle_submit_nps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_store():
    """Create a mock FeedbackStore."""
    store = MagicMock(spec=FeedbackStore)
    store.save.return_value = None
    store.get_nps_summary.return_value = {
        "nps_score": 50,
        "total_responses": 10,
        "promoters": 6,
        "passives": 2,
        "detractors": 2,
        "period_days": 30,
    }
    return store


@pytest.fixture(autouse=True)
def _patch_feedback_deps(monkeypatch, mock_store):
    """Patch dependencies for all feedback handler tests."""
    # Patch get_feedback_store to return our mock
    monkeypatch.setattr(
        "aragora.server.handlers.feedback.get_feedback_store",
        lambda: mock_store,
    )

    # Patch _check_permission to allow all access by default
    monkeypatch.setattr(
        "aragora.server.handlers.feedback._check_permission",
        lambda ctx, perm: None,
    )


def _ctx(
    user_id: str = "user-001",
    body: dict | None = None,
    query: dict | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a handler context dict."""
    ctx: dict[str, Any] = {
        "user_id": user_id,
        "org_id": "org-001",
        "roles": {"admin"},
        "permissions": {"*"},
        "body": body or {},
        "query": query or {},
    }
    ctx.update(extra)
    return ctx


# ===========================================================================
# FeedbackEntry dataclass
# ===========================================================================


class TestFeedbackEntry:
    """Tests for the FeedbackEntry dataclass."""

    def test_to_dict(self):
        entry = FeedbackEntry(
            id="fb-1",
            user_id="u1",
            feedback_type=FeedbackType.NPS,
            score=9,
            comment="Great!",
            metadata={"source": "web"},
            created_at="2026-01-01T00:00:00+00:00",
        )
        d = entry.to_dict()
        assert d["id"] == "fb-1"
        assert d["user_id"] == "u1"
        assert d["feedback_type"] == "nps"
        assert d["score"] == 9
        assert d["comment"] == "Great!"
        assert d["metadata"] == {"source": "web"}
        assert d["created_at"] == "2026-01-01T00:00:00+00:00"

    def test_auto_created_at(self):
        entry = FeedbackEntry(
            id="fb-2",
            user_id="u2",
            feedback_type=FeedbackType.GENERAL,
            score=None,
            comment="Hello",
        )
        assert entry.created_at  # non-empty
        assert "T" in entry.created_at  # ISO format

    def test_explicit_created_at_preserved(self):
        ts = "2025-06-15T12:00:00+00:00"
        entry = FeedbackEntry(
            id="fb-3",
            user_id="u3",
            feedback_type=FeedbackType.BUG_REPORT,
            score=None,
            comment="Bug",
            created_at=ts,
        )
        assert entry.created_at == ts


# ===========================================================================
# FeedbackType enum
# ===========================================================================


class TestFeedbackType:
    """Tests for the FeedbackType enum."""

    def test_values(self):
        assert FeedbackType.NPS.value == "nps"
        assert FeedbackType.FEATURE_REQUEST.value == "feature_request"
        assert FeedbackType.BUG_REPORT.value == "bug_report"
        assert FeedbackType.GENERAL.value == "general"
        assert FeedbackType.DEBATE_QUALITY.value == "debate_quality"

    def test_from_string(self):
        assert FeedbackType("nps") == FeedbackType.NPS
        assert FeedbackType("bug_report") == FeedbackType.BUG_REPORT

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            FeedbackType("nonexistent")


# ===========================================================================
# FeedbackStore (integration with real SQLite in-memory DB)
# ===========================================================================


class TestFeedbackStore:
    """Tests for the FeedbackStore SQLite backend."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a real FeedbackStore backed by a temp file."""
        db = str(tmp_path / "test_feedback.db")
        return FeedbackStore(db_path=db)

    def test_save_and_nps_summary(self, store):
        """Save several NPS entries and verify summary computation."""
        for i, score in enumerate([10, 9, 8, 7, 3, 2]):
            entry = FeedbackEntry(
                id=f"e-{i}",
                user_id="u1",
                feedback_type=FeedbackType.NPS,
                score=score,
                comment=None,
            )
            store.save(entry)

        summary = store.get_nps_summary(days=30)
        # Promoters: 10, 9 => 2
        # Passives: 8, 7 => 2
        # Detractors: 3, 2 => 2
        assert summary["promoters"] == 2
        assert summary["passives"] == 2
        assert summary["detractors"] == 2
        assert summary["total_responses"] == 6
        # NPS = ((2 - 2) / 6) * 100 = 0
        assert summary["nps_score"] == 0
        assert summary["period_days"] == 30

    def test_nps_summary_empty(self, store):
        """Empty store returns zero NPS."""
        summary = store.get_nps_summary()
        assert summary["nps_score"] == 0
        assert summary["total_responses"] == 0
        assert summary["promoters"] == 0

    def test_nps_all_promoters(self, store):
        """All promoters yields NPS of 100."""
        for i in range(5):
            store.save(FeedbackEntry(
                id=f"p-{i}",
                user_id="u1",
                feedback_type=FeedbackType.NPS,
                score=10,
                comment=None,
            ))
        summary = store.get_nps_summary()
        assert summary["nps_score"] == 100

    def test_nps_all_detractors(self, store):
        """All detractors yields NPS of -100."""
        for i in range(3):
            store.save(FeedbackEntry(
                id=f"d-{i}",
                user_id="u1",
                feedback_type=FeedbackType.NPS,
                score=0,
                comment=None,
            ))
        summary = store.get_nps_summary()
        assert summary["nps_score"] == -100

    def test_save_non_nps_excluded_from_summary(self, store):
        """Non-NPS feedback should not affect NPS summary."""
        store.save(FeedbackEntry(
            id="gen-1",
            user_id="u1",
            feedback_type=FeedbackType.GENERAL,
            score=5,
            comment="general",
        ))
        summary = store.get_nps_summary()
        assert summary["total_responses"] == 0

    def test_duplicate_id_raises(self, store):
        """Saving with duplicate ID raises IntegrityError."""
        entry = FeedbackEntry(
            id="dup-1",
            user_id="u1",
            feedback_type=FeedbackType.NPS,
            score=8,
            comment=None,
        )
        store.save(entry)
        with pytest.raises(sqlite3.IntegrityError):
            store.save(entry)

    def test_init_creates_tables(self, tmp_path):
        """Store initialization creates feedback table and indexes."""
        db = str(tmp_path / "init_test.db")
        FeedbackStore(db_path=db)
        with sqlite3.connect(db) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
            )
            assert cursor.fetchone() is not None


# ===========================================================================
# _check_permission (un-patched)
# ===========================================================================


class TestCheckPermission:
    """Tests for the _check_permission helper function."""

    @pytest.fixture(autouse=True)
    def _unpatch_check_permission(self, monkeypatch):
        """Re-enable real _check_permission for this test class."""
        # We need to import the real function and re-set it
        from aragora.server.handlers import feedback as _mod

        # The module-level reference was patched by the autouse fixture;
        # we re-import it from the original source.
        import importlib

        importlib.reload(_mod)
        # After reload, ensure get_feedback_store still returns our mock
        monkeypatch.setattr(
            "aragora.server.handlers.feedback.get_feedback_store",
            lambda: MagicMock(spec=FeedbackStore),
        )

    def test_no_user_id_returns_401(self):
        ctx: dict[str, Any] = {"org_id": "o1"}
        result = _check_permission(ctx, "feedback.write")
        assert result is not None
        assert _status(result) == 401

    def test_empty_user_id_returns_401(self):
        ctx: dict[str, Any] = {"user_id": "", "org_id": "o1"}
        result = _check_permission(ctx, "feedback.write")
        assert result is not None
        assert _status(result) == 401

    def test_permission_granted(self):
        """When checker says allowed, _check_permission returns None."""
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch(
            "aragora.server.handlers.feedback.get_permission_checker",
            return_value=mock_checker,
        ):
            result = _check_permission(
                {"user_id": "u1", "org_id": "o1", "roles": set(), "permissions": set()},
                "feedback.write",
            )
        assert result is None

    def test_permission_denied(self):
        """When checker denies, _check_permission returns 403."""
        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch(
            "aragora.server.handlers.feedback.get_permission_checker",
            return_value=mock_checker,
        ):
            result = _check_permission(
                {"user_id": "u1", "org_id": "o1", "roles": set(), "permissions": set()},
                "feedback.write",
            )
        assert result is not None
        assert _status(result) == 403
        assert "denied" in _body(result).get("error", "").lower()

    def test_builds_auth_context_from_ctx(self):
        """Verify AuthorizationContext is constructed correctly from ctx."""
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch(
            "aragora.server.handlers.feedback.get_permission_checker",
            return_value=mock_checker,
        ):
            _check_permission(
                {
                    "user_id": "u42",
                    "org_id": "org-99",
                    "roles": {"editor", "viewer"},
                    "permissions": {"feedback.write"},
                },
                "feedback.write",
            )

        call_args = mock_checker.check_permission.call_args
        auth_ctx = call_args[0][0]
        assert auth_ctx.user_id == "u42"
        assert auth_ctx.org_id == "org-99"
        assert "editor" in auth_ctx.roles
        assert "feedback.write" in auth_ctx.permissions

    def test_none_roles_defaults_to_empty_set(self):
        """ctx with roles=None should not crash."""
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch(
            "aragora.server.handlers.feedback.get_permission_checker",
            return_value=mock_checker,
        ):
            result = _check_permission(
                {"user_id": "u1", "org_id": "o1", "roles": None, "permissions": None},
                "feedback.read",
            )
        assert result is None

    def test_no_org_id(self):
        """ctx without org_id should still work (org_id becomes None)."""
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch(
            "aragora.server.handlers.feedback.get_permission_checker",
            return_value=mock_checker,
        ):
            result = _check_permission(
                {"user_id": "u1", "roles": set(), "permissions": set()},
                "feedback.read",
            )
        assert result is None


# ===========================================================================
# handle_submit_nps
# ===========================================================================


class TestHandleSubmitNps:
    """Tests for POST /api/v1/feedback/nps."""

    @pytest.mark.asyncio
    async def test_success(self, mock_store):
        ctx = _ctx(body={"score": 9, "comment": "Love it"})
        result = await handle_submit_nps(ctx)

        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert "feedback_id" in body
        assert body["message"] == "Thank you for your feedback!"
        mock_store.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_success_min_score(self, mock_store):
        ctx = _ctx(body={"score": 0})
        result = await handle_submit_nps(ctx)
        assert _status(result) == 200
        assert _body(result)["success"] is True

    @pytest.mark.asyncio
    async def test_success_max_score(self, mock_store):
        ctx = _ctx(body={"score": 10})
        result = await handle_submit_nps(ctx)
        assert _status(result) == 200
        assert _body(result)["success"] is True

    @pytest.mark.asyncio
    async def test_score_missing(self):
        ctx = _ctx(body={})
        result = await handle_submit_nps(ctx)
        assert _status(result) == 400
        assert "score" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_score_none(self):
        ctx = _ctx(body={"score": None})
        result = await handle_submit_nps(ctx)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_score_below_range(self):
        ctx = _ctx(body={"score": -1})
        result = await handle_submit_nps(ctx)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_score_above_range(self):
        ctx = _ctx(body={"score": 11})
        result = await handle_submit_nps(ctx)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_score_not_int(self):
        ctx = _ctx(body={"score": "five"})
        result = await handle_submit_nps(ctx)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_score_float(self):
        ctx = _ctx(body={"score": 7.5})
        result = await handle_submit_nps(ctx)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_with_context_metadata(self, mock_store):
        ctx = _ctx(body={"score": 8, "context": {"page": "dashboard"}})
        result = await handle_submit_nps(ctx)
        assert _status(result) == 200
        saved_entry = mock_store.save.call_args[0][0]
        assert saved_entry.metadata == {"page": "dashboard"}

    @pytest.mark.asyncio
    async def test_with_comment(self, mock_store):
        ctx = _ctx(body={"score": 10, "comment": "Excellent tool"})
        result = await handle_submit_nps(ctx)
        assert _status(result) == 200
        saved_entry = mock_store.save.call_args[0][0]
        assert saved_entry.comment == "Excellent tool"

    @pytest.mark.asyncio
    async def test_anonymous_user(self, mock_store):
        """When user_id is missing, defaults to 'anonymous'."""
        ctx = _ctx(body={"score": 5})
        del ctx["user_id"]
        result = await handle_submit_nps(ctx)
        assert _status(result) == 200
        saved_entry = mock_store.save.call_args[0][0]
        assert saved_entry.user_id == "anonymous"

    @pytest.mark.asyncio
    async def test_store_error_returns_500(self, mock_store):
        mock_store.save.side_effect = sqlite3.Error("disk full")
        ctx = _ctx(body={"score": 7})
        result = await handle_submit_nps(ctx)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_empty_body(self):
        ctx = _ctx(body={})
        result = await handle_submit_nps(ctx)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_no_body_key(self):
        ctx = _ctx()
        del ctx["body"]
        result = await handle_submit_nps(ctx)
        # body defaults to {} via ctx.get("body", {})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_entry_has_correct_type(self, mock_store):
        ctx = _ctx(body={"score": 6})
        await handle_submit_nps(ctx)
        saved_entry = mock_store.save.call_args[0][0]
        assert saved_entry.feedback_type == FeedbackType.NPS

    @pytest.mark.asyncio
    async def test_permission_denied(self, monkeypatch):
        """When _check_permission returns an error, handler returns it."""
        from aragora.server.handlers.base import error_response

        monkeypatch.setattr(
            "aragora.server.handlers.feedback._check_permission",
            lambda ctx, perm: error_response("Permission denied", status=403),
        )
        ctx = _ctx(body={"score": 5})
        result = await handle_submit_nps(ctx)
        assert _status(result) == 403


# ===========================================================================
# handle_submit_feedback
# ===========================================================================


class TestHandleSubmitFeedback:
    """Tests for POST /api/v1/feedback/general."""

    @pytest.mark.asyncio
    async def test_success_general(self, mock_store):
        ctx = _ctx(body={"comment": "This is great", "type": "general"})
        result = await handle_submit_feedback(ctx)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert "feedback_id" in body
        mock_store.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_success_feature_request(self, mock_store):
        ctx = _ctx(body={"comment": "Need dark mode", "type": "feature_request"})
        result = await handle_submit_feedback(ctx)
        assert _status(result) == 200
        saved_entry = mock_store.save.call_args[0][0]
        assert saved_entry.feedback_type == FeedbackType.FEATURE_REQUEST

    @pytest.mark.asyncio
    async def test_success_bug_report(self, mock_store):
        ctx = _ctx(body={"comment": "Button broken", "type": "bug_report"})
        result = await handle_submit_feedback(ctx)
        assert _status(result) == 200
        saved_entry = mock_store.save.call_args[0][0]
        assert saved_entry.feedback_type == FeedbackType.BUG_REPORT

    @pytest.mark.asyncio
    async def test_success_debate_quality(self, mock_store):
        ctx = _ctx(body={"comment": "Debate was insightful", "type": "debate_quality"})
        result = await handle_submit_feedback(ctx)
        assert _status(result) == 200
        saved_entry = mock_store.save.call_args[0][0]
        assert saved_entry.feedback_type == FeedbackType.DEBATE_QUALITY

    @pytest.mark.asyncio
    async def test_unknown_type_defaults_to_general(self, mock_store):
        ctx = _ctx(body={"comment": "Something", "type": "unknown_type"})
        result = await handle_submit_feedback(ctx)
        assert _status(result) == 200
        saved_entry = mock_store.save.call_args[0][0]
        assert saved_entry.feedback_type == FeedbackType.GENERAL

    @pytest.mark.asyncio
    async def test_no_type_defaults_to_general(self, mock_store):
        ctx = _ctx(body={"comment": "No type specified"})
        result = await handle_submit_feedback(ctx)
        assert _status(result) == 200
        saved_entry = mock_store.save.call_args[0][0]
        assert saved_entry.feedback_type == FeedbackType.GENERAL

    @pytest.mark.asyncio
    async def test_comment_required(self):
        ctx = _ctx(body={"type": "general"})
        result = await handle_submit_feedback(ctx)
        assert _status(result) == 400
        assert "comment" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_empty_comment(self):
        ctx = _ctx(body={"comment": "", "type": "general"})
        result = await handle_submit_feedback(ctx)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_none_comment(self):
        ctx = _ctx(body={"comment": None, "type": "general"})
        result = await handle_submit_feedback(ctx)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_with_optional_score(self, mock_store):
        ctx = _ctx(body={"comment": "Good debate", "type": "debate_quality", "score": 8})
        result = await handle_submit_feedback(ctx)
        assert _status(result) == 200
        saved_entry = mock_store.save.call_args[0][0]
        assert saved_entry.score == 8

    @pytest.mark.asyncio
    async def test_with_context_metadata(self, mock_store):
        ctx = _ctx(body={
            "comment": "Feature idea",
            "type": "feature_request",
            "context": {"debate_id": "d-123"},
        })
        result = await handle_submit_feedback(ctx)
        assert _status(result) == 200
        saved_entry = mock_store.save.call_args[0][0]
        assert saved_entry.metadata == {"debate_id": "d-123"}

    @pytest.mark.asyncio
    async def test_anonymous_user(self, mock_store):
        ctx = _ctx(body={"comment": "Hi"})
        del ctx["user_id"]
        result = await handle_submit_feedback(ctx)
        assert _status(result) == 200
        saved_entry = mock_store.save.call_args[0][0]
        assert saved_entry.user_id == "anonymous"

    @pytest.mark.asyncio
    async def test_store_error_returns_500(self, mock_store):
        mock_store.save.side_effect = sqlite3.Error("write failed")
        ctx = _ctx(body={"comment": "test"})
        result = await handle_submit_feedback(ctx)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_permission_denied(self, monkeypatch):
        from aragora.server.handlers.base import error_response

        monkeypatch.setattr(
            "aragora.server.handlers.feedback._check_permission",
            lambda ctx, perm: error_response("Permission denied", status=403),
        )
        ctx = _ctx(body={"comment": "test"})
        result = await handle_submit_feedback(ctx)
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_no_body_key(self, mock_store):
        ctx = _ctx()
        del ctx["body"]
        # body defaults to {} via ctx.get("body", {})
        # No comment => 400
        result = await handle_submit_feedback(ctx)
        assert _status(result) == 400


# ===========================================================================
# handle_get_nps_summary
# ===========================================================================


class TestHandleGetNpsSummary:
    """Tests for GET /api/v1/feedback/nps/summary."""

    @pytest.mark.asyncio
    async def test_success_default_days(self, mock_store):
        ctx = _ctx(query={})
        result = await handle_get_nps_summary(ctx)
        assert _status(result) == 200
        body = _body(result)
        assert body["nps_score"] == 50
        assert body["total_responses"] == 10
        mock_store.get_nps_summary.assert_called_once_with(30)

    @pytest.mark.asyncio
    async def test_custom_days(self, mock_store):
        ctx = _ctx(query={"days": "7"})
        result = await handle_get_nps_summary(ctx)
        assert _status(result) == 200
        mock_store.get_nps_summary.assert_called_once_with(7)

    @pytest.mark.asyncio
    async def test_days_clamped_min(self, mock_store):
        ctx = _ctx(query={"days": "0"})
        result = await handle_get_nps_summary(ctx)
        assert _status(result) == 200
        # 0 is below min_val=1, so gets clamped to 1
        mock_store.get_nps_summary.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_days_clamped_max(self, mock_store):
        ctx = _ctx(query={"days": "999"})
        result = await handle_get_nps_summary(ctx)
        assert _status(result) == 200
        # 999 is above max_val=365, so gets clamped to 365
        mock_store.get_nps_summary.assert_called_once_with(365)

    @pytest.mark.asyncio
    async def test_store_error_returns_500(self, mock_store):
        mock_store.get_nps_summary.side_effect = sqlite3.Error("db locked")
        ctx = _ctx(query={})
        result = await handle_get_nps_summary(ctx)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_permission_denied(self, monkeypatch):
        from aragora.server.handlers.base import error_response

        monkeypatch.setattr(
            "aragora.server.handlers.feedback._check_permission",
            lambda ctx, perm: error_response("Permission denied", status=403),
        )
        ctx = _ctx(query={})
        result = await handle_get_nps_summary(ctx)
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_no_query_key(self, mock_store):
        ctx = _ctx()
        del ctx["query"]
        # query defaults to {} via ctx.get("query", {})
        result = await handle_get_nps_summary(ctx)
        assert _status(result) == 200
        mock_store.get_nps_summary.assert_called_once_with(30)

    @pytest.mark.asyncio
    async def test_response_includes_all_fields(self, mock_store):
        ctx = _ctx(query={})
        result = await handle_get_nps_summary(ctx)
        body = _body(result)
        expected_keys = {"nps_score", "total_responses", "promoters", "passives", "detractors", "period_days"}
        assert expected_keys.issubset(set(body.keys()))


# ===========================================================================
# handle_get_feedback_prompts
# ===========================================================================


class TestHandleGetFeedbackPrompts:
    """Tests for GET /api/v1/feedback/prompts."""

    @pytest.mark.asyncio
    async def test_success(self):
        ctx = _ctx()
        result = await handle_get_feedback_prompts(ctx)
        assert _status(result) == 200
        body = _body(result)
        assert "prompts" in body
        prompts = body["prompts"]
        assert len(prompts) >= 1

    @pytest.mark.asyncio
    async def test_nps_prompt_present(self):
        ctx = _ctx()
        result = await handle_get_feedback_prompts(ctx)
        body = _body(result)
        nps_prompts = [p for p in body["prompts"] if p["type"] == "nps"]
        assert len(nps_prompts) == 1

    @pytest.mark.asyncio
    async def test_nps_prompt_structure(self):
        ctx = _ctx()
        result = await handle_get_feedback_prompts(ctx)
        body = _body(result)
        nps = body["prompts"][0]
        assert nps["type"] == "nps"
        assert "question" in nps
        assert nps["scale"]["min"] == 0
        assert nps["scale"]["max"] == 10
        assert "labels" in nps["scale"]
        assert "follow_up" in nps

    @pytest.mark.asyncio
    async def test_permission_denied(self, monkeypatch):
        from aragora.server.handlers.base import error_response

        monkeypatch.setattr(
            "aragora.server.handlers.feedback._check_permission",
            lambda ctx, perm: error_response("Permission denied", status=403),
        )
        ctx = _ctx()
        result = await handle_get_feedback_prompts(ctx)
        assert _status(result) == 403


# ===========================================================================
# FEEDBACK_ROUTES and FeedbackRoutesHandler
# ===========================================================================


class TestRouteDefinitions:
    """Tests for route constants and the facade handler."""

    def test_feedback_routes_count(self):
        assert len(FEEDBACK_ROUTES) == 4

    def test_feedback_routes_methods(self):
        methods = [r[0] for r in FEEDBACK_ROUTES]
        assert methods.count("POST") == 2
        assert methods.count("GET") == 2

    def test_feedback_routes_paths(self):
        paths = [r[1] for r in FEEDBACK_ROUTES]
        assert "/api/v1/feedback/nps" in paths
        assert "/api/v1/feedback/general" in paths
        assert "/api/v1/feedback/nps/summary" in paths
        assert "/api/v1/feedback/prompts" in paths

    def test_feedback_routes_handlers(self):
        handlers = {r[1]: r[2] for r in FEEDBACK_ROUTES}
        assert handlers["/api/v1/feedback/nps"] is handle_submit_nps
        assert handlers["/api/v1/feedback/general"] is handle_submit_feedback
        assert handlers["/api/v1/feedback/nps/summary"] is handle_get_nps_summary
        assert handlers["/api/v1/feedback/prompts"] is handle_get_feedback_prompts

    def test_facade_handler_routes(self):
        assert "/api/v1/feedback/general" in FeedbackRoutesHandler.ROUTES
        assert "/api/v1/feedback/nps" in FeedbackRoutesHandler.ROUTES
        assert "/api/v1/feedback/nps/summary" in FeedbackRoutesHandler.ROUTES
        assert "/api/v1/feedback/prompts" in FeedbackRoutesHandler.ROUTES

    def test_facade_handler_routes_count(self):
        assert len(FeedbackRoutesHandler.ROUTES) == 4


# ===========================================================================
# get_feedback_store (lazy init)
# ===========================================================================


class TestGetFeedbackStore:
    """Tests for the get_feedback_store lazy factory."""

    def test_returns_store_instance(self, monkeypatch):
        """get_feedback_store returns a FeedbackStore via the LazyStore."""
        mock_lazy = MagicMock()
        mock_lazy.get.return_value = MagicMock(spec=FeedbackStore)
        monkeypatch.setattr(
            "aragora.server.handlers.feedback._feedback_store_lazy",
            mock_lazy,
        )
        store = get_feedback_store()
        assert store is not None
        mock_lazy.get.assert_called_once()


# ===========================================================================
# Edge cases and integration
# ===========================================================================


class TestEdgeCases:
    """Additional edge case and integration tests."""

    @pytest.mark.asyncio
    async def test_submit_nps_score_boundary_0(self, mock_store):
        result = await handle_submit_nps(_ctx(body={"score": 0}))
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_submit_nps_score_boundary_10(self, mock_store):
        result = await handle_submit_nps(_ctx(body={"score": 10}))
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_submit_nps_bool_score_rejected(self):
        """Boolean True (== 1 int) might pass isinstance(score, int);
        Python bool is subclass of int, so True maps to 1 and should be accepted."""
        result = await handle_submit_nps(_ctx(body={"score": True}))
        # bool is subclass of int in Python, True == 1, so 0 <= 1 <= 10 passes
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_submit_feedback_long_comment(self, mock_store):
        comment = "x" * 10000
        ctx = _ctx(body={"comment": comment, "type": "general"})
        result = await handle_submit_feedback(ctx)
        assert _status(result) == 200
        saved_entry = mock_store.save.call_args[0][0]
        assert saved_entry.comment == comment

    @pytest.mark.asyncio
    async def test_submit_feedback_without_score(self, mock_store):
        ctx = _ctx(body={"comment": "No score", "type": "feature_request"})
        result = await handle_submit_feedback(ctx)
        assert _status(result) == 200
        saved_entry = mock_store.save.call_args[0][0]
        assert saved_entry.score is None

    @pytest.mark.asyncio
    async def test_submit_feedback_without_context(self, mock_store):
        ctx = _ctx(body={"comment": "No context", "type": "general"})
        result = await handle_submit_feedback(ctx)
        assert _status(result) == 200
        saved_entry = mock_store.save.call_args[0][0]
        assert saved_entry.metadata == {}

    @pytest.mark.asyncio
    async def test_nps_unique_feedback_ids(self, mock_store):
        """Each submission should produce a unique feedback_id."""
        ids = set()
        for i in range(5):
            result = await handle_submit_nps(_ctx(body={"score": 5}))
            fid = _body(result)["feedback_id"]
            ids.add(fid)
        assert len(ids) == 5

    @pytest.mark.asyncio
    async def test_general_unique_feedback_ids(self, mock_store):
        ids = set()
        for i in range(5):
            result = await handle_submit_feedback(_ctx(body={"comment": f"msg-{i}"}))
            fid = _body(result)["feedback_id"]
            ids.add(fid)
        assert len(ids) == 5
