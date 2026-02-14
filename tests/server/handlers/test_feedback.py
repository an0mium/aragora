"""
Tests for aragora.server.handlers.feedback - User Feedback Handlers.

Tests cover:
- FeedbackType enum: values
- FeedbackEntry: dataclass behavior, to_dict
- FeedbackStore: init_db, save, get_nps_summary
- handle_submit_nps: success, invalid score, missing permission
- handle_submit_feedback: success, missing comment, invalid type
- handle_get_nps_summary: success, permission check
- handle_get_feedback_prompts: success, permission check
- _check_permission: auth required, permission denied, granted
- FEEDBACK_ROUTES: route definitions
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.feedback import (
    FEEDBACK_ROUTES,
    FeedbackEntry,
    FeedbackStore,
    FeedbackType,
    _check_permission,
    get_feedback_store,
    handle_get_feedback_prompts,
    handle_get_nps_summary,
    handle_submit_feedback,
    handle_submit_nps,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


def _make_ctx(
    user_id: str = "user-001",
    body: dict | None = None,
    query: dict | None = None,
    permissions: set | None = None,
    roles: set | None = None,
) -> dict[str, Any]:
    """Create a mock context dict for handler functions."""
    ctx: dict[str, Any] = {
        "user_id": user_id,
        "body": body or {},
        "query": query or {},
    }
    if permissions is not None:
        ctx["permissions"] = permissions
    if roles is not None:
        ctx["roles"] = roles
    return ctx


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database for FeedbackStore."""
    return str(tmp_path / "test_feedback.db")


@pytest.fixture
def store(tmp_db):
    """Create a FeedbackStore with temporary database."""
    return FeedbackStore(db_path=tmp_db)


@pytest.fixture
def mock_checker():
    """Mock the permission checker to always allow."""
    mock_decision = MagicMock()
    mock_decision.allowed = True
    checker = MagicMock()
    checker.check_permission.return_value = mock_decision
    return checker


@pytest.fixture(autouse=True)
def _patch_checker(mock_checker):
    """Auto-patch the permission checker."""
    with patch(
        "aragora.server.handlers.feedback.get_permission_checker",
        return_value=mock_checker,
    ):
        yield


# ===========================================================================
# Test FeedbackType Enum
# ===========================================================================


class TestFeedbackType:
    """Test the FeedbackType enum."""

    def test_nps_value(self):
        assert FeedbackType.NPS.value == "nps"

    def test_feature_request_value(self):
        assert FeedbackType.FEATURE_REQUEST.value == "feature_request"

    def test_bug_report_value(self):
        assert FeedbackType.BUG_REPORT.value == "bug_report"

    def test_general_value(self):
        assert FeedbackType.GENERAL.value == "general"

    def test_debate_quality_value(self):
        assert FeedbackType.DEBATE_QUALITY.value == "debate_quality"


# ===========================================================================
# Test FeedbackEntry
# ===========================================================================


class TestFeedbackEntry:
    """Test the FeedbackEntry dataclass."""

    def test_creation(self):
        entry = FeedbackEntry(
            id="fb-001",
            user_id="user-001",
            feedback_type=FeedbackType.NPS,
            score=9,
            comment="Great!",
        )
        assert entry.id == "fb-001"
        assert entry.score == 9
        assert entry.created_at  # Should be auto-set

    def test_to_dict(self):
        entry = FeedbackEntry(
            id="fb-001",
            user_id="user-001",
            feedback_type=FeedbackType.GENERAL,
            score=None,
            comment="Good product",
        )
        d = entry.to_dict()
        assert d["id"] == "fb-001"
        assert d["feedback_type"] == "general"
        assert d["comment"] == "Good product"
        assert "created_at" in d

    def test_default_metadata(self):
        entry = FeedbackEntry(
            id="fb-001",
            user_id="user-001",
            feedback_type=FeedbackType.NPS,
            score=5,
            comment=None,
        )
        assert entry.metadata == {}


# ===========================================================================
# Test FeedbackStore
# ===========================================================================


class TestFeedbackStore:
    """Tests for the SQLite feedback store."""

    def test_init_db(self, store):
        """Store initializes tables without error."""
        assert store is not None
        assert store.db_path is not None

    def test_save_and_retrieve_nps_summary(self, store):
        """Save NPS entries and get summary."""
        for score in [9, 10, 8, 3, 9]:
            entry = FeedbackEntry(
                id=f"fb-{score}",
                user_id="user-001",
                feedback_type=FeedbackType.NPS,
                score=score,
                comment=None,
            )
            store.save(entry)

        summary = store.get_nps_summary(days=30)
        assert summary["total_responses"] == 5
        assert summary["promoters"] == 3  # 9, 10, 9
        assert summary["passives"] == 1  # 8
        assert summary["detractors"] == 1  # 3

    def test_nps_score_calculation(self, store):
        """Test NPS score formula: ((promoters - detractors) / total) * 100."""
        # 2 promoters, 1 detractor, 1 passive -> (2-1)/4 = 25
        for score in [10, 9, 3, 8]:
            entry = FeedbackEntry(
                id=f"fb-calc-{score}",
                user_id="user-001",
                feedback_type=FeedbackType.NPS,
                score=score,
                comment=None,
            )
            store.save(entry)

        summary = store.get_nps_summary(days=30)
        assert summary["nps_score"] == 25

    def test_empty_nps_summary(self, store):
        """Empty store returns zero NPS."""
        summary = store.get_nps_summary(days=30)
        assert summary["total_responses"] == 0
        assert summary["nps_score"] == 0


# ===========================================================================
# Test _check_permission
# ===========================================================================


class TestCheckPermission:
    """Tests for the permission checking helper."""

    def test_no_user_id(self):
        result = _check_permission({}, "feedback.write")
        assert result is not None
        assert result.status_code == 401

    def test_permission_denied(self, mock_checker):
        mock_checker.check_permission.return_value = MagicMock(allowed=False)
        with patch(
            "aragora.server.handlers.feedback.get_permission_checker",
            return_value=mock_checker,
        ):
            result = _check_permission({"user_id": "user-001"}, "feedback.write")
            assert result is not None
            assert result.status_code == 403

    def test_permission_granted(self, mock_checker):
        result = _check_permission({"user_id": "user-001"}, "feedback.write")
        assert result is None


# ===========================================================================
# Test handle_submit_nps
# ===========================================================================


class TestHandleSubmitNps:
    """Tests for POST /api/v1/feedback/nps."""

    @pytest.mark.asyncio
    async def test_submit_nps_success(self, store):
        ctx = _make_ctx(body={"score": 9, "comment": "Love it"})
        with patch("aragora.server.handlers.feedback.get_feedback_store", return_value=store):
            result = await handle_submit_nps(ctx)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["success"] is True
            assert "feedback_id" in data

    @pytest.mark.asyncio
    async def test_submit_nps_invalid_score_too_high(self, store):
        ctx = _make_ctx(body={"score": 11})
        with patch("aragora.server.handlers.feedback.get_feedback_store", return_value=store):
            result = await handle_submit_nps(ctx)
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_nps_invalid_score_negative(self, store):
        ctx = _make_ctx(body={"score": -1})
        with patch("aragora.server.handlers.feedback.get_feedback_store", return_value=store):
            result = await handle_submit_nps(ctx)
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_nps_missing_score(self, store):
        ctx = _make_ctx(body={})
        with patch("aragora.server.handlers.feedback.get_feedback_store", return_value=store):
            result = await handle_submit_nps(ctx)
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_nps_string_score(self, store):
        ctx = _make_ctx(body={"score": "not_a_number"})
        with patch("aragora.server.handlers.feedback.get_feedback_store", return_value=store):
            result = await handle_submit_nps(ctx)
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_nps_no_auth(self):
        """Missing user_id should return 401."""
        ctx = {"body": {"score": 9}}
        result = await handle_submit_nps(ctx)
        assert result.status_code == 401


# ===========================================================================
# Test handle_submit_feedback
# ===========================================================================


class TestHandleSubmitFeedback:
    """Tests for POST /api/v1/feedback/general."""

    @pytest.mark.asyncio
    async def test_submit_feedback_success(self, store):
        ctx = _make_ctx(body={"type": "bug_report", "comment": "Found a bug"})
        with patch("aragora.server.handlers.feedback.get_feedback_store", return_value=store):
            result = await handle_submit_feedback(ctx)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_submit_feedback_missing_comment(self, store):
        ctx = _make_ctx(body={"type": "general"})
        with patch("aragora.server.handlers.feedback.get_feedback_store", return_value=store):
            result = await handle_submit_feedback(ctx)
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_feedback_invalid_type_defaults_to_general(self, store):
        ctx = _make_ctx(body={"type": "invalid_type", "comment": "Some feedback"})
        with patch("aragora.server.handlers.feedback.get_feedback_store", return_value=store):
            result = await handle_submit_feedback(ctx)
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_submit_feedback_with_context(self, store):
        ctx = _make_ctx(
            body={
                "type": "feature_request",
                "comment": "Need dark mode",
                "context": {"page": "dashboard"},
            }
        )
        with patch("aragora.server.handlers.feedback.get_feedback_store", return_value=store):
            result = await handle_submit_feedback(ctx)
            assert result.status_code == 200


# ===========================================================================
# Test handle_get_nps_summary
# ===========================================================================


class TestHandleGetNpsSummary:
    """Tests for GET /api/v1/feedback/nps/summary."""

    @pytest.mark.asyncio
    async def test_get_nps_summary_success(self, store):
        ctx = _make_ctx(query={"days": "30"})
        with patch("aragora.server.handlers.feedback.get_feedback_store", return_value=store):
            result = await handle_get_nps_summary(ctx)
            assert result.status_code == 200
            data = _parse_body(result)
            assert "nps_score" in data
            assert "total_responses" in data


# ===========================================================================
# Test handle_get_feedback_prompts
# ===========================================================================


class TestHandleGetFeedbackPrompts:
    """Tests for GET /api/v1/feedback/prompts."""

    @pytest.mark.asyncio
    async def test_get_prompts_success(self):
        ctx = _make_ctx()
        result = await handle_get_feedback_prompts(ctx)
        assert result.status_code == 200
        data = _parse_body(result)
        assert "prompts" in data
        assert len(data["prompts"]) > 0

    @pytest.mark.asyncio
    async def test_prompts_contain_nps(self):
        ctx = _make_ctx()
        result = await handle_get_feedback_prompts(ctx)
        data = _parse_body(result)
        nps_prompts = [p for p in data["prompts"] if p["type"] == "nps"]
        assert len(nps_prompts) == 1


# ===========================================================================
# Test FEEDBACK_ROUTES
# ===========================================================================


class TestFeedbackRoutes:
    """Test the route definitions."""

    def test_routes_defined(self):
        assert len(FEEDBACK_ROUTES) == 4

    def test_nps_route(self):
        methods = [r[0] for r in FEEDBACK_ROUTES]
        paths = [r[1] for r in FEEDBACK_ROUTES]
        assert "POST" in methods
        assert "/api/v1/feedback/nps" in paths

    def test_general_route(self):
        paths = [r[1] for r in FEEDBACK_ROUTES]
        assert "/api/v1/feedback/general" in paths

    def test_summary_route(self):
        paths = [r[1] for r in FEEDBACK_ROUTES]
        assert "/api/v1/feedback/nps/summary" in paths

    def test_prompts_route(self):
        paths = [r[1] for r in FEEDBACK_ROUTES]
        assert "/api/v1/feedback/prompts" in paths

    def test_all_routes_have_handlers(self):
        for method, path, handler_fn in FEEDBACK_ROUTES:
            assert callable(handler_fn)
