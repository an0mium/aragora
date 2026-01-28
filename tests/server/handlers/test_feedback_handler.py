"""
Tests for aragora.server.handlers.feedback - User Feedback Handler.

Tests cover:
- POST /api/v1/feedback/nps - Submit NPS feedback
- POST /api/v1/feedback/general - Submit general feedback
- GET /api/v1/feedback/nps/summary - Get NPS summary (admin)
- GET /api/v1/feedback/prompts - Get feedback prompts
- RBAC permission enforcement
"""

from __future__ import annotations

import json
import tempfile
from typing import Any, Dict, Optional, Set
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.feedback import (
    FeedbackEntry,
    FeedbackStore,
    FeedbackType,
    handle_get_feedback_prompts,
    handle_get_nps_summary,
    handle_submit_feedback,
    handle_submit_nps,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def mock_permission_checker():
    """Mock the permission checker for controlled testing."""
    with patch("aragora.server.handlers.feedback.get_permission_checker") as mock_get:
        mock_checker = MagicMock()
        mock_get.return_value = mock_checker
        yield mock_checker


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name


@pytest.fixture
def feedback_store(temp_db):
    """Create a FeedbackStore with a temporary database."""
    return FeedbackStore(db_path=temp_db)


def make_server_context(
    user_id: Optional[str] = "test-user-001",
    org_id: Optional[str] = "org-001",
    roles: Optional[Set[str]] = None,
    permissions: Optional[Set[str]] = None,
    body: Optional[Dict[str, Any]] = None,
    query: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a mock server context for testing."""
    return {
        "user_id": user_id,
        "org_id": org_id,
        "roles": roles or {"user"},
        "permissions": permissions or set(),
        "body": body or {},
        "query": query or {},
    }


# ===========================================================================
# FeedbackEntry Tests
# ===========================================================================


class TestFeedbackEntry:
    """Tests for FeedbackEntry dataclass."""

    def test_create_entry(self):
        """Test creating a feedback entry."""
        entry = FeedbackEntry(
            id="test-id",
            user_id="user-123",
            feedback_type=FeedbackType.NPS,
            score=9,
            comment="Great product!",
        )

        assert entry.id == "test-id"
        assert entry.user_id == "user-123"
        assert entry.feedback_type == FeedbackType.NPS
        assert entry.score == 9
        assert entry.comment == "Great product!"
        assert entry.created_at != ""

    def test_to_dict(self):
        """Test converting entry to dictionary."""
        entry = FeedbackEntry(
            id="test-id",
            user_id="user-123",
            feedback_type=FeedbackType.BUG_REPORT,
            score=None,
            comment="Found a bug",
            metadata={"page": "/dashboard"},
        )

        result = entry.to_dict()

        assert result["id"] == "test-id"
        assert result["feedback_type"] == "bug_report"
        assert result["metadata"]["page"] == "/dashboard"


# ===========================================================================
# FeedbackStore Tests
# ===========================================================================


class TestFeedbackStore:
    """Tests for FeedbackStore."""

    def test_save_entry(self, feedback_store):
        """Test saving a feedback entry."""
        entry = FeedbackEntry(
            id="entry-001",
            user_id="user-123",
            feedback_type=FeedbackType.NPS,
            score=8,
            comment="Good!",
        )

        feedback_store.save(entry)
        # No exception means success

    def test_get_nps_summary_empty(self, feedback_store):
        """Test NPS summary with no data."""
        summary = feedback_store.get_nps_summary(days=30)

        assert summary["nps_score"] == 0
        assert summary["total_responses"] == 0
        assert summary["promoters"] == 0
        assert summary["passives"] == 0
        assert summary["detractors"] == 0

    def test_get_nps_summary_with_data(self, feedback_store):
        """Test NPS summary with feedback data."""
        # Add promoters (9, 10)
        for i, score in enumerate([9, 10, 10]):
            entry = FeedbackEntry(
                id=f"promo-{i}",
                user_id="user-123",
                feedback_type=FeedbackType.NPS,
                score=score,
                comment=None,
            )
            feedback_store.save(entry)

        # Add passives (7, 8)
        entry = FeedbackEntry(
            id="passive-1",
            user_id="user-456",
            feedback_type=FeedbackType.NPS,
            score=7,
            comment=None,
        )
        feedback_store.save(entry)

        # Add detractors (0-6)
        entry = FeedbackEntry(
            id="detract-1",
            user_id="user-789",
            feedback_type=FeedbackType.NPS,
            score=3,
            comment="Not great",
        )
        feedback_store.save(entry)

        summary = feedback_store.get_nps_summary(days=30)

        assert summary["total_responses"] == 5
        assert summary["promoters"] == 3
        assert summary["passives"] == 1
        assert summary["detractors"] == 1
        # NPS = ((3 - 1) / 5) * 100 = 40
        assert summary["nps_score"] == 40


# ===========================================================================
# Submit NPS Tests
# ===========================================================================


class TestSubmitNPS:
    """Tests for POST /api/v1/feedback/nps."""

    @pytest.mark.asyncio
    async def test_submit_nps_success(self, mock_permission_checker, temp_db):
        """Test successful NPS submission."""
        # Allow permission
        mock_permission_checker.check_permission.return_value = MagicMock(allowed=True)

        ctx = make_server_context(body={"score": 9, "comment": "Love it!"})

        with patch("aragora.server.handlers.feedback.get_feedback_store") as mock_store:
            mock_store.return_value = FeedbackStore(db_path=temp_db)
            result = await handle_submit_nps(ctx)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert "feedback_id" in body

    @pytest.mark.asyncio
    async def test_submit_nps_permission_denied(self, mock_permission_checker):
        """Test NPS submission without permission."""
        mock_permission_checker.check_permission.return_value = MagicMock(
            allowed=False, reason="No feedback.write permission"
        )

        ctx = make_server_context(body={"score": 9})

        result = await handle_submit_nps(ctx)

        assert result.status_code == 403
        body = json.loads(result.body)
        assert "Permission denied" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_submit_nps_no_user(self, mock_permission_checker):
        """Test NPS submission without authenticated user."""
        ctx = make_server_context(user_id=None, body={"score": 9})

        result = await handle_submit_nps(ctx)

        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_submit_nps_invalid_score(self, mock_permission_checker, temp_db):
        """Test NPS submission with invalid score."""
        mock_permission_checker.check_permission.return_value = MagicMock(allowed=True)

        ctx = make_server_context(body={"score": 15})  # Invalid: > 10

        result = await handle_submit_nps(ctx)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "Score must be" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_submit_nps_missing_score(self, mock_permission_checker):
        """Test NPS submission without score."""
        mock_permission_checker.check_permission.return_value = MagicMock(allowed=True)

        ctx = make_server_context(body={})  # No score

        result = await handle_submit_nps(ctx)

        assert result.status_code == 400


# ===========================================================================
# Submit General Feedback Tests
# ===========================================================================


class TestSubmitFeedback:
    """Tests for POST /api/v1/feedback/general."""

    @pytest.mark.asyncio
    async def test_submit_feedback_success(self, mock_permission_checker, temp_db):
        """Test successful general feedback submission."""
        mock_permission_checker.check_permission.return_value = MagicMock(allowed=True)

        ctx = make_server_context(
            body={
                "type": "feature_request",
                "comment": "Would love dark mode!",
                "context": {"page": "/settings"},
            }
        )

        with patch("aragora.server.handlers.feedback.get_feedback_store") as mock_store:
            mock_store.return_value = FeedbackStore(db_path=temp_db)
            result = await handle_submit_feedback(ctx)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert "feedback_id" in body

    @pytest.mark.asyncio
    async def test_submit_feedback_permission_denied(self, mock_permission_checker):
        """Test feedback submission without permission."""
        mock_permission_checker.check_permission.return_value = MagicMock(
            allowed=False, reason="No feedback.write permission"
        )

        ctx = make_server_context(body={"type": "bug_report", "comment": "Bug found"})

        result = await handle_submit_feedback(ctx)

        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_submit_feedback_missing_comment(self, mock_permission_checker):
        """Test feedback submission without comment."""
        mock_permission_checker.check_permission.return_value = MagicMock(allowed=True)

        ctx = make_server_context(body={"type": "general"})  # No comment

        result = await handle_submit_feedback(ctx)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "Comment is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_submit_feedback_invalid_type(self, mock_permission_checker, temp_db):
        """Test feedback submission with unknown type defaults to general."""
        mock_permission_checker.check_permission.return_value = MagicMock(allowed=True)

        ctx = make_server_context(body={"type": "unknown_type", "comment": "Some feedback"})

        with patch("aragora.server.handlers.feedback.get_feedback_store") as mock_store:
            mock_store.return_value = FeedbackStore(db_path=temp_db)
            result = await handle_submit_feedback(ctx)

        # Should succeed with type defaulting to "general"
        assert result.status_code == 200


# ===========================================================================
# Get NPS Summary Tests
# ===========================================================================


class TestGetNPSSummary:
    """Tests for GET /api/v1/feedback/nps/summary."""

    @pytest.mark.asyncio
    async def test_get_summary_success(self, mock_permission_checker, temp_db):
        """Test successful NPS summary retrieval."""
        mock_permission_checker.check_permission.return_value = MagicMock(allowed=True)

        ctx = make_server_context(
            roles={"admin"},
            permissions={"feedback.update"},
            query={"days": "7"},
        )

        with patch("aragora.server.handlers.feedback.get_feedback_store") as mock_store:
            mock_store.return_value = FeedbackStore(db_path=temp_db)
            result = await handle_get_nps_summary(ctx)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "nps_score" in body
        assert "total_responses" in body
        assert body["period_days"] == 7

    @pytest.mark.asyncio
    async def test_get_summary_permission_denied(self, mock_permission_checker):
        """Test NPS summary access without admin permission."""
        mock_permission_checker.check_permission.return_value = MagicMock(
            allowed=False, reason="Requires feedback.update permission"
        )

        ctx = make_server_context(roles={"user"})

        result = await handle_get_nps_summary(ctx)

        assert result.status_code == 403


# ===========================================================================
# Get Feedback Prompts Tests
# ===========================================================================


class TestGetFeedbackPrompts:
    """Tests for GET /api/v1/feedback/prompts."""

    @pytest.mark.asyncio
    async def test_get_prompts_success(self, mock_permission_checker):
        """Test successful prompts retrieval."""
        mock_permission_checker.check_permission.return_value = MagicMock(allowed=True)

        ctx = make_server_context()

        result = await handle_get_feedback_prompts(ctx)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "prompts" in body
        assert len(body["prompts"]) > 0

        # Verify NPS prompt structure
        nps_prompt = body["prompts"][0]
        assert nps_prompt["type"] == "nps"
        assert "question" in nps_prompt
        assert "scale" in nps_prompt

    @pytest.mark.asyncio
    async def test_get_prompts_permission_denied(self, mock_permission_checker):
        """Test prompts access without permission."""
        mock_permission_checker.check_permission.return_value = MagicMock(
            allowed=False, reason="No feedback.read permission"
        )

        ctx = make_server_context()

        result = await handle_get_feedback_prompts(ctx)

        assert result.status_code == 403


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestFeedbackIntegration:
    """Integration tests for feedback flow."""

    @pytest.mark.asyncio
    async def test_full_nps_flow(self, mock_permission_checker, temp_db):
        """Test complete NPS submission and summary flow."""
        mock_permission_checker.check_permission.return_value = MagicMock(allowed=True)

        store = FeedbackStore(db_path=temp_db)

        with patch("aragora.server.handlers.feedback.get_feedback_store") as mock_store:
            mock_store.return_value = store

            # Submit several NPS scores
            for score in [10, 9, 8, 7, 5, 3]:
                ctx = make_server_context(body={"score": score})
                result = await handle_submit_nps(ctx)
                assert result.status_code == 200

            # Get summary
            ctx = make_server_context(
                roles={"admin"},
                permissions={"feedback.update"},
                query={"days": "30"},
            )
            result = await handle_get_nps_summary(ctx)

            body = json.loads(result.body)
            assert body["total_responses"] == 6
            assert body["promoters"] == 2  # 10, 9
            assert body["passives"] == 2  # 8, 7
            assert body["detractors"] == 2  # 5, 3


# ===========================================================================
# FeedbackType Enum Tests
# ===========================================================================


class TestFeedbackType:
    """Tests for FeedbackType enum."""

    def test_all_types_defined(self):
        """Test all feedback types are defined."""
        assert FeedbackType.NPS.value == "nps"
        assert FeedbackType.FEATURE_REQUEST.value == "feature_request"
        assert FeedbackType.BUG_REPORT.value == "bug_report"
        assert FeedbackType.GENERAL.value == "general"
        assert FeedbackType.DEBATE_QUALITY.value == "debate_quality"

    def test_type_from_string(self):
        """Test creating type from string."""
        assert FeedbackType("nps") == FeedbackType.NPS
        assert FeedbackType("bug_report") == FeedbackType.BUG_REPORT
