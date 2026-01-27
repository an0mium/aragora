"""
User Feedback Collection Handler.

Provides endpoints for collecting user feedback including:
- NPS (Net Promoter Score) surveys
- Feature feedback
- Bug reports
- General suggestions

Endpoints:
- POST /api/v1/feedback/nps - Submit NPS score (requires feedback.write)
- POST /api/v1/feedback/general - Submit general feedback (requires feedback.write)
- GET /api/v1/feedback/nps/summary - Get NPS summary (requires feedback.update - admin)
- GET /api/v1/feedback/prompts - Get active feedback prompts (requires feedback.read)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from aragora.config import resolve_db_path
from aragora.rbac.checker import get_permission_checker
from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.base import (
    HandlerResult,
    ServerContext,
    error_response,
    json_response,
)

logger = logging.getLogger(__name__)


def _check_permission(ctx: ServerContext, permission: str) -> Optional[HandlerResult]:
    """
    Check if the current user has the required permission.

    Args:
        ctx: Server context with user_id
        permission: Permission key to check (e.g., "feedback.write")

    Returns:
        None if permission granted, error_response if denied
    """
    user_id = ctx.get("user_id")
    if not user_id:
        return error_response("Authentication required", status=401)

    # Build minimal auth context from server context
    auth_context = AuthorizationContext(
        user_id=user_id,
        org_id=ctx.get("org_id"),  # type: ignore[arg-type]
        roles=ctx.get("roles", set()),  # type: ignore[arg-type]
        permissions=ctx.get("permissions", set()),  # type: ignore[arg-type]
    )

    checker = get_permission_checker()
    decision = checker.check_permission(auth_context, permission)

    if not decision.allowed:
        logger.warning(f"Permission denied: {permission} for user {user_id}")
        return error_response(f"Permission denied: {permission}", status=403)

    return None


class FeedbackType(str, Enum):
    """Types of feedback."""

    NPS = "nps"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    GENERAL = "general"
    DEBATE_QUALITY = "debate_quality"


@dataclass
class FeedbackEntry:
    """A feedback submission."""

    id: str
    user_id: Optional[str]
    feedback_type: FeedbackType
    score: Optional[int]  # For NPS: 0-10
    comment: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "feedback_type": self.feedback_type.value,
            "score": self.score,
            "comment": self.comment,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


class FeedbackStore:
    """SQLite store for user feedback."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or resolve_db_path("feedback.db")
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    feedback_type TEXT NOT NULL,
                    score INTEGER,
                    comment TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_type
                ON feedback(feedback_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_user
                ON feedback(user_id)
            """)

    def save(self, entry: FeedbackEntry) -> None:
        """Save feedback entry."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO feedback (id, user_id, feedback_type, score, comment, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.user_id,
                    entry.feedback_type.value,
                    entry.score,
                    entry.comment,
                    json.dumps(entry.metadata),
                    entry.created_at,
                ),
            )

    def get_nps_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get NPS summary for the last N days."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT score, COUNT(*) as count
                FROM feedback
                WHERE feedback_type = 'nps'
                AND created_at >= datetime('now', ?)
                GROUP BY score
                """,
                (f"-{days} days",),
            )
            scores = {row["score"]: row["count"] for row in cursor.fetchall()}

        # Calculate NPS
        promoters = sum(scores.get(s, 0) for s in [9, 10])
        passives = sum(scores.get(s, 0) for s in [7, 8])
        detractors = sum(scores.get(s, 0) for s in range(0, 7))
        total = promoters + passives + detractors

        nps = 0
        if total > 0:
            nps = round(((promoters - detractors) / total) * 100)

        return {
            "nps_score": nps,
            "total_responses": total,
            "promoters": promoters,
            "passives": passives,
            "detractors": detractors,
            "period_days": days,
        }


# Global store instance
_feedback_store: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    """Get or create the feedback store."""
    global _feedback_store
    if _feedback_store is None:
        _feedback_store = FeedbackStore()
    return _feedback_store


async def handle_submit_nps(ctx: ServerContext) -> HandlerResult:
    """
    Submit NPS feedback.

    POST /api/v1/feedback/nps

    Body:
        score: int (0-10, required)
        comment: str (optional)
        context: dict (optional metadata)

    Returns:
        {"success": true, "feedback_id": "..."}

    Requires: feedback.write permission
    """
    # Check permission
    perm_error = _check_permission(ctx, "feedback.write")
    if perm_error:
        return perm_error

    try:
        body = ctx.get("body", {})
        user_id = ctx.get("user_id", "anonymous")
        score = body.get("score")

        if score is None or not isinstance(score, int) or not 0 <= score <= 10:
            return error_response("Score must be an integer between 0 and 10", status=400)

        entry = FeedbackEntry(
            id=str(uuid.uuid4()),
            user_id=user_id,
            feedback_type=FeedbackType.NPS,
            score=score,
            comment=body.get("comment"),
            metadata=body.get("context", {}),
        )

        store = get_feedback_store()
        store.save(entry)

        logger.info(f"NPS feedback submitted: score={score}, user={user_id}")

        return json_response(
            {
                "success": True,
                "feedback_id": entry.id,
                "message": "Thank you for your feedback!",
            }
        )

    except Exception as e:
        logger.error(f"Error submitting NPS feedback: {e}")
        return error_response(str(e), status=500)


async def handle_submit_feedback(ctx: ServerContext) -> HandlerResult:
    """
    Submit general feedback.

    POST /api/v1/feedback/general

    Body:
        type: str (feature_request, bug_report, general, debate_quality)
        comment: str (required)
        score: int (optional, for ratings)
        context: dict (optional metadata)

    Returns:
        {"success": true, "feedback_id": "..."}

    Requires: feedback.write permission
    """
    # Check permission
    perm_error = _check_permission(ctx, "feedback.write")
    if perm_error:
        return perm_error

    try:
        body = ctx.get("body", {})
        user_id = ctx.get("user_id", "anonymous")

        feedback_type_str = body.get("type", "general")
        try:
            feedback_type = FeedbackType(feedback_type_str)
        except ValueError:
            feedback_type = FeedbackType.GENERAL

        comment = body.get("comment")
        if not comment:
            return error_response("Comment is required", status=400)

        entry = FeedbackEntry(
            id=str(uuid.uuid4()),
            user_id=user_id,
            feedback_type=feedback_type,
            score=body.get("score"),
            comment=comment,
            metadata=body.get("context", {}),
        )

        store = get_feedback_store()
        store.save(entry)

        logger.info(f"Feedback submitted: type={feedback_type.value}, user={user_id}")

        return json_response(
            {
                "success": True,
                "feedback_id": entry.id,
                "message": "Thank you for your feedback!",
            }
        )

    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return error_response(str(e), status=500)


async def handle_get_nps_summary(ctx: ServerContext) -> HandlerResult:
    """
    Get NPS summary (admin only).

    GET /api/v1/feedback/nps/summary

    Query params:
        days: int (default 30)

    Returns:
        {"nps_score": ..., "total_responses": ..., ...}

    Requires: feedback.update permission (admin only)
    """
    # Check permission - admin only (feedback.update maps to PERM_FEEDBACK_ALL)
    perm_error = _check_permission(ctx, "feedback.update")
    if perm_error:
        return perm_error

    try:
        query = ctx.get("query", {})
        days = int(query.get("days", 30))
        store = get_feedback_store()
        summary = store.get_nps_summary(days)
        return json_response(summary)

    except Exception as e:
        logger.error(f"Error getting NPS summary: {e}")
        return error_response(str(e), status=500)


async def handle_get_feedback_prompts(ctx: ServerContext) -> HandlerResult:
    """
    Get active feedback prompts for the user.

    GET /api/v1/feedback/prompts

    Returns prompts based on user activity and timing.

    Requires: feedback.read permission
    """
    # Check permission
    perm_error = _check_permission(ctx, "feedback.read")
    if perm_error:
        return perm_error

    # Simple prompt logic - can be expanded
    prompts = []

    # NPS prompt (show periodically)
    prompts.append(
        {
            "type": "nps",
            "question": "How likely are you to recommend Aragora to a colleague?",
            "scale": {"min": 0, "max": 10, "labels": {"0": "Not likely", "10": "Very likely"}},
            "follow_up": "What's the main reason for your score?",
        }
    )

    return json_response({"prompts": prompts})


# Route definitions for registration
FEEDBACK_ROUTES = [
    ("POST", "/api/v1/feedback/nps", handle_submit_nps),
    ("POST", "/api/v1/feedback/general", handle_submit_feedback),
    ("GET", "/api/v1/feedback/nps/summary", handle_get_nps_summary),
    ("GET", "/api/v1/feedback/prompts", handle_get_feedback_prompts),
]
