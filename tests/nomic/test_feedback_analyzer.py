"""Tests for the FeedbackAnalyzer -- user feedback to self-improvement bridge.

Verifies that user feedback (NPS, bugs, feature requests, debate quality)
is correctly categorized, deduplicated, and converted into improvement goals
for the Nomic Loop.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.nomic.feedback_analyzer import (
    AnalysisResult,
    FeedbackAnalyzer,
    FeedbackItem,
    _ProcessingStateStore,
    process_new_feedback,
)
from aragora.nomic.feedback_orchestrator import ImprovementGoal, ImprovementQueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_feedback_db(db_path: Path, entries: list[dict]) -> str:
    """Create a feedback SQLite database with the given entries."""
    fp = str(db_path / "feedback.db")
    conn = sqlite3.connect(fp)
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
    for entry in entries:
        conn.execute(
            "INSERT INTO feedback (id, user_id, feedback_type, score, comment, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                entry["id"],
                entry.get("user_id"),
                entry["feedback_type"],
                entry.get("score"),
                entry.get("comment"),
                json.dumps(entry.get("metadata", {})),
                entry.get("created_at", "2026-02-23T12:00:00+00:00"),
            ),
        )
    conn.commit()
    conn.close()
    return fp


def _make_analyzer(tmp_path: Path, feedback_db: str) -> FeedbackAnalyzer:
    """Create a FeedbackAnalyzer with isolated databases."""
    return FeedbackAnalyzer(
        feedback_db_path=feedback_db,
        state_db_path=tmp_path / "analyzer_state.db",
        queue_db_path=tmp_path / "improvement_queue.db",
    )


def _get_queued_goals(tmp_path: Path) -> list[ImprovementGoal]:
    """Peek all goals from the improvement queue."""
    queue = ImprovementQueue(db_path=tmp_path / "improvement_queue.db")
    return queue.peek(limit=100)


# ---------------------------------------------------------------------------
# Tests: Category mapping
# ---------------------------------------------------------------------------


class TestCategoryMapping:
    """Test that feedback types map to the correct improvement categories."""

    def test_bug_report_becomes_reliability_goal(self, tmp_path: Path):
        """Bug reports should create improvement goals with category=reliability."""
        db = _create_feedback_db(tmp_path, [
            {
                "id": "bug-1",
                "feedback_type": "bug_report",
                "comment": "The debate crashes when I submit long prompts",
            },
        ])
        analyzer = _make_analyzer(tmp_path, db)
        result = analyzer.process_new_feedback()

        assert result.goals_created == 1
        goals = _get_queued_goals(tmp_path)
        assert len(goals) == 1
        assert goals[0].context["category"] == "reliability"
        assert goals[0].source == "user_feedback"
        assert "bug" in goals[0].goal.lower() or "fix" in goals[0].goal.lower()

    def test_feature_request_becomes_features_goal(self, tmp_path: Path):
        """Feature requests should create goals with category=features."""
        db = _create_feedback_db(tmp_path, [
            {
                "id": "feat-1",
                "feedback_type": "feature_request",
                "comment": "Please add Slack integration for debate results",
            },
        ])
        analyzer = _make_analyzer(tmp_path, db)
        result = analyzer.process_new_feedback()

        assert result.goals_created == 1
        goals = _get_queued_goals(tmp_path)
        assert goals[0].context["category"] == "features"
        assert "feature" in goals[0].goal.lower() or "implement" in goals[0].goal.lower()

    def test_debate_quality_low_score_becomes_accuracy_goal(self, tmp_path: Path):
        """Low debate quality ratings should create accuracy improvement goals."""
        db = _create_feedback_db(tmp_path, [
            {
                "id": "dq-1",
                "feedback_type": "debate_quality",
                "score": 2,
                "comment": "The agents gave incorrect answers about Python asyncio",
            },
        ])
        analyzer = _make_analyzer(tmp_path, db)
        result = analyzer.process_new_feedback()

        assert result.goals_created == 1
        goals = _get_queued_goals(tmp_path)
        assert goals[0].context["category"] == "accuracy"
        assert goals[0].context["feedback_type"] == "debate_quality"

    def test_low_nps_triggers_investigation_goal(self, tmp_path: Path):
        """NPS scores <= 5 should trigger investigation goals."""
        db = _create_feedback_db(tmp_path, [
            {
                "id": "nps-1",
                "feedback_type": "nps",
                "score": 3,
                "comment": "Too slow and confusing",
            },
        ])
        analyzer = _make_analyzer(tmp_path, db)
        result = analyzer.process_new_feedback()

        assert result.goals_created == 1
        goals = _get_queued_goals(tmp_path)
        assert "nps" in goals[0].goal.lower() or "investigate" in goals[0].goal.lower()
        assert goals[0].context["score"] == 3

    def test_high_nps_without_comment_skipped(self, tmp_path: Path):
        """High NPS scores without comments produce no actionable goals."""
        db = _create_feedback_db(tmp_path, [
            {
                "id": "nps-high-1",
                "feedback_type": "nps",
                "score": 10,
                "comment": None,
            },
        ])
        analyzer = _make_analyzer(tmp_path, db)
        result = analyzer.process_new_feedback()

        assert result.goals_created == 0
        assert result.feedback_processed == 1


# ---------------------------------------------------------------------------
# Tests: Keyword-based category refinement
# ---------------------------------------------------------------------------


class TestKeywordRefinement:
    """Test that comment keywords override default category assignment."""

    def test_performance_keywords_override(self, tmp_path: Path):
        """Comments mentioning 'slow' should be categorized as performance."""
        db = _create_feedback_db(tmp_path, [
            {
                "id": "gen-perf-1",
                "feedback_type": "general",
                "comment": "The debates are too slow, takes minutes to get results",
            },
        ])
        analyzer = _make_analyzer(tmp_path, db)
        result = analyzer.process_new_feedback()

        assert result.goals_created == 1
        goals = _get_queued_goals(tmp_path)
        assert goals[0].context["category"] == "performance"

    def test_crash_keyword_maps_to_reliability(self, tmp_path: Path):
        """Comments mentioning 'crash' should be categorized as reliability."""
        db = _create_feedback_db(tmp_path, [
            {
                "id": "gen-crash-1",
                "feedback_type": "general",
                "comment": "The app crashes when I try to export results",
            },
        ])
        analyzer = _make_analyzer(tmp_path, db)
        result = analyzer.process_new_feedback()

        goals = _get_queued_goals(tmp_path)
        assert goals[0].context["category"] == "reliability"


# ---------------------------------------------------------------------------
# Tests: Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Test that duplicate feedback is detected and skipped."""

    def test_duplicate_feedback_skipped(self, tmp_path: Path):
        """Near-identical feedback items should be deduplicated."""
        db = _create_feedback_db(tmp_path, [
            {
                "id": "dup-1",
                "feedback_type": "bug_report",
                "comment": "The debate crashes when I submit long prompts",
            },
            {
                "id": "dup-2",
                "feedback_type": "bug_report",
                "comment": "The debate crashes when I submit long prompts!",
            },
        ])
        analyzer = _make_analyzer(tmp_path, db)
        result = analyzer.process_new_feedback()

        # First one should create a goal, second should be deduplicated
        assert result.goals_created == 1
        assert result.duplicates_skipped == 1

    def test_different_feedback_not_deduplicated(self, tmp_path: Path):
        """Distinct feedback items should each produce goals."""
        db = _create_feedback_db(tmp_path, [
            {
                "id": "diff-1",
                "feedback_type": "bug_report",
                "comment": "The debate crashes on long prompts",
            },
            {
                "id": "diff-2",
                "feedback_type": "feature_request",
                "comment": "Please add dark mode to the dashboard",
            },
        ])
        analyzer = _make_analyzer(tmp_path, db)
        result = analyzer.process_new_feedback()

        assert result.goals_created == 2
        assert result.duplicates_skipped == 0


# ---------------------------------------------------------------------------
# Tests: Processing state tracking
# ---------------------------------------------------------------------------


class TestProcessingState:
    """Test that already-processed feedback is not reprocessed."""

    def test_already_processed_feedback_skipped(self, tmp_path: Path):
        """Running the analyzer twice should not create duplicate goals."""
        db = _create_feedback_db(tmp_path, [
            {
                "id": "once-1",
                "feedback_type": "bug_report",
                "comment": "Login form is broken on mobile",
            },
        ])
        analyzer = _make_analyzer(tmp_path, db)

        result1 = analyzer.process_new_feedback()
        assert result1.goals_created == 1

        # Second run: already processed
        result2 = analyzer.process_new_feedback()
        assert result2.goals_created == 0
        assert result2.feedback_processed == 0  # None fetched (all processed)

    def test_processing_state_persists(self, tmp_path: Path):
        """Processing state should survive across FeedbackAnalyzer instances."""
        db = _create_feedback_db(tmp_path, [
            {
                "id": "persist-1",
                "feedback_type": "feature_request",
                "comment": "Add CSV export",
            },
        ])

        # First instance processes it
        analyzer1 = _make_analyzer(tmp_path, db)
        result1 = analyzer1.process_new_feedback()
        assert result1.goals_created == 1

        # New instance with same state DB should skip it
        analyzer2 = _make_analyzer(tmp_path, db)
        result2 = analyzer2.process_new_feedback()
        assert result2.goals_created == 0


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test graceful handling of empty, invalid, and edge-case feedback."""

    def test_empty_feedback_handled_gracefully(self, tmp_path: Path):
        """Feedback with no comment and no score should be skipped."""
        db = _create_feedback_db(tmp_path, [
            {
                "id": "empty-1",
                "feedback_type": "general",
                "comment": None,
                "score": None,
            },
        ])
        analyzer = _make_analyzer(tmp_path, db)
        result = analyzer.process_new_feedback()

        assert result.goals_created == 0
        assert result.feedback_processed == 1
        assert result.errors == []

    def test_no_feedback_returns_empty_result(self, tmp_path: Path):
        """Processing with no feedback in DB returns clean empty result."""
        db = _create_feedback_db(tmp_path, [])
        analyzer = _make_analyzer(tmp_path, db)
        result = analyzer.process_new_feedback()

        assert result.feedback_processed == 0
        assert result.goals_created == 0
        assert result.duplicates_skipped == 0
        assert result.errors == []

    def test_missing_feedback_db_returns_empty(self, tmp_path: Path):
        """If the feedback database doesn't exist, return empty result."""
        analyzer = FeedbackAnalyzer(
            feedback_db_path=str(tmp_path / "nonexistent.db"),
            state_db_path=tmp_path / "state.db",
            queue_db_path=tmp_path / "queue.db",
        )
        result = analyzer.process_new_feedback()

        assert result.feedback_processed == 0
        assert result.goals_created == 0

    def test_invalid_metadata_handled(self, tmp_path: Path):
        """Feedback with malformed JSON metadata should still be processed."""
        fp = str(tmp_path / "feedback.db")
        conn = sqlite3.connect(fp)
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
        conn.execute(
            "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("bad-meta-1", "user1", "bug_report", None, "Something is broken", "not-json{{{", "2026-02-23T12:00:00"),
        )
        conn.commit()
        conn.close()

        analyzer = _make_analyzer(tmp_path, fp)
        result = analyzer.process_new_feedback()

        assert result.goals_created == 1
        assert result.errors == []


# ---------------------------------------------------------------------------
# Tests: Goal content and source
# ---------------------------------------------------------------------------


class TestGoalContent:
    """Test that generated goals have correct source and content."""

    def test_goal_source_is_user_feedback(self, tmp_path: Path):
        """All goals from this analyzer should have source='user_feedback'."""
        db = _create_feedback_db(tmp_path, [
            {
                "id": "src-1",
                "feedback_type": "bug_report",
                "comment": "Memory leak in long debates",
            },
        ])
        analyzer = _make_analyzer(tmp_path, db)
        analyzer.process_new_feedback()

        goals = _get_queued_goals(tmp_path)
        assert all(g.source == "user_feedback" for g in goals)

    def test_goal_tracks_feedback_id(self, tmp_path: Path):
        """Goal context should include the originating feedback_id."""
        db = _create_feedback_db(tmp_path, [
            {
                "id": "track-me-123",
                "feedback_type": "feature_request",
                "comment": "Add PDF export",
            },
        ])
        analyzer = _make_analyzer(tmp_path, db)
        analyzer.process_new_feedback()

        goals = _get_queued_goals(tmp_path)
        assert goals[0].context["feedback_id"] == "track-me-123"

    def test_goal_priority_reflects_category(self, tmp_path: Path):
        """Reliability goals should have higher priority than feature goals."""
        db = _create_feedback_db(tmp_path, [
            {
                "id": "pri-bug",
                "feedback_type": "bug_report",
                "comment": "Critical crash in debate engine",
            },
            {
                "id": "pri-feat",
                "feedback_type": "feature_request",
                "comment": "Add emoji reactions to debate outcomes",
            },
        ])
        analyzer = _make_analyzer(tmp_path, db)
        analyzer.process_new_feedback()

        goals = _get_queued_goals(tmp_path)
        # Goals are ordered by priority DESC in peek
        bug_goal = next(g for g in goals if g.context["feedback_id"] == "pri-bug")
        feat_goal = next(g for g in goals if g.context["feedback_id"] == "pri-feat")
        # Reliability (1.0) > features (0.4)
        assert bug_goal.priority > feat_goal.priority


# ---------------------------------------------------------------------------
# Tests: Convenience function
# ---------------------------------------------------------------------------


class TestConvenienceFunction:
    """Test the module-level process_new_feedback() helper."""

    def test_process_new_feedback_function(self, tmp_path: Path):
        """The convenience function should work end-to-end."""
        db = _create_feedback_db(tmp_path, [
            {
                "id": "conv-1",
                "feedback_type": "general",
                "comment": "Great product but needs better docs",
            },
        ])
        result = process_new_feedback(
            feedback_db_path=db,
            state_db_path=tmp_path / "state.db",
            queue_db_path=tmp_path / "queue.db",
        )

        assert isinstance(result, AnalysisResult)
        assert result.goals_created == 1
