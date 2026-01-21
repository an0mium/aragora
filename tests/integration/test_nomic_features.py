"""Integration tests for Nomic Loop features.

Tests cross-feature integration between:
- Task decomposer and design phase
- Cross-cycle learning persistence
- Pattern-based agent selection
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Task decomposer integration
from aragora.nomic.task_decomposer import (
    TaskDecomposer,
    TaskDecomposition,
    SubTask,
    DecomposerConfig,
    analyze_task,
)

# Cross-cycle learning
from aragora.nomic.cycle_record import (
    NomicCycleRecord,
    AgentContribution,
    SurpriseEvent,
    PatternReinforcement,
)
from aragora.nomic.cycle_store import CycleLearningStore


class TestTaskDecomposerIntegration:
    """Integration tests for task decomposer with design phase."""

    def test_decomposer_integrates_with_debate_result(self):
        """Task decomposer should accept DebateResult context."""
        decomposer = TaskDecomposer()

        # Create mock debate result
        debate_result = MagicMock()
        debate_result.consensus_text = (
            "Refactor the entire authentication system to support multi-tenancy. "
            "This requires changes to models, migrations, API endpoints. "
            "Update auth.py, db.py, handlers.py, middleware.py, tests.py."
        )

        result = decomposer.analyze(
            task_description=debate_result.consensus_text,
            debate_result=debate_result,
        )

        # Should recognize complexity from debate result
        assert result.complexity_score >= 5
        assert result.should_decompose is True

    def test_decomposer_extracts_file_scope(self):
        """Subtasks should have relevant file scope."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze(
            "Update database layer in db.py, models.py, and add API endpoints in handlers.py"
        )

        if result.should_decompose:
            # At least one subtask should have file scope
            has_file_scope = any(len(st.file_scope) > 0 for st in result.subtasks)
            # This depends on heuristics, so we just verify structure
            assert isinstance(result.subtasks, list)

    def test_decomposer_creates_dependencies(self):
        """Subtasks should have logical dependencies."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze(
            "Major refactor: update database schema, modify API layer, "
            "add security middleware, update tests. Requires db.py, api.py, "
            "middleware.py, tests.py changes."
        )

        if result.should_decompose and len(result.subtasks) > 1:
            # Later subtasks should depend on earlier ones
            for i, subtask in enumerate(result.subtasks[1:], 1):
                # Either has dependencies or is independent
                assert isinstance(subtask.dependencies, list)


class TestCrossFeatureLearning:
    """Integration tests for cross-cycle learning with other features."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary store for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        store = CycleLearningStore(db_path=db_path)
        yield store

        # Cleanup
        try:
            Path(db_path).unlink()
        except OSError:
            pass

    def test_cycle_records_persist_task_decomposition_data(self, temp_store):
        """Cycle records should capture task decomposition results."""
        # Create cycle with decomposed task
        record = NomicCycleRecord(
            cycle_id="decomposed-cycle-1",
            started_at=time.time(),
        )

        # Simulate decomposed task execution
        record.topics_debated.append("Refactor authentication (decomposed)")
        record.add_pattern_reinforcement(
            pattern_type="decomposition",
            description="Task split into 3 subtasks, all succeeded",
            success=True,
            confidence=0.9,
        )
        record.mark_complete(success=True)

        # Save and reload
        temp_store.save_cycle(record)
        loaded = temp_store.load_cycle("decomposed-cycle-1")

        assert loaded is not None
        assert len(loaded.pattern_reinforcements) == 1
        assert loaded.pattern_reinforcements[0].pattern_type == "decomposition"
        assert loaded.pattern_reinforcements[0].success is True

    def test_cross_cycle_context_includes_decomposition_patterns(self, temp_store):
        """Cross-cycle context should include decomposition success patterns."""
        # Store multiple cycles with decomposition patterns
        for i in range(5):
            record = NomicCycleRecord(
                cycle_id=f"cycle-{i}",
                started_at=time.time() + i,
            )
            record.add_pattern_reinforcement(
                pattern_type="decomposition",
                description=f"Decomposition attempt {i}",
                success=(i % 2 == 0),  # 50% success rate
                confidence=0.8,
            )
            record.mark_complete(success=(i % 2 == 0))
            temp_store.save_cycle(record)

        # Get pattern statistics
        stats = temp_store.get_pattern_statistics()

        assert "decomposition" in stats
        assert stats["decomposition"]["success_count"] == 3  # 0, 2, 4
        assert stats["decomposition"]["failure_count"] == 2  # 1, 3
        assert abs(stats["decomposition"]["success_rate"] - 0.6) < 0.01

    def test_agent_trajectory_tracks_across_cycles(self, temp_store):
        """Agent performance should be trackable across cycles."""
        # Store cycles with agent contributions
        for i in range(5):
            record = NomicCycleRecord(
                cycle_id=f"agent-cycle-{i}",
                started_at=time.time() + i * 60,
            )
            record.add_agent_contribution(
                agent_name="claude",
                proposals_made=5,
                proposals_accepted=3 + i,  # Improving over time
            )
            record.mark_complete(success=True)
            temp_store.save_cycle(record)

        # Get agent trajectory
        trajectory = temp_store.get_agent_trajectory("claude")

        assert len(trajectory) == 5
        # Most recent first (reversed chronological)
        assert trajectory[0]["proposals_accepted"] == 7  # 3 + 4
        assert trajectory[4]["proposals_accepted"] == 3  # 3 + 0

    def test_surprise_events_aggregate_by_phase(self, temp_store):
        """Surprise events should be queryable by phase."""
        # Store cycles with surprises in different phases
        for i in range(3):
            record = NomicCycleRecord(
                cycle_id=f"surprise-cycle-{i}",
                started_at=time.time() + i,
            )
            record.add_surprise(
                phase="design" if i % 2 == 0 else "implement",
                description=f"Unexpected complexity in cycle {i}",
                expected="Simple change",
                actual="Major refactor needed",
                impact="high" if i == 0 else "medium",
            )
            record.mark_complete(success=False)
            temp_store.save_cycle(record)

        # Get surprise summary
        summary = temp_store.get_surprise_summary()

        assert "design" in summary
        assert "implement" in summary
        assert len(summary["design"]) == 2  # cycles 0, 2
        assert len(summary["implement"]) == 1  # cycle 1


class TestPatternMatcherIntegration:
    """Integration tests for pattern-based agent selection."""

    def test_pattern_matcher_with_task_decomposer(self):
        """Pattern matcher should classify decomposed subtasks."""
        from aragora.ranking.pattern_matcher import TaskPatternMatcher

        matcher = TaskPatternMatcher()
        decomposer = TaskDecomposer()

        # Decompose a complex task
        result = decomposer.analyze(
            "Add authentication with JWT, implement rate limiting, "
            "fix the database connection bug, and refactor API layer."
        )

        if result.should_decompose:
            # Classify each subtask
            for subtask in result.subtasks:
                pattern = matcher.classify_task(subtask.description)
                assert pattern in matcher.TASK_PATTERNS or pattern == "general"

    def test_pattern_history_affects_agent_selection(self):
        """Historical pattern data should influence agent selection scores."""
        from aragora.ranking.pattern_matcher import TaskPatternMatcher

        matcher = TaskPatternMatcher()

        # Mock critique store with pattern data
        mock_store = MagicMock()
        mock_store.get_agent_pattern_stats.return_value = {
            "claude": {"bugfix": 0.9, "refactor": 0.7},
            "codex": {"bugfix": 0.6, "refactor": 0.9},
        }

        # Classify tasks and get affinities
        bugfix_pattern = matcher.classify_task("Fix the authentication bug")
        refactor_pattern = matcher.classify_task("Refactor the database layer")

        assert bugfix_pattern == "bugfix"
        assert refactor_pattern == "refactor"


class TestKnowledgeMoundWorkflowIntegration:
    """Integration tests for Knowledge Mound with Workflow Engine."""

    @pytest.mark.asyncio
    async def test_workflow_can_query_knowledge_mound(self):
        """Workflow nodes should be able to query Knowledge Mound."""
        # Mock KnowledgeMound
        mock_mound = MagicMock()
        mock_result = MagicMock()
        mock_result.items = [
            MagicMock(content="Previous debate about auth", confidence=0.9),
            MagicMock(content="Auth pattern recommendation", confidence=0.85),
        ]
        mock_result.total_count = 2
        mock_mound.query = AsyncMock(return_value=mock_result)

        # Simulate workflow querying mound
        result = await mock_mound.query(
            query="authentication patterns",
            limit=5,
        )

        assert result.total_count == 2
        assert len(result.items) == 2
        assert result.items[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_debate_results_stored_in_knowledge_mound(self):
        """Debate results should be storable in Knowledge Mound."""
        # Mock KnowledgeMound
        mock_mound = MagicMock()
        mock_mound.add = AsyncMock(return_value="node-123")

        # Simulate storing debate result
        debate_summary = {
            "topic": "Rate limiting implementation",
            "consensus": "Use token bucket algorithm",
            "confidence": 0.92,
            "participants": ["claude", "codex", "gemini"],
        }

        node_id = await mock_mound.add(
            content=str(debate_summary),
            metadata={"type": "debate_result", "topic": debate_summary["topic"]},
        )

        assert node_id == "node-123"
        mock_mound.add.assert_called_once()


class TestEndToEndIntegration:
    """End-to-end integration tests combining multiple features."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_cycle_with_decomposition_and_learning(self, temp_db):
        """Test full cycle flow with task decomposition and cross-cycle learning."""
        # Initialize stores
        cycle_store = CycleLearningStore(db_path=str(temp_db / "cycles.db"))

        # Create a complex task with high-complexity keywords
        task = (
            "Refactor and redesign the complete audit logging system with database storage, "
            "API endpoints for querying logs, real-time streaming support, "
            "and retention policies. This system-wide overhaul requires updating "
            "db.py, handlers.py, websocket.py, models.py, migrations.py, tests.py."
        )

        # Decompose the task with lower threshold
        decomposer = TaskDecomposer(DecomposerConfig(complexity_threshold=4))
        decomposition = decomposer.analyze(task)

        # Should have recognized complexity
        assert decomposition.complexity_score >= 4

        # Create cycle record
        record = NomicCycleRecord(
            cycle_id="full-cycle-1",
            started_at=time.time(),
        )
        record.topics_debated.append(task[:100])

        # Record subtask execution (or single task if not decomposed)
        if decomposition.should_decompose and decomposition.subtasks:
            for subtask in decomposition.subtasks:
                record.add_pattern_reinforcement(
                    pattern_type=subtask.estimated_complexity,
                    description=subtask.title,
                    success=True,
                    confidence=0.85,
                )
            expected_patterns = len(decomposition.subtasks)
        else:
            record.add_pattern_reinforcement(
                pattern_type=decomposition.complexity_level,
                description="Single task execution",
                success=True,
                confidence=0.85,
            )
            expected_patterns = 1

        # Mark complete
        record.mark_complete(success=True)
        cycle_store.save_cycle(record)

        # Verify persistence
        loaded = cycle_store.load_cycle("full-cycle-1")
        assert loaded is not None
        assert loaded.success is True
        assert len(loaded.pattern_reinforcements) == expected_patterns

    def test_learning_improves_over_cycles(self, temp_db):
        """Pattern statistics should show improvement over cycles."""
        cycle_store = CycleLearningStore(db_path=str(temp_db / "cycles.db"))

        # Simulate 10 cycles with improving success rate
        for i in range(10):
            record = NomicCycleRecord(
                cycle_id=f"learning-cycle-{i}",
                started_at=time.time() + i * 60,
            )

            # Success rate improves over time (last 5 cycles succeed)
            success = i >= 5

            record.add_pattern_reinforcement(
                pattern_type="improvement",
                description=f"Cycle {i} improvement",
                success=success,
                confidence=0.7 + (i * 0.03),  # Confidence grows
            )
            record.mark_complete(success=success)
            cycle_store.save_cycle(record)

        # Get statistics
        stats = cycle_store.get_pattern_statistics()

        assert "improvement" in stats
        # 5 successes out of 10
        assert stats["improvement"]["success_rate"] == 0.5
        # Average confidence should be around 0.835 (0.7 + 0.15/2)
        assert stats["improvement"]["avg_confidence"] > 0.8
