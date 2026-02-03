"""Tests for SOAR curriculum integration with Nomic Loop."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.curriculum.integration import (
    CurriculumAwareFeedbackLoop,
    CurriculumConfig,
    integrate_curriculum_with_orchestrator,
)
from aragora.nomic.curriculum.soar_curriculum import (
    Curriculum,
    SteppingStone,
    SteppingStoneResult,
)


@dataclass
class MockSubTask:
    """Mock subtask for testing."""

    id: str
    description: str


@dataclass
class MockAssignment:
    """Mock assignment for testing."""

    subtask: MockSubTask


class TestCurriculumConfig:
    """Tests for CurriculumConfig."""

    def test_default_values(self):
        """Default configuration values."""
        config = CurriculumConfig()

        assert config.enable_curriculum is True
        assert config.max_stepping_stones == 5
        assert config.min_failures_for_curriculum == 2
        assert config.stone_success_threshold == 0.6
        assert config.persist_curricula is True

    def test_custom_values(self):
        """Custom configuration values."""
        config = CurriculumConfig(
            enable_curriculum=False,
            max_stepping_stones=3,
            min_failures_for_curriculum=1,
        )

        assert config.enable_curriculum is False
        assert config.max_stepping_stones == 3
        assert config.min_failures_for_curriculum == 1


class TestCurriculumAwareFeedbackLoop:
    """Tests for CurriculumAwareFeedbackLoop."""

    def test_initialization(self):
        """Feedback loop initializes correctly."""
        loop = CurriculumAwareFeedbackLoop(max_iterations=5)

        assert loop.max_iterations == 5
        assert loop.config.enable_curriculum is True
        assert loop.skill_profile is not None

    def test_initialization_with_config(self):
        """Feedback loop accepts custom config."""
        config = CurriculumConfig(max_stepping_stones=10)
        loop = CurriculumAwareFeedbackLoop(config=config)

        assert loop.config.max_stepping_stones == 10

    @pytest.mark.asyncio
    async def test_standard_analysis_early_failures(self):
        """Uses standard analysis for early failures."""
        loop = CurriculumAwareFeedbackLoop(max_iterations=3)
        assignment = MockAssignment(subtask=MockSubTask(id="task1", description="Test task"))

        # First failure - standard analysis
        result = await loop.analyze_failure(
            assignment, {"type": "test_failure", "message": "AssertionError"}
        )

        assert result["action"] == "retry_implement"
        assert "Test failures" in result["reason"]

    @pytest.mark.asyncio
    async def test_lint_error_quick_fix(self):
        """Lint errors trigger quick fix."""
        loop = CurriculumAwareFeedbackLoop()
        assignment = MockAssignment(subtask=MockSubTask(id="task1", description="Test task"))

        result = await loop.analyze_failure(
            assignment, {"type": "lint_error", "message": "unused import"}
        )

        assert result["action"] == "quick_fix"

    @pytest.mark.asyncio
    async def test_design_issue_redesign(self):
        """Design issues trigger redesign."""
        loop = CurriculumAwareFeedbackLoop()
        assignment = MockAssignment(subtask=MockSubTask(id="task1", description="Test task"))

        result = await loop.analyze_failure(
            assignment, {"type": "design_issue", "suggestion": "use factory pattern"}
        )

        assert result["action"] == "redesign"

    @pytest.mark.asyncio
    async def test_creates_curriculum_at_max_iterations(self):
        """Creates curriculum when max iterations reached."""
        config = CurriculumConfig(min_failures_for_curriculum=2)
        loop = CurriculumAwareFeedbackLoop(max_iterations=2, config=config)
        assignment = MockAssignment(
            subtask=MockSubTask(id="task1", description="Implement complex feature")
        )

        # Fail twice to reach max iterations
        await loop.analyze_failure(assignment, {"type": "test_failure"})
        result = await loop.analyze_failure(assignment, {"type": "test_failure"})

        # Should create curriculum
        assert result["action"] == "stepping_stone"
        assert "curriculum_id" in result
        assert "stone" in result
        assert result["stone_number"] == 1

    @pytest.mark.asyncio
    async def test_escalates_when_curriculum_disabled(self):
        """Escalates at max iterations when curriculum disabled."""
        config = CurriculumConfig(enable_curriculum=False, min_failures_for_curriculum=2)
        loop = CurriculumAwareFeedbackLoop(max_iterations=2, config=config)
        assignment = MockAssignment(subtask=MockSubTask(id="task1", description="Test task"))

        # Fail twice
        await loop.analyze_failure(assignment, {"type": "test_failure"})
        result = await loop.analyze_failure(assignment, {"type": "test_failure"})

        assert result["action"] == "escalate"
        assert result["require_human"] is True

    @pytest.mark.asyncio
    async def test_curriculum_progress_with_success(self):
        """Progresses through curriculum on stepping stone success."""
        loop = CurriculumAwareFeedbackLoop(
            max_iterations=2,
            config=CurriculumConfig(min_failures_for_curriculum=2),
        )
        assignment = MockAssignment(subtask=MockSubTask(id="task1", description="Complex task"))

        # Create curriculum
        await loop.analyze_failure(assignment, {"type": "test_failure"})
        result = await loop.analyze_failure(assignment, {"type": "test_failure"})
        stone_id = result["stone"]["id"]

        # Report success on first stone
        result = await loop.analyze_failure(
            assignment,
            {
                "type": "stone_result",
                "stone_result": {
                    "stone_id": stone_id,
                    "success": True,
                    "score": 0.9,
                },
            },
        )

        # Should either give next stone or retry main task
        assert result["action"] in ("stepping_stone", "retry_main_task")

    @pytest.mark.asyncio
    async def test_retry_main_task_after_curriculum_success(self):
        """Returns to main task after curriculum success."""
        loop = CurriculumAwareFeedbackLoop(
            max_iterations=2,
            config=CurriculumConfig(min_failures_for_curriculum=2),
        )
        assignment = MockAssignment(subtask=MockSubTask(id="task1", description="Complex task"))

        # Create curriculum and complete all stones successfully
        await loop.analyze_failure(assignment, {"type": "test_failure"})
        result = await loop.analyze_failure(assignment, {"type": "test_failure"})

        curriculum = loop.get_active_curriculum("task1")
        assert curriculum is not None

        # Complete all stones
        for stone in curriculum.stepping_stones:
            result = await loop.analyze_failure(
                assignment,
                {
                    "type": "stone_result",
                    "stone_result": {
                        "stone_id": stone.id,
                        "success": True,
                        "score": 0.9,
                    },
                },
            )

        # Should retry main task
        assert result["action"] == "retry_main_task"
        assert "curriculum_success_rate" in result

    @pytest.mark.asyncio
    async def test_escalates_after_curriculum_failure(self):
        """Escalates when curriculum doesn't help."""
        loop = CurriculumAwareFeedbackLoop(
            max_iterations=2,
            config=CurriculumConfig(min_failures_for_curriculum=2),
        )
        assignment = MockAssignment(subtask=MockSubTask(id="task1", description="Complex task"))

        # Create curriculum
        await loop.analyze_failure(assignment, {"type": "test_failure"})
        result = await loop.analyze_failure(assignment, {"type": "test_failure"})

        curriculum = loop.get_active_curriculum("task1")

        # Fail all stones
        for stone in curriculum.stepping_stones:
            result = await loop.analyze_failure(
                assignment,
                {
                    "type": "stone_result",
                    "stone_result": {
                        "stone_id": stone.id,
                        "success": False,
                        "score": 0.2,
                    },
                },
            )

        # Should escalate after curriculum failure
        assert result["action"] == "escalate"
        assert "curriculum_results" in result

    def test_get_active_curriculum(self):
        """Can retrieve active curriculum for a task."""
        loop = CurriculumAwareFeedbackLoop()

        # No curriculum initially
        assert loop.get_active_curriculum("task1") is None


class TestIntegrateWithOrchestrator:
    """Tests for orchestrator integration."""

    def test_integration_replaces_feedback_loop(self):
        """Integration replaces the orchestrator's feedback loop."""
        # Create mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.feedback_loop = MagicMock()
        mock_orchestrator.feedback_loop.max_iterations = 5

        # Integrate
        integrate_curriculum_with_orchestrator(mock_orchestrator)

        # Check replacement
        assert isinstance(mock_orchestrator.feedback_loop, CurriculumAwareFeedbackLoop)
        assert mock_orchestrator.feedback_loop.max_iterations == 5

    def test_integration_with_custom_config(self):
        """Integration accepts custom configuration."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.feedback_loop = MagicMock()
        mock_orchestrator.feedback_loop.max_iterations = 3

        config = CurriculumConfig(max_stepping_stones=10)
        integrate_curriculum_with_orchestrator(mock_orchestrator, config)

        assert mock_orchestrator.feedback_loop.config.max_stepping_stones == 10


class TestHintExtraction:
    """Tests for test hint extraction."""

    def test_extracts_assertion_errors(self):
        """Extracts AssertionError lines."""
        loop = CurriculumAwareFeedbackLoop()
        message = """
        test_foo.py::test_bar FAILED
        AssertionError: Expected 5 but got 3
        Another line
        """

        hints = loop._extract_test_hints(message)
        assert "AssertionError" in hints

    def test_extracts_expected_actual(self):
        """Extracts Expected/Actual lines."""
        loop = CurriculumAwareFeedbackLoop()
        message = """
        Expected: 10
        Actual: 5
        """

        hints = loop._extract_test_hints(message)
        assert "Expected" in hints
        assert "Actual" in hints

    def test_limits_hints(self):
        """Limits number of hint lines."""
        loop = CurriculumAwareFeedbackLoop()
        message = "\n".join([f"AssertionError: line {i}" for i in range(10)])

        hints = loop._extract_test_hints(message)
        lines = hints.split("\n")
        assert len(lines) <= 5

    def test_default_hint_when_empty(self):
        """Returns default hint when no patterns match."""
        loop = CurriculumAwareFeedbackLoop()
        hints = loop._extract_test_hints("Some random error")

        assert hints == "Review test output"
