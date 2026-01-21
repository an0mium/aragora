"""
Tests for Nomic Loop Design Phase.

Phase 2: Design
- Tests architecture planning
- Tests file identification
- Tests safety review
- Tests design approval
- Tests task decomposition integration
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.task_decomposer import TaskDecomposer, TaskDecomposition, SubTask


class TestDesignPhaseInitialization:
    """Tests for DesignPhase initialization."""

    def test_init_with_required_args(self, mock_aragora_path, mock_claude_agent):
        """Should initialize with required arguments."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
        )

        assert phase.aragora_path == mock_aragora_path
        assert phase.claude == mock_claude_agent

    def test_init_with_protected_files(self, mock_aragora_path, mock_claude_agent):
        """Should accept protected files list."""
        from aragora.nomic.phases.design import DesignPhase

        protected = ["CLAUDE.md", "core.py", ".env"]

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            protected_files=protected,
        )

        assert phase.protected_files == protected


class TestDesignPhaseArchitecture:
    """Tests for architecture planning."""

    @pytest.mark.asyncio
    async def test_generates_design_from_proposal(
        self, mock_aragora_path, mock_claude_agent, mock_log_fn
    ):
        """Should generate design from approved proposal."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            log_fn=mock_log_fn,
        )

        proposal = {
            "id": "p1",
            "proposal": "Add comprehensive error handling",
            "reasoning": "Improves reliability",
        }

        with patch.object(phase, "_generate_design", new_callable=AsyncMock) as mock_design:
            mock_design.return_value = {
                "components": ["ErrorHandler", "RetryLogic"],
                "files_to_modify": ["aragora/errors.py"],
                "files_to_create": ["aragora/retry.py"],
            }

            design = await phase.generate_design(proposal)

            assert "components" in design
            assert "files_to_modify" in design or "files_to_create" in design
            mock_design.assert_called_once()

    @pytest.mark.asyncio
    async def test_identifies_affected_files(
        self, mock_aragora_path, mock_claude_agent, mock_log_fn
    ):
        """Should identify files affected by design."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            log_fn=mock_log_fn,
        )

        design = {
            "components": ["ErrorHandler"],
            "description": "Add error handling to core module",
        }

        with patch.object(phase, "_identify_files", new_callable=AsyncMock) as mock_identify:
            mock_identify.return_value = [
                "aragora/core.py",
                "aragora/errors.py",
                "tests/test_errors.py",
            ]

            files = await phase.identify_affected_files(design)

            assert len(files) >= 1
            mock_identify.assert_called_once()


class TestDesignPhaseSafetyReview:
    """Tests for design safety review."""

    @pytest.mark.asyncio
    async def test_blocks_protected_file_modifications(
        self, mock_aragora_path, mock_claude_agent, mock_log_fn
    ):
        """Should block designs that modify protected files."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            log_fn=mock_log_fn,
            protected_files=["CLAUDE.md", "core.py"],
        )

        design = {
            "files_to_modify": ["CLAUDE.md", "aragora/utils.py"],
        }

        result = await phase.safety_review(design)

        assert result["safe"] is False
        assert "protected" in result.get("reason", "").lower()

    @pytest.mark.asyncio
    async def test_allows_safe_designs(self, mock_aragora_path, mock_claude_agent, mock_log_fn):
        """Should allow designs that don't touch protected files."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            log_fn=mock_log_fn,
            protected_files=["CLAUDE.md"],
        )

        design = {
            "files_to_modify": ["aragora/utils.py", "aragora/helpers.py"],
            "files_to_create": ["aragora/new_module.py"],
        }

        result = await phase.safety_review(design)

        assert result["safe"] is True

    @pytest.mark.asyncio
    async def test_flags_high_risk_patterns(
        self, mock_aragora_path, mock_claude_agent, mock_log_fn
    ):
        """Should flag designs with high-risk patterns."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            log_fn=mock_log_fn,
        )

        design = {
            "description": "Add eval() for dynamic code execution",
            "files_to_modify": ["aragora/executor.py"],
        }

        with patch.object(phase, "_check_risk_patterns") as mock_check:
            mock_check.return_value = {
                "high_risk": True,
                "patterns_found": ["eval", "exec"],
            }

            result = await phase.safety_review(design)

            # Should either block or flag for human review
            assert result.get("requires_review", False) or result.get("safe", True) is False


class TestDesignPhaseApproval:
    """Tests for design approval workflow."""

    @pytest.mark.asyncio
    async def test_auto_approves_low_risk_designs(
        self, mock_aragora_path, mock_claude_agent, mock_log_fn
    ):
        """Should auto-approve low-risk designs."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            log_fn=mock_log_fn,
            auto_approve_threshold=0.8,
        )

        design = {
            "risk_score": 0.2,
            "files_to_modify": ["aragora/utils.py"],
        }

        with patch.object(phase, "safety_review", new_callable=AsyncMock) as mock_review:
            mock_review.return_value = {"safe": True, "risk_score": 0.2}

            result = await phase.approve_design(design)

            assert result["approved"] is True
            assert result.get("auto_approved", False) is True

    @pytest.mark.asyncio
    async def test_requires_human_approval_for_high_risk(
        self, mock_aragora_path, mock_claude_agent, mock_log_fn
    ):
        """Should require human approval for high-risk designs."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            log_fn=mock_log_fn,
            auto_approve_threshold=0.5,
        )

        design = {
            "risk_score": 0.8,
            "files_to_modify": ["aragora/core.py"],
        }

        with patch.object(phase, "safety_review", new_callable=AsyncMock) as mock_review:
            mock_review.return_value = {"safe": True, "risk_score": 0.8}

            result = await phase.approve_design(design)

            assert result.get("requires_human_review", False) is True or result["approved"] is False


class TestDesignPhaseIntegration:
    """Integration tests for design phase."""

    @pytest.mark.asyncio
    async def test_full_design_flow(
        self, mock_aragora_path, mock_claude_agent, mock_log_fn, mock_debate_result
    ):
        """Should complete full design flow."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "generate_design", new_callable=AsyncMock) as mock_gen:
            with patch.object(phase, "safety_review", new_callable=AsyncMock) as mock_review:
                with patch.object(phase, "approve_design", new_callable=AsyncMock) as mock_approve:
                    mock_gen.return_value = {
                        "components": ["ErrorHandler"],
                        "files_to_modify": ["aragora/errors.py"],
                    }
                    mock_review.return_value = {"safe": True}
                    mock_approve.return_value = {"approved": True}

                    result = await phase.run(winning_proposal=mock_debate_result["proposals"][0])

                    assert result is not None
                    assert result.get("approved", False) is True

    @pytest.mark.asyncio
    async def test_design_flow_with_rejection(
        self, mock_aragora_path, mock_claude_agent, mock_log_fn
    ):
        """Should handle design rejection gracefully."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            log_fn=mock_log_fn,
            protected_files=["aragora/core.py"],
        )

        proposal = {
            "proposal": "Modify core module",
        }

        with patch.object(phase, "generate_design", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = {
                "files_to_modify": ["aragora/core.py"],  # Protected
            }

            result = await phase.run(winning_proposal=proposal)

            assert result is not None
            assert result.get("approved", True) is False


class TestDesignPhaseDecomposition:
    """Tests for task decomposition integration."""

    def test_decomposer_initialized_with_phase(self, mock_aragora_path, mock_claude_agent):
        """Should initialize TaskDecomposer with the phase."""
        from aragora.nomic.phases.design import DesignPhase, DesignConfig

        config = DesignConfig(
            enable_decomposition=True,
            decomposition_threshold=5,
            max_subtasks=3,
        )

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            config=config,
        )

        assert phase._decomposer is not None
        assert isinstance(phase._decomposer, TaskDecomposer)
        assert phase._decomposer.config.complexity_threshold == 5
        assert phase._decomposer.config.max_subtasks == 3

    def test_decomposition_disabled_by_config(self, mock_aragora_path, mock_claude_agent):
        """Should respect disabled decomposition config."""
        from aragora.nomic.phases.design import DesignPhase, DesignConfig

        config = DesignConfig(enable_decomposition=False)

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            config=config,
        )

        # Decomposer should still be created but not used when disabled
        assert phase._decomposer is not None
        assert phase.config.enable_decomposition is False

    def test_merge_subtask_designs_creates_header(self, mock_aragora_path, mock_claude_agent):
        """Should create proper header when merging subtask designs."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
        )

        decomposition = TaskDecomposition(
            original_task="Refactor the API and database layers",
            complexity_score=7,
            complexity_level="high",
            should_decompose=True,
            subtasks=[
                SubTask(
                    id="subtask_1",
                    title="API Changes",
                    description="Update API endpoints",
                    dependencies=[],
                    estimated_complexity="medium",
                ),
                SubTask(
                    id="subtask_2",
                    title="Database Changes",
                    description="Update database schema",
                    dependencies=["subtask_1"],
                    estimated_complexity="high",
                ),
            ],
            rationale="Task spans multiple concepts",
        )

        subtask_designs = [
            "### Subtask 1: API Changes\n\nDesign for API...",
            "### Subtask 2: Database Changes\n\nDesign for DB...",
        ]

        merged = phase._merge_subtask_designs(
            decomposition.original_task,
            subtask_designs,
            decomposition,
        )

        assert "# Decomposed Design" in merged
        assert "Refactor the API" in merged
        assert "API Changes" in merged
        assert "Database Changes" in merged
        assert "Integration Notes" in merged
        assert "subtask_1" in merged  # Dependency reference

    def test_merge_subtask_designs_empty(self, mock_aragora_path, mock_claude_agent):
        """Should handle empty subtask designs."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
        )

        decomposition = TaskDecomposition(
            original_task="Some task",
            complexity_score=3,
            complexity_level="low",
            should_decompose=False,
            rationale="Low complexity",
        )

        merged = phase._merge_subtask_designs(
            decomposition.original_task,
            [],
            decomposition,
        )

        assert merged == ""

    def test_decomposer_analyzes_low_complexity_correctly(
        self, mock_aragora_path, mock_claude_agent
    ):
        """Should analyze low-complexity tasks and not recommend decomposition."""
        from aragora.nomic.phases.design import DesignPhase, DesignConfig

        config = DesignConfig(
            enable_decomposition=True,
            decomposition_threshold=6,
        )

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            config=config,
        )

        # Simple task should NOT trigger decomposition
        simple_task = "Fix typo in README"
        result = phase._decomposer.analyze(simple_task)

        assert result.complexity_score < 6
        assert result.should_decompose is False
        assert result.complexity_level == "low"

    def test_decomposer_analyzes_high_complexity_correctly(
        self, mock_aragora_path, mock_claude_agent
    ):
        """Should analyze high-complexity tasks and recommend decomposition."""
        from aragora.nomic.phases.design import DesignPhase, DesignConfig

        config = DesignConfig(
            enable_decomposition=True,
            decomposition_threshold=5,
        )

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            config=config,
        )

        # Complex task should trigger decomposition
        complex_task = (
            "Refactor the entire authentication system and migrate the database "
            "schema while redesigning the API layer. Update handler.py, auth.py, "
            "database.py, and schema.py with new security patterns and performance "
            "optimizations for the backend frontend and testing infrastructure."
        )
        result = phase._decomposer.analyze(complex_task)

        assert result.complexity_score >= 5
        assert result.should_decompose is True
        assert len(result.subtasks) >= 2
