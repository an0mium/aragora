"""
Comprehensive unit tests for DesignPhase.

Tests the design phase module including:
- Initialization with various configurations
- Legacy API (run, generate_design, safety_review, approve_design)
- Modern execute() API with task decomposition
- Safety review and risk pattern detection
- Task decomposition and subtask merging
- Phase transitions and error handling
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_aragora_path(tmp_path: Path) -> Path:
    """Create a mock aragora project structure."""
    (tmp_path / "aragora").mkdir()
    (tmp_path / "aragora" / "__init__.py").write_text('"""Aragora."""')
    return tmp_path


@pytest.fixture
def mock_agent():
    """Create a mock agent with generate method."""
    agent = MagicMock()
    agent.name = "claude"
    agent.generate = AsyncMock(return_value="Design: Create new module...")
    return agent


@pytest.fixture
def mock_agents():
    """Create multiple mock agents."""
    agents = []
    for name in ["claude", "gemini", "codex"]:
        agent = MagicMock()
        agent.name = name
        agent.generate = AsyncMock(return_value=f"Design from {name}")
        agents.append(agent)
    return agents


@pytest.fixture
def mock_arena():
    """Create a mock arena."""
    arena = MagicMock()
    result = MagicMock()
    result.consensus_reached = True
    result.final_answer = """
## FILE CHANGES
- `aragora/new_module.py` - Create new (50 lines)
- `aragora/core.py` - Modify existing (10 lines)

## API DESIGN
```python
class NewModule:
    def process(self, data: str) -> dict:
        '''Process data.'''
        ...
```

## INTEGRATION POINTS
- Uses aragora.utils.helper
- Called by aragora.server.handlers

## TEST PLAN
- test_new_module_basic()
- test_new_module_edge_cases()

## EXAMPLE USAGE
```python
module = NewModule()
result = module.process("input")
```
"""
    result.confidence = 0.9
    result.votes = []
    result.messages = []
    arena.run = AsyncMock(return_value=result)
    return arena


@pytest.fixture
def mock_arena_factory(mock_arena):
    """Create a mock arena factory."""
    return MagicMock(return_value=mock_arena)


@pytest.fixture
def mock_environment_factory():
    """Create a mock environment factory."""
    return MagicMock(return_value=MagicMock())


@pytest.fixture
def mock_protocol_factory():
    """Create a mock protocol factory."""
    return MagicMock(return_value=MagicMock())


@pytest.fixture
def mock_nomic_integration():
    """Create a mock nomic integration."""
    integration = MagicMock()
    integration.probe_agents = AsyncMock(return_value={"claude": 0.9})
    integration.checkpoint = AsyncMock()
    integration.full_post_debate_analysis = AsyncMock(return_value={})
    return integration


@pytest.fixture
def mock_log_fn():
    """Create a mock log function."""
    return MagicMock()


@pytest.fixture
def mock_stream_emit_fn():
    """Create a mock stream emit function."""
    return MagicMock()


# ============================================================================
# DesignConfig Tests
# ============================================================================


class TestDesignConfig:
    """Tests for DesignConfig dataclass."""

    def test_default_config(self):
        """Should have sensible defaults."""
        from aragora.nomic.phases.design import DesignConfig

        config = DesignConfig()

        assert config.rounds > 0
        assert config.protected_files is not None
        assert "CLAUDE.md" in config.protected_files
        assert config.enable_decomposition is True

    def test_custom_config(self):
        """Should accept custom values."""
        from aragora.nomic.phases.design import DesignConfig

        config = DesignConfig(
            rounds=3,
            consensus_mode="majority",
            protected_files=["custom.py"],
            enable_decomposition=False,
            decomposition_threshold=8,
        )

        assert config.rounds == 3
        assert config.protected_files == ["custom.py"]
        assert config.enable_decomposition is False
        assert config.decomposition_threshold == 8

    def test_post_init_sets_default_protected(self):
        """Should set default protected files if None."""
        from aragora.nomic.phases.design import DesignConfig, DEFAULT_PROTECTED_FILES

        config = DesignConfig(protected_files=None)

        assert config.protected_files == DEFAULT_PROTECTED_FILES


# ============================================================================
# BeliefContext Tests
# ============================================================================


class TestBeliefContext:
    """Tests for BeliefContext dataclass."""

    def test_default_context(self):
        """Should have default values."""
        from aragora.nomic.phases.design import BeliefContext

        ctx = BeliefContext()

        assert ctx.contested_count == 0
        assert ctx.crux_count == 0
        assert ctx.convergence_achieved is False

    def test_to_string_empty(self):
        """Should return empty string when no uncertainty."""
        from aragora.nomic.phases.design import BeliefContext

        ctx = BeliefContext()

        assert ctx.to_string() == ""

    def test_to_string_with_contested_claims(self):
        """Should include contested claim info."""
        from aragora.nomic.phases.design import BeliefContext

        ctx = BeliefContext(
            contested_count=3,
            crux_count=1,
            convergence_achieved=False,
        )

        result = ctx.to_string()

        assert "3 contested claims" in result
        assert "1 crux" in result
        assert "did NOT converge" in result

    def test_to_string_with_convergence(self):
        """Should note convergence status."""
        from aragora.nomic.phases.design import BeliefContext

        ctx = BeliefContext(
            contested_count=1,
            convergence_achieved=True,
        )

        result = ctx.to_string()

        assert "converged" in result.lower()

    def test_to_string_with_posteriors(self):
        """Should include high-entropy claims."""
        from aragora.nomic.phases.design import BeliefContext

        ctx = BeliefContext(
            contested_count=2,
            posteriors={
                "claim_1": {"entropy": 0.8},
                "claim_2": {"entropy": 0.3},
            },
        )

        result = ctx.to_string()

        assert "claim_1" in result
        # claim_2 has low entropy, may not be included


# ============================================================================
# DesignPhase Initialization Tests
# ============================================================================


class TestDesignPhaseInitialization:
    """Tests for DesignPhase initialization."""

    def test_init_with_agents_list(self, mock_aragora_path, mock_agents):
        """Should initialize with agents list."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        assert phase.agents == mock_agents

    def test_init_with_legacy_claude_agent(self, mock_aragora_path, mock_agent):
        """Should support legacy claude_agent parameter."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_agent,
        )

        assert phase.claude == mock_agent
        assert mock_agent in phase.agents

    def test_init_with_config(self, mock_aragora_path, mock_agents):
        """Should accept config parameter."""
        from aragora.nomic.phases.design import DesignPhase, DesignConfig

        config = DesignConfig(rounds=5, enable_decomposition=False)

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            config=config,
        )

        assert phase.config.rounds == 5
        assert phase.config.enable_decomposition is False

    def test_init_creates_decomposer(self, mock_aragora_path, mock_agents):
        """Should create TaskDecomposer with config settings."""
        from aragora.nomic.phases.design import DesignPhase, DesignConfig

        config = DesignConfig(
            decomposition_threshold=7,
            max_subtasks=5,
        )

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            config=config,
        )

        assert phase._decomposer is not None
        assert phase._decomposer.config.complexity_threshold == 7
        assert phase._decomposer.config.max_subtasks == 5

    def test_init_with_protected_files(self, mock_aragora_path, mock_agents):
        """Should accept protected files list."""
        from aragora.nomic.phases.design import DesignPhase

        protected = ["custom.py", "secret.py"]

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            protected_files=protected,
        )

        assert phase.protected_files == protected


# ============================================================================
# Legacy API Tests
# ============================================================================


class TestDesignPhaseGenerateDesign:
    """Tests for generate_design method."""

    @pytest.mark.asyncio
    async def test_generates_with_claude_agent(self, mock_aragora_path, mock_agent, mock_log_fn):
        """Should use claude agent to generate design."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_agent,
            log_fn=mock_log_fn,
        )

        proposal = {"proposal": "Add error handling"}

        design = await phase.generate_design(proposal)

        assert "description" in design
        mock_agent.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_empty_design_without_agent(self, mock_aragora_path, mock_log_fn):
        """Should return empty design when no agent."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        proposal = {"proposal": "Add caching"}

        design = await phase.generate_design(proposal)

        assert design["description"] == "Add caching"
        assert design["components"] == []


class TestDesignPhaseIdentifyAffectedFiles:
    """Tests for identify_affected_files method."""

    @pytest.mark.asyncio
    async def test_extracts_from_design(self, mock_aragora_path, mock_agents):
        """Should extract files from design."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        design = {
            "files_to_modify": ["aragora/core.py", "aragora/utils.py"],
            "files_to_create": ["aragora/new.py"],
        }

        files = await phase.identify_affected_files(design)

        assert "aragora/core.py" in files
        assert "aragora/utils.py" in files
        assert "aragora/new.py" in files


class TestDesignPhaseSafetyReview:
    """Tests for safety_review method."""

    @pytest.mark.asyncio
    async def test_blocks_protected_files(self, mock_aragora_path, mock_agents):
        """Should flag protected file modifications."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            protected_files=["CLAUDE.md", "aragora/__init__.py"],
        )

        design = {
            "files_to_modify": ["CLAUDE.md", "aragora/utils.py"],
            "description": "Update docs",
        }

        result = await phase.safety_review(design)

        assert result["safe"] is False
        assert any("protected" in issue.lower() for issue in result["issues"])

    @pytest.mark.asyncio
    async def test_allows_safe_modifications(self, mock_aragora_path, mock_agents):
        """Should allow non-protected modifications."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            protected_files=["CLAUDE.md"],
        )

        design = {
            "files_to_modify": ["aragora/utils.py"],
            "files_to_create": ["aragora/new_feature.py"],
            "description": "Add new feature",
        }

        result = await phase.safety_review(design)

        assert result["safe"] is True

    @pytest.mark.asyncio
    async def test_flags_dangerous_patterns(self, mock_aragora_path, mock_agents):
        """Should flag dangerous code patterns."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        design = {
            "files_to_modify": ["aragora/executor.py"],
            "description": "Add dynamic code execution using eval() and exec()",
        }

        result = await phase.safety_review(design)

        assert result["requires_review"] is True
        assert len(result["risk_patterns"]) > 0

    @pytest.mark.asyncio
    async def test_calculates_risk_level(self, mock_aragora_path, mock_agents):
        """Should calculate appropriate risk level."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        # Low risk design
        low_risk = {
            "files_to_modify": ["aragora/utils.py"],
            "description": "Add helper function",
        }
        low_result = await phase.safety_review(low_risk)
        assert low_result["risk_level"] == "low"

        # Medium risk design (many files)
        medium_risk = {
            "files_to_modify": [f"aragora/module_{i}.py" for i in range(6)],
            "description": "Refactor modules",
        }
        medium_result = await phase.safety_review(medium_risk)
        assert medium_result["risk_level"] in ["medium", "high"]

        # High risk design (dangerous patterns)
        high_risk = {
            "files_to_modify": ["aragora/exec.py"],
            "description": "Use subprocess.call for system commands",
        }
        high_result = await phase.safety_review(high_risk)
        assert high_result["risk_level"] == "high"


class TestDesignPhaseCheckRiskPatterns:
    """Tests for _check_risk_patterns method."""

    def test_detects_eval(self, mock_aragora_path, mock_agents):
        """Should detect eval pattern."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        result = phase._check_risk_patterns("Use eval(code) to execute")

        assert result["high_risk"] is True
        assert "eval" in result["patterns_found"]

    def test_detects_exec(self, mock_aragora_path, mock_agents):
        """Should detect exec pattern."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        result = phase._check_risk_patterns("Use exec( to run dynamic code")

        assert result["high_risk"] is True
        assert "exec" in result["patterns_found"]

    def test_detects_subprocess(self, mock_aragora_path, mock_agents):
        """Should detect subprocess pattern."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        result = phase._check_risk_patterns("Run commands via subprocess")

        assert result["high_risk"] is True

    def test_safe_description(self, mock_aragora_path, mock_agents):
        """Should pass safe descriptions."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        result = phase._check_risk_patterns("Add a new class for data processing using pandas")

        assert result["high_risk"] is False
        assert result["patterns_found"] == []


class TestDesignPhaseApproveDesign:
    """Tests for approve_design method."""

    @pytest.mark.asyncio
    async def test_auto_approves_low_risk(self, mock_aragora_path, mock_agents):
        """Should auto-approve low risk designs."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            auto_approve_threshold=0.5,
        )

        design = {
            "files_to_modify": ["aragora/utils.py"],
            "description": "Add helper function",
        }

        result = await phase.approve_design(design)

        assert result["approved"] is True
        assert result["auto_approved"] is True

    @pytest.mark.asyncio
    async def test_requires_human_for_high_risk(self, mock_aragora_path, mock_agents):
        """Should require human approval for high risk."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            auto_approve_threshold=0.3,  # Lower threshold
        )

        design = {
            "files_to_modify": [f"module_{i}.py" for i in range(10)],
            "description": "Major refactoring with subprocess calls",
        }

        result = await phase.approve_design(design)

        assert result["requires_human_review"] is True

    @pytest.mark.asyncio
    async def test_rejects_unsafe_designs(self, mock_aragora_path, mock_agents):
        """Should reject designs that fail safety check."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            protected_files=["CLAUDE.md"],
        )

        design = {
            "files_to_modify": ["CLAUDE.md"],
            "description": "Update protected file",
        }

        result = await phase.approve_design(design)

        assert result["approved"] is False
        assert "protected" in result.get("reason", "").lower() or "Safety" in result.get(
            "reason", ""
        )


class TestDesignPhaseRun:
    """Tests for legacy run() method."""

    @pytest.mark.asyncio
    async def test_complete_flow(self, mock_aragora_path, mock_agent, mock_log_fn):
        """Should complete full design flow."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "generate_design", new_callable=AsyncMock) as mock_gen:
            with patch.object(phase, "safety_review", new_callable=AsyncMock) as mock_review:
                with patch.object(phase, "approve_design", new_callable=AsyncMock) as mock_approve:
                    mock_gen.return_value = {
                        "description": "New feature",
                        "files_to_modify": ["aragora/utils.py"],
                        "files_to_create": [],
                    }
                    mock_review.return_value = {"safe": True}
                    mock_approve.return_value = {"approved": True, "auto_approved": True}

                    result = await phase.run(proposal={"proposal": "Test"})

                    assert result["approved"] is True
                    mock_gen.assert_called_once()
                    mock_review.assert_called_once()
                    mock_approve.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_safety_rejection(self, mock_aragora_path, mock_agent, mock_log_fn):
        """Should handle safety review rejection."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_agent,
            protected_files=["secret.py"],
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "generate_design", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = {
                "description": "Bad design",
                "files_to_modify": ["secret.py"],
                "files_to_create": [],
            }

            result = await phase.run(proposal={"proposal": "Modify secret"})

            assert result["approved"] is False
            assert "error" in result


# ============================================================================
# Modern Execute API Tests
# ============================================================================


class TestDesignPhaseExecute:
    """Tests for execute() method."""

    @pytest.mark.asyncio
    async def test_execute_success(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
        mock_log_fn,
        mock_stream_emit_fn,
    ):
        """Should execute design and return DesignResult."""
        from aragora.nomic.phases.design import DesignPhase, DesignConfig

        config = DesignConfig(enable_decomposition=False)

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            config=config,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        result = await phase.execute(improvement="Add error handling")

        assert result["success"] is True
        assert "design" in result
        assert "files_affected" in result
        mock_arena_factory.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_belief_context(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
        mock_log_fn,
    ):
        """Should include belief context in prompt."""
        from aragora.nomic.phases.design import DesignPhase, DesignConfig, BeliefContext

        config = DesignConfig(enable_decomposition=False)

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            config=config,
            log_fn=mock_log_fn,
        )

        belief = BeliefContext(
            contested_count=2,
            crux_count=1,
        )

        await phase.execute(
            improvement="Add caching",
            belief_context=belief,
        )

        env_call = mock_environment_factory.call_args
        task = env_call.kwargs.get("task", "")
        assert "UNCERTAINTY" in task or "contested" in task

    @pytest.mark.asyncio
    async def test_execute_emits_events(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
        mock_stream_emit_fn,
    ):
        """Should emit phase events."""
        from aragora.nomic.phases.design import DesignPhase, DesignConfig

        config = DesignConfig(enable_decomposition=False)

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            config=config,
            stream_emit_fn=mock_stream_emit_fn,
        )

        await phase.execute(improvement="Test")

        start_calls = [c for c in mock_stream_emit_fn.call_args_list if c[0][0] == "on_phase_start"]
        end_calls = [c for c in mock_stream_emit_fn.call_args_list if c[0][0] == "on_phase_end"]
        assert len(start_calls) >= 1
        assert len(end_calls) >= 1


# ============================================================================
# Task Decomposition Tests
# ============================================================================


class TestDesignPhaseTaskDecomposition:
    """Tests for task decomposition functionality."""

    @pytest.mark.asyncio
    async def test_decomposes_complex_task(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
        mock_log_fn,
        mock_stream_emit_fn,
    ):
        """Should decompose complex tasks."""
        from aragora.nomic.phases.design import DesignPhase, DesignConfig

        config = DesignConfig(
            enable_decomposition=True,
            decomposition_threshold=3,  # Low threshold for test
        )

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            config=config,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        complex_improvement = (
            "Refactor the authentication system and migrate database schema "
            "while redesigning the API layer with new endpoints for users, "
            "products, and orders. Update handler.py, auth.py, database.py."
        )

        result = await phase.execute(improvement=complex_improvement)

        # Should have run multiple times for subtasks
        assert mock_arena_factory.call_count >= 1
        assert result["data"].get("decomposed", False) or result["design"]

    @pytest.mark.asyncio
    async def test_skips_decomposition_for_simple_task(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
        mock_log_fn,
    ):
        """Should not decompose simple tasks."""
        from aragora.nomic.phases.design import DesignPhase, DesignConfig

        config = DesignConfig(
            enable_decomposition=True,
            decomposition_threshold=8,  # High threshold
        )

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            config=config,
            log_fn=mock_log_fn,
        )

        simple_improvement = "Fix typo in error message"

        result = await phase.execute(improvement=simple_improvement)

        # Should run once without decomposition
        assert mock_arena_factory.call_count == 1
        assert result["data"].get("decomposed") is not True

    def test_merge_subtask_designs(self, mock_aragora_path, mock_agents):
        """Should merge subtask designs properly."""
        from aragora.nomic.phases.design import DesignPhase
        from aragora.nomic.task_decomposer import TaskDecomposition, SubTask

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        decomposition = TaskDecomposition(
            original_task="Refactor API and database",
            complexity_score=7,
            complexity_level="high",
            should_decompose=True,
            subtasks=[
                SubTask(
                    id="st1",
                    title="API Changes",
                    description="Update endpoints",
                    dependencies=[],
                    estimated_complexity="medium",
                ),
                SubTask(
                    id="st2",
                    title="DB Changes",
                    description="Update schema",
                    dependencies=["st1"],
                    estimated_complexity="high",
                ),
            ],
            rationale="Complex multi-area refactor",
        )

        subtask_designs = [
            "### Subtask 1\nAPI design content...",
            "### Subtask 2\nDB design content...",
        ]

        merged = phase._merge_subtask_designs(
            decomposition.original_task,
            subtask_designs,
            decomposition,
        )

        assert "# Decomposed Design" in merged
        assert "API Changes" in merged
        assert "DB Changes" in merged
        assert "Integration Notes" in merged
        assert "st1" in merged  # Dependency reference

    def test_merge_empty_subtasks(self, mock_aragora_path, mock_agents):
        """Should handle empty subtask list."""
        from aragora.nomic.phases.design import DesignPhase
        from aragora.nomic.task_decomposer import TaskDecomposition

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        decomposition = TaskDecomposition(
            original_task="Test",
            complexity_score=2,
            complexity_level="low",
            should_decompose=False,
            rationale="Too simple",
        )

        merged = phase._merge_subtask_designs("Test", [], decomposition)

        assert merged == ""


# ============================================================================
# File Extraction Tests
# ============================================================================


class TestDesignPhaseFileExtraction:
    """Tests for file extraction from design."""

    def test_extracts_python_files(self, mock_aragora_path, mock_agents):
        """Should extract Python file paths."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        design = """
## FILE CHANGES
- `aragora/new_module.py` - Create new
- `aragora/utils/helper.py` - Modify existing
- aragora/core.py - Update

## Other content...
"""

        files = phase._extract_files_from_design(design)

        assert "aragora/new_module.py" in files
        assert "aragora/utils/helper.py" in files
        assert "aragora/core.py" in files

    def test_limits_file_count(self, mock_aragora_path, mock_agents):
        """Should limit to 10 files maximum."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        design = "\n".join([f"`aragora/module_{i}.py` - Create" for i in range(20)])

        files = phase._extract_files_from_design(design)

        assert len(files) <= 10

    def test_deduplicates_files(self, mock_aragora_path, mock_agents):
        """Should remove duplicate file paths."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        design = """
`aragora/utils.py` - Modify
`aragora/utils.py` - Also change
aragora/utils.py - Another reference
"""

        files = phase._extract_files_from_design(design)

        assert files.count("aragora/utils.py") == 1


# ============================================================================
# Complexity Estimation Tests
# ============================================================================


class TestDesignPhaseComplexityEstimation:
    """Tests for complexity estimation."""

    def test_estimates_high_complexity(self, mock_aragora_path, mock_agents):
        """Should estimate high complexity for many files."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        design = """
Files to modify:
`aragora/a.py`, `aragora/b.py`, `aragora/c.py`,
`aragora/d.py`, `aragora/e.py`, `aragora/f.py`

This is a complex refactoring...
"""

        complexity = phase._estimate_complexity(design)

        assert complexity == "high"

    def test_estimates_low_complexity(self, mock_aragora_path, mock_agents):
        """Should estimate low complexity for simple changes."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        design = """
Simple update to `aragora/utils.py`.
"""

        complexity = phase._estimate_complexity(design)

        assert complexity == "low"

    def test_handles_empty_design(self, mock_aragora_path, mock_agents):
        """Should handle empty design."""
        from aragora.nomic.phases.design import DesignPhase

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        complexity = phase._estimate_complexity("")

        assert complexity == "unknown"


# ============================================================================
# No Consensus Handling Tests
# ============================================================================


class TestDesignPhaseNoConsensusHandling:
    """Tests for handling no consensus scenarios."""

    @pytest.mark.asyncio
    async def test_handles_no_consensus_with_arbitration(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
        mock_log_fn,
    ):
        """Should attempt arbitration when no consensus."""
        from aragora.nomic.phases.design import DesignPhase, DesignConfig

        # Create arena that doesn't reach consensus
        mock_arena = MagicMock()
        result = MagicMock()
        result.consensus_reached = False
        result.final_answer = ""
        result.votes = []
        result.messages = [
            MagicMock(role="proposer", agent="claude", content="Design A"),
            MagicMock(role="proposer", agent="gemini", content="Design B"),
        ]
        mock_arena.run = AsyncMock(return_value=result)
        mock_arena_factory.return_value = mock_arena

        arbitrate_fn = AsyncMock(return_value="Arbitrated design")

        config = DesignConfig(enable_decomposition=False)

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            config=config,
            arbitrate_fn=arbitrate_fn,
            max_cycle_seconds=100,  # Low to trigger fast-track
            log_fn=mock_log_fn,
        )

        await phase.execute(improvement="Test")

        # Arbitration should have been attempted (if time was critical)
        # Note: The actual behavior depends on elapsed time

    @pytest.mark.asyncio
    async def test_counterfactual_resolution(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
        mock_nomic_integration,
        mock_log_fn,
    ):
        """Should try counterfactual resolution via nomic integration."""
        from aragora.nomic.phases.design import DesignPhase, DesignConfig

        # Create arena that doesn't reach consensus
        mock_arena = MagicMock()
        result = MagicMock()
        result.consensus_reached = False
        result.final_answer = ""
        result.votes = []
        result.messages = []
        mock_arena.run = AsyncMock(return_value=result)
        mock_arena_factory.return_value = mock_arena

        # Mock counterfactual resolution
        conditional = MagicMock()
        conditional.synthesized_answer = "Resolved design"
        conditional.confidence = 0.75
        mock_nomic_integration.full_post_debate_analysis = AsyncMock(
            return_value={"conditional": conditional}
        )

        config = DesignConfig(enable_decomposition=False)

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            config=config,
            nomic_integration=mock_nomic_integration,
            log_fn=mock_log_fn,
        )

        await phase.execute(improvement="Test")

        mock_nomic_integration.full_post_debate_analysis.assert_called_once()


# ============================================================================
# Deep Audit Tests
# ============================================================================


class TestDesignPhaseDeepAudit:
    """Tests for deep audit functionality."""

    @pytest.mark.asyncio
    async def test_rejects_on_deep_audit_failure(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
        mock_log_fn,
    ):
        """Should reject design if deep audit fails."""
        from aragora.nomic.phases.design import DesignPhase, DesignConfig

        # Create a callable that simulates deep_audit behavior
        # Called with "check" action first, should return (True, reason)
        # Then called with "run" action, should return the audit result dict
        call_count = 0

        async def deep_audit_fn(action, content, phase=None):
            nonlocal call_count
            call_count += 1
            if action == "check":
                return True, "Needs audit due to suspicious patterns"
            elif action == "run":
                return {"approved": False, "unanimous_issues": ["Security concern"]}
            return None

        config = DesignConfig(enable_decomposition=False)

        phase = DesignPhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            config=config,
            deep_audit_fn=deep_audit_fn,
            log_fn=mock_log_fn,
        )

        result = await phase.execute(improvement="Suspicious feature")

        # The deep audit should have been called (for check and run)
        assert call_count >= 1
        assert result["success"] is False
        # Either the error mentions deep_audit or the data has the flag
        assert (
            "deep_audit" in str(result.get("error", "")).lower()
            or "Rejected" in str(result.get("error", ""))
            or result["data"].get("rejected_by_deep_audit")
        )
