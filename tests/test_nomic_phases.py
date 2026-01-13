"""
Tests for nomic loop phase factories and extracted phases.

Verifies that:
- Factory methods create valid phase instances
- Phase classes can be instantiated with minimal dependencies
- Result type conversions work correctly
"""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock

# Set environment variable before importing NomicLoop
os.environ["USE_EXTRACTED_PHASES"] = "1"


@pytest.fixture
def mock_agents():
    """Create mock agents for testing."""
    agents = []
    for name in ["claude", "codex", "gemini", "grok"]:
        agent = Mock()
        agent.name = name
        agent.generate = AsyncMock(return_value=f"Response from {name}")
        agents.append(agent)
    return agents


@pytest.fixture
def aragora_path(tmp_path):
    """Create a temporary aragora path."""
    return tmp_path


@pytest.fixture
def minimal_nomic_loop(aragora_path, mock_agents):
    """Create a minimal NomicLoop instance for testing factories."""
    # Import here to ensure env var is set
    from scripts.nomic_loop import NomicLoop

    loop = Mock(spec=NomicLoop)
    loop.aragora_path = aragora_path
    loop.cycle_count = 1
    loop.use_extracted_phases = True
    loop.agents = mock_agents
    loop.claude = mock_agents[0]
    loop.codex = mock_agents[1]
    loop.nomic_integration = None
    loop.max_cycle_seconds = 3600
    loop.initial_proposal = None
    loop._log = Mock()
    loop._stream_emit = Mock()
    loop._record_replay_event = Mock()
    loop.save_state = Mock()
    loop.get_current_features = Mock(return_value="Test features")

    return loop


class TestPhaseImports:
    """Test that phase classes can be imported."""

    def test_import_context_phase(self):
        """ContextPhase can be imported."""
        from scripts.nomic.phases import ContextPhase

        assert ContextPhase is not None

    def test_import_debate_phase(self):
        """DebatePhase can be imported."""
        from scripts.nomic.phases import DebatePhase, DebateConfig, PostDebateHooks

        assert DebatePhase is not None
        assert DebateConfig is not None
        assert PostDebateHooks is not None

    def test_import_design_phase(self):
        """DesignPhase can be imported."""
        from scripts.nomic.phases import DesignPhase, DesignConfig, BeliefContext

        assert DesignPhase is not None
        assert DesignConfig is not None
        assert BeliefContext is not None

    def test_import_implement_phase(self):
        """ImplementPhase can be imported."""
        from scripts.nomic.phases import ImplementPhase

        assert ImplementPhase is not None

    def test_import_verify_phase(self):
        """VerifyPhase can be imported."""
        from scripts.nomic.phases import VerifyPhase

        assert VerifyPhase is not None

    def test_import_commit_phase(self):
        """CommitPhase can be imported."""
        from scripts.nomic.phases import CommitPhase

        assert CommitPhase is not None


class TestPhaseInstantiation:
    """Test that phase classes can be instantiated."""

    def test_context_phase_instantiation(self, aragora_path, mock_agents):
        """ContextPhase can be instantiated with minimal deps."""
        from scripts.nomic.phases import ContextPhase

        phase = ContextPhase(
            aragora_path=aragora_path,
            claude_agent=mock_agents[0],
            codex_agent=mock_agents[1],
            cycle_count=1,
        )

        assert phase.aragora_path == aragora_path
        assert phase.cycle_count == 1

    def test_debate_phase_instantiation(self, aragora_path, mock_agents):
        """DebatePhase can be instantiated with minimal deps."""
        from scripts.nomic.phases import DebatePhase, DebateConfig

        phase = DebatePhase(
            aragora_path=aragora_path,
            agents=mock_agents,
            arena_factory=Mock(),
            environment_factory=Mock(),
            protocol_factory=Mock(),
            config=DebateConfig(),
            cycle_count=1,
        )

        assert phase.aragora_path == aragora_path
        assert len(phase.agents) == 4

    def test_design_phase_instantiation(self, aragora_path, mock_agents):
        """DesignPhase can be instantiated with minimal deps."""
        from scripts.nomic.phases import DesignPhase, DesignConfig

        phase = DesignPhase(
            aragora_path=aragora_path,
            agents=mock_agents,
            arena_factory=Mock(),
            environment_factory=Mock(),
            protocol_factory=Mock(),
            config=DesignConfig(),
            cycle_count=1,
        )

        assert phase.aragora_path == aragora_path

    def test_implement_phase_instantiation(self, aragora_path):
        """ImplementPhase can be instantiated with minimal deps."""
        from scripts.nomic.phases import ImplementPhase

        phase = ImplementPhase(
            aragora_path=aragora_path,
            cycle_count=1,
        )

        assert phase.aragora_path == aragora_path

    def test_verify_phase_instantiation(self, aragora_path):
        """VerifyPhase can be instantiated with minimal deps."""
        from scripts.nomic.phases import VerifyPhase

        phase = VerifyPhase(
            aragora_path=aragora_path,
            cycle_count=1,
        )

        assert phase.aragora_path == aragora_path

    def test_commit_phase_instantiation(self, aragora_path):
        """CommitPhase can be instantiated with minimal deps."""
        from scripts.nomic.phases import CommitPhase

        phase = CommitPhase(
            aragora_path=aragora_path,
            cycle_count=1,
        )

        assert phase.aragora_path == aragora_path


class TestResultTypes:
    """Test result type structures."""

    def test_context_result_structure(self):
        """ContextResult has expected fields."""
        from scripts.nomic.phases import ContextResult

        result = ContextResult(
            success=True,
            data={"agents_succeeded": 2},
            duration_seconds=10.5,
            codebase_summary="Test summary",
            recent_changes="",
            open_issues=[],
        )

        assert result["success"] is True
        assert result["codebase_summary"] == "Test summary"

    def test_debate_result_structure(self):
        """DebateResult has expected fields."""
        from scripts.nomic.phases import DebateResult

        result = DebateResult(
            success=True,
            data={},
            duration_seconds=60.0,
            improvement="Add new feature",
            consensus_reached=True,
            confidence=0.85,
            votes=[("agent1", "approve")],
        )

        assert result["consensus_reached"] is True
        assert result["confidence"] == 0.85

    def test_design_result_structure(self):
        """DesignResult has expected fields."""
        from scripts.nomic.phases import DesignResult

        result = DesignResult(
            success=True,
            data={},
            duration_seconds=45.0,
            design="Implementation design...",
            files_affected=["aragora/new_module.py"],
            complexity_estimate="medium",
        )

        assert result["files_affected"] == ["aragora/new_module.py"]

    def test_implement_result_structure(self):
        """ImplementResult has expected fields."""
        from scripts.nomic.phases import ImplementResult

        result = ImplementResult(
            success=True,
            data={},
            duration_seconds=120.0,
            files_modified=["aragora/new_module.py"],
            diff_summary="1 file changed, 50 insertions",
        )

        assert len(result["files_modified"]) == 1

    def test_verify_result_structure(self):
        """VerifyResult has expected fields."""
        from scripts.nomic.phases import VerifyResult

        result = VerifyResult(
            success=True,
            data={},
            duration_seconds=30.0,
            tests_passed=True,
            test_output="All tests passed",
            syntax_valid=True,
        )

        assert result["tests_passed"] is True

    def test_commit_result_structure(self):
        """CommitResult has expected fields."""
        from scripts.nomic.phases import CommitResult

        result = CommitResult(
            success=True,
            data={},
            duration_seconds=5.0,
            commit_hash="abc123",
            committed=True,
        )

        assert result["commit_hash"] == "abc123"


class TestDebateConfig:
    """Test DebateConfig defaults and customization."""

    def test_debate_config_defaults(self):
        """DebateConfig has sensible defaults."""
        from scripts.nomic.phases import DebateConfig

        config = DebateConfig()

        assert config.rounds == 2
        assert config.consensus_mode == "judge"
        assert config.proposer_count == 4

    def test_debate_config_customization(self):
        """DebateConfig can be customized."""
        from scripts.nomic.phases import DebateConfig

        config = DebateConfig(
            rounds=3,
            consensus_mode="unanimous",
            proposer_count=2,
        )

        assert config.rounds == 3
        assert config.consensus_mode == "unanimous"


class TestDesignConfig:
    """Test DesignConfig defaults and customization."""

    def test_design_config_defaults(self):
        """DesignConfig has sensible defaults."""
        from scripts.nomic.phases import DesignConfig

        config = DesignConfig()

        assert config.rounds == 2
        assert config.early_stopping is True
        assert "CLAUDE.md" in config.protected_files

    def test_design_config_protected_files(self):
        """DesignConfig protects critical files."""
        from scripts.nomic.phases import DesignConfig

        config = DesignConfig()

        assert "core.py" in config.protected_files
        assert "aragora/__init__.py" in config.protected_files
        assert ".env" in config.protected_files


class TestPostDebateHooks:
    """Test PostDebateHooks structure."""

    def test_hooks_all_optional(self):
        """All hooks are optional."""
        from scripts.nomic.phases import PostDebateHooks

        hooks = PostDebateHooks()

        assert hooks.on_consensus_stored is None
        assert hooks.on_calibration_recorded is None
        assert hooks.on_elo_recorded is None

    def test_hooks_can_be_set(self):
        """Hooks can be set to callables."""
        from scripts.nomic.phases import PostDebateHooks

        def my_hook(*args):
            pass

        hooks = PostDebateHooks(
            on_consensus_stored=my_hook,
            on_elo_recorded=my_hook,
        )

        assert hooks.on_consensus_stored is my_hook
        assert hooks.on_elo_recorded is my_hook


class TestBeliefContext:
    """Test BeliefContext structure."""

    def test_belief_context_defaults(self):
        """BeliefContext has sensible defaults."""
        from scripts.nomic.phases import BeliefContext

        ctx = BeliefContext()

        assert ctx.contested_count == 0
        assert ctx.crux_count == 0
        assert ctx.convergence_achieved is False

    def test_belief_context_to_string_empty(self):
        """BeliefContext.to_string() returns empty for no uncertainty."""
        from scripts.nomic.phases import BeliefContext

        ctx = BeliefContext()

        assert ctx.to_string() == ""

    def test_belief_context_to_string_with_data(self):
        """BeliefContext.to_string() includes uncertainty info."""
        from scripts.nomic.phases import BeliefContext

        ctx = BeliefContext(
            contested_count=3,
            crux_count=1,
            convergence_achieved=True,
        )

        result = ctx.to_string()
        assert "3 contested claims" in result
        assert "1 crux" in result
        assert "converged" in result.lower()


class TestLearningContext:
    """Test LearningContext structure."""

    def test_learning_context_defaults(self):
        """LearningContext has empty defaults."""
        from scripts.nomic.phases import LearningContext

        ctx = LearningContext()

        assert ctx.failure_lessons == ""
        assert ctx.successful_patterns == ""

    def test_learning_context_to_string(self):
        """LearningContext.to_string() combines all fields."""
        from scripts.nomic.phases import LearningContext

        ctx = LearningContext(
            failure_lessons="Don't repeat X",
            successful_patterns="Pattern Y works",
        )

        result = ctx.to_string()
        assert "Don't repeat X" in result
        assert "Pattern Y works" in result
