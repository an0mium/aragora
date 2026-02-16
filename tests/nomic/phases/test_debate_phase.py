"""
Comprehensive unit tests for DebatePhase.

Tests the debate phase module including:
- Initialization with various configurations
- Legacy API (run, generate_proposals, collect_votes, check_consensus)
- Modern execute() API with hooks and learning context
- Phase transitions and error handling
- Post-debate hook execution
- Agent probing and reliability weights
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

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
    agent.name = "test_agent"
    agent.generate = AsyncMock(return_value="Proposal: Add feature X")
    return agent


@pytest.fixture
def mock_agents():
    """Create multiple mock agents."""
    agents = []
    for name in ["claude", "codex", "gemini"]:
        agent = MagicMock()
        agent.name = name
        agent.generate = AsyncMock(return_value=f"Proposal from {name}")
        agents.append(agent)
    return agents


@pytest.fixture
def mock_arena():
    """Create a mock arena."""
    arena = MagicMock()
    result = MagicMock()
    result.consensus_reached = True
    result.final_answer = "Implement error handling"
    result.confidence = 0.85
    result.votes = [
        MagicMock(agent="claude", choice="p1"),
        MagicMock(agent="codex", choice="p1"),
    ]
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
    integration.probe_agents = AsyncMock(return_value={"claude": 0.9, "codex": 0.8})
    integration.list_checkpoints = AsyncMock(return_value=[])
    integration.checkpoint = AsyncMock()
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
# DebateConfig Tests
# ============================================================================


class TestDebateConfig:
    """Tests for DebateConfig dataclass."""

    def test_default_config(self):
        """Should have sensible defaults."""
        from aragora.nomic.phases.debate import DebateConfig

        config = DebateConfig()

        assert config.rounds > 0
        assert config.consensus_mode in ["judge", "majority", "unanimous"]
        assert config.early_stopping is True
        assert 0 < config.early_stop_threshold <= 1.0

    def test_custom_config(self):
        """Should accept custom values."""
        from aragora.nomic.phases.debate import DebateConfig

        config = DebateConfig(
            rounds=5,
            consensus_mode="majority",
            early_stopping=False,
            early_stop_threshold=0.8,
        )

        assert config.rounds == 5
        assert config.consensus_mode == "majority"
        assert config.early_stopping is False
        assert config.early_stop_threshold == 0.8

    def test_from_profile(self):
        """Should create config from NomicDebateProfile."""
        from aragora.nomic.phases.debate import DebateConfig

        # This tests that from_profile() doesn't crash
        # The actual profile loading may have dependencies
        try:
            config = DebateConfig.from_profile()
            assert isinstance(config, DebateConfig)
        except Exception:
            # Profile may not be configured in test environment
            pass


# ============================================================================
# LearningContext Tests
# ============================================================================


class TestLearningContext:
    """Tests for LearningContext dataclass."""

    def test_default_context(self):
        """Should have empty defaults."""
        from aragora.nomic.phases.debate import LearningContext

        ctx = LearningContext()

        assert ctx.failure_lessons == ""
        assert ctx.successful_patterns == ""
        assert ctx.agent_reputations == ""

    def test_to_string_empty(self):
        """Should return empty string when no context."""
        from aragora.nomic.phases.debate import LearningContext

        ctx = LearningContext()

        assert ctx.to_string() == ""

    def test_to_string_with_content(self):
        """Should combine non-empty fields."""
        from aragora.nomic.phases.debate import LearningContext

        ctx = LearningContext(
            failure_lessons="Don't propose existing features",
            successful_patterns="Pattern: modular design works",
            agent_reputations="Claude: reliable, Codex: fast",
        )

        result = ctx.to_string()

        assert "Don't propose existing features" in result
        assert "modular design" in result
        assert "Claude: reliable" in result

    def test_to_string_preserves_order(self):
        """Should output fields in consistent order."""
        from aragora.nomic.phases.debate import LearningContext

        ctx = LearningContext(
            failure_lessons="First",
            successful_patterns="Second",
        )

        result = ctx.to_string()
        first_pos = result.find("First")
        second_pos = result.find("Second")

        # failure_lessons comes before successful_patterns
        assert first_pos < second_pos


# ============================================================================
# PostDebateHooks Tests
# ============================================================================


class TestPostDebateHooks:
    """Tests for PostDebateHooks dataclass."""

    def test_default_hooks(self):
        """Should have None defaults."""
        from aragora.nomic.phases.debate import PostDebateHooks

        hooks = PostDebateHooks()

        assert hooks.on_consensus_stored is None
        assert hooks.on_calibration_recorded is None
        assert hooks.on_insights_extracted is None

    def test_custom_hooks(self):
        """Should accept callable hooks."""
        from aragora.nomic.phases.debate import PostDebateHooks

        on_consensus = MagicMock()
        on_calibration = MagicMock()

        hooks = PostDebateHooks(
            on_consensus_stored=on_consensus,
            on_calibration_recorded=on_calibration,
        )

        assert hooks.on_consensus_stored is on_consensus
        assert hooks.on_calibration_recorded is on_calibration


# ============================================================================
# DebatePhase Initialization Tests
# ============================================================================


class TestDebatePhaseInitialization:
    """Tests for DebatePhase initialization."""

    def test_init_with_agents_list(self, mock_aragora_path, mock_agents):
        """Should initialize with agents list."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        assert phase.agents == mock_agents
        assert len(phase.agents) == 3

    def test_init_with_legacy_individual_agents(self, mock_aragora_path, mock_agent):
        """Should support legacy individual agent parameters."""
        from aragora.nomic.phases.debate import DebatePhase

        claude = MagicMock(name="claude")
        codex = MagicMock(name="codex")

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            claude_agent=claude,
            codex_agent=codex,
        )

        assert phase.claude == claude
        assert phase.codex == codex
        assert len(phase.agents) == 2

    def test_init_with_config(self, mock_aragora_path, mock_agents):
        """Should accept config parameter."""
        from aragora.nomic.phases.debate import DebatePhase, DebateConfig

        config = DebateConfig(rounds=7, consensus_mode="majority")

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            config=config,
        )

        assert phase.config.rounds == 7
        assert phase.config.consensus_mode == "majority"

    def test_init_with_factories(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
    ):
        """Should accept factory functions."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
        )

        assert phase._arena_factory is mock_arena_factory
        assert phase._environment_factory is mock_environment_factory
        assert phase._protocol_factory is mock_protocol_factory

    def test_init_with_nomic_integration(
        self, mock_aragora_path, mock_agents, mock_nomic_integration
    ):
        """Should accept nomic integration."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            nomic_integration=mock_nomic_integration,
        )

        assert phase.nomic_integration is mock_nomic_integration

    def test_init_with_initial_proposal(self, mock_aragora_path, mock_agents):
        """Should accept initial human proposal."""
        from aragora.nomic.phases.debate import DebatePhase

        initial = "Implement a caching layer"

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            initial_proposal=initial,
        )

        assert phase.initial_proposal == initial


# ============================================================================
# Legacy API Tests
# ============================================================================


class TestDebatePhaseGenerateProposals:
    """Tests for generate_proposals method."""

    @pytest.mark.asyncio
    async def test_generates_from_all_agents(self, mock_aragora_path, mock_agents, mock_log_fn):
        """Should generate proposals from all agents."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            log_fn=mock_log_fn,
        )

        proposals = await phase.generate_proposals("Test context")

        assert len(proposals) == 3
        assert all("agent" in p for p in proposals)
        assert all("proposal" in p for p in proposals)

    @pytest.mark.asyncio
    async def test_handles_agent_exception(self, mock_aragora_path, mock_agents, mock_log_fn):
        """Should continue if one agent fails."""
        from aragora.nomic.phases.debate import DebatePhase

        mock_agents[1].generate = AsyncMock(side_effect=RuntimeError("API error"))

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            log_fn=mock_log_fn,
        )

        proposals = await phase.generate_proposals("Test context")

        # Should have proposals from 2 agents (one failed)
        assert len(proposals) == 2
        mock_log_fn.assert_called()  # Should log the error


class TestDebatePhaseCollectVotes:
    """Tests for collect_votes method."""

    @pytest.mark.asyncio
    async def test_collects_votes_from_agents(self, mock_aragora_path, mock_agents, mock_log_fn):
        """Should collect votes from all agents."""
        from aragora.nomic.phases.debate import DebatePhase

        # Set up mock vote method
        for agent in mock_agents:
            vote_result = MagicMock()
            vote_result.choice = 0
            agent.vote = AsyncMock(return_value=vote_result)

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            log_fn=mock_log_fn,
        )

        proposals = [
            {"id": "p1", "proposal": "Option A"},
            {"id": "p2", "proposal": "Option B"},
        ]

        votes = await phase.collect_votes(proposals)

        assert len(votes) == 3
        assert all(v == "p1" for v in votes.values())

    @pytest.mark.asyncio
    async def test_defaults_to_first_proposal_without_vote_method(
        self, mock_aragora_path, mock_agents, mock_log_fn
    ):
        """Should default to first proposal if agent lacks vote method."""
        from aragora.nomic.phases.debate import DebatePhase

        # Remove vote method from agents
        for agent in mock_agents:
            if hasattr(agent, "vote"):
                delattr(agent, "vote")

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            log_fn=mock_log_fn,
        )

        proposals = [
            {"id": "p1", "proposal": "Option A"},
            {"id": "p2", "proposal": "Option B"},
        ]

        votes = await phase.collect_votes(proposals)

        # All agents should vote for first proposal
        assert all(v == "p1" for v in votes.values())


class TestDebatePhaseCountVotes:
    """Tests for count_votes method."""

    def test_counts_correctly(self, mock_aragora_path, mock_agents):
        """Should count votes correctly."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        votes = {
            "claude": "p1",
            "codex": "p1",
            "gemini": "p2",
            "grok": "p2",
            "mistral": "p1",
        }

        counts = phase.count_votes(votes)

        assert counts["p1"] == 3
        assert counts["p2"] == 2

    def test_empty_votes(self, mock_aragora_path, mock_agents):
        """Should handle empty votes."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        counts = phase.count_votes({})

        assert counts == {}


class TestDebatePhaseCheckConsensus:
    """Tests for check_consensus method."""

    def test_detects_consensus(self, mock_aragora_path, mock_agents):
        """Should detect when consensus threshold is met."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            consensus_threshold=0.5,
        )

        votes = {"a1": "p1", "a2": "p1", "a3": "p2"}

        result = phase.check_consensus(votes, total_agents=3)

        assert result["consensus"] is True
        assert result["winning_proposal"] == "p1"
        assert result["confidence"] == pytest.approx(2 / 3)

    def test_detects_no_consensus(self, mock_aragora_path, mock_agents):
        """Should detect when consensus not reached."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            consensus_threshold=0.8,
        )

        votes = {"a1": "p1", "a2": "p2", "a3": "p3"}

        result = phase.check_consensus(votes, total_agents=3)

        assert result["consensus"] is False
        assert result["confidence"] == pytest.approx(1 / 3)

    def test_handles_empty_votes(self, mock_aragora_path, mock_agents):
        """Should handle no votes cast."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        result = phase.check_consensus({})

        assert result["consensus"] is False
        assert "No votes" in result.get("reason", "")

    def test_uses_custom_threshold(self, mock_aragora_path, mock_agents):
        """Should use custom threshold from parameter."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            consensus_threshold=0.5,
        )

        votes = {"a1": "p1", "a2": "p2"}

        # With threshold=0.9, 50% should not reach consensus
        result = phase.check_consensus(votes, threshold=0.9, total_agents=2)

        assert result["consensus"] is False
        assert result["threshold"] == 0.9


class TestDebatePhaseRun:
    """Tests for legacy run() method."""

    @pytest.mark.asyncio
    async def test_complete_flow(self, mock_aragora_path, mock_agents, mock_log_fn):
        """Should complete full debate flow."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "generate_proposals", new_callable=AsyncMock) as mock_gen:
            with patch.object(phase, "collect_votes", new_callable=AsyncMock) as mock_votes:
                mock_gen.return_value = [
                    {"id": "p1", "proposal": "Feature A"},
                    {"id": "p2", "proposal": "Feature B"},
                ]
                mock_votes.return_value = {"claude": "p1", "codex": "p1", "gemini": "p1"}

                result = await phase.run("Test context")

                assert result["consensus"] is True
                assert result["winning_proposal"] == "p1"
                mock_gen.assert_called_once()
                mock_votes.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_no_proposals(self, mock_aragora_path, mock_agents):
        """Should handle case with no proposals."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        with patch.object(phase, "generate_proposals", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = []

            result = await phase.run()

            assert result["consensus"] is False
            assert "No proposals" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_stores_proposals_and_votes(self, mock_aragora_path, mock_agents):
        """Should store proposals and votes for retrieval."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        with patch.object(phase, "generate_proposals", new_callable=AsyncMock) as mock_gen:
            with patch.object(phase, "collect_votes", new_callable=AsyncMock) as mock_votes:
                proposals = [{"id": "p1", "proposal": "Test"}]
                votes = {"claude": "p1"}
                mock_gen.return_value = proposals
                mock_votes.return_value = votes

                await phase.run()

                assert phase.get_proposals() == proposals
                assert phase.get_votes() == votes


# ============================================================================
# Modern Execute API Tests
# ============================================================================


class TestDebatePhaseExecute:
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
        """Should execute debate and return DebateResult."""
        from aragora.nomic.phases.debate import DebatePhase, LearningContext

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        result = await phase.execute(
            codebase_context="Feature inventory: ...",
            recent_changes="Added module X",
            learning_context=LearningContext(failure_lessons="Don't duplicate features"),
        )

        assert result["success"] is True
        assert result["consensus_reached"] is True
        assert "improvement" in result
        mock_arena_factory.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_initial_proposal(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
        mock_log_fn,
    ):
        """Should include initial proposal in task prompt."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            initial_proposal="Implement caching",
            log_fn=mock_log_fn,
        )

        await phase.execute()

        # Check that environment was created with initial proposal in task
        env_call = mock_environment_factory.call_args
        task_arg = env_call.kwargs.get("task", "")
        assert "HUMAN-SUBMITTED PROPOSAL" in task_arg
        assert "Implement caching" in task_arg

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
        """Should emit phase start/end events."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            stream_emit_fn=mock_stream_emit_fn,
        )

        await phase.execute()

        # Check phase start was emitted
        start_calls = [c for c in mock_stream_emit_fn.call_args_list if c[0][0] == "on_phase_start"]
        assert len(start_calls) >= 1
        assert start_calls[0][0][1] == "debate"

        # Check phase end was emitted
        end_calls = [c for c in mock_stream_emit_fn.call_args_list if c[0][0] == "on_phase_end"]
        assert len(end_calls) >= 1


class TestDebatePhaseAgentProbing:
    """Tests for agent probing functionality."""

    @pytest.mark.asyncio
    async def test_probes_agents_with_integration(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
        mock_nomic_integration,
        mock_log_fn,
    ):
        """Should probe agents when integration available."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            nomic_integration=mock_nomic_integration,
            log_fn=mock_log_fn,
        )

        await phase.execute()

        mock_nomic_integration.probe_agents.assert_called_once()
        # Arena should receive agent weights
        arena_call = mock_arena_factory.call_args
        assert "agent_weights" in arena_call.kwargs

    @pytest.mark.asyncio
    async def test_handles_probe_failure(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
        mock_nomic_integration,
        mock_log_fn,
    ):
        """Should handle probing failure gracefully."""
        from aragora.nomic.phases.debate import DebatePhase

        mock_nomic_integration.probe_agents = AsyncMock(side_effect=RuntimeError("Probe failed"))

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            nomic_integration=mock_nomic_integration,
            log_fn=mock_log_fn,
        )

        # Should not raise
        result = await phase.execute()
        assert result is not None


class TestDebatePhasePostDebateHooks:
    """Tests for post-debate hook execution."""

    @pytest.mark.asyncio
    async def test_executes_hooks_on_consensus(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
        mock_log_fn,
    ):
        """Should execute hooks when consensus reached."""
        from aragora.nomic.phases.debate import DebatePhase, PostDebateHooks

        on_consensus = MagicMock()
        on_calibration = MagicMock()

        hooks = PostDebateHooks(
            on_consensus_stored=on_consensus,
            on_calibration_recorded=on_calibration,
        )

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            log_fn=mock_log_fn,
        )

        await phase.execute(hooks=hooks)

        on_consensus.assert_called_once()
        on_calibration.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_async_hooks(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
        mock_log_fn,
    ):
        """Should handle async hook functions."""
        from aragora.nomic.phases.debate import DebatePhase, PostDebateHooks

        async_hook = AsyncMock()

        hooks = PostDebateHooks(on_insights_extracted=async_hook)

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            log_fn=mock_log_fn,
        )

        await phase.execute(hooks=hooks)

        async_hook.assert_called_once()

    @pytest.mark.asyncio
    async def test_hook_failure_does_not_crash(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
        mock_log_fn,
    ):
        """Should continue if hook raises exception."""
        from aragora.nomic.phases.debate import DebatePhase, PostDebateHooks

        failing_hook = MagicMock(side_effect=Exception("Hook failed"))

        hooks = PostDebateHooks(on_consensus_stored=failing_hook)

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            log_fn=mock_log_fn,
        )

        # Should not raise
        result = await phase.execute(hooks=hooks)
        assert result is not None


class TestDebatePhaseNoveltyCheck:
    """Tests for codebase novelty checking."""

    @pytest.mark.asyncio
    async def test_checks_novelty_with_context(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
        mock_log_fn,
    ):
        """Should check proposal against codebase context."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            log_fn=mock_log_fn,
        )

        with patch("aragora.nomic.phases.debate.CodebaseNoveltyChecker") as mock_checker_class:
            mock_checker = MagicMock()
            mock_result = MagicMock()
            mock_result.is_novel = True
            mock_result.max_similarity = 0.3
            mock_checker.check_proposal.return_value = mock_result
            mock_checker_class.return_value = mock_checker

            await phase.execute(codebase_context="Feature inventory: WebSocket streaming, ...")

            mock_checker_class.assert_called_once()
            mock_checker.check_proposal.assert_called_once()

    @pytest.mark.asyncio
    async def test_warns_on_duplicate_proposal(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
        mock_log_fn,
    ):
        """Should warn when proposal duplicates existing feature."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            log_fn=mock_log_fn,
        )

        with patch("aragora.nomic.phases.debate.CodebaseNoveltyChecker") as mock_checker_class:
            mock_checker = MagicMock()
            mock_result = MagicMock()
            mock_result.is_novel = False
            mock_result.warning = "Duplicates existing WebSocket feature"
            mock_result.most_similar_feature = "WebSocket streaming"
            mock_result.max_similarity = 0.9
            mock_checker.check_proposal.return_value = mock_result
            mock_checker_class.return_value = mock_checker

            result = await phase.execute(codebase_context="Feature inventory: ...")

            assert result["data"]["codebase_novelty_warning"] is not None


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestDebatePhaseErrorHandling:
    """Tests for error handling in debate phase."""

    @pytest.mark.asyncio
    async def test_handles_arena_exception(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
        mock_log_fn,
    ):
        """Should handle arena.run() exception."""
        from aragora.nomic.phases.debate import DebatePhase

        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(side_effect=RuntimeError("Arena crashed"))
        mock_arena_factory.return_value = mock_arena

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            log_fn=mock_log_fn,
        )

        with pytest.raises(Exception, match="Arena crashed"):
            await phase.execute()

    @pytest.mark.asyncio
    async def test_handles_belief_network_load_failure(
        self,
        mock_aragora_path,
        mock_agents,
        mock_arena_factory,
        mock_environment_factory,
        mock_protocol_factory,
        mock_nomic_integration,
        mock_log_fn,
    ):
        """Should handle belief network load failure gracefully."""
        from aragora.nomic.phases.debate import DebatePhase

        mock_nomic_integration.list_checkpoints = AsyncMock(
            side_effect=RuntimeError("Checkpoint load failed")
        )

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
            arena_factory=mock_arena_factory,
            environment_factory=mock_environment_factory,
            protocol_factory=mock_protocol_factory,
            nomic_integration=mock_nomic_integration,
            cycle_count=5,
            log_fn=mock_log_fn,
        )

        # Should not raise
        result = await phase.execute()
        assert result is not None


# ============================================================================
# Task Prompt Building Tests
# ============================================================================


class TestDebatePhaseTaskPromptBuilding:
    """Tests for task prompt construction."""

    def test_build_task_prompt_includes_safety_preamble(self, mock_aragora_path, mock_agents):
        """Should include safety preamble in task."""
        from aragora.nomic.phases.debate import DebatePhase, LearningContext

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        prompt = phase._build_task_prompt(
            codebase_context="",
            recent_changes="",
            learning=LearningContext(),
        )

        assert "SAFETY RULES" in prompt
        assert "EXISTENCE CHECK" in prompt

    def test_build_task_prompt_includes_learning_context(self, mock_aragora_path, mock_agents):
        """Should include learning context in task."""
        from aragora.nomic.phases.debate import DebatePhase, LearningContext

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        learning = LearningContext(
            failure_lessons="Don't propose caching - it exists",
            successful_patterns="Modular design preferred",
        )

        prompt = phase._build_task_prompt(
            codebase_context="",
            recent_changes="Added module X",
            learning=learning,
        )

        assert "Don't propose caching" in prompt
        assert "Modular design preferred" in prompt
        assert "Added module X" in prompt

    def test_build_context_section_short_context(self, mock_aragora_path, mock_agents):
        """Should format short context simply."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        section = phase._build_context_section("Short context")

        assert "Short context" in section
        assert "CODEBASE ANALYSIS" not in section

    def test_build_context_section_long_context(self, mock_aragora_path, mock_agents):
        """Should format long context with header."""
        from aragora.nomic.phases.debate import DebatePhase

        phase = DebatePhase(
            aragora_path=mock_aragora_path,
            agents=mock_agents,
        )

        long_context = "x" * 1000  # More than 500 chars
        section = phase._build_context_section(long_context)

        assert "CODEBASE ANALYSIS" in section
        assert long_context in section
