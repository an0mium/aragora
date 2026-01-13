"""
Tests for DebateHooks.

Tests cover:
- DebateHooks initialization
- Round completion hooks (position recording)
- Debate completion hooks (relationship updates, memory storage)
- Memory tracking
- Evidence storage
- Grounded verdict creation
- Error handling (graceful failures)
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from unittest.mock import Mock, MagicMock, AsyncMock, patch

from aragora.debate.debate_hooks import DebateHooks, HooksConfig


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_position_ledger():
    """Create a mock position ledger."""
    ledger = Mock()
    ledger.record_position = Mock()
    return ledger


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    elo = Mock()
    elo.update_relationships_batch = Mock()
    return elo


@pytest.fixture
def mock_memory_manager():
    """Create a mock memory manager."""
    manager = Mock()
    manager.store_debate_outcome = Mock()
    manager.store_evidence = Mock()
    manager.track_retrieved_ids = Mock()
    manager.update_memory_outcomes = Mock()
    return manager


@pytest.fixture
def mock_evidence_grounder():
    """Create a mock evidence grounder."""
    grounder = Mock()
    grounder.create_grounded_verdict = Mock(return_value={"verdict": "grounded"})
    grounder.verify_claims_formally = AsyncMock()
    return grounder


@pytest.fixture
def mock_context():
    """Create a mock debate context."""
    ctx = Mock()
    ctx.debate_id = "test-debate-123"
    ctx.task = "Discuss the best programming language"
    return ctx


@pytest.fixture
def mock_result():
    """Create a mock debate result."""
    result = Mock()
    result.consensus = "Python is versatile"
    result.consensus_confidence = 0.85
    result.final_answer = "Python is the best choice for beginners"
    result.confidence = 0.9
    result.winner = "claude"
    result.votes = [
        Mock(agent="claude", choice="python"),
        Mock(agent="grok", choice="rust"),
    ]
    result.grounded_verdict = {"claims": []}
    return result


@pytest.fixture
def mock_agents():
    """Create mock agents."""
    return [
        Mock(name="claude"),
        Mock(name="grok"),
        Mock(name="gemini"),
    ]


# ============================================================================
# Initialization Tests
# ============================================================================


class TestDebateHooksInit:
    """Tests for DebateHooks initialization."""

    def test_empty_init(self):
        """Test hooks can be created with no subsystems."""
        hooks = DebateHooks()

        assert hooks.position_ledger is None
        assert hooks.elo_system is None
        assert hooks.memory_manager is None
        assert hooks.evidence_grounder is None

    def test_init_with_subsystems(
        self, mock_position_ledger, mock_elo_system, mock_memory_manager
    ):
        """Test hooks accepts subsystems."""
        hooks = DebateHooks(
            position_ledger=mock_position_ledger,
            elo_system=mock_elo_system,
            memory_manager=mock_memory_manager,
        )

        assert hooks.position_ledger is mock_position_ledger
        assert hooks.elo_system is mock_elo_system
        assert hooks.memory_manager is mock_memory_manager


# ============================================================================
# Round Completion Tests
# ============================================================================


class TestRoundCompletionHooks:
    """Tests for round completion hooks."""

    def test_on_round_complete_records_positions(
        self, mock_context, mock_position_ledger
    ):
        """Test on_round_complete records all agent positions."""
        hooks = DebateHooks(position_ledger=mock_position_ledger)

        proposals = {
            "claude": "Python is great for beginners",
            "grok": "Rust is more performant",
        }

        hooks.on_round_complete(mock_context, round_num=1, proposals=proposals)

        assert mock_position_ledger.record_position.call_count == 2

    def test_on_round_complete_without_ledger(self, mock_context):
        """Test on_round_complete handles missing ledger gracefully."""
        hooks = DebateHooks()

        proposals = {"claude": "Some proposal"}

        # Should not raise
        hooks.on_round_complete(mock_context, round_num=1, proposals=proposals)

    def test_on_round_complete_with_domain(
        self, mock_context, mock_position_ledger
    ):
        """Test on_round_complete passes domain to ledger."""
        hooks = DebateHooks(position_ledger=mock_position_ledger)

        proposals = {"claude": "Security best practices"}

        hooks.on_round_complete(
            mock_context, round_num=1, proposals=proposals, domain="security"
        )

        call_args = mock_position_ledger.record_position.call_args
        assert call_args.kwargs.get("domain") == "security"

    def test_record_position_truncates_long_content(
        self, mock_position_ledger
    ):
        """Test position content is truncated to 1000 chars."""
        hooks = DebateHooks(position_ledger=mock_position_ledger)

        long_content = "x" * 2000

        hooks._record_position(
            agent_name="claude",
            content=long_content,
            debate_id="test-123",
            round_num=1,
        )

        call_args = mock_position_ledger.record_position.call_args
        assert len(call_args.kwargs.get("claim", "")) == 1000

    def test_record_position_handles_errors(self, mock_position_ledger):
        """Test position recording handles errors gracefully."""
        mock_position_ledger.record_position.side_effect = ValueError("Test error")
        hooks = DebateHooks(position_ledger=mock_position_ledger)

        # Should not raise
        hooks._record_position(
            agent_name="claude",
            content="Test",
            debate_id="test-123",
            round_num=1,
        )


# ============================================================================
# Debate Completion Tests
# ============================================================================


class TestDebateCompletionHooks:
    """Tests for debate completion hooks."""

    @pytest.mark.asyncio
    async def test_on_debate_complete_updates_relationships(
        self, mock_context, mock_result, mock_agents, mock_elo_system
    ):
        """Test on_debate_complete updates ELO relationships."""
        hooks = DebateHooks(elo_system=mock_elo_system)

        await hooks.on_debate_complete(
            ctx=mock_context,
            result=mock_result,
            agents=mock_agents,
            task="Test task",
        )

        mock_elo_system.update_relationships_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_debate_complete_stores_outcome(
        self, mock_context, mock_result, mock_agents, mock_memory_manager
    ):
        """Test on_debate_complete stores debate outcome."""
        hooks = DebateHooks(memory_manager=mock_memory_manager)

        await hooks.on_debate_complete(
            ctx=mock_context,
            result=mock_result,
            agents=mock_agents,
            task="Test task",
            belief_cruxes=["crux1", "crux2"],
        )

        mock_memory_manager.store_debate_outcome.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_debate_complete_verifies_claims(
        self, mock_context, mock_result, mock_agents, mock_evidence_grounder
    ):
        """Test on_debate_complete triggers formal verification."""
        hooks = DebateHooks(evidence_grounder=mock_evidence_grounder)

        await hooks.on_debate_complete(
            ctx=mock_context,
            result=mock_result,
            agents=mock_agents,
            task="Test task",
        )

        mock_evidence_grounder.verify_claims_formally.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_debate_complete_handles_all_errors(
        self, mock_context, mock_result, mock_agents
    ):
        """Test on_debate_complete handles subsystem errors gracefully."""
        elo = Mock()
        # Use specific exception types that are caught
        elo.update_relationships_batch.side_effect = RuntimeError("ELO error")

        memory = Mock()
        memory.store_debate_outcome.side_effect = ValueError("Memory error")
        memory.track_retrieved_ids = Mock()
        memory.update_memory_outcomes = Mock()

        hooks = DebateHooks(elo_system=elo, memory_manager=memory)

        # Should not raise - errors are logged and swallowed
        await hooks.on_debate_complete(
            ctx=mock_context,
            result=mock_result,
            agents=mock_agents,
            task="Test task",
        )


class TestRelationshipUpdates:
    """Tests for relationship update logic."""

    def test_update_relationships_batch_structure(self, mock_elo_system):
        """Test relationship updates have correct batch structure."""
        hooks = DebateHooks(elo_system=mock_elo_system)

        hooks._update_relationships(
            debate_id="test-123",
            participants=["claude", "grok", "gemini"],
            winner="claude",
            votes=[
                Mock(agent="claude", choice="python"),
                Mock(agent="grok", choice="python"),
                Mock(agent="gemini", choice="rust"),
            ],
        )

        call_args = mock_elo_system.update_relationships_batch.call_args
        updates = call_args[0][0]

        # 3 participants -> 3 pairs
        assert len(updates) == 3

        # Check structure
        for update in updates:
            assert "agent_a" in update
            assert "agent_b" in update
            assert "debate_increment" in update
            assert "agreement_increment" in update

    def test_update_relationships_without_elo(self):
        """Test relationship update handles missing ELO gracefully."""
        hooks = DebateHooks()

        # Should not raise
        hooks._update_relationships(
            debate_id="test-123",
            participants=["claude", "grok"],
            winner="claude",
            votes=[],
        )


# ============================================================================
# Memory Tracking Tests
# ============================================================================


class TestMemoryTracking:
    """Tests for memory tracking."""

    def test_track_retrieved_memories(self):
        """Test tracking of retrieved memory IDs."""
        hooks = DebateHooks()

        hooks.track_retrieved_memories(
            retrieved_ids=["mem1", "mem2", "mem3"],
            retrieved_tiers={"mem1": "fast", "mem2": "slow"},
        )

        assert hooks._continuum_retrieved_ids == ["mem1", "mem2", "mem3"]
        assert hooks._continuum_retrieved_tiers == {"mem1": "fast", "mem2": "slow"}

    @pytest.mark.asyncio
    async def test_memory_outcomes_updated_on_complete(
        self, mock_context, mock_result, mock_agents, mock_memory_manager
    ):
        """Test retrieved memories are updated on debate completion."""
        hooks = DebateHooks(memory_manager=mock_memory_manager)

        # Track some memories
        hooks.track_retrieved_memories(
            retrieved_ids=["mem1", "mem2"],
            retrieved_tiers={"mem1": "fast"},
        )

        await hooks.on_debate_complete(
            ctx=mock_context,
            result=mock_result,
            agents=mock_agents,
            task="Test task",
        )

        mock_memory_manager.track_retrieved_ids.assert_called_once()
        mock_memory_manager.update_memory_outcomes.assert_called_once()

        # Tracking should be cleared
        assert hooks._continuum_retrieved_ids == []


# ============================================================================
# Evidence Storage Tests
# ============================================================================


class TestEvidenceStorage:
    """Tests for evidence storage."""

    def test_store_evidence(self, mock_memory_manager):
        """Test evidence storage calls memory manager."""
        hooks = DebateHooks(memory_manager=mock_memory_manager)

        evidence = [
            {"source": "arxiv", "snippet": "Research finding"},
            {"source": "wikipedia", "snippet": "Background info"},
        ]

        hooks.store_evidence(evidence, task="Test task")

        mock_memory_manager.store_evidence.assert_called_once_with(
            evidence, "Test task"
        )

    def test_store_evidence_without_manager(self):
        """Test evidence storage handles missing manager gracefully."""
        hooks = DebateHooks()

        # Should not raise
        hooks.store_evidence([{"snippet": "test"}], task="Test task")


# ============================================================================
# Grounded Verdict Tests
# ============================================================================


class TestGroundedVerdict:
    """Tests for grounded verdict creation."""

    def test_create_grounded_verdict(self, mock_result, mock_evidence_grounder):
        """Test grounded verdict is created via evidence grounder."""
        hooks = DebateHooks(evidence_grounder=mock_evidence_grounder)

        verdict = hooks.create_grounded_verdict(mock_result)

        assert verdict == {"verdict": "grounded"}
        mock_evidence_grounder.create_grounded_verdict.assert_called_once()

    def test_create_grounded_verdict_without_answer(self, mock_evidence_grounder):
        """Test verdict creation skipped when no final answer."""
        hooks = DebateHooks(evidence_grounder=mock_evidence_grounder)

        result = Mock()
        result.final_answer = None

        verdict = hooks.create_grounded_verdict(result)

        assert verdict is None
        mock_evidence_grounder.create_grounded_verdict.assert_not_called()

    def test_create_grounded_verdict_without_grounder(self, mock_result):
        """Test verdict creation returns None without grounder."""
        hooks = DebateHooks()

        verdict = hooks.create_grounded_verdict(mock_result)

        assert verdict is None


# ============================================================================
# Diagnostics Tests
# ============================================================================


class TestDiagnostics:
    """Tests for diagnostic methods."""

    def test_get_status_all_subsystems(
        self,
        mock_position_ledger,
        mock_elo_system,
        mock_memory_manager,
        mock_evidence_grounder,
    ):
        """Test get_status returns comprehensive status."""
        hooks = DebateHooks(
            position_ledger=mock_position_ledger,
            elo_system=mock_elo_system,
            memory_manager=mock_memory_manager,
            evidence_grounder=mock_evidence_grounder,
        )

        status = hooks.get_status()

        assert status["subsystems"]["position_ledger"] is True
        assert status["subsystems"]["elo_system"] is True
        assert status["subsystems"]["memory_manager"] is True
        assert status["subsystems"]["evidence_grounder"] is True

    def test_get_status_empty(self):
        """Test get_status with no subsystems."""
        hooks = DebateHooks()

        status = hooks.get_status()

        assert all(v is False for v in status["subsystems"].values())
        assert status["tracking"]["retrieved_memory_count"] == 0

    def test_get_status_with_tracked_memories(self):
        """Test get_status shows tracked memory count."""
        hooks = DebateHooks()
        hooks.track_retrieved_memories(["mem1", "mem2", "mem3"], {})

        status = hooks.get_status()

        assert status["tracking"]["retrieved_memory_count"] == 3


# ============================================================================
# HooksConfig Tests
# ============================================================================


class TestHooksConfig:
    """Tests for HooksConfig helper class."""

    def test_create_hooks_basic(self):
        """Test create_hooks creates hooks instance."""
        config = HooksConfig()
        hooks = config.create_hooks()

        assert isinstance(hooks, DebateHooks)

    def test_create_hooks_with_subsystems(self, mock_elo_system, mock_memory_manager):
        """Test create_hooks passes through subsystems."""
        config = HooksConfig(
            elo_system=mock_elo_system,
            memory_manager=mock_memory_manager,
        )
        hooks = config.create_hooks()

        assert hooks.elo_system is mock_elo_system
        assert hooks.memory_manager is mock_memory_manager
