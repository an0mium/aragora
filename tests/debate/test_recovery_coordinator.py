"""
Tests for Recovery Coordinator Module.

Tests the recovery coordinator functionality including:
- RecoveryAction enum
- RecoveryDecision dataclass
- RecoveryEvent dataclass
- RecoveryConfig dataclass
- RecoveryCoordinator class
- Stall handling
- Deadlock handling
- Agent failure handling
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.recovery_coordinator import (
    RecoveryAction,
    RecoveryConfig,
    RecoveryCoordinator,
    RecoveryDecision,
    RecoveryEvent,
)


# =============================================================================
# RecoveryAction Enum Tests
# =============================================================================


class TestRecoveryAction:
    """Test RecoveryAction enum."""

    def test_none_action(self):
        """Test none action value."""
        assert RecoveryAction.NONE.value == "none"

    def test_nudge_action(self):
        """Test nudge action value."""
        assert RecoveryAction.NUDGE.value == "nudge"

    def test_replace_action(self):
        """Test replace action value."""
        assert RecoveryAction.REPLACE.value == "replace"

    def test_skip_action(self):
        """Test skip action value."""
        assert RecoveryAction.SKIP.value == "skip"

    def test_reset_round_action(self):
        """Test reset_round action value."""
        assert RecoveryAction.RESET_ROUND.value == "reset_round"

    def test_force_vote_action(self):
        """Test force_vote action value."""
        assert RecoveryAction.FORCE_VOTE.value == "force_vote"

    def test_inject_mediator_action(self):
        """Test inject_mediator action value."""
        assert RecoveryAction.INJECT_MEDIATOR.value == "inject_mediator"

    def test_escalate_action(self):
        """Test escalate action value."""
        assert RecoveryAction.ESCALATE.value == "escalate"

    def test_abort_action(self):
        """Test abort action value."""
        assert RecoveryAction.ABORT.value == "abort"


# =============================================================================
# RecoveryDecision Tests
# =============================================================================


class TestRecoveryDecision:
    """Test RecoveryDecision dataclass."""

    def test_create_simple_decision(self):
        """Test creating a simple decision."""
        decision = RecoveryDecision(action=RecoveryAction.NUDGE)

        assert decision.action == RecoveryAction.NUDGE
        assert decision.target_agent_id is None
        assert decision.confidence == 1.0

    def test_create_replacement_decision(self):
        """Test creating a replacement decision."""
        decision = RecoveryDecision(
            action=RecoveryAction.REPLACE,
            target_agent_id="agent-1",
            replacement_agent_id="backup-agent",
            reason="Agent failed too many times",
        )

        assert decision.action == RecoveryAction.REPLACE
        assert decision.target_agent_id == "agent-1"
        assert decision.replacement_agent_id == "backup-agent"

    def test_decision_with_approval(self):
        """Test decision requiring approval."""
        decision = RecoveryDecision(
            action=RecoveryAction.ABORT,
            reason="Critical failure",
            requires_approval=True,
        )

        assert decision.requires_approval is True

    def test_decision_with_metadata(self):
        """Test decision with metadata."""
        decision = RecoveryDecision(
            action=RecoveryAction.NUDGE,
            metadata={"nudge_type": "request_novel_perspective"},
        )

        assert decision.metadata["nudge_type"] == "request_novel_perspective"


# =============================================================================
# RecoveryEvent Tests
# =============================================================================


class TestRecoveryEvent:
    """Test RecoveryEvent dataclass."""

    def test_create_recovery_event(self):
        """Test creating a recovery event."""
        event = RecoveryEvent(
            id="recovery-001",
            debate_id="debate-123",
            action=RecoveryAction.NUDGE,
            trigger_type="stall",
        )

        assert event.id == "recovery-001"
        assert event.debate_id == "debate-123"
        assert event.action == RecoveryAction.NUDGE
        assert event.result == "pending"

    def test_event_with_target_agent(self):
        """Test event with target agent."""
        event = RecoveryEvent(
            id="recovery-002",
            debate_id="debate-123",
            action=RecoveryAction.REPLACE,
            trigger_type="failure",
            target_agent_id="claude",
        )

        assert event.target_agent_id == "claude"

    def test_event_default_timestamp(self):
        """Test event has default timestamp."""
        event = RecoveryEvent(
            id="recovery-003",
            debate_id="debate-123",
            action=RecoveryAction.SKIP,
            trigger_type="stall",
        )

        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)

    def test_event_with_details(self):
        """Test event with details."""
        event = RecoveryEvent(
            id="recovery-004",
            debate_id="debate-123",
            action=RecoveryAction.ESCALATE,
            trigger_type="deadlock",
            details={"severity": "critical"},
        )

        assert event.details["severity"] == "critical"


# =============================================================================
# RecoveryConfig Tests
# =============================================================================


class TestRecoveryConfig:
    """Test RecoveryConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RecoveryConfig()

        assert config.max_agent_failures == 3
        assert config.max_agent_stalls == 2
        assert config.nudge_before_replace is True
        assert config.approval_required_for_abort is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = RecoveryConfig(
            max_agent_failures=5,
            max_agent_stalls=3,
            replacement_pool=["backup-1", "backup-2"],
        )

        assert config.max_agent_failures == 5
        assert config.replacement_pool == ["backup-1", "backup-2"]

    def test_deadlock_strategies(self):
        """Test deadlock strategy configuration."""
        config = RecoveryConfig(
            cycle_resolution_strategy="force_vote",
            mutual_block_strategy="inject_mediator",
        )

        assert config.cycle_resolution_strategy == "force_vote"
        assert config.mutual_block_strategy == "inject_mediator"


# =============================================================================
# RecoveryCoordinator Initialization Tests
# =============================================================================


class TestRecoveryCoordinatorInit:
    """Test RecoveryCoordinator initialization."""

    def test_init_minimal(self):
        """Test minimal initialization."""
        coordinator = RecoveryCoordinator(debate_id="debate-123")

        assert coordinator.debate_id == "debate-123"
        assert coordinator.witness is None
        assert coordinator.config is not None

    def test_init_with_config(self):
        """Test initialization with config."""
        config = RecoveryConfig(max_agent_failures=5)
        coordinator = RecoveryCoordinator(
            debate_id="debate-123",
            config=config,
        )

        assert coordinator.config.max_agent_failures == 5

    def test_init_with_callbacks(self):
        """Test initialization with callbacks."""
        on_action = MagicMock()
        on_message = MagicMock()

        coordinator = RecoveryCoordinator(
            debate_id="debate-123",
            on_action=on_action,
            on_message=on_message,
        )

        assert coordinator.on_action == on_action
        assert coordinator.on_message == on_message

    def test_init_with_witness(self):
        """Test initialization with witness."""
        mock_witness = MagicMock()

        coordinator = RecoveryCoordinator(
            debate_id="debate-123",
            witness=mock_witness,
        )

        assert coordinator.witness == mock_witness


# =============================================================================
# Stall Recovery Tests
# =============================================================================


class TestStallRecovery:
    """Test stall recovery handling."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator for testing."""
        config = RecoveryConfig(
            max_agent_stalls=2,
            nudge_before_replace=True,
            replacement_pool=["backup-agent"],
        )
        return RecoveryCoordinator(debate_id="debate-test", config=config)

    @pytest.fixture
    def mock_stall_event(self):
        """Create mock stall event."""
        from aragora.debate.witness import StallEvent, StallReason

        return StallEvent(
            debate_id="debate-test",
            agent_id="claude",
            round_number=1,
            reason=StallReason.TIMEOUT,
        )

    def test_decide_stall_timeout_nudge_first(self, coordinator, mock_stall_event):
        """Test timeout stall nudges first."""
        decision = coordinator._decide_stall_recovery(mock_stall_event)

        assert decision.action == RecoveryAction.NUDGE
        assert decision.target_agent_id == "claude"

    def test_decide_stall_timeout_replace_after_max(self, coordinator, mock_stall_event):
        """Test timeout stall replaces after max nudges."""
        # Exhaust nudge attempts
        coordinator._agent_nudge_counts["claude"] = 2

        decision = coordinator._decide_stall_recovery(mock_stall_event)

        assert decision.action == RecoveryAction.REPLACE

    def test_decide_stall_repeated_content(self, coordinator):
        """Test repeated content stall."""
        from aragora.debate.witness import StallEvent, StallReason

        stall = StallEvent(
            debate_id="debate-test",
            agent_id="claude",
            round_number=1,
            reason=StallReason.REPEATED_CONTENT,
        )

        decision = coordinator._decide_stall_recovery(stall)

        assert decision.action == RecoveryAction.NUDGE
        assert decision.metadata.get("nudge_type") == "request_novel_perspective"

    def test_decide_stall_agent_failure(self, coordinator):
        """Test agent failure stall."""
        from aragora.debate.witness import StallEvent, StallReason

        stall = StallEvent(
            debate_id="debate-test",
            agent_id="claude",
            round_number=1,
            reason=StallReason.AGENT_FAILURE,
        )

        decision = coordinator._decide_stall_recovery(stall)

        assert decision.action == RecoveryAction.REPLACE

    def test_decide_stall_no_progress(self, coordinator):
        """Test no progress stall."""
        from aragora.debate.witness import StallEvent, StallReason

        stall = StallEvent(
            debate_id="debate-test",
            agent_id="claude",
            round_number=1,
            reason=StallReason.NO_PROGRESS,
        )

        decision = coordinator._decide_stall_recovery(stall)

        assert decision.action == RecoveryAction.SKIP


# =============================================================================
# Deadlock Recovery Tests
# =============================================================================


class TestDeadlockRecovery:
    """Test deadlock recovery handling."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator for testing."""
        config = RecoveryConfig(
            cycle_resolution_strategy="inject_mediator",
            mutual_block_strategy="nudge",
            semantic_loop_strategy="skip",
            convergence_failure_strategy="force_vote",
        )
        return RecoveryCoordinator(debate_id="debate-test", config=config)

    def test_decide_deadlock_cycle(self, coordinator):
        """Test cycle deadlock recovery."""
        from aragora.debate.deadlock_detector import Deadlock, DeadlockType

        deadlock = Deadlock(
            id="deadlock-1",
            deadlock_type=DeadlockType.CYCLE,
            debate_id="debate-test",
            involved_agents=["agent-1", "agent-2"],
            involved_arguments=["arg-1", "arg-2"],
            description="Circular dependency detected",
            severity="medium",
        )

        decision = coordinator._decide_deadlock_recovery(deadlock)

        assert decision.action == RecoveryAction.INJECT_MEDIATOR

    def test_decide_deadlock_mutual_block(self, coordinator):
        """Test mutual block deadlock recovery."""
        from aragora.debate.deadlock_detector import Deadlock, DeadlockType

        deadlock = Deadlock(
            id="deadlock-2",
            deadlock_type=DeadlockType.MUTUAL_BLOCK,
            debate_id="debate-test",
            involved_agents=["agent-1", "agent-2"],
            involved_arguments=["arg-1", "arg-2"],
            description="Agents blocking each other",
            severity="medium",
        )

        decision = coordinator._decide_deadlock_recovery(deadlock)

        assert decision.action == RecoveryAction.NUDGE
        assert decision.target_agent_id == "agent-1"

    def test_decide_deadlock_semantic_loop(self, coordinator):
        """Test semantic loop deadlock recovery."""
        from aragora.debate.deadlock_detector import Deadlock, DeadlockType

        deadlock = Deadlock(
            id="deadlock-3",
            deadlock_type=DeadlockType.SEMANTIC_LOOP,
            debate_id="debate-test",
            involved_agents=["agent-1"],
            involved_arguments=["arg-1", "arg-2", "arg-3"],
            description="Arguments repeating",
            severity="low",
        )

        decision = coordinator._decide_deadlock_recovery(deadlock)

        assert decision.action == RecoveryAction.SKIP

    def test_decide_deadlock_convergence_failure(self, coordinator):
        """Test convergence failure deadlock recovery."""
        from aragora.debate.deadlock_detector import Deadlock, DeadlockType

        deadlock = Deadlock(
            id="deadlock-4",
            deadlock_type=DeadlockType.CONVERGENCE_FAILURE,
            debate_id="debate-test",
            involved_agents=[],
            involved_arguments=["arg-1", "arg-2"],
            description="No consensus forming",
            severity="high",
        )

        decision = coordinator._decide_deadlock_recovery(deadlock)

        assert decision.action == RecoveryAction.FORCE_VOTE
        assert decision.requires_approval is True  # High severity

    def test_decide_deadlock_critical_requires_approval(self, coordinator):
        """Test critical deadlock requires approval."""
        from aragora.debate.deadlock_detector import Deadlock, DeadlockType

        deadlock = Deadlock(
            id="deadlock-5",
            deadlock_type=DeadlockType.CYCLE,
            debate_id="debate-test",
            involved_agents=[],
            involved_arguments=["arg-1"],
            description="Critical cycle",
            severity="critical",
        )

        decision = coordinator._decide_deadlock_recovery(deadlock)

        assert decision.requires_approval is True


# =============================================================================
# Failure Recovery Tests
# =============================================================================


class TestFailureRecovery:
    """Test agent failure recovery handling."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator with witness."""
        config = RecoveryConfig(
            max_agent_failures=3,
            replacement_pool=["backup-agent"],
        )
        return RecoveryCoordinator(debate_id="debate-test", config=config)

    def test_decide_failure_transient_error(self, coordinator):
        """Test transient error recovery."""
        decision = coordinator._decide_failure_recovery(
            "claude",
            "Request timeout after 30 seconds",
        )

        assert decision.action == RecoveryAction.NUDGE
        assert decision.metadata.get("retry") is True

    def test_decide_failure_rate_limit(self, coordinator):
        """Test rate limit error recovery."""
        decision = coordinator._decide_failure_recovery(
            "claude",
            "Rate limit exceeded: 429",
        )

        assert decision.action == RecoveryAction.NUDGE

    def test_decide_failure_permanent_error(self, coordinator):
        """Test permanent error recovery."""
        decision = coordinator._decide_failure_recovery(
            "claude",
            "Authentication failed: invalid API key",
        )

        assert decision.action == RecoveryAction.REPLACE


# =============================================================================
# Replacement Decision Tests
# =============================================================================


class TestReplacementDecision:
    """Test agent replacement decisions."""

    def test_replacement_with_pool(self):
        """Test replacement when pool available."""
        config = RecoveryConfig(replacement_pool=["backup-1", "backup-2"])
        coordinator = RecoveryCoordinator(debate_id="test", config=config)

        decision = coordinator._decide_replacement("agent-1", "failure")

        assert decision.action == RecoveryAction.REPLACE
        assert decision.replacement_agent_id == "backup-1"

    def test_replacement_no_pool(self):
        """Test replacement when no pool available."""
        config = RecoveryConfig(replacement_pool=[])
        coordinator = RecoveryCoordinator(debate_id="test", config=config)

        decision = coordinator._decide_replacement("agent-1", "failure")

        assert decision.action == RecoveryAction.ESCALATE
        assert decision.requires_approval is True

    def test_replacement_already_replaced(self):
        """Test skip when agent already replaced."""
        config = RecoveryConfig(replacement_pool=["backup-1"])
        coordinator = RecoveryCoordinator(debate_id="test", config=config)
        coordinator._replaced_agents.add("agent-1")

        decision = coordinator._decide_replacement("agent-1", "failure")

        assert decision.action == RecoveryAction.SKIP

    def test_replacement_pool_exhausted(self):
        """Test escalate when pool exhausted."""
        config = RecoveryConfig(replacement_pool=["backup-1"])
        coordinator = RecoveryCoordinator(debate_id="test", config=config)
        coordinator._replaced_agents.add("backup-1")  # Already used

        decision = coordinator._decide_replacement("agent-1", "failure")

        assert decision.action == RecoveryAction.ESCALATE


# =============================================================================
# Handle Methods Tests
# =============================================================================


class TestHandleMethods:
    """Test handle_* methods."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator for testing."""
        config = RecoveryConfig(
            nudge_before_replace=True,
            max_agent_stalls=2,
        )
        return RecoveryCoordinator(debate_id="debate-test", config=config)

    @pytest.mark.asyncio
    async def test_handle_stall_returns_event(self, coordinator):
        """Test handle_stall returns recovery event."""
        from aragora.debate.witness import StallEvent, StallReason

        stall = StallEvent(
            debate_id="debate-test",
            agent_id="claude",
            round_number=1,
            reason=StallReason.TIMEOUT,
        )

        event = await coordinator.handle_stall(stall)

        assert event is not None
        assert event.action == RecoveryAction.NUDGE
        assert event.trigger_type == "stall"

    @pytest.mark.asyncio
    async def test_handle_stall_none_action(self, coordinator):
        """Test handle_stall returns None for NONE action."""
        from aragora.debate.witness import StallEvent, StallReason

        # Mock to return NONE action
        with patch.object(
            coordinator,
            "_decide_stall_recovery",
            return_value=RecoveryDecision(action=RecoveryAction.NONE),
        ):
            stall = StallEvent(
                debate_id="debate-test",
                agent_id="claude",
                round_number=1,
                reason=StallReason.TIMEOUT,
            )

            event = await coordinator.handle_stall(stall)

            assert event is None

    @pytest.mark.asyncio
    async def test_handle_deadlock_returns_event(self, coordinator):
        """Test handle_deadlock returns recovery event."""
        from aragora.debate.deadlock_detector import Deadlock, DeadlockType

        deadlock = Deadlock(
            id="dl-1",
            deadlock_type=DeadlockType.CYCLE,
            debate_id="debate-test",
            involved_agents=["agent-1"],
            involved_arguments=["arg-1", "arg-2"],
            description="Test deadlock",
            severity="medium",
        )

        event = await coordinator.handle_deadlock(deadlock)

        assert event is not None
        assert event.trigger_type == "deadlock"

    @pytest.mark.asyncio
    async def test_handle_agent_failure_returns_event(self, coordinator):
        """Test handle_agent_failure returns recovery event."""
        event = await coordinator.handle_agent_failure(
            "claude",
            "Connection timeout",
        )

        assert event is not None
        assert event.trigger_type == "failure"
