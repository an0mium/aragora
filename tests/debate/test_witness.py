"""
Tests for the Debate Witness Pattern.

Tests cover:
- DebateWitness progress tracking
- DeadlockDetector cycle and block detection
- RecoveryCoordinator action decisions
- Integration between components
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.witness import (
    AgentProgress,
    DebateWitness,
    ProgressStatus,
    RoundProgress,
    StallEvent,
    StallReason,
    WitnessConfig,
    get_witness,
    remove_witness,
    reset_witnesses,
)
from aragora.debate.deadlock_detector import (
    ArgumentGraph,
    ArgumentNode,
    Deadlock,
    DeadlockDetector,
    DeadlockType,
    create_argument_node,
)
from aragora.debate.recovery_coordinator import (
    RecoveryAction,
    RecoveryConfig,
    RecoveryCoordinator,
    RecoveryDecision,
    RecoveryEvent,
    create_debate_observer,
)
from aragora.debate.protocol_messages import (
    ProtocolMessage,
    ProtocolMessageType,
)
from aragora.debate.protocol_messages.messages import (
    proposal_message,
    critique_message,
)


# ============================================================================
# Witness Tests
# ============================================================================


class TestAgentProgress:
    """Tests for AgentProgress tracking."""

    def test_initial_state(self):
        """Test initial agent progress state."""
        progress = AgentProgress(
            agent_id="claude-opus",
            debate_id="debate-123",
        )

        assert progress.proposals_submitted == 0
        assert progress.critiques_submitted == 0
        assert progress.status == ProgressStatus.HEALTHY
        assert progress.last_activity is None

    def test_record_activity(self):
        """Test recording different activity types."""
        progress = AgentProgress(
            agent_id="claude-opus",
            debate_id="debate-123",
        )

        progress.record_activity("proposal")
        assert progress.proposals_submitted == 1
        assert progress.last_activity is not None

        progress.record_activity("critique")
        assert progress.critiques_submitted == 1

        progress.record_activity("revision")
        assert progress.revisions_submitted == 1

        progress.record_activity("vote")
        assert progress.votes_cast == 1

    def test_time_since_activity(self):
        """Test time calculation since last activity."""
        progress = AgentProgress(
            agent_id="claude-opus",
            debate_id="debate-123",
        )

        assert progress.time_since_activity() is None

        progress.last_activity = datetime.now(timezone.utc) - timedelta(seconds=30)
        elapsed = progress.time_since_activity()
        assert elapsed is not None
        assert 29 <= elapsed <= 31


class TestRoundProgress:
    """Tests for RoundProgress tracking."""

    def test_initial_state(self):
        """Test initial round progress state."""
        progress = RoundProgress(
            round_number=1,
            debate_id="debate-123",
            proposals_expected=3,
            critiques_expected=6,
        )

        assert not progress.is_complete
        assert progress.duration_seconds is None
        assert progress.phase == "proposal"

    def test_completion(self):
        """Test round completion."""
        progress = RoundProgress(
            round_number=1,
            debate_id="debate-123",
        )

        progress.completed_at = datetime.now(timezone.utc)
        assert progress.is_complete
        assert progress.duration_seconds is not None


class TestDebateWitness:
    """Tests for DebateWitness."""

    def test_register_agent(self):
        """Test agent registration."""
        witness = DebateWitness(debate_id="debate-123")

        progress = witness.register_agent("claude-opus")
        assert progress.agent_id == "claude-opus"
        assert progress.debate_id == "debate-123"

        # Re-registering returns same progress
        progress2 = witness.register_agent("claude-opus")
        assert progress2 is progress

    def test_start_round(self):
        """Test starting a new round."""
        witness = DebateWitness(debate_id="debate-123")
        witness.register_agent("agent-1")

        round_progress = witness.start_round(
            round_number=1,
            proposals_expected=2,
            critiques_expected=4,
        )

        assert round_progress.round_number == 1
        assert round_progress.proposals_expected == 2
        assert witness._current_round == 1

    def test_observe_proposal_message(self):
        """Test observing a proposal message."""
        witness = DebateWitness(debate_id="debate-123")
        witness.register_agent("claude-opus")
        witness.start_round(1, proposals_expected=1)

        message = proposal_message(
            debate_id="debate-123",
            agent_id="claude-opus",
            proposal_id="prop-1",
            content="My proposal",
            model="claude-3-opus",
            round_number=1,
        )

        witness.observe(message)

        agent = witness.get_agent_progress("claude-opus")
        assert agent.proposals_submitted == 1
        assert agent.status == ProgressStatus.HEALTHY

        round_prog = witness.get_round_progress(1)
        assert round_prog.proposals_received == 1

    def test_observe_critique_message(self):
        """Test observing a critique message."""
        witness = DebateWitness(debate_id="debate-123")
        witness.register_agent("gpt-4")
        witness.start_round(1)

        message = critique_message(
            debate_id="debate-123",
            agent_id="gpt-4",
            critique_id="crit-1",
            proposal_id="prop-1",
            content="My critique",
            model="gpt-4",
            round_number=1,
        )

        witness.observe(message)

        agent = witness.get_agent_progress("gpt-4")
        assert agent.critiques_submitted == 1

    def test_detect_stalls_timeout(self):
        """Test stall detection based on timeout."""
        config = WitnessConfig(stall_threshold_seconds=1.0)
        witness = DebateWitness(debate_id="debate-123", config=config)
        witness.register_agent("claude-opus")
        witness.start_round(1)

        # Set last activity to past
        agent = witness.get_agent_progress("claude-opus")
        agent.last_activity = datetime.now(timezone.utc) - timedelta(seconds=5)

        stalls = witness.detect_stalls()

        assert len(stalls) == 1
        assert stalls[0].agent_id == "claude-opus"
        assert stalls[0].reason == StallReason.TIMEOUT

    def test_detect_slow_progress(self):
        """Test slow progress detection."""
        config = WitnessConfig(
            slow_threshold_seconds=1.0,
            stall_threshold_seconds=10.0,
        )
        witness = DebateWitness(debate_id="debate-123", config=config)
        witness.register_agent("claude-opus")
        witness.start_round(1)

        agent = witness.get_agent_progress("claude-opus")
        agent.last_activity = datetime.now(timezone.utc) - timedelta(seconds=2)

        witness.detect_stalls()

        assert agent.status == ProgressStatus.SLOW

    def test_content_repetition_detection(self):
        """Test detection of repeated content."""
        witness = DebateWitness(debate_id="debate-123")
        witness.register_agent("claude-opus")

        # First submission
        is_repeated, similarity = witness.check_content_repetition(
            "claude-opus", "This is my proposal"
        )
        assert not is_repeated

        # Record the content hash manually (simulating observe)
        agent = witness.get_agent_progress("claude-opus")
        agent.content_hashes.append(witness._hash_content("This is my proposal"))

        # Same content again
        is_repeated, similarity = witness.check_content_repetition(
            "claude-opus", "This is my proposal"
        )
        assert is_repeated
        assert similarity == 1.0

    def test_get_progress_summary(self):
        """Test getting progress summary."""
        witness = DebateWitness(debate_id="debate-123")
        witness.register_agent("agent-1")
        witness.register_agent("agent-2")
        witness.start_round(1, proposals_expected=2)

        summary = witness.get_progress_summary()

        assert summary["debate_id"] == "debate-123"
        assert summary["current_round"] == 1
        assert "agent-1" in summary["agents"]
        assert "1" in summary["rounds"]

    def test_stall_callback(self):
        """Test stall callback invocation."""
        callback = MagicMock()
        config = WitnessConfig(stall_threshold_seconds=0.1)
        witness = DebateWitness(
            debate_id="debate-123",
            config=config,
            on_stall=callback,
        )
        witness.register_agent("claude-opus")

        agent = witness.get_agent_progress("claude-opus")
        agent.last_activity = datetime.now(timezone.utc) - timedelta(seconds=1)

        witness.detect_stalls()

        callback.assert_called_once()
        stall_event = callback.call_args[0][0]
        assert isinstance(stall_event, StallEvent)

    def test_resolve_stall(self):
        """Test resolving a stall."""
        config = WitnessConfig(stall_threshold_seconds=0.1)
        witness = DebateWitness(debate_id="debate-123", config=config)
        witness.register_agent("claude-opus")

        agent = witness.get_agent_progress("claude-opus")
        agent.last_activity = datetime.now(timezone.utc) - timedelta(seconds=1)
        witness.detect_stalls()

        assert agent.status == ProgressStatus.STALLED

        resolved = witness.resolve_stall("claude-opus", "Agent recovered")
        assert resolved == 1
        assert agent.status == ProgressStatus.HEALTHY
        assert agent.recovery_count == 1


class TestWitnessGlobalRegistry:
    """Tests for global witness registry."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset witness registry before each test."""
        reset_witnesses()
        yield
        reset_witnesses()

    @pytest.mark.asyncio
    async def test_get_witness(self):
        """Test getting a witness from registry."""
        witness = await get_witness("debate-123")
        assert witness.debate_id == "debate-123"

        # Same debate returns same witness
        witness2 = await get_witness("debate-123")
        assert witness2 is witness

    @pytest.mark.asyncio
    async def test_remove_witness(self):
        """Test removing a witness."""
        witness = await get_witness("debate-123")
        await witness.start_monitoring()

        removed = await remove_witness("debate-123")
        assert removed

        # Should create new witness now
        witness2 = await get_witness("debate-123")
        assert witness2 is not witness


# ============================================================================
# Deadlock Detector Tests
# ============================================================================


class TestArgumentGraph:
    """Tests for ArgumentGraph."""

    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = ArgumentGraph(debate_id="debate-123")

        node = ArgumentNode(
            id="arg-1",
            agent_id="claude-opus",
            content_hash="abc123",
            round_number=1,
            argument_type="proposal",
        )

        graph.add_node(node)

        assert graph.get_node("arg-1") is not None
        assert "arg-1" in graph.get_agent_nodes("claude-opus")

    def test_add_edges(self):
        """Test edge creation."""
        graph = ArgumentGraph(debate_id="debate-123")

        # Add proposal
        proposal = ArgumentNode(
            id="prop-1",
            agent_id="claude-opus",
            content_hash="abc",
            round_number=1,
            argument_type="proposal",
        )
        graph.add_node(proposal)

        # Add critique targeting proposal
        critique = ArgumentNode(
            id="crit-1",
            agent_id="gpt-4",
            content_hash="def",
            round_number=1,
            argument_type="critique",
            targets=["prop-1"],
        )
        graph.add_node(critique)

        assert "prop-1" in graph.get_outgoing("crit-1")
        assert "crit-1" in graph.get_incoming("prop-1")

    def test_find_cycles_no_cycle(self):
        """Test cycle detection with no cycles."""
        graph = ArgumentGraph(debate_id="debate-123")

        # Linear chain: prop -> crit -> rebuttal
        graph.add_node(
            ArgumentNode(
                id="prop", agent_id="a", content_hash="1", round_number=1, argument_type="proposal"
            )
        )
        graph.add_node(
            ArgumentNode(
                id="crit",
                agent_id="b",
                content_hash="2",
                round_number=1,
                argument_type="critique",
                targets=["prop"],
            )
        )
        graph.add_node(
            ArgumentNode(
                id="rebuttal",
                agent_id="a",
                content_hash="3",
                round_number=1,
                argument_type="rebuttal",
                targets=["crit"],
            )
        )

        cycles = graph.find_cycles()
        assert len(cycles) == 0

    def test_find_cycles_with_cycle(self):
        """Test cycle detection with a cycle."""
        graph = ArgumentGraph(debate_id="debate-123")

        # Create cycle: A -> B -> C -> A
        graph.add_node(
            ArgumentNode(
                id="A", agent_id="a", content_hash="1", round_number=1, argument_type="proposal"
            )
        )
        graph.add_node(
            ArgumentNode(
                id="B",
                agent_id="b",
                content_hash="2",
                round_number=1,
                argument_type="critique",
                targets=["A"],
            )
        )
        graph.add_node(
            ArgumentNode(
                id="C",
                agent_id="c",
                content_hash="3",
                round_number=1,
                argument_type="critique",
                targets=["B"],
            )
        )
        # Add edge back to A to create cycle
        graph._add_edge("A", "C")

        cycles = graph.find_cycles()
        assert len(cycles) >= 1

    def test_find_mutual_blocks(self):
        """Test mutual block detection."""
        graph = ArgumentGraph(debate_id="debate-123")

        # For mutual blocking, we need both agents' LATEST arguments to target each other
        # Agent A's latest (crit-a-2) must target agent B's node
        # Agent B's latest (crit-b-2) must target agent A's node

        # Round 1: Initial proposals
        graph.add_node(
            ArgumentNode(
                id="prop-a",
                agent_id="agent-a",
                content_hash="1",
                round_number=1,
                argument_type="proposal",
            )
        )
        graph.add_node(
            ArgumentNode(
                id="prop-b",
                agent_id="agent-b",
                content_hash="2",
                round_number=1,
                argument_type="proposal",
            )
        )

        # Round 2: Both agents critique each other's proposals (this creates mutual blocking)
        # Agent A's latest targets Agent B's node
        graph.add_node(
            ArgumentNode(
                id="crit-a-2",
                agent_id="agent-a",
                content_hash="3",
                round_number=2,
                argument_type="critique",
                targets=["prop-b"],
            )
        )
        # Agent B's latest targets Agent A's node
        graph.add_node(
            ArgumentNode(
                id="crit-b-2",
                agent_id="agent-b",
                content_hash="4",
                round_number=2,
                argument_type="critique",
                targets=["prop-a"],
            )
        )

        blocks = graph.find_mutual_blocks()
        assert len(blocks) >= 1
        assert ("agent-a", "agent-b") in blocks or ("agent-b", "agent-a") in blocks


class TestDeadlockDetector:
    """Tests for DeadlockDetector."""

    def test_register_argument(self):
        """Test argument registration."""
        detector = DeadlockDetector(debate_id="debate-123")

        node = create_argument_node(
            node_id="arg-1",
            agent_id="claude-opus",
            content="My proposal",
            round_number=1,
            argument_type="proposal",
        )

        deadlocks = detector.register_argument(node)
        assert detector._graph.get_node("arg-1") is not None

    def test_detect_semantic_loop(self):
        """Test semantic loop detection."""
        detector = DeadlockDetector(debate_id="debate-123")

        # Same content in different rounds
        node1 = create_argument_node(
            node_id="arg-1",
            agent_id="claude-opus",
            content="This is my argument about X",
            round_number=1,
            argument_type="proposal",
        )
        node2 = create_argument_node(
            node_id="arg-2",
            agent_id="claude-opus",
            content="This is my argument about X",  # Same content
            round_number=2,
            argument_type="proposal",
        )

        detector.register_argument(node1)
        deadlocks = detector.register_argument(node2)

        semantic_loops = [d for d in deadlocks if d.deadlock_type == DeadlockType.SEMANTIC_LOOP]
        assert len(semantic_loops) >= 1

    def test_detect_convergence_failure(self):
        """Test convergence failure detection."""
        detector = DeadlockDetector(debate_id="debate-123")

        # Strictly increasing argument counts across 4 rounds: 1, 2, 3, 4
        # This indicates debate is escalating, not converging
        rounds_args = [(1, 1), (2, 2), (3, 3), (4, 4)]
        for round_num, arg_count in rounds_args:
            for i in range(arg_count):
                node = create_argument_node(
                    node_id=f"arg-{round_num}-{i}",
                    agent_id=f"agent-{i % 2}",
                    content=f"Unique argument {round_num} {i}",  # Unique to avoid semantic loop
                    round_number=round_num,
                    argument_type="proposal",
                )
                detector.register_argument(node)

        # Use get_deadlocks() since convergence failure was detected during
        # register_argument calls (detect_deadlocks only returns NEW deadlocks)
        all_deadlocks = detector.get_deadlocks()
        convergence_failures = [
            d for d in all_deadlocks if d.deadlock_type == DeadlockType.CONVERGENCE_FAILURE
        ]
        # Convergence failure is detected when counts don't decrease
        # across 3+ rounds (e.g., [1,2,2] or [2,3,4])
        assert len(convergence_failures) >= 1

    def test_resolve_deadlock(self):
        """Test deadlock resolution."""
        detector = DeadlockDetector(debate_id="debate-123")

        # Create a deadlock (semantic loop)
        node1 = create_argument_node("arg-1", "agent-1", "Same content", 1, "proposal")
        node2 = create_argument_node("arg-2", "agent-1", "Same content", 2, "proposal")

        detector.register_argument(node1)
        deadlocks = detector.register_argument(node2)

        if deadlocks:
            resolved = detector.resolve_deadlock(deadlocks[0].id, "Introduced new evidence")
            assert resolved
            assert deadlocks[0].resolved

    def test_get_statistics(self):
        """Test getting detector statistics."""
        detector = DeadlockDetector(debate_id="debate-123")

        node = create_argument_node("arg-1", "agent-1", "Content", 1, "proposal")
        detector.register_argument(node)

        stats = detector.get_statistics()
        assert stats["debate_id"] == "debate-123"
        assert stats["total_arguments"] == 1


# ============================================================================
# Recovery Coordinator Tests
# ============================================================================


class TestRecoveryCoordinator:
    """Tests for RecoveryCoordinator."""

    def test_init(self):
        """Test coordinator initialization."""
        coordinator = RecoveryCoordinator(debate_id="debate-123")
        assert coordinator.debate_id == "debate-123"
        assert len(coordinator._recovery_history) == 0

    @pytest.mark.asyncio
    async def test_handle_stall_timeout_with_nudge(self):
        """Test handling timeout stall with nudge."""
        config = RecoveryConfig(
            nudge_before_replace=True,
            max_agent_stalls=2,
        )
        coordinator = RecoveryCoordinator(debate_id="debate-123", config=config)

        stall = StallEvent(
            debate_id="debate-123",
            agent_id="claude-opus",
            round_number=1,
            reason=StallReason.TIMEOUT,
        )

        event = await coordinator.handle_stall(stall)

        assert event is not None
        assert event.action == RecoveryAction.NUDGE
        assert event.target_agent_id == "claude-opus"
        assert event.result == "success"

    @pytest.mark.asyncio
    async def test_handle_stall_timeout_with_replace(self):
        """Test handling timeout stall with replacement."""
        config = RecoveryConfig(
            nudge_before_replace=True,
            max_agent_stalls=1,
            replacement_pool=["backup-agent"],
        )
        coordinator = RecoveryCoordinator(debate_id="debate-123", config=config)

        stall = StallEvent(
            debate_id="debate-123",
            agent_id="claude-opus",
            round_number=1,
            reason=StallReason.TIMEOUT,
        )

        # First stall triggers nudge
        event1 = await coordinator.handle_stall(stall)
        assert event1.action == RecoveryAction.NUDGE

        # Second stall triggers replace
        event2 = await coordinator.handle_stall(stall)
        assert event2.action == RecoveryAction.REPLACE

    @pytest.mark.asyncio
    async def test_handle_deadlock_cycle(self):
        """Test handling cycle deadlock."""
        config = RecoveryConfig(cycle_resolution_strategy="inject_mediator")
        coordinator = RecoveryCoordinator(debate_id="debate-123", config=config)

        deadlock = Deadlock(
            id="deadlock-1",
            deadlock_type=DeadlockType.CYCLE,
            debate_id="debate-123",
            involved_agents=["agent-1", "agent-2"],
            involved_arguments=["arg-1", "arg-2", "arg-3"],
            cycle_path=["arg-1", "arg-2", "arg-3", "arg-1"],
            severity="medium",
            description="Circular argument chain",
        )

        event = await coordinator.handle_deadlock(deadlock)

        assert event is not None
        assert event.action == RecoveryAction.INJECT_MEDIATOR

    @pytest.mark.asyncio
    async def test_handle_agent_failure(self):
        """Test handling agent failure."""
        config = RecoveryConfig(
            max_agent_failures=2,
            replacement_pool=["backup-1", "backup-2"],
        )
        witness = DebateWitness(debate_id="debate-123")
        witness.register_agent("failing-agent")

        coordinator = RecoveryCoordinator(
            debate_id="debate-123",
            witness=witness,
            config=config,
        )

        # First failure
        event1 = await coordinator.handle_agent_failure("failing-agent", "Connection timeout")
        assert event1.action == RecoveryAction.NUDGE  # Transient error

        # Mark agent as having failures
        agent = witness.get_agent_progress("failing-agent")
        agent.failure_count = 3

        # Now should replace
        event2 = await coordinator.handle_agent_failure("failing-agent", "Permanent error")
        assert event2.action == RecoveryAction.REPLACE

    @pytest.mark.asyncio
    async def test_no_replacement_available(self):
        """Test behavior when no replacement is available."""
        config = RecoveryConfig(replacement_pool=[])
        coordinator = RecoveryCoordinator(debate_id="debate-123", config=config)

        # Force replacement decision
        coordinator._agent_nudge_counts["agent-1"] = 10

        stall = StallEvent(
            debate_id="debate-123",
            agent_id="agent-1",
            round_number=1,
            reason=StallReason.TIMEOUT,
        )

        event = await coordinator.handle_stall(stall)

        assert event.action == RecoveryAction.ESCALATE
        assert event.details.get("requires_approval")

    @pytest.mark.asyncio
    async def test_approval_workflow(self):
        """Test approval workflow for recovery actions."""
        config = RecoveryConfig(
            approval_required_for_replace=True,
            replacement_pool=["backup"],
        )
        coordinator = RecoveryCoordinator(debate_id="debate-123", config=config)
        coordinator._agent_nudge_counts["agent-1"] = 10

        stall = StallEvent(
            debate_id="debate-123",
            agent_id="agent-1",
            round_number=1,
            reason=StallReason.TIMEOUT,
        )

        event = await coordinator.handle_stall(stall)
        assert event.result == "pending_approval"

        # Approve the action
        approved = await coordinator.approve_recovery(event.id, approved=True)
        assert approved
        assert event.result == "success"

    @pytest.mark.asyncio
    async def test_queue_and_process(self):
        """Test queuing and processing issues."""
        coordinator = RecoveryCoordinator(debate_id="debate-123")

        stall = StallEvent(
            debate_id="debate-123",
            agent_id="agent-1",
            round_number=1,
            reason=StallReason.NO_PROGRESS,
        )

        coordinator.queue_stall(stall)
        assert len(coordinator._pending_stalls) == 1

        events = await coordinator.process_pending_issues()
        assert len(events) == 1
        assert len(coordinator._pending_stalls) == 0

    def test_get_statistics(self):
        """Test getting coordinator statistics."""
        coordinator = RecoveryCoordinator(debate_id="debate-123")

        stats = coordinator.get_statistics()
        assert stats["debate_id"] == "debate-123"
        assert stats["total_recoveries"] == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestWitnessIntegration:
    """Integration tests for witness pattern components."""

    @pytest.mark.asyncio
    async def test_create_debate_observer(self):
        """Test creating integrated observer system."""
        witness, detector, coordinator = await create_debate_observer(
            debate_id="debate-123",
            agents=["agent-1", "agent-2"],
            replacement_pool=["backup-1"],
        )

        assert witness.debate_id == "debate-123"
        assert detector.debate_id == "debate-123"
        assert coordinator.debate_id == "debate-123"

        # Verify agents registered
        assert witness.get_agent_progress("agent-1") is not None
        assert witness.get_agent_progress("agent-2") is not None

    @pytest.mark.asyncio
    async def test_end_to_end_stall_handling(self):
        """Test end-to-end stall detection and recovery."""
        config = WitnessConfig(stall_threshold_seconds=0.1)
        recovery_config = RecoveryConfig(replacement_pool=["backup"])

        witness = DebateWitness(debate_id="debate-123", config=config)
        witness.register_agent("slow-agent")

        coordinator = RecoveryCoordinator(
            debate_id="debate-123",
            witness=witness,
            config=recovery_config,
        )

        # Wire callback
        witness.on_stall = coordinator.queue_stall

        # Simulate stall
        agent = witness.get_agent_progress("slow-agent")
        agent.last_activity = datetime.now(timezone.utc) - timedelta(seconds=1)

        # Detect stalls
        stalls = witness.detect_stalls()
        assert len(stalls) == 1

        # Process recovery
        events = await coordinator.process_pending_issues()
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_deadlock_to_recovery_flow(self):
        """Test flow from deadlock detection to recovery."""
        detector = DeadlockDetector(debate_id="debate-123")
        coordinator = RecoveryCoordinator(debate_id="debate-123")

        # Create semantic loop deadlock
        node1 = create_argument_node("arg-1", "agent-1", "Repeated content", 1, "proposal")
        node2 = create_argument_node("arg-2", "agent-1", "Repeated content", 2, "proposal")

        detector.register_argument(node1)
        deadlocks = detector.register_argument(node2)

        # Handle each deadlock
        for deadlock in deadlocks:
            coordinator.queue_deadlock(deadlock)

        events = await coordinator.process_pending_issues()
        # Should have handled the semantic loop
        assert len(events) >= len(
            [d for d in deadlocks if d.deadlock_type == DeadlockType.SEMANTIC_LOOP]
        )


@pytest.mark.asyncio
async def test_monitoring_lifecycle():
    """Test witness monitoring start/stop."""
    witness = DebateWitness(debate_id="debate-123")

    await witness.start_monitoring()
    assert witness._running

    await witness.stop_monitoring()
    assert not witness._running
