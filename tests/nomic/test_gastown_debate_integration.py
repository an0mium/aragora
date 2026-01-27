"""Integration tests for Gastown patterns with debate workflows.

Tests the interaction between Gastown work coordination patterns and
debate workflows, focusing on bead/convoy structures for debate tracking.
"""

from datetime import datetime, timezone

import pytest

# Gastown patterns - core
from aragora.nomic.beads import (
    Bead,
    BeadPriority,
    BeadStatus,
    BeadType,
)
from aragora.nomic.convoys import (
    Convoy,
    ConvoyPriority,
    ConvoyStatus,
)


# Note: Bead uses:
#  - metadata instead of data
#  - BeadStatus.RUNNING instead of ACTIVE
#  - BeadType.DEBATE_DECISION instead of DECISION
#  - error_message instead of error
#  - claimed_at instead of started_at (for when work begins)
#  - No result attribute (use metadata to store results)


# =============================================================================
# Debate-Bead Integration Tests
# =============================================================================


class TestDebateBeadStructures:
    """Tests for creating debate structures using beads."""

    def test_create_proposal_bead(self):
        """Test creating a bead for a debate proposal."""
        bead = Bead.create(
            bead_type=BeadType.TASK,
            title="Proposal: Implement rate limiter design",
            description="Agent must propose a rate limiter design",
            metadata={
                "debate_id": "debate_123",
                "round": 1,
                "agent": "claude",
                "operation": "proposal",
            },
            priority=BeadPriority.HIGH,
        )

        assert bead.id is not None
        assert bead.status == BeadStatus.PENDING
        assert bead.metadata["operation"] == "proposal"
        assert bead.metadata["agent"] == "claude"
        assert bead.priority == BeadPriority.HIGH

    def test_create_critique_bead_with_dependency(self):
        """Test creating a critique bead with dependency on proposal."""
        # Create proposal first
        proposal_bead = Bead.create(
            bead_type=BeadType.TASK,
            title="Proposal: Rate limiter",
            description="Original proposal",
            metadata={"debate_id": "debate_123", "operation": "proposal"},
        )

        # Create critique that depends on proposal
        critique_bead = Bead.create(
            bead_type=BeadType.TASK,
            title="Critique: Rate limiter proposal",
            description="Critique of the proposal",
            metadata={
                "debate_id": "debate_123",
                "round": 1,
                "agent": "gpt4",
                "operation": "critique",
                "target_proposal_id": proposal_bead.id,
            },
            dependencies=[proposal_bead.id],
        )

        assert critique_bead.dependencies == [proposal_bead.id]
        assert not critique_bead.can_start(set())  # Can't start without deps
        assert critique_bead.can_start({proposal_bead.id})  # Can start with dep

    def test_create_vote_bead(self):
        """Test creating a voting bead for consensus."""
        vote_bead = Bead.create(
            bead_type=BeadType.DEBATE_DECISION,
            title="Vote: Rate limiter design",
            description="Agent votes on the final design",
            metadata={
                "debate_id": "debate_123",
                "round": 3,
                "agent": "claude",
                "operation": "vote",
                "choice": "proposal_A",
                "confidence": 0.85,
            },
        )

        assert vote_bead.bead_type == BeadType.DEBATE_DECISION
        assert vote_bead.metadata["operation"] == "vote"
        assert vote_bead.metadata["confidence"] == 0.85

    def test_debate_round_as_convoy(self):
        """Test grouping a debate round's work as a convoy."""
        debate_id = "debate_456"
        round_num = 2

        # Create proposal beads for multiple agents
        proposal_beads = []
        for agent in ["claude", "gpt4", "gemini"]:
            bead = Bead.create(
                bead_type=BeadType.TASK,
                title=f"Proposal from {agent}",
                description=f"Round {round_num} proposal",
                metadata={
                    "debate_id": debate_id,
                    "round": round_num,
                    "agent": agent,
                    "operation": "proposal",
                },
            )
            proposal_beads.append(bead)

        # Create convoy for the round
        convoy = Convoy.create(
            title=f"Debate {debate_id} Round {round_num}",
            description="All proposals for this round",
            bead_ids=[b.id for b in proposal_beads],
            priority=ConvoyPriority.HIGH,
            metadata={
                "debate_id": debate_id,
                "round": round_num,
                "phase": "proposal",
            },
        )

        assert len(convoy.bead_ids) == 3
        assert convoy.metadata["debate_id"] == debate_id
        assert convoy.metadata["round"] == round_num
        assert convoy.priority == ConvoyPriority.HIGH


# =============================================================================
# Bead Status Lifecycle Tests
# =============================================================================


class TestBeadDebateLifecycle:
    """Tests for bead status transitions in debate context."""

    def test_proposal_lifecycle(self):
        """Test proposal bead lifecycle from pending to completed."""
        bead = Bead.create(
            bead_type=BeadType.TASK,
            title="Debate proposal",
            description="A proposal for the debate",
            metadata={"debate_id": "lifecycle_test"},
        )

        # Initial status
        assert bead.status == BeadStatus.PENDING
        assert not bead.is_terminal()

        # Transition to running (agent starts working)
        bead.status = BeadStatus.RUNNING
        bead.claimed_at = datetime.now(timezone.utc)
        assert bead.status == BeadStatus.RUNNING
        assert not bead.is_terminal()

        # Transition to completed (agent submitted)
        bead.status = BeadStatus.COMPLETED
        bead.completed_at = datetime.now(timezone.utc)
        assert bead.status == BeadStatus.COMPLETED
        assert bead.is_terminal()

    def test_proposal_timeout_failure(self):
        """Test proposal bead failure due to timeout."""
        bead = Bead.create(
            bead_type=BeadType.TASK,
            title="Timed out proposal",
            description="This agent was too slow",
            metadata={"debate_id": "timeout_test", "agent": "slow_agent"},
        )

        # Start processing
        bead.status = BeadStatus.RUNNING
        bead.claimed_at = datetime.now(timezone.utc)

        # Simulate timeout
        bead.status = BeadStatus.FAILED
        bead.error_message = "Agent timeout: no response within 60 seconds"

        assert bead.status == BeadStatus.FAILED
        assert bead.error_message == "Agent timeout: no response within 60 seconds"
        assert bead.is_terminal()

    def test_critique_blocked_by_dependency(self):
        """Test critique bead blocked until proposal completes."""
        # Create proposal
        proposal = Bead.create(
            bead_type=BeadType.TASK,
            title="Proposal",
            metadata={"operation": "proposal"},
        )

        # Create dependent critique
        critique = Bead.create(
            bead_type=BeadType.TASK,
            title="Critique",
            metadata={"operation": "critique"},
            dependencies=[proposal.id],
        )

        # Critique can't start without completed proposal
        assert not critique.can_start(set())

        # After proposal completes
        proposal.status = BeadStatus.COMPLETED
        assert critique.can_start({proposal.id})


# =============================================================================
# Convoy Lifecycle Tests
# =============================================================================


class TestConvoyDebateLifecycle:
    """Tests for convoy management in debate context."""

    def test_round_convoy_creation(self):
        """Test creating a convoy for a debate round."""
        debate_id = "convoy_test"

        # Create beads for round 1
        round1_beads = []
        for i in range(3):
            bead = Bead.create(
                bead_type=BeadType.TASK,
                title=f"Round 1 Task {i}",
                metadata={"debate_id": debate_id, "round": 1, "index": i},
            )
            round1_beads.append(bead)

        # Create convoy
        convoy = Convoy.create(
            title="Debate Round 1",
            description="Round 1 tasks",
            bead_ids=[b.id for b in round1_beads],
            priority=ConvoyPriority.HIGH,
            metadata={"debate_id": debate_id, "round": 1},
        )

        assert convoy.id is not None
        assert len(convoy.bead_ids) == 3
        assert convoy.status == ConvoyStatus.PENDING

    def test_convoy_status_transitions(self):
        """Test convoy status transitions during debate."""
        beads = [Bead.create(bead_type=BeadType.TASK, title=f"Task {i}") for i in range(2)]

        convoy = Convoy.create(
            title="Test Convoy",
            description="Test",
            bead_ids=[b.id for b in beads],
        )

        # Initial status
        assert convoy.status == ConvoyStatus.PENDING

        # Start convoy
        convoy.status = ConvoyStatus.ACTIVE
        assert convoy.status == ConvoyStatus.ACTIVE

        # Complete convoy
        convoy.status = ConvoyStatus.COMPLETED
        assert convoy.status == ConvoyStatus.COMPLETED

    def test_multiple_round_convoys(self):
        """Test creating convoys for multiple debate rounds."""
        debate_id = "multi_round"
        convoys = []

        for round_num in range(1, 4):
            # Create beads for this round
            beads = [
                Bead.create(
                    bead_type=BeadType.TASK,
                    title=f"Round {round_num} Bead",
                    metadata={"round": round_num},
                )
                for _ in range(3)
            ]

            # Create convoy
            convoy = Convoy.create(
                title=f"Round {round_num}",
                description=f"Debate round {round_num}",
                bead_ids=[b.id for b in beads],
                metadata={"debate_id": debate_id, "round": round_num},
            )
            convoys.append(convoy)

        # Verify 3 convoys created
        assert len(convoys) == 3

        # Each convoy has distinct metadata
        for i, convoy in enumerate(convoys):
            assert convoy.metadata["round"] == i + 1


# =============================================================================
# End-to-End Debate Round Tests
# =============================================================================


class TestEndToEndDebateRound:
    """End-to-end tests for debate rounds using Gastown patterns."""

    def test_complete_proposal_phase(self):
        """Test complete proposal phase with multiple agents."""
        debate_id = "e2e_debate"
        round_num = 1
        agents = ["claude", "gpt4", "gemini"]

        # Create proposal beads for each agent
        proposals = []
        for agent in agents:
            bead = Bead.create(
                bead_type=BeadType.TASK,
                title=f"Proposal from {agent}",
                description=f"Round {round_num} proposal",
                metadata={
                    "debate_id": debate_id,
                    "round": round_num,
                    "agent": agent,
                    "operation": "proposal",
                },
                priority=BeadPriority.HIGH,
            )
            proposals.append(bead)

        # Group as convoy
        proposal_convoy = Convoy.create(
            title=f"Round {round_num} Proposals",
            bead_ids=[b.id for b in proposals],
            priority=ConvoyPriority.HIGH,
            metadata={"debate_id": debate_id, "round": round_num, "phase": "proposal"},
        )

        # Start processing
        proposal_convoy.status = ConvoyStatus.ACTIVE
        for bead in proposals:
            bead.status = BeadStatus.RUNNING
            bead.claimed_at = datetime.now(timezone.utc)

        # Complete all proposals
        for bead in proposals:
            bead.status = BeadStatus.COMPLETED
            bead.completed_at = datetime.now(timezone.utc)
            # Store result in metadata
            bead.metadata["result"] = f"Proposal content from {bead.metadata['agent']}"

        proposal_convoy.status = ConvoyStatus.COMPLETED

        # Verify all completed
        assert proposal_convoy.status == ConvoyStatus.COMPLETED
        assert all(b.status == BeadStatus.COMPLETED for b in proposals)

    def test_complete_critique_phase(self):
        """Test critique phase with dependencies on proposals."""
        debate_id = "critique_test"

        # Create completed proposals
        proposals = []
        for agent in ["claude", "gpt4"]:
            bead = Bead.create(
                bead_type=BeadType.TASK,
                title=f"Proposal from {agent}",
                metadata={"debate_id": debate_id, "agent": agent, "phase": "proposal"},
            )
            bead.status = BeadStatus.COMPLETED
            proposals.append(bead)

        # Create critiques with dependencies
        critiques = []
        for i, proposal in enumerate(proposals):
            critic_agent = "gpt4" if i == 0 else "claude"
            critique = Bead.create(
                bead_type=BeadType.TASK,
                title=f"Critique from {critic_agent}",
                metadata={
                    "debate_id": debate_id,
                    "agent": critic_agent,
                    "phase": "critique",
                    "target_id": proposal.id,
                },
                dependencies=[proposal.id],
            )
            critiques.append(critique)

        # Create critique convoy
        critique_convoy = Convoy.create(
            title="Critique Phase",
            bead_ids=[b.id for b in critiques],
            metadata={"debate_id": debate_id, "phase": "critique"},
        )

        # Verify all critiques can start (deps are complete)
        completed_ids = {p.id for p in proposals}
        for critique in critiques:
            assert critique.can_start(completed_ids)

        # Complete critiques
        for critique in critiques:
            critique.status = BeadStatus.COMPLETED
        critique_convoy.status = ConvoyStatus.COMPLETED

        assert critique_convoy.status == ConvoyStatus.COMPLETED

    def test_voting_phase_with_decision_beads(self):
        """Test voting phase using decision beads."""
        debate_id = "vote_test"
        agents = ["claude", "gpt4", "gemini"]

        # Create vote beads
        votes = []
        for agent in agents:
            vote = Bead.create(
                bead_type=BeadType.DEBATE_DECISION,
                title=f"Vote from {agent}",
                metadata={
                    "debate_id": debate_id,
                    "agent": agent,
                    "phase": "vote",
                    "choice": "proposal_A" if agent != "gemini" else "proposal_B",
                    "confidence": 0.9 if agent != "gemini" else 0.6,
                },
            )
            votes.append(vote)

        # Create voting convoy
        vote_convoy = Convoy.create(
            title="Voting Phase",
            bead_ids=[b.id for b in votes],
            priority=ConvoyPriority.URGENT,
            metadata={"debate_id": debate_id, "phase": "vote"},
        )

        # Complete all votes
        for vote in votes:
            vote.status = BeadStatus.COMPLETED
        vote_convoy.status = ConvoyStatus.COMPLETED

        # Calculate consensus
        choices = [v.metadata["choice"] for v in votes]
        winner = max(set(choices), key=choices.count)

        assert winner == "proposal_A"
        assert vote_convoy.status == ConvoyStatus.COMPLETED
