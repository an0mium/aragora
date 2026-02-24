"""
E2E test for the complete debate flow: start debate -> stream events -> get receipt.

Validates the critical user path end-to-end with mocked agents (no real API calls).

Test Coverage:
1. Full debate lifecycle via Arena API with mock agents
2. Round events arrive in correct chronological order
3. Decision receipt is generated after debate completion
4. Receipt contains consensus proof and vote history
5. Spectator event stream captures all expected event types
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import Agent, Critique, Environment, Message, Vote
from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import DebateProtocol
from aragora.gauntlet.receipt import (
    ConsensusProof,
    DecisionReceipt,
    ProvenanceRecord,
)
from aragora.spectate.events import SpectatorEvents
from aragora.spectate.stream import SpectatorStream


# =============================================================================
# Mock Agent for Complete Flow Testing
# =============================================================================


@dataclass
class FlowAgentConfig:
    """Configuration for a mock agent in the complete flow test."""

    name: str
    response: str = "Implement caching with Redis for read-heavy endpoints."
    vote_choice: str | None = None
    vote_confidence: float = 0.85
    critique_severity: float = 0.2


class FlowMockAgent(Agent):
    """Mock agent for complete flow testing without real LLM calls.

    Tracks all method calls to allow assertion on debate progression.
    """

    def __init__(self, config: FlowAgentConfig):
        super().__init__(
            name=config.name,
            model="mock-model",
            role="proposer",
        )
        self.agent_type = "mock"
        self.config = config
        self.call_log: list[str] = []
        # Token tracking attributes required by extensions
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.metrics = None
        self.provider = None

    async def generate(self, prompt: str, context: list | None = None) -> str:
        self.call_log.append("generate")
        return self.config.response

    async def generate_stream(self, prompt: str, context: list | None = None):
        self.call_log.append("generate_stream")
        yield self.config.response

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list | None = None,
        target_agent: str | None = None,
    ) -> Critique:
        self.call_log.append("critique")
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal[:100] if proposal else "",
            issues=[],
            suggestions=[],
            severity=self.config.critique_severity,
            reasoning="Reviewed and agree with the approach.",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        self.call_log.append("vote")
        choice = self.config.vote_choice
        if choice is None:
            choice = list(proposals.keys())[0] if proposals else self.name
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning="This proposal aligns with best practices.",
            confidence=self.config.vote_confidence,
            continue_debate=False,
        )


# =============================================================================
# Event Collector for Spectator Stream Testing
# =============================================================================


class EventCollector:
    """Collects spectator events for assertion during tests.

    Wraps a SpectatorStream to capture all emitted events in order,
    enabling assertions on event sequencing and content.
    """

    def __init__(self):
        self.events: list[dict[str, Any]] = []
        self._event_types: list[str] = []

    def capture(
        self,
        event_type: str,
        agent: str = "",
        details: str = "",
        metric: float | None = None,
        round_number: int | None = None,
    ) -> None:
        """Capture a spectator event."""
        self.events.append(
            {
                "event_type": event_type,
                "agent": agent,
                "details": details,
                "metric": metric,
                "round_number": round_number,
            }
        )
        self._event_types.append(event_type)

    @property
    def event_types(self) -> list[str]:
        """Return ordered list of event types."""
        return list(self._event_types)

    def events_of_type(self, event_type: str) -> list[dict[str, Any]]:
        """Return all events of a specific type."""
        return [e for e in self.events if e["event_type"] == event_type]

    def has_event(self, event_type: str) -> bool:
        """Check if an event of given type was emitted."""
        return event_type in self._event_types


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def flow_agents() -> list[FlowMockAgent]:
    """Create three mock agents that converge on a shared response."""
    shared_response = "Use Redis caching with TTL-based invalidation for read-heavy endpoints."
    return [
        FlowMockAgent(
            FlowAgentConfig(
                name="agent-claude",
                response=shared_response,
                vote_confidence=0.9,
            )
        ),
        FlowMockAgent(
            FlowAgentConfig(
                name="agent-gpt",
                response=shared_response,
                vote_confidence=0.85,
            )
        ),
        FlowMockAgent(
            FlowAgentConfig(
                name="agent-gemini",
                response=shared_response,
                vote_confidence=0.88,
            )
        ),
    ]


@pytest.fixture
def debate_environment() -> Environment:
    """Create a test environment with a clear debate topic."""
    return Environment(
        task="Should we implement caching for our read-heavy API endpoints?"
    )


@pytest.fixture
def minimal_protocol() -> DebateProtocol:
    """Create a minimal protocol that completes quickly with mocked agents."""
    return DebateProtocol(
        rounds=2,
        consensus="majority",
        enable_calibration=False,
        enable_rhetorical_observer=False,
        enable_trickster=False,
    )


@pytest.fixture
def event_collector() -> EventCollector:
    """Create an event collector for capturing spectator events."""
    return EventCollector()


@pytest.fixture
def spectator_with_collector(event_collector: EventCollector) -> SpectatorStream:
    """Create a spectator stream that forwards events to the collector."""
    output = io.StringIO()
    spectator = SpectatorStream(enabled=True, output=output, format="plain")
    # Monkey-patch the emit method to also collect events
    original_emit = spectator.emit

    def capturing_emit(
        event_type: str,
        agent: str = "",
        details: str = "",
        metric: float | None = None,
        round_number: int | None = None,
    ) -> None:
        event_collector.capture(event_type, agent, details, metric, round_number)
        original_emit(event_type, agent=agent, details=details, metric=metric, round_number=round_number)

    spectator.emit = capturing_emit  # type: ignore[assignment]
    return spectator


# =============================================================================
# Test: Complete Debate Flow (Start -> Events -> Receipt)
# =============================================================================


@pytest.mark.e2e
class TestCompleteDebateFlow:
    """E2E tests for the complete debate flow: start -> stream -> receipt."""

    @pytest.mark.asyncio
    async def test_debate_creates_and_completes(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
    ):
        """Test that a debate can be created and runs to completion."""
        arena = Arena(debate_environment, flow_agents, minimal_protocol)
        result = await arena.run()

        assert result is not None
        assert result.rounds_completed > 0
        assert result.final_answer is not None
        assert len(result.final_answer) > 0
        assert result.task == debate_environment.task

    @pytest.mark.asyncio
    async def test_debate_with_spectator_emits_events(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
        spectator_with_collector: SpectatorStream,
        event_collector: EventCollector,
    ):
        """Test that a debate with spectator emits events during execution."""
        arena = Arena(
            debate_environment,
            flow_agents,
            minimal_protocol,
            spectator=spectator_with_collector,
        )
        result = await arena.run()

        assert result is not None
        # Events should have been collected
        assert len(event_collector.events) > 0
        # There should be at least a debate_start event
        assert event_collector.has_event(SpectatorEvents.DEBATE_START)

    @pytest.mark.asyncio
    async def test_debate_agents_are_called(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
    ):
        """Test that all agents participate in the debate by verifying call logs."""
        arena = Arena(debate_environment, flow_agents, minimal_protocol)
        result = await arena.run()

        assert result is not None
        # Each agent should have been called at least once
        for agent in flow_agents:
            assert len(agent.call_log) > 0, (
                f"Agent {agent.name} was never called during the debate"
            )
            # At minimum, generate should be called for proposals
            assert "generate" in agent.call_log, (
                f"Agent {agent.name} never had generate() called"
            )

    @pytest.mark.asyncio
    async def test_full_flow_debate_to_receipt(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
    ):
        """Test the full critical path: start debate -> get result -> generate receipt."""
        # Step 1: Run the debate
        arena = Arena(debate_environment, flow_agents, minimal_protocol)
        result = await arena.run()

        assert result is not None
        assert result.rounds_completed > 0

        # Step 2: Generate receipt from the debate result
        receipt = DecisionReceipt.from_debate_result(result)

        # Step 3: Validate receipt is complete and well-formed
        assert receipt.receipt_id is not None
        assert len(receipt.receipt_id) > 0
        assert receipt.gauntlet_id is not None
        assert receipt.timestamp is not None
        assert receipt.confidence >= 0.0
        assert receipt.verdict in ("PASS", "CONDITIONAL", "FAIL")
        assert receipt.artifact_hash is not None
        assert len(receipt.artifact_hash) == 64  # SHA-256 hex digest length

    @pytest.mark.asyncio
    async def test_full_flow_with_events_and_receipt(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
        spectator_with_collector: SpectatorStream,
        event_collector: EventCollector,
    ):
        """Test the entire critical path with event streaming and receipt generation."""
        # Step 1: Create and run debate with spectator
        arena = Arena(
            debate_environment,
            flow_agents,
            minimal_protocol,
            spectator=spectator_with_collector,
        )
        result = await arena.run()

        assert result is not None

        # Step 2: Verify events were emitted
        assert len(event_collector.events) > 0

        # Step 3: Generate and verify receipt
        receipt = DecisionReceipt.from_debate_result(result)
        assert receipt is not None
        assert receipt.receipt_id
        assert receipt.verdict in ("PASS", "CONDITIONAL", "FAIL")
        assert receipt.consensus_proof is not None

        # Step 4: Verify receipt integrity (tamper-evident hash)
        assert receipt.verify_integrity() is True


# =============================================================================
# Test: Round Events Arrive in Correct Order
# =============================================================================


@pytest.mark.e2e
class TestEventOrdering:
    """Tests that debate events arrive in the correct chronological order."""

    @pytest.mark.asyncio
    async def test_debate_start_before_debate_end(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
        spectator_with_collector: SpectatorStream,
        event_collector: EventCollector,
    ):
        """Test that debate_start comes before debate_end."""
        arena = Arena(
            debate_environment,
            flow_agents,
            minimal_protocol,
            spectator=spectator_with_collector,
        )
        await arena.run()

        types = event_collector.event_types
        if SpectatorEvents.DEBATE_START in types and SpectatorEvents.DEBATE_END in types:
            start_idx = types.index(SpectatorEvents.DEBATE_START)
            end_idx = types.index(SpectatorEvents.DEBATE_END)
            assert start_idx < end_idx, (
                f"debate_start (index {start_idx}) should come before "
                f"debate_end (index {end_idx})"
            )

    @pytest.mark.asyncio
    async def test_round_events_ordered_sequentially(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
        spectator_with_collector: SpectatorStream,
        event_collector: EventCollector,
    ):
        """Test that round_start events are in sequential order by round number."""
        arena = Arena(
            debate_environment,
            flow_agents,
            minimal_protocol,
            spectator=spectator_with_collector,
        )
        await arena.run()

        round_starts = event_collector.events_of_type(SpectatorEvents.ROUND_START)
        if len(round_starts) >= 2:
            round_numbers = [
                e["round_number"] for e in round_starts if e["round_number"] is not None
            ]
            # Round numbers should be monotonically non-decreasing
            for i in range(1, len(round_numbers)):
                assert round_numbers[i] >= round_numbers[i - 1], (
                    f"Round numbers out of order: {round_numbers}"
                )

    @pytest.mark.asyncio
    async def test_proposals_come_after_round_start(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
        spectator_with_collector: SpectatorStream,
        event_collector: EventCollector,
    ):
        """Test that proposal events come after their round_start."""
        arena = Arena(
            debate_environment,
            flow_agents,
            minimal_protocol,
            spectator=spectator_with_collector,
        )
        await arena.run()

        types = event_collector.event_types
        # If we have both round_start and proposal events, proposals must come after
        if SpectatorEvents.ROUND_START in types and SpectatorEvents.PROPOSAL in types:
            first_round_start = types.index(SpectatorEvents.ROUND_START)
            first_proposal = types.index(SpectatorEvents.PROPOSAL)
            assert first_round_start < first_proposal, (
                f"First round_start (index {first_round_start}) should come before "
                f"first proposal (index {first_proposal})"
            )

    @pytest.mark.asyncio
    async def test_votes_come_after_proposals(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
        spectator_with_collector: SpectatorStream,
        event_collector: EventCollector,
    ):
        """Test that vote events come after proposal events."""
        arena = Arena(
            debate_environment,
            flow_agents,
            minimal_protocol,
            spectator=spectator_with_collector,
        )
        await arena.run()

        types = event_collector.event_types
        if SpectatorEvents.PROPOSAL in types and SpectatorEvents.VOTE in types:
            last_proposal = len(types) - 1 - types[::-1].index(SpectatorEvents.PROPOSAL)
            first_vote = types.index(SpectatorEvents.VOTE)
            # The first vote should come after at least one proposal
            first_proposal = types.index(SpectatorEvents.PROPOSAL)
            assert first_proposal < first_vote, (
                f"First proposal (index {first_proposal}) should come before "
                f"first vote (index {first_vote})"
            )


# =============================================================================
# Test: Receipt Contains Consensus and Vote History
# =============================================================================


@pytest.mark.e2e
class TestReceiptContents:
    """Tests that the decision receipt captures consensus and vote data."""

    @pytest.mark.asyncio
    async def test_receipt_has_consensus_proof(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
    ):
        """Test that the receipt includes a consensus proof section."""
        arena = Arena(debate_environment, flow_agents, minimal_protocol)
        result = await arena.run()

        receipt = DecisionReceipt.from_debate_result(result)

        assert receipt.consensus_proof is not None
        assert isinstance(receipt.consensus_proof, ConsensusProof)
        assert isinstance(receipt.consensus_proof.reached, bool)
        assert receipt.consensus_proof.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_receipt_consensus_tracks_agents(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
    ):
        """Test that the consensus proof identifies supporting agents."""
        arena = Arena(debate_environment, flow_agents, minimal_protocol)
        result = await arena.run()

        receipt = DecisionReceipt.from_debate_result(result)

        assert receipt.consensus_proof is not None
        all_agents = (
            receipt.consensus_proof.supporting_agents
            + receipt.consensus_proof.dissenting_agents
        )
        # If there are participants, at least some should be tracked
        if result.participants:
            assert len(all_agents) > 0

    @pytest.mark.asyncio
    async def test_receipt_has_provenance_chain(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
    ):
        """Test that the receipt includes a provenance chain for audit trail."""
        arena = Arena(debate_environment, flow_agents, minimal_protocol)
        result = await arena.run()

        receipt = DecisionReceipt.from_debate_result(result)

        # At minimum, the verdict event should be in the provenance chain
        assert len(receipt.provenance_chain) >= 1
        # The last provenance record should be the verdict
        verdict_records = [
            r for r in receipt.provenance_chain if r.event_type == "verdict"
        ]
        assert len(verdict_records) > 0

    @pytest.mark.asyncio
    async def test_receipt_captures_vote_events_in_provenance(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
    ):
        """Test that vote events appear in the provenance chain."""
        arena = Arena(debate_environment, flow_agents, minimal_protocol)
        result = await arena.run()

        # If votes were recorded on the result, they should appear in provenance
        if result.votes:
            receipt = DecisionReceipt.from_debate_result(result)
            vote_records = [
                r for r in receipt.provenance_chain if r.event_type == "vote"
            ]
            assert len(vote_records) > 0
            # Each vote record should have an agent attribution
            for record in vote_records:
                assert record.agent is not None or record.description

    @pytest.mark.asyncio
    async def test_receipt_maps_rounds_to_probes(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
    ):
        """Test that receipt probes_run reflects debate rounds used."""
        arena = Arena(debate_environment, flow_agents, minimal_protocol)
        result = await arena.run()

        receipt = DecisionReceipt.from_debate_result(result)

        assert receipt.probes_run == result.rounds_used

    @pytest.mark.asyncio
    async def test_receipt_input_summary_matches_task(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
    ):
        """Test that the receipt input_summary captures the original debate task."""
        arena = Arena(debate_environment, flow_agents, minimal_protocol)
        result = await arena.run()

        receipt = DecisionReceipt.from_debate_result(result)

        # The input summary should contain the original task
        assert receipt.input_summary
        assert debate_environment.task[:100] in receipt.input_summary

    @pytest.mark.asyncio
    async def test_receipt_verdict_reflects_consensus(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
    ):
        """Test that receipt verdict reflects the consensus outcome."""
        arena = Arena(debate_environment, flow_agents, minimal_protocol)
        result = await arena.run()

        # Force known state for deterministic assertion
        result.consensus_reached = True
        result.confidence = 0.9
        receipt = DecisionReceipt.from_debate_result(result)
        assert receipt.verdict == "PASS"

        # Test CONDITIONAL verdict
        result.confidence = 0.5
        result.consensus_reached = True
        receipt2 = DecisionReceipt.from_debate_result(result)
        assert receipt2.verdict == "CONDITIONAL"

        # Test FAIL verdict
        result.consensus_reached = False
        result.confidence = 0.3
        receipt3 = DecisionReceipt.from_debate_result(result)
        assert receipt3.verdict == "FAIL"

    @pytest.mark.asyncio
    async def test_receipt_integrity_hash_is_valid(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
    ):
        """Test that the receipt artifact hash is a valid SHA-256 hex string."""
        arena = Arena(debate_environment, flow_agents, minimal_protocol)
        result = await arena.run()

        receipt = DecisionReceipt.from_debate_result(result)

        assert receipt.artifact_hash
        assert len(receipt.artifact_hash) == 64
        assert all(c in "0123456789abcdef" for c in receipt.artifact_hash)
        assert receipt.verify_integrity() is True

    @pytest.mark.asyncio
    async def test_receipt_tamper_detection(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
    ):
        """Test that tampering with the receipt is detected via hash verification."""
        arena = Arena(debate_environment, flow_agents, minimal_protocol)
        result = await arena.run()

        receipt = DecisionReceipt.from_debate_result(result)
        original_hash = receipt.artifact_hash

        # Tamper with the verdict
        receipt.verdict = "FAIL" if receipt.verdict != "FAIL" else "PASS"

        # Hash should be unchanged (not auto-recalculated)
        assert receipt.artifact_hash == original_hash
        # But verification should now fail
        assert receipt.verify_integrity() is False


# =============================================================================
# Test: Receipt Config Captures Debate Metadata
# =============================================================================


@pytest.mark.e2e
class TestReceiptMetadata:
    """Tests that receipt config_used captures debate execution metadata."""

    @pytest.mark.asyncio
    async def test_receipt_config_has_rounds(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
    ):
        """Test that the receipt config_used records the number of rounds."""
        arena = Arena(debate_environment, flow_agents, minimal_protocol)
        result = await arena.run()

        receipt = DecisionReceipt.from_debate_result(result)

        assert "rounds" in receipt.config_used
        assert receipt.config_used["rounds"] == result.rounds_used

    @pytest.mark.asyncio
    async def test_receipt_config_has_participants(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
    ):
        """Test that the receipt config_used records participants."""
        arena = Arena(debate_environment, flow_agents, minimal_protocol)
        result = await arena.run()

        receipt = DecisionReceipt.from_debate_result(result)

        assert "participants" in receipt.config_used

    @pytest.mark.asyncio
    async def test_receipt_config_has_duration(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
    ):
        """Test that the receipt config_used records debate duration."""
        arena = Arena(debate_environment, flow_agents, minimal_protocol)
        result = await arena.run()

        receipt = DecisionReceipt.from_debate_result(result)

        assert "duration_seconds" in receipt.config_used

    @pytest.mark.asyncio
    async def test_receipt_to_dict_is_serializable(
        self,
        debate_environment: Environment,
        minimal_protocol: DebateProtocol,
        flow_agents: list[FlowMockAgent],
    ):
        """Test that receipt can be serialized to JSON (export-ready)."""
        arena = Arena(debate_environment, flow_agents, minimal_protocol)
        result = await arena.run()

        receipt = DecisionReceipt.from_debate_result(result)

        # to_dict should produce a JSON-serializable structure
        receipt_dict = receipt.to_dict()
        json_str = json.dumps(receipt_dict)
        assert len(json_str) > 0

        # Round-trip should preserve key fields
        parsed = json.loads(json_str)
        assert parsed["receipt_id"] == receipt.receipt_id
        assert parsed["verdict"] == receipt.verdict
