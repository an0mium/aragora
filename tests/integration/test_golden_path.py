"""
End-to-end golden path integration tests: debate creation -> execution ->
receipt generation -> delivery routing.

Validates the WIRING between core components without real LLM calls:
1. Debate with mocked agents runs to completion through Arena
2. DecisionReceipt is generated from the debate result with audit fields
3. Result routing dispatches to the originating channel
4. Debate outcome persists to KnowledgeMound via DebateAdapter
5. Full pipeline from idea to receipt via IdeaToExecutionPipeline

GitHub issue: #309
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import Agent, Critique, Environment, Message, Vote
from aragora.core_types import DebateResult
from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import DebateProtocol
from aragora.gauntlet.receipt import (
    ConsensusProof,
    DecisionReceipt,
    ProvenanceRecord,
)
from aragora.knowledge.mound.adapters.debate_adapter import (
    DebateAdapter,
    DebateOutcome,
)


# =============================================================================
# Mock agent that returns deterministic responses (no real API calls)
# =============================================================================


class _GoldenPathAgent(Agent):
    """Lightweight mock agent for golden-path wiring tests.

    Mocks at the LLM boundary: generate/critique/vote produce canned
    responses so the Arena orchestration logic runs for real.
    """

    def __init__(self, name: str, proposal: str, vote_for: str | None = None):
        super().__init__(name=name, model="mock-golden-path", role="proposer")
        self.agent_type = "mock"
        self._proposal = proposal
        self._vote_for = vote_for  # None => vote for first proposal
        # Token tracking attributes expected by extensions
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.metrics = None
        self.provider = None

    async def generate(self, prompt: str, context: list | None = None) -> str:
        return self._proposal

    async def generate_stream(self, prompt: str, context: list | None = None):
        yield self._proposal

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list | None = None,
        target_agent: str | None = None,
    ) -> Critique:
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal[:100] if proposal else "",
            issues=[],
            suggestions=[],
            severity=0.1,
            reasoning="Minor remarks only",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        choice = self._vote_for
        if choice is None:
            choice = list(proposals.keys())[0] if proposals else self.name
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning="I concur with this position",
            confidence=0.88,
            continue_debate=False,
        )


# =============================================================================
# Shared fixtures
# =============================================================================


@pytest.fixture
def task_description() -> str:
    return "Should we migrate our primary datastore from PostgreSQL to CockroachDB?"


@pytest.fixture
def consensus_agents() -> list[_GoldenPathAgent]:
    """Three agents that will converge on a shared proposal.

    All agents explicitly vote for agent-claude so majority voting
    reaches consensus deterministically regardless of proposal dict
    key ordering.
    """
    shared = (
        "Keep PostgreSQL as primary; add CockroachDB as a secondary "
        "read-replica for geo-distributed queries."
    )
    return [
        _GoldenPathAgent("agent-claude", proposal=shared, vote_for="agent-claude"),
        _GoldenPathAgent("agent-gpt", proposal=shared, vote_for="agent-claude"),
        _GoldenPathAgent("agent-gemini", proposal=shared, vote_for="agent-claude"),
    ]


@pytest.fixture
def environment(task_description: str) -> Environment:
    return Environment(task=task_description)


@pytest.fixture
def fast_protocol() -> DebateProtocol:
    """Minimal protocol: 2 rounds, no heavy subsystems."""
    return DebateProtocol(
        rounds=2,
        consensus="majority",
        enable_calibration=False,
        enable_rhetorical_observer=False,
        enable_trickster=False,
    )


# =============================================================================
# 1. Debate to Receipt
# =============================================================================


class TestDebateToReceipt:
    """Create a debate with mocked agents, run to completion, and verify
    the receipt is generated with proper audit fields."""

    @pytest.mark.asyncio
    async def test_debate_to_receipt(
        self,
        environment: Environment,
        fast_protocol: DebateProtocol,
        consensus_agents: list[_GoldenPathAgent],
    ):
        """Arena.run() -> DebateResult -> DecisionReceipt round-trip."""
        arena = Arena(environment, consensus_agents, fast_protocol)
        result = await arena.run()

        # -- Debate completed successfully --
        assert result is not None, "Arena.run() should return a DebateResult"
        assert result.final_answer, "final_answer should be non-empty"
        assert result.rounds_completed > 0, "At least one round executed"

        # -- Generate receipt --
        receipt = DecisionReceipt.from_debate_result(result)

        assert receipt.receipt_id, "receipt_id must be populated"
        assert receipt.gauntlet_id, "gauntlet_id (debate id) must be set"
        assert receipt.timestamp, "timestamp must be populated"
        assert receipt.verdict in ("PASS", "CONDITIONAL", "FAIL")
        assert 0.0 <= receipt.confidence <= 1.0

        # Receipt fields must map back to debate result
        assert receipt.probes_run == result.rounds_used
        assert receipt.confidence == result.confidence

        # Consensus proof wired correctly
        assert receipt.consensus_proof is not None
        assert isinstance(receipt.consensus_proof.reached, bool)
        assert receipt.consensus_proof.confidence == result.confidence

    @pytest.mark.asyncio
    async def test_receipt_contains_audit_fields(
        self,
        environment: Environment,
        fast_protocol: DebateProtocol,
        consensus_agents: list[_GoldenPathAgent],
    ):
        """Verify receipt has SHA-256 hash, agent list, vote tally,
        winning proposal, and provenance chain."""
        arena = Arena(environment, consensus_agents, fast_protocol)
        result = await arena.run()
        receipt = DecisionReceipt.from_debate_result(result)

        # --- SHA-256 artifact hash ---
        assert receipt.artifact_hash, "artifact_hash must be populated"
        assert len(receipt.artifact_hash) == 64, (
            f"SHA-256 hash must be 64 hex chars, got {len(receipt.artifact_hash)}"
        )
        assert all(c in "0123456789abcdef" for c in receipt.artifact_hash)

        # Integrity verification passes (hash matches content)
        assert receipt.verify_integrity() is True

        # --- Agent list in consensus proof ---
        proof = receipt.consensus_proof
        assert proof is not None
        all_agents = proof.supporting_agents + proof.dissenting_agents
        assert len(all_agents) > 0, "At least one agent should be tracked"

        # --- Provenance chain ---
        assert len(receipt.provenance_chain) >= 1, "Must have at least the verdict event"
        verdict_events = [p for p in receipt.provenance_chain if p.event_type == "verdict"]
        assert len(verdict_events) >= 1, "Provenance must include a verdict event"

        # --- Config captures round count ---
        assert receipt.config_used.get("rounds") == result.rounds_used

        # --- JSON serialization round-trip ---
        json_str = receipt.to_json()
        data = json.loads(json_str)
        assert "receipt_id" in data
        assert "artifact_hash" in data
        assert "consensus_proof" in data
        assert "provenance_chain" in data

        # Restore from JSON preserves integrity
        restored = DecisionReceipt.from_dict(data)
        assert restored.artifact_hash == receipt.artifact_hash
        assert restored.verify_integrity() is True


# =============================================================================
# 2. Tamper detection
# =============================================================================


class TestReceiptTamperDetection:
    """Verify SHA-256 integrity detects post-hoc modifications."""

    def test_tamper_detection(self):
        """Changing verdict after receipt creation invalidates the hash."""
        result = DebateResult(
            task="Should we use Rust or Go?",
            final_answer="Go for the team's productivity",
            confidence=0.85,
            consensus_reached=True,
            rounds_used=3,
            participants=["alice", "bob", "carol"],
            dissenting_views=[],
        )
        receipt = DecisionReceipt.from_debate_result(result)

        # Original receipt should be PASS (high confidence + consensus)
        assert receipt.verdict == "PASS"
        original_hash = receipt.artifact_hash
        assert receipt.verify_integrity() is True

        # Tamper with verdict
        receipt.verdict = "FAIL"

        # Hash was computed at construction and is now stale
        assert receipt.artifact_hash == original_hash
        assert receipt.verify_integrity() is False, "Integrity check should fail after tampering"

    def test_tamper_detection_confidence(self):
        """Changing confidence after receipt creation invalidates the hash."""
        result = DebateResult(
            task="Which CI system?",
            final_answer="GitHub Actions",
            confidence=0.9,
            consensus_reached=True,
            rounds_used=2,
        )
        receipt = DecisionReceipt.from_debate_result(result)

        assert receipt.verify_integrity() is True

        receipt.confidence = 0.1
        assert receipt.verify_integrity() is False


# =============================================================================
# 3. Debate result routing
# =============================================================================


class TestDebateResultRouting:
    """Verify that debate completion triggers result routing back to the
    originating channel (all delivery channels mocked)."""

    @pytest.mark.asyncio
    async def test_debate_result_routing(
        self,
        environment: Environment,
        fast_protocol: DebateProtocol,
        consensus_agents: list[_GoldenPathAgent],
    ):
        """Register an origin, route the result, verify delivery."""
        arena = Arena(environment, consensus_agents, fast_protocol)
        result = await arena.run()

        debate_id = getattr(result, "debate_id", None) or "test-debate-routing-001"

        # Simulate a Slack origin registration
        from aragora.server.debate_origin import register_debate_origin

        origin = register_debate_origin(
            debate_id=debate_id,
            platform="slack",
            channel_id="C_GENERAL",
            user_id="U_TESTER",
            metadata={"username": "golden_path_bot"},
        )
        assert origin is not None
        assert origin.platform == "slack"

        # Build the result dict that route_debate_result expects
        result_dict: dict[str, Any] = {
            "consensus_reached": result.consensus_reached,
            "final_answer": result.final_answer,
            "confidence": result.confidence,
            "participants": result.participants or [],
        }

        # Mock the Slack sender so no real HTTP calls are made
        with patch(
            "aragora.server.debate_origin.router._send_slack_result",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_send:
            from aragora.server.debate_origin.router import route_debate_result

            routed = await route_debate_result(debate_id, result_dict)

            assert routed is True, "route_debate_result should return True on success"
            assert mock_send.called, "_send_slack_result should have been invoked"

            # Verify the sender received the debate_id's origin
            call_args = mock_send.call_args
            # The sender receives (origin, formatted_message) or similar
            assert call_args is not None

    @pytest.mark.asyncio
    async def test_routing_returns_false_when_no_origin(self):
        """Routing a debate with no registered origin returns False."""
        from aragora.server.result_router import route_result

        # Use a random debate_id that was never registered
        routed = await route_result("nonexistent-debate-999", {"final_answer": "x"})
        assert routed is False


# =============================================================================
# 4. Debate to Knowledge Mound
# =============================================================================


class TestDebateToKnowledgeMound:
    """Verify debate outcome writes to KnowledgeMound via DebateAdapter."""

    @pytest.mark.asyncio
    async def test_debate_to_knowledge_mound(
        self,
        environment: Environment,
        fast_protocol: DebateProtocol,
        consensus_agents: list[_GoldenPathAgent],
    ):
        """store_outcome() + sync_to_km() persists the debate to KM."""
        arena = Arena(environment, consensus_agents, fast_protocol)
        result = await arena.run()

        # --- Store outcome in adapter ---
        adapter = DebateAdapter()
        adapter.store_outcome(result)

        assert len(adapter._pending_outcomes) == 1
        outcome = adapter._pending_outcomes[0]
        assert outcome.task == result.task
        assert outcome.confidence == result.confidence
        assert outcome.consensus_reached == result.consensus_reached
        assert outcome.metadata.get("km_sync_pending") is True

        # --- Sync to a mock KnowledgeMound ---
        mock_mound = AsyncMock()
        mock_mound.store_item = AsyncMock()

        sync_result = await adapter.sync_to_km(mock_mound)

        assert sync_result.records_synced == 1
        assert sync_result.records_failed == 0
        assert len(adapter._pending_outcomes) == 0, "Pending list should be drained"

        # Verify the KM store was called
        mock_mound.store_item.assert_called_once()
        stored_item = mock_mound.store_item.call_args[0][0]
        assert stored_item.id is not None
        assert stored_item.source == "debate"

    @pytest.mark.asyncio
    async def test_low_confidence_skipped(
        self,
        environment: Environment,
        fast_protocol: DebateProtocol,
        consensus_agents: list[_GoldenPathAgent],
    ):
        """Debates below the confidence threshold are skipped during sync."""
        arena = Arena(environment, consensus_agents, fast_protocol)
        result = await arena.run()

        # Force very low confidence
        result.confidence = 0.1

        adapter = DebateAdapter()
        adapter.store_outcome(result)

        mock_mound = AsyncMock()
        mock_mound.store_item = AsyncMock()

        # Sync with a high confidence threshold
        sync_result = await adapter.sync_to_km(mock_mound, min_confidence=0.5)

        assert sync_result.records_synced == 0
        assert sync_result.records_skipped == 1
        mock_mound.store_item.assert_not_called()


# =============================================================================
# 5. Full pipeline: idea to receipt
# =============================================================================


class TestFullPipelineIdeaToReceipt:
    """Create ideas -> plan -> workflow -> orchestrate debate -> receipt.

    Uses the IdeaToExecutionPipeline's synchronous from_ideas() path
    followed by DecisionReceipt generation from a simulated debate result.
    """

    @pytest.mark.asyncio
    async def test_full_pipeline_idea_to_receipt(
        self,
        task_description: str,
    ):
        from aragora.pipeline.idea_to_execution import (
            IdeaToExecutionPipeline,
            PipelineResult,
        )

        # --- Stage A: Pipeline from ideas ---
        pipeline = IdeaToExecutionPipeline()
        ideas = [
            "[high] Evaluate CockroachDB vs PostgreSQL for geo-distributed reads",
            "[high] Design read-replica failover with automatic promotion",
            "[medium] Benchmark write throughput under multi-region load",
            "[low] Document migration rollback procedure",
        ]
        pipeline_result = pipeline.from_ideas(ideas, auto_advance=True)

        assert isinstance(pipeline_result, PipelineResult)
        assert pipeline_result.pipeline_id.startswith("pipe-")
        assert pipeline_result.ideas_canvas is not None
        assert len(pipeline_result.ideas_canvas.nodes) >= len(ideas)
        assert pipeline_result.goal_graph is not None
        assert len(pipeline_result.goal_graph.goals) >= 1
        assert pipeline_result.actions_canvas is not None

        # All stages completed
        for stage in ("ideas", "goals", "actions", "orchestration"):
            assert pipeline_result.stage_status.get(stage) == "complete", (
                f"Stage '{stage}' should be 'complete', "
                f"got '{pipeline_result.stage_status.get(stage)}'"
            )

        # Pipeline integrity hash
        result_dict = pipeline_result.to_dict()
        integrity_hash = result_dict["integrity_hash"]
        assert integrity_hash is not None
        assert len(integrity_hash) == 16
        assert all(c in "0123456789abcdef" for c in integrity_hash)

        # Deterministic
        assert pipeline_result._compute_integrity_hash() == integrity_hash

        # --- Stage B: Simulate debate on the pipeline goals ---
        debate_result = DebateResult(
            task=task_description,
            final_answer=(
                "Keep PostgreSQL as primary datastore. Add CockroachDB as "
                "read-replica for geo-distributed queries only."
            ),
            confidence=0.91,
            consensus_reached=True,
            rounds_used=3,
            participants=["agent-claude", "agent-gpt", "agent-gemini"],
            dissenting_views=[],
        )

        # --- Stage C: Generate receipt ---
        receipt = DecisionReceipt.from_debate_result(debate_result)

        assert receipt.verdict == "PASS"
        assert receipt.confidence == 0.91
        assert receipt.verify_integrity() is True

        # Provenance chain contains verdict
        verdict_events = [p for p in receipt.provenance_chain if p.event_type == "verdict"]
        assert len(verdict_events) >= 1

        # JSON round-trip works
        data = json.loads(receipt.to_json())
        assert data["verdict"] == "PASS"
        assert data["confidence"] == 0.91


# =============================================================================
# 6. Dissent capture and conditional verdicts
# =============================================================================


class TestDissentAndConditionalVerdicts:
    """Verify that dissenting views propagate through the receipt and that
    low-confidence consensus yields a CONDITIONAL verdict."""

    def test_dissent_captured_in_receipt(self):
        result = DebateResult(
            task="Use microservices or monolith?",
            final_answer="Start monolith, extract services later",
            confidence=0.72,
            consensus_reached=True,
            rounds_used=4,
            participants=["claude", "gpt", "contrarian"],
            dissenting_views=[
                "contrarian: Microservices from day one avoids future migration pain"
            ],
        )
        receipt = DecisionReceipt.from_debate_result(result)

        assert len(receipt.dissenting_views) == 1
        assert "contrarian" in receipt.dissenting_views[0]
        assert receipt.consensus_proof is not None
        assert "contrarian" in receipt.consensus_proof.dissenting_agents

    def test_low_confidence_conditional_verdict(self):
        result = DebateResult(
            task="Which cloud provider?",
            final_answer="AWS",
            confidence=0.55,
            consensus_reached=True,
            rounds_used=3,
        )
        receipt = DecisionReceipt.from_debate_result(result)
        assert receipt.verdict == "CONDITIONAL"

    def test_no_consensus_fail_verdict(self):
        result = DebateResult(
            task="Which language?",
            final_answer="No agreement",
            confidence=0.3,
            consensus_reached=False,
            rounds_used=5,
        )
        receipt = DecisionReceipt.from_debate_result(result)
        assert receipt.verdict == "FAIL"
        assert receipt.consensus_proof is not None
        assert receipt.consensus_proof.reached is False
