"""
End-to-end integration tests for the Aragora value chain.

Tests the complete flow from debate creation through receipt generation,
post-debate processing, Knowledge Mound persistence, semantic search,
and approval workflows.

Steps tested:
1. Create a debate -- Arena with mock agents, 2-round debate
2. Generate receipt -- DecisionReceipt from debate result with SHA-256 hash
3. Run post-debate coordinator -- PostDebateCoordinator.run() produces results
4. Persist to KM -- Receipt ingested into KnowledgeMound via ReceiptAdapter
5. Search KM -- Persisted receipt retrieved via semantic search
6. Approval flow -- ApprovalWorkflow request creation and approval

Also tests failure paths:
- Debate with no consensus -> receipt still generated
- KM unavailable -> graceful degradation
- Receipt with missing fields -> validation error
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import (
    Agent,
    Critique,
    DebateResult,
    Environment,
    Message,
    Vote,
)
from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import DebateProtocol
from aragora.debate.post_debate_coordinator import (
    PostDebateConfig,
    PostDebateCoordinator,
    PostDebateResult,
)
from aragora.gauntlet.receipt_models import (
    ConsensusProof,
    DecisionReceipt,
    ProvenanceRecord,
)
from aragora.knowledge.mound.adapters.receipt_adapter import (
    ReceiptAdapter,
    ReceiptIngestionResult,
)
from aragora.knowledge.unified.types import (
    ConfidenceLevel,
    KnowledgeItem,
    KnowledgeSource,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


# =============================================================================
# Fixtures
# =============================================================================


class _MockAgent(Agent):
    """Deterministic mock agent for integration testing."""

    def __init__(
        self,
        name: str,
        role: str = "proposer",
        responses: list[str] | None = None,
    ):
        super().__init__(name, "mock-model", role)
        self.agent_type = "mock"
        self._responses = responses or [f"Response from {name}"]
        self._call_idx = 0

    async def generate(self, prompt: str, context: list | None = None) -> str:
        resp = self._responses[self._call_idx % len(self._responses)]
        self._call_idx += 1
        return resp

    async def critique(
        self, proposal: str, task: str, context: list | None = None
    ) -> Critique:
        self._call_idx += 1
        return Critique(
            agent=self.name,
            target_agent="target",
            target_content=proposal[:80],
            issues=["Minor concern"],
            suggestions=["Iterate further"],
            severity=0.2,
            reasoning="Reasonable but can improve",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        self._call_idx += 1
        choice = list(proposals.keys())[0] if proposals else self.name
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning=f"Vote from {self.name}",
            confidence=0.88,
            continue_debate=False,
        )


@pytest.fixture
def proposer_agent() -> _MockAgent:
    return _MockAgent(
        name="proposer_alpha",
        role="proposer",
        responses=[
            "I propose using a token bucket algorithm for rate limiting. "
            "It supports burst traffic while enforcing steady-state limits.",
            "Revised proposal: token bucket with sliding window fallback "
            "for distributed environments.",
        ],
    )


@pytest.fixture
def critic_agent() -> _MockAgent:
    return _MockAgent(
        name="critic_beta",
        role="critic",
        responses=[
            "The token bucket approach is reasonable but needs to address "
            "distributed coordination across nodes.",
            "Revised approach addresses my concerns. I support this design.",
        ],
    )


@pytest.fixture
def synthesizer_agent() -> _MockAgent:
    return _MockAgent(
        name="synthesizer_gamma",
        role="synthesizer",
        responses=[
            "Synthesis: use token bucket with sliding window fallback, "
            "coordinated via Redis for multi-node consistency.",
        ],
    )


@pytest.fixture
def consensus_agents(proposer_agent, critic_agent, synthesizer_agent):
    """Agents that will converge toward consensus."""
    return [proposer_agent, critic_agent, synthesizer_agent]


@pytest.fixture
def split_agents() -> list[_MockAgent]:
    """Agents that produce a split vote (no consensus)."""
    return [
        _MockAgent(
            name="agent_a",
            role="proposer",
            responses=["We should go with option A."],
        ),
        _MockAgent(
            name="agent_b",
            role="critic",
            responses=["Option A is fundamentally flawed. Option B is better."],
        ),
        _MockAgent(
            name="agent_c",
            role="synthesizer",
            responses=["Both options have merit but are irreconcilable."],
        ),
    ]


@pytest.fixture
def quick_protocol() -> DebateProtocol:
    """Two-round debate with majority consensus, simple phases."""
    return DebateProtocol(
        rounds=2,
        consensus="majority",
        critique_required=False,
        use_structured_phases=False,
    )


@pytest.fixture
def simple_env() -> Environment:
    return Environment(
        task="Design a rate limiter API for a multi-tenant SaaS platform",
        context="The platform serves 10,000 tenants with varying traffic profiles.",
    )


@pytest.fixture
def mock_mound():
    """Mock KnowledgeMound with store_sync and query support."""
    mound = MagicMock()
    mound._stored_items: dict[str, KnowledgeItem] = {}

    def _store_sync(item: KnowledgeItem):
        mound._stored_items[item.id] = item

    async def _store(item: KnowledgeItem):
        mound._stored_items[item.id] = item
        return item

    async def _query(query: str, tags=None, workspace_id=None, limit=5):
        # Simple substring search over stored items
        matches = []
        for item in mound._stored_items.values():
            if query.lower() in item.content.lower():
                matches.append(item)
            elif tags:
                item_tags = item.metadata.get("tags", [])
                if any(t in item_tags for t in tags):
                    matches.append(item)
        result = MagicMock()
        result.items = matches[:limit]
        return result

    mound.store_sync = MagicMock(side_effect=_store_sync)
    mound.store = AsyncMock(side_effect=_store)
    mound.query = AsyncMock(side_effect=_query)
    return mound


# =============================================================================
# Helper: run a debate using Arena with mock externals
# =============================================================================


async def _run_debate(
    agents: list[_MockAgent],
    env: Environment,
    protocol: DebateProtocol,
    timeout: float = 30.0,
) -> DebateResult:
    """Run a debate with all external dependencies mocked."""
    arena = Arena(env, agents, protocol)
    return await asyncio.wait_for(arena.run(), timeout=timeout)


# =============================================================================
# Step 1: Create a debate
# =============================================================================


class TestStep1CreateDebate:
    """Verify Arena runs a debate with mock agents and produces a DebateResult.

    Note: The integration conftest patches ContextInitializer.initialize (autouse)
    to prevent external calls. This means the proposal phase may not assign
    proposer roles, so proposals can be empty. We verify Arena completes and
    returns a well-formed DebateResult.
    """

    async def test_debate_completes_with_result(
        self, consensus_agents, simple_env, quick_protocol
    ):
        result = await _run_debate(consensus_agents, simple_env, quick_protocol)

        assert result is not None
        assert isinstance(result, DebateResult)
        assert result.rounds_completed >= 1
        assert result.task == simple_env.task

    async def test_debate_records_participants(
        self, consensus_agents, simple_env, quick_protocol
    ):
        result = await _run_debate(consensus_agents, simple_env, quick_protocol)

        # Participants should include agent names
        agent_names = {a.name for a in consensus_agents}
        result_participants = set(result.participants)
        assert result_participants & agent_names, (
            f"Expected at least some of {agent_names} in participants {result_participants}"
        )

    async def test_debate_generates_votes(
        self, consensus_agents, simple_env, quick_protocol
    ):
        result = await _run_debate(consensus_agents, simple_env, quick_protocol)

        assert len(result.votes) > 0, "Debate should generate at least one vote"

    async def test_debate_has_final_answer(
        self, consensus_agents, simple_env, quick_protocol
    ):
        result = await _run_debate(consensus_agents, simple_env, quick_protocol)

        # Arena always produces a final_answer (may be a summary fallback)
        assert result.final_answer, "Debate must produce a final_answer"


# =============================================================================
# Step 2: Generate receipt from debate result
# =============================================================================


class TestStep2GenerateReceipt:
    """Verify DecisionReceipt is created from DebateResult with SHA-256 hash."""

    @pytest.fixture
    def debate_result(self) -> DebateResult:
        """Pre-built debate result for receipt generation tests."""
        return DebateResult(
            debate_id="debate-e2e-001",
            task="Design a rate limiter API",
            final_answer="Use token bucket with sliding window fallback.",
            confidence=0.88,
            consensus_reached=True,
            rounds_used=2,
            rounds_completed=2,
            participants=["proposer_alpha", "critic_beta", "synthesizer_gamma"],
            proposals={
                "proposer_alpha": "Token bucket algorithm",
                "critic_beta": "Sliding window approach",
            },
            votes=[
                Vote(
                    agent="proposer_alpha",
                    choice="proposer_alpha",
                    reasoning="My proposal is best",
                    confidence=0.9,
                ),
                Vote(
                    agent="critic_beta",
                    choice="proposer_alpha",
                    reasoning="Strong proposal",
                    confidence=0.85,
                ),
                Vote(
                    agent="synthesizer_gamma",
                    choice="proposer_alpha",
                    reasoning="Combined approach",
                    confidence=0.92,
                ),
            ],
            dissenting_views=[],
            winner="proposer_alpha",
            duration_seconds=12.5,
        )

    def test_receipt_from_debate_result(self, debate_result):
        receipt = DecisionReceipt.from_debate_result(debate_result)

        assert receipt.receipt_id, "Receipt must have an ID"
        assert receipt.gauntlet_id == "debate-e2e-001"
        assert receipt.confidence == 0.88
        assert receipt.verdict == "PASS"  # High-confidence consensus -> PASS

    def test_receipt_has_sha256_artifact_hash(self, debate_result):
        receipt = DecisionReceipt.from_debate_result(debate_result)

        assert receipt.artifact_hash, "Receipt must have artifact_hash"
        assert len(receipt.artifact_hash) == 64, "SHA-256 hash should be 64 hex chars"

    def test_receipt_integrity_verification(self, debate_result):
        receipt = DecisionReceipt.from_debate_result(debate_result)

        assert receipt.verify_integrity(), "Fresh receipt should pass integrity check"

    def test_receipt_detects_tampering(self, debate_result):
        receipt = DecisionReceipt.from_debate_result(debate_result)
        original_hash = receipt.artifact_hash

        # Tamper with the verdict
        receipt.verdict = "FAIL"
        # Re-check -- the stored hash no longer matches
        assert not receipt.verify_integrity(), "Tampered receipt should fail integrity"

    def test_receipt_has_consensus_proof(self, debate_result):
        receipt = DecisionReceipt.from_debate_result(debate_result)

        assert receipt.consensus_proof is not None
        assert receipt.consensus_proof.reached is True
        assert receipt.consensus_proof.confidence == 0.88

    def test_receipt_has_provenance_chain(self, debate_result):
        receipt = DecisionReceipt.from_debate_result(debate_result)

        assert len(receipt.provenance_chain) > 0, "Receipt must have provenance records"
        # Should have a verdict event
        verdict_events = [
            p for p in receipt.provenance_chain if p.event_type == "verdict"
        ]
        assert len(verdict_events) >= 1

    def test_receipt_has_vote_provenance(self, debate_result):
        receipt = DecisionReceipt.from_debate_result(debate_result)

        vote_events = [
            p for p in receipt.provenance_chain if p.event_type == "vote"
        ]
        assert len(vote_events) == 3, "Should have one provenance record per vote"

    def test_receipt_serialization_roundtrip(self, debate_result):
        receipt = DecisionReceipt.from_debate_result(debate_result)
        data = receipt.to_dict()

        reconstructed = DecisionReceipt.from_dict(data)
        assert reconstructed.receipt_id == receipt.receipt_id
        assert reconstructed.verdict == receipt.verdict
        assert reconstructed.confidence == receipt.confidence
        assert reconstructed.artifact_hash == receipt.artifact_hash


# =============================================================================
# Step 3: Run PostDebateCoordinator
# =============================================================================


class TestStep3PostDebateCoordinator:
    """Verify PostDebateCoordinator.run() produces results."""

    @pytest.fixture
    def debate_result(self) -> DebateResult:
        return DebateResult(
            debate_id="debate-e2e-002",
            task="Design a caching strategy",
            final_answer="Multi-tier caching with Redis and local LRU.",
            confidence=0.85,
            consensus_reached=True,
            rounds_used=2,
            rounds_completed=2,
            participants=["agent_x", "agent_y"],
            proposals={"agent_x": "Redis caching", "agent_y": "LRU local cache"},
        )

    def test_coordinator_produces_result(self, debate_result):
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_persist_receipt=False,
            auto_outcome_feedback=False,
            auto_queue_improvement=False,
            auto_trigger_canvas=False,
            auto_execution_bridge=False,
            auto_llm_judge=False,
        )
        coordinator = PostDebateCoordinator(config=config)
        result = coordinator.run(
            debate_id="debate-e2e-002",
            debate_result=debate_result,
            confidence=0.85,
            task="Design a caching strategy",
        )

        assert isinstance(result, PostDebateResult)
        assert result.debate_id == "debate-e2e-002"

    def test_coordinator_receipt_persistence_step(self, debate_result):
        """Test that auto_persist_receipt triggers ReceiptAdapter.ingest."""
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_persist_receipt=True,
            auto_outcome_feedback=False,
            auto_queue_improvement=False,
            auto_trigger_canvas=False,
            auto_execution_bridge=False,
            auto_llm_judge=False,
        )
        coordinator = PostDebateCoordinator(config=config)

        # Patch get_receipt_adapter at the source module where it's imported from
        mock_adapter = MagicMock()
        mock_adapter.ingest.return_value = True

        with patch(
            "aragora.knowledge.mound.adapters.receipt_adapter.get_receipt_adapter",
            return_value=mock_adapter,
        ):
            result = coordinator.run(
                debate_id="debate-e2e-002",
                debate_result=debate_result,
                confidence=0.85,
                task="Design a caching strategy",
            )

        assert result.receipt_persisted is True
        mock_adapter.ingest.assert_called_once()
        call_args = mock_adapter.ingest.call_args[0][0]
        assert call_args["debate_id"] == "debate-e2e-002"
        assert call_args["task"] == "Design a caching strategy"

    def test_coordinator_handles_step_failures_gracefully(self, debate_result):
        """Failed steps should not prevent subsequent steps from running."""
        config = PostDebateConfig(
            auto_explain=True,  # Will fail (no real ExplanationBuilder)
            auto_create_plan=False,
            auto_notify=False,
            auto_persist_receipt=False,
            auto_outcome_feedback=False,
            auto_queue_improvement=False,
            auto_trigger_canvas=False,
            auto_execution_bridge=False,
            auto_llm_judge=False,
        )
        coordinator = PostDebateCoordinator(config=config)
        result = coordinator.run(
            debate_id="debate-e2e-002",
            debate_result=debate_result,
            confidence=0.85,
            task="Design a caching strategy",
        )

        # Should complete without raising, explanation may be None
        assert isinstance(result, PostDebateResult)


# =============================================================================
# Step 4: Persist to Knowledge Mound via ReceiptAdapter
# =============================================================================


class TestStep4PersistToKM:
    """Verify receipt ingestion into KnowledgeMound via ReceiptAdapter."""

    def test_receipt_adapter_sync_ingest(self, mock_mound):
        adapter = ReceiptAdapter(mound=mock_mound)

        success = adapter.ingest({
            "debate_id": "debate-e2e-003",
            "task": "Design a rate limiter API",
            "confidence": 0.9,
            "consensus_reached": True,
            "final_answer": "Token bucket with sliding window fallback.",
            "participants": ["alpha", "beta", "gamma"],
        })

        assert success is True
        mock_mound.store_sync.assert_called_once()

        # Verify the stored item has expected content
        stored_item = mock_mound.store_sync.call_args[0][0]
        assert isinstance(stored_item, KnowledgeItem)
        assert "rate limiter" in stored_item.content.lower()
        assert stored_item.source == KnowledgeSource.DEBATE
        assert stored_item.metadata["debate_id"] == "debate-e2e-003"
        assert stored_item.metadata["consensus_reached"] is True

    def test_receipt_adapter_tracks_ingestion(self, mock_mound):
        adapter = ReceiptAdapter(mound=mock_mound)

        adapter.ingest({
            "debate_id": "debate-track-001",
            "task": "Test tracking",
            "confidence": 0.75,
            "consensus_reached": True,
            "final_answer": "Answer here.",
            "participants": ["a"],
        })

        result = adapter.get_ingestion_result("debate-track-001")
        assert result is not None
        assert isinstance(result, ReceiptIngestionResult)
        assert result.claims_ingested == 1
        assert result.receipt_id == "debate-track-001"
        assert result.success

    async def test_receipt_adapter_async_ingest(self, mock_mound):
        """Test the full async ingest_receipt path with a DecisionReceipt."""
        adapter = ReceiptAdapter(mound=mock_mound)

        receipt = DecisionReceipt(
            receipt_id="rcpt-async-001",
            gauntlet_id="debate-async-001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Async ingest test for rate limiter design",
            input_hash=hashlib.sha256(b"test input").hexdigest(),
            risk_summary={"critical": 0, "high": 0, "medium": 0, "low": 0, "total": 0},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=2,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.92,
            robustness_score=0.88,
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.92,
                supporting_agents=["alpha", "beta"],
                method="majority",
            ),
        )

        result = await adapter.ingest_receipt(receipt, workspace_id="ws-test")

        assert isinstance(result, ReceiptIngestionResult)
        assert result.receipt_id == "rcpt-async-001"
        # Should at least have the summary item
        assert len(result.knowledge_item_ids) >= 1

    def test_receipt_adapter_stats(self, mock_mound):
        adapter = ReceiptAdapter(mound=mock_mound)

        adapter.ingest({
            "debate_id": "stats-001",
            "task": "Stats test",
            "confidence": 0.8,
            "consensus_reached": True,
            "final_answer": "Answer.",
            "participants": [],
        })

        stats = adapter.get_stats()
        assert stats["receipts_processed"] == 1
        assert stats["total_claims_ingested"] == 1
        assert stats["mound_connected"] is True


# =============================================================================
# Step 5: Search KM for persisted receipt
# =============================================================================


class TestStep5SearchKM:
    """Verify persisted receipts can be retrieved via semantic search."""

    async def test_find_related_decisions_returns_ingested_receipt(self, mock_mound):
        adapter = ReceiptAdapter(mound=mock_mound)

        # Ingest a receipt about rate limiting
        adapter.ingest({
            "debate_id": "search-001",
            "task": "Design a rate limiter API",
            "confidence": 0.9,
            "consensus_reached": True,
            "final_answer": "Token bucket with sliding window fallback.",
            "participants": ["alpha", "beta"],
        })

        # Search for related decisions
        results = await adapter.find_related_decisions("rate limiter", limit=5)

        assert len(results) >= 1, "Should find the ingested receipt via search"
        found = results[0]
        assert "rate limiter" in found.content.lower()

    async def test_search_with_multiple_receipts_returns_matching(self, mock_mound):
        adapter = ReceiptAdapter(mound=mock_mound)

        adapter.ingest({
            "debate_id": "search-002a",
            "task": "Database schema design",
            "confidence": 0.8,
            "consensus_reached": True,
            "final_answer": "Use normalized schema with indexes.",
            "participants": ["a"],
        })
        adapter.ingest({
            "debate_id": "search-002b",
            "task": "API rate limiting approach",
            "confidence": 0.85,
            "consensus_reached": True,
            "final_answer": "Token bucket algorithm.",
            "participants": ["b"],
        })

        # Search for rate limiting - should match the second receipt
        results = await adapter.find_related_decisions("rate limiting")
        matching_content = [r.content for r in results]
        assert any("rate limiting" in c.lower() for c in matching_content)

    async def test_search_with_no_mound_returns_empty(self):
        adapter = ReceiptAdapter(mound=None)

        results = await adapter.find_related_decisions("anything")
        assert results == []


# =============================================================================
# Step 6: Approval flow
# =============================================================================


class TestStep6ApprovalFlow:
    """Test that an approval request can be created and approved."""

    async def test_create_approval_request(self):
        from aragora.rbac.approvals import ApprovalWorkflow, ApprovalStatus

        workflow = ApprovalWorkflow()

        request = await workflow.request_access(
            requester_id="user-e2e-001",
            permission="debates:delete",
            resource_type="debates",
            resource_id="debate-e2e-005",
            justification="Cleaning up test data",
            approvers=["admin-001"],
            required_approvals=1,
        )

        assert request.id.startswith("req-")
        assert request.status == ApprovalStatus.PENDING
        assert request.requester_id == "user-e2e-001"
        assert request.permission == "debates:delete"

    async def test_approve_request(self):
        from aragora.rbac.approvals import ApprovalWorkflow, ApprovalStatus

        workflow = ApprovalWorkflow()

        request = await workflow.request_access(
            requester_id="user-e2e-002",
            permission="receipts:read",
            resource_type="receipts",
            justification="Need audit trail access",
            approvers=["admin-001", "admin-002"],
            required_approvals=1,
        )

        updated = await workflow.approve(
            approver_id="admin-001",
            request_id=request.id,
            comment="Approved for audit purposes",
        )

        assert updated.status == ApprovalStatus.APPROVED
        assert updated.approval_count == 1
        assert updated.is_approved

    async def test_reject_prevents_approval(self):
        from aragora.rbac.approvals import ApprovalWorkflow, ApprovalStatus

        workflow = ApprovalWorkflow()

        request = await workflow.request_access(
            requester_id="user-e2e-003",
            permission="debates:write",
            resource_type="debates",
            justification="Want to modify debates",
            approvers=["admin-001"],
            required_approvals=1,
        )

        # Reject instead of approve
        updated = await workflow.reject(
            approver_id="admin-001",
            request_id=request.id,
            reason="Not authorized for this action",
        )

        assert updated.status == ApprovalStatus.REJECTED


# =============================================================================
# Full pipeline integration (data flows step to step)
# =============================================================================


class TestFullPipelineIntegration:
    """End-to-end test: debate -> receipt -> coordinator -> KM -> search."""

    async def test_debate_to_km_full_pipeline(
        self, consensus_agents, simple_env, quick_protocol, mock_mound
    ):
        # Step 1: Run debate
        debate_result = await _run_debate(
            consensus_agents, simple_env, quick_protocol
        )
        assert debate_result is not None
        assert debate_result.rounds_completed >= 1

        # Step 2: Generate receipt from debate result
        receipt = DecisionReceipt.from_debate_result(debate_result)
        assert receipt.artifact_hash
        assert receipt.verify_integrity()
        assert receipt.consensus_proof is not None

        # Step 3: Run post-debate coordinator
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_persist_receipt=True,
            auto_outcome_feedback=False,
            auto_queue_improvement=False,
            auto_trigger_canvas=False,
            auto_execution_bridge=False,
            auto_llm_judge=False,
        )
        coordinator = PostDebateCoordinator(config=config)

        mock_adapter = ReceiptAdapter(mound=mock_mound)
        with patch(
            "aragora.knowledge.mound.adapters.receipt_adapter.get_receipt_adapter",
            return_value=mock_adapter,
        ):
            post_result = coordinator.run(
                debate_id=debate_result.debate_id,
                debate_result=debate_result,
                confidence=debate_result.confidence,
                task=debate_result.task,
            )

        assert post_result.receipt_persisted is True

        # Step 4: Verify persisted in KM
        ingestion = mock_adapter.get_ingestion_result(debate_result.debate_id)
        assert ingestion is not None
        assert ingestion.claims_ingested >= 1

        # Step 5: Search KM for the persisted receipt
        results = await mock_adapter.find_related_decisions("rate limiter")
        assert len(results) >= 1, "Should find the debate receipt via KM search"


# =============================================================================
# Failure paths
# =============================================================================


class TestFailurePaths:
    """Test graceful degradation and error handling."""

    async def test_no_consensus_still_generates_receipt(
        self, split_agents, simple_env, quick_protocol
    ):
        """Debate without consensus should still produce a valid receipt."""
        result = await _run_debate(split_agents, simple_env, quick_protocol)

        receipt = DecisionReceipt.from_debate_result(result)
        assert receipt.receipt_id
        assert receipt.artifact_hash
        assert receipt.verify_integrity()
        # Without high confidence consensus, verdict should be CONDITIONAL or FAIL
        assert receipt.verdict in ("CONDITIONAL", "FAIL")

    def test_km_unavailable_graceful_degradation(self):
        """ReceiptAdapter should degrade gracefully when KM is None."""
        adapter = ReceiptAdapter(mound=None)

        success = adapter.ingest({
            "debate_id": "fail-001",
            "task": "Test",
            "confidence": 0.5,
            "consensus_reached": False,
            "final_answer": "N/A",
            "participants": [],
        })

        # Should still track locally even without mound
        result = adapter.get_ingestion_result("fail-001")
        assert result is not None

    async def test_km_unavailable_async_ingest_reports_error(self):
        """Async ingest with no mound should report an error."""
        adapter = ReceiptAdapter(mound=None)

        receipt = DecisionReceipt(
            receipt_id="rcpt-fail-002",
            gauntlet_id="fail-002",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Test failure",
            input_hash=hashlib.sha256(b"fail").hexdigest(),
            risk_summary={"critical": 0, "high": 0, "medium": 0, "low": 0, "total": 0},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.5,
            robustness_score=0.5,
        )

        result = await adapter.ingest_receipt(receipt)

        assert not result.success
        assert len(result.errors) > 0
        assert "Knowledge Mound not configured" in result.errors[0]

    def test_receipt_missing_required_fields_raises(self):
        """DecisionReceipt should validate required fields."""
        # receipt_id, gauntlet_id, timestamp, input_summary, input_hash are required
        # Dataclass will raise TypeError if any required positional field is missing
        with pytest.raises(TypeError):
            DecisionReceipt(
                receipt_id="partial",
                # Missing gauntlet_id, timestamp, input_summary, input_hash, etc.
            )

    def test_coordinator_persists_receipt_failure_no_crash(self):
        """Coordinator should not crash when receipt persistence fails."""
        debate_result = DebateResult(
            debate_id="fail-003",
            task="Failure test",
            final_answer="Should not crash",
            confidence=0.9,
            consensus_reached=True,
            rounds_used=1,
            rounds_completed=1,
        )

        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_persist_receipt=True,
            auto_outcome_feedback=False,
            auto_queue_improvement=False,
            auto_trigger_canvas=False,
            auto_execution_bridge=False,
            auto_llm_judge=False,
        )
        coordinator = PostDebateCoordinator(config=config)

        # Patch adapter to raise an error the coordinator catches
        mock_adapter = MagicMock()
        mock_adapter.ingest.side_effect = OSError("KM connection lost")

        with patch(
            "aragora.knowledge.mound.adapters.receipt_adapter.get_receipt_adapter",
            return_value=mock_adapter,
        ):
            result = coordinator.run(
                debate_id="fail-003",
                debate_result=debate_result,
                confidence=0.9,
                task="Failure test",
            )

        # Should complete without raising
        assert isinstance(result, PostDebateResult)
        assert result.receipt_persisted is False

    def test_receipt_from_debate_result_no_consensus(self):
        """Receipt from a no-consensus debate should have verdict FAIL."""
        result = DebateResult(
            debate_id="no-consensus-001",
            task="Contentious topic",
            final_answer="",
            confidence=0.3,
            consensus_reached=False,
            rounds_used=2,
            rounds_completed=2,
            participants=["a", "b"],
            dissenting_views=["Agent A disagrees with B", "Agent B disagrees with A"],
        )

        receipt = DecisionReceipt.from_debate_result(result)

        assert receipt.verdict == "FAIL"
        assert receipt.consensus_proof is not None
        assert receipt.consensus_proof.reached is False
        assert len(receipt.dissenting_views) == 2

    async def test_mound_store_failure_during_async_ingest(self):
        """If mound.store raises, adapter should capture the error."""
        broken_mound = MagicMock()
        broken_mound.store = AsyncMock(side_effect=RuntimeError("DB connection lost"))

        adapter = ReceiptAdapter(mound=broken_mound)
        receipt = DecisionReceipt(
            receipt_id="rcpt-broken-001",
            gauntlet_id="broken-001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Broken mound test",
            input_hash=hashlib.sha256(b"broken").hexdigest(),
            risk_summary={"critical": 0, "high": 0, "medium": 0, "low": 0, "total": 0},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.7,
            robustness_score=0.6,
        )

        result = await adapter.ingest_receipt(receipt)

        # Should capture the error rather than crash
        assert len(result.errors) > 0
