"""
Production Smoke Test Suite.

A focused set of tests that verify critical Aragora code paths work
correctly. Designed to run post-deploy to catch broken imports,
misconfigured subsystems, or regressions in core logic.

These are unit/integration tests using mocks -- they do NOT require a
running server, database, or API keys.

Run:
    pytest -m smoke tests/smoke/test_smoke.py -v --timeout=30
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core_types import Critique, DebateResult, Message, Vote

pytestmark = [pytest.mark.smoke]


# ============================================================================
# 1. Server Health Endpoint
# ============================================================================


class TestServerHealthEndpoint:
    """Verify the health check handler returns the expected response.

    This exercises the handler code path without starting a real HTTP server.
    """

    def test_health_handler_returns_ok(self):
        """The /api/health endpoint code path returns {"status": "ok"}."""
        from aragora.server.api import DebateAPIHandler

        handler = MagicMock(spec=DebateAPIHandler)
        handler.static_dir = None
        captured: dict = {}

        def fake_send_json(data: dict, **kw) -> None:
            captured.update(data)

        handler._send_json = fake_send_json

        # Call the method directly on the class, passing our mock as self
        DebateAPIHandler._health_check(handler)

        assert captured == {"status": "ok"}

    def test_health_endpoint_path_routing(self):
        """The auth system exempts health paths from authentication."""
        from aragora.server.auth_checks import AuthChecksMixin

        exempt = AuthChecksMixin.AUTH_EXEMPT_PATHS

        # Core health paths must be exempt from auth
        health_paths = {"/healthz", "/api/health", "/api/v1/health"}
        for hp in health_paths:
            assert hp in exempt, f"{hp} not in AUTH_EXEMPT_PATHS"


# ============================================================================
# 2. Debate Creation and Completion
# ============================================================================


class TestDebateCreationAndCompletion:
    """Create an Arena with mock agents, run a 1-round debate, verify completion."""

    @pytest.mark.asyncio
    async def test_arena_creates_and_completes(self, smoke_env, smoke_agents, smoke_protocol):
        """Arena initialises, runs _run_inner (mocked), and returns a DebateResult."""
        from aragora.debate.orchestrator import Arena

        arena = Arena(
            environment=smoke_env,
            agents=smoke_agents,
            protocol=smoke_protocol,
        )

        # Verify initialisation set core attributes
        assert arena.env.task == smoke_env.task
        assert len(arena.agents) >= len(smoke_agents)
        assert arena.protocol.rounds == 1

        # Mock _run_inner to return a synthetic result without hitting any API
        expected_result = DebateResult(
            task=smoke_env.task,
            final_answer="Use a token bucket with sliding window fallback",
            confidence=0.88,
            consensus_reached=True,
            rounds_used=1,
            status="consensus_reached",
            participants=[a.name for a in smoke_agents],
            proposals={a.name: f"Proposal from {a.name}" for a in smoke_agents},
            messages=[
                Message(role="proposer", agent=a.name, content=f"Proposal from {a.name}")
                for a in smoke_agents
            ],
            votes=[
                Vote(
                    agent=smoke_agents[0].name,
                    choice=smoke_agents[1].name,
                    reasoning="Better approach",
                    confidence=0.9,
                ),
            ],
        )
        arena._run_inner = AsyncMock(return_value=expected_result)

        result = await arena.run()

        assert isinstance(result, DebateResult)
        assert result.consensus_reached is True
        assert result.confidence == pytest.approx(0.88)
        assert result.final_answer != ""
        assert result.rounds_used == 1
        assert len(result.participants) == len(smoke_agents)
        assert len(result.messages) >= 1

    @pytest.mark.asyncio
    async def test_arena_timeout_returns_partial_result(self, smoke_env, smoke_agents):
        """When a debate times out, Arena returns a partial DebateResult."""
        import asyncio
        from aragora.debate.protocol import DebateProtocol
        from aragora.debate.orchestrator import Arena

        protocol = DebateProtocol(
            rounds=1,
            consensus="majority",
            timeout_seconds=1,
            use_structured_phases=False,
            convergence_detection=False,
            early_stopping=False,
            enable_trickster=False,
            enable_rhetorical_observer=False,
            enable_calibration=False,
            enable_evolution=False,
            enable_research=False,
            role_rotation=False,
            role_matching=False,
            enable_breakpoints=False,
            enable_evidence_weighting=False,
            verify_claims_during_consensus=False,
            enable_molecule_tracking=False,
            enable_agent_channels=False,
        )

        arena = Arena(
            environment=smoke_env,
            agents=smoke_agents,
            protocol=protocol,
        )

        # Mock _run_inner to sleep longer than timeout
        async def slow_run(correlation_id: str = "") -> DebateResult:
            await asyncio.sleep(10)
            return DebateResult(task=smoke_env.task)

        arena._run_inner = slow_run  # type: ignore[assignment]

        result = await arena.run()

        # Should get a partial result (timeout path)
        assert isinstance(result, DebateResult)
        assert result.task == smoke_env.task


# ============================================================================
# 3. Receipt Generation
# ============================================================================


class TestReceiptGeneration:
    """Verify a debate result produces a valid receipt with SHA-256 hash."""

    def test_decision_receipt_hash_integrity(self):
        """DecisionReceipt auto-computes an artifact_hash and verify_integrity passes."""
        from aragora.gauntlet.receipt_models import (
            ConsensusProof,
            DecisionReceipt,
            ProvenanceRecord,
        )

        receipt = DecisionReceipt(
            receipt_id="smoke-receipt-001",
            gauntlet_id="smoke-gauntlet-001",
            timestamp="2026-02-24T00:00:00Z",
            input_summary="Design a rate limiter",
            input_hash=hashlib.sha256(b"Design a rate limiter").hexdigest(),
            risk_summary={"critical": 0, "high": 0, "medium": 1, "low": 2},
            attacks_attempted=5,
            attacks_successful=0,
            probes_run=10,
            vulnerabilities_found=1,
            verdict="PASS",
            confidence=0.92,
            robustness_score=0.88,
            verdict_reasoning="All critical paths verified",
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.92,
                supporting_agents=["agent-alpha", "agent-beta"],
                dissenting_agents=[],
                method="majority",
            ),
            provenance_chain=[
                ProvenanceRecord(
                    timestamp="2026-02-24T00:00:01Z",
                    event_type="verdict",
                    agent="agent-alpha",
                    description="Final verdict issued",
                ),
            ],
        )

        # artifact_hash should be auto-computed in __post_init__
        assert receipt.artifact_hash != ""
        assert len(receipt.artifact_hash) == 64  # SHA-256 hex digest length

        # verify_integrity should pass on a freshly created receipt
        assert receipt.verify_integrity() is True

    def test_receipt_tamper_detection(self):
        """Modifying receipt fields after creation causes verify_integrity to fail."""
        from aragora.gauntlet.receipt_models import DecisionReceipt

        receipt = DecisionReceipt(
            receipt_id="smoke-receipt-002",
            gauntlet_id="smoke-gauntlet-002",
            timestamp="2026-02-24T00:00:00Z",
            input_summary="Original question",
            input_hash=hashlib.sha256(b"Original question").hexdigest(),
            risk_summary={"critical": 0, "high": 0, "medium": 0, "low": 0},
            attacks_attempted=3,
            attacks_successful=0,
            probes_run=5,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.95,
            robustness_score=0.9,
        )

        original_hash = receipt.artifact_hash
        assert receipt.verify_integrity() is True

        # Tamper with the verdict
        receipt.verdict = "FAIL"
        # artifact_hash is stale now -- verify_integrity should fail
        assert receipt.verify_integrity() is False

    def test_consensus_proof_to_dict(self):
        """ConsensusProof serialises cleanly to a dictionary."""
        from aragora.gauntlet.receipt_models import ConsensusProof

        proof = ConsensusProof(
            reached=True,
            confidence=0.85,
            supporting_agents=["alpha", "beta"],
            dissenting_agents=["gamma"],
            method="supermajority",
        )

        d = proof.to_dict()
        assert d["reached"] is True
        assert d["confidence"] == 0.85
        assert "alpha" in d["supporting_agents"]
        assert "gamma" in d["dissenting_agents"]


# ============================================================================
# 4. Consensus Detection
# ============================================================================


class TestConsensusDetection:
    """Verify consensus can be detected from matching agent positions."""

    def test_consensus_builder_creates_proof(self):
        """ConsensusBuilder produces a ConsensusProof with checksum."""
        from aragora.debate.consensus import (
            ConsensusBuilder,
            ConsensusProof,
            VoteType,
        )

        builder = ConsensusBuilder(debate_id="smoke-debate-001", task="Rate limiter design")

        # Add claims
        claim = builder.add_claim(
            statement="Token bucket is the best approach",
            author="agent-alpha",
            confidence=0.9,
            round_num=1,
        )

        # Add evidence
        builder.add_evidence(
            claim_id=claim.claim_id,
            source="agent-alpha",
            content="Token bucket handles bursty traffic well",
            evidence_type="argument",
            supports=True,
            strength=0.8,
        )

        # Record votes
        builder.record_vote(
            agent="agent-alpha",
            vote=VoteType.AGREE,
            confidence=0.9,
            reasoning="Token bucket is well-proven",
        )
        builder.record_vote(
            agent="agent-beta",
            vote=VoteType.AGREE,
            confidence=0.85,
            reasoning="Concur with the approach",
        )
        builder.record_vote(
            agent="agent-gamma",
            vote=VoteType.DISAGREE,
            confidence=0.6,
            reasoning="Prefer sliding window",
        )

        proof = builder.build(
            final_claim="Use token bucket algorithm",
            confidence=0.87,
            consensus_reached=True,
            reasoning_summary="2 out of 3 agents agreed on token bucket",
            rounds=1,
        )

        assert isinstance(proof, ConsensusProof)
        assert proof.consensus_reached is True
        assert proof.confidence == 0.87
        assert "agent-alpha" in proof.supporting_agents
        assert "agent-beta" in proof.supporting_agents
        assert "agent-gamma" in proof.dissenting_agents
        assert len(proof.claims) == 1
        assert len(proof.evidence_chain) == 1
        assert proof.rounds_to_consensus == 1

        # Checksum should be a non-empty hex string (SHA-256 truncated to 16 chars)
        assert proof.checksum
        assert len(proof.checksum) == 16

    def test_consensus_proof_strong_consensus(self):
        """has_strong_consensus returns True when agreement > 80% and confidence > 0.7."""
        from aragora.debate.consensus import ConsensusProof

        proof = ConsensusProof(
            proof_id="proof-strong",
            debate_id="debate-strong",
            task="Test task",
            final_claim="Agreed answer",
            confidence=0.85,
            consensus_reached=True,
            votes=[],
            supporting_agents=["a", "b", "c", "d", "e"],
            dissenting_agents=[],
            claims=[],
            dissents=[],
            unresolved_tensions=[],
            evidence_chain=[],
            reasoning_summary="All agents agreed",
        )

        assert proof.has_strong_consensus is True
        assert proof.agreement_ratio == 1.0

    def test_partial_consensus_from_debate_result(self):
        """build_partial_consensus extracts sub-topic agreement from a DebateResult."""
        from aragora.debate.consensus import build_partial_consensus

        result = DebateResult(
            debate_id="smoke-partial",
            task="Design a rate limiter",
            final_answer="Use token bucket for rate limiting",
            confidence=0.75,
            consensus_reached=True,
            participants=["alpha", "beta"],
        )

        partial = build_partial_consensus(result)

        assert partial.debate_id == "smoke-partial"
        assert partial.overall_consensus is True
        assert partial.overall_confidence == 0.75


# ============================================================================
# 5. Memory Write and Read
# ============================================================================


class TestMemoryWriteAndRead:
    """Verify ContinuumMemory can store and retrieve data."""

    def test_add_and_retrieve_memory(self, tmp_db_path):
        """ContinuumMemory stores an entry and retrieves it by content match."""
        from aragora.memory.continuum import ContinuumMemory, MemoryTier

        cms = ContinuumMemory(db_path=tmp_db_path)

        # Store a memory
        entry = cms.add(
            id="smoke-memory-001",
            content="Token bucket handles bursty traffic effectively",
            tier=MemoryTier.FAST,
            importance=0.9,
            metadata={"source": "smoke_test"},
        )

        assert entry is not None
        assert entry.id == "smoke-memory-001"
        assert entry.content == "Token bucket handles bursty traffic effectively"
        assert entry.tier == MemoryTier.FAST
        assert entry.importance == 0.9

        # Retrieve by ID (direct get)
        fetched = cms.get("smoke-memory-001")
        assert fetched is not None
        assert fetched.content == "Token bucket handles bursty traffic effectively"

    def test_retrieve_returns_entries_sorted_by_importance(self, tmp_db_path):
        """retrieve() returns entries ordered by importance (descending)."""
        from aragora.memory.continuum import ContinuumMemory, MemoryTier

        cms = ContinuumMemory(db_path=tmp_db_path)

        cms.add("low", "Low importance item", tier=MemoryTier.FAST, importance=0.2)
        cms.add("high", "High importance item", tier=MemoryTier.FAST, importance=0.95)
        cms.add("mid", "Medium importance item", tier=MemoryTier.FAST, importance=0.5)

        results = cms.retrieve(limit=10)

        assert len(results) == 3
        # First result should be the highest importance
        assert results[0].id == "high"
        assert results[0].importance == 0.95

    def test_memory_tier_isolation(self, tmp_db_path):
        """Entries in different tiers are filtered correctly."""
        from aragora.memory.continuum import ContinuumMemory, MemoryTier

        cms = ContinuumMemory(db_path=tmp_db_path)

        cms.add("fast-1", "Fast tier entry", tier=MemoryTier.FAST, importance=0.8)
        cms.add("slow-1", "Slow tier entry", tier=MemoryTier.SLOW, importance=0.8)

        fast_only = cms.retrieve(tiers=[MemoryTier.FAST], limit=10)
        assert all(e.tier == MemoryTier.FAST for e in fast_only)

        slow_only = cms.retrieve(tiers=[MemoryTier.SLOW], limit=10)
        assert all(e.tier == MemoryTier.SLOW for e in slow_only)


# ============================================================================
# 6. Knowledge Mound Ingest and Query
# ============================================================================


class TestKnowledgeMoundIngestAndQuery:
    """Verify KnowledgeMound can ingest and return data.

    Uses the SQLite backend with a temp directory to avoid external dependencies.
    """

    @pytest.mark.asyncio
    async def test_ingest_and_query(self, tmp_path):
        """KnowledgeMound stores knowledge and retrieves it via query."""
        from aragora.knowledge.mound import (
            KnowledgeMound,
            KnowledgeSource,
            MoundBackend,
            MoundConfig,
            IngestionRequest,
        )

        config = MoundConfig(
            backend=MoundBackend.SQLITE,
            sqlite_path=str(tmp_path / "smoke_km.db"),
            enable_staleness_detection=False,
        )
        mound = KnowledgeMound(config=config, workspace_id="smoke-test")
        await mound.initialize()

        try:
            # Ingest a knowledge item
            request = IngestionRequest(
                content="Rate limiters should use token bucket for bursty traffic",
                workspace_id="smoke-test",
                source_type=KnowledgeSource.DEBATE,
                debate_id="smoke-debate-001",
                confidence=0.9,
                node_type="fact",
                topics=["rate-limiting", "architecture"],
            )
            result = await mound.store(request)

            assert result.success is True
            assert result.node_id != ""

            # Query back
            query_result = await mound.query(
                "rate limiter token bucket",
                workspace_id="smoke-test",
                limit=5,
            )

            # Should find at least the item we just ingested
            assert query_result.total_count >= 1
            found_content = [item.content for item in query_result.items]
            assert any("token bucket" in c.lower() for c in found_content)
        finally:
            await mound.close()

    @pytest.mark.asyncio
    async def test_ingest_deduplication(self, tmp_path):
        """Storing the same content twice is handled gracefully."""
        from aragora.knowledge.mound import (
            KnowledgeMound,
            KnowledgeSource,
            MoundBackend,
            MoundConfig,
            IngestionRequest,
        )

        config = MoundConfig(
            backend=MoundBackend.SQLITE,
            sqlite_path=str(tmp_path / "smoke_km_dedup.db"),
            enable_staleness_detection=False,
        )
        mound = KnowledgeMound(config=config, workspace_id="smoke-test")
        await mound.initialize()

        try:
            request = IngestionRequest(
                content="Exactly the same content for dedup test",
                workspace_id="smoke-test",
                source_type=KnowledgeSource.FACT,
                confidence=0.8,
            )

            result1 = await mound.store(request)
            assert result1.success is True

            result2 = await mound.store(request)
            # Second store should succeed (dedup or upsert)
            assert result2.success is True
        finally:
            await mound.close()

    @pytest.mark.asyncio
    async def test_mound_requires_initialization(self):
        """Calling query before initialize raises RuntimeError."""
        from aragora.knowledge.mound import KnowledgeMound, MoundConfig, MoundBackend

        config = MoundConfig(backend=MoundBackend.SQLITE)
        mound = KnowledgeMound(config=config, workspace_id="smoke-test")

        # Should raise because we haven't called initialize()
        with pytest.raises(RuntimeError, match="not initialized"):
            await mound.query("anything")
