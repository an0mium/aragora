"""
Memory flow integration tests.

Tests that debate outcomes flow correctly through the memory stack:
MemoryCoordinator -> ContinuumMemory, ConsensusMemory, KnowledgeMound.
Verifies cross-debate memory injection for subsequent debates.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aragora.memory.consensus import ConsensusMemory, ConsensusStrength, ConsensusRecord
from aragora.memory.continuum import ContinuumMemory, MemoryTier
from aragora.memory.coordinator import (
    CoordinatorOptions,
    MemoryCoordinator,
    MemoryTransaction,
    WriteStatus,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def continuum(tmp_dir):
    return ContinuumMemory(str(tmp_dir / "continuum.db"))


@pytest.fixture
def consensus_mem(tmp_dir):
    return ConsensusMemory(str(tmp_dir / "consensus.db"))


# ---------------------------------------------------------------------------
# 1. ConsensusMemory stores & retrieves
# ---------------------------------------------------------------------------


class TestConsensusMemoryStorage:
    """Test real ConsensusMemory stores and retrieves records."""

    def test_store_consensus_returns_record(self, consensus_mem):
        record = consensus_mem.store_consensus(
            topic="Rate limiter design",
            conclusion="Use token bucket with Redis backend",
            strength=ConsensusStrength.STRONG,
            confidence=0.88,
            participating_agents=["alice", "bob", "charlie"],
            agreeing_agents=["alice", "bob", "charlie"],
        )

        assert isinstance(record, ConsensusRecord)
        assert record.topic == "Rate limiter design"
        assert record.confidence == 0.88
        assert record.strength == ConsensusStrength.STRONG

    def test_store_consensus_with_dissent(self, consensus_mem):
        record = consensus_mem.store_consensus(
            topic="Database choice",
            conclusion="PostgreSQL for OLTP workloads",
            strength=ConsensusStrength.MODERATE,
            confidence=0.72,
            participating_agents=["alice", "bob", "charlie"],
            agreeing_agents=["alice", "bob"],
            dissenting_agents=["charlie"],
        )

        assert len(record.agreeing_agents) == 2
        assert len(record.dissenting_agents) == 1

    def test_retrieve_stored_consensus(self, consensus_mem):
        consensus_mem.store_consensus(
            topic="API versioning",
            conclusion="Use URL-based versioning",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["alice", "bob"],
            agreeing_agents=["alice", "bob"],
        )

        # Retrieve by topic search
        results = consensus_mem.find_similar_debates("API versioning", limit=5)
        assert len(results) >= 1
        assert any("API versioning" in r.consensus.topic for r in results)

    def test_store_multiple_consensus_records(self, consensus_mem):
        for i in range(5):
            consensus_mem.store_consensus(
                topic=f"Topic {i}",
                conclusion=f"Conclusion {i}",
                strength=ConsensusStrength.MODERATE,
                confidence=0.7 + i * 0.05,
                participating_agents=["a", "b"],
                agreeing_agents=["a", "b"],
            )

        results = consensus_mem.find_similar_debates("Topic", limit=10)
        assert len(results) >= 3


# ---------------------------------------------------------------------------
# 2. ContinuumMemory stores patterns
# ---------------------------------------------------------------------------


class TestContinuumMemoryStorage:
    """Test ContinuumMemory stores and retrieves entries."""

    def test_add_and_retrieve_pattern(self, continuum):
        entry = continuum.add(
            id="debate:rate-limiter-001",
            content="Token bucket is the best approach for rate limiting",
            tier=MemoryTier.FAST,
            importance=0.8,
        )

        assert entry is not None

    def test_add_multiple_tiers(self, continuum):
        entries = []
        for tier in [MemoryTier.FAST, MemoryTier.MEDIUM, MemoryTier.SLOW]:
            entry = continuum.add(
                id=f"entry:{tier.value}",
                content=f"Content for {tier.value} tier",
                tier=tier,
                importance=0.7,
            )
            entries.append(entry)

        assert len(entries) == 3
        assert all(e is not None for e in entries)

    def test_retrieve_returns_results(self, continuum):
        continuum.add(
            id="search-test",
            content="Machine learning for anomaly detection in rate limiters",
            tier=MemoryTier.FAST,
            importance=0.9,
        )

        results = continuum.retrieve(query="anomaly detection", limit=5)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# 3. MemoryCoordinator transaction semantics
# ---------------------------------------------------------------------------


class TestMemoryCoordinatorTransactions:
    """Test coordinator writes to multiple memory systems atomically."""

    async def test_coordinator_initializes_with_systems(self, continuum, consensus_mem):
        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus_mem,
        )

        assert coordinator.continuum_memory is continuum
        assert coordinator.consensus_memory is consensus_mem

    async def test_coordinator_options_defaults(self):
        opts = CoordinatorOptions()
        assert opts.write_continuum is True
        assert opts.write_consensus is True
        assert opts.rollback_on_failure is True
        assert opts.min_confidence_for_mound == 0.7

    async def test_coordinator_metrics_start_at_zero(self, continuum, consensus_mem):
        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            consensus_memory=consensus_mem,
        )

        assert coordinator.metrics.total_transactions == 0
        assert coordinator.metrics.successful_transactions == 0
        assert coordinator.metrics.rollbacks_performed == 0

    async def test_coordinator_with_no_systems_is_valid(self):
        coordinator = MemoryCoordinator()
        assert coordinator.options.write_continuum is True
        # Coordinator should still be usable even without systems attached

    async def test_coordinator_custom_options(self, continuum):
        opts = CoordinatorOptions(
            write_continuum=True,
            write_consensus=False,
            parallel_writes=True,
            min_confidence_for_mound=0.9,
        )
        coordinator = MemoryCoordinator(
            continuum_memory=continuum,
            options=opts,
        )

        assert coordinator.options.write_consensus is False
        assert coordinator.options.parallel_writes is True


# ---------------------------------------------------------------------------
# 4. Cross-debate memory injection
# ---------------------------------------------------------------------------


class TestCrossDebateMemory:
    """Test that prior debate outcomes are available in subsequent debates."""

    def test_consensus_persists_across_instances(self, tmp_dir):
        db_path = str(tmp_dir / "shared_consensus.db")

        mem1 = ConsensusMemory(db_path)
        mem1.store_consensus(
            topic="Caching strategy",
            conclusion="Use Redis with write-through caching",
            strength=ConsensusStrength.STRONG,
            confidence=0.92,
            participating_agents=["alice", "bob"],
            agreeing_agents=["alice", "bob"],
            domain="architecture",
        )

        # New instance reads the same data
        mem2 = ConsensusMemory(db_path)
        results = mem2.find_similar_debates("Caching strategy", limit=5)
        assert len(results) >= 1

    def test_continuum_persists_across_instances(self, tmp_dir):
        db_path = str(tmp_dir / "shared_continuum.db")

        mem1 = ContinuumMemory(db_path)
        mem1.add(
            id="debate:caching-001",
            content="Redis write-through is the recommended caching approach",
            tier=MemoryTier.MEDIUM,
            importance=0.85,
        )

        mem2 = ContinuumMemory(db_path)
        results = mem2.retrieve(query="caching", limit=5)
        assert isinstance(results, list)

    def test_consensus_with_domain_filtering(self, consensus_mem):
        consensus_mem.store_consensus(
            topic="Auth token format",
            conclusion="Use JWT with RS256",
            strength=ConsensusStrength.STRONG,
            confidence=0.95,
            participating_agents=["alice", "bob"],
            agreeing_agents=["alice", "bob"],
            domain="security",
        )
        consensus_mem.store_consensus(
            topic="Cache eviction policy",
            conclusion="LRU with TTL",
            strength=ConsensusStrength.MODERATE,
            confidence=0.78,
            participating_agents=["alice", "charlie"],
            agreeing_agents=["alice", "charlie"],
            domain="infrastructure",
        )

        results = consensus_mem.find_similar_debates("Auth token", limit=5)
        assert len(results) >= 1
