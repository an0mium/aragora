"""
Tests for Provenance API handlers.

Tests cover:
- GET /api/debates/{debate_id}/provenance - Full provenance graph
- GET /api/debates/{debate_id}/provenance/timeline - Timeline view
- GET /api/debates/{debate_id}/provenance/verify - Chain verification
- GET /api/debates/{debate_id}/provenance/export - Compliance export
- GET /api/debates/{debate_id}/claims/{claim_id}/provenance - Claim provenance
- GET /api/debates/{debate_id}/provenance/contributions - Agent contributions
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from aragora.reasoning.provenance import (
    ProvenanceManager,
    SourceType,
    TransformationType,
)
from aragora.server.handlers.features.provenance import (
    get_provenance_manager,
    register_provenance_manager,
    handle_get_debate_provenance,
    handle_get_provenance_timeline,
    handle_verify_provenance_chain,
    handle_get_claim_provenance,
    handle_get_agent_contributions,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def sample_manager():
    """Create a sample ProvenanceManager with data."""
    manager = ProvenanceManager(debate_id="debate-001")

    # Record evidence from different sources
    evidence1 = manager.record_evidence(
        content="The contract clause requires 30-day notice for termination.",
        source_type=SourceType.DOCUMENT,
        source_id="contract.pdf",
        metadata={"round_number": 1},
    )

    evidence2 = manager.record_evidence(
        content="Based on my analysis, the 30-day clause is standard practice.",
        source_type=SourceType.AGENT_GENERATED,
        source_id="claude",
        transformation=TransformationType.SUMMARIZED,
        metadata={"round_number": 1, "position": "supports"},
    )

    evidence3 = manager.record_evidence(
        content="Industry benchmarks show 14-day notice is more common.",
        source_type=SourceType.WEB_SEARCH,
        source_id="industry-report",
        metadata={"round_number": 2},
    )

    agent_claim = manager.record_evidence(
        content="The 30-day notice period is appropriate given the complexity.",
        source_type=SourceType.AGENT_GENERATED,
        source_id="gemini",
        transformation=TransformationType.AGGREGATED,
        parent_ids=[evidence1.id, evidence2.id],
        metadata={"round_number": 2, "position": "supports"},
    )

    # Create citations
    manager.cite_evidence(
        claim_id=agent_claim.id,
        evidence_id=evidence1.id,
        relevance=0.95,
        support_type="supports",
        citation_text="30-day notice clause",
    )

    manager.cite_evidence(
        claim_id=agent_claim.id,
        evidence_id=evidence3.id,
        relevance=0.6,
        support_type="contradicts",
        citation_text="14-day industry standard",
    )

    return manager


@pytest.fixture
def registered_manager(sample_manager):
    """Register a sample manager and return it."""
    register_provenance_manager(sample_manager.debate_id, sample_manager)
    yield sample_manager

    # Clean up
    from aragora.server.handlers.features.provenance import _provenance_managers

    _provenance_managers.pop(sample_manager.debate_id, None)


# ===========================================================================
# Manager Registration Tests
# ===========================================================================


class TestManagerRegistration:
    """Tests for manager get/register operations."""

    def test_get_creates_new_manager(self):
        """Test that get_provenance_manager creates a new manager if none exists."""
        from aragora.server.handlers.features.provenance import _provenance_managers

        debate_id = "new-debate-001"

        # Ensure clean state
        _provenance_managers.pop(debate_id, None)

        manager = get_provenance_manager(debate_id)
        assert manager is not None
        assert manager.debate_id == debate_id
        assert len(manager.chain.records) == 0

        # Clean up
        _provenance_managers.pop(debate_id, None)

    def test_register_manager(self, sample_manager):
        """Test registering an external manager."""
        from aragora.server.handlers.features.provenance import _provenance_managers

        debate_id = sample_manager.debate_id

        # Clean state
        _provenance_managers.pop(debate_id, None)

        register_provenance_manager(debate_id, sample_manager)

        retrieved = get_provenance_manager(debate_id)
        assert retrieved is sample_manager
        assert len(retrieved.chain.records) == len(sample_manager.chain.records)

        # Clean up
        _provenance_managers.pop(debate_id, None)


# ===========================================================================
# Handler Tests
# ===========================================================================


class TestGetDebateProvenance:
    """Tests for handle_get_debate_provenance."""

    @pytest.mark.asyncio
    async def test_returns_provenance_graph(self, registered_manager):
        """Test that the handler returns a valid provenance graph."""
        result = await handle_get_debate_provenance(registered_manager.debate_id)

        assert result.status_code == 200
        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body

        assert body["debate_id"] == registered_manager.debate_id
        assert "nodes" in body
        assert "edges" in body
        assert "metadata" in body

        # Check node count matches records
        assert len(body["nodes"]) == len(registered_manager.chain.records)

    @pytest.mark.asyncio
    async def test_includes_verification_status(self, registered_manager):
        """Test that the graph includes verification status."""
        result = await handle_get_debate_provenance(registered_manager.debate_id)

        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body

        assert "verified" in body["metadata"]
        assert "verification_errors" in body["metadata"]
        assert "genesis_hash" in body["metadata"]

    @pytest.mark.asyncio
    async def test_empty_manager(self):
        """Test handling an empty manager."""
        from aragora.server.handlers.features.provenance import _provenance_managers

        debate_id = "empty-debate"
        _provenance_managers.pop(debate_id, None)

        result = await handle_get_debate_provenance(debate_id)

        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body

        assert body["debate_id"] == debate_id
        assert len(body["nodes"]) == 0
        assert len(body["edges"]) == 0

        _provenance_managers.pop(debate_id, None)


class TestGetProvenanceTimeline:
    """Tests for handle_get_provenance_timeline."""

    @pytest.mark.asyncio
    async def test_returns_timeline(self, registered_manager):
        """Test that the handler returns a timeline view."""
        result = await handle_get_provenance_timeline(registered_manager.debate_id)

        assert result.status_code == 200
        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body

        assert body["debate_id"] == registered_manager.debate_id
        assert "rounds" in body
        assert "agent_positions" in body
        assert "total_records" in body

    @pytest.mark.asyncio
    async def test_filter_by_round(self, registered_manager):
        """Test filtering timeline by round number."""
        result = await handle_get_provenance_timeline(
            registered_manager.debate_id,
            round_number=1,
        )

        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body

        # Should only have round 1 records
        for round_data in body["rounds"]:
            assert round_data["round"] == 1


class TestVerifyProvenanceChain:
    """Tests for handle_verify_provenance_chain."""

    @pytest.mark.asyncio
    async def test_verify_valid_chain(self, registered_manager):
        """Test verifying a valid chain."""
        result = await handle_verify_provenance_chain(registered_manager.debate_id)

        assert result.status_code == 200
        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body

        assert body["debate_id"] == registered_manager.debate_id
        assert body["chain_valid"] is True
        assert body["content_valid"] is True
        assert "verified_at" in body
        assert body["record_count"] == len(registered_manager.chain.records)

    @pytest.mark.asyncio
    async def test_verify_empty_chain(self):
        """Test verifying an empty chain."""
        from aragora.server.handlers.features.provenance import _provenance_managers

        debate_id = "verify-empty"
        _provenance_managers.pop(debate_id, None)

        result = await handle_verify_provenance_chain(debate_id)

        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body

        assert body["chain_valid"] is True  # Empty chain is technically valid
        assert body["record_count"] == 0

        _provenance_managers.pop(debate_id, None)


class TestGetClaimProvenance:
    """Tests for handle_get_claim_provenance."""

    @pytest.mark.asyncio
    async def test_get_claim_with_evidence(self, registered_manager):
        """Test getting provenance for a claim with evidence."""
        # Get the claim ID (the aggregated record)
        claim_id = registered_manager.chain.records[-1].id

        result = await handle_get_claim_provenance(
            registered_manager.debate_id,
            claim_id,
        )

        assert result.status_code == 200
        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body

        assert body["debate_id"] == registered_manager.debate_id
        assert body["claim_id"] == claim_id
        assert "supporting_evidence" in body
        assert "contradicting_evidence" in body
        assert "verification_status" in body

    @pytest.mark.asyncio
    async def test_get_nonexistent_claim(self, registered_manager):
        """Test getting provenance for a nonexistent claim."""
        result = await handle_get_claim_provenance(
            registered_manager.debate_id,
            "nonexistent-claim",
        )

        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body

        assert body["claim_id"] == "nonexistent-claim"
        assert body["claim_text"] == ""
        assert len(body["supporting_evidence"]) == 0


class TestGetAgentContributions:
    """Tests for handle_get_agent_contributions."""

    @pytest.mark.asyncio
    async def test_get_all_agent_contributions(self, registered_manager):
        """Test getting contributions from all agents."""
        result = await handle_get_agent_contributions(registered_manager.debate_id)

        assert result.status_code == 200
        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body

        assert body["debate_id"] == registered_manager.debate_id
        assert "contributions" in body
        assert "summary" in body
        assert body["summary"]["unique_agents"] >= 1

    @pytest.mark.asyncio
    async def test_get_specific_agent_contributions(self, registered_manager):
        """Test getting contributions from a specific agent."""
        result = await handle_get_agent_contributions(
            registered_manager.debate_id,
            agent_id="claude",
        )

        body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body

        assert body["agent_id"] == "claude"

        # Should only have contributions from claude
        for contrib in body["contributions"]:
            assert contrib["agent_id"] == "claude"


# ===========================================================================
# Integration Tests with Store
# ===========================================================================


class TestStoreIntegration:
    """Tests for integration with ProvenanceStore."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from aragora.storage.provenance_store import ProvenanceStore

            store = ProvenanceStore(db_path=Path(tmpdir) / "test.db")
            yield store

    def test_register_manager_saves_to_store(self, temp_store, sample_manager):
        """Test that register_provenance_manager saves to the store."""
        from aragora.server.handlers.features import provenance

        # Patch the global store
        original_store = provenance._provenance_store
        provenance._provenance_store = temp_store

        try:
            # Register the manager (should also save to store)
            register_provenance_manager(sample_manager.debate_id, sample_manager)

            # Verify it was saved to store
            loaded = temp_store.load_manager(sample_manager.debate_id)
            assert loaded is not None
            assert len(loaded.chain.records) == len(sample_manager.chain.records)

        finally:
            provenance._provenance_store = original_store
            provenance._provenance_managers.pop(sample_manager.debate_id, None)

    def test_get_manager_loads_from_store(self, temp_store, sample_manager):
        """Test that get_provenance_manager loads from store."""
        from aragora.server.handlers.features import provenance

        # Save to store first
        temp_store.save_manager(sample_manager)

        # Patch the global store
        original_store = provenance._provenance_store
        provenance._provenance_store = temp_store

        # Clear the in-memory cache
        provenance._provenance_managers.pop(sample_manager.debate_id, None)

        try:
            # Get should load from store
            manager = get_provenance_manager(sample_manager.debate_id)
            assert len(manager.chain.records) == len(sample_manager.chain.records)

        finally:
            provenance._provenance_store = original_store
            provenance._provenance_managers.pop(sample_manager.debate_id, None)
