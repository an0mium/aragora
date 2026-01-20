"""Tests for the BeliefAdapter."""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from aragora.knowledge.mound.adapters.belief_adapter import (
    BeliefAdapter,
    BeliefSearchResult,
    CruxSearchResult,
)
from aragora.knowledge.unified.types import ConfidenceLevel, KnowledgeSource


class TestBeliefSearchResult:
    """Tests for BeliefSearchResult dataclass."""

    def test_basic_creation(self):
        """Create a basic search result."""
        result = BeliefSearchResult(
            belief={"id": "bl_1", "claim_statement": "test"},
            relevance_score=0.8,
        )
        assert result.belief["id"] == "bl_1"
        assert result.relevance_score == 0.8
        assert result.matched_topics == []


class TestCruxSearchResult:
    """Tests for CruxSearchResult dataclass."""

    def test_basic_creation(self):
        """Create a basic crux search result."""
        result = CruxSearchResult(
            crux={"id": "cx_1", "statement": "test crux"},
            relevance_score=0.9,
        )
        assert result.crux["id"] == "cx_1"
        assert result.debate_ids == []


class TestBeliefAdapterInit:
    """Tests for BeliefAdapter initialization."""

    def test_init_without_network(self):
        """Initialize without a network."""
        adapter = BeliefAdapter()
        assert adapter.network is None

    def test_init_with_network(self):
        """Initialize with a network."""
        mock_network = Mock()
        adapter = BeliefAdapter(network=mock_network)
        assert adapter.network is mock_network

    def test_set_network(self):
        """Set network after initialization."""
        adapter = BeliefAdapter()
        mock_network = Mock()
        adapter.set_network(mock_network)
        assert adapter.network is mock_network

    def test_constants(self):
        """Verify adapter constants."""
        assert BeliefAdapter.BELIEF_PREFIX == "bl_"
        assert BeliefAdapter.CRUX_PREFIX == "cx_"
        assert BeliefAdapter.PROVENANCE_PREFIX == "pv_"
        assert BeliefAdapter.MIN_BELIEF_CONFIDENCE == 0.8
        assert BeliefAdapter.MIN_CRUX_SCORE == 0.3


class TestBeliefAdapterStoreConvergedBelief:
    """Tests for store_converged_belief method."""

    def test_store_high_confidence_belief(self):
        """Store a high-confidence belief."""
        adapter = BeliefAdapter()

        mock_node = Mock()
        mock_node.node_id = "bn-0001"
        mock_node.claim_id = "claim_123"
        mock_node.claim_statement = "Test claim"
        mock_node.author = "agent1"
        mock_node.posterior = Mock(p_true=0.9)
        mock_node.prior = Mock(p_true=0.5)
        mock_node.status = Mock(value="converged")
        mock_node.centrality = 0.7
        mock_node.update_count = 5
        mock_node.parent_ids = []
        mock_node.child_ids = []
        mock_node.metadata = {}

        belief_id = adapter.store_converged_belief(mock_node, debate_id="debate_1")

        assert belief_id is not None
        assert belief_id.startswith("bl_")
        assert belief_id in adapter._beliefs

    def test_store_below_threshold(self):
        """Don't store beliefs below confidence threshold."""
        adapter = BeliefAdapter()

        mock_node = Mock()
        mock_node.node_id = "bn-0001"
        mock_node.posterior = Mock(p_true=0.5)  # Below threshold

        belief_id = adapter.store_converged_belief(mock_node)

        assert belief_id is None

    def test_store_updates_indices(self):
        """Verify indices are updated on store."""
        adapter = BeliefAdapter()

        mock_node = Mock()
        mock_node.node_id = "bn-0001"
        mock_node.claim_id = "claim_123"
        mock_node.claim_statement = "Test"
        mock_node.author = "agent1"
        mock_node.posterior = Mock(p_true=0.9)
        mock_node.prior = Mock(p_true=0.5)
        mock_node.status = Mock(value="converged")
        mock_node.centrality = 0.5
        mock_node.update_count = 1
        mock_node.parent_ids = []
        mock_node.child_ids = []
        mock_node.metadata = {}

        adapter.store_converged_belief(mock_node, debate_id="debate_1")

        assert "debate_1" in adapter._debate_beliefs
        assert len(adapter._debate_beliefs["debate_1"]) == 1


class TestBeliefAdapterStoreConvergedBeliefs:
    """Tests for store_converged_beliefs method."""

    def test_store_all_from_network(self):
        """Store all converged beliefs from network."""
        mock_network = Mock()
        mock_network.debate_id = "debate_1"

        mock_node1 = Mock()
        mock_node1.node_id = "bn-0001"
        mock_node1.claim_id = "c1"
        mock_node1.claim_statement = "Claim 1"
        mock_node1.author = "agent1"
        mock_node1.posterior = Mock(p_true=0.9)
        mock_node1.prior = Mock(p_true=0.5)
        mock_node1.status = Mock(value="converged")
        mock_node1.centrality = 0.5
        mock_node1.update_count = 1
        mock_node1.parent_ids = []
        mock_node1.child_ids = []
        mock_node1.metadata = {}

        mock_node2 = Mock()
        mock_node2.node_id = "bn-0002"
        mock_node2.claim_id = "c2"
        mock_node2.claim_statement = "Claim 2"
        mock_node2.author = "agent2"
        mock_node2.posterior = Mock(p_true=0.5)  # Below threshold
        mock_node2.prior = Mock(p_true=0.5)

        mock_network.nodes = {"bn-0001": mock_node1, "bn-0002": mock_node2}

        adapter = BeliefAdapter(network=mock_network)
        stored = adapter.store_converged_beliefs()

        assert len(stored) == 1  # Only high-confidence one

    def test_store_without_network(self):
        """Return empty list when no network set."""
        adapter = BeliefAdapter()
        stored = adapter.store_converged_beliefs()
        assert stored == []


class TestBeliefAdapterStoreCrux:
    """Tests for store_crux method."""

    def test_store_crux_above_threshold(self):
        """Store a crux above score threshold."""
        adapter = BeliefAdapter()

        mock_crux = Mock()
        mock_crux.claim_id = "claim_123"
        mock_crux.statement = "This is a pivotal claim"
        mock_crux.author = "agent1"
        mock_crux.crux_score = 0.7
        mock_crux.influence_score = 0.8
        mock_crux.disagreement_score = 0.6
        mock_crux.uncertainty_score = 0.5
        mock_crux.centrality_score = 0.4
        mock_crux.affected_claims = ["c1", "c2"]
        mock_crux.contesting_agents = ["agent2", "agent3"]
        mock_crux.resolution_impact = 0.9

        crux_id = adapter.store_crux(mock_crux, debate_id="debate_1", topics=["legal"])

        assert crux_id is not None
        assert crux_id.startswith("cx_")
        assert crux_id in adapter._cruxes

    def test_store_crux_below_threshold(self):
        """Don't store crux below score threshold."""
        adapter = BeliefAdapter()

        mock_crux = Mock()
        mock_crux.claim_id = "claim_123"
        mock_crux.crux_score = 0.2  # Below 0.3 threshold

        crux_id = adapter.store_crux(mock_crux)
        assert crux_id is None

    def test_store_crux_updates_topic_index(self):
        """Verify topic index is updated."""
        adapter = BeliefAdapter()

        mock_crux = Mock()
        mock_crux.claim_id = "claim_123"
        mock_crux.statement = "Test"
        mock_crux.author = "agent1"
        mock_crux.crux_score = 0.5
        mock_crux.influence_score = 0.5
        mock_crux.disagreement_score = 0.5
        mock_crux.uncertainty_score = 0.5
        mock_crux.centrality_score = 0.5
        mock_crux.affected_claims = []
        mock_crux.contesting_agents = []
        mock_crux.resolution_impact = 0.5

        adapter.store_crux(mock_crux, topics=["Legal", "Contracts"])

        assert "legal" in adapter._topic_cruxes
        assert "contracts" in adapter._topic_cruxes


class TestBeliefAdapterStoreProvenance:
    """Tests for store_provenance method."""

    def test_store_verified_provenance(self):
        """Store a verified provenance chain."""
        adapter = BeliefAdapter()

        prov_id = adapter.store_provenance(
            chain_id="chain_123",
            source_id="source_456",
            claim_ids=["c1", "c2", "c3"],
            verified=True,
            verification_method="formal_proof",
        )

        assert prov_id is not None
        assert prov_id.startswith("pv_")

    def test_skip_unverified_provenance(self):
        """Don't store unverified provenance."""
        adapter = BeliefAdapter()

        prov_id = adapter.store_provenance(
            chain_id="chain_123",
            source_id="source_456",
            claim_ids=["c1"],
            verified=False,
            verification_method="manual",
        )

        assert prov_id is None


class TestBeliefAdapterSearchSimilarCruxes:
    """Tests for search_similar_cruxes method."""

    def test_search_finds_matching_cruxes(self):
        """Find cruxes matching query."""
        adapter = BeliefAdapter()

        # Add some cruxes
        adapter._cruxes["cx_1"] = {
            "id": "cx_1",
            "statement": "The contract terms are ambiguous",
            "crux_score": 0.7,
        }
        adapter._cruxes["cx_2"] = {
            "id": "cx_2",
            "statement": "Performance metrics are unclear",
            "crux_score": 0.6,
        }

        results = adapter.search_similar_cruxes("contract terms", limit=5)

        assert len(results) >= 1
        assert any("contract" in r["statement"].lower() for r in results)

    def test_search_respects_min_score(self):
        """Respect minimum crux score filter."""
        adapter = BeliefAdapter()

        adapter._cruxes["cx_1"] = {
            "id": "cx_1",
            "statement": "Test crux",
            "crux_score": 0.4,
        }

        results = adapter.search_similar_cruxes("test", min_score=0.5)
        assert len(results) == 0


class TestBeliefAdapterSearchCruxesByTopic:
    """Tests for search_cruxes_by_topic method."""

    def test_search_by_topic(self):
        """Find cruxes by topic."""
        adapter = BeliefAdapter()

        adapter._cruxes["cx_1"] = {"id": "cx_1", "statement": "Test"}
        adapter._topic_cruxes["legal"] = ["cx_1"]

        results = adapter.search_cruxes_by_topic("legal")

        assert len(results) == 1
        assert results[0]["id"] == "cx_1"

    def test_search_empty_topic(self):
        """Return empty for unknown topic."""
        adapter = BeliefAdapter()

        results = adapter.search_cruxes_by_topic("unknown_topic")
        assert results == []


class TestBeliefAdapterGetDebateBeliefs:
    """Tests for get_debate_beliefs method."""

    def test_get_beliefs_for_debate(self):
        """Get all beliefs for a debate."""
        adapter = BeliefAdapter()

        adapter._beliefs["bl_1"] = {"id": "bl_1", "confidence": 0.9}
        adapter._beliefs["bl_2"] = {"id": "bl_2", "confidence": 0.7}
        adapter._debate_beliefs["debate_1"] = ["bl_1", "bl_2"]

        results = adapter.get_debate_beliefs("debate_1")

        assert len(results) == 2

    def test_get_beliefs_with_min_confidence(self):
        """Filter beliefs by minimum confidence."""
        adapter = BeliefAdapter()

        adapter._beliefs["bl_1"] = {"id": "bl_1", "confidence": 0.9}
        adapter._beliefs["bl_2"] = {"id": "bl_2", "confidence": 0.7}
        adapter._debate_beliefs["debate_1"] = ["bl_1", "bl_2"]

        results = adapter.get_debate_beliefs("debate_1", min_confidence=0.8)

        assert len(results) == 1
        assert results[0]["confidence"] == 0.9


class TestBeliefAdapterToKnowledgeItem:
    """Tests for to_knowledge_item method."""

    def test_convert_high_confidence_belief(self):
        """Convert high-confidence belief to knowledge item."""
        adapter = BeliefAdapter()

        belief = {
            "id": "bl_bn-0001",
            "node_id": "bn-0001",
            "claim_statement": "Test claim",
            "author": "agent1",
            "confidence": 0.95,
            "prior_confidence": 0.5,
            "centrality": 0.7,
            "update_count": 5,
            "created_at": "2024-01-01T00:00:00Z",
        }

        item = adapter.to_knowledge_item(belief)

        assert item.id == "bl_bn-0001"
        assert item.content == "Test claim"
        assert item.source == KnowledgeSource.BELIEF
        assert item.confidence == ConfidenceLevel.VERIFIED
        assert item.importance == 0.95

    def test_convert_medium_confidence_belief(self):
        """Convert medium-confidence belief."""
        adapter = BeliefAdapter()

        belief = {
            "id": "bl_1",
            "claim_statement": "Test",
            "confidence": 0.65,
        }

        item = adapter.to_knowledge_item(belief)
        assert item.confidence == ConfidenceLevel.MEDIUM


class TestBeliefAdapterCruxToKnowledgeItem:
    """Tests for crux_to_knowledge_item method."""

    def test_convert_crux(self):
        """Convert crux to knowledge item."""
        adapter = BeliefAdapter()

        crux = {
            "id": "cx_claim_123",
            "claim_id": "claim_123",
            "statement": "Pivotal claim about contract interpretation",
            "author": "agent1",
            "crux_score": 0.8,
            "influence_score": 0.9,
            "disagreement_score": 0.7,
            "resolution_impact": 0.85,
            "topics": ["legal", "contracts"],
            "created_at": "2024-01-01T00:00:00Z",
        }

        item = adapter.crux_to_knowledge_item(crux)

        assert item.id == "cx_claim_123"
        assert "Pivotal claim" in item.content
        assert item.source == KnowledgeSource.BELIEF
        assert item.metadata["is_crux"] is True
        assert item.metadata["crux_score"] == 0.8


class TestBeliefAdapterGetStats:
    """Tests for get_stats method."""

    def test_get_stats(self):
        """Get adapter statistics."""
        adapter = BeliefAdapter()

        adapter._beliefs["bl_1"] = {"id": "bl_1"}
        adapter._cruxes["cx_1"] = {"id": "cx_1"}
        adapter._provenance["pv_1"] = {"id": "pv_1"}
        adapter._debate_beliefs["debate_1"] = ["bl_1"]
        adapter._topic_cruxes["legal"] = ["cx_1"]

        stats = adapter.get_stats()

        assert stats["total_beliefs"] == 1
        assert stats["total_cruxes"] == 1
        assert stats["total_provenance_chains"] == 1
        assert stats["debates_with_beliefs"] == 1
        assert stats["topics_indexed"] == 1
