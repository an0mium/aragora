"""
Tests for BeliefAdapter - Bridges BeliefNetwork to Knowledge Mound.

Tests cover:
- Adapter initialization and configuration
- Belief node creation and retrieval
- Crux storage and search
- Provenance chain handling
- Evidence linking and propagation calculations
- Reverse flow (KM -> BeliefNetwork)
- Threshold updates from KM patterns
- KM validation and prior computation
- FusionMixin integration
- Edge cases and error handling
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass


# ============================================================================
# Test Fixtures and Mock Classes
# ============================================================================


@dataclass
class MockBeliefDistribution:
    """Mock belief distribution for testing."""

    p_true: float = 0.5
    p_false: float = 0.5
    p_unknown: float = 0.0


@dataclass
class MockBeliefNode:
    """Mock belief node for testing."""

    node_id: str = "node-123"
    claim_id: str = "claim-123"
    claim_statement: str = "Test claim statement"
    author: str = "test-agent"
    prior: MockBeliefDistribution = None
    posterior: MockBeliefDistribution = None
    status: MagicMock = None
    centrality: float = 0.5
    update_count: int = 1
    parent_ids: list = None
    child_ids: list = None
    metadata: dict = None

    def __post_init__(self):
        if self.prior is None:
            self.prior = MockBeliefDistribution(p_true=0.5, p_false=0.5)
        if self.posterior is None:
            self.posterior = MockBeliefDistribution(p_true=0.85, p_false=0.15)
        if self.status is None:
            self.status = MagicMock()
            self.status.value = "converged"
        if self.parent_ids is None:
            self.parent_ids = []
        if self.child_ids is None:
            self.child_ids = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MockCruxClaim:
    """Mock crux claim for testing."""

    claim_id: str = "crux-123"
    statement: str = "This is a pivotal claim"
    author: str = "test-agent"
    crux_score: float = 0.75
    influence_score: float = 0.6
    disagreement_score: float = 0.5
    uncertainty_score: float = 0.4
    centrality_score: float = 0.3
    affected_claims: list = None
    contesting_agents: list = None
    resolution_impact: float = 0.7

    def __post_init__(self):
        if self.affected_claims is None:
            self.affected_claims = ["claim-1", "claim-2"]
        if self.contesting_agents is None:
            self.contesting_agents = ["claude", "gpt"]


# ============================================================================
# Adapter Initialization Tests
# ============================================================================


class TestBeliefAdapterInit:
    """Tests for BeliefAdapter initialization."""

    def test_init_without_network(self):
        """Should initialize without a network."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        assert adapter._network is None
        assert adapter._enable_dual_write is False
        assert adapter._event_callback is None
        assert adapter._beliefs == {}
        assert adapter._cruxes == {}
        assert adapter._provenance == {}

    def test_init_with_network(self):
        """Should initialize with a network."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        mock_network = MagicMock()
        mock_network.debate_id = "debate-123"

        adapter = BeliefAdapter(network=mock_network)

        assert adapter._network is mock_network
        assert adapter.network is mock_network

    def test_init_with_options(self):
        """Should accept optional parameters."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        mock_network = MagicMock()
        callback = MagicMock()

        adapter = BeliefAdapter(
            network=mock_network,
            enable_dual_write=True,
            event_callback=callback,
            enable_resilience=False,
        )

        assert adapter._enable_dual_write is True
        assert adapter._event_callback is callback

    def test_set_network(self):
        """Should allow setting network after init."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        mock_network = MagicMock()

        adapter.set_network(mock_network)

        assert adapter._network is mock_network

    def test_class_constants(self):
        """Should have correct class constants."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        assert BeliefAdapter.BELIEF_PREFIX == "bl_"
        assert BeliefAdapter.CRUX_PREFIX == "cx_"
        assert BeliefAdapter.PROVENANCE_PREFIX == "pv_"
        assert BeliefAdapter.MIN_BELIEF_CONFIDENCE == 0.8
        assert BeliefAdapter.MIN_CRUX_SCORE == 0.3


# ============================================================================
# Event Callback Tests
# ============================================================================


class TestEventCallback:
    """Tests for event callback functionality."""

    def test_set_event_callback(self):
        """Should set event callback."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        callback = MagicMock()

        adapter.set_event_callback(callback)

        assert adapter._event_callback is callback

    def test_emit_event(self):
        """Should emit event via callback."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        callback = MagicMock()
        adapter = BeliefAdapter(event_callback=callback)

        adapter._emit_event("test_event", {"key": "value"})

        callback.assert_called_once_with("test_event", {"key": "value"})

    def test_emit_event_handles_callback_error(self):
        """Should handle callback errors gracefully."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        callback = MagicMock(side_effect=Exception("Callback failed"))
        adapter = BeliefAdapter(event_callback=callback)

        # Should not raise
        adapter._emit_event("test_event", {})

    def test_emit_event_without_callback(self):
        """Should not fail when no callback is set."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        # Should not raise
        adapter._emit_event("test_event", {"data": "test"})


# ============================================================================
# Store Converged Belief Tests
# ============================================================================


class TestStoreConvergedBelief:
    """Tests for store_converged_belief method."""

    def test_store_converged_belief(self):
        """Should store a converged belief node."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        node = MockBeliefNode(
            node_id="node-001",
            claim_statement="AI will transform healthcare",
            posterior=MockBeliefDistribution(p_true=0.9, p_false=0.1),
        )

        belief_id = adapter.store_converged_belief(node, debate_id="debate-123")

        assert belief_id == "bl_node-001"
        assert belief_id in adapter._beliefs
        assert adapter._beliefs[belief_id]["confidence"] == 0.9
        assert adapter._beliefs[belief_id]["debate_id"] == "debate-123"

    def test_store_belief_below_threshold(self):
        """Should reject beliefs below confidence threshold."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        node = MockBeliefNode(
            node_id="node-low",
            posterior=MockBeliefDistribution(p_true=0.5, p_false=0.5),
        )

        belief_id = adapter.store_converged_belief(node)

        assert belief_id is None

    def test_store_belief_high_false_confidence(self):
        """Should store beliefs with high false confidence."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        node = MockBeliefNode(
            node_id="node-false",
            posterior=MockBeliefDistribution(p_true=0.1, p_false=0.9),
        )

        belief_id = adapter.store_converged_belief(node)

        # 1 - 0.1 = 0.9 >= 0.8 threshold
        assert belief_id == "bl_node-false"

    def test_store_belief_uses_network_debate_id(self):
        """Should use network's debate_id if not provided."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        mock_network = MagicMock()
        mock_network.debate_id = "network-debate-456"

        adapter = BeliefAdapter(network=mock_network)
        node = MockBeliefNode(posterior=MockBeliefDistribution(p_true=0.9))

        belief_id = adapter.store_converged_belief(node)

        assert adapter._beliefs[belief_id]["debate_id"] == "network-debate-456"

    def test_store_belief_updates_debate_index(self):
        """Should update debate-to-beliefs index."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        node = MockBeliefNode(
            node_id="node-indexed",
            posterior=MockBeliefDistribution(p_true=0.85),
        )

        belief_id = adapter.store_converged_belief(node, debate_id="debate-idx")

        assert "debate-idx" in adapter._debate_beliefs
        assert belief_id in adapter._debate_beliefs["debate-idx"]

    def test_store_belief_emits_event(self):
        """Should emit belief_converged event."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        callback = MagicMock()
        adapter = BeliefAdapter(event_callback=callback)
        node = MockBeliefNode(
            claim_statement="Important claim",
            posterior=MockBeliefDistribution(p_true=0.9),
        )

        adapter.store_converged_belief(node, debate_id="debate-evt")

        callback.assert_called()
        call_args = callback.call_args[0]
        assert call_args[0] == "belief_converged"
        assert call_args[1]["confidence"] == 0.9


# ============================================================================
# Store Converged Beliefs (Batch) Tests
# ============================================================================


class TestStoreConvergedBeliefs:
    """Tests for store_converged_beliefs batch method."""

    def test_store_converged_beliefs_batch(self):
        """Should store all high-confidence beliefs from network."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        mock_network = MagicMock()
        mock_network.debate_id = "batch-debate"
        mock_network.nodes = {
            "n1": MockBeliefNode(node_id="n1", posterior=MockBeliefDistribution(p_true=0.9)),
            "n2": MockBeliefNode(node_id="n2", posterior=MockBeliefDistribution(p_true=0.5)),
            "n3": MockBeliefNode(node_id="n3", posterior=MockBeliefDistribution(p_true=0.85)),
        }

        adapter = BeliefAdapter(network=mock_network)
        stored_ids = adapter.store_converged_beliefs()

        assert len(stored_ids) == 2
        assert "bl_n1" in stored_ids
        assert "bl_n3" in stored_ids

    def test_store_converged_beliefs_without_network(self):
        """Should return empty list without network."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        stored_ids = adapter.store_converged_beliefs()

        assert stored_ids == []

    def test_store_converged_beliefs_custom_threshold(self):
        """Should respect custom min_confidence threshold."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        mock_network = MagicMock()
        mock_network.debate_id = "custom-thresh"
        mock_network.nodes = {
            "n1": MockBeliefNode(node_id="n1", posterior=MockBeliefDistribution(p_true=0.7)),
            "n2": MockBeliefNode(node_id="n2", posterior=MockBeliefDistribution(p_true=0.65)),
        }

        adapter = BeliefAdapter(network=mock_network)
        stored_ids = adapter.store_converged_beliefs(min_confidence=0.6)

        assert len(stored_ids) == 2


# ============================================================================
# Store Crux Tests
# ============================================================================


class TestStoreCrux:
    """Tests for store_crux method."""

    def test_store_crux(self):
        """Should store a crux claim."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        crux = MockCruxClaim(
            claim_id="crux-001",
            statement="Key pivotal claim",
            crux_score=0.8,
        )

        crux_id = adapter.store_crux(crux, debate_id="debate-crux")

        assert crux_id == "cx_crux-001"
        assert crux_id in adapter._cruxes
        assert adapter._cruxes[crux_id]["crux_score"] == 0.8

    def test_store_crux_below_threshold(self):
        """Should reject cruxes below score threshold."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        crux = MockCruxClaim(crux_score=0.1)

        crux_id = adapter.store_crux(crux)

        assert crux_id is None

    def test_store_crux_with_topics(self):
        """Should index crux by topics."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        crux = MockCruxClaim(claim_id="crux-topics", crux_score=0.5)

        crux_id = adapter.store_crux(crux, topics=["security", "Authentication"])

        assert "security" in adapter._topic_cruxes
        assert "authentication" in adapter._topic_cruxes  # lowercase
        assert crux_id in adapter._topic_cruxes["security"]

    def test_store_crux_updates_debate_index(self):
        """Should update debate-to-cruxes index."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        crux = MockCruxClaim(claim_id="crux-idx", crux_score=0.5)

        crux_id = adapter.store_crux(crux, debate_id="debate-crux-idx")

        assert "debate-crux-idx" in adapter._debate_cruxes
        assert crux_id in adapter._debate_cruxes["debate-crux-idx"]

    def test_store_crux_emits_event(self):
        """Should emit crux_detected event."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        callback = MagicMock()
        adapter = BeliefAdapter(event_callback=callback)
        crux = MockCruxClaim(statement="Pivotal claim", crux_score=0.6)

        adapter.store_crux(crux, topics=["api"])

        callback.assert_called()
        event_type = callback.call_args[0][0]
        assert event_type == "crux_detected"


# ============================================================================
# Store Provenance Tests
# ============================================================================


class TestStoreProvenance:
    """Tests for store_provenance method."""

    def test_store_provenance(self):
        """Should store a verified provenance chain."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        prov_id = adapter.store_provenance(
            chain_id="chain-001",
            source_id="src-001",
            claim_ids=["claim-1", "claim-2"],
            verified=True,
            verification_method="cryptographic",
            debate_id="debate-prov",
        )

        assert prov_id == "pv_chain-001"
        assert prov_id in adapter._provenance
        assert adapter._provenance[prov_id]["verified"] is True

    def test_store_provenance_unverified(self):
        """Should reject unverified provenance chains."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        prov_id = adapter.store_provenance(
            chain_id="chain-unverified",
            source_id="src-002",
            claim_ids=["claim-3"],
            verified=False,
            verification_method="manual",
        )

        assert prov_id is None

    def test_store_provenance_with_metadata(self):
        """Should store provenance with metadata."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        prov_id = adapter.store_provenance(
            chain_id="chain-meta",
            source_id="src-003",
            claim_ids=["claim-4"],
            verified=True,
            verification_method="hash",
            metadata={"algorithm": "SHA-256"},
        )

        assert adapter._provenance[prov_id]["metadata"]["algorithm"] == "SHA-256"


# ============================================================================
# Get Belief/Crux Tests
# ============================================================================


class TestGetBeliefAndCrux:
    """Tests for get_belief and get_crux methods."""

    def test_get_belief_by_id(self):
        """Should get belief by ID."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._beliefs["bl_test-id"] = {"id": "bl_test-id", "confidence": 0.9}

        result = adapter.get_belief("bl_test-id")

        assert result is not None
        assert result["confidence"] == 0.9

    def test_get_belief_with_prefix_added(self):
        """Should add prefix if not present."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._beliefs["bl_node-x"] = {"id": "bl_node-x"}

        result = adapter.get_belief("node-x")

        assert result is not None

    def test_get_belief_not_found(self):
        """Should return None for missing belief."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        result = adapter.get_belief("nonexistent")

        assert result is None

    def test_get_crux_by_id(self):
        """Should get crux by ID."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._cruxes["cx_crux-test"] = {"id": "cx_crux-test", "crux_score": 0.7}

        result = adapter.get_crux("cx_crux-test")

        assert result is not None
        assert result["crux_score"] == 0.7

    def test_get_crux_with_prefix_added(self):
        """Should add prefix if not present."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._cruxes["cx_crux-y"] = {"id": "cx_crux-y"}

        result = adapter.get_crux("crux-y")

        assert result is not None


# ============================================================================
# Search Similar Cruxes Tests
# ============================================================================


class TestSearchSimilarCruxes:
    """Tests for search_similar_cruxes method."""

    def test_search_similar_cruxes(self):
        """Should find cruxes matching query keywords."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._cruxes = {
            "cx_1": {
                "statement": "Security is critical for API design",
                "crux_score": 0.8,
            },
            "cx_2": {
                "statement": "Performance matters most",
                "crux_score": 0.6,
            },
            "cx_3": {
                "statement": "API security best practices",
                "crux_score": 0.7,
            },
        }

        results = adapter.search_similar_cruxes("security API", limit=10)

        assert len(results) == 2
        # Both should match "security" or "API"
        assert any("security" in r["statement"].lower() for r in results)

    def test_search_similar_cruxes_with_min_score(self):
        """Should filter by minimum crux score."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._cruxes = {
            "cx_1": {"statement": "test claim", "crux_score": 0.3},
            "cx_2": {"statement": "test claim", "crux_score": 0.7},
        }

        results = adapter.search_similar_cruxes("test claim", min_score=0.5)

        assert len(results) == 1
        assert results[0]["crux_score"] == 0.7

    def test_search_similar_cruxes_limit(self):
        """Should respect limit parameter."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._cruxes = {
            f"cx_{i}": {"statement": "test statement", "crux_score": 0.5} for i in range(10)
        }

        results = adapter.search_similar_cruxes("test", limit=3)

        assert len(results) == 3

    def test_search_similar_cruxes_empty(self):
        """Should return empty list for no matches."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._cruxes = {"cx_1": {"statement": "completely different", "crux_score": 0.8}}

        results = adapter.search_similar_cruxes("xyz123")

        assert results == []


# ============================================================================
# Search Cruxes By Topic Tests
# ============================================================================


class TestSearchCruxesByTopic:
    """Tests for search_cruxes_by_topic method."""

    def test_search_cruxes_by_topic(self):
        """Should find cruxes by indexed topic."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._cruxes = {
            "cx_1": {"id": "cx_1", "statement": "Security crux"},
            "cx_2": {"id": "cx_2", "statement": "Other crux"},
        }
        adapter._topic_cruxes = {"security": ["cx_1"]}

        results = adapter.search_cruxes_by_topic("security")

        assert len(results) == 1
        assert results[0]["id"] == "cx_1"

    def test_search_cruxes_by_topic_case_insensitive(self):
        """Should be case insensitive."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._cruxes = {"cx_1": {"id": "cx_1"}}
        adapter._topic_cruxes = {"authentication": ["cx_1"]}

        results = adapter.search_cruxes_by_topic("AUTHENTICATION")

        assert len(results) == 1

    def test_search_cruxes_by_topic_not_found(self):
        """Should return empty for unknown topic."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        results = adapter.search_cruxes_by_topic("unknown-topic")

        assert results == []


# ============================================================================
# Get Debate Beliefs/Cruxes Tests
# ============================================================================


class TestGetDebateData:
    """Tests for get_debate_beliefs and get_debate_cruxes methods."""

    def test_get_debate_beliefs(self):
        """Should get all beliefs for a debate."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._beliefs = {
            "bl_1": {"id": "bl_1", "confidence": 0.9},
            "bl_2": {"id": "bl_2", "confidence": 0.7},
        }
        adapter._debate_beliefs = {"debate-123": ["bl_1", "bl_2"]}

        results = adapter.get_debate_beliefs("debate-123")

        assert len(results) == 2

    def test_get_debate_beliefs_with_min_confidence(self):
        """Should filter by minimum confidence."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._beliefs = {
            "bl_1": {"id": "bl_1", "confidence": 0.9},
            "bl_2": {"id": "bl_2", "confidence": 0.5},
        }
        adapter._debate_beliefs = {"debate-123": ["bl_1", "bl_2"]}

        results = adapter.get_debate_beliefs("debate-123", min_confidence=0.8)

        assert len(results) == 1

    def test_get_debate_cruxes(self):
        """Should get all cruxes for a debate."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._cruxes = {
            "cx_1": {"id": "cx_1", "crux_score": 0.8},
            "cx_2": {"id": "cx_2", "crux_score": 0.5},
        }
        adapter._debate_cruxes = {"debate-456": ["cx_1", "cx_2"]}

        results = adapter.get_debate_cruxes("debate-456")

        assert len(results) == 2

    def test_get_debate_cruxes_with_min_score(self):
        """Should filter by minimum crux score."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._cruxes = {
            "cx_1": {"id": "cx_1", "crux_score": 0.8},
            "cx_2": {"id": "cx_2", "crux_score": 0.3},
        }
        adapter._debate_cruxes = {"debate-456": ["cx_1", "cx_2"]}

        results = adapter.get_debate_cruxes("debate-456", min_score=0.5)

        assert len(results) == 1


# ============================================================================
# Knowledge Item Conversion Tests
# ============================================================================


class TestKnowledgeItemConversion:
    """Tests for to_knowledge_item and crux_to_knowledge_item methods."""

    def test_to_knowledge_item(self):
        """Should convert belief to KnowledgeItem."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        belief = {
            "id": "bl_test",
            "claim_statement": "Test claim",
            "node_id": "node-123",
            "author": "claude",
            "confidence": 0.9,
            "centrality": 0.5,
            "prior_confidence": 0.6,
            "update_count": 2,
            "debate_id": "debate-km",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        item = adapter.to_knowledge_item(belief)

        assert item.id == "bl_test"
        assert item.content == "Test claim"
        assert item.metadata["author"] == "claude"

    def test_to_knowledge_item_confidence_levels(self):
        """Should map confidence to correct levels."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter
        from aragora.knowledge.unified.types import ConfidenceLevel

        adapter = BeliefAdapter()

        # Test different confidence levels
        for conf, expected in [
            (0.95, ConfidenceLevel.VERIFIED),
            (0.85, ConfidenceLevel.HIGH),
            (0.65, ConfidenceLevel.MEDIUM),
            (0.45, ConfidenceLevel.LOW),
            (0.25, ConfidenceLevel.UNVERIFIED),
        ]:
            belief = {
                "id": f"bl_{conf}",
                "claim_statement": "Test",
                "confidence": conf,
            }
            item = adapter.to_knowledge_item(belief)
            assert item.confidence == expected

    def test_crux_to_knowledge_item(self):
        """Should convert crux to KnowledgeItem."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        crux = {
            "id": "cx_test",
            "claim_id": "claim-crux",
            "statement": "Pivotal claim statement",
            "author": "gpt",
            "crux_score": 0.75,
            "influence_score": 0.6,
            "disagreement_score": 0.5,
            "resolution_impact": 0.7,
            "topics": ["security"],
            "debate_id": "debate-crux-km",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        item = adapter.crux_to_knowledge_item(crux)

        assert item.id == "cx_test"
        assert item.content == "Pivotal claim statement"
        assert item.metadata["is_crux"] is True
        assert item.metadata["crux_score"] == 0.75


# ============================================================================
# Statistics Tests
# ============================================================================


class TestGetStats:
    """Tests for get_stats method."""

    def test_get_stats(self):
        """Should return comprehensive statistics."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._beliefs = {"bl_1": {}, "bl_2": {}}
        adapter._cruxes = {"cx_1": {}}
        adapter._provenance = {"pv_1": {}, "pv_2": {}, "pv_3": {}}
        adapter._debate_beliefs = {"d1": [], "d2": []}
        adapter._debate_cruxes = {"d3": []}
        adapter._topic_cruxes = {"t1": [], "t2": []}

        stats = adapter.get_stats()

        assert stats["total_beliefs"] == 2
        assert stats["total_cruxes"] == 1
        assert stats["total_provenance_chains"] == 3
        assert stats["debates_with_beliefs"] == 2
        assert stats["debates_with_cruxes"] == 1
        assert stats["topics_indexed"] == 2


# ============================================================================
# Reverse Flow Tests (KM -> BeliefNetwork)
# ============================================================================


class TestRecordOutcome:
    """Tests for record_outcome method."""

    def test_record_outcome(self):
        """Should record outcome for a belief."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        adapter.record_outcome(
            belief_id="bl_test",
            debate_id="debate-outcome",
            was_successful=True,
            confidence=0.8,
        )

        assert "bl_test" in adapter._outcome_history
        assert len(adapter._outcome_history["bl_test"]) == 1
        assert adapter._outcome_history["bl_test"][0]["was_successful"] is True

    def test_record_multiple_outcomes(self):
        """Should accumulate multiple outcomes."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        adapter.record_outcome("bl_multi", "d1", True, 0.9)
        adapter.record_outcome("bl_multi", "d2", False, 0.7)
        adapter.record_outcome("bl_multi", "d3", True, 0.8)

        assert len(adapter._outcome_history["bl_multi"]) == 3


class TestUpdateBeliefThresholdsFromKM:
    """Tests for update_belief_thresholds_from_km method."""

    @pytest.mark.asyncio
    async def test_update_thresholds_from_km(self):
        """Should analyze KM patterns and update thresholds."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        # Create KM items with outcome data
        km_items = [
            {"metadata": {"confidence": 0.85, "outcome_success": True}} for _ in range(50)
        ] + [{"metadata": {"confidence": 0.65, "outcome_success": False}} for _ in range(30)]

        result = await adapter.update_belief_thresholds_from_km(km_items)

        assert result.patterns_analyzed == 80
        assert result.old_belief_confidence_threshold == 0.8

    @pytest.mark.asyncio
    async def test_threshold_update_low_confidence(self):
        """Should not update thresholds with low confidence."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        # Too few items for high confidence
        km_items = [{"metadata": {"confidence": 0.9, "outcome_success": True}} for _ in range(5)]

        result = await adapter.update_belief_thresholds_from_km(km_items, min_confidence=0.8)

        # Should not change threshold with low sample size
        assert result.confidence < 0.8


class TestGetKMValidatedPriors:
    """Tests for get_km_validated_priors method."""

    @pytest.mark.asyncio
    async def test_get_km_validated_priors(self):
        """Should compute priors from KM items."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        km_items = [
            {
                "metadata": {
                    "claim_type": "factual",
                    "outcome_success": True,
                    "confidence": 0.9,
                    "debate_id": "d1",
                }
            },
            {
                "metadata": {
                    "claim_type": "factual",
                    "outcome_success": True,
                    "confidence": 0.8,
                    "debate_id": "d2",
                }
            },
            {
                "metadata": {
                    "claim_type": "factual",
                    "outcome_success": False,
                    "confidence": 0.7,
                    "debate_id": "d3",
                }
            },
        ]

        result = await adapter.get_km_validated_priors("factual", km_items)

        assert result.claim_type == "factual"
        assert result.sample_count == 3
        # 2 successes out of 3 = 0.667 prior
        assert 0.6 <= result.recommended_prior <= 0.7

    @pytest.mark.asyncio
    async def test_get_priors_caches_result(self):
        """Should cache prior recommendations."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        km_items = [{"metadata": {"claim_type": "opinion", "outcome_success": True}}]

        await adapter.get_km_validated_priors("opinion", km_items)

        # Second call without items should use cache
        cached = await adapter.get_km_validated_priors("opinion")
        assert cached.claim_type == "opinion"

    @pytest.mark.asyncio
    async def test_get_priors_unknown_type(self):
        """Should return default prior for unknown types."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        result = await adapter.get_km_validated_priors("unknown_type", [])

        assert result.recommended_prior == 0.5
        assert result.sample_count == 0


class TestValidateBeliefFromKM:
    """Tests for validate_belief_from_km method."""

    @pytest.mark.asyncio
    async def test_validate_belief_from_km(self):
        """Should validate belief based on cross-references."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._beliefs["bl_validate"] = {"id": "bl_validate", "confidence": 0.7}

        cross_refs = [
            {"metadata": {"relationship": "supports", "debate_id": "d1"}},
            {"metadata": {"relationship": "supports", "debate_id": "d2"}},
            {"metadata": {"relationship": "supports", "debate_id": "d3"}},
            {
                "metadata": {
                    "relationship": "supports",
                    "debate_id": "d4",
                    "outcome_success": True,
                }
            },
        ]

        result = await adapter.validate_belief_from_km("bl_validate", cross_refs)

        assert result.belief_id == "bl_validate"
        assert result.was_supported is True
        assert result.was_contradicted is False

    @pytest.mark.asyncio
    async def test_validate_belief_not_found(self):
        """Should handle missing belief."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        result = await adapter.validate_belief_from_km("nonexistent", [])

        assert result.recommendation == "review"
        assert result.metadata.get("error") == "belief_not_found"

    @pytest.mark.asyncio
    async def test_validate_belief_contradicted(self):
        """Should detect contradicted beliefs."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._beliefs["bl_contradict"] = {"id": "bl_contradict", "confidence": 0.8}

        cross_refs = [{"metadata": {"relationship": "contradicts"}} for _ in range(5)]

        result = await adapter.validate_belief_from_km("bl_contradict", cross_refs)

        assert result.was_contradicted is True
        assert result.recommendation == "penalize"


class TestApplyKMValidation:
    """Tests for apply_km_validation method."""

    @pytest.mark.asyncio
    async def test_apply_km_validation(self):
        """Should apply validation to update belief confidence."""
        from aragora.knowledge.mound.adapters.belief_adapter import (
            BeliefAdapter,
            KMBeliefValidation,
        )

        adapter = BeliefAdapter()
        adapter._beliefs["bl_apply"] = {"id": "bl_apply", "confidence": 0.7}

        validation = KMBeliefValidation(
            belief_id="bl_apply",
            km_confidence=0.85,
            recommendation="boost",
            adjustment=0.1,
        )

        success = await adapter.apply_km_validation(validation)

        assert success is True
        assert adapter._beliefs["bl_apply"]["confidence"] == 0.8
        assert adapter._beliefs["bl_apply"]["km_validated"] is True

    @pytest.mark.asyncio
    async def test_apply_km_validation_not_found(self):
        """Should return False for missing belief."""
        from aragora.knowledge.mound.adapters.belief_adapter import (
            BeliefAdapter,
            KMBeliefValidation,
        )

        adapter = BeliefAdapter()

        validation = KMBeliefValidation(
            belief_id="missing",
            km_confidence=0.5,
            recommendation="keep",
            adjustment=0.0,
        )

        success = await adapter.apply_km_validation(validation)

        assert success is False


class TestSyncValidationsFromKM:
    """Tests for sync_validations_from_km method."""

    @pytest.mark.asyncio
    async def test_sync_validations_from_km(self):
        """Should batch sync validations."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._beliefs = {
            "bl_sync1": {"id": "bl_sync1", "confidence": 0.6},
            "bl_sync2": {"id": "bl_sync2", "confidence": 0.7},
        }

        km_items = [
            {
                "metadata": {
                    "belief_id": "bl_sync1",
                    "relationship": "supports",
                    "outcome_success": True,
                }
            },
            {"metadata": {"belief_id": "bl_sync2", "relationship": "contradicts"}},
        ]

        result = await adapter.sync_validations_from_km(km_items)

        assert result.beliefs_analyzed > 0
        assert result.duration_ms > 0
        assert len(result.threshold_updates) > 0


# ============================================================================
# Reverse Flow Stats and Cleanup Tests
# ============================================================================


class TestReverseFlowStats:
    """Tests for reverse flow statistics and cleanup."""

    def test_get_reverse_flow_stats(self):
        """Should return reverse flow statistics."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._km_validations_applied = 5
        adapter._km_threshold_updates = 2
        adapter._km_validated_priors = {"factual": MagicMock()}

        stats = adapter.get_reverse_flow_stats()

        assert stats["km_validations_applied"] == 5
        assert stats["km_threshold_updates"] == 2
        assert stats["km_priors_computed"] == 1

    def test_clear_reverse_flow_state(self):
        """Should clear all reverse flow state."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._km_validations_applied = 10
        adapter._km_threshold_updates = 5
        adapter._km_validated_priors = {"test": MagicMock()}
        adapter._km_validations = [MagicMock()]
        adapter._outcome_history = {"bl_1": []}

        adapter.clear_reverse_flow_state()

        assert adapter._km_validations_applied == 0
        assert adapter._km_threshold_updates == 0
        assert adapter._km_validated_priors == {}
        assert adapter._km_validations == []
        assert adapter._outcome_history == {}
        assert adapter.MIN_BELIEF_CONFIDENCE == 0.8
        assert adapter.MIN_CRUX_SCORE == 0.3


# ============================================================================
# FusionMixin Integration Tests
# ============================================================================


class TestFusionMixinIntegration:
    """Tests for FusionMixin abstract method implementations."""

    def test_get_fusion_sources(self):
        """Should return list of source adapters."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        sources = adapter._get_fusion_sources()

        assert "consensus" in sources
        assert "elo" in sources
        assert "evidence" in sources
        assert "continuum" in sources
        assert "insights" in sources

    def test_extract_fusible_data(self):
        """Should extract data from KM item."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        km_item = {
            "id": "item-123",
            "confidence": 0.85,
            "metadata": {
                "is_crux": True,
                "crux_score": 0.7,
                "centrality": 0.5,
                "sources": ["src1", "src2"],
                "reasoning": "Test reasoning",
            },
        }

        data = adapter._extract_fusible_data(km_item)

        assert data["confidence"] == 0.85
        assert data["is_crux"] is True
        assert data["crux_score"] == 0.7
        assert data["centrality"] == 0.5
        assert data["is_valid"] is True  # 0.85 >= 0.8

    def test_apply_fusion_result(self):
        """Should apply fusion result to belief record."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        record = {"id": "bl_fusion", "confidence": 0.7}
        fusion_result = MagicMock()
        fusion_result.fused_confidence = 0.85

        success = adapter._apply_fusion_result(record, fusion_result)

        assert success is True
        assert record["confidence"] == 0.85
        assert record["km_fused"] is True

    def test_apply_fusion_result_updates_stored_belief(self):
        """Should update stored belief when applying fusion."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        adapter._beliefs["bl_stored"] = {"id": "bl_stored", "confidence": 0.6}

        record = {"id": "bl_stored", "confidence": 0.6}
        fusion_result = MagicMock()
        fusion_result.fused_confidence = 0.9

        adapter._apply_fusion_result(record, fusion_result)

        assert adapter._beliefs["bl_stored"]["confidence"] == 0.9

    def test_apply_fusion_result_no_fused_confidence(self):
        """Should return False when no fused_confidence."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        record = {"id": "bl_no_fusion"}
        fusion_result = MagicMock(spec=[])  # No fused_confidence attribute

        success = adapter._apply_fusion_result(record, fusion_result)

        assert success is False


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestDataclasses:
    """Tests for exported dataclasses."""

    def test_km_threshold_update(self):
        """Should create KMThresholdUpdate."""
        from aragora.knowledge.mound.adapters.belief_adapter import KMThresholdUpdate

        update = KMThresholdUpdate(
            old_belief_confidence_threshold=0.8,
            new_belief_confidence_threshold=0.75,
            old_crux_score_threshold=0.3,
            new_crux_score_threshold=0.25,
        )

        assert update.recommendation == "keep"
        assert update.patterns_analyzed == 0

    def test_km_belief_validation(self):
        """Should create KMBeliefValidation."""
        from aragora.knowledge.mound.adapters.belief_adapter import KMBeliefValidation

        validation = KMBeliefValidation(
            belief_id="bl_test",
            km_confidence=0.85,
            outcome_success_rate=0.7,
            was_supported=True,
        )

        assert validation.recommendation == "keep"
        assert validation.adjustment == 0.0

    def test_km_prior_recommendation(self):
        """Should create KMPriorRecommendation."""
        from aragora.knowledge.mound.adapters.belief_adapter import KMPriorRecommendation

        prior = KMPriorRecommendation(
            claim_type="factual",
            recommended_prior=0.6,
            sample_count=50,
        )

        assert prior.confidence == 0.7
        assert prior.supporting_debates == []

    def test_belief_search_result(self):
        """Should create BeliefSearchResult."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefSearchResult

        result = BeliefSearchResult(
            belief={"id": "bl_1"},
            relevance_score=0.9,
        )

        assert result.matched_topics == []

    def test_crux_search_result(self):
        """Should create CruxSearchResult."""
        from aragora.knowledge.mound.adapters.belief_adapter import CruxSearchResult

        result = CruxSearchResult(
            crux={"id": "cx_1"},
            relevance_score=0.85,
        )

        assert result.debate_ids == []

    def test_belief_threshold_sync_result(self):
        """Should create BeliefThresholdSyncResult."""
        from aragora.knowledge.mound.adapters.belief_adapter import (
            BeliefThresholdSyncResult,
        )

        result = BeliefThresholdSyncResult()

        assert result.beliefs_analyzed == 0
        assert result.threshold_updates == []
        assert result.errors == []


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_store_belief_with_missing_posterior_attribute(self):
        """Should handle nodes with missing posterior attribute."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        # Node with posterior that doesn't have p_true
        node = MagicMock()
        node.node_id = "edge-1"
        node.claim_id = "claim-edge"
        node.claim_statement = "Test"
        node.author = "test"
        node.posterior = MagicMock(spec=[])  # No p_true
        node.prior = MockBeliefDistribution()
        node.status = MagicMock()
        node.status.value = "prior"
        node.centrality = 0.0
        node.update_count = 0
        node.parent_ids = []
        node.child_ids = []

        # Should use default confidence of 0.5, which is below threshold
        result = adapter.store_converged_belief(node)
        assert result is None

    def test_datetime_string_parsing(self):
        """Should handle various datetime string formats."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        # ISO format with Z
        belief_z = {
            "id": "bl_z",
            "created_at": "2024-01-15T10:30:00Z",
        }
        item = adapter.to_knowledge_item(belief_z)
        assert item.created_at is not None

        # ISO format with timezone
        belief_tz = {
            "id": "bl_tz",
            "created_at": "2024-01-15T10:30:00+00:00",
        }
        item = adapter.to_knowledge_item(belief_tz)
        assert item.created_at is not None

    def test_empty_claim_statement(self):
        """Should handle empty claim statements."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()
        node = MockBeliefNode(
            claim_statement="",
            posterior=MockBeliefDistribution(p_true=0.9),
        )

        belief_id = adapter.store_converged_belief(node)

        assert belief_id is not None
        # Event emission should handle empty string
        assert adapter._beliefs[belief_id]["claim_statement"] == ""

    def test_confidence_boundary_values(self):
        """Should handle boundary confidence values."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        # Exactly at threshold
        node = MockBeliefNode(
            node_id="boundary",
            posterior=MockBeliefDistribution(p_true=0.8),
        )
        result = adapter.store_converged_belief(node)
        assert result is not None

        # Just below threshold
        node_below = MockBeliefNode(
            node_id="below",
            posterior=MockBeliefDistribution(p_true=0.79),
        )
        result = adapter.store_converged_belief(node_below)
        assert result is None

    def test_metric_recording_handles_import_error(self):
        """Should handle missing metrics gracefully."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        # This should not raise even if metrics aren't available
        adapter._record_metric("test_op", True, 0.1)


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Should export all expected classes."""
        from aragora.knowledge.mound.adapters import belief_adapter

        exports = belief_adapter.__all__

        assert "BeliefAdapter" in exports
        assert "BeliefSearchResult" in exports
        assert "CruxSearchResult" in exports
        assert "KMThresholdUpdate" in exports
        assert "KMBeliefValidation" in exports
        assert "KMPriorRecommendation" in exports
        assert "BeliefThresholdSyncResult" in exports
