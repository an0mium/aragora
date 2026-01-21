"""Tests for the Evidence-Provenance Bridge.

Tests cover:
- Evidence registration as provenance records
- Evidence-claim linking
- Belief updates from evidence
- Evidence chain creation
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from aragora.reasoning.evidence_bridge import (
    EvidenceProvenanceBridge,
    EvidenceLink,
    EvidenceImpact,
    get_evidence_bridge,
    reset_evidence_bridge,
)
from aragora.reasoning.belief import BeliefDistribution
from aragora.reasoning.provenance import SourceType
from aragora.evidence.collector import EvidenceSnippet


def make_evidence_snippet(
    id: str = "ev_001",
    source: str = "local_docs",
    title: str = "Test Evidence",
    snippet: str = "This is test evidence content.",
    url: str = "https://example.com/doc",
    reliability_score: float = 0.8,
) -> EvidenceSnippet:
    """Factory for creating test evidence snippets."""
    return EvidenceSnippet(
        id=id,
        source=source,
        title=title,
        snippet=snippet,
        url=url,
        reliability_score=reliability_score,
        metadata={},
        fetched_at=datetime.now(),
    )


class TestEvidenceProvenanceBridge:
    """Test EvidenceProvenanceBridge class."""

    def setup_method(self):
        """Reset global state before each test."""
        reset_evidence_bridge()

    def test_register_evidence_creates_provenance(self):
        """Test registering evidence creates a provenance record."""
        bridge = EvidenceProvenanceBridge()
        snippet = make_evidence_snippet()

        record = bridge.register_evidence(snippet)

        assert record is not None
        assert record.id == f"ev_{snippet.id}"
        assert record.content == snippet.snippet
        assert record.source_type == SourceType.DOCUMENT
        assert record.confidence == snippet.reliability_score

    def test_register_evidence_maps_source_types(self):
        """Test source types are correctly mapped."""
        bridge = EvidenceProvenanceBridge()

        # Test different source mappings
        mappings = [
            ("local_docs", SourceType.DOCUMENT),
            ("github", SourceType.CODE_ANALYSIS),
            ("web", SourceType.WEB_SEARCH),
            ("api", SourceType.EXTERNAL_API),
            ("user", SourceType.USER_PROVIDED),
            ("unknown_source", SourceType.UNKNOWN),
        ]

        for source, expected_type in mappings:
            snippet = make_evidence_snippet(id=f"ev_{source}", source=source)
            record = bridge.register_evidence(snippet)
            assert record.source_type == expected_type, f"Failed for source: {source}"

    def test_get_provenance_for_evidence(self):
        """Test retrieving provenance for registered evidence."""
        bridge = EvidenceProvenanceBridge()
        snippet = make_evidence_snippet()

        bridge.register_evidence(snippet)
        record = bridge.get_provenance_for_evidence(snippet.id)

        assert record is not None
        assert record.content == snippet.snippet

    def test_get_provenance_for_unregistered_returns_none(self):
        """Test getting provenance for unregistered evidence returns None."""
        bridge = EvidenceProvenanceBridge()
        record = bridge.get_provenance_for_evidence("nonexistent")
        assert record is None


class TestEvidenceClaimLinking:
    """Test evidence-to-claim linking functionality."""

    def setup_method(self):
        reset_evidence_bridge()

    def test_link_evidence_to_claim(self):
        """Test linking evidence to a claim."""
        bridge = EvidenceProvenanceBridge()
        snippet = make_evidence_snippet()
        claim_id = "claim_001"

        link = bridge.link_evidence_to_claim(snippet, claim_id)

        assert isinstance(link, EvidenceLink)
        assert link.evidence_id == snippet.id
        assert link.claim_id == claim_id
        assert link.relevance == 1.0
        assert link.support_direction == 1.0

    def test_link_with_custom_relevance(self):
        """Test linking with custom relevance and direction."""
        bridge = EvidenceProvenanceBridge()
        snippet = make_evidence_snippet()

        link = bridge.link_evidence_to_claim(
            snippet,
            "claim_001",
            relevance=0.7,
            support_direction=-1.0,  # Contradicting
        )

        assert link.relevance == 0.7
        assert link.support_direction == -1.0

    def test_link_auto_registers_evidence(self):
        """Test linking auto-registers unregistered evidence."""
        bridge = EvidenceProvenanceBridge()
        snippet = make_evidence_snippet()

        # Don't register first
        link = bridge.link_evidence_to_claim(snippet, "claim_001")

        # Should still work - auto-registration
        assert link.provenance_id is not None
        assert bridge.get_provenance_for_evidence(snippet.id) is not None

    def test_get_evidence_for_claim(self):
        """Test getting all evidence linked to a claim."""
        bridge = EvidenceProvenanceBridge()
        claim_id = "claim_001"

        # Link multiple evidence to same claim
        for i in range(3):
            snippet = make_evidence_snippet(id=f"ev_{i}")
            bridge.link_evidence_to_claim(snippet, claim_id)

        links = bridge.get_evidence_for_claim(claim_id)

        assert len(links) == 3
        assert all(link.claim_id == claim_id for link in links)

    def test_get_claims_for_evidence(self):
        """Test getting all claims linked to evidence."""
        bridge = EvidenceProvenanceBridge()
        snippet = make_evidence_snippet()

        # Link same evidence to multiple claims
        for i in range(3):
            bridge.link_evidence_to_claim(snippet, f"claim_{i}")

        links = bridge.get_claims_for_evidence(snippet.id)

        assert len(links) == 3
        assert all(link.evidence_id == snippet.id for link in links)


class TestBeliefUpdates:
    """Test belief distribution updates from evidence."""

    def setup_method(self):
        reset_evidence_bridge()

    def test_update_belief_no_evidence(self):
        """Test updating belief with no evidence returns original."""
        bridge = EvidenceProvenanceBridge()
        belief = BeliefDistribution(p_true=0.5, p_false=0.5)

        impact = bridge.update_belief_from_evidence(belief, [])

        assert impact.original_belief == belief
        assert impact.updated_belief == belief
        assert impact.evidence_count == 0
        assert impact.direction == "neutral"

    def test_update_belief_supporting_evidence(self):
        """Test supporting evidence increases p_true."""
        bridge = EvidenceProvenanceBridge()
        belief = BeliefDistribution(p_true=0.5, p_false=0.5)

        # High reliability supporting evidence
        snippets = [
            make_evidence_snippet(id="ev_1", reliability_score=0.9),
            make_evidence_snippet(id="ev_2", reliability_score=0.8),
        ]
        for s in snippets:
            s.metadata["support_direction"] = 1.0

        impact = bridge.update_belief_from_evidence(belief, snippets)

        assert impact.updated_belief.p_true > belief.p_true
        assert impact.direction == "supporting"

    def test_update_belief_contradicting_evidence(self):
        """Test contradicting evidence increases p_false."""
        bridge = EvidenceProvenanceBridge()
        belief = BeliefDistribution(p_true=0.5, p_false=0.5)

        snippets = [make_evidence_snippet(reliability_score=0.9)]
        snippets[0].metadata["support_direction"] = -1.0

        impact = bridge.update_belief_from_evidence(belief, snippets)

        assert impact.updated_belief.p_false > belief.p_false
        assert impact.direction == "contradicting"

    def test_update_belief_mixed_evidence(self):
        """Test mixed evidence shows as mixed direction."""
        bridge = EvidenceProvenanceBridge()
        belief = BeliefDistribution(p_true=0.5, p_false=0.5)

        snippets = [
            make_evidence_snippet(id="ev_1", reliability_score=0.8),
            make_evidence_snippet(id="ev_2", reliability_score=0.8),
        ]
        snippets[0].metadata["support_direction"] = 1.0
        snippets[1].metadata["support_direction"] = -1.0

        impact = bridge.update_belief_from_evidence(belief, snippets)

        assert impact.direction == "mixed"

    def test_update_belief_respects_reliability(self):
        """Test evidence weight respects reliability score."""
        bridge = EvidenceProvenanceBridge()
        belief = BeliefDistribution(p_true=0.5, p_false=0.5)

        # Low reliability evidence
        low_rel = [make_evidence_snippet(reliability_score=0.2)]
        low_rel[0].metadata["support_direction"] = 1.0
        impact_low = bridge.update_belief_from_evidence(belief, low_rel)

        # High reliability evidence
        high_rel = [make_evidence_snippet(reliability_score=0.9)]
        high_rel[0].metadata["support_direction"] = 1.0
        impact_high = bridge.update_belief_from_evidence(belief, high_rel)

        # High reliability should have more impact
        assert impact_high.total_weight > impact_low.total_weight


class TestEvidenceChains:
    """Test evidence chain creation."""

    def setup_method(self):
        reset_evidence_bridge()

    def test_create_evidence_chain(self):
        """Test creating a chain from multiple snippets."""
        bridge = EvidenceProvenanceBridge()
        snippets = [
            make_evidence_snippet(id="ev_1"),
            make_evidence_snippet(id="ev_2"),
            make_evidence_snippet(id="ev_3"),
        ]

        chain_id = bridge.create_evidence_chain(snippets)

        assert chain_id is not None
        assert len(chain_id) == 12  # UUID prefix length

    def test_create_chain_links_to_claim(self):
        """Test chain creation with claim linking."""
        bridge = EvidenceProvenanceBridge()
        snippets = [
            make_evidence_snippet(id="ev_1"),
            make_evidence_snippet(id="ev_2"),
        ]
        claim_id = "claim_001"

        bridge.create_evidence_chain(snippets, claim_id=claim_id)

        # All snippets should be linked to claim
        links = bridge.get_evidence_for_claim(claim_id)
        assert len(links) == 2

    def test_create_chain_empty_raises(self):
        """Test creating chain from empty list raises."""
        bridge = EvidenceProvenanceBridge()

        with pytest.raises(ValueError):
            bridge.create_evidence_chain([])

    def test_get_chain_summary(self):
        """Test getting chain summary."""
        bridge = EvidenceProvenanceBridge()
        snippets = [
            make_evidence_snippet(id="ev_1", source="local_docs"),
            make_evidence_snippet(id="ev_2", source="github"),
        ]

        chain_id = bridge.create_evidence_chain(snippets)
        summary = bridge.get_chain_summary(chain_id)

        assert summary["chain_id"] == chain_id
        assert summary["length"] == 2
        assert "document" in summary["sources"] or "code_analysis" in summary["sources"]


class TestGlobalBridge:
    """Test global bridge instance management."""

    def setup_method(self):
        reset_evidence_bridge()

    def test_get_evidence_bridge_returns_singleton(self):
        """Test get_evidence_bridge returns same instance."""
        bridge1 = get_evidence_bridge()
        bridge2 = get_evidence_bridge()
        assert bridge1 is bridge2

    def test_reset_clears_bridge(self):
        """Test reset creates new bridge instance."""
        bridge1 = get_evidence_bridge()
        reset_evidence_bridge()
        bridge2 = get_evidence_bridge()
        assert bridge1 is not bridge2
