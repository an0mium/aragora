"""
Tests for source attribution chain and reputation scoring.

Tests the SourceReputationManager, ReputationScorer, AttributionChain,
and related components of the evidence attribution system.
"""

import pytest
from datetime import datetime, timedelta

from aragora.evidence.attribution import (
    AttributionChain,
    AttributionChainEntry,
    ReputationScorer,
    ReputationTier,
    SourceReputation,
    SourceReputationManager,
    VerificationOutcome,
    VerificationRecord,
)


# =============================================================================
# VerificationOutcome Tests
# =============================================================================


class TestVerificationOutcome:
    """Tests for VerificationOutcome enum."""

    def test_verification_outcome_values(self):
        """Test all outcome values."""
        assert VerificationOutcome.VERIFIED.value == "verified"
        assert VerificationOutcome.PARTIALLY_VERIFIED.value == "partial"
        assert VerificationOutcome.UNVERIFIED.value == "unverified"
        assert VerificationOutcome.CONTESTED.value == "contested"
        assert VerificationOutcome.REFUTED.value == "refuted"


# =============================================================================
# ReputationTier Tests
# =============================================================================


class TestReputationTier:
    """Tests for ReputationTier enum."""

    def test_reputation_tier_values(self):
        """Test all tier values."""
        assert ReputationTier.AUTHORITATIVE.value == "authoritative"
        assert ReputationTier.RELIABLE.value == "reliable"
        assert ReputationTier.STANDARD.value == "standard"
        assert ReputationTier.UNCERTAIN.value == "uncertain"
        assert ReputationTier.UNRELIABLE.value == "unreliable"

    def test_reputation_tier_from_score_authoritative(self):
        """Test authoritative tier classification."""
        assert ReputationTier.from_score(0.90) == ReputationTier.AUTHORITATIVE
        assert ReputationTier.from_score(0.85) == ReputationTier.AUTHORITATIVE
        assert ReputationTier.from_score(1.0) == ReputationTier.AUTHORITATIVE

    def test_reputation_tier_from_score_reliable(self):
        """Test reliable tier classification."""
        assert ReputationTier.from_score(0.70) == ReputationTier.RELIABLE
        assert ReputationTier.from_score(0.80) == ReputationTier.RELIABLE
        assert ReputationTier.from_score(0.84) == ReputationTier.RELIABLE

    def test_reputation_tier_from_score_standard(self):
        """Test standard tier classification."""
        assert ReputationTier.from_score(0.50) == ReputationTier.STANDARD
        assert ReputationTier.from_score(0.60) == ReputationTier.STANDARD
        assert ReputationTier.from_score(0.69) == ReputationTier.STANDARD

    def test_reputation_tier_from_score_uncertain(self):
        """Test uncertain tier classification."""
        assert ReputationTier.from_score(0.30) == ReputationTier.UNCERTAIN
        assert ReputationTier.from_score(0.40) == ReputationTier.UNCERTAIN
        assert ReputationTier.from_score(0.49) == ReputationTier.UNCERTAIN

    def test_reputation_tier_from_score_unreliable(self):
        """Test unreliable tier classification."""
        assert ReputationTier.from_score(0.0) == ReputationTier.UNRELIABLE
        assert ReputationTier.from_score(0.20) == ReputationTier.UNRELIABLE
        assert ReputationTier.from_score(0.29) == ReputationTier.UNRELIABLE


# =============================================================================
# VerificationRecord Tests
# =============================================================================


class TestVerificationRecord:
    """Tests for VerificationRecord dataclass."""

    def test_verification_record_creation(self):
        """Test creating a verification record."""
        record = VerificationRecord(
            record_id="rec-123",
            source_id="arxiv.org",
            debate_id="debate-456",
            outcome=VerificationOutcome.VERIFIED,
        )
        assert record.record_id == "rec-123"
        assert record.source_id == "arxiv.org"
        assert record.outcome == VerificationOutcome.VERIFIED
        assert record.confidence == 1.0

    def test_verification_record_to_dict(self):
        """Test serialization."""
        record = VerificationRecord(
            record_id="rec-123",
            source_id="arxiv.org",
            debate_id="debate-456",
            outcome=VerificationOutcome.VERIFIED,
            confidence=0.9,
            notes="Confirmed by external source",
        )
        data = record.to_dict()
        assert data["record_id"] == "rec-123"
        assert data["outcome"] == "verified"
        assert data["confidence"] == 0.9
        assert data["notes"] == "Confirmed by external source"

    def test_verification_record_from_dict(self):
        """Test deserialization."""
        data = {
            "record_id": "rec-123",
            "source_id": "arxiv.org",
            "debate_id": "debate-456",
            "outcome": "refuted",
            "confidence": 0.8,
            "timestamp": datetime.now().isoformat(),
        }
        record = VerificationRecord.from_dict(data)
        assert record.record_id == "rec-123"
        assert record.outcome == VerificationOutcome.REFUTED
        assert record.confidence == 0.8


# =============================================================================
# SourceReputation Tests
# =============================================================================


class TestSourceReputation:
    """Tests for SourceReputation dataclass."""

    def test_source_reputation_defaults(self):
        """Test default values."""
        rep = SourceReputation(
            source_id="test-source",
            source_type="web",
        )
        assert rep.reputation_score == 0.5
        assert rep.verification_count == 0
        assert rep.verified_count == 0
        assert rep.refuted_count == 0
        assert rep.tier == ReputationTier.STANDARD

    def test_source_reputation_verification_rate(self):
        """Test verification rate calculation."""
        rep = SourceReputation(
            source_id="test",
            source_type="web",
            verification_count=10,
            verified_count=8,
        )
        assert rep.verification_rate == 0.8

    def test_source_reputation_verification_rate_zero(self):
        """Test verification rate with no verifications."""
        rep = SourceReputation(
            source_id="test",
            source_type="web",
        )
        assert rep.verification_rate == 0.5  # Neutral for new sources

    def test_source_reputation_refutation_rate(self):
        """Test refutation rate calculation."""
        rep = SourceReputation(
            source_id="test",
            source_type="web",
            verification_count=10,
            refuted_count=3,
        )
        assert rep.refutation_rate == 0.3

    def test_source_reputation_to_dict(self):
        """Test serialization."""
        rep = SourceReputation(
            source_id="arxiv.org",
            source_type="academic",
            reputation_score=0.9,
            verified_count=100,
            verification_count=110,
        )
        data = rep.to_dict()
        assert data["source_id"] == "arxiv.org"
        assert data["source_type"] == "academic"
        assert data["reputation_score"] == 0.9
        assert data["tier"] == "authoritative"
        assert "verification_rate" in data

    def test_source_reputation_from_dict(self):
        """Test deserialization."""
        data = {
            "source_id": "test",
            "source_type": "web",
            "reputation_score": 0.75,
            "first_seen": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }
        rep = SourceReputation.from_dict(data)
        assert rep.source_id == "test"
        assert rep.reputation_score == 0.75


# =============================================================================
# ReputationScorer Tests
# =============================================================================


class TestReputationScorer:
    """Tests for ReputationScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create a ReputationScorer instance."""
        return ReputationScorer()

    def test_compute_score_empty(self, scorer):
        """Test scoring with no verifications."""
        overall, recent, trend = scorer.compute_score([])
        assert overall == 0.5  # Default
        assert recent == 0.5
        assert trend == 0.0

    def test_compute_score_verified(self, scorer):
        """Test scoring with verified records."""
        verifications = [
            VerificationRecord(
                record_id=f"rec-{i}",
                source_id="test",
                debate_id="debate",
                outcome=VerificationOutcome.VERIFIED,
            )
            for i in range(5)
        ]
        overall, recent, trend = scorer.compute_score(verifications)
        assert overall > 0.5  # Should improve

    def test_compute_score_refuted(self, scorer):
        """Test scoring with refuted records."""
        verifications = [
            VerificationRecord(
                record_id=f"rec-{i}",
                source_id="test",
                debate_id="debate",
                outcome=VerificationOutcome.REFUTED,
            )
            for i in range(5)
        ]
        overall, recent, trend = scorer.compute_score(verifications)
        assert overall < 0.5  # Should decrease

    def test_compute_score_mixed(self, scorer):
        """Test scoring with mixed outcomes."""
        verifications = [
            VerificationRecord(
                record_id="rec-1",
                source_id="test",
                debate_id="debate",
                outcome=VerificationOutcome.VERIFIED,
            ),
            VerificationRecord(
                record_id="rec-2",
                source_id="test",
                debate_id="debate",
                outcome=VerificationOutcome.REFUTED,
            ),
            VerificationRecord(
                record_id="rec-3",
                source_id="test",
                debate_id="debate",
                outcome=VerificationOutcome.VERIFIED,
            ),
        ]
        overall, recent, trend = scorer.compute_score(verifications)
        # 2 verified, 1 refuted - should be slightly positive
        assert 0.4 <= overall <= 0.6

    def test_compute_incremental_update_verified(self, scorer):
        """Test incremental update with verified outcome."""
        rep = SourceReputation(source_id="test", source_type="web")
        verification = VerificationRecord(
            record_id="rec-1",
            source_id="test",
            debate_id="debate",
            outcome=VerificationOutcome.VERIFIED,
        )
        updated = scorer.compute_incremental_update(rep, verification)
        assert updated.verified_count == 1
        assert updated.verification_count == 1
        assert updated.reputation_score > 0.5

    def test_compute_incremental_update_refuted(self, scorer):
        """Test incremental update with refuted outcome."""
        rep = SourceReputation(source_id="test", source_type="web")
        verification = VerificationRecord(
            record_id="rec-1",
            source_id="test",
            debate_id="debate",
            outcome=VerificationOutcome.REFUTED,
        )
        updated = scorer.compute_incremental_update(rep, verification)
        assert updated.refuted_count == 1
        assert updated.reputation_score < 0.5

    def test_compute_incremental_update_tracks_history(self, scorer):
        """Test that incremental updates track score history."""
        rep = SourceReputation(source_id="test", source_type="web")
        assert len(rep.score_history) == 0

        verification = VerificationRecord(
            record_id="rec-1",
            source_id="test",
            debate_id="debate",
            outcome=VerificationOutcome.VERIFIED,
        )
        updated = scorer.compute_incremental_update(rep, verification)
        assert len(updated.score_history) == 1


# =============================================================================
# SourceReputationManager Tests
# =============================================================================


class TestSourceReputationManager:
    """Tests for SourceReputationManager class."""

    @pytest.fixture
    def manager(self):
        """Create a SourceReputationManager instance."""
        return SourceReputationManager()

    def test_get_or_create_reputation(self, manager):
        """Test creating new reputation."""
        rep = manager.get_or_create_reputation("arxiv.org", "academic")
        assert rep.source_id == "arxiv.org"
        assert rep.source_type == "academic"
        assert rep.reputation_score == 0.5

        # Should return same instance
        rep2 = manager.get_or_create_reputation("arxiv.org")
        assert rep2 is rep

    def test_record_verification(self, manager):
        """Test recording a verification."""
        verification = manager.record_verification(
            record_id="rec-123",
            source_id="arxiv.org",
            debate_id="debate-1",
            outcome=VerificationOutcome.VERIFIED,
            source_type="academic",
        )
        assert verification.source_id == "arxiv.org"
        assert verification.outcome == VerificationOutcome.VERIFIED

        # Check reputation updated
        rep = manager.get_reputation("arxiv.org")
        assert rep is not None
        assert rep.verification_count == 1
        assert rep.verified_count == 1

    def test_record_multiple_verifications(self, manager):
        """Test recording multiple verifications."""
        for i in range(5):
            manager.record_verification(
                record_id=f"rec-{i}",
                source_id="test-source",
                debate_id="debate-1",
                outcome=VerificationOutcome.VERIFIED,
            )

        rep = manager.get_reputation("test-source")
        assert rep.verification_count == 5
        assert rep.verified_count == 5

    def test_record_verification_cross_debate(self, manager):
        """Test tracking debate participation."""
        manager.record_verification(
            record_id="rec-1",
            source_id="test",
            debate_id="debate-1",
            outcome=VerificationOutcome.VERIFIED,
        )
        manager.record_verification(
            record_id="rec-2",
            source_id="test",
            debate_id="debate-2",
            outcome=VerificationOutcome.VERIFIED,
        )

        rep = manager.get_reputation("test")
        assert rep.debate_count == 2

    def test_get_source_history(self, manager):
        """Test getting verification history."""
        for i in range(10):
            manager.record_verification(
                record_id=f"rec-{i}",
                source_id="test",
                debate_id="debate-1",
                outcome=VerificationOutcome.VERIFIED,
            )

        history = manager.get_source_history("test", limit=5)
        assert len(history) == 5
        # Should be most recent first
        assert history[0].record_id == "rec-9"

    def test_get_debate_sources(self, manager):
        """Test getting sources for a debate."""
        manager.record_verification(
            record_id="rec-1",
            source_id="source-a",
            debate_id="debate-1",
            outcome=VerificationOutcome.VERIFIED,
        )
        manager.record_verification(
            record_id="rec-2",
            source_id="source-b",
            debate_id="debate-1",
            outcome=VerificationOutcome.VERIFIED,
        )

        sources = manager.get_debate_sources("debate-1")
        source_ids = [s.source_id for s in sources]
        assert "source-a" in source_ids
        assert "source-b" in source_ids

    def test_get_top_sources(self, manager):
        """Test getting top-rated sources."""
        # Create sources with different reputations
        for i in range(5):
            manager.record_verification(
                record_id=f"rec-{i}",
                source_id="good-source",
                debate_id="debate-1",
                outcome=VerificationOutcome.VERIFIED,
            )
        for i in range(5):
            manager.record_verification(
                record_id=f"rec-bad-{i}",
                source_id="bad-source",
                debate_id="debate-1",
                outcome=VerificationOutcome.REFUTED,
            )

        top = manager.get_top_sources(limit=10)
        assert len(top) == 2
        assert top[0].source_id == "good-source"

    def test_get_unreliable_sources(self, manager):
        """Test getting unreliable sources."""
        for i in range(5):
            manager.record_verification(
                record_id=f"rec-{i}",
                source_id="unreliable",
                debate_id="debate-1",
                outcome=VerificationOutcome.REFUTED,
            )

        unreliable = manager.get_unreliable_sources(threshold=0.4)
        assert len(unreliable) >= 1
        assert unreliable[0].source_id == "unreliable"

    def test_export_import_state(self, manager):
        """Test state export and import."""
        manager.record_verification(
            record_id="rec-1",
            source_id="test",
            debate_id="debate-1",
            outcome=VerificationOutcome.VERIFIED,
        )

        state = manager.export_state()
        assert "reputations" in state
        assert "verifications" in state
        assert "test" in state["reputations"]

        # Import into new manager
        new_manager = SourceReputationManager()
        new_manager.import_state(state)

        rep = new_manager.get_reputation("test")
        assert rep is not None
        assert rep.verified_count == 1


# =============================================================================
# AttributionChainEntry Tests
# =============================================================================


class TestAttributionChainEntry:
    """Tests for AttributionChainEntry dataclass."""

    def test_attribution_entry_creation(self):
        """Test creating an attribution entry."""
        entry = AttributionChainEntry(
            evidence_id="ev-123",
            source_id="arxiv.org",
            debate_id="debate-1",
            content_hash="abc123",
            reputation_at_use=0.9,
        )
        assert entry.evidence_id == "ev-123"
        assert entry.reputation_at_use == 0.9
        assert entry.verification_outcome is None

    def test_attribution_entry_to_dict(self):
        """Test serialization."""
        entry = AttributionChainEntry(
            evidence_id="ev-123",
            source_id="arxiv.org",
            debate_id="debate-1",
            content_hash="abc123",
            reputation_at_use=0.9,
            verification_outcome=VerificationOutcome.VERIFIED,
        )
        data = entry.to_dict()
        assert data["evidence_id"] == "ev-123"
        assert data["reputation_at_use"] == 0.9
        assert data["verification_outcome"] == "verified"


# =============================================================================
# AttributionChain Tests
# =============================================================================


class TestAttributionChain:
    """Tests for AttributionChain class."""

    @pytest.fixture
    def chain(self):
        """Create an AttributionChain instance."""
        return AttributionChain()

    def test_add_entry(self, chain):
        """Test adding an entry to the chain."""
        entry = chain.add_entry(
            evidence_id="ev-123",
            source_id="arxiv.org",
            debate_id="debate-1",
            content="Test evidence content",
            source_type="academic",
        )
        assert entry.evidence_id == "ev-123"
        assert len(entry.content_hash) == 64  # SHA-256 hex
        assert len(chain.entries) == 1

    def test_add_entry_captures_reputation(self, chain):
        """Test that entry captures source reputation at time of use."""
        # First set up some reputation
        chain.reputation_manager.record_verification(
            record_id="setup-1",
            source_id="arxiv.org",
            debate_id="setup",
            outcome=VerificationOutcome.VERIFIED,
            source_type="academic",
        )

        entry = chain.add_entry(
            evidence_id="ev-123",
            source_id="arxiv.org",
            debate_id="debate-1",
            content="Test content",
        )

        # Reputation should be captured
        assert entry.reputation_at_use > 0.5

    def test_record_verification(self, chain):
        """Test recording verification for chain entry."""
        chain.add_entry(
            evidence_id="ev-123",
            source_id="test",
            debate_id="debate-1",
            content="Content",
        )

        verification = chain.record_verification(
            evidence_id="ev-123",
            outcome=VerificationOutcome.VERIFIED,
        )

        assert verification is not None
        assert verification.outcome == VerificationOutcome.VERIFIED

        # Entry should be updated
        entries = chain.get_evidence_chain("ev-123")
        assert entries[0].verification_outcome == VerificationOutcome.VERIFIED

    def test_get_evidence_chain(self, chain):
        """Test getting chain for an evidence item."""
        # Add same evidence in multiple debates
        chain.add_entry(
            evidence_id="ev-123",
            source_id="test",
            debate_id="debate-1",
            content="Content",
        )
        chain.add_entry(
            evidence_id="ev-123",
            source_id="test",
            debate_id="debate-2",
            content="Content",
        )

        evidence_chain = chain.get_evidence_chain("ev-123")
        assert len(evidence_chain) == 2

    def test_get_source_chain(self, chain):
        """Test getting all evidence from a source."""
        for i in range(3):
            chain.add_entry(
                evidence_id=f"ev-{i}",
                source_id="arxiv.org",
                debate_id="debate-1",
                content=f"Content {i}",
            )

        source_chain = chain.get_source_chain("arxiv.org")
        assert len(source_chain) == 3

    def test_get_debate_attributions(self, chain):
        """Test getting attributions for a debate."""
        chain.add_entry(
            evidence_id="ev-1",
            source_id="source-a",
            debate_id="debate-1",
            content="Content A",
        )
        chain.add_entry(
            evidence_id="ev-2",
            source_id="source-b",
            debate_id="debate-1",
            content="Content B",
        )
        chain.add_entry(
            evidence_id="ev-3",
            source_id="source-c",
            debate_id="debate-2",
            content="Content C",
        )

        attrs = chain.get_debate_attributions("debate-1")
        assert len(attrs) == 2

    def test_compute_debate_reliability_empty(self, chain):
        """Test reliability metrics for empty debate."""
        metrics = chain.compute_debate_reliability("nonexistent")
        assert metrics["evidence_count"] == 0
        assert metrics["reliability_score"] == 0.5

    def test_compute_debate_reliability(self, chain):
        """Test reliability metrics calculation."""
        # Add evidence with known reputations
        chain.add_entry(
            evidence_id="ev-1",
            source_id="good",
            debate_id="debate-1",
            content="Content",
        )
        chain.add_entry(
            evidence_id="ev-2",
            source_id="good",
            debate_id="debate-1",
            content="Content 2",
        )

        # Verify one entry
        chain.record_verification("ev-1", VerificationOutcome.VERIFIED)

        metrics = chain.compute_debate_reliability("debate-1")
        assert metrics["evidence_count"] == 2
        assert metrics["verified_count"] == 1
        assert 0 <= metrics["reliability_score"] <= 1

    def test_find_reused_evidence(self, chain):
        """Test finding evidence reused across debates."""
        # Same evidence in multiple debates
        chain.add_entry(
            evidence_id="ev-popular",
            source_id="test",
            debate_id="debate-1",
            content="Popular content",
        )
        chain.add_entry(
            evidence_id="ev-popular",
            source_id="test",
            debate_id="debate-2",
            content="Popular content",
        )
        chain.add_entry(
            evidence_id="ev-popular",
            source_id="test",
            debate_id="debate-3",
            content="Popular content",
        )

        # Unique evidence
        chain.add_entry(
            evidence_id="ev-unique",
            source_id="test",
            debate_id="debate-1",
            content="Unique content",
        )

        reused = chain.find_reused_evidence(min_uses=2)
        assert "ev-popular" in reused
        assert "ev-unique" not in reused
        assert len(reused["ev-popular"]) == 3

    def test_export_chain(self, chain):
        """Test exporting chain state."""
        chain.add_entry(
            evidence_id="ev-1",
            source_id="test",
            debate_id="debate-1",
            content="Content",
        )

        exported = chain.export_chain()
        assert "entries" in exported
        assert "reputation_state" in exported
        assert len(exported["entries"]) == 1
