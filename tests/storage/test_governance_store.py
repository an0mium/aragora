"""
Tests for GovernanceStore.

Tests persistence and retrieval of governance artifacts:
- Approvals
- Verifications
- Decisions
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from aragora.storage.governance_store import (
    GovernanceStore,
    ApprovalRecord,
    VerificationRecord,
    DecisionRecord,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_governance.db"


@pytest.fixture
def store(temp_db_path):
    """Create a governance store for testing."""
    return GovernanceStore(db_path=temp_db_path)


class TestApprovalRecords:
    """Tests for approval record persistence."""

    def test_save_and_get_approval(self, store):
        """Should save and retrieve an approval record."""
        approval_id = store.save_approval(
            approval_id="appr_test_123",
            title="Test Approval",
            description="Test description",
            risk_level="medium",
            status="pending",
            requested_by="user@test.com",
            changes=[{"file": "test.py", "action": "modify"}],
            timeout_seconds=3600,
        )

        assert approval_id == "appr_test_123"

        record = store.get_approval("appr_test_123")
        assert record is not None
        assert record.title == "Test Approval"
        assert record.risk_level == "medium"
        assert record.status == "pending"

    def test_get_nonexistent_approval(self, store):
        """Should return None for nonexistent approval."""
        record = store.get_approval("nonexistent")
        assert record is None

    def test_update_approval_status(self, store):
        """Should update approval status."""
        store.save_approval(
            approval_id="appr_update_test",
            title="Update Test",
            description="Test",
            risk_level="low",
            status="pending",
            requested_by="user@test.com",
            changes=[],
        )

        store.update_approval_status(
            approval_id="appr_update_test",
            status="approved",
            approved_by="admin@test.com",
        )

        record = store.get_approval("appr_update_test")
        assert record is not None
        assert record.status == "approved"
        assert record.approved_by == "admin@test.com"

    def test_list_approvals(self, store):
        """Should list approvals with filtering."""
        # Create multiple approvals
        for i in range(5):
            store.save_approval(
                approval_id=f"appr_list_{i}",
                title=f"Approval {i}",
                description="Test",
                risk_level="low" if i % 2 == 0 else "high",
                status="pending" if i < 3 else "approved",
                requested_by="user@test.com",
                changes=[],
            )

        # List all pending
        pending = store.list_approvals(status="pending")
        assert len(pending) == 3

        # List all approved
        approved = store.list_approvals(status="approved")
        assert len(approved) == 2


class TestVerificationRecords:
    """Tests for verification record persistence."""

    def test_save_and_get_verification(self, store):
        """Should save and retrieve a verification record."""
        verification_id = store.save_verification(
            verification_id="ver_test_123",
            claim="x > 0 implies x >= 0",
            context="arithmetic proof",
            result={"valid": True, "prover": "z3"},
            verified_by="z3_prover",
            claim_type="formal",
            confidence=0.95,
        )

        assert verification_id == "ver_test_123"

        record = store.get_verification("ver_test_123")
        assert record is not None
        assert "x > 0" in record.claim
        assert record.confidence == 0.95

    def test_get_nonexistent_verification(self, store):
        """Should return None for nonexistent verification."""
        record = store.get_verification("nonexistent")
        assert record is None

    def test_list_verifications(self, store):
        """Should list verifications with filtering."""
        # Create verifications
        for i in range(5):
            store.save_verification(
                verification_id=f"ver_list_{i}",
                claim=f"Claim {i}",
                context="test",
                result={"valid": i % 2 == 0},
                verified_by="test",
                claim_type="formal" if i < 3 else "runtime",
            )

        # List all
        all_records = store.list_verifications()
        assert len(all_records) >= 5

        # List by claim type
        formal = store.list_verifications(claim_type="formal")
        assert len(formal) >= 3


class TestDecisionRecords:
    """Tests for decision record persistence."""

    def test_save_and_get_decision(self, store):
        """Should save and retrieve a decision record."""
        decision_id = store.save_decision(
            decision_id="dec_test_123",
            debate_id="debate_456",
            conclusion="Consensus reached on option A",
            consensus_reached=True,
            confidence=0.85,
            evidence_chain=[{"claim": "A is better", "source": "agent1"}],
            agents_involved=["claude", "gpt4", "gemini"],
        )

        assert decision_id == "dec_test_123"

        record = store.get_decision("dec_test_123")
        assert record is not None
        assert record.debate_id == "debate_456"
        assert record.consensus_reached is True
        assert record.confidence == 0.85

    def test_get_nonexistent_decision(self, store):
        """Should return None for nonexistent decision."""
        record = store.get_decision("nonexistent")
        assert record is None

    def test_list_decisions(self, store):
        """Should list decisions."""
        # Create decisions
        for i in range(3):
            store.save_decision(
                decision_id=f"dec_list_{i}",
                debate_id=f"debate_{i}",
                conclusion=f"Conclusion {i}",
                consensus_reached=i % 2 == 0,
                confidence=0.7 + i * 0.1,
            )

        decisions = store.list_decisions()
        assert len(decisions) >= 3


class TestPersistence:
    """Tests for data persistence across store instances."""

    def test_data_persists_across_instances(self, temp_db_path):
        """Data should persist after store is recreated."""
        # Create and populate first store
        store1 = GovernanceStore(db_path=temp_db_path)
        store1.save_approval(
            approval_id="persist_test",
            title="Persistence Test",
            description="Test",
            risk_level="low",
            status="pending",
            requested_by="user@test.com",
            changes=[],
        )

        # Create new store instance
        store2 = GovernanceStore(db_path=temp_db_path)

        # Verify data persists
        record = store2.get_approval("persist_test")
        assert record is not None
        assert record.title == "Persistence Test"


class TestMultiTenant:
    """Tests for multi-tenant isolation."""

    def test_approvals_isolated_by_org(self, store):
        """Approvals should be isolated by org_id."""
        store.save_approval(
            approval_id="org_a_approval",
            title="Org A Approval",
            description="Test",
            risk_level="low",
            status="pending",
            requested_by="user@orga.com",
            changes=[],
            org_id="org_a",
        )

        store.save_approval(
            approval_id="org_b_approval",
            title="Org B Approval",
            description="Test",
            risk_level="low",
            status="pending",
            requested_by="user@orgb.com",
            changes=[],
            org_id="org_b",
        )

        # List by org
        org_a_approvals = store.list_approvals(org_id="org_a")
        org_b_approvals = store.list_approvals(org_id="org_b")

        assert len(org_a_approvals) == 1
        assert len(org_b_approvals) == 1
        assert org_a_approvals[0].title == "Org A Approval"


class TestApprovalRecordDataclass:
    """Tests for ApprovalRecord dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        record = ApprovalRecord(
            approval_id="test_123",
            title="Test",
            description="Description",
            risk_level="medium",
            status="pending",
            requested_by="user@test.com",
            requested_at=datetime(2024, 1, 15, 10, 30, 0),
            changes_json='[{"file": "test.py"}]',
        )

        d = record.to_dict()
        assert d["approval_id"] == "test_123"
        assert d["risk_level"] == "medium"
        assert d["changes"] == [{"file": "test.py"}]


class TestVerificationRecordDataclass:
    """Tests for VerificationRecord dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        record = VerificationRecord(
            verification_id="ver_123",
            claim="x > 0",
            claim_type="formal",
            context="test",
            result_json='{"valid": true}',
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            verified_by="z3",
            confidence=0.95,
        )

        d = record.to_dict()
        assert d["verification_id"] == "ver_123"
        assert d["result"]["valid"] is True


class TestDecisionRecordDataclass:
    """Tests for DecisionRecord dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        record = DecisionRecord(
            decision_id="dec_123",
            debate_id="debate_456",
            conclusion="Test conclusion",
            consensus_reached=True,
            confidence=0.85,
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            evidence_chain_json='[{"claim": "A", "source": "agent1"}]',
            vote_pivots_json='[]',
            belief_changes_json='[]',
            agents_involved_json='["claude", "gpt4"]',
        )

        d = record.to_dict()
        assert d["decision_id"] == "dec_123"
        assert d["consensus_reached"] is True
        assert d["evidence_chain"] == [{"claim": "A", "source": "agent1"}]
        assert d["agents_involved"] == ["claude", "gpt4"]
