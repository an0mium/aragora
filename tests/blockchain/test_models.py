"""
Tests for blockchain data models.

Tests cover:
- ValidationResponse enum values and construction
- MetadataEntry creation and edge cases
- OnChainAgentIdentity defaults, full creation, and field independence
- ReputationFeedback normalized_value property (all branches)
- ReputationSummary normalized_value property (all branches)
- ValidationRecord properties (is_passed, is_pending, all response codes)
- ValidationSummary creation
- Module __all__ exports
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aragora.blockchain.models import (
    MetadataEntry,
    OnChainAgentIdentity,
    ReputationFeedback,
    ReputationSummary,
    ValidationRecord,
    ValidationResponse,
    ValidationSummary,
)


class TestValidationResponse:
    """Tests for ValidationResponse enum."""

    def test_pending_value(self):
        assert ValidationResponse.PENDING.value == 0

    def test_pass_value(self):
        assert ValidationResponse.PASS.value == 1

    def test_fail_value(self):
        assert ValidationResponse.FAIL.value == 2

    def test_revoked_value(self):
        assert ValidationResponse.REVOKED.value == 3

    def test_create_from_int(self):
        assert ValidationResponse(0) == ValidationResponse.PENDING
        assert ValidationResponse(1) == ValidationResponse.PASS
        assert ValidationResponse(2) == ValidationResponse.FAIL
        assert ValidationResponse(3) == ValidationResponse.REVOKED

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ValidationResponse(99)

    def test_is_int_enum(self):
        """All members are integers for arithmetic use."""
        for member in ValidationResponse:
            assert isinstance(member, int)

    def test_member_count(self):
        """Exactly 4 response codes."""
        assert len(ValidationResponse) == 4

    def test_negative_value_raises(self):
        with pytest.raises(ValueError):
            ValidationResponse(-1)


class TestMetadataEntry:
    """Tests for MetadataEntry model."""

    def test_create_entry(self):
        entry = MetadataEntry(key="model", value=b"claude-3")
        assert entry.key == "model"
        assert entry.value == b"claude-3"

    def test_bytes_value(self):
        entry = MetadataEntry(key="config", value=b"\x00\x01\x02\x03")
        assert len(entry.value) == 4
        assert entry.value[0] == 0

    def test_empty_key(self):
        entry = MetadataEntry(key="", value=b"data")
        assert entry.key == ""

    def test_empty_value(self):
        entry = MetadataEntry(key="test", value=b"")
        assert entry.value == b""

    def test_large_value(self):
        data = b"\xff" * 10000
        entry = MetadataEntry(key="big", value=data)
        assert len(entry.value) == 10000

    def test_unicode_key(self):
        entry = MetadataEntry(key="description\u2603", value=b"test")
        assert "\u2603" in entry.key


class TestOnChainAgentIdentity:
    """Tests for OnChainAgentIdentity model."""

    def test_create_basic_identity(self):
        identity = OnChainAgentIdentity(
            token_id=42,
            owner="0x1234567890123456789012345678901234567890",
        )
        assert identity.token_id == 42
        assert identity.owner == "0x1234567890123456789012345678901234567890"

    def test_identity_defaults(self):
        identity = OnChainAgentIdentity(
            token_id=1,
            owner="0x1234567890123456789012345678901234567890",
        )
        assert identity.agent_uri == ""
        assert identity.wallet_address is None
        assert identity.aragora_agent_id is None
        assert identity.chain_id == 1
        assert identity.registered_at is None
        assert identity.metadata == {}
        assert identity.on_chain_metadata == []
        assert identity.tx_hash is None

    def test_identity_with_uri(self):
        identity = OnChainAgentIdentity(
            token_id=1,
            owner="0x1234567890123456789012345678901234567890",
            agent_uri="ipfs://QmTest123456789",
        )
        assert identity.agent_uri == "ipfs://QmTest123456789"

    def test_identity_with_wallet_address(self):
        identity = OnChainAgentIdentity(
            token_id=1,
            owner="0x1234567890123456789012345678901234567890",
            wallet_address="0xabcdef1234567890abcdef1234567890abcdef12",
        )
        assert identity.wallet_address == "0xabcdef1234567890abcdef1234567890abcdef12"

    def test_identity_with_metadata(self):
        identity = OnChainAgentIdentity(
            token_id=1,
            owner="0x1234567890123456789012345678901234567890",
            metadata={"capabilities": ["reasoning"], "model": "claude-3"},
        )
        assert "capabilities" in identity.metadata
        assert "reasoning" in identity.metadata["capabilities"]

    def test_identity_with_on_chain_metadata(self):
        entries = [
            MetadataEntry(key="model", value=b"claude-3"),
            MetadataEntry(key="version", value=b"1.0"),
        ]
        identity = OnChainAgentIdentity(
            token_id=1,
            owner="0x1234567890123456789012345678901234567890",
            on_chain_metadata=entries,
        )
        assert len(identity.on_chain_metadata) == 2
        assert identity.on_chain_metadata[0].key == "model"

    def test_identity_with_chain_id(self):
        identity = OnChainAgentIdentity(
            token_id=1,
            owner="0x1234567890123456789012345678901234567890",
            chain_id=137,
        )
        assert identity.chain_id == 137

    def test_identity_with_timestamp(self):
        now = datetime.now(timezone.utc)
        identity = OnChainAgentIdentity(
            token_id=1,
            owner="0x1234567890123456789012345678901234567890",
            registered_at=now,
        )
        assert identity.registered_at == now

    def test_metadata_is_independent(self):
        """Metadata dicts are independent across instances."""
        id1 = OnChainAgentIdentity(token_id=1, owner="0x1")
        id2 = OnChainAgentIdentity(token_id=2, owner="0x2")
        id1.metadata["key"] = "value"
        assert "key" not in id2.metadata

    def test_on_chain_metadata_is_independent(self):
        """On-chain metadata lists are independent across instances."""
        id1 = OnChainAgentIdentity(token_id=1, owner="0x1")
        id2 = OnChainAgentIdentity(token_id=2, owner="0x2")
        id1.on_chain_metadata.append(MetadataEntry(key="a", value=b"b"))
        assert len(id2.on_chain_metadata) == 0

    def test_zero_token_id(self):
        identity = OnChainAgentIdentity(token_id=0, owner="0x0")
        assert identity.token_id == 0

    def test_full_creation(self):
        now = datetime.now(timezone.utc)
        meta_entry = MetadataEntry(key="type", value=b"assistant")
        identity = OnChainAgentIdentity(
            token_id=42,
            owner="0x1234567890abcdef1234567890abcdef12345678",
            agent_uri="https://example.com/agent.json",
            wallet_address="0xfedcba9876543210fedcba9876543210fedcba98",
            aragora_agent_id="claude-3",
            chain_id=137,
            registered_at=now,
            metadata={"model": "claude"},
            on_chain_metadata=[meta_entry],
            tx_hash="0xabc123",
        )
        assert identity.token_id == 42
        assert identity.chain_id == 137
        assert identity.tx_hash == "0xabc123"
        assert identity.aragora_agent_id == "claude-3"


class TestReputationFeedback:
    """Tests for ReputationFeedback model."""

    def test_create_feedback(self):
        feedback = ReputationFeedback(
            agent_id=42,
            client_address="0x1234567890123456789012345678901234567890",
        )
        assert feedback.agent_id == 42
        assert feedback.client_address == "0x1234567890123456789012345678901234567890"

    def test_feedback_defaults(self):
        feedback = ReputationFeedback(
            agent_id=1,
            client_address="0x1234567890123456789012345678901234567890",
        )
        assert feedback.feedback_index == 0
        assert feedback.value == 0
        assert feedback.value_decimals == 0
        assert feedback.tag1 == ""
        assert feedback.tag2 == ""
        assert feedback.endpoint == ""
        assert feedback.feedback_uri == ""
        assert feedback.feedback_hash == ""
        assert feedback.is_revoked is False
        assert feedback.timestamp is None
        assert feedback.tx_hash is None

    def test_positive_feedback(self):
        feedback = ReputationFeedback(
            agent_id=1,
            client_address="0x1234567890123456789012345678901234567890",
            value=100,
            tag1="accuracy",
        )
        assert feedback.value > 0

    def test_negative_feedback(self):
        feedback = ReputationFeedback(
            agent_id=1,
            client_address="0x1234567890123456789012345678901234567890",
            value=-50,
            tag1="reliability",
        )
        assert feedback.value < 0

    def test_revoked_feedback(self):
        feedback = ReputationFeedback(
            agent_id=1,
            client_address="0x1234567890123456789012345678901234567890",
            is_revoked=True,
        )
        assert feedback.is_revoked is True

    def test_normalized_value_no_decimals(self):
        feedback = ReputationFeedback(
            agent_id=1,
            client_address="0x1234567890123456789012345678901234567890",
            value=100,
            value_decimals=0,
        )
        assert feedback.normalized_value == 100.0
        assert isinstance(feedback.normalized_value, float)

    def test_normalized_value_with_decimals(self):
        feedback = ReputationFeedback(
            agent_id=1,
            client_address="0x1234567890123456789012345678901234567890",
            value=12345,
            value_decimals=2,
        )
        assert feedback.normalized_value == 123.45

    def test_normalized_value_with_18_decimals(self):
        """Handle Ethereum-standard 18 decimals."""
        fb = ReputationFeedback(
            agent_id=1,
            client_address="0x1",
            value=1_000_000_000_000_000_000,
            value_decimals=18,
        )
        assert fb.normalized_value == pytest.approx(1.0)

    def test_normalized_value_negative(self):
        fb = ReputationFeedback(agent_id=1, client_address="0x1", value=-500, value_decimals=1)
        assert fb.normalized_value == pytest.approx(-50.0)

    def test_normalized_value_zero(self):
        fb = ReputationFeedback(agent_id=1, client_address="0x1", value=0, value_decimals=5)
        assert fb.normalized_value == 0.0

    def test_normalized_value_small_fraction(self):
        fb = ReputationFeedback(agent_id=1, client_address="0x1", value=1, value_decimals=3)
        assert fb.normalized_value == pytest.approx(0.001)

    def test_feedback_with_tags(self):
        feedback = ReputationFeedback(
            agent_id=1,
            client_address="0x1234567890123456789012345678901234567890",
            tag1="accuracy",
            tag2="reasoning",
        )
        assert feedback.tag1 == "accuracy"
        assert feedback.tag2 == "reasoning"

    def test_feedback_with_endpoint(self):
        feedback = ReputationFeedback(
            agent_id=1,
            client_address="0x1234567890123456789012345678901234567890",
            endpoint="/api/v1/chat",
        )
        assert feedback.endpoint == "/api/v1/chat"

    def test_full_creation(self):
        now = datetime.now(timezone.utc)
        fb = ReputationFeedback(
            agent_id=42,
            client_address="0xClient",
            feedback_index=3,
            value=850,
            value_decimals=2,
            tag1="accuracy",
            tag2="math",
            endpoint="/api/solve",
            feedback_uri="https://example.com/feedback",
            feedback_hash="abc123hash",
            is_revoked=True,
            timestamp=now,
            tx_hash="0xtx",
        )
        assert fb.is_revoked is True
        assert fb.tag1 == "accuracy"
        assert fb.normalized_value == pytest.approx(8.5)
        assert fb.timestamp == now


class TestReputationSummary:
    """Tests for ReputationSummary model."""

    def test_create_summary(self):
        summary = ReputationSummary(agent_id=42)
        assert summary.agent_id == 42

    def test_summary_defaults(self):
        summary = ReputationSummary(agent_id=1)
        assert summary.count == 0
        assert summary.summary_value == 0
        assert summary.summary_value_decimals == 0
        assert summary.tag1 == ""
        assert summary.tag2 == ""

    def test_summary_with_count(self):
        summary = ReputationSummary(agent_id=1, count=15, summary_value=850)
        assert summary.count == 15
        assert summary.summary_value == 850

    def test_summary_normalized_value_no_decimals(self):
        summary = ReputationSummary(agent_id=1, summary_value=850, summary_value_decimals=0)
        assert summary.normalized_value == 850.0
        assert isinstance(summary.normalized_value, float)

    def test_summary_normalized_value_with_decimals(self):
        summary = ReputationSummary(agent_id=1, summary_value=85025, summary_value_decimals=2)
        assert summary.normalized_value == 850.25

    def test_summary_normalized_value_negative(self):
        summary = ReputationSummary(agent_id=1, summary_value=-200, summary_value_decimals=1)
        assert summary.normalized_value == pytest.approx(-20.0)

    def test_summary_normalized_value_zero(self):
        summary = ReputationSummary(agent_id=1, summary_value=0, summary_value_decimals=3)
        assert summary.normalized_value == 0.0

    def test_full_creation(self):
        summary = ReputationSummary(
            agent_id=10,
            count=100,
            summary_value=5000,
            summary_value_decimals=2,
            tag1="accuracy",
            tag2="math",
        )
        assert summary.count == 100
        assert summary.normalized_value == pytest.approx(50.0)


class TestValidationRecord:
    """Tests for ValidationRecord model."""

    def test_create_record(self):
        record = ValidationRecord(request_hash="abc123", agent_id=42)
        assert record.request_hash == "abc123"
        assert record.agent_id == 42

    def test_record_defaults(self):
        record = ValidationRecord(request_hash="abc123", agent_id=1)
        assert record.validator_address == ""
        assert record.request_uri == ""
        assert record.response == ValidationResponse.PENDING
        assert record.response_uri == ""
        assert record.response_hash == ""
        assert record.tag == ""
        assert record.last_update is None
        assert record.tx_hash is None

    def test_is_pending_true(self):
        record = ValidationRecord(
            request_hash="abc123", agent_id=1, response=ValidationResponse.PENDING
        )
        assert record.is_pending is True
        assert record.is_passed is False

    def test_is_passed_true(self):
        record = ValidationRecord(
            request_hash="abc123", agent_id=1, response=ValidationResponse.PASS
        )
        assert record.is_passed is True
        assert record.is_pending is False

    def test_is_passed_false_for_fail(self):
        record = ValidationRecord(
            request_hash="abc123", agent_id=1, response=ValidationResponse.FAIL
        )
        assert record.is_passed is False
        assert record.is_pending is False

    def test_is_passed_false_for_revoked(self):
        record = ValidationRecord(
            request_hash="abc123", agent_id=1, response=ValidationResponse.REVOKED
        )
        assert record.is_passed is False
        assert record.is_pending is False

    def test_record_with_validator(self):
        record = ValidationRecord(
            request_hash="abc123",
            agent_id=1,
            validator_address="0x1234567890123456789012345678901234567890",
        )
        assert record.validator_address == "0x1234567890123456789012345678901234567890"

    def test_record_with_tag(self):
        record = ValidationRecord(request_hash="abc123", agent_id=1, tag="capability")
        assert record.tag == "capability"

    def test_full_creation(self):
        now = datetime.now(timezone.utc)
        record = ValidationRecord(
            request_hash="0xreqhash",
            agent_id=42,
            validator_address="0xValidator",
            request_uri="https://example.com/request",
            response=ValidationResponse.FAIL,
            response_uri="https://example.com/response",
            response_hash="0xresphash",
            tag="security",
            last_update=now,
            tx_hash="0xtx",
        )
        assert record.is_passed is False
        assert record.is_pending is False
        assert record.tag == "security"
        assert record.last_update == now


class TestValidationSummary:
    """Tests for ValidationSummary model."""

    def test_create_summary(self):
        summary = ValidationSummary(agent_id=42)
        assert summary.agent_id == 42

    def test_summary_defaults(self):
        summary = ValidationSummary(agent_id=1)
        assert summary.count == 0
        assert summary.average_response == 0
        assert summary.tag == ""

    def test_summary_with_count(self):
        summary = ValidationSummary(agent_id=1, count=10, average_response=1)
        assert summary.count == 10
        assert summary.average_response == 1

    def test_summary_with_tag(self):
        summary = ValidationSummary(agent_id=1, count=5, tag="safety")
        assert summary.tag == "safety"


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports(self):
        from aragora.blockchain.models import __all__

        expected = [
            "MetadataEntry",
            "OnChainAgentIdentity",
            "ReputationFeedback",
            "ReputationSummary",
            "ValidationRecord",
            "ValidationResponse",
            "ValidationSummary",
        ]
        for name in expected:
            assert name in __all__, f"{name} missing from __all__"

    def test_all_count(self):
        from aragora.blockchain.models import __all__

        assert len(__all__) == 7
