"""Tests for ReceiptSettlementService -- receipt-to-blockchain anchoring integration.

Tests cover:
1. Receipt anchoring (local mode, no chain configured)
2. Settlement status population on receipts
3. Graceful skip when no chain is configured
4. Identity bridge integration for agent reputation
5. Round-trip serialization of settlement_status
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.blockchain.receipt_anchor import AnchorRecord, ReceiptAnchor
from aragora.blockchain.receipt_settlement import (
    ReceiptSettlementService,
    SettlementResult,
    anchor_receipt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeConsensusProof:
    """Minimal consensus proof for testing."""

    reached: bool = True
    confidence: float = 0.9
    supporting_agents: list[str] = field(default_factory=lambda: ["claude", "gpt-6-astra"])
    dissenting_agents: list[str] = field(default_factory=lambda: ["gemini"])


@dataclass
class FakeReceipt:
    """Minimal DecisionReceipt for testing."""

    receipt_id: str = "test-receipt-001"
    gauntlet_id: str = "gauntlet-001"
    timestamp: str = "2026-03-23T00:00:00Z"
    input_summary: str = "Test decision"
    input_hash: str = "abc123"
    risk_summary: dict = field(default_factory=dict)
    verdict: str = "PASS"
    confidence: float = 0.95
    robustness_score: float = 0.9
    artifact_hash: str = ""
    settlement_status: dict[str, Any] | None = None
    settlement_metadata: dict[str, Any] | None = None
    consensus_proof: FakeConsensusProof | None = None

    def __post_init__(self) -> None:
        if not self.artifact_hash:
            content = json.dumps(
                {
                    "receipt_id": self.receipt_id,
                    "gauntlet_id": self.gauntlet_id,
                    "input_hash": self.input_hash,
                    "verdict": self.verdict,
                    "confidence": self.confidence,
                },
                sort_keys=True,
            )
            self.artifact_hash = hashlib.sha256(content.encode()).hexdigest()


@pytest.fixture
def receipt() -> FakeReceipt:
    """Create a fake receipt for testing."""
    return FakeReceipt(consensus_proof=FakeConsensusProof())


@pytest.fixture
def service() -> ReceiptSettlementService:
    """Create a ReceiptSettlementService in local mode (no provider)."""
    return ReceiptSettlementService()


# ---------------------------------------------------------------------------
# SettlementResult tests
# ---------------------------------------------------------------------------


class TestSettlementResult:
    """Tests for the SettlementResult dataclass."""

    def test_default_values(self):
        result = SettlementResult()
        assert result.anchored is False
        assert result.local_only is True
        assert result.chain_id is None
        assert result.block_number is None
        assert result.error is None

    def test_to_dict_minimal(self):
        result = SettlementResult(
            anchored=True,
            anchor_id="local:abc123",
            receipt_hash="deadbeef",
            timestamp=1234567890.0,
        )
        d = result.to_dict()
        assert d["anchored"] is True
        assert d["anchor_id"] == "local:abc123"
        assert d["local_only"] is True
        assert d["receipt_hash"] == "deadbeef"
        assert d["timestamp"] == 1234567890.0
        assert "chain_id" not in d
        assert "block_number" not in d
        assert "error" not in d

    def test_to_dict_with_chain(self):
        result = SettlementResult(
            anchored=True,
            anchor_id="0xdeadbeef",
            chain_id=11155111,
            block_number=42,
            local_only=False,
            receipt_hash="cafebabe",
            timestamp=1234567890.0,
        )
        d = result.to_dict()
        assert d["chain_id"] == 11155111
        assert d["block_number"] == 42
        assert d["local_only"] is False

    def test_to_dict_with_error(self):
        result = SettlementResult(
            anchored=False,
            error="Connection refused",
            receipt_hash="abc",
        )
        d = result.to_dict()
        assert d["anchored"] is False
        assert d["error"] == "Connection refused"


# ---------------------------------------------------------------------------
# ReceiptSettlementService -- local mode
# ---------------------------------------------------------------------------


class TestReceiptSettlementLocal:
    """Tests for ReceiptSettlementService in local mode (no blockchain)."""

    @pytest.mark.asyncio
    async def test_anchor_receipt_local(self, service, receipt):
        """Anchoring without provider creates a local anchor."""
        result = await service.anchor_receipt(receipt)
        assert result is receipt
        assert receipt.settlement_status is not None
        assert receipt.settlement_status["anchored"] is True
        assert receipt.settlement_status["local_only"] is True
        assert receipt.settlement_status["anchor_id"].startswith("local:")
        assert receipt.settlement_status["receipt_hash"] == receipt.artifact_hash

    @pytest.mark.asyncio
    async def test_anchor_receipt_populates_timestamp(self, service, receipt):
        """Anchoring sets a timestamp."""
        await service.anchor_receipt(receipt)
        assert receipt.settlement_status["timestamp"] > 0

    @pytest.mark.asyncio
    async def test_anchor_receipt_no_chain_id_for_local(self, service, receipt):
        """Local anchors have no chain_id."""
        await service.anchor_receipt(receipt)
        assert "chain_id" not in receipt.settlement_status

    @pytest.mark.asyncio
    async def test_anchor_receipt_computes_hash_if_missing(self, service):
        """If artifact_hash is empty, computes one."""
        receipt = FakeReceipt(artifact_hash="")
        # artifact_hash gets set in __post_init__
        assert receipt.artifact_hash != ""
        await service.anchor_receipt(receipt)
        assert receipt.settlement_status["anchored"] is True

    @pytest.mark.asyncio
    async def test_get_settlement_status_from_receipt(self, service, receipt):
        """get_settlement_status returns stored settlement_status."""
        await service.anchor_receipt(receipt)
        status = service.get_settlement_status(receipt)
        assert status["anchored"] is True
        assert status["local_only"] is True

    @pytest.mark.asyncio
    async def test_get_settlement_status_unanchored(self, service, receipt):
        """get_settlement_status for unanchored receipt returns from verify_anchor."""
        status = service.get_settlement_status(receipt)
        # Receipt hasn't been anchored yet; verify_anchor returns "not found"
        assert status["anchored"] is False

    @pytest.mark.asyncio
    async def test_anchor_receipt_idempotent(self, service, receipt):
        """Anchoring the same receipt twice produces two separate statuses."""
        await service.anchor_receipt(receipt)
        first_status = dict(receipt.settlement_status)
        await service.anchor_receipt(receipt)
        second_status = receipt.settlement_status
        # Both should be anchored but may have different anchor_ids (local is time-based)
        assert first_status["anchored"] is True
        assert second_status["anchored"] is True


# ---------------------------------------------------------------------------
# ReceiptSettlementService -- graceful degradation
# ---------------------------------------------------------------------------


class TestReceiptSettlementGraceful:
    """Tests for graceful degradation when components fail."""

    @pytest.mark.asyncio
    async def test_anchor_receipt_handles_anchor_failure(self, receipt):
        """If ReceiptAnchor raises, settlement_status records the error."""
        service = ReceiptSettlementService()
        # Monkey-patch the anchor to raise
        original = service._anchor.anchor_receipt

        async def failing_anchor(*args, **kwargs):
            raise ConnectionError("No RPC available")

        service._anchor.anchor_receipt = failing_anchor

        result = await service.anchor_receipt(receipt)
        assert result is receipt
        assert receipt.settlement_status is not None
        assert receipt.settlement_status["anchored"] is False
        assert "error" in receipt.settlement_status

    @pytest.mark.asyncio
    async def test_push_reputation_no_bridge(self, service, receipt):
        """push_agent_reputation returns 0 when no bridge is available."""
        count = await service.push_agent_reputation(receipt, agent_elo_ratings={"claude": 1800.0})
        assert count == 0

    @pytest.mark.asyncio
    async def test_push_reputation_no_ratings(self, service, receipt):
        """push_agent_reputation returns 0 when no ELO ratings provided."""
        count = await service.push_agent_reputation(receipt, agent_elo_ratings=None)
        assert count == 0


# ---------------------------------------------------------------------------
# ReceiptSettlementService -- agent reputation
# ---------------------------------------------------------------------------


class TestAgentReputation:
    """Tests for agent reputation pushing via identity bridge."""

    @pytest.mark.asyncio
    async def test_push_reputation_with_linked_agent(self, receipt):
        """Pushing reputation for a linked agent calls push_reputation."""
        mock_bridge = MagicMock()
        mock_link = MagicMock()
        mock_link.token_id = 42
        mock_link.chain_id = 1
        mock_bridge.get_link.return_value = mock_link

        mock_adapter = MagicMock()

        service = ReceiptSettlementService(identity_bridge=mock_bridge)

        with patch.object(service, "_get_erc8004_adapter", return_value=mock_adapter):
            count = await service.push_agent_reputation(
                receipt, agent_elo_ratings={"claude": 1800.0, "gpt-6-astra": 1750.0}
            )

        # Both claude and gpt-6-astra are in supporting_agents, gemini is dissenting
        # but we only provided ELO for claude and gpt-6-astra
        assert count >= 1
        assert mock_adapter.push_reputation.called

    @pytest.mark.asyncio
    async def test_push_reputation_skips_unlinked_agents(self, receipt):
        """Agents not linked to blockchain identity are skipped."""
        mock_bridge = MagicMock()
        mock_bridge.get_link.return_value = None  # No link

        service = ReceiptSettlementService(identity_bridge=mock_bridge)

        count = await service.push_agent_reputation(receipt, agent_elo_ratings={"claude": 1800.0})
        assert count == 0


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestConvenienceFunction:
    """Tests for the module-level anchor_receipt convenience function."""

    @pytest.mark.asyncio
    async def test_anchor_receipt_convenience(self, receipt):
        """The module-level anchor_receipt function works."""
        result = await anchor_receipt(receipt)
        assert result is receipt
        assert receipt.settlement_status is not None
        assert receipt.settlement_status["anchored"] is True


# ---------------------------------------------------------------------------
# Integration with DecisionReceipt
# ---------------------------------------------------------------------------


class TestDecisionReceiptIntegration:
    """Tests for settlement_status field on the real DecisionReceipt."""

    def test_settlement_status_field_exists(self):
        """DecisionReceipt has a settlement_status field."""
        from aragora.gauntlet.receipt_models import DecisionReceipt

        receipt = DecisionReceipt(
            receipt_id="test",
            gauntlet_id="g1",
            timestamp="2026-01-01T00:00:00Z",
            input_summary="test",
            input_hash="abc",
            risk_summary={},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.8,
        )
        assert receipt.settlement_status is None

    def test_settlement_status_in_to_dict(self):
        """settlement_status appears in to_dict output."""
        from aragora.gauntlet.receipt_models import DecisionReceipt

        receipt = DecisionReceipt(
            receipt_id="test",
            gauntlet_id="g1",
            timestamp="2026-01-01T00:00:00Z",
            input_summary="test",
            input_hash="abc",
            risk_summary={},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.8,
            settlement_status={
                "anchored": True,
                "anchor_id": "local:abc",
                "local_only": True,
            },
        )
        d = receipt.to_dict()
        assert "settlement_status" in d
        assert d["settlement_status"]["anchored"] is True

    def test_settlement_status_round_trip(self):
        """settlement_status survives to_dict -> from_dict round trip."""
        from aragora.gauntlet.receipt_models import DecisionReceipt

        original = DecisionReceipt(
            receipt_id="test",
            gauntlet_id="g1",
            timestamp="2026-01-01T00:00:00Z",
            input_summary="test",
            input_hash="abc",
            risk_summary={},
            attacks_attempted=0,
            attacks_successful=0,
            probes_run=0,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.8,
            settlement_status={
                "anchored": True,
                "anchor_id": "local:test123",
                "local_only": True,
                "chain_id": None,
                "receipt_hash": "deadbeef",
            },
        )
        d = original.to_dict()
        restored = DecisionReceipt.from_dict(d)
        assert restored.settlement_status == original.settlement_status

    @pytest.mark.asyncio
    async def test_anchor_real_receipt(self):
        """Anchoring a real DecisionReceipt works end-to-end."""
        from aragora.gauntlet.receipt_models import DecisionReceipt

        receipt = DecisionReceipt(
            receipt_id="real-test",
            gauntlet_id="g1",
            timestamp="2026-01-01T00:00:00Z",
            input_summary="test decision",
            input_hash="abc123",
            risk_summary={"Critical": 0, "High": 0, "Medium": 1, "Low": 2},
            attacks_attempted=5,
            attacks_successful=1,
            probes_run=3,
            vulnerabilities_found=3,
            verdict="CONDITIONAL",
            confidence=0.85,
            robustness_score=0.8,
        )
        service = ReceiptSettlementService()
        result = await service.anchor_receipt(receipt)
        assert result is receipt
        assert receipt.settlement_status is not None
        assert receipt.settlement_status["anchored"] is True
        assert receipt.settlement_status["local_only"] is True
        assert receipt.settlement_status["receipt_hash"] == receipt.artifact_hash


# ---------------------------------------------------------------------------
# PostDebateCoordinator integration
# ---------------------------------------------------------------------------


class TestPostDebateCoordinatorConfig:
    """Tests that the auto_anchor_receipt config option exists."""

    def test_config_default_false(self):
        """auto_anchor_receipt defaults to False."""
        from aragora.debate.post_debate_coordinator import PostDebateConfig

        config = PostDebateConfig()
        assert config.auto_anchor_receipt is False

    def test_config_can_enable(self):
        """auto_anchor_receipt can be enabled."""
        from aragora.debate.post_debate_coordinator import PostDebateConfig

        config = PostDebateConfig(auto_anchor_receipt=True)
        assert config.auto_anchor_receipt is True

    def test_result_has_settlement_field(self):
        """PostDebateResult has a receipt_settlement field."""
        from aragora.debate.post_debate_coordinator import PostDebateResult

        result = PostDebateResult()
        assert result.receipt_settlement is None
