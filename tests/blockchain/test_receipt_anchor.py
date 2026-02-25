"""
Tests for ReceiptAnchor -- on-chain receipt anchoring.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import pytest

from aragora.blockchain.receipt_anchor import (
    AnchorRecord,
    ReceiptAnchor,
)


class TestReceiptAnchorLocal:
    """Tests for local (no blockchain) receipt anchoring."""

    @pytest.fixture
    def anchor(self) -> ReceiptAnchor:
        """Create a ReceiptAnchor in local mode."""
        return ReceiptAnchor()

    @pytest.mark.asyncio
    async def test_anchor_locally(self, anchor: ReceiptAnchor):
        """Anchoring without provider creates a local anchor."""
        receipt_hash = hashlib.sha256(b"test receipt").hexdigest()
        result = await anchor.anchor_receipt(receipt_hash, metadata={"debate_id": "d1"})
        assert result.startswith("local:")
        assert len(result) > 10

    @pytest.mark.asyncio
    async def test_anchor_locally_returns_unique_ids(self, anchor: ReceiptAnchor):
        """Each local anchor gets a unique ID."""
        h1 = hashlib.sha256(b"receipt1").hexdigest()
        h2 = hashlib.sha256(b"receipt2").hexdigest()
        id1 = await anchor.anchor_receipt(h1)
        id2 = await anchor.anchor_receipt(h2)
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_get_anchors_all(self, anchor: ReceiptAnchor):
        """get_anchors with no filter returns all anchors."""
        h1 = hashlib.sha256(b"r1").hexdigest()
        h2 = hashlib.sha256(b"r2").hexdigest()
        await anchor.anchor_receipt(h1)
        await anchor.anchor_receipt(h2)
        all_anchors = anchor.get_anchors()
        assert len(all_anchors) == 2

    @pytest.mark.asyncio
    async def test_get_anchors_filtered(self, anchor: ReceiptAnchor):
        """get_anchors with receipt_hash filters correctly."""
        h1 = hashlib.sha256(b"r1").hexdigest()
        h2 = hashlib.sha256(b"r2").hexdigest()
        await anchor.anchor_receipt(h1)
        await anchor.anchor_receipt(h2)
        filtered = anchor.get_anchors(receipt_hash=h1)
        assert len(filtered) == 1
        assert filtered[0].receipt_hash == h1

    @pytest.mark.asyncio
    async def test_local_anchor_record_fields(self, anchor: ReceiptAnchor):
        """Local anchor records have correct field values."""
        receipt_hash = hashlib.sha256(b"test").hexdigest()
        await anchor.anchor_receipt(receipt_hash, metadata={"key": "val"})
        records = anchor.get_anchors(receipt_hash)
        assert len(records) == 1
        record = records[0]
        assert record.local_only is True
        assert record.tx_hash is None
        assert record.chain_id is None
        assert record.metadata == {"key": "val"}
        assert record.timestamp > 0


class TestReceiptAnchorOnChain:
    """Tests for on-chain receipt anchoring with mocked provider."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock Web3Provider."""
        provider = MagicMock()
        mock_w3 = MagicMock()
        mock_contract = MagicMock()

        mock_config = MagicMock()
        mock_config.chain_id = 1
        mock_config.has_validation_registry = True
        mock_config.validation_registry_address = "0x3333333333333333333333333333333333333333"
        mock_config.gas_limit = 500_000

        provider.get_web3.return_value = mock_w3
        provider.get_config.return_value = mock_config
        mock_w3.to_checksum_address.side_effect = lambda x: x
        mock_w3.eth.contract.return_value = mock_contract
        mock_w3.eth.get_transaction_count.return_value = 0

        mock_contract.functions.validationRequest.return_value.build_transaction.return_value = {
            "to": "0x3333333333333333333333333333333333333333",
            "data": "0x...",
            "gas": 100_000,
            "nonce": 0,
        }

        return provider, mock_contract

    @pytest.fixture
    def mock_signer(self):
        """Create a mock signer."""
        signer = MagicMock()
        signer.address = "0xABCDEF1234567890ABCDEF1234567890ABCDEF12"
        signer.sign_and_send.return_value = "0x" + "ab" * 32
        return signer

    @pytest.mark.asyncio
    async def test_anchor_on_chain(self, mock_provider, mock_signer):
        """Anchoring with provider submits to chain."""
        provider, mock_contract = mock_provider
        anchor = ReceiptAnchor(provider=provider, chain_id=1)

        receipt_hash = hashlib.sha256(b"test receipt").hexdigest()
        tx_hash = await anchor.anchor_receipt(
            receipt_hash,
            metadata={"debate_id": "d1"},
            signer=mock_signer,
        )

        assert tx_hash == "0x" + "ab" * 32
        mock_contract.functions.validationRequest.assert_called_once()
        provider.record_success.assert_called()

    @pytest.mark.asyncio
    async def test_anchor_on_chain_failure_fallback(self, mock_signer):
        """When on-chain anchoring fails, falls back to local."""
        provider = MagicMock()
        provider.get_web3.side_effect = ConnectionError("No connection")

        anchor = ReceiptAnchor(provider=provider, chain_id=1)
        receipt_hash = hashlib.sha256(b"test").hexdigest()
        result = await anchor.anchor_receipt(receipt_hash, signer=mock_signer)

        assert result.startswith("local:")


class TestAnchorRecord:
    """Tests for the AnchorRecord dataclass."""

    def test_default_values(self):
        """AnchorRecord has sensible defaults."""
        record = AnchorRecord(receipt_hash="abc123")
        assert record.tx_hash is None
        assert record.chain_id is None
        assert record.timestamp == 0.0
        assert record.metadata == {}
        assert record.local_only is False

    def test_with_all_fields(self):
        """AnchorRecord stores all fields correctly."""
        record = AnchorRecord(
            receipt_hash="abc123",
            tx_hash="0xdef456",
            chain_id=1,
            timestamp=1234567890.0,
            metadata={"key": "val"},
            local_only=False,
        )
        assert record.receipt_hash == "abc123"
        assert record.tx_hash == "0xdef456"
        assert record.chain_id == 1
