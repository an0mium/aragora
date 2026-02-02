"""
Tests for blockchain models.

Tests cover:
- BlockchainSearchResult dataclass
- BlockchainEvidence dataclass
- make_block_explorer_url utility
"""

import pytest

from aragora.connectors.blockchain.models import (
    BlockchainEvidence,
    BlockchainSearchResult,
    make_block_explorer_url,
)


class TestBlockchainSearchResult:
    """Tests for BlockchainSearchResult dataclass."""

    def test_basic_initialization(self):
        """Should initialize with required fields."""
        result = BlockchainSearchResult(
            id="identity:1:42",
            title="Agent Identity #42",
            snippet="Verified agent with 95% reputation",
        )

        assert result.id == "identity:1:42"
        assert result.title == "Agent Identity #42"
        assert result.snippet == "Verified agent with 95% reputation"

    def test_default_values(self):
        """Should have sensible defaults."""
        result = BlockchainSearchResult(
            id="test:1:1",
            title="Test",
            snippet="Snippet",
        )

        assert result.source_url == ""
        assert result.score == 1.0
        assert result.metadata == {}

    def test_custom_values(self):
        """Should accept custom values."""
        result = BlockchainSearchResult(
            id="reputation:137:100",
            title="Reputation Record",
            snippet="High reputation score",
            source_url="https://polygonscan.com/token/0x123",
            score=0.95,
            metadata={"category": "agent", "verified": True},
        )

        assert result.source_url == "https://polygonscan.com/token/0x123"
        assert result.score == 0.95
        assert result.metadata["category"] == "agent"


class TestBlockchainSearchResultProperties:
    """Tests for BlockchainSearchResult properties."""

    def test_evidence_type_identity(self):
        """Should extract identity type from ID."""
        result = BlockchainSearchResult(
            id="identity:1:42",
            title="Test",
            snippet="Test",
        )

        assert result.evidence_type == "identity"

    def test_evidence_type_reputation(self):
        """Should extract reputation type from ID."""
        result = BlockchainSearchResult(
            id="reputation:137:100",
            title="Test",
            snippet="Test",
        )

        assert result.evidence_type == "reputation"

    def test_evidence_type_validation(self):
        """Should extract validation type from ID."""
        result = BlockchainSearchResult(
            id="validation:1:500",
            title="Test",
            snippet="Test",
        )

        assert result.evidence_type == "validation"

    def test_evidence_type_empty_id(self):
        """Should handle empty ID."""
        result = BlockchainSearchResult(
            id="",
            title="Test",
            snippet="Test",
        )

        assert result.evidence_type == ""

    def test_chain_id_extraction(self):
        """Should extract chain ID from ID."""
        result = BlockchainSearchResult(
            id="identity:11155111:42",
            title="Test",
            snippet="Test",
        )

        assert result.chain_id == 11155111

    def test_chain_id_default(self):
        """Should default to 1 for missing chain ID."""
        result = BlockchainSearchResult(
            id="type",
            title="Test",
            snippet="Test",
        )

        assert result.chain_id == 1

    def test_token_id_extraction(self):
        """Should extract token ID from ID."""
        result = BlockchainSearchResult(
            id="identity:1:12345",
            title="Test",
            snippet="Test",
        )

        assert result.token_id == 12345

    def test_token_id_default(self):
        """Should default to 0 for missing token ID."""
        result = BlockchainSearchResult(
            id="type:1",
            title="Test",
            snippet="Test",
        )

        assert result.token_id == 0


class TestBlockchainEvidence:
    """Tests for BlockchainEvidence dataclass."""

    def test_basic_initialization(self):
        """Should initialize with required fields."""
        evidence = BlockchainEvidence(
            id="identity:1:42",
            evidence_type="identity",
            chain_id=1,
            token_id=42,
            title="Agent Identity",
            content="Verified agent identity record",
        )

        assert evidence.id == "identity:1:42"
        assert evidence.evidence_type == "identity"
        assert evidence.chain_id == 1
        assert evidence.token_id == 42
        assert evidence.title == "Agent Identity"
        assert evidence.content == "Verified agent identity record"

    def test_default_values(self):
        """Should have sensible defaults."""
        evidence = BlockchainEvidence(
            id="test:1:1",
            evidence_type="test",
            chain_id=1,
            token_id=1,
            title="Test",
            content="Content",
        )

        assert evidence.raw_data == {}
        assert evidence.contract_address == ""
        assert evidence.tx_hash is None
        assert evidence.block_number is None
        assert evidence.block_explorer_url == ""
        assert evidence.confidence == 0.9
        assert evidence.freshness == 1.0
        assert evidence.authority == 0.85

    def test_full_initialization(self):
        """Should accept all optional fields."""
        evidence = BlockchainEvidence(
            id="reputation:137:100",
            evidence_type="reputation",
            chain_id=137,
            token_id=100,
            title="Reputation Score",
            content="Agent has 98% reputation",
            raw_data={"score": 98, "validations": 150},
            contract_address="0xabcdef1234567890abcdef1234567890abcdef12",
            tx_hash="0x123456789abcdef123456789abcdef123456789abcdef123456789abcdef1234",
            block_number=50000000,
            block_explorer_url="https://polygonscan.com/tx/0x123...",
            confidence=0.95,
            freshness=0.8,
            authority=0.9,
        )

        assert evidence.raw_data["score"] == 98
        assert evidence.block_number == 50000000
        assert evidence.confidence == 0.95


class TestBlockchainEvidenceToDict:
    """Tests for to_evidence_dict method."""

    def test_basic_conversion(self):
        """Should convert to standard evidence dict format."""
        evidence = BlockchainEvidence(
            id="identity:1:42",
            evidence_type="identity",
            chain_id=1,
            token_id=42,
            title="Agent Identity",
            content="Verified identity",
            contract_address="0x1234567890123456789012345678901234567890",
        )

        result = evidence.to_evidence_dict()

        assert result["id"] == "identity:1:42"
        assert result["source_type"] == "blockchain"
        assert result["content"] == "Verified identity"
        assert result["title"] == "Agent Identity"

    def test_source_id_format(self):
        """Should format source_id correctly."""
        evidence = BlockchainEvidence(
            id="test:137:1",
            evidence_type="test",
            chain_id=137,
            token_id=1,
            title="Test",
            content="Test",
            contract_address="0xcontract",
        )

        result = evidence.to_evidence_dict()

        assert result["source_id"] == "erc8004:137:0xcontract"

    def test_includes_scores(self):
        """Should include confidence/freshness/authority scores."""
        evidence = BlockchainEvidence(
            id="test:1:1",
            evidence_type="test",
            chain_id=1,
            token_id=1,
            title="Test",
            content="Test",
            confidence=0.95,
            freshness=0.8,
            authority=0.85,
        )

        result = evidence.to_evidence_dict()

        assert result["confidence"] == 0.95
        assert result["freshness"] == 0.8
        assert result["authority"] == 0.85

    def test_includes_url(self):
        """Should include block explorer URL."""
        evidence = BlockchainEvidence(
            id="test:1:1",
            evidence_type="test",
            chain_id=1,
            token_id=1,
            title="Test",
            content="Test",
            block_explorer_url="https://etherscan.io/tx/0x123",
        )

        result = evidence.to_evidence_dict()

        assert result["url"] == "https://etherscan.io/tx/0x123"

    def test_metadata_structure(self):
        """Should include proper metadata structure."""
        evidence = BlockchainEvidence(
            id="identity:1:42",
            evidence_type="identity",
            chain_id=1,
            token_id=42,
            title="Test",
            content="Test",
            contract_address="0xcontract",
            tx_hash="0xtxhash",
            block_number=12345,
            raw_data={"key": "value"},
        )

        result = evidence.to_evidence_dict()
        metadata = result["metadata"]

        assert metadata["chain_id"] == 1
        assert metadata["token_id"] == 42
        assert metadata["evidence_type"] == "identity"
        assert metadata["contract_address"] == "0xcontract"
        assert metadata["tx_hash"] == "0xtxhash"
        assert metadata["block_number"] == 12345
        assert metadata["raw_data"]["key"] == "value"


class TestMakeBlockExplorerUrl:
    """Tests for make_block_explorer_url utility."""

    def test_ethereum_mainnet_tx(self):
        """Should create Etherscan URL for mainnet transaction."""
        url = make_block_explorer_url(
            chain_id=1,
            tx_hash="0x123abc",
        )

        assert url == "https://etherscan.io/tx/0x123abc"

    def test_ethereum_mainnet_address(self):
        """Should create Etherscan URL for mainnet address."""
        url = make_block_explorer_url(
            chain_id=1,
            address="0xcontract",
        )

        assert url == "https://etherscan.io/address/0xcontract"

    def test_sepolia_tx(self):
        """Should create Sepolia Etherscan URL."""
        url = make_block_explorer_url(
            chain_id=11155111,
            tx_hash="0xsepolia_tx",
        )

        assert url == "https://sepolia.etherscan.io/tx/0xsepolia_tx"

    def test_polygon_tx(self):
        """Should create Polygonscan URL."""
        url = make_block_explorer_url(
            chain_id=137,
            tx_hash="0xpolygon_tx",
        )

        assert url == "https://polygonscan.com/tx/0xpolygon_tx"

    def test_arbitrum_tx(self):
        """Should create Arbiscan URL."""
        url = make_block_explorer_url(
            chain_id=42161,
            tx_hash="0xarb_tx",
        )

        assert url == "https://arbiscan.io/tx/0xarb_tx"

    def test_base_tx(self):
        """Should create Basescan URL."""
        url = make_block_explorer_url(
            chain_id=8453,
            tx_hash="0xbase_tx",
        )

        assert url == "https://basescan.org/tx/0xbase_tx"

    def test_optimism_tx(self):
        """Should create Optimistic Etherscan URL."""
        url = make_block_explorer_url(
            chain_id=10,
            tx_hash="0xop_tx",
        )

        assert url == "https://optimistic.etherscan.io/tx/0xop_tx"

    def test_base_sepolia_address(self):
        """Should create Base Sepolia URL."""
        url = make_block_explorer_url(
            chain_id=84532,
            address="0xbase_sepolia_addr",
        )

        assert url == "https://sepolia.basescan.org/address/0xbase_sepolia_addr"

    def test_unknown_chain_defaults_to_etherscan(self):
        """Should default to Etherscan for unknown chains."""
        url = make_block_explorer_url(
            chain_id=99999,  # Unknown chain
            tx_hash="0xunknown_tx",
        )

        assert url == "https://etherscan.io/tx/0xunknown_tx"

    def test_no_tx_or_address(self):
        """Should return base URL when neither tx nor address provided."""
        url = make_block_explorer_url(chain_id=1)

        assert url == "https://etherscan.io"

    def test_tx_takes_precedence(self):
        """Should prefer tx_hash over address when both provided."""
        url = make_block_explorer_url(
            chain_id=1,
            tx_hash="0xtx",
            address="0xaddr",
        )

        assert url == "https://etherscan.io/tx/0xtx"


class TestBlockchainModelsAll:
    """Tests for module __all__ export."""

    def test_exports(self):
        """Should export correct classes and functions."""
        from aragora.connectors.blockchain import models

        assert hasattr(models, "BlockchainEvidence")
        assert hasattr(models, "BlockchainSearchResult")
        assert hasattr(models, "make_block_explorer_url")
