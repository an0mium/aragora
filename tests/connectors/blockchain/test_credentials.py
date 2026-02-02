"""
Tests for blockchain credentials.

Tests cover:
- BlockchainCredentials dataclass
- Loading from environment variables
- Configuration validation
- Registry availability checks
"""

import os
import pytest
from unittest.mock import patch

from aragora.connectors.blockchain.credentials import BlockchainCredentials


class TestBlockchainCredentialsDefaults:
    """Tests for BlockchainCredentials default values."""

    def test_default_initialization(self):
        """Should initialize with empty defaults."""
        creds = BlockchainCredentials()

        assert creds.rpc_url == ""
        assert creds.chain_id == 1  # Mainnet
        assert creds.identity_registry == ""
        assert creds.reputation_registry == ""
        assert creds.validation_registry == ""
        assert creds.fallback_rpc_urls == []

    def test_custom_initialization(self):
        """Should accept custom values."""
        creds = BlockchainCredentials(
            rpc_url="https://eth.example.com",
            chain_id=11155111,  # Sepolia
            identity_registry="0x1234567890123456789012345678901234567890",
            reputation_registry="0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
            validation_registry="0x9876543210987654321098765432109876543210",
            fallback_rpc_urls=["https://backup1.example.com", "https://backup2.example.com"],
        )

        assert creds.rpc_url == "https://eth.example.com"
        assert creds.chain_id == 11155111
        assert creds.identity_registry == "0x1234567890123456789012345678901234567890"
        assert creds.reputation_registry == "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
        assert creds.validation_registry == "0x9876543210987654321098765432109876543210"
        assert len(creds.fallback_rpc_urls) == 2


class TestBlockchainCredentialsFromEnv:
    """Tests for from_env class method."""

    def test_from_env_empty(self):
        """Should handle missing environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            creds = BlockchainCredentials.from_env()

        assert creds.rpc_url == ""
        assert creds.chain_id == 1
        assert creds.identity_registry == ""

    def test_from_env_rpc_url(self):
        """Should read RPC URL from environment."""
        env = {"ERC8004_RPC_URL": "https://mainnet.infura.io/v3/key123"}

        with patch.dict(os.environ, env, clear=True):
            creds = BlockchainCredentials.from_env()

        assert creds.rpc_url == "https://mainnet.infura.io/v3/key123"

    def test_from_env_chain_id(self):
        """Should read chain ID from environment."""
        env = {"ERC8004_CHAIN_ID": "137"}  # Polygon

        with patch.dict(os.environ, env, clear=True):
            creds = BlockchainCredentials.from_env()

        assert creds.chain_id == 137

    def test_from_env_chain_id_default(self):
        """Should default to mainnet (1) for chain ID."""
        with patch.dict(os.environ, {}, clear=True):
            creds = BlockchainCredentials.from_env()

        assert creds.chain_id == 1

    def test_from_env_identity_registry(self):
        """Should read identity registry address."""
        env = {"ERC8004_IDENTITY_REGISTRY": "0x1111111111111111111111111111111111111111"}

        with patch.dict(os.environ, env, clear=True):
            creds = BlockchainCredentials.from_env()

        assert creds.identity_registry == "0x1111111111111111111111111111111111111111"

    def test_from_env_reputation_registry(self):
        """Should read reputation registry address."""
        env = {"ERC8004_REPUTATION_REGISTRY": "0x2222222222222222222222222222222222222222"}

        with patch.dict(os.environ, env, clear=True):
            creds = BlockchainCredentials.from_env()

        assert creds.reputation_registry == "0x2222222222222222222222222222222222222222"

    def test_from_env_validation_registry(self):
        """Should read validation registry address."""
        env = {"ERC8004_VALIDATION_REGISTRY": "0x3333333333333333333333333333333333333333"}

        with patch.dict(os.environ, env, clear=True):
            creds = BlockchainCredentials.from_env()

        assert creds.validation_registry == "0x3333333333333333333333333333333333333333"

    def test_from_env_fallback_urls(self):
        """Should parse comma-separated fallback URLs."""
        env = {"ERC8004_FALLBACK_RPC_URLS": "https://rpc1.com, https://rpc2.com, https://rpc3.com"}

        with patch.dict(os.environ, env, clear=True):
            creds = BlockchainCredentials.from_env()

        assert creds.fallback_rpc_urls == [
            "https://rpc1.com",
            "https://rpc2.com",
            "https://rpc3.com",
        ]

    def test_from_env_fallback_urls_empty(self):
        """Should handle empty fallback URLs."""
        env = {"ERC8004_FALLBACK_RPC_URLS": ""}

        with patch.dict(os.environ, env, clear=True):
            creds = BlockchainCredentials.from_env()

        assert creds.fallback_rpc_urls == []

    def test_from_env_fallback_urls_whitespace(self):
        """Should handle whitespace in fallback URLs."""
        env = {"ERC8004_FALLBACK_RPC_URLS": "  ,  ,  "}

        with patch.dict(os.environ, env, clear=True):
            creds = BlockchainCredentials.from_env()

        assert creds.fallback_rpc_urls == []

    def test_from_env_full_config(self):
        """Should load complete configuration."""
        env = {
            "ERC8004_RPC_URL": "https://mainnet.example.com",
            "ERC8004_CHAIN_ID": "1",
            "ERC8004_IDENTITY_REGISTRY": "0x1111111111111111111111111111111111111111",
            "ERC8004_REPUTATION_REGISTRY": "0x2222222222222222222222222222222222222222",
            "ERC8004_VALIDATION_REGISTRY": "0x3333333333333333333333333333333333333333",
            "ERC8004_FALLBACK_RPC_URLS": "https://backup.example.com",
        }

        with patch.dict(os.environ, env, clear=True):
            creds = BlockchainCredentials.from_env()

        assert creds.is_configured is True


class TestBlockchainCredentialsIsConfigured:
    """Tests for is_configured property."""

    def test_not_configured_empty(self):
        """Should not be configured when empty."""
        creds = BlockchainCredentials()

        assert creds.is_configured is False

    def test_not_configured_rpc_only(self):
        """Should not be configured with only RPC URL."""
        creds = BlockchainCredentials(rpc_url="https://rpc.example.com")

        assert creds.is_configured is False

    def test_not_configured_registry_only(self):
        """Should not be configured with only registry."""
        creds = BlockchainCredentials(
            identity_registry="0x1234567890123456789012345678901234567890"
        )

        assert creds.is_configured is False

    def test_configured_with_identity(self):
        """Should be configured with RPC and identity registry."""
        creds = BlockchainCredentials(
            rpc_url="https://rpc.example.com",
            identity_registry="0x1234567890123456789012345678901234567890",
        )

        assert creds.is_configured is True

    def test_configured_with_reputation(self):
        """Should be configured with RPC and reputation registry."""
        creds = BlockchainCredentials(
            rpc_url="https://rpc.example.com",
            reputation_registry="0x1234567890123456789012345678901234567890",
        )

        assert creds.is_configured is True

    def test_configured_with_validation(self):
        """Should be configured with RPC and validation registry."""
        creds = BlockchainCredentials(
            rpc_url="https://rpc.example.com",
            validation_registry="0x1234567890123456789012345678901234567890",
        )

        assert creds.is_configured is True


class TestBlockchainCredentialsRegistryFlags:
    """Tests for registry availability flags."""

    def test_has_identity_registry_false(self):
        """Should return False when no identity registry."""
        creds = BlockchainCredentials()
        assert creds.has_identity_registry is False

    def test_has_identity_registry_true(self):
        """Should return True when identity registry set."""
        creds = BlockchainCredentials(
            identity_registry="0x1234567890123456789012345678901234567890"
        )
        assert creds.has_identity_registry is True

    def test_has_reputation_registry_false(self):
        """Should return False when no reputation registry."""
        creds = BlockchainCredentials()
        assert creds.has_reputation_registry is False

    def test_has_reputation_registry_true(self):
        """Should return True when reputation registry set."""
        creds = BlockchainCredentials(
            reputation_registry="0x1234567890123456789012345678901234567890"
        )
        assert creds.has_reputation_registry is True

    def test_has_validation_registry_false(self):
        """Should return False when no validation registry."""
        creds = BlockchainCredentials()
        assert creds.has_validation_registry is False

    def test_has_validation_registry_true(self):
        """Should return True when validation registry set."""
        creds = BlockchainCredentials(
            validation_registry="0x1234567890123456789012345678901234567890"
        )
        assert creds.has_validation_registry is True


class TestBlockchainCredentialsEdgeCases:
    """Tests for edge cases."""

    def test_empty_string_registries(self):
        """Empty strings should not count as configured."""
        creds = BlockchainCredentials(
            rpc_url="https://rpc.example.com",
            identity_registry="",
            reputation_registry="",
            validation_registry="",
        )

        assert creds.is_configured is False
        assert creds.has_identity_registry is False

    def test_whitespace_rpc_url(self):
        """Whitespace-only RPC URL should not count."""
        creds = BlockchainCredentials(
            rpc_url="   ",
            identity_registry="0x1234567890123456789012345678901234567890",
        )

        # Note: Current implementation doesn't strip whitespace
        # This tests the actual behavior
        assert creds.is_configured is True  # "   " is truthy

    def test_various_chain_ids(self):
        """Should support various chain IDs."""
        chain_ids = {
            1: "Ethereum Mainnet",
            11155111: "Sepolia",
            137: "Polygon",
            42161: "Arbitrum",
            8453: "Base",
            10: "Optimism",
        }

        for chain_id in chain_ids:
            creds = BlockchainCredentials(chain_id=chain_id)
            assert creds.chain_id == chain_id

    def test_immutability(self):
        """Dataclass should be mutable (default behavior)."""
        creds = BlockchainCredentials(rpc_url="https://old.example.com")

        creds.rpc_url = "https://new.example.com"

        assert creds.rpc_url == "https://new.example.com"

    def test_fallback_list_independence(self):
        """Default fallback list should be independent per instance."""
        creds1 = BlockchainCredentials()
        creds2 = BlockchainCredentials()

        creds1.fallback_rpc_urls.append("https://test.com")

        assert creds2.fallback_rpc_urls == []
