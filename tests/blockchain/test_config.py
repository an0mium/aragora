"""
Tests for blockchain configuration module.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from aragora.blockchain.config import (
    CHAIN_ARBITRUM,
    CHAIN_BASE,
    CHAIN_BASE_SEPOLIA,
    CHAIN_ETHEREUM_MAINNET,
    CHAIN_OPTIMISM,
    CHAIN_POLYGON,
    CHAIN_SEPOLIA,
    DEFAULT_RPC_URLS,
    ChainConfig,
    get_chain_config,
    get_default_chain_config,
)


class TestChainConstants:
    """Tests for chain ID constants."""

    def test_mainnet_chain_id(self):
        assert CHAIN_ETHEREUM_MAINNET == 1

    def test_polygon_chain_id(self):
        assert CHAIN_POLYGON == 137

    def test_arbitrum_chain_id(self):
        assert CHAIN_ARBITRUM == 42161

    def test_base_chain_id(self):
        assert CHAIN_BASE == 8453

    def test_optimism_chain_id(self):
        assert CHAIN_OPTIMISM == 10

    def test_sepolia_chain_id(self):
        assert CHAIN_SEPOLIA == 11155111

    def test_base_sepolia_chain_id(self):
        assert CHAIN_BASE_SEPOLIA == 84532


class TestChainConfig:
    """Tests for ChainConfig dataclass."""

    def test_create_basic_config(self):
        config = ChainConfig(chain_id=1, rpc_url="https://eth.llamarpc.com")
        assert config.chain_id == 1
        assert config.rpc_url == "https://eth.llamarpc.com"
        assert config.identity_registry_address == ""
        assert config.reputation_registry_address == ""
        assert config.validation_registry_address == ""
        assert config.fallback_rpc_urls == []
        assert config.block_confirmations == 12
        assert config.gas_limit == 500_000

    def test_create_full_config(self):
        config = ChainConfig(
            chain_id=137,
            rpc_url="https://polygon-rpc.com",
            identity_registry_address="0x1234567890123456789012345678901234567890",
            reputation_registry_address="0xabcdef1234567890abcdef1234567890abcdef12",
            validation_registry_address="0xfedcba0987654321fedcba0987654321fedcba09",
            fallback_rpc_urls=["https://backup1.polygon.com", "https://backup2.polygon.com"],
            block_confirmations=20,
            gas_limit=1_000_000,
        )
        assert config.chain_id == 137
        assert config.has_identity_registry
        assert config.has_reputation_registry
        assert config.has_validation_registry
        assert len(config.fallback_rpc_urls) == 2
        assert config.block_confirmations == 20
        assert config.gas_limit == 1_000_000

    def test_config_is_frozen(self):
        config = ChainConfig(chain_id=1, rpc_url="https://test.rpc")
        with pytest.raises(Exception):
            config.chain_id = 2

    def test_invalid_chain_id_zero(self):
        with pytest.raises(ValueError, match="chain_id must be positive"):
            ChainConfig(chain_id=0, rpc_url="https://example.com")

    def test_invalid_chain_id_negative(self):
        with pytest.raises(ValueError, match="chain_id must be positive"):
            ChainConfig(chain_id=-1, rpc_url="https://example.com")

    def test_empty_rpc_url(self):
        with pytest.raises(ValueError, match="rpc_url is required"):
            ChainConfig(chain_id=1, rpc_url="")

    def test_negative_block_confirmations(self):
        with pytest.raises(ValueError, match="block_confirmations must be non-negative"):
            ChainConfig(chain_id=1, rpc_url="https://example.com", block_confirmations=-1)

    def test_gas_limit_too_low(self):
        with pytest.raises(ValueError, match="gas_limit must be at least 21000"):
            ChainConfig(chain_id=1, rpc_url="https://example.com", gas_limit=20000)

    def test_has_identity_registry_false(self):
        config = ChainConfig(chain_id=1, rpc_url="https://example.com")
        assert not config.has_identity_registry

    def test_has_identity_registry_true(self):
        config = ChainConfig(
            chain_id=1,
            rpc_url="https://example.com",
            identity_registry_address="0x1234567890123456789012345678901234567890",
        )
        assert config.has_identity_registry

    def test_has_reputation_registry_false(self):
        config = ChainConfig(chain_id=1, rpc_url="https://example.com")
        assert not config.has_reputation_registry

    def test_has_validation_registry_false(self):
        config = ChainConfig(chain_id=1, rpc_url="https://example.com")
        assert not config.has_validation_registry

    def test_all_rpc_urls_primary_only(self):
        config = ChainConfig(chain_id=1, rpc_url="https://primary.com")
        assert config.all_rpc_urls == ["https://primary.com"]

    def test_all_rpc_urls_with_fallbacks(self):
        config = ChainConfig(
            chain_id=1,
            rpc_url="https://primary.com",
            fallback_rpc_urls=["https://backup1.com", "https://backup2.com"],
        )
        assert config.all_rpc_urls == [
            "https://primary.com",
            "https://backup1.com",
            "https://backup2.com",
        ]


class TestGetChainConfig:
    """Tests for get_chain_config function."""

    def test_default_chain_id(self):
        env = {"ERC8004_RPC_URL": "https://custom.com"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("ERC8004_CHAIN_ID", None)
            config = get_chain_config()
            assert config.chain_id == 1

    def test_chain_id_from_env(self):
        env = {"ERC8004_CHAIN_ID": "137", "ERC8004_RPC_URL": "https://custom-polygon.com"}
        with patch.dict(os.environ, env, clear=False):
            config = get_chain_config()
            assert config.chain_id == 137
            assert config.rpc_url == "https://custom-polygon.com"

    def test_override_chain_id(self):
        env = {"ERC8004_RPC_URL": "https://custom.com"}
        with patch.dict(os.environ, env, clear=False):
            config = get_chain_config(chain_id=42161)
            assert config.chain_id == 42161

    def test_default_rpc_for_known_chain(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ERC8004_RPC_URL", None)
            os.environ.pop("ERC8004_CHAIN_ID", None)
            config = get_chain_config(chain_id=1)
            assert config.chain_id == 1
            assert "llamarpc" in config.rpc_url

    def test_no_rpc_for_unknown_chain(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ERC8004_RPC_URL", None)
            with pytest.raises(ValueError, match="No RPC URL configured"):
                get_chain_config(chain_id=99999)

    def test_registry_addresses_from_env(self):
        env = {
            "ERC8004_RPC_URL": "https://example.com",
            "ERC8004_IDENTITY_REGISTRY": "0x1111111111111111111111111111111111111111",
            "ERC8004_REPUTATION_REGISTRY": "0x2222222222222222222222222222222222222222",
            "ERC8004_VALIDATION_REGISTRY": "0x3333333333333333333333333333333333333333",
        }
        with patch.dict(os.environ, env, clear=False):
            config = get_chain_config()
            assert config.identity_registry_address == "0x1111111111111111111111111111111111111111"
            assert (
                config.reputation_registry_address == "0x2222222222222222222222222222222222222222"
            )
            assert (
                config.validation_registry_address == "0x3333333333333333333333333333333333333333"
            )

    def test_fallback_urls_from_env(self):
        env = {
            "ERC8004_RPC_URL": "https://primary.com",
            "ERC8004_FALLBACK_RPC_URLS": "https://backup1.com, https://backup2.com",
        }
        with patch.dict(os.environ, env, clear=False):
            config = get_chain_config()
            assert config.fallback_rpc_urls == ["https://backup1.com", "https://backup2.com"]

    def test_empty_fallback_urls(self):
        env = {"ERC8004_RPC_URL": "https://primary.com", "ERC8004_FALLBACK_RPC_URLS": ""}
        with patch.dict(os.environ, env, clear=False):
            config = get_chain_config()
            assert config.fallback_rpc_urls == []

    def test_block_confirmations_from_env(self):
        env = {"ERC8004_RPC_URL": "https://example.com", "ERC8004_BLOCK_CONFIRMATIONS": "6"}
        with patch.dict(os.environ, env, clear=False):
            config = get_chain_config()
            assert config.block_confirmations == 6

    def test_gas_limit_from_env(self):
        env = {"ERC8004_RPC_URL": "https://example.com", "ERC8004_GAS_LIMIT": "1000000"}
        with patch.dict(os.environ, env, clear=False):
            config = get_chain_config()
            assert config.gas_limit == 1_000_000


class TestGetDefaultChainConfig:
    """Tests for get_default_chain_config function."""

    def test_default_config_chain_id(self):
        config = get_default_chain_config()
        assert config.chain_id == CHAIN_ETHEREUM_MAINNET

    def test_default_config_rpc_url(self):
        config = get_default_chain_config()
        assert "llamarpc" in config.rpc_url

    def test_default_config_no_registries(self):
        config = get_default_chain_config()
        assert not config.has_identity_registry
        assert not config.has_reputation_registry
        assert not config.has_validation_registry

    def test_default_config_defaults(self):
        config = get_default_chain_config()
        assert config.block_confirmations == 12
        assert config.gas_limit == 500_000


class TestDefaultRPCUrls:
    """Tests for DEFAULT_RPC_URLS mapping."""

    def test_mainnet_default(self):
        assert CHAIN_ETHEREUM_MAINNET in DEFAULT_RPC_URLS
        assert "llamarpc" in DEFAULT_RPC_URLS[CHAIN_ETHEREUM_MAINNET]

    def test_sepolia_default(self):
        assert CHAIN_SEPOLIA in DEFAULT_RPC_URLS
        assert "sepolia" in DEFAULT_RPC_URLS[CHAIN_SEPOLIA].lower()

    def test_polygon_default(self):
        assert CHAIN_POLYGON in DEFAULT_RPC_URLS
        assert "polygon" in DEFAULT_RPC_URLS[CHAIN_POLYGON].lower()

    def test_base_default(self):
        assert CHAIN_BASE in DEFAULT_RPC_URLS
        assert "base" in DEFAULT_RPC_URLS[CHAIN_BASE].lower()

    def test_arbitrum_default(self):
        assert CHAIN_ARBITRUM in DEFAULT_RPC_URLS
        assert "arbitrum" in DEFAULT_RPC_URLS[CHAIN_ARBITRUM].lower()

    def test_optimism_default(self):
        assert CHAIN_OPTIMISM in DEFAULT_RPC_URLS
        assert "optimism" in DEFAULT_RPC_URLS[CHAIN_OPTIMISM].lower()

    def test_base_sepolia_default(self):
        assert CHAIN_BASE_SEPOLIA in DEFAULT_RPC_URLS
