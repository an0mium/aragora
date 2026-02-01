"""
Blockchain configuration for ERC-8004 integration.

Provides chain configuration, contract addresses, and environment variable
overrides for connecting to Ethereum-compatible networks.

Environment variables:
    ERC8004_RPC_URL: Primary RPC endpoint
    ERC8004_CHAIN_ID: Chain ID (default: 1)
    ERC8004_IDENTITY_REGISTRY: Identity registry contract address
    ERC8004_REPUTATION_REGISTRY: Reputation registry contract address
    ERC8004_VALIDATION_REGISTRY: Validation registry contract address
    ERC8004_FALLBACK_RPC_URLS: Comma-separated fallback RPC URLs
    ERC8004_BLOCK_CONFIRMATIONS: Block confirmations required (default: 12)
    ERC8004_GAS_LIMIT: Gas limit for transactions (default: 500000)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Well-known chain IDs
CHAIN_ETHEREUM_MAINNET = 1
CHAIN_POLYGON = 137
CHAIN_ARBITRUM = 42161
CHAIN_BASE = 8453
CHAIN_OPTIMISM = 10
CHAIN_SEPOLIA = 11155111
CHAIN_BASE_SEPOLIA = 84532
CHAIN_LINEA_SEPOLIA = 59141

# Default public RPC endpoints (rate-limited, for development only)
DEFAULT_RPC_URLS: dict[int, str] = {
    CHAIN_ETHEREUM_MAINNET: "https://eth.llamarpc.com",
    CHAIN_SEPOLIA: "https://rpc.sepolia.org",
    CHAIN_BASE_SEPOLIA: "https://sepolia.base.org",
    CHAIN_POLYGON: "https://polygon-rpc.com",
    CHAIN_ARBITRUM: "https://arb1.arbitrum.io/rpc",
    CHAIN_BASE: "https://mainnet.base.org",
    CHAIN_OPTIMISM: "https://mainnet.optimism.io",
}


@dataclass(frozen=True)
class ChainConfig:
    """Configuration for connecting to an ERC-8004 deployment on a specific chain.

    Attributes:
        chain_id: Ethereum chain ID.
        rpc_url: Primary JSON-RPC endpoint URL.
        identity_registry_address: Deployed Identity Registry contract address.
        reputation_registry_address: Deployed Reputation Registry contract address.
        validation_registry_address: Deployed Validation Registry contract address.
        fallback_rpc_urls: Additional RPC endpoints for failover.
        block_confirmations: Blocks to wait before considering a tx finalized.
        gas_limit: Default gas limit for write transactions.
    """

    chain_id: int
    rpc_url: str
    identity_registry_address: str = ""
    reputation_registry_address: str = ""
    validation_registry_address: str = ""
    fallback_rpc_urls: list[str] = field(default_factory=list)
    block_confirmations: int = 12
    gas_limit: int = 500_000

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.chain_id < 1:
            raise ValueError(f"chain_id must be positive, got {self.chain_id}")
        if not self.rpc_url:
            raise ValueError("rpc_url is required")
        if self.block_confirmations < 0:
            raise ValueError("block_confirmations must be non-negative")
        if self.gas_limit < 21_000:
            raise ValueError("gas_limit must be at least 21000 (base tx cost)")

    @property
    def has_identity_registry(self) -> bool:
        """Whether an identity registry address is configured."""
        return bool(self.identity_registry_address)

    @property
    def has_reputation_registry(self) -> bool:
        """Whether a reputation registry address is configured."""
        return bool(self.reputation_registry_address)

    @property
    def has_validation_registry(self) -> bool:
        """Whether a validation registry address is configured."""
        return bool(self.validation_registry_address)

    @property
    def all_rpc_urls(self) -> list[str]:
        """All RPC URLs including fallbacks, primary first."""
        return [self.rpc_url, *self.fallback_rpc_urls]


def get_chain_config(chain_id: int | None = None) -> ChainConfig:
    """Build a ChainConfig from environment variables.

    Args:
        chain_id: Override chain ID. If None, reads from ERC8004_CHAIN_ID env var.

    Returns:
        ChainConfig populated from environment variables.
    """
    resolved_chain_id = chain_id or int(os.getenv("ERC8004_CHAIN_ID", "1"))

    rpc_url = os.getenv(
        "ERC8004_RPC_URL",
        DEFAULT_RPC_URLS.get(resolved_chain_id, ""),
    )
    if not rpc_url:
        raise ValueError(
            f"No RPC URL configured for chain {resolved_chain_id}. "
            "Set ERC8004_RPC_URL environment variable."
        )

    fallback_raw = os.getenv("ERC8004_FALLBACK_RPC_URLS", "")
    fallback_urls = [u.strip() for u in fallback_raw.split(",") if u.strip()]

    return ChainConfig(
        chain_id=resolved_chain_id,
        rpc_url=rpc_url,
        identity_registry_address=os.getenv("ERC8004_IDENTITY_REGISTRY", ""),
        reputation_registry_address=os.getenv("ERC8004_REPUTATION_REGISTRY", ""),
        validation_registry_address=os.getenv("ERC8004_VALIDATION_REGISTRY", ""),
        fallback_rpc_urls=fallback_urls,
        block_confirmations=int(os.getenv("ERC8004_BLOCK_CONFIRMATIONS", "12")),
        gas_limit=int(os.getenv("ERC8004_GAS_LIMIT", "500000")),
    )


def get_default_chain_config() -> ChainConfig:
    """Get a default config for Ethereum mainnet (read-only, no contract addresses).

    Useful for testing connectivity without requiring deployed contracts.
    """
    return ChainConfig(
        chain_id=CHAIN_ETHEREUM_MAINNET,
        rpc_url=DEFAULT_RPC_URLS[CHAIN_ETHEREUM_MAINNET],
    )
