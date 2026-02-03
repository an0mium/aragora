"""
Credentials for blockchain connector.

Provides configuration for connecting to ERC-8004 registries,
including RPC endpoints and contract addresses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class BlockchainCredentials:
    """Credentials and configuration for the ERC-8004 connector.

    Attributes:
        rpc_url: Ethereum RPC endpoint URL.
        chain_id: Chain ID (1 = mainnet, 11155111 = sepolia, etc.).
        identity_registry: Address of the Identity Registry contract.
        reputation_registry: Address of the Reputation Registry contract.
        validation_registry: Address of the Validation Registry contract.
        fallback_rpc_urls: Backup RPC endpoints for failover.
    """

    rpc_url: str = ""
    chain_id: int = 1
    identity_registry: str = ""
    reputation_registry: str = ""
    validation_registry: str = ""
    private_key: str = field(default="", repr=False)
    fallback_rpc_urls: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> BlockchainCredentials:
        """Create credentials from environment variables.

        Returns:
            Configured BlockchainCredentials.
        """
        fallback_raw = os.getenv("ERC8004_FALLBACK_RPC_URLS", "")
        fallback_urls = [u.strip() for u in fallback_raw.split(",") if u.strip()]

        return cls(
            rpc_url=os.getenv("ERC8004_RPC_URL", ""),
            chain_id=int(os.getenv("ERC8004_CHAIN_ID", "1")),
            identity_registry=os.getenv("ERC8004_IDENTITY_REGISTRY", ""),
            reputation_registry=os.getenv("ERC8004_REPUTATION_REGISTRY", ""),
            validation_registry=os.getenv("ERC8004_VALIDATION_REGISTRY", ""),
            private_key=os.getenv("ERC8004_WALLET_KEY", ""),
            fallback_rpc_urls=fallback_urls,
        )

    @property
    def is_configured(self) -> bool:
        """Check if credentials are sufficiently configured.

        Returns True if we have an RPC URL and at least one registry address.
        """
        has_rpc = bool(self.rpc_url)
        has_registry = bool(
            self.identity_registry or self.reputation_registry or self.validation_registry
        )
        return has_rpc and has_registry

    @property
    def has_identity_registry(self) -> bool:
        return bool(self.identity_registry)

    @property
    def has_reputation_registry(self) -> bool:
        return bool(self.reputation_registry)

    @property
    def has_validation_registry(self) -> bool:
        return bool(self.validation_registry)


__all__ = ["BlockchainCredentials"]
