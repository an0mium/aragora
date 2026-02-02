"""
Blockchain integration for ERC-8004 Trustless Autonomous Agents.

Provides multi-chain Ethereum connectivity for agent identity, reputation,
and validation registries defined by ERC-8004.

Usage:
    from aragora.blockchain import Web3Provider, OnChainAgentIdentity
    from aragora.blockchain.config import get_chain_config

    provider = Web3Provider.from_env()
    identity = provider.identity_registry.get_agent(token_id=42)

Requires optional dependencies:
    pip install aragora[blockchain]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.blockchain.provider import Web3Provider
    from aragora.blockchain.wallet import WalletSigner

from aragora.blockchain.config import (
    ChainConfig,
    get_chain_config,
    get_default_chain_config,
)
from aragora.blockchain.models import (
    OnChainAgentIdentity,
    ReputationFeedback,
    ValidationRecord,
)

__all__ = [
    "ChainConfig",
    "OnChainAgentIdentity",
    "ReputationFeedback",
    "ValidationRecord",
    "get_chain_config",
    "get_default_chain_config",
]

# Lazy imports for optional web3 dependency
_provider_cls = None
_wallet_cls = None


def get_web3_provider(**kwargs: Any) -> Web3Provider:
    """Get a Web3Provider instance (lazy import to avoid requiring web3)."""
    global _provider_cls
    if _provider_cls is None:
        from aragora.blockchain.provider import Web3Provider as _Web3Provider

        _provider_cls = _Web3Provider
    return _provider_cls(**kwargs)


def get_wallet_signer(**kwargs: Any) -> WalletSigner:
    """Get a WalletSigner instance (lazy import to avoid requiring web3)."""
    global _wallet_cls
    if _wallet_cls is None:
        from aragora.blockchain.wallet import WalletSigner as _WalletSigner

        _wallet_cls = _WalletSigner
    return _wallet_cls(**kwargs)
