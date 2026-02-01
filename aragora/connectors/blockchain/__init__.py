"""
Blockchain connector for ERC-8004 on-chain agent data.

Provides an evidence connector that retrieves agent identities,
reputation records, and validation status from ERC-8004 registries.

Usage:
    from aragora.connectors.blockchain import ERC8004Connector

    connector = ERC8004Connector.from_env()
    results = await connector.search("agent:123")
    evidence = await connector.fetch("identity:1:42")
"""

from __future__ import annotations

from aragora.connectors.blockchain.connector import ERC8004Connector
from aragora.connectors.blockchain.credentials import BlockchainCredentials
from aragora.connectors.blockchain.models import (
    BlockchainEvidence,
    BlockchainSearchResult,
)

__all__ = [
    "BlockchainCredentials",
    "BlockchainEvidence",
    "BlockchainSearchResult",
    "ERC8004Connector",
]
