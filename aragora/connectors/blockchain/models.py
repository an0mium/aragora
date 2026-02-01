"""
Models for blockchain connector evidence.

Extends the base Evidence model with blockchain-specific metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BlockchainSearchResult:
    """A search result from the blockchain connector.

    Represents a matching item from on-chain data queries.
    """

    id: str  # Format: "{type}:{chain_id}:{token_id}" e.g., "identity:1:42"
    title: str
    snippet: str
    source_url: str = ""  # Block explorer URL
    score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def evidence_type(self) -> str:
        """Extract the evidence type from the ID."""
        parts = self.id.split(":")
        return parts[0] if parts else "unknown"

    @property
    def chain_id(self) -> int:
        """Extract the chain ID from the ID."""
        parts = self.id.split(":")
        return int(parts[1]) if len(parts) > 1 else 1

    @property
    def token_id(self) -> int:
        """Extract the token ID from the ID."""
        parts = self.id.split(":")
        return int(parts[2]) if len(parts) > 2 else 0


@dataclass
class BlockchainEvidence:
    """Evidence fetched from ERC-8004 registries.

    Contains structured on-chain data about agent identity,
    reputation, or validation records.
    """

    id: str
    evidence_type: str  # "identity", "reputation", "validation"
    chain_id: int
    token_id: int

    # Core content
    title: str
    content: str
    raw_data: dict[str, Any] = field(default_factory=dict)

    # Blockchain metadata
    contract_address: str = ""
    tx_hash: str | None = None
    block_number: int | None = None
    block_explorer_url: str = ""

    # Reliability indicators
    confidence: float = 0.9  # On-chain data is highly reliable
    freshness: float = 1.0  # Current block
    authority: float = 0.85  # Blockchain is authoritative

    def to_evidence_dict(self) -> dict[str, Any]:
        """Convert to standard Evidence dictionary format."""
        return {
            "id": self.id,
            "source_type": "blockchain",
            "source_id": f"erc8004:{self.chain_id}:{self.contract_address}",
            "content": self.content,
            "title": self.title,
            "url": self.block_explorer_url,
            "confidence": self.confidence,
            "freshness": self.freshness,
            "authority": self.authority,
            "metadata": {
                "chain_id": self.chain_id,
                "token_id": self.token_id,
                "evidence_type": self.evidence_type,
                "contract_address": self.contract_address,
                "tx_hash": self.tx_hash,
                "block_number": self.block_number,
                "raw_data": self.raw_data,
            },
        }


def make_block_explorer_url(
    chain_id: int, tx_hash: str | None = None, address: str | None = None
) -> str:
    """Generate a block explorer URL for a transaction or address.

    Args:
        chain_id: Ethereum chain ID.
        tx_hash: Transaction hash (if viewing a transaction).
        address: Contract/wallet address (if viewing an address).

    Returns:
        Block explorer URL.
    """
    explorers = {
        1: "https://etherscan.io",
        11155111: "https://sepolia.etherscan.io",
        137: "https://polygonscan.com",
        42161: "https://arbiscan.io",
        8453: "https://basescan.org",
        10: "https://optimistic.etherscan.io",
        84532: "https://sepolia.basescan.org",
    }

    base_url = explorers.get(chain_id, "https://etherscan.io")

    if tx_hash:
        return f"{base_url}/tx/{tx_hash}"
    if address:
        return f"{base_url}/address/{address}"
    return base_url


__all__ = [
    "BlockchainEvidence",
    "BlockchainSearchResult",
    "make_block_explorer_url",
]
