"""
Data models for ERC-8004 on-chain entities.

Provides Python dataclasses mirroring the three ERC-8004 registries:
- OnChainAgentIdentity (Identity Registry)
- ReputationFeedback (Reputation Registry)
- ValidationRecord (Validation Registry)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any


class ValidationResponse(IntEnum):
    """Validation response codes from the ERC-8004 Validation Registry."""

    PENDING = 0
    PASS = 1
    FAIL = 2
    REVOKED = 3


@dataclass
class MetadataEntry:
    """A key-value metadata entry for an agent identity.

    Maps to the Solidity struct:
        struct MetadataEntry { string metadataKey; bytes metadataValue; }
    """

    key: str
    value: bytes


@dataclass
class OnChainAgentIdentity:
    """An agent identity from the ERC-8004 Identity Registry.

    Each agent is an ERC-721 token with a URI pointing to its off-chain AgentCard
    and optional key-value metadata.

    Attributes:
        token_id: ERC-721 token ID (agentId in the contract).
        owner: Ethereum address that owns the agent NFT.
        agent_uri: URI pointing to the agent's off-chain metadata (AgentCard).
        wallet_address: Optional separate wallet address for transactions.
        aragora_agent_id: Link to internal Aragora AgentInfo ID.
        chain_id: Chain where the identity is registered.
        registered_at: Block timestamp of registration.
        metadata: Off-chain metadata parsed from the agent URI.
        on_chain_metadata: On-chain key-value metadata entries.
        tx_hash: Transaction hash of the registration.
    """

    token_id: int
    owner: str
    agent_uri: str = ""
    wallet_address: str | None = None
    aragora_agent_id: str | None = None
    chain_id: int = 1
    registered_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    on_chain_metadata: list[MetadataEntry] = field(default_factory=list)
    tx_hash: str | None = None


@dataclass
class ReputationFeedback:
    """A feedback record from the ERC-8004 Reputation Registry.

    Attributes:
        agent_id: Token ID of the agent being rated.
        client_address: Ethereum address of the feedback submitter.
        feedback_index: Index of this feedback for the (agentId, client) pair.
        value: Feedback value as int128 (positive or negative).
        value_decimals: Decimal precision for the value.
        tag1: Primary category tag (e.g., "accuracy", "reliability").
        tag2: Secondary tag for finer categorization.
        endpoint: The agent endpoint/service being rated.
        feedback_uri: URI pointing to detailed off-chain feedback data.
        feedback_hash: SHA-256 hash of the off-chain feedback content.
        is_revoked: Whether this feedback has been revoked.
        timestamp: Block timestamp of the feedback submission.
        tx_hash: Transaction hash.
    """

    agent_id: int
    client_address: str
    feedback_index: int = 0
    value: int = 0
    value_decimals: int = 0
    tag1: str = ""
    tag2: str = ""
    endpoint: str = ""
    feedback_uri: str = ""
    feedback_hash: str = ""
    is_revoked: bool = False
    timestamp: datetime | None = None
    tx_hash: str | None = None

    @property
    def normalized_value(self) -> float:
        """Get the feedback value as a float with decimal precision applied."""
        if self.value_decimals == 0:
            return float(self.value)
        return self.value / (10**self.value_decimals)


@dataclass
class ReputationSummary:
    """Aggregated reputation summary for an agent.

    Returned by IReputationRegistry.getSummary().
    """

    agent_id: int
    count: int = 0
    summary_value: int = 0
    summary_value_decimals: int = 0
    tag1: str = ""
    tag2: str = ""

    @property
    def normalized_value(self) -> float:
        """Get the summary value as a float with decimal precision applied."""
        if self.summary_value_decimals == 0:
            return float(self.summary_value)
        return self.summary_value / (10**self.summary_value_decimals)


@dataclass
class ValidationRecord:
    """A validation record from the ERC-8004 Validation Registry.

    Attributes:
        request_hash: Unique hash identifying the validation request.
        agent_id: Token ID of the agent being validated.
        validator_address: Ethereum address of the validator.
        request_uri: URI pointing to the validation request details.
        response: Validation response code (0=pending, 1=pass, 2=fail, 3=revoked).
        response_uri: URI pointing to detailed response data.
        response_hash: Hash of the response content.
        tag: Category tag for the validation.
        last_update: Block timestamp of the last update.
        tx_hash: Transaction hash.
    """

    request_hash: str
    agent_id: int
    validator_address: str = ""
    request_uri: str = ""
    response: ValidationResponse = ValidationResponse.PENDING
    response_uri: str = ""
    response_hash: str = ""
    tag: str = ""
    last_update: datetime | None = None
    tx_hash: str | None = None

    @property
    def is_passed(self) -> bool:
        """Whether the validation passed."""
        return self.response == ValidationResponse.PASS

    @property
    def is_pending(self) -> bool:
        """Whether the validation is still pending."""
        return self.response == ValidationResponse.PENDING


@dataclass
class ValidationSummary:
    """Aggregated validation summary for an agent.

    Returned by IValidationRegistry.getSummary().
    """

    agent_id: int
    count: int = 0
    average_response: int = 0
    tag: str = ""


__all__ = [
    "MetadataEntry",
    "OnChainAgentIdentity",
    "ReputationFeedback",
    "ReputationSummary",
    "ValidationRecord",
    "ValidationResponse",
    "ValidationSummary",
]
