"""
Reputation Registry contract wrapper for ERC-8004.

Wraps the IReputationRegistry Solidity interface with typed Python methods.
Handles on-chain feedback submission, revocation, and aggregated summaries.

Key functions:
    giveFeedback(agentId, value, valueDecimals, tag1, tag2, endpoint, feedbackURI, feedbackHash)
    revokeFeedback(agentId, feedbackIndex)
    getSummary(agentId, clientAddresses, tag1, tag2) -> (count, summaryValue, summaryValueDecimals)
    readFeedback(agentId, clientAddress, feedbackIndex) -> (value, valueDecimals, tag1, tag2, isRevoked)
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.blockchain.models import ReputationFeedback, ReputationSummary

logger = logging.getLogger(__name__)

REPUTATION_REGISTRY_ABI: list[dict[str, Any]] = [
    # Events
    {
        "type": "event",
        "name": "NewFeedback",
        "inputs": [
            {"name": "agentId", "type": "uint256", "indexed": True},
            {"name": "clientAddress", "type": "address", "indexed": True},
            {"name": "feedbackIndex", "type": "uint64", "indexed": False},
            {"name": "value", "type": "int128", "indexed": False},
            {"name": "valueDecimals", "type": "uint8", "indexed": False},
            {"name": "indexedTag1", "type": "string", "indexed": True},
            {"name": "tag1", "type": "string", "indexed": False},
            {"name": "tag2", "type": "string", "indexed": False},
            {"name": "endpoint", "type": "string", "indexed": False},
            {"name": "feedbackURI", "type": "string", "indexed": False},
            {"name": "feedbackHash", "type": "bytes32", "indexed": False},
        ],
    },
    {
        "type": "event",
        "name": "FeedbackRevoked",
        "inputs": [
            {"name": "agentId", "type": "uint256", "indexed": True},
            {"name": "clientAddress", "type": "address", "indexed": True},
            {"name": "feedbackIndex", "type": "uint64", "indexed": True},
        ],
    },
    {
        "type": "event",
        "name": "ResponseAppended",
        "inputs": [
            {"name": "agentId", "type": "uint256", "indexed": True},
            {"name": "clientAddress", "type": "address", "indexed": True},
            {"name": "feedbackIndex", "type": "uint64", "indexed": False},
            {"name": "responder", "type": "address", "indexed": True},
            {"name": "responseURI", "type": "string", "indexed": False},
            {"name": "responseHash", "type": "bytes32", "indexed": False},
        ],
    },
    # giveFeedback
    {
        "type": "function",
        "name": "giveFeedback",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "value", "type": "int128"},
            {"name": "valueDecimals", "type": "uint8"},
            {"name": "tag1", "type": "string"},
            {"name": "tag2", "type": "string"},
            {"name": "endpoint", "type": "string"},
            {"name": "feedbackURI", "type": "string"},
            {"name": "feedbackHash", "type": "bytes32"},
        ],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    # revokeFeedback
    {
        "type": "function",
        "name": "revokeFeedback",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "feedbackIndex", "type": "uint64"},
        ],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    # appendResponse
    {
        "type": "function",
        "name": "appendResponse",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "clientAddress", "type": "address"},
            {"name": "feedbackIndex", "type": "uint64"},
            {"name": "responseURI", "type": "string"},
            {"name": "responseHash", "type": "bytes32"},
        ],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    # getSummary
    {
        "type": "function",
        "name": "getSummary",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "clientAddresses", "type": "address[]"},
            {"name": "tag1", "type": "string"},
            {"name": "tag2", "type": "string"},
        ],
        "outputs": [
            {"name": "count", "type": "uint64"},
            {"name": "summaryValue", "type": "int128"},
            {"name": "summaryValueDecimals", "type": "uint8"},
        ],
        "stateMutability": "view",
    },
    # readFeedback
    {
        "type": "function",
        "name": "readFeedback",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "clientAddress", "type": "address"},
            {"name": "feedbackIndex", "type": "uint64"},
        ],
        "outputs": [
            {"name": "value", "type": "int128"},
            {"name": "valueDecimals", "type": "uint8"},
            {"name": "tag1", "type": "string"},
            {"name": "tag2", "type": "string"},
            {"name": "isRevoked", "type": "bool"},
        ],
        "stateMutability": "view",
    },
    # getClients
    {
        "type": "function",
        "name": "getClients",
        "inputs": [{"name": "agentId", "type": "uint256"}],
        "outputs": [{"name": "", "type": "address[]"}],
        "stateMutability": "view",
    },
    # getLastIndex
    {
        "type": "function",
        "name": "getLastIndex",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "clientAddress", "type": "address"},
        ],
        "outputs": [{"name": "", "type": "uint64"}],
        "stateMutability": "view",
    },
    # getIdentityRegistry
    {
        "type": "function",
        "name": "getIdentityRegistry",
        "inputs": [],
        "outputs": [{"name": "identityRegistry", "type": "address"}],
        "stateMutability": "view",
    },
]


class ReputationRegistryContract:
    """Typed wrapper around the ERC-8004 Reputation Registry contract.

    Provides methods for submitting feedback, querying reputation summaries,
    and reading individual feedback records.
    """

    def __init__(self, provider: Any, chain_id: int | None = None) -> None:
        self._provider = provider
        self._chain_id = chain_id
        self._contract: Any = None

    def _get_contract(self) -> Any:
        if self._contract is None:
            w3 = self._provider.get_web3(self._chain_id)
            config = self._provider.get_config(self._chain_id)
            if not config.has_reputation_registry:
                raise ValueError(
                    f"No Reputation Registry address configured for chain {config.chain_id}. "
                    "Set ERC8004_REPUTATION_REGISTRY environment variable."
                )
            self._contract = w3.eth.contract(
                address=w3.to_checksum_address(config.reputation_registry_address),
                abi=REPUTATION_REGISTRY_ABI,
            )
        return self._contract

    def give_feedback(
        self,
        agent_id: int,
        value: int,
        signer: Any,
        value_decimals: int = 0,
        tag1: str = "",
        tag2: str = "",
        endpoint: str = "",
        feedback_uri: str = "",
        feedback_hash: bytes = b"\x00" * 32,
    ) -> str:
        """Submit feedback for an agent.

        Args:
            agent_id: Token ID of the agent to rate.
            value: Feedback value (int128, positive or negative).
            signer: WalletSigner for signing.
            value_decimals: Decimal precision for the value.
            tag1: Primary category tag.
            tag2: Secondary tag.
            endpoint: Agent endpoint being rated.
            feedback_uri: URI to detailed feedback data.
            feedback_hash: Hash of the feedback content.

        Returns:
            Transaction hash.
        """
        contract = self._get_contract()
        w3 = self._provider.get_web3(self._chain_id)
        config = self._provider.get_config(self._chain_id)

        tx = contract.functions.giveFeedback(
            agent_id, value, value_decimals, tag1, tag2, endpoint, feedback_uri, feedback_hash
        ).build_transaction(
            {
                "from": signer.address,
                "gas": config.gas_limit,
                "nonce": w3.eth.get_transaction_count(signer.address),
            }
        )

        tx_hash = signer.sign_and_send(w3, tx)
        self._provider.record_success(self._chain_id)
        return tx_hash

    def revoke_feedback(self, agent_id: int, feedback_index: int, signer: Any) -> str:
        """Revoke a previously submitted feedback.

        Args:
            agent_id: Token ID of the agent.
            feedback_index: Index of the feedback to revoke.
            signer: WalletSigner for signing.

        Returns:
            Transaction hash.
        """
        contract = self._get_contract()
        w3 = self._provider.get_web3(self._chain_id)
        config = self._provider.get_config(self._chain_id)

        tx = contract.functions.revokeFeedback(agent_id, feedback_index).build_transaction(
            {
                "from": signer.address,
                "gas": config.gas_limit,
                "nonce": w3.eth.get_transaction_count(signer.address),
            }
        )

        tx_hash = signer.sign_and_send(w3, tx)
        self._provider.record_success(self._chain_id)
        return tx_hash

    def get_summary(
        self,
        agent_id: int,
        client_addresses: list[str] | None = None,
        tag1: str = "",
        tag2: str = "",
    ) -> ReputationSummary:
        """Get aggregated reputation summary for an agent.

        Args:
            agent_id: Token ID of the agent.
            client_addresses: Filter by specific clients. None = all.
            tag1: Filter by primary tag.
            tag2: Filter by secondary tag.

        Returns:
            ReputationSummary with count and aggregated value.
        """
        contract = self._get_contract()
        addresses = client_addresses or []

        count, summary_value, summary_decimals = contract.functions.getSummary(
            agent_id, addresses, tag1, tag2
        ).call()

        self._provider.record_success(self._chain_id)
        return ReputationSummary(
            agent_id=agent_id,
            count=count,
            summary_value=summary_value,
            summary_value_decimals=summary_decimals,
            tag1=tag1,
            tag2=tag2,
        )

    def read_feedback(
        self,
        agent_id: int,
        client_address: str,
        feedback_index: int,
    ) -> ReputationFeedback:
        """Read a single feedback record.

        Args:
            agent_id: Token ID of the agent.
            client_address: Address of the feedback submitter.
            feedback_index: Index of the feedback.

        Returns:
            ReputationFeedback record.
        """
        contract = self._get_contract()

        value, value_decimals, tag1, tag2, is_revoked = contract.functions.readFeedback(
            agent_id, client_address, feedback_index
        ).call()

        self._provider.record_success(self._chain_id)
        return ReputationFeedback(
            agent_id=agent_id,
            client_address=client_address,
            feedback_index=feedback_index,
            value=value,
            value_decimals=value_decimals,
            tag1=tag1,
            tag2=tag2,
            is_revoked=is_revoked,
        )

    def get_clients(self, agent_id: int) -> list[str]:
        """Get all client addresses that have submitted feedback for an agent.

        Args:
            agent_id: Token ID of the agent.

        Returns:
            List of client Ethereum addresses.
        """
        contract = self._get_contract()
        result = contract.functions.getClients(agent_id).call()
        self._provider.record_success(self._chain_id)
        return result

    def get_last_index(self, agent_id: int, client_address: str) -> int:
        """Get the last feedback index for an agent-client pair.

        Args:
            agent_id: Token ID of the agent.
            client_address: Client address.

        Returns:
            Last feedback index.
        """
        contract = self._get_contract()
        result = contract.functions.getLastIndex(agent_id, client_address).call()
        self._provider.record_success(self._chain_id)
        return result


__all__ = [
    "REPUTATION_REGISTRY_ABI",
    "ReputationRegistryContract",
]
