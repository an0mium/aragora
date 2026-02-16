"""
Identity Registry contract wrapper for ERC-8004.

Wraps the IIdentityRegistry Solidity interface with typed Python methods.
Each agent is an ERC-721 token with a URI pointing to its AgentCard.

Solidity interface:
    function register(string memory agentURI, MetadataEntry[] calldata metadata) returns (uint256 agentId)
    function register(string memory agentURI) returns (uint256 agentId)
    function register() returns (uint256 agentId)
    function setAgentURI(uint256 agentId, string calldata newURI)
    function getMetadata(uint256 agentId, string memory metadataKey) returns (bytes memory)
    function setMetadata(uint256 agentId, string memory metadataKey, bytes memory metadataValue)
    function setAgentWallet(uint256 agentId, address newWallet, uint256 deadline, bytes calldata signature)
    function getAgentWallet(uint256 agentId) returns (address)
    function unsetAgentWallet(uint256 agentId)
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.blockchain.models import MetadataEntry, OnChainAgentIdentity

logger = logging.getLogger(__name__)

# Minimal ABI for the ERC-8004 Identity Registry
# Covers the IIdentityRegistry interface + ERC-721 tokenURI
IDENTITY_REGISTRY_ABI: list[dict[str, Any]] = [
    # Events
    {
        "type": "event",
        "name": "Registered",
        "inputs": [
            {"name": "agentId", "type": "uint256", "indexed": True},
            {"name": "agentURI", "type": "string", "indexed": False},
            {"name": "owner", "type": "address", "indexed": True},
        ],
    },
    {
        "type": "event",
        "name": "URIUpdated",
        "inputs": [
            {"name": "agentId", "type": "uint256", "indexed": True},
            {"name": "newURI", "type": "string", "indexed": False},
            {"name": "updatedBy", "type": "address", "indexed": True},
        ],
    },
    {
        "type": "event",
        "name": "MetadataSet",
        "inputs": [
            {"name": "agentId", "type": "uint256", "indexed": True},
            {"name": "indexedMetadataKey", "type": "string", "indexed": True},
            {"name": "metadataKey", "type": "string", "indexed": False},
            {"name": "metadataValue", "type": "bytes", "indexed": False},
        ],
    },
    # register(string, tuple[])
    {
        "type": "function",
        "name": "register",
        "inputs": [
            {"name": "agentURI", "type": "string"},
            {
                "name": "metadata",
                "type": "tuple[]",
                "components": [
                    {"name": "metadataKey", "type": "string"},
                    {"name": "metadataValue", "type": "bytes"},
                ],
            },
        ],
        "outputs": [{"name": "agentId", "type": "uint256"}],
        "stateMutability": "nonpayable",
    },
    # setAgentURI(uint256, string)
    {
        "type": "function",
        "name": "setAgentURI",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "newURI", "type": "string"},
        ],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    # getMetadata(uint256, string) -> bytes
    {
        "type": "function",
        "name": "getMetadata",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "metadataKey", "type": "string"},
        ],
        "outputs": [{"name": "", "type": "bytes"}],
        "stateMutability": "view",
    },
    # setMetadata(uint256, string, bytes)
    {
        "type": "function",
        "name": "setMetadata",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "metadataKey", "type": "string"},
            {"name": "metadataValue", "type": "bytes"},
        ],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    # setAgentWallet(uint256, address, uint256, bytes)
    {
        "type": "function",
        "name": "setAgentWallet",
        "inputs": [
            {"name": "agentId", "type": "uint256"},
            {"name": "newWallet", "type": "address"},
            {"name": "deadline", "type": "uint256"},
            {"name": "signature", "type": "bytes"},
        ],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    # getAgentWallet(uint256) -> address
    {
        "type": "function",
        "name": "getAgentWallet",
        "inputs": [{"name": "agentId", "type": "uint256"}],
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
    },
    # unsetAgentWallet(uint256)
    {
        "type": "function",
        "name": "unsetAgentWallet",
        "inputs": [{"name": "agentId", "type": "uint256"}],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    # ERC-721: tokenURI(uint256) -> string
    {
        "type": "function",
        "name": "tokenURI",
        "inputs": [{"name": "tokenId", "type": "uint256"}],
        "outputs": [{"name": "", "type": "string"}],
        "stateMutability": "view",
    },
    # ERC-721: ownerOf(uint256) -> address
    {
        "type": "function",
        "name": "ownerOf",
        "inputs": [{"name": "tokenId", "type": "uint256"}],
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
    },
    # ERC-721: totalSupply() -> uint256
    {
        "type": "function",
        "name": "totalSupply",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
]


class IdentityRegistryContract:
    """Typed wrapper around the ERC-8004 Identity Registry contract.

    Provides Python methods for registering agents, querying identities,
    and managing agent metadata on-chain.

    Usage:
        from aragora.blockchain.provider import Web3Provider
        from aragora.blockchain.wallet import WalletSigner

        provider = Web3Provider.from_env()
        signer = WalletSigner.from_env()
        registry = IdentityRegistryContract(provider)

        # Register a new agent
        token_id = registry.register_agent("https://example.com/agent.json", signer)

        # Query an agent
        identity = registry.get_agent(token_id)
    """

    def __init__(self, provider: Any, chain_id: int | None = None) -> None:
        """Initialize the Identity Registry wrapper.

        Args:
            provider: Web3Provider instance.
            chain_id: Chain to use. Defaults to provider's default.
        """
        self._provider = provider
        self._chain_id = chain_id
        self._contract: Any = None

    def _get_contract(self) -> Any:
        """Get or create the contract instance."""
        if self._contract is None:
            w3 = self._provider.get_web3(self._chain_id)
            config = self._provider.get_config(self._chain_id)
            if not config.has_identity_registry:
                raise ValueError(
                    f"No Identity Registry address configured for chain {config.chain_id}. "
                    "Set ERC8004_IDENTITY_REGISTRY environment variable."
                )
            self._contract = w3.eth.contract(
                address=w3.to_checksum_address(config.identity_registry_address),
                abi=IDENTITY_REGISTRY_ABI,
            )
        return self._contract

    def register_agent(
        self,
        agent_uri: str,
        signer: Any,
        metadata: list[MetadataEntry] | None = None,
    ) -> int:
        """Register a new agent on the Identity Registry.

        Args:
            agent_uri: URI pointing to the agent's off-chain AgentCard.
            signer: WalletSigner for signing the transaction.
            metadata: Optional on-chain metadata entries.

        Returns:
            The newly assigned agent token ID.
        """
        contract = self._get_contract()
        w3 = self._provider.get_web3(self._chain_id)
        config = self._provider.get_config(self._chain_id)

        metadata_tuples = [(m.key, m.value) for m in (metadata or [])]

        tx = contract.functions.register(agent_uri, metadata_tuples).build_transaction(
            {
                "from": signer.address,
                "gas": config.gas_limit,
                "nonce": w3.eth.get_transaction_count(signer.address),
            }
        )

        tx_hash = signer.sign_and_send(w3, tx)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        # Extract agentId from Registered event
        logs = contract.events.Registered().process_receipt(receipt)
        if logs:
            self._provider.record_success(self._chain_id)
            return logs[0]["args"]["agentId"]

        self._provider.record_success(self._chain_id)
        logger.warning("No Registered event found in receipt, returning 0")
        return 0

    def get_agent(self, token_id: int) -> OnChainAgentIdentity:
        """Get agent identity by token ID.

        Args:
            token_id: The agent's ERC-721 token ID.

        Returns:
            OnChainAgentIdentity with data from the contract.
        """
        contract = self._get_contract()
        config = self._provider.get_config(self._chain_id)

        try:
            owner = contract.functions.ownerOf(token_id).call()
            agent_uri = contract.functions.tokenURI(token_id).call()
            wallet = contract.functions.getAgentWallet(token_id).call()

            self._provider.record_success(self._chain_id)
            return OnChainAgentIdentity(
                token_id=token_id,
                owner=owner,
                agent_uri=agent_uri,
                wallet_address=wallet if wallet != "0x" + "0" * 40 else None,
                chain_id=config.chain_id,
            )
        except (RuntimeError, ConnectionError, ValueError, OSError) as e:
            self._provider.record_failure(self._chain_id)
            raise RuntimeError(f"Failed to get agent {token_id}: {e}") from e

    def set_agent_uri(self, token_id: int, new_uri: str, signer: Any) -> str:
        """Update an agent's URI.

        Args:
            token_id: Agent token ID.
            new_uri: New URI for the agent.
            signer: WalletSigner for signing.

        Returns:
            Transaction hash.
        """
        contract = self._get_contract()
        w3 = self._provider.get_web3(self._chain_id)
        config = self._provider.get_config(self._chain_id)

        tx = contract.functions.setAgentURI(token_id, new_uri).build_transaction(
            {
                "from": signer.address,
                "gas": config.gas_limit,
                "nonce": w3.eth.get_transaction_count(signer.address),
            }
        )

        tx_hash = signer.sign_and_send(w3, tx)
        self._provider.record_success(self._chain_id)
        return tx_hash

    def get_metadata(self, token_id: int, key: str) -> bytes:
        """Get on-chain metadata for an agent.

        Args:
            token_id: Agent token ID.
            key: Metadata key.

        Returns:
            Metadata value as bytes.
        """
        contract = self._get_contract()
        result = contract.functions.getMetadata(token_id, key).call()
        self._provider.record_success(self._chain_id)
        return result

    def set_metadata(self, token_id: int, key: str, value: bytes, signer: Any) -> str:
        """Set on-chain metadata for an agent.

        Args:
            token_id: Agent token ID.
            key: Metadata key.
            value: Metadata value as bytes.
            signer: WalletSigner for signing.

        Returns:
            Transaction hash.
        """
        contract = self._get_contract()
        w3 = self._provider.get_web3(self._chain_id)
        config = self._provider.get_config(self._chain_id)

        tx = contract.functions.setMetadata(token_id, key, value).build_transaction(
            {
                "from": signer.address,
                "gas": config.gas_limit,
                "nonce": w3.eth.get_transaction_count(signer.address),
            }
        )

        tx_hash = signer.sign_and_send(w3, tx)
        self._provider.record_success(self._chain_id)
        return tx_hash

    def get_total_supply(self) -> int:
        """Get the total number of registered agents.

        Returns:
            Total supply count.
        """
        contract = self._get_contract()
        result = contract.functions.totalSupply().call()
        self._provider.record_success(self._chain_id)
        return result


__all__ = [
    "IDENTITY_REGISTRY_ABI",
    "IdentityRegistryContract",
]
