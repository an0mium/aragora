"""
ERC-8004 Connector - Evidence source for on-chain agent data.

Implements the BaseConnector interface for retrieving agent identity,
reputation, and validation records from ERC-8004 registries.

Search query formats:
- "agent:{token_id}" - Search by agent token ID
- "owner:{address}" - Search by owner address
- "reputation:{token_id}" - Get reputation for an agent
- "validation:{token_id}" - Get validations for an agent

Fetch ID formats:
- "identity:{chain_id}:{token_id}" - Fetch agent identity
- "reputation:{chain_id}:{token_id}:{client}:{index}" - Fetch feedback
- "validation:{chain_id}:{request_hash}" - Fetch validation record
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.connectors.base import BaseConnector, Evidence
from aragora.connectors.blockchain.credentials import BlockchainCredentials
from aragora.connectors.blockchain.models import (
    BlockchainEvidence,
    BlockchainSearchResult,
    make_block_explorer_url,
)

logger = logging.getLogger(__name__)

# Check for web3 availability
_web3_available: bool | None = None


def _check_web3() -> bool:
    global _web3_available
    if _web3_available is None:
        try:
            import web3  # noqa: F401

            _web3_available = True
        except ImportError:
            _web3_available = False
    return _web3_available


class ERC8004Connector(BaseConnector):
    """Connector for ERC-8004 on-chain agent data.

    Retrieves agent identities, reputation feedback, and validation
    records from deployed ERC-8004 registry contracts.

    Usage:
        connector = ERC8004Connector.from_env()

        # Search for an agent
        results = await connector.search("agent:42")

        # Fetch agent identity
        evidence = await connector.fetch("identity:1:42")
    """

    @property
    def name(self) -> str:
        return "erc8004"

    @property
    def source_type(self) -> Any:
        # Return string for now; will integrate with SourceType.BLOCKCHAIN after Phase 5
        return "blockchain"

    @property
    def is_available(self) -> bool:
        return _check_web3()

    @property
    def is_configured(self) -> bool:
        return self._credentials.is_configured

    def __init__(
        self,
        credentials: BlockchainCredentials | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ERC-8004 connector.

        Args:
            credentials: Blockchain credentials. Defaults to from_env().
            **kwargs: Additional BaseConnector arguments.
        """
        super().__init__(**kwargs)
        self._credentials = credentials or BlockchainCredentials.from_env()
        self._provider: Any = None
        self._identity_contract: Any = None
        self._reputation_contract: Any = None
        self._validation_contract: Any = None

    @classmethod
    def from_env(cls, **kwargs: Any) -> ERC8004Connector:
        """Create a connector from environment variables.

        Returns:
            Configured ERC8004Connector.
        """
        return cls(credentials=BlockchainCredentials.from_env(), **kwargs)

    def _get_provider(self) -> Any:
        """Lazy-initialize the Web3 provider."""
        if self._provider is None:
            if not _check_web3():
                raise ImportError(
                    "web3 is required for blockchain connector. "
                    "Install with: pip install aragora[blockchain]"
                )
            from aragora.blockchain.config import ChainConfig
            from aragora.blockchain.provider import Web3Provider

            config = ChainConfig(
                chain_id=self._credentials.chain_id,
                rpc_url=self._credentials.rpc_url,
                identity_registry_address=self._credentials.identity_registry,
                reputation_registry_address=self._credentials.reputation_registry,
                validation_registry_address=self._credentials.validation_registry,
                fallback_rpc_urls=self._credentials.fallback_rpc_urls,
            )
            self._provider = Web3Provider.from_config(config)
        return self._provider

    def _get_identity_contract(self) -> Any:
        """Lazy-initialize the Identity Registry contract."""
        if self._identity_contract is None:
            from aragora.blockchain.contracts.identity import IdentityRegistryContract

            self._identity_contract = IdentityRegistryContract(self._get_provider())
        return self._identity_contract

    def _get_reputation_contract(self) -> Any:
        """Lazy-initialize the Reputation Registry contract."""
        if self._reputation_contract is None:
            from aragora.blockchain.contracts.reputation import ReputationRegistryContract

            self._reputation_contract = ReputationRegistryContract(self._get_provider())
        return self._reputation_contract

    def _get_validation_contract(self) -> Any:
        """Lazy-initialize the Validation Registry contract."""
        if self._validation_contract is None:
            from aragora.blockchain.contracts.validation import ValidationRegistryContract

            self._validation_contract = ValidationRegistryContract(self._get_provider())
        return self._validation_contract

    async def _perform_health_check(self, timeout: float) -> bool:
        """Check blockchain connectivity."""
        try:
            provider = self._get_provider()
            return provider.is_connected()
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

    async def search(  # type: ignore[override]  # blockchain connector returns BlockchainSearchResult instead of Evidence
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> list[BlockchainSearchResult]:
        """Search for on-chain agent data.

        Query formats:
        - "agent:{token_id}" - Get agent by ID
        - "owner:{address}" - Get agents by owner
        - "reputation:{token_id}" - Get reputation summary
        - "validation:{token_id}" - Get validation summary

        Args:
            query: Search query string.
            max_results: Maximum results to return.

        Returns:
            List of BlockchainSearchResult objects.
        """
        results: list[BlockchainSearchResult] = []

        parts = query.lower().split(":")
        query_type = parts[0] if parts else ""

        try:
            if query_type == "agent" and len(parts) > 1:
                token_id = int(parts[1])
                identity = self._get_identity_contract().get_agent(token_id)
                results.append(
                    BlockchainSearchResult(
                        id=f"identity:{self._credentials.chain_id}:{token_id}",
                        title=f"Agent #{token_id}",
                        snippet=f"Owner: {identity.owner[:10]}... URI: {identity.agent_uri[:50]}...",
                        source_url=make_block_explorer_url(
                            self._credentials.chain_id,
                            address=self._credentials.identity_registry,
                        ),
                        metadata={"owner": identity.owner, "uri": identity.agent_uri},
                    )
                )

            elif query_type == "reputation" and len(parts) > 1:
                token_id = int(parts[1])
                if self._credentials.has_reputation_registry:
                    summary = self._get_reputation_contract().get_summary(token_id)
                    results.append(
                        BlockchainSearchResult(
                            id=f"reputation:{self._credentials.chain_id}:{token_id}",
                            title=f"Reputation for Agent #{token_id}",
                            snippet=f"Count: {summary.count}, Score: {summary.normalized_value:.2f}",
                            source_url=make_block_explorer_url(
                                self._credentials.chain_id,
                                address=self._credentials.reputation_registry,
                            ),
                            metadata={
                                "count": summary.count,
                                "value": summary.summary_value,
                            },
                        )
                    )

            elif query_type == "validation" and len(parts) > 1:
                token_id = int(parts[1])
                if self._credentials.has_validation_registry:
                    summary = self._get_validation_contract().get_summary(token_id)
                    results.append(
                        BlockchainSearchResult(
                            id=f"validation:{self._credentials.chain_id}:{token_id}",
                            title=f"Validations for Agent #{token_id}",
                            snippet=f"Count: {summary.count}, Avg Response: {summary.average_response}",
                            source_url=make_block_explorer_url(
                                self._credentials.chain_id,
                                address=self._credentials.validation_registry,
                            ),
                            metadata={
                                "count": summary.count,
                                "average_response": summary.average_response,
                            },
                        )
                    )

        except Exception as e:
            logger.error(f"Search error for query '{query}': {e}")

        return results[:max_results]

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch specific evidence by ID.

        ID formats:
        - "identity:{chain_id}:{token_id}"
        - "reputation:{chain_id}:{token_id}"
        - "validation:{chain_id}:{request_hash}"

        Args:
            evidence_id: Evidence identifier.

        Returns:
            Evidence object or None if not found.
        """
        # Check cache first
        cached = self._cache_get(evidence_id)
        if cached:
            return cached

        parts = evidence_id.split(":")
        if len(parts) < 3:
            logger.warning(f"Invalid evidence ID format: {evidence_id}")
            return None

        evidence_type = parts[0]
        chain_id = int(parts[1])

        try:
            if evidence_type == "identity":
                token_id = int(parts[2])
                identity = self._get_identity_contract().get_agent(token_id)

                content = (
                    f"Agent #{token_id}\n"
                    f"Owner: {identity.owner}\n"
                    f"URI: {identity.agent_uri}\n"
                    f"Wallet: {identity.wallet_address or 'Not set'}\n"
                    f"Chain: {chain_id}"
                )

                blockchain_evidence = BlockchainEvidence(
                    id=evidence_id,
                    evidence_type="identity",
                    chain_id=chain_id,
                    token_id=token_id,
                    title=f"Agent #{token_id} Identity",
                    content=content,
                    raw_data={
                        "token_id": identity.token_id,
                        "owner": identity.owner,
                        "agent_uri": identity.agent_uri,
                        "wallet_address": identity.wallet_address,
                    },
                    contract_address=self._credentials.identity_registry,
                    tx_hash=identity.tx_hash,
                    block_explorer_url=make_block_explorer_url(
                        chain_id, address=self._credentials.identity_registry
                    ),
                )

                evidence = self._to_evidence(blockchain_evidence)
                self._cache_set(evidence_id, evidence)
                return evidence

            elif evidence_type == "reputation":
                token_id = int(parts[2])
                summary = self._get_reputation_contract().get_summary(token_id)

                content = (
                    f"Reputation Summary for Agent #{token_id}\n"
                    f"Feedback Count: {summary.count}\n"
                    f"Aggregate Score: {summary.normalized_value:.4f}\n"
                    f"Tag1: {summary.tag1 or 'All'}\n"
                    f"Tag2: {summary.tag2 or 'All'}"
                )

                blockchain_evidence = BlockchainEvidence(
                    id=evidence_id,
                    evidence_type="reputation",
                    chain_id=chain_id,
                    token_id=token_id,
                    title=f"Reputation for Agent #{token_id}",
                    content=content,
                    raw_data={
                        "agent_id": summary.agent_id,
                        "count": summary.count,
                        "summary_value": summary.summary_value,
                        "summary_value_decimals": summary.summary_value_decimals,
                    },
                    contract_address=self._credentials.reputation_registry,
                    block_explorer_url=make_block_explorer_url(
                        chain_id, address=self._credentials.reputation_registry
                    ),
                )

                evidence = self._to_evidence(blockchain_evidence)
                self._cache_set(evidence_id, evidence)
                return evidence

            elif evidence_type == "validation":
                # For validation, parts[2] is the request hash
                request_hash = bytes.fromhex(parts[2].replace("0x", ""))
                record = self._get_validation_contract().get_validation_status(request_hash)

                content = (
                    f"Validation Record\n"
                    f"Agent: #{record.agent_id}\n"
                    f"Validator: {record.validator_address}\n"
                    f"Response: {record.response.name}\n"
                    f"Tag: {record.tag or 'None'}"
                )

                blockchain_evidence = BlockchainEvidence(
                    id=evidence_id,
                    evidence_type="validation",
                    chain_id=chain_id,
                    token_id=record.agent_id,
                    title=f"Validation for Agent #{record.agent_id}",
                    content=content,
                    raw_data={
                        "request_hash": record.request_hash,
                        "agent_id": record.agent_id,
                        "validator_address": record.validator_address,
                        "response": record.response.value,
                        "tag": record.tag,
                    },
                    contract_address=self._credentials.validation_registry,
                    tx_hash=record.tx_hash,
                    block_explorer_url=make_block_explorer_url(
                        chain_id, address=self._credentials.validation_registry
                    ),
                )

                evidence = self._to_evidence(blockchain_evidence)
                self._cache_set(evidence_id, evidence)
                return evidence

        except Exception as e:
            logger.error(f"Fetch error for '{evidence_id}': {e}")

        return None

    def _to_evidence(self, blockchain_evidence: BlockchainEvidence) -> Evidence:
        """Convert BlockchainEvidence to standard Evidence."""
        # Import here to avoid circular dependency issues
        source_type: Any = "blockchain"  # default fallback as string
        try:
            from aragora.reasoning.provenance import SourceType

            source_type = SourceType.EXTERNAL_API  # Use EXTERNAL_API until BLOCKCHAIN is added
        except ImportError:
            pass  # source_type stays as fallback string

        return Evidence(
            id=blockchain_evidence.id,
            source_type=source_type,
            source_id=f"erc8004:{blockchain_evidence.chain_id}:{blockchain_evidence.contract_address}",
            content=blockchain_evidence.content,
            title=blockchain_evidence.title,
            url=blockchain_evidence.block_explorer_url,
            confidence=blockchain_evidence.confidence,
            freshness=blockchain_evidence.freshness,
            authority=blockchain_evidence.authority,
            metadata=blockchain_evidence.raw_data,
        )

    def _cache_set(self, key: str, evidence: Evidence) -> None:
        """Add to cache with TTL."""
        import time

        self._cache[key] = (time.time(), evidence)
        # Prune if needed
        while len(self._cache) > self._max_cache_entries:
            self._cache.popitem(last=False)


__all__ = ["ERC8004Connector"]
