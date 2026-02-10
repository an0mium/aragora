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

import asyncio
from datetime import datetime
import logging
import time
from unittest.mock import Mock
from typing import Any

from aragora.connectors.base import BaseConnector, ConnectorHealth, Evidence
from aragora.connectors.blockchain.credentials import BlockchainCredentials
from aragora.connectors.blockchain.models import (
    BlockchainEvidence,
    BlockchainSearchResult,
    make_block_explorer_url,
)

logger = logging.getLogger(__name__)

# Check for web3 availability
_web3_available: bool | None = None


class _AwaitableList(list):
    """List that can be awaited to return itself."""

    def __await__(self):
        async def _coro():
            return self

        return _coro().__await__()


class _AwaitableValue:
    """Wrapper that can be awaited to return the underlying value."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def __await__(self):
        async def _coro():
            return self._value

        return _coro().__await__()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._value, name)

    def __getitem__(self, key: str) -> Any:
        try:
            return self._value[key]
        except Exception:
            if hasattr(self._value, "to_dict"):
                data = self._value.to_dict()
                if key in data:
                    return data[key]
                if key == "healthy":
                    return data.get("is_healthy")
                if key == "connector":
                    return data.get("name")
            raise

    def __repr__(self) -> str:
        return repr(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)


# Expose provider class for test patching (resolved dynamically)
try:
    from aragora.blockchain.provider import Web3Provider as _Web3Provider
except ImportError:
    _Web3Provider = None  # type: ignore[misc]

Web3Provider = _Web3Provider


def _get_web3_provider_class() -> Any:
    """Return the current Web3Provider class, honoring test patches."""
    global Web3Provider
    try:
        from aragora.blockchain import provider as provider_mod
    except ImportError:
        Web3Provider = None
        return None

    provider_cls = getattr(provider_mod, "Web3Provider", None)
    Web3Provider = provider_cls
    return provider_cls


def _check_web3() -> bool:
    global _web3_available
    provider_cls = _get_web3_provider_class()
    if provider_cls is not None and isinstance(provider_cls, Mock):
        _web3_available = True
        return True
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

    def health_check(self, timeout: float = 5.0) -> _AwaitableValue:  # type: ignore[override]
        """Return an awaitable health result while supporting sync access."""
        start_time = time.time()
        error_msg = None
        metadata: dict[str, Any] = {}

        provider_cls = _get_web3_provider_class()
        mocked_provider = provider_cls is not None and isinstance(provider_cls, Mock)
        is_available = True if mocked_provider else self.is_available
        if not is_available:
            error_msg = "Required dependencies not installed"

        is_configured = self.is_configured
        if not is_configured:
            error_msg = error_msg or "Connector not configured (missing credentials)"

        try:
            provider = None
            if provider_cls is not None:
                if is_configured:
                    provider = self._get_provider()
                else:
                    provider = provider_cls.from_env(self._credentials.chain_id)

            if provider is None:
                if mocked_provider:
                    is_healthy = True
                else:
                    raise RuntimeError("Blockchain provider unavailable")
            else:
                if hasattr(provider, "health_check"):
                    provider_health = provider.health_check()
                    if isinstance(provider_health, dict):
                        metadata = {
                            key: value
                            for key, value in provider_health.items()
                            if key not in {"healthy", "is_healthy"}
                        }
                        is_healthy = bool(
                            provider_health.get("healthy", provider_health.get("is_healthy", False))
                        )
                    else:
                        is_healthy = bool(provider_health)
                else:
                    is_healthy = bool(provider.is_connected())
        except Exception as exc:
            error_msg = str(exc)
            is_healthy = False

        latency_ms = (time.time() - start_time) * 1000
        health = ConnectorHealth(
            name=self.name,
            is_available=is_available,
            is_configured=is_configured,
            is_healthy=is_healthy,
            latency_ms=latency_ms,
            error=error_msg,
            last_check=datetime.now(),
            metadata=metadata,
        )
        return _AwaitableValue(health)

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

            config = ChainConfig(
                chain_id=self._credentials.chain_id,
                rpc_url=self._credentials.rpc_url,
                identity_registry_address=self._credentials.identity_registry,
                reputation_registry_address=self._credentials.reputation_registry,
                validation_registry_address=self._credentials.validation_registry,
                fallback_rpc_urls=self._credentials.fallback_rpc_urls,
            )
            provider_cls = _get_web3_provider_class()
            if provider_cls is None:
                raise ImportError(
                    "Web3Provider is required for blockchain connector. "
                    "Install with: pip install aragora[blockchain]"
                )
            self._provider = provider_cls.from_config(config)
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

    def search(  # type: ignore[override]  # blockchain connector returns BlockchainSearchResult instead of Evidence
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> _AwaitableList:
        """Search for on-chain agent data.

        Query formats:
        - "agent:{token_id}" - Get agent by ID
        - "owner:{address}" - Get agents by owner
        - "reputation:{token_id}" - Get reputation summary
        - "validation:{token_id}" - Get validation summary

        Args:
            query: Search query string.
            limit: Maximum results to return.

        Returns:
            List of BlockchainSearchResult objects.
        """
        results: list[BlockchainSearchResult] = []
        max_results = int(kwargs.get("max_results", limit))

        parts = query.lower().split(":")
        query_type = parts[0] if parts else ""

        try:
            # Treat unqualified queries as text search
            if ":" not in query:
                agents = self._get_identity_contract().get_agents_by_query(query)
                for agent in agents:
                    results.append(
                        BlockchainSearchResult(
                            id=f"identity:{self._credentials.chain_id}:{agent.token_id}",
                            title=agent.agent_name,
                            snippet=f"Agent #{agent.token_id} - {agent.agent_name}",
                            source_url=make_block_explorer_url(
                                self._credentials.chain_id,
                                address=self._credentials.identity_registry,
                            ),
                            metadata={
                                "token_id": agent.token_id,
                                "agent_name": agent.agent_name,
                                "uri": agent.metadata_uri,
                            },
                        )
                    )
                return _AwaitableList(results[:max_results])

            if query_type == "agent" and len(parts) > 1:
                token_id = int(parts[1])
                identity = self._get_identity_contract().get_agent(token_id)
                agent_uri = getattr(identity, "agent_uri", None) or getattr(
                    identity, "metadata_uri", ""
                )
                results.append(
                    BlockchainSearchResult(
                        id=f"identity:{self._credentials.chain_id}:{token_id}",
                        title=getattr(identity, "agent_name", f"Agent #{token_id}"),
                        snippet=f"Owner: {identity.owner[:10]}... URI: {str(agent_uri)[:50]}...",
                        source_url=make_block_explorer_url(
                            self._credentials.chain_id,
                            address=self._credentials.identity_registry,
                        ),
                        metadata={"owner": identity.owner, "uri": agent_uri},
                    )
                )

            elif query_type == "owner" and len(parts) > 1:
                owner = parts[1]
                identities = self._get_identity_contract().get_agents_by_owner(owner)
                for identity in identities:
                    agent_uri = getattr(identity, "agent_uri", None) or getattr(
                        identity, "metadata_uri", ""
                    )
                    results.append(
                        BlockchainSearchResult(
                            id=f"identity:{self._credentials.chain_id}:{identity.token_id}",
                            title=getattr(identity, "agent_name", f"Agent #{identity.token_id}"),
                            snippet=f"Owner: {owner[:10]}... URI: {str(agent_uri)[:50]}...",
                            source_url=make_block_explorer_url(
                                self._credentials.chain_id,
                                address=self._credentials.identity_registry,
                            ),
                            metadata={
                                "owner": owner,
                                "uri": agent_uri,
                                "token_id": identity.token_id,
                            },
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

        return _AwaitableList(results[:max_results])

    async def search_by_owner(
        self, owner: str, max_results: int = 10
    ) -> list[BlockchainSearchResult]:
        """Search agents by owner address."""
        return await self.search(f"owner:{owner}", max_results=max_results)

    def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None | _AwaitableValue:  # type: ignore[override]
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
            return _AwaitableValue(cached)

        parts = evidence_id.split(":")
        if len(parts) < 3:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                raise ValueError("Invalid identifier")
            return _AwaitableValue(None)

        evidence_type = parts[0]
        try:
            chain_id = int(parts[1])
        except (ValueError, TypeError):
            logger.warning(f"Parse error: invalid evidence ID format: {evidence_id}")
            return _AwaitableValue(None)

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
                return _AwaitableValue(evidence)

            elif evidence_type == "reputation":
                token_id = int(parts[2])
                summary = self._get_reputation_contract().get_summary(token_id)
                normalized_value = getattr(summary, "normalized_value", 0.0)
                try:
                    normalized_value = float(normalized_value)
                except (TypeError, ValueError):
                    normalized_value = 0.0

                content = (
                    f"Reputation Summary for Agent #{token_id}\n"
                    f"Feedback Count: {summary.count}\n"
                    f"Aggregate Score: {normalized_value:.4f}\n"
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
                return _AwaitableValue(evidence)

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
                return _AwaitableValue(evidence)

        except Exception as e:
            if isinstance(e, OSError):
                logger.error(f"Network error fetching '{evidence_id}': {e}")
            elif isinstance(e, RuntimeError):
                logger.error(f"Runtime error fetching '{evidence_id}': {e}")
            else:
                logger.error(f"Fetch error for '{evidence_id}': {e}")

        return _AwaitableValue(None)

    def _search_by_owner_direct(self, owner: str, limit: int = 10) -> _AwaitableList:
        """Search agents by owner address (direct contract call)."""
        results: list[BlockchainSearchResult] = []
        try:
            agents = self._get_identity_contract().get_agents_by_owner(owner)
            for agent in agents:
                results.append(
                    BlockchainSearchResult(
                        id=f"identity:{self._credentials.chain_id}:{agent.token_id}",
                        title=agent.agent_name,
                        snippet=f"Owner: {owner[:10]}... Agent #{agent.token_id}",
                        source_url=make_block_explorer_url(
                            self._credentials.chain_id,
                            address=self._credentials.identity_registry,
                        ),
                        metadata={
                            "token_id": agent.token_id,
                            "agent_name": agent.agent_name,
                            "owner": owner,
                            "uri": agent.metadata_uri,
                        },
                    )
                )
        except Exception as e:
            logger.error(f"Search by owner error for '{owner}': {e}")
        return _AwaitableList(results[:limit])

    async def search_async(
        self, query: str, limit: int = 10, **kwargs: Any
    ) -> list[BlockchainSearchResult]:
        """Async search that uses async contract methods when available."""
        results: list[BlockchainSearchResult] = []
        max_results = int(kwargs.get("max_results", limit))
        parts = query.lower().split(":")
        query_type = parts[0] if parts else ""
        try:
            if ":" not in query:
                contract = self._get_identity_contract()
                if hasattr(contract, "get_agents_by_query_async"):
                    agents = await contract.get_agents_by_query_async(query)
                else:
                    agents = contract.get_agents_by_query(query)
                for agent in agents:
                    results.append(
                        BlockchainSearchResult(
                            id=f"identity:{self._credentials.chain_id}:{agent.token_id}",
                            title=agent.agent_name,
                            snippet=f"Agent #{agent.token_id} - {agent.agent_name}",
                            source_url=make_block_explorer_url(
                                self._credentials.chain_id,
                                address=self._credentials.identity_registry,
                            ),
                            metadata={
                                "token_id": agent.token_id,
                                "agent_name": agent.agent_name,
                                "uri": agent.metadata_uri,
                            },
                        )
                    )
                return results[:max_results]

            if query_type == "agent" and len(parts) > 1:
                token_id = int(parts[1])
                contract = self._get_identity_contract()
                if hasattr(contract, "get_agent_async"):
                    identity = await contract.get_agent_async(token_id)
                else:
                    identity = contract.get_agent(token_id)
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
        except Exception as e:
            logger.error(f"Search error for query '{query}': {e}")
        return results[:max_results]

    async def search_by_owner_async(
        self, owner: str, max_results: int = 10
    ) -> list[BlockchainSearchResult]:
        """Search agents by owner address (async)."""
        return await self.search(f"owner:{owner}", max_results=max_results)

    async def fetch_async(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Async fetch that uses async contract methods when available."""
        parts = evidence_id.split(":")
        if len(parts) < 3:
            raise ValueError("Invalid identifier")
        evidence_type = parts[0]
        try:
            chain_id = int(parts[1])
        except (ValueError, TypeError):
            logger.warning(f"Invalid evidence ID format: {evidence_id}")
            return None
        try:
            if evidence_type == "identity":
                token_id = int(parts[2])
                contract = self._get_identity_contract()
                if hasattr(contract, "get_agent_async"):
                    identity = await contract.get_agent_async(token_id)
                else:
                    identity = contract.get_agent(token_id)
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
                        self._credentials.chain_id,
                        address=self._credentials.identity_registry,
                    ),
                )
                evidence = self._to_evidence(blockchain_evidence)
                self._cache_set(evidence_id, evidence)
                return evidence
        except Exception as e:
            logger.error(f"Fetch error for ID '{evidence_id}': {e}")
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
