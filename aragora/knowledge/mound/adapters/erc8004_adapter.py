"""
ERC-8004 Adapter for Knowledge Mound.

Provides bidirectional synchronization between ERC-8004 on-chain registries
and Knowledge Mound:

Forward sync (blockchain -> KM):
- Agent identities become knowledge nodes
- Reputation feedback informs confidence scores
- Validation records provide trust attestations

Reverse sync (KM -> blockchain):
- ELO ratings contribute to on-chain reputation
- Gauntlet receipts become validation records
- Debate outcomes inform feedback

Usage:
    from aragora.knowledge.mound.adapters.erc8004_adapter import ERC8004Adapter
    from aragora.blockchain.provider import Web3Provider

    provider = Web3Provider.from_env()
    adapter = ERC8004Adapter(provider=provider)

    # Sync on-chain data to KM
    result = await adapter.sync_to_km()

    # Push KM ratings to blockchain
    result = await adapter.sync_from_km()
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional

from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter
from aragora.knowledge.mound.adapters._types import SyncResult, ValidationSyncResult

if TYPE_CHECKING:
    from aragora.blockchain.provider import Web3Provider
    from aragora.blockchain.wallet import WalletSigner

logger = logging.getLogger(__name__)


class ERC8004Adapter(KnowledgeMoundAdapter):
    """Bidirectional adapter between ERC-8004 registries and Knowledge Mound.

    Synchronizes on-chain agent data (identity, reputation, validation)
    with Knowledge Mound nodes, enabling trustless verification and
    cross-system confidence calibration.
    """

    adapter_name = "erc8004"

    def __init__(
        self,
        provider: Optional["Web3Provider"] = None,
        signer: Optional["WalletSigner"] = None,
        km_store: Optional[Any] = None,
        enable_reverse_sync: bool = False,
        min_elo_for_reputation: float = 1500.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the ERC-8004 adapter.

        Args:
            provider: Web3Provider for blockchain access.
            signer: WalletSigner for write operations (optional).
            km_store: Knowledge Mound store instance.
            enable_reverse_sync: If True, enables KM->blockchain sync.
            min_elo_for_reputation: Minimum ELO score to push reputation.
            **kwargs: Additional KnowledgeMoundAdapter arguments.
        """
        super().__init__(**kwargs)
        self._provider = provider
        self._signer = signer
        self._km_store = km_store
        self._enable_reverse_sync = enable_reverse_sync
        self._min_elo_for_reputation = min_elo_for_reputation
        self._identity_contract: Any = None
        self._reputation_contract: Any = None
        self._validation_contract: Any = None

    def _get_provider(self) -> "Web3Provider":
        """Get or create the Web3 provider."""
        if self._provider is None:
            from aragora.blockchain.provider import Web3Provider

            self._provider = Web3Provider.from_env()
        return self._provider

    def _get_identity_contract(self) -> Any:
        """Get or create the Identity Registry contract."""
        if self._identity_contract is None:
            from aragora.blockchain.contracts.identity import IdentityRegistryContract

            self._identity_contract = IdentityRegistryContract(self._get_provider())
        return self._identity_contract

    def _get_reputation_contract(self) -> Any:
        """Get or create the Reputation Registry contract."""
        if self._reputation_contract is None:
            from aragora.blockchain.contracts.reputation import ReputationRegistryContract

            self._reputation_contract = ReputationRegistryContract(self._get_provider())
        return self._reputation_contract

    def _get_validation_contract(self) -> Any:
        """Get or create the Validation Registry contract."""
        if self._validation_contract is None:
            from aragora.blockchain.contracts.validation import ValidationRegistryContract

            self._validation_contract = ValidationRegistryContract(self._get_provider())
        return self._validation_contract

    async def sync_to_km(
        self,
        agent_ids: list[int] | None = None,
        sync_identities: bool = True,
        sync_reputation: bool = True,
        sync_validations: bool = True,
    ) -> SyncResult:
        """Sync on-chain data to Knowledge Mound.

        Retrieves agent identities, reputation summaries, and validation
        records from ERC-8004 registries and stores them as KM nodes.

        Args:
            agent_ids: Specific agent IDs to sync. None = all known agents.
            sync_identities: Whether to sync identity records.
            sync_reputation: Whether to sync reputation summaries.
            sync_validations: Whether to sync validation records.

        Returns:
            SyncResult with counts and errors.
        """
        start_time = time.time()
        synced = 0
        skipped = 0
        failed = 0
        errors: list[str] = []

        try:
            async with self._resilient_call("sync_to_km"):
                provider = self._get_provider()
                config = provider.get_config()

                # Get agent IDs to sync
                if agent_ids is None and config.has_identity_registry:
                    # Get all registered agents (up to a reasonable limit)
                    try:
                        total = self._get_identity_contract().get_total_supply()
                        agent_ids = list(range(1, min(total + 1, 1001)))
                    except Exception as e:
                        logger.warning(f"Could not get total supply: {e}")
                        agent_ids = []

                for agent_id in agent_ids or []:
                    try:
                        # Sync identity
                        if sync_identities and config.has_identity_registry:
                            identity = self._get_identity_contract().get_agent(agent_id)
                            await self._store_identity_node(identity)
                            synced += 1

                        # Sync reputation summary
                        if sync_reputation and config.has_reputation_registry:
                            try:
                                summary = self._get_reputation_contract().get_summary(agent_id)
                                await self._store_reputation_node(summary)
                                synced += 1
                            except Exception as e:
                                logger.debug(f"No reputation for agent {agent_id}: {e}")
                                skipped += 1

                        # Sync validations
                        if sync_validations and config.has_validation_registry:
                            try:
                                validation_hashes = (
                                    self._get_validation_contract().get_agent_validations(agent_id)
                                )
                                for req_hash in validation_hashes[:10]:  # Limit per agent
                                    record = self._get_validation_contract().get_validation_status(
                                        req_hash
                                    )
                                    await self._store_validation_node(record)
                                    synced += 1
                            except Exception as e:
                                logger.debug(f"No validations for agent {agent_id}: {e}")
                                skipped += 1

                    except Exception as e:
                        failed += 1
                        errors.append(f"Agent {agent_id}: {str(e)}")
                        logger.warning(f"Failed to sync agent {agent_id}: {e}")

        except Exception as e:
            errors.append(f"Sync failed: {str(e)}")
            logger.error(f"sync_to_km failed: {e}")

        duration_ms = (time.time() - start_time) * 1000
        return SyncResult(
            records_synced=synced,
            records_skipped=skipped,
            records_failed=failed,
            errors=errors,
            duration_ms=duration_ms,
        )

    async def sync_from_km(
        self,
        push_elo_ratings: bool = True,
        push_receipts: bool = True,
    ) -> ValidationSyncResult:
        """Sync Knowledge Mound data to blockchain (reverse flow).

        Pushes ELO ratings as reputation feedback and gauntlet receipts
        as validation records to ERC-8004 registries.

        Requires a signer to be configured.

        Args:
            push_elo_ratings: Whether to push ELO ratings as reputation.
            push_receipts: Whether to push receipts as validations.

        Returns:
            ValidationSyncResult with counts and errors.
        """
        start_time = time.time()
        analyzed = 0
        updated = 0
        skipped = 0
        errors: list[str] = []

        if not self._enable_reverse_sync:
            return ValidationSyncResult(
                records_analyzed=0,
                records_updated=0,
                records_skipped=0,
                errors=["Reverse sync is disabled"],
                duration_ms=0.0,
            )

        if self._signer is None:
            return ValidationSyncResult(
                records_analyzed=0,
                records_updated=0,
                records_skipped=0,
                errors=["No signer configured for write operations"],
                duration_ms=0.0,
            )

        try:
            async with self._resilient_call("sync_from_km"):
                # TODO: Implement reverse sync logic
                # This would:
                # 1. Query ELO rankings from ELO adapter
                # 2. Map Aragora agent IDs to blockchain token IDs
                # 3. Push reputation feedback for high-performing agents
                # 4. Query gauntlet receipts
                # 5. Push validation records for receipts
                logger.info("Reverse sync not yet implemented")
                skipped = 1

        except Exception as e:
            errors.append(f"Reverse sync failed: {str(e)}")
            logger.error(f"sync_from_km failed: {e}")

        duration_ms = (time.time() - start_time) * 1000
        return ValidationSyncResult(
            records_analyzed=analyzed,
            records_updated=updated,
            records_skipped=skipped,
            errors=errors,
            duration_ms=duration_ms,
        )

    async def _store_identity_node(self, identity: Any) -> None:
        """Store an agent identity as a KM node."""
        if self._km_store is None:
            return

        node_data = {
            "type": "agent_identity",
            "source": "erc8004",
            "token_id": identity.token_id,
            "owner": identity.owner,
            "agent_uri": identity.agent_uri,
            "wallet_address": identity.wallet_address,
            "chain_id": identity.chain_id,
            "aragora_agent_id": identity.aragora_agent_id,
        }

        self._emit_event("identity_synced", node_data)
        logger.debug(f"Stored identity node for agent #{identity.token_id}")

    async def _store_reputation_node(self, summary: Any) -> None:
        """Store a reputation summary as a KM node."""
        if self._km_store is None:
            return

        node_data = {
            "type": "reputation_summary",
            "source": "erc8004",
            "agent_id": summary.agent_id,
            "count": summary.count,
            "value": summary.normalized_value,
            "tag1": summary.tag1,
            "tag2": summary.tag2,
        }

        self._emit_event("reputation_synced", node_data)
        logger.debug(f"Stored reputation node for agent #{summary.agent_id}")

    async def _store_validation_node(self, record: Any) -> None:
        """Store a validation record as a KM node."""
        if self._km_store is None:
            return

        node_data = {
            "type": "validation_record",
            "source": "erc8004",
            "request_hash": record.request_hash,
            "agent_id": record.agent_id,
            "validator": record.validator_address,
            "response": record.response.name,
            "tag": record.tag,
        }

        self._emit_event("validation_synced", node_data)
        logger.debug(f"Stored validation node for hash {record.request_hash[:16]}...")

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of the adapter."""
        try:
            provider = self._get_provider()
            connected = provider.is_connected()
            config = provider.get_config()

            return {
                "adapter": self.adapter_name,
                "connected": connected,
                "chain_id": config.chain_id,
                "has_identity_registry": config.has_identity_registry,
                "has_reputation_registry": config.has_reputation_registry,
                "has_validation_registry": config.has_validation_registry,
                "reverse_sync_enabled": self._enable_reverse_sync,
                "has_signer": self._signer is not None,
                "rpc_health": provider.get_health_status(),
            }
        except Exception as e:
            return {
                "adapter": self.adapter_name,
                "connected": False,
                "error": str(e),
            }


__all__ = ["ERC8004Adapter"]
