"""
Sync Operations Mixin for Knowledge Mound.

Provides cross-system synchronization:
- sync_from_continuum: Sync from ContinuumMemory
- sync_from_consensus: Sync from ConsensusMemory
- sync_from_facts: Sync from FactStore
- sync_from_evidence: Sync from EvidenceStore
- sync_from_critique: Sync from CritiqueStore
- sync_all: Sync from all connected systems
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import (
        IngestionRequest,
        MoundConfig,
        SyncResult,
    )
    from aragora.memory.continuum import ContinuumMemory
    from aragora.memory.consensus import ConsensusMemory
    from aragora.knowledge.fact_store import FactStore
    from aragora.evidence.store import EvidenceStore
    from aragora.memory.store import CritiqueStore

logger = logging.getLogger(__name__)


class SyncProtocol(Protocol):
    """Protocol defining expected interface for Sync mixin."""

    config: "MoundConfig"
    workspace_id: str
    _continuum: Optional["ContinuumMemory"]
    _consensus: Optional["ConsensusMemory"]
    _facts: Optional["FactStore"]
    _evidence: Optional["EvidenceStore"]
    _critique: Optional["CritiqueStore"]
    _initialized: bool

    def _ensure_initialized(self) -> None: ...
    async def store(self, request: "IngestionRequest") -> Any: ...


class SyncOperationsMixin:
    """Mixin providing sync operations for KnowledgeMound."""

    async def sync_from_continuum(
        self: SyncProtocol,
        continuum: "ContinuumMemory",
        incremental: bool = True,
        batch_size: int = 100,
    ) -> "SyncResult":
        """
        Sync knowledge from ContinuumMemory.

        Iterates through memory entries and stores them as knowledge nodes.
        Uses content hash deduplication to avoid duplicates.

        Args:
            continuum: ContinuumMemory instance to sync from
            incremental: If True, only sync entries updated since last sync
            batch_size: Number of entries to process per batch

        Returns:
            SyncResult with counts of synced/updated/skipped nodes
        """
        from aragora.knowledge.mound.types import (
            IngestionRequest,
            KnowledgeSource,
            SyncResult,
        )

        self._ensure_initialized()

        start_time = time.time()
        self._continuum = continuum
        nodes_synced = 0
        nodes_updated = 0
        nodes_skipped = 0
        errors: List[str] = []

        try:
            # Retrieve all entries from continuum (using high limit for full sync)
            entries = continuum.retrieve(
                query=None,
                tiers=None,
                limit=10000,  # High limit for sync
                min_importance=0.0,
                include_glacial=True,
            )

            for entry in entries:
                try:
                    # Create ingestion request from continuum entry
                    request = IngestionRequest(
                        content=entry.content,
                        workspace_id=self.workspace_id,
                        source_type=KnowledgeSource.CONTINUUM,
                        node_type="memory",
                        confidence=entry.importance,
                        tier=entry.tier.value,
                        metadata={
                            "continuum_id": entry.id,
                            "surprise_score": entry.surprise_score,
                            "consolidation_score": entry.consolidation_score,
                            "update_count": entry.update_count,
                            "success_rate": entry.success_rate,
                            "original_metadata": entry.metadata,
                        },
                    )

                    result = await self.store(request)

                    if result.deduplicated:
                        nodes_updated += 1
                    else:
                        nodes_synced += 1

                except Exception as e:
                    nodes_skipped += 1
                    errors.append(f"continuum:{entry.id}: {str(e)}")
                    logger.warning(f"Failed to sync continuum entry {entry.id}: {e}")

        except Exception as e:
            errors.append(f"continuum:retrieve: {str(e)}")
            logger.error(f"Failed to retrieve continuum entries: {e}")

        return SyncResult(
            source="continuum",
            nodes_synced=nodes_synced,
            nodes_updated=nodes_updated,
            nodes_skipped=nodes_skipped,
            relationships_created=0,
            duration_ms=int((time.time() - start_time) * 1000),
            errors=errors,
        )

    async def sync_from_consensus(
        self: SyncProtocol,
        consensus: "ConsensusMemory",
        incremental: bool = True,
        batch_size: int = 100,
    ) -> "SyncResult":
        """
        Sync knowledge from ConsensusMemory.

        Stores consensus records as high-confidence knowledge nodes.

        Args:
            consensus: ConsensusMemory instance to sync from
            incremental: If True, only sync entries since last sync
            batch_size: Number of entries to process per batch

        Returns:
            SyncResult with counts of synced/updated/skipped nodes
        """
        from aragora.knowledge.mound.types import (
            IngestionRequest,
            KnowledgeSource,
            SyncResult,
        )

        self._ensure_initialized()

        start_time = time.time()
        self._consensus = consensus
        nodes_synced = 0
        nodes_updated = 0
        nodes_skipped = 0
        relationships_created = 0
        errors: List[str] = []

        try:
            # Get recent consensus records from the store
            # ConsensusMemory stores records in SQLite, we query directly
            if hasattr(consensus, "_store") and consensus._store:
                with consensus._store.connection() as conn:
                    cursor = conn.execute(
                        """
                        SELECT id, topic, conclusion, strength, confidence,
                               participating_agents, agreeing_agents, domain, tags,
                               timestamp, supersedes, metadata
                        FROM consensus_records
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (10000,),
                    )
                    rows = cursor.fetchall()

                for row in rows:
                    try:
                        record_id = row[0]
                        topic = row[1]
                        conclusion = row[2]
                        strength = row[3]
                        confidence = row[4]
                        domain = row[7]
                        tags_json = row[8]
                        supersedes = row[10]
                        metadata_json = row[11]

                        # Parse JSON fields
                        from aragora.utils.json_helpers import safe_json_loads

                        tags: list[str] = safe_json_loads(tags_json, [])
                        metadata: dict[str, Any] = safe_json_loads(metadata_json, {})

                        # Create ingestion request
                        request = IngestionRequest(
                            content=f"{topic}: {conclusion}",
                            workspace_id=self.workspace_id,
                            source_type=KnowledgeSource.CONSENSUS,
                            debate_id=record_id,
                            node_type="consensus",
                            confidence=confidence,
                            tier="slow",  # Consensus is stable knowledge
                            topics=tags,
                            metadata={
                                "consensus_id": record_id,
                                "strength": strength,
                                "domain": domain,
                                "original_metadata": metadata,
                            },
                        )

                        # Add supersession relationship
                        if supersedes:
                            request.derived_from = [f"cs_{supersedes}"]

                        result = await self.store(request)

                        if result.deduplicated:
                            nodes_updated += 1
                        else:
                            nodes_synced += 1

                        relationships_created += result.relationships_created

                    except Exception as e:
                        nodes_skipped += 1
                        errors.append(f"consensus:{row[0]}: {str(e)}")
                        logger.warning(f"Failed to sync consensus record {row[0]}: {e}")

        except Exception as e:
            errors.append(f"consensus:query: {str(e)}")
            logger.error(f"Failed to query consensus records: {e}")

        return SyncResult(
            source="consensus",
            nodes_synced=nodes_synced,
            nodes_updated=nodes_updated,
            nodes_skipped=nodes_skipped,
            relationships_created=relationships_created,
            duration_ms=int((time.time() - start_time) * 1000),
            errors=errors,
        )

    async def sync_from_facts(
        self: SyncProtocol,
        facts: "FactStore",
        incremental: bool = True,
        batch_size: int = 100,
    ) -> "SyncResult":
        """
        Sync knowledge from FactStore.

        Stores facts as knowledge nodes with evidence relationships.

        Args:
            facts: FactStore instance to sync from
            incremental: If True, only sync since last sync
            batch_size: Number of entries to process per batch

        Returns:
            SyncResult with counts of synced/updated/skipped nodes
        """
        from aragora.knowledge.mound.types import (
            IngestionRequest,
            KnowledgeSource,
            SyncResult,
        )

        self._ensure_initialized()

        start_time = time.time()
        self._facts = facts
        nodes_synced = 0
        nodes_updated = 0
        nodes_skipped = 0
        relationships_created = 0
        errors: List[str] = []

        try:
            # FactStore has query_facts method
            if hasattr(facts, "query_facts"):
                all_facts = facts.query_facts(  # type: ignore[call-arg]
                    query="",
                    workspace_id=self.workspace_id,
                    limit=10000,
                )

                for fact in all_facts:
                    try:
                        request = IngestionRequest(
                            content=fact.statement,
                            workspace_id=self.workspace_id,
                            source_type=KnowledgeSource.FACT,
                            document_id=fact.source_documents[0] if fact.source_documents else None,
                            node_type="fact",
                            confidence=fact.confidence,
                            tier="slow",
                            topics=fact.topics,
                            metadata={
                                "fact_id": fact.id,
                                "validation_status": (
                                    fact.validation_status.value
                                    if hasattr(fact.validation_status, "value")
                                    else str(fact.validation_status)
                                ),
                                "evidence_ids": fact.evidence_ids,
                                "source_documents": fact.source_documents,
                            },
                        )

                        result = await self.store(request)

                        if result.deduplicated:
                            nodes_updated += 1
                        else:
                            nodes_synced += 1

                    except Exception as e:
                        nodes_skipped += 1
                        errors.append(f"facts:{fact.id}: {str(e)}")
                        logger.warning(f"Failed to sync fact {fact.id}: {e}")

        except Exception as e:
            errors.append(f"facts:query: {str(e)}")
            logger.error(f"Failed to query facts: {e}")

        return SyncResult(
            source="facts",
            nodes_synced=nodes_synced,
            nodes_updated=nodes_updated,
            nodes_skipped=nodes_skipped,
            relationships_created=relationships_created,
            duration_ms=int((time.time() - start_time) * 1000),
            errors=errors,
        )

    async def sync_from_evidence(
        self: SyncProtocol,
        evidence: "EvidenceStore",
        incremental: bool = True,
        batch_size: int = 100,
    ) -> "SyncResult":
        """
        Sync knowledge from EvidenceStore.

        Stores evidence snippets as knowledge nodes.

        Args:
            evidence: EvidenceStore instance to sync from
            incremental: If True, only sync since last sync
            batch_size: Number of entries to process per batch

        Returns:
            SyncResult with counts of synced/updated/skipped nodes
        """
        from aragora.knowledge.mound.types import (
            IngestionRequest,
            KnowledgeSource,
            SyncResult,
        )

        self._ensure_initialized()

        start_time = time.time()
        self._evidence = evidence
        nodes_synced = 0
        nodes_updated = 0
        nodes_skipped = 0
        errors: List[str] = []

        try:
            # EvidenceStore has search method
            if hasattr(evidence, "search"):
                all_evidence = evidence.search("", limit=10000)

                for ev in all_evidence:
                    try:
                        request = IngestionRequest(
                            content=ev.content,
                            workspace_id=self.workspace_id,
                            source_type=KnowledgeSource.EVIDENCE,
                            debate_id=getattr(ev, "debate_id", None),
                            agent_id=getattr(ev, "agent_id", None),
                            node_type="evidence",
                            confidence=getattr(ev, "quality_score", 0.5),
                            tier="medium",
                            metadata={
                                "evidence_id": ev.id,
                                "source_url": getattr(ev, "source_url", None),
                            },
                        )

                        result = await self.store(request)

                        if result.deduplicated:
                            nodes_updated += 1
                        else:
                            nodes_synced += 1

                    except Exception as e:
                        nodes_skipped += 1
                        errors.append(f"evidence:{ev.id}: {str(e)}")
                        logger.warning(f"Failed to sync evidence {ev.id}: {e}")

        except Exception as e:
            errors.append(f"evidence:search: {str(e)}")
            logger.error(f"Failed to search evidence: {e}")

        return SyncResult(
            source="evidence",
            nodes_synced=nodes_synced,
            nodes_updated=nodes_updated,
            nodes_skipped=nodes_skipped,
            relationships_created=0,
            duration_ms=int((time.time() - start_time) * 1000),
            errors=errors,
        )

    async def sync_from_critique(
        self: SyncProtocol,
        critique: "CritiqueStore",
        incremental: bool = True,
        batch_size: int = 100,
    ) -> "SyncResult":
        """
        Sync knowledge from CritiqueStore (critique patterns).

        Stores successful critique patterns as knowledge nodes.

        Args:
            critique: CritiqueStore instance to sync from
            incremental: If True, only sync since last sync
            batch_size: Number of entries to process per batch

        Returns:
            SyncResult with counts of synced/updated/skipped nodes
        """
        from aragora.knowledge.mound.types import (
            IngestionRequest,
            KnowledgeSource,
            SyncResult,
        )

        self._ensure_initialized()

        start_time = time.time()
        self._critique = critique
        nodes_synced = 0
        nodes_updated = 0
        nodes_skipped = 0
        errors: List[str] = []

        try:
            # CritiqueStore has search_patterns method
            if hasattr(critique, "search_patterns"):
                patterns = critique.search_patterns("", limit=10000)

                for pattern in patterns:
                    try:
                        content = getattr(pattern, "pattern", "") or getattr(pattern, "content", "")
                        if not content:
                            nodes_skipped += 1
                            continue

                        request = IngestionRequest(
                            content=content,
                            workspace_id=self.workspace_id,
                            source_type=KnowledgeSource.CRITIQUE,
                            agent_id=getattr(pattern, "agent_name", None),
                            node_type="critique",
                            confidence=getattr(pattern, "success_rate", 0.5),
                            tier="slow",
                            metadata={
                                "pattern_id": pattern.id,
                                "success_count": getattr(pattern, "success_count", 0),
                            },
                        )

                        result = await self.store(request)

                        if result.deduplicated:
                            nodes_updated += 1
                        else:
                            nodes_synced += 1

                    except Exception as e:
                        nodes_skipped += 1
                        errors.append(f"critique:{pattern.id}: {str(e)}")
                        logger.warning(f"Failed to sync critique pattern {pattern.id}: {e}")

        except Exception as e:
            errors.append(f"critique:search: {str(e)}")
            logger.error(f"Failed to search critique patterns: {e}")

        return SyncResult(
            source="critique",
            nodes_synced=nodes_synced,
            nodes_updated=nodes_updated,
            nodes_skipped=nodes_skipped,
            relationships_created=0,
            duration_ms=int((time.time() - start_time) * 1000),
            errors=errors,
        )

    async def sync_all(self: SyncProtocol) -> Dict[str, "SyncResult"]:
        """
        Sync from all connected memory systems.

        Returns a dict mapping source name to SyncResult.
        Only syncs from sources that have been connected.
        """

        self._ensure_initialized()
        results: Dict[str, SyncResult] = {}

        if self._continuum:
            results["continuum"] = await self.sync_from_continuum(self._continuum)  # type: ignore[arg-type,attr-defined]

        if self._consensus:
            results["consensus"] = await self.sync_from_consensus(self._consensus)  # type: ignore[arg-type,attr-defined]

        if self._facts:
            results["facts"] = await self.sync_from_facts(self._facts)  # type: ignore[arg-type,attr-defined]

        if self._evidence:
            results["evidence"] = await self.sync_from_evidence(self._evidence)  # type: ignore[arg-type,attr-defined]

        if self._critique:
            results["critique"] = await self.sync_from_critique(self._critique)  # type: ignore[arg-type,attr-defined]

        logger.info(
            "Sync complete: %d sources, %d total nodes synced",
            len(results),
            sum(r.nodes_synced for r in results.values()),
        )

        return results

    # =========================================================================
    # Handler-Compatible Sync Methods
    # These methods use internally connected stores and support incremental sync
    # with `since` parameter for API handler integration.
    # =========================================================================

    async def sync_continuum_incremental(
        self: SyncProtocol,
        workspace_id: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 1000,
    ) -> "SyncResult":
        """
        Handler-compatible incremental sync from ContinuumMemory.

        Uses the internally connected _continuum store. Supports incremental
        sync via the `since` parameter (ISO timestamp or entry ID).

        Args:
            workspace_id: Override workspace ID for this sync
            since: ISO timestamp or entry ID to sync from (incremental)
            limit: Maximum entries to sync in this batch

        Returns:
            SyncResult with counts of synced/updated/skipped nodes
        """
        from datetime import datetime
        from aragora.knowledge.mound.types import (
            IngestionRequest,
            KnowledgeSource,
            SyncResult,
        )

        self._ensure_initialized()

        if not self._continuum:
            return SyncResult(
                source="continuum",
                nodes_synced=0,
                nodes_updated=0,
                nodes_skipped=0,
                relationships_created=0,
                duration_ms=0,
                errors=["ContinuumMemory not connected"],
            )

        start_time = time.time()
        ws_id = workspace_id or self.workspace_id
        nodes_synced = 0
        nodes_updated = 0
        nodes_skipped = 0
        errors: List[str] = []

        try:
            # Parse since timestamp if provided
            since_dt = None
            if since:
                try:
                    since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
                except ValueError:
                    pass  # Not a timestamp, could be an ID

            # Retrieve entries from continuum
            entries = self._continuum.retrieve(
                query=None,
                tiers=None,
                limit=limit,
                min_importance=0.0,
                include_glacial=True,
            )

            # Filter by since if provided
            if since_dt:
                entries = [
                    e
                    for e in entries
                    if hasattr(e, "updated_at") and e.updated_at >= since_dt.isoformat()
                ]

            for entry in entries:
                try:
                    request = IngestionRequest(
                        content=entry.content,
                        workspace_id=ws_id,
                        source_type=KnowledgeSource.CONTINUUM,
                        node_type="memory",
                        confidence=entry.importance,
                        tier=entry.tier.value,
                        metadata={
                            "continuum_id": entry.id,
                            "surprise_score": entry.surprise_score,
                            "consolidation_score": entry.consolidation_score,
                            "update_count": entry.update_count,
                            "success_rate": entry.success_rate,
                            "original_metadata": entry.metadata,
                        },
                    )

                    result = await self.store(request)

                    if result.deduplicated:
                        nodes_updated += 1
                    else:
                        nodes_synced += 1

                except Exception as e:
                    nodes_skipped += 1
                    errors.append(f"continuum:{entry.id}: {str(e)}")

        except Exception as e:
            errors.append(f"continuum:retrieve: {str(e)}")

        return SyncResult(
            source="continuum",
            nodes_synced=nodes_synced,
            nodes_updated=nodes_updated,
            nodes_skipped=nodes_skipped,
            relationships_created=0,
            duration_ms=int((time.time() - start_time) * 1000),
            errors=errors,
        )

    async def sync_consensus_incremental(
        self: SyncProtocol,
        workspace_id: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 1000,
    ) -> "SyncResult":
        """
        Handler-compatible incremental sync from ConsensusMemory.

        Uses the internally connected _consensus store. Supports incremental
        sync via the `since` parameter (ISO timestamp).

        Args:
            workspace_id: Override workspace ID for this sync
            since: ISO timestamp to sync from (incremental)
            limit: Maximum entries to sync in this batch

        Returns:
            SyncResult with counts of synced/updated/skipped nodes
        """
        from aragora.knowledge.mound.types import (
            IngestionRequest,
            KnowledgeSource,
            SyncResult,
        )

        self._ensure_initialized()

        if not self._consensus:
            return SyncResult(
                source="consensus",
                nodes_synced=0,
                nodes_updated=0,
                nodes_skipped=0,
                relationships_created=0,
                duration_ms=0,
                errors=["ConsensusMemory not connected"],
            )

        start_time = time.time()
        ws_id = workspace_id or self.workspace_id
        nodes_synced = 0
        nodes_updated = 0
        nodes_skipped = 0
        relationships_created = 0
        errors: List[str] = []

        try:
            # Build query with since filter
            query = """
                SELECT id, topic, conclusion, strength, confidence,
                       participating_agents, agreeing_agents, domain, tags,
                       timestamp, supersedes, metadata
                FROM consensus_records
            """
            params: List[Any] = []

            if since:
                query += " WHERE timestamp >= ?"
                params.append(since)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            if hasattr(self._consensus, "_store") and self._consensus._store:
                with self._consensus._store.connection() as conn:
                    cursor = conn.execute(query, params)
                    rows = cursor.fetchall()

                from aragora.utils.json_helpers import safe_json_loads

                for row in rows:
                    try:
                        record_id = row[0]
                        topic = row[1]
                        conclusion = row[2]
                        strength = row[3]
                        confidence = row[4]
                        domain = row[7]
                        tags_json = row[8]
                        supersedes = row[10]
                        metadata_json = row[11]

                        tags: list[str] = safe_json_loads(tags_json, [])
                        metadata: dict[str, Any] = safe_json_loads(metadata_json, {})

                        request = IngestionRequest(
                            content=f"{topic}: {conclusion}",
                            workspace_id=ws_id,
                            source_type=KnowledgeSource.CONSENSUS,
                            debate_id=record_id,
                            node_type="consensus",
                            confidence=confidence,
                            tier="slow",
                            topics=tags,
                            metadata={
                                "consensus_id": record_id,
                                "strength": strength,
                                "domain": domain,
                                "original_metadata": metadata,
                            },
                        )

                        if supersedes:
                            request.derived_from = [f"cs_{supersedes}"]

                        result = await self.store(request)

                        if result.deduplicated:
                            nodes_updated += 1
                        else:
                            nodes_synced += 1

                        relationships_created += result.relationships_created

                    except Exception as e:
                        nodes_skipped += 1
                        errors.append(f"consensus:{row[0]}: {str(e)}")

        except Exception as e:
            errors.append(f"consensus:query: {str(e)}")

        return SyncResult(
            source="consensus",
            nodes_synced=nodes_synced,
            nodes_updated=nodes_updated,
            nodes_skipped=nodes_skipped,
            relationships_created=relationships_created,
            duration_ms=int((time.time() - start_time) * 1000),
            errors=errors,
        )

    async def sync_facts_incremental(
        self: SyncProtocol,
        workspace_id: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 1000,
    ) -> "SyncResult":
        """
        Handler-compatible incremental sync from FactStore.

        Uses the internally connected _facts store. Supports incremental
        sync via the `since` parameter.

        Args:
            workspace_id: Override workspace ID for this sync
            since: ISO timestamp to sync from (incremental)
            limit: Maximum entries to sync in this batch

        Returns:
            SyncResult with counts of synced/updated/skipped nodes
        """
        from aragora.knowledge.mound.types import (
            IngestionRequest,
            KnowledgeSource,
            SyncResult,
        )

        self._ensure_initialized()

        if not self._facts:
            return SyncResult(
                source="facts",
                nodes_synced=0,
                nodes_updated=0,
                nodes_skipped=0,
                relationships_created=0,
                duration_ms=0,
                errors=["FactStore not connected"],
            )

        start_time = time.time()
        ws_id = workspace_id or self.workspace_id
        nodes_synced = 0
        nodes_updated = 0
        nodes_skipped = 0
        errors: List[str] = []

        try:
            if hasattr(self._facts, "query_facts"):
                all_facts = self._facts.query_facts(  # type: ignore[call-arg]
                    query="",
                    workspace_id=ws_id,
                    limit=limit,
                    since=since,  # Pass since to the query if supported
                )

                for fact in all_facts:
                    try:
                        request = IngestionRequest(
                            content=fact.statement,
                            workspace_id=ws_id,
                            source_type=KnowledgeSource.FACT,
                            document_id=fact.source_documents[0] if fact.source_documents else None,
                            node_type="fact",
                            confidence=fact.confidence,
                            tier="slow",
                            topics=fact.topics,
                            metadata={
                                "fact_id": fact.id,
                                "validation_status": (
                                    fact.validation_status.value
                                    if hasattr(fact.validation_status, "value")
                                    else str(fact.validation_status)
                                ),
                                "evidence_ids": fact.evidence_ids,
                                "source_documents": fact.source_documents,
                            },
                        )

                        result = await self.store(request)

                        if result.deduplicated:
                            nodes_updated += 1
                        else:
                            nodes_synced += 1

                    except Exception as e:
                        nodes_skipped += 1
                        errors.append(f"facts:{fact.id}: {str(e)}")

        except Exception as e:
            errors.append(f"facts:query: {str(e)}")

        return SyncResult(
            source="facts",
            nodes_synced=nodes_synced,
            nodes_updated=nodes_updated,
            nodes_skipped=nodes_skipped,
            relationships_created=0,
            duration_ms=int((time.time() - start_time) * 1000),
            errors=errors,
        )

    async def connect_memory_stores(
        self: SyncProtocol,
        continuum: Optional["ContinuumMemory"] = None,
        consensus: Optional["ConsensusMemory"] = None,
        facts: Optional["FactStore"] = None,
        evidence: Optional["EvidenceStore"] = None,
        critique: Optional["CritiqueStore"] = None,
    ) -> Dict[str, bool]:
        """
        Connect memory stores for use with incremental sync methods.

        Args:
            continuum: ContinuumMemory instance
            consensus: ConsensusMemory instance
            facts: FactStore instance
            evidence: EvidenceStore instance
            critique: CritiqueStore instance

        Returns:
            Dict mapping store name to connection status
        """
        status = {}

        if continuum:
            self._continuum = continuum
            status["continuum"] = True

        if consensus:
            self._consensus = consensus
            status["consensus"] = True

        if facts:
            self._facts = facts
            status["facts"] = True

        if evidence:
            self._evidence = evidence
            status["evidence"] = True

        if critique:
            self._critique = critique
            status["critique"] = True

        logger.info(
            "Connected %d memory stores to Knowledge Mound",
            len(status),
        )

        return status

    def get_connected_stores(self: SyncProtocol) -> List[str]:
        """Get list of connected memory store names."""
        connected = []
        if self._continuum:
            connected.append("continuum")
        if self._consensus:
            connected.append("consensus")
        if self._facts:
            connected.append("facts")
        if self._evidence:
            connected.append("evidence")
        if self._critique:
            connected.append("critique")
        return connected
