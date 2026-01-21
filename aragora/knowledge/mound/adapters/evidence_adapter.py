"""
EvidenceAdapter - Bridges EvidenceStore to the Knowledge Mound.

This adapter enables bidirectional integration between the Evidence system
and the Knowledge Mound:

- Data flow IN: Evidence snippets with quality scores are stored in KM
- Data flow OUT: Similar evidence is retrieved for deduplication/enrichment
- Reverse flow: KM validation feeds back to evidence reliability scores

The adapter provides:
- Unified search interface (search_by_topic, search_similar)
- Bidirectional sync with dual-write support
- Reliability-to-confidence mapping
- Provenance tracking for evidence chains
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.knowledge.unified.types import KnowledgeItem
    from aragora.evidence.store import EvidenceStore
    from aragora.knowledge.mound.types import IngestionRequest

# Type alias for event callback
EventCallback = Callable[[str, Dict[str, Any]], None]

logger = logging.getLogger(__name__)

# Try to import SLO metrics
try:
    from aragora.observability.metrics.slo import check_and_record_slo

    SLO_AVAILABLE = True
except ImportError:
    SLO_AVAILABLE = False


class EvidenceAdapterError(Exception):
    """Base exception for evidence adapter errors."""

    pass


class EvidenceStoreUnavailableError(EvidenceAdapterError):
    """Raised when evidence store is not configured."""

    pass


class EvidenceNotFoundError(EvidenceAdapterError):
    """Raised when evidence item is not found."""

    pass


@dataclass
class EvidenceSearchResult:
    """Wrapper for evidence search results with adapter metadata."""

    evidence: Dict[str, Any]
    relevance_score: float = 0.0
    matched_topics: List[str] = None

    def __post_init__(self) -> None:
        if self.matched_topics is None:
            self.matched_topics = []


class EvidenceAdapter:
    """
    Adapter that bridges EvidenceStore to the Knowledge Mound.

    Provides methods for the Knowledge Mound's federated query system:
    - search_by_topic: Topic-based evidence search
    - search_similar: Find similar evidence for deduplication
    - to_knowledge_item: Convert evidence to unified format
    - store: Store content with KM sync

    Usage:
        from aragora.evidence.store import EvidenceStore
        from aragora.knowledge.mound.adapters import EvidenceAdapter

        store = EvidenceStore()
        adapter = EvidenceAdapter(store)

        # Search for evidence
        results = adapter.search_by_topic("contract law", limit=10)

        # Convert to knowledge items
        items = [adapter.to_knowledge_item(r) for r in results]
    """

    ID_PREFIX = "ev_"
    MIN_RELIABILITY = 0.6
    MIN_QUALITY = 0.7

    def __init__(
        self,
        store: Optional["EvidenceStore"] = None,
        enable_dual_write: bool = False,
        event_callback: Optional[EventCallback] = None,
    ):
        """
        Initialize the adapter.

        Args:
            store: Optional EvidenceStore instance to wrap
            enable_dual_write: If True, writes go to both systems during migration
            event_callback: Optional callback for emitting events (event_type, data)
                          Used for WebSocket updates when adapter is in server context
        """
        self._store = store
        self._enable_dual_write = enable_dual_write
        self._event_callback = event_callback

    def set_event_callback(self, callback: EventCallback) -> None:
        """Set the event callback for WebSocket notifications.

        Args:
            callback: Function taking (event_type: str, data: Dict)
        """
        self._event_callback = callback

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event if callback is configured.

        Args:
            event_type: Type of event (e.g., 'knowledge_indexed')
            data: Event data payload
        """
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Failed to emit event {event_type}: {e}")

    def _record_metric(self, operation: str, success: bool, latency: float) -> None:
        """Record Prometheus metric for adapter operation.

        Args:
            operation: Operation name (search, store, sync)
            success: Whether operation succeeded
            latency: Operation latency in seconds
        """
        try:
            from aragora.observability.metrics.km import (
                record_km_operation,
                record_km_adapter_sync,
            )

            record_km_operation(operation, success, latency)
            if operation in ("store", "sync"):
                record_km_adapter_sync("evidence", "forward", success)
        except ImportError:
            pass  # Metrics not available
        except Exception as e:
            logger.debug(f"Failed to record metric: {e}")

    @property
    def evidence_store(self) -> Optional["EvidenceStore"]:
        """Access the underlying EvidenceStore."""
        return self._store

    def _ensure_store(self) -> "EvidenceStore":
        """Ensure evidence store is available.

        Returns:
            The evidence store instance

        Raises:
            EvidenceStoreUnavailableError: If store is not configured
        """
        if self._store is None:
            raise EvidenceStoreUnavailableError(
                "EvidenceStore not configured. Initialize adapter with a store instance."
            )
        return self._store

    def search_by_topic(
        self,
        query: str,
        limit: int = 10,
        min_reliability: float = 0.0,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search evidence by topic query.

        This method wraps EvidenceStore.search_evidence() to provide the
        interface expected by KnowledgeMound._query_evidence().

        Args:
            query: Search query (keywords are OR'd)
            limit: Maximum results to return
            min_reliability: Minimum reliability score threshold
            source: Optional source filter (e.g., "github", "web")

        Returns:
            List of evidence dicts matching the query

        Raises:
            EvidenceStoreUnavailableError: If store is not configured
        """
        import time

        start = time.time()
        success = False
        store = self._ensure_store()

        try:
            results = store.search_evidence(
                query=query,
                limit=limit,
                min_reliability=min_reliability,
            )

            # Filter by source if specified
            if source:
                results = [r for r in results if r.get("source") == source]

            success = True

            # Check SLO if available
            latency_ms = (time.time() - start) * 1000
            if SLO_AVAILABLE:
                check_and_record_slo("evidence_search", latency_ms)

            return results

        except EvidenceStoreUnavailableError:
            raise
        except Exception as e:
            logger.error(f"Evidence search failed for query '{query}': {e}")
            raise EvidenceAdapterError(f"Search failed: {e}") from e
        finally:
            self._record_metric("search", success, time.time() - start)

    def search_similar(
        self,
        content: str,
        limit: int = 5,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Find similar evidence for deduplication.

        Args:
            content: Content to find similar evidence for
            limit: Maximum results
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar evidence items

        Raises:
            EvidenceStoreUnavailableError: If store is not configured
            EvidenceAdapterError: If search fails
        """
        import hashlib
        import time

        start = time.time()
        store = self._ensure_store()

        try:
            # Use content hash-based lookup for exact matches
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:32]

            try:
                existing = store.get_evidence_by_hash(content_hash)
                if existing:
                    # Check SLO for hash lookup
                    latency_ms = (time.time() - start) * 1000
                    if SLO_AVAILABLE:
                        check_and_record_slo("evidence_hash_lookup", latency_ms)
                    return [existing]
            except AttributeError:
                # Store doesn't support hash lookup, fall through to text search
                logger.debug("Evidence store does not support hash lookup, using text search")
            except Exception as e:
                # Log but don't fail - fall back to text search
                logger.warning(f"Hash lookup failed, falling back to text search: {e}")

            # Fall back to text search for partial matches
            # Extract key terms for search
            words = content.split()[:10]  # First 10 words
            query = " ".join(words)

            return self.search_by_topic(query, limit=limit)

        except EvidenceStoreUnavailableError:
            raise
        except EvidenceAdapterError:
            raise
        except Exception as e:
            logger.error(f"Similar evidence search failed: {e}")
            raise EvidenceAdapterError(f"Similar search failed: {e}") from e

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.6,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic vector search over evidence items.

        Uses the Knowledge Mound's SemanticStore for embedding-based similarity
        search, falling back to keyword search if embeddings aren't available.

        Args:
            query: The search query
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold (0.0-1.0)
            tenant_id: Optional tenant filter

        Returns:
            List of matching evidence with similarity scores

        Raises:
            EvidenceStoreUnavailableError: If store is not configured
            EvidenceAdapterError: If search fails
        """
        import time

        start = time.time()
        success = False

        try:
            # Try semantic search first
            try:
                from aragora.knowledge.mound.semantic_store import SemanticStore

                # Get or create semantic store
                store = SemanticStore()

                # Search using embeddings
                results = await store.search_similar(
                    query=query,
                    tenant_id=tenant_id or "default",
                    limit=limit,
                    min_similarity=min_similarity,
                    source_type="evidence",
                )

                # Enrich results with full evidence items
                enriched = []
                for r in results:
                    # Try to get the full evidence from store
                    evidence_id = r.source_id
                    if evidence_id.startswith("ev_"):
                        evidence_id = evidence_id[3:]

                    evidence = self.get(evidence_id)
                    if evidence:
                        evidence["similarity"] = r.similarity
                        evidence["domain"] = r.domain
                        enriched.append(evidence)
                    else:
                        # Evidence may not be in store
                        enriched.append(
                            {
                                "id": r.source_id,
                                "similarity": r.similarity,
                                "domain": r.domain,
                                "importance": r.importance,
                                "metadata": r.metadata,
                            }
                        )

                success = True
                logger.debug(f"Semantic search returned {len(enriched)} results for '{query[:50]}'")

                # Emit event
                self._emit_event(
                    "km_adapter_semantic_search",
                    {
                        "source": "evidence",
                        "query_preview": query[:50],
                        "results_count": len(enriched),
                        "search_type": "vector",
                    },
                )

                return enriched

            except ImportError:
                logger.debug("SemanticStore not available, falling back to keyword search")
            except Exception as e:
                logger.debug(f"Semantic search failed, falling back: {e}")

            # Fallback to keyword search
            results = self.search_similar(query, limit=limit, min_similarity=min_similarity)
            success = True
            return results

        except EvidenceStoreUnavailableError:
            raise
        except EvidenceAdapterError:
            raise
        except Exception as e:
            logger.error(f"Semantic evidence search failed: {e}")
            raise EvidenceAdapterError(f"Semantic search failed: {e}") from e
        finally:
            self._record_metric("semantic_search", success, time.time() - start)

    def get(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific evidence item by ID.

        Args:
            evidence_id: The evidence ID (may be prefixed with "ev_" from mound)

        Returns:
            Evidence dict or None

        Raises:
            EvidenceStoreUnavailableError: If store is not configured
            EvidenceAdapterError: If retrieval fails
        """
        store = self._ensure_store()

        # Strip mound prefix if present
        if evidence_id.startswith(self.ID_PREFIX):
            evidence_id = evidence_id[len(self.ID_PREFIX) :]

        try:
            return store.get_evidence(evidence_id)
        except Exception as e:
            logger.error(f"Failed to get evidence {evidence_id}: {e}")
            raise EvidenceAdapterError(f"Failed to get evidence: {e}") from e

    def to_knowledge_item(self, evidence: Dict[str, Any]) -> "KnowledgeItem":
        """
        Convert evidence dict to a KnowledgeItem.

        Args:
            evidence: The evidence dictionary

        Returns:
            KnowledgeItem for unified knowledge mound API
        """
        from aragora.knowledge.unified.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        # Map reliability score to confidence level
        reliability = evidence.get("reliability_score", 0.5)
        if reliability >= 0.9:
            confidence = ConfidenceLevel.VERIFIED
        elif reliability >= 0.7:
            confidence = ConfidenceLevel.HIGH
        elif reliability >= 0.5:
            confidence = ConfidenceLevel.MEDIUM
        elif reliability >= 0.3:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.UNVERIFIED

        # Parse quality scores if available
        quality_scores = {}
        if evidence.get("quality_scores_json"):
            import json

            try:
                quality_scores = json.loads(evidence["quality_scores_json"])
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse quality_scores_json for evidence {evidence.get('id')}: {e}"
                )
            except TypeError as e:
                logger.warning(
                    f"Invalid quality_scores_json type for evidence {evidence.get('id')}: {e}"
                )

        # Parse enriched metadata if available
        enriched_metadata = {}
        if evidence.get("enriched_metadata_json"):
            import json

            try:
                enriched_metadata = json.loads(evidence["enriched_metadata_json"])
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse enriched_metadata_json for evidence {evidence.get('id')}: {e}"
                )
            except TypeError as e:
                logger.warning(
                    f"Invalid enriched_metadata_json type for evidence {evidence.get('id')}: {e}"
                )

        # Build metadata
        metadata: Dict[str, Any] = {
            "source": evidence.get("source", "unknown"),
            "title": evidence.get("title", ""),
            "url": evidence.get("url", ""),
            "reliability_score": reliability,
            "quality_scores": quality_scores,
            "enriched": enriched_metadata,
        }

        # Parse timestamps
        created_at = evidence.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = datetime.now(timezone.utc)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        updated_at = evidence.get("updated_at")
        if isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            except ValueError:
                updated_at = created_at
        elif updated_at is None:
            updated_at = created_at

        return KnowledgeItem(
            id=f"{self.ID_PREFIX}{evidence['id']}",
            content=evidence.get("snippet", ""),
            source=KnowledgeSource.EVIDENCE,
            source_id=evidence["id"],
            confidence=confidence,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
            importance=reliability,
        )

    def from_ingestion_request(
        self,
        request: "IngestionRequest",
        evidence_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert an IngestionRequest to EvidenceStore save_evidence() parameters.

        Args:
            request: The ingestion request from Knowledge Mound
            evidence_id: Optional ID to use (generates one if not provided)

        Returns:
            Dict of parameters for EvidenceStore.save_evidence()
        """
        import uuid

        return {
            "evidence_id": evidence_id or f"mound_{uuid.uuid4().hex[:12]}",
            "source": request.metadata.get("source", "knowledge_mound"),
            "title": request.metadata.get("title", "Knowledge Mound Entry"),
            "snippet": request.content,
            "url": request.metadata.get("url", ""),
            "reliability_score": request.confidence,
            "metadata": {
                "source_type": (
                    request.source_type.value
                    if hasattr(request.source_type, "value")
                    else str(request.source_type)
                ),
                "debate_id": request.debate_id,
                "document_id": request.document_id,
                "agent_id": request.agent_id,
                "user_id": request.user_id,
                "workspace_id": request.workspace_id,
                "topics": request.topics,
                "mound_metadata": request.metadata,
            },
        }

    def store(
        self,
        evidence_id: str,
        source: str,
        title: str,
        snippet: str,
        url: str = "",
        reliability_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        debate_id: Optional[str] = None,
    ) -> str:
        """
        Store evidence with optional KM sync.

        Args:
            evidence_id: Unique evidence ID
            source: Source name (e.g., "github", "web")
            title: Evidence title
            snippet: Evidence content
            url: Source URL
            reliability_score: Reliability score (0-1)
            metadata: Additional metadata
            debate_id: Optional debate association

        Returns:
            The evidence ID (may be deduplicated)

        Raises:
            EvidenceStoreUnavailableError: If store is not configured
            EvidenceAdapterError: If storage fails
        """
        import time

        start = time.time()
        success = False
        store = self._ensure_store()

        try:
            result_id = store.save_evidence(
                evidence_id=evidence_id,
                source=source,
                title=title,
                snippet=snippet,
                url=url,
                reliability_score=reliability_score,
                metadata=metadata,
                debate_id=debate_id,
            )

            # Emit event for WebSocket updates
            self._emit_event(
                "knowledge_indexed",
                {
                    "source": "evidence",
                    "evidence_id": result_id,
                    "title": title,
                    "reliability": reliability_score,
                    "debate_id": debate_id,
                },
            )

            success = True

            # Check SLO if available
            latency_ms = (time.time() - start) * 1000
            if SLO_AVAILABLE:
                check_and_record_slo("evidence_store", latency_ms)

            return result_id

        except EvidenceStoreUnavailableError:
            raise
        except Exception as e:
            logger.error(f"Failed to store evidence {evidence_id}: {e}")
            raise EvidenceAdapterError(f"Storage failed: {e}") from e
        finally:
            self._record_metric("store", success, time.time() - start)

    def mark_used_in_consensus(
        self,
        evidence_id: str,
        debate_id: str,
    ) -> None:
        """
        Mark evidence as used in consensus (boost reliability).

        Args:
            evidence_id: The evidence ID
            debate_id: The debate ID where consensus was reached

        Raises:
            EvidenceStoreUnavailableError: If store is not configured
            EvidenceAdapterError: If marking fails
        """
        store = self._ensure_store()

        try:
            store.mark_used_in_consensus(debate_id, evidence_id)
            logger.debug(
                f"Marked evidence {evidence_id} as used in consensus for debate {debate_id}"
            )
        except AttributeError:
            # Store doesn't support this method
            logger.debug("Evidence store does not support mark_used_in_consensus")
        except Exception as e:
            logger.error(f"Failed to mark evidence {evidence_id} in consensus: {e}")
            raise EvidenceAdapterError(f"Failed to mark consensus usage: {e}") from e

    async def update_reliability_from_km(
        self,
        evidence_id: str,
        km_validation: Dict[str, Any],
    ) -> None:
        """
        Update evidence reliability based on KM validation feedback.

        This is the reverse flow: KM validation improves evidence scores.

        Args:
            evidence_id: The evidence ID
            km_validation: Validation data from Knowledge Mound

        Raises:
            EvidenceStoreUnavailableError: If store is not configured
            EvidenceNotFoundError: If evidence item is not found
            EvidenceAdapterError: If update fails
        """
        store = self._ensure_store()

        # Strip prefix if present
        if evidence_id.startswith(self.ID_PREFIX):
            evidence_id = evidence_id[len(self.ID_PREFIX) :]

        try:
            # Get current evidence
            evidence = store.get_evidence(evidence_id)
            if not evidence:
                raise EvidenceNotFoundError(f"Evidence not found: {evidence_id}")

            # Calculate new reliability based on KM feedback
            current_reliability = evidence.get("reliability_score", 0.5)
            km_confidence = km_validation.get("confidence", 0.5)
            validation_count = km_validation.get("validation_count", 1)

            # Weighted average: more validations = more weight on KM confidence
            weight = min(0.5, validation_count * 0.1)  # Max 50% weight
            new_reliability = current_reliability * (1 - weight) + km_confidence * weight

            # Update the evidence
            store.update_evidence(
                evidence_id,
                reliability_score=new_reliability,
            )

            logger.info(
                f"Updated evidence reliability from KM: {evidence_id} "
                f"{current_reliability:.2f} -> {new_reliability:.2f}"
            )

        except EvidenceNotFoundError:
            logger.warning(f"Evidence not found for KM validation: {evidence_id}")
            raise
        except EvidenceStoreUnavailableError:
            raise
        except Exception as e:
            logger.error(f"Failed to update reliability for evidence {evidence_id}: {e}")
            raise EvidenceAdapterError(f"Reliability update failed: {e}") from e

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the evidence store.

        Returns:
            Dict with store statistics

        Raises:
            EvidenceStoreUnavailableError: If store is not configured
            EvidenceAdapterError: If stats retrieval fails
        """
        store = self._ensure_store()

        try:
            return store.get_stats()
        except AttributeError:
            # Store doesn't support get_stats
            logger.debug("Evidence store does not support get_stats")
            return {"error": "stats not supported", "store_type": type(store).__name__}
        except Exception as e:
            logger.error(f"Failed to get evidence store stats: {e}")
            raise EvidenceAdapterError(f"Stats retrieval failed: {e}") from e

    def get_debate_evidence(
        self,
        debate_id: str,
        min_relevance: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Get all evidence associated with a debate.

        Args:
            debate_id: The debate ID
            min_relevance: Minimum relevance score

        Returns:
            List of evidence items for the debate

        Raises:
            EvidenceStoreUnavailableError: If store is not configured
            EvidenceAdapterError: If retrieval fails
        """
        store = self._ensure_store()

        try:
            return store.get_debate_evidence(debate_id, min_relevance)
        except AttributeError:
            # Store doesn't support debate evidence lookup
            logger.debug("Evidence store does not support get_debate_evidence")
            return []
        except Exception as e:
            logger.error(f"Failed to get debate evidence for {debate_id}: {e}")
            raise EvidenceAdapterError(f"Debate evidence retrieval failed: {e}") from e


__all__ = [
    "EvidenceAdapter",
    "EvidenceSearchResult",
    "EvidenceAdapterError",
    "EvidenceStoreUnavailableError",
    "EvidenceNotFoundError",
]
