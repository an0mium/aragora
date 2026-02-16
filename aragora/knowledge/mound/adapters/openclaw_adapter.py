"""
OpenClawAdapter - Bridges OpenClaw to the Knowledge Mound.

This adapter enables bidirectional integration between OpenClaw (the external
task execution agent) and the Knowledge Mound:

**Forward Sync (OpenClaw -> KM):**
- Sync OpenClaw action results to KM as knowledge items
- Store execution logs for auditability and pattern analysis
- Index OpenClaw session context for debate retrieval

**Reverse Sync (KM -> OpenClaw):**
- Push debate decisions to OpenClaw for task prioritization
- Update OpenClaw context with KM insights and patterns
- Sync knowledge items for action context enrichment

**Bidirectional Learning:**
- Extract patterns from successful actions for reuse
- Identify failure patterns for avoidance
- Cross-debate learning from action outcomes

**Integration Points:**
- Subscribe to OpenClaw action events
- Emit KM validation events for downstream systems
- Support batch sync operations for efficiency

ID Prefix: oc_
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter
from aragora.knowledge.mound.adapters._fusion_mixin import FusionMixin
from aragora.knowledge.mound.adapters._semantic_mixin import SemanticSearchMixin

# Import learning mixin from split module
from aragora.knowledge.mound.adapters.openclaw_adapter_learning import (
    OpenClawLearningMixin,
)

# Import dataclasses/enums from split models module
from aragora.knowledge.mound.adapters.openclaw_adapter_models import (
    ActionPattern,
    ActionStatus,
    KMContextUpdate,
    KMValidationResult,
    OpenClawKMSyncResult,
    OpenClawKnowledgeItem,
    PatternType,
    SyncResult,
    TaskPrioritizationUpdate,
)

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import KnowledgeItem

# Type alias for event callback
EventCallback = Callable[[str, dict[str, Any]], None]

logger = logging.getLogger(__name__)


# Dataclass definitions (ActionStatus, PatternType, OpenClawKnowledgeItem,
# ActionPattern, SyncResult, KMContextUpdate, TaskPrioritizationUpdate,
# KMValidationResult, OpenClawKMSyncResult) have been moved to
# openclaw_adapter_models.py and are imported above for backward compatibility.


# =============================================================================
# OpenClaw Adapter Implementation
# =============================================================================


class OpenClawAdapter(OpenClawLearningMixin, FusionMixin, SemanticSearchMixin, KnowledgeMoundAdapter):
    """
    Adapter that bridges OpenClaw to the Knowledge Mound.

    Provides bidirectional synchronization between OpenClaw's action
    execution system and the Knowledge Mound's knowledge storage:

    **Forward Sync (OpenClaw -> KM):**
    - `store_action_result()`: Store individual action results
    - `sync_actions_to_mound()`: Batch sync multiple actions
    - `store_execution_log()`: Store detailed execution logs
    - `index_session_context()`: Index session context for retrieval

    **Reverse Sync (KM -> OpenClaw):**
    - `push_debate_decisions()`: Push debate decisions for task prioritization
    - `update_openclaw_context()`: Push KM insights to OpenClaw context
    - `sync_knowledge_for_action()`: Sync relevant knowledge for an action

    **Bidirectional Learning:**
    - `extract_action_patterns()`: Extract patterns from action outcomes
    - `get_failure_patterns()`: Get patterns to avoid
    - `get_success_patterns()`: Get patterns to replicate
    - `cross_debate_learning()`: Learn from action outcomes across debates

    Usage:
        adapter = OpenClawAdapter()

        # Store action result
        action = OpenClawKnowledgeItem(
            action_id="action_123",
            result=ActionStatus.SUCCESS,
            context={"task": "web_search", "query": "latest news"},
            capabilities_used=["web_search"],
        )
        item_id = adapter.store_action_result(action)

        # Extract patterns from outcomes
        patterns = await adapter.extract_action_patterns(
            workspace_id="ws_123",
            min_observations=5,
        )

        # Push debate decisions to OpenClaw
        await adapter.push_debate_decisions(
            debate_id="debate_456",
            decisions=[{"task_id": "task_1", "priority": 0.9}],
        )
    """

    ID_PREFIX = "oc_"

    # Mixin configuration
    adapter_name = "openclaw"
    source_type = "openclaw"

    # Quality thresholds
    MIN_ACTION_CONFIDENCE = 0.5
    MIN_PATTERN_OBSERVATIONS = 3
    PATTERN_CONFIDENCE_DECAY = 0.95  # Daily decay rate

    def __init__(
        self,
        openclaw_client: Any | None = None,
        enable_dual_write: bool = False,
        event_callback: EventCallback | None = None,
        enable_tracing: bool = True,
        enable_resilience: bool = True,
        resilience_timeout: float = 30.0,
    ) -> None:
        """
        Initialize the OpenClaw adapter.

        Args:
            openclaw_client: Optional OpenClaw client instance for reverse sync.
            enable_dual_write: If True, writes go to both systems during migration.
            event_callback: Optional callback for emitting events (event_type, data).
            enable_tracing: If True, OpenTelemetry tracing is enabled.
            enable_resilience: If True, enables circuit breaker and bulkhead protection.
            resilience_timeout: Default timeout for resilient operations in seconds.
        """
        super().__init__(
            enable_dual_write=enable_dual_write,
            event_callback=event_callback,
            enable_tracing=enable_tracing,
            enable_resilience=enable_resilience,
            resilience_timeout=resilience_timeout,
        )

        self._openclaw_client = openclaw_client

        # In-memory storage (will be replaced by KM backend in production)
        self._actions: dict[str, dict[str, Any]] = {}
        self._patterns: dict[str, dict[str, Any]] = {}
        self._execution_logs: dict[str, list[dict[str, Any]]] = {}
        self._session_contexts: dict[str, dict[str, Any]] = {}

        # Reverse flow state
        self._context_updates: dict[str, KMContextUpdate] = {}
        self._prioritization_updates: dict[str, TaskPrioritizationUpdate] = {}
        self._km_validations: dict[str, KMValidationResult] = {}

        # Indices for fast lookup
        self._capability_actions: dict[str, list[str]] = {}
        self._debate_actions: dict[str, list[str]] = {}
        self._action_hash_map: dict[str, str] = {}

        # Initialize fusion state
        self._init_fusion_state()

    def set_openclaw_client(self, client: Any) -> None:
        """Set the OpenClaw client for reverse sync operations.

        Args:
            client: OpenClaw client instance.
        """
        self._openclaw_client = client

    # =========================================================================
    # FusionMixin Required Method Implementations
    # =========================================================================

    def _get_fusion_sources(self) -> list[str]:
        """Return list of source adapters this adapter can fuse data from."""
        return ["consensus", "evidence", "belief", "continuum", "insights"]

    def _extract_fusible_data(
        self,
        km_item: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Extract fusible data from a KM item."""
        metadata = km_item.get("metadata", {})
        confidence = km_item.get("confidence") or metadata.get("confidence")

        if confidence is None:
            return None

        return {
            "confidence": float(confidence),
            "is_valid": float(confidence) >= 0.5,
            "source_id": km_item.get("id") or metadata.get("source_id"),
            "action_status": metadata.get("action_status"),
            "capabilities": metadata.get("capabilities_used", []),
            "sources": metadata.get("sources", []),
            "reasoning": metadata.get("reasoning"),
        }

    def _apply_fusion_result(
        self,
        record: Any,
        fusion_result: Any,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Apply a fusion result to an OpenClaw action record."""
        try:
            fused_confidence = getattr(fusion_result, "fused_confidence", None)
            if fused_confidence is None:
                return False

            if isinstance(record, dict):
                record["km_fused"] = True
                record["km_fused_confidence"] = fused_confidence
                record["km_fusion_time"] = datetime.now(timezone.utc).isoformat()
                if metadata:
                    record["km_fusion_metadata"] = metadata

                logger.debug(
                    f"Applied fusion result to OpenClaw action: "
                    f"fused_confidence={fused_confidence:.3f}"
                )
                return True

            return False
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.warning(f"Failed to apply fusion result to OpenClaw record: {e}")
            return False

    # =========================================================================
    # SemanticSearchMixin Required Method Implementations
    # =========================================================================

    def _get_record_by_id(self, record_id: str) -> Any | None:
        """Get an action record by ID."""
        raw_id = record_id
        if record_id.startswith(self.ID_PREFIX):
            raw_id = record_id[len(self.ID_PREFIX) :]

        # Check actions
        for storage in (self._actions, self._patterns, self._session_contexts):
            if record_id in storage:
                return storage[record_id]
            if raw_id in storage:
                return storage[raw_id]
            prefixed = f"{self.ID_PREFIX}{raw_id}"
            if prefixed in storage:
                return storage[prefixed]

        return None

    def _record_to_dict(self, record: Any, similarity: float = 0.0) -> dict[str, Any]:
        """Convert an OpenClaw record to dict."""
        if isinstance(record, dict):
            result = dict(record)
            result["similarity"] = similarity
            return result

        if isinstance(record, OpenClawKnowledgeItem):
            result = record.to_dict()
            result["similarity"] = similarity
            return result

        return {
            "id": getattr(record, "id", getattr(record, "action_id", None)),
            "content": getattr(record, "output", ""),
            "confidence": getattr(record, "confidence", 0.0),
            "similarity": similarity,
            "metadata": getattr(record, "metadata", {}),
        }

    def _extract_record_id(self, source_id: str) -> str:
        """Extract record ID from prefixed source ID."""
        if source_id.startswith(self.ID_PREFIX):
            return source_id[len(self.ID_PREFIX) :]
        return source_id

    # =========================================================================
    # Forward Sync Methods (OpenClaw -> KM)
    # =========================================================================

    def store_action_result(
        self,
        action: OpenClawKnowledgeItem,
    ) -> str:
        """
        Store an OpenClaw action result in the Knowledge Mound.

        Args:
            action: The OpenClaw action result to store.

        Returns:
            The action ID in the Knowledge Mound.
        """
        start = time.time()
        success = False

        try:
            # Generate ID
            action_hash = hashlib.sha256(
                f"{action.action_id}:{action.created_at.isoformat()}".encode()
            ).hexdigest()[:16]
            item_id = f"{self.ID_PREFIX}action_{action_hash}"

            # Build action data
            action_data = action.to_dict()
            action_data["id"] = item_id
            action_data["action_hash"] = action_hash

            # Calculate confidence based on result
            confidence = self._calculate_action_confidence(action)
            action_data["confidence"] = confidence

            # Store
            self._actions[item_id] = action_data
            self._action_hash_map[action_hash] = item_id

            # Update indices
            for capability in action.capabilities_used:
                if capability not in self._capability_actions:
                    self._capability_actions[capability] = []
                self._capability_actions[capability].append(item_id)

            if action.debate_id:
                if action.debate_id not in self._debate_actions:
                    self._debate_actions[action.debate_id] = []
                self._debate_actions[action.debate_id].append(item_id)

            # Emit event
            self._emit_event(
                "km_adapter_forward_sync",
                {
                    "source": "openclaw",
                    "action_id": item_id,
                    "result": action.result.value,
                    "capabilities": action.capabilities_used,
                    "confidence": confidence,
                },
            )

            logger.info(
                f"Stored OpenClaw action: {item_id} "
                f"(result={action.result.value}, confidence={confidence:.2f})"
            )

            success = True
            return item_id

        finally:
            self._record_metric("store", success, time.time() - start)

    def _calculate_action_confidence(self, action: OpenClawKnowledgeItem) -> float:
        """Calculate confidence score for an action based on its outcome."""
        base_confidence = 0.5

        # Adjust based on result
        result_adjustments = {
            ActionStatus.SUCCESS: 0.3,
            ActionStatus.FAILED: -0.2,
            ActionStatus.TIMEOUT: -0.1,
            ActionStatus.CANCELLED: 0.0,
            ActionStatus.PENDING: 0.0,
            ActionStatus.RUNNING: 0.0,
        }
        base_confidence += result_adjustments.get(action.result, 0.0)

        # Adjust based on execution time (faster is generally better)
        if action.execution_time_ms > 0:
            if action.execution_time_ms < 1000:
                base_confidence += 0.1
            elif action.execution_time_ms > 30000:
                base_confidence -= 0.1

        return max(0.0, min(1.0, base_confidence))

    async def sync_actions_to_mound(
        self,
        mound: Any,
        workspace_id: str,
        min_confidence: float = 0.5,
        limit: int = 100,
    ) -> SyncResult:
        """
        Batch sync OpenClaw actions to the Knowledge Mound.

        Args:
            mound: KnowledgeMound instance to sync to.
            workspace_id: Workspace ID for the KM entries.
            min_confidence: Minimum confidence threshold.
            limit: Maximum entries to sync.

        Returns:
            SyncResult with sync statistics.
        """
        start_time = time.time()
        result = SyncResult(direction="forward")

        actions = list(self._actions.values())[:limit]

        for action_data in actions:
            try:
                confidence = action_data.get("confidence", 0.5)
                if confidence < min_confidence:
                    result.items_skipped += 1
                    continue

                # Check if already synced
                if action_data.get("km_synced"):
                    result.items_skipped += 1
                    continue

                # Create ingestion request
                from aragora.knowledge.mound.types import (
                    IngestionRequest,
                    SourceType,
                )

                content = self._build_action_content(action_data)
                request = IngestionRequest(
                    content=content,
                    source_type=SourceType.FACT,
                    workspace_id=workspace_id,
                    confidence=confidence,
                    metadata={
                        "openclaw_action_id": action_data.get("action_id"),
                        "action_result": action_data.get("result"),
                        "capabilities_used": action_data.get("capabilities_used", []),
                        "execution_time_ms": action_data.get("execution_time_ms", 0),
                    },
                )

                # Ingest into KM
                km_id = await mound.ingest(request)

                # Mark as synced
                action_data["km_synced"] = True
                action_data["km_node_id"] = km_id

                result.items_synced += 1
                logger.debug(f"Synced OpenClaw action to KM: {action_data['id']} -> {km_id}")

            except (RuntimeError, ValueError, OSError, AttributeError) as e:
                error_msg = f"Error syncing action {action_data.get('id')}: {e}"
                logger.warning(error_msg)
                result.errors.append(error_msg)
                result.items_failed += 1

        result.duration_ms = (time.time() - start_time) * 1000

        logger.info(
            f"OpenClaw to KM sync complete: synced={result.items_synced}, "
            f"skipped={result.items_skipped}, failed={result.items_failed}"
        )

        return result

    def _build_action_content(self, action_data: dict[str, Any]) -> str:
        """Build content string for an action for KM storage."""
        parts = []

        if action_data.get("output"):
            parts.append(f"Action output: {action_data['output'][:500]}")

        result = action_data.get("result", "unknown")
        parts.append(f"Result: {result}")

        capabilities = action_data.get("capabilities_used", [])
        if capabilities:
            parts.append(f"Capabilities: {', '.join(capabilities)}")

        context = action_data.get("context", {})
        if context.get("task"):
            parts.append(f"Task: {context['task']}")

        return " | ".join(parts) if parts else "OpenClaw action execution"

    def store_execution_log(
        self,
        action_id: str,
        log_entries: list[dict[str, Any]],
    ) -> str:
        """
        Store detailed execution log for an action.

        Args:
            action_id: The action ID.
            log_entries: List of log entry dicts.

        Returns:
            The log storage ID.
        """
        log_id = f"{self.ID_PREFIX}log_{action_id}"

        if action_id not in self._execution_logs:
            self._execution_logs[action_id] = []

        self._execution_logs[action_id].extend(log_entries)

        self._emit_event(
            "km_openclaw_log_stored",
            {
                "action_id": action_id,
                "entries_count": len(log_entries),
            },
        )

        return log_id

    def index_session_context(
        self,
        session_id: str,
        context: dict[str, Any],
    ) -> str:
        """
        Index OpenClaw session context for debate retrieval.

        Args:
            session_id: The session ID.
            context: Session context to index.

        Returns:
            The context storage ID.
        """
        context_id = f"{self.ID_PREFIX}ctx_{session_id}"

        context_data = {
            "id": context_id,
            "session_id": session_id,
            "context": context,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }

        self._session_contexts[context_id] = context_data

        self._emit_event(
            "km_openclaw_context_indexed",
            {
                "session_id": session_id,
                "context_keys": list(context.keys()),
            },
        )

        return context_id

    # =========================================================================
    # Reverse Sync Methods (KM -> OpenClaw)
    # =========================================================================

    async def push_debate_decisions(
        self,
        debate_id: str,
        decisions: list[dict[str, Any]],
    ) -> int:
        """
        Push debate decisions to OpenClaw for task prioritization.

        Args:
            debate_id: The debate ID.
            decisions: List of decision dicts with task_id and priority.

        Returns:
            Number of decisions pushed.
        """
        pushed_count = 0

        for decision in decisions:
            task_id = decision.get("task_id")
            if not task_id:
                continue

            update = TaskPrioritizationUpdate(
                task_id=task_id,
                debate_id=debate_id,
                original_priority=decision.get("original_priority", 0.5),
                new_priority=decision.get("priority", 0.5),
                reason=decision.get("reason", "debate_decision"),
                confidence=decision.get("confidence", 0.7),
            )

            # Store for tracking
            self._prioritization_updates[task_id] = update

            # Push to OpenClaw if client available
            if self._openclaw_client is not None:
                try:
                    # Assuming client has a method for this
                    if hasattr(self._openclaw_client, "update_task_priority"):
                        await self._openclaw_client.update_task_priority(
                            task_id=task_id,
                            priority=update.new_priority,
                            reason=update.reason,
                        )
                        update.applied = True
                        update.applied_at = datetime.now(timezone.utc)
                except (RuntimeError, ValueError, TypeError, AttributeError) as e:  # noqa: BLE001 - adapter isolation
                    logger.warning(f"Failed to push priority update to OpenClaw: {e}")

            pushed_count += 1

        self._emit_event(
            "km_openclaw_decisions_pushed",
            {
                "debate_id": debate_id,
                "decisions_count": pushed_count,
            },
        )

        return pushed_count

    async def update_openclaw_context(
        self,
        context_updates: list[KMContextUpdate],
    ) -> int:
        """
        Push KM insights to OpenClaw context.

        Args:
            context_updates: List of context updates to push.

        Returns:
            Number of updates applied.
        """
        applied_count = 0

        for update in context_updates:
            # Store for tracking
            self._context_updates[update.update_id] = update

            # Push to OpenClaw if client available
            if self._openclaw_client is not None:
                try:
                    if hasattr(self._openclaw_client, "add_context"):
                        await self._openclaw_client.add_context(
                            context_type=update.context_type,
                            content=update.content,
                            priority=update.priority,
                        )
                        applied_count += 1
                except (RuntimeError, ValueError, TypeError, AttributeError) as e:  # noqa: BLE001 - adapter isolation
                    logger.warning(f"Failed to push context update to OpenClaw: {e}")

        self._emit_event(
            "km_openclaw_context_updated",
            {
                "updates_count": len(context_updates),
                "applied_count": applied_count,
            },
        )

        return applied_count

    async def sync_knowledge_for_action(
        self,
        action_context: dict[str, Any],
        km_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Sync relevant KM knowledge items for an action context.

        Args:
            action_context: Context of the action being planned.
            km_items: KM items to consider for the action.

        Returns:
            List of relevant knowledge items for the action.
        """
        relevant_items = []

        # Extract action context features
        task_type = action_context.get("task_type", "")
        capabilities = action_context.get("capabilities", [])
        query = action_context.get("query", "")

        for item in km_items:
            relevance = self._calculate_item_relevance(item, task_type, capabilities, query)

            if relevance >= 0.5:
                item_copy = dict(item)
                item_copy["action_relevance"] = relevance
                relevant_items.append(item_copy)

        # Sort by relevance
        relevant_items.sort(key=lambda x: x.get("action_relevance", 0), reverse=True)

        return relevant_items[:20]  # Return top 20

    def _calculate_item_relevance(
        self,
        item: dict[str, Any],
        task_type: str,
        capabilities: list[str],
        query: str,
    ) -> float:
        """Calculate relevance of a KM item to an action context."""
        relevance = 0.0
        metadata = item.get("metadata", {})

        # Check capability overlap
        item_capabilities = metadata.get("capabilities_used", [])
        if item_capabilities and capabilities:
            overlap = len(set(item_capabilities) & set(capabilities))
            relevance += overlap * 0.2

        # Check task type match
        item_task = metadata.get("task_type", "")
        if item_task and task_type and item_task.lower() == task_type.lower():
            relevance += 0.3

        # Check content relevance (simple keyword matching)
        content = item.get("content", "")
        if query and content:
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            word_overlap = len(query_words & content_words)
            relevance += min(0.3, word_overlap * 0.05)

        # Boost by confidence
        confidence = item.get("confidence", 0.5)
        relevance *= 0.5 + confidence * 0.5

        return min(1.0, relevance)

    # Bidirectional Learning Methods and Batch Sync Operations are provided
    # by OpenClawLearningMixin (openclaw_adapter_learning.py):
    # - extract_action_patterns()
    # - get_failure_patterns()
    # - get_success_patterns()
    # - cross_debate_learning()
    # - sync_validations_from_km()

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_action(self, action_id: str) -> dict[str, Any] | None:
        """Get a specific action by ID."""
        if not action_id.startswith(self.ID_PREFIX):
            action_id = f"{self.ID_PREFIX}action_{action_id}"
        return self._actions.get(action_id)

    def get_pattern(self, pattern_id: str) -> dict[str, Any] | None:
        """Get a specific pattern by ID."""
        if not pattern_id.startswith(self.ID_PREFIX):
            pattern_id = f"{self.ID_PREFIX}pattern_{pattern_id}"
        return self._patterns.get(pattern_id)

    def search_actions_by_capability(
        self,
        capability: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search actions by capability."""
        action_ids = self._capability_actions.get(capability, [])
        return [self._actions[aid] for aid in action_ids[:limit] if aid in self._actions]

    def search_actions_by_debate(
        self,
        debate_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search actions by debate ID."""
        action_ids = self._debate_actions.get(debate_id, [])
        return [self._actions[aid] for aid in action_ids[:limit] if aid in self._actions]

    def to_knowledge_item(self, action: dict[str, Any]) -> KnowledgeItem:
        """Convert an action dict to a KnowledgeItem."""
        from aragora.knowledge.mound.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        confidence_value = action.get("confidence", 0.5)
        if confidence_value >= 0.8:
            confidence = ConfidenceLevel.HIGH
        elif confidence_value >= 0.6:
            confidence = ConfidenceLevel.MEDIUM
        elif confidence_value >= 0.4:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.UNVERIFIED

        created_at = action.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = datetime.now(timezone.utc)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return KnowledgeItem(
            id=action.get("id", ""),
            content=self._build_action_content(action),
            source=KnowledgeSource.FACT,
            source_id=action.get("action_id", action.get("id", "")),
            confidence=confidence,
            created_at=created_at,
            updated_at=created_at,
            metadata={
                "openclaw_action_id": action.get("action_id"),
                "result": action.get("result"),
                "capabilities_used": action.get("capabilities_used", []),
                "execution_time_ms": action.get("execution_time_ms", 0),
                "debate_id": action.get("debate_id"),
            },
            importance=confidence_value,
        )

    # =========================================================================
    # Statistics and Health
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about stored OpenClaw data."""
        return {
            "total_actions": len(self._actions),
            "total_patterns": len(self._patterns),
            "total_execution_logs": sum(len(logs) for logs in self._execution_logs.values()),
            "total_session_contexts": len(self._session_contexts),
            "capabilities_tracked": len(self._capability_actions),
            "debates_tracked": len(self._debate_actions),
            "km_validations": len(self._km_validations),
            "context_updates": len(self._context_updates),
            "prioritization_updates": len(self._prioritization_updates),
        }

    def get_reverse_flow_stats(self) -> dict[str, Any]:
        """Get statistics about reverse flow operations."""
        return {
            "validations_stored": len(self._km_validations),
            "context_updates": len(self._context_updates),
            "prioritization_updates": len(self._prioritization_updates),
            "applied_prioritizations": sum(
                1 for p in self._prioritization_updates.values() if p.applied
            ),
        }

    def clear_reverse_flow_state(self) -> None:
        """Clear all reverse flow state."""
        self._context_updates = {}
        self._prioritization_updates = {}
        self._km_validations = {}


__all__ = [
    "OpenClawAdapter",
    # Dataclasses
    "OpenClawKnowledgeItem",
    "ActionPattern",
    "SyncResult",
    "ActionStatus",
    "PatternType",
    # Reverse flow dataclasses
    "KMContextUpdate",
    "TaskPrioritizationUpdate",
    "KMValidationResult",
    "OpenClawKMSyncResult",
]
