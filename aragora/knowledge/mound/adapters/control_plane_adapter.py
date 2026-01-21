"""
Control Plane Adapter for Knowledge Mound Integration.

Bridges the Control Plane with Knowledge Mound to enable:
- Cross-workspace knowledge sharing
- Task performance history
- Agent capability learning
- Organizational knowledge federation

ID Prefixes:
- cp_task_: Task outcome records
- cp_agent_: Agent performance data
- cp_capability_: Capability patterns
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.knowledge.mound.core import KnowledgeMound
    from aragora.control_plane.coordinator import ControlPlaneCoordinator
    from aragora.control_plane.scheduler import Task

logger = logging.getLogger(__name__)

# Type alias for event callbacks
EventCallback = Callable[[str, Dict[str, Any]], None]


@dataclass
class TaskOutcome:
    """Record of a completed task for KM storage."""

    task_id: str
    task_type: str
    agent_id: str
    success: bool
    duration_seconds: float
    workspace_id: str = "default"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCapabilityRecord:
    """Record of agent capability performance."""

    agent_id: str
    capability: str
    success_count: int
    failure_count: int
    avg_duration_seconds: float
    workspace_id: str = "default"
    confidence: float = 0.8


@dataclass
class CrossWorkspaceInsight:
    """Insight shared across workspaces."""

    insight_id: str
    source_workspace: str
    target_workspaces: List[str]
    task_type: str
    content: str
    confidence: float
    created_at: str


class ControlPlaneAdapter:
    """
    Adapter that bridges Control Plane to the Knowledge Mound.

    Provides bidirectional sync between Control Plane operations and KM:
    - Forward: Task outcomes and agent performance → KM
    - Reverse: Historical patterns → Control Plane decisions

    Usage:
        from aragora.control_plane.coordinator import ControlPlaneCoordinator
        from aragora.knowledge.mound.adapters import ControlPlaneAdapter

        coordinator = await ControlPlaneCoordinator.create()
        adapter = ControlPlaneAdapter(coordinator)

        # Store task outcome
        await adapter.store_task_outcome(task_outcome)

        # Get agent capability recommendations
        recommendations = await adapter.get_capability_recommendations("debate")
    """

    def __init__(
        self,
        coordinator: Optional["ControlPlaneCoordinator"] = None,
        knowledge_mound: Optional["KnowledgeMound"] = None,
        workspace_id: str = "default",
        event_callback: Optional[EventCallback] = None,
        min_task_confidence: float = 0.6,
        min_capability_sample_size: int = 5,
    ):
        """
        Initialize the adapter.

        Args:
            coordinator: The ControlPlaneCoordinator to wrap
            knowledge_mound: Optional KnowledgeMound for direct storage
            workspace_id: Workspace ID for multi-tenancy
            event_callback: Optional callback for emitting events
            min_task_confidence: Minimum confidence to store task outcomes
            min_capability_sample_size: Minimum samples before capability recommendations
        """
        self._coordinator = coordinator
        self._knowledge_mound = knowledge_mound
        self._workspace_id = workspace_id
        self._event_callback = event_callback
        self._min_task_confidence = min_task_confidence
        self._min_capability_sample_size = min_capability_sample_size

        # Caches
        self._capability_cache: Dict[str, List[AgentCapabilityRecord]] = {}
        self._cache_ttl: float = 300  # 5 minutes
        self._cache_times: Dict[str, float] = {}

        # Stats
        self._stats = {
            "task_outcomes_stored": 0,
            "capability_records_stored": 0,
            "capability_queries": 0,
            "cross_workspace_shares": 0,
        }

    def set_event_callback(self, callback: EventCallback) -> None:
        """Set the event callback for WebSocket notifications."""
        self._event_callback = callback

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event if callback is configured."""
        if self._event_callback:
            try:
                self._event_callback(event_type, data)
            except Exception as e:
                logger.debug(f"Event emission failed: {e}")

    # =========================================================================
    # Forward Sync: Control Plane → KM
    # =========================================================================

    async def store_task_outcome(
        self,
        outcome: TaskOutcome,
    ) -> Optional[str]:
        """
        Store a task outcome in the Knowledge Mound.

        Args:
            outcome: Task outcome to store

        Returns:
            KM item ID if stored, None if below threshold
        """
        if not self._knowledge_mound:
            logger.debug("No KnowledgeMound configured, skipping task outcome storage")
            return None

        try:
            from aragora.knowledge.unified.types import (
                KnowledgeItem,
                KnowledgeSource,
                ConfidenceLevel,
            )

            # Calculate confidence based on consistency
            confidence_val = 0.8 if outcome.success else 0.5

            if confidence_val < self._min_task_confidence:
                logger.debug(
                    f"Task outcome below confidence threshold: {confidence_val:.2f}"
                )
                return None

            # Map confidence value to enum
            confidence = ConfidenceLevel.HIGH if outcome.success else ConfidenceLevel.MEDIUM
            now = datetime.now()

            # Create knowledge item
            item = KnowledgeItem(
                id=f"cp_task_{outcome.task_id}",
                content=(
                    f"Task {outcome.task_type} completed by {outcome.agent_id}: "
                    f"{'success' if outcome.success else 'failure'} in "
                    f"{outcome.duration_seconds:.2f}s"
                ),
                source=KnowledgeSource.CONTINUUM,  # Use existing source type
                source_id=outcome.task_id,
                confidence=confidence,
                created_at=now,
                updated_at=now,
                metadata={
                    "type": "control_plane_task_outcome",
                    "task_id": outcome.task_id,
                    "task_type": outcome.task_type,
                    "agent_id": outcome.agent_id,
                    "success": outcome.success,
                    "duration_seconds": outcome.duration_seconds,
                    "error_message": outcome.error_message,
                    "workspace_id": outcome.workspace_id,
                    **outcome.metadata,
                },
            )

            item_id = await self._knowledge_mound.ingest(item)

            self._stats["task_outcomes_stored"] += 1
            self._emit_event("cp_task_outcome_stored", {
                "task_id": outcome.task_id,
                "agent_id": outcome.agent_id,
                "success": outcome.success,
            })

            logger.debug(
                f"Stored task outcome: task={outcome.task_id} "
                f"agent={outcome.agent_id} success={outcome.success}"
            )

            return item_id

        except Exception as e:
            logger.error(f"Failed to store task outcome: {e}")
            return None

    async def store_capability_record(
        self,
        record: AgentCapabilityRecord,
    ) -> Optional[str]:
        """
        Store an agent capability record in the Knowledge Mound.

        Args:
            record: Capability record to store

        Returns:
            KM item ID if stored, None if below threshold
        """
        if not self._knowledge_mound:
            return None

        try:
            from aragora.knowledge.unified.types import (
                KnowledgeItem,
                KnowledgeSource,
                ConfidenceLevel,
            )

            total_tasks = record.success_count + record.failure_count
            if total_tasks < self._min_capability_sample_size:
                logger.debug(
                    f"Capability record below sample threshold: {total_tasks}"
                )
                return None

            success_rate = record.success_count / total_tasks if total_tasks > 0 else 0

            # Map confidence value to enum
            if record.confidence >= 0.8:
                confidence = ConfidenceLevel.HIGH
            elif record.confidence >= 0.6:
                confidence = ConfidenceLevel.MEDIUM
            else:
                confidence = ConfidenceLevel.LOW
            now = datetime.now()

            item = KnowledgeItem(
                id=f"cp_capability_{record.agent_id}_{record.capability}",
                content=(
                    f"Agent {record.agent_id} capability '{record.capability}': "
                    f"{success_rate:.1%} success rate over {total_tasks} tasks, "
                    f"avg duration {record.avg_duration_seconds:.2f}s"
                ),
                source=KnowledgeSource.ELO,  # Use existing source type
                source_id=f"{record.agent_id}_{record.capability}",
                confidence=confidence,
                created_at=now,
                updated_at=now,
                metadata={
                    "type": "control_plane_capability",
                    "agent_id": record.agent_id,
                    "capability": record.capability,
                    "success_count": record.success_count,
                    "failure_count": record.failure_count,
                    "success_rate": success_rate,
                    "avg_duration_seconds": record.avg_duration_seconds,
                    "workspace_id": record.workspace_id,
                },
            )

            item_id = await self._knowledge_mound.ingest(item)

            self._stats["capability_records_stored"] += 1

            # Invalidate cache for this capability
            if record.capability in self._capability_cache:
                del self._capability_cache[record.capability]

            return item_id

        except Exception as e:
            logger.error(f"Failed to store capability record: {e}")
            return None

    # =========================================================================
    # Reverse Flow: KM → Control Plane
    # =========================================================================

    async def get_capability_recommendations(
        self,
        capability: str,
        limit: int = 5,
        use_cache: bool = True,
    ) -> List[AgentCapabilityRecord]:
        """
        Get agent recommendations for a capability from KM.

        Queries historical capability performance to recommend agents.

        Args:
            capability: Capability to query
            limit: Maximum agents to return
            use_cache: Whether to use cached results

        Returns:
            List of AgentCapabilityRecord sorted by success rate
        """
        self._stats["capability_queries"] += 1

        # Check cache
        if use_cache:
            cache_key = capability.lower()
            if cache_key in self._capability_cache:
                cache_time = self._cache_times.get(cache_key, 0)
                if time.time() - cache_time < self._cache_ttl:
                    return self._capability_cache[cache_key][:limit]

        if not self._knowledge_mound:
            return []

        try:
            # Query KM for capability records
            results = await self._knowledge_mound.query(
                query=f"agent capability {capability}",
                limit=limit * 2,  # Query more, then filter
                workspace_id=self._workspace_id,
            )

            records = []
            for result in results:
                metadata = result.get("metadata", {})
                if metadata.get("type") != "control_plane_capability":
                    continue

                record = AgentCapabilityRecord(
                    agent_id=metadata.get("agent_id", ""),
                    capability=metadata.get("capability", ""),
                    success_count=metadata.get("success_count", 0),
                    failure_count=metadata.get("failure_count", 0),
                    avg_duration_seconds=metadata.get("avg_duration_seconds", 0),
                    workspace_id=self._workspace_id,
                    confidence=result.get("confidence", 0.8),
                )
                if record.agent_id:
                    records.append(record)

            # Sort by success rate
            records.sort(
                key=lambda r: r.success_count / max(1, r.success_count + r.failure_count),
                reverse=True,
            )

            # Cache results
            if use_cache:
                cache_key = capability.lower()
                self._capability_cache[cache_key] = records
                self._cache_times[cache_key] = time.time()

            return records[:limit]

        except Exception as e:
            logger.error(f"Failed to get capability recommendations: {e}")
            return []

    async def get_task_history(
        self,
        task_type: str,
        limit: int = 20,
    ) -> List[TaskOutcome]:
        """
        Get historical task outcomes for a task type.

        Args:
            task_type: Type of task to query
            limit: Maximum outcomes to return

        Returns:
            List of TaskOutcome sorted by recency
        """
        if not self._knowledge_mound:
            return []

        try:
            results = await self._knowledge_mound.query(
                query=f"task {task_type}",
                limit=limit,
                workspace_id=self._workspace_id,
            )

            outcomes = []
            for result in results:
                metadata = result.get("metadata", {})
                if metadata.get("type") != "control_plane_task_outcome":
                    continue

                outcome = TaskOutcome(
                    task_id=metadata.get("task_id", ""),
                    task_type=metadata.get("task_type", ""),
                    agent_id=metadata.get("agent_id", ""),
                    success=metadata.get("success", False),
                    duration_seconds=metadata.get("duration_seconds", 0),
                    workspace_id=self._workspace_id,
                    error_message=metadata.get("error_message"),
                )
                if outcome.task_id:
                    outcomes.append(outcome)

            return outcomes

        except Exception as e:
            logger.error(f"Failed to get task history: {e}")
            return []

    # =========================================================================
    # Cross-Workspace Federation
    # =========================================================================

    async def share_insight_cross_workspace(
        self,
        insight: CrossWorkspaceInsight,
    ) -> bool:
        """
        Share an insight across workspaces via KM federation.

        Args:
            insight: Insight to share

        Returns:
            True if shared successfully
        """
        if not self._knowledge_mound:
            return False

        try:
            from aragora.knowledge.unified.types import (
                KnowledgeItem,
                KnowledgeSource,
                ConfidenceLevel,
            )

            # Map confidence value to enum
            if insight.confidence >= 0.8:
                confidence = ConfidenceLevel.HIGH
            elif insight.confidence >= 0.6:
                confidence = ConfidenceLevel.MEDIUM
            else:
                confidence = ConfidenceLevel.LOW
            now = datetime.now()

            # Create item with organization-wide visibility
            item = KnowledgeItem(
                id=f"cp_insight_{insight.insight_id}",
                content=insight.content,
                source=KnowledgeSource.INSIGHT,
                source_id=insight.insight_id,
                confidence=confidence,
                created_at=now,
                updated_at=now,
                metadata={
                    "type": "cross_workspace_insight",
                    "insight_id": insight.insight_id,
                    "source_workspace": insight.source_workspace,
                    "target_workspaces": insight.target_workspaces,
                    "task_type": insight.task_type,
                    "workspace_id": insight.source_workspace,
                },
            )

            # Ingest with organization visibility
            await self._knowledge_mound.ingest(item)

            self._stats["cross_workspace_shares"] += 1
            self._emit_event("cp_insight_shared", {
                "insight_id": insight.insight_id,
                "source_workspace": insight.source_workspace,
                "target_count": len(insight.target_workspaces),
            })

            return True

        except Exception as e:
            logger.error(f"Failed to share cross-workspace insight: {e}")
            return False

    async def get_cross_workspace_insights(
        self,
        task_type: str,
        limit: int = 10,
    ) -> List[CrossWorkspaceInsight]:
        """
        Get insights from other workspaces for a task type.

        Args:
            task_type: Type of task to query insights for
            limit: Maximum insights to return

        Returns:
            List of CrossWorkspaceInsight from other workspaces
        """
        if not self._knowledge_mound:
            return []

        try:
            # Query for organization-wide insights
            results = await self._knowledge_mound.query(
                query=f"cross workspace insight {task_type}",
                limit=limit,
                workspace_id="__organization__",  # Query across workspaces
            )

            insights = []
            for result in results:
                metadata = result.get("metadata", {})
                if metadata.get("type") != "cross_workspace_insight":
                    continue

                # Skip insights from our own workspace
                if metadata.get("source_workspace") == self._workspace_id:
                    continue

                insight = CrossWorkspaceInsight(
                    insight_id=metadata.get("insight_id", ""),
                    source_workspace=metadata.get("source_workspace", ""),
                    target_workspaces=metadata.get("target_workspaces", []),
                    task_type=metadata.get("task_type", ""),
                    content=result.get("content", ""),
                    confidence=result.get("confidence", 0.8),
                    created_at=metadata.get("created_at", ""),
                )
                if insight.insight_id:
                    insights.append(insight)

            return insights

        except Exception as e:
            logger.error(f"Failed to get cross-workspace insights: {e}")
            return []

    # =========================================================================
    # Stats and Metrics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            **self._stats,
            "workspace_id": self._workspace_id,
            "cache_size": len(self._capability_cache),
            "has_knowledge_mound": self._knowledge_mound is not None,
            "has_coordinator": self._coordinator is not None,
        }

    def clear_cache(self) -> int:
        """Clear all caches and return count of cleared items."""
        count = len(self._capability_cache)
        self._capability_cache.clear()
        self._cache_times.clear()
        return count


__all__ = [
    "ControlPlaneAdapter",
    "TaskOutcome",
    "AgentCapabilityRecord",
    "CrossWorkspaceInsight",
]
