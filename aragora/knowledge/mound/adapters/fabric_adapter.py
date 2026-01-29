"""
Fabric Adapter for Knowledge Mound Integration.

Bridges the Agent Fabric with Knowledge Mound to enable:
- Pool performance history and optimization
- Task scheduling patterns and recommendations
- Budget usage tracking and forecasting
- Policy decision audit trails
- Cross-fabric knowledge sharing

ID Prefixes:
- fabric_pool_: Pool configuration and statistics
- fabric_task_: Task scheduling outcomes
- fabric_budget_: Budget usage snapshots
- fabric_policy_: Policy decision records
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Optional

from ._base import KnowledgeMoundAdapter

if TYPE_CHECKING:
    from aragora.knowledge.mound.core import KnowledgeMound  # type: ignore[attr-defined]
    from aragora.fabric import AgentFabric

logger = logging.getLogger(__name__)

# Type alias for event callbacks
EventCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class PoolSnapshot:
    """Snapshot of pool state for KM storage."""

    pool_id: str
    name: str
    model: str
    current_agents: int
    min_agents: int
    max_agents: int
    tasks_pending: int = 0
    tasks_completed: int = 0
    avg_task_duration_seconds: float = 0.0
    workspace_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSchedulingOutcome:
    """Record of a task scheduling event for KM storage."""

    task_id: str
    task_type: str
    agent_id: str
    pool_id: str | None
    priority: int
    scheduled_at: float
    completed_at: float | None = None
    success: bool = False
    duration_seconds: float = 0.0
    error_message: str | None = None
    workspace_id: str = "default"


@dataclass
class BudgetUsageSnapshot:
    """Snapshot of budget usage for KM storage."""

    entity_id: str
    entity_type: str  # "agent", "pool", "workspace"
    tokens_used: int
    tokens_limit: int
    cost_used_usd: float
    cost_limit_usd: float
    period_start: float
    period_end: float
    alerts_triggered: int = 0
    workspace_id: str = "default"


@dataclass
class PolicyDecisionRecord:
    """Record of a policy decision for audit trails."""

    decision_id: str
    agent_id: str
    action: str
    allowed: bool
    policy_id: str | None
    reason: str
    timestamp: float
    context: dict[str, Any] = field(default_factory=dict)
    workspace_id: str = "default"


class FabricAdapter(KnowledgeMoundAdapter):
    """
    Adapter that bridges Agent Fabric to the Knowledge Mound.

    Provides bidirectional sync between Fabric operations and KM:
    - Forward: Pool stats, task outcomes, budget usage, policy decisions → KM
    - Reverse: Historical patterns → Fabric scheduling decisions

    Usage:
        from aragora.fabric import AgentFabric
        from aragora.knowledge.mound.adapters import FabricAdapter

        fabric = AgentFabric()
        adapter = FabricAdapter(fabric)

        # Store pool snapshot
        await adapter.store_pool_snapshot(pool_snapshot)

        # Get pool recommendations
        recommendations = await adapter.get_pool_recommendations("debate")
    """

    adapter_name = "fabric"

    def __init__(
        self,
        fabric: Optional["AgentFabric"] = None,
        knowledge_mound: Optional["KnowledgeMound"] = None,
        workspace_id: str = "default",
        event_callback: EventCallback | None = None,
        min_confidence_threshold: float = 0.6,
        enable_dual_write: bool = False,
    ):
        """
        Initialize the adapter.

        Args:
            fabric: The AgentFabric instance to wrap
            knowledge_mound: Optional KnowledgeMound for direct storage
            workspace_id: Workspace ID for multi-tenancy
            event_callback: Optional callback for emitting events
            min_confidence_threshold: Minimum confidence to store records
            enable_dual_write: If True, writes to both systems during migration
        """
        super().__init__(
            enable_dual_write=enable_dual_write,
            event_callback=event_callback,
        )
        self._fabric = fabric
        self._knowledge_mound = knowledge_mound
        self._workspace_id = workspace_id
        self._min_confidence_threshold = min_confidence_threshold

        # Caches for reverse flow
        self._pool_performance_cache: dict[str, list[PoolSnapshot]] = {}
        self._task_patterns_cache: dict[str, list[TaskSchedulingOutcome]] = {}
        self._cache_ttl: float = 300  # 5 minutes
        self._cache_times: dict[str, float] = {}

        # Statistics
        self._stats = {
            "pool_snapshots_stored": 0,
            "task_outcomes_stored": 0,
            "budget_snapshots_stored": 0,
            "policy_decisions_stored": 0,
            "pool_queries": 0,
            "task_pattern_queries": 0,
        }

    # =========================================================================
    # Forward Sync: Fabric → KM
    # =========================================================================

    async def store_pool_snapshot(
        self,
        snapshot: PoolSnapshot,
    ) -> str | None:
        """
        Store a pool snapshot in the Knowledge Mound.

        Args:
            snapshot: Pool snapshot to store

        Returns:
            KM item ID if stored, None if below threshold
        """
        if not self._knowledge_mound:
            logger.debug("No KnowledgeMound configured, skipping pool snapshot storage")
            return None

        with self._timed_operation("store_pool_snapshot", pool_id=snapshot.pool_id):
            try:
                from aragora.knowledge.unified.types import (
                    KnowledgeItem,
                    KnowledgeSource,
                    ConfidenceLevel,
                )

                # Calculate confidence based on data completeness
                confidence_val = 0.8 if snapshot.tasks_completed > 0 else 0.6
                if confidence_val < self._min_confidence_threshold:
                    return None

                confidence = (
                    ConfidenceLevel.HIGH if confidence_val >= 0.8 else ConfidenceLevel.MEDIUM
                )
                now = datetime.now()

                utilization = snapshot.current_agents / max(1, snapshot.max_agents)

                item = KnowledgeItem(
                    id=f"fabric_pool_{snapshot.pool_id}_{int(time.time())}",
                    content=(
                        f"Pool '{snapshot.name}' ({snapshot.model}): "
                        f"{snapshot.current_agents}/{snapshot.max_agents} agents, "
                        f"{snapshot.tasks_pending} pending, {snapshot.tasks_completed} completed, "
                        f"avg duration {snapshot.avg_task_duration_seconds:.2f}s"
                    ),
                    source=KnowledgeSource.CONTINUUM,
                    source_id=snapshot.pool_id,
                    confidence=confidence,
                    created_at=now,
                    updated_at=now,
                    metadata={
                        "type": "fabric_pool_snapshot",
                        "pool_id": snapshot.pool_id,
                        "pool_name": snapshot.name,
                        "model": snapshot.model,
                        "current_agents": snapshot.current_agents,
                        "min_agents": snapshot.min_agents,
                        "max_agents": snapshot.max_agents,
                        "utilization": utilization,
                        "tasks_pending": snapshot.tasks_pending,
                        "tasks_completed": snapshot.tasks_completed,
                        "avg_task_duration_seconds": snapshot.avg_task_duration_seconds,
                        "workspace_id": snapshot.workspace_id,
                        **snapshot.metadata,
                    },
                )

                item_id = await self._knowledge_mound.ingest(item)

                self._stats["pool_snapshots_stored"] += 1
                self._emit_event(
                    "fabric_pool_snapshot_stored",
                    {
                        "pool_id": snapshot.pool_id,
                        "current_agents": snapshot.current_agents,
                        "utilization": utilization,
                    },
                )

                # Invalidate cache
                if snapshot.pool_id in self._pool_performance_cache:
                    del self._pool_performance_cache[snapshot.pool_id]

                return item_id

            except Exception as e:
                logger.error(f"Failed to store pool snapshot: {e}")
                return None

    async def store_task_outcome(
        self,
        outcome: TaskSchedulingOutcome,
    ) -> str | None:
        """
        Store a task scheduling outcome in the Knowledge Mound.

        Args:
            outcome: Task outcome to store

        Returns:
            KM item ID if stored, None if below threshold
        """
        if not self._knowledge_mound:
            return None

        with self._timed_operation("store_task_outcome", task_id=outcome.task_id):
            try:
                from aragora.knowledge.unified.types import (
                    KnowledgeItem,
                    KnowledgeSource,
                    ConfidenceLevel,
                )

                confidence = ConfidenceLevel.HIGH if outcome.success else ConfidenceLevel.MEDIUM
                now = datetime.now()

                status = "success" if outcome.success else "failed"
                pool_info = f" in pool {outcome.pool_id}" if outcome.pool_id else ""

                item = KnowledgeItem(
                    id=f"fabric_task_{outcome.task_id}",
                    content=(
                        f"Task {outcome.task_type} ({status}) by agent {outcome.agent_id}"
                        f"{pool_info}: {outcome.duration_seconds:.2f}s"
                    ),
                    source=KnowledgeSource.CONTINUUM,
                    source_id=outcome.task_id,
                    confidence=confidence,
                    created_at=now,
                    updated_at=now,
                    metadata={
                        "type": "fabric_task_outcome",
                        "task_id": outcome.task_id,
                        "task_type": outcome.task_type,
                        "agent_id": outcome.agent_id,
                        "pool_id": outcome.pool_id,
                        "priority": outcome.priority,
                        "scheduled_at": outcome.scheduled_at,
                        "completed_at": outcome.completed_at,
                        "success": outcome.success,
                        "duration_seconds": outcome.duration_seconds,
                        "error_message": outcome.error_message,
                        "workspace_id": outcome.workspace_id,
                    },
                )

                item_id = await self._knowledge_mound.ingest(item)

                self._stats["task_outcomes_stored"] += 1
                self._emit_event(
                    "fabric_task_outcome_stored",
                    {
                        "task_id": outcome.task_id,
                        "task_type": outcome.task_type,
                        "success": outcome.success,
                    },
                )

                # Invalidate task pattern cache
                cache_key = outcome.task_type.lower()
                if cache_key in self._task_patterns_cache:
                    del self._task_patterns_cache[cache_key]

                return item_id

            except Exception as e:
                logger.error(f"Failed to store task outcome: {e}")
                return None

    async def store_budget_snapshot(
        self,
        snapshot: BudgetUsageSnapshot,
    ) -> str | None:
        """
        Store a budget usage snapshot in the Knowledge Mound.

        Args:
            snapshot: Budget snapshot to store

        Returns:
            KM item ID if stored
        """
        if not self._knowledge_mound:
            return None

        with self._timed_operation("store_budget_snapshot", entity_id=snapshot.entity_id):
            try:
                from aragora.knowledge.unified.types import (
                    KnowledgeItem,
                    KnowledgeSource,
                    ConfidenceLevel,
                )

                token_pct = (snapshot.tokens_used / max(1, snapshot.tokens_limit)) * 100
                cost_pct = (snapshot.cost_used_usd / max(0.01, snapshot.cost_limit_usd)) * 100

                # Higher confidence for more complete data
                confidence = (
                    ConfidenceLevel.HIGH if snapshot.tokens_used > 0 else ConfidenceLevel.MEDIUM
                )
                now = datetime.now()

                item = KnowledgeItem(
                    id=f"fabric_budget_{snapshot.entity_id}_{int(snapshot.period_end)}",
                    content=(
                        f"Budget for {snapshot.entity_type} {snapshot.entity_id}: "
                        f"{snapshot.tokens_used:,}/{snapshot.tokens_limit:,} tokens ({token_pct:.1f}%), "
                        f"${snapshot.cost_used_usd:.2f}/${snapshot.cost_limit_usd:.2f} ({cost_pct:.1f}%)"
                    ),
                    source=KnowledgeSource.CONTINUUM,
                    source_id=snapshot.entity_id,
                    confidence=confidence,
                    created_at=now,
                    updated_at=now,
                    metadata={
                        "type": "fabric_budget_snapshot",
                        "entity_id": snapshot.entity_id,
                        "entity_type": snapshot.entity_type,
                        "tokens_used": snapshot.tokens_used,
                        "tokens_limit": snapshot.tokens_limit,
                        "token_utilization_pct": token_pct,
                        "cost_used_usd": snapshot.cost_used_usd,
                        "cost_limit_usd": snapshot.cost_limit_usd,
                        "cost_utilization_pct": cost_pct,
                        "period_start": snapshot.period_start,
                        "period_end": snapshot.period_end,
                        "alerts_triggered": snapshot.alerts_triggered,
                        "workspace_id": snapshot.workspace_id,
                    },
                )

                item_id = await self._knowledge_mound.ingest(item)

                self._stats["budget_snapshots_stored"] += 1
                self._emit_event(
                    "fabric_budget_snapshot_stored",
                    {
                        "entity_id": snapshot.entity_id,
                        "token_pct": token_pct,
                        "cost_pct": cost_pct,
                    },
                )

                return item_id

            except Exception as e:
                logger.error(f"Failed to store budget snapshot: {e}")
                return None

    async def store_policy_decision(
        self,
        record: PolicyDecisionRecord,
    ) -> str | None:
        """
        Store a policy decision record for audit trails.

        Args:
            record: Policy decision to store

        Returns:
            KM item ID if stored
        """
        if not self._knowledge_mound:
            return None

        with self._timed_operation("store_policy_decision", decision_id=record.decision_id):
            try:
                from aragora.knowledge.unified.types import (
                    KnowledgeItem,
                    KnowledgeSource,
                    ConfidenceLevel,
                )

                decision_str = "allowed" if record.allowed else "denied"
                confidence = ConfidenceLevel.HIGH  # Policy decisions are always authoritative
                now = datetime.now()

                item = KnowledgeItem(
                    id=f"fabric_policy_{record.decision_id}",
                    content=(
                        f"Policy decision: {record.action} by {record.agent_id} was {decision_str}. "
                        f"Reason: {record.reason}"
                    ),
                    source=KnowledgeSource.CONTINUUM,
                    source_id=record.decision_id,
                    confidence=confidence,
                    created_at=now,
                    updated_at=now,
                    metadata={
                        "type": "fabric_policy_decision",
                        "decision_id": record.decision_id,
                        "agent_id": record.agent_id,
                        "action": record.action,
                        "allowed": record.allowed,
                        "policy_id": record.policy_id,
                        "reason": record.reason,
                        "timestamp": record.timestamp,
                        "context": record.context,
                        "workspace_id": record.workspace_id,
                    },
                )

                item_id = await self._knowledge_mound.ingest(item)

                self._stats["policy_decisions_stored"] += 1
                self._emit_event(
                    "fabric_policy_decision_stored",
                    {
                        "decision_id": record.decision_id,
                        "action": record.action,
                        "allowed": record.allowed,
                    },
                )

                return item_id

            except Exception as e:
                logger.error(f"Failed to store policy decision: {e}")
                return None

    # =========================================================================
    # Reverse Flow: KM → Fabric
    # =========================================================================

    async def get_pool_performance_history(
        self,
        pool_id: str,
        limit: int = 20,
        use_cache: bool = True,
    ) -> list[PoolSnapshot]:
        """
        Get historical pool performance from KM.

        Args:
            pool_id: Pool ID to query
            limit: Maximum snapshots to return
            use_cache: Whether to use cached results

        Returns:
            List of PoolSnapshot sorted by recency
        """
        self._stats["pool_queries"] += 1

        # Check cache
        if use_cache:
            cache_key = pool_id
            if cache_key in self._pool_performance_cache:
                cache_time = self._cache_times.get(cache_key, 0)
                if time.time() - cache_time < self._cache_ttl:
                    return self._pool_performance_cache[cache_key][:limit]

        if not self._knowledge_mound:
            return []

        with self._timed_operation("get_pool_performance", pool_id=pool_id):
            try:
                results = await self._knowledge_mound.query(
                    query=f"fabric pool {pool_id}",
                    limit=limit * 2,
                    workspace_id=self._workspace_id,
                )

                snapshots = []
                for result in results:
                    metadata = result.get("metadata", {})
                    if metadata.get("type") != "fabric_pool_snapshot":
                        continue
                    if metadata.get("pool_id") != pool_id:
                        continue

                    snapshot = PoolSnapshot(
                        pool_id=metadata.get("pool_id", ""),
                        name=metadata.get("pool_name", ""),
                        model=metadata.get("model", ""),
                        current_agents=metadata.get("current_agents", 0),
                        min_agents=metadata.get("min_agents", 0),
                        max_agents=metadata.get("max_agents", 10),
                        tasks_pending=metadata.get("tasks_pending", 0),
                        tasks_completed=metadata.get("tasks_completed", 0),
                        avg_task_duration_seconds=metadata.get("avg_task_duration_seconds", 0.0),
                        workspace_id=self._workspace_id,
                    )
                    snapshots.append(snapshot)

                # Cache results
                if use_cache:
                    self._pool_performance_cache[pool_id] = snapshots
                    self._cache_times[pool_id] = time.time()

                return snapshots[:limit]

            except Exception as e:
                logger.error(f"Failed to get pool performance history: {e}")
                return []

    async def get_task_patterns(
        self,
        task_type: str,
        limit: int = 50,
        use_cache: bool = True,
    ) -> list[TaskSchedulingOutcome]:
        """
        Get historical task patterns for a task type from KM.

        Args:
            task_type: Task type to query
            limit: Maximum outcomes to return
            use_cache: Whether to use cached results

        Returns:
            List of TaskSchedulingOutcome sorted by recency
        """
        self._stats["task_pattern_queries"] += 1

        cache_key = task_type.lower()

        # Check cache
        if use_cache:
            if cache_key in self._task_patterns_cache:
                cache_time = self._cache_times.get(f"task_{cache_key}", 0)
                if time.time() - cache_time < self._cache_ttl:
                    return self._task_patterns_cache[cache_key][:limit]

        if not self._knowledge_mound:
            return []

        with self._timed_operation("get_task_patterns", task_type=task_type):
            try:
                results = await self._knowledge_mound.query(
                    query=f"fabric task {task_type}",
                    limit=limit * 2,
                    workspace_id=self._workspace_id,
                )

                outcomes = []
                for result in results:
                    metadata = result.get("metadata", {})
                    if metadata.get("type") != "fabric_task_outcome":
                        continue
                    if metadata.get("task_type") != task_type:
                        continue

                    outcome = TaskSchedulingOutcome(
                        task_id=metadata.get("task_id", ""),
                        task_type=metadata.get("task_type", ""),
                        agent_id=metadata.get("agent_id", ""),
                        pool_id=metadata.get("pool_id"),
                        priority=metadata.get("priority", 2),
                        scheduled_at=metadata.get("scheduled_at", 0),
                        completed_at=metadata.get("completed_at"),
                        success=metadata.get("success", False),
                        duration_seconds=metadata.get("duration_seconds", 0.0),
                        error_message=metadata.get("error_message"),
                        workspace_id=self._workspace_id,
                    )
                    if outcome.task_id:
                        outcomes.append(outcome)

                # Sort by scheduled_at descending (most recent first)
                outcomes.sort(key=lambda o: o.scheduled_at, reverse=True)

                # Cache results
                if use_cache:
                    self._task_patterns_cache[cache_key] = outcomes
                    self._cache_times[f"task_{cache_key}"] = time.time()

                return outcomes[:limit]

            except Exception as e:
                logger.error(f"Failed to get task patterns: {e}")
                return []

    async def get_pool_recommendations(
        self,
        task_type: str,
        available_pools: Optional[list[str]] = None,
        top_n: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Get pool recommendations for a task type based on historical performance.

        Args:
            task_type: Type of task to schedule
            available_pools: Optional list of pool IDs to consider
            top_n: Number of recommendations to return

        Returns:
            List of pool recommendations with scores
        """
        # Get historical task patterns
        patterns = await self.get_task_patterns(task_type, limit=100, use_cache=True)

        # Aggregate performance by pool
        pool_stats: dict[str, dict[str, Any]] = {}
        for outcome in patterns:
            if outcome.pool_id is None:
                continue
            if available_pools and outcome.pool_id not in available_pools:
                continue

            if outcome.pool_id not in pool_stats:
                pool_stats[outcome.pool_id] = {
                    "success_count": 0,
                    "failure_count": 0,
                    "total_duration": 0.0,
                }

            stats = pool_stats[outcome.pool_id]
            if outcome.success:
                stats["success_count"] += 1
            else:
                stats["failure_count"] += 1
            stats["total_duration"] += outcome.duration_seconds

        # Calculate scores
        recommendations = []
        for pool_id, stats in pool_stats.items():
            total = stats["success_count"] + stats["failure_count"]
            if total == 0:
                continue

            success_rate = stats["success_count"] / total
            avg_duration = stats["total_duration"] / total

            # Score: weighted by success rate (70%) and speed (30%)
            # Normalize duration: faster is better, cap at 300s
            duration_score = max(0, 1 - (avg_duration / 300))
            combined_score = (success_rate * 0.7) + (duration_score * 0.3)

            recommendations.append(
                {
                    "pool_id": pool_id,
                    "combined_score": combined_score,
                    "success_rate": success_rate,
                    "avg_duration_seconds": avg_duration,
                    "sample_size": total,
                    "confidence": min(1.0, total / 10),  # Higher samples = higher confidence
                }
            )

        # Sort by combined score
        recommendations.sort(key=lambda r: r["combined_score"], reverse=True)

        return recommendations[:top_n]

    async def get_agent_recommendations_for_pool(
        self,
        pool_id: str,
        task_type: str,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Get agent recommendations within a pool for a specific task type.

        Args:
            pool_id: Pool to analyze
            task_type: Task type to optimize for
            top_n: Number of recommendations

        Returns:
            List of agent recommendations with scores
        """
        patterns = await self.get_task_patterns(task_type, limit=200, use_cache=True)

        # Filter to pool and aggregate by agent
        agent_stats: dict[str, dict[str, Any]] = {}
        for outcome in patterns:
            if outcome.pool_id != pool_id:
                continue

            if outcome.agent_id not in agent_stats:
                agent_stats[outcome.agent_id] = {
                    "success_count": 0,
                    "failure_count": 0,
                    "total_duration": 0.0,
                }

            stats = agent_stats[outcome.agent_id]
            if outcome.success:
                stats["success_count"] += 1
            else:
                stats["failure_count"] += 1
            stats["total_duration"] += outcome.duration_seconds

        # Calculate recommendations
        recommendations = []
        for agent_id, stats in agent_stats.items():
            total = stats["success_count"] + stats["failure_count"]
            if total == 0:
                continue

            success_rate = stats["success_count"] / total
            avg_duration = stats["total_duration"] / total

            recommendations.append(
                {
                    "agent_id": agent_id,
                    "pool_id": pool_id,
                    "success_rate": success_rate,
                    "avg_duration_seconds": avg_duration,
                    "sample_size": total,
                }
            )

        recommendations.sort(key=lambda r: r["success_rate"], reverse=True)
        return recommendations[:top_n]

    async def get_budget_forecast(
        self,
        entity_id: str,
        forecast_days: int = 7,
    ) -> dict[str, Any]:
        """
        Forecast budget usage based on historical patterns.

        Args:
            entity_id: Entity to forecast for
            forecast_days: Days to forecast ahead

        Returns:
            Forecast with projected usage and alerts
        """
        if not self._knowledge_mound:
            return {"forecast_available": False}

        with self._timed_operation("get_budget_forecast", entity_id=entity_id):
            try:
                results = await self._knowledge_mound.query(
                    query=f"fabric budget {entity_id}",
                    limit=30,  # Last 30 snapshots
                    workspace_id=self._workspace_id,
                )

                snapshots = []
                for result in results:
                    metadata = result.get("metadata", {})
                    if metadata.get("type") != "fabric_budget_snapshot":
                        continue
                    if metadata.get("entity_id") != entity_id:
                        continue

                    snapshots.append(
                        {
                            "tokens_used": metadata.get("tokens_used", 0),
                            "tokens_limit": metadata.get("tokens_limit", 0),
                            "cost_used_usd": metadata.get("cost_used_usd", 0),
                            "period_end": metadata.get("period_end", 0),
                        }
                    )

                if len(snapshots) < 3:
                    return {
                        "forecast_available": False,
                        "reason": "Insufficient historical data",
                        "sample_size": len(snapshots),
                    }

                # Sort by period_end
                snapshots.sort(key=lambda s: s["period_end"])

                # Calculate daily averages from recent snapshots
                recent = snapshots[-7:]  # Last 7 data points
                daily_tokens = sum(s["tokens_used"] for s in recent) / len(recent)
                daily_cost = sum(s["cost_used_usd"] for s in recent) / len(recent)

                # Get current limits
                current_limits = snapshots[-1]
                tokens_limit = current_limits.get("tokens_limit", 0)

                # Forecast
                projected_tokens = daily_tokens * forecast_days
                projected_cost = daily_cost * forecast_days
                days_until_limit = (
                    (tokens_limit / daily_tokens) if daily_tokens > 0 else float("inf")
                )

                return {
                    "forecast_available": True,
                    "entity_id": entity_id,
                    "forecast_days": forecast_days,
                    "daily_avg_tokens": daily_tokens,
                    "daily_avg_cost_usd": daily_cost,
                    "projected_tokens": projected_tokens,
                    "projected_cost_usd": projected_cost,
                    "days_until_token_limit": days_until_limit,
                    "at_risk": days_until_limit < forecast_days,
                    "sample_size": len(snapshots),
                }

            except Exception as e:
                logger.error(f"Failed to get budget forecast: {e}")
                return {"forecast_available": False, "error": str(e)}

    # =========================================================================
    # Sync from Fabric
    # =========================================================================

    async def sync_from_fabric(self) -> dict[str, Any]:
        """
        Sync current fabric state to Knowledge Mound.

        Captures pool snapshots, recent task outcomes, and budget status.

        Returns:
            Dict with counts of synced items, or error dict on failure
        """
        if not self._fabric:
            logger.warning("No fabric configured for sync")
            return {"error": "No fabric configured"}

        synced = {
            "pools": 0,
            "tasks": 0,
            "budgets": 0,
        }

        with self._timed_operation("sync_from_fabric"):
            try:
                # Sync pools
                pools = await self._fabric.list_pools()
                for pool in pools:
                    scheduler_stats = await self._fabric.scheduler.get_stats()
                    snapshot = PoolSnapshot(
                        pool_id=pool.id,
                        name=pool.name,
                        model=pool.model,
                        current_agents=len(pool.current_agents),
                        min_agents=pool.min_agents,
                        max_agents=pool.max_agents,
                        tasks_pending=scheduler_stats.get("tasks_pending", 0),
                        tasks_completed=scheduler_stats.get("tasks_completed", 0),
                        workspace_id=self._workspace_id,
                    )
                    if await self.store_pool_snapshot(snapshot):
                        synced["pools"] += 1

                # Sync budget status for tracked entities
                budget_stats = await self._fabric.budget.get_stats()
                for entity_id in budget_stats.get("tracked_entities", []):
                    try:
                        report = await self._fabric.get_usage_report(entity_id, period_days=1)
                        # Map UsageReport attributes to BudgetUsageSnapshot
                        # UsageReport uses total_tokens/total_cost_usd; BudgetUsageSnapshot uses tokens_used/cost_used_usd
                        # Convert datetime to timestamp if needed
                        period_start_ts = (
                            report.period_start.timestamp()
                            if hasattr(report.period_start, "timestamp")
                            else float(report.period_start)  # type: ignore[arg-type]
                        )
                        period_end_ts = (
                            report.period_end.timestamp()
                            if hasattr(report.period_end, "timestamp")
                            else float(report.period_end)  # type: ignore[arg-type]
                        )

                        budget_snapshot = BudgetUsageSnapshot(
                            entity_id=entity_id,
                            entity_type="agent",  # Default, could be improved
                            tokens_used=getattr(report, "total_tokens", 0),
                            tokens_limit=getattr(report, "tokens_limit", 0),  # May not exist
                            cost_used_usd=getattr(report, "total_cost_usd", 0.0),
                            cost_limit_usd=getattr(report, "cost_limit_usd", 0.0),  # May not exist
                            period_start=period_start_ts,
                            period_end=period_end_ts,
                            alerts_triggered=getattr(report, "alerts_count", 0),
                            workspace_id=self._workspace_id,
                        )
                        if await self.store_budget_snapshot(budget_snapshot):
                            synced["budgets"] += 1
                    except Exception as e:
                        logger.debug(f"Failed to sync budget for {entity_id}: {e}")

                logger.info(f"Synced from fabric: {synced}")
                return synced

            except Exception as e:
                logger.error(f"Failed to sync from fabric: {e}")
                return {"error": str(e)}

    # =========================================================================
    # Stats and Health
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        return {
            **self._stats,
            "workspace_id": self._workspace_id,
            "pool_cache_size": len(self._pool_performance_cache),
            "task_cache_size": len(self._task_patterns_cache),
            "has_knowledge_mound": self._knowledge_mound is not None,
            "has_fabric": self._fabric is not None,
        }

    def clear_cache(self) -> int:
        """Clear all caches and return count of cleared items."""
        count = len(self._pool_performance_cache) + len(self._task_patterns_cache)
        self._pool_performance_cache.clear()
        self._task_patterns_cache.clear()
        self._cache_times.clear()
        return count


__all__ = [
    "FabricAdapter",
    "PoolSnapshot",
    "TaskSchedulingOutcome",
    "BudgetUsageSnapshot",
    "PolicyDecisionRecord",
]
