"""
Computer-Use Adapter for Knowledge Mound Integration.

Bridges the Computer-Use Orchestrator with Knowledge Mound to enable:
- Task completion history and patterns
- Action success/failure analysis
- Policy effectiveness monitoring
- Goal-based recommendations

ID Prefixes:
- cu_task_: Task execution records
- cu_action_: Action performance data
- cu_policy_: Policy block records
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, cast
from collections.abc import Callable

from ._base import KnowledgeMoundAdapter

if TYPE_CHECKING:
    from aragora.computer_use import ComputerUseOrchestrator, TaskResult

# KnowledgeMound is used dynamically, use Any for typing
KnowledgeMound = Any

logger = logging.getLogger(__name__)

# Type alias for event callbacks
EventCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class TaskExecutionRecord:
    """Record of a computer-use task execution for KM storage."""

    task_id: str
    goal: str
    status: str  # TaskStatus value
    total_steps: int
    successful_steps: int
    failed_steps: int
    blocked_steps: int
    duration_seconds: float
    agent_id: str | None = None
    error_message: str | None = None
    workspace_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionPerformanceRecord:
    """Aggregated action performance for KM storage."""

    action_type: str  # click, type, scroll, key, screenshot, wait
    total_executions: int
    successful_executions: int
    failed_executions: int
    avg_duration_ms: float
    policy_blocked_count: int = 0
    workspace_id: str = "default"


@dataclass
class PolicyBlockRecord:
    """Record of a policy-blocked action for analysis."""

    block_id: str
    task_id: str
    action_type: str
    element_selector: str | None
    domain: str | None
    policy_rule: str
    reason: str
    timestamp: float
    workspace_id: str = "default"


class ComputerUseAdapter(KnowledgeMoundAdapter):
    """
    Adapter that bridges Computer-Use Orchestrator to the Knowledge Mound.

    Provides bidirectional sync between computer-use operations and KM:
    - Forward: Task outcomes, action stats, policy blocks → KM
    - Reverse: Historical patterns → Task planning

    Usage:
        from aragora.computer_use import ComputerUseOrchestrator
        from aragora.knowledge.mound.adapters import ComputerUseAdapter

        orchestrator = ComputerUseOrchestrator()
        adapter = ComputerUseAdapter(orchestrator)

        # Store task result
        await adapter.store_task_result(task_result)

        # Get task recommendations
        recommendations = await adapter.get_similar_tasks("open settings")
    """

    adapter_name = "computer_use"

    def __init__(
        self,
        orchestrator: ComputerUseOrchestrator | None = None,
        knowledge_mound: KnowledgeMound | None = None,
        workspace_id: str = "default",
        event_callback: EventCallback | None = None,
        min_confidence_threshold: float = 0.6,
        enable_dual_write: bool = False,
    ):
        """
        Initialize the adapter.

        Args:
            orchestrator: The ComputerUseOrchestrator instance to wrap
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
        self._orchestrator = orchestrator
        self._knowledge_mound = knowledge_mound
        self._workspace_id = workspace_id
        self._min_confidence_threshold = min_confidence_threshold

        # Caches for reverse flow
        self._task_patterns_cache: dict[str, list[TaskExecutionRecord]] = {}
        self._action_stats_cache: dict[str, ActionPerformanceRecord] = {}
        self._cache_ttl: float = 300  # 5 minutes
        self._cache_times: dict[str, float] = {}

        # Statistics
        self._stats = {
            "task_records_stored": 0,
            "action_records_stored": 0,
            "policy_blocks_stored": 0,
            "task_queries": 0,
            "action_queries": 0,
        }

    # =========================================================================
    # Forward Sync: Computer-Use → KM
    # =========================================================================

    async def store_task_result(
        self,
        result: TaskResult,
        agent_id: str | None = None,
    ) -> str | None:
        """
        Store a task execution result in the Knowledge Mound.

        Args:
            result: TaskResult from the orchestrator
            agent_id: Optional agent that executed the task

        Returns:
            KM item ID if stored, None if below threshold
        """
        if not self._knowledge_mound:
            logger.debug("No KnowledgeMound configured, skipping task result storage")
            return None

        with self._timed_operation("store_task_result", task_id=result.task_id):
            try:
                from aragora.knowledge.unified.types import (
                    KnowledgeItem,
                    KnowledgeSource,
                    ConfidenceLevel,
                )

                # Calculate step statistics
                successful_steps = sum(1 for s in result.steps if s.status.value == "success")
                failed_steps = sum(1 for s in result.steps if s.status.value == "failed")
                blocked_steps = sum(1 for s in result.steps if s.status.value == "blocked")
                total_steps = len(result.steps)

                success = result.status.value in ("completed", "success")
                confidence_val = 0.8 if success else 0.5
                if confidence_val < self._min_confidence_threshold:
                    return None

                confidence = ConfidenceLevel.HIGH if success else ConfidenceLevel.MEDIUM
                now = datetime.now()

                duration = (
                    result.end_time - result.start_time
                    if result.end_time
                    else time.time() - result.start_time
                )

                item = KnowledgeItem(
                    id=f"cu_task_{result.task_id}",
                    content=(
                        f"Computer-use task ({result.status.value}): {result.goal[:100]}... "
                        f"Steps: {successful_steps}/{total_steps} succeeded, "
                        f"{blocked_steps} blocked, {duration:.1f}s"
                    ),
                    source=KnowledgeSource.CONTINUUM,
                    source_id=result.task_id,
                    confidence=confidence,
                    created_at=now,
                    updated_at=now,
                    metadata={
                        "type": "computer_use_task",
                        "task_id": result.task_id,
                        "goal": result.goal,
                        "status": result.status.value,
                        "total_steps": total_steps,
                        "successful_steps": successful_steps,
                        "failed_steps": failed_steps,
                        "blocked_steps": blocked_steps,
                        "duration_seconds": duration,
                        "agent_id": agent_id,
                        "error_message": result.error,
                        "workspace_id": self._workspace_id,
                        "action_types": list(set(s.action.action_type.value for s in result.steps)),
                    },
                )

                item_id = await self._knowledge_mound.ingest(item)

                self._stats["task_records_stored"] += 1
                self._emit_event(
                    "cu_task_stored",
                    {
                        "task_id": result.task_id,
                        "status": result.status.value,
                        "steps": total_steps,
                    },
                )

                # Invalidate cache
                self._task_patterns_cache.clear()

                return item_id

            except Exception as e:
                logger.error(f"Failed to store task result: {e}")
                return None

    async def store_task_execution_record(
        self,
        record: TaskExecutionRecord,
    ) -> str | None:
        """
        Store a task execution record in the Knowledge Mound.

        Args:
            record: Task execution record to store

        Returns:
            KM item ID if stored
        """
        if not self._knowledge_mound:
            return None

        with self._timed_operation("store_task_execution_record", task_id=record.task_id):
            try:
                from aragora.knowledge.unified.types import (
                    KnowledgeItem,
                    KnowledgeSource,
                    ConfidenceLevel,
                )

                success = record.status in ("completed", "success")
                confidence = ConfidenceLevel.HIGH if success else ConfidenceLevel.MEDIUM
                now = datetime.now()

                success_rate = 0.0
                if record.total_steps > 0:
                    success_rate = record.successful_steps / record.total_steps

                item = KnowledgeItem(
                    id=f"cu_task_{record.task_id}",
                    content=(
                        f"Computer-use task ({record.status}): {record.goal[:100]}... "
                        f"Steps: {record.successful_steps}/{record.total_steps} ({success_rate:.0%}), "
                        f"{record.duration_seconds:.1f}s"
                    ),
                    source=KnowledgeSource.CONTINUUM,
                    source_id=record.task_id,
                    confidence=confidence,
                    created_at=now,
                    updated_at=now,
                    metadata={
                        "type": "computer_use_task",
                        "task_id": record.task_id,
                        "goal": record.goal,
                        "status": record.status,
                        "total_steps": record.total_steps,
                        "successful_steps": record.successful_steps,
                        "failed_steps": record.failed_steps,
                        "blocked_steps": record.blocked_steps,
                        "success_rate": success_rate,
                        "duration_seconds": record.duration_seconds,
                        "agent_id": record.agent_id,
                        "error_message": record.error_message,
                        "workspace_id": record.workspace_id,
                        **record.metadata,
                    },
                )

                item_id = await self._knowledge_mound.ingest(item)

                self._stats["task_records_stored"] += 1
                self._emit_event(
                    "cu_task_record_stored",
                    {
                        "task_id": record.task_id,
                        "status": record.status,
                        "success_rate": success_rate,
                    },
                )

                return item_id

            except Exception as e:
                logger.error(f"Failed to store task execution record: {e}")
                return None

    async def store_action_performance(
        self,
        record: ActionPerformanceRecord,
    ) -> str | None:
        """
        Store aggregated action performance in the Knowledge Mound.

        Args:
            record: Action performance record to store

        Returns:
            KM item ID if stored
        """
        if not self._knowledge_mound:
            return None

        with self._timed_operation("store_action_performance", action_type=record.action_type):
            try:
                from aragora.knowledge.unified.types import (
                    KnowledgeItem,
                    KnowledgeSource,
                    ConfidenceLevel,
                )

                confidence = (
                    ConfidenceLevel.HIGH
                    if record.total_executions >= 10
                    else ConfidenceLevel.MEDIUM
                )
                now = datetime.now()

                success_rate = 0.0
                if record.total_executions > 0:
                    success_rate = record.successful_executions / record.total_executions

                item = KnowledgeItem(
                    id=f"cu_action_{record.action_type}_{int(time.time())}",
                    content=(
                        f"Action '{record.action_type}' performance: "
                        f"{record.successful_executions}/{record.total_executions} ({success_rate:.0%} success), "
                        f"avg {record.avg_duration_ms:.0f}ms, {record.policy_blocked_count} blocked"
                    ),
                    source=KnowledgeSource.ELO,  # Performance data
                    source_id=f"action_{record.action_type}",
                    confidence=confidence,
                    created_at=now,
                    updated_at=now,
                    metadata={
                        "type": "computer_use_action_performance",
                        "action_type": record.action_type,
                        "total_executions": record.total_executions,
                        "successful_executions": record.successful_executions,
                        "failed_executions": record.failed_executions,
                        "success_rate": success_rate,
                        "avg_duration_ms": record.avg_duration_ms,
                        "policy_blocked_count": record.policy_blocked_count,
                        "workspace_id": record.workspace_id,
                    },
                )

                item_id = await self._knowledge_mound.ingest(item)

                self._stats["action_records_stored"] += 1

                # Invalidate action stats cache
                if record.action_type in self._action_stats_cache:
                    del self._action_stats_cache[record.action_type]

                return item_id

            except Exception as e:
                logger.error(f"Failed to store action performance: {e}")
                return None

    async def store_policy_block(
        self,
        record: PolicyBlockRecord,
    ) -> str | None:
        """
        Store a policy block record for audit and analysis.

        Args:
            record: Policy block record to store

        Returns:
            KM item ID if stored
        """
        if not self._knowledge_mound:
            return None

        with self._timed_operation("store_policy_block", block_id=record.block_id):
            try:
                from aragora.knowledge.unified.types import (
                    KnowledgeItem,
                    KnowledgeSource,
                    ConfidenceLevel,
                )

                confidence = ConfidenceLevel.HIGH  # Policy blocks are authoritative
                now = datetime.now()

                location = record.domain or record.element_selector or "unknown"

                item = KnowledgeItem(
                    id=f"cu_policy_{record.block_id}",
                    content=(
                        f"Policy blocked {record.action_type} at {location}: "
                        f"{record.reason} (rule: {record.policy_rule})"
                    ),
                    source=KnowledgeSource.CONTINUUM,
                    source_id=record.block_id,
                    confidence=confidence,
                    created_at=now,
                    updated_at=now,
                    metadata={
                        "type": "computer_use_policy_block",
                        "block_id": record.block_id,
                        "task_id": record.task_id,
                        "action_type": record.action_type,
                        "element_selector": record.element_selector,
                        "domain": record.domain,
                        "policy_rule": record.policy_rule,
                        "reason": record.reason,
                        "timestamp": record.timestamp,
                        "workspace_id": record.workspace_id,
                    },
                )

                item_id = await self._knowledge_mound.ingest(item)

                self._stats["policy_blocks_stored"] += 1
                self._emit_event(
                    "cu_policy_block_stored",
                    {
                        "block_id": record.block_id,
                        "action_type": record.action_type,
                        "policy_rule": record.policy_rule,
                    },
                )

                return item_id

            except Exception as e:
                logger.error(f"Failed to store policy block: {e}")
                return None

    # =========================================================================
    # Reverse Flow: KM → Computer-Use
    # =========================================================================

    async def get_similar_tasks(
        self,
        goal: str,
        limit: int = 10,
        success_only: bool = False,
    ) -> list[TaskExecutionRecord]:
        """
        Find similar task executions from KM based on goal.

        Args:
            goal: Goal description to search for
            limit: Maximum tasks to return
            success_only: Only return successful tasks

        Returns:
            List of TaskExecutionRecord sorted by similarity
        """
        self._stats["task_queries"] += 1

        if not self._knowledge_mound:
            return []

        with self._timed_operation("get_similar_tasks", goal=goal[:50]):
            try:
                results = await self._knowledge_mound.query(
                    query=f"computer-use task {goal[:100]}",
                    limit=limit * 2,
                    workspace_id=self._workspace_id,
                )

                records = []
                for result in results:
                    metadata = result.get("metadata", {})
                    if metadata.get("type") != "computer_use_task":
                        continue

                    status = metadata.get("status", "")
                    if success_only and status not in ("completed", "success"):
                        continue

                    record = TaskExecutionRecord(
                        task_id=metadata.get("task_id", ""),
                        goal=metadata.get("goal", ""),
                        status=status,
                        total_steps=metadata.get("total_steps", 0),
                        successful_steps=metadata.get("successful_steps", 0),
                        failed_steps=metadata.get("failed_steps", 0),
                        blocked_steps=metadata.get("blocked_steps", 0),
                        duration_seconds=metadata.get("duration_seconds", 0.0),
                        agent_id=metadata.get("agent_id"),
                        error_message=metadata.get("error_message"),
                        workspace_id=self._workspace_id,
                        metadata={"similarity_score": result.get("score", 0.0)},
                    )
                    if record.task_id:
                        records.append(record)

                # Sort by similarity score
                records.sort(
                    key=lambda r: r.metadata.get("similarity_score", 0),
                    reverse=True,
                )

                return records[:limit]

            except Exception as e:
                logger.error(f"Failed to get similar tasks: {e}")
                return []

    async def get_action_statistics(
        self,
        action_type: str | None = None,
    ) -> dict[str, ActionPerformanceRecord]:
        """
        Get action performance statistics from KM.

        Args:
            action_type: Optional specific action type to query

        Returns:
            Dict mapping action_type to performance record
        """
        self._stats["action_queries"] += 1

        if not self._knowledge_mound:
            return {}

        with self._timed_operation("get_action_statistics", action_type=action_type):
            try:
                query = (
                    f"computer-use action {action_type}"
                    if action_type
                    else "computer-use action performance"
                )
                results = await self._knowledge_mound.query(
                    query=query,
                    limit=50,
                    workspace_id=self._workspace_id,
                )

                stats: dict[str, ActionPerformanceRecord] = {}
                for result in results:
                    metadata = result.get("metadata", {})
                    if metadata.get("type") != "computer_use_action_performance":
                        continue

                    at = metadata.get("action_type", "")
                    if not at:
                        continue
                    if action_type and at != action_type:
                        continue

                    # Aggregate if multiple records for same action type
                    if at in stats:
                        existing = stats[at]
                        existing.total_executions += metadata.get("total_executions", 0)
                        existing.successful_executions += metadata.get("successful_executions", 0)
                        existing.failed_executions += metadata.get("failed_executions", 0)
                        existing.policy_blocked_count += metadata.get("policy_blocked_count", 0)
                    else:
                        stats[at] = ActionPerformanceRecord(
                            action_type=at,
                            total_executions=metadata.get("total_executions", 0),
                            successful_executions=metadata.get("successful_executions", 0),
                            failed_executions=metadata.get("failed_executions", 0),
                            avg_duration_ms=metadata.get("avg_duration_ms", 0.0),
                            policy_blocked_count=metadata.get("policy_blocked_count", 0),
                            workspace_id=self._workspace_id,
                        )

                return stats

            except Exception as e:
                logger.error(f"Failed to get action statistics: {e}")
                return {}

    async def get_task_recommendations(
        self,
        goal: str,
        top_n: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Get recommendations for executing a task based on historical patterns.

        Args:
            goal: Task goal to get recommendations for
            top_n: Number of recommendations to return

        Returns:
            List of recommendations with expected steps and success factors
        """
        similar = await self.get_similar_tasks(goal, limit=20, success_only=True)

        if not similar:
            return [
                {
                    "recommendation": "No similar tasks found",
                    "confidence": 0.0,
                }
            ]

        # Analyze successful patterns
        avg_steps = sum(t.total_steps for t in similar) / len(similar)
        avg_duration = sum(t.duration_seconds for t in similar) / len(similar)
        common_agents: dict[str, int] = {}
        for t in similar:
            if t.agent_id:
                common_agents[t.agent_id] = common_agents.get(t.agent_id, 0) + 1

        # Build recommendations
        recommendations = []

        # Step count recommendation
        recommendations.append(
            {
                "type": "steps",
                "recommendation": f"Expect approximately {avg_steps:.0f} steps",
                "based_on": len(similar),
                "confidence": min(1.0, len(similar) / 10),
            }
        )

        # Duration recommendation
        recommendations.append(
            {
                "type": "duration",
                "recommendation": f"Expect approximately {avg_duration:.0f}s execution time",
                "based_on": len(similar),
                "confidence": min(1.0, len(similar) / 10),
            }
        )

        # Agent recommendation (if patterns exist)
        if common_agents:
            best_agent = max(common_agents, key=lambda k: common_agents[k])
            agent_count = common_agents[best_agent]
            recommendations.append(
                {
                    "type": "agent",
                    "recommendation": f"Agent '{best_agent}' has completed {agent_count} similar tasks",
                    "agent_id": best_agent,
                    "confidence": agent_count / len(similar),
                }
            )

        return recommendations[:top_n]

    async def get_policy_block_analysis(
        self,
        days_back: int = 7,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Analyze policy blocks to identify patterns and potential policy adjustments.

        Args:
            days_back: How many days to analyze
            limit: Maximum blocks to analyze

        Returns:
            Analysis with block patterns and recommendations
        """
        if not self._knowledge_mound:
            return {"analysis_available": False}

        with self._timed_operation("get_policy_block_analysis"):
            try:
                results = await self._knowledge_mound.query(
                    query="computer-use policy blocked",
                    limit=limit,
                    workspace_id=self._workspace_id,
                )

                blocks_by_rule: dict[str, int] = {}
                blocks_by_action: dict[str, int] = {}
                blocks_by_domain: dict[str, int] = {}

                cutoff = time.time() - (days_back * 86400)

                for result in results:
                    metadata = result.get("metadata", {})
                    if metadata.get("type") != "computer_use_policy_block":
                        continue

                    timestamp = metadata.get("timestamp", 0)
                    if timestamp < cutoff:
                        continue

                    rule = metadata.get("policy_rule", "unknown")
                    blocks_by_rule[rule] = blocks_by_rule.get(rule, 0) + 1

                    action = metadata.get("action_type", "unknown")
                    blocks_by_action[action] = blocks_by_action.get(action, 0) + 1

                    domain = metadata.get("domain")
                    if domain:
                        blocks_by_domain[domain] = blocks_by_domain.get(domain, 0) + 1

                total_blocks = sum(blocks_by_rule.values())

                return {
                    "analysis_available": True,
                    "days_analyzed": days_back,
                    "total_blocks": total_blocks,
                    "blocks_by_rule": dict(
                        sorted(blocks_by_rule.items(), key=lambda x: -x[1])[:10]
                    ),
                    "blocks_by_action": dict(
                        sorted(blocks_by_action.items(), key=lambda x: -x[1])[:10]
                    ),
                    "blocks_by_domain": dict(
                        sorted(blocks_by_domain.items(), key=lambda x: -x[1])[:10]
                    ),
                    "recommendations": self._generate_policy_recommendations(
                        blocks_by_rule, blocks_by_action, total_blocks
                    ),
                }

            except Exception as e:
                logger.error(f"Failed to get policy block analysis: {e}")
                return {"analysis_available": False, "error": str(e)}

    def _generate_policy_recommendations(
        self,
        blocks_by_rule: dict[str, int],
        blocks_by_action: dict[str, int],
        total_blocks: int,
    ) -> list[str]:
        """Generate policy adjustment recommendations based on block patterns."""
        recommendations = []

        if total_blocks == 0:
            return ["No policy blocks to analyze"]

        # High-frequency rule blocks might indicate overly restrictive policies
        for rule, count in blocks_by_rule.items():
            if count / total_blocks > 0.3:  # More than 30% of blocks
                recommendations.append(
                    f"Rule '{rule}' accounts for {count}/{total_blocks} blocks - consider reviewing"
                )

        # Check if certain actions are frequently blocked
        for action, count in blocks_by_action.items():
            if count / total_blocks > 0.25:  # More than 25% of blocks
                recommendations.append(
                    f"Action '{action}' is frequently blocked ({count} times) - may need policy adjustment"
                )

        if not recommendations:
            recommendations.append(
                "Policy blocks are distributed evenly - no adjustments recommended"
            )

        return recommendations

    # =========================================================================
    # Sync from Orchestrator
    # =========================================================================

    async def sync_from_orchestrator(self) -> dict[str, Any]:
        """
        Sync current orchestrator metrics to Knowledge Mound.

        Returns:
            Dict with counts of synced items
        """
        if not self._orchestrator:
            logger.warning("No orchestrator configured for sync")
            return {"error": "No orchestrator configured"}

        synced = {
            "action_records": 0,
        }

        with self._timed_operation("sync_from_orchestrator"):
            try:
                metrics = cast(Any, self._orchestrator).get_metrics()

                # Store action performance aggregates
                action_types = ["click", "type", "scroll", "key", "screenshot", "wait"]
                for action_type in action_types:
                    # Get action-specific metrics from orchestrator
                    record = ActionPerformanceRecord(
                        action_type=action_type,
                        total_executions=metrics.total_actions,  # Would need per-action breakdown
                        successful_executions=metrics.successful_actions,
                        failed_executions=metrics.failed_actions,
                        avg_duration_ms=metrics.total_latency_ms / max(1, metrics.total_actions),
                        policy_blocked_count=metrics.policy_blocked_actions,
                        workspace_id=self._workspace_id,
                    )
                    if await self.store_action_performance(record):
                        synced["action_records"] += 1

                logger.info(f"Synced from orchestrator: {synced}")
                return synced

            except Exception as e:
                logger.error(f"Failed to sync from orchestrator: {e}")
                return {"error": str(e)}

    # =========================================================================
    # Stats and Health
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        return {
            **self._stats,
            "workspace_id": self._workspace_id,
            "task_cache_size": len(self._task_patterns_cache),
            "action_cache_size": len(self._action_stats_cache),
            "has_knowledge_mound": self._knowledge_mound is not None,
            "has_orchestrator": self._orchestrator is not None,
        }

    def clear_cache(self) -> int:
        """Clear all caches and return count of cleared items."""
        count = len(self._task_patterns_cache) + len(self._action_stats_cache)
        self._task_patterns_cache.clear()
        self._action_stats_cache.clear()
        self._cache_times.clear()
        return count


__all__ = [
    "ComputerUseAdapter",
    "TaskExecutionRecord",
    "ActionPerformanceRecord",
    "PolicyBlockRecord",
]
