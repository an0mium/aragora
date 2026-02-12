"""
Workspace Adapter for Knowledge Mound Integration.

Bridges the Workspace Manager with Knowledge Mound to enable:
- Rig performance history and optimization
- Convoy completion patterns
- Merge success/failure analysis
- Cross-workspace knowledge sharing

ID Prefixes:
- workspace_rig_: Rig configuration and statistics
- workspace_convoy_: Convoy completion records
- workspace_merge_: Merge queue outcomes
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional
from collections.abc import Callable

from ._base import KnowledgeMoundAdapter

if TYPE_CHECKING:
    from aragora.workspace import WorkspaceManager

# KnowledgeMound is used dynamically, use Any for typing
KnowledgeMound = Any

logger = logging.getLogger(__name__)

# Type alias for event callbacks
EventCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class RigSnapshot:
    """Snapshot of rig state for KM storage."""

    rig_id: str
    name: str
    workspace_id: str
    status: str  # RigStatus value
    repo_url: str = ""
    branch: str = "main"
    assigned_agents: int = 0
    max_agents: int = 10
    active_convoys: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConvoyOutcome:
    """Record of a convoy completion for KM storage."""

    convoy_id: str
    workspace_id: str
    rig_id: str
    name: str
    status: str  # ConvoyStatus value
    total_beads: int
    completed_beads: int = 0
    assigned_agents: int = 0
    duration_seconds: float = 0.0
    merge_success: bool | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MergeOutcome:
    """Record of a merge queue outcome for KM storage."""

    merge_id: str
    convoy_id: str
    rig_id: str
    workspace_id: str
    success: bool
    conflicts_resolved: int = 0
    files_changed: int = 0
    tests_passed: bool = False
    review_approved: bool = False
    duration_seconds: float = 0.0
    error_message: str | None = None


class WorkspaceAdapter(KnowledgeMoundAdapter):
    """
    Adapter that bridges Workspace Manager to the Knowledge Mound.

    Provides bidirectional sync between Workspace operations and KM:
    - Forward: Rig stats, convoy outcomes, merge results → KM
    - Reverse: Historical patterns → Workspace decisions

    Usage:
        from aragora.workspace import WorkspaceManager
        from aragora.knowledge.mound.adapters import WorkspaceAdapter

        ws = WorkspaceManager()
        adapter = WorkspaceAdapter(ws)

        # Store rig snapshot
        await adapter.store_rig_snapshot(rig_snapshot)

        # Get rig recommendations
        recommendations = await adapter.get_rig_recommendations("backend")
    """

    adapter_name = "workspace"

    def __init__(
        self,
        workspace_manager: WorkspaceManager | None = None,
        knowledge_mound: KnowledgeMound | None = None,
        workspace_id: str = "default",
        event_callback: EventCallback | None = None,
        min_confidence_threshold: float = 0.6,
        enable_dual_write: bool = False,
    ):
        """
        Initialize the adapter.

        Args:
            workspace_manager: The WorkspaceManager instance to wrap
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
        self._workspace_manager = workspace_manager
        self._knowledge_mound = knowledge_mound
        self._workspace_id = workspace_id
        self._min_confidence_threshold = min_confidence_threshold

        # Caches for reverse flow
        self._rig_performance_cache: dict[str, list[RigSnapshot]] = {}
        self._convoy_patterns_cache: dict[str, list[ConvoyOutcome]] = {}
        self._cache_ttl: float = 300  # 5 minutes
        self._cache_times: dict[str, float] = {}

        # Statistics
        self._stats = {
            "rig_snapshots_stored": 0,
            "convoy_outcomes_stored": 0,
            "merge_outcomes_stored": 0,
            "rig_queries": 0,
            "convoy_pattern_queries": 0,
        }

    # =========================================================================
    # Forward Sync: Workspace → KM
    # =========================================================================

    async def store_rig_snapshot(
        self,
        snapshot: RigSnapshot,
    ) -> str | None:
        """
        Store a rig snapshot in the Knowledge Mound.

        Args:
            snapshot: Rig snapshot to store

        Returns:
            KM item ID if stored, None if below threshold
        """
        if not self._knowledge_mound:
            logger.debug("No KnowledgeMound configured, skipping rig snapshot storage")
            return None

        with self._timed_operation("store_rig_snapshot", rig_id=snapshot.rig_id):
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

                success_rate = 0.0
                total_tasks = snapshot.tasks_completed + snapshot.tasks_failed
                if total_tasks > 0:
                    success_rate = snapshot.tasks_completed / total_tasks

                item = KnowledgeItem(
                    id=f"workspace_rig_{snapshot.rig_id}_{int(time.time())}",
                    content=(
                        f"Rig '{snapshot.name}' ({snapshot.status}): "
                        f"{snapshot.assigned_agents}/{snapshot.max_agents} agents, "
                        f"{snapshot.active_convoys} active convoys, "
                        f"{snapshot.tasks_completed}/{total_tasks} tasks ({success_rate:.1%} success)"
                    ),
                    source=KnowledgeSource.CONTINUUM,
                    source_id=snapshot.rig_id,
                    confidence=confidence,
                    created_at=now,
                    updated_at=now,
                    metadata={
                        "type": "workspace_rig_snapshot",
                        "rig_id": snapshot.rig_id,
                        "rig_name": snapshot.name,
                        "workspace_id": snapshot.workspace_id,
                        "status": snapshot.status,
                        "repo_url": snapshot.repo_url,
                        "branch": snapshot.branch,
                        "assigned_agents": snapshot.assigned_agents,
                        "max_agents": snapshot.max_agents,
                        "active_convoys": snapshot.active_convoys,
                        "tasks_completed": snapshot.tasks_completed,
                        "tasks_failed": snapshot.tasks_failed,
                        "success_rate": success_rate,
                        **snapshot.metadata,
                    },
                )

                item_id = await self._knowledge_mound.ingest(item)

                self._stats["rig_snapshots_stored"] += 1
                self._emit_event(
                    "workspace_rig_snapshot_stored",
                    {
                        "rig_id": snapshot.rig_id,
                        "status": snapshot.status,
                        "success_rate": success_rate,
                    },
                )

                # Invalidate cache
                if snapshot.rig_id in self._rig_performance_cache:
                    del self._rig_performance_cache[snapshot.rig_id]

                return item_id

            except Exception as e:
                logger.error(f"Failed to store rig snapshot: {e}")
                return None

    async def store_convoy_outcome(
        self,
        outcome: ConvoyOutcome,
    ) -> str | None:
        """
        Store a convoy outcome in the Knowledge Mound.

        Args:
            outcome: Convoy outcome to store

        Returns:
            KM item ID if stored
        """
        if not self._knowledge_mound:
            return None

        with self._timed_operation("store_convoy_outcome", convoy_id=outcome.convoy_id):
            try:
                from aragora.knowledge.unified.types import (
                    KnowledgeItem,
                    KnowledgeSource,
                    ConfidenceLevel,
                )

                success = outcome.status in ("done", "completed")
                confidence = ConfidenceLevel.HIGH if success else ConfidenceLevel.MEDIUM
                now = datetime.now()

                completion_pct = 0.0
                if outcome.total_beads > 0:
                    completion_pct = outcome.completed_beads / outcome.total_beads * 100

                item = KnowledgeItem(
                    id=f"workspace_convoy_{outcome.convoy_id}",
                    content=(
                        f"Convoy '{outcome.name}' ({outcome.status}) in rig {outcome.rig_id}: "
                        f"{outcome.completed_beads}/{outcome.total_beads} beads ({completion_pct:.0f}%), "
                        f"{outcome.duration_seconds:.1f}s"
                    ),
                    source=KnowledgeSource.CONTINUUM,
                    source_id=outcome.convoy_id,
                    confidence=confidence,
                    created_at=now,
                    updated_at=now,
                    metadata={
                        "type": "workspace_convoy_outcome",
                        "convoy_id": outcome.convoy_id,
                        "convoy_name": outcome.name,
                        "workspace_id": outcome.workspace_id,
                        "rig_id": outcome.rig_id,
                        "status": outcome.status,
                        "total_beads": outcome.total_beads,
                        "completed_beads": outcome.completed_beads,
                        "completion_pct": completion_pct,
                        "assigned_agents": outcome.assigned_agents,
                        "duration_seconds": outcome.duration_seconds,
                        "merge_success": outcome.merge_success,
                        "error_message": outcome.error_message,
                        **outcome.metadata,
                    },
                )

                item_id = await self._knowledge_mound.ingest(item)

                self._stats["convoy_outcomes_stored"] += 1
                self._emit_event(
                    "workspace_convoy_outcome_stored",
                    {
                        "convoy_id": outcome.convoy_id,
                        "status": outcome.status,
                        "completion_pct": completion_pct,
                    },
                )

                # Invalidate convoy patterns cache for this rig
                cache_key = outcome.rig_id
                if cache_key in self._convoy_patterns_cache:
                    del self._convoy_patterns_cache[cache_key]

                return item_id

            except Exception as e:
                logger.error(f"Failed to store convoy outcome: {e}")
                return None

    async def store_merge_outcome(
        self,
        outcome: MergeOutcome,
    ) -> str | None:
        """
        Store a merge outcome in the Knowledge Mound.

        Args:
            outcome: Merge outcome to store

        Returns:
            KM item ID if stored
        """
        if not self._knowledge_mound:
            return None

        with self._timed_operation("store_merge_outcome", merge_id=outcome.merge_id):
            try:
                from aragora.knowledge.unified.types import (
                    KnowledgeItem,
                    KnowledgeSource,
                    ConfidenceLevel,
                )

                confidence = ConfidenceLevel.HIGH  # Merge outcomes are authoritative
                now = datetime.now()

                status_str = "success" if outcome.success else "failed"
                checks = []
                if outcome.tests_passed:
                    checks.append("tests passed")
                if outcome.review_approved:
                    checks.append("review approved")
                checks_str = ", ".join(checks) if checks else "no checks"

                item = KnowledgeItem(
                    id=f"workspace_merge_{outcome.merge_id}",
                    content=(
                        f"Merge {status_str} for convoy {outcome.convoy_id}: "
                        f"{outcome.files_changed} files, {outcome.conflicts_resolved} conflicts resolved, "
                        f"{checks_str}, {outcome.duration_seconds:.1f}s"
                    ),
                    source=KnowledgeSource.CONTINUUM,
                    source_id=outcome.merge_id,
                    confidence=confidence,
                    created_at=now,
                    updated_at=now,
                    metadata={
                        "type": "workspace_merge_outcome",
                        "merge_id": outcome.merge_id,
                        "convoy_id": outcome.convoy_id,
                        "rig_id": outcome.rig_id,
                        "workspace_id": outcome.workspace_id,
                        "success": outcome.success,
                        "conflicts_resolved": outcome.conflicts_resolved,
                        "files_changed": outcome.files_changed,
                        "tests_passed": outcome.tests_passed,
                        "review_approved": outcome.review_approved,
                        "duration_seconds": outcome.duration_seconds,
                        "error_message": outcome.error_message,
                    },
                )

                item_id = await self._knowledge_mound.ingest(item)

                self._stats["merge_outcomes_stored"] += 1
                self._emit_event(
                    "workspace_merge_outcome_stored",
                    {
                        "merge_id": outcome.merge_id,
                        "success": outcome.success,
                        "files_changed": outcome.files_changed,
                    },
                )

                return item_id

            except Exception as e:
                logger.error(f"Failed to store merge outcome: {e}")
                return None

    # =========================================================================
    # Reverse Flow: KM → Workspace
    # =========================================================================

    async def get_rig_performance_history(
        self,
        rig_id: str,
        limit: int = 20,
        use_cache: bool = True,
    ) -> list[RigSnapshot]:
        """
        Get historical rig performance from KM.

        Args:
            rig_id: Rig ID to query
            limit: Maximum snapshots to return
            use_cache: Whether to use cached results

        Returns:
            List of RigSnapshot sorted by recency
        """
        self._stats["rig_queries"] += 1

        # Check cache
        if use_cache:
            if rig_id in self._rig_performance_cache:
                cache_time = self._cache_times.get(f"rig_{rig_id}", 0)
                if time.time() - cache_time < self._cache_ttl:
                    return self._rig_performance_cache[rig_id][:limit]

        if not self._knowledge_mound:
            return []

        with self._timed_operation("get_rig_performance", rig_id=rig_id):
            try:
                results = await self._knowledge_mound.query(
                    query=f"workspace rig {rig_id}",
                    limit=limit * 2,
                    workspace_id=self._workspace_id,
                )

                snapshots = []
                for result in results:
                    metadata = result.get("metadata", {})
                    if metadata.get("type") != "workspace_rig_snapshot":
                        continue
                    if metadata.get("rig_id") != rig_id:
                        continue

                    snapshot = RigSnapshot(
                        rig_id=metadata.get("rig_id", ""),
                        name=metadata.get("rig_name", ""),
                        workspace_id=metadata.get("workspace_id", ""),
                        status=metadata.get("status", ""),
                        repo_url=metadata.get("repo_url", ""),
                        branch=metadata.get("branch", "main"),
                        assigned_agents=metadata.get("assigned_agents", 0),
                        max_agents=metadata.get("max_agents", 10),
                        active_convoys=metadata.get("active_convoys", 0),
                        tasks_completed=metadata.get("tasks_completed", 0),
                        tasks_failed=metadata.get("tasks_failed", 0),
                    )
                    snapshots.append(snapshot)

                # Cache results
                if use_cache:
                    self._rig_performance_cache[rig_id] = snapshots
                    self._cache_times[f"rig_{rig_id}"] = time.time()

                return snapshots[:limit]

            except Exception as e:
                logger.error(f"Failed to get rig performance history: {e}")
                return []

    async def get_convoy_patterns(
        self,
        rig_id: str,
        limit: int = 50,
        use_cache: bool = True,
    ) -> list[ConvoyOutcome]:
        """
        Get historical convoy patterns for a rig from KM.

        Args:
            rig_id: Rig ID to query
            limit: Maximum outcomes to return
            use_cache: Whether to use cached results

        Returns:
            List of ConvoyOutcome sorted by recency
        """
        self._stats["convoy_pattern_queries"] += 1

        # Check cache
        if use_cache:
            if rig_id in self._convoy_patterns_cache:
                cache_time = self._cache_times.get(f"convoy_{rig_id}", 0)
                if time.time() - cache_time < self._cache_ttl:
                    return self._convoy_patterns_cache[rig_id][:limit]

        if not self._knowledge_mound:
            return []

        with self._timed_operation("get_convoy_patterns", rig_id=rig_id):
            try:
                results = await self._knowledge_mound.query(
                    query=f"workspace convoy rig {rig_id}",
                    limit=limit * 2,
                    workspace_id=self._workspace_id,
                )

                outcomes = []
                for result in results:
                    metadata = result.get("metadata", {})
                    if metadata.get("type") != "workspace_convoy_outcome":
                        continue
                    if metadata.get("rig_id") != rig_id:
                        continue

                    outcome = ConvoyOutcome(
                        convoy_id=metadata.get("convoy_id", ""),
                        workspace_id=metadata.get("workspace_id", ""),
                        rig_id=metadata.get("rig_id", ""),
                        name=metadata.get("convoy_name", ""),
                        status=metadata.get("status", ""),
                        total_beads=metadata.get("total_beads", 0),
                        completed_beads=metadata.get("completed_beads", 0),
                        assigned_agents=metadata.get("assigned_agents", 0),
                        duration_seconds=metadata.get("duration_seconds", 0.0),
                        merge_success=metadata.get("merge_success"),
                        error_message=metadata.get("error_message"),
                    )
                    if outcome.convoy_id:
                        outcomes.append(outcome)

                # Cache results
                if use_cache:
                    self._convoy_patterns_cache[rig_id] = outcomes
                    self._cache_times[f"convoy_{rig_id}"] = time.time()

                return outcomes[:limit]

            except Exception as e:
                logger.error(f"Failed to get convoy patterns: {e}")
                return []

    async def get_rig_recommendations(
        self,
        project_type: str,
        available_rigs: list[str] | None = None,
        top_n: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Get rig recommendations for a project type based on historical performance.

        Args:
            project_type: Type of project (backend, frontend, data, etc.)
            available_rigs: Optional list of rig IDs to consider
            top_n: Number of recommendations to return

        Returns:
            List of rig recommendations with scores
        """
        if not self._knowledge_mound:
            return []

        with self._timed_operation("get_rig_recommendations", project_type=project_type):
            try:
                results = await self._knowledge_mound.query(
                    query=f"workspace rig {project_type}",
                    limit=100,
                    workspace_id=self._workspace_id,
                )

                # Aggregate performance by rig
                rig_stats: dict[str, dict[str, Any]] = {}
                for result in results:
                    metadata = result.get("metadata", {})
                    if metadata.get("type") != "workspace_rig_snapshot":
                        continue

                    rig_id = metadata.get("rig_id", "")
                    if not rig_id:
                        continue
                    if available_rigs and rig_id not in available_rigs:
                        continue

                    if rig_id not in rig_stats:
                        rig_stats[rig_id] = {
                            "name": metadata.get("rig_name", ""),
                            "total_tasks": 0,
                            "success_tasks": 0,
                            "convoys": 0,
                        }

                    stats = rig_stats[rig_id]
                    stats["total_tasks"] += metadata.get("tasks_completed", 0) + metadata.get(
                        "tasks_failed", 0
                    )
                    stats["success_tasks"] += metadata.get("tasks_completed", 0)
                    stats["convoys"] += metadata.get("active_convoys", 0)

                # Calculate recommendations
                recommendations = []
                for rig_id, stats in rig_stats.items():
                    if stats["total_tasks"] == 0:
                        continue

                    success_rate = stats["success_tasks"] / stats["total_tasks"]
                    throughput = stats["total_tasks"]  # Total tasks as proxy for throughput

                    # Score: 70% success rate, 30% throughput (normalized)
                    throughput_score = min(1.0, throughput / 100)  # Cap at 100 tasks
                    combined_score = (success_rate * 0.7) + (throughput_score * 0.3)

                    recommendations.append(
                        {
                            "rig_id": rig_id,
                            "rig_name": stats["name"],
                            "combined_score": combined_score,
                            "success_rate": success_rate,
                            "total_tasks": stats["total_tasks"],
                            "confidence": min(1.0, stats["total_tasks"] / 20),
                        }
                    )

                recommendations.sort(key=lambda r: r["combined_score"], reverse=True)
                return recommendations[:top_n]

            except Exception as e:
                logger.error(f"Failed to get rig recommendations: {e}")
                return []

    async def get_optimal_agent_count(
        self,
        rig_id: str,
        convoy_size: int,
    ) -> dict[str, Any]:
        """
        Recommend optimal agent count for a rig based on historical patterns.

        Args:
            rig_id: Rig to analyze
            convoy_size: Expected number of beads in convoy

        Returns:
            Recommendation with optimal agent count and confidence
        """
        patterns = await self.get_convoy_patterns(rig_id, limit=50, use_cache=True)

        if len(patterns) < 3:
            return {
                "recommendation_available": False,
                "reason": "Insufficient historical data",
                "sample_size": len(patterns),
            }

        # Analyze agent count vs performance
        successful = [p for p in patterns if p.status in ("done", "completed")]
        if not successful:
            return {
                "recommendation_available": False,
                "reason": "No successful convoys found",
            }

        # Find optimal agent ratio per bead
        agent_ratios = []
        for convoy in successful:
            if convoy.total_beads > 0 and convoy.assigned_agents > 0:
                ratio = convoy.assigned_agents / convoy.total_beads
                duration_per_bead = convoy.duration_seconds / convoy.total_beads
                agent_ratios.append(
                    {
                        "ratio": ratio,
                        "duration_per_bead": duration_per_bead,
                        "agents": convoy.assigned_agents,
                        "beads": convoy.total_beads,
                    }
                )

        if not agent_ratios:
            return {
                "recommendation_available": False,
                "reason": "No convoy data with agent/bead counts",
            }

        # Find best ratio (lowest duration per bead)
        best = min(agent_ratios, key=lambda r: r["duration_per_bead"])
        optimal_ratio = best["ratio"]
        recommended_agents = max(1, int(convoy_size * optimal_ratio))

        return {
            "recommendation_available": True,
            "rig_id": rig_id,
            "convoy_size": convoy_size,
            "recommended_agents": recommended_agents,
            "optimal_ratio": optimal_ratio,
            "expected_duration_per_bead": best["duration_per_bead"],
            "sample_size": len(agent_ratios),
            "confidence": min(1.0, len(successful) / 10),
        }

    async def get_merge_success_factors(
        self,
        rig_id: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Analyze factors that contribute to merge success for a rig.

        Args:
            rig_id: Rig to analyze
            limit: Maximum merge outcomes to consider

        Returns:
            Analysis of success factors
        """
        if not self._knowledge_mound:
            return {"analysis_available": False}

        with self._timed_operation("get_merge_success_factors", rig_id=rig_id):
            try:
                results = await self._knowledge_mound.query(
                    query=f"workspace merge rig {rig_id}",
                    limit=limit,
                    workspace_id=self._workspace_id,
                )

                successful = []
                failed = []

                for result in results:
                    metadata = result.get("metadata", {})
                    if metadata.get("type") != "workspace_merge_outcome":
                        continue
                    if metadata.get("rig_id") != rig_id:
                        continue

                    outcome = {
                        "tests_passed": metadata.get("tests_passed", False),
                        "review_approved": metadata.get("review_approved", False),
                        "conflicts_resolved": metadata.get("conflicts_resolved", 0),
                        "files_changed": metadata.get("files_changed", 0),
                    }

                    if metadata.get("success"):
                        successful.append(outcome)
                    else:
                        failed.append(outcome)

                if not successful and not failed:
                    return {"analysis_available": False, "reason": "No merge data"}

                # Calculate success factors
                total = len(successful) + len(failed)
                success_rate = len(successful) / total if total > 0 else 0

                factors = {
                    "tests_passed_rate": 0.0,
                    "review_approved_rate": 0.0,
                    "avg_files_changed_success": 0.0,
                    "avg_conflicts_success": 0.0,
                }

                if successful:
                    factors["tests_passed_rate"] = sum(
                        1 for s in successful if s["tests_passed"]
                    ) / len(successful)
                    factors["review_approved_rate"] = sum(
                        1 for s in successful if s["review_approved"]
                    ) / len(successful)
                    factors["avg_files_changed_success"] = sum(
                        s["files_changed"] for s in successful
                    ) / len(successful)
                    factors["avg_conflicts_success"] = sum(
                        s["conflicts_resolved"] for s in successful
                    ) / len(successful)

                return {
                    "analysis_available": True,
                    "rig_id": rig_id,
                    "sample_size": total,
                    "success_rate": success_rate,
                    "successful_count": len(successful),
                    "failed_count": len(failed),
                    "factors": factors,
                    "recommendations": [
                        "Run tests before merge" if factors["tests_passed_rate"] > 0.8 else None,
                        "Require review approval"
                        if factors["review_approved_rate"] > 0.8
                        else None,
                    ],
                }

            except Exception as e:
                logger.error(f"Failed to get merge success factors: {e}")
                return {"analysis_available": False, "error": str(e)}

    # =========================================================================
    # Sync from Workspace Manager
    # =========================================================================

    async def sync_from_workspace(self) -> dict[str, Any]:
        """
        Sync current workspace state to Knowledge Mound.

        Captures rig snapshots and recent convoy outcomes.

        Returns:
            Dict with counts of synced items
        """
        if not self._workspace_manager:
            logger.warning("No workspace manager configured for sync")
            return {"error": "No workspace manager configured"}

        synced = {
            "rigs": 0,
            "convoys": 0,
        }

        with self._timed_operation("sync_from_workspace"):
            try:
                # Sync rigs
                rigs = await self._workspace_manager.list_rigs()
                for rig in rigs:
                    snapshot = RigSnapshot(
                        rig_id=rig.rig_id,
                        name=rig.name,
                        workspace_id=rig.workspace_id,
                        status=rig.status.value,
                        repo_url=rig.config.repo_url,
                        branch=rig.config.branch,
                        assigned_agents=len(rig.assigned_agents),
                        max_agents=rig.config.max_agents,
                        active_convoys=len(rig.active_convoys),
                        tasks_completed=rig.tasks_completed,
                        tasks_failed=rig.tasks_failed,
                    )
                    if await self.store_rig_snapshot(snapshot):
                        synced["rigs"] += 1

                # Sync recent convoys
                convoys = await self._workspace_manager.list_convoys()
                for convoy in convoys:
                    outcome = ConvoyOutcome(
                        convoy_id=convoy.convoy_id,
                        workspace_id=convoy.workspace_id,
                        rig_id=convoy.rig_id,
                        name=convoy.name,
                        status=convoy.status.value,
                        total_beads=convoy.total_beads,
                        completed_beads=len([b for b in convoy.bead_ids if b]),  # Simplification
                        assigned_agents=len(convoy.assigned_agents),
                        duration_seconds=(
                            (convoy.completed_at or time.time())
                            - (convoy.started_at or convoy.created_at)
                            if convoy.started_at
                            else 0.0
                        ),
                        error_message=convoy.error,
                    )
                    if await self.store_convoy_outcome(outcome):
                        synced["convoys"] += 1

                logger.info(f"Synced from workspace: {synced}")
                return synced

            except Exception as e:
                logger.error(f"Failed to sync from workspace: {e}")
                return {"error": str(e)}

    # =========================================================================
    # Stats and Health
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        return {
            **self._stats,
            "workspace_id": self._workspace_id,
            "rig_cache_size": len(self._rig_performance_cache),
            "convoy_cache_size": len(self._convoy_patterns_cache),
            "has_knowledge_mound": self._knowledge_mound is not None,
            "has_workspace_manager": self._workspace_manager is not None,
        }

    def clear_cache(self) -> int:
        """Clear all caches and return count of cleared items."""
        count = len(self._rig_performance_cache) + len(self._convoy_patterns_cache)
        self._rig_performance_cache.clear()
        self._convoy_patterns_cache.clear()
        self._cache_times.clear()
        return count


__all__ = [
    "WorkspaceAdapter",
    "RigSnapshot",
    "ConvoyOutcome",
    "MergeOutcome",
]
