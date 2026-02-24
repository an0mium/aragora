"""
PipelineAdapter - Bridges Idea-to-Execution Pipeline outcomes to Knowledge Mound.

Enables cross-pipeline learning:
- Data flow IN: Pipeline results stored as knowledge items
- Data flow IN: Stage transitions and provenance recorded
- Reverse flow: GoalExtractor queries past pipelines for pattern learning

Usage:
    adapter = PipelineAdapter(mound)
    result = await adapter.ingest_pipeline_result(pipeline_result, workspace_id="pipeline")
    similar = await adapter.find_similar_pipelines("rate limiter design", limit=5)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from collections.abc import Callable

if TYPE_CHECKING:
    pass

from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter
from aragora.knowledge.unified.types import KnowledgeSource

logger = logging.getLogger(__name__)

EventCallback = Callable[[str, dict[str, Any]], None]


class PipelineStatus:
    """Pipeline execution status constants."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    PLANNED = "planned"


@dataclass
class PipelineIngestionResult:
    """Result of ingesting a pipeline result into Knowledge Mound."""

    pipeline_id: str
    items_ingested: int
    transitions_recorded: int
    provenance_links_recorded: int
    knowledge_item_ids: list[str]
    errors: list[str]

    @property
    def success(self) -> bool:
        """Check if ingestion was successful."""
        return len(self.errors) == 0 and self.items_ingested > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "items_ingested": self.items_ingested,
            "transitions_recorded": self.transitions_recorded,
            "provenance_links_recorded": self.provenance_links_recorded,
            "knowledge_item_ids": self.knowledge_item_ids,
            "errors": self.errors,
            "success": self.success,
        }


@dataclass
class SimilarPipeline:
    """A past pipeline similar to the current query."""

    pipeline_id: str
    description: str
    similarity: float
    status: str
    stages_completed: int
    goals_extracted: int
    tasks_executed: int
    what_worked: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "description": self.description,
            "similarity": self.similarity,
            "status": self.status,
            "stages_completed": self.stages_completed,
            "goals_extracted": self.goals_extracted,
            "tasks_executed": self.tasks_executed,
            "what_worked": self.what_worked,
        }


class PipelineAdapterError(Exception):
    """Base exception for pipeline adapter errors."""


class PipelineAdapter(KnowledgeMoundAdapter):
    """Adapter that bridges Idea-to-Execution Pipeline results to the Knowledge Mound.

    Provides methods to:
    - Ingest pipeline results (all 4 stages) as knowledge items
    - Record stage transitions with provenance links
    - Record execution outcomes (success/failure per task)
    - Find similar past pipelines for cross-pipeline learning
    - Extract high-ROI goal patterns for goal extraction improvement

    Usage:
        from aragora.knowledge.mound.adapters import PipelineAdapter

        adapter = PipelineAdapter(mound)

        # After a pipeline completes
        result = await adapter.ingest_pipeline_result(
            pipeline_data, workspace_id="pipeline"
        )

        # When starting a new pipeline
        similar = await adapter.find_similar_pipelines(
            "rate limiter design", limit=5
        )
    """

    adapter_name = "pipeline"

    ID_PREFIX = "pipeline_"

    def __init__(
        self,
        mound: Any = None,
        on_event: EventCallback | None = None,
    ):
        """Initialize the adapter.

        Args:
            mound: KnowledgeMound instance (optional, will use singleton if not provided)
            on_event: Callback for emitting events
        """
        super().__init__()
        self._mound = mound
        self._on_event = on_event

    @property
    def mound(self) -> Any:
        """Get the Knowledge Mound instance (lazy initialization)."""
        if self._mound is None:
            try:
                from aragora.knowledge.mound import get_knowledge_mound

                self._mound = get_knowledge_mound(workspace_id="pipeline")
            except ImportError:
                logger.warning("Knowledge Mound not available")
        return self._mound

    def _emit_event(self, event: str, data: dict[str, Any]) -> None:
        """Emit an event if callback is registered."""
        if self._on_event:
            try:
                self._on_event(event, data)
            except (RuntimeError, ValueError, TypeError, AttributeError):
                pass

    def _generate_pipeline_km_id(self, pipeline_id: str) -> str:
        """Generate a unique KM document ID for a pipeline."""
        content_hash = hashlib.sha256(pipeline_id.encode()).hexdigest()[:12]
        return f"{self.ID_PREFIX}{content_hash}"

    async def ingest_pipeline_result(
        self,
        pipeline_data: dict[str, Any],
        workspace_id: str = "pipeline",
    ) -> PipelineIngestionResult:
        """Ingest a pipeline result into the Knowledge Mound.

        Creates knowledge items for:
        - The pipeline summary
        - Individual stage transitions
        - Goal data for pattern learning
        - Execution outcomes

        Args:
            pipeline_data: PipelineResult.to_dict() output
            workspace_id: Workspace to store in

        Returns:
            PipelineIngestionResult with ingestion details
        """
        async with self._resilient_call("ingest_pipeline_result"):
            return await self._do_ingest(pipeline_data, workspace_id)

    async def _do_ingest(
        self,
        pipeline_data: dict[str, Any],
        workspace_id: str,
    ) -> PipelineIngestionResult:
        """Internal ingestion implementation."""
        errors: list[str] = []
        knowledge_item_ids: list[str] = []
        items_ingested = 0
        transitions_recorded = 0
        provenance_recorded = 0

        mound = self.mound
        if mound is None:
            errors.append("Knowledge Mound not available")
            return PipelineIngestionResult(
                pipeline_id=pipeline_data.get("pipeline_id", "unknown"),
                items_ingested=0,
                transitions_recorded=0,
                provenance_links_recorded=0,
                knowledge_item_ids=[],
                errors=errors,
            )

        try:
            from aragora.knowledge.mound import IngestionRequest

            pipeline_id = pipeline_data.get("pipeline_id", "unknown")
            km_id = self._generate_pipeline_km_id(pipeline_id)

            # 1. Ingest pipeline summary
            summary = self._build_pipeline_summary(pipeline_data)
            stages_completed = sum(
                1 for s in pipeline_data.get("stage_status", {}).values() if s == "complete"
            )
            summary_request = IngestionRequest(
                content=summary,
                workspace_id=workspace_id,
                source_type=KnowledgeSource.INSIGHT,
                document_id=km_id,
                confidence=0.9,
                topics=["pipeline", "idea_to_execution"],
                metadata={
                    "type": "pipeline_result",
                    "pipeline_id": pipeline_id,
                    "stages_completed": stages_completed,
                    "provenance_count": pipeline_data.get("provenance_count", 0),
                    "integrity_hash": pipeline_data.get("integrity_hash", ""),
                    "duration": pipeline_data.get("duration", 0),
                },
            )
            result = await mound.store(summary_request)
            if result and hasattr(result, "item_id"):
                knowledge_item_ids.append(result.item_id)
                items_ingested += 1

            # 2. Ingest stage transitions
            for transition in pipeline_data.get("transitions", []):
                trans_id = f"{km_id}_trans_{transition.get('id', '')}"
                from_stage = transition.get("from_stage", "")
                to_stage = transition.get("to_stage", "")
                confidence = transition.get("confidence", 0.5)
                ai_rationale = transition.get("ai_rationale", "")

                trans_content = (
                    f"STAGE TRANSITION: {from_stage} -> {to_stage}\n"
                    f"Confidence: {confidence:.1%}\n"
                    f"Rationale: {ai_rationale}"
                )
                trans_request = IngestionRequest(
                    content=trans_content,
                    workspace_id=workspace_id,
                    source_type=KnowledgeSource.INSIGHT,
                    document_id=trans_id,
                    confidence=confidence,
                    topics=["pipeline", "transition"],
                    metadata={
                        "type": "pipeline_transition",
                        "parent_pipeline_id": pipeline_id,
                        "from_stage": from_stage,
                        "to_stage": to_stage,
                    },
                )
                result = await mound.store(trans_request)
                if result and not getattr(result, "deduplicated", False):
                    transitions_recorded += 1
                    if hasattr(result, "item_id"):
                        knowledge_item_ids.append(result.item_id)

            # 3. Ingest goal data for pattern learning
            goals_data = pipeline_data.get("goals", {})
            if goals_data and isinstance(goals_data, dict) and goals_data.get("goals"):
                for goal in goals_data["goals"]:
                    goal_id_val = goal.get("id", "")
                    goal_km_id = f"{km_id}_goal_{goal_id_val}"
                    goal_title = goal.get("title", "")
                    goal_type = goal.get("type", "goal")
                    goal_priority = goal.get("priority", "medium")
                    goal_desc = goal.get("description", "")

                    goal_content = (
                        f"PIPELINE GOAL: {goal_title}\n"
                        f"Type: {goal_type}\n"
                        f"Priority: {goal_priority}\n"
                        f"Description: {goal_desc}"
                    )
                    goal_request = IngestionRequest(
                        content=goal_content,
                        workspace_id=workspace_id,
                        source_type=KnowledgeSource.INSIGHT,
                        document_id=goal_km_id,
                        confidence=goal.get("confidence", 0.5),
                        topics=["pipeline", "goal"],
                        metadata={
                            "type": "pipeline_goal",
                            "parent_pipeline_id": pipeline_id,
                            "goal_type": goal_type,
                            "priority": goal_priority,
                        },
                    )
                    result = await mound.store(goal_request)
                    if result and not getattr(result, "deduplicated", False):
                        items_ingested += 1
                        if hasattr(result, "item_id"):
                            knowledge_item_ids.append(result.item_id)

            # 4. Ingest execution outcomes
            orch = pipeline_data.get("orchestration_result", {})
            if orch and isinstance(orch, dict):
                for task_result in orch.get("results", []):
                    task_id = task_result.get("task_id", "")
                    task_km_id = f"{km_id}_task_{task_id}"
                    task_name = task_result.get("name", "")
                    task_status = task_result.get("status", "unknown")

                    task_content = (
                        f"PIPELINE TASK OUTCOME: {task_name}\n"
                        f"Status: {task_status}\n"
                        f"Task ID: {task_id}"
                    )
                    task_request = IngestionRequest(
                        content=task_content,
                        workspace_id=workspace_id,
                        source_type=KnowledgeSource.INSIGHT,
                        document_id=task_km_id,
                        confidence=0.9 if task_status == "completed" else 0.5,
                        topics=["pipeline", "task_outcome"],
                        metadata={
                            "type": "pipeline_task_outcome",
                            "parent_pipeline_id": pipeline_id,
                            "task_status": task_status,
                        },
                    )
                    result = await mound.store(task_request)
                    if result and not getattr(result, "deduplicated", False):
                        items_ingested += 1
                        if hasattr(result, "item_id"):
                            knowledge_item_ids.append(result.item_id)

            # 5. Record provenance count
            provenance_recorded = pipeline_data.get("provenance_count", 0)

            self._emit_event(
                "pipeline_ingested",
                {
                    "pipeline_id": pipeline_id,
                    "items_ingested": items_ingested,
                    "transitions_recorded": transitions_recorded,
                },
            )

        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            error_msg = f"Failed to ingest pipeline: {e}"
            errors.append(error_msg)
            logger.exception(error_msg)

        return PipelineIngestionResult(
            pipeline_id=pipeline_data.get("pipeline_id", "unknown"),
            items_ingested=items_ingested,
            transitions_recorded=transitions_recorded,
            provenance_links_recorded=provenance_recorded,
            knowledge_item_ids=knowledge_item_ids,
            errors=errors,
        )

    def _build_pipeline_summary(self, pipeline_data: dict[str, Any]) -> str:
        """Build a text summary of the pipeline for knowledge storage."""
        lines = [
            f"PIPELINE: {pipeline_data.get('pipeline_id', 'unknown')}",
        ]
        stage_status = pipeline_data.get("stage_status", {})
        completed = sum(1 for s in stage_status.values() if s == "complete")
        lines.append(f"Stages Completed: {completed}/{len(stage_status)}")

        goals = pipeline_data.get("goals", {})
        if goals and isinstance(goals, dict) and goals.get("goals"):
            lines.append(f"Goals Extracted: {len(goals['goals'])}")
            for g in goals["goals"][:5]:
                lines.append(f"  - [{g.get('priority', '?')}] {g.get('title', '')}")

        orch = pipeline_data.get("orchestration_result", {})
        if orch and isinstance(orch, dict):
            lines.append(f"Orchestration: {orch.get('status', 'unknown')}")
            lines.append(f"Tasks: {orch.get('tasks_completed', 0)}/{orch.get('tasks_total', 0)}")

        lines.append(f"Integrity: {pipeline_data.get('integrity_hash', 'none')}")
        return "\n".join(lines)

    async def find_similar_pipelines(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.3,
        workspace_id: str = "pipeline",
    ) -> list[SimilarPipeline]:
        """Find past pipelines similar to the given query.

        Uses semantic search to find pipelines with similar content,
        useful for cross-pipeline learning when starting new pipelines.

        Args:
            query: Description of the current pipeline intent
            limit: Maximum number of similar pipelines to return
            min_similarity: Minimum similarity score (0-1)
            workspace_id: Workspace to search in

        Returns:
            List of SimilarPipeline objects ordered by relevance
        """
        async with self._resilient_call("find_similar_pipelines"):
            return await self._do_find_similar(query, limit, min_similarity, workspace_id)

    async def _do_find_similar(
        self,
        query: str,
        limit: int,
        min_similarity: float,
        workspace_id: str,
    ) -> list[SimilarPipeline]:
        """Internal implementation of finding similar pipelines."""
        mound = self.mound
        if mound is None:
            return []

        similar: list[SimilarPipeline] = []
        try:
            results = await mound.search(
                query=query,
                workspace_id=workspace_id,
                limit=limit * 2,
                filters={"type": "pipeline_result"},
            )
            for result in results:
                metadata = getattr(result, "metadata", {})
                if metadata.get("type") != "pipeline_result":
                    continue
                similarity = getattr(result, "score", 0.5)
                if similarity < min_similarity:
                    continue

                stages = metadata.get("stages_completed", 0)
                similar.append(
                    SimilarPipeline(
                        pipeline_id=metadata.get("pipeline_id", ""),
                        description=getattr(result, "content", "")[:200],
                        similarity=similarity,
                        status="complete" if stages == 4 else "partial",
                        stages_completed=stages,
                        goals_extracted=0,
                        tasks_executed=0,
                        what_worked=[],
                    )
                )
            similar.sort(key=lambda p: p.similarity, reverse=True)
            return similar[:limit]
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.warning("Failed to find similar pipelines: %s", e)
            return []

    async def get_high_roi_patterns(
        self,
        limit: int = 5,
        workspace_id: str = "pipeline",
    ) -> list[dict[str, Any]]:
        """Find goal patterns that historically lead to successful pipeline execution.

        Queries past pipeline goals, groups by type and priority, and ranks
        by frequency. This helps GoalExtractor focus on goal types that
        have historically been most commonly pursued.

        Args:
            limit: Maximum number of patterns to return
            workspace_id: Workspace to search in

        Returns:
            List of dicts: [{goal_type, priority, count, examples}]
        """
        mound = self.mound
        if mound is None:
            return []

        try:
            results = await mound.search(
                query="pipeline goal success high priority",
                workspace_id=workspace_id,
                limit=50,
                filters={"type": "pipeline_goal"},
            )

            pattern_data: dict[str, dict[str, Any]] = {}
            for result in results:
                metadata = getattr(result, "metadata", {})
                if metadata.get("type") != "pipeline_goal":
                    continue
                goal_type = metadata.get("goal_type", "goal")
                priority = metadata.get("priority", "medium")
                pattern_key = f"{goal_type}_{priority}"

                if pattern_key not in pattern_data:
                    pattern_data[pattern_key] = {
                        "goal_type": goal_type,
                        "priority": priority,
                        "count": 0,
                        "examples": [],
                    }
                entry = pattern_data[pattern_key]
                entry["count"] += 1
                content = getattr(result, "content", "")
                if len(entry["examples"]) < 3:
                    entry["examples"].append(content[:100])

            ranked = sorted(pattern_data.values(), key=lambda x: x["count"], reverse=True)
            return ranked[:limit]
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.warning("Failed to find high-ROI patterns: %s", e)
            return []

    async def get_execution_success_rate(
        self,
        workspace_id: str = "pipeline",
    ) -> dict[str, Any]:
        """Get overall task execution success rate across all pipelines.

        Args:
            workspace_id: Workspace to search in

        Returns:
            Dict with total, completed, failed, planned counts and rate
        """
        mound = self.mound
        if mound is None:
            return {"total": 0, "completed": 0, "failed": 0, "planned": 0, "rate": 0.0}

        try:
            results = await mound.search(
                query="pipeline task outcome",
                workspace_id=workspace_id,
                limit=100,
                filters={"type": "pipeline_task_outcome"},
            )

            total = 0
            completed = 0
            failed = 0
            planned = 0

            for result in results:
                metadata = getattr(result, "metadata", {})
                if metadata.get("type") != "pipeline_task_outcome":
                    continue
                total += 1
                status = metadata.get("task_status", "unknown")
                if status == "completed":
                    completed += 1
                elif status == "failed":
                    failed += 1
                elif status == "planned":
                    planned += 1

            rate = completed / total if total > 0 else 0.0
            return {
                "total": total,
                "completed": completed,
                "failed": failed,
                "planned": planned,
                "rate": round(rate, 3),
            }
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.warning("Failed to get execution success rate: %s", e)
            return {"total": 0, "completed": 0, "failed": 0, "planned": 0, "rate": 0.0}


    async def query_precedents(
        self,
        task_type: str,
        limit: int = 5,
        workspace_id: str = "pipeline",
    ) -> list[dict[str, Any]]:
        """Find historical task outcomes similar to the given task type.

        Returns precedent dicts with outcome, task_type, and status keys
        that can inform new execution with historical lessons learned.
        """
        mound = self.mound
        if mound is None:
            return []

        try:
            results = await mound.search(
                query=f"pipeline task {task_type} outcome lessons",
                workspace_id=workspace_id,
                limit=limit * 2,
                filters={"type": "pipeline_task_outcome"},
            )

            precedents: list[dict[str, Any]] = []
            for result in results:
                metadata = getattr(result, "metadata", {})
                if metadata.get("type") != "pipeline_task_outcome":
                    continue
                precedents.append({
                    "task_type": metadata.get("task_type", task_type),
                    "outcome": getattr(result, "content", "")[:200],
                    "status": metadata.get("task_status", "unknown"),
                    "agent_type": metadata.get("agent_type", ""),
                })
                if len(precedents) >= limit:
                    break

            return precedents
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.warning("Failed to query precedents for task_type=%s: %s", task_type, e)
            return []

    async def get_agent_performance(
        self,
        agent_type: str,
        domain: str = "general",
        workspace_id: str = "pipeline",
    ) -> dict[str, Any]:
        """Get historical performance metrics for an agent type in a domain.

        Returns dict with success_rate, total_tasks, and avg_duration.
        """
        if not agent_type:
            return {}

        mound = self.mound
        if mound is None:
            return {}

        try:
            results = await mound.search(
                query=f"agent {agent_type} {domain} performance outcome",
                workspace_id=workspace_id,
                limit=50,
                filters={"type": "pipeline_task_outcome"},
            )

            total = 0
            completed = 0
            for result in results:
                metadata = getattr(result, "metadata", {})
                if metadata.get("type") != "pipeline_task_outcome":
                    continue
                if metadata.get("agent_type", "") != agent_type:
                    continue
                total += 1
                if metadata.get("task_status") == "completed":
                    completed += 1

            if total == 0:
                return {}

            return {
                "agent_type": agent_type,
                "domain": domain,
                "total_tasks": total,
                "success_rate": completed / total,
            }
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.warning("Failed to get agent performance for %s: %s", agent_type, e)
            return {}


def get_pipeline_adapter(workspace_id: str = "pipeline") -> PipelineAdapter:
    """Get or create a PipelineAdapter instance."""
    return PipelineAdapter()


__all__ = [
    "PipelineAdapter",
    "PipelineAdapterError",
    "PipelineIngestionResult",
    "SimilarPipeline",
    "PipelineStatus",
    "get_pipeline_adapter",
]
