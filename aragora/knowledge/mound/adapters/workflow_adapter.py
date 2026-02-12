"""
WorkflowAdapter - Bridges workflow execution outcomes to the Knowledge Mound.

This adapter persists workflow execution results, step outcomes, and template
usage patterns to enable cross-workflow learning and analytics.

The adapter provides:
- Workflow result persistence with step-level granularity
- Template usage pattern tracking
- Category-based search (legal, healthcare, finance, etc.)
- Execution metric aggregation for optimization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import KnowledgeItem
    from aragora.workflow.types import WorkflowResult

EventCallback = Callable[[str, dict[str, Any]], None]

logger = logging.getLogger(__name__)

from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter
from aragora.knowledge.mound.adapters._semantic_mixin import SemanticSearchMixin
from aragora.knowledge.mound.adapters._reverse_flow_base import ReverseFlowMixin
from aragora.knowledge.mound.adapters._fusion_mixin import FusionMixin
from aragora.knowledge.mound.adapters._types import SyncResult


@dataclass
class WorkflowSearchResult:
    """Wrapper for workflow search results with similarity metadata."""

    workflow_id: str
    definition_id: str
    success: bool
    total_duration_ms: float
    step_count: int
    failed_steps: int
    similarity: float = 0.0
    category: str = ""
    error: str | None = None


@dataclass
class WorkflowOutcome:
    """Lightweight representation of a workflow execution for adapter storage.

    Decouples the adapter from the full WorkflowResult/WorkflowEngine types.
    """

    workflow_id: str
    definition_id: str
    success: bool
    total_duration_ms: float
    step_count: int
    failed_steps: int
    step_summaries: list[dict[str, Any]] = field(default_factory=list)
    category: str = ""
    template_name: str = ""
    error: str | None = None
    checkpoints_created: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_workflow_result(cls, result: WorkflowResult, **extra: Any) -> WorkflowOutcome:
        """Create a WorkflowOutcome from a WorkflowResult."""
        step_summaries = []
        failed_steps = 0
        for step in getattr(result, "steps", []):
            summary = {
                "step_id": getattr(step, "step_id", ""),
                "step_name": getattr(step, "step_name", ""),
                "status": str(getattr(step, "status", "")),
                "duration_ms": getattr(step, "duration_ms", 0.0),
                "retry_count": getattr(step, "retry_count", 0),
            }
            if getattr(step, "error", None):
                summary["error"] = str(step.error)[:200]
                failed_steps += 1
            step_summaries.append(summary)

        return cls(
            workflow_id=getattr(result, "workflow_id", ""),
            definition_id=getattr(result, "definition_id", ""),
            success=getattr(result, "success", False),
            total_duration_ms=getattr(result, "total_duration_ms", 0.0),
            step_count=len(step_summaries),
            failed_steps=failed_steps,
            step_summaries=step_summaries,
            error=getattr(result, "error", None),
            checkpoints_created=getattr(result, "checkpoints_created", 0),
            category=extra.get("category", ""),
            template_name=extra.get("template_name", ""),
        )


class WorkflowAdapter(FusionMixin, ReverseFlowMixin, SemanticSearchMixin, KnowledgeMoundAdapter):
    """
    Adapter that bridges workflow execution outcomes to the Knowledge Mound.

    Provides workflow result persistence, template pattern tracking, and
    category-based analytics for cross-workflow learning.

    Usage:
        adapter = WorkflowAdapter()
        adapter.store_execution(workflow_result, category="legal")
        await adapter.sync_to_km(mound)
        results = await adapter.search_by_template("contract_review")
    """

    adapter_name = "workflow"
    source_type = "workflow"

    def __init__(
        self,
        enable_dual_write: bool = False,
        event_callback: EventCallback | None = None,
        enable_resilience: bool = True,
    ):
        super().__init__(
            enable_dual_write=enable_dual_write,
            event_callback=event_callback,
            enable_resilience=enable_resilience,
        )
        self._pending_outcomes: list[WorkflowOutcome] = []
        self._synced_outcomes: dict[str, WorkflowOutcome] = {}

    def store_execution(self, result: Any, **extra: Any) -> None:
        """Store a workflow execution result for KM sync.

        Args:
            result: A WorkflowResult or WorkflowOutcome object.
            **extra: Additional metadata (category, template_name, etc.)
        """
        if isinstance(result, WorkflowOutcome):
            outcome = result
        else:
            outcome = WorkflowOutcome.from_workflow_result(result, **extra)

        outcome.metadata["km_sync_pending"] = True
        outcome.metadata["km_sync_requested_at"] = datetime.now(timezone.utc).isoformat()
        self._pending_outcomes.append(outcome)

        self._emit_event(
            "km_adapter_forward_sync",
            {
                "adapter": self.adapter_name,
                "workflow_id": outcome.workflow_id,
                "definition_id": outcome.definition_id,
                "success": outcome.success,
            },
        )

    def get(self, record_id: str) -> WorkflowOutcome | None:
        """Get a workflow outcome by ID."""
        clean_id = record_id[3:] if record_id.startswith("wf_") else record_id
        return self._synced_outcomes.get(clean_id)

    async def get_async(self, record_id: str) -> WorkflowOutcome | None:
        """Async version of get."""
        return self.get(record_id)

    async def search_by_template(
        self,
        template_id: str,
        limit: int = 10,
        success_only: bool = False,
    ) -> list[WorkflowSearchResult]:
        """Search workflow outcomes by template/definition ID.

        Args:
            template_id: Template or definition ID to search for
            limit: Max results
            success_only: Only return successful executions

        Returns:
            List of WorkflowSearchResult sorted by recency.
        """
        results: list[WorkflowSearchResult] = []
        all_outcomes = list(self._synced_outcomes.values()) + self._pending_outcomes

        for outcome in all_outcomes:
            if outcome.definition_id != template_id and outcome.template_name != template_id:
                continue
            if success_only and not outcome.success:
                continue

            results.append(
                WorkflowSearchResult(
                    workflow_id=outcome.workflow_id,
                    definition_id=outcome.definition_id,
                    success=outcome.success,
                    total_duration_ms=outcome.total_duration_ms,
                    step_count=outcome.step_count,
                    failed_steps=outcome.failed_steps,
                    similarity=1.0,
                    category=outcome.category,
                    error=outcome.error,
                )
            )

        results.sort(key=lambda r: r.total_duration_ms)
        return results[:limit]

    async def search_by_category(
        self,
        category: str,
        limit: int = 10,
        min_success_rate: float = 0.0,
    ) -> list[WorkflowSearchResult]:
        """Search workflow outcomes by category (legal, healthcare, etc.)."""
        results: list[WorkflowSearchResult] = []
        all_outcomes = list(self._synced_outcomes.values()) + self._pending_outcomes

        for outcome in all_outcomes:
            if outcome.category != category:
                continue

            results.append(
                WorkflowSearchResult(
                    workflow_id=outcome.workflow_id,
                    definition_id=outcome.definition_id,
                    success=outcome.success,
                    total_duration_ms=outcome.total_duration_ms,
                    step_count=outcome.step_count,
                    failed_steps=outcome.failed_steps,
                    similarity=0.9,
                    category=outcome.category,
                    error=outcome.error,
                )
            )

        return results[:limit]

    def to_knowledge_item(self, outcome: WorkflowOutcome) -> KnowledgeItem:
        """Convert a WorkflowOutcome to a KnowledgeItem."""
        from aragora.knowledge.mound.types import KnowledgeItem, KnowledgeSource
        from aragora.knowledge.unified.types import ConfidenceLevel

        status = "succeeded" if outcome.success else "failed"
        content = (
            f"Workflow {outcome.definition_id} {status} "
            f"({outcome.step_count} steps, {outcome.total_duration_ms:.0f}ms)"
        )
        if outcome.error:
            content += f"\nError: {outcome.error[:200]}"
        if outcome.failed_steps > 0:
            content += f"\nFailed steps: {outcome.failed_steps}/{outcome.step_count}"

        return KnowledgeItem(
            id=f"wf_{outcome.workflow_id}",
            content=content,
            source=KnowledgeSource.WORKFLOW,
            source_id=outcome.workflow_id,
            confidence=ConfidenceLevel.VERIFIED if outcome.success else ConfidenceLevel.LOW,
            created_at=outcome.created_at,
            updated_at=outcome.created_at,
            metadata={
                "definition_id": outcome.definition_id,
                "success": outcome.success,
                "total_duration_ms": outcome.total_duration_ms,
                "step_count": outcome.step_count,
                "failed_steps": outcome.failed_steps,
                "category": outcome.category,
                "template_name": outcome.template_name,
                "checkpoints_created": outcome.checkpoints_created,
                "step_summaries": outcome.step_summaries,
            },
        )

    async def sync_to_km(
        self,
        mound: Any,
        min_confidence: float = 0.0,
        batch_size: int = 50,
    ) -> SyncResult:
        """Sync pending workflow outcomes to Knowledge Mound."""
        start = datetime.now(timezone.utc)
        synced = 0
        skipped = 0
        failed = 0
        errors: list[str] = []

        pending = self._pending_outcomes[:batch_size]

        for outcome in pending:
            try:
                km_item = self.to_knowledge_item(outcome)

                if hasattr(mound, "store_item"):
                    await mound.store_item(km_item)
                elif hasattr(mound, "store"):
                    await mound.store(km_item)
                elif hasattr(mound, "_semantic_store"):
                    await mound._semantic_store.store(km_item)

                outcome.metadata["km_sync_pending"] = False
                outcome.metadata["km_synced_at"] = datetime.now(timezone.utc).isoformat()
                outcome.metadata["km_item_id"] = km_item.id

                self._synced_outcomes[outcome.workflow_id] = outcome
                synced += 1

                self._emit_event(
                    "km_adapter_forward_sync_complete",
                    {
                        "adapter": self.adapter_name,
                        "workflow_id": outcome.workflow_id,
                        "km_item_id": km_item.id,
                    },
                )

            except Exception as e:
                failed += 1
                error_msg = f"Failed to sync workflow {outcome.workflow_id}: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
                outcome.metadata["km_sync_error"] = str(e)

        synced_ids = {o.workflow_id for o in pending if o.metadata.get("km_sync_pending") is False}
        self._pending_outcomes = [
            o for o in self._pending_outcomes if o.workflow_id not in synced_ids
        ]

        duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        return SyncResult(
            records_synced=synced,
            records_skipped=skipped,
            records_failed=failed,
            errors=errors,
            duration_ms=duration_ms,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about stored workflow outcomes."""
        all_outcomes = list(self._synced_outcomes.values())
        return {
            "total_synced": len(self._synced_outcomes),
            "pending_sync": len(self._pending_outcomes),
            "success_rate": (
                sum(1 for o in all_outcomes if o.success) / len(all_outcomes)
                if all_outcomes
                else 0.0
            ),
            "avg_duration_ms": (
                sum(o.total_duration_ms for o in all_outcomes) / len(all_outcomes)
                if all_outcomes
                else 0.0
            ),
            "categories": list({o.category for o in all_outcomes if o.category}),
        }

    # --- SemanticSearchMixin required methods ---

    def _get_record_by_id(self, record_id: str) -> WorkflowOutcome | None:
        return self.get(record_id)

    def _record_to_dict(self, record: Any, similarity: float = 0.0) -> dict[str, Any]:
        return {
            "id": record.workflow_id,
            "definition_id": record.definition_id,
            "success": record.success,
            "total_duration_ms": record.total_duration_ms,
            "step_count": record.step_count,
            "failed_steps": record.failed_steps,
            "similarity": similarity,
        }

    # --- ReverseFlowMixin required methods ---

    def _get_record_for_validation(self, source_id: str) -> WorkflowOutcome | None:
        return self.get(source_id)

    def _apply_km_validation(
        self,
        record: Any,
        km_confidence: float,
        cross_refs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        record.metadata["km_validated"] = True
        record.metadata["km_validation_confidence"] = km_confidence
        record.metadata["km_validation_timestamp"] = datetime.now(timezone.utc).isoformat()
        if cross_refs:
            record.metadata["km_cross_references"] = cross_refs
        return True

    def _extract_source_id(self, item: dict[str, Any]) -> str | None:
        source_id = item.get("source_id", "")
        if source_id.startswith("wf_"):
            return source_id[3:]
        return source_id or None

    # --- FusionMixin required methods ---

    def _get_fusion_sources(self) -> list[str]:
        return ["debate", "compliance"]

    def _extract_fusible_data(self, km_item: dict[str, Any]) -> dict[str, Any] | None:
        if km_item.get("source") == "workflow":
            return {
                "success": km_item.get("metadata", {}).get("success", False),
                "duration_ms": km_item.get("metadata", {}).get("total_duration_ms", 0.0),
            }
        return None

    def _apply_fusion_result(
        self,
        record: Any,
        fusion_result: Any,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        record.metadata["fusion_applied"] = True
        record.metadata["fusion_timestamp"] = datetime.now(timezone.utc).isoformat()
        return True
