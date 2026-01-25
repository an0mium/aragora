"""
Knowledge Pruning Step for scheduled knowledge maintenance.

Provides workflow steps for:
- Automatic pruning of stale knowledge items
- Deduplication of similar content
- Confidence decay application
- Knowledge quality maintenance

These steps integrate with the workflow scheduler for periodic execution.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from aragora.workflow.step import BaseStep, WorkflowContext

logger = logging.getLogger(__name__)


class KnowledgePruningStep(BaseStep):
    """
    Workflow step for automatic knowledge pruning.

    Executes pruning operations on the Knowledge Mound based on
    configurable policies (staleness, confidence, usage).

    Config options:
        workspace_id: str - Workspace to prune (default: from context)
        policy_id: str - Existing policy ID to use (optional)
        staleness_threshold: float - Staleness threshold (default: 0.9)
        min_age_days: int - Minimum item age (default: 30)
        action: str - Pruning action: archive, delete, demote, flag (default: archive)
        dry_run: bool - Report only without changes (default: True)
        max_items: int - Maximum items to prune per run (default: 100)
        tier_exceptions: List[str] - Tiers to exclude (default: ["glacial"])

    Usage:
        step = KnowledgePruningStep(
            name="Nightly Pruning",
            config={
                "workspace_id": "production",
                "staleness_threshold": 0.85,
                "action": "archive",
                "dry_run": False,
            }
        )
        result = await step.execute(context)
    """

    VALID_ACTIONS = ["archive", "delete", "demote", "flag"]

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._mound = None

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the pruning step."""
        config = {**self._config, **context.current_step_config}

        # Extract configuration
        workspace_id = config.get("workspace_id", context.get_input("workspace_id", "default"))
        policy_id = config.get("policy_id")
        staleness_threshold = config.get("staleness_threshold", 0.9)
        min_age_days = config.get("min_age_days", 30)
        action = config.get("action", "archive")
        dry_run = config.get("dry_run", True)
        max_items = config.get("max_items", 100)
        tier_exceptions = config.get("tier_exceptions", ["glacial"])

        # Validate action
        if action not in self.VALID_ACTIONS:
            logger.warning(f"Invalid action '{action}', using 'archive'")
            action = "archive"

        try:
            from aragora.knowledge.mound import KnowledgeMound
            from aragora.knowledge.mound.ops.pruning import PruningAction, PruningPolicy

            mound = KnowledgeMound(workspace_id=workspace_id)
            await mound.initialize()

            # Create policy
            policy = PruningPolicy(
                policy_id=policy_id or f"workflow_{self.name}_{datetime.now().timestamp()}",
                workspace_id=workspace_id,
                name=f"Workflow: {self.name}",
                staleness_threshold=staleness_threshold,
                min_age_days=min_age_days,
                action=PruningAction(action),
                tier_exceptions=tier_exceptions,
                enabled=True,
            )

            # Get prunable items first
            items = await mound.get_prunable_items(
                workspace_id=workspace_id,
                staleness_threshold=staleness_threshold,
                min_age_days=min_age_days,
                limit=max_items,
            )

            if not items:
                logger.info(f"No items to prune in workspace '{workspace_id}'")
                return {
                    "success": True,
                    "workspace_id": workspace_id,
                    "dry_run": dry_run,
                    "items_found": 0,
                    "items_pruned": 0,
                    "message": "No items matched pruning criteria",
                }

            # Execute pruning
            result = await mound.auto_prune(
                workspace_id=workspace_id,
                policy=policy,
                dry_run=dry_run,
            )

            logger.info(
                f"Pruning complete for '{workspace_id}': "
                f"{result.items_pruned}/{result.items_analyzed} items "
                f"({'dry run' if dry_run else 'executed'})"
            )

            return {
                "success": True,
                "workspace_id": workspace_id,
                "policy_id": policy.policy_id,
                "dry_run": dry_run,
                "items_analyzed": result.items_analyzed,
                "items_pruned": result.items_pruned,
                "items_archived": result.items_archived,
                "items_deleted": result.items_deleted,
                "items_demoted": result.items_demoted,
                "items_flagged": result.items_flagged,
                "errors": result.errors if hasattr(result, "errors") else [],
            }

        except Exception as e:
            logger.error(f"Pruning step failed: {e}")
            return {
                "success": False,
                "workspace_id": workspace_id,
                "error": str(e),
            }


class KnowledgeDedupStep(BaseStep):
    """
    Workflow step for automatic knowledge deduplication.

    Finds and merges duplicate knowledge items based on content similarity.

    Config options:
        workspace_id: str - Workspace to dedupe (default: from context)
        similarity_threshold: float - Minimum similarity (default: 0.95)
        auto_merge: bool - Auto-merge exact duplicates (default: False)
        dry_run: bool - Report only without changes (default: True)
        max_clusters: int - Maximum clusters to process (default: 50)

    Usage:
        step = KnowledgeDedupStep(
            name="Weekly Dedup",
            config={
                "workspace_id": "production",
                "similarity_threshold": 0.95,
                "auto_merge": True,
                "dry_run": False,
            }
        )
        result = await step.execute(context)
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the deduplication step."""
        config = {**self._config, **context.current_step_config}

        workspace_id = config.get("workspace_id", context.get_input("workspace_id", "default"))
        similarity_threshold = config.get("similarity_threshold", 0.95)
        auto_merge = config.get("auto_merge", False)
        dry_run = config.get("dry_run", True)
        config.get("max_clusters", 50)

        try:
            from aragora.knowledge.mound import KnowledgeMound

            mound = KnowledgeMound(workspace_id=workspace_id)
            await mound.initialize()

            # Generate dedup report
            report = await mound.generate_dedup_report(
                workspace_id=workspace_id,
                similarity_threshold=similarity_threshold,
            )

            result_data = {
                "success": True,
                "workspace_id": workspace_id,
                "dry_run": dry_run,
                "total_nodes_analyzed": report.total_nodes_analyzed,
                "duplicate_clusters_found": report.duplicate_clusters_found,
                "estimated_reduction_percent": report.estimated_reduction_percent,
                "merges_performed": 0,
            }

            # Auto-merge if requested
            if auto_merge and not dry_run:
                merge_result = await mound.auto_merge_exact_duplicates(
                    workspace_id=workspace_id,
                    dry_run=False,
                )
                result_data["merges_performed"] = merge_result.get("merges_performed", 0)
                result_data["duplicates_removed"] = merge_result.get("duplicates_found", 0)

            logger.info(
                f"Dedup complete for '{workspace_id}': "
                f"{report.duplicate_clusters_found} clusters found, "
                f"{result_data.get('merges_performed', 0)} merged"
            )

            return result_data

        except Exception as e:
            logger.error(f"Dedup step failed: {e}")
            return {
                "success": False,
                "workspace_id": workspace_id,
                "error": str(e),
            }


class ConfidenceDecayStep(BaseStep):
    """
    Workflow step for applying confidence decay to knowledge items.

    Gradually reduces confidence scores over time to reflect
    potential staleness of information.

    Config options:
        workspace_id: str - Workspace to process (default: from context)
        decay_rate: float - Decay rate per day (default: 0.01 = 1%)
        min_confidence: float - Minimum confidence floor (default: 0.1)

    Usage:
        step = ConfidenceDecayStep(
            name="Daily Confidence Decay",
            config={
                "workspace_id": "production",
                "decay_rate": 0.005,  # 0.5% per day
                "min_confidence": 0.1,
            }
        )
        result = await step.execute(context)
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the confidence decay step."""
        config = {**self._config, **context.current_step_config}

        workspace_id = config.get("workspace_id", context.get_input("workspace_id", "default"))
        decay_rate = config.get("decay_rate", 0.01)
        min_confidence = config.get("min_confidence", 0.1)

        # Validate parameters
        if not 0 < decay_rate < 1:
            return {
                "success": False,
                "workspace_id": workspace_id,
                "error": "decay_rate must be between 0 and 1",
            }

        if not 0 < min_confidence < 1:
            return {
                "success": False,
                "workspace_id": workspace_id,
                "error": "min_confidence must be between 0 and 1",
            }

        try:
            from aragora.knowledge.mound import KnowledgeMound

            mound = KnowledgeMound(workspace_id=workspace_id)
            await mound.initialize()

            items_decayed = await mound.apply_confidence_decay(
                workspace_id=workspace_id,
                decay_rate=decay_rate,
                min_confidence=min_confidence,
            )

            logger.info(
                f"Confidence decay applied to '{workspace_id}': {items_decayed} items decayed"
            )

            return {
                "success": True,
                "workspace_id": workspace_id,
                "decay_rate": decay_rate,
                "min_confidence": min_confidence,
                "items_decayed": items_decayed,
            }

        except Exception as e:
            logger.error(f"Confidence decay step failed: {e}")
            return {
                "success": False,
                "workspace_id": workspace_id,
                "error": str(e),
            }


# Register task handlers for use with TaskStep
def _register_pruning_handlers():
    """Register pruning task handlers."""
    from aragora.workflow.nodes.task import register_task_handler

    async def prune_handler(context: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for pruning tasks."""
        step = KnowledgePruningStep("prune_task", context)
        from aragora.workflow.step import WorkflowContext

        wf_context = WorkflowContext(workflow_id="task_handler", definition_id="knowledge_prune")
        return await step.execute(wf_context)

    async def dedup_handler(context: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for dedup tasks."""
        step = KnowledgeDedupStep("dedup_task", context)
        from aragora.workflow.step import WorkflowContext

        wf_context = WorkflowContext(workflow_id="task_handler", definition_id="knowledge_dedup")
        return await step.execute(wf_context)

    async def decay_handler(context: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for confidence decay tasks."""
        step = ConfidenceDecayStep("decay_task", context)
        from aragora.workflow.step import WorkflowContext

        wf_context = WorkflowContext(workflow_id="task_handler", definition_id="knowledge_decay")
        return await step.execute(wf_context)

    register_task_handler("knowledge_prune", prune_handler)
    register_task_handler("knowledge_dedup", dedup_handler)
    register_task_handler("knowledge_decay", decay_handler)

    logger.debug("Registered knowledge maintenance task handlers")


# Auto-register handlers on module load
try:
    _register_pruning_handlers()
except Exception as e:
    logger.warning(f"Could not register pruning handlers: {e}")
