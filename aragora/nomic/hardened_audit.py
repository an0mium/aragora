"""
Audit reconciliation mixin for HardenedOrchestrator.

Extracted from hardened_orchestrator.py for maintainability.
Handles cross-agent file overlap detection and audit logging.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.nomic.autonomous_orchestrator import AgentAssignment

logger = logging.getLogger("aragora.nomic.hardened_orchestrator")


class AuditMixin:
    """Mixin providing audit reconciliation methods for HardenedOrchestrator."""

    def _reconcile_audits(self, assignments: list[AgentAssignment]) -> None:
        """Detect cross-agent file overlaps and log reconciliation report."""
        completed = [a for a in assignments if a.status == "completed"]
        if len(completed) < 2:
            return

        # Build file -> assignments mapping
        file_assignments: dict[str, list[str]] = {}
        for a in completed:
            for f in a.subtask.file_scope:
                file_assignments.setdefault(f, []).append(a.subtask.id)

        # Find overlaps (files touched by multiple assignments)
        overlaps = {
            f: subtask_ids for f, subtask_ids in file_assignments.items() if len(subtask_ids) > 1
        }

        if not overlaps:
            logger.info("audit_reconciliation no_overlaps")
            return

        logger.warning(
            "audit_reconciliation overlaps=%d files=%s",
            len(overlaps),
            list(overlaps.keys()),
        )

        # Try to log via AuditLog
        try:
            from aragora.audit.log import AuditLog

            audit = AuditLog()
            audit.log(  # type: ignore[call-arg]
                event="orchestration_reconciliation",  # type: ignore[arg-type]
                data={
                    "overlapping_files": overlaps,
                    "assignment_count": len(completed),
                    "overlap_count": len(overlaps),
                },
            )
        except (ImportError, Exception) as e:
            logger.debug("AuditLog unavailable for reconciliation: %s", e)


__all__ = ["AuditMixin"]
