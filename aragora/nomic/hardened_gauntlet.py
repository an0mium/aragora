"""
Gauntlet validation mixin for HardenedOrchestrator.

Extracted from hardened_orchestrator.py for maintainability.
Handles gauntlet validation runs, content building, and constraint extraction.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.nomic.autonomous_orchestrator import AgentAssignment

logger = logging.getLogger("aragora.nomic.hardened_orchestrator")


class GauntletMixin:
    """Mixin providing gauntlet validation methods for HardenedOrchestrator."""

    # These attributes are expected to be set by the host class __init__
    hardened_config: Any
    _gauntlet_constraints: list[str]

    async def _run_gauntlet_validation(self, assignment: AgentAssignment) -> None:
        """Run lightweight gauntlet validation on completed assignment."""
        try:
            from aragora.gauntlet.runner import GauntletConfig, GauntletRunner

            # Build content summary from assignment result
            content = self._build_gauntlet_content(assignment)
            if not content:
                return

            config = GauntletConfig(
                attack_rounds=1,
            )
            runner = GauntletRunner(config)
            result = await runner.run(content, context=assignment.subtask.description)

            # Check for critical findings
            critical = [
                f
                for f in getattr(result, "findings", [])
                if getattr(f, "severity", "").lower() in ("critical", "high")
            ]

            if critical:
                logger.warning(
                    "gauntlet_critical subtask=%s findings=%d",
                    assignment.subtask.id,
                    len(critical),
                )
                assignment.status = "failed"
                assignment.result = {
                    **(assignment.result or {}),
                    "gauntlet_findings": [str(f) for f in critical[:5]],
                }

                # Feed findings back as constraints for next iteration
                new_constraints = self._extract_gauntlet_constraints(
                    critical, assignment.subtask.description
                )
                self._gauntlet_constraints.extend(new_constraints)

        except ImportError:
            logger.debug("Gauntlet unavailable, skipping validation")
        except (RuntimeError, OSError, ValueError) as e:
            logger.warning(
                "gauntlet_error subtask=%s: %s",
                assignment.subtask.id,
                e,
            )

    def _build_gauntlet_content(self, assignment: AgentAssignment) -> str:
        """Build content string for gauntlet validation from assignment."""
        parts = [f"Task: {assignment.subtask.title}"]
        parts.append(f"Description: {assignment.subtask.description}")
        if assignment.subtask.file_scope:
            parts.append(f"Files modified: {', '.join(assignment.subtask.file_scope)}")
        if assignment.result:
            workflow_result = assignment.result.get("workflow_result", "")
            if isinstance(workflow_result, str):
                parts.append(f"Output: {workflow_result[:2000]}")
        return "\n".join(parts)

    def _extract_gauntlet_constraints(
        self,
        findings: list[Any],
        subtask_description: str,
    ) -> list[str]:
        """Convert gauntlet findings into debate constraints for the next iteration.

        Each finding is transformed into a natural-language constraint that the
        MetaPlanner / ContextInit phase can inject into the next cycle's debate,
        ensuring agents address vulnerabilities discovered by the Gauntlet.

        Args:
            findings: List of GauntletFinding objects (or similar) with
                description/severity attributes.
            subtask_description: Description of the subtask that was validated.

        Returns:
            List of constraint strings suitable for injection into debate context.
        """
        constraints: list[str] = []

        for finding in findings[:10]:  # Cap at 10 to avoid context bloat
            description = getattr(finding, "description", str(finding))
            severity = getattr(finding, "severity", "unknown")
            category = getattr(finding, "category", "")

            # Truncate long descriptions
            if len(description) > 300:
                description = description[:297] + "..."

            constraint = (
                f"Previous iteration found [{severity}]: {description}. "
                f"Address this in the new design."
            )
            if category:
                constraint = (
                    f"Previous iteration found [{severity}/{category}]: "
                    f"{description}. Address this in the new design."
                )

            constraints.append(constraint)

        if constraints:
            logger.info(
                "gauntlet_constraints_extracted count=%d subtask=%s",
                len(constraints),
                subtask_description[:80],
            )

        return constraints


__all__ = ["GauntletMixin"]
