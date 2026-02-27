"""DecisionExporter — orchestrates export of DecisionPlan to external tools.

Takes a completed DecisionPlan, converts its implementation tasks into
normalized TicketData, and dispatches to all registered adapters.

Usage:
    exporter = DecisionExporter()
    exporter.register_adapter(JiraAdapter(base_url=..., project_key=...))
    exporter.register_adapter(WebhookAdapter(url=...))

    receipts = await exporter.export(plan)
    for r in receipts:
        print(r.adapter_name, r.status, r.tickets_exported)
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.integrations.exporters.base import (
    ExportAdapter,
    ExportReceipt,
    ExportStatus,
    TicketData,
    TicketPriority,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Priority mapping from risk levels
# ---------------------------------------------------------------------------

_RISK_TO_PRIORITY = {
    "critical": TicketPriority.CRITICAL,
    "high": TicketPriority.HIGH,
    "medium": TicketPriority.MEDIUM,
    "low": TicketPriority.LOW,
}


def _complexity_to_priority(complexity: str) -> TicketPriority:
    """Map task complexity to ticket priority as a fallback."""
    mapping = {
        "complex": TicketPriority.HIGH,
        "moderate": TicketPriority.MEDIUM,
        "simple": TicketPriority.LOW,
    }
    return mapping.get(complexity, TicketPriority.MEDIUM)


# ---------------------------------------------------------------------------
# DecisionExporter
# ---------------------------------------------------------------------------


class DecisionExporter:
    """Orchestrates export of DecisionPlan tasks to external tools.

    Converts each ImplementTask into a TicketData with:
    - Title derived from the task description
    - Description with full context including plan summary
    - Priority derived from risk register or task complexity
    - Acceptance criteria derived from verification plan test cases
    - Labels for categorization
    """

    def __init__(self) -> None:
        self._adapters: list[ExportAdapter] = []

    # -- Adapter management --------------------------------------------------

    def register_adapter(self, adapter: ExportAdapter) -> None:
        """Register an export adapter."""
        self._adapters.append(adapter)
        logger.info("Registered export adapter: %s", adapter.name)

    def unregister_adapter(self, name: str) -> bool:
        """Remove adapter by name. Returns True if found and removed."""
        before = len(self._adapters)
        self._adapters = [a for a in self._adapters if a.name != name]
        return len(self._adapters) < before

    @property
    def adapters(self) -> list[ExportAdapter]:
        return list(self._adapters)

    # -- Ticket extraction ---------------------------------------------------

    def extract_tickets(self, plan: Any) -> list[TicketData]:
        """Convert a DecisionPlan's tasks into normalized TicketData.

        Args:
            plan: A DecisionPlan instance (or duck-typed equivalent).

        Returns:
            List of TicketData, one per implementation task.
        """
        implement_plan = getattr(plan, "implement_plan", None)
        tasks = getattr(implement_plan, "tasks", []) if implement_plan else []

        if not tasks:
            logger.debug("DecisionPlan has no implementation tasks to export")
            return []

        plan_id = getattr(plan, "id", "")
        debate_id = getattr(plan, "debate_id", "")
        plan_task = getattr(plan, "task", "")

        # Build risk lookup: task_id -> highest risk level
        risk_lookup = self._build_risk_lookup(plan)

        # Build acceptance criteria from verification plan
        criteria_lookup = self._build_criteria_lookup(plan)

        tickets: list[TicketData] = []
        for task in tasks:
            task_id = getattr(task, "id", "")
            description = getattr(task, "description", "")
            complexity = getattr(task, "complexity", "moderate")
            files = getattr(task, "files", [])
            task_type = getattr(task, "task_type", None)

            # Determine priority
            risk_level = risk_lookup.get(task_id)
            if risk_level:
                priority = _RISK_TO_PRIORITY.get(risk_level, TicketPriority.MEDIUM)
            else:
                priority = _complexity_to_priority(complexity)

            # Build title
            title = self._make_title(description, plan_task)

            # Build description body
            body = self._make_description(
                description=description,
                plan_task=plan_task,
                plan_id=plan_id,
                debate_id=debate_id,
                complexity=complexity,
                files=files,
                task_type=task_type,
            )

            # Acceptance criteria
            criteria = criteria_lookup.get(task_id, [])

            # Labels
            labels = ["aragora", f"complexity:{complexity}"]
            if task_type:
                labels.append(f"type:{task_type}")

            tickets.append(
                TicketData(
                    title=title,
                    description=body,
                    priority=priority,
                    acceptance_criteria=criteria,
                    labels=labels,
                    plan_id=plan_id,
                    debate_id=debate_id,
                    task_id=task_id,
                    metadata={
                        "complexity": complexity,
                        "files": files,
                        "task_type": task_type,
                    },
                )
            )

        logger.info("Extracted %d ticket(s) from DecisionPlan %s", len(tickets), plan_id)
        return tickets

    # -- Export orchestration -------------------------------------------------

    async def export(self, plan: Any) -> list[ExportReceipt]:
        """Export a DecisionPlan to all registered adapters.

        Args:
            plan: A DecisionPlan instance.

        Returns:
            List of ExportReceipt, one per adapter.
        """
        if not self._adapters:
            logger.warning("No export adapters registered; nothing to export")
            return []

        tickets = self.extract_tickets(plan)
        if not tickets:
            return [
                ExportReceipt(
                    adapter_name=a.name,
                    status=ExportStatus.SKIPPED,
                    plan_id=getattr(plan, "id", ""),
                    debate_id=getattr(plan, "debate_id", ""),
                )
                for a in self._adapters
            ]

        receipts: list[ExportReceipt] = []
        for adapter in self._adapters:
            try:
                receipt = await adapter.export_tickets(tickets)
                receipts.append(receipt)
                logger.info(
                    "Export via %s: %d succeeded, %d failed",
                    adapter.name,
                    receipt.tickets_exported,
                    receipt.tickets_failed,
                )
            except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
                error_msg = f"Adapter {adapter.name} failed"
                logger.warning("%s: %s", error_msg, exc)
                receipt = ExportReceipt(
                    adapter_name=adapter.name,
                    plan_id=getattr(plan, "id", ""),
                    debate_id=getattr(plan, "debate_id", ""),
                )
                receipt.mark_failed(error_msg)
                receipts.append(receipt)

        return receipts

    # -- Internal helpers ----------------------------------------------------

    @staticmethod
    def _make_title(description: str, plan_task: str) -> str:
        """Produce a concise ticket title from the task description."""
        # Use the first line of description, capped at 120 chars
        first_line = description.split("\n")[0].strip()
        if len(first_line) > 120:
            first_line = first_line[:117] + "..."
        return f"[Aragora] {first_line}"

    @staticmethod
    def _make_description(
        *,
        description: str,
        plan_task: str,
        plan_id: str,
        debate_id: str,
        complexity: str,
        files: list[str],
        task_type: str | None,
    ) -> str:
        """Build the full ticket description body."""
        parts = [
            f"## Decision Context\n\n**Decision question:** {plan_task}",
            f"**Plan ID:** {plan_id}",
            f"**Debate ID:** {debate_id}",
            f"**Complexity:** {complexity}",
        ]
        if task_type:
            parts.append(f"**Task type:** {task_type}")
        parts.append("")
        parts.append(f"## Task Description\n\n{description}")
        if files:
            parts.append("")
            parts.append("## Affected Files\n")
            for f in files:
                parts.append(f"- `{f}`")
        parts.append("")
        parts.append(
            "*Auto-generated by Aragora Decision Exporter from multi-agent debate outcome.*"
        )
        return "\n".join(parts)

    @staticmethod
    def _build_risk_lookup(plan: Any) -> dict[str, str]:
        """Map task IDs to their highest associated risk level string."""
        risk_register = getattr(plan, "risk_register", None)
        if not risk_register:
            return {}
        risks = getattr(risk_register, "risks", [])
        lookup: dict[str, str] = {}
        level_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        for risk in risks:
            level_str = getattr(risk, "level", None)
            if hasattr(level_str, "value"):
                level_str = level_str.value
            # Associate with all tasks if risk metadata contains task_id
            related_task = getattr(risk, "task_id", None) or (
                getattr(risk, "metadata", {}) or {}
            ).get("task_id")
            if related_task and level_str:
                existing = lookup.get(related_task)
                if not existing or level_order.get(level_str, 0) > level_order.get(existing, 0):
                    lookup[related_task] = level_str
        return lookup

    @staticmethod
    def _build_criteria_lookup(plan: Any) -> dict[str, list[str]]:
        """Map task IDs to acceptance criteria from the verification plan."""
        verification_plan = getattr(plan, "verification_plan", None)
        if not verification_plan:
            return {}
        test_cases = getattr(verification_plan, "test_cases", [])
        lookup: dict[str, list[str]] = {}
        for tc in test_cases:
            task_id = getattr(tc, "task_id", None) or (getattr(tc, "metadata", {}) or {}).get(
                "task_id"
            )
            name = getattr(tc, "name", "") or getattr(tc, "description", "")
            if task_id and name:
                lookup.setdefault(task_id, []).append(name)
        return lookup
