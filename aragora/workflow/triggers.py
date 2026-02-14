"""
Receipt Workflow Triggers.

Maps receipt properties to workflow templates, enabling automatic
workflow execution when decision receipts are generated.

Usage:
    trigger = ReceiptWorkflowTrigger()
    trigger.register_rule(
        name="escalate_failures",
        condition=lambda r: r.get("verdict") == "FAIL",
        workflow_template="escalation_review",
    )
    triggered = trigger.evaluate(receipt_data)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class TriggerRule:
    """A rule that maps receipt properties to a workflow template."""

    name: str
    condition: Callable[[dict[str, Any]], bool]
    workflow_template: str
    priority: int = 0  # Higher = evaluated first
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerResult:
    """Result of evaluating triggers against a receipt."""

    rule_name: str
    workflow_template: str
    matched: bool
    receipt_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "workflow_template": self.workflow_template,
            "matched": self.matched,
            "receipt_id": self.receipt_id,
            "metadata": self.metadata,
        }


class ReceiptWorkflowTrigger:
    """Evaluates receipt data against configured trigger rules.

    Rules are evaluated in priority order (highest first). Multiple
    rules can match the same receipt.
    """

    def __init__(self) -> None:
        self._rules: list[TriggerRule] = []

    @property
    def rules(self) -> list[TriggerRule]:
        return list(self._rules)

    def register_rule(
        self,
        name: str,
        condition: Callable[[dict[str, Any]], bool],
        workflow_template: str,
        priority: int = 0,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> TriggerRule:
        """Register a new trigger rule.

        Args:
            name: Unique name for the rule.
            condition: Callable that takes receipt dict and returns True if rule matches.
            workflow_template: Name of workflow template to trigger on match.
            priority: Evaluation priority (higher = first).
            enabled: Whether the rule is active.
            metadata: Extra metadata to attach to trigger results.

        Returns:
            The created TriggerRule.
        """
        rule = TriggerRule(
            name=name,
            condition=condition,
            workflow_template=workflow_template,
            priority=priority,
            enabled=enabled,
            metadata=metadata or {},
        )
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)
        logger.debug("Registered trigger rule: %s -> %s", name, workflow_template)
        return rule

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name.

        Returns:
            True if the rule was found and removed.
        """
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < before

    def enable_rule(self, name: str) -> bool:
        """Enable a rule by name."""
        for rule in self._rules:
            if rule.name == name:
                rule.enabled = True
                return True
        return False

    def disable_rule(self, name: str) -> bool:
        """Disable a rule by name."""
        for rule in self._rules:
            if rule.name == name:
                rule.enabled = False
                return True
        return False

    def evaluate(self, receipt_data: dict[str, Any]) -> list[TriggerResult]:
        """Evaluate all enabled rules against a receipt.

        Args:
            receipt_data: Dict representation of a decision receipt.

        Returns:
            List of TriggerResult for all matching rules.
        """
        receipt_id = receipt_data.get("receipt_id", "unknown")
        results = []

        for rule in self._rules:
            if not rule.enabled:
                continue

            try:
                matched = rule.condition(receipt_data)
            except (KeyError, TypeError, ValueError) as e:
                logger.warning("Trigger rule '%s' raised error: %s", rule.name, e)
                matched = False

            if matched:
                results.append(TriggerResult(
                    rule_name=rule.name,
                    workflow_template=rule.workflow_template,
                    matched=True,
                    receipt_id=receipt_id,
                    metadata=rule.metadata,
                ))
                logger.info(
                    "Trigger '%s' matched receipt %s -> workflow '%s'",
                    rule.name,
                    receipt_id,
                    rule.workflow_template,
                )

        return results

    def evaluate_first(self, receipt_data: dict[str, Any]) -> TriggerResult | None:
        """Evaluate rules and return only the first (highest priority) match.

        Args:
            receipt_data: Dict representation of a decision receipt.

        Returns:
            First matching TriggerResult or None.
        """
        results = self.evaluate(receipt_data)
        return results[0] if results else None


# Default trigger rules for common patterns
def create_default_triggers() -> ReceiptWorkflowTrigger:
    """Create a trigger with standard rules for common receipt patterns."""
    trigger = ReceiptWorkflowTrigger()

    trigger.register_rule(
        name="escalate_failures",
        condition=lambda r: r.get("verdict") == "FAIL",
        workflow_template="escalation_review",
        priority=100,
        metadata={"severity": "high"},
    )

    trigger.register_rule(
        name="review_conditional",
        condition=lambda r: r.get("verdict") == "CONDITIONAL",
        workflow_template="conditional_review",
        priority=50,
        metadata={"severity": "medium"},
    )

    trigger.register_rule(
        name="archive_passes",
        condition=lambda r: r.get("verdict") == "PASS" and r.get("confidence", 0) > 0.9,
        workflow_template="auto_archive",
        priority=10,
        metadata={"severity": "low"},
    )

    return trigger


__all__ = [
    "ReceiptWorkflowTrigger",
    "TriggerResult",
    "TriggerRule",
    "create_default_triggers",
]
