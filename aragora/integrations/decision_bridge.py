"""Decision Bridge -- routes DecisionPlans to external project management tools.

After a debate produces a DecisionPlan, this bridge can automatically:
- Create Jira issues from implementation tasks
- Create Linear issues from implementation tasks
- Trigger n8n workflows via webhook

Usage:
    bridge = DecisionBridge()
    results = await bridge.handle_decision_plan(plan)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BridgeResult:
    """Result of routing a DecisionPlan to external tools."""

    jira_issues: list[dict[str, Any]] = field(default_factory=list)
    linear_issues: list[dict[str, Any]] = field(default_factory=list)
    n8n_triggered: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "jira_issues": self.jira_issues,
            "linear_issues": self.linear_issues,
            "n8n_triggered": self.n8n_triggered,
            "errors": self.errors,
        }


class DecisionBridge:
    """Routes DecisionPlans to configured external integrations.

    Integrations are determined by:
    1. plan.metadata["integrations"] list (e.g., ["jira", "linear", "n8n"])
    2. Environment variables (ARAGORA_DECISION_BRIDGE_TARGETS)
    3. Defaults to no integrations if neither is configured
    """

    def __init__(self, default_targets: list[str] | None = None):
        env_targets = os.environ.get("ARAGORA_DECISION_BRIDGE_TARGETS", "")
        self._default_targets = default_targets or (
            [t.strip() for t in env_targets.split(",") if t.strip()]
            if env_targets
            else []
        )

    def _get_targets(self, plan: Any) -> list[str]:
        """Determine which integrations to route to."""
        metadata = getattr(plan, "metadata", None) or {}
        explicit = metadata.get("integrations", [])
        if explicit:
            return list(explicit)
        return list(self._default_targets)

    async def handle_decision_plan(self, plan: Any) -> BridgeResult:
        """Route a DecisionPlan to configured integrations.

        Args:
            plan: DecisionPlan instance with implement_plan.tasks and metadata.

        Returns:
            BridgeResult with created issues and status.
        """
        result = BridgeResult()
        targets = self._get_targets(plan)

        if not targets:
            logger.debug("No decision bridge targets configured; skipping")
            return result

        for target in targets:
            target_lower = target.lower()
            try:
                if target_lower == "jira":
                    result.jira_issues = await self._create_jira_issues(plan)
                elif target_lower == "linear":
                    result.linear_issues = await self._create_linear_issues(plan)
                elif target_lower == "n8n":
                    result.n8n_triggered = await self._trigger_n8n_workflow(plan)
                else:
                    logger.warning("Unknown decision bridge target: %s", target)
            except (ImportError, ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
                error_msg = f"Decision bridge failed for {target}"
                logger.warning("%s: %s", error_msg, e)
                result.errors.append(error_msg)

        return result

    async def _create_jira_issues(self, plan: Any) -> list[dict[str, Any]]:
        """Create Jira issues from DecisionPlan tasks."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector()
        created: list[dict[str, Any]] = []

        implement_plan = getattr(plan, "implement_plan", None)
        tasks = getattr(implement_plan, "tasks", []) if implement_plan else []
        plan_title = getattr(plan, "title", "Decision Plan")

        for task in tasks:
            title = getattr(task, "title", "") or getattr(task, "description", "Untitled task")
            description = getattr(task, "description", "")
            issue_data = {
                "fields": {
                    "summary": f"[Aragora] {title}",
                    "description": (
                        f"Auto-created from decision plan: {plan_title}\n\n"
                        f"{description}"
                    ),
                    "issuetype": {"name": "Task"},
                }
            }
            try:
                response = await connector._api_request("POST", "/rest/api/3/issue", json=issue_data)
                created.append(
                    {"key": response.get("key", ""), "id": response.get("id", ""), "summary": title}
                )
            except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
                logger.warning("Failed to create Jira issue for task '%s': %s", title, e)

        logger.info("Created %d Jira issues from decision plan", len(created))
        return created

    async def _create_linear_issues(self, plan: Any) -> list[dict[str, Any]]:
        """Create Linear issues from DecisionPlan tasks."""
        from aragora.connectors.enterprise.collaboration.linear import LinearConnector

        connector = LinearConnector()
        created: list[dict[str, Any]] = []

        implement_plan = getattr(plan, "implement_plan", None)
        tasks = getattr(implement_plan, "tasks", []) if implement_plan else []
        plan_title = getattr(plan, "title", "Decision Plan")

        # Get default team for issue creation
        teams = await connector.get_teams()
        if not teams:
            logger.warning("No Linear teams found; cannot create issues")
            return created
        default_team_id = teams[0].id

        for task in tasks:
            title = getattr(task, "title", "") or getattr(task, "description", "Untitled task")
            description = getattr(task, "description", "")
            try:
                issue = await connector.create_issue(
                    team_id=default_team_id,
                    title=f"[Aragora] {title}",
                    description=(
                        f"Auto-created from decision plan: {plan_title}\n\n"
                        f"{description}"
                    ),
                )
                created.append(
                    {"id": issue.id, "identifier": issue.identifier, "title": title}
                )
            except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
                logger.warning("Failed to create Linear issue for task '%s': %s", title, e)

        logger.info("Created %d Linear issues from decision plan", len(created))
        return created

    async def _trigger_n8n_workflow(self, plan: Any) -> bool:
        """Trigger n8n workflow via event dispatch for a DecisionPlan."""
        from aragora.integrations.n8n import get_n8n_integration

        integration = get_n8n_integration()

        plan_title = getattr(plan, "title", "Decision Plan")
        plan_id = getattr(plan, "id", "")
        metadata = getattr(plan, "metadata", {}) or {}

        implement_plan = getattr(plan, "implement_plan", None)
        task_count = len(getattr(implement_plan, "tasks", []) or [])

        event_data = {
            "plan_id": str(plan_id),
            "title": plan_title,
            "status": getattr(plan, "status", "created"),
            "task_count": task_count,
            "metadata": metadata,
        }

        triggered = await integration.dispatch_event("decision_made", event_data)
        if triggered > 0:
            logger.info("Triggered %d n8n webhook(s) for decision plan %s", triggered, plan_id)
        return triggered > 0
