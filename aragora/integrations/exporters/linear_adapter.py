"""Linear adapter for Decision Plan export.

Formats TicketData as Linear issues and creates them via the Linear
GraphQL API.

Usage:
    adapter = LinearAdapter(
        api_key="lin_api_xxx",
        team_id="TEAM-UUID",
    )
    result = await adapter.export_ticket(ticket)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import aiohttp
from aiohttp import ClientTimeout

from aragora.integrations.exporters.base import (
    ExportAdapter,
    TicketData,
    TicketPriority,
)

logger = logging.getLogger(__name__)

# Linear API endpoint
_LINEAR_API_URL = "https://api.linear.app/graphql"

# Default timeout for Linear API calls
_DEFAULT_TIMEOUT = ClientTimeout(total=30, connect=10, sock_read=20)

# Linear priority mapping (0=No priority, 1=Urgent, 2=High, 3=Medium, 4=Low)
_PRIORITY_MAP: dict[TicketPriority, int] = {
    TicketPriority.CRITICAL: 1,  # Urgent
    TicketPriority.HIGH: 2,
    TicketPriority.MEDIUM: 3,
    TicketPriority.LOW: 4,
}

# GraphQL mutation for creating an issue
_CREATE_ISSUE_MUTATION = """
mutation CreateIssue($input: IssueCreateInput!) {
    issueCreate(input: $input) {
        success
        issue {
            id
            identifier
            title
            url
        }
    }
}
"""

# GraphQL mutation for adding a label
_ADD_LABEL_MUTATION = """
mutation CreateLabel($input: IssueLabelCreateInput!) {
    issueLabelCreate(input: $input) {
        success
        issueLabel {
            id
            name
        }
    }
}
"""


class LinearAdapter(ExportAdapter):
    """Export adapter that creates Linear issues from TicketData.

    Uses the Linear GraphQL API with API key authentication.

    Args:
        api_key: Linear API key (starts with ``lin_api_``).
        team_id: Target team UUID in Linear.
        project_id: Optional project UUID to assign issues to.
        label_ids: Optional list of label UUIDs to apply to all issues.
        api_url: Override Linear API URL (for testing).
    """

    def __init__(
        self,
        api_key: str,
        team_id: str,
        *,
        project_id: str | None = None,
        label_ids: list[str] | None = None,
        api_url: str | None = None,
        timeout: ClientTimeout | None = None,
    ) -> None:
        self._api_key = api_key
        self._team_id = team_id
        self._project_id = project_id
        self._label_ids = label_ids or []
        self._api_url = api_url or _LINEAR_API_URL
        self._timeout = timeout or _DEFAULT_TIMEOUT
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        return "linear"

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # -- Auth headers --------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": self._api_key,
            "User-Agent": "Aragora-DecisionExporter/1.0",
        }

    # -- Ticket export -------------------------------------------------------

    async def export_ticket(self, ticket: TicketData) -> dict[str, Any]:
        """Create a Linear issue from a TicketData."""
        variables = self._format_variables(ticket)
        return await self._execute_mutation(_CREATE_ISSUE_MUTATION, variables, ticket)

    async def _execute_mutation(
        self,
        query: str,
        variables: dict[str, Any],
        ticket: TicketData,
    ) -> dict[str, Any]:
        """Execute a GraphQL mutation against the Linear API."""
        payload = {"query": query, "variables": variables}
        body = json.dumps(payload).encode("utf-8")

        session = await self._get_session()
        try:
            async with session.post(
                self._api_url,
                data=body,
                headers=self._headers(),
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    error_msg = f"Linear API error {response.status}: {text[:200]}"
                    logger.warning(error_msg)
                    return {"success": False, "error": error_msg}

                data = await response.json()

                # Check for GraphQL errors
                if data.get("errors"):
                    errors = data["errors"]
                    error_msg = errors[0].get("message", "Unknown GraphQL error")
                    logger.warning("Linear GraphQL error: %s", error_msg)
                    return {"success": False, "error": error_msg}

                # Extract issue data
                issue_create = (data.get("data") or {}).get("issueCreate", {})
                if not issue_create.get("success"):
                    return {"success": False, "error": "Issue creation reported failure"}

                issue = issue_create.get("issue", {})
                return {
                    "success": True,
                    "issue_id": issue.get("id", ""),
                    "identifier": issue.get("identifier", ""),
                    "issue_url": issue.get("url", ""),
                }
        except aiohttp.ClientError as exc:
            logger.warning("Linear API request failed: %s", exc)
            return {"success": False, "error": str(exc)}

    # -- Payload formatting --------------------------------------------------

    def _format_variables(self, ticket: TicketData) -> dict[str, Any]:
        """Build GraphQL variables for issue creation."""
        description = self._format_markdown_description(ticket)

        issue_input: dict[str, Any] = {
            "teamId": self._team_id,
            "title": ticket.title,
            "description": description,
            "priority": _PRIORITY_MAP.get(ticket.priority, 3),
        }

        if self._project_id:
            issue_input["projectId"] = self._project_id

        if self._label_ids:
            issue_input["labelIds"] = self._label_ids

        return {"input": issue_input}

    @staticmethod
    def _format_markdown_description(ticket: TicketData) -> str:
        """Format TicketData as Markdown for Linear's description field."""
        parts = [ticket.description]

        if ticket.acceptance_criteria:
            parts.append("")
            parts.append("### Acceptance Criteria")
            for criterion in ticket.acceptance_criteria:
                parts.append(f"- [ ] {criterion}")

        parts.append("")
        parts.append("---")
        parts.append(
            f"*Plan: {ticket.plan_id} | Debate: {ticket.debate_id} | Task: {ticket.task_id}*"
        )

        return "\n".join(parts)
