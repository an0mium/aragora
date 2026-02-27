"""Jira adapter for Decision Plan export.

Formats TicketData as Jira issues and creates them via the Jira REST API v3.
Supports Atlassian Cloud and Server/Data Center.

Usage:
    adapter = JiraAdapter(
        base_url="https://myorg.atlassian.net",
        project_key="PROJ",
        email="bot@company.com",
        api_token="token-abc",
    )
    result = await adapter.export_ticket(ticket)
"""

from __future__ import annotations

import base64
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

# Default timeout for Jira API calls
_DEFAULT_TIMEOUT = ClientTimeout(total=30, connect=10, sock_read=20)

# Jira priority name mapping (Jira uses 1-5 where 1=Highest)
_PRIORITY_MAP: dict[TicketPriority, str] = {
    TicketPriority.CRITICAL: "Highest",
    TicketPriority.HIGH: "High",
    TicketPriority.MEDIUM: "Medium",
    TicketPriority.LOW: "Low",
}


class JiraAdapter(ExportAdapter):
    """Export adapter that creates Jira issues from TicketData.

    Uses Jira REST API v3 with Basic Auth (email + API token)
    or Bearer token authentication.

    Args:
        base_url: Jira instance URL (e.g., 'https://myorg.atlassian.net').
        project_key: Target Jira project key (e.g., 'PROJ').
        email: Atlassian account email for basic auth.
        api_token: API token (Atlassian Cloud) or PAT (Data Center).
        issue_type: Default issue type name. Defaults to 'Task'.
        component: Optional component name to add to issues.
        labels: Additional Jira labels to apply.
        bearer_token: If set, use Bearer auth instead of Basic.
    """

    def __init__(
        self,
        base_url: str,
        project_key: str,
        email: str = "",
        api_token: str = "",
        *,
        issue_type: str = "Task",
        component: str | None = None,
        labels: list[str] | None = None,
        bearer_token: str | None = None,
        timeout: ClientTimeout | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._project_key = project_key
        self._email = email
        self._api_token = api_token
        self._bearer_token = bearer_token
        self._issue_type = issue_type
        self._component = component
        self._extra_labels = labels or []
        self._timeout = timeout or _DEFAULT_TIMEOUT
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        return "jira"

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # -- Auth headers --------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Aragora-DecisionExporter/1.0",
        }
        if self._bearer_token:
            headers["Authorization"] = f"Bearer {self._bearer_token}"
        elif self._email and self._api_token:
            creds = base64.b64encode(f"{self._email}:{self._api_token}".encode()).decode()
            headers["Authorization"] = f"Basic {creds}"
        return headers

    # -- Ticket export -------------------------------------------------------

    async def export_ticket(self, ticket: TicketData) -> dict[str, Any]:
        """Create a Jira issue from a TicketData."""
        issue_payload = self._format_issue(ticket)

        session = await self._get_session()
        url = f"{self._base_url}/rest/api/3/issue"

        try:
            async with session.post(
                url,
                data=json.dumps(issue_payload).encode("utf-8"),
                headers=self._auth_headers(),
            ) as response:
                if response.status in (200, 201):
                    data = await response.json()
                    issue_key = data.get("key", "")
                    issue_id = data.get("id", "")
                    return {
                        "success": True,
                        "issue_key": issue_key,
                        "issue_id": issue_id,
                        "issue_url": f"{self._base_url}/browse/{issue_key}",
                    }
                else:
                    body_text = await response.text()
                    error_msg = f"Jira API error {response.status}: {body_text[:200]}"
                    logger.warning(error_msg)
                    return {"success": False, "error": error_msg}
        except aiohttp.ClientError as exc:
            logger.warning("Jira API request failed: %s", exc)
            return {"success": False, "error": str(exc)}

    # -- Jira payload formatting ---------------------------------------------

    def _format_issue(self, ticket: TicketData) -> dict[str, Any]:
        """Format TicketData as a Jira issue creation payload."""
        # Build Atlassian Document Format (ADF) description
        description_adf = self._to_adf(ticket)

        fields: dict[str, Any] = {
            "project": {"key": self._project_key},
            "summary": ticket.title,
            "description": description_adf,
            "issuetype": {"name": self._issue_type},
        }

        # Priority
        jira_priority = _PRIORITY_MAP.get(ticket.priority, "Medium")
        fields["priority"] = {"name": jira_priority}

        # Labels — combine ticket labels with adapter-level extras
        all_labels = list(ticket.labels) + self._extra_labels
        if all_labels:
            # Jira labels can't have spaces
            fields["labels"] = [lbl.replace(" ", "_") for lbl in all_labels]

        # Component
        if self._component:
            fields["components"] = [{"name": self._component}]

        return {"fields": fields}

    @staticmethod
    def _to_adf(ticket: TicketData) -> dict[str, Any]:
        """Build an Atlassian Document Format body from TicketData.

        ADF is required by Jira REST API v3 for description fields.
        """
        content: list[dict[str, Any]] = []

        # Description text
        content.append(
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": ticket.description}],
            }
        )

        # Acceptance criteria as a bullet list
        if ticket.acceptance_criteria:
            content.append(
                {
                    "type": "heading",
                    "attrs": {"level": 3},
                    "content": [{"type": "text", "text": "Acceptance Criteria"}],
                }
            )
            list_items = []
            for criterion in ticket.acceptance_criteria:
                list_items.append(
                    {
                        "type": "listItem",
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [{"type": "text", "text": criterion}],
                            }
                        ],
                    }
                )
            content.append({"type": "bulletList", "content": list_items})

        # Provenance panel
        content.append(
            {
                "type": "panel",
                "attrs": {"panelType": "info"},
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Plan: {ticket.plan_id} | "
                                    f"Debate: {ticket.debate_id} | "
                                    f"Task: {ticket.task_id}"
                                ),
                            }
                        ],
                    }
                ],
            }
        )

        return {
            "type": "doc",
            "version": 1,
            "content": content,
        }
