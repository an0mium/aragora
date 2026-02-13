"""
Type Protocols for External Connectors used by orchestration.

Defines structural typing interfaces for Confluence, GitHub, Jira,
email, and Knowledge Mound connectors.
"""

from __future__ import annotations

from typing import Any, Protocol
from collections.abc import Callable


class ConfluenceConnectorProtocol(Protocol):
    """Protocol for Confluence connector with page content fetching."""

    async def get_page_content(self, page_id: str) -> str | None:
        """Fetch content from a Confluence page."""
        ...


class GitHubConnectorProtocol(Protocol):
    """Protocol for GitHub connector with PR/issue content fetching."""

    async def get_pr_content(self, owner: str, repo: str, number: int) -> str | None:
        """Fetch content from a GitHub PR."""
        ...

    async def get_issue_content(self, owner: str, repo: str, number: int) -> str | None:
        """Fetch content from a GitHub issue."""
        ...


class JiraConnectorProtocol(Protocol):
    """Protocol for Jira connector with issue fetching."""

    async def get_issue(self, issue_key: str) -> dict[str, Any] | None:
        """Fetch a Jira issue."""
        ...


class EmailSenderProtocol(Protocol):
    """Protocol for email sending function."""

    async def __call__(self, to: str, subject: str, body: str) -> None:
        """Send an email."""
        ...


class KnowledgeMoundProtocol(Protocol):
    """Protocol for Knowledge Mound search interface."""

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search the knowledge mound."""
        ...


# Type alias for recommend_agents function
RecommendAgentsFunc = Callable[[str], "Any"]
