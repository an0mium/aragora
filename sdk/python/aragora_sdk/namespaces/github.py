"""
GitHub Namespace API

Provides methods for GitHub integration:
- Repository analysis
- PR review
- Issue tracking
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class GithubAPI:
    """Synchronous GitHub API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_audit_issues(self) -> dict[str, Any]:
        """List audit issues from GitHub."""
        return self._client.request("GET", "/api/v1/github/audit/issues")

    def create_audit_issue(self, **kwargs: Any) -> dict[str, Any]:
        """Create an audit issue on GitHub."""
        return self._client.request("POST", "/api/v1/github/audit/issues", json=kwargs)

    def create_audit_issues_bulk(self, **kwargs: Any) -> dict[str, Any]:
        """Create audit issues in bulk on GitHub."""
        return self._client.request("POST", "/api/v1/github/audit/issues/bulk", json=kwargs)

    def create_audit_pr(self, **kwargs: Any) -> dict[str, Any]:
        """Create an audit PR on GitHub."""
        return self._client.request("POST", "/api/v1/github/audit/pr", json=kwargs)

    def review_pr(self, **kwargs: Any) -> dict[str, Any]:
        """Submit a PR review."""
        return self._client.request("POST", "/api/v1/github/pr/review", json=kwargs)


class AsyncGithubAPI:
    """Asynchronous GitHub API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_audit_issues(self) -> dict[str, Any]:
        """List audit issues from GitHub."""
        return await self._client.request("GET", "/api/v1/github/audit/issues")

    async def create_audit_issue(self, **kwargs: Any) -> dict[str, Any]:
        """Create an audit issue on GitHub."""
        return await self._client.request("POST", "/api/v1/github/audit/issues", json=kwargs)

    async def create_audit_issues_bulk(self, **kwargs: Any) -> dict[str, Any]:
        """Create audit issues in bulk on GitHub."""
        return await self._client.request("POST", "/api/v1/github/audit/issues/bulk", json=kwargs)

    async def create_audit_pr(self, **kwargs: Any) -> dict[str, Any]:
        """Create an audit PR on GitHub."""
        return await self._client.request("POST", "/api/v1/github/audit/pr", json=kwargs)

    async def review_pr(self, **kwargs: Any) -> dict[str, Any]:
        """Submit a PR review."""
        return await self._client.request("POST", "/api/v1/github/pr/review", json=kwargs)
