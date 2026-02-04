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

    def list_repos(self, limit: int = 20) -> dict[str, Any]:
        """List connected repositories."""
        return self._client.request("GET", "/api/v1/github/repos", params={"limit": limit})

    def get_repo(self, owner: str, repo: str) -> dict[str, Any]:
        """Get repository details."""
        return self._client.request("GET", f"/api/v1/github/repos/{owner}/{repo}")

    def analyze_pr(self, owner: str, repo: str, pr_number: int) -> dict[str, Any]:
        """Analyze a pull request."""
        return self._client.request(
            "POST", f"/api/v1/github/repos/{owner}/{repo}/pulls/{pr_number}/analyze"
        )

    def review_pr(
        self, owner: str, repo: str, pr_number: int, agents: list[str] | None = None
    ) -> dict[str, Any]:
        """Run AI review on a pull request."""
        data: dict[str, Any] = {}
        if agents:
            data["agents"] = agents
        return self._client.request(
            "POST", f"/api/v1/github/repos/{owner}/{repo}/pulls/{pr_number}/review", json=data
        )

    def list_issues(
        self, owner: str, repo: str, state: str = "open", limit: int = 20
    ) -> dict[str, Any]:
        """List repository issues."""
        return self._client.request(
            "GET",
            f"/api/v1/github/repos/{owner}/{repo}/issues",
            params={
                "state": state,
                "limit": limit,
            },
        )

    def analyze_issue(self, owner: str, repo: str, issue_number: int) -> dict[str, Any]:
        """Analyze an issue."""
        return self._client.request(
            "POST", f"/api/v1/github/repos/{owner}/{repo}/issues/{issue_number}/analyze"
        )

    def connect(self, installation_id: str) -> dict[str, Any]:
        """Connect GitHub installation."""
        return self._client.request(
            "POST", "/api/v1/github/connect", json={"installation_id": installation_id}
        )

    def disconnect(self, installation_id: str) -> dict[str, Any]:
        """Disconnect GitHub installation."""
        return self._client.request("DELETE", f"/api/v1/github/installations/{installation_id}")


class AsyncGithubAPI:
    """Asynchronous GitHub API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_repos(self, limit: int = 20) -> dict[str, Any]:
        """List connected repositories."""
        return await self._client.request("GET", "/api/v1/github/repos", params={"limit": limit})

    async def get_repo(self, owner: str, repo: str) -> dict[str, Any]:
        """Get repository details."""
        return await self._client.request("GET", f"/api/v1/github/repos/{owner}/{repo}")

    async def analyze_pr(self, owner: str, repo: str, pr_number: int) -> dict[str, Any]:
        """Analyze a pull request."""
        return await self._client.request(
            "POST", f"/api/v1/github/repos/{owner}/{repo}/pulls/{pr_number}/analyze"
        )

    async def review_pr(
        self, owner: str, repo: str, pr_number: int, agents: list[str] | None = None
    ) -> dict[str, Any]:
        """Run AI review on a pull request."""
        data: dict[str, Any] = {}
        if agents:
            data["agents"] = agents
        return await self._client.request(
            "POST", f"/api/v1/github/repos/{owner}/{repo}/pulls/{pr_number}/review", json=data
        )

    async def list_issues(
        self, owner: str, repo: str, state: str = "open", limit: int = 20
    ) -> dict[str, Any]:
        """List repository issues."""
        return await self._client.request(
            "GET",
            f"/api/v1/github/repos/{owner}/{repo}/issues",
            params={
                "state": state,
                "limit": limit,
            },
        )

    async def analyze_issue(self, owner: str, repo: str, issue_number: int) -> dict[str, Any]:
        """Analyze an issue."""
        return await self._client.request(
            "POST", f"/api/v1/github/repos/{owner}/{repo}/issues/{issue_number}/analyze"
        )

    async def connect(self, installation_id: str) -> dict[str, Any]:
        """Connect GitHub installation."""
        return await self._client.request(
            "POST", "/api/v1/github/connect", json={"installation_id": installation_id}
        )

    async def disconnect(self, installation_id: str) -> dict[str, Any]:
        """Disconnect GitHub installation."""
        return await self._client.request(
            "DELETE", f"/api/v1/github/installations/{installation_id}"
        )
