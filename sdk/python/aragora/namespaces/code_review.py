"""
Code Review Namespace API.

Provides multi-agent code review capabilities including:
- Code snippet review
- Diff/patch review
- GitHub PR review
- Security scanning
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

FindingSeverity = Literal["critical", "high", "medium", "low", "info"]
FindingCategory = Literal[
    "security", "performance", "maintainability", "test_coverage", "style", "logic"
]


class CodeReviewAPI:
    """
    Synchronous Code Review API.

    Provides methods for:
    - Code snippet review
    - Diff/patch review
    - GitHub PR review
    - Security scanning
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def review_code(
        self,
        code: str,
        language: str | None = None,
        file_path: str | None = None,
        review_types: list[FindingCategory] | None = None,
        context: str | None = None,
    ) -> dict[str, Any]:
        """
        Review a code snippet.

        Args:
            code: Code to review.
            language: Programming language (auto-detected if not provided).
            file_path: File path for context.
            review_types: Types of review to perform.
            context: Additional context.

        Returns:
            Review result with findings.
        """
        data: dict[str, Any] = {"code": code}
        if language:
            data["language"] = language
        if file_path:
            data["file_path"] = file_path
        if review_types:
            data["review_types"] = review_types
        if context:
            data["context"] = context

        return self._client._request("POST", "/api/v1/code-review/review", json=data)

    def review_diff(
        self,
        diff: str,
        base_branch: str | None = None,
        head_branch: str | None = None,
        review_types: list[FindingCategory] | None = None,
        context: str | None = None,
    ) -> dict[str, Any]:
        """
        Review a diff/patch.

        Args:
            diff: Unified diff format.
            base_branch: Base branch name.
            head_branch: Head branch name.
            review_types: Types of review to perform.
            context: Additional context.

        Returns:
            Review result with findings.
        """
        data: dict[str, Any] = {"diff": diff}
        if base_branch:
            data["base_branch"] = base_branch
        if head_branch:
            data["head_branch"] = head_branch
        if review_types:
            data["review_types"] = review_types
        if context:
            data["context"] = context

        return self._client._request("POST", "/api/v1/code-review/diff", json=data)

    def review_pr(
        self,
        pr_url: str,
        review_types: list[FindingCategory] | None = None,
        post_comments: bool = False,
    ) -> dict[str, Any]:
        """
        Review a GitHub pull request.

        Args:
            pr_url: GitHub PR URL.
            review_types: Types of review to perform.
            post_comments: Post comments to PR (requires GitHub token).

        Returns:
            Review result with findings.

        Raises:
            ValueError: If PR URL is invalid.
        """
        if "github.com" not in pr_url or "/pull/" not in pr_url:
            raise ValueError(
                "Invalid PR URL. Expected format: https://github.com/owner/repo/pull/123"
            )

        data: dict[str, Any] = {"pr_url": pr_url}
        if review_types:
            data["review_types"] = review_types
        if post_comments:
            data["post_comments"] = post_comments

        return self._client._request("POST", "/api/v1/code-review/pr", json=data)

    def get_result(self, result_id: str) -> dict[str, Any]:
        """
        Get a review result by ID.

        Args:
            result_id: Review result identifier.

        Returns:
            Review result details.
        """
        return self._client._request("GET", f"/api/v1/code-review/results/{result_id}")

    def get_history(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """
        Get review history.

        Args:
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            List of review results.
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        return self._client._request("GET", "/api/v1/code-review/history", params=params)

    def security_scan(
        self,
        code: str,
        language: str | None = None,
    ) -> dict[str, Any]:
        """
        Quick security-focused code scan.

        Args:
            code: Code to scan.
            language: Programming language (optional).

        Returns:
            Security scan result with findings.
        """
        data: dict[str, Any] = {"code": code}
        if language:
            data["language"] = language

        return self._client._request("POST", "/api/v1/code-review/security-scan", json=data)


class AsyncCodeReviewAPI:
    """Asynchronous Code Review API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def review_code(
        self,
        code: str,
        language: str | None = None,
        file_path: str | None = None,
        review_types: list[FindingCategory] | None = None,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Review a code snippet."""
        data: dict[str, Any] = {"code": code}
        if language:
            data["language"] = language
        if file_path:
            data["file_path"] = file_path
        if review_types:
            data["review_types"] = review_types
        if context:
            data["context"] = context

        return await self._client._request("POST", "/api/v1/code-review/review", json=data)

    async def review_diff(
        self,
        diff: str,
        base_branch: str | None = None,
        head_branch: str | None = None,
        review_types: list[FindingCategory] | None = None,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Review a diff/patch."""
        data: dict[str, Any] = {"diff": diff}
        if base_branch:
            data["base_branch"] = base_branch
        if head_branch:
            data["head_branch"] = head_branch
        if review_types:
            data["review_types"] = review_types
        if context:
            data["context"] = context

        return await self._client._request("POST", "/api/v1/code-review/diff", json=data)

    async def review_pr(
        self,
        pr_url: str,
        review_types: list[FindingCategory] | None = None,
        post_comments: bool = False,
    ) -> dict[str, Any]:
        """Review a GitHub pull request."""
        if "github.com" not in pr_url or "/pull/" not in pr_url:
            raise ValueError(
                "Invalid PR URL. Expected format: https://github.com/owner/repo/pull/123"
            )

        data: dict[str, Any] = {"pr_url": pr_url}
        if review_types:
            data["review_types"] = review_types
        if post_comments:
            data["post_comments"] = post_comments

        return await self._client._request("POST", "/api/v1/code-review/pr", json=data)

    async def get_result(self, result_id: str) -> dict[str, Any]:
        """Get a review result by ID."""
        return await self._client._request("GET", f"/api/v1/code-review/results/{result_id}")

    async def get_history(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Get review history."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        return await self._client._request("GET", "/api/v1/code-review/history", params=params)

    async def security_scan(
        self,
        code: str,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Quick security-focused code scan."""
        data: dict[str, Any] = {"code": code}
        if language:
            data["language"] = language

        return await self._client._request("POST", "/api/v1/code-review/security-scan", json=data)
