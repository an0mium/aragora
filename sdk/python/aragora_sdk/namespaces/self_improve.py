"""
Self-Improve namespace for autonomous self-improvement run management.

Provides API access to starting, listing, and cancelling self-improvement
runs, as well as managing git worktrees used during execution.

Endpoints:
- POST /api/self-improve/start              - Start a new self-improvement run
- GET  /api/self-improve/runs               - List all runs
- GET  /api/self-improve/runs/:id           - Get run status and progress
- POST /api/self-improve/runs/:id/cancel    - Cancel a running run
- GET  /api/self-improve/history            - Get run history (alias for /runs)
- GET  /api/self-improve/worktrees          - List active worktrees
- POST /api/self-improve/worktrees/cleanup  - Clean up all worktrees
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

_List = list  # Preserve builtin list for type annotations


class SelfImproveAPI:
    """Synchronous self-improvement API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def start(
        self,
        goal: str,
        *,
        tracks: _List[str] | None = None,
        mode: str = "flat",
        budget_limit_usd: float | None = None,
        max_cycles: int = 5,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Start a new self-improvement run.

        POST /api/self-improve/start

        Args:
            goal: The improvement goal to pursue.
            tracks: Optional list of tracks to focus on.
            mode: Execution mode ('flat' or 'hierarchical').
            budget_limit_usd: Optional budget limit in USD.
            max_cycles: Maximum number of improvement cycles (default 5).
            dry_run: If True, generate a plan without executing.

        Returns:
            Dict with run_id and status ('started' or 'preview').
        """
        data: dict[str, Any] = {
            "goal": goal,
            "mode": mode,
            "max_cycles": max_cycles,
            "dry_run": dry_run,
        }
        if tracks is not None:
            data["tracks"] = tracks
        if budget_limit_usd is not None:
            data["budget_limit_usd"] = budget_limit_usd

        return self._client.request("POST", "/api/v1/self-improve/start", json=data)

    def list_runs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> dict[str, Any]:
        """
        List self-improvement runs with pagination.

        GET /api/self-improve/runs

        Args:
            limit: Maximum number of runs to return.
            offset: Number of runs to skip.
            status: Filter by run status.

        Returns:
            Dict with runs array, total count, limit, and offset.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        return self._client.request("GET", "/api/v1/self-improve/runs", params=params)

    def get_run(self, run_id: str) -> dict[str, Any]:
        """
        Get a specific run's status and progress.

        GET /api/self-improve/runs/:run_id

        Args:
            run_id: The run identifier.

        Returns:
            Run details including status, progress, and summary.
        """
        return self._client.request("GET", f"/api/v1/self-improve/runs/{run_id}")

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        """
        Cancel a running self-improvement run.

        POST /api/self-improve/runs/:run_id/cancel

        Args:
            run_id: The run identifier.

        Returns:
            Dict with run_id and status 'cancelled'.
        """
        return self._client.request("POST", f"/api/v1/self-improve/runs/{run_id}/cancel")

    def get_history(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> dict[str, Any]:
        """
        Get run history (alias for list_runs).

        GET /api/self-improve/history

        Args:
            limit: Maximum number of runs to return.
            offset: Number of runs to skip.
            status: Filter by run status.

        Returns:
            Dict with runs array, total count, limit, and offset.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        return self._client.request("GET", "/api/v1/self-improve/history", params=params)

    def list_worktrees(self) -> dict[str, Any]:
        """
        List active git worktrees managed by the branch coordinator.

        GET /api/self-improve/worktrees

        Returns:
            Dict with worktrees array and total count.
        """
        return self._client.request("GET", "/api/v1/self-improve/worktrees")

    def cleanup_worktrees(self) -> dict[str, Any]:
        """
        Clean up all managed worktrees.

        POST /api/self-improve/worktrees/cleanup

        Returns:
            Dict with removed count and status 'cleaned'.
        """
        return self._client.request("POST", "/api/v1/self-improve/worktrees/cleanup")


class AsyncSelfImproveAPI:
    """Asynchronous self-improvement API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def start(
        self,
        goal: str,
        *,
        tracks: _List[str] | None = None,
        mode: str = "flat",
        budget_limit_usd: float | None = None,
        max_cycles: int = 5,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Start a new self-improvement run. POST /api/self-improve/start"""
        data: dict[str, Any] = {
            "goal": goal,
            "mode": mode,
            "max_cycles": max_cycles,
            "dry_run": dry_run,
        }
        if tracks is not None:
            data["tracks"] = tracks
        if budget_limit_usd is not None:
            data["budget_limit_usd"] = budget_limit_usd

        return await self._client.request("POST", "/api/v1/self-improve/start", json=data)

    async def list_runs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> dict[str, Any]:
        """List self-improvement runs. GET /api/self-improve/runs"""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        return await self._client.request("GET", "/api/v1/self-improve/runs", params=params)

    async def get_run(self, run_id: str) -> dict[str, Any]:
        """Get run status and progress. GET /api/self-improve/runs/:run_id"""
        return await self._client.request("GET", f"/api/v1/self-improve/runs/{run_id}")

    async def cancel_run(self, run_id: str) -> dict[str, Any]:
        """Cancel a running run. POST /api/self-improve/runs/:run_id/cancel"""
        return await self._client.request("POST", f"/api/v1/self-improve/runs/{run_id}/cancel")

    async def get_history(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> dict[str, Any]:
        """Get run history. GET /api/self-improve/history"""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        return await self._client.request("GET", "/api/v1/self-improve/history", params=params)

    async def list_worktrees(self) -> dict[str, Any]:
        """List active worktrees. GET /api/self-improve/worktrees"""
        return await self._client.request("GET", "/api/v1/self-improve/worktrees")

    async def cleanup_worktrees(self) -> dict[str, Any]:
        """Clean up all worktrees. POST /api/self-improve/worktrees/cleanup"""
        return await self._client.request("POST", "/api/v1/self-improve/worktrees/cleanup")
