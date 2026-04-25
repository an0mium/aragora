"""Agent Bridge namespace API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class AgentBridgeAPI:
    """Synchronous Agent Bridge API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def list_runs(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List recorded agent-bridge runs."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._client.request("GET", "/api/v1/agent-bridge/runs", params=params)

    def start_run(
        self,
        *,
        task: str,
        actors: list[dict[str, Any]],
        run_id: str | None = None,
        next_actor: str | None = None,
        worktree_path: str | None = None,
        worktree_agent_slug: str | None = None,
        repair_budget_per_turn: int | None = None,
    ) -> dict[str, Any]:
        """Start an agent-bridge run without dispatching a turn."""
        data: dict[str, Any] = {"task": task, "actors": actors}
        if run_id is not None:
            data["run_id"] = run_id
        if next_actor is not None:
            data["next_actor"] = next_actor
        if worktree_path is not None:
            data["worktree_path"] = worktree_path
        if worktree_agent_slug is not None:
            data["worktree_agent_slug"] = worktree_agent_slug
        if repair_budget_per_turn is not None:
            data["repair_budget_per_turn"] = repair_budget_per_turn
        return self._client.request("POST", "/api/v1/agent-bridge/runs", json=data)

    def dispatch_turn(self, run_id: str, *, role: str, prompt: str) -> dict[str, Any]:
        """Dispatch one bridge turn to a run role."""
        return self._client.request(
            "POST",
            f"/api/v1/agent-bridge/runs/{run_id}/dispatch",
            json={"role": role, "prompt": prompt},
        )

    def auto_step(
        self,
        run_id: str,
        *,
        prompt: str | None = None,
        context_turns: int | None = None,
    ) -> dict[str, Any]:
        """Dispatch one turn to the run's next_actor."""
        data: dict[str, Any] = {}
        if prompt is not None:
            data["prompt"] = prompt
        if context_turns is not None:
            data["context_turns"] = context_turns
        return self._client.request(
            "POST",
            f"/api/v1/agent-bridge/runs/{run_id}/auto-step",
            json=data,
        )


class AsyncAgentBridgeAPI:
    """Asynchronous Agent Bridge API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def list_runs(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List recorded agent-bridge runs."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._client.request("GET", "/api/v1/agent-bridge/runs", params=params)

    async def start_run(
        self,
        *,
        task: str,
        actors: list[dict[str, Any]],
        run_id: str | None = None,
        next_actor: str | None = None,
        worktree_path: str | None = None,
        worktree_agent_slug: str | None = None,
        repair_budget_per_turn: int | None = None,
    ) -> dict[str, Any]:
        """Start an agent-bridge run without dispatching a turn."""
        data: dict[str, Any] = {"task": task, "actors": actors}
        if run_id is not None:
            data["run_id"] = run_id
        if next_actor is not None:
            data["next_actor"] = next_actor
        if worktree_path is not None:
            data["worktree_path"] = worktree_path
        if worktree_agent_slug is not None:
            data["worktree_agent_slug"] = worktree_agent_slug
        if repair_budget_per_turn is not None:
            data["repair_budget_per_turn"] = repair_budget_per_turn
        return await self._client.request("POST", "/api/v1/agent-bridge/runs", json=data)

    async def dispatch_turn(self, run_id: str, *, role: str, prompt: str) -> dict[str, Any]:
        """Dispatch one bridge turn to a run role."""
        return await self._client.request(
            "POST",
            f"/api/v1/agent-bridge/runs/{run_id}/dispatch",
            json={"role": role, "prompt": prompt},
        )

    async def auto_step(
        self,
        run_id: str,
        *,
        prompt: str | None = None,
        context_turns: int | None = None,
    ) -> dict[str, Any]:
        """Dispatch one turn to the run's next_actor."""
        data: dict[str, Any] = {}
        if prompt is not None:
            data["prompt"] = prompt
        if context_turns is not None:
            data["context_turns"] = context_turns
        return await self._client.request(
            "POST",
            f"/api/v1/agent-bridge/runs/{run_id}/auto-step",
            json=data,
        )
