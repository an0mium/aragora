"""Main client for the Aragora SDK."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from aragora_client.audit import AuditAPI
from aragora_client.auth import AuthAPI
from aragora_client.control_plane import ControlPlaneAPI
from aragora_client.onboarding import OnboardingAPI
from aragora_client.exceptions import (
    AragoraAuthenticationError,
    AragoraConnectionError,
    AragoraError,
    AragoraNotFoundError,
    AragoraTimeoutError,
    AragoraValidationError,
)
from aragora_client.rbac import RBACAPI
from aragora_client.tenancy import TenancyAPI
from aragora_client.tournaments import TournamentsAPI
from aragora_client.types import (
    AgentProfile,
    AgentScore,
    CreateDebateRequest,
    CreateGraphDebateRequest,
    CreateMatrixDebateRequest,
    Debate,
    GauntletReceipt,
    GraphBranch,
    GraphDebate,
    HealthStatus,
    MatrixConclusion,
    MatrixDebate,
    MemoryAnalytics,
    MemoryTierStats,
    RunGauntletRequest,
    ScoreAgentsRequest,
    SelectionPlugins,
    SelectTeamRequest,
    TeamSelection,
    VerificationResult,
    VerifyClaimRequest,
)


class DebatesAPI:
    """API for debate operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def create(
        self,
        task: str,
        *,
        agents: list[str] | None = None,
        max_rounds: int = 5,
        consensus_threshold: float = 0.8,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a new debate."""
        request = CreateDebateRequest(
            task=task,
            agents=agents,
            max_rounds=max_rounds,
            consensus_threshold=consensus_threshold,
            metadata=kwargs.get("metadata", {}),
        )
        return await self._client._post("/api/v1/debates", request.model_dump())

    async def get(self, debate_id: str) -> Debate:
        """Get a debate by ID."""
        data = await self._client._get(f"/api/v1/debates/{debate_id}")
        return Debate.model_validate(data)

    async def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> list[Debate]:
        """List debates."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        data = await self._client._get("/api/v1/debates", params=params)
        return [Debate.model_validate(d) for d in data.get("debates", [])]

    async def run(
        self,
        task: str,
        *,
        agents: list[str] | None = None,
        max_rounds: int = 5,
        consensus_threshold: float = 0.8,
        poll_interval: float = 1.0,
        timeout: float = 300.0,
        **kwargs: Any,
    ) -> Debate:
        """Run a debate and wait for completion."""
        response = await self.create(
            task,
            agents=agents,
            max_rounds=max_rounds,
            consensus_threshold=consensus_threshold,
            **kwargs,
        )
        debate_id = response["id"]

        elapsed = 0.0
        while elapsed < timeout:
            debate = await self.get(debate_id)
            if debate.status.value in ("completed", "failed", "cancelled"):
                return debate
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise AragoraTimeoutError(
            f"Debate {debate_id} did not complete within {timeout}s"
        )


class GraphDebatesAPI:
    """API for graph debate operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def create(
        self,
        task: str,
        *,
        agents: list[str] | None = None,
        max_rounds: int = 5,
        branch_threshold: float = 0.5,
        max_branches: int = 10,
    ) -> dict[str, Any]:
        """Create a new graph debate."""
        request = CreateGraphDebateRequest(
            task=task,
            agents=agents,
            max_rounds=max_rounds,
            branch_threshold=branch_threshold,
            max_branches=max_branches,
        )
        return await self._client._post("/api/v1/graph-debates", request.model_dump())

    async def get(self, debate_id: str) -> GraphDebate:
        """Get a graph debate by ID."""
        data = await self._client._get(f"/api/v1/graph-debates/{debate_id}")
        return GraphDebate.model_validate(data)

    async def get_branches(self, debate_id: str) -> list[GraphBranch]:
        """Get branches for a graph debate."""
        data = await self._client._get(f"/api/v1/graph-debates/{debate_id}/branches")
        return [GraphBranch.model_validate(b) for b in data.get("branches", [])]


class MatrixDebatesAPI:
    """API for matrix debate operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def create(
        self,
        task: str,
        scenarios: list[dict[str, Any]],
        *,
        agents: list[str] | None = None,
        max_rounds: int = 3,
    ) -> dict[str, Any]:
        """Create a new matrix debate."""
        request = CreateMatrixDebateRequest(
            task=task,
            scenarios=scenarios,
            agents=agents,
            max_rounds=max_rounds,
        )
        return await self._client._post("/api/v1/matrix-debates", request.model_dump())

    async def get(self, debate_id: str) -> MatrixDebate:
        """Get a matrix debate by ID."""
        data = await self._client._get(f"/api/v1/matrix-debates/{debate_id}")
        return MatrixDebate.model_validate(data)

    async def get_conclusions(self, debate_id: str) -> MatrixConclusion:
        """Get conclusions for a matrix debate."""
        data = await self._client._get(
            f"/api/v1/matrix-debates/{debate_id}/conclusions"
        )
        return MatrixConclusion.model_validate(data)


class AgentsAPI:
    """API for agent operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def list(self) -> list[AgentProfile]:
        """List all available agents."""
        data = await self._client._get("/api/v1/agents")
        return [AgentProfile.model_validate(a) for a in data.get("agents", [])]

    async def get(self, agent_id: str) -> AgentProfile:
        """Get an agent profile."""
        data = await self._client._get(f"/api/v1/agents/{agent_id}")
        return AgentProfile.model_validate(data)

    async def history(self, agent_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get match history for an agent."""
        data = await self._client._get(
            f"/api/v1/agents/{agent_id}/history", params={"limit": limit}
        )
        return data.get("matches", [])

    async def rivals(self, agent_id: str) -> list[dict[str, Any]]:
        """Get rivals for an agent."""
        data = await self._client._get(f"/api/v1/agents/{agent_id}/rivals")
        return data.get("rivals", [])

    async def allies(self, agent_id: str) -> list[dict[str, Any]]:
        """Get allies for an agent."""
        data = await self._client._get(f"/api/v1/agents/{agent_id}/allies")
        return data.get("allies", [])


class VerificationAPI:
    """API for formal verification."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def verify(
        self,
        claim: str,
        *,
        backend: str = "z3",
        timeout: int = 30,
    ) -> VerificationResult:
        """Verify a claim using formal methods."""
        request = VerifyClaimRequest(claim=claim, backend=backend, timeout=timeout)
        data = await self._client._post(
            "/api/v1/verification/verify", request.model_dump()
        )
        return VerificationResult.model_validate(data)

    async def status(self) -> dict[str, Any]:
        """Get verification backend status."""
        return await self._client._get("/api/v1/verification/status")


class GauntletAPI:
    """API for gauntlet validation."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def run(
        self,
        input_content: str,
        *,
        input_type: str = "spec",
        persona: str = "security",
    ) -> dict[str, Any]:
        """Run gauntlet validation."""
        request = RunGauntletRequest(
            input_content=input_content,
            input_type=input_type,
            persona=persona,
        )
        return await self._client._post("/api/v1/gauntlet/run", request.model_dump())

    async def get_receipt(self, gauntlet_id: str) -> GauntletReceipt:
        """Get a gauntlet receipt."""
        data = await self._client._get(f"/api/v1/gauntlet/{gauntlet_id}/receipt")
        return GauntletReceipt.model_validate(data)

    async def run_and_wait(
        self,
        input_content: str,
        *,
        input_type: str = "spec",
        persona: str = "security",
        poll_interval: float = 1.0,
        timeout: float = 120.0,
    ) -> GauntletReceipt:
        """Run gauntlet and wait for completion."""
        response = await self.run(input_content, input_type=input_type, persona=persona)
        gauntlet_id = response["gauntlet_id"]

        elapsed = 0.0
        while elapsed < timeout:
            try:
                return await self.get_receipt(gauntlet_id)
            except AragoraNotFoundError:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

        raise AragoraTimeoutError(
            f"Gauntlet {gauntlet_id} did not complete within {timeout}s"
        )


class MemoryAPI:
    """API for memory system."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def analytics(self, days: int = 30) -> MemoryAnalytics:
        """Get memory analytics."""
        data = await self._client._get(
            "/api/v1/memory/analytics", params={"days": days}
        )
        return MemoryAnalytics.model_validate(data)

    async def tier_stats(self, tier: str) -> MemoryTierStats:
        """Get stats for a specific memory tier."""
        data = await self._client._get(f"/api/v1/memory/tiers/{tier}")
        return MemoryTierStats.model_validate(data)

    async def snapshot(self) -> dict[str, Any]:
        """Take a manual memory snapshot."""
        return await self._client._post("/api/v1/memory/snapshot", {})


class SelectionAPI:
    """API for agent selection plugins."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def list_plugins(self) -> SelectionPlugins:
        """List available selection plugins."""
        data = await self._client._get("/api/v1/selection/plugins")
        return SelectionPlugins.model_validate(data)

    async def get_defaults(self) -> dict[str, str]:
        """Get default plugin configuration."""
        return await self._client._get("/api/v1/selection/defaults")

    async def score_agents(
        self,
        task_description: str,
        *,
        primary_domain: str | None = None,
        scorer: str | None = None,
    ) -> list[AgentScore]:
        """Score agents for a task."""
        request = ScoreAgentsRequest(
            task_description=task_description,
            primary_domain=primary_domain,
            scorer=scorer,
        )
        data = await self._client._post("/api/v1/selection/score", request.model_dump())
        return [AgentScore.model_validate(a) for a in data.get("agents", [])]

    async def select_team(
        self,
        task_description: str,
        *,
        min_agents: int = 2,
        max_agents: int = 5,
        diversity_preference: float = 0.5,
        quality_priority: float = 0.5,
        scorer: str | None = None,
        team_selector: str | None = None,
        role_assigner: str | None = None,
    ) -> TeamSelection:
        """Select an optimal team for a task."""
        request = SelectTeamRequest(
            task_description=task_description,
            min_agents=min_agents,
            max_agents=max_agents,
            diversity_preference=diversity_preference,
            quality_priority=quality_priority,
            scorer=scorer,
            team_selector=team_selector,
            role_assigner=role_assigner,
        )
        data = await self._client._post("/api/v1/selection/team", request.model_dump())
        return TeamSelection.model_validate(data)


class ReplaysAPI:
    """API for replay management."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def list(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """List replays."""
        data = await self._client._get(
            "/api/v1/replays", params={"limit": limit, "offset": offset}
        )
        return data.get("replays", [])

    async def get(self, replay_id: str) -> dict[str, Any]:
        """Get a replay by ID."""
        return await self._client._get(f"/api/v1/replays/{replay_id}")

    async def export(self, replay_id: str, format: str = "json") -> bytes:
        """Export a replay."""
        return await self._client._get_raw(
            f"/api/v1/replays/{replay_id}/export", params={"format": format}
        )

    async def delete(self, replay_id: str) -> None:
        """Delete a replay."""
        await self._client._delete(f"/api/v1/replays/{replay_id}")


class AragoraClient:
    """
    Client for the Aragora API.

    Example:
        >>> client = AragoraClient("http://localhost:8080")
        >>> debate = await client.debates.run(task="Should we use microservices?")
        >>> print(debate.consensus.conclusion)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the Aragora client.

        Args:
            base_url: Base URL of the Aragora server.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
            headers: Optional additional headers.
        """
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

        default_headers = {"User-Agent": "aragora-client-python/2.0.0"}
        if api_key:
            default_headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            default_headers.update(headers)

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=default_headers,
            timeout=timeout,
        )

        # Initialize API namespaces
        self.debates = DebatesAPI(self)
        self.graph_debates = GraphDebatesAPI(self)
        self.matrix_debates = MatrixDebatesAPI(self)
        self.agents = AgentsAPI(self)
        self.verification = VerificationAPI(self)
        self.gauntlet = GauntletAPI(self)
        self.memory = MemoryAPI(self)
        self.selection = SelectionAPI(self)
        self.replays = ReplaysAPI(self)
        self.control_plane = ControlPlaneAPI(self)
        # Enterprise APIs
        self.auth = AuthAPI(self)
        self.tenants = TenancyAPI(self)
        self.rbac = RBACAPI(self)
        self.tournaments = TournamentsAPI(self)
        self.audit = AuditAPI(self)
        self.onboarding = OnboardingAPI(self)

    async def __aenter__(self) -> AragoraClient:
        """Enter async context."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context."""
        await self.close()

    async def close(self) -> None:
        """Close the client."""
        await self._client.aclose()

    async def health(self) -> HealthStatus:
        """Get server health status."""
        data = await self._get("/api/v1/health")
        return HealthStatus.model_validate(data)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Make an HTTP request."""
        try:
            response = await self._client.request(
                method, path, params=params, json=json
            )
        except httpx.ConnectError as e:
            raise AragoraConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise AragoraTimeoutError(str(e)) from e

        if response.status_code == 401:
            raise AragoraAuthenticationError()
        if response.status_code == 404:
            raise AragoraNotFoundError("Resource", path)
        if response.status_code == 400:
            data = response.json() if response.content else {}
            raise AragoraValidationError(
                data.get("error", "Validation error"),
                details=data.get("details"),
            )
        if response.status_code >= 400:
            data = response.json() if response.content else {}
            raise AragoraError(
                data.get("error", f"Request failed with status {response.status_code}"),
                code=data.get("code"),
                status=response.status_code,
                details=data.get("details"),
            )

        return response

    async def _get(
        self, path: str, *, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make a GET request."""
        response = await self._request("GET", path, params=params)
        return response.json()

    async def _get_raw(
        self, path: str, *, params: dict[str, Any] | None = None
    ) -> bytes:
        """Make a GET request and return raw bytes."""
        response = await self._request("GET", path, params=params)
        return response.content

    async def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make a POST request."""
        response = await self._request("POST", path, json=data)
        return response.json()

    async def _patch(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make a PATCH request."""
        response = await self._request("PATCH", path, json=data)
        return response.json()

    async def _put(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make a PUT request."""
        response = await self._request("PUT", path, json=data)
        return response.json()

    async def _delete(self, path: str) -> None:
        """Make a DELETE request."""
        await self._request("DELETE", path)

    async def _delete_with_body(self, path: str, data: dict[str, Any]) -> None:
        """Make a DELETE request with a body."""
        await self._request("DELETE", path, json=data)
