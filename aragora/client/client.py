"""
Aragora Python SDK Client.

Provides a type-safe interface for interacting with the Aragora API.

Usage:
    from aragora.client import AragoraClient

    # Synchronous usage
    client = AragoraClient(base_url="http://localhost:8080")
    debate = client.debates.create(task="Should we use microservices?")
    result = client.debates.get(debate.debate_id)

    # Async usage
    async with AragoraClient(base_url="http://localhost:8080") as client:
        debate = await client.debates.create_async(task="...")
        result = await client.debates.get_async(debate.debate_id)
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, NoReturn, Optional, TYPE_CHECKING
from urllib.parse import urljoin

from .models import (
    Debate,
    DebateCreateRequest,
    DebateCreateResponse,
    DebateStatus,
    ConsensusType,
    AgentProfile,
    LeaderboardEntry,
    GauntletReceipt,
    GauntletRunRequest,
    GauntletRunResponse,
    HealthCheck,
    APIError,
    # Graph debates
    GraphDebate,
    GraphDebateCreateRequest,
    GraphDebateCreateResponse,
    GraphDebateBranch,
    # Matrix debates
    MatrixDebate,
    MatrixDebateCreateRequest,
    MatrixDebateCreateResponse,
    MatrixScenario,
    MatrixConclusion,
    # Verification
    VerifyClaimRequest,
    VerifyClaimResponse,
    VerifyStatusResponse,
    # Memory analytics
    MemoryAnalyticsResponse,
    MemorySnapshotResponse,
    # Replays
    Replay,
    ReplaySummary,
)

if TYPE_CHECKING:
    import aiohttp
    import httpx

logger = logging.getLogger(__name__)


class AragoraAPIError(Exception):
    """Exception raised for API errors."""

    def __init__(self, message: str, code: str = "UNKNOWN", status_code: int = 500):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class DebatesAPI:
    """API interface for debates."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def create(
        self,
        task: str,
        agents: list[str] | None = None,
        rounds: int = 3,
        consensus: str = "majority",
        context: str | None = None,
        **kwargs: Any,
    ) -> DebateCreateResponse:
        """
        Create and start a new debate.

        Args:
            task: The question or topic to debate.
            agents: List of agent IDs to participate (default: anthropic-api, openai-api).
            rounds: Number of debate rounds (default: 3).
            consensus: Consensus mechanism (unanimous, majority, supermajority, hybrid).
            context: Additional context for the debate.

        Returns:
            DebateCreateResponse with debate_id and status.
        """
        request = DebateCreateRequest(
            task=task,
            agents=agents or ["anthropic-api", "openai-api"],
            rounds=rounds,
            consensus=ConsensusType(consensus),
            context=context,
            metadata=kwargs,
        )

        response = self._client._post("/api/debates", request.model_dump())
        return DebateCreateResponse(**response)

    async def create_async(
        self,
        task: str,
        agents: list[str] | None = None,
        rounds: int = 3,
        consensus: str = "majority",
        context: str | None = None,
        **kwargs: Any,
    ) -> DebateCreateResponse:
        """Async version of create()."""
        request = DebateCreateRequest(
            task=task,
            agents=agents or ["anthropic-api", "openai-api"],
            rounds=rounds,
            consensus=ConsensusType(consensus),
            context=context,
            metadata=kwargs,
        )

        response = await self._client._post_async("/api/debates", request.model_dump())
        return DebateCreateResponse(**response)

    def get(self, debate_id: str) -> Debate:
        """
        Get debate details by ID.

        Args:
            debate_id: The debate ID.

        Returns:
            Debate with full details including rounds and consensus.
        """
        response = self._client._get(f"/api/debates/{debate_id}")
        return Debate(**response)

    async def get_async(self, debate_id: str) -> Debate:
        """Async version of get()."""
        response = await self._client._get_async(f"/api/debates/{debate_id}")
        return Debate(**response)

    def list(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
    ) -> list[Debate]:
        """
        List recent debates.

        Args:
            limit: Maximum number of debates to return.
            offset: Number of debates to skip.
            status: Filter by status (pending, running, completed, failed).

        Returns:
            List of Debate objects.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        response = self._client._get("/api/debates", params=params)
        debates = response.get("debates", response) if isinstance(response, dict) else response
        return [Debate(**d) for d in debates]

    async def list_async(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
    ) -> list[Debate]:
        """Async version of list()."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        response = await self._client._get_async("/api/debates", params=params)
        debates = response.get("debates", response) if isinstance(response, dict) else response
        return [Debate(**d) for d in debates]

    def run(
        self,
        task: str,
        agents: list[str] | None = None,
        rounds: int = 3,
        consensus: str = "majority",
        timeout: int = 600,
        **kwargs: Any,
    ) -> Debate:
        """
        Create a debate and wait for completion.

        This is a convenience method that creates a debate and polls
        until it completes or times out.

        Args:
            task: The question or topic to debate.
            agents: List of agent IDs to participate.
            rounds: Number of debate rounds.
            consensus: Consensus mechanism.
            timeout: Maximum wait time in seconds.

        Returns:
            Completed Debate with full results.
        """
        import time

        response = self.create(task, agents, rounds, consensus, **kwargs)
        debate_id = response.debate_id

        start = time.time()
        while time.time() - start < timeout:
            debate = self.get(debate_id)
            if debate.status in (DebateStatus.COMPLETED, DebateStatus.FAILED):
                return debate
            time.sleep(2)

        raise TimeoutError(f"Debate {debate_id} did not complete within {timeout}s")

    async def run_async(
        self,
        task: str,
        agents: list[str] | None = None,
        rounds: int = 3,
        consensus: str = "majority",
        timeout: int = 600,
        **kwargs: Any,
    ) -> Debate:
        """Async version of run()."""
        import asyncio

        response = await self.create_async(task, agents, rounds, consensus, **kwargs)
        debate_id = response.debate_id

        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            debate = await self.get_async(debate_id)
            if debate.status in (DebateStatus.COMPLETED, DebateStatus.FAILED):
                return debate
            await asyncio.sleep(2)

        raise TimeoutError(f"Debate {debate_id} did not complete within {timeout}s")

    def wait_for_completion(
        self,
        debate_id: str,
        timeout: int = 600,
        poll_interval: float = 2.0,
    ) -> Debate:
        """
        Wait for an existing debate to complete.

        Args:
            debate_id: The debate ID to wait for.
            timeout: Maximum wait time in seconds (default: 600).
            poll_interval: Time between status checks in seconds (default: 2.0).

        Returns:
            Completed Debate with full results.

        Raises:
            TimeoutError: If debate doesn't complete within timeout.
        """
        import time

        start = time.time()
        while time.time() - start < timeout:
            debate = self.get(debate_id)
            if debate.status in (DebateStatus.COMPLETED, DebateStatus.FAILED):
                return debate
            time.sleep(poll_interval)

        raise TimeoutError(f"Debate {debate_id} did not complete within {timeout}s")

    async def wait_for_completion_async(
        self,
        debate_id: str,
        timeout: int = 600,
        poll_interval: float = 2.0,
    ) -> Debate:
        """
        Async version of wait_for_completion().

        Args:
            debate_id: The debate ID to wait for.
            timeout: Maximum wait time in seconds (default: 600).
            poll_interval: Time between status checks in seconds (default: 2.0).

        Returns:
            Completed Debate with full results.

        Raises:
            TimeoutError: If debate doesn't complete within timeout.
        """
        import asyncio

        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            debate = await self.get_async(debate_id)
            if debate.status in (DebateStatus.COMPLETED, DebateStatus.FAILED):
                return debate
            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"Debate {debate_id} did not complete within {timeout}s")

    def compare(
        self,
        debate_ids: list[str],
    ) -> list[Debate]:
        """
        Get multiple debates for side-by-side comparison.

        Args:
            debate_ids: List of debate IDs to compare.

        Returns:
            List of Debate objects.
        """
        return [self.get(debate_id) for debate_id in debate_ids]

    async def compare_async(
        self,
        debate_ids: list[str],
    ) -> list[Debate]:
        """
        Async version of compare().

        Args:
            debate_ids: List of debate IDs to compare.

        Returns:
            List of Debate objects.
        """
        import asyncio

        return await asyncio.gather(
            *[self.get_async(debate_id) for debate_id in debate_ids]
        )


class AgentsAPI:
    """API interface for agents."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def list(self) -> list[AgentProfile]:
        """
        List all available agents.

        Returns:
            List of AgentProfile objects.
        """
        response = self._client._get("/api/agents")
        agents = response.get("agents", response) if isinstance(response, dict) else response
        return [AgentProfile(**a) for a in agents]

    async def list_async(self) -> list[AgentProfile]:
        """Async version of list()."""
        response = await self._client._get_async("/api/agents")
        agents = response.get("agents", response) if isinstance(response, dict) else response
        return [AgentProfile(**a) for a in agents]

    def get(self, agent_id: str) -> AgentProfile:
        """
        Get agent profile by ID.

        Args:
            agent_id: The agent ID.

        Returns:
            AgentProfile with details.
        """
        response = self._client._get(f"/api/agent/{agent_id}")
        return AgentProfile(**response)

    async def get_async(self, agent_id: str) -> AgentProfile:
        """Async version of get()."""
        response = await self._client._get_async(f"/api/agent/{agent_id}")
        return AgentProfile(**response)


class LeaderboardAPI:
    """API interface for leaderboard."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def get(self, limit: int = 10) -> list[LeaderboardEntry]:
        """
        Get leaderboard rankings.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of LeaderboardEntry objects.
        """
        response = self._client._get("/api/leaderboard", params={"limit": limit})
        rankings = response.get("rankings", response) if isinstance(response, dict) else response
        return [LeaderboardEntry(**r) for r in rankings]

    async def get_async(self, limit: int = 10) -> list[LeaderboardEntry]:
        """Async version of get()."""
        response = await self._client._get_async("/api/leaderboard", params={"limit": limit})
        rankings = response.get("rankings", response) if isinstance(response, dict) else response
        return [LeaderboardEntry(**r) for r in rankings]


class GauntletAPI:
    """API interface for gauntlet (adversarial validation)."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def run(
        self,
        input_content: str,
        input_type: str = "text",
        persona: str = "security",
        profile: str = "default",
    ) -> GauntletRunResponse:
        """
        Start a gauntlet analysis run.

        Args:
            input_content: Content to analyze.
            input_type: Type of content (text, policy, code).
            persona: Analysis persona (security, gdpr, hipaa, etc).
            profile: Analysis depth (quick, default, thorough).

        Returns:
            GauntletRunResponse with gauntlet_id.
        """
        request = GauntletRunRequest(
            input_content=input_content,
            input_type=input_type,
            persona=persona,
            profile=profile,
        )

        response = self._client._post("/api/gauntlet/run", request.model_dump())
        return GauntletRunResponse(**response)

    async def run_async(
        self,
        input_content: str,
        input_type: str = "text",
        persona: str = "security",
        profile: str = "default",
    ) -> GauntletRunResponse:
        """Async version of run()."""
        request = GauntletRunRequest(
            input_content=input_content,
            input_type=input_type,
            persona=persona,
            profile=profile,
        )

        response = await self._client._post_async("/api/gauntlet/run", request.model_dump())
        return GauntletRunResponse(**response)

    def get_receipt(self, gauntlet_id: str) -> GauntletReceipt:
        """
        Get the decision receipt for a gauntlet run.

        Args:
            gauntlet_id: The gauntlet run ID.

        Returns:
            GauntletReceipt with verdict and findings.
        """
        response = self._client._get(f"/api/gauntlet/{gauntlet_id}/receipt")
        return GauntletReceipt(**response)

    async def get_receipt_async(self, gauntlet_id: str) -> GauntletReceipt:
        """Async version of get_receipt()."""
        response = await self._client._get_async(f"/api/gauntlet/{gauntlet_id}/receipt")
        return GauntletReceipt(**response)

    def run_and_wait(
        self,
        input_content: str,
        input_type: str = "text",
        persona: str = "security",
        profile: str = "default",
        timeout: int = 900,
    ) -> GauntletReceipt:
        """
        Run gauntlet and wait for completion.

        Args:
            input_content: Content to analyze.
            input_type: Type of content.
            persona: Analysis persona.
            profile: Analysis depth.
            timeout: Maximum wait time in seconds.

        Returns:
            GauntletReceipt with full results.
        """
        import time

        response = self.run(input_content, input_type, persona, profile)
        gauntlet_id = response.gauntlet_id

        start = time.time()
        while time.time() - start < timeout:
            try:
                return self.get_receipt(gauntlet_id)
            except AragoraAPIError as e:
                if e.status_code != 404:
                    raise
            time.sleep(5)

        raise TimeoutError(f"Gauntlet {gauntlet_id} did not complete within {timeout}s")


class GraphDebatesAPI:
    """API interface for graph-structured debates with branching."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def create(
        self,
        task: str,
        agents: list[str] | None = None,
        max_rounds: int = 5,
        branch_threshold: float = 0.5,
        max_branches: int = 5,
    ) -> GraphDebateCreateResponse:
        """
        Create and start a graph-structured debate.

        Graph debates allow for automatic branching when agents
        identify fundamentally different approaches.

        Args:
            task: The question or topic to debate.
            agents: List of agent IDs to participate.
            max_rounds: Maximum rounds per branch (1-20).
            branch_threshold: Divergence threshold for branching (0-1).
            max_branches: Maximum number of branches allowed.

        Returns:
            GraphDebateCreateResponse with debate_id.
        """
        request = GraphDebateCreateRequest(
            task=task,
            agents=agents or ["anthropic-api", "openai-api"],
            max_rounds=max_rounds,
            branch_threshold=branch_threshold,
            max_branches=max_branches,
        )

        response = self._client._post("/api/debates/graph", request.model_dump())
        return GraphDebateCreateResponse(**response)

    async def create_async(
        self,
        task: str,
        agents: list[str] | None = None,
        max_rounds: int = 5,
        branch_threshold: float = 0.5,
        max_branches: int = 5,
    ) -> GraphDebateCreateResponse:
        """Async version of create()."""
        request = GraphDebateCreateRequest(
            task=task,
            agents=agents or ["anthropic-api", "openai-api"],
            max_rounds=max_rounds,
            branch_threshold=branch_threshold,
            max_branches=max_branches,
        )

        response = await self._client._post_async("/api/debates/graph", request.model_dump())
        return GraphDebateCreateResponse(**response)

    def get(self, debate_id: str) -> GraphDebate:
        """
        Get graph debate details by ID.

        Args:
            debate_id: The graph debate ID.

        Returns:
            GraphDebate with full details including branches.
        """
        response = self._client._get(f"/api/debates/graph/{debate_id}")
        return GraphDebate(**response)

    async def get_async(self, debate_id: str) -> GraphDebate:
        """Async version of get()."""
        response = await self._client._get_async(f"/api/debates/graph/{debate_id}")
        return GraphDebate(**response)

    def get_branches(self, debate_id: str) -> list[GraphDebateBranch]:
        """
        Get all branches for a graph debate.

        Args:
            debate_id: The graph debate ID.

        Returns:
            List of GraphDebateBranch objects.
        """
        response = self._client._get(f"/api/debates/graph/{debate_id}/branches")
        branches = response.get("branches", response) if isinstance(response, dict) else response
        return [GraphDebateBranch(**b) for b in branches]

    async def get_branches_async(self, debate_id: str) -> list[GraphDebateBranch]:
        """Async version of get_branches()."""
        response = await self._client._get_async(f"/api/debates/graph/{debate_id}/branches")
        branches = response.get("branches", response) if isinstance(response, dict) else response
        return [GraphDebateBranch(**b) for b in branches]


class MatrixDebatesAPI:
    """API interface for matrix debates with parallel scenarios."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def create(
        self,
        task: str,
        agents: list[str] | None = None,
        scenarios: list[dict] | None = None,
        max_rounds: int = 3,
    ) -> MatrixDebateCreateResponse:
        """
        Create and start a matrix debate with parallel scenarios.

        Matrix debates run the same debate across different scenarios
        to identify universal vs conditional conclusions.

        Args:
            task: The base question or topic to debate.
            agents: List of agent IDs to participate.
            scenarios: List of scenario configurations.
                Each scenario can have: name, parameters, constraints, is_baseline.
            max_rounds: Maximum rounds per scenario (1-10).

        Returns:
            MatrixDebateCreateResponse with matrix_id.
        """
        scenario_models = []
        if scenarios:
            for s in scenarios:
                scenario_models.append(MatrixScenario(**s))

        request = MatrixDebateCreateRequest(
            task=task,
            agents=agents or ["anthropic-api", "openai-api"],
            scenarios=scenario_models,
            max_rounds=max_rounds,
        )

        response = self._client._post("/api/debates/matrix", request.model_dump())
        return MatrixDebateCreateResponse(**response)

    async def create_async(
        self,
        task: str,
        agents: list[str] | None = None,
        scenarios: list[dict] | None = None,
        max_rounds: int = 3,
    ) -> MatrixDebateCreateResponse:
        """Async version of create()."""
        scenario_models = []
        if scenarios:
            for s in scenarios:
                scenario_models.append(MatrixScenario(**s))

        request = MatrixDebateCreateRequest(
            task=task,
            agents=agents or ["anthropic-api", "openai-api"],
            scenarios=scenario_models,
            max_rounds=max_rounds,
        )

        response = await self._client._post_async("/api/debates/matrix", request.model_dump())
        return MatrixDebateCreateResponse(**response)

    def get(self, matrix_id: str) -> MatrixDebate:
        """
        Get matrix debate details by ID.

        Args:
            matrix_id: The matrix debate ID.

        Returns:
            MatrixDebate with full details including scenario results.
        """
        response = self._client._get(f"/api/debates/matrix/{matrix_id}")
        return MatrixDebate(**response)

    async def get_async(self, matrix_id: str) -> MatrixDebate:
        """Async version of get()."""
        response = await self._client._get_async(f"/api/debates/matrix/{matrix_id}")
        return MatrixDebate(**response)

    def get_conclusions(self, matrix_id: str) -> MatrixConclusion:
        """
        Get universal and conditional conclusions from a matrix debate.

        Args:
            matrix_id: The matrix debate ID.

        Returns:
            MatrixConclusion with universal, conditional, and contradictory findings.
        """
        response = self._client._get(f"/api/debates/matrix/{matrix_id}/conclusions")
        return MatrixConclusion(**response)

    async def get_conclusions_async(self, matrix_id: str) -> MatrixConclusion:
        """Async version of get_conclusions()."""
        response = await self._client._get_async(f"/api/debates/matrix/{matrix_id}/conclusions")
        return MatrixConclusion(**response)


class VerificationAPI:
    """API interface for formal verification of claims."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def verify(
        self,
        claim: str,
        context: str | None = None,
        backend: str = "z3",
        timeout: int = 30,
    ) -> VerifyClaimResponse:
        """
        Verify a claim using formal methods.

        Args:
            claim: The claim to verify in natural language.
            context: Optional context for the claim.
            backend: Verification backend (z3, lean, coq).
            timeout: Verification timeout in seconds.

        Returns:
            VerifyClaimResponse with status, proof, or counterexample.
        """
        request = VerifyClaimRequest(
            claim=claim,
            context=context,
            backend=backend,
            timeout=timeout,
        )

        response = self._client._post("/api/verify/claim", request.model_dump())
        return VerifyClaimResponse(**response)

    async def verify_async(
        self,
        claim: str,
        context: str | None = None,
        backend: str = "z3",
        timeout: int = 30,
    ) -> VerifyClaimResponse:
        """Async version of verify()."""
        request = VerifyClaimRequest(
            claim=claim,
            context=context,
            backend=backend,
            timeout=timeout,
        )

        response = await self._client._post_async("/api/verify/claim", request.model_dump())
        return VerifyClaimResponse(**response)

    def status(self) -> VerifyStatusResponse:
        """
        Check verification backend availability.

        Returns:
            VerifyStatusResponse with available backends.
        """
        response = self._client._get("/api/verify/status")
        return VerifyStatusResponse(**response)

    async def status_async(self) -> VerifyStatusResponse:
        """Async version of status()."""
        response = await self._client._get_async("/api/verify/status")
        return VerifyStatusResponse(**response)


class MemoryAPI:
    """API interface for memory tier analytics."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def analytics(self, days: int = 30) -> MemoryAnalyticsResponse:
        """
        Get comprehensive memory tier analytics.

        Args:
            days: Number of days to analyze (1-365).

        Returns:
            MemoryAnalyticsResponse with tier stats and recommendations.
        """
        response = self._client._get("/api/memory/analytics", params={"days": days})
        return MemoryAnalyticsResponse(**response)

    async def analytics_async(self, days: int = 30) -> MemoryAnalyticsResponse:
        """Async version of analytics()."""
        response = await self._client._get_async("/api/memory/analytics", params={"days": days})
        return MemoryAnalyticsResponse(**response)

    def tier_stats(self, tier_name: str, days: int = 30) -> dict:
        """
        Get statistics for a specific memory tier.

        Args:
            tier_name: Name of the tier (fast, medium, slow, glacial).
            days: Number of days to analyze.

        Returns:
            Dict with tier-specific statistics.
        """
        response = self._client._get(
            f"/api/memory/analytics/tier/{tier_name}",
            params={"days": days}
        )
        return response

    async def tier_stats_async(self, tier_name: str, days: int = 30) -> dict:
        """Async version of tier_stats()."""
        response = await self._client._get_async(
            f"/api/memory/analytics/tier/{tier_name}",
            params={"days": days}
        )
        return response

    def snapshot(self) -> MemorySnapshotResponse:
        """
        Take a manual memory analytics snapshot.

        Returns:
            MemorySnapshotResponse with snapshot details.
        """
        response = self._client._post("/api/memory/analytics/snapshot", {})
        return MemorySnapshotResponse(**response)

    async def snapshot_async(self) -> MemorySnapshotResponse:
        """Async version of snapshot()."""
        response = await self._client._post_async("/api/memory/analytics/snapshot", {})
        return MemorySnapshotResponse(**response)


class ReplayAPI:
    """API interface for debate replays."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def list(
        self,
        limit: int = 20,
        debate_id: str | None = None,
    ) -> list[ReplaySummary]:
        """
        List available debate replays.

        Args:
            limit: Maximum number of replays to return.
            debate_id: Optional filter by debate ID.

        Returns:
            List of ReplaySummary objects.
        """
        params: dict[str, Any] = {"limit": limit}
        if debate_id:
            params["debate_id"] = debate_id

        response = self._client._get("/api/replays", params=params)
        replays = response.get("replays", response) if isinstance(response, dict) else response
        return [ReplaySummary(**r) for r in replays]

    async def list_async(
        self,
        limit: int = 20,
        debate_id: str | None = None,
    ) -> list[ReplaySummary]:
        """Async version of list()."""
        params: dict[str, Any] = {"limit": limit}
        if debate_id:
            params["debate_id"] = debate_id

        response = await self._client._get_async("/api/replays", params=params)
        replays = response.get("replays", response) if isinstance(response, dict) else response
        return [ReplaySummary(**r) for r in replays]

    def get(self, replay_id: str) -> Replay:
        """
        Get full replay by ID.

        Args:
            replay_id: The replay ID.

        Returns:
            Replay with full event timeline.
        """
        response = self._client._get(f"/api/replays/{replay_id}")
        return Replay(**response)

    async def get_async(self, replay_id: str) -> Replay:
        """Async version of get()."""
        response = await self._client._get_async(f"/api/replays/{replay_id}")
        return Replay(**response)

    def delete(self, replay_id: str) -> bool:
        """
        Delete a replay.

        Args:
            replay_id: The replay ID to delete.

        Returns:
            True if deleted successfully.
        """
        self._client._delete(f"/api/replays/{replay_id}")
        return True

    async def delete_async(self, replay_id: str) -> bool:
        """Async version of delete()."""
        await self._client._delete_async(f"/api/replays/{replay_id}")
        return True

    def export(self, replay_id: str, format: str = "json") -> str:
        """
        Export replay data in specified format.

        Args:
            replay_id: The replay ID.
            format: Export format (json, csv).

        Returns:
            Exported data as string.
        """
        response = self._client._get(
            f"/api/replays/{replay_id}/export",
            params={"format": format}
        )
        return response.get("data", "") if isinstance(response, dict) else str(response)

    async def export_async(self, replay_id: str, format: str = "json") -> str:
        """Async version of export()."""
        response = await self._client._get_async(
            f"/api/replays/{replay_id}/export",
            params={"format": format}
        )
        return response.get("data", "") if isinstance(response, dict) else str(response)


class AragoraClient:
    """
    Aragora API client.

    Provides synchronous and asynchronous access to the Aragora API.

    Available API interfaces:
        - debates: Standard debates (create, get, list, run)
        - graph_debates: Graph-structured debates with branching
        - matrix_debates: Parallel scenario debates
        - verification: Formal claim verification
        - memory: Memory tier analytics
        - agents: Agent discovery and profiles
        - leaderboard: ELO rankings
        - gauntlet: Adversarial validation
        - replays: Debate replay viewing and export

    Usage:
        # Synchronous
        client = AragoraClient(base_url="http://localhost:8080")
        debate = client.debates.run(task="Should we use microservices?")

        # Graph debate with branching
        result = client.graph_debates.create(task="Design a distributed system")

        # Matrix debate with scenarios
        result = client.matrix_debates.create(
            task="Should we adopt microservices?",
            scenarios=[
                {"name": "small team", "parameters": {"team_size": 5}},
                {"name": "large team", "parameters": {"team_size": 50}},
            ]
        )

        # Verify a claim
        result = client.verification.verify(claim="All primes > 2 are odd")

        # Memory analytics
        analytics = client.memory.analytics(days=30)

        # Asynchronous
        async with AragoraClient(base_url="http://localhost:8080") as client:
            debate = await client.debates.run_async(task="...")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout: int = 60,
    ):
        """
        Initialize the Aragora client.

        Args:
            base_url: Base URL of the Aragora API server.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

        # Lazy-loaded HTTP clients
        self._session: Optional["aiohttp.ClientSession"] = None
        self._sync_client: Any = None

        # API interfaces
        self.debates = DebatesAPI(self)
        self.agents = AgentsAPI(self)
        self.leaderboard = LeaderboardAPI(self)
        self.gauntlet = GauntletAPI(self)

        # Extended API interfaces
        self.graph_debates = GraphDebatesAPI(self)
        self.matrix_debates = MatrixDebatesAPI(self)
        self.verification = VerificationAPI(self)
        self.memory = MemoryAPI(self)
        self.replays = ReplayAPI(self)

    def _get_headers(self) -> dict[str, str]:
        """Get common request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get(self, path: str, params: dict | None = None) -> dict:
        """Make a synchronous GET request."""
        import urllib.request
        import urllib.parse
        import urllib.error

        url = urljoin(self.base_url, path)
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            self._handle_http_error(e)

    def _post(self, path: str, data: dict) -> dict:
        """Make a synchronous POST request."""
        import urllib.request
        import urllib.error

        url = urljoin(self.base_url, path)
        req = urllib.request.Request(
            url,
            headers=self._get_headers(),
            method="POST",
            data=json.dumps(data).encode(),
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            self._handle_http_error(e)

    def _delete(self, path: str, params: dict | None = None) -> dict:
        """Make a synchronous DELETE request."""
        import urllib.request
        import urllib.parse
        import urllib.error

        url = urljoin(self.base_url, path)
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(url, headers=self._get_headers(), method="DELETE")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            self._handle_http_error(e)

    async def _delete_async(self, path: str, params: dict | None = None) -> dict:
        """Make an asynchronous DELETE request."""
        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession()

        url = urljoin(self.base_url, path)
        async with self._session.delete(
            url,
            headers=self._get_headers(),
            params=params,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as resp:
            if resp.status >= 400:
                body = await resp.json()
                raise AragoraAPIError(
                    body.get("error", "Unknown error"),
                    body.get("code", "HTTP_ERROR"),
                    resp.status,
                )
            return await resp.json()

    def _handle_http_error(self, e: Any) -> NoReturn:
        """Handle HTTP errors."""
        try:
            body = json.loads(e.read().decode())
            error_msg = body.get("error", str(e))
            error_code = body.get("code", "HTTP_ERROR")
        except Exception as parse_err:
            # Failed to parse error response body - use raw error
            logger.debug(f"Could not parse HTTP error body: {parse_err}")
            error_msg = str(e)
            error_code = "HTTP_ERROR"

        raise AragoraAPIError(error_msg, error_code, e.code)

    async def _get_async(self, path: str, params: dict | None = None) -> dict:
        """Make an asynchronous GET request."""
        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession()

        url = urljoin(self.base_url, path)
        async with self._session.get(
            url,
            headers=self._get_headers(),
            params=params,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as resp:
            if resp.status >= 400:
                body = await resp.json()
                raise AragoraAPIError(
                    body.get("error", "Unknown error"),
                    body.get("code", "HTTP_ERROR"),
                    resp.status,
                )
            return await resp.json()

    async def _post_async(self, path: str, data: dict) -> dict:
        """Make an asynchronous POST request."""
        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession()

        url = urljoin(self.base_url, path)
        async with self._session.post(
            url,
            headers=self._get_headers(),
            json=data,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as resp:
            if resp.status >= 400:
                body = await resp.json()
                raise AragoraAPIError(
                    body.get("error", "Unknown error"),
                    body.get("code", "HTTP_ERROR"),
                    resp.status,
                )
            return await resp.json()

    def health(self) -> HealthCheck:
        """Check API health."""
        response = self._get("/api/health")
        return HealthCheck(**response)

    async def health_async(self) -> HealthCheck:
        """Async version of health()."""
        response = await self._get_async("/api/health")
        return HealthCheck(**response)

    async def __aenter__(self) -> "AragoraClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close_async()

    async def close_async(self) -> None:
        """Close async HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    def close(self) -> None:
        """Close any open resources."""
        # Sync client cleanup if needed
        pass
