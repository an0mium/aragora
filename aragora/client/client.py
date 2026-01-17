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

    # With retry and rate limiting
    from aragora.client import AragoraClient, RetryConfig
    client = AragoraClient(
        base_url="http://localhost:8080",
        retry_config=RetryConfig(max_retries=3, backoff_factor=0.5),
        rate_limit_rps=10,
    )

    # Batch fetching
    debates = await client.debates.batch_get_async(["id1", "id2", "id3"])

    # Pagination
    async for debate in client.debates.iterate_async(status="completed"):
        print(debate.task)
"""

from __future__ import annotations

import json
import logging
import time as time_module
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, List, NoReturn, Optional
from urllib.parse import urljoin

# Import from refactored modules
from .errors import (
    AragoraAPIError,
    AuthenticationError,
    NotFoundError,
    QuotaExceededError,
    RateLimitError,
    ValidationError,
)
from .transport import RateLimiter, RetryConfig
from .resources import AgentsAPI, AuditAPI, DocumentsAPI, LeaderboardAPI, MemoryAPI, VerificationAPI
from .models import (
    ConsensusType,
    Debate,
    DebateCreateRequest,
    DebateCreateResponse,
    DebateStatus,
    GauntletReceipt,
    GauntletRunRequest,
    GauntletRunResponse,
    # Graph debates
    GraphDebate,
    GraphDebateBranch,
    GraphDebateCreateRequest,
    GraphDebateCreateResponse,
    HealthCheck,
    MatrixConclusion,
    # Matrix debates
    MatrixDebate,
    MatrixDebateCreateRequest,
    MatrixDebateCreateResponse,
    MatrixScenario,
    # Memory analytics
    Replay,
    ReplaySummary,
)

if TYPE_CHECKING:
    import aiohttp

logger = logging.getLogger(__name__)


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
    ) -> List[Debate]:
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
    ) -> List[Debate]:
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
        agents: List[str] | None = None,
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
        agents: List[str] | None = None,
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
        debate_ids: List[str],
    ) -> List[Debate]:
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
        debate_ids: List[str],
    ) -> List[Debate]:
        """
        Async version of compare().

        Args:
            debate_ids: List of debate IDs to compare.

        Returns:
            List of Debate objects.
        """
        import asyncio

        return await asyncio.gather(*[self.get_async(debate_id) for debate_id in debate_ids])

    def batch_get(
        self,
        debate_ids: List[str],
        max_concurrent: int = 10,
    ) -> List[Debate]:
        """
        Batch fetch multiple debates efficiently.

        Fetches debates sequentially in sync mode but allows controlling
        the batch size to avoid overwhelming the server.

        Args:
            debate_ids: List of debate IDs to fetch.
            max_concurrent: Maximum concurrent requests (for pacing).

        Returns:
            List of Debate objects (in same order as input IDs).
        """
        results = []
        for i, debate_id in enumerate(debate_ids):
            results.append(self.get(debate_id))
            # Add small delay every max_concurrent requests
            if (i + 1) % max_concurrent == 0 and i < len(debate_ids) - 1:
                import time

                time.sleep(0.1)
        return results

    async def batch_get_async(
        self,
        debate_ids: List[str],
        max_concurrent: int = 10,
    ) -> List[Debate]:
        """
        Batch fetch multiple debates with concurrency control.

        Uses asyncio.Semaphore to limit concurrent requests.

        Args:
            debate_ids: List of debate IDs to fetch.
            max_concurrent: Maximum concurrent requests (default: 10).

        Returns:
            List of Debate objects (in same order as input IDs).
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_limit(debate_id: str) -> Debate:
            async with semaphore:
                return await self.get_async(debate_id)

        return await asyncio.gather(*[fetch_with_limit(did) for did in debate_ids])

    def iterate(
        self,
        status: str | None = None,
        page_size: int = 50,
        max_items: int | None = None,
    ) -> Iterator[Debate]:
        """
        Iterate through all debates with automatic pagination.

        Lazily fetches pages as needed, making it memory-efficient
        for large result sets.

        Args:
            status: Optional status filter.
            page_size: Number of items per page (default: 50).
            max_items: Maximum total items to return (default: unlimited).

        Yields:
            Debate objects one at a time.
        """
        offset = 0
        count = 0

        while True:
            debates = self.list(limit=page_size, offset=offset, status=status)
            if not debates:
                break

            for debate in debates:
                yield debate
                count += 1
                if max_items and count >= max_items:
                    return

            if len(debates) < page_size:
                break
            offset += page_size

    async def iterate_async(
        self,
        status: str | None = None,
        page_size: int = 50,
        max_items: int | None = None,
    ) -> AsyncIterator[Debate]:
        """
        Async iterate through all debates with automatic pagination.

        Lazily fetches pages as needed, making it memory-efficient
        for large result sets.

        Args:
            status: Optional status filter.
            page_size: Number of items per page (default: 50).
            max_items: Maximum total items to return (default: unlimited).

        Yields:
            Debate objects one at a time.
        """
        offset = 0
        count = 0

        while True:
            debates = await self.list_async(limit=page_size, offset=offset, status=status)
            if not debates:
                break

            for debate in debates:
                yield debate
                count += 1
                if max_items and count >= max_items:
                    return

            if len(debates) < page_size:
                break
            offset += page_size


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
    ) -> List[ReplaySummary]:
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
        response = self._client._get(f"/api/replays/{replay_id}/export", params={"format": format})
        return response.get("data", "") if isinstance(response, dict) else str(response)

    async def export_async(self, replay_id: str, format: str = "json") -> str:
        """Async version of export()."""
        response = await self._client._get_async(
            f"/api/replays/{replay_id}/export", params={"format": format}
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
        - documents: Document management, batch processing, and auditing
        - audit: Enterprise audit features (presets, workflow, quick audit)
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
        retry_config: RetryConfig | None = None,
        rate_limit_rps: float = 0,
    ):
        """
        Initialize the Aragora client.

        Args:
            base_url: Base URL of the Aragora API server.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
            retry_config: Optional retry configuration for resilient requests.
            rate_limit_rps: Client-side rate limit in requests per second (0 = disabled).
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retry_config = retry_config
        self._rate_limiter = RateLimiter(rate_limit_rps) if rate_limit_rps > 0 else None

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

        # Document management and auditing
        self.documents = DocumentsAPI(self)

        # Enterprise audit features (presets, workflow, quick audit)
        self.audit = AuditAPI(self)

    def _get_headers(self) -> dict[str, str]:
        """Get common request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get(self, path: str, params: dict | None = None) -> dict:
        """Make a synchronous GET request with retry and rate limiting."""
        import urllib.error
        import urllib.parse
        import urllib.request

        url = urljoin(self.base_url, path)
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        last_error: Optional[Exception] = None
        max_attempts = (self.retry_config.max_retries + 1) if self.retry_config else 1

        for attempt in range(max_attempts):
            # Apply rate limiting
            if self._rate_limiter:
                self._rate_limiter.wait()

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                last_error = e
                # Check if we should retry
                if self.retry_config and e.code in self.retry_config.retry_statuses:
                    if attempt < max_attempts - 1:
                        delay = self.retry_config.get_delay(attempt)
                        logger.debug(
                            f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (HTTP {e.code})"
                        )
                        time_module.sleep(delay)
                        continue
                self._handle_http_error(e)

        # Should not reach here, but handle gracefully
        if last_error:
            self._handle_http_error(last_error)
        raise AragoraAPIError("Request failed after retries", "RETRY_EXHAUSTED", 0)

    def _post(self, path: str, data: dict, headers: dict | None = None) -> dict:
        """Make a synchronous POST request with retry and rate limiting."""
        import urllib.error
        import urllib.request

        url = urljoin(self.base_url, path)
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        req = urllib.request.Request(
            url,
            headers=request_headers,
            method="POST",
            data=json.dumps(data).encode(),
        )
        last_error: Optional[Exception] = None
        max_attempts = (self.retry_config.max_retries + 1) if self.retry_config else 1

        for attempt in range(max_attempts):
            # Apply rate limiting
            if self._rate_limiter:
                self._rate_limiter.wait()

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                last_error = e
                # Check if we should retry
                if self.retry_config and e.code in self.retry_config.retry_statuses:
                    if attempt < max_attempts - 1:
                        delay = self.retry_config.get_delay(attempt)
                        logger.debug(
                            f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (HTTP {e.code})"
                        )
                        time_module.sleep(delay)
                        continue
                self._handle_http_error(e)

        # Should not reach here, but handle gracefully
        if last_error:
            self._handle_http_error(last_error)
        raise AragoraAPIError("Request failed after retries", "RETRY_EXHAUSTED", 0)

    def _delete(self, path: str, params: dict | None = None) -> dict:
        """Make a synchronous DELETE request with retry and rate limiting."""
        import urllib.error
        import urllib.parse
        import urllib.request

        url = urljoin(self.base_url, path)
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(url, headers=self._get_headers(), method="DELETE")
        last_error: Optional[Exception] = None
        max_attempts = (self.retry_config.max_retries + 1) if self.retry_config else 1

        for attempt in range(max_attempts):
            # Apply rate limiting
            if self._rate_limiter:
                self._rate_limiter.wait()

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                last_error = e
                # Check if we should retry
                if self.retry_config and e.code in self.retry_config.retry_statuses:
                    if attempt < max_attempts - 1:
                        delay = self.retry_config.get_delay(attempt)
                        logger.debug(
                            f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (HTTP {e.code})"
                        )
                        time_module.sleep(delay)
                        continue
                self._handle_http_error(e)

        # Should not reach here, but handle gracefully
        if last_error:
            self._handle_http_error(last_error)
        raise AragoraAPIError("Request failed after retries", "RETRY_EXHAUSTED", 0)

    async def _delete_async(self, path: str, params: dict | None = None) -> dict:
        """Make an asynchronous DELETE request with retry and rate limiting."""
        import asyncio

        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession()

        url = urljoin(self.base_url, path)
        last_error: Optional[Exception] = None
        max_attempts = (self.retry_config.max_retries + 1) if self.retry_config else 1

        for attempt in range(max_attempts):
            # Apply rate limiting
            if self._rate_limiter:
                await self._rate_limiter.wait_async()

            try:
                async with self._session.delete(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.json()
                        # Check if we should retry
                        if self.retry_config and resp.status in self.retry_config.retry_statuses:
                            if attempt < max_attempts - 1:
                                delay = self.retry_config.get_delay(attempt)
                                logger.debug(
                                    f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (HTTP {resp.status})"
                                )
                                await asyncio.sleep(delay)
                                continue
                        raise AragoraAPIError(
                            body.get("error", "Unknown error"),
                            body.get("code", "HTTP_ERROR"),
                            resp.status,
                        )
                    return await resp.json()
            except aiohttp.ClientError as e:
                last_error = e
                if self.retry_config and attempt < max_attempts - 1:
                    delay = self.retry_config.get_delay(attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (connection error)"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise AragoraAPIError(str(e), "CONNECTION_ERROR", 0)

        # Should not reach here
        raise AragoraAPIError(
            str(last_error) if last_error else "Unknown error", "RETRY_EXHAUSTED", 0
        )

    def _patch(self, path: str, data: dict, headers: dict | None = None) -> dict:
        """Make a synchronous PATCH request with retry and rate limiting."""
        import urllib.error
        import urllib.request

        url = urljoin(self.base_url, path)
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        req = urllib.request.Request(
            url,
            headers=request_headers,
            method="PATCH",
            data=json.dumps(data).encode(),
        )
        last_error: Optional[Exception] = None
        max_attempts = (self.retry_config.max_retries + 1) if self.retry_config else 1

        for attempt in range(max_attempts):
            # Apply rate limiting
            if self._rate_limiter:
                self._rate_limiter.wait()

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                last_error = e
                # Check if we should retry
                if self.retry_config and e.code in self.retry_config.retry_statuses:
                    if attempt < max_attempts - 1:
                        delay = self.retry_config.get_delay(attempt)
                        logger.debug(
                            f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (HTTP {e.code})"
                        )
                        time_module.sleep(delay)
                        continue
                self._handle_http_error(e)

        # Should not reach here, but handle gracefully
        if last_error:
            self._handle_http_error(last_error)
        raise AragoraAPIError("Request failed after retries", "RETRY_EXHAUSTED", 0)

    async def _patch_async(self, path: str, data: dict, headers: dict | None = None) -> dict:
        """Make an asynchronous PATCH request with retry and rate limiting."""
        import asyncio

        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession()

        url = urljoin(self.base_url, path)
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        last_error: Optional[Exception] = None
        max_attempts = (self.retry_config.max_retries + 1) if self.retry_config else 1

        for attempt in range(max_attempts):
            # Apply rate limiting
            if self._rate_limiter:
                await self._rate_limiter.wait_async()

            try:
                async with self._session.patch(
                    url,
                    headers=request_headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.json()
                        # Check if we should retry
                        if self.retry_config and resp.status in self.retry_config.retry_statuses:
                            if attempt < max_attempts - 1:
                                delay = self.retry_config.get_delay(attempt)
                                logger.debug(
                                    f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (HTTP {resp.status})"
                                )
                                await asyncio.sleep(delay)
                                continue
                        raise AragoraAPIError(
                            body.get("error", "Unknown error"),
                            body.get("code", "HTTP_ERROR"),
                            resp.status,
                        )
                    return await resp.json()
            except aiohttp.ClientError as e:
                last_error = e
                if self.retry_config and attempt < max_attempts - 1:
                    delay = self.retry_config.get_delay(attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (connection error)"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise AragoraAPIError(str(e), "CONNECTION_ERROR", 0)

        # Should not reach here
        raise AragoraAPIError(
            str(last_error) if last_error else "Unknown error", "RETRY_EXHAUSTED", 0
        )

    def _handle_http_error(self, e: Any) -> NoReturn:
        """Handle HTTP errors with specific error classes."""
        try:
            body = json.loads(e.read().decode())
            error_msg = body.get("error", str(e))
            error_code = body.get("code", "HTTP_ERROR")
        except Exception as parse_err:
            # Failed to parse error response body - use raw error
            logger.debug(f"Could not parse HTTP error body: {parse_err}")
            error_msg = str(e)
            error_code = "HTTP_ERROR"

        # Map HTTP status codes to specific error classes
        status_code = getattr(e, "code", 500)
        if status_code == 401:
            raise AuthenticationError(error_msg)
        elif status_code == 402:
            raise QuotaExceededError(error_msg)
        elif status_code == 404:
            raise NotFoundError(error_msg)
        elif status_code == 429:
            raise RateLimitError(error_msg)
        elif status_code == 400:
            raise ValidationError(error_msg)
        else:
            raise AragoraAPIError(error_msg, error_code, status_code)

    async def _get_async(self, path: str, params: dict | None = None) -> dict:
        """Make an asynchronous GET request with retry and rate limiting."""
        import asyncio

        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession()

        url = urljoin(self.base_url, path)
        last_error: Optional[Exception] = None
        max_attempts = (self.retry_config.max_retries + 1) if self.retry_config else 1

        for attempt in range(max_attempts):
            # Apply rate limiting
            if self._rate_limiter:
                await self._rate_limiter.wait_async()

            try:
                async with self._session.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.json()
                        # Check if we should retry
                        if self.retry_config and resp.status in self.retry_config.retry_statuses:
                            if attempt < max_attempts - 1:
                                delay = self.retry_config.get_delay(attempt)
                                logger.debug(
                                    f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (HTTP {resp.status})"
                                )
                                await asyncio.sleep(delay)
                                continue
                        raise AragoraAPIError(
                            body.get("error", "Unknown error"),
                            body.get("code", "HTTP_ERROR"),
                            resp.status,
                        )
                    return await resp.json()
            except aiohttp.ClientError as e:
                last_error = e
                if self.retry_config and attempt < max_attempts - 1:
                    delay = self.retry_config.get_delay(attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (connection error)"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise AragoraAPIError(str(e), "CONNECTION_ERROR", 0)

        # Should not reach here
        raise AragoraAPIError(
            str(last_error) if last_error else "Unknown error", "RETRY_EXHAUSTED", 0
        )

    async def _post_async(self, path: str, data: dict, headers: dict | None = None) -> dict:
        """Make an asynchronous POST request with retry and rate limiting."""
        import asyncio

        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession()

        url = urljoin(self.base_url, path)
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        last_error: Optional[Exception] = None
        max_attempts = (self.retry_config.max_retries + 1) if self.retry_config else 1

        for attempt in range(max_attempts):
            # Apply rate limiting
            if self._rate_limiter:
                await self._rate_limiter.wait_async()

            try:
                async with self._session.post(
                    url,
                    headers=request_headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status >= 400:
                        body = await resp.json()
                        # Check if we should retry
                        if self.retry_config and resp.status in self.retry_config.retry_statuses:
                            if attempt < max_attempts - 1:
                                delay = self.retry_config.get_delay(attempt)
                                logger.debug(
                                    f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (HTTP {resp.status})"
                                )
                                await asyncio.sleep(delay)
                                continue
                        raise AragoraAPIError(
                            body.get("error", "Unknown error"),
                            body.get("code", "HTTP_ERROR"),
                            resp.status,
                        )
                    return await resp.json()
            except aiohttp.ClientError as e:
                last_error = e
                if self.retry_config and attempt < max_attempts - 1:
                    delay = self.retry_config.get_delay(attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{max_attempts} after {delay:.2f}s (connection error)"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise AragoraAPIError(str(e), "CONNECTION_ERROR", 0)

        # Should not reach here
        raise AragoraAPIError(
            str(last_error) if last_error else "Unknown error", "RETRY_EXHAUSTED", 0
        )

    def health(self) -> HealthCheck:
        """Check API health."""
        response = self._get("/api/health")
        return HealthCheck(**response)

    async def health_async(self) -> HealthCheck:
        """Async version of health()."""
        response = await self._get_async("/api/health")
        return HealthCheck(**response)

    def __enter__(self) -> "AragoraClient":
        """Sync context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Sync context manager exit."""
        self.close()

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
        self._sync_client = None
