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
from typing import Any, Optional, TYPE_CHECKING
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
        params = {"limit": limit, "offset": offset}
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
        params = {"limit": limit, "offset": offset}
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


class AragoraClient:
    """
    Aragora API client.

    Provides synchronous and asynchronous access to the Aragora API.

    Usage:
        # Synchronous
        client = AragoraClient(base_url="http://localhost:8080")
        debate = client.debates.run(task="Should we use microservices?")

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

    def _handle_http_error(self, e: Any) -> None:
        """Handle HTTP errors."""
        try:
            body = json.loads(e.read().decode())
            error_msg = body.get("error", str(e))
            error_code = body.get("code", "HTTP_ERROR")
        except Exception:
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
