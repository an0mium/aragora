"""
Aragora SDK Client.

Async client for interacting with the Aragora API.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncIterator
from urllib.parse import urljoin

import aiohttp

from .exceptions import (
    AragoraError,
    AuthenticationError,
    AuthorizationError,
    ConnectionError,
    NotFoundError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError,
)
from .models import (
    HealthStatus,
    ReviewRequest,
    ReviewResult,
    UsageInfo,
)


class AragoraClient:
    """
    Async client for the Aragora API.

    Usage:
        async with AragoraClient(api_key="ara_...") as client:
            result = await client.review(
                spec="Your design document here...",
                personas=["security", "sox", "performance"],
                rounds=3,
            )
            print(result.consensus)
            print(result.dissenting_opinions)
    """

    DEFAULT_BASE_URL = "https://api.aragora.ai"
    DEFAULT_TIMEOUT = 300  # 5 minutes for long-running reviews

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
    ):
        """
        Initialize the Aragora client.

        Args:
            api_key: API key for authentication (or set ARAGORA_API_KEY env var)
            base_url: Base URL for the API (defaults to https://api.aragora.ai)
            timeout: Request timeout in seconds (default: 300)
        """
        self.api_key = api_key or os.getenv("ARAGORA_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "API key required. Pass api_key or set ARAGORA_API_KEY environment variable."
            )

        self.base_url = base_url or os.getenv("ARAGORA_BASE_URL", self.DEFAULT_BASE_URL)
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "AragoraClient":
        """Enter async context manager."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        await self.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"ApiKey {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "aragora-sdk/0.1.0",
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
        return self._session

    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _url(self, path: str) -> str:
        """Build full URL from path."""
        return urljoin(self.base_url, path)

    async def _handle_response(self, response: aiohttp.ClientResponse) -> dict[str, Any]:
        """Handle API response and raise appropriate exceptions."""
        try:
            data = await response.json()
        except (json.JSONDecodeError, aiohttp.ContentTypeError):
            data = {"error": await response.text()}

        if response.status == 200:
            return data
        elif response.status == 401:
            raise AuthenticationError(data.get("error", "Authentication failed"))
        elif response.status == 403:
            raise AuthorizationError(data.get("error", "Access denied"))
        elif response.status == 404:
            raise NotFoundError(data.get("error", "Resource not found"))
        elif response.status == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                data.get("error", "Rate limit exceeded"),
                retry_after=int(retry_after) if retry_after else None,
            )
        elif response.status == 400:
            raise ValidationError(data.get("error", "Validation failed"))
        elif response.status >= 500:
            raise ServerError(data.get("error", "Server error"))
        else:
            raise AragoraError(data.get("error", "Unknown error"), status_code=response.status)

    async def _request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an API request."""
        session = await self._ensure_session()

        try:
            async with session.request(
                method,
                self._url(path),
                json=data,
                params=params,
            ) as response:
                return await self._handle_response(response)
        except aiohttp.ClientConnectorError as e:
            raise ConnectionError(f"Failed to connect: {e}")
        except asyncio.TimeoutError:
            raise TimeoutError("Request timed out")

    # =========================================================================
    # Health & Info
    # =========================================================================

    async def health(self) -> HealthStatus:
        """Check API health status."""
        data = await self._request("GET", "/api/health")
        return HealthStatus(**data)

    async def get_usage(self) -> UsageInfo:
        """Get current usage information."""
        data = await self._request("GET", "/api/billing/usage")
        return UsageInfo(**data.get("usage", {}))

    # =========================================================================
    # Review API
    # =========================================================================

    async def review(
        self,
        spec: str,
        personas: list[str] | None = None,
        rounds: int = 3,
        task: str | None = None,
        agents: list[str] | None = None,
        include_receipt: bool = True,
    ) -> ReviewResult:
        """
        Review a design specification using multi-agent debate.

        Args:
            spec: The design specification or document to review
            personas: List of personas to use (e.g., ["security", "sox", "hipaa"])
                     Defaults to ["security", "performance"]
            rounds: Number of debate rounds (1-10, default: 3)
            task: Optional task description
            agents: Optional specific agents to use
            include_receipt: Include full decision receipt (default: True)

        Returns:
            ReviewResult with consensus, dissenting opinions, and findings

        Example:
            result = await client.review(
                spec=open("design.md").read(),
                personas=["sox", "pci_dss", "security"],
                rounds=3,
            )
            print(f"Consensus: {result.consensus.status}")
            for dissent in result.dissenting_opinions:
                print(f"Dissent from {dissent.agent}: {dissent.position}")
        """
        request = ReviewRequest(
            spec=spec,
            personas=personas or ["security", "performance"],
            rounds=rounds,
            task=task,
            agents=agents,
            include_receipt=include_receipt,
        )

        data = await self._request(
            "POST",
            "/api/review",
            data=request.model_dump(exclude_none=True),
        )

        return ReviewResult(**data)

    async def review_file(
        self,
        file_path: str,
        personas: list[str] | None = None,
        rounds: int = 3,
        **kwargs,
    ) -> ReviewResult:
        """
        Review a file by reading its contents.

        Args:
            file_path: Path to the file to review
            personas: List of personas to use
            rounds: Number of debate rounds
            **kwargs: Additional arguments passed to review()

        Returns:
            ReviewResult
        """
        with open(file_path, "r") as f:
            spec = f.read()
        return await self.review(spec, personas=personas, rounds=rounds, **kwargs)

    # =========================================================================
    # Streaming API
    # =========================================================================

    async def review_stream(
        self,
        spec: str,
        personas: list[str] | None = None,
        rounds: int = 3,
        task: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Review with streaming events.

        Yields events as the debate progresses:
        - {"event": "debate_start", "debate_id": "..."}
        - {"event": "round_start", "round": 1}
        - {"event": "position", "agent": "...", "content": "..."}
        - {"event": "critique", "agent": "...", "content": "..."}
        - {"event": "consensus", "status": "...", "position": "..."}
        - {"event": "debate_end", "result": {...}}

        Example:
            async for event in client.review_stream(spec, personas=["sox"]):
                if event["event"] == "position":
                    print(f"{event['agent']}: {event['content'][:100]}...")
        """
        session = await self._ensure_session()

        request = ReviewRequest(
            spec=spec,
            personas=personas or ["security", "performance"],
            rounds=rounds,
            task=task,
            include_receipt=True,
        )

        try:
            async with session.post(
                self._url("/api/review/stream"),
                json=request.model_dump(exclude_none=True),
            ) as response:
                if response.status != 200:
                    await self._handle_response(response)

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        try:
                            event = json.loads(line[6:])
                            yield event
                        except json.JSONDecodeError:
                            continue
        except aiohttp.ClientConnectorError as e:
            raise ConnectionError(f"Failed to connect: {e}")
        except asyncio.TimeoutError:
            raise TimeoutError("Request timed out")

    # =========================================================================
    # Debate API (lower-level)
    # =========================================================================

    async def create_debate(
        self,
        task: str,
        agents: list[str] | None = None,
        rounds: int = 3,
        protocol: str = "propose_critique_revise",
    ) -> dict[str, Any]:
        """
        Create a new debate.

        Args:
            task: The task or question to debate
            agents: List of agent names to include
            rounds: Number of debate rounds
            protocol: Debate protocol to use

        Returns:
            Debate creation response with debate_id
        """
        data = await self._request(
            "POST",
            "/api/debates",
            data={
                "task": task,
                "agents": agents,
                "rounds": rounds,
                "protocol": protocol,
            },
        )
        return data

    async def get_debate(self, debate_id: str) -> dict[str, Any]:
        """Get debate status and results."""
        return await self._request("GET", f"/api/debates/{debate_id}")

    async def list_debates(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
    ) -> dict[str, Any]:
        """List debates."""
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return await self._request("GET", "/api/debates", params=params)

    # =========================================================================
    # Personas API
    # =========================================================================

    async def list_personas(self) -> list[dict[str, Any]]:
        """List available personas."""
        data = await self._request("GET", "/api/personas")
        return data.get("personas", [])

    async def get_persona(self, name: str) -> dict[str, Any]:
        """Get persona details."""
        return await self._request("GET", f"/api/personas/{name}")


# Convenience function for one-off reviews
async def review(
    spec: str,
    personas: list[str] | None = None,
    rounds: int = 3,
    api_key: str | None = None,
    **kwargs,
) -> ReviewResult:
    """
    Convenience function for one-off reviews.

    Example:
        from aragora_sdk import review

        result = await review(
            spec="My design document...",
            personas=["sox", "security"],
            api_key="ara_..."
        )
    """
    async with AragoraClient(api_key=api_key) as client:
        return await client.review(spec, personas=personas, rounds=rounds, **kwargs)
