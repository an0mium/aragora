"""
A2A Protocol Client.

Client for invoking external agents via the A2A protocol.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from aragora.protocols.a2a.types import (
    AgentCard,
    AgentCapability,
    ContextItem,
    TaskRequest,
    TaskResult,
    TaskStatus,
    TaskPriority,
)

logger = logging.getLogger(__name__)


class A2AClientError(Exception):
    """Error from A2A client operations."""

    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        task_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.agent_name = agent_name
        self.task_id = task_id


class A2AClient:
    """
    Client for invoking external agents via A2A protocol.

    Supports:
    - Agent discovery from registries
    - Synchronous task invocation
    - Streaming task invocation
    - Task status polling
    """

    def __init__(
        self,
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        """
        Initialize A2A client.

        Args:
            timeout: Default timeout for requests
            max_retries: Maximum retries for failed requests
        """
        self._timeout = timeout
        self._max_retries = max_retries

        # Agent registry cache
        self._agents: Dict[str, AgentCard] = {}

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "A2AClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def discover_agents(
        self,
        registry_url: str,
        capability: Optional[AgentCapability] = None,
    ) -> List[AgentCard]:
        """
        Discover agents from a registry.

        Args:
            registry_url: URL of the agent registry
            capability: Optional capability filter

        Returns:
            List of discovered agent cards
        """
        client = self._get_client()

        try:
            url = f"{registry_url}/agents"
            if capability:
                url += f"?capability={capability.value}"

            response = await client.get(url)
            response.raise_for_status()

            data = response.json()
            agents = []

            for agent_data in data.get("agents", []):
                agent = AgentCard.from_dict(agent_data)
                agents.append(agent)
                self._agents[agent.name] = agent

            logger.info(f"Discovered {len(agents)} agents from {registry_url}")

            return agents

        except httpx.HTTPError as e:
            logger.error(f"Failed to discover agents from {registry_url}: {e}")
            raise A2AClientError(f"Discovery failed: {e}")

    def register_agent(self, agent: AgentCard) -> None:
        """
        Manually register an agent card.

        Args:
            agent: Agent card to register
        """
        self._agents[agent.name] = agent
        logger.debug(f"Registered agent: {agent.name}")

    def get_agent(self, name: str) -> Optional[AgentCard]:
        """Get a registered agent by name."""
        return self._agents.get(name)

    async def invoke(
        self,
        agent_name: str,
        instruction: str,
        context: Optional[List[ContextItem]] = None,
        capability: Optional[AgentCapability] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskResult:
        """
        Invoke an agent synchronously.

        Args:
            agent_name: Name of the agent to invoke
            instruction: Task instruction
            context: Optional context items
            capability: Required capability
            priority: Task priority
            timeout_ms: Task timeout in milliseconds
            metadata: Additional metadata

        Returns:
            Task result
        """
        agent = self._agents.get(agent_name)
        if not agent:
            raise A2AClientError(f"Agent not found: {agent_name}", agent_name=agent_name)

        if not agent.endpoint:
            raise A2AClientError(f"Agent has no endpoint: {agent_name}", agent_name=agent_name)

        # Create task request
        task_id = f"t_{uuid.uuid4().hex[:12]}"
        request = TaskRequest(
            task_id=task_id,
            instruction=instruction,
            context=context or [],
            capability=capability,
            priority=priority,
            timeout_ms=timeout_ms or int(self._timeout * 1000),
            metadata=metadata or {},
        )

        # Send request
        client = self._get_client()

        try:
            response = await client.post(
                f"{agent.endpoint}/tasks",
                json=request.to_dict(),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            result_data = response.json()
            return TaskResult.from_dict(result_data)

        except httpx.HTTPError as e:
            logger.error(f"Task invocation failed for {agent_name}: {e}")
            return TaskResult(
                task_id=task_id,
                agent_name=agent_name,
                status=TaskStatus.FAILED,
                error_message=str(e),
            )

    async def stream_invoke(
        self,
        agent_name: str,
        instruction: str,
        context: Optional[List[ContextItem]] = None,
        capability: Optional[AgentCapability] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Invoke an agent with streaming output.

        Args:
            agent_name: Name of the agent to invoke
            instruction: Task instruction
            context: Optional context items
            capability: Required capability

        Yields:
            Stream events from the agent
        """
        agent = self._agents.get(agent_name)
        if not agent:
            raise A2AClientError(f"Agent not found: {agent_name}", agent_name=agent_name)

        if not agent.endpoint:
            raise A2AClientError(f"Agent has no endpoint: {agent_name}", agent_name=agent_name)

        # Create task request
        task_id = f"t_{uuid.uuid4().hex[:12]}"
        request = TaskRequest(
            task_id=task_id,
            instruction=instruction,
            context=context or [],
            capability=capability,
            stream_output=True,
        )

        # Stream request
        client = self._get_client()

        try:
            async with client.stream(
                "POST",
                f"{agent.endpoint}/tasks/stream",
                json=request.to_dict(),
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        yield data

        except httpx.HTTPError as e:
            logger.error(f"Stream invocation failed for {agent_name}: {e}")
            yield {
                "type": "error",
                "task_id": task_id,
                "error": str(e),
            }

    async def get_task_status(
        self,
        agent_name: str,
        task_id: str,
    ) -> TaskResult:
        """
        Get the status of a task.

        Args:
            agent_name: Name of the agent
            task_id: Task ID to check

        Returns:
            Current task result/status
        """
        agent = self._agents.get(agent_name)
        if not agent:
            raise A2AClientError(f"Agent not found: {agent_name}", agent_name=agent_name)

        if not agent.endpoint:
            raise A2AClientError(f"Agent has no endpoint: {agent_name}", agent_name=agent_name)

        client = self._get_client()

        try:
            response = await client.get(f"{agent.endpoint}/tasks/{task_id}")
            response.raise_for_status()

            return TaskResult.from_dict(response.json())

        except httpx.HTTPError as e:
            logger.error(f"Failed to get task status: {e}")
            raise A2AClientError(
                f"Status check failed: {e}",
                agent_name=agent_name,
                task_id=task_id,
            )

    async def cancel_task(
        self,
        agent_name: str,
        task_id: str,
    ) -> bool:
        """
        Cancel a running task.

        Args:
            agent_name: Name of the agent
            task_id: Task ID to cancel

        Returns:
            True if cancellation was successful
        """
        agent = self._agents.get(agent_name)
        if not agent or not agent.endpoint:
            return False

        client = self._get_client()

        try:
            response = await client.delete(f"{agent.endpoint}/tasks/{task_id}")
            return response.status_code in (200, 202, 204)

        except httpx.HTTPError as e:
            logger.error(f"Failed to cancel task: {e}")
            return False

    def list_agents(
        self,
        capability: Optional[AgentCapability] = None,
    ) -> List[AgentCard]:
        """
        List registered agents.

        Args:
            capability: Optional capability filter

        Returns:
            List of matching agent cards
        """
        agents = list(self._agents.values())

        if capability:
            agents = [a for a in agents if a.supports_capability(capability)]

        return agents


__all__ = [
    "A2AClient",
    "A2AClientError",
]
