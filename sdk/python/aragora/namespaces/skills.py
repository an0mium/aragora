"""
Skills Namespace API

Provides skill management and invocation:
- List available skills
- Get skill details and metrics
- Invoke skills with input data

Features:
- Skill discovery and inspection
- Skill execution with RBAC
- Metrics and monitoring
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


SkillCapability = Literal[
    "READ_LOCAL",
    "WRITE_LOCAL",
    "READ_DATABASE",
    "WRITE_DATABASE",
    "EXTERNAL_API",
    "WEB_SEARCH",
    "WEB_FETCH",
    "CODE_EXECUTION",
    "SHELL_EXECUTION",
    "LLM_INFERENCE",
    "EMBEDDING",
    "DEBATE_CONTEXT",
    "EVIDENCE_COLLECTION",
    "KNOWLEDGE_QUERY",
    "SYSTEM_INFO",
    "NETWORK",
]

SkillStatus = Literal[
    "SUCCESS",
    "FAILURE",
    "PARTIAL",
    "TIMEOUT",
    "RATE_LIMITED",
    "PERMISSION_DENIED",
    "INVALID_INPUT",
    "NOT_IMPLEMENTED",
]


class SkillsAPI:
    """
    Synchronous Skills API.

    Provides methods for managing and invoking skills:
    - List all available skills
    - Get skill details and metrics
    - Invoke skills with custom input

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> skills = client.skills.list()
        >>> result = client.skills.invoke(skill="web-search", input={"query": "TypeScript"})
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(self) -> dict[str, Any]:
        """
        List all available skills.

        Returns:
            Dict with:
            - skills: List of skill manifests
            - total: Total count
        """
        return self._client.request("GET", "/api/skills")

    def get(self, name: str) -> dict[str, Any]:
        """
        Get skill details.

        Args:
            name: Skill name

        Returns:
            Dict with full skill details including:
            - name, version, description
            - capabilities
            - input_schema, output_schema
            - rate_limit_per_minute
            - timeout_seconds
        """
        return self._client.request("GET", f"/api/skills/{name}")

    def get_metrics(self, name: str) -> dict[str, Any]:
        """
        Get skill metrics.

        Args:
            name: Skill name

        Returns:
            Dict with:
            - skill: Skill name
            - total_invocations: Total calls
            - successful_invocations: Success count
            - failed_invocations: Failure count
            - average_latency_ms: Average latency
            - last_invoked: Last invocation time
        """
        return self._client.request("GET", f"/api/skills/{name}/metrics")

    def invoke(
        self,
        skill: str,
        input: dict[str, Any] | None = None,
        user_id: str | None = None,
        permissions: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Invoke a skill.

        Args:
            skill: Skill name to invoke
            input: Input data for the skill
            user_id: User ID for RBAC
            permissions: User permissions
            metadata: Additional metadata
            timeout: Timeout in seconds (max 60)

        Returns:
            Dict with:
            - status: success or error
            - output: Skill output
            - error: Error message if failed
            - execution_time_ms: Execution time
        """
        data: dict[str, Any] = {"skill": skill}
        if input:
            data["input"] = input
        if user_id:
            data["user_id"] = user_id
        if permissions:
            data["permissions"] = permissions
        if metadata:
            data["metadata"] = metadata
        if timeout:
            data["timeout"] = timeout
        return self._client.request("POST", "/api/skills/invoke", json=data)


class AsyncSkillsAPI:
    """
    Asynchronous Skills API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     skills = await client.skills.list()
        ...     result = await client.skills.invoke(skill="web-search", input={"query": "AI"})
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(self) -> dict[str, Any]:
        """List all available skills."""
        return await self._client.request("GET", "/api/skills")

    async def get(self, name: str) -> dict[str, Any]:
        """Get skill details."""
        return await self._client.request("GET", f"/api/skills/{name}")

    async def get_metrics(self, name: str) -> dict[str, Any]:
        """Get skill metrics."""
        return await self._client.request("GET", f"/api/skills/{name}/metrics")

    async def invoke(
        self,
        skill: str,
        input: dict[str, Any] | None = None,
        user_id: str | None = None,
        permissions: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Invoke a skill."""
        data: dict[str, Any] = {"skill": skill}
        if input:
            data["input"] = input
        if user_id:
            data["user_id"] = user_id
        if permissions:
            data["permissions"] = permissions
        if metadata:
            data["metadata"] = metadata
        if timeout:
            data["timeout"] = timeout
        return await self._client.request("POST", "/api/skills/invoke", json=data)
