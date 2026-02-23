"""
CrewAI Agent - Integration with CrewAI multi-agent orchestration framework.

Provides secure enterprise access to CrewAI's capabilities:
- Crew execution (sequential, hierarchical)
- Task orchestration
- Agent coordination

With Aragora security controls:
- RBAC permission checking
- Audit logging
- Rate limiting via max_rpm
- Tool whitelist enforcement
- Response sanitization
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from aragora.agents.api_agents import common as api_common

create_client_session = api_common.create_client_session
from aragora.agents.api_agents.common import (
    AgentAPIError,
    AgentConnectionError,
    AgentRateLimitError,
    AgentTimeoutError,
    _sanitize_error_message,
)
from aragora.agents.errors import AgentError
from aragora.agents.api_agents.external_framework import (
    ExternalFrameworkAgent,
    ExternalFrameworkConfig,
)
from aragora.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


@dataclass
class CrewAIConfig(ExternalFrameworkConfig):
    """Configuration for CrewAI integration.

    Extends ExternalFrameworkConfig with CrewAI-specific settings for
    controlling crew execution and agent behavior.

    Attributes:
        process: Crew execution process mode ('sequential' or 'hierarchical').
            Sequential executes tasks one at a time, hierarchical uses a
            manager agent to delegate tasks.
        verbose: Enable verbose logging from CrewAI. Default False.
        memory: Enable CrewAI memory for agent learning. Default True.
        max_rpm: Maximum requests per minute to respect rate limits.
            Prevents overloading the CrewAI server. Default 10.
        allowed_tools: Whitelist of tools agents can use. Empty list means
            all tools are blocked for security. Must explicitly enable tools.
        crew_timeout: Overall timeout for crew execution in seconds.
            Prevents runaway crews from blocking resources. Default 600 (10 min).
    """

    # CrewAI-specific settings
    process: str = "sequential"  # sequential, hierarchical
    verbose: bool = False  # Enable verbose CrewAI logging
    memory: bool = True  # Enable CrewAI memory
    max_rpm: int = 10  # Max requests per minute
    allowed_tools: list[str] = field(default_factory=list)  # Tool whitelist
    crew_timeout: int = 600  # Overall crew execution timeout
    audit_all_requests: bool = True  # Log all crew executions for compliance

    def __post_init__(self) -> None:
        """Set CrewAI-specific defaults after initialization."""
        # Set CrewAI-specific defaults if not already set
        if not self.base_url:
            self.base_url = os.environ.get("CREWAI_URL", "http://localhost:8000")
        if self.generate_endpoint == "/generate":
            # Override default to CrewAI's kickoff endpoint
            self.generate_endpoint = "/v1/crew/kickoff"
        if self.health_endpoint == "/health":
            # CrewAI uses /health by default, which matches
            pass

    def validate_process(self) -> bool:
        """Validate the process mode is valid.

        Returns:
            True if process mode is valid, False otherwise.
        """
        return self.process in ("sequential", "hierarchical")


@AgentRegistry.register(
    "crewai",
    default_model="crewai",
    default_name="crewai",
    agent_type="API",
    requires="CrewAI server running at CREWAI_URL",
    env_vars="CREWAI_URL, CREWAI_API_KEY",
    description="Integration with CrewAI multi-agent orchestration framework",
    accepts_api_key=True,
)
class CrewAIAgent(ExternalFrameworkAgent):
    """
    Agent for CrewAI multi-agent orchestration framework.

    Wraps CrewAI's REST API with enterprise security controls.
    CrewAI is a multi-agent framework that orchestrates teams of AI agents
    to accomplish complex tasks through collaboration.

    Security Model:
        - Tools must be explicitly whitelisted via allowed_tools config
        - Rate limiting via max_rpm prevents API abuse
        - All crew executions can be audited for compliance
        - Response sanitization is always enabled
        - Crew timeout prevents runaway executions

    Example:
        >>> config = CrewAIConfig(
        ...     process="sequential",
        ...     allowed_tools=["search", "calculator"],
        ...     max_rpm=5,
        ... )
        >>> agent = CrewAIAgent(config=config, api_key="your-key")
        >>> response = await agent.kickoff("Research and summarize topic X")
    """

    def __init__(
        self,
        name: str = "crewai",
        model: str = "crewai",
        config: CrewAIConfig | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CrewAI agent.

        Args:
            name: Agent instance name.
            model: Model identifier (passed to CrewAI).
            config: CrewAI-specific configuration.
            api_key: API key for authentication. If not provided,
                reads from CREWAI_API_KEY environment variable.
            **kwargs: Additional arguments passed to ExternalFrameworkAgent.
        """
        if config is None:
            config = CrewAIConfig(base_url="")  # Will be set in __post_init__
            config.__post_init__()

        # Validate process mode
        if not config.validate_process():
            raise ValueError(
                f"Invalid process mode: {config.process}. Must be 'sequential' or 'hierarchical'."
            )

        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("CREWAI_API_KEY")

        super().__init__(
            name=name,
            model=model,
            config=config,
            api_key=api_key,
            **kwargs,
        )
        self.crewai_config = config
        self.agent_type = "crewai"

        # Rate limiting state
        self._request_timestamps: list[float] = []
        self._crew_status: dict[str, Any] = {}

    def _get_allowed_tools(self) -> list[str]:
        """Get list of allowed tools.

        Returns:
            List of tool names that are allowed.
        """
        return list(self.crewai_config.allowed_tools)

    def _is_tool_allowed(self, tool: str) -> bool:
        """Check if a tool is in the allowed list.

        Args:
            tool: Tool name to check.

        Returns:
            True if tool is allowed, False otherwise.
        """
        if not self.crewai_config.allowed_tools:
            return False
        return tool.lower() in [t.lower() for t in self.crewai_config.allowed_tools]

    def _filter_tools(self, requested_tools: list[str]) -> list[str]:
        """Filter tools to only include allowed ones.

        Args:
            requested_tools: List of requested tool names.

        Returns:
            List of allowed tool names only.
        """
        if not self.crewai_config.allowed_tools:
            return []
        return [t for t in requested_tools if self._is_tool_allowed(t)]

    def _build_capability_prefix(self) -> str:
        """Build capability restriction prefix for prompts.

        Returns:
            String prefix describing allowed tools and constraints.
        """
        tools = self._get_allowed_tools()
        if tools:
            return f"[Allowed tools: {', '.join(tools)}]\n\n"
        return "[No tools allowed - tool access is disabled]\n\n"

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits.

        Uses a sliding window to track requests per minute.

        Returns:
            True if within limits, False if rate limited.
        """
        now = time.time()
        window_start = now - 60.0  # 1 minute window

        # Remove timestamps outside the window
        self._request_timestamps = [ts for ts in self._request_timestamps if ts > window_start]

        # Check if we're at the limit
        return len(self._request_timestamps) < self.crewai_config.max_rpm

    def _record_request(self) -> None:
        """Record a request timestamp for rate limiting."""
        self._request_timestamps.append(time.time())

    def _get_rate_limit_wait(self) -> float:
        """Calculate how long to wait before next request is allowed.

        Returns:
            Seconds to wait, or 0 if no wait needed.
        """
        if not self._request_timestamps:
            return 0.0

        now = time.time()
        window_start = now - 60.0

        # Find oldest timestamp in window
        timestamps_in_window = [ts for ts in self._request_timestamps if ts > window_start]

        if len(timestamps_in_window) < self.crewai_config.max_rpm:
            return 0.0

        # Wait until oldest request falls outside window
        oldest = min(timestamps_in_window)
        return max(0.0, (oldest + 60.0) - now)

    async def generate(
        self,
        prompt: str,
        context: list | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate response from CrewAI with tool filtering and rate limiting.

        Adds tool restrictions to the prompt and enforces rate limits
        before delegating to the parent class.

        Args:
            prompt: The prompt/task to send to CrewAI.
            context: Optional conversation context.
            **kwargs: Additional generation parameters.

        Returns:
            Generated response text from CrewAI.

        Raises:
            AgentRateLimitError: If rate limit exceeded.
        """
        # Check rate limit
        if not self._check_rate_limit():
            wait_time = self._get_rate_limit_wait()
            raise AgentRateLimitError(
                f"Rate limit exceeded ({self.crewai_config.max_rpm} rpm). Wait {wait_time:.1f}s.",
                agent_name=self.name,
                retry_after=wait_time,
            )

        # Add capability restrictions to prompt
        prefixed_prompt = self._build_capability_prefix() + prompt

        if self.crewai_config.audit_all_requests:
            logger.info(
                "[%s] CrewAI request",
                self.name,
                extra={
                    "prompt_length": len(prompt),
                    "allowed_tools": self._get_allowed_tools(),
                    "crewai_process": self.crewai_config.process,
                    "memory_enabled": self.crewai_config.memory,
                },
            )

        # Record the request for rate limiting
        self._record_request()

        return await super().generate(prefixed_prompt, context, **kwargs)

    async def kickoff(
        self,
        task: str,
        inputs: dict[str, Any] | None = None,
        tools: list[str] | None = None,
    ) -> dict[str, Any]:
        """Start crew execution via CrewAI kickoff endpoint.

        This is the primary method for executing CrewAI crews with
        structured inputs and tool selection.

        Args:
            task: The task description for the crew to accomplish.
            inputs: Optional dict of inputs to pass to the crew.
            tools: Optional list of tools to request (will be filtered
                by allowed_tools whitelist).

        Returns:
            Dict with keys:
                - success: bool indicating if crew completed
                - output: crew output or error message
                - agent: agent name
                - tools_used: list of tools that were allowed and used
                - process: the process mode used
                - execution_time: time taken in seconds

        Raises:
            AgentRateLimitError: If rate limit exceeded.
            AgentConnectionError: If connection to CrewAI fails.
            AgentTimeoutError: If crew execution times out.
        """
        start_time = time.time()

        # Check rate limit
        if not self._check_rate_limit():
            wait_time = self._get_rate_limit_wait()
            raise AgentRateLimitError(
                f"Rate limit exceeded ({self.crewai_config.max_rpm} rpm). Wait {wait_time:.1f}s.",
                agent_name=self.name,
                retry_after=wait_time,
            )

        # Filter tools to only allowed ones
        filtered_tools = []
        if tools:
            filtered_tools = self._filter_tools(tools)
            blocked_tools = set(tools) - set(filtered_tools)
            if blocked_tools:
                logger.warning(
                    "[%s] Blocked tools: %s. Allowed: %s",
                    self.name,
                    blocked_tools,
                    self.crewai_config.allowed_tools,
                )

        if self.crewai_config.audit_all_requests:
            logger.info(
                "[%s] CrewAI kickoff",
                self.name,
                extra={
                    "task_length": len(task),
                    "inputs": list(inputs.keys()) if inputs else [],
                    "requested_tools": tools or [],
                    "allowed_tools": filtered_tools,
                    "crewai_process": self.crewai_config.process,
                },
            )

        # Record the request for rate limiting
        self._record_request()

        # Build request payload
        payload = {
            "task": task,
            "process": self.crewai_config.process,
            "verbose": self.crewai_config.verbose,
            "memory": self.crewai_config.memory,
        }

        if inputs:
            payload["inputs"] = inputs
        if filtered_tools:
            payload["tools"] = filtered_tools

        url = f"{self.base_url}{self.config.generate_endpoint}"

        try:
            async with create_client_session(
                timeout=float(self.crewai_config.crew_timeout)
            ) as session:
                async with session.post(
                    url, json=payload, headers=self._build_headers()
                ) as response:
                    if response.status == 429:
                        error_text = await response.text()
                        retry_after = self._parse_retry_after(response)
                        raise AgentRateLimitError(
                            f"Rate limited by CrewAI server: {_sanitize_error_message(error_text)}",
                            agent_name=self.name,
                            retry_after=retry_after,
                        )

                    if response.status != 200:
                        error_text = await response.text()
                        sanitized = _sanitize_error_message(error_text)
                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        raise AgentAPIError(
                            f"CrewAI API error {response.status}: {sanitized}",
                            agent_name=self.name,
                            status_code=response.status,
                        )

                    data = await response.json()

                    # Record success for circuit breaker
                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()

                    execution_time = time.time() - start_time

                    # Extract output from response
                    output = self._extract_response_text(data)

                    return {
                        "success": True,
                        "output": self._sanitize_response(output),
                        "agent": self.name,
                        "tools_used": filtered_tools,
                        "process": self.crewai_config.process,
                        "execution_time": execution_time,
                    }

        except aiohttp.ClientConnectorError as e:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            raise AgentConnectionError(
                f"Cannot connect to CrewAI server at {self.base_url}: {e}",
                agent_name=self.name,
                cause=e,
            )
        except TimeoutError as e:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            raise AgentTimeoutError(
                f"CrewAI crew execution timed out after {self.crewai_config.crew_timeout}s",
                agent_name=self.name,
                cause=e,
            )
        except (AgentRateLimitError, AgentAPIError, AgentConnectionError, AgentTimeoutError):
            raise
        except (AgentError, ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
            execution_time = time.time() - start_time
            logger.error("[%s] Crew execution failed: %s", self.name, e)
            return {
                "success": False,
                "output": "Crew execution failed",
                "agent": self.name,
                "tools_used": [],
                "process": self.crewai_config.process,
                "execution_time": execution_time,
            }

    async def get_crew_status(self, crew_id: str | None = None) -> dict[str, Any]:
        """Get status of crew execution.

        Queries the CrewAI server for the status of a running or
        completed crew execution.

        Args:
            crew_id: Optional crew execution ID. If not provided,
                returns status of the most recent execution.

        Returns:
            Dict with status information:
                - status: 'running', 'completed', 'failed', or 'unknown'
                - progress: float 0.0-1.0 indicating completion
                - message: human-readable status message
                - crew_id: the crew execution ID
        """
        status_url = f"{self.base_url}/v1/crew/status"
        if crew_id:
            status_url += f"/{crew_id}"

        try:
            async with create_client_session(timeout=30.0) as session:
                async with session.get(status_url, headers=self._build_headers()) as response:
                    if response.status == 404:
                        return {
                            "status": "unknown",
                            "progress": 0.0,
                            "message": "Crew execution not found",
                            "crew_id": crew_id,
                        }

                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "status": "error",
                            "progress": 0.0,
                            "message": f"Error fetching status: {_sanitize_error_message(error_text)}",
                            "crew_id": crew_id,
                        }

                    data = await response.json()

                    return {
                        "status": data.get("status", "unknown"),
                        "progress": data.get("progress", 0.0),
                        "message": data.get("message", ""),
                        "crew_id": data.get("crew_id", crew_id),
                    }

        except (aiohttp.ClientError, OSError, TimeoutError) as e:
            return {
                "status": "error",
                "progress": 0.0,
                "message": f"Connection error: {e}",
                "crew_id": crew_id,
            }

    async def is_available(self) -> bool:
        """Check if CrewAI server is accessible.

        Returns:
            True if CrewAI server responds to health check, False otherwise.
        """
        available = await super().is_available()
        if available:
            logger.debug(
                "[%s] CrewAI available at %s (process=%s, tools=%s)",
                self.name,
                self.base_url,
                self.crewai_config.process,
                len(self.crewai_config.allowed_tools),
            )
        return available

    def get_config_status(self) -> dict[str, Any]:
        """Get current configuration status.

        Returns:
            Dict describing current configuration settings.
        """
        return {
            "process": self.crewai_config.process,
            "verbose": self.crewai_config.verbose,
            "memory_enabled": self.crewai_config.memory,
            "max_rpm": self.crewai_config.max_rpm,
            "allowed_tools": self.crewai_config.allowed_tools,
            "crew_timeout": self.crewai_config.crew_timeout,
            "audit_enabled": self.crewai_config.audit_all_requests,
            "base_url": self.base_url,
            "current_rpm_usage": len(self._request_timestamps),
        }

    def reset_rate_limit(self) -> None:
        """Reset rate limit counters (for testing)."""
        self._request_timestamps.clear()


__all__ = ["CrewAIAgent", "CrewAIConfig"]
