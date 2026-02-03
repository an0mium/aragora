"""
LangGraph Agent - Integration with LangGraph state machine framework.

LangGraph is a framework for building agent workflows as state machines.
This integration provides secure access to LangGraph Cloud API with:
- Graph execution (invoke, stream)
- State management (get, update)
- Node filtering for security
- Interrupt points for human-in-the-loop

With Aragora security controls:
- Node whitelist enforcement
- State size limits
- Recursion limits
- Audit logging
- Response sanitization
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import aiohttp

from aragora.agents.api_agents import common as api_common
from aragora.agents.api_agents.common import (
    AgentAPIError,
    AgentConnectionError,
    AgentTimeoutError,
    Message,
    _sanitize_error_message,
)
from aragora.agents.api_agents.external_framework import (
    ExternalFrameworkAgent,
    ExternalFrameworkConfig,
)
from aragora.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


@dataclass
class LangGraphConfig(ExternalFrameworkConfig):
    """Configuration for LangGraph integration.

    Extends ExternalFrameworkConfig with LangGraph-specific settings for
    controlling graph execution and state management.

    Attributes:
        graph_id: Specific graph to use. None means use default.
        checkpoint_ns: Checkpoint namespace for state persistence.
        recursion_limit: Maximum recursion depth to prevent infinite loops.
        stream_mode: Streaming mode - 'values', 'updates', or 'debug'.
        interrupt_before: List of node names to pause before executing.
        interrupt_after: List of node names to pause after executing.
        allowed_nodes: Whitelist of allowed nodes. Empty list means all allowed.
        max_state_size: Maximum state size in bytes (1MB default).
    """

    # LangGraph-specific settings
    graph_id: str | None = None
    checkpoint_ns: str | None = None
    recursion_limit: int = 50
    stream_mode: str = "values"  # values, updates, debug
    interrupt_before: list[str] = field(default_factory=list)
    interrupt_after: list[str] = field(default_factory=list)
    allowed_nodes: list[str] = field(default_factory=list)
    max_state_size: int = 1048576  # 1MB default

    def __post_init__(self) -> None:
        """Set LangGraph-specific defaults after initialization."""
        # Set LangGraph-specific defaults if not already set
        if not self.base_url:
            self.base_url = os.environ.get("LANGGRAPH_URL", "http://localhost:8123")
        if self.generate_endpoint == "/generate":
            # Override default to LangGraph's streaming endpoint
            self.generate_endpoint = "/runs/stream"
        if self.health_endpoint == "/health":
            # LangGraph Cloud uses /health
            pass

    def validate_stream_mode(self) -> None:
        """Validate stream_mode value."""
        valid_modes = {"values", "updates", "debug"}
        if self.stream_mode not in valid_modes:
            raise ValueError(
                f"Invalid stream_mode '{self.stream_mode}'. "
                f"Must be one of: {', '.join(valid_modes)}"
            )


@AgentRegistry.register(
    "langgraph",
    default_model="langgraph",
    default_name="langgraph",
    agent_type="API",
    requires="LangGraph Cloud or self-hosted LangGraph server",
    env_vars="LANGGRAPH_URL, LANGGRAPH_API_KEY",
    description="Integration with LangGraph state machine framework",
    accepts_api_key=True,
)
class LangGraphAgent(ExternalFrameworkAgent):
    """
    Agent for LangGraph state machine framework.

    Wraps LangGraph's API with enterprise security controls.
    LangGraph enables building agent workflows as state machines with
    support for checkpointing, branching, and human-in-the-loop patterns.

    Security Model:
        - Node whitelist enforcement (allowed_nodes)
        - State size limits to prevent memory exhaustion
        - Recursion limits to prevent infinite loops
        - All requests are auditable
        - Response sanitization is always enabled

    Example:
        >>> config = LangGraphConfig(
        ...     graph_id="my-agent-graph",
        ...     allowed_nodes=["generate", "review"],
        ...     recursion_limit=25,
        ... )
        >>> agent = LangGraphAgent(config=config, api_key="your-key")
        >>> response = await agent.generate("Process this task")
    """

    def __init__(
        self,
        name: str = "langgraph",
        model: str = "langgraph",
        config: LangGraphConfig | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize LangGraph agent.

        Args:
            name: Agent instance name.
            model: Model identifier (passed to LangGraph).
            config: LangGraph-specific configuration.
            api_key: API key for authentication. If not provided,
                reads from LANGGRAPH_API_KEY environment variable.
            **kwargs: Additional arguments passed to ExternalFrameworkAgent.
        """
        if config is None:
            config = LangGraphConfig(base_url="")
            config.__post_init__()

        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("LANGGRAPH_API_KEY")

        super().__init__(
            name=name,
            model=model,
            config=config,
            api_key=api_key,
            **kwargs,
        )
        self.langgraph_config = config
        self.agent_type = "langgraph"

        # Thread ID for stateful conversations
        self._thread_id: str | None = None

    def _validate_state_size(self, state: dict[str, Any]) -> None:
        """Validate state size against configured limit.

        Args:
            state: State dictionary to validate.

        Raises:
            AgentAPIError: If state exceeds max_state_size.
        """
        state_json = json.dumps(state)
        state_size = len(state_json.encode("utf-8"))

        if state_size > self.langgraph_config.max_state_size:
            raise AgentAPIError(
                f"State size ({state_size} bytes) exceeds limit "
                f"({self.langgraph_config.max_state_size} bytes)",
                agent_name=self.name,
            )

    def _validate_node_allowed(self, node_name: str) -> bool:
        """Check if a node is in the allowed list.

        Args:
            node_name: Name of the node to check.

        Returns:
            True if node is allowed (or no whitelist configured).
        """
        if not self.langgraph_config.allowed_nodes:
            # Empty list means all nodes allowed
            return True
        return node_name in self.langgraph_config.allowed_nodes

    def _filter_response_nodes(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter streaming events to only include allowed nodes.

        Args:
            events: List of streaming events.

        Returns:
            Filtered list with only allowed node events.
        """
        if not self.langgraph_config.allowed_nodes:
            return events

        filtered = []
        for event in events:
            node = event.get("node")
            if node is None or self._validate_node_allowed(node):
                filtered.append(event)
            else:
                logger.debug(f"[{self.name}] Filtered out event from disallowed node: {node}")
        return filtered

    def _build_run_payload(
        self,
        input_data: dict[str, Any] | str,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Build payload for LangGraph run request.

        Args:
            input_data: Input for the graph (dict or string).
            thread_id: Optional thread ID for stateful execution.

        Returns:
            Request payload dictionary.
        """
        # Convert string input to standard message format
        if isinstance(input_data, str):
            input_data = {"messages": [{"role": "user", "content": input_data}]}

        payload: dict[str, Any] = {
            "input": input_data,
            "config": {
                "recursion_limit": self.langgraph_config.recursion_limit,
            },
        }

        # Add graph_id if specified
        if self.langgraph_config.graph_id:
            payload["assistant_id"] = self.langgraph_config.graph_id

        # Add thread_id for stateful execution
        if thread_id:
            payload["thread_id"] = thread_id
        elif self._thread_id:
            payload["thread_id"] = self._thread_id

        # Add checkpoint namespace if specified
        if self.langgraph_config.checkpoint_ns:
            payload["config"]["configurable"] = {
                "checkpoint_ns": self.langgraph_config.checkpoint_ns
            }

        # Add interrupt points
        if self.langgraph_config.interrupt_before:
            payload["interrupt_before"] = self.langgraph_config.interrupt_before
        if self.langgraph_config.interrupt_after:
            payload["interrupt_after"] = self.langgraph_config.interrupt_after

        # Add stream mode
        payload["stream_mode"] = self.langgraph_config.stream_mode

        return payload

    async def generate(
        self,
        prompt: str,
        context: list[Message] | None = None,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate response by invoking the LangGraph workflow.

        Args:
            prompt: The prompt to process.
            context: Optional conversation context.
            thread_id: Optional thread ID for stateful execution.
            **kwargs: Additional parameters.

        Returns:
            Generated response text.
        """
        logger.info(
            f"[{self.name}] LangGraph request",
            extra={
                "prompt_length": len(prompt),
                "graph_id": self.langgraph_config.graph_id,
                "recursion_limit": self.langgraph_config.recursion_limit,
            },
        )

        # Use invoke endpoint for non-streaming generation
        url = f"{self.base_url}/runs"

        payload = self._build_run_payload(prompt, thread_id)
        # For generate(), we want wait mode (synchronous)
        payload["stream_mode"] = "values"  # Get final values
        # Remove streaming to get synchronous response
        del payload["stream_mode"]

        async with api_common.create_client_session(timeout=float(self.timeout)) as session:
            try:
                async with session.post(
                    url, json=payload, headers=self._build_headers()
                ) as response:
                    if response.status == 429:
                        error_text = await response.text()
                        raise AgentAPIError(
                            f"Rate limited by LangGraph: {_sanitize_error_message(error_text)}",
                            agent_name=self.name,
                            status_code=429,
                        )

                    if response.status != 200:
                        error_text = await response.text()
                        sanitized = _sanitize_error_message(error_text)
                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        raise AgentAPIError(
                            f"LangGraph API error {response.status}: {sanitized}",
                            agent_name=self.name,
                            status_code=response.status,
                        )

                    data = await response.json()

                    # Record success
                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()

                    # Store thread_id if returned
                    if "thread_id" in data:
                        self._thread_id = data["thread_id"]

                    # Extract response from output
                    return self._extract_langgraph_response(data)

            except aiohttp.ClientConnectorError as e:
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                raise AgentConnectionError(
                    f"Cannot connect to LangGraph at {self.base_url}: {e}",
                    agent_name=self.name,
                    cause=e,
                )
            except TimeoutError as e:
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                raise AgentTimeoutError(
                    f"LangGraph request timed out after {self.timeout}s",
                    agent_name=self.name,
                    cause=e,
                )

    def _extract_langgraph_response(self, data: dict[str, Any]) -> str:
        """Extract response text from LangGraph output.

        Args:
            data: Response data from LangGraph.

        Returns:
            Extracted text content.
        """
        # Handle run result format
        output = data.get("output") or data.get("values") or data

        # Extract messages from output
        if isinstance(output, dict):
            messages = output.get("messages", [])
            if messages and isinstance(messages, list):
                # Get last AI message
                for msg in reversed(messages):
                    if isinstance(msg, dict):
                        role = msg.get("role", "").lower()
                        if role in ("assistant", "ai"):
                            content = msg.get("content", "")
                            if isinstance(content, str):
                                return self._sanitize_response(content)
                            elif isinstance(content, list):
                                # Handle content blocks
                                texts = []
                                for block in content:
                                    if isinstance(block, dict):
                                        texts.append(block.get("text", ""))
                                    elif isinstance(block, str):
                                        texts.append(block)
                                return self._sanitize_response("".join(texts))

            # Fallback: try to get result directly
            if "result" in output:
                result = output["result"]
                if isinstance(result, str):
                    return self._sanitize_response(result)

        # Last resort: stringify the output
        if isinstance(output, str):
            return self._sanitize_response(output)

        return self._sanitize_response(json.dumps(output))

    async def invoke(
        self,
        input_data: dict[str, Any] | str,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Invoke the LangGraph workflow synchronously.

        Provides full access to graph output including intermediate states.

        Args:
            input_data: Input for the graph.
            thread_id: Optional thread ID for stateful execution.
            **kwargs: Additional parameters.

        Returns:
            Full response dictionary including output and metadata.
        """
        logger.info(
            f"[{self.name}] LangGraph invoke",
            extra={
                "graph_id": self.langgraph_config.graph_id,
                "thread_id": thread_id,
            },
        )

        # Validate input state size
        if isinstance(input_data, dict):
            self._validate_state_size(input_data)

        url = f"{self.base_url}/runs"
        payload = self._build_run_payload(input_data, thread_id)

        async with api_common.create_client_session(timeout=float(self.timeout)) as session:
            try:
                async with session.post(
                    url, json=payload, headers=self._build_headers()
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        sanitized = _sanitize_error_message(error_text)
                        raise AgentAPIError(
                            f"LangGraph invoke error {response.status}: {sanitized}",
                            agent_name=self.name,
                            status_code=response.status,
                        )

                    data = await response.json()

                    # Store thread_id if returned
                    if "thread_id" in data:
                        self._thread_id = data["thread_id"]

                    return data

            except aiohttp.ClientConnectorError as e:
                raise AgentConnectionError(
                    f"Cannot connect to LangGraph: {e}",
                    agent_name=self.name,
                    cause=e,
                )

    async def stream(
        self,
        input_data: dict[str, Any] | str,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream execution events from the LangGraph workflow.

        Args:
            input_data: Input for the graph.
            thread_id: Optional thread ID for stateful execution.
            **kwargs: Additional parameters.

        Yields:
            Streaming events from graph execution.
        """
        logger.info(
            f"[{self.name}] LangGraph stream",
            extra={
                "graph_id": self.langgraph_config.graph_id,
                "stream_mode": self.langgraph_config.stream_mode,
            },
        )

        # Validate input state size
        if isinstance(input_data, dict):
            self._validate_state_size(input_data)

        url = f"{self.base_url}{self.config.generate_endpoint}"
        payload = self._build_run_payload(input_data, thread_id)

        async with api_common.create_client_session(timeout=float(self.timeout)) as session:
            try:
                async with session.post(
                    url, json=payload, headers=self._build_headers()
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        sanitized = _sanitize_error_message(error_text)
                        raise AgentAPIError(
                            f"LangGraph stream error {response.status}: {sanitized}",
                            agent_name=self.name,
                            status_code=response.status,
                        )

                    # Process SSE stream
                    buffer = ""
                    async for chunk in response.content.iter_any():
                        buffer += chunk.decode("utf-8", errors="ignore")

                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if not line or not line.startswith("data: "):
                                continue

                            data_str = line[6:]  # Remove 'data: ' prefix

                            if data_str == "[DONE]":
                                return

                            try:
                                event = json.loads(data_str)

                                # Filter by allowed nodes
                                node = event.get("node")
                                if node and not self._validate_node_allowed(node):
                                    logger.debug(f"[{self.name}] Skipping disallowed node: {node}")
                                    continue

                                yield event

                            except json.JSONDecodeError:
                                logger.debug(f"[{self.name}] Malformed JSON in stream")
                                continue

            except aiohttp.ClientConnectorError as e:
                raise AgentConnectionError(
                    f"Cannot connect to LangGraph: {e}",
                    agent_name=self.name,
                    cause=e,
                )

    async def get_state(
        self,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Get the current state of a thread.

        Args:
            thread_id: Thread ID to get state for. Uses stored thread_id if not provided.

        Returns:
            Current state dictionary.

        Raises:
            AgentAPIError: If thread_id not provided and none stored.
        """
        tid = thread_id or self._thread_id
        if not tid:
            raise AgentAPIError(
                "No thread_id provided and none stored from previous execution",
                agent_name=self.name,
            )

        url = f"{self.base_url}/threads/{tid}/state"

        logger.info(
            f"[{self.name}] LangGraph get_state",
            extra={"thread_id": tid},
        )

        async with api_common.create_client_session(timeout=float(self.timeout)) as session:
            try:
                async with session.get(url, headers=self._build_headers()) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        sanitized = _sanitize_error_message(error_text)
                        raise AgentAPIError(
                            f"LangGraph get_state error {response.status}: {sanitized}",
                            agent_name=self.name,
                            status_code=response.status,
                        )

                    data = await response.json()
                    return data

            except aiohttp.ClientConnectorError as e:
                raise AgentConnectionError(
                    f"Cannot connect to LangGraph: {e}",
                    agent_name=self.name,
                    cause=e,
                )

    async def update_state(
        self,
        state: dict[str, Any],
        thread_id: str | None = None,
        as_node: str | None = None,
    ) -> dict[str, Any]:
        """Update the state of a thread.

        Args:
            state: New state values to merge.
            thread_id: Thread ID to update. Uses stored thread_id if not provided.
            as_node: Node to attribute the update to (for graph routing).

        Returns:
            Updated state dictionary.

        Raises:
            AgentAPIError: If thread_id not provided, state too large,
                or node not allowed.
        """
        tid = thread_id or self._thread_id
        if not tid:
            raise AgentAPIError(
                "No thread_id provided and none stored from previous execution",
                agent_name=self.name,
            )

        # Validate state size
        self._validate_state_size(state)

        # Validate node if specified
        if as_node and not self._validate_node_allowed(as_node):
            raise AgentAPIError(
                f"Node '{as_node}' not in allowed_nodes list",
                agent_name=self.name,
            )

        url = f"{self.base_url}/threads/{tid}/state"

        payload: dict[str, Any] = {"values": state}
        if as_node:
            payload["as_node"] = as_node

        logger.info(
            f"[{self.name}] LangGraph update_state",
            extra={"thread_id": tid, "as_node": as_node},
        )

        async with api_common.create_client_session(timeout=float(self.timeout)) as session:
            try:
                async with session.post(
                    url, json=payload, headers=self._build_headers()
                ) as response:
                    if response.status not in (200, 201):
                        error_text = await response.text()
                        sanitized = _sanitize_error_message(error_text)
                        raise AgentAPIError(
                            f"LangGraph update_state error {response.status}: {sanitized}",
                            agent_name=self.name,
                            status_code=response.status,
                        )

                    data = await response.json()
                    return data

            except aiohttp.ClientConnectorError as e:
                raise AgentConnectionError(
                    f"Cannot connect to LangGraph: {e}",
                    agent_name=self.name,
                    cause=e,
                )

    def set_thread_id(self, thread_id: str) -> None:
        """Set the thread ID for subsequent operations.

        Args:
            thread_id: Thread ID to use.
        """
        self._thread_id = thread_id

    def get_thread_id(self) -> str | None:
        """Get the current thread ID.

        Returns:
            Current thread ID or None.
        """
        return self._thread_id

    def clear_thread(self) -> None:
        """Clear the stored thread ID."""
        self._thread_id = None

    def get_config_status(self) -> dict[str, Any]:
        """Get current configuration status.

        Returns:
            Dict describing current LangGraph configuration.
        """
        return {
            "graph_id": self.langgraph_config.graph_id,
            "checkpoint_ns": self.langgraph_config.checkpoint_ns,
            "recursion_limit": self.langgraph_config.recursion_limit,
            "stream_mode": self.langgraph_config.stream_mode,
            "interrupt_before": self.langgraph_config.interrupt_before,
            "interrupt_after": self.langgraph_config.interrupt_after,
            "allowed_nodes": self.langgraph_config.allowed_nodes,
            "max_state_size": self.langgraph_config.max_state_size,
            "thread_id": self._thread_id,
            "base_url": self.base_url,
        }


__all__ = ["LangGraphAgent", "LangGraphConfig"]
