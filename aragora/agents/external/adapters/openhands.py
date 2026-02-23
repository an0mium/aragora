"""OpenHands adapter for Aragora external agent integration.

Provides integration with OpenHands (formerly OpenDevin) autonomous
coding agent through Aragora's gateway layer.

OpenHands is an open-source platform for software development agents
with support for browser, terminal, and file editing tools.

For more information: https://github.com/All-Hands-AI/OpenHands
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any
from collections.abc import AsyncIterator

from aragora.agents.external.base import (
    ExternalAgentAdapter,
    ExternalAgentError,
    TaskNotCompleteError,
    TaskNotFoundError,
)
from aragora.agents.external.config import OpenHandsConfig
from aragora.agents.external.models import (
    HealthStatus,
    TaskProgress,
    TaskRequest,
    TaskResult,
    TaskStatus,
    ToolPermission,
)
from aragora.agents.external.registry import ExternalAgentRegistry

logger = logging.getLogger(__name__)


# Status mapping from OpenHands to Aragora
_STATUS_MAP: dict[str, TaskStatus] = {
    "running": TaskStatus.RUNNING,
    "finished": TaskStatus.COMPLETED,
    "stopped": TaskStatus.CANCELLED,
    "error": TaskStatus.FAILED,
    "awaiting_user_input": TaskStatus.PAUSED,
    "init": TaskStatus.INITIALIZING,
    "paused": TaskStatus.PAUSED,
}


@ExternalAgentRegistry.register(
    "openhands",
    config_class=OpenHandsConfig,
    description="OpenHands autonomous coding agent (formerly OpenDevin)",
    requires="OpenHands server (docker or local)",
    env_vars="OPENHANDS_URL",
)
class OpenHandsAdapter(ExternalAgentAdapter):
    """Adapter for OpenHands autonomous coding agent.

    OpenHands provides autonomous software engineering capabilities
    with browser, terminal, and code editing tools.

    Usage:
        config = OpenHandsConfig(base_url="http://localhost:3000")
        adapter = OpenHandsAdapter(config)

        task_id = await adapter.submit_task(TaskRequest(
            task_type="code",
            prompt="Create a Python function that...",
            tool_permissions={ToolPermission.FILE_WRITE, ToolPermission.SHELL_EXECUTE},
        ))

        result = await adapter.get_task_result(task_id)
    """

    adapter_name = "openhands"

    def __init__(self, config: OpenHandsConfig, **kwargs: Any):
        """Initialize the OpenHands adapter.

        Args:
            config: OpenHands configuration.
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(config, **kwargs)
        self._config: OpenHandsConfig = config
        self._client: Any = None  # Lazy-initialized HTTP client
        self._active_tasks: dict[str, dict[str, Any]] = {}

    async def _ensure_client(self) -> Any:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            try:
                import httpx

                self._client = httpx.AsyncClient(
                    base_url=self._config.base_url,
                    timeout=httpx.Timeout(30.0, connect=10.0),
                )
            except ImportError:
                raise ExternalAgentError(
                    "httpx is required for OpenHands adapter. Install with: pip install httpx",
                    adapter_name=self.adapter_name,
                )
        return self._client

    async def submit_task(self, request: TaskRequest) -> str:
        """Submit a task to OpenHands.

        Args:
            request: Task request with prompt and permissions.

        Returns:
            Task ID for tracking.

        Raises:
            ExternalAgentError: If submission fails.
        """
        # Validate task
        is_valid, error = await self.validate_task(request)
        if not is_valid:
            raise ExternalAgentError(
                error or "Task validation failed",
                adapter_name=self.adapter_name,
            )

        client = await self._ensure_client()

        # Build OpenHands agent configuration
        agent_config = self._build_agent_config(request)

        try:
            response = await client.post(
                "/api/agents",
                json={
                    "task": request.prompt,
                    "config": agent_config,
                    "workspace": request.context.get("workspace_path"),
                    "max_iterations": request.max_steps,
                },
            )
            response.raise_for_status()

            data = response.json()
            task_id = data.get("agent_id") or data.get("id")

            if not task_id:
                raise ExternalAgentError(
                    "OpenHands did not return task ID",
                    adapter_name=self.adapter_name,
                )

            # Track locally
            self._active_tasks[task_id] = {
                "request": request,
                "status": TaskStatus.RUNNING,
                "started_at": datetime.now(timezone.utc),
            }

            self._tasks_submitted += 1
            self._record_success()
            self._emit_event(
                "external_agent_task_submitted",
                {
                    "adapter": self.adapter_name,
                    "task_id": task_id,
                },
            )

            return task_id

        except (RuntimeError, OSError, ConnectionError, ValueError) as e:
            self._record_failure(e)
            logger.error("OpenHands task submission failed: %s", e)
            raise ExternalAgentError(
                "Task submission failed",
                adapter_name=self.adapter_name,
            ) from e

    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get status of an OpenHands task.

        Args:
            task_id: Task ID to check.

        Returns:
            Current task status.

        Raises:
            TaskNotFoundError: If task doesn't exist.
        """
        client = await self._ensure_client()

        try:
            response = await client.get(f"/api/agents/{task_id}/status")
            response.raise_for_status()

            data = response.json()
            status_str = data.get("status", "running")
            status = _STATUS_MAP.get(status_str, TaskStatus.RUNNING)

            # Update local tracking
            if task_id in self._active_tasks:
                self._active_tasks[task_id]["status"] = status

            return status

        except (RuntimeError, OSError, ConnectionError, ValueError) as e:
            if "404" in str(e) or "not found" in str(e).lower():
                raise TaskNotFoundError(
                    f"Task {task_id} not found",
                    adapter_name=self.adapter_name,
                    task_id=task_id,
                )
            logger.error("OpenHands status check failed: %s", e)
            raise

    async def get_task_result(self, task_id: str) -> TaskResult:
        """Get result of a completed OpenHands task.

        Args:
            task_id: Task ID to get result for.

        Returns:
            Task result with output, artifacts, and metrics.

        Raises:
            TaskNotFoundError: If task doesn't exist.
            TaskNotCompleteError: If task is still running.
        """
        # Check status first
        status = await self.get_task_status(task_id)
        if status == TaskStatus.RUNNING:
            raise TaskNotCompleteError(
                f"Task {task_id} is still running",
                adapter_name=self.adapter_name,
                task_id=task_id,
            )

        client = await self._ensure_client()

        try:
            response = await client.get(f"/api/agents/{task_id}")
            response.raise_for_status()

            data = response.json()

            # Parse OpenHands response format
            result = TaskResult(
                task_id=task_id,
                status=(TaskStatus.COMPLETED if data.get("success") else TaskStatus.FAILED),
                output=data.get("final_response", ""),
                artifacts=self._extract_artifacts(data),
                steps_executed=data.get("iterations", 0),
                tokens_used=data.get("tokens_used", 0),
                cost_usd=data.get("cost", 0.0),
                error=data.get("error"),
                logs=data.get("history", []),
            )

            # Get timing from local tracking if available
            local_task = self._active_tasks.get(task_id)
            if local_task:
                result.started_at = local_task.get("started_at")
                result.completed_at = datetime.now(timezone.utc)

            # Update metrics
            self._update_metrics(result)

            # Clean up local tracking
            self._active_tasks.pop(task_id, None)

            return result

        except TaskNotCompleteError:
            raise
        except (RuntimeError, OSError, ConnectionError, ValueError) as e:
            if "404" in str(e) or "not found" in str(e).lower():
                raise TaskNotFoundError(
                    f"Task {task_id} not found",
                    adapter_name=self.adapter_name,
                    task_id=task_id,
                )
            logger.error("OpenHands result retrieval failed: %s", e)
            raise

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running OpenHands task.

        Args:
            task_id: Task ID to cancel.

        Returns:
            True if cancelled, False otherwise.
        """
        client = await self._ensure_client()

        try:
            response = await client.post(f"/api/agents/{task_id}/stop")
            response.raise_for_status()

            self._active_tasks.pop(task_id, None)
            return True

        except (RuntimeError, OSError, ConnectionError, ValueError) as e:
            logger.error("OpenHands task cancellation failed: %s", e)
            return False

    async def health_check(self) -> HealthStatus:
        """Check OpenHands server health.

        Returns:
            Health status with latency and version info.
        """
        start = time.time()
        try:
            client = await self._ensure_client()
            response = await client.get("/api/health")
            response.raise_for_status()

            latency_ms = (time.time() - start) * 1000
            data = response.json()

            return HealthStatus(
                adapter_name=self.adapter_name,
                healthy=True,
                last_check=datetime.now(timezone.utc),
                response_time_ms=latency_ms,
                framework_version=data.get("version"),
                metadata={"sandbox": self._config.sandbox_type},
            )

        except (RuntimeError, OSError, ConnectionError, TimeoutError, ValueError):
            return HealthStatus(
                adapter_name=self.adapter_name,
                healthy=False,
                last_check=datetime.now(timezone.utc),
                response_time_ms=0.0,
                error="Health check failed",
            )

    async def stream_progress(self, task_id: str) -> AsyncIterator[TaskProgress]:
        """Stream progress from OpenHands.

        OpenHands supports WebSocket streaming for real-time updates.
        Falls back to polling if WebSocket is not available.

        Args:
            task_id: Task ID to stream.

        Yields:
            TaskProgress updates.
        """
        # Try WebSocket streaming first
        try:
            async for progress in self._stream_via_websocket(task_id):
                yield progress
            return
        except (RuntimeError, OSError, ConnectionError, TimeoutError, ValueError, ImportError) as e:
            logger.debug("WebSocket streaming failed, falling back to polling: %s", e)

        # Fall back to polling
        async for progress in super().stream_progress(task_id):
            yield progress

    async def _stream_via_websocket(self, task_id: str) -> AsyncIterator[TaskProgress]:
        """Stream via WebSocket if available."""
        try:
            import websockets
        except ImportError:
            raise ImportError("websockets package required for streaming")

        ws_url = self._config.base_url.replace("http", "ws")

        async with websockets.connect(f"{ws_url}/api/agents/{task_id}/stream") as ws:
            import json

            async for message in ws:
                data = json.loads(message)

                yield TaskProgress(
                    task_id=task_id,
                    status=TaskStatus.RUNNING,
                    current_step=data.get("iteration", 0),
                    total_steps=data.get("max_iterations"),
                    message=data.get("message", ""),
                )

                if data.get("status") in ("finished", "error", "stopped"):
                    break

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available OpenHands tools.

        Returns:
            List of tool definitions.
        """
        # OpenHands built-in tools
        return [
            {
                "name": "TerminalTool",
                "description": "Execute shell commands in the workspace",
                "permission": "computer_use.shell",
                "risk_level": "high",
            },
            {
                "name": "FileEditorTool",
                "description": "Read and write files in the workspace",
                "permission": "computer_use.file_write",
                "risk_level": "high",
            },
            {
                "name": "BrowserTool",
                "description": "Browse the web and interact with pages",
                "permission": "computer_use.browser",
                "risk_level": "high",
            },
            {
                "name": "TaskTrackerTool",
                "description": "Track task progress and status",
                "permission": "computer_use.read",
                "risk_level": "low",
            },
        ]

    def _build_agent_config(self, request: TaskRequest) -> dict[str, Any]:
        """Build OpenHands agent configuration from request.

        Args:
            request: Task request with permissions.

        Returns:
            OpenHands agent configuration dict.
        """
        config: dict[str, Any] = {
            "agent": "CodeActAgent",
            "timeout": int(request.timeout_seconds),
            "model": self._config.model,
        }

        # Map tool permissions to OpenHands configuration
        if ToolPermission.BROWSER_USE in request.tool_permissions:
            config["enable_browsing"] = True
        if ToolPermission.SHELL_EXECUTE in request.tool_permissions:
            config["enable_bash"] = True
        if ToolPermission.FILE_WRITE in request.tool_permissions:
            config["enable_file_edit"] = True

        return config

    def _extract_artifacts(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract artifacts from OpenHands response.

        Args:
            data: OpenHands response data.

        Returns:
            List of artifact dicts.
        """
        artifacts: list[dict[str, Any]] = []

        for file_change in data.get("file_changes", []):
            artifacts.append(
                {
                    "type": "file",
                    "path": file_change.get("path"),
                    "action": file_change.get("action"),
                    "content": file_change.get("content"),
                }
            )

        return artifacts

    async def close(self) -> None:
        """Close the adapter and release resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
