"""
OpenClaw Agent - Integration with OpenClaw personal AI assistant.

Provides secure enterprise access to OpenClaw's capabilities:
- Task execution (shell, file operations)
- Channel integrations (Slack, Teams, etc.)
- Voice and browser control

With Aragora security controls:
- RBAC permission checking
- Audit logging
- Rate limiting
- Response sanitization
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from aragora.agents.api_agents.external_framework import (
    ExternalFrameworkAgent,
    ExternalFrameworkConfig,
)
from aragora.agents.errors import AgentError
from aragora.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


@dataclass
class OpenClawConfig(ExternalFrameworkConfig):
    """Configuration for OpenClaw integration.

    Extends ExternalFrameworkConfig with OpenClaw-specific settings for
    controlling which capabilities are exposed to the agent.

    Attributes:
        gateway_mode: Deployment mode - 'local' for local instance,
            'cloud' for OpenClaw cloud service.
        enable_shell: Allow shell command execution. Default False for safety.
        enable_file_ops: Allow file operations (read/write/delete). Default False.
        enable_browser: Allow browser control and automation. Default False.
        allowed_channels: List of allowed messaging channels for send_message().
            Empty list means all channels are blocked.
        task_timeout: Timeout for task execution in seconds.
        audit_all_requests: Log all requests for compliance. Default True.
    """

    # OpenClaw-specific settings
    gateway_mode: str = "local"  # local, cloud
    enable_shell: bool = False  # Allow shell command execution
    enable_file_ops: bool = False  # Allow file operations
    enable_browser: bool = False  # Allow browser control
    allowed_channels: list[str] = field(default_factory=list)  # Allowed messaging channels
    task_timeout: int = 300  # Task execution timeout
    audit_all_requests: bool = True  # Log all requests for compliance

    def __post_init__(self) -> None:
        """Set OpenClaw-specific defaults after initialization."""
        # Set OpenClaw-specific defaults if not already set
        if not self.base_url:
            self.base_url = os.environ.get("OPENCLAW_URL", "http://localhost:3000")
        if self.generate_endpoint == "/generate":
            # Override default to OpenClaw's chat endpoint
            self.generate_endpoint = "/api/chat"
        if self.health_endpoint == "/health":
            # OpenClaw uses /health by default, which matches
            pass


@AgentRegistry.register(
    "openclaw",
    default_model="openclaw",
    default_name="openclaw",
    agent_type="API",
    requires="OpenClaw running at OPENCLAW_URL",
    env_vars="OPENCLAW_URL, OPENCLAW_API_KEY",
    description="Integration with OpenClaw personal AI assistant",
    accepts_api_key=True,
)
class OpenClawAgent(ExternalFrameworkAgent):
    """
    Agent for OpenClaw personal AI assistant.

    Wraps OpenClaw's gateway API with enterprise security controls.
    OpenClaw is a personal AI assistant that can execute tasks, control
    browsers, manage files, and integrate with messaging platforms.

    Security Model:
        - All dangerous capabilities (shell, files, browser) are disabled by default
        - Channels must be explicitly whitelisted
        - All requests can be audited for compliance
        - Response sanitization is always enabled

    Example:
        >>> config = OpenClawConfig(
        ...     enable_shell=True,  # Allow shell commands
        ...     allowed_channels=["slack"],  # Allow Slack messaging
        ... )
        >>> agent = OpenClawAgent(config=config, api_key="your-key")
        >>> response = await agent.generate("Check disk usage")
    """

    def __init__(
        self,
        name: str = "openclaw",
        model: str = "openclaw",
        config: OpenClawConfig | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenClaw agent.

        Args:
            name: Agent instance name.
            model: Model identifier (passed to OpenClaw).
            config: OpenClaw-specific configuration.
            api_key: API key for authentication. If not provided,
                reads from OPENCLAW_API_KEY environment variable.
            **kwargs: Additional arguments passed to ExternalFrameworkAgent.
        """
        if config is None:
            config = OpenClawConfig(base_url="")  # Will be set in __post_init__
            config.__post_init__()

        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("OPENCLAW_API_KEY")

        super().__init__(
            name=name,
            model=model,
            config=config,
            api_key=api_key,
            **kwargs,
        )
        self.openclaw_config = config
        self.agent_type = "openclaw"

    def _get_enabled_capabilities(self) -> list[str]:
        """Get list of enabled capabilities.

        Returns:
            List of capability names that are enabled.
        """
        capabilities = []
        if self.openclaw_config.enable_shell:
            capabilities.append("shell")
        if self.openclaw_config.enable_file_ops:
            capabilities.append("files")
        if self.openclaw_config.enable_browser:
            capabilities.append("browser")
        return capabilities

    def _build_capability_prefix(self) -> str:
        """Build capability restriction prefix for prompts.

        Returns:
            String prefix describing allowed/disallowed capabilities.
        """
        capabilities = self._get_enabled_capabilities()
        if capabilities:
            return f"[Allowed capabilities: {', '.join(capabilities)}]\n\n"
        return "[No dangerous capabilities allowed - shell, files, and browser access are disabled]\n\n"

    async def generate(
        self,
        prompt: str,
        context: list | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate response from OpenClaw with capability filtering.

        Adds capability restrictions to the prompt to ensure OpenClaw
        respects the security configuration.

        Args:
            prompt: The prompt to send to OpenClaw.
            context: Optional conversation context.
            **kwargs: Additional generation parameters.

        Returns:
            Generated response text from OpenClaw.
        """
        # Add capability restrictions to prompt
        prefixed_prompt = self._build_capability_prefix() + prompt

        if self.openclaw_config.audit_all_requests:
            logger.info(
                f"[{self.name}] OpenClaw request",
                extra={
                    "prompt_length": len(prompt),
                    "capabilities": self._get_enabled_capabilities(),
                    "gateway_mode": self.openclaw_config.gateway_mode,
                },
            )

        return await super().generate(prefixed_prompt, context, **kwargs)

    async def execute_task(
        self,
        task: str,
        context: list | None = None,
        require_capabilities: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute a task via OpenClaw with full capabilities.

        This method is for when task execution is explicitly allowed
        and the caller wants structured results.

        Args:
            task: The task description to execute.
            context: Optional conversation context.
            require_capabilities: Capabilities required for this task.
                If any required capability is not enabled, returns error.

        Returns:
            Dict with keys:
                - success: bool indicating if task completed
                - output: task output or error message
                - agent: agent name
                - capabilities_used: list of capabilities that were available
        """
        # Check required capabilities
        if require_capabilities:
            enabled = set(self._get_enabled_capabilities())
            required = set(require_capabilities)
            missing = required - enabled
            if missing:
                return {
                    "success": False,
                    "output": f"Required capabilities not enabled: {', '.join(missing)}",
                    "agent": self.name,
                    "capabilities_used": [],
                }

        try:
            result = await self.generate(task, context)
            return {
                "success": True,
                "output": result,
                "agent": self.name,
                "capabilities_used": self._get_enabled_capabilities(),
            }
        except (AgentError, ValueError, KeyError, TypeError, RuntimeError, OSError, Exception) as e:
            logger.error(f"[{self.name}] Task execution failed: {e}")
            return {
                "success": False,
                "output": "Task execution failed",
                "agent": self.name,
                "capabilities_used": [],
            }

    async def send_message(
        self,
        channel: str,
        message: str,
        recipient: str | None = None,
    ) -> dict[str, Any]:
        """Send message via OpenClaw channel integration.

        Sends a message through OpenClaw to an external messaging platform.
        The channel must be in the allowed_channels list.

        Args:
            channel: Target channel (e.g., 'slack', 'teams', 'telegram').
            message: Message content to send.
            recipient: Optional recipient identifier (user, channel, or group).

        Returns:
            Dict with keys:
                - success: bool indicating if message was sent
                - output: response or error message
                - channel: the target channel
                - error: error message if success is False
        """
        # Validate channel is allowed
        if self.openclaw_config.allowed_channels:
            if channel.lower() not in [c.lower() for c in self.openclaw_config.allowed_channels]:
                logger.warning(
                    f"[{self.name}] Channel '{channel}' not in allowed list: "
                    f"{self.openclaw_config.allowed_channels}"
                )
                return {
                    "success": False,
                    "error": f"Channel '{channel}' not allowed. "
                    f"Allowed channels: {self.openclaw_config.allowed_channels}",
                    "channel": channel,
                }
        else:
            # No channels allowed if list is empty
            return {
                "success": False,
                "error": "No messaging channels are allowed. Configure allowed_channels.",
                "channel": channel,
            }

        # Build the send message prompt
        prompt = f"Send message to {channel}"
        if recipient:
            prompt += f" recipient {recipient}"
        prompt += f": {message}"

        try:
            result = await self.generate(prompt)
            return {
                "success": True,
                "output": result,
                "channel": channel,
            }
        except (AgentError, ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
            logger.error(f"[{self.name}] Failed to send message to {channel}: {e}")
            return {
                "success": False,
                "error": "Message send failed",
                "channel": channel,
            }

    async def is_available(self) -> bool:
        """Check if OpenClaw server is accessible.

        Returns:
            True if OpenClaw server responds to health check, False otherwise.
        """
        available = await super().is_available()
        if available:
            logger.debug(
                f"[{self.name}] OpenClaw available at {self.base_url} "
                f"(mode={self.openclaw_config.gateway_mode})"
            )
        return available

    def get_capability_status(self) -> dict[str, Any]:
        """Get current capability configuration status.

        Returns:
            Dict describing which capabilities are enabled/disabled.
        """
        return {
            "gateway_mode": self.openclaw_config.gateway_mode,
            "shell_enabled": self.openclaw_config.enable_shell,
            "file_ops_enabled": self.openclaw_config.enable_file_ops,
            "browser_enabled": self.openclaw_config.enable_browser,
            "allowed_channels": self.openclaw_config.allowed_channels,
            "audit_enabled": self.openclaw_config.audit_all_requests,
            "base_url": self.base_url,
        }


__all__ = ["OpenClawAgent", "OpenClawConfig"]
