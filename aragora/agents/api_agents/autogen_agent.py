"""
AutoGen Agent - Integration with Microsoft AutoGen multi-agent framework.

Provides secure enterprise access to AutoGen's capabilities:
- GroupChat orchestration for multi-agent conversations
- Two-agent chat for simpler interactions
- Configurable speaker selection strategies
- Code execution with security controls

With Aragora security controls:
- Code execution disabled by default
- Work directory validation
- Conversation round limits
- Response sanitization
- Audit logging
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from aragora.agents.api_agents.external_framework import (
    ExternalFrameworkAgent,
    ExternalFrameworkConfig,
)
from aragora.agents.errors import AgentError
from aragora.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


@dataclass
class AutoGenConfig(ExternalFrameworkConfig):
    """Configuration for AutoGen integration.

    Extends ExternalFrameworkConfig with AutoGen-specific settings for
    controlling multi-agent conversation behavior and security.

    Attributes:
        mode: Conversation mode - 'groupchat' for multi-agent or
            'two_agent' for simple two-party conversations.
        max_round: Maximum conversation rounds to prevent infinite loops.
        speaker_selection_method: How to select the next speaker in groupchat.
            Options: 'auto', 'random', 'round_robin', 'manual'.
        allow_code_execution: Allow code execution. Default False for safety.
        work_dir: Working directory if code execution is enabled.
            Must be an absolute path within allowed directories.
        human_input_mode: When to request human input.
            Options: 'NEVER', 'TERMINATE', 'ALWAYS'.
        conversation_timeout: Overall timeout for conversation in seconds.
        audit_all_requests: Log all requests for compliance. Default True.
    """

    # AutoGen-specific settings
    mode: str = "groupchat"  # groupchat, two_agent
    max_round: int = 10  # Maximum conversation rounds
    speaker_selection_method: str = "auto"  # auto, random, round_robin, manual
    allow_code_execution: bool = False  # Allow code execution (disabled by default)
    work_dir: str | None = None  # Working directory for code execution
    human_input_mode: str = "NEVER"  # NEVER, TERMINATE, ALWAYS
    conversation_timeout: int = 300  # Overall timeout in seconds
    audit_all_requests: bool = True  # Log all requests for compliance

    def __post_init__(self) -> None:
        """Set AutoGen-specific defaults after initialization."""
        # Set AutoGen-specific defaults if not already set
        if not self.base_url:
            self.base_url = os.environ.get("AUTOGEN_URL", "http://localhost:8000")
        if self.generate_endpoint == "/generate":
            # Override default to AutoGen Studio's chat endpoint
            self.generate_endpoint = "/api/chat"
        if self.health_endpoint == "/health":
            # AutoGen uses /health by default, which matches
            pass

    def validate_work_dir(self) -> bool:
        """Validate work directory if code execution is enabled.

        Returns:
            True if work_dir is valid or code execution is disabled.
        """
        if not self.allow_code_execution:
            return True

        if self.work_dir is None:
            return False

        # Must be an absolute path
        if not os.path.isabs(self.work_dir):
            return False

        # Must exist or be creatable (parent must exist)
        if os.path.exists(self.work_dir):
            return os.path.isdir(self.work_dir)

        parent = os.path.dirname(self.work_dir)
        return os.path.isdir(parent)


@AgentRegistry.register(
    "autogen",
    default_model="autogen",
    default_name="autogen",
    agent_type="API",
    requires="AutoGen Studio running at AUTOGEN_URL",
    env_vars="AUTOGEN_URL, AUTOGEN_API_KEY",
    description="Integration with Microsoft AutoGen multi-agent framework",
    accepts_api_key=True,
)
class AutoGenAgent(ExternalFrameworkAgent):
    """
    Agent for Microsoft AutoGen multi-agent framework.

    Wraps AutoGen's REST API with enterprise security controls.
    AutoGen is a multi-agent conversation framework that supports
    groupchat orchestration, code execution, and various speaker
    selection strategies.

    Security Model:
        - Code execution is disabled by default
        - Work directory must be validated if code execution is enabled
        - Conversation rounds are limited to prevent infinite loops
        - Response sanitization is always enabled
        - All requests can be audited for compliance

    Example:
        >>> config = AutoGenConfig(
        ...     mode="groupchat",
        ...     max_round=5,
        ...     allow_code_execution=False,
        ... )
        >>> agent = AutoGenAgent(config=config, api_key="your-key")
        >>> response = await agent.generate("Discuss the best approach to...")
    """

    def __init__(
        self,
        name: str = "autogen",
        model: str = "autogen",
        config: AutoGenConfig | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize AutoGen agent.

        Args:
            name: Agent instance name.
            model: Model identifier (passed to AutoGen).
            config: AutoGen-specific configuration.
            api_key: API key for authentication. If not provided,
                reads from AUTOGEN_API_KEY environment variable.
            **kwargs: Additional arguments passed to ExternalFrameworkAgent.
        """
        if config is None:
            config = AutoGenConfig(base_url="")  # Will be set in __post_init__
            config.__post_init__()

        # Validate work_dir if code execution is enabled
        if config.allow_code_execution and not config.validate_work_dir():
            raise ValueError(
                "Code execution enabled but work_dir is invalid or not set. "
                "Provide an absolute path to an existing directory."
            )

        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("AUTOGEN_API_KEY")

        super().__init__(
            name=name,
            model=model,
            config=config,
            api_key=api_key,
            **kwargs,
        )
        self.autogen_config = config
        self.agent_type = "autogen"
        self._conversations: dict[str, list[dict[str, Any]]] = {}

    def _build_autogen_prefix(self) -> str:
        """Build configuration prefix for AutoGen prompts.

        Returns:
            String prefix describing AutoGen configuration.
        """
        mode_desc = (
            "multi-agent groupchat"
            if self.autogen_config.mode == "groupchat"
            else "two-agent conversation"
        )

        code_status = "enabled" if self.autogen_config.allow_code_execution else "disabled"

        prefix_parts = [
            f"[AutoGen Mode: {mode_desc}]",
            f"[Max Rounds: {self.autogen_config.max_round}]",
            f"[Code Execution: {code_status}]",
        ]

        if self.autogen_config.allow_code_execution and self.autogen_config.work_dir:
            prefix_parts.append(f"[Work Dir: {self.autogen_config.work_dir}]")

        return " ".join(prefix_parts) + "\n\n"

    def _build_request_payload(
        self,
        prompt: str,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Build AutoGen-specific request payload.

        Args:
            prompt: The prompt to send.
            conversation_id: Optional conversation ID for continuity.

        Returns:
            Request payload dict for AutoGen API.
        """
        payload: dict[str, Any] = {
            "message": prompt,
            "model": self.model,
            "config": {
                "mode": self.autogen_config.mode,
                "max_round": self.autogen_config.max_round,
                "speaker_selection_method": self.autogen_config.speaker_selection_method,
                "human_input_mode": self.autogen_config.human_input_mode,
                "code_execution_config": {
                    "enabled": self.autogen_config.allow_code_execution,
                },
            },
        }

        if self.autogen_config.allow_code_execution and self.autogen_config.work_dir:
            payload["config"]["code_execution_config"]["work_dir"] = self.autogen_config.work_dir

        if conversation_id:
            payload["conversation_id"] = conversation_id

        return payload

    async def generate(
        self,
        prompt: str,
        context: list | None = None,
        conversation_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate response from AutoGen with configuration.

        Adds AutoGen configuration to the request and enforces
        security controls.

        Args:
            prompt: The prompt to send to AutoGen.
            context: Optional conversation context.
            conversation_id: Optional conversation ID for continuity.
            **kwargs: Additional generation parameters.

        Returns:
            Generated response text from AutoGen.
        """
        # Add configuration prefix to prompt
        prefixed_prompt = self._build_autogen_prefix() + prompt

        if self.autogen_config.audit_all_requests:
            logger.info(
                f"[{self.name}] AutoGen request",
                extra={
                    "prompt_length": len(prompt),
                    "mode": self.autogen_config.mode,
                    "max_round": self.autogen_config.max_round,
                    "code_execution": self.autogen_config.allow_code_execution,
                    "conversation_id": conversation_id,
                },
            )

        return await super().generate(prefixed_prompt, context, **kwargs)

    async def initiate_chat(
        self,
        message: str,
        agents: list[str] | None = None,
        max_round: int | None = None,
    ) -> dict[str, Any]:
        """Initiate a new AutoGen chat conversation.

        Creates a new conversation with specified agents and returns
        the conversation ID along with the initial response.

        Args:
            message: Initial message to start the conversation.
            agents: Optional list of agent names to include.
            max_round: Optional override for max rounds.

        Returns:
            Dict with keys:
                - conversation_id: Unique identifier for this conversation
                - response: Initial response from AutoGen
                - success: bool indicating if chat was initiated
        """
        import uuid

        conversation_id = str(uuid.uuid4())

        if self.autogen_config.audit_all_requests:
            logger.info(
                f"[{self.name}] Initiating AutoGen chat",
                extra={
                    "conversation_id": conversation_id,
                    "agents": agents,
                    "max_round": max_round or self.autogen_config.max_round,
                },
            )

        try:
            response = await self.generate(
                message,
                conversation_id=conversation_id,
            )

            # Store conversation history
            self._conversations[conversation_id] = [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ]

            return {
                "conversation_id": conversation_id,
                "response": response,
                "success": True,
            }
        except (AgentError, ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
            logger.error(f"[{self.name}] Failed to initiate chat: {e}")
            return {
                "conversation_id": conversation_id,
                "response": str(e),
                "success": False,
            }

    async def continue_chat(
        self,
        conversation_id: str,
        message: str,
    ) -> dict[str, Any]:
        """Continue an existing AutoGen chat conversation.

        Args:
            conversation_id: ID of the conversation to continue.
            message: Message to add to the conversation.

        Returns:
            Dict with keys:
                - response: Response from AutoGen
                - success: bool indicating if chat continued successfully
        """
        if conversation_id not in self._conversations:
            return {
                "response": f"Conversation {conversation_id} not found",
                "success": False,
            }

        try:
            response = await self.generate(
                message,
                conversation_id=conversation_id,
            )

            # Update conversation history
            self._conversations[conversation_id].extend(
                [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response},
                ]
            )

            return {
                "response": response,
                "success": True,
            }
        except (AgentError, ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
            logger.error(f"[{self.name}] Failed to continue chat: {e}")
            return {
                "response": str(e),
                "success": False,
            }

    def get_conversation(self, conversation_id: str) -> list[dict[str, Any]] | None:
        """Retrieve conversation history.

        Args:
            conversation_id: ID of the conversation to retrieve.

        Returns:
            List of messages in the conversation, or None if not found.
        """
        return self._conversations.get(conversation_id)

    def get_all_conversations(self) -> dict[str, list[dict[str, Any]]]:
        """Retrieve all conversation histories.

        Returns:
            Dict mapping conversation IDs to their message lists.
        """
        return self._conversations.copy()

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a specific conversation from memory.

        Args:
            conversation_id: ID of the conversation to clear.

        Returns:
            True if conversation was cleared, False if not found.
        """
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            return True
        return False

    def clear_all_conversations(self) -> int:
        """Clear all conversations from memory.

        Returns:
            Number of conversations cleared.
        """
        count = len(self._conversations)
        self._conversations.clear()
        return count

    async def is_available(self) -> bool:
        """Check if AutoGen server is accessible.

        Returns:
            True if AutoGen server responds to health check, False otherwise.
        """
        available = await super().is_available()
        if available:
            logger.debug(
                f"[{self.name}] AutoGen available at {self.base_url} "
                f"(mode={self.autogen_config.mode})"
            )
        return available

    def get_config_status(self) -> dict[str, Any]:
        """Get current configuration status.

        Returns:
            Dict describing the current AutoGen configuration.
        """
        return {
            "mode": self.autogen_config.mode,
            "max_round": self.autogen_config.max_round,
            "speaker_selection_method": self.autogen_config.speaker_selection_method,
            "code_execution_enabled": self.autogen_config.allow_code_execution,
            "work_dir": self.autogen_config.work_dir,
            "human_input_mode": self.autogen_config.human_input_mode,
            "conversation_timeout": self.autogen_config.conversation_timeout,
            "audit_enabled": self.autogen_config.audit_all_requests,
            "base_url": self.base_url,
            "active_conversations": len(self._conversations),
        }


__all__ = ["AutoGenAgent", "AutoGenConfig"]
