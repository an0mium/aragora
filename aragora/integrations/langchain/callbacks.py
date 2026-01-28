"""
LangChain Callbacks for Aragora.

Provides callback handlers for streaming debate progress
and integrating with LangChain's callback system.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

# LangChain imports with fallback
try:
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema import AgentAction, AgentFinish, LLMResult

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

    class BaseCallbackHandler:  # type: ignore
        """Stub BaseCallbackHandler when LangChain not installed."""

        pass

    class LLMResult:  # type: ignore
        """Stub LLMResult."""

        pass

    class AgentAction:  # type: ignore
        """Stub AgentAction."""

        pass

    class AgentFinish:  # type: ignore
        """Stub AgentFinish."""

        pass


class AragoraCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler for Aragora integration.

    Streams events from LangChain to Aragora for:
    - Debate progress tracking
    - Token usage monitoring
    - Error reporting

    Example:
        handler = AragoraCallbackHandler(
            aragora_url="http://localhost:8080",
            debate_id="debate-123",
        )

        llm = ChatOpenAI(callbacks=[handler])
    """

    aragora_url: str = "http://localhost:8080"
    api_token: Optional[str] = None
    debate_id: Optional[str] = None
    session_id: Optional[str] = None
    timeout_seconds: float = 10.0

    def __init__(
        self,
        aragora_url: str = "http://localhost:8080",
        api_token: Optional[str] = None,
        debate_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the Aragora callback handler.

        Args:
            aragora_url: Base URL for Aragora API
            api_token: Optional API token
            debate_id: Optional debate ID to associate events with
            session_id: Optional session ID for tracking
            **kwargs: Additional handler arguments
        """
        super().__init__(**kwargs)
        self.aragora_url = aragora_url
        self.api_token = api_token
        self.debate_id = debate_id
        self.session_id = session_id
        self._events: List[Dict[str, Any]] = []

    def _send_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Send event to Aragora (fire and forget)."""

        event = {
            "type": event_type,
            "debate_id": self.debate_id,
            "session_id": self.session_id,
            "data": data,
        }
        self._events.append(event)

        # In production, this would send to Aragora's event endpoint
        # For now, just log the event
        logger.debug(f"[AragoraCallback] {event_type}: {data}")

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts."""
        self._send_event(
            "llm_start",
            {
                "run_id": str(run_id),
                "model": serialized.get("name", "unknown"),
                "prompt_count": len(prompts),
            },
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM ends."""
        token_usage = {}
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})

        self._send_event(
            "llm_end",
            {
                "run_id": str(run_id),
                "token_usage": token_usage,
            },
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called on LLM error."""
        self._send_event(
            "llm_error",
            {
                "run_id": str(run_id),
                "error": str(error),
            },
        )

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain starts."""
        self._send_event(
            "chain_start",
            {
                "run_id": str(run_id),
                "chain_type": serialized.get("name", "unknown"),
            },
        )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain ends."""
        self._send_event(
            "chain_end",
            {
                "run_id": str(run_id),
                "output_keys": list(outputs.keys()) if outputs else [],
            },
        )

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool starts."""
        self._send_event(
            "tool_start",
            {
                "run_id": str(run_id),
                "tool": serialized.get("name", "unknown"),
            },
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool ends."""
        self._send_event(
            "tool_end",
            {
                "run_id": str(run_id),
                "output_length": len(output) if output else 0,
            },
        )

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called on agent action."""
        self._send_event(
            "agent_action",
            {
                "run_id": str(run_id),
                "tool": action.tool if hasattr(action, "tool") else "unknown",
            },
        )

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent finishes."""
        self._send_event(
            "agent_finish",
            {
                "run_id": str(run_id),
            },
        )

    def get_events(self) -> List[Dict[str, Any]]:
        """Get all recorded events."""
        return self._events.copy()

    def clear_events(self) -> None:
        """Clear recorded events."""
        self._events.clear()
