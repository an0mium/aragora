"""
Session Management Mixin for Chat Platform Connectors.

Contains methods for integrating with the debate session management system.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import SendMessageResponse

logger = logging.getLogger(__name__)


class SessionMixin:
    """
    Mixin providing session management integration for chat connectors.

    Includes:
    - Session creation and retrieval
    - Debate linking
    - Result routing
    """

    # These methods are expected from the base class/other mixins
    send_message: Any

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the platform identifier."""
        ...

    # ==========================================================================
    # Session Management Integration
    # ==========================================================================

    def _get_session_manager(self) -> Any:
        """
        Get the debate session manager (lazy initialization).

        Returns:
            DebateSessionManager instance, or None if unavailable
        """
        if not hasattr(self, "_session_manager"):
            try:
                from aragora.connectors.debate_session import get_debate_session_manager

                self._session_manager = get_debate_session_manager()
            except ImportError:
                logger.warning("Session manager not available")
                self._session_manager = None
        return self._session_manager

    async def get_or_create_session(
        self,
        user_id: str,
        context: dict[str, Any] | None = None,
    ) -> Any | None:
        """
        Get or create a debate session for a user on this platform.

        This method integrates the chat connector with the session management
        system, enabling cross-channel session tracking and debate routing.

        Args:
            user_id: Platform-specific user identifier
            context: Optional context metadata for the session

        Returns:
            DebateSession if session manager is available, None otherwise

        Example:
            session = await slack.get_or_create_session("U123456")
            if session:
                await session_manager.link_debate(session.session_id, "debate-abc")
        """
        manager = self._get_session_manager()
        if not manager:
            return None

        session_context = {
            "platform": self.platform_name,
            **(context or {}),
        }

        return await manager.get_or_create_session(
            channel=self.platform_name,
            user_id=user_id,
            context=session_context,
        )

    async def link_debate_to_session(
        self,
        user_id: str,
        debate_id: str,
        context: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Create or get a session and link it to a debate.

        Convenience method that combines session creation with debate linking.
        Used when starting a debate from a chat platform.

        Args:
            user_id: Platform-specific user identifier
            debate_id: Debate to link to the session
            context: Optional session context

        Returns:
            Session ID if successful, None if session manager unavailable

        Example:
            session_id = await telegram.link_debate_to_session(
                user_id="user123",
                debate_id="debate-abc",
                context={"message_id": "msg123", "chat_id": "chat456"}
            )
        """
        manager = self._get_session_manager()
        if not manager:
            return None

        session = await self.get_or_create_session(user_id, context)
        if not session:
            return None

        await manager.link_debate(session.session_id, debate_id)
        logger.debug(
            "Linked debate %s to %s session for user %s", debate_id[:8], self.platform_name, user_id
        )
        return session.session_id

    async def find_sessions_for_debate(self, debate_id: str) -> list[Any]:
        """
        Find all sessions on this platform linked to a debate.

        Used for routing debate results back to the originating channel.

        Args:
            debate_id: The debate ID to search for

        Returns:
            List of sessions for this debate on this platform
        """
        manager = self._get_session_manager()
        if not manager:
            return []

        all_sessions = await manager.find_sessions_for_debate(debate_id)
        return [s for s in all_sessions if s.channel == self.platform_name]

    async def route_debate_result(
        self,
        debate_id: str,
        result: str,
        channel_id: str | None = None,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> list[SendMessageResponse]:
        """
        Route a debate result to all sessions on this platform.

        Finds all sessions linked to the debate and sends the result to
        each session's channel/thread.

        Args:
            debate_id: The completed debate ID
            result: The debate result/consensus text
            channel_id: Override channel ID (uses session context if not provided)
            thread_id: Override thread ID (uses session context if not provided)
            **kwargs: Additional arguments passed to send_message

        Returns:
            List of SendMessageResponse for each message sent

        Example:
            responses = await slack.route_debate_result(
                debate_id="debate-abc",
                result="The consensus is to use token bucket algorithm..."
            )
        """
        sessions = await self.find_sessions_for_debate(debate_id)
        responses = []

        for session in sessions:
            # Get channel/thread from session context
            ctx = session.context or {}
            target_channel = channel_id or ctx.get("channel_id") or ctx.get("chat_id")
            target_thread = thread_id or ctx.get("thread_id") or ctx.get("message_id")

            if not target_channel:
                logger.warning(
                    "No channel found in session %s context, skipping", session.session_id
                )
                continue

            try:
                response = await self.send_message(
                    channel_id=target_channel,
                    text=result,
                    thread_id=target_thread,
                    **kwargs,
                )
                responses.append(response)
                logger.debug(
                    "Routed debate %s result to %s channel %s",
                    debate_id[:8],
                    self.platform_name,
                    target_channel,
                )
            except (RuntimeError, OSError, ValueError, Exception) as e:
                logger.error(
                    "Failed to route debate result to %s channel %s: %s",
                    self.platform_name,
                    target_channel,
                    e,
                )

        return responses


__all__ = ["SessionMixin"]
