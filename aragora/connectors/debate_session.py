"""Debate session management for multi-channel conversations.

Provides a higher-level interface for managing debate sessions across
channels (Slack, Telegram, WhatsApp, API). Builds on the session store
infrastructure to enable:

- Cross-channel session tracking and handoff
- Linking debates to user sessions
- Session lifecycle management

Usage:
    from aragora.connectors.debate_session import DebateSessionManager

    manager = DebateSessionManager()

    # Create a session for a user on Telegram
    session = await manager.create_session("telegram", "user123")

    # Link an active debate to the session
    await manager.link_debate(session.session_id, "debate-abc")

    # Find all sessions for a debate (for routing results)
    sessions = await manager.find_sessions_for_debate("debate-abc")

    # Handoff session to a different channel
    new_session = await manager.handoff(session.session_id, "slack")
"""

from __future__ import annotations

import logging
import secrets
from typing import Optional

from aragora.server.session_store import (
    DebateSession,
    get_session_store,
)

logger = logging.getLogger(__name__)


class DebateSessionManager:
    """Manage debate sessions across channels.

    Thread-safe session manager for multi-channel debate conversations.
    Integrates with the session store (in-memory or Redis) for persistence.
    """

    def __init__(self):
        """Initialize the session manager."""
        self._store = get_session_store()

    async def create_session(
        self,
        channel: str,
        user_id: str,
        context: Optional[dict] = None,
    ) -> DebateSession:
        """Create a new debate session.

        Args:
            channel: Source channel ("slack", "telegram", "api", "whatsapp")
            user_id: User identifier from the channel
            context: Optional additional context for the session

        Returns:
            The created DebateSession
        """
        # Generate unique session ID
        random_suffix = secrets.token_hex(4)
        session_id = f"{channel}:{user_id}:{random_suffix}"

        session = DebateSession(
            session_id=session_id,
            channel=channel,
            user_id=user_id,
            context=context or {},
        )

        self._store.set_debate_session(session)
        logger.debug(f"Created debate session: {session_id}")
        return session

    async def get_session(self, session_id: str) -> Optional[DebateSession]:
        """Get a session by ID.

        Args:
            session_id: The session ID

        Returns:
            The DebateSession if found, None otherwise
        """
        return self._store.get_debate_session(session_id)

    async def update_session(self, session: DebateSession) -> None:
        """Update an existing session.

        Args:
            session: The session to update
        """
        self._store.set_debate_session(session)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deleted, False if not found
        """
        result = self._store.delete_debate_session(session_id)
        if result:
            logger.debug(f"Deleted debate session: {session_id}")
        return result

    async def link_debate(self, session_id: str, debate_id: str) -> bool:
        """Link a debate to a session.

        Args:
            session_id: The session ID
            debate_id: The debate ID to link

        Returns:
            True if successful, False if session not found
        """
        session = self._store.get_debate_session(session_id)
        if not session:
            logger.warning(f"Session not found for linking: {session_id}")
            return False

        old_debate = session.debate_id
        session.link_debate(debate_id)
        self._store.set_debate_session(session)

        logger.debug(
            f"Linked debate {debate_id[:8] if debate_id else 'None'} to session {session_id}"
            + (f" (was {old_debate[:8]})" if old_debate else "")
        )
        return True

    async def unlink_debate(self, session_id: str) -> bool:
        """Unlink the current debate from a session.

        Args:
            session_id: The session ID

        Returns:
            True if successful, False if session not found
        """
        session = self._store.get_debate_session(session_id)
        if not session:
            return False

        session.unlink_debate()
        self._store.set_debate_session(session)
        logger.debug(f"Unlinked debate from session {session_id}")
        return True

    async def find_sessions_for_user(
        self, user_id: str, channel: Optional[str] = None
    ) -> list[DebateSession]:
        """Find all sessions for a user.

        Args:
            user_id: The user ID
            channel: Optional channel filter

        Returns:
            List of matching sessions
        """
        return self._store.find_sessions_by_user(user_id, channel)

    async def find_sessions_for_debate(self, debate_id: str) -> list[DebateSession]:
        """Find all sessions linked to a debate.

        Args:
            debate_id: The debate ID

        Returns:
            List of sessions linked to the debate
        """
        return self._store.find_sessions_by_debate(debate_id)

    async def get_or_create_session(
        self,
        channel: str,
        user_id: str,
        context: Optional[dict] = None,
    ) -> DebateSession:
        """Get an existing session or create a new one.

        Finds the most recent active session for the user on the channel,
        or creates a new one if none exists.

        Args:
            channel: Source channel
            user_id: User identifier
            context: Optional context for new session

        Returns:
            The existing or newly created session
        """
        # Look for existing session
        sessions = self._store.find_sessions_by_user(user_id, channel)
        if sessions:
            # Return most recently active session
            sessions.sort(key=lambda s: s.last_active, reverse=True)
            session = sessions[0]
            session.touch()
            self._store.set_debate_session(session)
            return session

        # Create new session
        return await self.create_session(channel, user_id, context)

    async def handoff(
        self,
        session_id: str,
        new_channel: str,
        preserve_debate: bool = True,
    ) -> Optional[DebateSession]:
        """Handoff a session to a different channel.

        Creates a new session on the target channel, optionally preserving
        the debate link and context.

        Args:
            session_id: The source session ID
            new_channel: The target channel
            preserve_debate: Whether to preserve the debate link

        Returns:
            The new session on the target channel, or None if source not found
        """
        old_session = self._store.get_debate_session(session_id)
        if not old_session:
            logger.warning(f"Session not found for handoff: {session_id}")
            return None

        # Create new session on target channel
        random_suffix = secrets.token_hex(4)
        new_session_id = f"{new_channel}:{old_session.user_id}:{random_suffix}"

        new_session = DebateSession(
            session_id=new_session_id,
            channel=new_channel,
            user_id=old_session.user_id,
            debate_id=old_session.debate_id if preserve_debate else None,
            context={
                **old_session.context,
                "handoff_from": old_session.session_id,
                "handoff_channel": old_session.channel,
            },
        )

        self._store.set_debate_session(new_session)
        logger.info(
            f"Handoff session from {old_session.channel} to {new_channel}: "
            f"{session_id} -> {new_session_id}"
        )
        return new_session

    async def update_context(self, session_id: str, context_updates: dict) -> bool:
        """Update session context.

        Args:
            session_id: The session ID
            context_updates: Dictionary of context updates to merge

        Returns:
            True if successful, False if session not found
        """
        session = self._store.get_debate_session(session_id)
        if not session:
            return False

        session.context.update(context_updates)
        session.touch()
        self._store.set_debate_session(session)
        return True

    async def get_active_debate(self, session_id: str) -> Optional[str]:
        """Get the active debate ID for a session.

        Args:
            session_id: The session ID

        Returns:
            The debate ID if linked, None otherwise
        """
        session = self._store.get_debate_session(session_id)
        return session.debate_id if session else None


# Singleton instance
_manager: Optional[DebateSessionManager] = None


def get_debate_session_manager() -> DebateSessionManager:
    """Get the singleton DebateSessionManager instance."""
    global _manager
    if _manager is None:
        _manager = DebateSessionManager()
    return _manager
