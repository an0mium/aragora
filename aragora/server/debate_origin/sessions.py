"""Debate session management for multi-channel routing.

Provides session creation, linking, and lookup for debates that may
span multiple chat platforms.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_debate_session_manager():
    """Get the debate session manager.

    Defined here for test patching and to avoid repeated import sites.
    """
    from aragora.connectors.debate_session import get_debate_session_manager as _get_manager

    return _get_manager()


async def _create_and_link_session(
    manager,
    platform: str,
    user_id: str,
    metadata: dict[str, Any] | None,
    debate_id: str,
) -> None:
    """Helper to create and link session in async context."""
    try:
        session = await manager.create_session(platform, user_id, metadata)
        await manager.link_debate(session.session_id, debate_id)
    except (OSError, RuntimeError, ValueError, KeyError, ConnectionError, TimeoutError) as e:
        logger.debug(f"Async session creation failed: {e}")


async def get_sessions_for_debate(debate_id: str) -> list:
    """Get all sessions linked to a debate for multi-channel routing.

    Args:
        debate_id: The debate ID

    Returns:
        List of DebateSession objects linked to the debate
    """
    try:
        manager = get_debate_session_manager()
        return await manager.find_sessions_for_debate(debate_id)
    except ImportError:
        return []
    except (OSError, RuntimeError, ValueError, KeyError, ConnectionError, TimeoutError) as e:
        logger.debug(f"Session lookup failed: {e}")
        return []
