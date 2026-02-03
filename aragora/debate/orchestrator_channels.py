"""Agent-to-agent channel integration for debate orchestration.

Handles setup and teardown of communication channels between agents
during a debate session.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.debate.context import DebateContext
    from aragora.debate.protocol import DebateProtocol
    from aragora.core import Agent

logger = logging.getLogger(__name__)


async def setup_agent_channels(
    protocol: "DebateProtocol",
    agents: list["Agent"],
    debate_id: str,
    ctx: "DebateContext",
) -> Optional[Any]:
    """Initialize agent-to-agent channels for the current debate.

    Returns the channel_integration instance if setup succeeds, else None.
    """
    if not getattr(protocol, "enable_agent_channels", False):
        return None
    try:
        from aragora.debate.channel_integration import create_channel_integration

        channel_integration = create_channel_integration(
            debate_id=debate_id,
            agents=agents,
            protocol=protocol,
        )
        if await channel_integration.setup():
            ctx.channel_integration = channel_integration
            return channel_integration
        return None
    except (ImportError, ConnectionError, OSError, ValueError, TypeError, AttributeError) as e:
        logger.debug(f"[channels] Channel setup failed (non-critical): {e}")
        return None


async def teardown_agent_channels(channel_integration: Optional[Any]) -> None:
    """Tear down agent channels after debate completion."""
    if not channel_integration:
        return
    try:
        await channel_integration.teardown()
    except (ConnectionError, OSError, RuntimeError) as e:
        logger.debug(f"[channels] Channel teardown failed (non-critical): {e}")
