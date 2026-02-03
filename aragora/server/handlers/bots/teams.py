"""
Microsoft Teams Bot endpoint handler -- backward-compatibility shim.

The implementation has been decomposed into the ``teams/`` package:
  - ``teams/handler.py``  -- TeamsBot, TeamsHandler
  - ``teams/cards.py``    -- TeamsCardActions
  - ``teams/channels.py`` -- TeamsChannelManager
  - ``teams/events.py``   -- TeamsEventProcessor
  - ``teams/oauth.py``    -- TeamsOAuth

Shared state and card-builder utilities live in ``teams_utils.py``.

This file is kept only for any edge-case loader that resolves the ``.py``
file directly rather than the package directory.  Normal ``import`` will
prefer the ``teams/`` package.
"""

from __future__ import annotations

# Re-export everything from the package so the public surface is identical.
from aragora.server.handlers.bots.teams import (  # noqa: F401
    AGENT_DISPLAY_NAMES,
    MENTION_PATTERN,
    PERM_TEAMS_ADMIN,
    PERM_TEAMS_CARDS_RESPOND,
    PERM_TEAMS_DEBATES_CREATE,
    PERM_TEAMS_DEBATES_VOTE,
    PERM_TEAMS_MESSAGES_READ,
    PERM_TEAMS_MESSAGES_SEND,
    TEAMS_APP_ID,
    TEAMS_APP_PASSWORD,
    TEAMS_TENANT_ID,
    TeamsBot,
    TeamsCardActions,
    TeamsChannelManager,
    TeamsEventProcessor,
    TeamsHandler,
    TeamsOAuth,
    _active_debates,
    _check_botframework_available,
    _check_connector_available,
    _conversation_references,
    _start_teams_debate,
    _store_conversation_reference,
    _user_votes,
    _verify_teams_token,
    build_consensus_card,
    build_debate_card,
    get_conversation_reference,
    get_debate_vote_counts,
)

__all__ = [
    "TeamsHandler",
    "TeamsBot",
    "TeamsEventProcessor",
    "TeamsCardActions",
    "TeamsOAuth",
    "TeamsChannelManager",
    "build_debate_card",
    "build_consensus_card",
    "get_debate_vote_counts",
    "get_conversation_reference",
    "_active_debates",
    "_user_votes",
    "_conversation_references",
    "_verify_teams_token",
    "_check_botframework_available",
    "_check_connector_available",
    "_start_teams_debate",
    "_store_conversation_reference",
    "AGENT_DISPLAY_NAMES",
    "MENTION_PATTERN",
    "TEAMS_APP_ID",
    "TEAMS_APP_PASSWORD",
    "TEAMS_TENANT_ID",
    "PERM_TEAMS_MESSAGES_READ",
    "PERM_TEAMS_MESSAGES_SEND",
    "PERM_TEAMS_DEBATES_CREATE",
    "PERM_TEAMS_DEBATES_VOTE",
    "PERM_TEAMS_CARDS_RESPOND",
    "PERM_TEAMS_ADMIN",
]
