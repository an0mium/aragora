"""
Microsoft Teams bot handler package.

Re-exports the public API from decomposed submodules so that existing
imports like ``from aragora.server.handlers.bots.teams import TeamsHandler``
continue to work unchanged.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core handler classes
# ---------------------------------------------------------------------------
from aragora.server.handlers.bots.teams.handler import (  # noqa: F401
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
    TeamsHandler,
)

# ---------------------------------------------------------------------------
# Decomposed module classes
# ---------------------------------------------------------------------------
from aragora.server.handlers.bots.teams.cards import TeamsCardActions  # noqa: F401
from aragora.server.handlers.bots.teams.channels import TeamsChannelManager  # noqa: F401
from aragora.server.handlers.bots.teams.events import TeamsEventProcessor  # noqa: F401
from aragora.server.handlers.bots.teams.oauth import TeamsOAuth  # noqa: F401
from aragora.audit.unified import audit_data  # noqa: F401

# ---------------------------------------------------------------------------
# Shared state and utilities (from teams_utils)
# ---------------------------------------------------------------------------
from aragora.server.handlers.bots.teams_utils import (  # noqa: F401
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
    # Main classes
    "TeamsHandler",
    "TeamsBot",
    # Decomposed module classes
    "TeamsEventProcessor",
    "TeamsCardActions",
    "TeamsOAuth",
    "TeamsChannelManager",
    # Card builders (from teams_utils)
    "build_debate_card",
    "build_consensus_card",
    "get_debate_vote_counts",
    "get_conversation_reference",
    # Shared state (for testing/integration)
    "_active_debates",
    "_user_votes",
    "_conversation_references",
    # Utilities
    "_verify_teams_token",
    "_check_botframework_available",
    "_check_connector_available",
    "_start_teams_debate",
    "_store_conversation_reference",
    # Constants
    "AGENT_DISPLAY_NAMES",
    "MENTION_PATTERN",
    "TEAMS_APP_ID",
    "TEAMS_APP_PASSWORD",
    "TEAMS_TENANT_ID",
    # RBAC permissions
    "PERM_TEAMS_MESSAGES_READ",
    "PERM_TEAMS_MESSAGES_SEND",
    "PERM_TEAMS_DEBATES_CREATE",
    "PERM_TEAMS_DEBATES_VOTE",
    "PERM_TEAMS_CARDS_RESPOND",
    "PERM_TEAMS_ADMIN",
    "audit_data",
]
