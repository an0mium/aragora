"""
Slack Bot Handler State Management.

This module manages the shared state for active debates, user votes,
and the Slack integration singleton.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Store active debate sessions for Slack
# In production, this would be in Redis/database
_active_debates: dict[str, dict[str, Any]] = {}
_user_votes: dict[str, dict[str, str]] = {}  # debate_id -> {user_id: vote}

# Slack integration singleton
_slack_integration: Any | None = None


def get_active_debates() -> dict[str, dict[str, Any]]:
    """Get the active debates dictionary."""
    return _active_debates


def get_user_votes() -> dict[str, dict[str, str]]:
    """Get the user votes dictionary."""
    return _user_votes


def get_slack_integration() -> Any | None:
    """Get the Slack integration singleton.

    Returns None if Slack is not configured (no webhook URL).
    """
    global _slack_integration

    # Respect overrides on the slack module for test patching.
    try:
        import sys

        slack_module = sys.modules.get("aragora.server.handlers.bots.slack")
        if slack_module is not None and hasattr(slack_module, "_slack_integration"):
            override = getattr(slack_module, "_slack_integration")
            if override is not None:
                _slack_integration = override
                return _slack_integration
    except Exception:
        pass

    # Return cached if available
    if _slack_integration is not None:
        return _slack_integration

    # Check if Slack is configured
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not webhook_url:
        return None

    # Create integration (lazy import to avoid circular dependencies)
    try:
        from aragora.connectors.slack import SlackConnector

        _slack_integration = SlackConnector(webhook_url=webhook_url)
        return _slack_integration
    except ImportError:
        logger.debug("SlackConnector not available")
        return None


def get_debate_vote_counts(debate_id: str) -> dict[str, int]:
    """Get vote counts for a debate."""
    votes = _user_votes.get(debate_id, {})
    counts: dict[str, int] = {}
    for agent in votes.values():
        counts[agent] = counts.get(agent, 0) + 1
    return counts


__all__ = [
    "_active_debates",
    "_user_votes",
    "get_active_debates",
    "get_user_votes",
    "get_slack_integration",
    "get_debate_vote_counts",
]
