"""
Slack handler module alias.

Re-exports the Slack bot handler from the bots directory for backward compatibility.
"""

from aragora.server.handlers.bots.slack import (
    SlackHandler,
    COMMAND_PATTERN,
    TOPIC_PATTERN,
    get_slack_integration,
)

# Re-export for backward compatibility
__all__ = [
    "SlackHandler",
    "COMMAND_PATTERN",
    "TOPIC_PATTERN",
    "get_slack_integration",
]
