"""
Slack Bot Handler for bi-directional integration.

This module is a backward compatibility re-export layer.
The implementation has been decomposed into the slack/ package.

See aragora/server/handlers/bots/slack/__init__.py for the full implementation.

Handles:
- Incoming webhooks (slash commands, events)
- Interactive components (button clicks, votes)
- Two-way debate participation from Slack

Implements:
- Signature verification for security
- Block Kit interactive messages with voting
- Threaded debate updates
- User vote counting in consensus
- RBAC permission enforcement for all handler methods

RBAC Permissions:
- slack.commands.read: View command status and help
- slack.commands.execute: Execute slash commands
- slack.debates.create: Create new debates from Slack
- slack.votes.record: Record votes in debates
- slack.interactive.respond: Respond to interactive components
- slack.admin: Full administrative access to Slack integration
"""

# =============================================================================
# Re-export all public symbols from the slack package
# =============================================================================
# This maintains backward compatibility with existing imports like:
#   from aragora.server.handlers.bots.slack import SlackHandler
#   from aragora.server.handlers.bots import slack

from aragora.server.handlers.bots.slack import (
    # Handler classes
    SlackHandler,
    # Handler functions
    handle_slack_events,
    handle_slack_interactions,
    handle_slack_commands,
    # Block Kit builders
    build_debate_message_blocks,
    build_consensus_message_blocks,
    get_debate_vote_counts,
    register_slack_routes,
    PERM_SLACK_COMMANDS_READ,
    PERM_SLACK_COMMANDS_EXECUTE,
    PERM_SLACK_DEBATES_CREATE,
    PERM_SLACK_VOTES_RECORD,
    PERM_SLACK_INTERACTIVE,
    PERM_SLACK_ADMIN,
)


__all__ = [
    # Handler classes
    "SlackHandler",
    # Handler functions
    "handle_slack_events",
    "handle_slack_interactions",
    "handle_slack_commands",
    # Block Kit builders
    "build_debate_message_blocks",
    "build_consensus_message_blocks",
    # Utility functions
    "get_debate_vote_counts",
    "register_slack_routes",
    # RBAC Permission constants
    "PERM_SLACK_COMMANDS_READ",
    "PERM_SLACK_COMMANDS_EXECUTE",
    "PERM_SLACK_DEBATES_CREATE",
    "PERM_SLACK_VOTES_RECORD",
    "PERM_SLACK_INTERACTIVE",
    "PERM_SLACK_ADMIN",
]


# =============================================================================
# Import Verification
# =============================================================================
# Verify that imports work correctly:
#   python -c "from aragora.server.handlers.bots.slack import SlackHandler; print('OK')"
#   python -c "from aragora.server.handlers.bots import slack; print(slack.SlackHandler)"
