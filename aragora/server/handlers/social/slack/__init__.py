"""
Slack handler package.

Provides Slack integration endpoints for slash commands, interactive components,
and event handling. The handler is split into logical modules for better maintainability:

- handler.py: Main SlackHandler class (re-exported from _slack_impl.py)
- commands/: Slash command implementations
- interactive/: Interactive component handlers
- events/: Event API handlers
- blocks/: Slack block building utilities
- utils/: Response helpers and utilities

For backward compatibility, import SlackHandler from this package:

    from aragora.server.handlers.social.slack import SlackHandler
"""

from .handler import (
    SlackHandler,
    get_slack_handler,
    get_slack_integration,
    get_workspace_store,
    resolve_workspace,
    create_tracked_task,
    SLACK_SIGNING_SECRET,
    SLACK_BOT_TOKEN,
    SLACK_WEBHOOK_URL,
    SLACK_ALLOWED_DOMAINS,
    ARAGORA_API_BASE_URL,
)

__all__ = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "ARAGORA_API_BASE_URL",
]
