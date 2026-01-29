# mypy: ignore-errors
"""
Slack integration endpoint handlers.

Endpoints:
- POST /api/integrations/slack/commands - Handle Slack slash commands
- POST /api/integrations/slack/interactive - Handle interactive components
- POST /api/integrations/slack/events - Handle Slack events

Environment Variables:
- SLACK_SIGNING_SECRET - Required for webhook verification
- SLACK_BOT_TOKEN - Optional for advanced API calls
- SLACK_WEBHOOK_URL - For sending notifications
- ARAGORA_API_BASE_URL - Base URL for internal API calls (default: http://localhost:8080)

This package was refactored from a single _slack_impl.py module into sub-modules:
- config.py: Constants, lazy singletons, environment variables
- messaging.py: Response helpers, async message posting
- blocks.py: Slack Block Kit message construction
- commands.py: Slash command implementations
- events.py: Events API handlers (app_mention, message, DM)
- interactive.py: Interactive component handlers (votes, details)
- handler.py: Main SlackHandler class composing all mixins
"""

from __future__ import annotations

from .config import (
    ARAGORA_API_BASE_URL,
    BOTS_READ_PERMISSION,
    COMMAND_PATTERN,
    SLACK_ALLOWED_DOMAINS,
    SLACK_BOT_TOKEN,
    SLACK_SIGNING_SECRET,
    SLACK_WEBHOOK_URL,
    SLACK_WORKSPACE_RATE_LIMIT_RPM,
    TOPIC_PATTERN,
    _get_audit_logger,
    _get_user_rate_limiter,
    _get_workspace_rate_limiter,
    _validate_slack_url,
    create_tracked_task,
    get_slack_integration,
    get_workspace_store,
    resolve_workspace,
)
from .handler import SlackHandler, get_slack_handler

__all__ = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "_get_audit_logger",
    "_get_user_rate_limiter",
    "_get_workspace_rate_limiter",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "ARAGORA_API_BASE_URL",
    "SLACK_WORKSPACE_RATE_LIMIT_RPM",
    "BOTS_READ_PERMISSION",
    "COMMAND_PATTERN",
    "TOPIC_PATTERN",
]
