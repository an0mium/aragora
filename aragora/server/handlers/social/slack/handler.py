"""
Slack handler - main entry point.

Re-exports SlackHandler from the implementation module for backward compatibility.
The modular structure is in place for future incremental migration of methods.
"""

# Re-export SlackHandler from the implementation module
from .._slack_impl import (
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
