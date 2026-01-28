"""
Slack handler package.

Provides Slack integration endpoints for slash commands, interactive components,
and event handling. The handler is split into logical modules for better maintainability:

- handler.py: Main SlackHandler class (re-exported from _slack_impl.py)
- config.py: Configuration constants
- security.py: Signature verification (SignatureVerifierMixin)
- commands/: Slash command implementations (CommandsMixin)
- events/: Event API handlers (EventsMixin)
- interactive/: Interactive component handlers
- blocks/: Slack block building utilities
- utils/: Response helpers and utilities

For backward compatibility, import SlackHandler from this package:

    from aragora.server.handlers.social.slack import SlackHandler

Mixins for custom handlers:

    from aragora.server.handlers.social.slack import (
        CommandsMixin,
        EventsMixin,
        SignatureVerifierMixin,
    )
"""

from .handler import (
    SlackHandler,
    get_slack_handler,
    get_slack_integration,
    get_workspace_store,
    resolve_workspace,
    create_tracked_task,
    _validate_slack_url,
    SLACK_SIGNING_SECRET,
    SLACK_BOT_TOKEN,
    SLACK_WEBHOOK_URL,
    SLACK_ALLOWED_DOMAINS,
    ARAGORA_API_BASE_URL,
)
from .commands import CommandsMixin
from .events import EventsMixin
from .security import SignatureVerifierMixin

__all__ = [
    # Main handler
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    # Mixins
    "CommandsMixin",
    "EventsMixin",
    "SignatureVerifierMixin",
    # Configuration
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "ARAGORA_API_BASE_URL",
]
