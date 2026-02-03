"""
Telegram Bot integration endpoint handlers.

Endpoints:
- POST /api/integrations/telegram/webhook - Handle Telegram webhook updates
- GET  /api/integrations/telegram/status  - Get integration status
- POST /api/integrations/telegram/set-webhook - Configure webhook URL

Environment Variables:
- TELEGRAM_BOT_TOKEN - Required for all API calls
- TELEGRAM_WEBHOOK_SECRET - Optional secret for webhook verification

Telegram Bot Commands:
- /start - Welcome message
- /help - Show available commands
 - /debate <topic> - Start a multi-agent debate
 - /plan <topic> - Debate with an implementation plan
 - /implement <topic> - Debate with plan + context snapshot
 - /gauntlet <statement> - Run adversarial validation
- /status - Get system status
- /agents - List available agents

RBAC Permissions:
- telegram:read - View status and read-only operations
- telegram:messages:send - Send messages to chats
- telegram:debates:create - Create debates via Telegram
- telegram:debates:read - View debate details
- telegram:gauntlet:run - Run gauntlet stress-tests
- telegram:votes:record - Record user votes
- telegram:commands:execute - Execute bot commands
- telegram:callbacks:handle - Handle callback queries
- telegram:admin - Administrative operations (webhook config)
"""

from ._common import (
    PERM_TELEGRAM_ADMIN,
    PERM_TELEGRAM_CALLBACKS_HANDLE,
    PERM_TELEGRAM_COMMANDS_EXECUTE,
    PERM_TELEGRAM_DEBATES_CREATE,
    PERM_TELEGRAM_DEBATES_READ,
    PERM_TELEGRAM_GAUNTLET_RUN,
    PERM_TELEGRAM_MESSAGES_SEND,
    PERM_TELEGRAM_READ,
    PERM_TELEGRAM_VOTES_RECORD,
    RBAC_AVAILABLE,  # noqa: F401
    TELEGRAM_API_BASE,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_WEBHOOK_SECRET,
    _handle_task_exception,
    create_tracked_task,
)
from .messages import TTS_VOICE_ENABLED  # noqa: F401
from .handler import TelegramHandler, get_telegram_handler

# Re-export telemetry and chat_events for test patching compatibility
from ..telemetry import (
    record_api_call,  # noqa: F401
    record_api_latency,  # noqa: F401
    record_command,  # noqa: F401
    record_debate_completed,  # noqa: F401
    record_debate_failed,  # noqa: F401
    record_debate_started,  # noqa: F401
    record_error,  # noqa: F401
    record_gauntlet_completed,  # noqa: F401
    record_gauntlet_failed,  # noqa: F401
    record_gauntlet_started,  # noqa: F401
    record_message,  # noqa: F401
    record_vote,  # noqa: F401
    record_webhook_latency,  # noqa: F401
    record_webhook_request,  # noqa: F401
)
from ..chat_events import (
    emit_command_received,  # noqa: F401
    emit_debate_completed,  # noqa: F401
    emit_debate_started,  # noqa: F401
    emit_gauntlet_completed,  # noqa: F401
    emit_gauntlet_started,  # noqa: F401
    emit_message_received,  # noqa: F401
    emit_vote_received,  # noqa: F401
)

__all__ = [
    # Handler classes
    "TelegramHandler",
    "get_telegram_handler",
    # Utility functions
    "create_tracked_task",
    "_handle_task_exception",
    # Constants
    "TELEGRAM_API_BASE",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_WEBHOOK_SECRET",
    # RBAC Permission Constants
    "PERM_TELEGRAM_READ",
    "PERM_TELEGRAM_MESSAGES_SEND",
    "PERM_TELEGRAM_DEBATES_CREATE",
    "PERM_TELEGRAM_DEBATES_READ",
    "PERM_TELEGRAM_GAUNTLET_RUN",
    "PERM_TELEGRAM_VOTES_RECORD",
    "PERM_TELEGRAM_COMMANDS_EXECUTE",
    "PERM_TELEGRAM_CALLBACKS_HANDLE",
    "PERM_TELEGRAM_ADMIN",
]
