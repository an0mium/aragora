# Bot Handlers

Multi-channel bot integrations delivering debate results and AI decisions to Discord, Slack, Teams, Telegram, WhatsApp, Zoom, and more.

## Modules

| Module | Purpose |
|--------|---------|
| `base.py` | BotHandlerMixin with shared utilities |
| `cache.py` | Bot response caching with TTL and heap cleanup |
| `discord.py` | Discord slash commands and interactions |
| `slack.py` | Slack bot (wrapper for slack/ subpackage) |
| `slack/` | Modular Slack handlers (commands, events, oauth) |
| `teams.py` | Microsoft Teams bot (wrapper for teams/ subpackage) |
| `teams/` | Modular Teams handlers (cards, events, oauth) |
| `teams_cards.py` | Adaptive Card templates for Teams |
| `teams_utils.py` | Teams API utilities |
| `telegram.py` | Telegram Bot API integration |
| `whatsapp.py` | WhatsApp Business API integration |
| `google_chat.py` | Google Chat webhook and slash commands |
| `email_webhook.py` | Email-to-debate webhook integration |
| `zoom.py` | Zoom chatbot integration |

## Supported Platforms

| Platform | Webhook | Commands | OAuth | Events |
|----------|---------|----------|-------|--------|
| Discord | Yes | Slash | Yes | Yes |
| Slack | Yes | Slash | Yes | Yes |
| Teams | Yes | Bot | Yes | Yes |
| Telegram | Yes | /commands | Bot Token | Yes |
| WhatsApp | Yes | N/A | App Token | Yes |
| Google Chat | Yes | Slash | Service | Yes |
| Zoom | Yes | Bot | Yes | Yes |
| Email | Webhook | N/A | OAuth/SMTP | N/A |

## Endpoints

### Common Pattern
Each platform follows this endpoint pattern:

- `POST /api/v1/bots/{platform}/webhook` - Incoming messages/events
- `POST /api/v1/bots/{platform}/commands/{command}` - Slash commands
- `GET /api/v1/bots/{platform}/oauth/authorize` - OAuth start
- `GET /api/v1/bots/{platform}/oauth/callback` - OAuth callback

### Platform-Specific

**Discord:**
- `POST /api/v1/bots/discord/interactions` - Slash command interactions

**Slack:**
- `POST /api/v1/bots/slack/events` - Event subscriptions
- `POST /api/v1/bots/slack/commands` - Slash commands

**Teams:**
- `POST /api/v1/bots/teams/messages` - Bot messages
- `POST /api/v1/bots/teams/adaptive-card` - Card submissions

**Telegram:**
- `POST /api/v1/bots/telegram/webhook` - Update webhook

**WhatsApp:**
- `POST /api/v1/bots/whatsapp/webhook` - Message webhook
- `GET /api/v1/bots/whatsapp/webhook` - Webhook verification

## RBAC Permissions

| Permission | Description |
|------------|-------------|
| `bot:read` | View bot configurations |
| `bot:write` | Configure bot integrations |
| `bot:send` | Send messages via bots |
| `bot:admin` | Manage OAuth and credentials |

## Usage

```python
from aragora.server.handlers.bots import (
    TelegramHandler,
    DiscordHandler,
    TeamsHandler,
)

# Initialize handler
telegram = TelegramHandler(bot_token="...", workspace_id="...")

# Handle incoming update
result = await telegram.handle_update(update_data)
```

## Features

- **Multi-Channel Delivery**: Send debate results to any platform
- **Bidirectional Chat**: Query debates from chat channels
- **OAuth Integration**: Secure workspace connections
- **Response Caching**: Heap-based TTL cache for performance
- **Adaptive Cards**: Rich interactive cards for Teams
- **Rate Limiting**: Platform-specific rate limit handling
- **Event Processing**: Real-time event handling per platform
