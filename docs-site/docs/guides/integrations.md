---
title: Integrations
description: Integrations
---

# Integrations

Aragora supports posting debate notifications to Discord and Slack channels via webhooks.

## Discord Integration

### Setup

1. **Create a Discord Webhook**
   - Go to your Discord server settings
   - Navigate to Integrations > Webhooks
   - Click "New Webhook"
   - Choose a channel and copy the webhook URL

2. **Configure Aragora**

```python
from aragora.integrations.discord import DiscordConfig, DiscordIntegration

config = DiscordConfig(
    webhook_url="https://discord.com/api/webhooks/...",
    username="Aragora Debates",        # Bot display name
    avatar_url="",                      # Optional avatar URL
    enabled=True,
    include_agent_details=True,         # Show participating agents
    include_vote_breakdown=True,        # Show vote counts
    max_summary_length=1900,            # Discord embed limit
    rate_limit_per_minute=30,           # Internal rate limiting
    retry_count=3,                      # Retry failed requests
    retry_delay=1.0,                    # Seconds between retries
)

discord = DiscordIntegration(config)
```

### Sending Notifications

```python
import asyncio

async def notify():
    # Debate started
    await discord.send_debate_start(
        debate_id="abc123",
        topic="Should we adopt microservices?",
        agents=["anthropic-api", "openai-api", "gemini"],
        config={"rounds": 3, "consensus_mode": "majority"},
    )

    # Consensus reached
    await discord.send_consensus_reached(
        debate_id="abc123",
        topic="Should we adopt microservices?",
        consensus_type="majority",
        result={
            "winner": "Yes, with caveats",
            "confidence": 0.85,
            "votes": {"Yes": 2, "No": 1},
        },
    )

    # No consensus
    await discord.send_no_consensus(
        debate_id="abc123",
        topic="...",
        final_state={
            "rounds_completed": 3,
            "final_votes": {"Option A": 1, "Option B": 1, "Option C": 1},
        },
    )

    # Round summary
    await discord.send_round_summary(
        debate_id="abc123",
        round_number=2,
        total_rounds=3,
        summary="Agents debated trade-offs of monolith vs microservices...",
        agent_positions={
            "anthropic-api": "Recommends starting with modular monolith",
            "openai-api": "Suggests microservices for team scaling",
        },
    )

    # Error notification
    await discord.send_error(
        error_type="Agent Timeout",
        message="anthropic-api failed to respond within 60s",
        debate_id="abc123",
        details={"agent": "anthropic-api", "timeout": 60},
    )

    # Clean up
    await discord.close()

asyncio.run(notify())
```

### Event Colors

Discord embeds use color-coded borders:

| Event | Color | Hex |
|-------|-------|-----|
| Debate Start | Green | `#57F287` |
| Consensus | Blurple | `#5865F2` |
| No Consensus | Yellow | `#FEE75C` |
| Error | Red | `#ED4245` |
| Agent Join | Light Green | `#3BA55C` |
| Round Complete | Pink | `#EB459E` |

### Multi-Webhook Manager

Broadcast to multiple Discord channels:

```python
from aragora.integrations.discord import (
    DiscordWebhookManager,
    DiscordConfig,
    discord_manager,  # Global instance
)

# Register multiple webhooks
discord_manager.register(
    "alerts",
    DiscordConfig(webhook_url="https://discord.com/api/webhooks/alerts/...")
)
discord_manager.register(
    "logs",
    DiscordConfig(webhook_url="https://discord.com/api/webhooks/logs/...")
)

# Broadcast to all
results = await discord_manager.broadcast(
    "send_consensus_reached",
    debate_id="abc123",
    topic="...",
    consensus_type="majority",
    result={"winner": "Yes", "confidence": 0.9},
)
print(results)  # {"alerts": True, "logs": True}

# Clean up all connections
await discord_manager.close_all()
```

---

## Slack Integration

### Setup

1. **Create a Slack App**
   - Go to [api.slack.com/apps](https://api.slack.com/apps)
   - Create a new app or use an existing one
   - Enable Incoming Webhooks
   - Add a webhook to your workspace
   - Copy the webhook URL

2. **Configure Aragora**

```python
from aragora.integrations.slack import SlackConfig, SlackIntegration

config = SlackConfig(
    webhook_url="https://hooks.slack.com/services/...",
    channel="#debates",                  # Default channel
    bot_name="Aragora",                  # Display name
    icon_emoji=":speech_balloon:",       # Bot icon
    notify_on_consensus=True,            # Alert on consensus
    notify_on_debate_end=True,           # Alert on completion
    notify_on_error=True,                # Alert on errors
    min_consensus_confidence=0.7,        # Minimum confidence for alerts
    max_messages_per_minute=10,          # Rate limiting
)

slack = SlackIntegration(config)
```

### Sending Notifications

```python
import asyncio
from aragora.core import DebateResult

async def notify():
    # Post debate summary
    result = DebateResult(
        debate_id="abc123",
        task="Design a rate limiter for 1M requests/sec",
        consensus_reached=True,
        winner="Token bucket with Redis backend",
        confidence=0.85,
        rounds_completed=3,
        final_proposal="Use a distributed token bucket algorithm...",
    )
    await slack.post_debate_summary(result)

    # Consensus alert
    await slack.send_consensus_alert(
        debate_id="abc123",
        confidence=0.92,
        winner="Token bucket approach",
        task="Design a rate limiter",
    )

    # Error alert
    await slack.send_error_alert(
        error_type="Agent Failure",
        error_message="openai-api returned 429 rate limit error",
        debate_id="abc123",
        severity="warning",  # info, warning, error, critical
    )

    # Leaderboard update
    await slack.send_leaderboard_update(
        rankings=[
            {"name": "anthropic-api", "elo": 1650, "wins": 42},
            {"name": "openai-api", "elo": 1620, "wins": 38},
            {"name": "gemini", "elo": 1580, "wins": 31},
        ],
        top_n=5,
    )

    # Clean up
    await slack.close()

asyncio.run(notify())
```

### Block Kit Formatting

Slack messages use Block Kit for rich formatting:

**Debate Summary:**
```
+--------------------------------------------------+
| :white_check_mark: Debate Completed              |
+--------------------------------------------------+
| *Task:* Design a rate limiter for 1M req/sec     |
|--------------------------------------------------|
| *Consensus:* Reached    | *Winner:* Token bucket |
| *Confidence:* 85%       | *Rounds:* 3            |
|--------------------------------------------------|
| *Final Proposal:*                                |
| ```Use a distributed token bucket algorithm...```|
|--------------------------------------------------|
| :robot_face: Aragora AI Debate System | 2024-... |
+--------------------------------------------------+
```

**Consensus Alert:**
```
+--------------------------------------------------+
| :tada: Consensus Reached!                        |
+--------------------------------------------------+
| *Debate:* abc123...  | *Confidence:* 92%         |
| *Winning Position:* Token bucket approach        |
+--------------------------------------------------+
```

### Severity Indicators

| Severity | Emoji | Use Case |
|----------|-------|----------|
| `info` | :information_source: | Status updates |
| `warning` | :warning: | Rate limits, retries |
| `error` | :x: | Failed operations |
| `critical` | :rotating_light: | System failures |

### Slack App Endpoints (Server)

Expose Slack slash commands and interactive actions via the server handler:

- `POST /api/integrations/slack/commands`
- `POST /api/integrations/slack/interactive`
- `POST /api/integrations/slack/events`
- `GET /api/integrations/slack/status`

Set `SLACK_SIGNING_SECRET` to verify signatures. `SLACK_BOT_TOKEN` enables
optional API calls, and `SLACK_WEBHOOK_URL` enables outbound notifications.

---

## Telegram Integration

Send debate summaries and consensus alerts to Telegram via a bot.

```python
import asyncio
from aragora.integrations.telegram import TelegramConfig, TelegramIntegration

async def notify(result):
    telegram = TelegramIntegration(
        TelegramConfig(
            bot_token="123456:ABC-DEF1234...",
            chat_id="-1001234567890",
            notify_on_consensus=True,
            notify_on_debate_end=True,
        )
    )

    # Post a debate summary
    await telegram.post_debate_summary(result)
    await telegram.close()

asyncio.run(notify(result))
```

---

## Email Integration

Send summaries and alerts via SMTP.

```python
import asyncio
from aragora.integrations.email import EmailConfig, EmailIntegration, EmailRecipient

async def notify(result):
    email = EmailIntegration(
        EmailConfig(
            smtp_host="smtp.sendgrid.net",
            smtp_username="apikey",
            smtp_password="your-api-key",
            from_email="debates@aragora.ai",
        )
    )
    email.add_recipient(EmailRecipient(email="ops@example.com", name="Ops Team"))

    await email.send_debate_summary(result)
    await email.close()

asyncio.run(notify(result))
```

---

## Outbound Webhooks

Use the webhook dispatcher for low-latency event delivery to external systems.
Configure webhooks via `ARAGORA_WEBHOOKS` or `ARAGORA_WEBHOOKS_CONFIG`.

```json
[
  {
    "name": "alerts",
    "url": "https://hooks.example.com/aragora",
    "secret": "hmac-secret",
    "event_types": ["consensus", "debate_end"],
    "loop_ids": ["loop_abc123"]
  }
]
```

```python
from aragora.integrations.webhooks import init_dispatcher
from aragora.server.stream import SyncEventEmitter, create_arena_hooks

emitter = SyncEventEmitter()
hooks = create_arena_hooks(emitter)

dispatcher = init_dispatcher()
if dispatcher:
    emitter.subscribe(lambda event: dispatcher.enqueue(event.to_dict()))
```

---

## Environment Variables

Integrations are configured programmatically; you can load configs from env vars
if desired. The server-side Slack handler uses these variables:

```bash
# Slack server integration
export SLACK_SIGNING_SECRET="..."
export SLACK_BOT_TOKEN="xoxb-..."
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."

# Optional app-level config for integration classes
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
export SLACK_CHANNEL="#debates"
export SLACK_BOT_NAME="Aragora"
```

---

## Server Integration

Hook into debate events by subscribing to the stream emitter used by Arena:

```python
import asyncio
from aragora.debate.orchestrator import Arena
from aragora.integrations.discord import DiscordConfig, DiscordIntegration
from aragora.integrations.slack import SlackConfig, SlackIntegration
from aragora.server.stream import SyncEventEmitter, create_arena_hooks
from aragora.server.stream.events import StreamEventType

# Create integrations
discord = DiscordIntegration(DiscordConfig(
    webhook_url=os.environ["DISCORD_WEBHOOK_URL"]
))
slack = SlackIntegration(SlackConfig(
    webhook_url=os.environ["SLACK_WEBHOOK_URL"]
))

# Wire emitter -> hooks
emitter = SyncEventEmitter(loop_id="debate_123")
hooks = create_arena_hooks(emitter)
debate_tasks = {}

async def handle_event(event):
    if event.type == StreamEventType.DEBATE_START:
        debate_tasks[event.loop_id] = event.data.get("task", "")
    if event.type == StreamEventType.CONSENSUS:
        task = debate_tasks.get(event.loop_id, "Debate")
        await discord.send_consensus_reached(
            debate_id=event.loop_id,
            topic=task,
            consensus_type="majority",
            result={
                "winner": event.data.get("answer", ""),
                "confidence": event.data.get("confidence", 0.0),
            },
        )
        await slack.send_consensus_alert(
            debate_id=event.loop_id,
            confidence=event.data.get("confidence", 0.0),
            winner=event.data.get("answer", ""),
            task=task,
        )

def enqueue(event):
    # Fan-out async work in your own scheduler/task runner
    asyncio.create_task(handle_event(event))

emitter.subscribe(enqueue)

# Start an arena with the hooks
arena = Arena(env, agents, event_hooks=hooks, event_emitter=emitter)
```

---

## Best Practices

1. **Rate Limiting**: Both integrations implement internal rate limiting. Discord limits to 30 messages/minute by default, Slack to 10/minute.

2. **Retry Logic**: Discord integration retries failed requests up to 3 times with exponential backoff.

3. **Message Truncation**: Long messages are automatically truncated to platform limits (2048 chars for Discord embeds, 1000 chars for Slack code blocks).

4. **Webhook Security**: Never commit webhook URLs to version control. Use environment variables or secrets management.

5. **Selective Notifications**: Configure `notify_on_*` flags to avoid notification fatigue.

---

## Related Documentation

- [WebSocket Events](./websocket-events) - Real-time event streaming
- [API Reference](../api/reference) - REST API endpoints
- [Environment Variables](../getting-started/environment) - Configuration reference
