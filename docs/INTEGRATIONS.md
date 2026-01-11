# Chat Platform Integrations

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

---

## Environment Variables

Configure integrations via environment variables:

```bash
# Discord
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
export DISCORD_BOT_NAME="Aragora"
export DISCORD_RATE_LIMIT=30

# Slack
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
export SLACK_CHANNEL="#debates"
export SLACK_BOT_NAME="Aragora"
```

---

## Server Integration

Use integrations with the Aragora server:

```python
from aragora.server import UnifiedServer
from aragora.integrations.discord import DiscordConfig, DiscordIntegration
from aragora.integrations.slack import SlackConfig, SlackIntegration

# Create integrations
discord = DiscordIntegration(DiscordConfig(
    webhook_url=os.environ["DISCORD_WEBHOOK_URL"]
))
slack = SlackIntegration(SlackConfig(
    webhook_url=os.environ["SLACK_WEBHOOK_URL"]
))

# Hook into debate events
async def on_debate_complete(result):
    await discord.send_consensus_reached(
        debate_id=result.debate_id,
        topic=result.task,
        consensus_type="majority",
        result={
            "winner": result.winner,
            "confidence": result.confidence,
        },
    )
    await slack.post_debate_summary(result)

# Register handler
server.on_debate_complete(on_debate_complete)
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

- [WebSocket Events](WEBSOCKET_EVENTS.md) - Real-time event streaming
- [API Reference](API_REFERENCE.md) - REST API endpoints
- [Environment Variables](ENVIRONMENT.md) - Configuration reference
