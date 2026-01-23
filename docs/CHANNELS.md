# Aragora Channels

**Bidirectional communication across organizational channels**

---

## Overview

Aragora is the control plane for multi-agent robust decisionmaking across organizational **knowledge and channels**. This document describes the channel integrations that enable:

- **Inbound Queries**: Receive deliberation requests from any channel
- **Outbound Results**: Automatically route results back to the originating channel
- **Real-time Updates**: Stream deliberation progress via WebSocket
- **Voice Integration**: Speech-to-text input, text-to-speech output

---

## Channel Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INBOUND CHANNELS                            │
├─────────────────────────────────────────────────────────────────┤
│  Slack │ Discord │ Teams │ Telegram │ WhatsApp │ Email │ Voice │ API │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ARAGORA CONTROL PLANE                         │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Chat Router │  │  Debate     │  │  Result     │             │
│  │             │  │  Origin     │  │  Router     │             │
│  │ Platform    │  │  Tracker    │  │             │             │
│  │ detection   │  │             │  │ Auto-route  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTBOUND CHANNELS                            │
├─────────────────────────────────────────────────────────────────┤
│  Slack │ Discord │ Teams │ Telegram │ WhatsApp │ Email │ Voice │ API │
└─────────────────────────────────────────────────────────────────┘
```

---

## Inbound Channels

### Chat Platforms

| Platform | Handler | Features |
|----------|---------|----------|
| **Slack** | `/api/bots/slack/webhook` | Slash commands, app mentions, threads |
| **Discord** | `/api/bots/discord/webhook` | Slash commands, bot mentions |
| **Microsoft Teams** | `/api/bots/teams/webhook` | Bot Framework, Adaptive Cards |
| **Telegram** | `/api/bots/telegram/webhook` | Bot commands, inline queries |
| **WhatsApp** | `/api/bots/whatsapp/webhook` | Cloud API integration |
| **Google Chat** | `/api/bots/google-chat/webhook` | Space messages |
| **Email (Gmail)** | `/api/v1/email/inbox` | Gmail ingestion + prioritization |
| **Email (Outlook)** | `/api/v1/outlook/messages` | Outlook/M365 ingestion |
| **Shared Inbox** | `/api/v1/inbox/shared` | Collaborative inbox triage |

### Configuration

```bash
# Slack
export SLACK_BOT_TOKEN=xoxb-...
export SLACK_SIGNING_SECRET=...

# Discord
export DISCORD_APPLICATION_ID=...
export DISCORD_PUBLIC_KEY=...
export DISCORD_BOT_TOKEN=...

# Teams
export TEAMS_APP_ID=...
export TEAMS_APP_PASSWORD=...

# Telegram
export TELEGRAM_BOT_TOKEN=...

# WhatsApp
export WHATSAPP_ACCESS_TOKEN=...
export WHATSAPP_PHONE_NUMBER_ID=...

# Gmail (email ingestion + prioritization)
export GMAIL_CLIENT_ID=...
export GMAIL_CLIENT_SECRET=...
```

### Unified Chat Router

The chat router auto-detects the originating platform and normalizes requests:

**Location:** `aragora/server/handlers/chat/router.py`

```python
# Auto-detect platform
POST /api/chat/webhook
Content-Type: application/json

{
  "message": "Review this API design",
  "attachments": [...],
  "metadata": {...}
}
```

The router extracts:
- Platform type
- User/channel identity
- Thread context (for replies)
- Attachments

---

## Voice Integration

Aragora supports bidirectional voice I/O over WebSocket:

- **STT**: Live transcription via Whisper connector
- **TTS**: Optional synthesized responses from agent messages

**Endpoint:** `/ws/voice/{debate_id}`

**Location:** `aragora/server/stream/voice_stream.py`

**Key settings:**

```bash
ARAGORA_VOICE_TTS_ENABLED=true
ARAGORA_VOICE_TTS_DEFAULT_VOICE=narrator
ARAGORA_VOICE_MAX_SESSION=300
ARAGORA_VOICE_INTERVAL=3000
```

---

## Outbound Routing

### Result Router

When a deliberation completes, results automatically route back to the originating channel.

**Location:** `aragora/server/result_router.py`

```python
from aragora.server import ResultRouter

router = ResultRouter()

# Results are automatically sent to the originating channel
await router.route_result(
    debate_id="dbt_abc123",
    result={
        "consensus": True,
        "confidence": 0.85,
        "answer": "...",
        "evidence": [...],
    }
)
```

### Debate Origin Tracking

Every debate tracks its origin for result routing:

**Location:** `aragora/server/debate_origin.py`

```python
from aragora.server import DebateOrigin

origin = DebateOrigin(
    platform="slack",
    channel_id="C123456",
    thread_ts="1234567890.123456",
    user_id="U123456",
)

# Store with debate
debate = await create_debate(topic="...", origin=origin)

# Results automatically route back
await debate.complete(result)  # Sends to Slack thread
```

### Platform-Specific Formatting

Results are formatted for each platform:

| Platform | Format |
|----------|--------|
| Slack | Rich blocks with expandable sections |
| Discord | Embeds with color-coded confidence |
| Teams | Adaptive Cards |
| Telegram | Markdown with inline buttons |
| WhatsApp | Plain text with links |

---

## Real-time Streaming

### WebSocket Events

80+ WebSocket events for real-time updates:

**Location:** `aragora/server/stream/`

| Event Category | Events |
|----------------|--------|
| Debate lifecycle | `debate_start`, `debate_end`, `round_start`, `round_end` |
| Agent activity | `agent_message`, `agent_thinking`, `critique` |
| Voting | `vote_cast`, `vote_tally`, `consensus_reached` |
| Progress | `progress_update`, `phase_change` |

### Subscription

```javascript
const ws = new WebSocket('wss://api.aragora.ai/ws');

ws.send(JSON.stringify({
  type: 'subscribe',
  debate_id: 'dbt_abc123'
}));

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.type, data.payload);
};
```

---

## Voice Integration

### Speech-to-Text

**Location:** `aragora/connectors/chat/voice_bridge.py`

```python
from aragora.connectors.chat import VoiceBridge

bridge = VoiceBridge()

# Transcribe audio to text
text = await bridge.transcribe(audio_bytes)

# Start a debate from voice
debate = await create_debate(topic=text, origin=voice_origin)
```

Supported formats: WAV, MP3, M4A, FLAC, OGG
Max file size: 25MB

### Text-to-Speech

**Location:** `aragora/connectors/chat/tts_bridge.py`

```python
from aragora.connectors.chat import TTSBridge

tts = TTSBridge()

# Generate audio from debate result
audio = await tts.synthesize(
    text=debate_result.summary,
    voice="nova",  # OpenAI TTS voice
)

# Send audio response
await send_voice_message(channel, audio)
```

---

## Email Integration

### Inbound Email

**Location:** `aragora/server/handlers/bots/email_webhook.py`

| Provider | Endpoint |
|----------|----------|
| SendGrid | `POST /api/bots/email/webhook/sendgrid` |
| AWS SES | `POST /api/bots/email/webhook/ses` |

### Gmail Integration

**Location:** `aragora/connectors/enterprise/communication/gmail.py`

```python
from aragora.connectors.enterprise.communication import GmailConnector

gmail = GmailConnector()

# Sync emails for knowledge ingestion
await gmail.sync(
    query="label:important",
    max_results=100,
)

# Emails are indexed and available for deliberation context
```

---

## External Integrations

### Zapier / Make / n8n

**Location:** `aragora/server/handlers/external_integrations.py`

| Platform | Endpoint |
|----------|----------|
| Zapier | `POST /api/integrations/zapier/triggers` |
| Make | `POST /api/integrations/make/webhooks` |
| n8n | `POST /api/integrations/n8n/webhooks` |

These integrations allow:
- Triggering debates from external workflows
- Receiving results in external systems
- Chaining debates with other automations

---

## Channel Statistics

Aragora tracks channel usage:

```
GET /api/channels/stats

{
  "total_messages": 12456,
  "by_platform": {
    "slack": 5234,
    "discord": 3122,
    "teams": 2100,
    "telegram": 1000,
    "api": 1000
  },
  "avg_response_time_ms": 450,
  "debates_triggered": 789
}
```

---

## Rate Limiting

Per-channel rate limits:

| Channel | Limit | Window |
|---------|-------|--------|
| Slack | 60 req/min | Per workspace |
| Discord | 50 req/min | Per guild |
| Teams | 60 req/min | Per tenant |
| Telegram | 30 req/min | Per chat |
| WhatsApp | 80 req/min | Per number |
| API | 100 req/min | Per API key |

---

## Related Documentation

- [CONTROL_PLANE.md](./CONTROL_PLANE.md) - Control plane architecture
- [CONNECTORS.md](./CONNECTORS.md) - Data connectors for knowledge ingestion
- [BOT_INTEGRATIONS.md](./BOT_INTEGRATIONS.md) - Detailed bot setup guides
