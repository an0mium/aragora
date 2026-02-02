# Chat Platform Connectors

Unified interface for chat platform integrations in Aragora. This module provides a consistent API for interacting with various chat platforms including Slack, Microsoft Teams, Discord, Telegram, WhatsApp, Google Chat, Signal, and iMessage.

## Overview

The chat connectors module abstracts away platform-specific differences, providing:

- **Unified message format** - `ChatMessage`, `ChatUser`, `ChatChannel` work across all platforms
- **Registry pattern** - Auto-discover and manage connectors for configured platforms
- **Resilience built-in** - Circuit breakers, retries, and timeouts on all API calls
- **Rich formatting** - Platform-native blocks, cards, and embeds
- **Thread management** - Consistent threading APIs across platforms
- **Voice integration** - TTS synthesis and STT transcription bridges
- **Evidence collection** - Gather chat messages as debate evidence

## Supported Platforms

| Platform | Connector | Rich Content | Threads | Voice | Commands | File Upload |
|----------|-----------|--------------|---------|-------|----------|-------------|
| Slack | `SlackConnector` | Block Kit | thread_ts | - | Slash commands | files.upload |
| Teams | `TeamsConnector` | Adaptive Cards | replyToId | - | Bot commands | OneDrive/Graph |
| Discord | `DiscordConnector` | Embeds + Components | Forum threads | - | Slash commands | Attachments |
| Telegram | `TelegramConnector` | Inline keyboards | reply_to | Voice notes | Bot commands | sendDocument |
| WhatsApp | `WhatsAppConnector` | Interactive messages | Context | Audio messages | Text prefix | Cloud API |
| Google Chat | `GoogleChatConnector` | Cards | Thread keys | - | Slash commands | Drive |
| Signal | `SignalConnector` | Basic text | Reply quote | Voice | - | Attachments |
| iMessage | `IMessageConnector` | Basic text | Reply | - | - | BlueBubbles |

## Architecture

### Base Connector

All connectors inherit from `ChatPlatformConnector`, which composes several mixins:

```
ChatPlatformConnector
    |-- HTTPResilienceMixin      # Circuit breaker, retry, HTTP handling
    |-- MessagingMixin           # Send/update/delete messages
    |-- FileOperationsMixin      # File upload/download, voice messages
    |-- ChannelUserMixin         # Channel/user info, reactions, pinning
    |-- WebhookMixin             # Webhook verification and event parsing
    |-- EvidenceMixin            # Evidence collection from channels
    |-- RichContextMixin         # Context fetching for deliberation
    |-- SessionMixin             # Session management integration
```

### Registry Pattern

The `ChatPlatformRegistry` provides factory and management for connectors:

```python
from aragora.connectors.chat import get_connector, get_registry

# Get a specific connector
slack = get_connector("slack")
await slack.send_message(channel_id, "Hello!")

# Get all configured connectors
registry = get_registry()
for platform, connector in registry.all().items():
    await connector.send_message(channel, f"Hello from {platform}!")

# Broadcast to multiple platforms
await registry.broadcast(
    "Important announcement!",
    channels={"slack": "C123", "teams": "channel-guid", "discord": "456"}
)
```

Connectors are lazy-loaded on first access and cached as singletons.

## Message Normalization

All platforms use unified data models from `models.py`:

### ChatMessage

```python
@dataclass
class ChatMessage:
    id: str                    # Platform message ID
    platform: str              # "slack", "teams", etc.
    channel: ChatChannel       # Channel info
    author: ChatUser           # Message author
    content: str               # Text content
    message_type: MessageType  # TEXT, RICH, FILE, VOICE, COMMAND
    thread_id: str | None      # Thread/reply reference
    timestamp: datetime
    blocks: list[dict] | None  # Platform-specific rich content
    attachments: list[dict]
    metadata: dict             # Platform-specific extras
```

### ChatUser

```python
@dataclass
class ChatUser:
    id: str
    platform: str
    username: str | None
    display_name: str | None
    email: str | None
    avatar_url: str | None
    is_bot: bool
    timezone: str | None       # IANA timezone for enrichment
    language: str | None       # ISO 639-1 code
    role: UserRole             # OWNER, ADMIN, MEMBER, GUEST
```

### ChatChannel

```python
@dataclass
class ChatChannel:
    id: str
    platform: str
    name: str | None
    is_private: bool
    is_dm: bool
    team_id: str | None        # Workspace/Guild/Organization
    channel_type: ChannelType  # PUBLIC, PRIVATE, DM, GROUP_DM, THREAD
    topic: str | None
    member_count: int | None
```

## Platform-Specific Features

### Slack - Block Kit

```python
slack = get_connector("slack")

# Using format_blocks helper
blocks = slack.format_blocks(
    title="Debate Results",
    body="The consensus was reached with 85% confidence.",
    fields=[("Rounds", "3"), ("Agents", "5")],
    actions=[
        MessageButton(text="View Details", action_id="view_details"),
        MessageButton(text="Approve", action_id="approve", style="primary"),
    ],
)
await slack.send_message(channel_id, "Results ready", blocks=blocks)

# Additional Slack features
await slack.add_reaction(channel_id, message_ts, "thumbsup")
await slack.pin_message(channel_id, message_ts)
await slack.open_modal(trigger_id, view_payload)
```

### Teams - Adaptive Cards

```python
from aragora.connectors.chat.teams_adaptive_cards import TeamsAdaptiveCards, AgentContribution

teams = get_connector("teams")

# Rich verdict card
card = TeamsAdaptiveCards.verdict_card(
    topic="Should we migrate to microservices?",
    verdict="Yes, for services needing independent scaling",
    confidence=0.85,
    agents=[
        AgentContribution(name="Claude", position="for", key_point="Better scaling"),
        AgentContribution(name="GPT-4", position="for", key_point="Team autonomy"),
        AgentContribution(name="Gemini", position="against", key_point="Complexity"),
    ],
    receipt_id="rec_abc123",
)

await teams.send_message(conversation_id, "Debate complete", blocks=[card])
```

### Discord - Embeds and Components

```python
discord = get_connector("discord")

# Embed with buttons
embed = discord.format_blocks(
    title="Vote Required",
    body="AI agents have reached a verdict.",
    fields=[("Confidence", "85%"), ("Rounds", "3")],
    color=0x00FF00,  # Green
)

# Components (buttons) go in kwargs
await discord.send_message(
    channel_id,
    "Please vote:",
    blocks=embed,
    components=[{
        "type": 1,  # ACTION_ROW
        "components": [
            discord.format_button("Agree", "vote_agree", style="primary"),
            discord.format_button("Disagree", "vote_disagree", style="danger"),
        ]
    }]
)
```

### Telegram - Inline Keyboards

```python
telegram = get_connector("telegram")

# Inline keyboard buttons
blocks = [
    {"type": "button", "text": "Approve", "action_id": "approve"},
    {"type": "button", "text": "Reject", "action_id": "reject"},
    {"type": "url_button", "text": "View Details", "url": "https://..."},
]

await telegram.send_message(chat_id, "Decision ready:", blocks=blocks)

# Rich media
await telegram.send_photo(chat_id, photo_url, caption="Debate visualization")
await telegram.send_voice_message(chat_id, audio_bytes)
```

### WhatsApp - Interactive Messages

```python
whatsapp = get_connector("whatsapp")

# Button reply
blocks = whatsapp.format_blocks(
    body="Do you approve this decision?",
    actions=[
        MessageButton(text="Yes", action_id="approve"),
        MessageButton(text="No", action_id="reject"),
    ],
)
await whatsapp.send_message(phone_number, "Vote needed", blocks=blocks)

# Template messages (pre-approved)
await whatsapp.send_template(
    phone_number,
    template_name="decision_notification",
    language_code="en",
    components=[{"type": "body", "parameters": [{"type": "text", "text": "Budget approval"}]}],
)
```

## Thread Management

Each platform has different threading models. The `ThreadManager` abstraction provides a unified interface:

```python
from aragora.connectors.chat import ThreadInfo, ThreadStats

# Platform-specific managers
from aragora.connectors.chat.slack import SlackThreadManager
from aragora.connectors.chat.teams import TeamsThreadManager

manager = SlackThreadManager(connector=slack)

# Get thread info
thread = await manager.get_thread(thread_ts, channel_id)
print(f"Thread has {thread.message_count} messages")

# Get thread messages with pagination
messages, cursor = await manager.get_thread_messages(
    thread_id=thread_ts,
    channel_id=channel_id,
    limit=50,
)

# Reply to thread
await manager.reply_to_thread(thread_ts, channel_id, "Adding to discussion...")

# Get context for AI prompts
context = await manager.get_thread_context(thread_ts, channel_id, max_messages=20)
```

### Platform Threading Models

| Platform | Thread ID | Start Thread | Reply | Notes |
|----------|-----------|--------------|-------|-------|
| Slack | `thread_ts` (timestamp) | First message ts | `thread_ts` param | Threads are message replies |
| Teams | `replyToId` | Conversation start | `replyToId` param | Uses Bot Framework |
| Discord | Channel ID | Create thread channel | Send to thread | Threads are channels |
| Telegram | `message_id` | N/A | `reply_to_message_id` | Reply chains only |
| WhatsApp | `context.id` | N/A | `context` object | Reply context only |

## Voice/TTS Integration

### Text-to-Speech (TTSBridge)

Synthesize text responses as audio for chat platforms:

```python
from aragora.connectors.chat import get_tts_bridge

bridge = get_tts_bridge()

# Simple synthesis
audio_path = await bridge.synthesize_response(
    "The debate concluded with consensus on option A.",
    voice="narrator"
)

# Send to chat platform
with open(audio_path, "rb") as f:
    await telegram.send_voice_message(chat_id, f.read())

# Or use the convenience method
await bridge.send_voice_response(
    connector=telegram,
    channel_id=chat_id,
    text="Consensus reached with 85% confidence.",
    voice="consensus",
)

# Specialized synthesizers
audio = await bridge.synthesize_debate_summary(
    task="Should we approve the budget?",
    final_answer="Yes, with minor amendments",
    consensus_reached=True,
    confidence=0.85,
    rounds_used=3,
)
```

### Speech-to-Text (VoiceBridge)

Transcribe voice messages from users:

```python
from aragora.connectors.chat import get_voice_bridge, VoiceMessage

bridge = get_voice_bridge()

# From voice message object
voice_msg = VoiceMessage(
    id="msg_123",
    channel=channel,
    author=user,
    duration_seconds=15.5,
    file=file_attachment,
)
transcription = await bridge.transcribe_voice_message(
    voice_msg,
    connector=telegram,  # Used to download if needed
    language="en",
)

# Or directly from file
transcription = await bridge.process_chat_audio(
    connector=whatsapp,
    file_id=media_id,
    language="en",
)
```

## Adding New Platforms

### Step 1: Create Connector Class

```python
# aragora/connectors/chat/myplatform.py
from aragora.connectors.chat.base import ChatPlatformConnector
from aragora.connectors.chat.models import (
    ChatMessage, ChatUser, ChatChannel, SendMessageResponse,
    BotCommand, UserInteraction, WebhookEvent, MessageButton,
)

class MyPlatformConnector(ChatPlatformConnector):
    """Connector for MyPlatform."""

    def __init__(self, api_key: str | None = None, **config):
        super().__init__(
            bot_token=api_key or os.environ.get("MYPLATFORM_API_KEY"),
            **config,
        )

    @property
    def platform_name(self) -> str:
        return "myplatform"

    @property
    def platform_display_name(self) -> str:
        return "My Platform"
```

### Step 2: Implement Required Methods

```python
    async def send_message(
        self,
        channel_id: str,
        text: str,
        blocks: list[dict] | None = None,
        thread_id: str | None = None,
        **kwargs,
    ) -> SendMessageResponse:
        """Send a message."""
        # Use self._http_request for resilience
        success, data, error = await self._http_request(
            method="POST",
            url=f"{API_BASE}/messages",
            json={"channel": channel_id, "text": text},
            operation="send_message",
        )
        if success:
            return SendMessageResponse(success=True, message_id=data["id"])
        return SendMessageResponse(success=False, error=error)

    async def update_message(self, channel_id, message_id, text, blocks=None, **kwargs):
        ...

    async def delete_message(self, channel_id, message_id, **kwargs) -> bool:
        ...

    def format_blocks(self, title=None, body=None, fields=None, actions=None, **kwargs):
        """Convert to platform-native format."""
        ...

    def format_button(self, text, action_id, value=None, style=None, url=None):
        ...

    def verify_webhook(self, headers: dict, body: bytes) -> bool:
        """Verify webhook signature."""
        ...

    def parse_webhook_event(self, headers: dict, body: bytes) -> WebhookEvent:
        """Parse incoming webhook."""
        ...

    async def respond_to_command(self, command, text, blocks=None, ephemeral=False, **kwargs):
        ...

    async def respond_to_interaction(self, interaction, text, blocks=None, replace_original=False, **kwargs):
        ...
```

### Step 3: Register in Registry

Add to `registry.py`:

```python
def _lazy_load_connector(platform: str):
    ...
    elif platform == "myplatform":
        from aragora.connectors.chat.myplatform import MyPlatformConnector
        _CONNECTOR_CLASSES["myplatform"] = MyPlatformConnector
        return MyPlatformConnector
```

Add to `get_configured_platforms()`:

```python
if os.environ.get("MYPLATFORM_API_KEY"):
    configured.append("myplatform")
```

### Step 4: Add to __init__.py

```python
elif name == "MyPlatformConnector":
    from .myplatform import MyPlatformConnector
    return MyPlatformConnector
```

## Configuration

### Environment Variables

| Platform | Required Variables | Optional |
|----------|-------------------|----------|
| Slack | `SLACK_BOT_TOKEN` | `SLACK_SIGNING_SECRET`, `SLACK_WEBHOOK_URL` |
| Teams | `TEAMS_APP_ID`, `TEAMS_APP_PASSWORD` | `TEAMS_TENANT_ID`, `TEAMS_REQUEST_TIMEOUT` |
| Discord | `DISCORD_BOT_TOKEN` | `DISCORD_APPLICATION_ID`, `DISCORD_PUBLIC_KEY` |
| Telegram | `TELEGRAM_BOT_TOKEN` | `TELEGRAM_WEBHOOK_URL`, `TELEGRAM_WEBHOOK_SECRET` |
| WhatsApp | `WHATSAPP_ACCESS_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID` | `WHATSAPP_BUSINESS_ACCOUNT_ID`, `WHATSAPP_APP_SECRET` |
| Google Chat | `GOOGLE_CHAT_CREDENTIALS` | - |
| Signal | `SIGNAL_CLI_URL`, `SIGNAL_PHONE_NUMBER` | - |
| iMessage | `BLUEBUBBLES_URL`, `BLUEBUBBLES_PASSWORD` | - |

### TTS Configuration

```bash
ARAGORA_TTS_DEFAULT_VOICE=narrator
ARAGORA_TTS_MAX_TEXT=4000
ARAGORA_TTS_CACHE_DIR=/tmp/tts_cache
```

### Circuit Breaker Settings

```python
connector = SlackConnector(
    enable_circuit_breaker=True,     # Enable fault tolerance
    circuit_breaker_threshold=5,      # Failures before opening
    circuit_breaker_cooldown=60.0,    # Seconds before recovery
    request_timeout=30.0,             # HTTP timeout
)
```

## Examples

### Basic Message Sending

```python
from aragora.connectors.chat import get_connector

async def notify_all_platforms(message: str):
    for platform in ["slack", "teams", "discord"]:
        connector = get_connector(platform)
        if connector and connector.is_configured:
            await connector.send_message(
                channel_id=config[f"{platform}_channel"],
                text=message,
            )
```

### Handling Webhooks

```python
from aragora.connectors.chat import get_connector

async def handle_webhook(platform: str, headers: dict, body: bytes):
    connector = get_connector(platform)

    # Verify signature
    if not connector.verify_webhook(headers, body):
        return {"error": "Invalid signature"}, 401

    # Parse event
    event = connector.parse_webhook_event(headers, body)

    # Handle URL verification (Slack, WhatsApp)
    if event.is_verification:
        return event.challenge

    # Handle message
    if event.event_type == "message":
        message = await connector.parse_message(event.raw_payload)
        # Process message...

    # Handle command
    if event.command:
        await connector.respond_to_command(
            event.command,
            text="Command received!",
            ephemeral=True,
        )

    # Handle interaction (button click)
    if event.interaction:
        await connector.respond_to_interaction(
            event.interaction,
            text="Processing...",
            replace_original=True,
        )

    return {"ok": True}
```

### Collecting Evidence for Debates

```python
from aragora.connectors.chat import get_connector

async def gather_context(platform: str, channel_id: str, query: str):
    connector = get_connector(platform)

    # Collect messages as evidence
    evidence = await connector.collect_evidence(
        channel_id=channel_id,
        query=query,
        limit=100,
        include_threads=True,
        min_relevance=0.3,
    )

    # Use in debate
    context = "\n".join([
        f"[{e.author_name}]: {e.content}"
        for e in evidence[:20]
    ])
    return context
```

### Multi-Platform Broadcast

```python
from aragora.connectors.chat import get_registry

async def broadcast_decision(decision: str, channels: dict):
    registry = get_registry()

    # Build platform-specific blocks
    blocks = {}
    for platform, connector in registry.all().items():
        blocks[platform] = connector.format_blocks(
            title="Decision Announced",
            body=decision,
            actions=[
                MessageButton(text="Acknowledge", action_id="ack"),
            ],
        )

    # Broadcast
    results = await registry.broadcast(
        text=decision,
        channels=channels,
        blocks=blocks,
    )

    return results
```
