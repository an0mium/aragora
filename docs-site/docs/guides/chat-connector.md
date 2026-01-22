---
title: Chat Connector Developer Guide
description: Chat Connector Developer Guide
---

# Chat Connector Developer Guide

This guide covers creating and using chat platform connectors in Aragora. Chat connectors enable debates to be triggered and results delivered via platforms like Slack, Teams, Discord, Telegram, and more.

## Architecture Overview

```
                       ChatPlatformConnector (base.py)
                                  |
    +------------+-------+--------+-------+-----------+-------+
    |            |       |        |       |           |       |
SlackConnector  Teams  Discord  Telegram  WhatsApp  Google   ...
```

All connectors inherit from `ChatPlatformConnector` and implement a standardized interface for:
- Sending/updating/deleting messages
- Handling slash commands and interactions
- File upload/download
- Webhook verification
- Voice message support

## Quick Start: Using an Existing Connector

```python
from aragora.connectors.chat import SlackConnector

# Initialize connector
slack = SlackConnector(
    bot_token="xoxb-your-token",
    signing_secret="your-signing-secret",
)

# Send a message
response = await slack.send_message(
    channel_id="C123456",
    text="Debate result: REST API is recommended",
    blocks=slack.format_blocks(
        title="Debate Complete",
        body="REST API with OpenAPI spec is recommended",
        actions=[
            MessageButton(text="View Details", action_id="view_details"),
        ],
    ),
)

if response.success:
    print(f"Message sent: {response.message_id}")
```

## Available Connectors

| Connector | Platform | Import |
|-----------|----------|--------|
| `SlackConnector` | Slack | `aragora.connectors.chat.slack` |
| `TeamsConnector` | Microsoft Teams | `aragora.connectors.chat.teams` |
| `DiscordConnector` | Discord | `aragora.connectors.chat.discord` |
| `TelegramConnector` | Telegram | `aragora.connectors.chat.telegram` |
| `WhatsAppConnector` | WhatsApp | `aragora.connectors.chat.whatsapp` |
| `GoogleChatConnector` | Google Chat | `aragora.connectors.chat.google_chat` |

## Creating a Custom Connector

### Step 1: Inherit from ChatPlatformConnector

```python
from aragora.connectors.chat.base import ChatPlatformConnector
from aragora.connectors.chat.models import (
    BotCommand,
    FileAttachment,
    MessageButton,
    SendMessageResponse,
    UserInteraction,
    WebhookEvent,
)

class MyPlatformConnector(ChatPlatformConnector):
    """Connector for MyPlatform chat."""

    @property
    def platform_name(self) -> str:
        return "myplatform"

    @property
    def platform_display_name(self) -> str:
        return "My Platform"
```

### Step 2: Implement Required Abstract Methods

#### Message Operations (Required)

```python
async def send_message(
    self,
    channel_id: str,
    text: str,
    blocks: Optional[list[dict]] = None,
    thread_id: Optional[str] = None,
    **kwargs,
) -> SendMessageResponse:
    """Send a message to a channel."""
    try:
        response = await self._api_call(
            "send",
            channel=channel_id,
            text=text,
            blocks=blocks,
            reply_to=thread_id,
        )
        self._record_success()
        return SendMessageResponse(
            success=True,
            message_id=response["id"],
        )
    except Exception as e:
        self._record_failure(e)
        return SendMessageResponse(success=False, error=str(e))

async def update_message(
    self,
    channel_id: str,
    message_id: str,
    text: str,
    blocks: Optional[list[dict]] = None,
    **kwargs,
) -> SendMessageResponse:
    """Update an existing message."""
    # Similar implementation...

async def delete_message(
    self,
    channel_id: str,
    message_id: str,
    **kwargs,
) -> bool:
    """Delete a message."""
    # Similar implementation...
```

#### Command/Interaction Handling (Required)

```python
async def respond_to_command(
    self,
    command: BotCommand,
    text: str,
    blocks: Optional[list[dict]] = None,
    ephemeral: bool = True,
    **kwargs,
) -> SendMessageResponse:
    """Respond to a slash command."""
    channel_id = command.channel_id
    if ephemeral:
        return await self.send_ephemeral(
            channel_id, command.user_id, text, blocks, **kwargs
        )
    return await self.send_message(channel_id, text, blocks, **kwargs)

async def respond_to_interaction(
    self,
    interaction: UserInteraction,
    text: str,
    blocks: Optional[list[dict]] = None,
    replace_original: bool = False,
    **kwargs,
) -> SendMessageResponse:
    """Respond to button click, menu select, etc."""
    # Implementation depends on platform...
```

#### File Operations (Required)

```python
async def upload_file(
    self,
    channel_id: str,
    content: bytes,
    filename: str,
    content_type: str = "application/octet-stream",
    title: Optional[str] = None,
    thread_id: Optional[str] = None,
    **kwargs,
) -> FileAttachment:
    """Upload a file to a channel."""
    # Implementation...

async def download_file(
    self,
    file_id: str,
    **kwargs,
) -> FileAttachment:
    """Download a file by ID."""
    # Implementation...
```

#### Rich Content Formatting (Required)

```python
def format_blocks(
    self,
    title: Optional[str] = None,
    body: Optional[str] = None,
    fields: Optional[list[tuple[str, str]]] = None,
    actions: Optional[list[MessageButton]] = None,
    **kwargs,
) -> list[dict]:
    """Format content into platform-specific blocks."""
    blocks = []
    if title:
        blocks.append({"type": "header", "text": title})
    if body:
        blocks.append({"type": "section", "text": body})
    if fields:
        blocks.append({
            "type": "fields",
            "fields": [{"label": k, "value": v} for k, v in fields],
        })
    if actions:
        blocks.append({
            "type": "actions",
            "elements": [self.format_button(a.text, a.action_id, a.value) for a in actions],
        })
    return blocks

def format_button(
    self,
    text: str,
    action_id: str,
    value: Optional[str] = None,
    style: str = "default",
    url: Optional[str] = None,
) -> dict:
    """Format a button element."""
    return {
        "type": "button",
        "text": text,
        "action_id": action_id,
        "value": value or action_id,
        "style": style,
    }
```

#### Webhook Handling (Required)

```python
def verify_webhook(
    self,
    headers: dict[str, str],
    body: bytes,
) -> bool:
    """Verify webhook signature."""
    import hmac
    import hashlib

    signature = headers.get("X-MyPlatform-Signature")
    if not signature or not self.signing_secret:
        return False

    expected = hmac.new(
        self.signing_secret.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(signature, expected)

def parse_webhook_event(
    self,
    headers: dict[str, str],
    body: bytes,
) -> WebhookEvent:
    """Parse webhook payload."""
    import json
    data = json.loads(body)
    return WebhookEvent(
        event_type=data.get("type", "unknown"),
        platform=self.platform_name,
        raw_payload=data,
        # Map platform fields to standard fields...
    )
```

### Step 3: Use Built-in Resilience Features

The base class provides circuit breaker and retry support:

```python
async def send_message(self, channel_id: str, text: str, **kwargs):
    # Use _with_retry for automatic retry with exponential backoff
    return await self._with_retry(
        "send_message",
        self._do_send_message,
        channel_id,
        text,
        **kwargs,
        max_retries=3,
        base_delay=1.0,
    )

async def _do_send_message(self, channel_id: str, text: str, **kwargs):
    # Actual API call
    response = await self._http_client.post(...)
    if not response.ok and self._is_retryable_status_code(response.status):
        raise ConnectionError(f"API returned {response.status}")
    return response.json()
```

## Data Models

### SendMessageResponse

```python
@dataclass
class SendMessageResponse:
    success: bool
    message_id: Optional[str] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None
```

### BotCommand

```python
@dataclass
class BotCommand:
    command: str           # e.g., "/debate"
    text: str             # Arguments after command
    user_id: str
    user_name: str
    channel_id: str
    response_url: Optional[str] = None
```

### UserInteraction

```python
@dataclass
class UserInteraction:
    interaction_id: str
    action_id: str         # Button/menu action ID
    value: Optional[str]
    user_id: str
    channel_id: str
    message_id: Optional[str] = None
    response_url: Optional[str] = None
```

### FileAttachment

```python
@dataclass
class FileAttachment:
    id: str
    filename: str
    content_type: str
    size_bytes: int
    url: Optional[str] = None
    content: Optional[bytes] = None  # Populated on download
```

### WebhookEvent

```python
@dataclass
class WebhookEvent:
    event_type: str        # "message", "command", "interaction"
    platform: str
    timestamp: datetime
    raw_payload: dict
    message: Optional[ChatMessage] = None
    command: Optional[BotCommand] = None
    interaction: Optional[UserInteraction] = None
```

## Circuit Breaker Configuration

```python
connector = SlackConnector(
    bot_token="...",
    enable_circuit_breaker=True,      # Default: True
    circuit_breaker_threshold=5,       # Failures before opening
    circuit_breaker_cooldown=60.0,    # Seconds before retry
    request_timeout=30.0,              # HTTP timeout
)
```

### Circuit Breaker States

- **CLOSED**: Normal operation, requests proceed
- **OPEN**: Too many failures, requests blocked
- **HALF_OPEN**: Testing if service recovered

```python
# Check circuit breaker manually
can_proceed, error = connector._check_circuit_breaker()
if not can_proceed:
    print(f"Circuit open: \{error\}")
```

## Voice Message Support

```python
# Send voice response
await connector.send_voice_message(
    channel_id="C123456",
    audio_content=audio_bytes,
    filename="response.mp3",
    content_type="audio/mpeg",
)

# Get voice message for transcription
voice = await connector.get_voice_message(file_id="F123456")
if voice:
    transcript = await transcribe(voice.audio_content)
```

## Evidence Collection

Connectors can collect chat history as debate evidence:

```python
evidence = await connector.collect_evidence(
    channel_id="C123456",
    query="API design discussion",
    limit=100,
    include_threads=True,
    min_relevance=0.5,
)

for item in evidence:
    print(f"{item.author}: {item.content} (relevance: {item.relevance})")
```

## Registering Custom Connectors

```python
from aragora.connectors.chat.registry import register_connector

register_connector("myplatform", MyPlatformConnector)

# Now available via registry
from aragora.connectors.chat.registry import get_connector

connector = get_connector("myplatform", bot_token="...")
```

## Testing Connectors

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.fixture
def mock_connector():
    connector = MyPlatformConnector(bot_token="test-token")
    connector._api_call = AsyncMock()
    return connector

async def test_send_message(mock_connector):
    mock_connector._api_call.return_value = {"id": "msg_123"}

    response = await mock_connector.send_message(
        channel_id="ch_1",
        text="Hello",
    )

    assert response.success
    assert response.message_id == "msg_123"
    mock_connector._api_call.assert_called_once()

async def test_circuit_breaker_opens(mock_connector):
    # Force circuit breaker to open
    for _ in range(5):
        mock_connector._record_failure(Exception("API error"))

    can_proceed, _ = mock_connector._check_circuit_breaker()
    assert not can_proceed
```

## Error Handling Best Practices

1. **Always use _with_retry for API calls** - Built-in exponential backoff
2. **Record success/failure** - Keeps circuit breaker accurate
3. **Check retryable status codes** - 429, 500, 502, 503, 504
4. **Log with platform context** - Use `self.platform_name` in logs
5. **Return standardized responses** - Use `SendMessageResponse` dataclass

```python
async def api_call_example(self, endpoint: str, data: dict):
    try:
        result = await self._with_retry(
            f"api_call_\{endpoint\}",
            self._http_post,
            endpoint,
            data,
            retryable_exceptions=(ConnectionError, TimeoutError),
        )
        return result
    except Exception as e:
        logger.error(f"{self.platform_name} API call failed: \{e\}")
        raise
```

## Related Documentation

- [Debate Origin Routing](../core-concepts/architecture#debate-routing) - How debates are triggered from chat
- [Evidence Collection](./EVIDENCE_COLLECTION.md) - Collecting evidence from platforms
- [Voice TTS Bridge](./VOICE_TTS.md) - Voice message transcription and synthesis
