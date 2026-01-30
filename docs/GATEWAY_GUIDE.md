# Gateway Guide

Device-local message routing and unified inbox for OpenClaw parity.

## Overview

The `aragora.gateway` module provides local-first communication:

- **LocalGateway**: Device-local HTTP/WebSocket server
- **InboxAggregator**: Unified inbox across all channels
- **DeviceRegistry**: Device registration and capability management
- **AgentRouter**: Per-channel routing to appropriate agents

## Quick Start

```python
from aragora.gateway import LocalGateway, GatewayConfig

# Create gateway with configuration
config = GatewayConfig(
    host="127.0.0.1",
    port=8090,
    api_key="your-secret-key",
    enable_auth=True,
)

gateway = LocalGateway(config=config)

# Start HTTP server
runner = await gateway.start_http()

# Gateway is now accepting requests at http://localhost:8090
print(f"Gateway running: {gateway.is_running}")

# Stop when done
await gateway.stop_http()
```

## Core Concepts

### Session Persistence (OpenClaw WS)

Gateway WebSocket sessions can be persisted by setting:

- `ARAGORA_GATEWAY_SESSION_STORE=memory|file|redis|auto`
- `ARAGORA_GATEWAY_SESSION_PATH` (for file backend)
- `ARAGORA_GATEWAY_SESSION_REDIS_URL` or `REDIS_URL` (for redis backend)

### LocalGateway

The central server that handles all message routing:

```python
from aragora.gateway import LocalGateway, GatewayConfig, InboxMessage

# Configure gateway
config = GatewayConfig(
    host="0.0.0.0",  # Listen on all interfaces
    port=8090,
    api_key="secret-api-key",
    enable_auth=True,
    cloud_proxy_url="https://api.aragora.ai",  # Optional cloud proxy
    max_inbox_size=10000,
    allowed_channels=["slack", "telegram", "email"],
)

gateway = LocalGateway(config=config)

# Route a message
message = InboxMessage(
    message_id="msg-123",
    channel="slack",
    sender="user@example.com",
    content="Help me with this task",
    metadata={"api_key": config.api_key},
)

response = await gateway.route_message("slack", message)
print(f"Routed to agent: {response.agent_id}")
```

### InboxAggregator

Unified inbox across all communication channels:

```python
from aragora.gateway import InboxAggregator, InboxMessage, MessagePriority

# Create inbox
inbox = InboxAggregator(max_size=10000)

# Add messages from different channels
await inbox.add_message(InboxMessage(
    message_id="msg-1",
    channel="slack",
    sender="alice",
    content="Urgent: Server down!",
    priority=MessagePriority.HIGH,
))

await inbox.add_message(InboxMessage(
    message_id="msg-2",
    channel="email",
    sender="bob@example.com",
    content="Weekly report",
    priority=MessagePriority.LOW,
))

# Get messages with filters
urgent = await inbox.get_messages(
    channel=None,  # All channels
    limit=10,
    is_read=False,
)

# Mark as read
await inbox.mark_read("msg-1")
```

### DeviceRegistry

Register and manage devices:

```python
from aragora.gateway import DeviceRegistry, DeviceNode, DeviceStatus

# Create registry
registry = DeviceRegistry()

# Register a device
device = DeviceNode(
    name="MacBook Pro",
    device_type="laptop",
    capabilities=["browser", "terminal", "camera"],
)

device_id = await registry.register(device)

# Get device info
info = await registry.get(device_id)
print(f"Device: {info.name}, Status: {info.status}")

# Update status
await registry.update_status(device_id, DeviceStatus.ONLINE)

# List all devices
devices = await registry.list_devices(status=DeviceStatus.ONLINE)
```

### AgentRouter

Route messages to appropriate agents:

```python
from aragora.gateway import AgentRouter, RoutingRule

# Create router
router = AgentRouter()

# Add routing rules
await router.add_rule(RoutingRule(
    channel="slack",
    pattern=r"#support.*",  # Messages from support channels
    agent_id="support-agent",
))

await router.add_rule(RoutingRule(
    channel="telegram",
    pattern=r".*",  # All Telegram messages
    agent_id="personal-agent",
))

# Route a message
agent_id = await router.route("slack", message)
```

### CapabilityRouter

Route messages based on device capabilities with fallback agents:

```python
from aragora.gateway import CapabilityRouter, CapabilityRule

router = CapabilityRouter(default_agent="default", device_registry=registry)

await router.add_capability_rule(CapabilityRule(
    rule_id="video-support",
    agent_id="video-agent",
    channel_pattern="slack",
    required_capabilities=["camera", "mic"],
    fallback_capabilities=["mic"],
    fallback_agent_id="audio-agent",
))

result = await router.route_with_details("slack", message, device_id="device-1")
print(result.agent_id, result.used_fallback, result.missing_capabilities)
```

## HTTP API Endpoints

The gateway exposes these HTTP endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (no auth) |
| `GET` | `/stats` | Gateway statistics |
| `GET` | `/inbox` | Get inbox messages |
| `POST` | `/route` | Route a message to agent |
| `POST` | `/device` | Register a device |
| `GET` | `/device/{id}` | Get device info |
| `WS` | `/ws` | Real-time inbox updates |

### Authentication

Include API key in headers:

```bash
# Using X-API-Key header
curl -H "X-API-Key: your-secret-key" http://localhost:8090/stats

# Using Bearer token
curl -H "Authorization: Bearer your-secret-key" http://localhost:8090/inbox
```

### Route Message

```bash
curl -X POST http://localhost:8090/route \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "channel": "slack",
    "sender": "user@example.com",
    "content": "Help me with this task",
    "thread_id": "thread-123"
  }'
```

Response:
```json
{
  "message_id": "msg-abc123",
  "agent_id": "support-agent",
  "channel": "slack",
  "success": true
}
```

### WebSocket Events

Connect to `/ws` for real-time updates:

```javascript
const ws = new WebSocket("ws://localhost:8090/ws");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "new_message") {
    console.log("New message:", data.message);
  }
};

// Send ping
ws.send(JSON.stringify({ type: "ping" }));
```

## API Reference

### GatewayConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | str | "127.0.0.1" | Listen host |
| `port` | int | 8090 | Listen port |
| `api_key` | str | "" | API key for auth |
| `enable_auth` | bool | True | Enable authentication |
| `cloud_proxy_url` | str | None | Cloud proxy URL |
| `max_inbox_size` | int | 10000 | Max inbox messages |
| `allowed_channels` | list | [] | Allowed channel names |

### InboxMessage

| Field | Type | Description |
|-------|------|-------------|
| `message_id` | str | Unique message ID |
| `channel` | str | Source channel (slack, telegram, etc.) |
| `sender` | str | Sender identifier |
| `content` | str | Message content |
| `timestamp` | float | Unix timestamp |
| `thread_id` | str | Thread/conversation ID |
| `is_read` | bool | Read status |
| `is_replied` | bool | Reply status |
| `priority` | MessagePriority | Message priority |
| `metadata` | dict | Additional metadata |

### AgentResponse

| Field | Type | Description |
|-------|------|-------------|
| `message_id` | str | Original message ID |
| `agent_id` | str | Handling agent ID |
| `channel` | str | Source channel |
| `content` | str | Response content |
| `success` | bool | Success status |
| `error` | str | Error message if failed |

## Integration

### With Workspace

```python
from aragora.gateway import LocalGateway
from aragora.workspace import WorkspaceManager

gateway = LocalGateway()
ws = WorkspaceManager()

# Create bead from incoming message
async def on_message(message: InboxMessage):
    convoy = await ws.create_convoy(
        rig_id="rig-inbox",
        bead_specs=[{
            "title": f"Handle: {message.content[:50]}",
            "payload": {
                "message_id": message.message_id,
                "channel": message.channel,
                "sender": message.sender,
            },
        }],
    )
    await ws.start_convoy(convoy.convoy_id)
```

### With Agent Fabric

```python
from aragora.gateway import LocalGateway
from aragora.fabric import AgentFabric

gateway = LocalGateway()
fabric = AgentFabric()

# Route to fabric for scheduling
async def route_to_fabric(channel: str, message: InboxMessage):
    # Find best agent from pool
    agent = await fabric.get_available_agent(
        pool_id="pool-support",
        task_type="message",
    )

    # Schedule task
    task = await fabric.schedule(
        agent_id=agent.id,
        task_type="handle_message",
        payload={"message": message.to_dict()},
    )
    return task.id
```

## Examples

### Multi-Channel Inbox

```python
# Start gateway
gateway = LocalGateway(config)
await gateway.start_http()

# Get unified inbox
messages = await gateway.get_inbox(limit=50)

# Filter by channel
slack_messages = await gateway.get_inbox(channel="slack", limit=20)

# Get unread only
unread = await gateway._inbox.get_messages(is_read=False, limit=100)
```

### Device Registration

```python
# Register mobile device
device = DeviceNode(
    name="iPhone 15",
    device_type="mobile",
    capabilities=["notifications", "camera", "location"],
)

device_id = await gateway.register_device(device)

# Register desktop
desktop = DeviceNode(
    name="Work Laptop",
    device_type="desktop",
    capabilities=["browser", "terminal", "screen_share"],
)

await gateway.register_device(desktop)
```

### Statistics Monitoring

```python
# Get gateway stats
stats = await gateway.get_stats()

print(f"Running: {stats['running']}")
print(f"Messages routed: {stats['messages_routed']}")
print(f"Messages failed: {stats['messages_failed']}")
print(f"Inbox size: {stats['inbox_size']}")
print(f"Devices registered: {stats['devices_registered']}")
print(f"Routing rules: {stats['routing_rules']}")
```

---

*Part of Aragora control plane for multi-agent robust decisionmaking*
