# Gateway Subsystem

The Gateway subsystem provides a device-local routing and authentication service for Aragora. It aggregates messages from multiple communication channels into a unified inbox, routes them to appropriate agents, and manages device registrations.

## Overview

The gateway implements the OpenClaw consumer interface model on top of the existing Aragora connector infrastructure, providing:

- **Unified Inbox**: Aggregate messages from Slack, Telegram, WhatsApp, email, and other channels
- **Agent Routing**: Route messages to the right agent based on configurable rules
- **Device Management**: Register and track devices with capability-based routing
- **Session Management**: WebSocket-based presence and session tracking
- **Persistence**: Multiple storage backends (memory, file, Redis)

## Architecture

```
LocalGateway (server.py)
    |
    +-- InboxAggregator (inbox.py)
    |       Unified message inbox with threading
    |
    +-- DeviceRegistry (device_registry.py)
    |       Device registration and capability tracking
    |
    +-- AgentRouter / CapabilityRouter (router.py, capability_router.py)
    |       Rule-based message routing with capability awareness
    |
    +-- GatewayProtocolAdapter (protocol.py)
    |       WebSocket session and presence management
    |
    +-- GatewayStore (persistence.py)
            Persistence backends (memory, file, Redis)
```

## Key Concepts

### Messages and Inbox

The `InboxAggregator` collects messages from all channels into a single inbox:

```python
from aragora.gateway import InboxAggregator, InboxMessage, MessagePriority

inbox = InboxAggregator(max_size=10000)

# Add a message
await inbox.add_message(InboxMessage(
    message_id="msg-123",
    channel="slack",
    sender="user@example.com",
    content="Hello, I need help",
    priority=MessagePriority.HIGH,
    thread_id="thread-456"
))

# Query messages
unread = await inbox.get_messages(is_read=False, limit=10)
slack_msgs = await inbox.get_messages(channel="slack")

# Mark as read
await inbox.mark_read(["msg-123"])
```

### Device Registration

The `DeviceRegistry` tracks connected devices and their capabilities:

```python
from aragora.gateway import DeviceRegistry, DeviceNode, DeviceStatus

registry = DeviceRegistry()

# Register a device
device = DeviceNode(
    name="Work Laptop",
    device_type="laptop",
    capabilities=["browser", "shell", "notifications"]
)
device_id = await registry.register(device)

# Send heartbeats to stay online
await registry.heartbeat(device_id)

# Query devices
online = await registry.list_devices(status=DeviceStatus.ONLINE)
has_camera = await registry.has_capability(device_id, "camera")
```

### Message Routing

The `AgentRouter` and `CapabilityRouter` determine which agent handles each message:

```python
from aragora.gateway import CapabilityRouter, CapabilityRule, RoutingRule

router = CapabilityRouter(default_agent="claude", device_registry=registry)

# Add routing rules
await router.add_rule(RoutingRule(
    rule_id="urgent",
    agent_id="priority-agent",
    content_pattern="URGENT",
    priority=100
))

# Add capability-aware rules
await router.add_capability_rule(CapabilityRule(
    rule_id="video-calls",
    agent_id="video-agent",
    channel_pattern="zoom*",
    required_capabilities=["camera", "mic"],
    fallback_capabilities=["mic"],
    fallback_agent_id="audio-agent"
))

# Route a message
result = await router.route_with_details("slack", message, device_id="dev-123")
print(f"Routed to {result.agent_id}, fallback={result.used_fallback}")
```

### Local Gateway Server

The `LocalGateway` provides HTTP and WebSocket endpoints:

```python
from aragora.gateway import LocalGateway, GatewayConfig

# Configure the gateway
config = GatewayConfig(
    host="0.0.0.0",
    port=8090,
    api_key="secret-key",
    enable_auth=True,
    max_inbox_size=5000
)

# Start the HTTP server
gw = LocalGateway(config=config)
runner = await gw.start_http()

# Or use programmatically
await gw.start()
response = await gw.route_message("slack", message)
inbox = await gw.get_inbox(limit=50)
stats = await gw.get_stats()
```

### Session Management

The `GatewayProtocolAdapter` manages WebSocket sessions:

```python
from aragora.gateway import GatewayProtocolAdapter, GatewayWebSocketProtocol

adapter = GatewayProtocolAdapter(gateway, store=FileGatewayStore())

# Open a session
session = await adapter.open_session(
    user_id="user-1",
    device_id="dev-laptop",
    metadata={"role": "admin"}
)

# Update presence
await adapter.update_presence(session.session_id, status="active")

# List active sessions
sessions = await adapter.list_sessions(status="active")

# Close session
await adapter.close_session(session.session_id, reason="logout")
```

### Secure Device Pairing

The `SecureDeviceRegistry` adds verification-code based pairing:

```python
from aragora.gateway import SecureDeviceRegistry

secure_registry = SecureDeviceRegistry(
    pairing_timeout=300.0,  # 5 minutes
    max_requests_per_minute=5,
    on_device_offline=lambda dev_id: print(f"{dev_id} went offline")
)

# Start presence monitoring
await secure_registry.start()

# Pairing flow
request = await secure_registry.request_pairing(
    device_name="New Phone",
    device_type="phone",
    capabilities=["notifications", "camera"]
)
print(f"Enter this code on the device: {request.verification_code}")

# User approves on trusted device
await secure_registry.approve_pairing(request.request_id)

# Device confirms with code
device = await secure_registry.confirm_pairing(
    request.request_id,
    verification_code="123456"
)
```

### Persistence

Multiple storage backends are available:

```python
from aragora.gateway import (
    get_gateway_store,
    InMemoryGatewayStore,
    FileGatewayStore,
    RedisGatewayStore
)

# In-memory (testing)
store = InMemoryGatewayStore()

# File-based (local-first)
store = FileGatewayStore(path="~/.aragora/gateway.json")

# Redis (production)
store = RedisGatewayStore(
    redis_url="redis://localhost:6379",
    message_ttl_seconds=86400 * 7,  # 7 days
)

# Auto-detect based on environment
store = get_gateway_store("auto")
```

## HTTP API Endpoints

When running as an HTTP server, the gateway exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (no auth) |
| `/stats` | GET | Gateway statistics |
| `/inbox` | GET | Get inbox messages |
| `/route` | POST | Route a message |
| `/device` | POST | Register a device |
| `/device/{id}` | GET | Get device info |
| `/ws` | GET | WebSocket connection |

## WebSocket Protocol

The WebSocket endpoint supports both simple and OpenClaw-compatible message formats:

```json
// Simple format
{"type": "session.open", "user_id": "user-1", "device_id": "dev-1"}

// OpenClaw format
{"type": "req", "id": "123", "method": "connect", "params": {...}}
```

Supported message types:
- `session.open`, `session.close`, `session.get`, `session.list`
- `session.resume`, `session.bind`
- `presence.update`
- `config.get`
- `ping` / `pong`

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ARAGORA_GATEWAY_SESSION_STORE` | Storage backend (memory, file, redis) |
| `ARAGORA_GATEWAY_SESSION_PATH` | Path for file backend |
| `ARAGORA_GATEWAY_SESSION_REDIS_URL` | Redis URL for redis backend |
| `REDIS_URL` | Fallback Redis URL |

## Module Reference

| Module | Description |
|--------|-------------|
| `server.py` | LocalGateway HTTP/WebSocket server |
| `inbox.py` | InboxAggregator, InboxMessage, InboxThread |
| `device_registry.py` | DeviceRegistry, DeviceNode, DeviceStatus |
| `device_node.py` | DeviceNodeRuntime for device-side operations |
| `device_security.py` | SecureDeviceRegistry with pairing ceremony |
| `router.py` | AgentRouter with rule-based routing |
| `capability_router.py` | CapabilityRouter for device-aware routing |
| `protocol.py` | GatewayProtocolAdapter, GatewayWebSocketProtocol |
| `persistence.py` | GatewayStore, InMemory/File/Redis backends |
| `canonical_api.py` | GatewayAPI protocol, GatewayRuntime |
