# Extension Contracts (Workspace/Gastown + Gateway/OpenClaw)

This document captures the current cross-layer contracts used by the
Gastown (developer orchestration) and OpenClaw (consumer assistant)
extensions. The goal is to keep Aragora's enterprise control plane stable
while allowing compatibility adapters to evolve independently.

## Canonical primitives

### Beads (atomic work units)

- Canonical store: `aragora.nomic.beads.BeadStore` (via `NomicBeadStore`).
- Workspace adapter: `aragora.workspace.bead.BeadManager` (Nomic-backed).
- Persistence: canonical stores persist only when `ARAGORA_CANONICAL_STORE_PERSIST=1` (or
  `ARAGORA_STORE_DIR` is set). Otherwise, workspace adapters use an ephemeral temp store
  for local/dev safety.
- Identity:
  - `bead_id`: Nomic bead ID (opaque string)
  - `workspace_id`: owning workspace
  - `convoy_id`: owning convoy

Status mapping (workspace -> Nomic):
- PENDING -> PENDING
- ASSIGNED -> CLAIMED
- RUNNING -> RUNNING
- DONE -> COMPLETED
- FAILED -> FAILED
- SKIPPED -> CANCELLED

Metadata keys stored in Nomic:
- `workspace_id`
- `convoy_id`
- `workspace_status`
- `payload` (for workspace compatibility)
- `result` (for workspace compatibility)

### Convoys (work tracking units)

- Canonical store: `aragora.nomic.convoys.ConvoyManager`.
- Workspace adapter: `aragora.workspace.convoy.ConvoyTracker`.
- Gastown adapter: `aragora.extensions.gastown.convoy.ConvoyManager`.

Status mapping (workspace -> Nomic):
- CREATED -> PENDING
- ASSIGNING -> ACTIVE
- EXECUTING -> ACTIVE
- MERGING -> ACTIVE
- DONE -> COMPLETED
- FAILED -> FAILED
- CANCELLED -> CANCELLED

Metadata keys used by adapters:
- `workspace_id`
- `rig_id`
- `workspace_status`
- `gastown_status`
- `handoff_count`
- `handoffs`
- `artifacts`

ID contract:
- `convoy_id` is stable across adapters and used as the Nomic convoy ID.
- `rig_id` identifies the Gastown rig/crew owner of the convoy.

## Gateway + device runtime

### Gateway protocol adapter

- Adapter: `aragora.gateway.protocol.GatewayProtocolAdapter`.
- Session model: `aragora.gateway.protocol.GatewaySession`.
- Core methods:
  - `open_session(user_id, device_id, metadata)`
  - `close_session(session_id, reason)`
  - `update_presence(session_id, status)`
  - `get_session(session_id)`
  - `list_sessions(user_id, device_id, status)`

Session fields:
- `session_id`, `user_id`, `device_id`, `status`, `created_at`, `last_seen`, `metadata`, `end_reason`.

### Gateway HTTP surface (current)

These are the current Local Gateway endpoints (see `aragora.gateway.server`).

`GET /health` response:
```json
{
  "status": "healthy",
  "service": "aragora-gateway",
  "uptime_seconds": 120.5
}
```

`GET /stats` response (subset):
```json
{
  "running": true,
  "started_at": 1738101000.0,
  "messages_routed": 5,
  "messages_failed": 0,
  "inbox_size": 2,
  "devices_registered": 1,
  "routing_rules": 3
}
```

`POST /route` request:
```json
{
  "message_id": "msg-123",
  "channel": "slack",
  "sender": "user@example.com",
  "content": "Hello",
  "thread_id": "thread-1",
  "metadata": {
    "api_key": "optional-key"
  }
}
```

`POST /route` response:
```json
{
  "message_id": "msg-123",
  "agent_id": "default",
  "channel": "slack",
  "success": true,
  "error": null
}
```

`POST /device` request:
```json
{
  "device_id": "optional-id",
  "name": "MacBook Pro",
  "device_type": "macos",
  "capabilities": ["voice", "canvas"]
}
```

`POST /device` response:
```json
{
  "device_id": "dev-abc123",
  "status": "registered"
}
```

`GET /device/{device_id}` response (subset):
```json
{
  "device_id": "dev-abc123",
  "name": "MacBook Pro",
  "device_type": "macos",
  "status": "active",
  "capabilities": ["voice", "canvas"],
  "last_seen": 1738101123.4
}
```

### Gateway WebSocket events (current)

WebSocket endpoint: `GET /ws`.

Inbound ping (client → gateway):
```json
{"type": "ping"}
```

Outbound pong (gateway → client):
```json
{"type": "pong"}
```

Outbound inbox notification (gateway → client):
```json
{
  "type": "new_message",
  "message": {
    "message_id": "msg-123",
    "channel": "slack",
    "sender": "user@example.com",
    "content": "Hello",
    "timestamp": 1738101123.4
  }
}
```

### Gateway WS protocol draft (compatibility target)

These fields are proposed for OpenClaw parity and are not yet enforced.

- Envelope:
  - `type`: string
  - `protocol_version`: string (e.g., `"0.1"`)
  - `payload`: object
  - `request_id`: optional string for request/response pairing

### Device telemetry (draft)

These are proposed device telemetry messages for parity and are not yet enforced.

Heartbeat payload:
```json
{
  "device_id": "dev-abc123",
  "status": "active",
  "capabilities": ["voice", "canvas"],
  "last_seen": 1738101123.4,
  "telemetry": {
    "battery": 0.76,
    "network": "wifi",
    "os_version": "14.2"
  }
}
```

### Device node runtime

- Runtime: `aragora.gateway.device_node.DeviceNodeRuntime`.
- Config: `aragora.gateway.device_node.DeviceNodeRuntimeConfig`.
- Core methods:
  - `pair()` -> registers in DeviceRegistry
  - `heartbeat()`
  - `supports(capability)`
  - `unregister()`

Device registry contract:
- `DeviceNode` includes `name`, `device_type`, `capabilities`, `allowed_channels`, `metadata`.

## Capability routing + policy gating

- Routing: `aragora.gateway.capability_router` selects device endpoints
  based on declared capabilities + allowed channels.
- Policy gates: `aragora.computer_use.approval` and
  `aragora.computer_use.sandbox` are the canonical approval/sandbox
  controls for device and browser actions.

## Test hooks

- Beads/convoys parity: `tests/workspace/test_workspace_nomic.py`,
  `tests/extensions/test_gastown.py`.
- Gateway sessions: `tests/gateway/test_protocol_adapter.py`.
- Device runtime: `tests/gateway/test_device_node_runtime.py`.
- OpenClaw surface (moltbot module paths): `tests/extensions/moltbot/test_canvas.py`,
  `tests/extensions/moltbot/test_voice_wake.py`.

## Stability guidelines

- Prefer compatibility adapters over invasive rewrites.
- Keep Nomic stores authoritative; adapters translate status/metadata.
- Avoid regenerating SDKs until endpoint surfaces stabilize.
 - Add `protocol_version` to new gateway/device messages once parity surfaces are finalized.
