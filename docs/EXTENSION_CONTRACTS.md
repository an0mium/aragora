# Extension Contracts (Workspace/Gastown + Gateway/Moltbot)

This document captures the current cross-layer contracts used by the
Gastown (developer orchestration) and Moltbot (consumer assistant)
extensions. The goal is to keep Aragora's enterprise control plane stable
while allowing compatibility adapters to evolve independently.

## Canonical primitives

### Beads (atomic work units)

- Canonical store: `aragora.nomic.beads.BeadStore` (via `NomicBeadStore`).
- Workspace adapter: `aragora.workspace.bead.BeadManager` (Nomic-backed).
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
- Moltbot surface: `tests/extensions/moltbot/test_canvas.py`,
  `tests/extensions/moltbot/test_voice_wake.py`.

## Stability guidelines

- Prefer compatibility adapters over invasive rewrites.
- Keep Nomic stores authoritative; adapters translate status/metadata.
- Avoid regenerating SDKs until endpoint surfaces stabilize.
