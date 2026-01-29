# Gastown + Moltbot Parity Backlog

This backlog tracks parity work for the Gastown (developer orchestration) and
Moltbot (consumer assistant) extension layers. It focuses on gaps between
existing Aragora implementations and parity targets, and provides acceptance
criteria for each item.

Status legend:
- implemented: present and usable in the codebase
- partial: scaffolding exists, but lacks production wiring or protocol parity
- missing: no implementation yet

## Parity Matrix Snapshot (Code-Backed)

### Gastown (Developer Orchestration)

| Capability | Status | Code refs | Notes |
| --- | --- | --- | --- |
| Workspace + rig management | partial | `aragora/extensions/gastown/workspace.py`, `aragora/workspace/manager.py` | Two parallel managers; needs unification |
| Convoy lifecycle | partial | `aragora/extensions/gastown/convoy.py`, `aragora/workspace/convoy.py`, `aragora/nomic/convoys.py` | Three implementations |
| Beads (atomic work units) | partial | `aragora/workspace/bead.py`, `aragora/nomic/beads.py` | Two stores + schema mismatch |
| Hooks + worktree persistence | partial | `aragora/extensions/gastown/hooks.py`, `aragora/nomic/hook_queue.py` | Hooks + GUPP queue exist, need wiring |
| Mayor/Witness roles | partial | `aragora/nomic/agent_roles.py`, `aragora/nomic/mayor_coordinator.py`, `aragora/nomic/witness_behavior.py` | Core logic exists, needs runtime integration |
| CLI workflows | partial | `aragora/cli/gt.py` | CLI uses nomic convoys/beads |
| Dashboard endpoints | partial | `aragora/server/handlers/gastown_dashboard.py` | Mostly stubs |

### Moltbot (Consumer Assistant)

| Capability | Status | Code refs | Notes |
| --- | --- | --- | --- |
| Local Gateway server | partial | `aragora/gateway/server.py`, `aragora/gateway/router.py` | No Moltbot WS protocol adapter |
| Unified inbox | partial | `aragora/gateway/inbox.py`, `aragora/extensions/moltbot/inbox.py`, `aragora/storage/unified_inbox_store.py` | Multiple inbox layers |
| Device registry + pairing | partial | `aragora/gateway/device_registry.py`, `aragora/server/handlers/devices.py`, `aragora/onboarding/wizard.py` | No device runtime clients |
| Capability routing | partial | `aragora/gateway/capability_router.py`, `aragora/extensions/moltbot/capabilities.py` | Needs device node telemetry |
| Voice wake + voice sessions | partial | `aragora/extensions/moltbot/voice_wake.py`, `aragora/extensions/moltbot/voice.py`, `aragora/server/stream/voice_stream.py` | Wake engines mocked |
| Canvas / A2UI | partial | `aragora/extensions/moltbot/canvas.py`, `aragora/canvas/manager.py`, `aragora/server/stream/canvas_stream.py` | Protocol parity missing |
| Browser control | partial | `aragora/mcp/tools_module/browser.py`, `aragora/workflow/nodes/browser.py` | Needs device-level policy gates |
| Channel coverage | partial | `aragora/connectors/`, `aragora/integrations/` | Consumer channels incomplete |

## Priority Backlog (Best Order)

### P0: Unify duplicated primitives (foundation)
1. **Unify bead storage and status mapping**
   - Acceptance: single storage backend for beads (Nomic BeadStore), workspace APIs preserved, consistent status mapping
   - Current: `aragora.workspace` now uses Nomic BeadStore via adapter
2. **Unify convoy storage and IDs**
   - Acceptance: convoys share the same store backend, workspace status preserved via metadata, IDs stable across adapters
   - Current: `aragora.workspace` + `aragora.extensions.gastown` convoys use Nomic ConvoyManager adapters
3. **Canonical ownership of Gastown primitives**
   - Acceptance: clear “source of truth” module for beads/convoys/workspaces and compatibility shims for legacy APIs

### P1: Protocol compatibility & device runtime
4. **Gateway WS protocol adapter (Moltbot-compatible)**
   - Acceptance: sessions/presence/config endpoints mapped to gateway server; test fixture for compatibility
5. **Device node runtime reference client**
   - Acceptance: local device node can pair, heartbeat, and report capabilities; end-to-end test with DeviceRegistry
6. **Voice wake engine integration**
   - Acceptance: real wake engine binding (Porcupine/Vosk), device mic integration, integration test
7. **Canvas A2UI contract**
   - Acceptance: push/reset/snapshot messages handled; compatibility test suite

### P2: Expansion & hardening
8. **Consumer channel expansion (Signal/iMessage/Matrix/Zalo/Google Chat)**
   - Acceptance: connector + inbox routing + RBAC gate for each channel
9. **Policy + approval unification**
   - Acceptance: device + computer-use actions use the same approval and audit flow
10. **Agent Fabric MVP**
   - Acceptance: scheduler + isolation + budgets + telemetry for 50–100 concurrent agents

## Implementation Notes
- Avoid SDK regeneration until endpoint stability improves.
- Prefer adapters over invasive rewrites; keep enterprise core stable.
- Add acceptance tests alongside each parity item to lock behavior.
