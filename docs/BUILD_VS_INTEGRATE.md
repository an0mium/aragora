# Build vs Integrate Decisions (Gastown + Moltbot Extensions)

**Decision Date:** 2026-01-28
**Decision:** BUILD all features independently in Aragora, adopting patterns with attribution.

This document captures finalized decisions for parity work. The guiding principle is to
**reimplement features in Aragora** rather than depend on external repos, while adopting
good patterns from Gastown and Moltbot with proper attribution.

---

## Attribution Requirements

All reimplemented patterns must include attribution in docstrings:

```python
"""
Pattern: [Pattern Name]
Inspired by: [Gastown/Moltbot] (https://github.com/[repo])
Aragora adaptation: [Description of how adapted for Aragora]
"""
```

---

## Gastown Extension

| Capability | Decision | Status | Attribution |
| --- | --- | --- | --- |
| Git worktree hook persistence | **BUILD** | DONE | `fabric/hooks.py` - Gastown GUPP |
| Rig / workspace abstraction | **BUILD** | DONE | `workspace/rig.py` - Gastown Rigs |
| Convoy tracking | **BUILD** | DONE | `workspace/convoy.py` - Gastown Convoys |
| Mayor coordinator | **BUILD** | DONE | `nomic/mayor_coordinator.py` - Gastown Mayor |
| CLI UX compatibility | **BUILD** | DONE | Native Aragora CLI with familiar semantics |
| Refinery (merge queue) | **BUILD** | PENDING | New `workspace/refinery.py` - Gastown Refinery |
| Nudge (inter-agent mail) | **BUILD** | PENDING | New `fabric/nudge.py` - Gastown Nudge |

## Moltbot Extension

| Capability | Decision | Status | Attribution |
| --- | --- | --- | --- |
| Local gateway daemon | **BUILD** | IN PROGRESS | `gateway/server.py` - Moltbot local-first |
| Multi-channel inbox | **BUILD** | DONE | `gateway/inbox.py` - Moltbot unified inbox |
| Multi-agent routing | **BUILD** | DONE | `gateway/router.py` |
| Device registry | **BUILD** | DONE | `gateway/device_registry.py` |
| Live Canvas UI | **BUILD** | EXISTS | `canvas/manager.py` - use existing infra |
| Voice wake word | **BUILD** | PARTIAL | Integrate proven framework (e.g., Picovoice) |
| Onboarding wizard | **BUILD** | PENDING | New `onboarding/wizard.py` - Moltbot pattern |
| Device nodes | **BUILD** | DONE | `gateway/device_registry.py` |
| Computer-use sandbox | **BUILD** | PENDING | New `sandbox/computer_use.py` |
| Security pairing | **BUILD** | PARTIAL | Extend `gateway/device_registry.py` |

## Cross-cutting

| Capability | Decision | Status | Notes |
| --- | --- | --- | --- |
| Agent Fabric | **BUILD** | DONE | `aragora/fabric/` - 100% complete |
| Policy engine | **BUILD** | DONE | `fabric/policy.py` |
| Budget enforcement | **BUILD** | DONE | `fabric/budget.py` |
| Audit + replay | **BUILD** | DONE | `gauntlet/receipt.py` |
| RBAC gates for computer use | **BUILD** | PENDING | Integrate with `rbac/decorators.py` |

---

## Resolved Questions

1. **CLI compatibility:** Native Aragora CLI. No external dependency.
2. **Gateway approach:** Build minimal secure gateway from scratch.
3. **Canvas/voice stacks:** Use permissive-licensed frameworks (Picovoice for voice wake, existing canvas infra).

---

## Implementation Priority

| Priority | Module | Effort | Pattern Source |
| --- | --- | --- | --- |
| P0 | `gateway/server.py` - HTTP server | 300-400 lines | Moltbot |
| P1 | `gateway/persistence.py` | 400-500 lines | Moltbot |
| P2 | `sandbox/computer_use.py` | 600-800 lines | Anthropic patterns |
| P3 | `workspace/refinery.py` | 300-400 lines | Gastown |
| P4 | `onboarding/wizard.py` | 300-400 lines | Moltbot |
| P5 | `fabric/nudge.py` | 200-300 lines | Gastown |
