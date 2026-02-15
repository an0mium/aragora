# Parity Epics (Gastown + OpenClaw Extensions)

These epics track the work required to reach Gastown and OpenClaw parity as
extensions, while keeping Aragora as the enterprise decision control plane.

## Epics

### EPIC-1: Agent Fabric (Core Scale)
Owner: Platform
Milestone: M1 (2026-02-16 to 2026-03-13)
Goal: Scale to large-N agents with isolation, quotas, and lifecycle controls.
Status: **READY FOR M1 LAUNCH** (95%)
- 6,035 LOC across 11 modules (fabric.py, lifecycle.py, scheduler.py, policy.py, budget.py, models.py, hooks.py, isolation.py, telemetry.py, audit.py, nudge.py)
- 163 tests passing including 18 load tests
- Load tested to 100 concurrent agents, p99 scheduling latency < 50ms, throughput > 200 tasks/sec
- Remaining: server integration end-to-end test, GUPP patrol cycle (non-blocking)

### EPIC-2: Policy + Safety Gate
Owner: Security + Runtime
Milestone: M1 (2026-02-16 to 2026-03-13)
Goal: Unified policy enforcement and approvals for tools, devices, and sandboxed actions.
Status: **READY FOR M1 LAUNCH** (100%)
- 1,268 LOC policy engine with safe AST-based condition evaluation
- 5 risk levels x 5 blast radii = 25 risk profiles
- 144 tests passing (73 policy + 71 integration)
- 1,059 LOC HTTP API handlers (policy.py + control_plane/policy.py)
- Features complete: risk budgets, approval workflows, rate limiting, audit trails, default policies

### EPIC-3: Gastown Workspace Extension
Owner: Workspace
Milestone: M2 (2026-03-16 to 2026-03-27)
Goal: Workspace manager, git worktree persistence, convoys, and Mayor-style coordination.

### EPIC-4: OpenClaw Local Gateway + Inbox
Owner: Gateway
Milestone: M3 (2026-03-30 to 2026-04-10)
Goal: Local gateway daemon, device registry, and unified inbox surface.

### EPIC-5: Live Canvas (A2UI) Surface
Owner: Live UI
Milestone: M3 (2026-03-30 to 2026-04-10)
Goal: Consumer-grade canvas collaboration surface integrated with agent state.

### EPIC-6: Computer-Use Sandbox
Owner: Runtime + Security
Milestone: M4 (2026-04-13 to 2026-04-24)
Goal: Policy-gated browser/shell/screen actions with audit trails.

### EPIC-7: Voice Wake + Device Nodes
Owner: Device Platform
Milestone: M4 (2026-04-13 to 2026-04-24)
Goal: On-device wake word + device capability model.

### EPIC-8: Hardening + Observability
Owner: SRE + QA
Milestone: M5 (2026-04-27 to 2026-05-08)
Goal: Performance targets, tracing, tests, and integration docs.
