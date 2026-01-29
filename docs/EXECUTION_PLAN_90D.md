# 90-Day Execution Plan (Aragora Core + Gastown + Moltbot Extensions)

Objective: Preserve Aragora as the enterprise decision control plane while
building extension layers that can reach parity with Gastown (developer
orchestration) and Moltbot (consumer/device interface). This plan targets
proof-of-viability milestones; full parity will likely extend beyond 90 days.

## Guiding principles
- Keep Aragora core stable; extensions must not destabilize the enterprise plane.
- Build a shared Agent Fabric and policy engine before extension UX.
- Favor integration with Gastown/Moltbot where feasible to reach parity faster.

## Milestone map

### Weeks 1-2: Parity scoping + architecture lock
- Finalize parity matrix for Gastown and Moltbot
- Define three-layer architecture: Core, Dev-Orchestration extension, Consumer extension
- Decide build vs integrate strategy for Gastown and Moltbot
- Define Agent Fabric requirements (scheduling, isolation, budgets, policy)

### Weeks 3-4: Agent Fabric foundation
- Implement scheduler interface and agent lifecycle manager
- Add policy enforcement hooks (tool access, approvals, sandboxing)
- Add cost/budget enforcement at agent boundary
- Establish telemetry and traceability for agent runs

### Weeks 5-6: Gastown parity prototype
- Implement workspace manager service (projects, repos, rigs)
- Add git worktree persistence + hook runner
- Create convoy tracker (task state machine + artifacts)
- Expose CLI workflows for convoy/rig lifecycle

### Weeks 7-8: Moltbot parity prototype
- Implement local gateway service (routing + auth)
- Add device node registry and capability model
- Stub multi-channel inbox surface (unified entry point)
- Create onboarding wizard flow (CLI or minimal UI)

### Weeks 9-10: Safe computer-use MVP
- Build sandboxed execution surface (browser + shell)
- Add explicit approval gates and audit logs
- Integrate with Agent Fabric and policy engine

### Weeks 11-12: Integration hardening
- Document integration contracts for Gastown + Moltbot surfaces
- Add tests around extension APIs
- Validate performance with large-N agent simulation

## Success metrics (90 days)
- Agent Fabric supports 50+ concurrent agents with isolation and policy enforcement
- Gastown extension can persist work via git worktree hooks and run a convoy
- Moltbot extension can accept user input via local gateway + inbox stub
- Safe computer-use MVP is auditable and policy-gated

## Risks and mitigations
- Scope creep: keep extension work behind feature flags
- Security surface: enforce strict allowlists and approvals
- Operational complexity: ship minimal viable flows first

