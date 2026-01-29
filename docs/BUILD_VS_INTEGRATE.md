# Build vs Integrate Decisions (Gastown + Moltbot Extensions)

This document captures recommended decisions for parity work. Items marked
"Decision Needed" require explicit product approval.

## Gastown Extension

| Capability | Recommendation | Rationale | Decision Needed |
| --- | --- | --- | --- |
| Git worktree hook persistence | **Build** | Deeply tied to Aragora audit + policy model; needs enterprise-grade controls | No |
| Rig / workspace abstraction | **Build** | Maps to TenantContext and core orchestration | No |
| Convoy tracking | **Build** | Extends workflow engine + task decomposer | No |
| Mayor coordinator | **Build** | Must respect Aragora governance + approval flows | No |
| CLI UX compatibility | **Integrate / mimic** | Keep CLI semantics familiar; can map commands to Aragora endpoints | **Yes** |

## Moltbot Extension

| Capability | Recommendation | Rationale | Decision Needed |
| --- | --- | --- | --- |
| Local gateway daemon | **Build or Fork** | Must be trusted; Moltbot gateway could bootstrap | **Yes** |
| Live Canvas UI | **Integrate** | UI-heavy; reuse existing open-source canvas where possible | **Yes** |
| Voice wake word | **Integrate** | Hardware/ML heavy; use established frameworks | **Yes** |
| Onboarding wizard | **Build** | Product-specific, enterprise-friendly | No |
| Device nodes | **Build** | Requires strict policy integration | No |
| Computer-use sandbox | **Build** | Security-critical; enforce policy/audit by design | No |

## Cross-cutting

| Capability | Recommendation | Rationale | Decision Needed |
| --- | --- | --- | --- |
| Agent Fabric | **Build** | Core infrastructure; must be first-party | No |
| Policy engine extensions | **Build** | Enterprise-grade guardrails | No |
| Audit + replay | **Build** | Compliance requirement | No |

## Open Questions

1. Do we want Gastown CLI compatibility (same commands) or a native Aragora CLI?
2. Should Moltbot gateway be forked for speed or built for security/compliance?
3. Which canvas/voice stacks are acceptable from a security and licensing standpoint?
