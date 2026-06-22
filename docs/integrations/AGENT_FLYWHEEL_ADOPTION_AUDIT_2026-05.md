# Agent Flywheel Adoption Audit

Aragora should treat Agent Flywheel repositories as optional external tooling
and pattern sources, not vendored dependencies. Aragora's product boundary
remains governance: vetted decisions, receipts, dissent, calibration, and
settlement gates. Flywheel-style tools can improve local orchestration substrate.

## Capability Map

| Tool family | Useful pattern | Aragora adoption path |
| --- | --- | --- |
| Agent Mail | Agent inbox/outbox, file reservations, coordination messages | Compare with Aragora bridge lanes and shared outbox; consider an optional adapter after local validation |
| NTM | tmux swarm lifecycle, session inventory, local API | Mine lifecycle and session-control patterns for `scripts/agent_bridge.py` and worktree sessions |
| CASS / session search | Cross-agent transcript search and procedural memory | Compare against Aragora bridge snapshots, automation memory, and receipt search |
| Beads | Task graph, dependency ranking, critical-path focus | Evaluate as optional task-DAG input for queue triage and review-lane prioritization |
| Destructive command guard / SLB | Guardrails, confirmation policy, two-person rule patterns | Reimplement narrow safety gates for risky local commands; do not wrap all shell use blindly |
| ACFS | Manifested bootstrap, checksums, idempotent setup | Reuse the manifest/checksum idea for Aragora local lab setup; do not run broad installers by default |

## Licensing And Source Boundary

The first Aragora spike must not vendor Flywheel source, add submodules, or copy
substantial implementation code. The safe path is:

- use repos privately as external local tools where license terms permit;
- cite repository URLs in docs;
- reimplement generic ideas and interfaces in Aragora-owned code;
- require a separate legal/licensing gate before redistribution or vendoring.

## Local-First Validation

Local validation is required before AWS because local CLI auth state is part of
the experiment. The first Aragora-owned deliverable is intentionally small:

- `scripts/flywheel_tools_probe.py --json` for read-only local detection;
- `aragora.integrations.flywheel` for optional, guarded subprocess adapters;
- a runbook for scratch-directory lab work.

This does not install Flywheel, run agents, call model APIs, mutate H2 receipts,
or change GitHub runner behavior.

## Candidate Follow-Ups

- Add one optional adapter for a proven local tool, starting with read-only
  session inventory or search.
- Add a receipt around local tool probes if the output becomes part of Aragora
  operator decisions.
- Run an AWS portability pilot only after local value is demonstrated.
