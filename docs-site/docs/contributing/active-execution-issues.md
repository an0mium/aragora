---
title: Active Execution Issues
description: Active Execution Issues
---

# Active Execution Issues

Last updated: 2026-03-23

This document links Aragora's current execution program to the live GitHub issue tracker.

- Docs explain thesis, roadmap, and capability posture.
- GitHub issues track durable backlog items and acceptance criteria.
- [NEXT_STEPS_CANONICAL](./next-steps-canonical) defines execution order.
- [PMF_DOGFOOD_EXECUTION_PLAN](./pmf-dogfood-execution-plan) defines the current founder-loop proof and blocker-harvest runbook.

## Program Status: Structural PMF Slices Landed, Live Dogfood Gate Still Open

The repo moved forward materially on March 21-23:

- Provider routing is wired into the runtime path on `main`
- KM retrieval and writeback are wired into the debate path on `main`
- onboarding, API-key management, demo, dashboard, and integrations surfaces are substantially more truthful on `main`
- quickstart now fails fast on bad TLS and emits structured receipts with inline provider-key support on `main`
- the repo includes a current mocked founder-loop E2E proof:

  ```bash
  python3 -m pytest tests/e2e/test_user_journey.py tests/cli/test_quickstart.py -q
  ```

  Result on March 23, 2026: `57 passed`

That is enough to say the structural PMF slices are present.
It is **not** enough to say the live PMF loop is proven for external users.

## Live GitHub State

GitHub is no longer carrying an open PMF blocker set. As of March 23, 2026, the only open issues are enterprise-assurance items:

| Issue | State | Priority | Scope |
|-------|-------|----------|-------|
| [#273](https://github.com/synaptent/aragora/issues/273) | Open | `priority:critical` | Enterprise assurance closure epic |
| [#274](https://github.com/synaptent/aragora/issues/274) | Open | `priority:critical` | External penetration test and remediation |
| [#509](https://github.com/synaptent/aragora/issues/509) | Open | `priority:critical` | Pentest vendor selection and scope sign-off |

## Doc-Driven PMF Program (Current Active Work)

Because the PMF issue tree was closed ahead of live proof, the near-term PMF execution program is temporarily tracked in docs rather than as open GitHub issues.

| Track | Current status | Source of truth | Notes |
|------|----------------|-----------------|-------|
| Founder-loop live proof | Current gate | [PMF_DOGFOOD_EXECUTION_PLAN](./pmf-dogfood-execution-plan) | Must prove live debate -> receipt -> result -> KM path without manual rescue |
| Live blocker harvest | Active when proof fails | command transcript + new issue | Reopen or create issues only after reproducing a concrete failure |
| PMF blocker pipeline compile | Ready | `aragora pipeline dogfood` | Use the PMF plan doc as the bounded source file |
| Bounded swarm / Ralph repair lanes | Ready | generated manifest + review receipts | Only for proven PMF blockers |
| Inbox trust wedge dogfood | Deferred | docs/plans + live proof | Run only after founder loop is stable |

## Historical PMF Lineage (Closed Too Early To Be The Current Gate)

These issues and epics matter as historical lineage, but they do not substitute for current live proof:

| Issue | Status | Why it is not enough by itself |
|-------|--------|--------------------------------|
| [#806](https://github.com/synaptent/aragora/issues/806) | Closed | Surface productization landed structurally, but live external continuity still needs proof |
| [#820](https://github.com/synaptent/aragora/issues/820) | Closed | Wave 2 slices landed, but the current gate is the founder loop, not surface breadth |
| [#989](https://github.com/synaptent/aragora/issues/989) | Closed | The workbench exists, but it now needs to be used to finish Aragora itself |
| [#990](https://github.com/synaptent/aragora/issues/990) | Closed | Dogfood infrastructure exists, but the new obligation is to feed it PMF blockers only |
| [#1011](https://github.com/synaptent/aragora/issues/1011) | Closed | Design-partner motion should follow repeatable live proof, not precede it |
| [#1036](https://github.com/synaptent/aragora/issues/1036) | Closed | Self-assessment cadence exists, but it should be pointed at founder-loop failures |

## Historical Execution Link Map

These links are retained because the repo's status reconciliation contract still expects the March execution issue lineage to remain reachable from this document, even though they are not the current active gate:

[#804](https://github.com/synaptent/aragora/issues/804), [#805](https://github.com/synaptent/aragora/issues/805), [#807](https://github.com/synaptent/aragora/issues/807), [#808](https://github.com/synaptent/aragora/issues/808), [#809](https://github.com/synaptent/aragora/issues/809), [#810](https://github.com/synaptent/aragora/issues/810), [#811](https://github.com/synaptent/aragora/issues/811), [#812](https://github.com/synaptent/aragora/issues/812), [#813](https://github.com/synaptent/aragora/issues/813), [#814](https://github.com/synaptent/aragora/issues/814), [#815](https://github.com/synaptent/aragora/issues/815), [#816](https://github.com/synaptent/aragora/issues/816), [#817](https://github.com/synaptent/aragora/issues/817), [#818](https://github.com/synaptent/aragora/issues/818), [#819](https://github.com/synaptent/aragora/issues/819)

## Issue Recreation Rule

When the founder-loop dogfood run exposes a blocker:

1. capture the exact command transcript and truthful stop condition
2. create or reopen a narrow GitHub issue for that specific blocker
3. run pipeline/swarm only against that bounded blocker
4. close the issue only after the founder loop is re-run successfully

## Operating Rule

Until the PMF blocker set exists again as live issues, do not mistake "few open issues" for "nothing important left."
