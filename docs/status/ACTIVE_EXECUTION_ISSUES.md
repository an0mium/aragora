# Active Execution Issues

Last updated: 2026-03-23

This document links Aragora's current execution program to the live GitHub issue tracker.

- Docs explain thesis, roadmap, and capability posture.
- GitHub issues track live execution status, owners, priorities, and acceptance criteria.
- Use [NEXT_STEPS_CANONICAL](NEXT_STEPS_CANONICAL.md) for execution order and this file for the current issue map.

## Current Execution Order

1. Close the default product loop inside the PMF surfaces: provider routing, one truthful journey, and Knowledge Mound retrieval in the default flow
2. Sequential surface productization and PMF harvest
3. Developer swarm control plane and truthful unattended execution
4. Decision Integrity Kernel scale-out
5. Truthfulness and documentation hygiene
6. Idea-to-execution workbench
7. Enterprise readiness stays warm, not the main product lane

Immediate merge discipline for step 1:

- `main` already contains supporting PMF slices through `#1164`.
- The active merge gate is the overlapping open PR stack [#1166](https://github.com/synaptent/aragora/pull/1166), [#1167](https://github.com/synaptent/aragora/pull/1167), [#1168](https://github.com/synaptent/aragora/pull/1168), [#1169](https://github.com/synaptent/aragora/pull/1169), and [#1170](https://github.com/synaptent/aragora/pull/1170).
- `#1166` overlaps the narrower `#1167-#1170` slices; merge order must choose one coherent lane instead of landing overlapping PMF work twice.
- Recommended sequence: merge the narrow runtime/user-journey slices first ([#1167](https://github.com/synaptent/aragora/pull/1167) -> [#1168](https://github.com/synaptent/aragora/pull/1168) -> [#1169](https://github.com/synaptent/aragora/pull/1169) -> [#1170](https://github.com/synaptent/aragora/pull/1170)), then either harvest the remaining net-new pieces from [#1166](https://github.com/synaptent/aragora/pull/1166) or close it as superseded.

## Current PMF Proof Reality

The March 19-22 tranche/live-proof cycle generated real output and several merged proof-surface closures:

| Issue | Current reality | Output / note |
|------|-----------------|---------------|
| [#1011](https://github.com/synaptent/aragora/issues/1011) | First queue artifact recovered and published | [#1108](https://github.com/synaptent/aragora/pull/1108) merged |
| [#1046](https://github.com/synaptent/aragora/issues/1046) | Multiple user-journey support slices are now merged on `main` | [#1110](https://github.com/synaptent/aragora/pull/1110), [#1146](https://github.com/synaptent/aragora/pull/1146), and [#1147](https://github.com/synaptent/aragora/pull/1147) are merged; remaining gap is one repeatable end-to-end default proof and an explicit decision on `#1166` vs `#1169`/`#1170` |
| [#1048](https://github.com/synaptent/aragora/issues/1048) | Retrieval and writeback slices are partially merged on `main` | [#1111](https://github.com/synaptent/aragora/pull/1111) merged; KM writeback/settlement closures landed via [#1131](https://github.com/synaptent/aragora/pull/1131), [#1132](https://github.com/synaptent/aragora/pull/1132), and [#1134](https://github.com/synaptent/aragora/pull/1134); pre-debate precedent loading landed in [#1151](https://github.com/synaptent/aragora/pull/1151); default debate-factory wiring remains the active merge decision in `#1168` / `#1166` |
| [#1047](https://github.com/synaptent/aragora/issues/1047) | Still active, but the frontier changed | Partial-public status landed in [#1148](https://github.com/synaptent/aragora/pull/1148) and visible golden-path summary landed in [#1150](https://github.com/synaptent/aragora/pull/1150); the gap is now continuity across five truthful pages, not just page-shell diagnosis |
| [#818](https://github.com/synaptent/aragora/issues/818) | Truthful public proof slice is merged on `main` | [#1136](https://github.com/synaptent/aragora/pull/1136) merged; remaining gap is repeated external use |
| [#819](https://github.com/synaptent/aragora/issues/819) | First truthful integrations slice is merged on `main` | [#1119](https://github.com/synaptent/aragora/pull/1119) merged; broader integrations trustworthiness still active |

## Sequential Surface Productization

Epic: [#806](https://github.com/synaptent/aragora/issues/806)

Current tranche:

- the queue runs and follow-on merges now directly underpin the real PMF surfaces
- `#1046`, `#1048`, `#818`, and `#819` now all have at least one merged truthful slice on `main`
- `#1047` remains the active page-truthfulness/core-loop proof lane, not because nothing landed, but because the merged slices still do not add up to one complete default journey
- March 23 added live get-started/debate/settings/status/pipeline support on `main`, which means the immediate work is continuity and merge discipline rather than broadening page count

| Issue | State | Priority | Owner | Milestone | Scope |
|------|-------|----------|-------|-----------|-------|
| [#817](https://github.com/synaptent/aragora/issues/817) | Open | `priority:high` | `owner:team-integrations` | `2026-M2 Surface Productization` | Consolidate inbox and shared inbox onto the trust wedge |
| [#818](https://github.com/synaptent/aragora/issues/818) | Open | `priority:high` | `owner:team-integrations` | `2026-M2 Surface Productization` | Turn the public demo into a live proof surface |
| [#819](https://github.com/synaptent/aragora/issues/819) | Open | `priority:high` | `owner:team-integrations` | `2026-M2 Surface Productization` | Make the integrations UI trustworthy and non-demo by default |
| [#820](https://github.com/synaptent/aragora/issues/820) | Open | `priority:medium` | `owner:team-integrations` | `2026-M2 Surface Productization` | Productize Wave 2 surfaces: SME onboarding, spectate, and conditional public endpoints |

## Close The Product Loop (Immediate)

This is a cross-epic tranche front-loaded ahead of broader control-plane and workbench expansion.

- [#813](https://github.com/synaptent/aragora/issues/813) is the provider-routing blocker inside the first truthful loop.
- [#1046](https://github.com/synaptent/aragora/issues/1046) and [#1048](https://github.com/synaptent/aragora/issues/1048) already have merged base slices on `main` plus a new overlapping PR stack on top.
- The current goal is not more breadth. It is one default flow that truthfully routes agents, completes the user journey, and reads Knowledge Mound context back into debates.
- The active merge problem is explicit: [#1166](https://github.com/synaptent/aragora/pull/1166) overlaps [#1167](https://github.com/synaptent/aragora/pull/1167), [#1168](https://github.com/synaptent/aragora/pull/1168), [#1169](https://github.com/synaptent/aragora/pull/1169), and [#1170](https://github.com/synaptent/aragora/pull/1170). Treat it as a lane-selection problem, not a queue-more-work problem.
- Working recommendation: do not merge [#1166](https://github.com/synaptent/aragora/pull/1166) ahead of the narrower slices. Use [#1167](https://github.com/synaptent/aragora/pull/1167), [#1168](https://github.com/synaptent/aragora/pull/1168), [#1169](https://github.com/synaptent/aragora/pull/1169), and [#1170](https://github.com/synaptent/aragora/pull/1170) as the mainline sequence, then reassess what remains unique in [#1166](https://github.com/synaptent/aragora/pull/1166).

| Issue | Why it is first | Current reality |
|------|------------------|-----------------|
| [#813](https://github.com/synaptent/aragora/issues/813) | Provider routing blocks the first truthful product loop | ProviderRouter Phase 1 shipped on `main`; the active runtime agent-selection integration lane is [#1167](https://github.com/synaptent/aragora/pull/1167), with overlapping umbrella coverage in [#1166](https://github.com/synaptent/aragora/pull/1166) |
| [#1046](https://github.com/synaptent/aragora/issues/1046) | One working user journey is the first PMF proof | Base slices are merged on `main` via [#1110](https://github.com/synaptent/aragora/pull/1110), [#1146](https://github.com/synaptent/aragora/pull/1146), and [#1147](https://github.com/synaptent/aragora/pull/1147); the active continuity/onboarding lanes are [#1169](https://github.com/synaptent/aragora/pull/1169), [#1170](https://github.com/synaptent/aragora/pull/1170), and overlapping umbrella [#1166](https://github.com/synaptent/aragora/pull/1166) |
| [#1048](https://github.com/synaptent/aragora/issues/1048) | Knowledge retrieval must become a default read path | Base retrieval and writeback slices are merged on `main` via [#1111](https://github.com/synaptent/aragora/pull/1111), [#1131](https://github.com/synaptent/aragora/pull/1131), [#1132](https://github.com/synaptent/aragora/pull/1132), [#1134](https://github.com/synaptent/aragora/pull/1134), and [#1151](https://github.com/synaptent/aragora/pull/1151); the default debate-factory lane is [#1168](https://github.com/synaptent/aragora/pull/1168), with overlapping umbrella [#1166](https://github.com/synaptent/aragora/pull/1166) |

## Demonstrate The Value Prop (Q2 2026)

Epic: [#806](https://github.com/synaptent/aragora/issues/806)

Current tranche:

- [#1047](https://github.com/synaptent/aragora/issues/1047) is the active reduced proof lane; the latest `queue-v4b` attempt is truthfully `needs_human` / review-blocked.
- [#819](https://github.com/synaptent/aragora/issues/819) remains queued behind `#1047` in the same reduced proof run.
- The near-term proof target is five truthful surfaces, not a broad shell-heavy frontend.
- [#814](https://github.com/synaptent/aragora/issues/814) and [#815](https://github.com/synaptent/aragora/issues/815) stay in this tranche because action dispatch and higher-agent coordination matter only after the first loop is real.

| Issue | State | Priority | Owner | Milestone | Scope |
|------|-------|----------|-------|-----------|-------|
| [#814](https://github.com/synaptent/aragora/issues/814) | Open | `priority:high` | `owner:team-core` | `2026-M3 Strategic Moat Scale-Out` | Make OpenClaw action dispatch real |
| [#815](https://github.com/synaptent/aragora/issues/815) | Open | `priority:high` | `owner:team-core` | `2026-M3 Strategic Moat Scale-Out` | Scale adversarial orchestration to 10+ agents |
| [#817](https://github.com/synaptent/aragora/issues/817) | Open | `priority:high` | `owner:team-integrations` | `2026-M2 Surface Productization` | Consolidate inbox and shared inbox onto the trust wedge |
| [#818](https://github.com/synaptent/aragora/issues/818) | Open | `priority:high` | `owner:team-integrations` | `2026-M2 Surface Productization` | Turn the public demo into a live proof surface |
| [#819](https://github.com/synaptent/aragora/issues/819) | Open | `priority:high` | `owner:team-integrations` | `2026-M2 Surface Productization` | Make the integrations UI trustworthy and non-demo by default |
| [#820](https://github.com/synaptent/aragora/issues/820) | Open | `priority:medium` | `owner:team-integrations` | `2026-M2 Surface Productization` | Productize Wave 2 surfaces: SME onboarding, spectate, and conditional public endpoints |

## Enterprise Readiness (Kept Warm)

These remain open and real, but they do not outrank the PMF loop.

| Issue | State | Priority | Owner | Milestone | Scope |
|------|-------|----------|-------|-----------|-------|
| [#816](https://github.com/synaptent/aragora/issues/816) | Open | `priority:high` | `owner:team-core` | `2026-M3 Strategic Moat Scale-Out` | Deploy ERC-8004 identity and settlement integration |
| [#273](https://github.com/synaptent/aragora/issues/273) | Open | `priority:critical` | `owner:team-risk` | `2026-M2 Channel and FinOps` | Enterprise Assurance Closure epic |
| [#274](https://github.com/synaptent/aragora/issues/274) | Open | `priority:critical` | `owner:team-risk` | `2026-M2 Channel and FinOps` | Execute external penetration test and remediate findings |
| [#509](https://github.com/synaptent/aragora/issues/509) | Open | `priority:critical` | `owner:team-risk` | `none` | Pentest vendor selection and scope sign-off |

## Developer Swarm Control Plane And Truthful Execution

Epic context: [#836](https://github.com/synaptent/aragora/issues/836), [#1036](https://github.com/synaptent/aragora/issues/1036), [#989](https://github.com/synaptent/aragora/issues/989)

Recent reality on `main`:

- file-scope ownership and canonical PR tracking landed earlier via [#840](https://github.com/synaptent/aragora/issues/840) and [#841](https://github.com/synaptent/aragora/issues/841)
- March 19-20 hardening added queue compile and run, dead-worker recovery, deliverable sync, verification propagation, stale fleet-claim reaping, and queue-state persistence through merged PRs `#1109`, `#1112`, `#1115`, `#1116`, and `#1117`
- March 21-22 closure added preserved verification evidence, terminal tranche reconciliation, authoritative lane view, completed-lane publish, and remote-head PR review through `#1124`, `#1126`, `#1127`, `#1133`, and `#1138`
- March 23 added queue harvest support on `main` through [#1164](https://github.com/synaptent/aragora/pull/1164) and left two active supporting PRs open: [#1157](https://github.com/synaptent/aragora/pull/1157) for blocker-context persistence and [#1165](https://github.com/synaptent/aragora/pull/1165) for tranche queue Claude worker routing
- this work remains active, but it now serves the PMF slices above rather than defining the first execution lane by itself
- the current active proof lane is no longer "can a queue dispatch?" but "can every blocker and deliverable carry forward canonical receipts, review evidence, and operator-grade state?"

| Issue | State | Priority | Owner | Milestone | Scope |
|------|-------|----------|-------|-----------|-------|
| [#836](https://github.com/synaptent/aragora/issues/836) | Open | `priority:critical` | `owner:team-platform` | `2026-M3 Scale and Reliability` | Developer swarm control plane epic |
| [#837](https://github.com/synaptent/aragora/issues/837) | Open | `priority:high` | `owner:team-platform` | `2026-M3 Scale and Reliability` | Add tranche/task queue and claim protocol |
| [#840](https://github.com/synaptent/aragora/issues/840) | Closed | `priority:high` | `owner:team-platform` | `2026-M3 Scale and Reliability` | Enforce file-scope ownership on agent lanes |
| [#841](https://github.com/synaptent/aragora/issues/841) | Closed | `priority:high` | `owner:team-platform` | `2026-M3 Scale and Reliability` | Add canonical PR and supersession protocol |
| [#842](https://github.com/synaptent/aragora/issues/842) | Open | `priority:high` | `owner:team-platform` | `2026-M3 Scale and Reliability` | Emit receipts and provenance for every agent run |
| [#843](https://github.com/synaptent/aragora/issues/843) | Open | `priority:high` | `owner:team-platform` | `2026-M3 Scale and Reliability` | Build integrator view for active swarm lanes |
| [#871](https://github.com/synaptent/aragora/issues/871) | Open | `priority:high` | `owner:team-platform` | `none` | Autonomous Repo Maintenance MVP via Boss loop |
| [#990](https://github.com/synaptent/aragora/issues/990) | Open | `priority:high` | `owner:team-core` | `2026-M3 Strategic Moat Scale-Out` | Dogfood the pipeline to build more of Aragora itself |
| [#1036](https://github.com/synaptent/aragora/issues/1036) | Open | `priority:high` | `owner:team-core` | `2026-M3 Strategic Moat Scale-Out` | Continuous self-assessment and autonomous improvement cadence epic |
| [#1037](https://github.com/synaptent/aragora/issues/1037) | Closed | `priority:high` | `owner:team-core` | `2026-M3 Strategic Moat Scale-Out` | Compile a canonical repo assessment into pipeline-ready backlog artifacts |
| [#1038](https://github.com/synaptent/aragora/issues/1038) | Closed | `priority:high` | `owner:team-platform` | `2026-M3 Scale and Reliability` | Add pause-refresh checkpoints for long unattended self-improvement shifts |

## Decision Integrity Kernel Scale-Out

Epic: [#805](https://github.com/synaptent/aragora/issues/805)

Current tranche: the base kernel is on `main` through [#811](https://github.com/synaptent/aragora/issues/811) and [#812](https://github.com/synaptent/aragora/issues/812). The kernel-linked issues that block PMF are already front-loaded above: [#813](https://github.com/synaptent/aragora/issues/813) in the product-loop tranche, [#814](https://github.com/synaptent/aragora/issues/814) and [#815](https://github.com/synaptent/aragora/issues/815) in value-prop proof, and [#816](https://github.com/synaptent/aragora/issues/816) in enterprise readiness.

| Issue | State | Priority | Owner | Milestone | Scope |
|------|-------|----------|-------|-----------|-------|
| [#810](https://github.com/synaptent/aragora/issues/810) | Closed | `priority:high` | `owner:team-core` | `2026-M1 Truthfulness + Decision Integrity Core` | Add prompt -> specification -> DecisionPlan bridge |
| [#811](https://github.com/synaptent/aragora/issues/811) | Closed | `priority:high` | `owner:team-core` | `2026-M1 Truthfulness + Decision Integrity Core` | Collapse prompt/canvas/pipeline to one canonical execution runtime |
| [#812](https://github.com/synaptent/aragora/issues/812) | Closed | `priority:high` | `owner:team-core` | `2026-M1 Truthfulness + Decision Integrity Core` | Require cryptographic decision receipts before all action-taking |

## Truthfulness And Documentation Hygiene

Epic: [#804](https://github.com/synaptent/aragora/issues/804)

Tranche status: complete on `main` through [#809](https://github.com/synaptent/aragora/issues/809)

| Issue | State | Priority | Owner | Milestone | Scope |
|------|-------|----------|-------|-----------|-------|
| [#807](https://github.com/synaptent/aragora/issues/807) | Closed | `priority:critical` | `owner:team-platform` | `2026-M1 Truthfulness + Decision Integrity Core` | Make launch truthfulness blocking |
| [#808](https://github.com/synaptent/aragora/issues/808) | Closed | `priority:high` | `owner:team-platform` | `2026-M1 Truthfulness + Decision Integrity Core` | Make self-host readiness truthful and PR-gated |
| [#809](https://github.com/synaptent/aragora/issues/809) | Closed | `priority:high` | `owner:team-platform` | `2026-M1 Truthfulness + Decision Integrity Core` | Canonicalize the active backlog into GitHub issues |

## Idea-to-Execution Workbench

Epic: [#989](https://github.com/synaptent/aragora/issues/989)

Current planning reference: [ARAGORA_IDEA_TO_EXECUTION_STRATEGY](../plans/ARAGORA_IDEA_TO_EXECUTION_STRATEGY.md)

This remains strategically important, but it stays sequenced after PMF closure, enterprise-readiness warmup, control-plane truthfulness, and documentation hygiene.

| Issue | State | Priority | Owner | Milestone | Scope |
|------|-------|----------|-------|-----------|-------|
| [#989](https://github.com/synaptent/aragora/issues/989) | Open | `priority:high` | `owner:team-core` | `2026-M3 Strategic Moat Scale-Out` | Bootstrapped local-first idea-to-execution workbench |

## Operational Incidents

Operational incidents are not part of the planned execution epics, but they can preempt them when `main` is degraded.

| Issue | State | Priority | Scope |
|------|-------|----------|-------|
| [#829](https://github.com/synaptent/aragora/issues/829) | Open | `outage` | Service outage detected on 2026-03-07 |

## Operating Rule

When the execution program changes:

1. update the GitHub issues first
2. update [NEXT_STEPS_CANONICAL](NEXT_STEPS_CANONICAL.md)
3. update this issue map and any linked summary docs
4. if multiple open PRs cover the same PMF slice, choose a single merge lane and close or supersede the rest with proof
