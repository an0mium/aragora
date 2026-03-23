# Active Execution Issues

Last updated: 2026-03-23

This document links Aragora's current execution program to the live GitHub issue tracker.

- Docs explain thesis, roadmap, and capability posture.
- GitHub issues track live execution status, owners, priorities, and acceptance criteria.
- Use [NEXT_STEPS_CANONICAL](NEXT_STEPS_CANONICAL.md) for execution order and this file for the current issue map.

## Current Execution Order

1. **Product loop structurally closed** — prove it end-to-end with real users and harden edge cases
2. Sequential surface productization and PMF harvest
3. Developer swarm control plane and truthful unattended execution
4. Decision Integrity Kernel scale-out
5. Truthfulness and documentation hygiene
6. Idea-to-execution workbench
7. Enterprise readiness stays warm, not the main product lane

Step 1 status: the narrow PMF slices ([#1167](https://github.com/synaptent/aragora/pull/1167), [#1168](https://github.com/synaptent/aragora/pull/1168), [#1169](https://github.com/synaptent/aragora/pull/1169), [#1170](https://github.com/synaptent/aragora/pull/1170)) all merged on March 23, along with five additional product-loop PRs ([#1171](https://github.com/synaptent/aragora/pull/1171), [#1172](https://github.com/synaptent/aragora/pull/1172), [#1175](https://github.com/synaptent/aragora/pull/1175), [#1176](https://github.com/synaptent/aragora/pull/1176), [#1177](https://github.com/synaptent/aragora/pull/1177)). The merge-order discipline problem around [#1166](https://github.com/synaptent/aragora/pull/1166) is resolved; the narrow slices won.

## Current PMF Proof Reality

The March 23 merge wave closed the structural product-loop gaps. Nine PRs merged in one day:

| Issue | Current reality | Output / note |
|------|-----------------|---------------|
| [#1011](https://github.com/synaptent/aragora/issues/1011) | First queue artifact recovered and published | [#1108](https://github.com/synaptent/aragora/pull/1108) merged |
| [#813](https://github.com/synaptent/aragora/issues/813) | **ProviderRouter wired into DebateFactory** | [#1167](https://github.com/synaptent/aragora/pull/1167) merged; debates now use cost/quality/latency routing |
| [#1046](https://github.com/synaptent/aragora/issues/1046) | **Complete user journey exists** | [#1110](https://github.com/synaptent/aragora/pull/1110), [#1146](https://github.com/synaptent/aragora/pull/1146), [#1147](https://github.com/synaptent/aragora/pull/1147), [#1169](https://github.com/synaptent/aragora/pull/1169) (real API key mgmt), and [#1170](https://github.com/synaptent/aragora/pull/1170) (onboarding wizard) all merged; remaining gap is proving it with a real user |
| [#1048](https://github.com/synaptent/aragora/issues/1048) | **KM bidirectional flow closed** | Read path: [#1168](https://github.com/synaptent/aragora/pull/1168) wires KM retrieval into DebateFactory. Write path: [#1176](https://github.com/synaptent/aragora/pull/1176) ingests debate outcomes back into KM. Full read-write loop is now on `main`. |
| [#1047](https://github.com/synaptent/aragora/issues/1047) | **Dashboard and demo now live** | [#1175](https://github.com/synaptent/aragora/pull/1175) adds live debates section to dashboard; [#1177](https://github.com/synaptent/aragora/pull/1177) wires demo to real backend. The five truthful pages are now structurally connected. |
| [#818](https://github.com/synaptent/aragora/issues/818) | **Demo hits real backend** | [#1177](https://github.com/synaptent/aragora/pull/1177) merged; demo is no longer static |
| [#819](https://github.com/synaptent/aragora/issues/819) | Integrations surface improved | [#1119](https://github.com/synaptent/aragora/pull/1119) merged; broader integrations trustworthiness still active |
| [#871](https://github.com/synaptent/aragora/issues/871) | **Boss loop can run unattended** | [#1172](https://github.com/synaptent/aragora/pull/1172) adds label filter for scoped autonomous dispatch |
| — | **Queue v5 seeded** | [#1171](https://github.com/synaptent/aragora/pull/1171) seeds 5 fresh workbench/integrator issues (no longer replaying March work) |

## Sequential Surface Productization

Epic: [#806](https://github.com/synaptent/aragora/issues/806)

Current tranche:

- the product loop is structurally closed on `main`: onboarding -> credentials -> routed debate -> KM-enriched context -> receipt -> KM writeback -> live dashboard
- `#1046`, `#1048`, `#818`, and `#819` all have merged implementations on `main`
- `#1047` is materially closed: the demo, dashboard, and onboarding surfaces are all live and connected to real backends
- the immediate work is proving the closed loop with real users and hardening edge cases, not wiring more pages

| Issue | State | Priority | Owner | Milestone | Scope |
|------|-------|----------|-------|-----------|-------|
| [#817](https://github.com/synaptent/aragora/issues/817) | Open | `priority:high` | `owner:team-integrations` | `2026-M2 Surface Productization` | Consolidate inbox and shared inbox onto the trust wedge |
| [#818](https://github.com/synaptent/aragora/issues/818) | Open | `priority:high` | `owner:team-integrations` | `2026-M2 Surface Productization` | Turn the public demo into a live proof surface |
| [#819](https://github.com/synaptent/aragora/issues/819) | Open | `priority:high` | `owner:team-integrations` | `2026-M2 Surface Productization` | Make the integrations UI trustworthy and non-demo by default |
| [#820](https://github.com/synaptent/aragora/issues/820) | Open | `priority:medium` | `owner:team-integrations` | `2026-M2 Surface Productization` | Productize Wave 2 surfaces: SME onboarding, spectate, and conditional public endpoints |

## Close The Product Loop (Structurally Complete)

This cross-epic tranche is structurally complete as of March 23. All nine narrow PMF slices merged.

- [#813](https://github.com/synaptent/aragora/issues/813): **resolved** — ProviderRouter wired into DebateFactory via [#1167](https://github.com/synaptent/aragora/pull/1167).
- [#1046](https://github.com/synaptent/aragora/issues/1046): **resolved** — complete user journey from onboarding to debate to visible result via [#1169](https://github.com/synaptent/aragora/pull/1169) + [#1170](https://github.com/synaptent/aragora/pull/1170).
- [#1048](https://github.com/synaptent/aragora/issues/1048): **resolved** — KM bidirectional flow: read via [#1168](https://github.com/synaptent/aragora/pull/1168), write via [#1176](https://github.com/synaptent/aragora/pull/1176).
- The merge-order problem around [#1166](https://github.com/synaptent/aragora/pull/1166) is resolved: the narrow slices ([#1167](https://github.com/synaptent/aragora/pull/1167)-[#1170](https://github.com/synaptent/aragora/pull/1170)) won.

| Issue | Status | Current reality |
|------|--------|-----------------|
| [#813](https://github.com/synaptent/aragora/issues/813) | **Closed on `main`** | ProviderRouter wired into DebateFactory ([#1167](https://github.com/synaptent/aragora/pull/1167)); debates use cost/quality/latency routing |
| [#1046](https://github.com/synaptent/aragora/issues/1046) | **Closed on `main`** | Real API key management ([#1169](https://github.com/synaptent/aragora/pull/1169)), interactive onboarding ([#1170](https://github.com/synaptent/aragora/pull/1170)), live dashboard ([#1175](https://github.com/synaptent/aragora/pull/1175)), real demo ([#1177](https://github.com/synaptent/aragora/pull/1177)) |
| [#1048](https://github.com/synaptent/aragora/issues/1048) | **Closed on `main`** | KM retrieval in DebateFactory ([#1168](https://github.com/synaptent/aragora/pull/1168)) + debate outcome ingestion back to KM ([#1176](https://github.com/synaptent/aragora/pull/1176)); full read-write loop |

## Demonstrate The Value Prop (Q2 2026)

Epic: [#806](https://github.com/synaptent/aragora/issues/806)

Current tranche:

- [#1047](https://github.com/synaptent/aragora/issues/1047) is materially closed: demo hits real backend ([#1177](https://github.com/synaptent/aragora/pull/1177)), dashboard shows live state ([#1175](https://github.com/synaptent/aragora/pull/1175)), onboarding provides a complete first-run experience ([#1170](https://github.com/synaptent/aragora/pull/1170)).
- [#819](https://github.com/synaptent/aragora/issues/819) has a merged truthful integrations slice; broader trustworthiness still active.
- The product loop is closed; the near-term proof target is repeatable external usage.
- [#814](https://github.com/synaptent/aragora/issues/814) and [#815](https://github.com/synaptent/aragora/issues/815) stay in this tranche for action dispatch and higher-agent coordination.

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
- March 23 added queue harvest support through [#1164](https://github.com/synaptent/aragora/pull/1164), boss-loop label filter for unattended dispatch through [#1172](https://github.com/synaptent/aragora/pull/1172), and a fresh queue v5 with 5 new workbench/integrator issues through [#1171](https://github.com/synaptent/aragora/pull/1171)
- this work now serves a structurally closed product loop rather than gating one
- the current active proof lane is "can the boss loop run unattended with label-scoped dispatch and carry forward canonical receipts?"

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
