# Next Steps (Canonical)

Last updated: 2026-03-23

This is the single source of truth for short-horizon execution priorities.
[CANONICAL_GOALS](../CANONICAL_GOALS.md) defines what Aragora is and why.
[ARAGORA_EVOLUTION_ROADMAP](../plans/ARAGORA_EVOLUTION_ROADMAP.md) defines the long-range architecture and moat.
[FEATURE_GAP_LIST](../FEATURE_GAP_LIST.md) is the capability and backlog truth.
[ACTIVE_EXECUTION_ISSUES](ACTIVE_EXECUTION_ISSUES.md) maps the current live GitHub issue set.

## Current Reality

- The March 18 product cohesion assessment flagged shell-heavy pages and the lack of one complete user journey. As of March 23, the majority of those product-loop gaps are now closed on `main`.
- Nine PRs merged on March 23 that collectively close the default product loop:
  - [#1167](https://github.com/synaptent/aragora/pull/1167): ProviderRouter wired into DebateFactory (debates now use cost/quality/latency routing instead of hardcoded model selection)
  - [#1168](https://github.com/synaptent/aragora/pull/1168): KnowledgeMound retrieval wired into DebateFactory (debates enriched with org knowledge by default)
  - [#1169](https://github.com/synaptent/aragora/pull/1169): Versioned API key management endpoints (real backend auth, no more client-side fakes)
  - [#1170](https://github.com/synaptent/aragora/pull/1170): Interactive 3-step onboarding wizard (first complete user journey from landing to debate)
  - [#1171](https://github.com/synaptent/aragora/pull/1171): Fresh tranche queue seeded with 5 new workbench/integrator issues (queue v5)
  - [#1172](https://github.com/synaptent/aragora/pull/1172): Boss-loop label filter for unattended dispatch
  - [#1175](https://github.com/synaptent/aragora/pull/1175): Dashboard live debates section with active tracking (dashboard shows live state, not just historical)
  - [#1176](https://github.com/synaptent/aragora/pull/1176): Debate outcome ingested back into KnowledgeMound (closes the read-write KM feedback loop)
  - [#1177](https://github.com/synaptent/aragora/pull/1177): Demo surface wired to real backend debate endpoint (no more static demo)
- The product loop is now structurally closed: credentials -> provider routing -> KM-enriched debate -> receipt -> KM writeback -> visible result on dashboard. The merge-order discipline problem around [#1166](https://github.com/synaptent/aragora/pull/1166) vs [#1167](https://github.com/synaptent/aragora/pull/1167)-[#1170](https://github.com/synaptent/aragora/pull/1170) is resolved: the narrow slices won and are all merged.
- The immediate priority shifts from "close the product loop" to "prove the loop end-to-end with a real user" and "harden the surfaces that are now live."
- Queue and control-plane work still matter, but they now serve a running product loop rather than gating one.
- GitHub issues remain the live backlog. These docs summarize the active order and capability posture.
- The product strategy narrative is now maintained in [ARAGORA_IDEA_TO_EXECUTION_STRATEGY](../plans/ARAGORA_IDEA_TO_EXECUTION_STRATEGY.md).

## Execution Order

### 1) Close The Product Loop (Structurally Complete)
- Tracking: [#813](https://github.com/synaptent/aragora/issues/813), [#1046](https://github.com/synaptent/aragora/issues/1046), [#1047](https://github.com/synaptent/aragora/issues/1047), [#1048](https://github.com/synaptent/aragora/issues/1048), [#819](https://github.com/synaptent/aragora/issues/819)
- Goal: make one truthful default loop work end to end: credentials/provider routing, one complete user journey, KM-enriched debate context, receipt, and visible result.
- Status: **structurally closed on `main`** as of March 23. ProviderRouter wired ([#1167](https://github.com/synaptent/aragora/pull/1167)), KM bidirectional flow wired ([#1168](https://github.com/synaptent/aragora/pull/1168) read + [#1176](https://github.com/synaptent/aragora/pull/1176) write), real API key management ([#1169](https://github.com/synaptent/aragora/pull/1169)), onboarding journey ([#1170](https://github.com/synaptent/aragora/pull/1170)), live dashboard ([#1175](https://github.com/synaptent/aragora/pull/1175)), and real demo backend ([#1177](https://github.com/synaptent/aragora/pull/1177)). The remaining work is proving the loop end-to-end with a real user and hardening edge cases.

### 2) Demonstrate The Value Prop (Q2 2026)
- Tracking: [#806](https://github.com/synaptent/aragora/issues/806), [#814](https://github.com/synaptent/aragora/issues/814), [#817](https://github.com/synaptent/aragora/issues/817), [#818](https://github.com/synaptent/aragora/issues/818), [#819](https://github.com/synaptent/aragora/issues/819), [#820](https://github.com/synaptent/aragora/issues/820)
- Goal: prove Aragora on three bounded user-visible surfaces: trust wedge, truthful public/default debate surface, and swarm/OpenClaw execution.
- Current tranche: the demo now hits a real backend debate endpoint ([#1177](https://github.com/synaptent/aragora/pull/1177)), the dashboard shows live debate state ([#1175](https://github.com/synaptent/aragora/pull/1175)), and the onboarding wizard provides a complete first-run experience ([#1170](https://github.com/synaptent/aragora/pull/1170)). The remaining gap is repeatable partner usage and proving the full loop with external users, not wiring more pages.

### 3) Developer Swarm Control Plane And Truthful Execution
- Tracking: [#836](https://github.com/synaptent/aragora/issues/836), [#837](https://github.com/synaptent/aragora/issues/837), [#840](https://github.com/synaptent/aragora/issues/840), [#841](https://github.com/synaptent/aragora/issues/841), [#842](https://github.com/synaptent/aragora/issues/842), [#843](https://github.com/synaptent/aragora/issues/843), [#871](https://github.com/synaptent/aragora/issues/871), [#990](https://github.com/synaptent/aragora/issues/990), [#1036](https://github.com/synaptent/aragora/issues/1036), [#1037](https://github.com/synaptent/aragora/issues/1037), [#1038](https://github.com/synaptent/aragora/issues/1038)
- Goal: make unattended repo-improvement lanes truthful by keeping queue state, lane ownership, receipts, review outcomes, publish behavior, and operator visibility canonical.
- Current tranche: queue-backed execution is real on `main`; authoritative lane view, preserved review evidence, completed-lane publish, and remote-head review are now merged. March 23 added boss-loop label filtering for unattended dispatch ([#1172](https://github.com/synaptent/aragora/pull/1172)) and a fresh queue v5 with 5 new workbench/integrator issues ([#1171](https://github.com/synaptent/aragora/pull/1171)). The remaining gap is universal per-lane receipts/provenance and a canonical claim/integrator contract across all lanes.

### 4) Decision Integrity Kernel Scale-Out
- Tracking: [#805](https://github.com/synaptent/aragora/issues/805), [#810](https://github.com/synaptent/aragora/issues/810), [#811](https://github.com/synaptent/aragora/issues/811), [#812](https://github.com/synaptent/aragora/issues/812)
- Goal: keep the receipt-gated decision kernel canonical while the product loop closes on top of it.
- Current tranche: the base landed earlier; remaining scale-out matters, but it is not the first user-visible blocker.

### 5) Idea-to-Execution Workbench
- Tracking: [#989](https://github.com/synaptent/aragora/issues/989)
- Goal: unify ideas, goals, actions, and orchestration into one local-first product shell with auditable stage transitions.
- Why later: the product loop is now structurally closed, so the workbench can extend a running system instead of bridging disconnected shells. The first obligation is still to prove the closed loop with real users before widening the shell.

### 6) Truthfulness And Documentation Hygiene
- Tracking: [#804](https://github.com/synaptent/aragora/issues/804), [#807](https://github.com/synaptent/aragora/issues/807), [#808](https://github.com/synaptent/aragora/issues/808), [#809](https://github.com/synaptent/aragora/issues/809)
- Goal: keep `main` docs, readiness claims, and current-source status truthful as priorities shift.
- Current status: complete as a tranche, but still an active discipline. The PMF-first reframe is part of this work.

### 7) Enterprise Readiness (Only After PMF)
- Tracking: [#273](https://github.com/synaptent/aragora/issues/273), [#274](https://github.com/synaptent/aragora/issues/274), [#509](https://github.com/synaptent/aragora/issues/509), [#816](https://github.com/synaptent/aragora/issues/816)
- Goal: keep pentest, SOC 2, chain deployment, and marketplace readiness real without letting them outrank PMF closure.
- Current tranche: assurance materials stay warm, but certification and listings should follow a usable repeating product loop, not precede it.

### Operational Incidents (Interrupt-Driven)
- Tracking: [#829](https://github.com/synaptent/aragora/issues/829) and any future incident tickets
- Rule: incidents can preempt the planned order, but they do not replace the canonical program. Once mitigated, execution returns to the order above.

## Operating Rules

- GitHub issues are the live execution backlog; docs summarize context, order, and capability posture.
- [ACTIVE_EXECUTION_ISSUES](ACTIVE_EXECUTION_ISSUES.md) must stay aligned with the current issue set.
- [FEATURE_GAP_LIST](../FEATURE_GAP_LIST.md) remains the capability and backlog truth for planned and partial features.
- No document should claim "only one blocker remains" unless `main` CI, deployment, and proof-run evidence support it.
- Productize exposed surfaces sequentially; do not broaden active implementation lanes faster than the kernel and control plane can support.
- If priorities change, update the GitHub issues first, then update this file and the linked summaries.
