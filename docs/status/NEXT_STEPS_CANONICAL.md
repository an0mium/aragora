# Next Steps (Canonical)

Last updated: 2026-03-21

This is the single source of truth for short-horizon execution priorities.
[CANONICAL_GOALS](../CANONICAL_GOALS.md) defines what Aragora is and why.
[ARAGORA_EVOLUTION_ROADMAP](../plans/ARAGORA_EVOLUTION_ROADMAP.md) defines the long-range architecture and moat.
[FEATURE_GAP_LIST](../FEATURE_GAP_LIST.md) is the capability and backlog truth.
[ACTIVE_EXECUTION_ISSUES](ACTIVE_EXECUTION_ISSUES.md) maps the current live GitHub issue set.

## Current Reality

- The March 18 product cohesion assessment found ~25% effective feature completeness for actual use, no complete user journey, provider routing still not wired to Arena, Knowledge Mound reads not enriching default debates, and roughly 140 of 149 frontend pages still acting as shells.
- Because of that, the immediate priority is product cohesion and PMF proof, not enterprise certification or broader architectural widening.
- `main` already contains enough substrate to attack these gaps: queue hardening through [#1117](https://github.com/synaptent/aragora/pull/1117), smart provider routing Phase 1, OpenClaw core-loop scaffolding, and one merged queue artifact plus additional candidate PMF PRs ([#1108](https://github.com/synaptent/aragora/pull/1108), [#1110](https://github.com/synaptent/aragora/pull/1110), [#1111](https://github.com/synaptent/aragora/pull/1111), [#1113](https://github.com/synaptent/aragora/pull/1113), [#1114](https://github.com/synaptent/aragora/pull/1114)).
- Queue and control-plane work still matter, but only insofar as they help close the PMF slices truthfully and without hidden operator repair.
- GitHub issues remain the live backlog. These docs summarize the active order and capability posture.
- The product strategy narrative is now maintained in [ARAGORA_IDEA_TO_EXECUTION_STRATEGY](../plans/ARAGORA_IDEA_TO_EXECUTION_STRATEGY.md).

## Execution Order

### 1) Close The Product Loop (Immediate)
- Tracking: [#813](https://github.com/synaptent/aragora/issues/813), [#1046](https://github.com/synaptent/aragora/issues/1046), [#1048](https://github.com/synaptent/aragora/issues/1048)
- Goal: make one truthful loop work end to end: real provider routing, one complete user journey, and Knowledge Mound retrieval enriching debate context.
- Current tranche: provider routing is still not wired to Arena, no complete user journey exists yet, and KM remains effectively write-only in the default product flow.

### 2) Demonstrate The Value Prop (Q2 2026)
- Tracking: [#806](https://github.com/synaptent/aragora/issues/806), [#814](https://github.com/synaptent/aragora/issues/814), [#815](https://github.com/synaptent/aragora/issues/815), [#817](https://github.com/synaptent/aragora/issues/817), [#818](https://github.com/synaptent/aragora/issues/818), [#819](https://github.com/synaptent/aragora/issues/819), [#820](https://github.com/synaptent/aragora/issues/820), [#1047](https://github.com/synaptent/aragora/issues/1047)
- Goal: prove Aragora on a small set of user-visible surfaces: debate to execution, five functional frontend paths, and higher-agent coordination once the core loop is real.
- Current tranche: OpenClaw dispatch is incomplete, the frontend is still mostly shell pages, and the priority is five working paths rather than broader UI sprawl.

### 3) Enterprise Readiness (Only After PMF)
- Tracking: [#273](https://github.com/synaptent/aragora/issues/273), [#274](https://github.com/synaptent/aragora/issues/274), [#509](https://github.com/synaptent/aragora/issues/509), [#816](https://github.com/synaptent/aragora/issues/816)
- Goal: keep pentest, SOC 2, chain deployment, and marketplace readiness real without letting them outrank PMF closure.
- Current tranche: assurance materials stay warm, but certification and listings should follow a usable product loop, not precede it.

### 4) Developer Swarm Control Plane And Truthful Execution
- Tracking: [#836](https://github.com/synaptent/aragora/issues/836), [#837](https://github.com/synaptent/aragora/issues/837), [#840](https://github.com/synaptent/aragora/issues/840), [#841](https://github.com/synaptent/aragora/issues/841), [#842](https://github.com/synaptent/aragora/issues/842), [#843](https://github.com/synaptent/aragora/issues/843), [#871](https://github.com/synaptent/aragora/issues/871), [#990](https://github.com/synaptent/aragora/issues/990), [#1036](https://github.com/synaptent/aragora/issues/1036), [#1037](https://github.com/synaptent/aragora/issues/1037), [#1038](https://github.com/synaptent/aragora/issues/1038)
- Goal: make unattended repo-improvement lanes truthful by keeping queue state, lane ownership, receipts, review outcomes, publish behavior, and operator visibility canonical.
- Current tranche: queue-backed execution is real on `main`; the remaining gap is finishing proof runs without ambiguous stuck states and preserving every blocker truthfully.

### 5) Decision Integrity Kernel Scale-Out
- Tracking: [#805](https://github.com/synaptent/aragora/issues/805), [#810](https://github.com/synaptent/aragora/issues/810), [#811](https://github.com/synaptent/aragora/issues/811), [#812](https://github.com/synaptent/aragora/issues/812)
- Goal: keep the receipt-gated decision kernel canonical while the product loop closes on top of it.
- Current tranche: the base landed earlier; remaining scale-out matters, but it is not the first user-visible blocker.

### 6) Truthfulness And Documentation Hygiene
- Tracking: [#804](https://github.com/synaptent/aragora/issues/804), [#807](https://github.com/synaptent/aragora/issues/807), [#808](https://github.com/synaptent/aragora/issues/808), [#809](https://github.com/synaptent/aragora/issues/809)
- Goal: keep `main` docs, readiness claims, and current-source status truthful as priorities shift.
- Current status: complete as a tranche, but still an active discipline. The PMF-first reframe is part of this work.

### 7) Idea-to-Execution Workbench
- Tracking: [#989](https://github.com/synaptent/aragora/issues/989)
- Goal: unify ideas, goals, actions, and orchestration into one local-first product shell with auditable stage transitions.
- Why later: the backend substrate is stronger than the UI, but the first obligation is to close the PMF loop before widening the shell.

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
