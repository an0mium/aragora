# Next Steps (Canonical)

Last updated: 2026-03-22

This is the single source of truth for short-horizon execution priorities.
[CANONICAL_GOALS](../CANONICAL_GOALS.md) defines what Aragora is and why.
[ARAGORA_EVOLUTION_ROADMAP](../plans/ARAGORA_EVOLUTION_ROADMAP.md) defines the long-range architecture and moat.
[FEATURE_GAP_LIST](../FEATURE_GAP_LIST.md) is the capability and backlog truth.
[ACTIVE_EXECUTION_ISSUES](ACTIVE_EXECUTION_ISSUES.md) maps the current live GitHub issue set.

## Current Reality

- The March 18 product cohesion assessment is still directionally right about shell-heavy pages and the lack of one complete user journey, but `main` moved materially again on March 21-22.
- `main` now contains more than queue hardening through [#1117](https://github.com/synaptent/aragora/pull/1117): it also includes the first merged API-key/user-journey slice ([#1110](https://github.com/synaptent/aragora/pull/1110)), default KM retrieval and later writeback/settlement wiring ([#1111](https://github.com/synaptent/aragora/pull/1111), [#1131](https://github.com/synaptent/aragora/pull/1131), [#1132](https://github.com/synaptent/aragora/pull/1132), [#1134](https://github.com/synaptent/aragora/pull/1134)), truthful integrations/public/operator surfaces ([#1118](https://github.com/synaptent/aragora/pull/1118), [#1119](https://github.com/synaptent/aragora/pull/1119), [#1127](https://github.com/synaptent/aragora/pull/1127), [#1136](https://github.com/synaptent/aragora/pull/1136), [#1137](https://github.com/synaptent/aragora/pull/1137)), and real OpenClaw dispatch ([#1135](https://github.com/synaptent/aragora/pull/1135)).
- Because of that, the immediate priority is not broad platform breadth. It is stitching the merged slices into three repeatable wedges: trust wedge, truthful default/public debate surfaces, and bounded repo execution.
- Queue and control-plane work still matter, but only insofar as they help close those PMF slices truthfully and without hidden operator repair.
- GitHub issues remain the live backlog. These docs summarize the active order and capability posture.
- The product strategy narrative is now maintained in [ARAGORA_IDEA_TO_EXECUTION_STRATEGY](../plans/ARAGORA_IDEA_TO_EXECUTION_STRATEGY.md).

## Execution Order

### 1) Close The Product Loop (Immediate)
- Tracking: [#813](https://github.com/synaptent/aragora/issues/813), [#1046](https://github.com/synaptent/aragora/issues/1046), [#1047](https://github.com/synaptent/aragora/issues/1047), [#1048](https://github.com/synaptent/aragora/issues/1048), [#819](https://github.com/synaptent/aragora/issues/819)
- Goal: make one truthful default loop work end to end: credentials/provider routing, one complete user journey, KM-enriched debate context, receipt, and visible result.
- Current tranche: the first `#1046` and `#1048` slices are merged on `main`, but provider routing, surface continuity, and the shell-heavy frontend still prevent a genuinely complete default journey.

### 2) Demonstrate The Value Prop (Q2 2026)
- Tracking: [#806](https://github.com/synaptent/aragora/issues/806), [#814](https://github.com/synaptent/aragora/issues/814), [#817](https://github.com/synaptent/aragora/issues/817), [#818](https://github.com/synaptent/aragora/issues/818), [#819](https://github.com/synaptent/aragora/issues/819), [#820](https://github.com/synaptent/aragora/issues/820)
- Goal: prove Aragora on three bounded user-visible surfaces: trust wedge, truthful public/default debate surface, and swarm/OpenClaw execution.
- Current tranche: trust wedge core is real, public proof is materially truthful, and OpenClaw dispatch is real on `main`; the remaining gap is repeatable partner usage and five actually functional frontend paths.

### 3) Developer Swarm Control Plane And Truthful Execution
- Tracking: [#836](https://github.com/synaptent/aragora/issues/836), [#837](https://github.com/synaptent/aragora/issues/837), [#840](https://github.com/synaptent/aragora/issues/840), [#841](https://github.com/synaptent/aragora/issues/841), [#842](https://github.com/synaptent/aragora/issues/842), [#843](https://github.com/synaptent/aragora/issues/843), [#871](https://github.com/synaptent/aragora/issues/871), [#990](https://github.com/synaptent/aragora/issues/990), [#1036](https://github.com/synaptent/aragora/issues/1036), [#1037](https://github.com/synaptent/aragora/issues/1037), [#1038](https://github.com/synaptent/aragora/issues/1038)
- Goal: make unattended repo-improvement lanes truthful by keeping queue state, lane ownership, receipts, review outcomes, publish behavior, and operator visibility canonical.
- Current tranche: queue-backed execution is real on `main`; authoritative lane view, preserved review evidence, completed-lane publish, and remote-head review are now merged. The remaining gap is universal per-lane receipts/provenance and a canonical claim/integrator contract across all lanes.

### 4) Decision Integrity Kernel Scale-Out
- Tracking: [#805](https://github.com/synaptent/aragora/issues/805), [#810](https://github.com/synaptent/aragora/issues/810), [#811](https://github.com/synaptent/aragora/issues/811), [#812](https://github.com/synaptent/aragora/issues/812)
- Goal: keep the receipt-gated decision kernel canonical while the product loop closes on top of it.
- Current tranche: the base landed earlier; remaining scale-out matters, but it is not the first user-visible blocker.

### 5) Idea-to-Execution Workbench
- Tracking: [#989](https://github.com/synaptent/aragora/issues/989)
- Goal: unify ideas, goals, actions, and orchestration into one local-first product shell with auditable stage transitions.
- Why later: live state and one transition-review slice are now worth extending, but the first obligation is still to make the current wedges repeatable before widening the shell.

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
