# Next Steps (Canonical)

Last updated: 2026-03-20

This is the single source of truth for short-horizon execution priorities.
[CANONICAL_GOALS](../CANONICAL_GOALS.md) defines what Aragora is and why.
[ARAGORA_EVOLUTION_ROADMAP](../plans/ARAGORA_EVOLUTION_ROADMAP.md) defines the long-range architecture and moat.
[FEATURE_GAP_LIST](../FEATURE_GAP_LIST.md) is the capability and backlog truth.
[ACTIVE_EXECUTION_ISSUES](ACTIVE_EXECUTION_ISSUES.md) maps the current live GitHub issue set.

## Current Reality

- The long-range vision is still the unified idea-to-execution system, but the immediate execution frontier is now the tranche queue and unattended developer swarm proof lane.
- `main` includes the March 19-20 overnight hardening through [#1117](https://github.com/synaptent/aragora/pull/1117): queue compile/run, dead-worker recovery, deliverable sync, verification propagation, stale fleet-claim reaping, and queue-state persistence.
- Aragora has already produced one merged autonomous queue artifact ([#1108](https://github.com/synaptent/aragora/pull/1108)) and four additional candidate PMF PRs ([#1110](https://github.com/synaptent/aragora/pull/1110), [#1111](https://github.com/synaptent/aragora/pull/1111), [#1113](https://github.com/synaptent/aragora/pull/1113), [#1114](https://github.com/synaptent/aragora/pull/1114)).
- The active reduced proof lane is `queue-v4b` for [#1047](https://github.com/synaptent/aragora/issues/1047) and [#819](https://github.com/synaptent/aragora/issues/819). `#1047` is already truthfully `needs_human`; `#819` remains pending.
- The main remaining execution questions are no longer queue preflight or stale-claim dispatch. They are truthful finalization, publish and integrate edge cases, and harvesting the best PMF outputs already generated.
- GitHub issues remain the live backlog. These docs summarize the active order and capability posture.
- The product strategy narrative is now maintained in [ARAGORA_IDEA_TO_EXECUTION_STRATEGY](../plans/ARAGORA_IDEA_TO_EXECUTION_STRATEGY.md).

## Execution Order

### 1) Developer Swarm Control Plane And Autonomous Proof Lanes
- Tracking: [#836](https://github.com/synaptent/aragora/issues/836), [#837](https://github.com/synaptent/aragora/issues/837), [#840](https://github.com/synaptent/aragora/issues/840), [#841](https://github.com/synaptent/aragora/issues/841), [#842](https://github.com/synaptent/aragora/issues/842), [#843](https://github.com/synaptent/aragora/issues/843), [#871](https://github.com/synaptent/aragora/issues/871), [#990](https://github.com/synaptent/aragora/issues/990), [#1036](https://github.com/synaptent/aragora/issues/1036), [#1037](https://github.com/synaptent/aragora/issues/1037), [#1038](https://github.com/synaptent/aragora/issues/1038)
- Goal: make unattended repo-improvement lanes truthful by keeping queue state, lane ownership, receipts, review outcomes, publish behavior, and operator visibility canonical.
- Current tranche: queue-backed execution is real on `main`; the remaining gap is finishing proof runs without ambiguous stuck states and preserving every blocker truthfully.

### 2) Sequential Surface Productization And PMF Harvest
- Tracking: [#806](https://github.com/synaptent/aragora/issues/806), [#817](https://github.com/synaptent/aragora/issues/817), [#818](https://github.com/synaptent/aragora/issues/818), [#819](https://github.com/synaptent/aragora/issues/819), [#820](https://github.com/synaptent/aragora/issues/820), [#1011](https://github.com/synaptent/aragora/issues/1011), [#1046](https://github.com/synaptent/aragora/issues/1046), [#1047](https://github.com/synaptent/aragora/issues/1047), [#1048](https://github.com/synaptent/aragora/issues/1048)
- Goal: turn the most visible user-facing surfaces into truthful proof points and harvest the best outputs already produced by the overnight runs.
- Current tranche: `#1108` is merged; `#1110`, `#1111`, `#1113`, and `#1114` are awaiting review and selection; `#1047` and `#819` remain the active proof slices.

### 3) Idea-to-Execution Workbench
- Tracking: [#989](https://github.com/synaptent/aragora/issues/989)
- Goal: unify ideas, goals, actions, and orchestration into one local-first product shell with auditable stage transitions.
- Why now:
  - The backend substrate is stronger than the UI.
  - The strategy narrative is now stable enough to justify a maintained product plan.

### 4) Decision Integrity Kernel Scale-Out
- Tracking: [#805](https://github.com/synaptent/aragora/issues/805), [#810](https://github.com/synaptent/aragora/issues/810), [#811](https://github.com/synaptent/aragora/issues/811), [#812](https://github.com/synaptent/aragora/issues/812), [#813](https://github.com/synaptent/aragora/issues/813), [#814](https://github.com/synaptent/aragora/issues/814), [#815](https://github.com/synaptent/aragora/issues/815), [#816](https://github.com/synaptent/aragora/issues/816)
- Goal: keep the receipt-gated decision kernel canonical while the product and autonomy layers build on top of it.
- Current tranche: the base landed earlier; remaining scale-out is important, but not the first blocker on the current proof lane.

### 5) Truthfulness And Documentation Hygiene
- Tracking: [#804](https://github.com/synaptent/aragora/issues/804), [#807](https://github.com/synaptent/aragora/issues/807), [#808](https://github.com/synaptent/aragora/issues/808), [#809](https://github.com/synaptent/aragora/issues/809)
- Goal: keep `main` docs, readiness claims, and current-source status truthful as the system evolves quickly.
- Current status: complete as a tranche, but still an active discipline. The March 20 doc refresh is part of this work.

### 6) Assurance And GTM Closeout
- Tracking: [#273](https://github.com/synaptent/aragora/issues/273), [#274](https://github.com/synaptent/aragora/issues/274), [#509](https://github.com/synaptent/aragora/issues/509)
- Goal: keep enterprise assurance work real without letting it displace the current autonomy and PMF proof lanes.

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
