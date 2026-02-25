# Execution Plan (Next 6 Weeks)

Last updated: 2026-02-25
Owner: Platform program (Backend, Frontend, QA, SRE)
Scope: Operational plan for epistemic hygiene + settlement loop reliability.

This plan does not supersede `/docs/status/NEXT_STEPS_CANONICAL.md`. It translates current priorities into a short execution window with measurable gates.

## Program Outcomes

1. Epistemic hygiene debates produce structured falsifiers, confidence, alternatives, and unknowns by default.
2. Settlement metadata is captured at decision time, reviewed on schedule, and fed into calibration.
3. Debate outputs are measured against eventual outcomes with visible quality signals.
4. Streaming and intervention UX are reliable enough for design-partner usage.

## Target KPIs (6-week window)

1. Epistemic hygiene gate pass rate in CI: `100%` on main.
2. Settlement scaffolding coverage in hygiene mode receipts: `>=95%`.
3. Due-settlement review run success rate: `>=99%`.
4. Calibration updates from settled outcomes: non-zero on every weekly review run.
5. Oracle stream stall rate (first-token + inactivity): `<=2%` on monitored environments.

## Execution Order

### Week 1: Gate and Baseline

Owner: QA + Platform

1. Add dedicated CI gate for epistemic hygiene + settlement tests.
2. Establish baseline test timing and failure signatures for that lane.
3. Record baseline in status docs and CI summary.

Acceptance:

1. Blocking job exists in `.github/workflows/test.yml`.
2. Gate script exists under `scripts/` and runs locally/CI.
3. Gate suite passes on branch and on merge to main.

### Week 2: Settlement Capture Hardening

Owner: Backend

1. Enforce normalized settlement metadata in all hygiene-mode receipts.
2. Ensure production validation rejects incomplete settlement definitions.
3. Verify settlement metadata propagation in API completed payloads.

Acceptance:

1. Receipt store snapshots include claim/falsifier/metric/horizon.
2. Production-mode validation behavior is covered by tests.
3. Completed payloads include mode + settlement snapshot.

### Week 3: Scheduler + Calibration Reliability

Owner: Backend + SRE

1. Harden scheduled review behavior for due and unresolved settlements.
2. Validate calibration writebacks are idempotent and observable.
3. Add dashboard-facing status fields for scheduler health.

Acceptance:

1. Scheduler tests cover due, unresolved, settled, and exception paths.
2. Calibration updates are emitted exactly once per resolved settlement.
3. Dashboard endpoint exposes scheduler status and last-run result.

### Week 4: Adversarial Evaluation Layer

Owner: QA + Reasoning

1. Add stress tests for protocol-gaming attempts in hygiene mode.
2. Add regression fixtures for sycophancy/overconfidence failure modes.
3. Track score deltas across model/provider mixes.

Acceptance:

1. Red-team test fixtures are in CI.
2. Gate fails when compliance regressions cross thresholds.
3. Artifact includes per-model epistemic compliance summary.

### Week 5: Product Surface Tightening

Owner: Frontend + Backend

1. Surface unresolved cruxes and settlement state in debate/outcome UI.
2. Add operator controls for stale streams and settlement review visibility.
3. Ensure intervention + settlement UX is coherent in live debate flow.

Acceptance:

1. Outcome dashboard shows settlement lifecycle status.
2. Live debate view shows mode + settlement metadata when present.
3. E2E checks cover intervention and stream recovery paths.

### Week 6: Release Readiness and Rollout

Owner: Platform + SRE

1. Run full gate bundle with release-grade configuration.
2. Publish operational runbook for hygiene + settlement incidents.
3. Freeze acceptance criteria and promote to default program lane.

Acceptance:

1. All related gates green for release candidate.
2. Runbook is linked from status docs and on-call docs.
3. Program KPIs captured in weekly status update.

## Risks and Mitigations

1. Risk: Gate flakiness from slow tests.
Mitigation: Keep lane focused and marker-filtered; isolate from e2e/load classes.

2. Risk: Settlement outcomes remain unresolved.
Mitigation: Keep explicit unresolved state, scheduled retries, and operator reporting.

3. Risk: Persuasion-optimized outputs pass structure checks but fail substance.
Mitigation: Add adversarial fixtures and outcome-based calibration feedback.

