# Execution Plan (Next 6 Weeks)

Last updated: 2026-03-22
Owner: Platform program (Product, Backend, Integrations, QA, SRE)

This is the active short-horizon plan in the agreed priority order.
It complements (does not replace) `docs/status/NEXT_STEPS_CANONICAL.md`.

## Priority Order

1. Close the default product loop on current `main`
2. Make the current proof surfaces truthful by default
3. Finish the bounded execution operator contract
4. Run design-partner PMF loops on the real wedges
5. Expand one truthful workbench/stage-transition slice
6. Decide scale/iterate/narrow from proof metrics, not page count or backlog volume

## Current Merged Proof Basis

The next six weeks should be built on what is already merged:

1. `#1108` proved the queue can recover and publish a real output.
2. `#1110` and `#1111` merged the first user-journey and KM retrieval slices onto `main`.
3. `#1118` and `#1119` made receipts and integrations flows materially more truthful.
4. `#1124`, `#1126`, `#1127`, `#1133`, and `#1138` materially improved the operator contract for bounded repo execution.
5. `#1135`, `#1136`, and `#1137` turned OpenClaw dispatch, the public proof surface, and pipeline live state into real wedge components.

## Week-by-Week Execution

### Week 1: Default Product Loop Closure

Issue-sized tasks:
1. Freeze the canonical guided path: credentials/provider setup -> debate -> receipt -> visible result.
2. Make sure the path uses merged KM retrieval by default and surfaces truthful state at every step.
3. Publish the exact happy path and the first manual step when the path blocks.

Acceptance:
1. One canonical default path is documented and dogfooded internally.
2. The path produces a receipt and a visible result without hidden operator repair.

### Week 2: Truthful Public And Operator Surfaces

Issue-sized tasks:
1. Remove or truthfully mark any remaining optimistic state on `/demo`, integrations status/edit, receipts, and pipeline live state.
2. Tighten the bounded-lane operator view so state, evidence, and next action are all authoritative.
3. Verify remote-head review is the default review target for publishable PRs.

Acceptance:
1. Core proof surfaces have no known demo-only or misleading states.
2. Publishable lanes are reviewable from authoritative operator state.

### Week 3: Bounded Execution Contract

Issue-sized tasks:
1. Ensure completed-lane publish, blocked-lane terminalization, and evidence persistence hold across real runs.
2. Fill the biggest gaps in per-lane provenance/receipt coverage.
3. Capture one concise operator handoff format for blocked runs.

Acceptance:
1. Bounded runs end in deliverable or explicit blocked reason with evidence.
2. No lane requires manual reconstruction to explain what happened.

### Week 4: Design Partner Pilot Start

Issue-sized tasks:
1. Pick 3-5 partners matched to one of the real wedges: trust wedge, public proof/review, or swarm/OpenClaw.
2. Run one guided activation session per partner on a real artifact.
3. Start a weekly PMF scorecard and proof log per partner.

Acceptance:
1. Every partner has one bounded recurring workflow chosen.
2. First-week receipts, overrides, or bounded-lane outcomes are captured in scorecards.

### Week 5: Five Functional Paths + Workbench Slice

Issue-sized tasks:
1. Keep reducing shell-heavy product surfaces by focusing on five functional paths.
2. Extend one stage-transition/workbench slice so it is live, reviewable, and tied to real execution state.
3. Tie workbench state back to canonical receipts/provenance rather than separate demo data.

Acceptance:
1. Five core paths are usable enough for weekly dogfood.
2. At least one stage transition is truthfully represented in the UI.

### Week 6: PMF Decision Gate

Issue-sized tasks:
1. Review six weeks of wedge metrics and partner scorecards.
2. Decide whether to scale, iterate, or narrow each wedge.
3. Promote the successful proof metrics into the default product/program dashboard.

Acceptance:
1. The next six-week plan is generated from measured repeatability and truthfulness.
2. Any wedge that is not repeating gets explicitly narrowed instead of hand-waved forward.

## CI/Gate Commands (Required Weekly)

1. `python scripts/reconcile_status_docs.py --strict --output /tmp/reconciliation_report.md`
2. `python scripts/check_version_alignment.py`
3. `python scripts/check_agent_registry_sync.py`
4. `python scripts/check_connector_exception_handling.py`
5. `python scripts/check_self_host_compose.py`
6. `python scripts/check_pentest_findings.py`
7. `bash scripts/run_offline_golden_path.sh`
