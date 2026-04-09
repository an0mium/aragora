# Reverse-Stage Bootstrap Plan (2026-04-09)

## Intent
Bootstrap autonomous capability as a reverse-staged rocket: each stage increases the system’s ability to build the next stage with minimal human intervention. Reliability is the wedge.

## Stage 0: Minimal Booster (1–2 weeks)
Goal: boss-loop completes bounded issues without manual rescue.

Checklist:
- Merge hermetic worker contract (#4148).
- Enable interactive workers by default for autonomous runs.
- Enable boss-loop auto-update (`--boss-auto-update`).
- Add task sanitation v1 (auto-drop malformed auto-decomposed tasks).
- CI: reduce queue pressure (gate frontend E2E/typecheck by scope).

Acceptance:
- Benchmark corpus: ≥50% completion without human rescue.
- Mean attempts per issue < 1.7.

## Stage 1: Booster A (2–4 weeks)
Goal: reduce needs_human churn by repairing inputs instead of decomposing blindly.

Checklist:
- Task sanitation v2: auto-rewrite malformed issues into valid tasks.
- Failure taxonomy in receipts and run summaries.
- Auto-triage: re-queue failed tasks with enriched context + repair hints.

Acceptance:
- Benchmark corpus: ≥70% completion without human rescue.
- Auto-decomposed issues reduced by 50%.

## Stage 2: Booster B (4–8 weeks)
Goal: keep long-running autonomy healthy without restarts.

Checklist:
- Self-healing: profile rotation, permission fallback, rate-limit fallback, git push isolation.
- Stateful worker sessions with resume checkpoints.
- Reliability ledger: quarantine bad profiles automatically.

Acceptance:
- 12-hour unattended run with no manual intervention.
- ≥60% task completion on live boss-ready feed.

## Stage 3: Sustainer (8–12 weeks)
Goal: shift from tasks to goals.

Checklist:
- Goal-to-issue compiler with validation criteria.
- Milestone controller for dependency gating and budgets.
- Progress telemetry for autonomy gain and human intervention debt.

Acceptance:
- 30-day roadmap slice executed with <10 manual interventions.

## Benchmark Corpus & Nightly Dogfood
Benchmark runner:
- `python3 scripts/run_dogfood_benchmark.py --base-report docs/plans/dogfood_pipeline_self_improve_base_report.json --runs 1 --timeout 450 --enforce-pipeline-hard-checks --require-pipeline-hard-checks-presence`

Nightly schedule:
- Self-improve hard-checks run nightly via CI schedule to track reliability trends.

## Owner Notes
- Keep reliability gate metrics visible in receipts.
- Every repeated human intervention becomes a system feature.
