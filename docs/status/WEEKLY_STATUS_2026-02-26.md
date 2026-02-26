# Weekly Status - 2026-02-26

Scope: Epistemic hygiene + settlement reliability lane (runtime KPI automation).

## KPI Automation

1. Added weekly extraction script:
   - `scripts/extract_weekly_epistemic_kpis.py`
2. Added weekly scheduled workflow:
   - `.github/workflows/weekly-epistemic-kpis.yml`
3. Workflow output artifacts:
   - `weekly-epistemic-kpis.json`
   - `weekly-epistemic-kpis.md`

## Runtime Signals Tracked

1. Settlement review success rate.
2. Settlement unresolved due count (last run).
3. Calibration updates realized (`correct + incorrect`).
4. Oracle stream stall rate (`stalls_total / sessions_started`).

## Threshold Defaults

1. Settlement success rate: `>= 0.99`
2. Oracle stall rate: `<= 0.02`
3. Calibration updates realized: `>= 1`

## Notes

1. Workflow supports manual `strict=true` runs that fail on KPI threshold breaches.
2. Scheduled runs always publish artifacts and summary for weekly review.

## Execution Update (2026-02-26, auto-worktree branch)

1. CI signal quality hardening:
   - Added `scripts/check_required_check_priority_policy.py`.
   - Added `tests/scripts/test_check_required_check_priority_policy.py`.
   - Wired lint gate: `.github/workflows/lint.yml` now enforces required-check-priority keep-list policy.
2. Deploy rollback guard hardening:
   - Extended `scripts/check_deploy_secure_sha_guard.py` to validate rollback-gate and rollback-step invariants in `deploy-secure.yml` in addition to SHA verification markers.
   - Expanded unit tests in `tests/scripts/test_check_deploy_secure_sha_guard.py`.
3. Auto-worktree cleanup reliability:
   - Hardened `scripts/codex_worktree_autopilot.py` cleanup flow to keep state when active worktree removal fails and emit explicit failure counters.
   - Added coverage in `tests/scripts/test_codex_worktree_autopilot.py`.
4. Settlement/debate retrieval reliability:
   - Hardened active-debate lookup in `aragora/server/handlers/debates/crud.py` to fail safe on malformed in-memory state.
   - Added integration coverage in `tests/server/handlers/debates/test_handler_integration.py`.
5. Release-readiness discipline:
   - `scripts/ci_release_readiness.sh` now executes branch mutation, deploy SHA/rollback, and required-check-priority policy guards.
   - `.github/workflows/release-readiness.yml` path filters now include those guard scripts.
6. Production deploy rollback parity:
   - `.github/workflows/deploy-secure.yml` production lane now mirrors staging rollback behavior via `rollback_gate`, `Rollback on failure`, and explicit post-rollback failure signaling.
   - `scripts/check_deploy_secure_sha_guard.py` now enforces rollback parity markers for staging+production and validates production gate use of `steps.sha_verify.conclusion`.
7. Required-context mapping validation:
   - `scripts/check_required_check_priority_policy.py` now validates required branch-protection contexts against mapped workflow paths (`lint`, `typecheck`, `sdk-parity`, `Generate & Validate`, `TypeScript SDK Type Check`).
   - Added mapping regression coverage in `tests/scripts/test_check_required_check_priority_policy.py`.
