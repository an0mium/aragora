# 2-Week Hardening Sprint Checklist

Purpose: stabilize recently integrated research features before additional feature work.

Scope covered:
- LaRA routing and GraphRAG
- Adaptive stability detection and process verification
- Hilbert proofing path
- ClaimCheck evidence grounding
- A-HMAD role specialization
- SICA in legacy loop and state-machine handlers

## Ownership Model

Use role owners if individual names are not assigned yet.

- `Debate Owner`: debate engine, stability, process verification, proofing tests
- `Knowledge Owner`: LaRA, GraphRAG, ClaimCheck
- `Nomic Owner`: SICA runtime and state-machine wiring
- `QA Owner`: CI matrix, flaky test triage, regression checks
- `Ops Owner`: metrics dashboards, alerts, rollout controls
- `Security Owner`: self-modification guardrails and rollback policy

## Execution Order (Do Not Reorder)

## 1. Freeze and Baseline (Day 1)

- [ ] Enforce temporary feature freeze for new non-critical integrations.
- [ ] Capture baseline test and benchmark results.
- [ ] Confirm local Python runtime consistency.

Commands:
```bash
pyenv version
python --version
which -a python python3
python -m pytest tests/scripts/test_nomic_sica.py tests/nomic/test_handlers.py tests/reasoning/test_claim_check.py -q
python -m pytest tests/debate tests/knowledge tests/nomic -q
```

Owner: `QA Owner`
Exit criteria:
- Runtime is deterministic (`.python-version` respected).
- Baseline failures are documented with owner and ETA.

## 2. CI Stabilization and Regression Burn-Down (Days 1-4)

- [ ] Fix red tests in debate/knowledge/nomic lanes.
- [ ] Add missing guard tests for feature-gated paths.
- [ ] Ensure all new integrations have at least one disabled-path test (`flag off = old behavior`).

Required test lanes:
```bash
python -m pytest tests/reasoning/test_claim_check.py -q
python -m pytest tests/verification/test_hilbert.py -q
python -m pytest tests/debate/test_stability_detector.py -q
python -m pytest tests/knowledge/test_lara_router.py -q
python -m pytest tests/scripts/test_nomic_sica.py tests/nomic/test_handlers.py -q
```

Owner: `QA Owner` with `Debate Owner`, `Knowledge Owner`, `Nomic Owner`
Exit criteria:
- All targeted lanes green.
- No new flaky tests introduced in these files.

## 3. Runtime/Config Consolidation (Days 3-6)

Goal: align behavior between legacy nomic loop and state-machine handlers.

- [ ] Define single precedence order for SICA config:
  - env vars
  - script config defaults
  - code defaults
- [ ] Remove drift in parsing logic between `scripts/nomic_loop.py` and `aragora/nomic/handlers.py`.
- [ ] Add one shared helper (or documented parity test) for SICA config parsing.

Validation commands:
```bash
python -m pytest tests/scripts/test_nomic_sica.py tests/nomic/test_handlers.py -q
```

Owner: `Nomic Owner`
Exit criteria:
- No behavioral mismatch for same env flags across both execution paths.

## 4. Feature-Gate Matrix Completion (Days 4-8)

- [ ] Add/confirm tests for each integration in both modes:
  - `enable_lara_routing`
  - `enable_graph_rag`
  - `enable_stability_detection`
  - `enable_process_verification`
  - `enable_hilbert_proofing`
  - `NOMIC_SICA_ENABLED`
- [ ] Validate fallback semantics when optional dependencies are unavailable.

Recommended assertions:
- `flag off`: no new path invoked.
- `flag on`: path invoked, no recursion/loop regressions, safe fallback on errors.

Owner: `Debate Owner`, `Knowledge Owner`, `Nomic Owner`
Exit criteria:
- Matrix is complete and linked from PR descriptions.

## 5. Benchmark and Performance Gates (Days 6-9)

- [ ] Run routing and stopping benchmarks on stable corpus.
- [ ] Set thresholds and fail conditions for regressions.
- [ ] Store baseline artifacts for comparison.

Commands:
```bash
python scripts/benchmarks/bench_lara_routing.py --iterations 1000
python scripts/benchmarks/bench_adaptive_stopping.py --iterations 1000 --votes-per-round 7
```

Suggested minimum gates:
- LaRA routing p50 decision latency <= baseline + 10%
- Adaptive stopping false-early-stop proxy <= 5%
- No >10% regression in mean retrieval/consensus path latency

Owner: `QA Owner` + `Knowledge Owner` + `Debate Owner`
Exit criteria:
- Thresholds published in CI or release checklist.

## 6. Observability and Alerting (Days 8-10)

- [ ] Verify emitted metrics for:
  - LaRA route counts
  - debate stability scores
  - early-termination reason counts
- [ ] Add dashboards and initial alert thresholds.

Primary files:
- `aragora/observability/metrics/km.py`
- `aragora/observability/metrics/debate.py`

Owner: `Ops Owner`
Exit criteria:
- Dashboards show live counters/histograms for at least one staging run.

## 7. Safety and Security Review for Self-Modification (Days 9-11)

- [ ] Validate SICA approval gating in TTY and non-TTY contexts.
- [ ] Validate rollback behavior and patch rejection paths.
- [ ] Confirm protected-file policy and audit trail coverage.

Checks:
- approval required + non-TTY -> patch rejected
- failed validation -> rollback or no-apply
- cycle logs include decision metadata

Owner: `Security Owner` + `Nomic Owner`
Exit criteria:
- Safety checklist signed off before wider enablement.

## 8. Rollout Guide and Controlled Enablement (Days 11-14)

- [ ] Publish operator runbook for staged feature enablement.
- [ ] Define canary order and rollback triggers.
- [ ] Keep defaults conservative (all new flags off unless explicitly enabled).

Rollout order:
1. Telemetry only
2. Read-only/analysis mode where available
3. Feature flags in staging
4. Canary subset in production
5. Full rollout after 48h stable metrics

Owner: `Ops Owner` + `QA Owner`
Exit criteria:
- Runbook merged and linked in release notes.

## Daily Tracking Template

Use this in standups.

- Date:
- Blockers:
- Failing tests:
- Regression risk today:
- Planned merges today:
- Rollback risk:

## Sprint Completion Definition

All must be true:

- [ ] Targeted lanes green for debate/knowledge/nomic/reasoning
- [ ] Feature-gate matrix complete and validated
- [ ] Benchmarks recorded with thresholds
- [ ] SICA safety checklist complete
- [ ] Rollout/runbook merged
- [ ] No unresolved P0/P1 regressions from this integration wave
