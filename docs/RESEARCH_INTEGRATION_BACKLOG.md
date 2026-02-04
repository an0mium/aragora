# Research Integration Backlog (GitHub Issue Format)

This backlog is ordered by **impact × feasibility**, aligned with the integration plan.

---

## P0 – Foundation (Weeks 1–4)

### Issue 1: Adaptive Stability Detector (MAD/Judge)
- **Labels:** `research`, `debate`, `performance`, `foundation`
- **Owner:** TBD
- **Depends on:** none
- **Acceptance:**
  - `BetaBinomialStabilityDetector` integrated into `ConsensusEstimator`
  - Early‑stop uses stability score + ASCoT gate
  - Telemetry emitted: `stability_score`, `ks_distance`, `early_stop_reason`
  - Benchmarks show ≥20% round reduction without quality loss

### Issue 2: LaRA Router in Knowledge Mound
- **Labels:** `research`, `knowledge`, `routing`, `foundation`
- **Owner:** TBD
- **Depends on:** none
- **Acceptance:**
  - `LaRARouter` integrated into `QueryOperationsMixin.query()`
  - Config flag to enable + telemetry for route decisions
  - Fallback path preserves existing behavior
  - Benchmarks show no regression vs baseline

### Issue 3: Telemetry & Benchmark Harness
- **Labels:** `observability`, `benchmarks`, `foundation`
- **Owner:** TBD
- **Depends on:** none
- **Acceptance:**
  - Add benchmark scripts for stability + LaRA
  - Emit metrics to `observability/metrics` (route counts, early‑stop rate)
  - Documented in `docs/RESEARCH_INTEGRATION_PLAN.md`

---

## P1 – Verification (Weeks 5–8)

### Issue 4: ThinkPRM Step‑Verification
- **Labels:** `verification`, `debate`, `research`
- **Depends on:** Issue 3
- **Acceptance:**
  - `ThinkPRMVerifier` integrated into debate rounds
  - Per‑round verification scores stored in receipts
  - Gate final consensus if PRM confidence low

### Issue 5: Hilbert‑Style Proofing
- **Labels:** `verification`, `formal`, `research`
- **Depends on:** Issue 4
- **Acceptance:**
  - Recursive proof decomposition prototype
  - Z3/Lean4 proof generation for final consensus

---

## P2 – Knowledge (Weeks 9–12)

### Issue 6: GraphRAG Hybrid Retrieval
- **Labels:** `knowledge`, `rag`, `research`
- **Depends on:** Issue 2
- **Acceptance:**
  - Graph traversal + vector fusion
  - Config‑gated and benchmarked

### Issue 7: ClaimCheck Verification
- **Labels:** `knowledge`, `fact-check`, `research`
- **Depends on:** Issue 6
- **Acceptance:**
  - Atomic claim decomposition pipeline
  - Confidence scoring + evidence links

---

## P3 – Team & Self (Weeks 13–16)

### Issue 8: A‑HMAD Dynamic Roles
- **Labels:** `debate`, `team-selection`, `research`
- **Depends on:** Issue 3
- **Acceptance:**
  - Dynamic role assignment with diversity score
  - Fallback to `DOMAIN_CAPABILITY_MAP`

### Issue 9: SICA Self‑Improvement
- **Labels:** `nomic`, `self-improve`, `research`
- **Depends on:** Issue 3
- **Acceptance:**
  - TestFixer‑driven self‑improvement cycle
  - Sandboxed with rollback + validation

---

## P4 – Integration & Tuning (Weeks 17–20)

### Issue 10: Unified Config + Metrics Tuning
- **Labels:** `integration`, `ops`, `observability`
- **Depends on:** Issues 1–9
- **Acceptance:**
  - All toggles in config
  - Metrics dashboard and alerts for regressions
