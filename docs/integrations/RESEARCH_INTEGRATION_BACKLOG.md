# Research Integration Backlog

Prioritized GitHub issues for implementing research integrations. Issues are ordered by priority within each phase.

**Priority Key:**
- 🔴 P0: Critical path, blocks other work
- 🟠 P1: High impact, should be next
- 🟡 P2: Medium impact, nice to have early
- 🟢 P3: Lower priority, can defer

**Effort Key:**
- XS: < 1 day
- S: 1-2 days
- M: 3-5 days
- L: 1-2 weeks
- XL: > 2 weeks

---

## Phase 1: Foundation (Weeks 1-4)

### Issue #1: 🔴 P0 [S] Add BetaBinomialStabilityDetector for adaptive stopping

**Labels:** `enhancement`, `phase-1`, `debate`, `priority-critical`

**Description:**
Implement adaptive stability detection using Beta-Binomial mixture model to reduce debate compute costs by stopping when consensus stabilizes.

**Acceptance Criteria:**
- [ ] Create `aragora/debate/stability_detector.py` with `BetaBinomialStabilityDetector` class
- [ ] Implement KS-distance calculation between vote distributions
- [ ] Add MUSE divergence gating (don't stop if high disagreement)
- [ ] Add ASCoT fragility gating (don't stop if in late-stage fragile zone)
- [ ] Integrate with `ConsensusEstimator` in `ml_integration.py`
- [ ] Add config flags in `arena_sub_configs.py`
- [ ] Add telemetry events for stability checks
- [ ] Unit tests with >80% coverage

**Technical Notes:**
- Extends existing `ConsensusEstimator.should_terminate_early()`
- Wire into `Arena.run_debate_round()` orchestration loop
- See implementation scaffold in `RESEARCH_INTEGRATION_PLAN.md` Appendix A.1

**Dependencies:** None (foundational)

**Benchmark:** Must achieve ≥20% compute reduction with <5% false early stop rate

---

### Issue #2: 🔴 P0 [M] Implement MUSE ensemble uncertainty calculator

**Labels:** `enhancement`, `phase-1`, `calibration`, `priority-critical`

**Description:**
Add Jensen-Shannon Divergence based uncertainty quantification for multi-model consensus.

**Acceptance Criteria:**
- [ ] Create `aragora/ranking/muse_calibration.py` with `MUSECalculator` class
- [ ] Implement best-subset selection by historical calibration
- [ ] Calculate JSD across subset agent distributions
- [ ] Return `MUSEResult` with consensus confidence and best subset
- [ ] Integrate into `ConsensusPhase._apply_muse_adjustment()`
- [ ] Add `muse_weight` to weight calculation
- [ ] Add telemetry events for MUSE calculations
- [ ] Unit tests with >80% coverage

**Technical Notes:**
- Uses scipy's `jensenshannon` for divergence calculation
- Feeds into adaptive stopping (MUSE divergence gates stability)
- See implementation scaffold in `RESEARCH_INTEGRATION_PLAN.md` Section 1.1

**Dependencies:** None (but integrates with #1)

**Benchmark:** Must achieve ≥10% calibration error reduction

---

### Issue #3: 🟠 P1 [M] Implement LaRA retrieval router

**Labels:** `enhancement`, `phase-1`, `knowledge-mound`, `priority-high`

**Description:**
Add dynamic routing between RAG, RLM, Graph, and Long-Context retrieval based on query and document characteristics.

**Acceptance Criteria:**
- [ ] Create `aragora/knowledge/mound/api/router.py` with `LaRARouter` class
- [ ] Implement query feature extraction (factual, analytical, comparative, etc.)
- [ ] Implement document feature gathering (token count, relationship density)
- [ ] Create routing decision matrix based on LaRA paper findings
- [ ] Integrate into `QueryOperationsMixin.query()` as routing layer
- [ ] Add explicit mode override for deterministic behavior
- [ ] Add config flags for safe rollout (`auto_routing`, etc.)
- [ ] Add telemetry for routing decisions
- [ ] Unit tests with >80% coverage

**Technical Notes:**
- Keep existing `query_semantic()`, `query_graph()`, `query_with_rlm()` unchanged
- Router sits above them as decision layer
- See implementation scaffold in `RESEARCH_INTEGRATION_PLAN.md` Appendix A.2

**Dependencies:** None

**Benchmark:** Must achieve ≥15% retrieval relevance improvement

---

### Issue #4: 🟠 P1 [S] Add ASCoT late-stage fragility detection

**Labels:** `enhancement`, `phase-1`, `debate`, `priority-high`

**Description:**
Implement round-aware fragility scoring that applies higher scrutiny to later debate rounds.

**Acceptance Criteria:**
- [ ] Create `aragora/debate/ascot_fragility.py` with `ASCoTFragilityAnalyzer`
- [ ] Implement exponential fragility curve based on round position
- [ ] Calculate dependency depth for compound error risk
- [ ] Return scrutiny level (LOW, MEDIUM, HIGH, CRITICAL)
- [ ] Integrate into `Arena.run_debate_round()` for verification intensity
- [ ] Feed fragility score into stability detection (#1)
- [ ] Unit tests with >80% coverage

**Technical Notes:**
- Later rounds (>70% of max) get CRITICAL scrutiny
- Fragility gates adaptive stopping
- See implementation scaffold in `RESEARCH_INTEGRATION_PLAN.md` Section 1.2

**Dependencies:** Integrates with #1

---

### Issue #5: 🟡 P2 [S] Add RLM iterative refinement primitives

**Labels:** `enhancement`, `phase-1`, `rlm`, `priority-medium`

**Description:**
Extend existing RLM REPL with checkpoint/rollback and branching primitives for iterative refinement.

**Acceptance Criteria:**
- [ ] Add `CHECKPOINT(state, label)` primitive to save computation state
- [ ] Add `ROLLBACK(label)` primitive to restore state
- [ ] Add `VERIFY_STEP(claim, evidence)` for step-level verification requests
- [ ] Add `BRANCH(alternatives)` for parallel reasoning paths
- [ ] Add `MERGE(branches, strategy)` for combining branch results
- [ ] Update `RLMEnvironment.SAFE_BUILTINS` with new primitives
- [ ] Unit tests for each new primitive

**Technical Notes:**
- Build on existing `aragora/rlm/repl.py` implementation
- Maintain security sandbox (no new attack vectors)
- See implementation scaffold in `RESEARCH_INTEGRATION_PLAN.md` Section 1.3

**Dependencies:** None

---

### Issue #6: 🟡 P2 [XS] Add telemetry infrastructure for research integrations

**Labels:** `enhancement`, `phase-1`, `telemetry`, `priority-medium`

**Description:**
Create telemetry event types and collector for tracking integration health.

**Acceptance Criteria:**
- [ ] Create `aragora/telemetry/research_events.py` with event types
- [ ] Create `aragora/telemetry/collector.py` with `TelemetryCollector`
- [ ] Implement buffering and flush to backend
- [ ] Implement running aggregates for key metrics
- [ ] Add `get_summary()` for dashboard consumption
- [ ] Unit tests

**Technical Notes:**
- See specification in `RESEARCH_INTEGRATION_PLAN.md` Telemetry section
- Should be lightweight, async-safe

**Dependencies:** None (infrastructure)

---

### Issue #7: 🟢 P3 [S] Add benchmark scripts for Phase 1 integrations

**Labels:** `testing`, `phase-1`, `benchmarks`, `priority-low`

**Description:**
Create benchmark scripts to validate Phase 1 integration impact.

**Acceptance Criteria:**
- [ ] Create `scripts/benchmarks/bench_adaptive_stopping.py`
- [ ] Create `scripts/benchmarks/bench_lara_routing.py`
- [ ] Create `scripts/benchmarks/bench_muse_calibration.py`
- [ ] Add benchmark corpus (10-20 sample debates)
- [ ] Add CI job to run benchmarks on PR
- [ ] Document benchmark thresholds in README

**Dependencies:** #1, #2, #3 (benchmarks test those features)

---

## Phase 2: Process Verification (Weeks 5-8)

### Issue #8: 🔴 P0 [L] Implement ThinkPRM step-by-step verifier

**Labels:** `enhancement`, `phase-2`, `verification`, `priority-critical`

**Description:**
Add process reward model for step-by-step debate verification.

**Acceptance Criteria:**
- [ ] Create `aragora/verification/think_prm.py` with `ThinkPRMVerifier`
- [ ] Implement step verification prompt template
- [ ] Parse verification response (VERDICT, CONFIDENCE, REASONING)
- [ ] Implement `verify_debate_process()` for full debate verification
- [ ] Integrate with Arena to trigger targeted revisions on errors
- [ ] Add `enable_process_verification` config flag
- [ ] Add telemetry for PRM verifications
- [ ] Unit tests with >80% coverage

**Technical Notes:**
- Uses verifier agent (default: claude) for step evaluation
- Critical errors in late-stage rounds trigger revision phase
- See implementation scaffold in `RESEARCH_INTEGRATION_PLAN.md` Section 2.1

**Dependencies:** #4 (uses ASCoT fragility to identify late-stage)

**Benchmark:** Must detect >70% of errors with <15% false positive rate

---

### Issue #9: 🟠 P1 [XL] Implement Hilbert recursive formal prover

**Labels:** `enhancement`, `phase-2`, `verification`, `priority-high`

**Description:**
Add recursive informal→formal proof generation for decision verification.

**Acceptance Criteria:**
- [ ] Create `aragora/verification/hilbert_prover.py` with `HilbertProver`
- [ ] Implement claim decomposition into atomic sub-claims
- [ ] Implement claim formalization to Lean4
- [ ] Implement recursive proof tree construction
- [ ] Integrate with existing Lean4 runner
- [ ] Add proof composition for verified sub-claims
- [ ] Add config flags (`hilbert_enabled`, `hilbert_max_depth`)
- [ ] Unit tests with >80% coverage

**Technical Notes:**
- Builds on existing Z3/Lean4 verification infrastructure
- Recursive approach aligns with RLM hierarchical context
- See implementation scaffold in `RESEARCH_INTEGRATION_PLAN.md` Section 2.2

**Dependencies:** Existing Lean4 runner

**Benchmark:** Target 90%+ miniF2F-style verification accuracy

---

## Phase 3: Knowledge & Evidence (Weeks 9-12)

### Issue #10: 🔴 P0 [L] Implement GraphRAG hybrid retrieval

**Labels:** `enhancement`, `phase-3`, `knowledge-mound`, `priority-critical`

**Description:**
Add graph traversal layer to existing vector search for hybrid retrieval.

**Acceptance Criteria:**
- [ ] Create `aragora/knowledge/mound/ops/graph_rag.py` with `GraphRAGRetriever`
- [ ] Implement graph expansion from seed nodes via relationship edges
- [ ] Implement multi-hop reasoning path discovery
- [ ] Implement simple community detection
- [ ] Integrate into `QueryOperationsMixin.hybrid_retrieve()`
- [ ] Add to LaRA router as `HYBRID` mode
- [ ] Unit tests with >80% coverage

**Technical Notes:**
- Uses existing Knowledge Mound relationship edges (supports/contradicts)
- Combined scoring: vector similarity + relationship paths
- See implementation scaffold in `RESEARCH_INTEGRATION_PLAN.md` Section 3.1

**Dependencies:** #3 (LaRA router)

---

### Issue #11: 🟠 P1 [M] Implement ClaimCheck atomic verification

**Labels:** `enhancement`, `phase-3`, `evidence`, `priority-high`

**Description:**
Replace keyword-based evidence grounding with atomic claim decomposition and stepwise verification.

**Acceptance Criteria:**
- [ ] Create `aragora/evidence/claim_decomposer.py` with `ClaimDecomposer`
- [ ] Implement rule-based decomposition for common patterns
- [ ] Implement LLM-based decomposition for complex claims
- [ ] Create `ClaimChecker` for atomic claim verification
- [ ] Implement multi-strategy verification (exact, semantic, inference)
- [ ] Integrate into evidence grounding pipeline
- [ ] Unit tests with >80% coverage

**Technical Notes:**
- Keep existing reliability scoring, upgrade matching
- Atomic claims enable finer-grained verification
- See implementation scaffold in `RESEARCH_INTEGRATION_PLAN.md` Section 3.2

**Dependencies:** None

---

## Phase 4: Team Selection & Self-Improvement (Weeks 13-16)

### Issue #12: 🔴 P0 [L] Implement A-HMAD dynamic role specialization

**Labels:** `enhancement`, `phase-4`, `team-selection`, `priority-critical`

**Description:**
Replace static domain mapping with learned dynamic role specialization.

**Acceptance Criteria:**
- [ ] Create `aragora/debate/role_specializer.py` with `AHMADRoleSpecializer`
- [ ] Implement topic analysis to determine role importance
- [ ] Implement agent capability profiling from ELO/calibration
- [ ] Implement agent-role matching with diversity enforcement
- [ ] Integrate into `TeamSelector.select_team()`
- [ ] Keep static mapping as fallback
- [ ] Add telemetry for role assignments
- [ ] Unit tests with >80% coverage

**Technical Notes:**
- Dynamic roles adapt to specific debate topics
- Diversity score ≥0.6 enforced
- See implementation scaffold in `RESEARCH_INTEGRATION_PLAN.md` Section 4.1

**Dependencies:** Existing ELO system

**Benchmark:** Maintain >0.6 diversity score while improving accuracy 4-6%

---

### Issue #13: 🟠 P1 [XL] Implement SICA self-improvement for Nomic Loop

**Labels:** `enhancement`, `phase-4`, `nomic-loop`, `priority-high`

**Description:**
Add self-editing capabilities to TestFixer and Nomic Loop.

**Acceptance Criteria:**
- [ ] Create `aragora/nomic/sica_improver.py` with `SICAImprover`
- [ ] Implement opportunity identification (performance, reliability, readability)
- [ ] Implement patch generation via LLM
- [ ] Implement validation (tests, type check, lint)
- [ ] Implement backup/restore for safe rollback
- [ ] Integrate with existing TestFixer infrastructure
- [ ] Add human approval gate before applying patches
- [ ] Unit tests with >80% coverage

**Technical Notes:**
- Sandboxed execution with validation gates
- SWE-Bench style self-improvement
- See implementation scaffold in `RESEARCH_INTEGRATION_PLAN.md` Section 4.2

**Dependencies:** Existing TestFixer

**Benchmark:** Target 5% test coverage increase per cycle

---

## Phase 5: Integration & Optimization (Weeks 17-20)

### Issue #14: 🟠 P1 [M] Create unified research integration config

**Labels:** `enhancement`, `phase-5`, `config`, `priority-high`

**Description:**
Create master configuration for all research integrations with feature flags and presets.

**Acceptance Criteria:**
- [ ] Create `aragora/config/research_integration.py` with `ResearchIntegrationConfig`
- [ ] Implement `IntegrationLevel` presets (MINIMAL, STANDARD, FULL, CUSTOM)
- [ ] Add feature flags for each integration
- [ ] Integrate into main config loading
- [ ] Add CLI flags for integration level
- [ ] Documentation

**Dependencies:** All Phase 1-4 issues

---

### Issue #15: 🟡 P2 [M] Add integration metrics dashboard

**Labels:** `enhancement`, `phase-5`, `telemetry`, `priority-medium`

**Description:**
Create Grafana dashboard for monitoring research integration health.

**Acceptance Criteria:**
- [ ] Create dashboard JSON for Grafana
- [ ] Add panels for each integration area
- [ ] Configure alert thresholds
- [ ] Add runbook for common alerts
- [ ] Documentation

**Dependencies:** #6 (telemetry infrastructure)

---

### Issue #16: 🟢 P3 [S] End-to-end integration tests

**Labels:** `testing`, `phase-5`, `integration`, `priority-low`

**Description:**
Create comprehensive integration tests for full pipeline.

**Acceptance Criteria:**
- [ ] Create `tests/research/integration/test_full_debate_pipeline.py`
- [ ] Create `tests/research/integration/test_self_improvement_cycle.py`
- [ ] Test all integration combinations
- [ ] Add to CI pipeline
- [ ] Performance regression tests

**Dependencies:** All Phase 1-4 issues

---

## Issue Dependency Graph

```
Phase 1 (Foundation)
├── #1 Adaptive Stopping ──────────┐
├── #2 MUSE Calibration ───────────┼──► #7 Benchmarks
├── #3 LaRA Router ────────────────┤
├── #4 ASCoT Fragility ◄───────────┤
├── #5 RLM Primitives              │
└── #6 Telemetry ──────────────────┴──► #15 Dashboard

Phase 2 (Verification)
├── #8 ThinkPRM ◄── #4
└── #9 Hilbert Prover

Phase 3 (Knowledge)
├── #10 GraphRAG ◄── #3
└── #11 ClaimCheck

Phase 4 (Team & Self)
├── #12 A-HMAD Roles
└── #13 SICA Improver

Phase 5 (Integration)
├── #14 Unified Config ◄── All above
├── #15 Dashboard ◄── #6
└── #16 E2E Tests ◄── All above
```

---

## Sprint Planning Suggestion

**Sprint 1 (Week 1-2):** #1, #2, #6 (Adaptive Stopping + MUSE + Telemetry)
**Sprint 2 (Week 3-4):** #3, #4, #5 (LaRA + ASCoT + RLM)
**Sprint 3 (Week 5-6):** #7, #8 (Benchmarks + ThinkPRM)
**Sprint 4 (Week 7-8):** #9 (Hilbert)
**Sprint 5 (Week 9-10):** #10, #11 (GraphRAG + ClaimCheck)
**Sprint 6 (Week 11-12):** Buffer / Bug fixes
**Sprint 7 (Week 13-14):** #12 (A-HMAD)
**Sprint 8 (Week 15-16):** #13 (SICA)
**Sprint 9 (Week 17-18):** #14, #15 (Config + Dashboard)
**Sprint 10 (Week 19-20):** #16, polish, release

---

## Quick Start Commands

```bash
# Create all issues (requires gh CLI)
gh issue create --title "[Phase 1] Add BetaBinomialStabilityDetector for adaptive stopping" \
  --body "$(cat docs/RESEARCH_INTEGRATION_BACKLOG.md | sed -n '/### Issue #1/,/### Issue #2/p')" \
  --label "enhancement,phase-1,debate,priority-critical"

# Or use the bulk create script
./scripts/create_research_issues.sh
```
