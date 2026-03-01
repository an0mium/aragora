# Dogfooding Spec: Aragora Self-Improvement Pipeline

Generated: 2026-02-28
Status: Execution-ready (plan/spec only)

## 0) Observed Runtime Reality (from live dogfood attempts)

Initial multi-model `aragora ask` runs failed before synthesis output. Blocking faults were:

- Claude agent timeout at 60s (`Agent claude timed out after 60.0s`)
- Debate timeout (`Debate timed out after 240-420s`)
- Anthropic credit/billing failure in background research path
- Context/evidence interface mismatches (`query_semantic`, `search_evidence` attribute errors)
- Calibration fusion adapter method mismatch (fixed)
- Consensus mode `hybrid` routing fallback to `none` (fixed)
- Post-debate async timeout too short for evaluator steps (fixed)

### Post-fix validation update (2026-02-28)

A subsequent live run completed successfully with the required four-model roster and `--consensus hybrid`:

- Command exited `0` after debate + pipeline + decision integrity package
- No `search_evidence` adapter crash
- No Grok terminal `410` failure (`Live search is deprecated`)
- No "unknown consensus mode: hybrid" downgrade

Residual issues still observed:

- `knowledge_mound` context injection still receives an incompatible object in one path (`query_semantic` warning)
- Pre-debate research path can still emit provider quota warnings (now fail-soft, no run abort)
- Final synthesized answer can truncate under heavy output; needs explicit output-length guard/continuation behavior

Implication: Phase P0 remains the top priority until these residual reliability/quality gaps are closed.

## 1) Clarified Intent Matrix

| Field | Spec |
|---|---|
| Objective function | Produce orchestration-ready self-improvement specs that are executable, testable, and auditable with reduced single-model bias. |
| Primary users | Founder/power-user first, then CTO/team mode. |
| Required model set | Claude Opus 4.6 (Claude Code), GPT-5.3 Codex, Gemini 3.1 Pro Preview, Grok 4.20 must be represented in protocol. |
| Non-goals | No broad feature expansion, no new UI surfaces until reliability + spec quality gates pass. |
| Hard constraints | Spec-first execution, explicit acceptance criteria, dissent preserved, rollback for every task, receipt/provenance for every stage. |
| Acceptance criteria | End-to-end run emits all required sections + orchestration JSON; impairment metric <= 0.50; zero blocker runtime errors for 3 consecutive runs. |
| Escalation triggers | Any timeout cascade, missing required section, missing rollback plan, missing test plan, or missing dissent record blocks execution. |

## 2) Debate Protocol Spec (Light Roles, Required Model Presence)

### 2.1 Model roster requirement

Every debate must include all four model lineages in some form:

- `claude` (Claude Code, Opus 4.6)
- `codex` (GPT-5.3 Codex)
- `gemini` (Gemini 3.1 Pro Preview)
- `grok` (Grok 4.20)

### 2.2 Role guidance (not rigid pigeonholing)

Roles are soft and rotate per round:

- Round A: default role assignment
- Round B: proposer/critic rotation
- Round C+: free-form adversarial + synthesis passes

Constraint: each model must contribute at least one proposal and one critique before final synthesis.

### 2.3 Debate phases

1. Intent lock: objective, constraints, non-goals, acceptance criteria
2. Proposal pass: each model proposes plan variant
3. Adversarial pass: each model critiques a different model's proposal
4. Cross-verification pass: challenge unverifiable claims and missing tests
5. Synthesis pass: merged plan + dissent ledger + unresolved risks

### 2.4 Dissent and truth-seeking requirements

- Dissent cannot be dropped; must appear in final artifact.
- Every high-impact claim must be tagged as `verifiable_now`, `verifiable_later`, or `non-verifiable`.
- Persuasion-vs-truth check is mandatory before finalization.

### 2.5 Fail-soft behavior

If one model is unavailable, the run continues in degraded mode but final output is labeled `incomplete_roster=true` and cannot auto-execute.

## 3) Ranked Task Plan (Owners, Tests, Rollback, Gates)

## P0: Reliability hardening (blockers)

### P0.1 Fix calibration fusion adapter registration mismatch (completed)
- Owner files:
  - `aragora/knowledge/mound/adapters/factory.py`
  - `tests/knowledge/mound/adapters/test_factory.py`
- Test plan:
  - `pytest -q tests/knowledge/mound/adapters/test_factory.py::TestAdapterSpecsRegistry::test_calibration_fusion_methods_exist`
- Rollback:
  - Revert the factory mapping change.
- Gate:
  - No adapter-method-missing warning for `calibration_fusion` in debate startup logs.

### P0.2 Remove 60s Claude timeout bottleneck for debate rounds
- Owner files:
  - `aragora/debate/config/defaults.py`
  - `aragora/debate/optimizations.py`
  - `aragora/cli/commands/debate.py`
- Suggested subtasks:
  - Make per-agent timeout explicitly configurable from CLI.
  - Raise default timeout for CLI-heavy model paths.
  - Ensure round timeout scales with roster and timeout.
- Test plan:
  - Debate integration tests with 4-agent roster and long responses.
- Rollback:
  - Restore old timeout defaults.
- Gate:
  - 3 consecutive debate runs complete with final synthesis, no agent timeout errors.

### P0.3 Harden context/evidence interfaces (`query_semantic`, `search_evidence`)
- Owner files:
  - `aragora/debate/knowledge_mound_ops.py`
  - `aragora/debate/context_gatherer/sources.py`
  - `aragora/pipeline/decision_integrity.py`
- Suggested subtasks:
  - Guard calls with capability checks.
  - Add typed adapter interfaces and safe fallback behavior.
- Test plan:
  - Debate + decision-integrity tests with mocked stores lacking optional methods.
- Rollback:
  - Revert interface guards.
- Gate:
  - Zero attribute errors across 10 dry-run debates.

### P0.4 Fail-safe background research when Anthropic credits/API unavailable
- Owner files:
  - `aragora/debate/context_gatherer/gatherer.py`
  - `aragora/debate/context_gatherer/sources.py`
- Suggested subtasks:
  - Detect unavailable providers pre-run.
  - Downgrade to available providers without crashing the run.
- Test plan:
  - Provider-failure simulation tests.
- Rollback:
  - Revert fail-safe branch.
- Gate:
  - No uncaught background task exceptions under provider outage.

## P1: Spec quality and orchestration readiness

### P1.1 Wire prompt-to-spec conductor into user-facing interrogation path
- Owner files:
  - `aragora/server/handlers/interrogation/handler.py`
  - `aragora/interrogation/engine.py`
  - `aragora/prompt_engine/conductor.py`
- Suggested subtasks:
  - Use conductor outputs for crystallized specs.
  - Ensure acceptance criteria, constraints, and rollback are first-class fields.
- Test plan:
  - Interrogation API end-to-end tests from vague prompt to structured spec.
- Rollback:
  - Keep legacy crystallization fallback.
- Gate:
  - Spec emitted in <=5 minutes with all required fields populated.

### P1.2 Enforce task-spec completeness validator
- Owner files:
  - `aragora/prompt_engine/spec_validator.py`
  - `aragora/pipeline/decision_plan/factory.py`
- Suggested subtasks:
  - Reject tasks missing owner files, tests, rollback, gates.
- Test plan:
  - Validator unit tests for pass/fail matrix.
- Rollback:
  - Feature-flag validator enforcement.
- Gate:
  - 100% of generated tasks pass completeness checks or run is blocked.

### P1.3 Standardize dissent-preserving synthesis artifact
- Owner files:
  - `aragora/debate/orchestrator.py`
  - `aragora/debate/post_debate_coordinator.py`
- Suggested subtasks:
  - Add `dissent_ledger` output schema.
  - Persist dissent in receipts.
- Test plan:
  - Debate tests validating dissent presence in final artifacts.
- Rollback:
  - Keep existing synthesis schema behind feature flag.
- Gate:
  - Every run includes dissent or explicit `no_material_dissent=true`.

## P2: Bias/quality evaluation and automation

### P2.1 Implement impairment evaluation harness
- Owner files:
  - `aragora/debate/metrics.py`
  - `aragora/pipeline/outcome_feedback.py`
  - `tests/debate/` (new impairment harness tests)
- Suggested subtasks:
  - Baseline single-model runs on benchmark set.
  - Team-run scoring with same benchmark.
  - Compute impairment and confidence intervals.
- Test plan:
  - Deterministic benchmark fixture with expected metric calculations.
- Rollback:
  - Keep old reporting path.
- Gate:
  - Impairment metric computed and reported for every dogfood cycle.

### P2.2 Add dogfood cycle report artifact
- Owner files:
  - `aragora/nomic/self_improve.py`
  - `aragora/server/handlers/self_improve.py`
- Suggested subtasks:
  - Emit cycle summary with blockers, gates, impairment, receipts.
- Test plan:
  - Self-improve API/CLI tests for report generation.
- Rollback:
  - Make report optional.
- Gate:
  - Report generated in every run and persisted to KM.

## 4) Evaluation Harness Specification

### 4.1 Benchmark setup

- Task count: 10 benchmark tasks
- Mix:
  - 4 reliability/debug tasks
  - 3 spec-quality/planning tasks
  - 3 orchestration/risk-control tasks

### 4.2 Scoring dimensions (0-1)

- correctness
- executability
- spec completeness
- verification quality
- rollback quality
- dissent quality

Composite score = weighted average (weights configurable; default equal weights).

### 4.3 Impairment metric

- `team_score`: composite score from heterogeneous team run
- `best_single_score`: max composite score from any single-model run on same task set
- `impairment = 1 - (team_score / best_single_score)`

Pass threshold:
- hard pass: `impairment <= 0.50`
- stretch pass: `impairment <= 0.25`

### 4.4 Additional gates

- Final synthesis emitted: required
- Runtime blocker count: 0
- Required section completeness: 100%
- Missing rollback/test fields: 0

## 5) Orchestration-Ready JSON Payload

```json
{
  "goal": "Dogfood Aragora self-improvement to produce execution-grade specs with reduced single-model bias and reliable synthesis output.",
  "constraints": {
    "required_models": ["claude-opus-4.6", "gpt-5.3-codex", "gemini-3.1-pro-preview", "grok-4.20"],
    "spec_first": true,
    "preserve_dissent": true,
    "require_test_and_rollback_per_task": true,
    "auto_execute": false
  },
  "acceptance_criteria": [
    "All 5 required sections emitted",
    "No runtime blocker errors",
    "Impairment <= 0.50",
    "Each task includes owner files, tests, rollback, and gates"
  ],
  "tasks": [
    {
      "id": "P0.2",
      "title": "Remove Claude timeout bottleneck",
      "owner_paths": [
        "aragora/debate/config/defaults.py",
        "aragora/debate/optimizations.py",
        "aragora/cli/commands/debate.py"
      ],
      "tests": ["debate integration timeout suite"],
      "rollback": "Restore prior timeout constants and CLI parsing",
      "gate": "3 consecutive successful multi-model syntheses"
    },
    {
      "id": "P0.3",
      "title": "Harden context/evidence interfaces",
      "owner_paths": [
        "aragora/debate/knowledge_mound_ops.py",
        "aragora/debate/context_gatherer/sources.py",
        "aragora/pipeline/decision_integrity.py"
      ],
      "tests": ["attribute-compat mock store tests"],
      "rollback": "Revert interface guard layer",
      "gate": "0 attribute errors in 10 dry runs"
    },
    {
      "id": "P1.1",
      "title": "Wire prompt conductor into interrogation path",
      "owner_paths": [
        "aragora/server/handlers/interrogation/handler.py",
        "aragora/interrogation/engine.py",
        "aragora/prompt_engine/conductor.py"
      ],
      "tests": ["interrogation end-to-end crystallization tests"],
      "rollback": "Feature-flag fallback to legacy crystallizer",
      "gate": "spec emitted with full required fields in <= 5 minutes"
    }
  ],
  "checks": [
    "required_sections_complete",
    "runtime_blockers_zero",
    "impairment_threshold_pass",
    "dissent_ledger_present",
    "receipt_generated"
  ],
  "stop_conditions": [
    "any_runtime_blocker_error",
    "missing_required_models_without_degraded_label",
    "missing_rollback_or_tests_in_any_task",
    "impairment_gt_0.50"
  ]
}
```

## 6) Recommended Dogfood Sequence

1. Reliability gates only (P0) until synthesis output is stable.
2. Spec-quality gates (P1) once runs complete reliably.
3. Impairment benchmarking + automation (P2) after P0/P1 pass.
