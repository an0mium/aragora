# Dialectical Runtime Loop Integration Audit
**Status:** read-only audit — text only, no implementation.
**Author:** worker droid overnight 2026-04-28.
**Scope:** Strategic audit of the seams that should join *decay-signal → crux
debate → arbitration → quarantine → repair → verification → receipt →
genealogy → stress probe* into a single receipt-carrying lineage, as
described in [`docs/plans/2026-04-18-dialectical-runtime-synthesis.md`](2026-04-18-dialectical-runtime-synthesis.md)
and [Pillar 5 of `docs/CANONICAL_GOALS.md`](../CANONICAL_GOALS.md#5-cryptographic-receipts-and-auditability).

## Reading guide

- Every claim cites `path:line` so it can be verified directly.
- "Wired" means there is a non-test production caller invoking the symbol.
- "Scaffolded" means the symbol is implemented and importable, but the only
  callers are unit tests (`tests/...`) or other scaffolded modules with no
  production caller of their own.
- "Orphan" means the symbol exists but has *zero* callers anywhere — even
  in tests — outside its own module.
- All modules in this audit live under `aragora/epistemic/` rather than a
  top-level `aragora/dialectical/` directory; the synthesis plan's
  "dialectical runtime" naming is metaphorical and the implementation
  namespace is `epistemic`.

## TL;DR

The Dialectical Runtime Loop's nine pipeline steps are individually
implemented under `aragora/epistemic/` (DIC-13..28 modules) with strong
unit-test coverage. **None of them are reachable from production code
paths.** The orchestrator (`run_dialectical_loop`,
`aragora/epistemic/runtime_loop.py:228`) is import-safe but has no caller
outside `tests/epistemic/test_runtime_loop.py`. The single integration
seam where a production-emitting subsystem touches the synthesis-plan
modules is the cruxset-emission hook in
`aragora/debate/phases/winner_selector.py:395-411`, and that hook writes
into a different `CruxReceipt` class than the one the Knowledge Mound
adapter ingests, so even when fully enabled it does not close the receipt
lineage. The remainder of the loop is dormant scaffolding gated behind
opt-in environment flags. See the per-step audit and the cross-cutting
findings for the precise seams.

## Pipeline state diagram (text)

The arrows below describe the *intended* flow per the synthesis plan.
Solid `==>` means the seam is wired in production code today; dotted
`-..->` means the function exists but no production caller invokes it;
`X` means the function does not exist.

```
                +-------------------------------+
                |  ProofCarryingCodeUnit YAML   |
                |  (DIC-19, schema-only,        |
                |  scanner is opt-in)           |
                +---------------+---------------+
                                |
                                |  load_proof_units_from_dir()  -..->
                                v   ARAGORA_PROOF_UNIT_SCAN_ENABLED
                +-------------------------------+
                | (1) Decay signal detection    |
                | aragora/epistemic/            |
                |   decay_monitor.py:90         |
                | evaluate_unit() — pure fn     |
                | NO PRODUCTION CALLER          |
                +---------------+---------------+
                                |
                                |  -..-> DecaySignal
                                v
                +-------------------------------+
                | (2) Crux probe (optional)     |
                | runtime_loop.py:_attempt_     |
                |   crux_probe (line 117)       |
                |    ↓ delegates                |
                | reasoning/cruxset_emission.py |
                |   :maybe_emit_cruxset (line   |
                |   57) — *production hook*     |
                |   exists in winner_selector   |
                +---------------+---------------+
                                |
                                | -..-> CruxSet (via runtime_loop)
                                |       OR ==> CruxSet (via debate)
                                v
                +-------------------------------+
                | (3) Operator Crux Arbitration |
                | aragora/epistemic/            |
                |   arbitration.py:184          |
                | build_arbitration() — pure fn |
                | NO PRODUCTION CALLER          |
                | NO `aragora crux arbitrate`   |
                |   CLI verb registered         |
                +---------------+---------------+
                                |
                                |  X (no persist; tuples only)
                                v
                +-------------------------------+
                | (4) Persistent crux storage   |
                | NOT IMPLEMENTED               |
                | No KM table, no SQLite, no    |
                |   JSONL, no scheduler         |
                +---------------+---------------+
                                |
                                |  -..-> QuarantinePolicy
                                v
                +-------------------------------+
                | (5) Quarantine policy         |
                | aragora/epistemic/            |
                |   quarantine_policy.py:105    |
                | apply_quarantine_policy()     |
                |   pure fn (only callsite is   |
                |   runtime_loop.py:309)        |
                +---------------+---------------+
                                |
                                |  -..-> RepairSpec
                                v
                +-------------------------------+
                | (6) Repair candidate          |
                | aragora/epistemic/repair.py   |
                |   :propose_repair (line 76)   |
                |   pure fn (only callsite is   |
                |   runtime_loop.py:312)        |
                | NO PR/SHADOW EXECUTOR         |
                +---------------+---------------+
                                |
                                |  X (no executor)
                                v
                +-------------------------------+
                | (7) Verification gate         |
                | scripts/verify_claims.py      |
                |   (DIC-14 CLI, runs in shell) |
                | aragora/epistemic/            |
                |   claim_verifier.py:          |
                |     ClaimVerifier (line 75)   |
                | NO BRIDGE FROM ITS OUTPUT     |
                |   BACK INTO decay_monitor     |
                +---------------+---------------+
                                |
                                |  X (no caller writes a chain)
                                v
                +-------------------------------+
                | (8) Receipt + provenance      |
                | aragora/gauntlet/receipt_     |
                |   models.py:282 CruxReceipt   |
                |   (production-wired; CLI &    |
                |   gauntlet)                   |
                |       VS                      |
                | aragora/epistemic/crux_       |
                |   receipt.py:62 CruxReceipt   |
                |   (different class; only      |
                |   used by KM adapter & tests) |
                |                               |
                | aragora/knowledge/mound/      |
                |   adapters/crux_receipt_      |
                |   adapter.py:63 — registered  |
                |   in __init__, never called   |
                |                               |
                | aragora/epistemic/            |
                |   genealogy.py:171 — pure fn, |
                |   InMemoryGenealogyStore only |
                +---------------+---------------+
                                |
                                |  X (no closed loop)
                                v
                +-------------------------------+
                | (9) Future-state probing      |
                | aragora/epistemic/            |
                |   stress_test.py:142          |
                | run_stress_test() — pure fn   |
                | NO CATALOG YAMLs IN REPO      |
                |   (docs/status/stress_test_   |
                |    scenarios/ DOES NOT EXIST) |
                +-------------------------------+
```

## Per-step audit

### Step 1: Decay signal detection

- **Implementation:** `aragora/epistemic/decay_monitor.py:90` —
  `evaluate_unit(unit, claim_results, unresolved_crux_ids) -> DecaySignal`
  (DIC-20 / #6031). Reason classes: `failed_claim`, `stale_evidence`,
  `unresolved_crux`, `missing_receipt`, `verifier_error`. Score weights at
  `aragora/epistemic/decay_monitor.py:42-48`.
- **Wiring:** **scaffolded only.** Imported only by
  `aragora/epistemic/__init__.py`, `aragora/epistemic/runtime_loop.py:48`,
  `aragora/epistemic/repair.py:27`,
  `aragora/epistemic/quarantine_policy.py:27`, and the test files
  `tests/epistemic/test_decay_monitor.py`,
  `tests/epistemic/test_runtime_loop.py`.
- **Caller:** none in production. `runtime_loop.run_dialectical_loop()`
  consumes a `DecaySignal` parameter but does *not* itself produce one;
  the caller of `run_dialectical_loop` is responsible for calling
  `evaluate_unit` first, and there is no such caller.
- **Output:** in-memory `DecaySignal` dataclass
  (`aragora/epistemic/decay_monitor.py:65-83`). No persistence, no event
  emission, no log line (the module never calls `logger`).
- **Observability:** none. The unified DAG/GUI does not surface decay
  signals; no Prometheus metric, no KM ingestion, no `event_bus` write.
- **Gap:** No subsystem loads
  `docs/status/proof_units/proof_first_shift.yaml` (the single proof unit
  fixture) and runs `evaluate_unit` against it. The
  `load_proof_units_from_dir` helper at
  `aragora/epistemic/proof_unit_scanner.py:96` is imported by tests only
  (`tests/epistemic/test_proof_unit.py`). The flag
  `ARAGORA_PROOF_UNIT_SCAN_ENABLED` is checked at
  `aragora/epistemic/proof_unit_scanner.py:38-49` but never set or read by
  production code. Step 1 cannot fire today — the input never gets
  loaded.

### Step 2: Crux-finder debate trigger

- **Implementation:**
  - Production hook: `aragora/debate/phases/winner_selector.py:395-411`
    (`maybe_emit_cruxset`, gated by `ARAGORA_CRUXSET_EMISSION_ENABLED`).
    Emission entry point at `aragora/reasoning/cruxset_emission.py:57`.
  - DIC-23 internal hook: `aragora/epistemic/runtime_loop.py:117-149`
    (`_attempt_crux_probe`).
  - CruxSet contract: `aragora/reasoning/cruxset.py` —
    `CruxSet` dataclass at line 128, `Crux` at line 70,
    `CruxPosition` at line 46, `build_cruxset_from_analysis`
    at line 302.
  - Crux-finder consensus mode: `aragora/debate/crux_mode.py` —
    `build_crux_finder_result` at line 75.
- **Wiring:**
  - The cruxset emission *from completed debates* is **wired** at
    `aragora/debate/phases/winner_selector.py:405` and is enabled by
    `ARAGORA_CRUXSET_EMISSION_ENABLED`. The flag check happens inside
    `aragora/reasoning/cruxset_emission.py:42` (`cruxset_emission_enabled`).
    This is the only seam in the entire pipeline that runs on a
    production code path.
  - The crux probe *from a decay signal* is scaffolded; the only path
    into `_attempt_crux_probe` is through `run_dialectical_loop`'s
    `enable_crux_probe=True` parameter
    (`aragora/epistemic/runtime_loop.py:307-321`), and `run_dialectical_loop`
    has no production caller.
  - The CLI command `aragora crux` (`aragora/cli/commands/crux.py:127`)
    invokes a crux-finder debate manually and emits a *gauntlet*
    `CruxReceipt`, not a CruxSet (see step 8 for the type-mismatch).
- **Caller:**
  - Production: `WinnerSelector` phase at
    `aragora/debate/phases/winner_selector.py:395-411` (gated, default
    off).
  - Manual: `aragora crux` CLI at `aragora/cli/commands/crux.py:127`.
  - Test/demo: `scripts/agt_pipeline_dry_run.py:310` (calls
    `propose_followup_for_cruxset` rather than emitting a fresh CruxSet).
- **Output:** When the gate is open, a `CruxSet` is set on
  `result.cruxset` as JSON
  (`aragora/debate/phases/winner_selector.py:412-417`). The CLI
  alternatively writes a JSON `CruxReceipt` artifact
  (`aragora/cli/commands/crux.py:106-110`).
- **Observability:** structured log line `cruxset_emitted cruxset_id=...
  cruxes=...` at `aragora/debate/phases/winner_selector.py:414-417`. No
  metric.
- **Gap:** Decay-signal-triggered cruxes are not emitted (step 1 doesn't
  produce signals). The CruxSet emitted from a winner-selector pass is
  written to the result object but never persisted to KM, the receipt
  store, or the genealogy ledger; see step 8.

### Step 3: CruxArbitration (operator persistent decision)

- **Implementation:** `aragora/epistemic/arbitration.py:184` —
  `build_arbitration(crux, *, operator, side, rationale, ...)
  -> CruxArbitration` (DIC-27 / #6221). Reversal helper at line 231
  (`build_reversal`). Persistent crux qualifier at line 80
  (`PersistentCrux.qualifies`). Threshold constants at lines 31-33:
  `PERSISTENT_CRUX_MIN_SCORE=0.6`, `PERSISTENT_CRUX_MIN_CONSECUTIVE=3`,
  `DEFAULT_EXPIRY_DAYS=90`. Flag gate
  `ARAGORA_CRUX_ARBITRATION_ENABLED` at lines 38-46.
- **Wiring:** **scaffolded only.** Imported only by
  `aragora/epistemic/__init__.py` and
  `tests/epistemic/test_arbitration.py`.
- **Caller:** none. The synthesis plan calls for `aragora crux arbitrate
  <cruxset-id>` (per
  `docs/plans/2026-04-18-dialectical-runtime-synthesis.md` lines 280-286),
  but the only `aragora crux` subcommand registered in
  `aragora/cli/commands/crux.py:127` is the crux-finder debate; no
  `arbitrate` verb is wired into `aragora/cli/parser.py:3304-3308`.
- **Output:** in-memory `CruxArbitration` and `CruxArbitrationReversal`
  dataclasses with SHA-256 `checksum` field
  (`aragora/epistemic/arbitration.py:104-129` and lines 152-165). No
  persistence path.
- **Observability:** none. No KM entry, no receipt store entry, no log
  line, no event.
- **Gap:** The CLI verb is unimplemented; the persistent-crux counter
  (which would track `consecutive_debate_count` across debates) does not
  exist anywhere in the repo (zero matches for `consecutive_debate_count`
  outside `arbitration.py` and its test). The arbitration cannot become a
  belief-network priors update because that wiring is explicitly
  out-of-scope per the module docstring at
  `aragora/epistemic/arbitration.py:18-23`.

### Step 4: Persistent crux storage

- **Implementation:** **does not exist.** No table, no SQLite schema, no
  JSONL writer, no in-memory registry survives across debates.
- **Wiring:** N/A (orphan-by-absence).
- **Caller:** N/A.
- **Output:** N/A. `CruxArbitration.to_dict()` exists at
  `aragora/epistemic/arbitration.py:131-144` but no caller writes the
  result anywhere.
- **Observability:** none.
- **Gap:** Without a persistent store, the
  `consecutive_debate_count` invariant on `PersistentCrux`
  (`aragora/epistemic/arbitration.py:80-86`) cannot be evaluated;
  arbitrations cannot be retrieved by future debates as pinned context;
  reversals cannot reference a prior arbitration. This is a foundational
  gap — without step 4, step 3 can only ever produce ephemeral receipts
  that nothing else can find.

### Step 5: Quarantine policy

- **Implementation:** `aragora/epistemic/quarantine_policy.py:105` —
  `apply_quarantine_policy(signal, policy=None, *, code_unit_class,
  request_live_swap) -> QuarantineDecision` (DIC-21 / #6032).
  Default-policies dict at `aragora/epistemic/quarantine_policy.py:60-76`
  (`live_dispatch`, `report_surface`, `demo`, `pure_policy`, `default`).
  SHA-256 provenance hash for non-`report_only` decisions at
  `aragora/epistemic/quarantine_policy.py:140-155`. Flag at lines 161-168
  (`ARAGORA_QUARANTINE_POLICY_ENABLED`). Permanent live-swap block via
  the `live_swap_blocked` flag at lines 134-141.
- **Wiring:** **scaffolded only.** The only callsite is
  `aragora/epistemic/runtime_loop.py:309`, and `runtime_loop` itself
  is scaffolded.
- **Caller:** `run_dialectical_loop` at
  `aragora/epistemic/runtime_loop.py:309` (no production caller of its
  own). Tests at `tests/epistemic/test_quarantine_policy.py`.
- **Output:** in-memory `QuarantineDecision`
  (`aragora/epistemic/quarantine_policy.py:79-101`). No live route is
  blocked; no degrade behavior is invoked; no `fail_closed` action is
  enforced. The decision is just data.
- **Observability:** none. No log, no metric, no DAG node.
- **Gap:** The "live routing always blocked" invariant
  (`aragora/epistemic/quarantine_policy.py:130-141`) is enforced *as a
  function return value*, not as an actual route block in the dispatch
  pipeline. Even with the flag flipped on, no caller asks the function
  before routing.

### Step 6: Repair candidate generation

- **Implementation:** `aragora/epistemic/repair.py:76` — `propose_repair(
  decay_signal, *, repair_kind="report_only", linked_claims, ...)
  -> RepairSpec` (DIC-22 / #6033). Repair-kinds whitelist at
  `aragora/epistemic/repair.py:30-32`: `report_only`,
  `shadow_candidate`, `pr_candidate`. Permanent block on `live_swap` at
  lines 90-94. Flag `ARAGORA_REPAIR_PIPELINE_ENABLED` at lines 35-43.
- **Wiring:** **scaffolded only.** Sole callsite is
  `aragora/epistemic/runtime_loop.py:312`.
- **Caller:** `run_dialectical_loop` at
  `aragora/epistemic/runtime_loop.py:312`. Tests at
  `tests/epistemic/test_repair.py`.
- **Output:** in-memory `RepairSpec` with optional 64-char SHA-256
  `provenance_hash` (`aragora/epistemic/repair.py:46-72`,
  `aragora/epistemic/repair.py:147-160`). The `proposed_patch` field is
  always empty in current code paths because no caller passes a non-empty
  string.
- **Observability:** none.
- **Gap:** `RepairSpec` carries `validation_commands` and
  `receipt_context` fields (`aragora/epistemic/repair.py:64-66`), but no
  module reads them. There is no PR-creation path, no shadow runner, no
  bridge into the DIC-17 follow-up proposer at
  `aragora/epistemic/followup.py:118` (which is itself only invoked from
  `scripts/agt_pipeline_dry_run.py:310,367`, not from any live debate or
  decay event).

### Step 7: Verification gate (proof-carrying replacement)

- **Implementation:**
  - DIC-14 verifier: `aragora/epistemic/claim_verifier.py` —
    `ClaimVerifier` class at line 75; `verify_manifest` at line 106;
    single-claim path `verify_claim` at line 113.
  - CLI runner: `scripts/verify_claims.py` — `main` at line 51, flagged
    behind `ARAGORA_EPISTEMIC_CLAIMS_ENABLED` per the module docstring at
    `scripts/verify_claims.py:1-13`. Reads `*.yaml` from
    `docs/status/claims/`.
  - There is no other "verification gate" module dedicated to repair
    candidates. The synthesis-plan mention of "verifier candidates" in
    `aragora/reasoning/cruxset.py` (`candidate_verifier` field on the
    `Crux` dataclass) is a free-form string field, not a runner.
- **Wiring:** verifier CLI is **partially wired** (a developer can run
  `python3 scripts/verify_claims.py` and it produces a JSON report). The
  output is *not* consumed by any other production module:
  `evaluate_unit` at `aragora/epistemic/decay_monitor.py:90` accepts a
  `claim_results: dict[str, ClaimResult]` parameter, but no caller wires
  the CLI output back into a decay invocation. There is no scheduled job,
  no pre-merge hook, no boss-loop step that runs the verifier and feeds
  results forward.
- **Caller:** `scripts/verify_claims.py` is invoked from CI/operator
  shells only. No Python module imports its `main` or
  `aragora/epistemic/claim_verifier.py:ClaimVerifier` for runtime use
  (the only importer is `aragora/epistemic/truth_map.py:160` from inside
  `build_truth_map_from_manifests`, which itself has no production
  caller — see step 8).
- **Output:** JSON document with per-claim `pass|fail|stale|unsupported|
  error` status (`aragora/epistemic/claim_verifier.py:31-37`), printed to
  stdout or written to a file via `--output`.
- **Observability:** stdout text only. No KM, no receipt store, no DAG.
- **Gap:** This is the most under-specified seam. The synthesis plan
  asks for "verified replacement candidate" output (per DIC-22 and the
  pipeline diagram at lines 32-44 of
  `docs/plans/2026-04-18-dialectical-runtime-synthesis.md`), but there is
  no module that takes a `RepairSpec`, runs its
  `validation_commands` through `ClaimVerifier`, and gates a downstream
  receipt on the result. `aragora/swarm/proof_first_queue.py:21-77`
  shares the "proof-first" *name* but is a keyword-classifier for the
  boss queue lane (matching `_BENCHMARK_TERMS`, `_RESCUE_TERMS`,
  `_DOCS_TERMS`, etc.); it imports nothing from `aragora.epistemic` and
  does not gate against verified-replacement specs.

### Step 8: Receipt + provenance link

- **Implementation:** **two parallel `CruxReceipt` definitions, no bridge.**
  - **Production (gauntlet):** `aragora/gauntlet/receipt_models.py:282`
    `CruxReceipt`. Built from a `CruxFinderResult` via `build_crux_receipt`
    at line 356 or from a `ConsensusProof` via
    `build_crux_receipt_from_proof` at line 385. SHA-256 `checksum`
    property at line 311. This is the receipt the CLI emits at
    `aragora/cli/commands/crux.py:72,87` and the gauntlet exporters
    consume at `aragora/gauntlet/receipt_exporters.py:21`.
  - **Synthesis-plan (epistemic):** `aragora/epistemic/crux_receipt.py:62`
    `CruxReceipt`. Different shape: `cruxes: list[CruxEntry]` (typed),
    full-length 64-char `checksum` (`aragora/epistemic/crux_receipt.py:78`),
    `convergence_barrier`, `metadata` dict. Built via
    `build_crux_receipt` at `aragora/epistemic/crux_receipt.py:96`.
  - **KM adapter:** `aragora/knowledge/mound/adapters/crux_receipt_adapter.py:63`
    `CruxReceiptAdapter`. Registered in
    `aragora/knowledge/mound/adapters/__init__.py:118-121,283`. The
    adapter's `ingest_crux_receipt` method at line 75 imports from
    `aragora.epistemic.crux_receipt`, **not** from `aragora.gauntlet`.
  - **Genealogy:** `aragora/epistemic/genealogy.py:171` `get_genealogy(
    code_unit_id, store, *, require_enabled=True) -> CodeUnitGenealogy`.
    Sole concrete store is `InMemoryGenealogyStore` at
    `aragora/epistemic/genealogy.py:154`, used in tests only.
  - **DIC-16 receipt provenance helpers:**
    `aragora/export/receipt_epistemic.py` — `receipt_verification_from_claim_result`
    at line 33, `receipt_verification_from_crux` at line 72. Not invoked by
    production code; only `tests/export/test_receipt_epistemic.py`
    references them.
- **Wiring:**
  - The gauntlet `CruxReceipt` *is* produced in production through
    `aragora crux` CLI runs, and `aragora/cli/commands/crux.py:106`
    writes the receipt to disk as JSON. That is the closest thing to a
    "wired" receipt path in the dialectical-runtime stack.
  - The epistemic `CruxReceipt` is **never built** by production code —
    every `from aragora.epistemic.crux_receipt import` in the repo is
    either inside `aragora/epistemic/`,
    `aragora/knowledge/mound/adapters/crux_receipt_adapter.py`, or
    `tests/...`. See `Grep` results in this audit's investigation.
  - `CruxReceiptAdapter.ingest_crux_receipt` has zero production callers.
    The string `ingest_crux_receipt` appears in the source tree only at
    the adapter itself (`aragora/knowledge/mound/adapters/crux_receipt_adapter.py:75`)
    and in `tests/knowledge/mound/adapters/test_crux_receipt_adapter.py`.
  - Genealogy `get_genealogy` has zero callers outside
    `tests/epistemic/test_genealogy.py`.
- **Caller:**
  - `aragora crux` CLI for the gauntlet variant.
  - None for the epistemic variant.
- **Output:** Either a JSON file on disk (CLI mode) or, for the tests of
  the KM adapter, `KnowledgeItem` records keyed by
  `crux_km_<sha256-prefix>` with metadata fields including
  `dic_issue: "DIC-16/#6026"`
  (`aragora/knowledge/mound/adapters/crux_receipt_adapter.py:113-152`).
- **Observability:** filesystem artifact for the CLI; nothing else.
- **Gap:** This is the most disconnected seam of the entire pipeline.
  Even with every flag turned on, a winner-selector debate emits a
  `CruxSet` (not a CruxReceipt), the CLI emits a *gauntlet*
  `CruxReceipt`, and the KM adapter expects an *epistemic* `CruxReceipt`.
  Without a bridge between the two CruxReceipt types, the receipt-and-KM
  provenance loop is structurally open. Genealogy is similarly orphan: it
  never sees a real production lineage because no production store
  implements the `GenealogyStore` protocol at
  `aragora/epistemic/genealogy.py:145`.

### Step 9: Future-state probing (stress test)

- **Implementation:** `aragora/epistemic/stress_test.py:142` —
  `run_stress_test(perturbations, proof_unit_integrities, *, enabled)
  -> StressTestResult` (DIC-25 / #6219). Perturbation kinds at
  `aragora/epistemic/stress_test.py:11-19` (`cve_drop`,
  `api_rate_limit_shift`, `corpus_revision`, `dependency_drop`, etc.).
  Recommended-action ladder at lines 113-118. Flag at lines 22-30
  (`ARAGORA_STRESS_TEST_ENABLED`).
- **Wiring:** **scaffolded only.** Imported only by
  `aragora/epistemic/__init__.py` and
  `tests/epistemic/test_stress_test.py`. The grep match at
  `aragora/gauntlet/orchestrator.py:527,737` is the unrelated method
  `_run_stress_test` on the gauntlet `Pipeline` class — same name, no
  shared symbols.
- **Caller:** none.
- **Output:** in-memory `StressTestResult` with a list of
  `FragilityReport` records (`aragora/epistemic/stress_test.py:60-72,
  78-94`).
- **Observability:** none. The synthesis plan calls for a markdown
  report at `docs/status/generated/stress_test_reports/`
  (`docs/plans/2026-04-18-dialectical-runtime-synthesis.md` lines 220-227);
  that directory does not exist in the repo.
- **Gap:** The synthesis plan also calls for a curated catalog under
  `docs/status/stress_test_scenarios/` (lines 218-220), but that
  directory does not exist either. There is no scheduler, no scripted
  run, no CI artifact. Without inputs (no perturbation YAMLs) and no
  output channel, the function is unreachable from any operator surface.

## Cross-cutting findings

### Pure orphans (no caller anywhere except tests / their own module)

The following symbols are imported nowhere outside `aragora/epistemic/`,
`aragora/knowledge/mound/adapters/crux_receipt_adapter.py`, or `tests/`:

| Symbol | Path:line |
|---|---|
| `evaluate_unit` (DIC-20) | `aragora/epistemic/decay_monitor.py:90` |
| `apply_quarantine_policy` (DIC-21) | `aragora/epistemic/quarantine_policy.py:105` |
| `propose_repair` (DIC-22) | `aragora/epistemic/repair.py:76` |
| `run_dialectical_loop` (DIC-23) | `aragora/epistemic/runtime_loop.py:228` |
| `get_genealogy` (DIC-24) | `aragora/epistemic/genealogy.py:171` |
| `run_stress_test` (DIC-25) | `aragora/epistemic/stress_test.py:142` |
| `scan_coherence` (DIC-26) | `aragora/epistemic/coherence.py:185` |
| `build_arbitration` (DIC-27) | `aragora/epistemic/arbitration.py:184` |
| `build_reversal` (DIC-27) | `aragora/epistemic/arbitration.py:231` |
| `run_gardening_pass` (DIC-28) | `aragora/epistemic/gardening.py:319` |
| `build_truth_map_from_manifests` (DIC-18) | `aragora/epistemic/truth_map.py:148` |
| `CruxReceiptAdapter.ingest_crux_receipt` (DIC-16) | `aragora/knowledge/mound/adapters/crux_receipt_adapter.py:75` |
| `receipt_verification_from_claim_result` (DIC-16) | `aragora/export/receipt_epistemic.py:33` |
| `receipt_verification_from_crux` (DIC-16) | `aragora/export/receipt_epistemic.py:72` |
| `propose_followup_for_crux` (DIC-17) | `aragora/epistemic/followup.py:118` |
| `propose_followup_for_failed_claim` (DIC-17) | `aragora/epistemic/followup.py:240` |
| `bridge_from_crux_position` (AGT-05) | `aragora/reputation/crux_bridge.py:222` |

The single non-test, non-self callsite for the entire DIC-13..28 stack is
`maybe_emit_cruxset` at
`aragora/debate/phases/winner_selector.py:405` (gated, default off, never
persisted).

### Receipts: separate ledgers, no join

There are two `CruxReceipt` classes that look interchangeable but are not:

- `aragora/gauntlet/receipt_models.py:282` — produced by
  `aragora/gauntlet/receipt_models.py:357` (`build_crux_receipt`) and
  `aragora/gauntlet/receipt_models.py:385` (`build_crux_receipt_from_proof`),
  consumed by `aragora/cli/commands/crux.py:72-87` and the gauntlet
  exporter chain. **This is what the live `aragora crux` CLI emits.**
- `aragora/epistemic/crux_receipt.py:62` — produced only by
  `aragora/epistemic/crux_receipt.py:105` (`build_crux_receipt`), which
  takes a `CruxFinderResult` argument; **no production code calls this
  function.** Consumed by
  `aragora/knowledge/mound/adapters/crux_receipt_adapter.py:76` and
  `aragora/epistemic/gardening.py:25-27`.

The `DecisionReceipt` lineage (`aragora/gauntlet/receipt.py`,
`aragora/gauntlet/receipt_store.py`,
`aragora/notifications/receipt_delivery.py`,
`aragora/inbox/receipt_gated_executor.py`) is genuinely production-wired
across many subsystems, but it does not chain into any DIC-23..28 module.
There is no `RepairSpec` → `DecisionReceipt` bridge; no `DialecticalEvent`
→ `DecisionReceipt` chain; no `CruxArbitration` → `DecisionReceipt`
record. The DIC-16 helpers in `aragora/export/receipt_epistemic.py` are
the conceptual bridge but they are imported by tests only.

### DIC-* PR delivery vs. integration

| Step | DIC | Module | Issue / PR | Wired? |
|---|---|---|---|---|
| 1 | DIC-19 / #6030 | `aragora/epistemic/proof_unit*.py` | scaffold | scaffolded |
| 1 | DIC-20 / #6031 | `aragora/epistemic/decay_monitor.py` | scaffold | scaffolded |
| 2 | DIC-15 / #6025 | `aragora/reasoning/cruxset.py`, `aragora/debate/crux_mode.py` | landed | partially wired (winner_selector) |
| 2 | DIC-23 / #6217 (#6600, #6760) | `aragora/epistemic/runtime_loop.py` | landed | scaffolded |
| 3 | DIC-27 / #6221 | `aragora/epistemic/arbitration.py` | landed | scaffolded |
| 4 | — | not implemented | — | absent |
| 5 | DIC-21 / #6032 | `aragora/epistemic/quarantine_policy.py` | landed | scaffolded |
| 6 | DIC-22 / #6033 | `aragora/epistemic/repair.py` | landed | scaffolded |
| 7 | DIC-13 / #6023, DIC-14 / #6024 | `aragora/epistemic/executable_claim.py`, `aragora/epistemic/claim_verifier.py`, `scripts/verify_claims.py` | landed | CLI-wired only |
| 8a | DIC-16 / #6431 (CruxReceipt+KM provenance) | `aragora/gauntlet/receipt_models.py:CruxReceipt` | landed | wired (CLI) |
| 8b | DIC-16 / #6635 (CruxReceiptAdapter — KM ingestion) | `aragora/knowledge/mound/adapters/crux_receipt_adapter.py` | landed | scaffolded |
| 8c | DIC-24 / #6218 (#6561 Epistemic Genealogy ledger) | `aragora/epistemic/genealogy.py` | landed | scaffolded |
| 8d | DIC-26 / #6220 | `aragora/epistemic/coherence.py` | landed | scaffolded |
| 9 | DIC-25 / #6219 | `aragora/epistemic/stress_test.py` | landed | scaffolded |
| 9 | DIC-28 / #6222 (#6459 Proactive Crux Gardening) | `aragora/epistemic/gardening.py` | landed | scaffolded |

(PR mapping above: #6431 introduced KM provenance fields on the gauntlet
`DecisionReceipt`/`CruxReceipt` family, #6442 introduced the DIC-27
arbitration module, #6456 introduced the DIC-13 `ExecutableClaim`
manifest, #6459 introduced proactive crux gardening, #6472 introduced the
DIC-19 proof-carrying code unit schema, #6497 introduced the DIC-14
verifier CLI, #6561 introduced the DIC-24 genealogy ledger, #6600
introduced the DIC-23 runtime-loop scaffold, #6635 introduced the
`CruxReceiptAdapter`, #6760 wired the crux-probe call inside
`run_dialectical_loop`. None of these PRs added a *production caller* for
their respective module.)

### Observability of the loop

Across the nine steps, the only telemetry surfaces are:

- `aragora/debate/phases/winner_selector.py:414-417` — `info` log
  `cruxset_emitted cruxset_id=... cruxes=...` (gated off by default).
- `aragora/debate/phases/winner_selector.py:418-419` — `debug` log on
  CruxSet emission failure.
- `aragora/knowledge/mound/adapters/crux_receipt_adapter.py:83,102` —
  `debug`/`warning` logs in the adapter (only fires in tests).
- `aragora/epistemic/runtime_loop.py:148,310` — `debug` logs on
  crux-probe suppression (only fires in tests).

The unified DAG / GUI does not surface DIC-23..28 events, decay signals,
quarantine decisions, repair specs, arbitration receipts, coherence
issues, gardening reports, or stress-test fragility deltas. None of the
modules emit Prometheus metrics or write to the receipt store.

### Documents-vs-code drift

- `docs/plans/2026-04-18-dialectical-runtime-synthesis.md:280-286`
  describes a CLI verb `aragora crux arbitrate <cruxset-id>`. No such verb
  is registered (`aragora/cli/parser.py:3304-3308` only documents
  `aragora crux <question>`).
- `docs/plans/2026-04-18-dialectical-runtime-synthesis.md:218-220`
  describes a curated perturbation catalog under
  `docs/status/stress_test_scenarios/`. That directory does not exist.
- `docs/plans/2026-04-18-dialectical-runtime-synthesis.md:312-318`
  describes a generated stress-test report under
  `docs/status/generated/stress_test_reports/` and a gardening report
  under `docs/status/generated/gardening_reports/`. Neither directory
  exists.
- The only proof-unit fixture in the repo is
  `docs/status/proof_units/proof_first_shift.yaml` (verified via Glob).
  No production code loads it.

## Recommended next bounded deliverable

**Single-PR proposal: bridge the gauntlet `CruxReceipt` to the epistemic
`CruxReceipt` so KM ingestion of crux-finder debate output becomes
reachable.** This closes the most disconnected seam (step 8) and makes
*one* receipt-carrying lineage observable end-to-end (winner_selector →
KM) the first time anyone enables both
`ARAGORA_CRUXSET_EMISSION_ENABLED` and
`ARAGORA_CRUX_RECEIPT_ENABLED`.

### Proposed scope (additive only, ~200–300 LOC)

1. Add a converter module
   `aragora/epistemic/crux_receipt_bridge.py` (~80 LOC) that takes an
   `aragora.gauntlet.receipt_models.CruxReceipt` and returns an
   `aragora.epistemic.crux_receipt.CruxReceipt`. The conversion is
   straightforward because the data is structurally compatible: each entry
   in `gauntlet_receipt.cruxes` (a `dict`) maps to a `CruxEntry`
   (`aragora/epistemic/crux_receipt.py:29-58`). Preserve `receipt_id`,
   `debate_id`, `question`, `convergence_barrier`, `agents`, `rounds`,
   and `metadata`. Recompute the SHA-256 over canonical JSON to keep the
   epistemic `CruxReceipt`'s 64-char checksum invariant
   (`aragora/epistemic/crux_receipt.py:73-78`). Raise `ValueError` if any
   crux dict is missing `claim_id` / `statement` rather than guessing.

2. Add an opt-in hook in the CLI command
   `aragora/cli/commands/crux.py:cmd_crux` (after the existing
   `_save_receipt_artifact` call) that, when both
   `ARAGORA_CRUX_RECEIPT_ENABLED` and `ARAGORA_KM_CRUX_INGESTION_ENABLED`
   (new flag) are set, calls the bridge, then `await
   CruxReceiptAdapter().ingest_crux_receipt(epistemic_receipt)`. The
   adapter already short-circuits when the existing
   `ARAGORA_CRUX_RECEIPT_ENABLED` flag is off
   (`aragora/knowledge/mound/adapters/crux_receipt_adapter.py:82-91`),
   so no behaviour change ships unless the new flag is also flipped.

3. Add a *unit* test (not an integration test) at
   `tests/epistemic/test_crux_receipt_bridge.py` covering:
   - round-trip equality on `cruxes`, `convergence_barrier`,
     `convergence_barrier=0.0` edge case;
   - `ValueError` on a malformed gauntlet receipt;
   - checksum stability across two conversions of the same input.

4. Add a one-paragraph note to
   `docs/plans/2026-04-18-dialectical-runtime-synthesis.md` (in the
   "What This Adds Versus What Already Exists" section) recording that
   the gauntlet-↔-epistemic CruxReceipt seam is now bridgeable.

### Why this PR (and not a larger one)

- It does not introduce a new subsystem; it adds one converter, one
  optional CLI hook, and one test.
- It is strictly additive: existing CLI runs are unchanged unless the new
  flag is set, and the existing `aragora/knowledge/mound/adapters/crux_receipt_adapter.py`
  is untouched.
- It closes the seam most likely to mislead: a future operator who reads
  the synthesis plan, enables the cruxset emitter, runs a debate, and
  expects KM ingestion to "just work" will find it does. Today, even
  every flag flipped on cannot make ingestion happen.
- It touches zero red workflows, zero `.github/workflows/*` paths, zero
  protected files, and adds no new top-level dependency.
- It is a believable single-evening implementation effort that produces a
  verifiable test artifact.

### What this PR explicitly does not do

- It does not call `evaluate_unit`, `run_dialectical_loop`,
  `apply_quarantine_policy`, `propose_repair`, `run_gardening_pass`,
  `run_stress_test`, `scan_coherence`, `build_arbitration`,
  `build_reversal`, or `get_genealogy` from any production path.
- It does not add a CLI verb such as `aragora crux arbitrate`.
- It does not create any directory under `docs/status/stress_test_scenarios/`,
  `docs/status/generated/`, or any other generated-artifact path.
- It does not change the default state of any flag.
- It does not register a new boss-loop step, work-order, or scheduler
  trigger.
- It does not add a `boss-ready` label to the proposed issue.

## Out of scope

- Implementation of any DIC-23..28 wiring beyond the single bridge above.
- Wide PRs touching more than the four files listed in the proposal.
- File edits to `.github/workflows/*` or any red CI workflow.
- Edits to `CLAUDE.md`, `aragora/__init__.py`, `.env`, or
  `scripts/nomic_loop.py` (protected per `CLAUDE.md`).
- Mutating the live boss queue or filing any new GitHub issue.

## Verification record

This audit was produced via read-only analysis using the following
evidence-gathering steps (each result independently verifiable):

- LS of `aragora/epistemic/` (19 files, all dataclasses + helper
  modules; no orchestration entrypoint other than `runtime_loop.py`).
- Grep for `run_dialectical_loop`, `evaluate_unit`,
  `apply_quarantine_policy`, `propose_repair`, `build_arbitration`,
  `build_reversal`, `run_stress_test`, `scan_coherence`,
  `run_gardening_pass`, `get_genealogy`,
  `CruxReceiptAdapter`, `ingest_crux_receipt` across the entire repo.
  All non-test matches are listed in this audit.
- Grep for the seven `ARAGORA_*_ENABLED` flags introduced by DIC-19..28;
  every one of them is set/read only inside its own module and tests.
- Read of `aragora/debate/phases/winner_selector.py` to confirm the
  single live cruxset-emission seam.
- Read of `aragora/cli/commands/crux.py` and `aragora/cli/parser.py`
  lines 3304-3308 to confirm the absence of an `arbitrate` verb.
- Read of `aragora/swarm/proof_first_queue.py` to confirm it is a
  keyword-classifier, not a verifier gate.
- Read of `docs/status/proof_units/proof_first_shift.yaml` (the only
  proof-unit fixture).
- Read of `docs/plans/2026-04-18-dialectical-runtime-synthesis.md`
  (the source of the nine-step pipeline expectation).
- Read of `docs/CANONICAL_GOALS.md` Pillar 5 (the doctrinal grounding
  for receipt lineage).

The audit makes no claim beyond what is directly cited.
