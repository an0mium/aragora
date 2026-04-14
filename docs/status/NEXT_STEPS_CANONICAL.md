# Next Steps (Canonical)

Last updated: 2026-04-14

This is the single source of truth for short-horizon execution priorities.
[CANONICAL_GOALS](../CANONICAL_GOALS.md) defines what Aragora is and why.
[ARAGORA_EVOLUTION_ROADMAP](../plans/ARAGORA_EVOLUTION_ROADMAP.md) defines the multi-stage architecture.
[ACTIVE_EXECUTION_ISSUES](ACTIVE_EXECUTION_ISSUES.md) holds the epic/milestone/issue tree.
[COMMERCIAL_OVERVIEW](../COMMERCIAL_OVERVIEW.md) translates proof into market language.

## Current Gate

The current gate is to keep `TW-01/TW-02` recurring and boring on current `main`, finish `TW-03` rescue productization, and keep `CS-01..03` narrower than measured proof before expanding the `B2` guard across the safest execution classes in [#804](https://github.com/synaptent/aragora/issues/804), [#805](https://github.com/synaptent/aragora/issues/805), and [#806](https://github.com/synaptent/aragora/issues/806).

What is already true:

- boss, supervisor, tranche, and swarm infrastructure exist
- host-side install and preflight scripts exist
- bounded product wedges such as prompt-to-spec and inbox workflows exist
- the approved reliability substrate spec identifies the missing layer clearly
- terminal-truth taxonomy, benchmark fixtures, and the benchmark scoring lane are now on `main`
- the 30-day B0 target is already exceeded on the tracked cohort at **86.7%** no-rescue success
- `WorkerContract` and `CredentialEnvelope` primitives exist on the live swarm path
- launcher-side contract admission, dispatch gating, and module-level contract-aware preflight are on `main`
- receipt-backed preflight is now the default operator and live dispatch admission truth on `main` via [#5514](https://github.com/synaptent/aragora/pull/5514)
- scratch and remote-publish preflight validation now run through the production preflight path and emit canonical terminal truth on `main`
- task sanitizer outcomes and success-rate filtering are shaping safer boss-loop intake
- original versus sanitized task text is already preserved for audit on `main`
- session state now persists across the live supervisor lease/dispatch lifecycle on `main` via [#5503](https://github.com/synaptent/aragora/pull/5503)
- retry dispatch now carries prior session resume context on `main` via [#5384](https://github.com/synaptent/aragora/pull/5384)
- failed and `needs_human` lanes now persist normalized `blocker_evidence` on `main` via [#5512](https://github.com/synaptent/aragora/pull/5512)
- the rescue loop can now record interventions, plan bounded recovery, and execute safe followups on `main` via [#5379](https://github.com/synaptent/aragora/pull/5379), [#5380](https://github.com/synaptent/aragora/pull/5380), and [#5383](https://github.com/synaptent/aragora/pull/5383)
- recurring benchmark scorecards are now bound to the frozen corpus revision on `main` via [#5582](https://github.com/synaptent/aragora/pull/5582) and [#5583](https://github.com/synaptent/aragora/pull/5583)
- repo-tracked recurring truth publication now lands in `docs/status/generated/benchmark_truth_artifacts/` and `docs/status/generated/benchmark_scorecards/`, with the stable status summary at `docs/status/B0_BENCHMARK_TRUTH_STATUS.md`
- repeated rescue-class reports now include fixture-or-issue productization status on `main` via [#5535](https://github.com/synaptent/aragora/pull/5535)

What is still missing:

- repeated rescue classes still need the broader live-loop conversion from repeated patterns into benchmark fixtures or bounded issues beyond the landed productization report ([#5330](https://github.com/synaptent/aragora/issues/5330))
- proof that the B2 guard holds under repeated bounded runs instead of one-off success stories
- broader repair-loop coverage on top of the existing audit trail
- lower-rescue unattended operation on bounded backlogs

The work now is not “add more speculative autonomy.” It is “make bounded unattended execution boring.”

Queue rule for this tranche:

- only roadmap codes in the **Do now** set may carry or be auto-created with `boss-ready`
- delayed-track issues may stay open for planning truth, but restock and auto-decomposition should strip them from the live dispatch queue

## 30-Day Success Metric

The 30-day target is intentionally narrow:

- fixed benchmark corpus of bounded issues
- context-enriched workers complete **>=50%** of that corpus without human rescue
- **100%** of failures land in truthful canonical buckets
- repeated rescue classes become explicit product work

Current status: the tracked B0 cohort is running at **86.7%** no-rescue success as of 2026-04-13. The benchmark target is no longer the blocker; recurring truth publication and rescue productization are.

Primary truth metric:

- issue-level truth success remains `mergeable_pr OR merged_pr`
- merged-only rate is a secondary truth metric, not the primary gate
- PR-signal counts and iteration counts are proxies only

If a task does not improve that metric, it is not first-tranche work.

## TW-01/TW-02: Benchmark Corpus and No-Rescue Scorecard ([#5329](https://github.com/synaptent/aragora/issues/5329))

`TW-01` (fixed benchmark corpus) and `TW-02` (no-rescue scorecard) are the measurement backbone for the execution wedge. Without them, progress claims are anecdotal.

### Benchmark corpus requirements (TW-01)

- The corpus is a fixed, versioned list of bounded issues checked into the repo (e.g. `docs/benchmarks/corpus.json`).
- Issues in the corpus are not swapped ad hoc between runs; additions and removals are tracked as explicit corpus revisions.
- Each corpus entry includes: issue identifier, expected execution class, and any known constraints.
- The corpus runs against current `main` on a recurring basis (at minimum weekly) using the existing benchmark scoring lane.
- Corpus membership criteria: issues must be bounded (clear scope, single-PR resolution, no external dependency chain).

### No-rescue scorecard requirements (TW-02)

The recurring scorecard records the following per corpus run:

| Metric | Definition | Primary or proxy |
|--------|-----------|-----------------|
| Truth success rate | `mergeable_pr OR merged_pr` per issue | **primary** |
| No-rescue success rate | Truth successes with zero human intervention | **primary** |
| Verification pass rate | Fraction of runs where `verify` stage passes without repair | secondary |
| Failure-class distribution | Count of failures per canonical terminal-truth class | secondary |
| Merged-only rate | Fraction where the PR is actually merged (not just mergeable) | secondary |
| Rescue count | Number of runs requiring human intervention, broken out by rescue type | secondary |

Scorecard output rules:

- Each run produces a timestamped scorecard artifact that is diffable against prior runs for week-over-week comparison.
- Human rescue is distinguished from autonomous completion at the issue level — a rescued issue is never counted as no-rescue success.
- Proxy metrics (PR count, iteration count, token spend) are reported separately and never mixed with truth metrics.
- The scorecard links back to the corpus revision it was run against.

### Current state

- Terminal-truth taxonomy, benchmark fixtures, and the benchmark scoring lane are on `main`.
- The tracked B0 cohort is at **86.7%** no-rescue success as of 2026-04-13.
- The frozen corpus manifest now lives at `docs/benchmarks/corpus.json`.
- The diffable truth artifact path is `scripts/build_benchmark_truth_artifact.py`, with GitHub-truth reconciliation provided by `scripts/reconcile_b0_pr_truth.py`.
- The stable recurring status surface is `docs/status/B0_BENCHMARK_TRUTH_STATUS.md`, backed by the latest JSON pointers under `docs/status/generated/benchmark_truth_artifacts/` and `docs/status/generated/benchmark_scorecards/`.

## 30-Day Canonical Backlog

This is the executable backlog for the next 30 days. Keep it to one bounded lane at a time for a founder budget of 5-10 hours per week.

| Order | Code | Why it matters to the wedge | Acceptance criteria | Proof metric | Layer | GitHub coverage |
|---|---|---|---|---|---|---|
| 1 | `TW-03` | Human rescues only create leverage when they become fixtures or product work. | Every repeated rescue class becomes a benchmark fixture or bounded substrate issue within one weekly cycle. | Repeated rescue classes trend down, and every repeated class has a linked fixture or bounded issue. | trust | [#5330](https://github.com/synaptent/aragora/issues/5330) |
| 2 | `CS-01..03` | The wedge fails commercially if external claims outrun measured proof. | Roadmap, status, and positioning docs keep the wedge-first story and gate claims on measured proof. | External-facing docs stay narrower than current truth metrics and current gate status. | trust | Covered by [#804](https://github.com/synaptent/aragora/issues/804), [#806](https://github.com/synaptent/aragora/issues/806), and the current docs; no dedicated lane issue exists yet. |

## Do Now / Delay / Avoid

### Do now

- `TW-03`
- `CS-01..03`

### Delay

- `BC-07..09` until the repair loop is truthful and resumable
- `RS-10..12` until the operator-surface guard is real
- `TW-07..09` until the bounded execution wedge is boringly reliable
- `UDW-01..06` except for thin read-only queue, receipt, lineage, replay, retry, pause, resume, and override views backed by live runtime truth
- `MCF-01..03` until the wedge needs permissioned memory to improve bounded execution instead of broad retrieval ambition

### Avoid in this tranche

- `UDW-07..12`
- `MCF-04..12`
- `CS-04..12`
- broad provider-surface expansion
- heavy DAG workbench work that is not backed by live runtime truth
- generalized memory fabric work that is not directly improving the execution wedge

## Live Boss-Ready Queue

- `TW-03` ([#5330](https://github.com/synaptent/aragora/issues/5330)) because repeated rescues only create leverage when the live loop turns them into fixtures or bounded issues.
- There is no second dedicated boss-ready issue in this tranche right now; keep `CS-01..03` enforced through the docs/status surfaces until a concrete bounded issue exists.

The live queue for this tranche should now be driven by `TW-03`. `TW-01` ([#5539](https://github.com/synaptent/aragora/issues/5539)) completed on 2026-04-14, `TW-02` is now published through the repo-tracked recurring truth surface at `docs/status/B0_BENCHMARK_TRUTH_STATUS.md`, and [#5516](https://github.com/synaptent/aragora/issues/5516) completed under `TW-03` via [#5535](https://github.com/synaptent/aragora/pull/5535). `RS-07`, `BC-01`, `BC-02`, and `BC-03` are already on `main`; do not recycle them as active blockers unless new evidence shows a concrete regression.

## Reverse-Staged Rocket Bootstrap

### Booster 0 — Corpus

Build the fixed benchmark corpus, enrich worker context, and record rescues honestly. This booster is already above target, so the remaining work is to keep it truthful while the guard layers catch up.

### Booster 1 — Assist

Have the system draft work orders, scope, and validation plans for the safest classes of tasks. Humans approve or edit the draft instead of writing everything from scratch.

### Booster 2 — Guard

Add worker contracts and production-equivalent preflight for the safe classes that already benchmark well. Auto-run only when those guards pass.

### Booster 3 — Repair

Add resumable sessions, retry/repair paths, salvage, and quarantine so common failure classes stop requiring repeated prompt surgery.

### Booster 4 — Multi

Extend the proven loops across hosts with truthful operator state. This is the bridge into early `Foreman` behavior.

## Execution Order

### 1) Corpus and context first

- benchmark corpus plus terminal-truth taxonomy
- context enrichment for the safest bounded issue classes
- honest measurement of no-rescue success rate

### 2) Assisted dispatch second

- auto-drafted work orders and validator plans
- human approval on the safe classes
- clearer issue/task shapes before execution starts

### 3) Guarded autonomy third

- `WorkerContract` plus `CredentialEnvelope`
- contract-aware preflight
- admission gates for the already-proven classes

### 4) Repair and salvage fourth

- resumable session journal
- verify/repair loop
- precise blocker evidence
- sanitizer outcomes persisted for audit

### 5) Multi-host truthful state fifth

- ledger-backed lane, host, and run status
- pause, resume, retry, and salvage controls
- first control-plane or DAG view backed by live state

## Stop / Go Rules

- Do not expand claims if the benchmark corpus is not moving.
- Do not ship GUI surfaces that are not backed by live receipts and contracts.
- Do not treat human rescue as success; convert it into benchmark cases or substrate work.
- If humans intervene twice for the same failure class, the next change should productize that rescue.
- Do not create broad GitHub tasks when the blocker can be stated narrowly.
- Do not let commercial positioning outrun measured proof.

## Done Criteria for This Tranche

This tranche is complete when:

1. a fixed benchmark corpus exists and runs regularly
2. context-enriched workers complete **>=50%** of it without rescue
3. all failures map to truthful canonical classes
4. at least one guarded admission path is real for the safest task class
5. repeated rescue classes are captured as explicit product work instead of hidden labor

## References

- [Evolution roadmap](../plans/ARAGORA_EVOLUTION_ROADMAP.md)
- [Active execution issues](ACTIVE_EXECUTION_ISSUES.md)
- [Commercial overview](../COMMERCIAL_OVERVIEW.md)
