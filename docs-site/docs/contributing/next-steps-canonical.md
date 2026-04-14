---
title: Next Steps (Canonical)
description: Next Steps (Canonical)
---

# Next Steps (Canonical)

Last updated: 2026-04-13

This is the single source of truth for short-horizon execution priorities.
[CANONICAL_GOALS](./canonical-goals) defines what Aragora is and why.
[ARAGORA_EVOLUTION_ROADMAP](./aragora-evolution-roadmap) defines the multi-stage architecture.
[ACTIVE_EXECUTION_ISSUES](./active-execution-issues) holds the epic/milestone/issue tree.
[COMMERCIAL_OVERVIEW](../enterprise/commercial-overview) translates proof into market language.

## Current Gate

The current gate is to finish `RS-07`, close the remaining truthful repair gaps in `BC-01/03`, and keep `TW-01/TW-02` publishing recurring truth artifacts before expanding the `B2` guard across the safest execution classes in [#804](https://github.com/synaptent/aragora/issues/804), [#805](https://github.com/synaptent/aragora/issues/805), and [#806](https://github.com/synaptent/aragora/issues/806).

What is already true:

- boss, supervisor, tranche, and swarm infrastructure exist
- host-side install and preflight scripts exist
- bounded product wedges such as prompt-to-spec and inbox workflows exist
- the approved reliability substrate spec identifies the missing layer clearly
- terminal-truth taxonomy, benchmark fixtures, and the benchmark scoring lane are now on `main`
- the 30-day B0 target is already exceeded on the tracked cohort at **86.7%** no-rescue success
- `WorkerContract` and `CredentialEnvelope` primitives exist on the live swarm path
- launcher-side contract admission, dispatch gating, and module-level contract-aware preflight are on `main`
- scratch and remote-publish preflight validation now run through the production preflight path and emit canonical terminal truth on `main`
- task sanitizer outcomes and success-rate filtering are shaping safer boss-loop intake
- original versus sanitized task text is already preserved for audit on `main`
- retry dispatch now carries prior session resume context on `main` via [#5384](https://github.com/synaptent/aragora/pull/5384)
- the rescue loop can now record interventions, plan bounded recovery, and execute safe followups on `main` via [#5379](https://github.com/synaptent/aragora/pull/5379), [#5380](https://github.com/synaptent/aragora/pull/5380), and [#5383](https://github.com/synaptent/aragora/pull/5383)

What is still missing:

- the last `RS-07` step is to make receipt-backed contract preflight the default operator admission truth everywhere, not just a substrate primitive — contract in, persisted receipt out, fail-closed on any mismatch ([#5327](https://github.com/synaptent/aragora/issues/5327))
- broader repair truth on the live swarm loop still depends on full `BC-01` persistence and precise `BC-03` blocker evidence
- recurring scheduled use of the frozen corpus and diffable truth artifact still needs to become routine status output ([#5329](https://github.com/synaptent/aragora/issues/5329))
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

Current status: the tracked B0 cohort is running at **86.7%** no-rescue success as of 2026-04-13. The benchmark target is no longer the blocker; truthful guard completion is.

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
- What remains is recurring scheduled use of that artifact path, not inventing a second benchmark definition.

## 30-Day Canonical Backlog

This is the executable backlog for the next 30 days. Keep it to one bounded lane at a time for a founder budget of 5-10 hours per week.

| Order | Code | Why it matters to the wedge | Acceptance criteria | Proof metric | Layer | GitHub coverage |
|---|---|---|---|---|---|---|
| 1 | `RS-07` | Preflight only deserves trust when it returns a receipt the supervisor can verify, not just a shell exit code. The gap is the admission path: contract in, receipt out, fail-closed on any mismatch. | `aragora swarm preflight run --contract ...` accepts a `WorkerContract`, runs production-equivalent git/auth/env checks, and returns a signed `PreflightReceipt` with pass/fail plus canonical terminal class on failure. Supervisor rejects admission when the receipt is absent or failed. | At least one guarded admission path is live through the operator surface with receipt-backed success and failure. Failures map to canonical terminal truth classes. | substrate | [#5327](https://github.com/synaptent/aragora/issues/5327); covered by [#804](https://github.com/synaptent/aragora/issues/804) and [#805](https://github.com/synaptent/aragora/issues/805). |
| 2 | `BC-01` | Session persistence is the prerequisite for truthful repair, retry, and operator control. | Session state survives `explore -> plan -> edit -> verify -> repair -> publish` and survives process restart. | Benchmark retry lanes show resumed state instead of cold restarts. | control plane | Covered by [#805](https://github.com/synaptent/aragora/issues/805); no dedicated lane issue exists yet. |
| 3 | `BC-03` | Founder time is wasted when a failed run does not say exactly what broke and what to try next. | Failed runs emit precise blocker evidence, canonical blocker class, and repair transcript or next-step evidence. | `100%` of failed bounded runs include receipt-backed blocker evidence mapped to canonical terminal truth. | control plane | Covered by [#805](https://github.com/synaptent/aragora/issues/805); no dedicated lane issue exists yet. |
| 4 | `BC-02` | Retry without state reuse just repeats prompt cost and rescue labor. | Retry resumes from prior state, contract, and repair evidence instead of re-prompting from scratch. | Retried runs emit resume-from-stage evidence and show lower repeated-rescue incidence on the same class. | control plane | Covered by [#805](https://github.com/synaptent/aragora/issues/805); no dedicated lane issue exists yet. |
| 5 | `TW-01` | The wedge only becomes real if the benchmark corpus keeps proving `prompt -> spec -> code -> verify -> PR` loops on bounded work. | A fixed benchmark corpus runs repeatedly on current `main` without ad hoc issue swapping. | Repeated benchmark runs report issue-level truth outcomes on the same bounded corpus. | trust | Covered by [#806](https://github.com/synaptent/aragora/issues/806); no dedicated lane issue exists yet. |
| 6 | `TW-02` | The project needs issue-level truth, not PR-count or iteration-count vanity metrics. | Weekly truth reporting uses `mergeable_pr OR merged_pr`, distinguishes proxy from truth, and stays linked from status docs. | Fresh `origin/main` truth reports publish issue-level truth success and no-rescue truth success. | trust | Covered by [#804](https://github.com/synaptent/aragora/issues/804) and [#806](https://github.com/synaptent/aragora/issues/806); no dedicated lane issue exists yet. |
| 7 | `TW-03` | Human rescues only create leverage when they become fixtures or product work. | Every repeated rescue class becomes a benchmark fixture or bounded substrate issue within one weekly cycle. | Repeated rescue classes trend down, and every repeated class has a linked fixture or bounded issue. | trust | Covered by [#804](https://github.com/synaptent/aragora/issues/804) and [#806](https://github.com/synaptent/aragora/issues/806); no dedicated lane issue exists yet. |
| 8 | `CS-01..03` | The wedge fails commercially if external claims outrun measured proof. | Roadmap, status, and positioning docs keep the wedge-first story and gate claims on measured proof. | External-facing docs stay narrower than current truth metrics and current gate status. | trust | Covered by [#804](https://github.com/synaptent/aragora/issues/804), [#806](https://github.com/synaptent/aragora/issues/806), and the current docs; no dedicated lane issue exists yet. |

## Do Now / Delay / Avoid

### Do now

- `RS-07`
- `BC-01`
- `BC-03`
- `BC-02`
- `TW-01`
- `TW-02`
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

## Top 3 Boss-Ready Next

1. `RS-07` ([#5327](https://github.com/synaptent/aragora/issues/5327)) because it closes the last missing guard on the live operator path and turns preflight into the admission truth the operator can actually trust.
2. `BC-01` because retries and repair loops cannot become truthful until session state survives restarts.
3. `BC-03` because founder leverage depends on precise blocker evidence before more retry logic is added.

There is no dedicated open GitHub issue yet for those three codes. Existing issue coverage is still at the epic level through [#804](https://github.com/synaptent/aragora/issues/804), [#805](https://github.com/synaptent/aragora/issues/805), and [#806](https://github.com/synaptent/aragora/issues/806). Do not invent duplicate roadmap issues in this tranche unless the canonical docs stop being enough to drive bounded execution.

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

- [Evolution roadmap](./aragora-evolution-roadmap)
- [Active execution issues](./active-execution-issues)
- [Commercial overview](../enterprise/commercial-overview)
