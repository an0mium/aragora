# Next Steps (Canonical)

Last updated: 2026-04-13

This is the single source of truth for short-horizon execution priorities.
[CANONICAL_GOALS](../CANONICAL_GOALS.md) defines what Aragora is and why.
[ARAGORA_EVOLUTION_ROADMAP](../plans/ARAGORA_EVOLUTION_ROADMAP.md) defines the multi-stage architecture.
[ACTIVE_EXECUTION_ISSUES](ACTIVE_EXECUTION_ISSUES.md) holds the epic/milestone/issue tree.
[COMMERCIAL_OVERVIEW](../COMMERCIAL_OVERVIEW.md) translates proof into market language.

## Current Gate

The current gate is to finish `RS-07` and `BC-01..03`, then prove the `B2` guard on the safest execution classes across the current execution epics [#804](https://github.com/synaptent/aragora/issues/804), [#805](https://github.com/synaptent/aragora/issues/805), and [#806](https://github.com/synaptent/aragora/issues/806).

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

What is still missing:

- production-equivalent preflight through the operator surface on the safest classes
- resumable session state, retry context, and precise blocker evidence on the live swarm loop
- proof that the B2 guard holds under repeated bounded runs instead of one-off success stories
- broader repair-loop coverage on top of the existing audit trail
- lower-rescue unattended operation on bounded backlogs

The work now is not “add more speculative autonomy.” It is “make bounded unattended execution boring.”

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

## 30-Day Canonical Backlog

This is the executable backlog for the next 30 days. Keep it to one bounded lane at a time for a founder budget of 5-10 hours per week.

| Order | Code | Why it matters to the wedge | Acceptance criteria | Proof metric | Layer | GitHub coverage |
|---|---|---|---|---|---|---|
| 1 | `RS-07` | The remaining guard gap is the top-level operator preflight surface, not the module substrate. | `aragora swarm preflight run --contract ...` (or the canonical operator equivalent) uses the same receipt-backed production preflight path and fails closed on the safe classes. | At least one guarded admission path is live through the operator surface with receipt-backed success and failure. | substrate | Covered by [#804](https://github.com/synaptent/aragora/issues/804) and [#805](https://github.com/synaptent/aragora/issues/805); no dedicated lane issue exists yet. |
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

1. `RS-07` because it closes the last missing guard on the live operator path.
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

- [Evolution roadmap](../plans/ARAGORA_EVOLUTION_ROADMAP.md)
- [Active execution issues](ACTIVE_EXECUTION_ISSUES.md)
- [Commercial overview](../COMMERCIAL_OVERVIEW.md)
