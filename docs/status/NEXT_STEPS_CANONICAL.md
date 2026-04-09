# Next Steps (Canonical)

Last updated: 2026-04-09

This is the single source of truth for short-horizon execution priorities.
[CANONICAL_GOALS](../CANONICAL_GOALS.md) defines what Aragora is and why.
[ARAGORA_EVOLUTION_ROADMAP](../plans/ARAGORA_EVOLUTION_ROADMAP.md) defines the multi-stage architecture.
[ACTIVE_EXECUTION_ISSUES](ACTIVE_EXECUTION_ISSUES.md) holds the epic/milestone/issue tree.
[COMMERCIAL_OVERVIEW](../COMMERCIAL_OVERVIEW.md) translates proof into market language.

## Current Gate

The current gate is the transition from early `Teammate` behavior to reliable `Foreman` behavior.

What is already true:

- boss, supervisor, tranche, and swarm infrastructure exist
- host-side install and preflight scripts exist
- bounded product wedges such as prompt-to-spec and inbox workflows exist
- the approved reliability substrate spec identifies the missing layer clearly

What is still missing:

- honest benchmarked proof of semi-autonomous success
- contract-backed worker admission on the safest classes of work
- production-equivalent preflight on those classes
- truthful failure taxonomy and repair loops
- lower-rescue unattended operation on bounded backlogs

The work now is not “add more speculative autonomy.” It is “make bounded unattended execution boring.”

## 30-Day Success Metric

The 30-day target is intentionally narrow:

- fixed benchmark corpus of bounded issues
- context-enriched workers complete **>=50%** of that corpus without human rescue
- **100%** of failures land in truthful canonical buckets
- repeated rescue classes become explicit product work

If a task does not improve that metric, it is not first-tranche work.

## Reverse-Staged Rocket Bootstrap

### Booster 0 — Corpus

Build the fixed benchmark corpus, enrich worker context, and record rescues honestly. This is the smallest booster and the only one that must clearly work in the first 30 days.

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
