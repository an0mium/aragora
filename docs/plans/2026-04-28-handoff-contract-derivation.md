# Handoff Contract Derivation Spec

**Status:** spec only — text proposal for operator review.
**Author:** droid (Factory) overnight 2026-04-28.
**Scope:** declarative consolidation of the local-Codex → GitHub automation handoff contract.
**Non-scope:** no implementation, no code, no PR. Stop at spec.

> This document is one of two artifacts requested as overnight read-only-friendly
> work after the 300-PR reassessment surfaced an automation-outbox churn pattern
> (17 `fix(automation)` PRs in 7 days, all touching variants of the same handoff
> contract). It is intentionally written to be reviewable on its own — code
> changes are deferred to the operator and to the agent who owns the automation
> lane (Codex per the 2026-04-28 coordination). Do not implement from this
> document without explicit operator approval.

## Why this exists

Between 2026-04-21 and 2026-04-28, the project shipped **17 separate
`fix(automation)` PRs** that all addressed the same surface — the local Codex
handoff outbox / receipts / branch-audit triple — but each addressed a
*different* edge case in the same underlying contract.

```
#6531  validate outbox handoff contract
#6537  dedupe outbox handoff ids
#6547  dedupe outbox handoffs by branch
#6551  ignore receipted branch handoffs in backlog audit
#6553  skip merged outbox handoffs
#6568  resolve audit handoff state root
#6581  protect unresolved outbox handoffs in backlog audit
#6594  resolve publisher outbox state root
#6595  fingerprint top-level outbox branch fields
#6596  protect top-level outbox handoffs
#6607  protect patch-matched handoffs in audit
#6618  protect superseded outbox branches
#6624  skip patch-equivalent outbox handoffs
#6642  honor receipt-only handoff branches
#6742  keep outbox reconcile dry-run readonly
#6747  dedupe handoffs by open PR branch
#6755  tolerate list evidence in outbox reconcile
```

Per Aragora's Operating Law (`docs/CANONICAL_GOALS.md`):
> If humans intervene twice for the same class of failure, the next system
> change should absorb that rescue as product behavior.

Each of these 17 PRs *did* absorb a rescue class. But the rate is climbing
(daily counts: 1, 0, 12, 9, 9, 14 across the seven-day window) and each fix is
a patch against an implicit invariant rather than a derivation from a written
contract. This spec is the inverse: write the invariants down once so that
future drift produces a contract violation diagnostic rather than a patch.

## Surface in scope

Three scripts and three on-disk directories form the active handoff
substrate today:

| Surface | Role |
|---|---|
| `scripts/publish_automation_handoffs.py` (~1362 lines) | reads outbox files, dedups against open issues, publishes new GitHub issues, writes terminal receipts |
| `scripts/reconcile_automation_outbox.py` (~363 lines) | scans outbox files, archives ones whose PR has merged or whose receipt is terminal, optionally writes synthetic receipts |
| `scripts/audit_codex_branch_backlog.py` (~1069 lines) | classifies local `codex/*` branches as cleanup-eligible vs protected, where "protected" includes any branch with an unresolved outbox handoff |
| `.aragora/automation-outbox/` | open handoff requests (JSON), one per work item awaiting publication |
| `.aragora/automation-outbox-archive/` | satisfied/superseded handoffs (already 123 entries) |
| `.aragora/automation-receipts/` | terminal-state receipts (already 124 entries) |

The `aragora.swarm.boss_loop`, `aragora.swarm.supervisor`, and the boss-loop
preflight gate (`scripts/automation_pr_preflight.sh`) all consume and produce
records on this substrate but don't currently carry the contract themselves.

## Implicit invariants the 17 PRs imply

Reading the diffs in order reveals the **eight latent invariants** the
publisher/reconciler/auditor are converging on. None are currently asserted in
one place:

### 1. Idempotency-key as primary key (PRs #6537, #6547, #6594)

A handoff is uniquely identified by `idempotency_key`. The publisher must dedup
on this key against:
- existing open GitHub issues whose body contains the key
- prior terminal receipts in `.aragora/automation-receipts/` whose status is in
  `{published, already_satisfied, completed, skipped}`
- other outbox files in the same scan window

A handoff with a duplicate key MUST be a no-op against GitHub even if the body
text differs. There is no ambiguity here once stated, but the PRs reveal four
separate dedup keys were tried before settling on `idempotency_key`: by-id, by
branch, by patch, by fingerprint of branch fields.

**Contract clause:** `idempotency_key` is the canonical handoff identity.
Every other identity (issue title, branch, patch hash) is a *secondary index*
used only to detect prior satisfaction when the key alone isn't sufficient.

### 2. Terminal-state precedence (PRs #6551, #6553, #6618, #6642)

A handoff is satisfied (and therefore archivable) if **any** of the following
are true, in this precedence order:

1. A terminal receipt for its `idempotency_key` exists in
   `.aragora/automation-receipts/` with status in
   `{published, already_satisfied}`.
2. Its associated branch's PR is merged or closed-as-superseded.
3. Its associated branch is patch-equivalent to a commit already on the base
   ref (typically `origin/main`), as detected by `git cherry`.
4. A receipt-only handoff branch (no PR opened, but receipt present) marks the
   branch as out-of-scope for the outbox.

**Contract clause:** the four signals above are the *only* satisfaction
predicates. Any new condition added to mark a handoff satisfied MUST be added
to this list, and MUST be enumerable from a single function. The current
implementation has these checks scattered across publisher (#6594/6595),
reconciler (#6553/6618), and auditor (#6551/6607/6624).

### 3. Outbox state-root resolution is a single function (PRs #6568, #6594)

The repo root for `.aragora/automation-outbox/` and `.aragora/automation-receipts/`
must be resolved identically by all three scripts. PR #6568 fixed the auditor
to use the same root as the publisher; PR #6594 fixed the publisher to use the
same root as the reconciler. There is now a third resolution path lurking in
`scripts/automation_pr_preflight.sh` that hasn't been audited.

**Contract clause:** a single utility (e.g.,
`scripts/automation_state_root.py::resolve()`) returns the canonical state-root
path. All three Python scripts and the bash preflight script call this
utility. No ad-hoc resolution.

### 4. Branch field fingerprinting is canonical (PRs #6595, #6596, #6618)

When two outbox files reference the same branch but at different commits or
through different schema (top-level `branch:` vs. nested
`local_evidence.branch`), the auditor and publisher must converge on the same
*fingerprint*. PR #6595 introduced fingerprint-of-top-level-fields; PR #6596
extended it to top-level handoff fields; PR #6618 broadened it to superseded
branches.

**Contract clause:** the fingerprint is a stable function of
`(idempotency_key, branch_name, head_sha)` *plus* a normalized representation
of the requested action. Schema drift in the outbox file (e.g., the
`local_evidence` field becoming a list, see PR #6755) MUST be tolerated by the
fingerprint extractor: it falls back to the top-level fields and emits a
diagnostic warning, never silently mismatches.

### 5. Patch-equivalence is a satisfaction signal, not a hint (PRs #6607, #6624)

If a branch's tip commit is patch-equivalent to a commit already on `main`
(detected via `git cherry origin/main <branch>`), then **the handoff is
satisfied even if no PR was opened and no receipt was written**. This handles
the case where a worker's commits land via a sibling PR that gets squashed or
rebased onto a different feature branch.

**Contract clause:** patch-equivalence on the base ref is a first-class
satisfaction signal, equivalent to terminal receipt or merged PR. The
reconciler MUST archive such handoffs and write a synthetic receipt with
status `already_satisfied`.

### 6. Open-PR identity beats outbox identity (PR #6747)

If two outbox files have different `idempotency_key`s but reference the same
*open* PR, only the first publication should proceed; the second is dedup'd as
"covered by open PR". This protects against worker churn that produces
multiple outbox entries for the same conceptual work.

**Contract clause:** when an outbox file's branch matches an open PR's head
branch, the publisher MUST treat that handoff as already-published and write
an `already_satisfied` receipt. The auditor MUST NOT mark such branches as
cleanup-eligible.

### 7. Dry-run is read-only or it's a bug (PR #6742)

The reconciler accepts `--dry-run` and `--apply` flags. Under `--dry-run`, **no
file system writes** may occur. PR #6742 fixed a regression where dry-run was
moving files. This is non-negotiable for safe overnight runs.

**Contract clause:** every script in this substrate MUST have a single
mutation chokepoint that branches on `--apply`/`--dry-run` and MUST NOT mutate
anywhere else. A test fixture asserts this invariant by running every script
under `--dry-run` against a clean checkout and confirming `git status` shows
no new untracked or modified files.

### 8. Evidence schema is loosely-typed but defensively-validated (PRs #6755, #6531)

Outbox file shape has drifted: `local_evidence` started as a dict, sometimes
became a list, sometimes a JSON string with embedded dict. PR #6531 added the
contract validator; PR #6755 added list-tolerance to the reconciler.

**Contract clause:** outbox files MUST validate against a forward-compatible
schema where:
- required keys (`task`, `requires_github`, `requested_action`, `repo`,
  `local_evidence`, `validation`, `idempotency_key`, `created_at`) are
  present and non-empty
- `local_evidence` is a `Mapping` OR a `Sequence[Mapping]` (with the first
  element treated as canonical)
- `requested_action` is either a known action verb (the union in
  `PR_OPEN_REQUEST_ACTIONS`) or a JSON-string of a Mapping with an `action`
  key
- unknown extra keys are preserved through the publish/archive/receipt
  pipeline (forward-compatible additive growth)

A handoff that fails validation MUST emit a diagnostic *and* be quarantined to
`.aragora/automation-outbox/.invalid/` rather than silently published or
silently dropped.

## Proposed declarative module

A single module (proposed: `aragora/swarm/handoff_contract.py`) would expose:

```python
@dataclass(frozen=True)
class HandoffIdentity:
    idempotency_key: str
    branch_name: str | None
    head_sha: str | None
    fingerprint: str  # derived

@dataclass(frozen=True)
class SatisfactionSignal:
    kind: Literal["terminal_receipt", "merged_pr", "patch_equivalent",
                  "receipt_only_branch", "open_pr_match"]
    evidence: Mapping[str, Any]

def parse_outbox_entry(payload: Mapping[str, Any]) -> HandoffIdentity | InvalidHandoff: ...
def evaluate_satisfaction(ident: HandoffIdentity, *, repo: Path,
                          base_ref: str, receipt_dir: Path,
                          open_pr_heads: Mapping[str, int]) -> SatisfactionSignal | None: ...
def is_dry_run_safe(plan: ReconcilePlan) -> bool: ...
```

Plus a single test fixture `tests/swarm/test_handoff_contract.py` that
exercises:
- every required key combination
- every satisfaction signal type, in precedence order
- schema drift cases (list-typed evidence, JSON-string action, missing
  `local_evidence`)
- dry-run no-write invariant for all three scripts
- fingerprint stability across `local_evidence` shape variants

The publisher, reconciler, and auditor become **thin consumers** of the
contract module. Their existing logic moves into the module's pure functions;
their CLI entry points retain only argument parsing, IO, and the single
`--apply` mutation chokepoint.

## Mapping the 17 PRs to contract clauses

| PR | Clause(s) addressed |
|---|---|
| #6531 | C8 (evidence schema validation) |
| #6537 | C1 (idempotency-key dedup) |
| #6547 | C1 + C4 (branch fingerprinting as secondary dedup) |
| #6551 | C2 (terminal-receipt precedence) |
| #6553 | C2 (merged-PR precedence) |
| #6568 | C3 (state-root resolution unified) |
| #6581 | C2 (unresolved-outbox protection) |
| #6594 | C3 (publisher state-root unified) |
| #6595 | C4 (top-level branch field fingerprint) |
| #6596 | C4 (top-level outbox field fingerprint) |
| #6607 | C5 (patch-equivalent as satisfaction) |
| #6618 | C2 + C5 (superseded branch protection) |
| #6624 | C5 (patch-equivalent skip in publisher) |
| #6642 | C2 (receipt-only branch satisfaction) |
| #6742 | C7 (dry-run no-write invariant) |
| #6747 | C6 (open-PR identity dedup) |
| #6755 | C8 (list-typed evidence tolerance) |

Every PR maps cleanly. **No PR addresses a 9th invariant** — which is
encouraging: the contract surface is closed, not still expanding into
unmapped territory.

## What this contract does NOT solve

To stay honest about scope:

1. **It does not change the rate of new automation features.** New requested
   actions (e.g., `open_or_update_pr` was added late in this window) still
   require ad-hoc additions to `PR_OPEN_REQUEST_ACTIONS`. The contract makes
   the addition point obvious; it does not eliminate the need for the
   addition.
2. **It does not solve the boss-loop preflight bash → Python parity.**
   `scripts/automation_pr_preflight.sh` still has its own state-root
   resolution. C3 calls this out but doesn't fix it.
3. **It does not address GA_CHECKLIST.md staleness** or any other concern
   surfaced in the 300-PR reassessment. Those are separate text-only items.
4. **It does not silence Codex's parser-fix PR work or any other in-flight
   work.** Implementation should be sequenced by the operator after Codex's
   parser PR lands and the four red main workflows clear.

## Sequencing suggestion (operator decision, not droid initiative)

If the operator approves implementation, a safe sequencing is:

1. Land a **read-only contract module** with no behavior change — just the
   dataclasses and pure functions, called by tests but not yet by the three
   scripts. ~300 LOC, additive only, no regression risk.
2. Land a **single test fixture** asserting the eight invariants on the
   current scripts. This will likely fail in 1-2 places (every contract
   derivation reveals at least one bug) — those failures become explicit
   tickets rather than implicit churn.
3. Land **three thin-wrapper PRs**, one per script, that delegate to the
   contract module. Each PR's diff size should be net-negative (deletions of
   patch-handler code).
4. Land a **deprecation note** in CHANGELOG and a single doc entry in
   `docs/swarm/` referencing this spec.

Total: 5 PRs across approximately 800 net lines, of which ~half is deletion.
The current trajectory (17 patch PRs across 7 days) should taper off as the
contract surface closes.

## Stop conditions

This spec is finalized when an operator approves it. It is text only. No code
exists yet. No PR is filed. If the spec is rejected, the existing patch
trajectory continues without harm — every clause above has already been
encoded as a working code path; the only question is whether to consolidate
them.

---

*End of spec. No implementation initiative is requested or implied.*
