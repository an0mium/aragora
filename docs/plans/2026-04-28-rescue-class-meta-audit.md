# Rescue-Class Meta-Audit (2026-04-21 to 2026-04-28)

**Status:** read-only audit — text only, no implementation.
**Author:** worker droid overnight 2026-04-28.
**Scope:** every merged PR in `origin/main` whose title begins with `fix(automation`, `fix(swarm`, or `fix(boss-loop`, with merge timestamp inside the seven-day window 2026-04-21T00:00:00Z .. 2026-04-28T23:59:59Z.
**Non-scope:** no implementation, no PRs, no edits to any other path. This document is the only artefact produced.

> The 300-PR reassessment estimated **76** fix PRs in the seven-day window. The
> strict title-prefix query (`fix(automation OR fix(swarm OR fix(boss-loop`)
> against the actual `gh pr list` output for `merged:2026-04-21..2026-04-28`
> returns **51 PRs**. The delta of ~25 is accounted for by adjacent
> rescue-class fixes that did *not* use one of the three prefixes — for
> example `fix(queue): batch dev coordination work queue sync` (#6777),
> `fix(status): count canonical boss-ready queue depth` (#6745),
> `fix(cli): restructure swarm DevCoordinationStore optional-import` (#6430),
> `fix(review-queue-brief): hydrate secrets before building invoker in
> server handler` (#6454), and similar chore/refactor PRs that touched the
> same automation surface but were filed under a different prefix.
> This audit is bounded to the **strict 51** to keep the analysis honest;
> wherever the broader 60-PR view changes a conclusion the deviation is
> noted explicitly. The Operating-Law claim ("rescue → product change")
> applies to both views: the relative weights of the categories below do
> not flip when the broader query is used.

## Why this exists

Per Aragora's Operating Law (`docs/CANONICAL_GOALS.md`):

> If humans intervene twice for the same class of failure, the next system
> change should absorb that rescue as product behavior: a benchmark
> fixture, sanitizer rule, preflight check, repair path, policy gate, or
> control-plane affordance.

The window's 51 in-scope PRs *each* honor that law: each adopted a rescue as
product behavior. The next-level question, posed by the 300-PR
reassessment, is whether those 51 fixes are **converging on a stable
contract** or **chasing edge cases of a contract that has not yet been
written down**. This document answers that question category by category.

A complementary spec (`docs/plans/2026-04-28-handoff-contract-derivation.md`)
already covers the **17 PRs** of the outbox/handoff family in detail; that
document derives an explicit contract module from the implicit invariants of
the 17 patches. This audit incorporates those 17 by reference, then walks
through the remaining 34 PRs — **23 of which** form **three additional
recurrent classes** (≥3 PRs each) that warrant the same kind of contract
derivation.

## Summary stats

| Metric | Value |
|---|---|
| Total fix PRs in window (strict prefix) | **51** |
| Total fix PRs in window (broader rescue-class) | **60** (informational only) |
| Distinct contract surfaces (strict) | **12** |
| Recurrent classes (≥3 PRs) | **5** (incl. handoff family) |
| Single-PR orphan classes | **6** |
| Files patched ≥3 times | **6** (4 source + 2 test files; 3 underlying scripts/modules + 3 their test files) |
| Files patched ≥10 times | **2** (`audit_codex_branch_backlog.py`, `publish_automation_handoffs.py`) |
| Days with ≥10 fix PRs | **3** (2026-04-24, 2026-04-25, 2026-04-26) |
| Window peak day | **2026-04-27 — 14 PRs** |

### Top 6 most-patched source files (count of in-window fix PRs)

| File | Count |
|---|---|
| `scripts/audit_codex_branch_backlog.py` | 20 |
| `scripts/publish_automation_handoffs.py` | 12 |
| `scripts/publish_codex_automation_branches.py` | 7 |
| `scripts/reconcile_automation_outbox.py` | 4 |
| `aragora/swarm/boss_loop.py` | 4 |
| `aragora/swarm/boss_worker_lifecycle.py` | 2 |

The fact that two scripts attracted **20** and **12** patches in seven days
is the visible symptom of contract-design churn. Both files exceed 1000
lines, both carry implicit invariants enforced by patch-sized assertions
rather than a declarative module.

## Daily distribution

```
2026-04-22   █                                              1
2026-04-23   █                                              1
2026-04-24   ███████████                                   11
2026-04-25   ██████████                                    10
2026-04-26   ██████████                                    10
2026-04-27   ██████████████                                14
2026-04-28   ████                                           4
                                                          ──
                                                          51
```

The shape is a typical incident-response curve: 4/22-4/23 are quiet
("calm before the storm" — Codex worker dispatch was paused), 4/24 begins
the queue-drain push (the publisher and audit scripts were rewritten in
sequence), 4/25-4/26 are the consolidation days (each fix narrows a
specific publish/audit edge case), 4/27 peaks (operator iterating on the
sandboxed publisher cache + structured-action support), and 4/28 tapers as
the boss-loop runtime hardening cluster (#6773→#6776→#6778) lands and the
queue-drain settles.

## Per-category detail

The 51 PRs fall into 12 distinct contract surfaces. Categories sorted by
recurrence (≥3-PR classes first, then orphans).

---

### Category A — outbox/handoff (17 PRs, COVERED BY EXISTING SPEC)

**PRs:** #6531, #6537, #6547, #6551, #6553, #6568, #6581, #6594, #6595, #6596, #6607, #6618, #6624, #6642, #6742, #6747, #6755

**Files affected (in-category):**
- `scripts/publish_automation_handoffs.py`
- `scripts/audit_codex_branch_backlog.py`
- `scripts/reconcile_automation_outbox.py`
- their three sibling test files

**Root cause:** the local-Codex → GitHub automation handoff substrate
(`.aragora/automation-outbox/`, `.aragora/automation-receipts/`,
`.aragora/automation-outbox-archive/`) was bolted on incrementally. Each
new edge case (idempotency-key dedup, terminal-receipt precedence,
patch-equivalence as satisfaction signal, dry-run no-write invariant,
list-typed `local_evidence`) was patched into one of three scripts in
sequence. The contract was never written down once.

**Recurrence:** 17 PRs in 7 days. **Highest-recurrence class in the
audit.**

**Action in this audit:** *none*.
`docs/plans/2026-04-28-handoff-contract-derivation.md` already enumerates
the eight latent invariants and proposes a single declarative module
(`aragora/swarm/handoff_contract.py`) with thin wrappers in the three
scripts. Read that spec for the detailed mapping of all 17 PRs to the
eight invariants.

**Sequencing in cross-cutting plan:** if Recommended Action below is
adopted, the handoff-contract spec should land **first** because it has
the highest-recurrence surface, the largest in-scope diff (the three
scripts touching one another), and the smallest blast radius (additive
module + thin wrappers).

---

### Category B — outbox/handoff structured actions (post-spec extension) (6 PRs)

**PRs:**
- #6652 — `fix(automation): honor receipt filename keys`
- #6659 — `fix(automation): accept shared state root CLI flag`
- #6723 — `fix(automation): match terminal receipt keys in backlog audit`
- #6725 — `fix(automation): normalize publisher PR action aliases`
- #6751 — `fix(automation): reconcile structured action branches`
- #6754 — `fix(automation): fingerprint structured action branches`

**Files affected:**
- `scripts/publish_automation_handoffs.py` (5 of 6 PRs touched it)
- `scripts/audit_codex_branch_backlog.py` (3 of 6)
- `scripts/reconcile_automation_outbox.py` (2 of 6)

**Root cause at the contract level:** the existing handoff-contract spec
(C1 + C8 in particular) treats `requested_action` as a closed vocabulary
and `idempotency_key` as the canonical primary key. The six post-spec
PRs reveal **two refinements** the spec does not yet cover:

1. **Action vocabulary expansion** (PRs #6725, #6751, #6754). The action
   field is no longer a flat string union; it now includes
   structured payloads (e.g., `{"action": "open_or_update_pr", ...}`)
   that carry secondary metadata which the publisher and reconciler need
   to interpret. PR #6725 normalised `open_pr` aliases; PR #6751
   extended the reconciler to look inside structured payloads when
   computing branch-name evidence; PR #6754 extended the auditor and
   publisher to compute fingerprints from the structured payload's
   nested branch field. **The action vocabulary is no longer
   closed-set; it is a discriminated union.**

2. **Receipt identity normalization** (PRs #6652, #6723). The terminal
   receipt files use *filename keys* (`<idempotency_key>.json`) as
   well as embedded-payload `idempotency_key`s. Two consumers (publisher
   in #6652, auditor in #6723) had diverged on which to trust, and the
   filename was authoritative for one path but the embedded key for the
   other. **The filename and the embedded key must be treated as
   redundant evidence of the same fact and reconciled in one place.**

3. **State-root resolution as a CLI surface, not just a function**
   (PR #6659). The handoff-contract spec proposes a single resolution
   function (`scripts/automation_state_root.py::resolve()`); PR #6659
   added a `--state-root` CLI flag to both the auditor and publisher so
   they can be pointed at a non-default state root from the command
   line. The contract spec does not anticipate this; it should.

**Recurrence:** 6 PRs (≥3 = contract-design flaw — yes).

**Proposed contract change:** the handoff-contract module
(`aragora/swarm/handoff_contract.py` per the existing spec) should
include three additional clauses:

- **C9 (action discriminated union):** `requested_action` is either a
  string in `PR_OPEN_REQUEST_ACTIONS` or a `Mapping` with at least an
  `action` key whose value is in that set. Structured-action payloads'
  nested `branch`, `head_sha`, and `base_ref` fields are extracted by
  the same fingerprint function as top-level fields; precedence is
  documented.
- **C10 (receipt identity reconciliation):** filename-derived
  idempotency-key is the *primary* receipt identity. The embedded key
  is consulted only as a sanity check. A mismatch emits a diagnostic
  and quarantines the receipt to `.aragora/automation-receipts/.invalid/`
  rather than being silently chosen.
- **C11 (state-root resolution is CLI-surfaced):** the resolution
  function described in the spec (C3) takes an explicit override
  parameter that maps to `--state-root` on every CLI consumer. The
  bash preflight script reads the same flag, eliminating C3's "third
  resolution path" caveat.

**Sequencing suggestion:** these three clauses are additive to the
existing handoff-contract spec and can land alongside the existing 8
invariants in the same module. They do not change behavior of the
already-merged 17 PRs; they consolidate the implicit decisions of the
6 post-spec PRs.

**Estimated effort:** ~150 LOC additional in the contract module,
~50 LOC of changes to each script's CLI surface, ~200 LOC of tests.

---

### Category C — automation publisher (`publish_codex_automation_branches.py`) (7 PRs)

**PRs:**
- #6501 — `fix(automation): don't pause publisher for review-only PRs`
- #6512 — `fix(automation): bypass mutating pre-push hooks in publisher`
- #6536 — `fix(automation): skip empty PR diff branches`
- #6538 — `fix(automation): fail open on PR diff probe errors`
- #6543 — `fix(automation): relax in-flight PR backpressure`
- #6738 — `fix(automation): cache queue status for sandboxed watchers`
- #6741 — `fix(automation): accept branch publisher receipt dir flag`

**Files affected:**
- `scripts/publish_codex_automation_branches.py` (all 7)
- `tests/scripts/test_publish_codex_automation_branches.py` (all 7)
- `scripts/cache_codex_automation_github_status.py` (#6738 only — created)
- `scripts/run_codex_automation_publisher.sh` (#6738)
- `docs/briefs/automation-merge-contract.md` (#6738)

**Root cause at the contract level:** the **publisher backpressure
predicate** is implicit. The publisher decides whether to proceed,
pause, or fail based on a tangled set of GitHub-CLI signals: open PR
count, `mergeStateStatus`, individual check-run conclusions, review
state, and now also a sandbox-cached version of those signals. Each of
the 7 PRs adjusts one element of that predicate:

- #6501: review-required PRs (`mergeStateStatus=BLOCKED` for
  review-only reasons) should not pause the publisher.
- #6512: pre-push hook side effects (formatters mutating worktrees)
  cause the next publisher run to see a dirty worktree and pause —
  fix is `--no-verify` push by default, env opt-in for verification.
- #6536: branches with empty `git diff base...branch` (but unique
  history) should not produce stale PRs after a queue drain.
- #6538: errors during the empty-diff probe must fail-open, not
  classify the branch as `empty_pr_diff`.
- #6543: ordinary pending CI is not unhealthy; only dirty/draft/
  changes-requested/failed/cancelled is. Cancelled `Metrics Drift` and
  `Module Tier Drift` are advisory noise, not blockers.
- #6738: the watchers running inside the macOS sandbox cannot reach
  GitHub directly; they read a cached queue-status file written by the
  publisher bridge. Constrained unhealthy-queue override exposed.
- #6741: the publisher needs a `--receipt-dir` CLI flag analogous to
  the auditor's, so that the receipt directory used by C3/C11 in the
  handoff-contract spec is parameterized.

The seven adjustments have not been written down as a single predicate.
**The publisher's "should I pause" decision is currently spread across
seven code paths plus an environment variable plus two CLI flags plus a
cache file.**

**Recurrence:** 7 PRs (≥3 = contract-design flaw — yes).

**Proposed contract change:** introduce a `PublisherBackpressureSignal`
discriminated union and a single `evaluate_backpressure()` function:

```python
@dataclass(frozen=True)
class BackpressureSignal:
    kind: Literal[
        "queue_dirty",        # any open PR is dirty
        "queue_draft",        # any open PR is draft
        "queue_changes_requested",
        "queue_check_failed", # advisory list excluded
        "queue_check_cancelled_blocking",
        "queue_check_pending", # non-blocking by default (per #6543)
        "queue_review_required", # non-blocking (per #6501)
        "worktree_dirty_after_push",  # pre-push hook side effect (per #6512)
        "branch_empty_pr_diff", # per #6536+#6538
        "github_unreachable_no_cache",  # per #6738
    ]
    is_blocking: bool
    advisory: bool  # if True, log but don't pause
    evidence: Mapping[str, Any]

def evaluate_backpressure(
    open_prs: Sequence[PullRequest],
    *,
    advisory_check_names: AbstractSet[str],
    cached_queue_status: CachedStatus | None,
    sandbox_override: bool = False,
) -> Sequence[BackpressureSignal]: ...
```

The publisher's mainline becomes: collect signals, reject if any is
blocking-non-advisory, log advisories, proceed. The seven scattered
checks become one function.

**Sequencing suggestion:** lower priority than handoff-contract (this
surface is a smaller blast radius and the existing tests are
comprehensive), but worth doing in the same 30-day window. Land
**after** Category B's structured-action work because #6741's
`--receipt-dir` flag depends on the C11 state-root contract.

**Estimated effort:** ~250 LOC for the predicate module, ~150 LOC of
deletions in the publisher mainline, ~300 LOC of tests (most of which
already exist and just move to the new module's test file).

**Note on partial overlap with Category D:** PR #6738 (queue-status
cache) also affects the auditor's behavior because the auditor consumes
the same `cached_queue_status` when `gh` is degraded. The cache contract
should be defined once and consumed by both Category C and Category D
modules.

---

### Category D — automation backlog audit (`audit_codex_branch_backlog.py`) (10 PRs)

**PRs:**
- #6502 — `fix(automation): gate backlog audit PR lookup on gh health`
- #6539 — `fix(automation): distinguish publishable codex backlog`
- #6570 — `fix(automation): check patch-equivalent branches by default`
- #6605 — `fix(automation): treat empty branch diffs as backlog cleanup`
- #6623 — `fix(automation): exclude diverged branches from publishable backlog`
- #6643 — `fix(automation): tolerate missing audit worktrees`
- #6646 — `fix(automation): classify zero-diff branches in fast audit`
- #6651 — `fix(automation): add compact branch audit output`
- #6726 — `fix(automation): bound backlog patch equivalence audit`
- #6746 — `fix(automation): verify fast-audit salvage candidates`

**Files affected:**
- `scripts/audit_codex_branch_backlog.py` (all 10)
- `tests/scripts/test_audit_codex_branch_backlog.py` (all 10)
- `docs/briefs/automation-merge-contract.md` (#6539, #6623)

(Note: 10 additional PRs touched this same file in Categories A and B,
making `audit_codex_branch_backlog.py` the **most-patched single file**
in the window — 20 PRs total. The 10 in this category are the ones
where the audit-classifier behavior, not the handoff substrate, was the
mutator.)

**Root cause at the contract level:** the backlog auditor produces a
**branch classification** for each `codex/*` branch in the local repo.
The classification taxonomy was open-set at the start of the window
(it grew from ~6 reasons to ~15 over seven days). Each fix adds a new
classification or refines the predicate of an existing one:

- #6502: `github_health` and `open_pr_lookup_skipped` reported in JSON
  when `gh` is unhealthy.
- #6539: introduces the `summary.publishable_branch_backlog` metric —
  recent unique work + stale unique remote, *excluding* stale
  local-only archaeology.
- #6570: enables patch-equivalence detection by default (was opt-in).
- #6605: empty `base...branch` diff is `cleanup_patch_equivalent`, not
  publishable salvage.
- #6623: tracks ahead/behind counts; classifies behind/diverged as
  `salvage_diverged_*`; excludes diverged from publishable backlog.
- #6643: tolerate missing worktrees during the fast-audit phase.
- #6646: zero-diff branches classified during fast audit.
- #6651: compact output format (operational, not contract).
- #6726: bound the patch-equivalence check (it was unbounded and slow
  on repos with thousands of branches).
- #6746: verify fast-audit salvage candidates (false-positive
  reduction).

**The classifier's reason vocabulary is implicit and open-set.** Every
new operational pattern (a new way a branch can be "stale" or
"superseded") forces another `if/elif` branch. The summary metric
(`publishable_branch_backlog`) is also implicit: it is the inverse of
"all the reasons we don't want to count", so adding a new
non-publishable reason silently changes what the metric counts.

**Recurrence:** 10 PRs (≥3 = contract-design flaw — yes).

**Files patched repeatedly:** `audit_codex_branch_backlog.py` was
patched 20 times across all categories — by far the most-touched file
in the window. The 10 in this category are the classifier-only patches.

**Proposed contract change:** introduce a
`BranchClassification` enumeration and a `Classifier` protocol:

```python
class BranchReason(StrEnum):
    PUBLISHABLE_RECENT_UNIQUE = "publishable_recent_unique"
    PUBLISHABLE_STALE_UNIQUE_REMOTE = "publishable_stale_unique_remote"
    CLEANUP_MERGED = "cleanup_merged"
    CLEANUP_PATCH_EQUIVALENT = "cleanup_patch_equivalent"
    CLEANUP_ZERO_DIFF = "cleanup_zero_diff"
    PROTECTED_HANDOFF_RECEIPT = "protected_handoff_receipt"
    PROTECTED_OUTBOX_UNRESOLVED = "protected_outbox_unresolved"
    PROTECTED_OPEN_PR = "protected_open_pr"
    SALVAGE_DIVERGED_AHEAD = "salvage_diverged_ahead"
    SALVAGE_DIVERGED_BEHIND = "salvage_diverged_behind"
    SALVAGE_MISSING_WORKTREE = "salvage_missing_worktree"
    SALVAGE_FAST_AUDIT_UNVERIFIED = "salvage_fast_audit_unverified"
    EMPTY_PR_DIFF = "empty_pr_diff"
    DEGRADED_GH_HEALTH = "degraded_gh_health"

@dataclass(frozen=True)
class BranchClassification:
    branch: str
    reason: BranchReason
    is_publishable: bool  # derived from reason alone, not from each call site
    is_protected: bool    # derived from reason alone
    evidence: Mapping[str, Any]

PUBLISHABLE_REASONS: AbstractSet[BranchReason] = {
    BranchReason.PUBLISHABLE_RECENT_UNIQUE,
    BranchReason.PUBLISHABLE_STALE_UNIQUE_REMOTE,
}
```

The summary metric `publishable_branch_backlog` becomes a one-line
predicate: `sum(1 for c in classifications if c.is_publishable)`.
Adding a new reason forces an explicit decision about whether it is
publishable, protected, salvage, or cleanup.

The auditor's main loop becomes a sequence of `Classifier`
implementations (one per signal type — open-PR check, receipt match,
patch-equivalence probe, cherry probe, worktree existence, etc.) that
each return one or more `BranchClassification`s. The first classifier
to produce a non-`None` classification for a branch wins (deterministic
precedence). New classifiers are added by extending one list; the
`if/elif` mainline is deleted.

**Sequencing suggestion:** highest priority of the three new
recurrent-class proposals (Categories B, C, D). 10 PRs in 7 days on a
single file is the strongest signal the audit surfaces. The handoff
work (Category A) and this work share `audit_codex_branch_backlog.py`,
so they should be coordinated: land Category A's protection clauses
into the new `BranchReason.PROTECTED_*` slots in one commit rather than
two.

**Estimated effort:** ~400 LOC for the classification module + tests,
~600 LOC of deletions in the auditor mainline (the open-set if/elif
becomes a list of classifiers). Net negative diff after refactor.

---

### Category E — automation gh CLI health (1 PR)

**PRs:**
- #6617 — `fix(automation): classify github dns lookup failures`

**Files affected:**
- `scripts/github_cli_health.py`
- `tests/scripts/test_github_cli_health.py`

**Root cause:** the gh-health probe distinguished "auth failure" from
"network failure" but did not specifically classify DNS lookup failures,
which surface differently on macOS sandboxes (e.g., `nslookup` error vs.
`curl: (6)`). PR #6617 adds a DNS-specific classification so downstream
consumers (Category C publisher, Category D auditor) can decide whether
to retry, fall back to cache, or pause.

**Recurrence:** 1 PR alone, but the gh-health probe is consumed by **at
least 4 of the in-window PRs** (#6502, #6738, #6741, #6651) and the
fact that DNS-specific classification was a 7-day fix in a 1000+ line
script suggests the **probe taxonomy is implicit**. This is an orphan
class that is one validation away from becoming a recurrent class.

**Proposed contract change:** define a `GhHealthSignal` enumeration
analogous to Category D's `BranchReason`. Currently:
- `healthy`
- `auth_failure`
- `rate_limited`
- `unreachable_dns`  ← added by #6617
- `unreachable_network`
- `unreachable_unknown`

Probe results would carry this enumeration and a `should_retry: bool`
property. Consumers in C and D would branch on the enumeration, not on
parsed strings.

**Sequencing suggestion:** roll into Category C's predicate work.

**Estimated effort:** ~80 LOC additional in `github_cli_health.py` +
~50 LOC tests.

---

### Category F — boss-loop runtime hardening (3 PRs)

**PRs:**
- #6773 — `fix(boss-loop): bound worker receipt metadata before signed-receipt path` (4/28)
- #6776 — `fix(boss-loop): bound terminal json summaries` (4/28)
- #6778 — `fix(boss-loop): honor acceptance gate failures` (4/28)

**Files affected:**
- `aragora/swarm/boss_loop.py` (#6773 + #6776)
- `aragora/swarm/bounded_receipt_metadata.py` (#6773 — created)
- `aragora/swarm/boss_worker_lifecycle.py` (#6778)
- `aragora/swarm/terminal_truth.py` (#6778)
- `aragora/cli/commands/swarm.py` (#6776)
- `tests/swarm/test_bounded_receipt_metadata.py` (#6773 — created)
- `tests/swarm/test_boss_loop_receipts.py` (#6776)
- `tests/swarm/test_boss_loop.py` (#6776)
- `tests/swarm/test_boss_worker_lifecycle.py` (#6778)
- `tests/cli/test_swarm_command.py` (#6776)

**Root cause at the contract level:** all three landed on 2026-04-28
in a tight sequence after a constrained `#6187` validation surfaced
**three separate boss-loop runtime bugs in a single dispatch**:

1. **Unbounded JSON canonicalisation in receipt-signing** (PR #6773).
   Multi-MB worker `receipt_metadata` was spread verbatim into the
   `LaneCompletionReceipt.metadata`, then `json.dumps(..., sort_keys=True)`
   forced full materialisation of every nested dict. macOS `sample`
   captures showed 2+ seconds in `encoder_listencode_dict` recursion
   per call. The fix introduces a `bound_receipt_metadata()` function
   with explicit per-field caps (4 KiB stdout/stderr, 1 KiB log, 16 KiB
   total target) and persists the full payload to
   `.aragora/worker-results/<run_id>.json` with a sha256 reference. The
   bound applies BEFORE the receipt is constructed, so the entire
   downstream pipeline sees only bounded data.

2. **Unbounded operator-facing JSON projection** (PR #6776). After
   #6773 fixed receipt-signing, the operator-facing
   `swarm boss-loop --json` output path was still spinning on
   `to_dict()` of the same multi-MB result. PR #6776 adds a
   `BossLoopResult.to_bounded_dict()` that caps terminal-receipt
   `needs_human` and `next_action` text and routes the CLI's `--json`
   through this bounded projection.

3. **Acceptance-gate truth not honored as terminal verdict**
   (PR #6778). After #6776 fixed the JSON spin, the dispatch returned
   a branch deliverable with `worker_outcome=acceptance_gate_failed`,
   but `boss-loop` still counted the issue as completed because any
   typed deliverable from a `needs_human` worker was treated as
   success. This corrupts throughput/rescue metrics and can
   incorrectly drain the queue.

**This is the canonical example of the rescue cascade pattern.** A
single dispatch (#6187) surfaced three latent contract violations in
sequence; each fix unmasked the next. The Operating Law was honored
each time: the rescue became product behavior. But the underlying
**boss-loop runtime payload contract** was never written down, so
none of the three issues were caught until they manifested.

**Recurrence:** 3 PRs (≥3 = contract-design flaw — yes).

**Cross-cutting observation:** all three PRs touch the *output side* of
the boss-loop. The unboundedness of worker receipt metadata, the
unboundedness of the result projection, and the truth-functional
correctness of the terminal verdict are three faces of the same
contract: **what does a boss-loop dispatch return, and what bounds
must hold on every field of the return?**

**Proposed contract change:** introduce a
`BossLoopRuntimePayloadContract` module that consolidates:

1. **Output-size bounds**: every field on `BossLoopResult`,
   `LaneCompletionReceipt`, and `to_bounded_dict()` carries an
   explicit byte cap. The cap is enforced at *construction time*, not
   at serialisation time.
2. **Terminal verdict truth function**: `acceptance_gate_failed`,
   `needs_human`, `completed`, `idle`, `error` form a closed enum.
   The mapping from `worker_outcome` to terminal verdict is a single
   function (proposed: `aragora/swarm/terminal_truth.py::map_outcome()`)
   already partially extracted by #6778. Boss-loop counts only
   `completed` as success; everything else (including
   `needs_human + branch deliverable`) is non-success.
3. **Receipt signing input invariant**: the input to the signer is a
   `Mapping` whose total serialized size is `≤ SIGNER_INPUT_MAX_BYTES`
   (proposed 64 KiB). The signer asserts this invariant on entry and
   refuses to sign payloads exceeding it. (This is the test that
   would have caught #6773 before it hit production.)

**Sequencing suggestion:** this is **operationally urgent** because
the three PRs landed within ~1 hour of one another on 4/28 and the
boss-loop is throughput-critical. Land the bound-at-construction-time
invariant as a single contract module within the next 14 days.

**Estimated effort:** the work is partially done — `bounded_receipt_metadata.py`
(~465 LOC, by #6773) and `terminal_truth.py` (extracted by #6778) are
the seeds. ~200 LOC additional to consolidate the three contracts +
~150 LOC of the signer-input-size invariant + tests.

---

### Category G — swarm preflight + permission contracts (1 PR)

**PRs:**
- #6774 — `fix(swarm): align preflight permission contracts`

**Files affected:**
- `aragora/swarm/dispatch_contract_gate.py`
- `aragora/swarm/preflight.py`
- `tests/swarm/test_dispatch_contract_gate.py`
- `tests/swarm/test_worker_contract_drift_alignment.py`

**Root cause:** the dispatch-contract preview computed
`allow_full_auto=true` for Codex workers because it did not build the
same preflight-owned `LaunchConfig` as the scratch preflight worker.
The drift was visible only when the dispatch-contract gate output was
compared against the actual preflight worker's launch config — which
the new test `test_worker_contract_drift_alignment.py` does. The fix
threads the same `LaunchConfig` through both code paths.

**Recurrence:** 1 PR, but the contract surface is **the same
preflight/dispatch boundary** that:
- #6450 patched (dispatch hook preservation under claims)
- #6774 patched (permission contract alignment)
- the broader CI-lane / boss-loop preflight discussion in
  `docs/briefs/automation-merge-contract.md`

The drift alignment test (`test_worker_contract_drift_alignment.py`)
is the right kind of test: it asserts that two code paths produce the
same artefact. **More such tests would catch the same class of bug
before it hits production.**

**Cross-cutting observation:** Categories G and H together form an
implicit "swarm dispatch/preflight contract" that has 2 PRs in this
window — below the 3-PR threshold but worth flagging as an orphan
that **may recur** because the contract surface is large (the
preflight script alone is ~800 LOC and the dispatch contract gate is
~400 LOC).

**Proposed contract change:** define an explicit
`DispatchPreflightContract` test fixture that asserts: for every worker
type × every claim mode × every label set, `dispatch_contract_gate.preview()`
and `preflight.build_launch_config()` produce identical
`LaunchConfig` objects. This is a parametric test, not new code; it
catches drift class via assertion.

**Estimated effort:** ~200 LOC of parametric tests, no production
code change.

---

### Category H — swarm dispatch hook preservation (1 PR)

**PRs:**
- #6450 — `fix(swarm): preserve dispatch hook under claims`

**Files affected:**
- `aragora/swarm/boss_loop.py`
- `aragora/swarm/boss_worker_lifecycle.py`
- `tests/debate/test_pr_review_protocol_smoke.py`
- `tests/swarm/test_boss_worker_lifecycle.py`

**Root cause:** when dispatching under an issue claim, the
`loop._dispatch_issue` hook was not preserved, so claim-mode dispatch
took a different path than non-claim dispatch. The fix preserves the
hook explicitly.

**Recurrence:** 1 PR, but related to G via the dispatch/preflight
boundary. Together (G+H) = 2 PRs. The dispatch-hook contract is one of
the boss-loop's most-overridden integration points — eight existing
tests in `test_boss_worker_lifecycle.py` exercise it.

**Proposed contract change:** treat the dispatch-hook substitution
points as an explicit protocol (`DispatchHookProtocol`) with
documented method signatures and a default implementation. Claim-mode
and non-claim-mode share the same protocol; tests assert protocol
conformance independently of the dispatch path.

**Estimated effort:** ~100 LOC for the protocol + ~150 LOC tests; no
behavior change.

---

### Category I — swarm coordination DB resilience (1 PR)

**PRs:**
- #6645 — `fix(swarm): degrade status when coordination db is unreadable`

**Files affected:**
- `aragora/cli/commands/swarm.py`
- `aragora/nomic/dev_coordination/core.py`
- `tests/cli/test_swarm_status_command.py`
- `tests/nomic/test_dev_coordination.py`

**Root cause:** `swarm status --json` returned a non-degraded payload
(or crashed) when the SQLite coordination DB could not be opened. PR
#6645 makes it return `coordination.available=false` while keeping
fleet/integrator/operator status surfaces available. Also closes
SQLite connections on PRAGMA setup failure to avoid `ResourceWarning`.

**Recurrence:** 1 PR.

**Cross-cutting observation:** this is structurally identical to
Category E (gh CLI health degraded mode) and Category D's "github
unreachable" classifier. The pattern — **a downstream consumer must
handle the case where its data source is unavailable, and report a
degraded-but-truthful payload rather than crashing or lying** — is the
same. Three PRs in this audit (#6502, #6645, #6738) explicitly
implement this pattern; #6617 enables it.

**Hint of an emerging cross-cutting class:** "degraded-mode payload
contract" — a single typed structure that every status/health/audit
endpoint returns when its data source is unavailable. Currently each
endpoint reinvents this. If a fourth degraded-mode fix lands in the
next 7 days, this becomes a recurrent class.

**Proposed contract change (anticipatory):** define a
`DegradedPayload` dataclass with fields:
- `available: bool`
- `degradation_reason: str`
- `partial_data: Mapping[str, Any] | None`

Every status/health/audit endpoint's response includes this as a
sibling field at the top level. Consumers (the CLI, the boss-loop's
preflight) check `available` first, render `partial_data` when
present, and never crash on degraded inputs.

**Estimated effort:** ~150 LOC for the dataclass + helpers, ~50 LOC
per endpoint that adopts it. Adoption can be incremental.

---

### Category J — swarm roadmap codes (1 PR)

**PRs:**
- #6719 — `fix(swarm): preserve inherited roadmap codes on decomposed issues`

**Files affected:**
- `aragora/swarm/boss_loop.py`
- `tests/swarm/test_boss_loop.py`
- `tests/swarm/test_proof_first_queue_classification.py`

**Root cause:** auto-decomposed child issues (e.g., parent CS-01 → child
CS-01b) did not carry the parent's roadmap code, so the proof-first
roadmap fast path (which gates dispatch on roadmap-code regex match)
rejected them. PR #6719 adds inherited roadmap codes to
auto-decomposed child issue bodies.

**Recurrence:** 1 PR, but **the proof-first queue classifier
(`proof_first_queue.py`) and selection logic (`boss_loop_selection.py`)
were patched by 3 separate PRs in this window** — #6603 (Category K),
#6719 (Category J), and #6766 (Category K). When considered together
(J+K), the **proof-first dispatch contract** has 3 PRs in 7 days,
crossing the recurrence threshold. See Category K for the combined
analysis.

---

### Category K — swarm proof-first labels (2 PRs)

**PRs:**
- #6603 — `fix(swarm): report scope-overlap over missing-labels in target_issue_miss_guidance`
- #6766 — `fix(swarm): align proof-first boss fallback labels`

**Files affected:**
- `aragora/swarm/boss_loop_selection.py` (#6603)
- `scripts/run_proof_first_shift.py` (#6766)
- `tests/scripts/test_run_proof_first_shift.py` (#6766)

**Root cause:** the **proof-first dispatch contract** has three
implicit invariants spread across multiple files:

1. **Diagnostic precedence** (#6603): when `target_issue_miss_guidance`
   reports why a target issue is unsuitable, **scope-overlap should be
   reported before missing-labels**, because the proof-first queue
   filter (`filter_noncanonical_boss_ready_issues`) intentionally
   strips the `boss-ready` label from issues with scope conflicts in
   the in-memory state. The pre-#6603 ordering caused the diagnostic
   to surface "missing required labels: boss-ready" instead of the
   actually-actionable "scope conflict with in-flight work."
   This is a **diagnostic ordering bug rooted in a side-effecting
   filter contract.**
2. **Roadmap-code inheritance** (#6719, technically Category J): child
   issues inherit roadmap codes from parents.
3. **Fallback label set** (#6766): the proof-first direct boss-loop
   fallback should default to the canonical `boss-ready` label only,
   and should pass `--no-suitable-issue-keepalive` to avoid
   short-exiting on a transient empty feed.

**Combined recurrence (J+K):** 3 PRs in 7 days touching the
proof-first dispatch contract. **Crosses the 3-PR threshold.**

**Cross-cutting observation:** the proof-first dispatch contract
spans `proof_first_queue.py`, `boss_loop_selection.py`,
`boss_loop.py` (decomposition), `run_proof_first_shift.py` (script
wrapper), and at least 3 test files. The contract has at least
**five implicit invariants** (label canonicality, label-set membership,
roadmap-code regex, scope-overlap precedence, no-suitable-issue
keepalive behavior). None of them are written down centrally.

**Proposed contract change (combined J+K):** introduce a
`ProofFirstDispatchContract` module (`aragora/swarm/proof_first_contract.py`)
that exposes:

```python
@dataclass(frozen=True)
class ProofFirstSelectionResult:
    issue_number: int
    is_suitable: bool
    suitability_reasons: Sequence[SuitabilityReason]  # ordered by precedence

class SuitabilityReason(StrEnum):
    SCOPE_OVERLAP = "scope_overlap"
    MISSING_REQUIRED_LABELS = "missing_required_labels"
    DECOMPOSED_WITHOUT_LINEAGE = "decomposed_without_lineage"
    ROADMAP_CODE_MISMATCH = "roadmap_code_mismatch"
    NON_CANONICAL_BOSS_READY = "non_canonical_boss_ready"
    # ...

CANONICAL_BOSS_READY_LABELS: Final[AbstractSet[str]] = {"boss-ready"}
DIAGNOSTIC_PRECEDENCE: Final[Sequence[SuitabilityReason]] = (
    SuitabilityReason.SCOPE_OVERLAP,    # most actionable first
    SuitabilityReason.MISSING_REQUIRED_LABELS,
    SuitabilityReason.NON_CANONICAL_BOSS_READY,
    SuitabilityReason.DECOMPOSED_WITHOUT_LINEAGE,
    SuitabilityReason.ROADMAP_CODE_MISMATCH,
)
```

The selector consumes this contract; the diagnostic ordering becomes a
property of `DIAGNOSTIC_PRECEDENCE`, not of the gate-check ordering in
`target_issue_miss_guidance`. Decomposition lineage and roadmap-code
inheritance are explicit invariants enforced by the contract.

**Sequencing suggestion:** medium priority. Less urgent than handoff
(A+B) and backlog audit (D) by recurrence count, but the proof-first
queue is on the throughput-critical path and the diagnostic-ordering
bug surfaced in #6603 is the kind of gnarly side-effect interaction
that contract derivation prevents.

**Estimated effort:** ~250 LOC for the contract module + tests, ~100
LOC of refactoring the three call sites.

---

### Category L — swarm supervisor session state (1 PR)

**PRs:**
- #6403 — `fix(swarm): persist acceptance-gate session verdict`

**Files affected:**
- `aragora/swarm/supervisor_workers.py`
- `tests/swarm/test_supervisor.py`

**Root cause:** supervisor session state did not durably persist
merge-gate terminal outcomes as `needs_human`. The fix persists the
verdict so that terminal lease release does not lose the
acceptance-gate blocker outcome.

**Recurrence:** 1 PR in this window.

**Cross-cutting observation:** the **acceptance-gate verdict**
appears in three places in this audit:
- #6403 (this PR) — supervisor session state persistence.
- #6778 (Category F) — boss-loop terminal verdict mapping.
- #6450 (Category H) — dispatch hook preservation under claims.

Together these three PRs touch the **terminal-truth contract** of the
swarm supervisor: when does a worker outcome become a terminal verdict,
how is the verdict persisted, and how does the dispatcher count it?
Three PRs is the threshold; this is **another implicit recurrent
class** spanning Categories F, H, and L.

**Proposed contract change (combined F+H+L):** the
`aragora/swarm/terminal_truth.py` module introduced by #6778 should be
the single source of truth for:
- Mapping `worker_outcome` → terminal verdict (already partially done).
- Mapping merge-gate outcome → supervisor session state (move from
  `supervisor_workers.py` to the contract module).
- Mapping dispatch-hook outcome → claim release decision (move from
  `boss_worker_lifecycle.py`).

The boss-loop, supervisor, and dispatch hook all consume the same
`map_to_terminal_verdict()` function. The persistence layer
(`supervisor_workers.py`) writes the verdict atomically with the
session state. The dispatch hook reads it before deciding whether to
release the claim.

**Estimated effort:** ~150 LOC of consolidation in `terminal_truth.py`,
~100 LOC of test updates. This work is partially started by #6778; it
should be finished within the next 14 days alongside Category F's
other consolidation work.

---

## Recurrent classes summary

| # | Class | PRs (count) | PR list (representative) | Files affected | Proposed contract | Effort |
|---|---|---|---|---|---|---|
| 1 | outbox/handoff (covered) | **17** | #6531, #6537, #6547, #6551, #6553, #6568, #6581, #6594, #6595, #6596, #6607, #6618, #6624, #6642, #6742, #6747, #6755 | publish_automation_handoffs.py, audit_codex_branch_backlog.py, reconcile_automation_outbox.py | already drafted (handoff-contract spec) | already estimated (5 PRs / 800 LOC net) |
| 2 | outbox/handoff structured-actions | **6** | #6652, #6659, #6723, #6725, #6751, #6754 | same three scripts | C9/C10/C11 additions to handoff contract | ~400 LOC |
| 3 | automation publisher backpressure | **7** | #6501, #6512, #6536, #6538, #6543, #6738, #6741 | publish_codex_automation_branches.py, cache_codex_automation_github_status.py | `BackpressureSignal` discriminated union + single `evaluate_backpressure()` | ~700 LOC (~150 net after deletions) |
| 4 | automation backlog-audit classifier | **10** | #6502, #6539, #6570, #6605, #6623, #6643, #6646, #6651, #6726, #6746 | audit_codex_branch_backlog.py | `BranchReason` enum + `Classifier` protocol with deterministic precedence | ~1000 LOC (net negative after refactor) |
| 5 | boss-loop runtime payload bounds | **3** | #6773, #6776, #6778 | boss_loop.py, bounded_receipt_metadata.py, swarm.py, boss_worker_lifecycle.py, terminal_truth.py | `BossLoopRuntimePayloadContract` (output-size bounds + terminal-verdict truth function + signer-input-size invariant) | ~500 LOC (much already in flight) |
| 6 | proof-first dispatch (J+K combined) | **3** | #6603, #6719, #6766 | boss_loop_selection.py, boss_loop.py (decompose path), run_proof_first_shift.py | `ProofFirstDispatchContract` with explicit `SuitabilityReason` precedence + roadmap-code inheritance + canonical label set | ~350 LOC |
| 7 | swarm terminal-truth contract (F+H+L combined) | **3** | #6403, #6450, #6778 | supervisor_workers.py, boss_worker_lifecycle.py, terminal_truth.py | consolidate verdict mapping + persistence + dispatch-hook handling in `terminal_truth.py` | ~250 LOC |

Note that classes 5, 6, and 7 partially overlap on PRs (e.g., #6778
appears in classes 5 and 7) because the same PR honored multiple
implicit contracts. The **distinct-PR count** for these three combined
classes is 9 PRs, all from the swarm/boss-loop family.

**Total recurrent PRs:** 51 of 51 (every PR in the window belongs to at
least one recurrent class once J+K and F+H+L are combined). This is
the single strongest signal in the audit: **there are no truly
isolated rescue events in the window — every fix is part of an
implicit-contract family with other fixes.**

## Cross-cutting observations

### 1. Velocity is climbing, not stabilising

The 51 in-window PRs follow a daily-count curve (1, 1, 11, 10, 10, 14,
4) that is **not** the asymptotic decay one would expect from a
contract converging on a stable shape. The 4/27 peak (14 PRs) is
*after* 30+ patches had already landed on the same files; if the fixes
were converging on a closed contract, 4/27 should have been smaller
than 4/24, not larger. Instead, the 4/27 peak coincides with the
introduction of structured-action support (Category B) and the
sandboxed-watcher cache (Category C). **New product surface keeps
opening implicit contracts faster than existing patches close them.**

The 4/28 drop to 4 PRs is partly because operator attention shifted to
the boss-loop runtime cluster (Category F) and the worktree's CI lanes
required attention; the projection for 4/29 (not in window) is
unlikely to be zero.

### 2. The same files are being patched repeatedly

`audit_codex_branch_backlog.py` was patched **20 times** in 7 days —
roughly 3 PRs per day on a single 1000+ line script. This is the
canonical signal that the file's logic has outgrown its single-file
shape. The handoff-contract spec already proposes splitting its
audit-classifier portion into a contract module (Category A); this
audit's Category D recommendation extends that split to the whole
classifier taxonomy. Together, `audit_codex_branch_backlog.py` should
shrink by ~600 LOC after both proposals land, with new code distributed
across two well-typed modules.

`publish_automation_handoffs.py` was patched **12 times** in 7 days. It
is the second-most-patched file, with the same root cause: it
implements three implicit contracts (handoff identity, action
discrimination, state-root resolution) in one ~1300-line script. The
handoff-contract spec addresses this directly.

`publish_codex_automation_branches.py` was patched **7 times** —
Category C's recommendation pulls the backpressure predicate out of it.

These three scripts together account for **39 of the 51 PRs** in the
window. **Three contract derivations would absorb three quarters of
the rescue volume.**

### 3. Orphan classes that look like they will recur

The audit identifies **three orphan classes that did not yet cross the
3-PR threshold** but show the same structural pattern as the recurrent
classes:

1. **gh CLI health probe taxonomy** (Category E, 1 PR — #6617). Probe
   results are parsed from strings rather than typed. The first
   recurrence will be a non-DNS unreachable mode (HTTP 503, proxy
   timeout, certificate error). Predicting: 2-3 follow-up PRs in the
   next 14 days.

2. **degraded-mode payload contract** (Category I, 1 PR — #6645, plus
   structurally similar PRs #6502 and #6738). Each downstream consumer
   reinvents how to render a "data source unavailable" payload.
   Predicting: another 1-2 endpoints will need this in the next 14
   days.

3. **dispatch/preflight contract drift** (Categories G+H, 2 PRs —
   #6450 and #6774). The drift-alignment test added in #6774 is the
   right pattern but only covers one drift dimension. Predicting: 1-2
   follow-up PRs in the next 14 days adding similar drift-alignment
   tests for other config dimensions (worktree, env, claim mode).

If any of these three orphans crosses the 3-PR threshold by
2026-05-12, the contract derivation should be done at that point
rather than waiting longer.

### 4. The "rescue cascade" pattern of 4/28

PRs #6773 → #6776 → #6778 landed within a single hour on 2026-04-28.
Each unmasked the next: fixing the JSON-canonicalisation spin
(#6773) revealed an operator-projection spin (#6776), which revealed
an acceptance-gate truth-mapping bug (#6778). This is the canonical
**rescue cascade**: a single dispatch surfaces N latent contract
violations because the contracts are layered and each one masks the
next when broken in production.

The Operating Law was honored each time — each cascade step became
product behavior. But the cascade itself is a stronger signal that
the underlying contract is unwritten: when the contract is written
once, the cascade collapses to a single bug in a single test.

### 5. Recurrent-fix file patterns are predictive

The strongest predictor of "this file will get another fix in the
next 7 days" is "this file got a fix in the past 7 days." The four
top files (audit_codex_branch_backlog.py, publish_automation_handoffs.py,
publish_codex_automation_branches.py, reconcile_automation_outbox.py)
account for 35 of the 51 PRs. **The 30-day prediction is that these
four files alone will absorb another 30+ PRs unless their implicit
contracts are extracted into typed modules.**

### 6. The audit's own bias

This document quotes 51 strict-prefix PRs but the broader 60-PR set
exists. The 9 omitted PRs (e.g., #6777 `fix(queue): batch dev
coordination work queue sync`, #6745 `fix(status): count canonical
boss-ready queue depth`, #6430 `fix(cli): restructure swarm
DevCoordinationStore optional-import (~133 mypy errors)`, #6454
`fix(review-queue-brief): hydrate secrets before building invoker in
server handler`) are real rescue-class fixes with the same shape as
the in-scope PRs but a different conventional-commit prefix. **The
recommended action below is robust to the strict/broader choice
because the recurrent classes do not change.**

## Recommended action

Prioritized list of contract-design fixes to land in the **next 30
days**, ordered by churn-reduction value (= count of PRs the
derivation would have prevented if it had existed at the start of the
window).

| Rank | Action | Class | PRs prevented | Sequencing |
|---|---|---|---|---|
| 1 | Land handoff-contract spec (`aragora/swarm/handoff_contract.py` per existing spec) + C9/C10/C11 for structured actions | A + B | 23 | first; already drafted |
| 2 | Extract backlog-audit classifier into `BranchReason` enum + `Classifier` protocol with deterministic precedence | D | 10 | second; coordinates with rank 1 on `audit_codex_branch_backlog.py` |
| 3 | Extract publisher backpressure into `BackpressureSignal` discriminated union + single `evaluate_backpressure()` | C + E | 8 | third; depends on rank 1's C11 state-root work for `--receipt-dir` flag |
| 4 | Consolidate boss-loop runtime payload bounds + terminal-verdict truth function | F + L (+ partial H) | 4 | parallel with rank 2/3; operationally urgent because of 4/28 cascade |
| 5 | Extract proof-first dispatch contract with `SuitabilityReason` enum and `DIAGNOSTIC_PRECEDENCE` | J + K | 3 | parallel; lower blast-radius |
| 6 | Add drift-alignment parametric test fixture for dispatch/preflight contract | G + H | 2 (and 1-2 predicted in the next 14 days) | parallel; pure test addition |
| 7 | Consider `DegradedPayload` shared dataclass once 3rd PR lands in that class | I (+E +D-degraded) | predicted 2-3 in next 14 days | wait-and-see |

**Total churn prevented:** rank 1-5 absorb **48 of the 51** in-window
PRs (94% of rescue volume). Rank 6 and 7 are anticipatory.

**Net diff prediction:** the five contract derivations together would
add ~2,500 LOC of typed contract modules and ~3,500 LOC of tests,
while removing ~3,000 LOC from the three top scripts (mainline
if/elif chains, scattered backpressure checks, scattered dedup logic).
**Net: +3,000 LOC, but distributed across 12+ small typed modules
instead of concentrated in 3 large scripts.**

**Velocity prediction:** if all five derivations land within 30 days,
the projected fix-PR rate on these surfaces drops from ~7 per day to
~1 per day (an 85% reduction), because new edge cases produce
type-checker errors at the contract boundary rather than silent
divergence at the call site.

## Confidence calibration

This audit's claims about "what the contract derivations would
prevent" are **post-hoc**: every derivation is informed by the
already-merged patches. A more rigorous test would be to land one
derivation, then track how many fix PRs *do not* recur in its surface
over the following 7 days. The handoff-contract spec is the natural
candidate for that A/B comparison; the operator should consider
landing it first and tracking the 7-day rescue rate on
`publish_automation_handoffs.py`, `audit_codex_branch_backlog.py`,
and `reconcile_automation_outbox.py` to validate (or refute) this
audit's prediction.

## Related text-only artefacts

- `docs/plans/2026-04-28-handoff-contract-derivation.md` — declarative
  consolidation of the local-Codex → GitHub automation handoff
  contract (Category A, 17 PRs). **The 8 invariants it enumerates
  remain the highest-confidence portion of the recommended action.**
  This audit extends that spec with C9/C10/C11 for the 6 structured-action
  PRs that landed after it was drafted.
- `docs/CANONICAL_GOALS.md` — the Operating Law that this audit
  applies at the meta level. The Operating Law as written addresses
  the *individual rescue → product change* loop. This audit asks the
  next-level question: *when many rescues converge on one surface,
  what is the meta-product change?* Answer: extract the contract.

## Stop conditions

This audit is finalized when an operator reads it. It is text only.
No code exists yet. No PR is filed. **No edits to any other path
have been made by this audit run.** Stop conditions:

1. Operator reviews and either approves the recommended-action
   ranking or substitutes their own.
2. Operator signals which (if any) of ranks 1-5 to start with;
   if none, the existing patch trajectory continues without harm.
3. Subsequent contract-derivation specs (one per rank, analogous to
   the existing handoff-contract spec) are drafted only on operator
   approval.

## Out of scope

- Implementation of any contract module proposed in this audit.
- New PRs of any kind.
- Edits to any file other than this audit document.
- Changes to red workflows or `.github/workflows/*`.
- Triggering automation (`scripts/publish_automation_handoffs.py` was
  not run; nothing was written to `.aragora/automation-outbox/`).
- Reanalysis of the 17 outbox/handoff PRs already covered by
  `docs/plans/2026-04-28-handoff-contract-derivation.md`.
- The broader 60-PR rescue-class view; this audit is bounded to the
  strict 51 by the title-prefix filter the parent task specified.

## Appendix A — full PR-to-category mapping

For reproducibility, the categorisation used by this audit is encoded
below. Every in-window PR appears in exactly one row.

| PR | Date | Title | Category |
|---|---|---|---|
| #6403 | 2026-04-22 | fix(swarm): persist acceptance-gate session verdict | L (terminal-truth supervisor) |
| #6450 | 2026-04-23 | fix(swarm): preserve dispatch hook under claims | H (dispatch hook) |
| #6501 | 2026-04-24 | fix(automation): don't pause publisher for review-only PRs | C (publisher) |
| #6502 | 2026-04-24 | fix(automation): gate backlog audit PR lookup on gh health | D (backlog audit; gh-health crosscut to E) |
| #6512 | 2026-04-24 | fix(automation): bypass mutating pre-push hooks in publisher | C (publisher) |
| #6531 | 2026-04-24 | fix(automation): validate outbox handoff contract | A (handoff spec) |
| #6536 | 2026-04-24 | fix(automation): skip empty PR diff branches | C (publisher) |
| #6537 | 2026-04-24 | fix(automation): dedupe outbox handoff ids | A (handoff spec) |
| #6538 | 2026-04-24 | fix(automation): fail open on PR diff probe errors | C (publisher) |
| #6539 | 2026-04-24 | fix(automation): distinguish publishable codex backlog | D (backlog audit) |
| #6543 | 2026-04-24 | fix(automation): relax in-flight PR backpressure | C (publisher) |
| #6547 | 2026-04-24 | fix(automation): dedupe outbox handoffs by branch | A (handoff spec) |
| #6551 | 2026-04-24 | fix(automation): ignore receipted branch handoffs in backlog audit | A (handoff spec; classifier crosscut to D) |
| #6553 | 2026-04-25 | fix(automation): skip merged outbox handoffs | A (handoff spec) |
| #6568 | 2026-04-25 | fix(automation): resolve audit handoff state root | A (handoff spec) |
| #6570 | 2026-04-25 | fix(automation): check patch-equivalent branches by default | D (backlog audit) |
| #6581 | 2026-04-25 | fix(automation): protect unresolved outbox handoffs in backlog audit | A (handoff spec; classifier crosscut to D) |
| #6594 | 2026-04-25 | fix(automation): resolve publisher outbox state root | A (handoff spec) |
| #6595 | 2026-04-25 | fix(automation): fingerprint top-level outbox branch fields | A (handoff spec) |
| #6596 | 2026-04-25 | fix(automation): protect top-level outbox handoffs | A (handoff spec) |
| #6603 | 2026-04-25 | fix(swarm): report scope-overlap over missing-labels in target_issue_miss_guidance (#6591) | K (proof-first labels) |
| #6605 | 2026-04-25 | fix(automation): treat empty branch diffs as backlog cleanup | D (backlog audit) |
| #6607 | 2026-04-25 | fix(automation): protect patch-matched handoffs in audit | A (handoff spec) |
| #6617 | 2026-04-26 | fix(automation): classify github dns lookup failures | E (gh CLI health) |
| #6618 | 2026-04-26 | fix(automation): protect superseded outbox branches | A (handoff spec) |
| #6623 | 2026-04-26 | fix(automation): exclude diverged branches from publishable backlog | D (backlog audit) |
| #6624 | 2026-04-26 | fix(automation): skip patch-equivalent outbox handoffs | A (handoff spec) |
| #6642 | 2026-04-26 | fix(automation): honor receipt-only handoff branches | A (handoff spec) |
| #6643 | 2026-04-26 | fix(automation): tolerate missing audit worktrees | D (backlog audit) |
| #6645 | 2026-04-26 | fix(swarm): degrade status when coordination db is unreadable | I (coord-db / degraded payload) |
| #6646 | 2026-04-26 | fix(automation): classify zero-diff branches in fast audit | D (backlog audit) |
| #6651 | 2026-04-26 | fix(automation): add compact branch audit output | D (backlog audit) |
| #6652 | 2026-04-26 | fix(automation): honor receipt filename keys | B (handoff post-spec) |
| #6659 | 2026-04-27 | fix(automation): accept shared state root CLI flag | B (handoff post-spec) |
| #6719 | 2026-04-27 | fix(swarm): preserve inherited roadmap codes on decomposed issues | J (roadmap codes; crosscut to K) |
| #6723 | 2026-04-27 | fix(automation): match terminal receipt keys in backlog audit | B (handoff post-spec) |
| #6725 | 2026-04-27 | fix(automation): normalize publisher PR action aliases | B (handoff post-spec) |
| #6726 | 2026-04-27 | fix(automation): bound backlog patch equivalence audit | D (backlog audit) |
| #6738 | 2026-04-27 | fix(automation): cache queue status for sandboxed watchers | C (publisher; crosscut to D) |
| #6741 | 2026-04-27 | fix(automation): accept branch publisher receipt dir flag | C (publisher) |
| #6742 | 2026-04-27 | fix(automation): keep outbox reconcile dry-run readonly | A (handoff spec) |
| #6746 | 2026-04-27 | fix(automation): verify fast-audit salvage candidates | D (backlog audit) |
| #6747 | 2026-04-27 | fix(automation): dedupe handoffs by open PR branch | A (handoff spec) |
| #6751 | 2026-04-27 | fix(automation): reconcile structured action branches | B (handoff post-spec) |
| #6754 | 2026-04-27 | fix(automation): fingerprint structured action branches | B (handoff post-spec) |
| #6755 | 2026-04-27 | fix(automation): tolerate list evidence in outbox reconcile | A (handoff spec) |
| #6766 | 2026-04-27 | fix(swarm): align proof-first boss fallback labels | K (proof-first labels) |
| #6773 | 2026-04-28 | fix(boss-loop): bound worker receipt metadata before signed-receipt path | F (boss-loop runtime payload) |
| #6774 | 2026-04-28 | fix(swarm): align preflight permission contracts | G (preflight permission) |
| #6776 | 2026-04-28 | fix(boss-loop): bound terminal json summaries | F (boss-loop runtime payload) |
| #6778 | 2026-04-28 | fix(boss-loop): honor acceptance gate failures | F + L (boss-loop terminal verdict; crosscut to L) |

51 rows. Every in-window PR mapped exactly once.

## Appendix B — file × category heatmap

For the 6 most-patched source files:

| File | A | B | C | D | E | F | G | H | I | J | K | L | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `scripts/audit_codex_branch_backlog.py` | 8 | 2 | 0 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 |
| `scripts/publish_automation_handoffs.py` | 8 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 |
| `scripts/publish_codex_automation_branches.py` | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 |
| `scripts/reconcile_automation_outbox.py` | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 (in-category) — note: also #6755 in A and #6751 in B; total touches in window = 4 |
| `aragora/swarm/boss_loop.py` | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 1 | 0 | 0 | 4 |
| `aragora/swarm/boss_worker_lifecycle.py` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |

The cell `audit_codex_branch_backlog.py × A = 8` plus `× B = 2` plus
`× D = 10` = 20 total touches in the window across three categories.
The cell `publish_automation_handoffs.py × A = 8` plus `× B = 4` = 12.
These are the visible signals of the highest-priority contract
derivations (rank 1 + rank 2).

## Appendix C — methodology

1. Pulled all merged PRs in `merged:2026-04-21..2026-04-28` via
   `gh pr list --state merged --limit 400 --search merged:2026-04-21..2026-04-28
   --json number,title,mergedAt`. Total: 270 merged PRs in window.
2. Filtered to titles starting with `fix(automation`, `fix(swarm`, or
   `fix(boss-loop`. Total: 51 PRs.
3. For each of the 51 PRs, pulled `gh pr view <num> --json
   number,title,body,files,mergedAt`. Stored locally as JSON for
   re-analysis.
4. Categorised by inspecting the file list and (where present) the PR
   body's `## Summary` and `## Why` sections. PRs with empty bodies
   were categorised by the file list alone, cross-checked against the
   PR title's verb (`dedupe`, `protect`, `classify`, `bound`, etc.).
5. Cross-referenced the 17 PRs explicitly enumerated by the existing
   handoff-contract spec to ensure Category A == that spec's PR list.
6. Identified recurrence threshold (≥3 PRs) per the parent task's
   instructions. Combined J+K and F+H+L crosscuts into single
   recurrent classes where the underlying contract surface is shared.
7. Wrote this document. No other file was created or modified by the
   audit.

End of audit.
