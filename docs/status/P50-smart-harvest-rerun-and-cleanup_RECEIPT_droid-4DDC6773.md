# P50 (smart-harvest-rerun + cleanup) receipt

- Session: `droid-4DDC6773`
- Lane: `P50-smart-harvest-rerun-and-cleanup`
- Branch: `droid/P50-smart-harvest-rerun-and-cleanup-20260518-165324`
- PR: none (hygiene + audit phase; no PR per v11 prompt)
- Started: 2026-05-18T16:53:24Z
- Completed: 2026-05-18T17:15:00Z (approximate)
- Outcome: shipped

## Bytes recovered

894 MB = 464 MB (`droid-20260518-041442-0e8aade8`) + 430 MB
(`codex-work-owner-lane-enrichment-20260518`).

Disk: 40 G to 39 G free, 96% full (APFS lazy block release means
the next df poll will reflect the full delta).

## Acceptance against v11 P50 spec

(Renumbered from v11's P47 per R18 to avoid collision with codex
#7324's already-shipped P47-operator-snapshot-active-lane-parity.)

| Spec | Implementation |
|---|---|
| Refresh inventory with `--smart-merge-detection` flag | Done; 10.5 s wall clock |
| Publish new inventory artifact | Done; `worktree-value-inventory-20260518T170310Z.json` |
| For reclassified false-positives, run cleanup | 2 worktrees confirmed removable + removed (894 MB total) |
| Record reclassification delta | Yes; see SESSION_BRIEF inventory diff table (44 vs 56 candidates) |
| Receipt + journal + release lane | This file + journal append + status=completed |
| Bounded <= 45 min | ~22 min wall clock |

## Tests

Hygiene phase; no new code; no unit tests written. Used the
already-tested `codex_worktree_value_inventory.py`,
`publish_worktree_value_inventory.py`, and
`safe_worktree_cleanup.py` scripts (each ship with their own
test suites under `tests/scripts/`).

## CI

No PR; no CI run for this lane.

## D2/R11/R12/R13/D5 compliance

- R11 (no delete without fresh inspect): both removals passed
  `safe_worktree_cleanup.py inspect` with `removable: True`.
- R12 (no delete of open-PR head_ref): both removed branches
  had `open_prs: 0` at inspect time (PRs #7306 and #7309
  already merged).
- R13 (respect active_session): both showed `active_session: False`.
- D1 (use safe_worktree_cleanup.py): yes, no manual `rm -rf`.
- D2 (no delete dirty): 3 dirty no_git_cache_residue candidates
  DEFERRED; see SESSION_BRIEF; surfaced as v12 prompt-bug #2
  (inspect's dirty-flag is over-eager on empty nested wrappers).
- D5 (squash-merge false-positive rule): the publisher's smart
  classifier didn't reclassify any harvest candidates beyond what
  basic `git cherry` already caught; see SESSION_BRIEF for
  "Smart-merge-detection efficacy" finding.

## Scope notes for v12

1. Renumber the audit-batch phase to P51+ (P47 was claimed by
   codex for `operator-snapshot-active-lane-parity` while v11 was
   being drafted; #7324 already merged with P47 receipts at
   commits 9555e9b02 and 633b664f4).
2. `--smart-merge-detection` is currently hobbled by `--skip-gh`.
   The classifier can't recognize open-PR branches without gh
   access, leading to false-positive harvest_candidates. Either
   recommend running without `--skip-gh` (despite the wall-clock
   cost on this machine) or supplement with a cached `gh pr list`
   lookup.
3. Inspect's dirty-flag detection should ignore empty nested
   `.worktrees/` wrapper directories; 3 such residue dirs in
   the current inventory are blocked from cleanup despite being
   provably empty.
4. Smart classifier should emit per-commit match info, not just
   a yes/no decision per branch; for 12-commit branches like
   #7292's, this would clarify whether the branch is fully
   shipped or has unique value.

## Lane

`P50-smart-harvest-rerun-and-cleanup` released at session close
(status `completed`, branch recorded; no PR).
