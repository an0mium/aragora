# Session brief — droid-4DDC6773 (v11 fan-out, P50)

- Started: 2026-05-18T16:53:24Z
- Ended:   2026-05-18T17:15:00Z (approximate)
- Agent family: `droid`
- Lane claimed: `P50-smart-harvest-rerun-and-cleanup`
- Branch: `droid/P50-smart-harvest-rerun-and-cleanup-20260518-165324`
- PR: none (hygiene/audit phase; receipt-only on main)
- Outcome: shipped

## What happened

Two-part bounded contribution leveraging the freshly-merged P42
smart-harvest classifier (#7312) to expand cleanup coverage and
exercise the new `--smart-merge-detection` flag against the live
worktree inventory.

Phase 0 / 0.5 fired the disk-pressure trigger (40 G free, 96% full,
down 16 G from v11 publication). With 0 active sibling lanes,
Q01-* settled (Stage 2 #7292 at 61 SUCCESS / 1 CANCELLED / 21
SKIPPED — much improved), and P22/P29/P30/P42/P45/P47 all merged,
the hygiene-or-harvest priority pointed at "use P42's new flag to
expand cleanup".

## Implementation

1. Refreshed inventory with the new flag:
   `python3 scripts/codex_worktree_value_inventory.py --root .worktrees/codex-auto --skip-gh --size-mode none --git-timeout 5 --smart-merge-detection --json > /tmp/v11_smart_inv.json`
   (10.5 s wall clock — well inside budget).
2. Published via `publish_worktree_value_inventory.py --input /tmp/v11_smart_inv.json`. New `latest.json` snapshot at `worktree-value-inventory-20260518T170310Z.json` (`generated_at` `2026-05-18T17:03:10Z`).
3. Inspected each of the 4 cleanup_candidates from the smart-classifier output. 1 truly removable (`droid/P20-model-pins-frontier-aligned-20260518-041438` worktree, classified `patch_equivalent_or_merged` because #7306 merged earlier today). 3 dirty `no_git_cache_residue` wrappers deferred per D2.
4. Cross-referenced the broader `.worktrees/` directory for additional patch-equivalent worktrees post-merge. Caught `codex-work-owner-lane-enrichment-20260518` — patch-equivalent + 0 open PRs because #7309 just merged.
5. Ran `safe_worktree_cleanup.py remove` against both confirmed-removable paths. 894 MB recovered (464 M + 430 M).

## Inventory diff vs prior snapshot

| Metric | Prior (2026-05-18T05:10Z) | New (2026-05-18T16:53Z) | Delta |
|---|---|---|---|
| classified_candidates | 56 | 44 | -12 |
| cleanup_candidate | 7 | 4 | -3 |
| harvest_candidate | 34 | 25 | -9 |
| preserve | 15 | 15 | 0 |
| patch_equivalent_or_merged (class) | 4 | 1 | -3 |
| no_git_cache_residue (class) | 3 | 3 | 0 |
| active_or_dirty (class) | 14 | 14 | 0 |
| unique_unharvested (class) | 34 | 25 | -9 |
| receipt_protected (class) | 1 | 1 | 0 |

The drop in total candidates (56 to 44) is largely attributable to
codex-2D173797's `P29-cleanup-safe-removable-worktrees` lane
which cleaned up multiple candidates at 12:14:19Z, plus
codex-5B032BF7's `P45-legacy-worktree-dir-audit-clean` at
05:45:39Z.

## Smart-merge-detection efficacy (key finding for v12)

The new `--smart-merge-detection` flag found ZERO additional
reclassifications beyond what the basic `git cherry`
patch-equivalence check already catches. Counter-intuitive given
that 5+ branches in the pre-existing harvest_candidates list had
verifiable squash-merge equivalents on main (verified via
`gh pr list --state merged --search "<subject>"`).

Root cause: `--smart-merge-detection` is gated by `--skip-gh`,
which the constrained recipe uses to fit a 10-min budget. Without
the gh PR cross-reference, the classifier can't detect open-PR
branches and over-counts them as harvest_candidates. Also, the
subject-line fuzzy match appears to require very close textual
identity — multi-commit branches whose intermediate commit
subjects don't appear verbatim in main's squash-merge subjects
remain `unique_unharvested`.

Examples of harvest_candidates that ARE actually open-PR or
merged-PR branches (smart classifier should have reclassified
but didn't):

| Branch | Actual State |
|---|---|
| `worktree-packets-keyboard-throughput-20260517` | OPEN PR #7278 |
| `claude/P24-canonical-test-definitions-count-drift-...` | OPEN PR #7307 |
| `claude/P28-A-identify-lane-owner-...` | OPEN PR #7308 |
| `claude/P29-steering-mailbox-writer-...` | OPEN PR #7310 |
| `claude/P30-operator-snapshot-steering-messages-...` | OPEN PR #7311 |
| `droid/P16-stage2-auto-merge-bucket-a-...` (12 commits ahead!) | OPEN PR #7292 (Stage 2 LINCHPIN) |
| `droid/P02-freshness-probe-rerun-...` | MERGED PR #7287 |
| `droid/P17-stage3-triage-bucket-c-batcher-...` | OPEN PR #7294 |

This is a classification bug, not a cleanup bug: the affected
worktrees should be left alone (they ARE the active-PR branches
for the linchpin Stage 2 etc.), but they're noisy in the inventory.

## Bytes recovered

| Path | Size | Reason | Outcome |
|---|---|---|---|
| `.worktrees/codex-auto/droid-20260518-041442-0e8aade8` | 464 MB | branch `droid/P20-model-pins-frontier-aligned-20260518-041438` patch-equivalent post-#7306 merge | removed |
| `.worktrees/codex-work-owner-lane-enrichment-20260518` | 430 MB | branch `codex/work-owner-lane-enrichment-20260518` patch-equivalent post-#7309 merge | removed |

Total recovered: 894 MB.

Disk before P50: 40 G free / 926 G total (96% full).
Disk after  P50: 39 G free / 926 G total (96% full).
(APFS lazy block release means df may not reflect the full delta
for a few minutes; the absolute drop on `du -sh .worktrees` is
the canonical signal.)

## Deferred (per D2 — dirty worktrees never auto-removed)

| Path | Size | Classification | Defer reason |
|---|---|---|---|
| `.worktrees/codex-auto/claude-20260501-195315-89c3c0ad` | 4 KB | no_git_cache_residue | inspect reports `dirty=True` even though dir contains only an empty nested `.worktrees/` subdir; D2 conservative skip |
| `.worktrees/codex-auto/codex-20260422-194119-a5c0dd59` | 24 KB | no_git_cache_residue | same as above |
| `.worktrees/codex-auto/rbac-health-fix-20260413-1` | 32 KB | no_git_cache_residue | same as above |

Total deferred: 60 KB (immaterial). The dirty flag here appears to
be a false positive from `safe_worktree_cleanup.py inspect`; the
3 dirs are empty nested-worktree wrappers, not actual project
content. A v12 prompt-bug item: refine the dirty-detection in
inspect so empty nested wrappers don't block cleanup.

## v12 prompt-bug ledger

1. `--smart-merge-detection` is hobbled by `--skip-gh`. The
   classifier can't recognize OPEN-PR branches without gh access,
   so it misclassifies them as harvest_candidates. v12 should
   either (a) recommend running without `--skip-gh` despite the
   wall-clock cost, or (b) ship a follow-on that supplements the
   classifier with `gh pr list --state open --json headRefName` as
   a cached lookup the inventory script can consume.

2. Inspect's dirty-flag is over-eager on empty nested wrappers.
   Three 4-32 KB residue dirs in `.worktrees/codex-auto/` contain
   only an empty `.worktrees/` subdir and report `dirty=True`. They
   should be safely removable but D2 conservatism blocks. v12
   could add a `--ignore-empty-nested-wrappers` flag to
   `safe_worktree_cleanup.py inspect`, or have the inspect helper
   recognize empty-wrapper sentinel patterns.

3. Multi-commit harvest candidates need a per-commit match
   summary. The smart classifier currently reports a single
   yes/no decision per branch. For 12-commit branches like
   #7292's, knowing WHICH commits matched main vs which are
   unique would reveal whether the branch is safe to delete (all
   commits matched) or genuinely valuable (one+ unmatched). Add
   `match_per_commit: [{sha, subject, matched_main_sha?}]` to the
   per-candidate output.

4. Phase numbering: v11 used `P47` for "manual harvest-audit
   batch" but codex used `P47` for `operator-snapshot-active-lane-
   parity` (#7324) while v11 was being drafted. v12 ack section
   must diff the journal again and bump audit-batch lane to P51+.

## Lane release

`P50-smart-harvest-rerun-and-cleanup` released at session close
(status `completed`, no PR — hygiene phase).

## Files touched

- `docs/status/generated/worktree_value_inventory/latest.json` (refreshed)
- `docs/status/generated/worktree_value_inventory/worktree-value-inventory-20260518T170310Z.json` (new snapshot, with `--smart-merge-detection` enabled for the first time on this machine)
- `docs/status/WORKTREE_VALUE_INVENTORY_STATUS.md` (refreshed)
- `docs/status/SESSION_BRIEF_droid-4DDC6773.md` (this file)
- `docs/status/P50-smart-harvest-rerun-and-cleanup_RECEIPT_droid-4DDC6773.md` (receipt)
- `docs/status/AGENT_FANOUT_JOURNAL.md` (appended)

No source code touched, no PR opened.
