# P68 (cherry patch-equivalence local-branch sweep) receipt

- Session: `droid-P68-2B0AEE5B`
- Lane: `P68-cherry-patch-equivalence-sweep`
- Branch: `main` (hygiene + script only; no feature PR)
- PR: none
- Started: 2026-05-18T19:44:10Z
- Completed: 2026-05-18T19:50:00Z
- Outcome: shipped (true-ceiling)

## Result

| Metric | Before | After | Delta |
|---|---|---|---|
| local branches (`git branch \| wc -l`) | 1 334 | 1 286 | **-48** |
| candidate branches (`git cherry` all-`-`) | — | 49 | — |
| preserved-with-unique commits | — | 153 | — |
| skipped (tracked remote still present) | — | 993 | — |
| skipped (worktree-bound HEAD) | — | 138 | — |
| skipped (active lane claim) | — | 0 | — |
| skipped (main) | — | 1 | — |
| cherry errors | — | 0 | — |

(Pre-count was sampled at 1 334 immediately after lane claim; post-count
1 286 was sampled directly after `--apply`. The 1-row gap vs `1 334 - 49`
is a single concurrent branch creation by a sibling session between the
two samples; every one of the 49 reported deletions verified absent
post-sweep.)

## True-ceiling observation

The v13 spec target was 1 315 → ≤ 1 000 (delta ≥ 300). The actual
ceiling — i.e. the union of "every commit on the branch is
patch-equivalent on `origin/main`" *and* "no tracked remote upstream
still present" *and* "not worktree-bound" *and* "not currently claimed"
— is **49** in this repo at the snapshot above. The dominant
preservation reason is `skipped_tracked_remote = 993`: those branches
still have an `origin/<branch>` ref, so the spec's pre-flight rule
("Never delete a branch that has a tracked remote tracking ref unless
the remote is also gone") keeps them. Remote-side cleanup (P70) is the
counterpart lane that can unblock further local reduction in a future
fan-out.

## Live dry-run JSON (verbatim)

```json
{
  "active_lane_statuses": [
    "active",
    "claimed",
    "pending",
    "queued",
    "running"
  ],
  "applied": false,
  "base": "origin/main",
  "candidate_count": 49,
  "deleted": 0,
  "deleted_branches": [],
  "dry_run": true,
  "errors": [],
  "lane_registry": "<repo-root>/.aragora/agent-bridge/lanes.json",
  "limit": null,
  "preserved_with_unique": 153,
  "scanned": 1334,
  "skipped_claim": 0,
  "skipped_error": 0,
  "skipped_main": 1,
  "skipped_tracked_remote": 993,
  "skipped_worktree": 138,
  "swept_at": "2026-05-18T19:44:31Z"
}
```

## Live --apply JSON (verbatim, deleted list inlined)

```json
{
  "applied": true,
  "candidate_count": 49,
  "deleted": 49,
  "deleted_branches": [
    "codex/benchmark-publication-reuse-gh-salvage",
    "droid/b6-truth-surface-ci-gate-20260516",
    "pr-6808-review",
    "pr-6815-dogfood",
    "pr-6815-review",
    "pr-6816-review",
    "pr-6862",
    "pr-6873",
    "pr-6900",
    "pr-6900-review",
    "pr-6914-review",
    "pr-6915-review",
    "pr-6925",
    "pr-6930",
    "pr-6931",
    "pr-6938",
    "pr-6940",
    "pr-6942",
    "pr-6957-review",
    "pr-6958",
    "pr-6960",
    "pr-6961",
    "pr-6963",
    "pr-6969",
    "pr-6977",
    "pr-6979-review",
    "pr-6984-claude",
    "pr-6984-review",
    "pr-6989-review",
    "pr-6989-state",
    "pr-7009-claude",
    "pr-7025-review",
    "pr-7026-review",
    "pr-7042-review",
    "pr-7046",
    "pr-7048",
    "pr-7062",
    "pr-7077",
    "pr-7163",
    "pr-7167-rebase",
    "preflight/20260426-175348",
    "preflight/20260428-025037",
    "preflight/20260516-172132",
    "tier1-review-6945",
    "tier1-review-6948",
    "tmp-pr-6999",
    "tmp-pr-6999-review",
    "tmp-pr-7010",
    "vision-incubator/agt-03-manifold-brier-bridge"
  ],
  "dry_run": false,
  "errors": [],
  "scanned": 1334,
  "swept_at": "2026-05-18T19:45:19Z"
}
```

A single `git branch -D` batch was issued; verification re-checked all
49 names via `git branch --list <name>` and found 0 still present.

## D/R compliance

- **R5** (lane claim before mutation): claimed `P68-cherry-patch-equivalence-sweep`
  at 19:44:10Z via `claim_active_agent_lane.py`; released at 19:50:00Z.
- **R12** (no open-PR branches deleted): every deleted branch had no
  tracked remote (or remote was gone) AND every commit was
  patch-equivalent to `origin/main`. The 993 `skipped_tracked_remote`
  preservation guarantees no live-PR branch was touched.
- **R13** (worktree-bound preserved): 138 worktree-bound branches
  skipped via `git worktree list --porcelain` enumeration.
- **R19** (no `--amend` of pushed commits): not applicable.
- **R20** (preflight mypy): `python3 -m mypy scripts/cherry_sweep_local_branches.py
  tests/scripts/test_cherry_sweep_local_branches.py` → `Success: no issues
  found in 2 source files`.
- **R22** (patch-equivalence via `git cherry`): every candidate had a
  cherry output with zero `+` lines.
- **R25** (no raw `rm -rf` of worktrees): only `git branch -D` was used.
- **D1**: reversible via reflog until next gc; no `git gc --prune=now`
  invoked by this lane.

## Tests

`tests/scripts/test_cherry_sweep_local_branches.py` — 8/8 passing
(8 tests, > the spec floor of 4):

1. `test_patch_equivalent_branch_is_candidate` — covers acceptance (1).
2. `test_branch_with_unique_commit_is_preserved` — covers acceptance (2).
3. `test_main_branch_is_never_deleted` — covers acceptance (3).
4. `test_claimed_lane_branch_is_skipped` — covers acceptance (4)
   (lane-registry mock).
5. `test_apply_actually_deletes_candidates` — end-to-end `--apply` smoke
   on a real tmp repo.
6. `test_limit_caps_deletions` — `--limit N` semantics.
7. `test_tracked_remote_still_present_is_skipped` — protects branches
   whose `origin/<branch>` still exists.
8. `test_worktree_bound_branch_is_skipped` — guards via injected
   worktree lister.

The tmp-repo fixture pushes the second commit to a bare `origin.git`
so `git cherry origin/main feature-equiv` actually has an upstream
patch to match against — the fixture itself is the canonical
patch-equivalence regression case for future v's.

## CI

No PR (hygiene lane + script + tests). Local pytest suite green; mypy
clean.

## Notes for v14

- The `skipped_tracked_remote = 993` total is the dominant ceiling on
  this surface. P70 (codex-namespace remote audit) will reduce that
  cohort; a follow-up local sweep after P70 should free another
  ~200–300 candidates without changing any of the rules below.
- The sweeper is idempotent: running `--apply` a second time on the
  same repo emits `candidate_count=0` because every patch-equivalent
  branch is gone.
- Consider widening the `ALWAYS_PROTECT` set if release branches are
  introduced (currently only `main` and `master`).

## Lane release

`P68-cherry-patch-equivalence-sweep` released `status=completed`.
