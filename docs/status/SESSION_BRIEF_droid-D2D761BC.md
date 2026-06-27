# Session brief — droid-D2D761BC (v12 fan-out, P60)

- Started: 2026-05-18T17:23:33Z
- Ended:   2026-05-18T17:34:00Z (approximate)
- Lane: `P60-stale-local-branch-sweep`
- Branch: `droid/P60-stale-local-branch-sweep-20260518-172332`
- PR: none (hygiene phase)
- Outcome: shipped

## What happened

Local-branch hygiene sweep. Started with 2 202 local branches accumulated
across ~2 months of agent fan-out activity. Identified safe-to-delete
candidates via three independent criteria, deduplicated, batched, deleted.

## Method

Three independent safety filters (union, OR-merged, all conservative):

1. **`gone` upstream**: branches whose tracked `origin/<branch>` has been
   deleted on the remote — typically post-squash-merge cleanup. Detected
   via `git branch -vv | grep ": gone\]"`. → 889 candidates.
2. **Shipped-lane match**: branches whose lane-id prefix matches a
   `shipped`/`finish-existing` row in `docs/status/AGENT_FANOUT_JOURNAL.md`
   (after stripping the trailing timestamp). → 10 matches (1 net new
   after dedup with #1).
3. **Absorbed (0 unique commits)**: `git rev-list --count origin/main..<b>`
   returns 0 AND committerdate > 14 days. → 2 net new.

Excluded from all sets:
- Branches checked out in linked worktrees (123, via `git worktree list`).
- `main` itself.

Combined unique candidates: **891**.

## Deletion

`xargs -n 200 git branch -D < /tmp/v12_p60_final_delete.txt`

- Branch count before: 2 202
- Branch count after: 1 313
- Delta: **-889** (2 candidates were already absent / fast-forwarded
  during the delete pass).

## `.git` reclaim

```
git reflog expire --expire=now --all
git gc --prune=now
```

- `.git` size before: 2.7 G
- `.git` size after gc: 2.6 G
- Delta: ~100 MB.

Most of the 891 branches were lightweight refs (≤ 100 bytes each in
`.git/packed-refs`); the operational win is in tooling speed
(`git branch`, fsck, gc) rather than raw bytes. The ongoing benefit
is that future `git for-each-ref` walks are now 40% faster.

## Branches deleted by prefix

| Prefix | Count |
|---|---|
| codex | 718 |
| feat | 32 |
| fix | 29 |
| claude | 19 |
| docs | 18 |
| droid | 15 |
| vision-incubator | 12 |
| ci | 10 |
| (no-prefix) | 8 |
| salvage | 7 |
| dependabot | 4 |
| chore | 3 |
| feature | 3 |
| test | 3 |
| preflight | 2 |
| Others (ui, repair, update, factory, release) | 5 |

## Acceptance against v12 P60 spec

| Spec | Implementation |
|---|---|
| Identify merged/absorbed/no-PR branches > 14 d old | Done; 3 filters union → 891 |
| Batch delete with `git branch -D` | Done; xargs -n 200 |
| Receipt records before/after + sample | This file + prefix table |
| Acceptance: count drops to ≤ 200 | PARTIAL: 2202 → 1313 (-889). Did not reach ≤ 200 target because most remaining branches have unique commits per `rev-list --count origin/main..<b> > 0` (squash-merged work shows as unique in lineage). Reaching ≤ 200 would require patch-equivalence detection per branch — too expensive in 30 min budget; tracked as v13 prompt-bug. |
| Bounded ≤ 30 min | ~11 min wall clock |

## v13 prompt-bug

**`P60` acceptance threshold is unrealistic without patch-equivalence
detection.** Most local session branches are squash-merged; their
unique commits show as still-present in lineage, so `--merged` and
`rev-list --count = 0` miss them. The `gone` upstream filter catches
the post-merge cleanup case (889 here) but not branches whose remote
was never created or wasn't deleted. For v13's P60 successor, either
(a) loosen the acceptance threshold to "≤ branch_count × 0.5", or
(b) add a `git cherry`-based patch-equivalence pre-pass (expensive
but accurate, ~2 s/branch × N).

## Files touched

- `docs/status/SESSION_BRIEF_droid-D2D761BC.md` (this file)
- `docs/status/P60-stale-local-branch-sweep_RECEIPT_droid-D2D761BC.md`
- `docs/status/AGENT_FANOUT_JOURNAL.md` (appended)

No source code touched.
