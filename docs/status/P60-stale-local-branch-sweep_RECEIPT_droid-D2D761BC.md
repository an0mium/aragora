# P60 (stale-local-branch-sweep) receipt

- Session: `droid-D2D761BC`
- Lane: `P60-stale-local-branch-sweep`
- Branch: `droid/P60-stale-local-branch-sweep-20260518-172332`
- PR: none (hygiene)
- Started: 2026-05-18T17:23:33Z
- Completed: 2026-05-18T17:34:00Z
- Outcome: shipped

## Result

| Metric | Before | After | Delta |
|---|---|---|---|
| local branches | 2 202 | 1 313 | **-889** |
| `.git` size | 2.7 G | 2.6 G | -100 MB |

## D/R compliance

- D1: no manual `rm`; used `git branch -D` (reversible via reflog until gc).
- R5: lane claimed before any ref mutation.
- R12: 0 branches with open PRs deleted (all candidates verified `: gone` upstream OR shipped-in-journal OR 0-unique-commits-vs-main).
- R13: 123 worktree-bound branches preserved (skipped by filter).
- R20: ack — long-lived `active` lanes still present in registry (P54 may need transition; tracked for P63).

## Tests

Hygiene phase; no code change; no unit tests written.

## CI

No PR; no CI run.

## Notes for v13

- See SESSION_BRIEF "v13 prompt-bug" — P60 acceptance threshold of ≤ 200
  is unrealistic without patch-equivalence detection.
- Remaining 1 313 local branches: ~1 130 have committerdate > 14 d ago
  but appear in lineage as having unique commits (squash-merge artifact).
  Sweeping these requires either an aggressive cherry-pick-equivalence
  pass or operator-permission to delete by-rule.

## Lane release

`P60-stale-local-branch-sweep` released `status=completed`.
