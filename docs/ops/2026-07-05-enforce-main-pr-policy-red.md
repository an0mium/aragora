# enforce-main-pr-policy Red Classification: 2026-07-05

## Scope

This note classifies the red `enforce-main-pr-policy` check on `origin/main`
at commit `e85873d2998edce4cbfc2bfddb57249410837d3f`.

No workflow rerun, settlement, merge, PR close, lane release, outbox archive, or
branch-protection mutation was performed while producing this note.

## Run Evidence

- Workflow: `Branch Discipline`
- Run ID: `28756606214`
- Job ID: `85264674162`
- Run URL: `https://github.com/synaptent/aragora/actions/runs/28756606214`
- Job URL: `https://github.com/synaptent/aragora/actions/runs/28756606214/job/85264674162`
- Event: `push`
- Head SHA: `e85873d2998edce4cbfc2bfddb57249410837d3f`
- Conclusion: `failure`

The job log reported:

```text
Direct push commits detected on main without associated PRs:
e85873d fix(lanes): make terminal mailbox receipts advisory-only
Use feature branches + PR merge. Emergency override tag: [allow-direct-main]
```

The log did not include the `(PR lookup failed)` suffix for this commit.

## Per-Commit Classification

| Commit | Workflow log status | Independent REST evidence | Verdict |
| --- | --- | --- | --- |
| `e85873d2998edce4cbfc2bfddb57249410837d3f` | Flagged as direct push | `GET /repos/synaptent/aragora/commits/e85873d2998edce4cbfc2bfddb57249410837d3f/pulls` returned PR `#8897`, `merged_at=2026-07-05T22:11:27Z`, `base=main`, `merge_commit_sha=e85873d2998edce4cbfc2bfddb57249410837d3f` | `FALSE-POSITIVE-ASSOCIATION-LAG` |
| `f1f76bac14368ef0ef1af72dd3d6202dae3e16ee` | Not flagged in this run | REST returned merged PR `#8888`, `base=main`, matching merge commit | `NOT-FLAGGED-MERGED-PR` |
| `18b5c0db30437b8c04bfc2d122ebb9e0001278ab` | Not flagged in this run | REST returned merged PR `#8887`, `base=main`, matching merge commit | `NOT-FLAGGED-MERGED-PR` |

## Interpretation

The current evidence does not support a true direct push for the flagged commit.
The flagged commit is the merge commit for PR `#8897`. The most likely failure
mode is a transient GitHub commit-to-PR association indexing race during the
immediate post-merge push event: the workflow checked the pushed merge commit at
`2026-07-05T22:11:34Z`, roughly seven seconds after PR `#8897` merged at
`2026-07-05T22:11:27Z`.

Because the workflow treats a negative associated-PR lookup as a violation, an
eventual-consistency miss can make main look red even when the commit is a
normal PR merge.

## Recommendation

Do not revert `e85873d2998edce4cbfc2bfddb57249410837d3f`.

The follow-up should be a reviewed workflow hardening PR, because workflow files
are governed surfaces. The likely fix is to make `branch-discipline.yml` robust
to immediate post-merge association lag, for example by retrying the exact
commit-to-PR lookup briefly before classifying a merge commit as direct-pushed,
or by verifying merge-commit title/PR metadata with a narrow `pulls.get` fallback
when the commit has fresh PR-merge characteristics.

Any workflow change needs explicit approval under
`docs/AGENT_OPERATING_CONTRACT.md`; this note intentionally proposes the repair
path without changing `.github/workflows/branch-discipline.yml`.
