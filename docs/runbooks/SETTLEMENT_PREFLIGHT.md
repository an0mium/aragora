# Settlement Gate Preflight

`scripts/settle_preflight.py` is a read-only classifier for conductor queue
selection. It answers one question before a lane spends review or settlement
effort: what is the next legal action for this PR under
`docs/AGENT_OPERATING_CONTRACT.md` §Conductor?

The classifier does not post comments, rerun checks, collect evidence, mark PRs
ready, merge, edit labels, or change branch protection.

## Usage

Classify one PR:

```bash
python3 scripts/settle_preflight.py --pr 8990 --repo synaptent/aragora --json
```

Classify the open queue:

```bash
python3 scripts/settle_preflight.py --queue --repo synaptent/aragora --json
```

Callers must run the main-health check first. If any material required context
on `origin/main` is red or missing, pass `--main-red` only to produce a
`MAIN_RED_HALT` report; do not keep advancing PRs.

Every verdict carries this recheck rule:

> recheck on next origin/main push; never poll in a loop.

## Verdicts

| Verdict | Meaning | Conductor action |
| --- | --- | --- |
| `MAIN_RED_HALT` | `origin/main` required checks are not green. | Stop queue work and enter main-red incident mode. |
| `DRAFT_SKIP` | The PR is draft. | Skip until the PR is explicitly ready for review. |
| `HUMAN_GATED` | Tier is above 2, or the merge packet requires human risk settlement without recorded preapproval. | Stop and request exact-head human settlement before evidence or merge. |
| `HEAD_BLOCKED` | The head is conflicting, behind, dirty, missing a satisfied packet, or has current-head blockers. | Park this head until the blocker clears or a repair head lands. |
| `GITHUB_UNSTABLE` | The model packet is authorized, but GitHub reports an unstable or non-mergeable state. | Do not merge; wait for settlement-stable GitHub state on a future main push. |
| `READY` | The PR is model-authorized and settlement-stable. | Run one final live check, then use normal exact-head protected squash merge. |

## Park vs. Wait

Park when the blocker is about the PR head: human gate, draft state, missing
model quorum, current-head dissent, dirty/conflicting state, or an unresolved
repair finding.

Wait when the blocker is GitHub's transient merge-state calculation after the
packet is already authorized. Do not poll continuously. Record the exact head,
merge packet status, check state, and next recheck trigger.

This composes with
`docs/plans/2026-07-07-repeat-blocker-park-policy.md`: a current-head park
record remains the source of truth for avoiding repeated evidence attempts, and
the preflight classifier provides the cheap first-pass skip signal before a
conductor spends a cycle.

## Worked Example: #8990

#8990 added `docs/plans/2026-07-07-repeat-blocker-park-policy.md`. The exact
head `427aacc893a1f508690296405e0bbcf233b17c56` first failed
`aragora-merge-quorum` because no countable Tier 0 model signal existed. After
an exact-head OpenAI PASS landed, the merge packet became satisfied and
required checks were green, but GitHub still reported `mergeStateStatus` as
`UNSTABLE`.

Under §Conductor, `UNSTABLE` is not settlement-stable, so the correct
preflight verdict during that interval was `GITHUB_UNSTABLE`: wait for the next
main-push recheck, do not run another evidence cycle, and do not merge by
conductor automation. The PR later merged normally at merge commit
`196bf38d540df5bd37a5a0918d8b8dd54604c2f6` once the unstable state cleared.
