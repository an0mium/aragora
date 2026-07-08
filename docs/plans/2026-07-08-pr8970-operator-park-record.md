# PR 8970 Operator Park Record

Date: 2026-07-08

PR: https://github.com/synaptent/aragora/pull/8970

Head: `12cfe60b144d8393d82e16daeb986a31ec2b80c8`

Context artifact: `.aragora/conductor_cycles/20260708T210727Z-cycle198-pr8970-dogfood-dryrun.md`

## Disposition

PR #8970 is parked on operator disposition, not on another autonomous evidence
spend.

The current head has one counted OpenAI signal at exact head
`12cfe60b144d8393d82e16daeb986a31ec2b80c8`. A later exact-head dry-run produced
one clean prepared OpenAI item and one Grok item with `verdict=unknown`, so the
Grok item was not clean enough to apply.

The `operator-review-required` label remains present. While that label
independently blocks the normal merge path, a second-family evidence spend does
not unlock a same-cycle Tier 0-2 merge. Under `docs/AGENT_OPERATING_CONTRACT.md`
section `Conductor`, evidence-only work should not continue unless it unlocks
the same-cycle merge path or the operator explicitly authorizes the evidence
despite the independent operator gate.

## Next Allowed Actions

One of these is required before additional autonomous evidence work on this
exact head:

1. The operator resolves or dispositions `operator-review-required` through the
   supported path.
2. The operator explicitly authorizes applying one more clean exact-head model
   signal while the label remains.
3. The head changes because a new repair is needed, in which case the normal
   dry-run first sequence starts over at the new head.

## Non-Actions

This record does not authorize:

- removing labels
- collecting countable evidence while the operator gate independently blocks
  merge
- recording settlement
- rerunning CI
- merging with `--admin`
- force-pushing or rewriting the branch

## Related Live State

At the time this record was created, #8970 was:

- open and non-draft
- `MERGEABLE` / `BLOCKED`
- green on all non-quorum required checks
- failing only `aragora-merge-quorum` among required checks
- counted at `1/2` model quorum by merge-packet
- blocked by `operator-review-required`
