# Repeat-Blocker Park Policy

**Date:** 2026-07-07
**Status:** proposed policy for queue-drain and steward settlement selection
**Tracking:** epic #8761; composes with the draft-skip filter tracked by #8987
**Scope:** documentation only; no checker, settlement-code, workflow, or branch-protection change

## Problem

Queue-drain sessions lose throughput when the same PR head receives repeated
repair or evidence attempts after the current head already has a material,
repo-visible blocker. The failure mode is not that agents are idle. It is that
agents keep treating a standing current-head park record as a fresh queue item.

Recent examples:

- #8948 reached head `be56200ae6a19c79263c492741c68173eddbf669` after many
  prompt-handoff repair and evidence rounds. The current-head dry-run recorded a
  Claude timeout and OpenAI `CHANGES-REQUESTED`, `would_count=false`, with a
  blocking P1. That head should be skipped until it changes or an operator
  explicitly reopens it.
- #8908 reached head `7848d6ad02551a03bb283b0e60e466a9bb2fd4bb` with a
  current-head evidence blocker in the ack/apply path. Running more evidence on
  that unchanged head is less useful than switching progress class.

The policy below turns that observation into an explicit queue contract.

## Policy

### 1. Countable-evidence attempt cap

For a single PR head SHA, at most **three** countable-evidence attempts may be
made before the PR must be parked for that head and the conductor must switch
progress class.

An attempt counts when it tries to obtain model-review quorum for the current
head, whether the result is:

- counted support;
- a transport timeout;
- `CHANGES-REQUESTED`;
- non-countable evidence;
- an evidence collector failure after reviewer execution started.

An attempt does not count when it exits before reviewer execution because of a
main-red gate, a head mismatch, a live owner conflict, a missing branch, or a
local environment precheck failure. Those stops should still be recorded, but
they are not evidence attempts against the PR head.

The cap is per exact head SHA. A new commit resets the cap, but it does not
erase the prior head's park record.

### 2. Mandatory park after repeated blockers

The PR must be parked for the current head when any of these conditions holds:

- three countable-evidence attempts have been made for the exact head without a
  mergeable quorum packet;
- a current-head attempt returns a P0 or P1 finding;
- a reviewer returns real, countable dissent that is in scope for the PR;
- a current-head attempt returns the same unresolved P2 class previously
  identified on that head, and the session is not explicitly scoped to repair
  that exact blocker;
- the collector times out and a fallback reviewer path has already been used
  for the same head in the current cycle.

Parking is not closing. It is a queue-selection state: do not run more evidence
or settlement on this exact head until the head changes or an operator gives a
repo-visible override.

### 3. Progress-class switch

After parking an unchanged head, the next conductor cycle should switch progress
class instead of immediately repairing the same lane again.

Preferred progress classes, in order:

1. a different ready PR with no standing current-head park record;
2. a policy, fixture, or docs artifact that prevents the same failure mode;
3. a focused issue update that records the exact blocker and next repair scope.

This avoids turning model review into an unbounded nit treadmill while preserving
the useful dissent as backlog.

## Valid Park Record

A valid park record must be repo-visible on the PR and, when the PR is part of a
queue-drain or steward run, linked from the controlling epic or queue issue.

It must include:

- PR number, title, branch, and exact head SHA;
- generated timestamp in UTC;
- the stop condition;
- evidence attempt count for that head, when known;
- artifact paths or run URLs for the latest attempt;
- per-reviewer verdicts and whether each verdict would count;
- the highest blocking severity, with the exact finding text or a concise
  paraphrase plus file or behavior reference;
- whether required checks were green, red, pending, or not rechecked;
- whether a live owner, steering message, or head mismatch affected the result;
- the next allowed action.

The next allowed action should be one of:

- `new head required`;
- `repair exact blocker`;
- `operator override required`;
- `safe to retry after transport recovery`;
- `superseded by replacement PR`;
- `close as obsolete`.

Records that omit the head SHA or the next allowed action are not valid park
records for queue-selection purposes.

## Steward Selection Rule

Steward selection should skip a PR when all of these are true:

- the PR is open;
- the current head SHA matches a valid standing park record;
- the park record's next allowed action is not `safe to retry after transport
  recovery`;
- no newer operator comment explicitly reopens the head for evidence or
  settlement.

The skip should be reported, not hidden. Queue summaries should show:

- `skipped_current_head_parked`;
- the park record URL;
- the exact head SHA;
- the next allowed action.

This composes with the #8987 draft-skip filter as a second exclusion layer:

1. skip draft PRs that are not eligible for settlement;
2. skip non-draft PRs with a valid current-head park record;
3. rank the remaining eligible PRs by the normal steward policy.

If #8987 lands a machine-readable skip surface, the repeat-blocker skip should
reuse that same shape rather than adding a second queue metadata format.

## Non-Goals

This document does not implement:

- a park-record parser;
- a steward-selection checker;
- new GitHub labels;
- branch-protection or workflow changes;
- any settlement authority change.

The intended follow-up is a small checker or queue helper that recognizes the
park-record fields above and reports the skip reason before collector work
starts.
