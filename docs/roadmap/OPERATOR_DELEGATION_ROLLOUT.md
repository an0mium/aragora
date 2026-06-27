# Operator Delegation Rollout

**Status:** active; tracks the staged rollout of
`docs/governance/OPERATOR_DELEGATION_POLICY.md`.

The policy defines four buckets (A=auto-merge, B=auto-close,
C=operator-y/n, D=strategic) plus an irreducible operator-only
tripwire list. This rollout takes the policy from "document" to
"automation actually running against the queue."

## Goal

Reduce the operator's per-PR review burden from "read 14 diffs" to
"type y/n on ~3 PRs per day" while preserving every hold, every
protected file, and every tripwire — and producing a durable audit
trail at every stage.

## Stages

Each stage is one PR. Stages are sequential — stage N+1 depends on
stage N landing first. Every stage default-OFF until the operator
opts in.

### Stage 0 — Policy doc (this PR)

Ship `docs/governance/OPERATOR_DELEGATION_POLICY.md` +
`docs/roadmap/OPERATOR_DELEGATION_ROLLOUT.md` (this file).

Acceptance: docs land on `main`, no behavior change.

### Stage 1 — `scripts/triage_open_prs.py` (read-only classifier)

Pure-stdlib CLI. Reads the live `gh pr list` + `gh pr view` data,
runs the four-bucket classification, prints the table per the
policy doc. **Never mutates.** Dogfoods
`scripts/list_active_agent_sessions.py` for hold detection +
authorship trust.

CLI:
```
python3 scripts/triage_open_prs.py [--json] [--bucket A|B|C|D]
                                    [--include-held] [--limit N]
```

Acceptance:
- Re-running it 24h apart on the same queue produces deterministic
  output (sort by bucket → number).
- Bucket classification matches the four-bucket criteria for every
  PR in the open queue.
- Default output is the human table; `--json` emits machine-readable.
- Test fixture covers each of the four buckets + each hard
  constraint (held, protected-file edit, flag flip, large diff,
  unresolved review, CI red, CI pending, external dep).

Tracking: GitHub issue tagged `operator-delegation`,
`infrastructure`.

### Stage 2 — `scripts/auto_merge_bucket_a.py` (opt-in enforcement)

Reads the Stage 1 classifier output, attempts `gh pr merge --squash`
on every Bucket A PR. Default `--dry-run`. Settling window (default
30 min after last commit) before merge. Hard-coded skip on tripwires
even if the classifier missed them (defense-in-depth).

CLI:
```
python3 scripts/auto_merge_bucket_a.py [--apply] [--settling-minutes N]
                                       [--only-pr N] [--json]
```

Acceptance:
- Default `--dry-run` enumerates intended merges, never mutates.
- `--apply` merges only PRs in Bucket A; never touches B/C/D.
- Aborts and exits non-zero on any single tripwire hit.
- Every merge writes a receipt to
  `docs/status/AUTO_MERGE_RECEIPT_<utc>.md` with the PR list, the
  policy version, and the classifier output sha256.

Tracking: GitHub issue tagged `operator-delegation`, `automation`.

### Stage 3 — `scripts/triage_bucket_c.py` (operator y/n batcher)

Reads the Stage 1 output, prints only Bucket C lines, accepts a
one-character response per PR (`y`/`n`/`d`) via stdin or a YAML
response file. Then executes the y/n decision per PR (advancing y,
closing n, no-op on d).

CLI:
```
python3 scripts/triage_bucket_c.py [--interactive] [--responses FILE]
                                   [--apply]
```

Acceptance:
- Without `--apply`, prints what would happen but mutates nothing.
- With `--apply`, advances `y` PRs via the appropriate `gh`
  command (label, mark-ready, comment) and closes `n` PRs.
- Tripwires for "advancing" anything onto held / protected surfaces.
- Receipt written to `docs/status/BUCKET_C_RECEIPT_<utc>.md`.

Tracking: GitHub issue tagged `operator-delegation`, `automation`.

### Stage 4 — Periodic scheduling (opt-in)

Once Stages 1–3 are battle-tested by manual invocation, ship a
LaunchAgent template (`scripts/launch_agents/com.aragora.delegation-policy.plist`)
that runs Stage 1 + Stage 2 + Stage 3 once daily at a configurable
hour. **Strictly opt-in via `make delegation-installed`** — never
auto-installed.

Acceptance: template lands; no `launchctl load` runs in CI; the
install Makefile target documents the consequences.

Tracking: GitHub issue tagged `operator-delegation`, `launchd`.

### Stage 5 — Bucket-D escalation surface

A small operator-facing surface (likely a `/review-queue/strategic`
route, or a single CLI command) that lists pending Bucket D items
with the agent's paragraph. Operator types one paragraph back per
item. Receipt-bound.

Tracking: GitHub issue tagged `operator-delegation`, `ui`.

## Holds + tripwires (cross-reference)

This rollout never violates any hold or tripwire from
`docs/governance/OPERATOR_DELEGATION_POLICY.md`. In particular:

- No automation in any stage edits a protected file.
- No automation lifts a hold.
- No automation labels with `boss-ready` / `autonomous`.
- No automation deploys to production.
- No automation closes another agent's PR.

Each stage's tests assert the above for the relevant code paths.

## Measurement

Each stage records its operational metrics into the receipt format
already in use by this repo:

- Per-bucket-classification counts per run
- Per-merge time-to-merge (Bucket A) for trend analysis
- Per-operator-y/n response time (Bucket C) — empirical baseline
- Tripwire-hit counts (target: low; rising trend = policy needs
  revision)

After two weeks of Stage 4 runs, compare:
- Before: average operator review time per PR (≥3 minutes
  historical)
- After: average operator review time per PR (target: <3 seconds in
  Bucket C; zero in A/B)

If the after-numbers don't match, revise bucket criteria.

## Sequencing

Stage 0 (this PR) → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5.

Each stage opens its own draft PR; tracking issues filed against
this rollout. Total estimated effort: 5–8 hours across 5 PRs.

## Tracking

| Issue | Stage | Status |
|---|---|---|
| [#7280](https://github.com/synaptent/aragora/issues/7280) | Stage 1: triage_open_prs.py | open |
| [#7281](https://github.com/synaptent/aragora/issues/7281) | Stage 2: auto_merge_bucket_a.py | open |
| [#7282](https://github.com/synaptent/aragora/issues/7282) | Stage 3: triage_bucket_c.py | open |
| _to be filed after Stage 3 lands_ | Stage 4: scheduling | not yet |
| _to be filed after Stage 4 lands_ | Stage 5: bucket-D surface | not yet |
