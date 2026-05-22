# P102 Harvest + SSD Recovery Receipt — claude-51C05A58

**Lane:** `P102-claude-harvest-recovery`
**Window:** 2026-05-21 15:13Z – (in progress)
**Plan source:** Factory paste-ready prompt + user request "harvest worktrees/branches/codex subfolders + free SSD to 80GB+"

## Identity caveat (structural finding)

`gh api user --jq .login` returns `an0mium`. Every PR opened via `gh` on the operator's machine shows author=an0mium. Per branch protection, **`an0mium`-authored PRs cannot be approved by this session**. Affects every PR opened by every Claude/Codex/Droid agent running here, including:
- #7389, #7396, #7397, #7398 (previous-session harvests)
- #7408–#7414, #7416–#7420 (this session's 12 harvest PRs)
- #7415 (this session's review-packets PR)

**Operator action needed:** add a non-author GitHub identity (bot account / co-maintainer) before any further PR can advance to merge. This is the binding bottleneck across the whole queue.

## Concurrent agents observed

Three live agents during the soak:
| PID | Agent | Activity |
|-----|-------|----------|
| 57572 | Factory droid | Worker subagent "Inspect disk candidates" (read-only) |
| 36431 | worktree_maintainer.sh | ff-only reconcile loop (active) |
| 59446/59483 | bash automation_pr_preflight.sh | **STALE** since 12:40 AM, holding git locks |

The Factory droid did read-only inventory in parallel — no conflict.

**Stale-process problem:** the 2 ancient `automation_pr_preflight.sh` processes (PIDs 59446, 59483) from codex worktree `4783` have been holding git locks for ~10 hours, causing intermittent `index.lock` failures during this session. Recommend operator kill these.

## Harvest deliverables — 12 PRs

Opened via classifier-driven selection from 244 candidate branches with unique commits + no existing PR:

### First batch (7 PRs, single-commit ≤45 LOC, docs/CI)

| PR | Branch | Scope |
|----|--------|-------|
| [#7408](https://github.com/synaptent/aragora/pull/7408) | codex/b2-guard-expansion-criteria-5887 | docs(status): define B2 guard expansion criteria |
| [#7409](https://github.com/synaptent/aragora/pull/7409) | codex/ci-main-skip-telemetry-docs-salvage | docs(ci): explain main-branch skip telemetry |
| [#7410](https://github.com/synaptent/aragora/pull/7410) | codex/debate-test-shard-timeout | ci: extend debate test shard timeout |
| [#7411](https://github.com/synaptent/aragora/pull/7411) | codex/frontend-e2e-test-workflow-scope | fix(ci): scope test workflow changes to frontend e2e |
| [#7412](https://github.com/synaptent/aragora/pull/7412) | codex/harvest-github-cli-health-unused-sys-20260516 | fix(scripts): update github_cli_health.py |
| [#7413](https://github.com/synaptent/aragora/pull/7413) | codex/review-gate-missing-artifact-fail-closed | fix(ci): fail review gate on missing artifact |
| [#7414](https://github.com/synaptent/aragora/pull/7414) | codex/verify-claims-script-import-root | fix(epistemic): allow direct claim verifier script runs |

### Second batch (5 PRs, larger scope)

| PR | Branch | Scope |
|----|--------|-------|
| [#7416](https://github.com/synaptent/aragora/pull/7416) | worktree-agent-a02c5402 | docs(partners): refresh design partner brief |
| [#7417](https://github.com/synaptent/aragora/pull/7417) | worktree-agent-ad685921 | fix(ci): ubuntu-latest for deploy test jobs (ARM Python path fix) |
| [#7418](https://github.com/synaptent/aragora/pull/7418) | worktree-agent-add735a0 | docs(status): reflect 9 merged PMF product-loop PRs |
| [#7419](https://github.com/synaptent/aragora/pull/7419) | droid/Q10-dependabot-triage-2026-05-18 | docs(status): Q10 dependabot triage receipt |
| [#7420](https://github.com/synaptent/aragora/pull/7420) | codex/tmux-launcher-metadata-helper-current-base | fix(automation): avoid tmux prompt heredoc hangs |

### Review-packet PR (1)

| PR | Scope |
|----|-------|
| [#7415](https://github.com/synaptent/aragora/pull/7415) | Non-author review packets for #7389/#7396/#7397/#7398 (committed to main + opened as PR) |

## Disk recovery

| Phase | Removed | Disk |
|-------|---------|------|
| Empty wrappers in ~/.codex/worktrees | 153 | 33→32 GiB (rounding) |
| Bulk codex worktree removal (255 candidates) | 137 of 255 | 29→78 GiB (+49) |
| /private/tmp/aragora-* (21 candidates) | 3 of 21 | 55→58 (+3) |
| codex-auto worktrees (22 candidates) | 5 of 22 | 71→78 (+7 incl parallel) |
| Retry pass on bulk-removal failures | (in progress) | TBD |
| Retry-inspect on 900 errored paths | (in progress) | TBD |
| **TOTAL** | **~298 paths** | **33 → (target 80+) GiB** |

**Removable-but-failed:** ~252 paths inspected as removable but failed during `safe_worktree_cleanup.py remove`. Errors were predominantly empty-stderr — pattern of git-lock contention (concurrent worktree_maintainer + stale preflight processes). Operator can retry after clearing stale processes.

## What's still on disk by size

(captured at session midpoint, may have changed)

| Path | Size | Notes |
|------|------|-------|
| `~/.codex/worktrees/` | 377 GB → ~330 GB | 1308 → ~1024 dirs after removals |
| `~/.codex/sessions/` | 6.9 GB | Raw transcripts — protected per no-touch |
| `aragora/.worktrees/` | 21 GB | Repo-local managed worktrees (codex-auto + named) |
| `/private/tmp/` | 9.4 GB → ~11 GB | aragora-* dirs; 88 of 109 not removable per inspect |

## Tooling findings (potential issues to file)

1. **`safe_worktree_cleanup.py inspect` 20s timeout causes 78% error rate at 16-way parallelism.** 900 of 1155 inspections timed out. Recommend either:
   - Default longer timeout in the script
   - Add `--timeout` flag (currently hardcoded)
   - Document the safe parallelism ceiling (~6 concurrent based on retry behavior)
2. **`automation_pr_preflight.sh` hangs on git lock contention.** When stale processes hold `.git/index.lock`, the script waits indefinitely instead of failing fast. Recommend a lock-age check + timeout.
3. **`gh pr list` GraphQL 504 transient errors.** Already tracked by codex `salvage-github-connectivity-tokens-20260521` worktree.

## No-touch compliance

This session did NOT touch:
- #7292, #7385 (held)
- ADC PRs #7358/#7360/#7361/#7367/#7376 (handled by codex `harvest-adc-follow-on-deepening`)
- #7297 branch (merged; left dependent worktrees alone)
- Labels, branch deletions, force-push, --amend on pushed
- `~/.codex/sessions/` raw transcripts (protected)
- launchd, automation.toml
- Dependabot branches
- The R02-automation-disk-steward-20260521 codex lane (orphan claim, no live process)

## Recursive next prompt (paste-ready)

After this session, the operator should next do **non-author approval pass**:

```
Start from live truth in /Users/armand/Development/aragora. Use a NON-AUTHOR GitHub identity (NOT @an0mium).

Verify identity first:
gh api user --jq .login

If login is an0mium, STOP — this identity authored every queued PR and cannot self-approve. Operator must switch to bot account / co-maintainer.

If login is non-author:
1. Read docs/status/review_packets/INDEX.md for #7389/#7396/#7397/#7398 evidence.
2. For each of #7389, #7396, #7397, #7398: mark ready (if still draft), approve via:
   gh pr review <PR> --approve --body "Reviewed exact head per docs/status/review_packets/REVIEW_PACKET_<PR>.md"
3. Enable protected auto-merge:
   gh pr merge <PR> --auto --squash
4. After those 4 are queued, evaluate Bucket A candidates for #7408-#7414, #7416-#7420 (12 P102-harvest PRs) using the same packet pattern.
5. Do not approve PRs #7415 (still draft this session; will become useful only after #7389-#7398 land).

Hard non-touches: #7292, #7385, ADC PRs #7358/#7360/#7361/#7367/#7376, labels, branch deletion, launchd, automation.toml, raw transcripts, Dependabot branches.

After approvals, run:
- df -h /System/Volumes/Data
- python3 scripts/codex_worktree_autopilot.py cleanup --base main --ttl-hours 24 --no-delete-branches --json
- python3 scripts/sweep_stale_lane_claims.py --apply --json (to clear orphan R02 lane)

Report: identity, PRs approved/auto-merged, disk free delta, lanes cleaned.
```

## Lane release

P102 will be released `--status completed` at end of session.
