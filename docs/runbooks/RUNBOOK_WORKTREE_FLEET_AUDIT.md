# Worktree Fleet Audit Runbook

Use this runbook to classify Aragora side worktrees before any cleanup or
harvest action. It is fleet-scoped: it covers the mixed `.worktrees/`,
`.claude/worktrees/`, `.codex/worktrees/`, and `/private/tmp/aragora-*`
population. For release-only hygiene, use
[`RELEASE_WORKTREE_HYGIENE.md`](RELEASE_WORKTREE_HYGIENE.md).

This runbook is intentionally conservative. A worktree that is clean but has
commits not reachable from `origin/main` is a harvest candidate, not a cleanup
candidate. A worktree with tracked or untracked work and no live owner signal
is blocked for human review, not removable.

## Read-Only Audit

Run the audit from the shared checkout. These commands must not modify any
worktree.

```bash
git fetch origin --prune
git worktree list --porcelain
git -C <worktree-path> status --porcelain=v1 --untracked-files=all
git -C <worktree-path> log -1 --format='%H %cr'
git merge-base --is-ancestor <worktree-head> origin/main
```

For each worktree, record:

- path
- namespace (`/private/tmp`, `.codex/worktrees`, `.claude/worktrees`,
  repo `.worktrees`, or repo root)
- branch name or detached state
- `HEAD` SHA
- tracked and untracked dirty status from
  `status --porcelain=v1 --untracked-files=all`
- last commit age
- whether `HEAD` is an ancestor of `origin/main`
- any open PR attached to the branch
- active-session signal files such as `.codex_session_active`,
  `.claude-session-active`, `.nomic-session-active`, or
  `.codex_session_meta.json`

## Classification Rubric

Classify every worktree into exactly one class.

| Class | Criteria | Action |
| --- | --- | --- |
| `KEEP` | Root worktree, active-session worktree, dirty or untracked work with an active owner signal, or recent branch tied to an open PR/epic. | Leave it alone and re-check owner state before any future action. |
| `HARVEST` | Clean worktree whose `HEAD` is not an ancestor of `origin/main`. | Preserve it until its unique commits are mapped to an open PR, branch handoff, or explicit retirement. |
| `STALE-CLEAN` | Clean worktree whose `HEAD` is an ancestor of `origin/main`, with last commit older than 24 hours. | Candidate for later helper-mediated removal, after a final `inspect` pass. |
| `BLOCKED` | Tracked changes or untracked files without a visible active owner signal. | Human review required; list dirty and untracked paths and do not remove. |

When signals conflict, choose the safer class in this order:

`BLOCKED` over `STALE-CLEAN`, `HARVEST` over `STALE-CLEAN`, and `KEEP` over
cleanup if there is a live owner or open PR.

## Safe Commands

Inspect a single worktree before removal:

```bash
python3 scripts/safe_worktree_cleanup.py --repo "$(git rev-parse --show-toplevel)" inspect <worktree-path>
```

Remove only after the helper says the worktree is safe:

```bash
python3 scripts/safe_worktree_cleanup.py --repo "$(git rev-parse --show-toplevel)" remove <worktree-path>
```

Clean managed Codex worktrees through the autopilot helper while preserving
branch refs for later harvest or retirement review:

```bash
python3 scripts/codex_worktree_autopilot.py cleanup --base main --ttl-hours 24 --no-delete-branches
```

Use branch deletion only as a separate, explicit retirement action after the
branch has no open PR, no unique commits that need harvest, and no active owner.

Prune stale Git worktree metadata only after helper-mediated removals:

```bash
git worktree prune --expire=now
```

Never use ad-hoc deletion such as:

```bash
find .worktrees -type d -maxdepth 1 -exec rm -rf {} +
rm -rf /private/tmp/aragora-*
```

Raw deletion bypasses active-session locks, tracked and untracked work checks,
open-PR checks, and unique-commit preservation.

## 2026-07-07 Snapshot

Read-only audit source:

- repo: `$HOME/Development/aragora`
- `origin/main`: `94d96322d0c967f4a84e6132f2251c39af059e99`
- total registered worktrees: 176
- no worktrees, branches, or PR state were removed or modified

### Counts By Class

| Class | Count |
| --- | ---: |
| `KEEP` | 9 |
| `HARVEST` | 151 |
| `STALE-CLEAN` | 10 |
| `BLOCKED` | 6 |

### Counts By Namespace

| Namespace | KEEP | HARVEST | STALE-CLEAN | BLOCKED |
| --- | ---: | ---: | ---: | ---: |
| `/private/tmp` | 4 | 65 | 3 | 2 |
| `.codex/worktrees` | 1 | 23 | 1 | 3 |
| `.claude/worktrees` | 1 | 40 | 4 | 0 |
| repo `.worktrees` | 2 | 23 | 2 | 1 |
| repo root | 1 | 0 | 0 | 0 |

The high `HARVEST` count is expected in a high-churn queue: clean side
worktrees often contain branch-local commits not reachable from `origin/main`.
Those are preservation candidates until mapped to a PR, merged branch, explicit
handoff, or retirement receipt.

### Blocked Worktrees

These worktrees had tracked dirty files and no active-session signal visible to
the read-only audit. Future audits must also block untracked-only worktrees.
Do not remove any blocked worktree without human review or a later owner
release.

| Path | Branch | Dirty tracked paths |
| --- | --- | --- |
| `/private/tmp/aragora-8920-p2-fixes-20260707-1416` | detached | `aragora/live/next-env.d.ts` |
| `/private/tmp/aragora-8948-repair-20260707T132000Z` | detached | `scripts/publish_automation_handoffs.py` |
| `$HOME/.codex/worktrees/8519-github-event-resolver-repair-20260704/aragora` | `codex/8519-github-event-resolver-repair-20260704` | `aragora/prediction/github_event_resolver.py`, `aragora/prediction/stakeable_claim.py`, `docs/METRICS.md`, `tests/prediction/test_github_event_resolver.py`, `tests/prediction/test_stakeable_claim.py` |
| `$HOME/.codex/worktrees/adjudicator-wiring-step1/aragora` | `codex/adjudicator-wiring-step1` | `aragora/swarm/quorum_evidence.py`, `tests/swarm/test_quorum_evidence.py` |
| `$HOME/.codex/worktrees/claude-openai-m0-adjudicator/aragora` | `codex/claude-openai-m0-adjudicator` | `.github/workflows/aragora-merge-quorum.yml`, `AGENTS.md`, `aragora/cli/commands/review_queue.py`, `aragora/cli/parser.py`, `aragora/config/legacy.py`, `aragora/config/settings.py`, `aragora/core/decision_types.py`, `aragora/live/src/app/(app)/features/page.tsx`, `aragora/live/src/components/BootSequence.tsx`, `aragora/live/src/components/__tests__/DebateInput.test.tsx`, `aragora/live/src/components/debate-viewer/TranscriptMessageCard.tsx`, `aragora/live/src/components/debate-viewer/__tests__/TranscriptMessageCard.test.tsx`, `aragora/live/src/components/settings-panel/types.ts`, `aragora/live/src/components/settings/types.ts`, `aragora/live/src/config.ts`, `aragora/live/src/store/settingsStore.ts`, `aragora/nomic/debate_profile.py`, `aragora/server/handlers/platform_config.py`, `aragora/swarm/quorum_evidence.py`, `scripts/auto_evidence_cycle.py`, `scripts/settle_pr.py`, `tests/cli/commands/test_review_queue.py`, `tests/config/test_settings.py`, `tests/debate/test_default_profile.py`, `tests/handlers/test_platform_config.py`, `tests/mcp/test_mcp_integration.py`, `tests/nomic/test_debate_profile.py`, `tests/swarm/test_quorum_evidence.py`, `tests/workflows/test_aragora_merge_quorum_workflow.py` |
| `$HOME/Development/aragora/.worktrees/codex-8726-preflight-repair-20260701` | detached | `aragora/swarm/quorum_evidence.py`, `tests/swarm/test_quorum_evidence.py` |

### Stale-Clean Examples

These are examples only. Re-run `safe_worktree_cleanup.py inspect` immediately
before any removal, and re-run the audit with
`status --porcelain=v1 --untracked-files=all` so untracked-only work cannot be
misclassified as stale-clean.

| Path | Branch | Last commit age |
| --- | --- | --- |
| `/private/tmp/aragora-8913-rerun-20260706T142154Z` | detached | 28 hours ago |
| `/private/tmp/aragora-prompt-handoff-pull-20260706` | `codex/prompt-handoff-pull-queue-20260706T1552Z` | 27 hours ago |
| `/private/tmp/aragora-transport-settlement-20260706T185959Z` | detached | 26 hours ago |
| `$HOME/.codex/worktrees/pr8460-main-baseline/aragora` | detached | 6 days ago |
| `$HOME/Development/aragora/.claude/worktrees/lane-8773/.worktrees/preflight-preflight-20260702-182336` | `preflight/20260702-182336` | 7 days ago |

### Anomalies To Review

- Many clean `/private/tmp/aragora-*` worktrees are detached but not ancestors
  of `origin/main`; classify these as `HARVEST` until their commits are mapped.
- Several blocked worktrees are detached and dirty. They need owner or human
  review because there is no branch name to use as the first preservation key.
- Multiple namespaces carry cleanup-relevant worktrees, so cleanup must not
  assume that `.codex/worktrees` is the whole fleet.
