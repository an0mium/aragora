# Operator Authorization Queue: 2026-07-05

Generated: 2026-07-05T22:37:15Z

## Context

This packet collects the current operator-blocked Aragora conductor items into
one queue. It does not authorize or perform any merge, settlement, evidence
posting, workflow rerun, PR close, owner release, outbox archive, or workflow
edit.

Live state used for this packet:

- `origin/main`: `e85873d2998edce4cbfc2bfddb57249410837d3f`
- Publisher: ready, `outbox=19`, `cache=19`
- Fleet sentinel breaches: `outbox_depth`, `stale_terminal_owner`
- Reconcile dry-run: zero archive-ready actions; all 19 outbox items remain
  protected active work
- PR `#8900`: draft, `MERGEABLE`/`CLEAN`, required checks passing
- PR `#8879`: non-draft, `MERGEABLE`/`BLOCKED`; `aragora-merge-quorum` fails
  because it is Tier 4, touches `aragora/swarm/quorum_evidence.py` and
  `tests/swarm/test_quorum_evidence.py`, and still needs model quorum / human
  settlement path

## Pending Rulings

| Priority | Link | Requested Action | One-word Reply |
| --- | --- | --- | --- |
| P1 | https://github.com/synaptent/aragora/pull/8900 | Decide whether to mark PR #8900 ready for review. It is docs-only and classifies the current `enforce-main-pr-policy` red on `origin/main` as a false positive for merged PR #8897 caused by commit-to-PR association lag. | `READY-8900` |
| P1 | https://github.com/synaptent/aragora/actions/runs/28756606214 | Authorize a separate workflow-hardening branch for `.github/workflows/branch-discipline.yml` so `enforce-main-pr-policy` can tolerate immediate post-merge commit-to-PR association lag without turning `main` red. Workflow edits are protected/governance-adjacent and should be reviewed separately. | `APPROVE-WORKFLOW-FIX` |
| P1 | https://github.com/synaptent/aragora/pull/8536 | Authorize the supported stale-terminal-owner resolver for PR #8536 / lane `branch-salvage-blocker-8536-httpx-restack-r2-20260622T1939Z`, after rerunning dry-run and confirming `terminal_safety_blockers=[]`. | `APPLY-8536` |
| P2 | https://github.com/synaptent/aragora/pull/8879 | Record an explicit steering outcome for lane `codex-pr8879-tier4-evidence-20260705T174839Z` before claiming clean routing. Current lane state says prepared Claude/OpenAI evidence is clean, but the canonical Tier-4 helper forced `action=prepare`; it needs repo-supported human publication path or explicit raw-comment publication authorization. | `RECORD-8879` |
| P2 | outbox-depth://current | Confirm that no direct outbox archive should run while reconcile reports zero archive-ready actions. The next outbox-depth movement must be a supported representation/owner/human action, not a blind archive. | `DEFER-OUTBOX` |

## Exact Commands Or Actions

### PR #8900: mark ready if approved

```bash
GH_TOKEN="$(python3 scripts/gh_app_env.py --print-token --quiet)" \
ARAGORA_GITHUB_AUTH_SOURCE=github_app_installation \
gh pr ready 8900
```

Expected receipt after action: PR `#8900` becomes non-draft and normal review /
merge-quorum gates continue. No merge is implied by this action.

### Branch discipline: workflow hardening if approved

Create a separate branch and draft PR that changes only
`.github/workflows/branch-discipline.yml` and focused tests/docs needed to prove
the race condition. The likely implementation is a bounded retry or exact
PR-number fallback when a fresh merge commit has not yet appeared in the
commit-to-PR association API.

Expected receipt after action: a draft PR exists with the workflow patch and
test evidence. No workflow rerun or branch-protection mutation is implied by
approval to create the branch.

### PR #8536: stale terminal owner resolver if approved

Rerun the dry-run first:

```bash
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
python3 scripts/resolve_lane_conflicts.py --merged-pr-lane-audit --pr 8536 \
  --registry-path "$REPO_ROOT/.aragora/agent-bridge/lanes.json" \
  --receipt-dir "$REPO_ROOT/.aragora/agent-bridge/conflict-resolution-receipts" \
  --heartbeat-path "$REPO_ROOT/.aragora/agent-bridge/heartbeats.json" \
  --steering-inbox-root "$REPO_ROOT/.aragora/operator-steering" \
  --heartbeat-fresh-seconds 900 --json
```

Only if the dry-run still reports `terminal_safety_blockers=[]`, apply:

```bash
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
python3 scripts/resolve_lane_conflicts.py --merged-pr-lane-audit --pr 8536 \
  --expected-closed-at 2026-06-30T06:09:47Z \
  --expected-head-sha 1ffcc9d91ecaaec512891ba9dd69046227b232b2 \
  --operator-authorized \
  --registry-path "$REPO_ROOT/.aragora/agent-bridge/lanes.json" \
  --receipt-dir "$REPO_ROOT/.aragora/agent-bridge/conflict-resolution-receipts" \
  --heartbeat-path "$REPO_ROOT/.aragora/agent-bridge/heartbeats.json" \
  --steering-inbox-root "$REPO_ROOT/.aragora/operator-steering" \
  --heartbeat-fresh-seconds 900 --apply --json
```

Expected receipt after action: PR `#8536` disappears from the first
`stale_terminal_owner` candidates, and
`scripts/fleet_sentinel.py --json --no-ledger` shows one fewer stale terminal
owner.

### PR #8879 lane outcome if approved

One safe explicit outcome is to keep the lane blocked but make the steering
outcome machine-visible:

```bash
python3 scripts/claim_active_agent_lane.py \
  --lane-id codex-pr8879-tier4-evidence-20260705T174839Z \
  --owner-session codex-pr8879-tier4-evidence-20260705T174839Z \
  --status blocked \
  --pr-number 8879 \
  --branch codex/pr8811-adjudication-stall-salvage-20260705 \
  --worktree "$HOME/.codex/worktrees/pr8811-leftover-salvage-20260705/aragora" \
  --goal "mark PR #8879 ready and collect authorized Tier-4 evidence" \
  --source codex \
  --next-action "waiting for repo-supported Tier-4 human settlement/publication path or explicit raw-comment publication authorization" \
  --last-steering-outcome held \
  --json
```

Expected receipt after action:
`scripts/agent_bridge.py operator-snapshot --json --summary-only` no longer
reports `lane_missing_steering_outcome` for
`codex-pr8879-tier4-evidence-20260705T174839Z`.

### Outbox depth

No direct archive command is listed because the live reconciler currently
reports zero archive-ready actions. The next outbox-depth movement should be one
of:

- a representation mutation proved by the handoff-state/reconciler tools,
- an owner/human release that makes a protected item archive-ready, or
- a supported stale-owner resolver that reduces queue-health noise before the
next archive attempt.

Expected receipt after action: a future reconcile dry-run exposes a specific
archive-ready item, or fleet sentinel reports a lower actionable/gated queue
depth after a supported owner/representation transition.
