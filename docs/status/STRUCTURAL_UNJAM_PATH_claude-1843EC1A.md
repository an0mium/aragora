# Structural Unjam Path — `an0mium` Author Block + Publisher Rate Limit

**Generated:** 2026-05-21T21:08Z by claude-1843EC1A (lane P104-unjam-coordination)
**Audience:** Operator + any future agent session inheriting this constraint

## TL;DR

The local-machine PR queue cannot drain past 4 ready PRs without operator intervention. Two compounding factors:

1. **Every PR opened from this machine has `author=an0mium`** (the operator's gh login). Per `docs/REVIEW_AUTHORITY_PRINCIPLES.md` (and the soon-to-be-enforcing `aragora-merge-quorum` workflow in draft PR #7423), bot/same-author approvals are not authorized. **No agent session running on this machine can satisfy `REVIEW_REQUIRED`.**
2. **`publish_codex_automation_branches.py` is rate-limited** with `--max-open-prs 4`. The publisher checks the open-ready PR count each cycle and skips new pushes when at limit. With 4 ready + 47 draft + 51 open, the publisher is at its ceiling — every cycle it sleeps and retries.

Result: codex automations produce work, queue it locally (outbox + worktree harvest ledger), but the work cannot reach merge. Every session inheriting the same gh login hits the same wall.

## Live state (verified 2026-05-21T21:00Z)

| Surface | State |
|---------|-------|
| `gh api user --jq .login` | `an0mium` |
| Open PRs | 51 |
| Draft PRs | 47 |
| Ready PRs | 4 (at publisher ceiling) |
| Held PRs | 9 (`#7173, #7215, #7240, #7243, #7245, #7249, #7252, #4990, #7209`) |
| Active lanes | 0 (after R03 stood down at 18:13 UTC) |
| Pending steering messages | 5 (4 stale + 1 fresh wake to R03 at 21:06 UTC) |
| Open codex outbox proposals | 3 (1 just resolved as `already_satisfied`) |
| Codex Desktop active automations | 17 (engineering-autopilot ×4, worktree-* ×4, etc.) |
| Codex CLI rollout files | 5,458 total; several updated within past hour |
| Codex `app_server` processes | 12 (one using 718% CPU — likely model inference) |
| Factory Droid helper processes | 7 (idle until this session dispatched audit at 21:07Z) |
| Recent commits to main (8h) | 14 — all `claude` (sessions P95-P104) |

## Why claude is the only family producing visible artifacts in this 8h window

Claude code sessions produce visible artifacts because they:
1. Claim a lane synchronously (`claim_active_agent_lane.py`)
2. Make and push commits directly
3. Open PRs immediately via `gh pr create`
4. Write status docs to `docs/status/` on main
5. Append journal rows for every phase

Codex automations follow a different lifecycle:
1. Cron triggers an automation
2. Automation does work locally (worktrees, outbox JSON)
3. Publishes only via `publish_codex_automation_branches.py` — which is rate-limited
4. Successful publication writes a `automation-receipts/already-satisfied-*.json`
5. Failed publications leave outbox entries dangling

When the publisher is at ceiling, every automation cycle for the 17 active automations turns into `writer_should_pause_for_branch_backlog=true` or `connectivity_failed` or `max-open-prs reached`. They write nothing visible to git but consume CPU. **This is the "loop producing more loop" failure mode the operator's substrate-freeze principle exists to prevent.**

Factory Droid produces zero artifacts because it requires explicit operator dispatch to start a mission, and none has been dispatched in the past 8h until claude-1843EC1A sent one at 21:07 UTC (this session, currently running).

## Three concrete unjam paths (in priority order)

### Path A — Configure a non-author GitHub identity (the only structural fix)

The root cause. Options:

1. **Bot account.** Create a second GitHub identity (e.g. `aragora-bot`) with reviewer access to `synaptent/aragora`. Configure `gh auth` with a separate `GH_CONFIG_DIR` so agents can `GH_CONFIG_DIR=~/.config/gh-bot gh ...` to act as the bot. Bot approvals are explicitly forbidden by `REVIEW_AUTHORITY_PRINCIPLES.md` for Tier 3-4, but for **Tier 0-2 docs-only changes** (the bulk of the queue) the model-quorum + bot-flip workflow could be authorized — this is what draft PR #7423 codifies.
2. **Second human reviewer.** Add a co-maintainer with reviewer access. Approvals from them satisfy the `REVIEW_REQUIRED` gate without violating the same-author rule.
3. **`aragora-merge-quorum` workflow on main.** Once PR #7423 lands, branch protection can stop requiring human review and instead require the merge-quorum check. The check evaluates the model quorum + head-SHA-bound operator settlement signal. This is the architecturally clean fix.

Path A1 + A3 together is the lowest-friction long-term path. A2 is the fastest manual unblock.

### Path B — Raise the publisher rate limit (band-aid, doesn't fix the wall)

`scripts/publish_codex_automation_branches.py` runs with `--max-open-prs 4`. Raising to 12 or 20 would let codex flush its current backlog into PRs. But those PRs would still hit the same `REVIEW_REQUIRED` wall — they'd just be drafts instead of staged outbox entries.

Useful only as a diagnostic — if codex pushes 8 more PRs, the queue grows to 59/55 draft and the same blocker dominates.

### Path C — Drain the held-PR list (operator-only)

The 9 held PRs are explicitly paused by the operator. Auditing them periodically and lifting holds when conditions clear would let agents act on a few. But the bulk of the blocker is not these — it's the entire `an0mium`-authored queue.

## What this session did inside the unjam

Lane `P104-unjam-coordination` (this session):

1. ✅ **Reconciled codex outbox duplicate** (`open-pr-codex-eu-ai-act-compliance-artifacts-primary-r2-...93c5c430.json`): the work was already covered by my PRs #7391/#7392, BUT codex had correctly caught a `file:///Users/armand/.claude/...` absolute-path leak in #7391's RECEIPT.md. Applied that sanitization to #7391 directly (commit `0855c00895`). Wrote `already-satisfied` receipt to stop codex from re-attempting. Codex's local branch `codex/eu-ai-act-compliance-artifacts-primary-r2-20260521` is safe to delete.
2. ✅ **Woke codex-R03** with high-priority steering message at `.aragora/operator-steering/codex-r03-.../2026-05-21T21-06-31-991Z-9b6a3335.json` summarizing the reconciliation and what's safe to pick up next.
3. ✅ **Dispatched Factory Droid** (PID 90915, `droid exec --auto low -m claude-opus-4-7 -r high`) for a read-only audit of the 17 codex desktop automations. Output to `/tmp/factory_codex_automation_audit.md`.
4. ✅ **Documented this structural blocker** (this file) so the operator + future agent sessions have a single canonical reference.

## What this session deliberately did NOT do

- Did **not** approve any PR (identity blocked, as documented).
- Did **not** flip any draft PR to ready (would burn CI for no progression).
- Did **not** kill any process (PID 59401/59446/59483 still holding `.git/index.lock` since 12:40 AM yesterday per prior session note — operator action).
- Did **not** touch the ADC chain, held PRs, Dependabot branches, or any other agent's worktrees.
- Did **not** edit the codex automation outbox entry directly (used the `automation-receipts/` ledger pattern instead).
- Did **not** raise the publisher rate limit (band-aid, doesn't fix the wall).

## Recommended next operator action

1. **(15 min)** Confirm whether a second GitHub identity is available or willing to be configured. If yes, configure it via `GH_CONFIG_DIR` and instruct future agent sessions to use it for review actions.
2. **(30 min)** Review + ready-flip PR #7423 (Tier 4: enforce model-review quorum). This is the architectural unjam — once on main, future review requirements can be satisfied by the model-quorum check rather than a second human.
3. **(parallel)** Triage the existing 4 ready PRs to merge (or close if not wanted). Each merged PR opens a slot in the publisher rate limit for codex to push the next backlog item.
4. **(if R03 doesn't respond to the steering message by 21:40 UTC)** Manually inspect / kill codex-R03 lane. The `R02-automation-disk-steward` lane has also been idle since 15:00 UTC — likely the same issue (orphan lane claim from a session that exited without releasing).

## Related artifacts shipped this session window

- This file: `docs/status/STRUCTURAL_UNJAM_PATH_claude-1843EC1A.md`
- Codex reconciliation receipt: `.aragora/automation-receipts/already-satisfied-codex-eu-ai-act-compliance-artifacts-primary-r2-20260521-93c5c430.json`
- Sanitization commit on #7391 branch: `0855c00895`
- R03 wake message: `.aragora/operator-steering/codex-r03-post-p102-harvest-followthrough-20260521T181200Z/2026-05-21T21-06-31-991Z-9b6a3335.json`
- Factory audit (in-progress, /tmp/factory_codex_automation_audit.md, PID 90915)
- Prior-session artifacts still relevant: `PACKET_7389_HANDOFF_claude-B06EEE32.md`, `P102_HARVEST_RECOVERY_RECEIPT_claude-51C05A58.md`, draft PR #7423 (Tier 4 governance)
