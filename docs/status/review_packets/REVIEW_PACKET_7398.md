# Review Packet — PR #7398

## Header
- **PR:** [#7398 — fix(automation): qualify unavailable PR state in outbox reconcile](https://github.com/synaptent/aragora/pull/7398)
- **Author:** an0mium (non-reviewer for this packet operator)
- **Branch:** `codex/salvage-reconcile-open-pr-unknown-state-20260520` → `main`
- **Head SHA:** `cd4a8bbdf072c76aa1364d80d04da19888ebc7b3`
- **Created:** 2026-05-21T03:48:23Z
- **State:** Draft, BLOCKED (REVIEW_REQUIRED)

## Scope
- **LOC:** +52 / -2
- **Files (2):**
  - `scripts/reconcile_automation_outbox.py` (+10/-2) — distinguishes "no open PR" from "open-PR state unavailable" in dry-run keep-reason text; un-discards the `open_pr_state_available` boolean that was previously thrown away
  - `tests/scripts/test_reconcile_automation_outbox.py` (+42/-0) — one new regression test
- **Areas touched:** outbox reconciler dry-run output. No mutation paths changed; `still_protecting_active_work` counter behavior is unchanged.
- **Single own commit:** `cd4a8bbdf0 fix(automation): qualify unavailable PR state in outbox reconcile`

## What the change does
Before: when GitHub was unhealthy, the reconciler kept the outbox handoff but reported the misleading reason `"branch has unique commits not on main, no open PR — actively protecting"`. There is no way to distinguish that from a genuine no-open-PR case.

After: when `open_pr_state_available is False`, the reason becomes `"branch has unique commits not on main, open PR state is unavailable — actively protecting"`. Behavior (keep vs. archive) is unchanged; this is an output-clarity fix.

Mechanism: replaces `_open_pr_state_available` (underscore-discarded) with `open_pr_state_available`, then branches the reason string.

## Check Status
| Check | Conclusion |
|---|---|
| lint | SUCCESS |
| typecheck | SUCCESS |
| Generate & Validate (OpenAPI) | SUCCESS |
| TypeScript SDK Type Check | SUCCESS |
| sdk-parity | SUCCESS |
| Aragora Code Review / PR Review | SUCCESS |
| Auto PR Publisher | SUCCESS |
| PR Scope (Tests) | SUCCESS |
| Test shards / Core Suites / Module Tier Drift / Release Readiness | SKIPPED |

No failures, no pending checks.

## Validation Evidence
- `git diff --check origin/main...origin/<branch>` → exit 0 (whitespace clean).
- Inline preflight-equivalent: `changed_files=2`, `forbidden_files=none`, `rescue_publish_files=none`, `docs_only=false`, `source_without_tests=false`, `whitespace=clean`. (Direct `automation_pr_preflight.sh` invocation was blocked on git lock contention from concurrent worktree-maintainer/reconcile/preflight processes — not a property of this PR.)
- PR body documents: `pytest -q tests/scripts/test_reconcile_automation_outbox.py`, ruff check/format, mypy — author claims all clean.
- New test `test_unique_branch_keep_reason_notes_unavailable_open_pr_state` asserts both presence of `"open PR state is unavailable"` and absence of the misleading `"no open PR"` substring.

## Recommendation
**SAFE to approve at head `cd4a8bbd`.** Minimal, surgical change: 10 lines in one function in one script that only affect dry-run output text — no decision (keep vs. archive) logic changes. The `counts["still_protecting_active_work"] += 1` line is unchanged; the path is unchanged; only the reason string is now conditional. Test directly asserts the new behavior. No protected files, workflows, automation.toml, or coordination artifacts touched. A non-author reviewer can approve once they confirm the diff context: line 640's `_open_pr_state_available` → `open_pr_state_available` is the only behavioral wire change and it just preserves a value that was already returned by `load_open_pr_state()`.
