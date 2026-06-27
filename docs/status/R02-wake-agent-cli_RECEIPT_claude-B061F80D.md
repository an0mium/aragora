# Receipt — R02-wake-agent-cli

**Session:** `claude-B061F80D`
**Lane:** `R02-wake-agent-cli`
**Branch:** `claude/R02-wake-agent-cli-20260519-040146`
**PR:** [#7348](https://github.com/synaptent/aragora/pull/7348) (draft)
**Outcome:** `shipped`
**Plan reference:** PR #7327 (P54) — Phase 2
**Depends-on:** PR #7336 (R01 contact_method) — degrades gracefully if R01 unmerged
**Bounded budget:** ≤45 min · Actual: ~30 min

## Acceptance

| Item | Status | Evidence |
|---|---|---|
| `scripts/wake_agent.sh` ships | ✅ | 230 LoC bash |
| Reads `contact_method` from lane registry via Phase A reader | ✅ | `identify_lane_owner.py --lane-id --json` with raw-registry fallback |
| Backend dispatcher with switch on contact_method | ✅ | tmux:, mailbox-only:, osascript:* (stub), factory-api:* (stub) |
| Default `--dry-run` (fail-closed) | ✅ | Per spec; --apply opts in |
| `--fallback {mailbox-only,fail}` policy | ✅ | mailbox-only default |
| Dispatch receipt with schema_version "aragora-wake-agent-receipt/1.0" | ✅ | Written to `.aragora/dispatch-receipts/<utc>-<lane>-<sha8>.json` |
| SHA-256-bound prompt | ✅ | `prompt_sha256` field |
| Exit code matrix | ✅ | 0/1/2/3/4 documented in script header |
| ≥8 tests | ✅ (24) | tests/scripts/test_wake_agent.py |
| Pure bash + stdlib Python | ✅ | No new pip deps |
| Degrades gracefully without R01 | ✅ | Missing contact_method → mailbox-only |
| Draft PR via `gh pr create --draft` | ✅ | #7348 |
| `[lane: R02-wake-agent-cli]` commit tag | ✅ | commit `d5da938658` |

## Tests (24 new)

```
$ python3 -m pytest tests/scripts/test_wake_agent.py -q
........................                                                 [100%]
24 passed in 6.59s
```

| Group | # | Coverage |
|---|---|---|
| Arg validation | 5 | --lane required, --prompt required, mutex, priority/fallback enums |
| Lane resolution | 1 | lane-not-found → exit 2 |
| Backend selection | 6 | tmux dry-run, mailbox dry-run, missing-method default, missing+fail exit 3, unknown→fallback, unknown+fail exit 3 |
| Receipt schema | 3 | SHA-256 binding, schema_version, persistence path |
| Apply mode | 3 | tmux success, tmux failure → exit 4, mailbox success |
| Priority/JSON wiring | 5 | low/normal/high/blocking + human-readable output |
| Prompt-file path | 1 | multi-line file content + correct SHA |

## Validation

```
$ python3 -m ruff check scripts/wake_agent.sh tests/scripts/test_wake_agent.py
All checks passed!

$ python3 -m ruff format --check tests/scripts/test_wake_agent.py
1 file already formatted

$ bash scripts/automation_pr_preflight.sh origin/main HEAD
preflight: ok
```

## Non-Touches

- No protected files (`CLAUDE.md`, `aragora/__init__.py`, `docs/AGENT_OPERATING_CONTRACT.md`, `.env`, `scripts/nomic_loop.py`).
- No `automation.toml` edits (writer pause I did earlier was reverted by operator; not re-touched).
- No labels, no `boss-ready`, no `autonomous` markers.
- No mark-ready transition — stays draft.
- No merges, no launchd installs.
- No held-PR mutations.
- No edits to other agents' active lane branches.

## Cross-references

- **Reach plan**: PR #7327 (draft, P54), Phase 2 section in `docs/governance/AGENT_DISPATCH_REACH_PLAN.md` (which is on the P54 branch, not yet on main).
- **R01**: PR #7336 (draft, contact_method field) — landing unblocks the tmux/mailbox auto-population path but R02 functions without it.
- **R03 follow-on**: posted findings comment on #7327 documenting the `codex app-server proxy --sock` IPC layer as the canonical R03 implementation path (replaces the original osascript-only plan).

## Operational context

- Disk recovery during this session: 47.8 → 104 Gi free (codex's
  background `codex_worktree_recovery --apply` cleared ~56 GiB; >80 Gi
  writer-resume threshold met).
- TW03 freshness restored in parallel commit `bb39676048`.
