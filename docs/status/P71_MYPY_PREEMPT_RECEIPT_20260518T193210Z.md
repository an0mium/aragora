# P71 — Pre-push mypy preempt helper (Receipt)

**Lane:** P71-mypy-preempt-helper
**Session:** droid-C4A021F3
**Timestamp (UTC):** 2026-05-18T19:32:10Z
**Branch:** codex/droid-20260518-192925-38cd85b1
**Worktree:** ~/aragora/.worktrees/codex-auto/droid-20260518-192925-38cd85b1

## Deliverables

- `scripts/preflight_mypy.sh` (executable, +x). POSIX-friendly bash; macOS bash
  3.2 compatible. Uses `git diff --name-only <base>...HEAD -- '*.py'`,
  prints a skip message when the set is empty, otherwise runs `mypy --pretty`
  on the changed files using the repo's existing `[tool.mypy]` config in
  `pyproject.toml`. Supports `--diff-base <ref>` (default `origin/main`),
  `--diff-base=<ref>`, and `-h|--help`.
- `docs/dev/MYPY_PREFLIGHT.md` — rationale, command, and Phase 2 integration
  suggestion. Out-of-scope items (pre-push hook auto-installation, cross-platform
  polish) are flagged for a future lane.

No mypy config changes, no new dependencies, no CI workflow changes.

## Smoke run 1 — no python diff vs `origin/main`

```
$ git diff --name-only origin/main...HEAD -- '*.py'
(empty)

$ bash scripts/preflight_mypy.sh
no python changes; mypy preflight skipped
exit=0
```

## Smoke run 2 — temporary file with a deliberate type error

Steps:

1. Wrote `_tmp_mypy_smoke.py` containing two intentional mypy errors
   (`return "not an int"` from a `-> int` function and an
   `add_one("also wrong")` call expecting `int`).
2. Staged + committed it on the lane branch so the `origin/main...HEAD` diff
   would include it: `wip: smoke-test bad file (will revert)`.
3. Ran the preflight.
4. Reset the temp commit with `git reset --hard HEAD~1`; verified the file is
   gone (`ls _tmp_mypy_smoke.py` → No such file or directory).

Output (verbatim):

```
$ bash scripts/preflight_mypy.sh
preflight_mypy: origin/main...HEAD changed python files:
  _tmp_mypy_smoke.py

_tmp_mypy_smoke.py:11: error: Incompatible return value type (got "str",
expected "int")  [return-value]
        return "not an int"
               ^~~~~~~~~~~~
_tmp_mypy_smoke.py:14: error: Argument 1 to "add_one" has incompatible type
"str"; expected "int"  [arg-type]
    result: int = add_one("also wrong")
                          ^~~~~~~~~~~~
Found 2 errors in 1 file (checked 1 source file)

preflight_mypy: mypy reported issues (exit 1).
hint:
  - run `mypy --pretty <file>` locally on the file(s) above to iterate,
  - or narrow with `mypy --pretty --show-error-codes <file>` for triage,
  - then re-run `scripts/preflight_mypy.sh` before pushing.
exit=1
```

Both errors are surfaced; the script's exit code matches mypy's exit code (1).

## Lint

`shellcheck` not present in this environment (`which shellcheck` → empty,
`brew list shellcheck` → No such keg). Per the lane spec ("if installed"),
shellcheck was skipped. Script was kept POSIX-friendly: no GNU-only flags,
no arrays, no `mapfile`, and explicit `IFS= read -r` loops fed by heredocs.

## Environment

- `mypy 1.19.1 (compiled: yes)` from `~/.pyenv/shims/mypy`.
- `git 2.48.1`, macOS Darwin 25.2.0.
- Lane registry claim acquired via
  `scripts/claim_active_agent_lane.py --lane-id P71-mypy-preempt-helper
  --owner-session <uuid> --status active --json`.

## Acceptance checklist

- [x] `scripts/preflight_mypy.sh` exists and is executable.
- [x] `docs/dev/MYPY_PREFLIGHT.md` exists (1–30 line dev snippet).
- [x] Smoke run 1 — no diff → exit 0 with skip message.
- [x] Smoke run 2 — temp `*.py` with type error → non-zero exit pinpointing
      the offending file; temp file reverted before commit.
- [x] No mypy config, dependency, or CI workflow changes.
