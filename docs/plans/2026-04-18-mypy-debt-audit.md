# mypy pre-push debt audit (2026-04-18)

## Context

The pre-push hook in `.pre-commit-config.yaml` ran mypy against `aragora/`
and `scripts/`. A whole-tree run on `origin/main` (commit `331ae714d`,
mypy `v1.10.0`, `--config-file=pyproject.toml --ignore-missing-imports`)
produces **4,142 errors across 806 files** (4,286 source files checked).

Because every push of any Python file under those directories triggered the
full 4k-error report, the gate always failed. Every Codex automation and
human contributor has been using `--no-verify` to bypass, which means mypy
was providing zero value while eroding trust in the pre-push contract.

This note captures the debt shape we're baselining and the approach we're
adopting.

## Debt distribution

### Top 15 files

```
 422 scripts/nomic_loop.py
 239 aragora/nomic/dev_receipts.py
 133 aragora/cli/commands/swarm.py
  86 aragora/debate/phases/consensus_phase.py
  85 aragora/server/handlers/social/notifications.py
  44 aragora/debate/phases/winner_selector.py
  41 aragora/server/handlers/social/telegram/commands.py
  37 aragora/knowledge/mound/core.py
  35 aragora/server/handlers/workspace/settings.py
  34 aragora/knowledge/mound/redis_cache.py
  33 scripts/demo_control_plane.py
  33 aragora/server/handlers/workspace/policies.py
  31 aragora/server/handlers/workspace/members.py
  30 aragora/server/handlers/external_integrations.py
  28 scripts/self_develop.py
```

The top 5 files alone account for ~23% of the debt. `scripts/nomic_loop.py`
(422) and `aragora/nomic/dev_receipts.py` (239) together contribute 16%.

### Distribution by top-level subsystem

```
 787 scripts/
 383 aragora/server/handlers/
 299 aragora/nomic/
 277 aragora/debate/phases/
 190 aragora/cli/commands/
 122 aragora/server/handlers/workspace/
  96 aragora/server/handlers/social/
  76 aragora/knowledge/mound/
  76 aragora/debate/
  73 aragora/server/
  ... (long tail across 90+ directories)
```

Debt is **spread broadly** across the codebase. No single subsystem can be
cheaply excluded via a filter; excluding `scripts/` alone (787 errors, the
largest chunk) would still leave ~3,300 errors in `aragora/`. Most
`aragora/` subdirectories have at least a few errors, so narrowing the
regex filter only delays the problem.

### Distribution by error code

```
1313 [union-attr]
 782 [arg-type]
 529 [attr-defined]
 470 [misc]
 461 [assignment]
 158 [return-value]
 112 [index]
  72 [operator]
  56 [var-annotated]
  38 [call-arg]
  31 [dict-item]
  24 [call-overload]
  22 [no-redef]
  20 [override]
  19 [valid-type]
  17 [list-item]
   6 [name-defined]
   5 [empty-body]
   2 [typeddict-item]
   1 [unused-coroutine,type-var,has-type,exit-return,abstract] (1 each)
```

The top 3 codes — `union-attr`, `arg-type`, `attr-defined` — are ~64% of
the debt. These are the classic pre-narrowed-Optional / wrong-arg / missing
attribute patterns that accumulate when a fast-moving codebase adds
features without backfilling annotations.

### Stability across runs

mypy **error** output is byte-identical across repeated runs (0 diff
between two full-tree runs). **Note** output (severity `note:`) contains
some nondeterministic phrasing (overload resolution hints use `__init__`
vs `dict` inconsistently). For this reason the baseline mechanism ignores
the `note` category entirely (both in sync and filter stages).

## Approach chosen: (A) full baseline via mypy-baseline

Decision: go with the baseline approach, not filter narrowing.

Why:

1. **Debt is too broad to filter.** Any `files:` regex that excludes the
   noisy subsystems still leaves thousands of errors in the "clean" set.
2. **Per-file mypy is strictly worse than whole-tree mypy.** Pre-existing
   errors would still fire when the author merely *touched* any file that
   imports a type from a debt-heavy module.
3. **mypy output is stable for errors.** That makes baselining reliable.
4. **Turn-key tooling exists.** `mypy-baseline` (PyPI, MIT licensed) does
   exactly what we need: sync + filter with stats, supports `--allow-unsynced`
   for the ergonomics we want, and integrates cleanly in pre-commit.

### Implementation sketch

* Add `.mypy-baseline` at the repo root (4,142 baseline entries, sorted).
* Swap the pre-commit hook from `mirrors-mypy` to a local hook that runs
  `scripts/ci/mypy_with_baseline.py`, which:
  * runs `python -m mypy aragora/ scripts/ --config-file=pyproject.toml
    --ignore-missing-imports` on the whole tree (pass_filenames: false);
  * pipes stdout+stderr into `python -m mypy_baseline filter
    --baseline-path .mypy-baseline --allow-unsynced --ignore-categories note`;
  * returns the filter's exit code (0 if no new violations, >0 otherwise).
* Pin the hook's `additional_dependencies` to `mypy==1.10.0` and
  `mypy-baseline==0.7.4` so baseline reproducibility is version-locked.
* Expose `--sync` via the same wrapper so future baseline refreshes are
  one command.

### Ergonomics

* Contributors no longer need `--no-verify` for pre-push on unrelated debt.
* A deliberate new error (e.g. wrong return type) fails the hook with a
  clear "new: N" / "unresolved: 4142" summary.
* `--allow-unsynced` means accidental fixes to baselined errors don't fail
  the hook; we just reconcile the baseline on cadence.

### Followups (out of scope for this PR)

* **Shrink baseline deliberately.** Pick the top-offender files
  (`scripts/nomic_loop.py`, `aragora/nomic/dev_receipts.py`,
  `aragora/cli/commands/swarm.py`) and submit typing PRs that `mypy-baseline
  sync` can shrink. The `--allow-unsynced` flag lets that happen
  incrementally.
* **Wire a weekly reconciliation job.** A scheduled GH Action can run
  `python scripts/ci/mypy_with_baseline.py --sync && git diff --exit-code
  .mypy-baseline` to detect drift and open a maintenance PR.
* **Consider promoting specific directories to "strict" mypy** once they
  reach zero errors (override-module rules in `pyproject.toml`).

## Maintenance commands

```bash
# Regenerate baseline after intentionally clearing errors
python scripts/ci/mypy_with_baseline.py --sync
git add .mypy-baseline
git commit -m "chore(mypy): refresh baseline after typing cleanup"

# Run the hook manually without staging a push
pre-commit run mypy-baseline --hook-stage pre-push --all-files

# Inspect current new-error summary
python scripts/ci/mypy_with_baseline.py
```
