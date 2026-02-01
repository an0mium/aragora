# Nomic Evaluation Harness

This document describes the evaluation harness that compares Nomic multi-agent
output against a single-agent baseline.

## Goals

- Provide repeatable A/B comparisons for the same task.
- Capture diffs, verification status, and basic metrics in JSON.
- Keep runs isolated using git worktrees.

## How it works

The harness runs each task in an isolated git worktree:

- **single**: single-agent baseline (Codex-only implement path)
- **multi**: normal Nomic multi-agent run
- **shadow**: runs both and compares results

Each run writes logs and a `result.json`. A combined `report.json` is produced.

## Usage

```bash
python scripts/nomic_eval.py --tasks examples/nomic_eval_tasks.json --mode shadow
```

Optional flags:

- `--task-id TASK_ID` to run a single task
- `--single-agent codex` to change the baseline agent
- `--timeout 3600` to cap each run
- `--context-timeout 600` to cap the context phase
- `--skip-codex-context` to skip Codex during context gathering
- `--skip-gemini` / `--skip-grok` to skip providers without keys
- `--cleanup` to remove worktrees after runs

## Output

Results are written under `.nomic/eval/`:

- `task_id/single/<timestamp>/result.json`
- `task_id/multi/<timestamp>/result.json`
- `report.json`

The report includes a `rubric_template` and each task includes a
`manual_review` section for human scoring.

## Notes

- The harness does **not** auto-commit or push.
- Use `--cleanup` only after reviewing artifacts.
- The comparison in `report.json` is intentionally conservative and should be
  combined with human review for architectural quality.
