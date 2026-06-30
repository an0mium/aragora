# Stage-Gate Conductor Log Target Contract

Use this contract in any Stage-Gate Conductor prompt or automation that writes
the recurring audit log.

## Canonical Target

- The canonical Stage-Gate Conductor Log is issue
  [#8671](https://github.com/synaptent/aragora/issues/8671).
- Prefer an open log issue labeled `stage-gate-log-canonical` when that label is
  present on exactly one matching issue.
- If multiple `[automation] Stage-Gate Conductor Log` issues exist and no
  canonical label is present, fall back only to #8671.
- If neither a unique canonical label nor #8671 is present, fail closed and
  report the ambiguity. Do not create a new log issue.

## Required Resolver

Before posting a log comment, resolve the target with the repo helper:

```python
from aragora.ops.stage_gate_conductor_log import (
    build_gh_issue_comment_args,
    build_gh_issue_list_args,
    resolve_stage_gate_conductor_log_issue,
)
```

Fetch open candidates with `build_gh_issue_list_args()`, pass the returned issue
objects to `resolve_stage_gate_conductor_log_issue()`, then comment on the
resolved number. Never choose the log by age, comment count, oldest issue,
newest issue, or raw GitHub search ordering.

If the resolver raises `StageGateLogResolutionError`, include the candidate
numbers in the operator brief and stop without opening another log issue.
