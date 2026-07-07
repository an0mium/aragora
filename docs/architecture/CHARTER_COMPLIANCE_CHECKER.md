# Charter Compliance Checker

`scripts/check_charter_compliance.py` is advisory tooling for issue #8942. It
reads `docs/architecture/charters.yaml` and a Git diff, then reports citable
`CHR-*` and related `ARCH-*` rows when a change appears to re-add a chartered
removed surface, excluded placement, or parked/pending surface that is
machine-checkable from added lines.

Example:

```bash
python scripts/check_charter_compliance.py --range origin/main...HEAD
python scripts/check_charter_compliance.py --range origin/main...HEAD --format json
```

While the architecture charter remains `DRAFT`, binding violations are limited
to rows that bind in draft: `CHR-P4A-001`, `CHR-P4A-002`, `CHR-P4A-003`,
`CHR-P4A-004`, `CHR-X-007`, and any row explicitly marked
`binding_in_draft: true`. Other detectable rows are reported as proposed-only
violations so reviewers can see the future charter pressure without treating it
as ratified policy.

`symbols` and `kept_symbols` provide symbol-level granularity. A path-level
PARK row can name kept symbols that should not false-positive while the rest of
the path remains parked or proposed. `CHR-X-040` uses this for the
`aragora.control_plane.registry` health/liveness API consumed by
`aragora/debate/team_selector.py`.

This checker is not wired into CI, branch protection, merge gates, settlement,
or evidence collection. Any required-check integration needs separate operator
approval.
