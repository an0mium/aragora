# P28 (worktree inventory refresh) receipt

- Session: `droid-3D81079C`
- Lane: `P28-worktree-inventory-refresh`
- Branch: `droid/P28-worktree-inventory-refresh-20260518-050330`
- PR: none (hygiene phase; no PR per v9 prompt)
- Started: 2026-05-18T05:03:30Z
- Completed: 2026-05-18T05:13:00Z (approximate)
- Outcome: **shipped**

## Acceptance against v9 P28 spec

| Spec | Implementation |
|---|---|
| Refresh `docs/status/generated/worktree_value_inventory/latest.json` | New `latest.json` at 2026-05-18T05:10:14Z (was 2026-05-17T04:08:00Z; 24.9 h refresh) |
| Run `codex_worktree_value_inventory.py --json > /tmp/inv.json` | Done with `--root .worktrees/codex-auto --skip-gh --size-mode none --git-timeout 5` to fit budget |
| Run `publish_worktree_value_inventory.py --input <inv>` | Done; wrote 116 575 byte JSON + status doc |
| Receipt + journal + release lane | This file + journal append + lane status=completed |
| Bounded ≤ 10 min | 9.5 min wall clock (run started 05:03:30Z, finished 05:13:00Z; inventory generation itself ~22 s) |

## Inventory diff

| Metric | Before | After | Δ |
|---|---|---|---|
| Total candidates | 45 | 56 | +11 |
| `cleanup_candidate` | 5 | **7** | +2 |
| `harvest_candidate` | 28 | **34** | +6 |
| `preserve` | 12 | 15 | +3 |
| Active sessions | 12 | 12 | 0 |
| Registered worktrees | 30 | 41 | +11 |

## Tests

Hygiene phase; no new code; no unit tests written. The publisher
script is already unit-tested in
`tests/scripts/test_publish_worktree_value_inventory.py` (existing).
Ruff is N/A.

## CI

No PR; no CI run for this lane. The published artifact will be
picked up by recurring publication-freshness probes
(`scripts/publication_freshness_probe.py`).

## Defense-in-depth observations

- `--input` flow used as recommended by the publisher's docstring
  ("the inventory script can be slow when the legacy
  `~/.codex/worktrees` root has thousands of entries, so capturing
  once and publishing afterwards is the recommended flow").
- Dry-run was executed first to verify schema before mutating the
  published artifact.
- The publisher is documented as "intentionally additive: it does
  not run cleanup, harvest, or any GitHub-mutating action; it does
  not import any aragora subpackage; it stays a pure stdlib script;
  it never deletes prior dated artifacts" — confirmed by listing
  `docs/status/generated/worktree_value_inventory/` before + after.
  Prior dated artifact `worktree-value-inventory-20260517T040800Z.json`
  is still present alongside the new one.
- The legacy `~/.codex/worktrees` root does not exist on this
  machine; the `du` step would otherwise traverse it and burn the
  wall-clock budget.

## Scope notes for v10

1. **Renumber to P40**: this lane shares "P28" with claude-79AAF84B's
   already-shipped `P28-A-identify-lane-owner`. v10 should rename the
   worktree-inventory-refresh phase to a clean number, leaving P28-A/
   B/C/D for the lane-owner family.
2. **Default-scan recipe**: codify the constrained scan args
   (`--root .worktrees/codex-auto --skip-gh --size-mode none
   --git-timeout 5`) in the v10 P28-equivalent prompt body. The
   default `--root` flag traverses the legacy directory (even when
   absent it walks every codex-auto subtree with size-mode `du`),
   which doesn't fit a 10-minute budget on this machine.
3. **Inventory-driven harvest follow-up**: with 34 fresh
   harvest_candidates, sibling agents can claim P30 lanes with the
   pattern `P30-harvest-<branch-suffix>`. The `top_unique_unharvested`
   slice of the published JSON is a ready-made worklist.

## Lane

`P28-worktree-inventory-refresh` released at session close
(status `completed`, branch recorded; no PR).

## Bytes-recoverable accounting

This phase does NOT free disk space directly; it produces the data
that downstream P29/P30/P31/P32 phases use to act safely. Expected
downstream impact: ~1–2 G freeable from the 7 cleanup_candidates,
plus ~2.5 G from the 4 legacy top-level worktree dirs (P32) once
that lane runs.
