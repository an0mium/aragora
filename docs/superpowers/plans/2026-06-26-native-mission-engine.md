# Native Mission Engine Plan

## Summary

Build a Factory-like Aragora Mission Engine on top of the existing `aragora/missions`
spine. The engine inventories messy work, classifies it, safely drains what is
drainable, parks what needs humans, and only then admits new production work.

Autonomy is staged:

- `report`: dry-run classification only.
- `safe-clean`: authorize only fresh, clean, already-merged cleanup candidates.
- `auto-drain`: additionally authorize Tier 0-2 exact-head PR drain candidates.

Tier 3/4 settlement, branch protection changes, admin merges, and uncertain cleanup
always park with an operator receipt.

## Key Changes

- Make `aragora/missions` the canonical mission runtime.
- Add `aragora mission seed|status|run|resume|reconcile` while keeping
  `aragora mission <goal>` as a seed alias.
- Extend `Feature` with forward-compatible `metadata`.
- Add reconciliation categories and admission control.
- Add validation feature injection and operator escalation receipts.
- Add a live `FleetGate` adapter that reads Aragora merge-gate state and only
  merges through exact-head, non-admin GitHub CLI calls.

## Test Plan

- Mission state metadata round-trip and older-state compatibility.
- CLI parser and command coverage for seed/status/run/reconcile.
- Preserve-first reconciliation classification and autonomy modes.
- Admission control under unresolved backlog pressure.
- Validation injection and operator receipt structure.
- Live gate exact-head/non-admin merge command construction.
