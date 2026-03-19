# Boss Lane Manifest Prototype

This document preserves the first tranche manifest that was assembled by hand while Boss-loop dogfooding was still being operated manually.

The machine-readable example is [docs/examples/boss-lane-manifest-2026-03-19.yaml](../examples/boss-lane-manifest-2026-03-19.yaml). It captures the exact planning shape that was needed for the first bounded live proof:

- `#1061` as the already-landed one-tick dispatch fix
- `#1065` as the runtime hardening gate
- `#1060` as an explicitly replaced PR, not something to merge unchanged
- `#1064` as the bounded `boss-loop-test` issue
- `#873` and `#909` as retired stale targets

Current main has since moved past that snapshot: `#1065` and `#1066` are merged, and `#1064` is closed. That is intentional. The manifest remains useful because `swarm tranche inspect` can now compare the captured tranche snapshot against live GitHub and local repo state and report the drift explicitly.

## Why This Shape

The prototype is intentionally narrow:

- `references` records the external PR and issue facts the tranche depends on
- `gates` turns those references into explicit prerequisites
- `lanes` records ownership, write scope, verification commands, stop conditions, and expected artifacts

That is enough to make the current boss process reproducible without jumping straight to full autonomous multi-lane execution.

## Current Use

The first tranche-v1 implementation uses this artifact in three ways:

1. loader and validation for the manifest structure
2. read-only tranche inspection against live GitHub state
3. artifact persistence for lane-level proofs, forensics, and runbooks

## Next Increment

The next useful step after this prototype is not auto-dispatch. It is a narrow claim/prepare capability that:

1. refuses read-only lanes
2. creates a managed worktree for a writable lane
3. records write-scope ownership through the existing coordination lease store

That keeps the control plane honest before any broader automation is attempted.
