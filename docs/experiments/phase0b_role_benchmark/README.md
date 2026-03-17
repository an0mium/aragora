# Phase 0B Role Benchmark

This directory stores normalized results for the Codex/Claude planner, worker,
and reviewer permutation benchmark.

Expected artifacts:

- `active_run.json`: lock file for the single authoritative live benchmark run
- `results.json`: machine-readable benchmark rows
- `results.csv`: flat table for quick comparison and spreadsheet import
- `runs/<experiment_id>/<config_id>.json`: per-config runtime preparation records

Use `scripts/phase0b_role_benchmark.py` to prepare runtime manifests, enforce
one active benchmark lane at a time, and record finished runs into the
normalized result tables.
