# Rescue productization — `blocked_auth_failure` scenario corpus

Lane: **P76 — productize blocked_auth_failure rescue class**

B0 (benchmark truth) reports that `blocked_auth_failure` accounts for ~25% of
recent ticks (7/28 on the rev-4 corpus; the production ledger holds 74 ticks
of the dominant shape). Per Operating Law — *"if humans intervene twice for
the same class of failure, the next system change should absorb that rescue
as product behavior"* — this lane productizes the class via a small,
fixture-driven scenario corpus that future B0 ticks can be matched against
without a per-tick human triage step.

## What this lane ships

Three new files, additive only:

| File | Purpose |
|------|---------|
| `docs/benchmarks/auth_failure_scenarios.json` | Canonical corpus of 5 auth-failure shapes (401 mid-tool-call, 403 quota exceeded, missing env var, expired-token refresh failed, vendor explicit block). Each scenario carries a descriptive `shape` block and a synthesizable `metrics_row` block. |
| `tests/benchmarks/test_auth_failure_scenarios.py` | Fixture-driven tests asserting each scenario's `metrics_row` folds into `TerminalClass.BLOCKED_AUTH_FAILURE` via the existing `classify_from_metrics`. Includes schema integrity + canonical trigger coverage tests. No worker subprocesses; no network calls; no AI key consumption. |
| `docs/benchmarks/rescue_productization_auth_failure.md` | This document. |

## Linkage

| Surface | How this corpus links in |
|---------|--------------------------|
| **Terminal-truth classifier** (`aragora.swarm.terminal_truth.classify_from_metrics`) | The test imports and invokes the live classifier with each scenario's `metrics_row`. Any classifier change that drops `BLOCKED_AUTH_FAILURE` mapping for these shapes fails the test. |
| **B0 status doc** (`docs/status/B0_BENCHMARK_TRUTH_STATUS.md`) | This corpus is the regression anchor for the `blocked_auth_failure` rescue class summarised there. P66 owns mutations of that status doc; this lane does not modify it. |
| **Rescue productization ledger** (`docs/benchmarks/rescue_productization.json`) | The ledger entry titled *"Productize blocked_auth_failure (B0 rev-4 largest failure class)"* (target `#7209`) is the durable record. This corpus and its test are the runtime proof that the entry's productization claim is honest. |
| **Rescue productization scorecard pipeline** (`scripts/publish_rescue_productization_report.py`) | The scorecard pipeline reads the ledger and emits the rescue-productization scorecard. This corpus adds a fixture-level audit trail under the same `class: blocked_auth_failure` key. The scorecard pipeline itself is unmodified by this lane (separate lane). |
| **Existing terminal-truth fixture** (`benchmarks/fixtures/swarm/terminal_truth/blocked_auth_failure.json`) and existing test (`tests/benchmarks/test_blocked_auth_failure_productization.py`) | These remain the canonical metrics-row corpus. The new `docs/benchmarks/auth_failure_scenarios.json` is the *scenario-level* corpus — it documents the human-readable trigger/error pattern alongside the synthesizable metrics row. Both layers cover the same five canonical shapes and stay in lock-step. |

## Why two corpora at two paths

* `benchmarks/fixtures/swarm/terminal_truth/blocked_auth_failure.json` is the
  metrics-row corpus consumed by the parametrized terminal-truth benchmark
  test that walks every class in the taxonomy.
* `docs/benchmarks/auth_failure_scenarios.json` is the scenario-level corpus
  that adds the descriptive `shape` block (`trigger`, `tool_name_pattern`,
  `error_pattern`, `agent_response_class`) so operator dashboards, the rescue
  productization scorecard, and incident triage can reference an auth-failure
  shape by a stable id without re-deriving it from raw metrics.

## Honesty / scope

What this lane did **not** do:

* No new classifier code. The existing `classify_from_metrics` (with its
  `"auth" in outcome` substring trigger) is the canonical surface; this lane
  reuses it.
* No mutation of `docs/status/B0_BENCHMARK_TRUTH_STATUS.md` (P66 owns it).
* No mutation of `scripts/publish_rescue_productization_report.py`
  (separate lane).
* No validation against a live B0 ledger run beyond
  `scripts/check_canonical_metrics.py --all --write-receipt` for regression
  containment. Live B0 corpus mutation belongs to the B0 ticker, not P76.
* No AI key consumption; no network calls; no workflow changes.

## Receipt

The lane receipt (SHA-256 + path) is recorded in
`docs/status/P76-blocked-auth-failure-productization_RECEIPT_<session-id>.md`
once landed and is referenced from the AGENT_FANOUT_JOURNAL row for this lane.
