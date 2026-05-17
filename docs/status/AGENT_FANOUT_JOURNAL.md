# Agent Fan-out Journal

Append-only ledger of autonomous agent sessions running the v2 (and later) idempotent fan-out prompt. One row per session, written by the session itself at Phase 4 receipt time. Read this file first when starting a new fan-out session to avoid duplicating recently completed work.

Columns: `timestamp_utc | session_id | agent_family | phase_id | pr_number | outcome`

`outcome` ∈ {`shipped`, `finish-existing`, `deferred`, `no-work`, `conflict`}

---

2026-05-17T16:56:00Z | droid-DC5A5821 | droid | P03-lane-registry-claim-helper-rebase | 7267 | finish-existing

2026-05-17T17:23:00Z | droid-A5312D6A | droid | P05-publication-freshness-probe-rebase | 7261 | finish-existing
2026-05-17T17:23:00Z | droid-A5312D6A | prompt-bug: v3 heredoc shim '/tmp/fanout_claim.py <<PYEOF ... PYEOF' still hangs in test shell; v4 must ship shim as tracked scripts/ file (see P12 deferral)
