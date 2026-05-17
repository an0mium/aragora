# Agent Fan-out Journal

Append-only ledger of autonomous agent sessions running the v2 (and later) idempotent fan-out prompt. One row per session, written by the session itself at Phase 4 receipt time. Read this file first when starting a new fan-out session to avoid duplicating recently completed work.

Columns: `timestamp_utc | session_id | agent_family | phase_id | pr_number | outcome`

`outcome` ∈ {`shipped`, `finish-existing`, `deferred`, `no-work`, `conflict`}

---

2026-05-17T16:56:00Z | droid-DC5A5821 | droid | P03-lane-registry-claim-helper-rebase | 7267 | finish-existing

2026-05-17T17:23:00Z | droid-A5312D6A | droid | P05-publication-freshness-probe-rebase | 7261 | finish-existing
2026-05-17T17:23:00Z | droid-A5312D6A | prompt-bug: v3 heredoc shim '/tmp/fanout_claim.py <<PYEOF ... PYEOF' still hangs in test shell; v4 must ship shim as tracked scripts/ file (see P12 deferral)
2026-05-17T20:53:00Z | droid-6916BE6B | droid | P02-freshness-probe-rerun | 7287 | shipped
2026-05-17T20:53:00Z | droid-6916BE6B | prompt-bug: v4 lists scripts/triage_open_prs.py as required observer but doesn't say what to do when it isn't yet on main (it's still in PR #7285); fell back to manual bucket classification against OPERATOR_DELEGATION_POLICY.md
