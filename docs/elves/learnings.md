# Learnings — close-the-loop-20260701

> Append-only. Things discovered during the run that the next context window must know.

## Seeded at staging (from session + memory)

- **Advisory-settle drain boundary (Jul 1):** the queue DRAINS only with all 3 CI flags ON
  (severity-gated + tiered + advisory). Advisory [P2]/[P3] are non-blocking BUT non-counting —
  a PR still needs one real western-frontier PASS (Tier 1-2). Good PRs DO reach clean 2-0 after
  revisions (#8730 proved it). Don't over-claim "reviewers never both PASS".
- **Reviewer reliability:** claude CLI reviewer can hang; reliable pair = claude+openai; fallback
  grok/deepseek via OpenRouter (`ARAGORA_ENABLE_OPENROUTER_REVIEWER_FALLBACK=1`). Codex is
  hardening collector preflight transport in #8726 — if that merges mid-run, the collector gains
  `--reviewer-timeout/--overall-timeout` flags; prefer them.
- **Historical Codex conductor freeze:** at staging, a long-running Codex lane owned #8726, #8720,
  and the timeout-family files (`aragora/swarm/quorum_evidence.py`,
  `aragora/cli/commands/review_queue.py`). Do not inherit that as current truth. Before every PR
  claim, re-check live owner/mailbox state; heads moved mid-cycle twice during this run.
- **Evidence-lane self-owner trap:** your own evidence session can show as a "competing owner" —
  verify the pane vs `$TMUX_PANE` before treating a PR as owned.
- **Tier-4 --merge-apply 403:** use `ARAGORA_DISABLE_GITHUB_APP_TOKEN=1`; quorum check needs a
  rerun after `--settle-only`.
- **Stale .pyc trap:** `NameError: field` in openclaw_adapter.py →
  `find aragora/knowledge/mound/adapters -name __pycache__ -exec rm -rf {} +`.
- **Test collection:** `tests/connectors/chat/test_telegram.py` has a pre-existing collection
  error → add `--ignore=tests/connectors` when running broad slices.
- **PR triage policy:** keep unless actively bad; closures need recorded rationale. The B6 cleanup
  authority comes from the operator's G1/G2 sign-off and is scoped to the cleanup plan's own
  classifications — not a general license to close PRs.
- **mypy truth:** repo baseline is `.mypy-baseline` (3,115 lines at staging); current full run
  2,646 errors → headroom exists; the gate is "no NEW errors above baseline", not zero.
