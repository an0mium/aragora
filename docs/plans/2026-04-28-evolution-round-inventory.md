# 2026-04-28 Evolution Round — Phase 0 Inventory

**Branch:** `docs/2026-04-28-evolution-round`
**Time:** 02:40 local
**Status:** complete

## API key availability

| Provider | Allowlisted agent | Available |
|---|---|---|
| Anthropic | `anthropic-api`, `claude` (CLI) | NO |
| OpenAI | `openai-api`, `codex` (CLI), `openai` (CLI) | NO |
| OpenRouter | `deepseek`, `llama`, `mistral`, `qwen`, `kimi`, ... | NO |
| Google | `gemini`, `gemini-cli` | YES (env) |
| xAI | `grok`, `grok-cli` | YES (.env + env) |
| Mistral direct | `mistral-api`, `codestral` (opt-in) | YES (.env) |
| Built-in | `demo` | YES (always) |

**Verdict:** debate is feasible with `gemini` + `grok` + `demo` (3 agents, two real LLMs and one offline judge). This is enough for a real Arena debate with majority consensus.

## Aragora subsystem importability (read-only spot-checks)

From `feat/handoff-contract-module-skeleton` worktree:
- `aragora.swarm.handoff_contract`: ok
- `aragora.knowledge.mound.metrics_health_bridge`: ok
- `aragora.debate.Arena`, `DebateProtocol`: ok
- `aragora.agents.create_agent`, `list_available_agents`: ok
- `aragora.core.Environment`: ok

From main:
- `aragora.swarm.handoff_contract`: NOT YET (PR #6785 unmerged) — expected.
- `aragora.knowledge.mound.metrics_health_bridge`: NOT YET (PR #6786 unmerged) — expected.

## Live outbox dogfood (handoff_contract module against real data)

Outbox path: `.aragora/automation-outbox/`

```
entries: 2
  - open-pr-codex-boss-loop-open-pr-maxed-retry-17ab4dff2.json
      -> HandoffIdentity(branch=codex/boss-loop-open-pr-maxed-retry,
                         action=open_pr, fingerprint=551d1cd7...)
  - open-pr-codex-review-queue-baseline-parser-6f2869ab0.json
      -> HandoffIdentity(branch=codex/review-queue-baseline-parser,
                         action=open_pr, fingerprint=06fd2afc...)
```

Reconcile plan from empty `SatisfactionContext`:
- `archive_count`: 0
- `skip_count`: 2
- `publish_count`: 0
- `is_dry_run_safe`: True

**Both live entries parse cleanly. Zero `InvalidHandoff` results. Module behaves identically on real outbox data as on the synthetic test fixture.** This is the first end-to-end validation of PR #6785 against production data.

Note on outbox state: both entries are Codex-owned, not Droid-owned. The
Droid outbox-clean carve-out is honored. The `codex/review-queue-baseline-parser`
entry corresponds to Codex's open PR #6783; reconciliation of that
state requires the open-PR list (which the contract module's
`SatisfactionContext.open_pr_heads` accommodates but we deliberately
did not populate in this read-only check).

## Candidate next-PR universe (6 options)

| Source | Proposal | Estimated LOC | Pillar |
|---|---|---|---|
| Worker A | Agent-readable receipts envelope | ~250 | P8 + P3 |
| Worker B | `CruxReceipt` unification bridge | ~250 | P5 + dialectical thesis |
| Worker C | Backlog-audit classifier consolidation | ~250 | P7 (rescue churn) |
| Spec #6785 follow-up | Migrate one legacy script to delegate to handoff_contract | ~200 | P5 |
| Spec P3 | Bridge Run Inspector first PR | ~350 | P3 |
| Spec P4 | Wire `KMMetricsHealthBridge` into postgres-store factory + add `aragora km status` CLI | ~150 | P4 |

## Halt readiness

- Local Arena debate is in-process only; no swarm dispatch.
- ≤$5 token budget (2 rounds, 3 agents, terse prompts).
- All artifacts persist to `docs/plans/` on this branch only; no production receipt store writes.
- If Phase 1 API call fails, fall back to Worker B's recommendation silently.

Phase 0 complete. Proceeding to Phase 1.
