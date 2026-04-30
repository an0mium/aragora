---
prompt_id: mse_07_authoritative_sources
class: multi_seeded_error
seeded_errors:
  - description: "Lists 'aragora/swarm/multi_agent_dialog.py outputs' as an authoritative source for #6375 baseline. The Round 30f spec §2.2 explicitly forbids this — heterogeneous-dialog transcripts are not settled human-decision evidence."
    category: factual
    verification_ref: "docs/plans/2026-04-30f-round-spec.md §2.2"
  - description: "Claims (1) and (4) tied means (1) wins. Spec actually says (4) wins (authoritative GitHub state)."
    category: rule-direction
    verification_ref: "docs/plans/2026-04-30f-round-spec.md §2.2"
expected_flags: 6
verification_refs:
  - "docs/plans/2026-04-30f-round-spec.md"
---

The Round 30f spec §2.2 names four authoritative event sources for #6375 baseline measurement, scanned in order:

1. `.aragora/overnight/boss_metrics.jsonl` (primary settlement evidence).
2. `.aragora/review-queue/briefs/*.json` (operator-reviewed briefs).
3. `.aragora/evolve-round/*/dogfood/unstick-receipts/applied.jsonl` (boss-loop unstick records).
4. `aragora/swarm/multi_agent_dialog.py` outputs (heterogeneous-dialog transcripts, since these capture multi-model consensus on settlement decisions).
5. GitHub PR/issue timeline.

If (1) and (5) disagree on outcome, (1) wins (it is the local source of truth).

Review for accuracy. List any errors you find.
