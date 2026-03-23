# Next Steps (Canonical)

Last updated: 2026-03-23

This is the single source of truth for short-horizon execution priorities.
[CANONICAL_GOALS](../CANONICAL_GOALS.md) defines what Aragora is and why.
[ARAGORA_EVOLUTION_ROADMAP](../plans/ARAGORA_EVOLUTION_ROADMAP.md) defines the long-range architecture and moat.
[FEATURE_GAP_LIST](../FEATURE_GAP_LIST.md) is the capability and backlog truth.
[ACTIVE_EXECUTION_ISSUES](ACTIVE_EXECUTION_ISSUES.md) maps the live GitHub issue set and the current doc-driven PMF program.
[PMF_DOGFOOD_EXECUTION_PLAN](../plans/PMF_DOGFOOD_EXECUTION_PLAN.md) is the operator runbook for the next live proof.

## Current Reality

- Historical program epics still matter as lineage even though they are no longer the live gate: [#804](https://github.com/synaptent/aragora/issues/804), [#805](https://github.com/synaptent/aragora/issues/805), and [#806](https://github.com/synaptent/aragora/issues/806).
- `main` now contains the structural product-loop slices that were missing earlier in March: ProviderRouter-backed debate selection, KM retrieval and writeback, versioned API-key endpoints, the onboarding/get-started flow, truthful dashboard/integrations state, live demo wiring, queue/workbench productization, and quickstart fail-closed behavior.
- Current repo proof is materially stronger than the previous docs claimed. On current `main`, this focused verification passes:

  ```bash
  python3 -m pytest tests/e2e/test_user_journey.py tests/cli/test_quickstart.py -q
  ```

  Result on March 23, 2026: `57 passed` in `33.75s`.

- That proof matters, but it is still a controlled proof:
  - `tests/e2e/test_user_journey.py` validates a mocked end-to-end founder loop
  - `tests/cli/test_quickstart.py` validates the quickstart contract and fail-closed behavior
- It does **not** prove that the live founder loop is ready for external users without operator babysitting.
- GitHub's open issue set is no longer a truthful PMF map. As of March 23, 2026, the only open issues are enterprise-assurance items: [#273](https://github.com/synaptent/aragora/issues/273), [#274](https://github.com/synaptent/aragora/issues/274), and [#509](https://github.com/synaptent/aragora/issues/509).
- Therefore the next live lane is **dogfood-first PMF proof**, not design-partner GTM, not sales, and not more generic substrate work.

## Execution Order

### 1) Prove The Canonical Founder Loop Live

- Run the live founder loop from the actual product surfaces and CLI path:
  - health / readiness
  - API-key or provider readiness
  - live debate execution
  - structured receipt save + inspect + verify
  - visible result surface
  - KM ingestion or a truthful explicit stop
- Use [PMF_DOGFOOD_EXECUTION_PLAN](../plans/PMF_DOGFOOD_EXECUTION_PLAN.md) as the runbook and acceptance checklist.
- Treat this as the current gate for PMF. Do not claim "operational" or "design-partner ready" until this live proof is repeatable.

### 2) Convert Live Failures Into Bounded PMF Blockers

- If the founder loop fails, capture the exact command transcript, observed stop condition, and affected surface.
- Reopen or create GitHub issues only after reproducing a concrete live blocker.
- Do not revive broad umbrella issues without current evidence.

### 3) Use Idea-To-Execution / Nomic Only On Those PMF Blockers

- Compile PMF blocker sources through:

  ```bash
  python3 -m aragora.cli.main pipeline dogfood \
    --source-file docs/plans/PMF_DOGFOOD_EXECUTION_PLAN.md \
    --output-dir .aragora/dogfood/pmf \
    --max-goals 3 \
    --budget-limit 10 \
    --time-limit-hours 4 \
    --json
  ```

- The pipeline is now a product-completion tool, not a reason to widen scope.
- Every generated lane must map to a founder-loop acceptance failure or a direct blocker to that failure.

### 4) Run Bounded Swarm / Ralph Repair Lanes

- Keep execution narrow, evidence-backed, and human-gated.
- Prefer one blocker tranche at a time.
- Re-run the founder loop after each landed blocker tranche before widening scope.

### 5) Dogfood The Second Workflow Only After The Founder Loop Holds

- Once the founder loop is repeatable, dogfood the inbox trust wedge and adjacent real-user workflows.
- The second workflow exists to test retention and repeatability, not to bypass the founder-loop gate.

### 6) Design Partner Outreach Comes After Repeatable Live Proof

- The right sales point is not "the repo is large" or "all epics are closed."
- The right sales point is a clean, repeatable founder loop with truthful receipts and bounded operator recovery when something fails.

### 7) Enterprise Assurance Remains Parked

- [#273](https://github.com/synaptent/aragora/issues/273), [#274](https://github.com/synaptent/aragora/issues/274), and [#509](https://github.com/synaptent/aragora/issues/509) are real work, but they follow PMF proof rather than precede it.

## Operating Rules

- Closed PMF issues do not equal live PMF proof.
- No new infra or orchestration lane is justified unless it maps directly to a founder-loop acceptance gap.
- No document should say "operational," "complete," or "ready for sales" unless live dogfood evidence supports that claim.
- The PMF backlog should be reconstituted from observed live failures, not from stale issue trees.
- GitHub issues still matter, but until the PMF blocker set is recreated truthfully, these docs are the current execution map.
