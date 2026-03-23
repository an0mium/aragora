# PMF Dogfood Execution Plan

Last updated: 2026-03-23

This is the bounded operator plan for using Aragora to finish Aragora's PMF-critical product loop.

## Objective

Prove one repeatable founder loop on current `main`, then use Aragora's own pipeline, queue, and review machinery to clear only the blockers exposed by that proof.

The target is not "more autonomy." The target is one user-visible path that works truthfully:

`readiness -> live question -> live debate -> receipt -> visible result -> KM ingestion -> explicit next step`

## Current Evidence On `main`

These structural slices are already landed:

- ProviderRouter-backed runtime debate selection
- KM retrieval into debate context and KM writeback from outcomes
- versioned API-key management endpoints
- onboarding/get-started product flow
- truthful integrations and dashboard surfaces
- quickstart TLS fail-fast behavior
- quickstart structured receipts and inline provider-key support
- real demo/backend wiring

Current focused proof on `main`:

```bash
python3 -m pytest tests/e2e/test_user_journey.py tests/cli/test_quickstart.py -q
```

Observed result on March 23, 2026:

- `57 passed`
- runtime: `33.75s`

Interpretation:

- the mocked founder loop is verified
- the quickstart contract is verified
- the live founder loop is **not yet** proven by this alone

## Canonical Founder-Loop Acceptance Checklist

The founder loop passes only when all items below are true in one bounded live run:

1. **Readiness is explicit**
   - health/readiness endpoints respond truthfully
   - credential/provider state is explicit before the run starts
2. **Quickstart enters the live path or fails closed quickly**
   - no silent fallback to demo
   - if TLS/credentials are broken, the command exits promptly with a direct reason
3. **A live debate completes or stops truthfully**
   - no hanging operator path
   - no hidden manual rescue
4. **A structured receipt is saved**
   - the receipt is inspectable and verifiable from the CLI
5. **The result is visible on a product surface**
   - CLI output alone is not enough
   - the relevant UI or API result surface must reflect the debate truthfully
6. **KM ingestion or KM stop condition is explicit**
   - either the outcome is written/read as designed
   - or the system returns a bounded, truthful stop
7. **Operator noise is bounded**
   - no wall of irrelevant retry/circuit-breaker noise
   - the final message must tell the operator what happened and what to do next

## Strict Execution Order

### 1) Re-run The Controlled Baseline

Use the current mocked proof before touching the live path:

```bash
python3 -m pytest tests/e2e/test_user_journey.py tests/cli/test_quickstart.py -q
```

### 2) Run The Live Founder Loop

Use local secure-store hydration or explicit one-shot provider injection. Do not rely on ambient shell exports.

Suggested bounded live run:

```bash
ARAGORA_USER_ID=an0mium \
python3 -m aragora.cli.main quickstart \
  --question "Should Aragora use its own dogfood pipeline to close the remaining PMF gaps?" \
  --no-browser
```

Then verify the emitted receipt:

```bash
python3 -m aragora.cli.main receipt inspect .aragora/receipts/quickstart-live-receipt.json
python3 -m aragora.cli.main receipt verify .aragora/receipts/quickstart-live-receipt.json
```

If the live path uses a product/API surface directly, also capture the matching visible result endpoint or UI evidence.

### 3) If The Run Fails, Capture A Truthful Blocker

Record:

- exact command
- runtime
- stop condition
- affected surface
- whether the failure is:
  - credentials / readiness
  - provider/TLS
  - runtime debate execution
  - receipt/result visibility
  - KM ingestion
  - noisy but truthful
  - noisy and untruthful

### 4) Compile Only Those Blockers Through The Pipeline

Use this plan file as the source input:

```bash
python3 -m aragora.cli.main pipeline dogfood \
  --source-file docs/plans/PMF_DOGFOOD_EXECUTION_PLAN.md \
  --output-dir .aragora/dogfood/pmf \
  --max-goals 3 \
  --budget-limit 10 \
  --time-limit-hours 4 \
  --max-parallel-ready-projects 1 \
  --planner-model claude \
  --worker-model codex \
  --review-model claude \
  --json
```

Review the generated objectives and manifest before any execution.

### 5) Run Bounded Repair Lanes

Only after the generated blocker list matches the observed live failures:

```bash
python3 -m aragora.cli.main swarm tranche inspect \
  --manifest .aragora/dogfood/pmf/campaign_manifest.yaml \
  --json
```

If the manifest is clean and narrow, start the supervisor under the normal human merge gate.

### 6) Re-run The Founder Loop After Every Landed Blocker Slice

The founder loop is the only scorecard that matters for this phase.

## Stop Conditions

Stop and re-plan if any of the following happen:

- the generated blocker list does not map to the observed founder-loop failure
- the queue proposes generic infra work not tied to the founder loop
- a lane widens into repo cleanup, enterprise certification, or unrelated product work
- the live proof cannot be reproduced with an exact command transcript

## Out Of Scope

These are explicitly not the current lane:

- broad repo cleanup
- enterprise assurance / pentest work
- design partner outreach
- Stripe / revenue operations
- new autonomy infrastructure not tied to a founder-loop failure

## Reporting Contract

Every dogfood or repair lane should report:

- exact commands run
- PR/issue/doc URL if created
- residual risk
