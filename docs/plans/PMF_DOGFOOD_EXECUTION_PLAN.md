# PMF Dogfood Execution Plan

Last updated: 2026-03-25

This is the bounded operator plan for using Aragora to finish Aragora's PMF-critical product loop.

## Objective

~~Prove one repeatable founder loop on current `main`.~~ **ACHIEVED** (March 24, 2026).

The founder loop is proven repeatable: 5/5 consecutive live runs, 35-62s range, all producing valid receipts visible via API/dashboard.

The current objective is to **dogfood the second workflow** (inbox trust wedge) and **prepare for design partner outreach**.

## Founder Metrics Tree

The founder scorecard should answer one question in decision order:

1. are external users completing consequential workflows and trusting the
   output enough to come back
2. if not, which execution bottleneck is preventing that proof
3. which numbers are merely narrative garnish and should not move roadmap

### Proof Metrics

These are the only metrics that should directly change roadmap priority,
resource allocation, or go-to-market confidence.

#### Root proof metric

- **Weekly trusted workflow completions**: number of design-partner or
  founder-operated runs that reach one of three truthful outcomes:
  accepted action taken, accepted deliverable produced, or explicit
  needs-human stop with the next action clear.

#### Proof branches

- **Repeat usage**
  - active design partners who run more than one consequential workflow in a
    week
  - percent of users who come back for a second real task within 7 days
- **Time-to-value**
  - median time from auth/setup to first trusted visible result with receipt
  - median time from starting a real inbox or review workflow to terminal truth
- **Outcome advantage**
  - measurable delta versus the incumbent path on quality, risk caught, or
    time saved
  - examples: review findings caught vs single-model baseline, inbox triage
    actions accepted vs manually deferred
- **Workflow-specific PMF proof**
  - founder loop remains green as a baseline acceptance gate
  - inbox trust wedge live runs on a real inbox
  - design-partner workflows completed without hidden operator rescue
- **Commercial pull**
  - design partners who request continued use, commit to a pilot, or supply
    more consequential workflows after the first proof

If proof metrics improve, keep investing. If execution metrics improve but
proof stays flat, the team is polishing plumbing instead of proving demand.

### Execution Metrics

These are leading indicators and diagnostics. They matter only insofar as they
lift the proof metrics above.

- **Default journey success**
  - live founder-loop pass rate
  - real inbox wedge pass rate
  - percent of runs with a visible result plus persisted receipt
- **Truthfulness quality**
  - percent of failures that end with the correct terminal state
  - percent of blockers that name the real next human action
  - false-success and false-block rates
- **Operator burden**
  - operator interventions per live run
  - manual minutes required to get from setup to terminal truth
  - number of hidden rescue steps discovered during reruns
- **Latency and reliability**
  - p50 and p95 runtime for quickstart, prompt-to-spec, and inbox flows
  - retry rate, timeout rate, and provider failure rate on canonical paths
  - receipt persistence success and result-surface freshness
- **Repair velocity**
  - time from failed proof run to landed fix
  - time from landed fix to validated rerun
  - number of reruns required before the proof turns green again

Execution metrics are the engineering dashboard. They are not PMF proof by
themselves.

### Vanity Metrics

These numbers can be useful as inventory or storytelling context, but they
must not drive roadmap decisions:

- total agent types, providers, adapters, or integrations
- total debates, runs, or receipts without accepted outcomes
- test count, file count, doc count, and repo size
- GitHub stars, social impressions, newsletter signups, or waitlist growth
- benchmark wins disconnected from real operator workflows
- compliance-completeness percentages before repeated workflow usage exists
- raw demo traffic or `/demo` visits without retained real-task usage

### Decision Rules

- Proof metrics outrank execution metrics.
- Execution metrics only justify work when they are a credible bottleneck to a
  proof metric.
- Vanity metrics never justify a roadmap pivot or shipping priority.
- The founder loop is now a guardrail metric. The next true proof metric is
  repeatable external workflow success on the inbox/design-partner wedge.

## Proven Evidence On `main`

These structural slices are landed and **proven live**:

- ProviderRouter-backed runtime debate selection
- KM retrieval into debate context and KM writeback from outcomes
- versioned API-key management endpoints
- onboarding/get-started product flow
- truthful integrations and dashboard surfaces
- quickstart TLS fail-fast behavior
- quickstart structured receipts and inline provider-key support
- real demo/backend wiring
- **Quickstart receipts persist to receipt store** (API/dashboard/share-link visible)
- **Embedding rate limit resilience** (hash-based fallback on 429 errors)
- **Summary preamble cleaning** (LLM chain-of-thought stripped from CLI output)
- **Prompt-to-spec CLI** (`aragora spec` completes in ~23s)
- **Phase 2 truth-seeking wired**: Prover-Estimator consensus, cross-verification, truth ratio vote weights
- **Inbox trust wedge CLI**: `aragora triage auth` (OAuth), `--dry-run`, `--auto-approve`

Current focused proof on `main`:

```bash
python3 -m pytest tests/e2e/test_user_journey.py tests/cli/test_quickstart.py -q
```

Result on March 24, 2026: `71 passed` in `34.2s`.

Extended suite: `125 passed` in `35.1s`.

Live founder loop: **5/5 consecutive runs pass** (35-62s range).

## Canonical Founder-Loop Acceptance Checklist

All items below are **PASSED** as of March 24, 2026:

1. **Readiness is explicit** -- **PASSED**
   - health/readiness endpoints respond truthfully
   - credential/provider state is explicit before the run starts
   - `.env` loaded and provider reported before debate starts
2. **Quickstart enters the live path or fails closed quickly** -- **PASSED**
   - no silent fallback to demo
   - TLS/credential failures exit promptly with a direct reason
3. **A live debate completes or stops truthfully** -- **PASSED**
   - 5/5 consecutive runs complete in 35-62s
   - no hanging operator path, no hidden manual rescue
4. **A structured receipt is saved** -- **PASSED**
   - receipt inspectable and verifiable via `aragora receipt inspect/verify`
   - receipt also persisted to receipt store (commit 97074e28c)
5. **The result is visible on a product surface** -- **PASSED**
   - receipts visible via `/api/v2/receipts`, dashboard ReceiptsBrowser, share links
   - `aragora receipt list` shows quickstart receipts
6. **KM ingestion or KM stop condition is explicit** -- **PASSED**
   - truthful message: "ingestion skipped (quickstart uses lightweight KM)"
   - guidance: "Use 'aragora ask' or 'aragora decide' for full KM writeback"
7. **Operator noise is bounded** -- **PASSED**
   - embedding dimension mismatch demoted to DEBUG (commit 5333ada7d)
   - LLM chain-of-thought preamble stripped from summary output
   - final message shows clear next steps

## Founder-Facing Failure Taxonomy

When the founder loop does not complete successfully, label the run with one
primary canonical state from
[`docs/strategy/PRECISION_AND_TERMS.md`](../strategy/PRECISION_AND_TERMS.md#part-2-founder-run-failure-taxonomy):

- `auth_failure`
- `no_evidence`
- `low_confidence`
- `conflicting_models`
- `blocked_integration`
- `truthful_stop`

Classification rules for this plan:

- choose the earliest blocking reason that best explains why the run stopped
- do not collapse `conflicting_models` into `low_confidence`
- do not collapse `blocked_integration` into `auth_failure` once access is valid
- use `truthful_stop` only for a correct human/policy boundary, not as a
  generic fallback

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
- primary taxonomy label:
  - `auth_failure`
  - `no_evidence`
  - `low_confidence`
  - `conflicting_models`
  - `blocked_integration`
  - `truthful_stop`
- secondary diagnostics, if any:
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
- the run evidence is insufficient to satisfy the internal dogfood exit proof above

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
