# Sprint 2 PR Plan (Weeks 3-4)

Last updated: 2026-02-12

Sprint 2 scope (from `docs/status/BACKLOG_Q1_2026.md`):
- SME-04: Implement guided onboarding wizard (CLI/Web)
- SME-05: First debate → receipt flow in <15 min
- SME-06: Decision receipt export (PDF/Markdown)
- DEV-04: TS SDK: debates/streaming/receipts parity
- DEV-05: Update SDK_PARITY.md with coverage metrics
- DEV-06: Add streaming example to SDK docs
- HOST-04: Health check endpoints documentation
- HOST-05: Smoke test script for self-hosted

This document turns that scope into a mergeable PR sequence with suggested `codex/…` branches and test gates.

Related:
- Canonical stability gates: `docs/status/NEXT_STEPS_CANONICAL.md`
- Implementation map: `docs/status/BACKLOG_Q1_2026_IMPLEMENTATION_MAP.md`

---

## Global Rules For Every PR

1. Keep canonical stability gates green (don’t merge work that regresses them):
   - Test pollution / randomized ordering gate: `.github/workflows/test.yml` `test-pollution-randomized`
   - Connector exception hygiene: `scripts/check_connector_exception_handling.py`
   - Offline/demo golden path: `tests/cli/test_offline_golden_path.py`
   - Agent registry drift checks: `scripts/check_agent_registry_sync.py`
   - SDK/version drift: `.github/workflows/sdk-parity.yml` + parity tests
   - Self-host readiness static check: `scripts/check_self_host_compose.py`
2. Prefer small PRs that can be reviewed in <30 minutes and reverted safely.
3. Every PR must list its exact local validation commands in the PR description.

---

## PR Sequence (Best Order)

### PR 1 (P0): Receipt API Supports Onboarding Retrieval

Branch: `codex/s2-receipts-filter-by-debate`

Goal:
- Make it possible to reliably fetch a receipt for a debate without needing the receipt id at create-time.

Scope:
- Add optional `debate_id` query param to receipts listing endpoint (v2).
- Thread the filter down to storage so it’s efficient and testable.

Primary files:
- Server: `aragora/server/handlers/receipts.py` (list endpoint)
- Storage: `aragora/storage/receipt_store.py` (list/count filters)
- OpenAPI (if needed): `docs/api/openapi.json` and/or `docs/api/openapi.yaml`
- Python tests: `tests/server/handlers/test_receipts.py`

Definition of done:
- `GET /api/v2/receipts?debate_id=...` returns receipts for that debate only.
- Pagination + other filters continue to work.

Local test gates:
- `pytest /Users/armand/Development/aragora/tests/server/handlers/test_receipts.py`
- `pytest /Users/armand/Development/aragora/tests/test_decision_receipt.py`

Notes:
- This PR unblocks SME-05/SME-06 UX without assuming the debate create response includes `receipt_id`.

---

### PR 2 (P0): Web Onboarding First Debate Produces a Receipt You Can Reach

Branch: `codex/s2-onboarding-receipt-review-step`

Goal:
- Deliver the “first debate → receipt” loop in the onboarding wizard.

Scope:
- Use existing onboarding wizard (`OnboardingWizardFlow`) as the canonical flow for Sprint 2.
- On debate completion, fetch the receipt via `debate_id` and present a receipt review UI state:
  - show receipt summary + verdict/confidence
  - provide export buttons for Markdown + PDF (via `/api/v2/receipts/{id}/export`)
- Store `receipt_id` in onboarding store for future use.

Primary files:
- UI flow: `aragora/live/src/components/onboarding/OnboardingWizardFlow.tsx`
- First debate step: `aragora/live/src/components/onboarding/FirstDebateStep.tsx`
- Store: `aragora/live/src/store/onboardingStore.ts`

Definition of done:
- A user can complete the onboarding wizard and end up on a verifiable receipt (no “dead link”).
- Works even when debate create response omits `receipt_id` (uses PR 1 filter).

Local test gates:
- UI unit: `cd /Users/armand/Development/aragora/aragora/live && npm test`
- UI e2e: `cd /Users/armand/Development/aragora/aragora/live && npm run test:e2e -- e2e/onboarding.spec.ts`

Dependencies:
- PR 1 merged first.

---

### PR 3 (P0): Receipt Export Clarity (PDF/MD) and UX-Safe Fallback

Branch: `codex/s2-receipts-export-polish`

Goal:
- Make SME-06 unambiguous: PDF and Markdown exports work, and PDF failure modes are non-confusing.

Scope:
- Ensure `/api/v2/receipts/{receipt_id}/export?format=md|pdf` behaves consistently:
  - correct `Content-Type`
  - sensible `Content-Disposition` filenames
  - when PDF rendering isn’t available, return printable HTML with a clear indicator header (already supported) and document it.
- Add/adjust tests to lock the contract.
- Update docs for “PDF export requirements” and fallback behavior.

Primary files:
- Server export: `aragora/server/handlers/receipts.py`
- Export impl: `aragora/export/decision_receipt.py`
- Tests: `tests/server/handlers/test_receipts.py`, `tests/test_decision_receipt.py`
- Docs: `docs/RECEIPT_CONTRACT.md` and/or `docs/api/API_REFERENCE.md` (choose one canonical place)

Local test gates:
- `pytest /Users/armand/Development/aragora/tests/server/handlers/test_receipts.py`
- `pytest /Users/armand/Development/aragora/tests/test_decision_receipt.py`

Parallelizable:
- Can run in parallel with PR 2 once PR 1 is merged (PR 2 can temporarily keep export buttons hidden behind feature flag if needed).

---

### PR 4 (P0): TS SDK Parity for Debates + Streaming + Receipts (Core Flows)

Branch: `codex/s2-ts-sdk-debate-stream-receipts`

Goal:
- DEV-04: ensure TypeScript SDK supports the onboarding-critical flows end-to-end:
  - create debate
  - stream events
  - fetch/export receipt

Scope:
- Audit and fix missing/incorrect endpoints in:
  - `sdk/typescript/src/namespaces/debates.ts`
  - `sdk/typescript/src/websocket.ts`
  - `sdk/typescript/src/namespaces/receipts.ts`
- Add TS tests for the above behavior where missing.
- If PR 1 adds `debate_id` filtering, expose it in TS receipt list params.

Primary files:
- SDK: `sdk/typescript/src/client.ts`, `sdk/typescript/src/websocket.ts`, `sdk/typescript/src/namespaces/debates.ts`, `sdk/typescript/src/namespaces/receipts.ts`
- SDK tests: `sdk/typescript/src/namespaces/__tests__/`

Local test gates:
- `cd /Users/armand/Development/aragora/sdk/typescript && npm run check:types`
- `cd /Users/armand/Development/aragora/sdk/typescript && npm run typecheck`
- `cd /Users/armand/Development/aragora/sdk/typescript && npm test`
- Python parity suite (to prevent drift): `pytest /Users/armand/Development/aragora/tests/sdk/test_contract_parity.py /Users/armand/Development/aragora/tests/sdk/test_endpoint_parity.py /Users/armand/Development/aragora/tests/sdk/test_sdk_parity.py /Users/armand/Development/aragora/tests/sdk/test_websocket.py`

Dependencies:
- PR 1 if you expose `debate_id` filter.

---

### PR 5 (P1): SDK_PARITY Metrics Refresh + CI Guard

Branch: `codex/s2-sdk-parity-metrics-refresh`

Goal:
- DEV-05: refresh parity metrics based on the actual shipped endpoint surface and ensure it’s easy to keep current.

Scope:
- Update `docs/guides/SDK_PARITY.md` (and any referenced parity docs) based on current SDK/server.
- Ensure the parity validation scripts remain blocking and documented.

Primary files:
- Docs: `docs/guides/SDK_PARITY.md`
- Tests/scripts: `tests/sdk/test_sdk_parity.py`, `scripts/verify_sdk_contracts.py`

Local test gates:
- `pytest /Users/armand/Development/aragora/tests/sdk/test_sdk_parity.py`
- `python /Users/armand/Development/aragora/scripts/verify_sdk_contracts.py`

Dependencies:
- Prefer after PR 4 so the parity update reflects the new coverage.

---

### PR 6 (P1): Streaming Example In SDK Docs (Copy/Paste Ready)

Branch: `codex/s2-sdk-streaming-example-docs`

Goal:
- DEV-06: ensure a developer can run streaming in <10 minutes with a local server.

Scope:
- Add a dedicated “Streaming Debate” section to `docs/SDK_GUIDE.md` referencing (or aligning with) the cookbook.
- Provide a minimal TS streaming example (if missing) alongside the existing cookbook artifacts.

Primary files:
- Docs: `docs/SDK_GUIDE.md`
- Cookbook: `sdk/cookbook/02_streaming_debate.py` (already exists)
- New TS example (suggested): `sdk/cookbook/11_streaming_debate.ts` (or update existing `sdk/cookbook/09_typescript_quickstart.ts`)

Local validation:
- Ensure examples compile/run (manual):
  - `cd /Users/armand/Development/aragora/sdk/typescript && npm run build`

Dependencies:
- Prefer after PR 4 so docs match the SDK API.

---

### PR 7 (P0): Self-Hosted Health Endpoints Documentation

Branch: `codex/s2-selfhost-health-docs`

Goal:
- HOST-04: document which endpoints to use for liveness/readiness and what they mean.

Scope:
- Add a “Health & Readiness” section to the self-host docs.
- Document the recommended endpoints and expected responses:
  - `/healthz`, `/readyz`, plus any `/api/metrics/health` details used in ops.

Primary files:
- Docs: `deploy/README.md` (or a dedicated doc under `docs/deployment/`)
- Reference handlers:
  - `aragora/server/handlers/admin/health/__init__.py`
  - `aragora/server/handlers/admin/system.py` (contains references)
  - `aragora/server/handlers/gateway_health_handler.py`

Local validation:
- Doc-only PR; validate by starting server and curling endpoints (manual checklist in PR description).

---

### PR 8 (P1): Runtime Smoke Test for Self-Hosted (Not Just Static YAML)

Branch: `codex/s2-selfhost-runtime-smoke`

Goal:
- HOST-05: provide a smoke test script that proves “compose is up and the API works”.

Scope:
- Add a script that runs against an already-running self-host deployment:
  - hits health/ready endpoints
  - hits OpenAPI/docs endpoint
  - optionally hits a lightweight authenticated endpoint if token is provided
- Keep `scripts/check_self_host_compose.py` as the static CI guard; this is complementary.

Primary files:
- New: `scripts/smoke_self_host_runtime.py` (suggested name)
- Existing static check: `scripts/check_self_host_compose.py`
- Docs: add usage section to `deploy/README.md`

Local validation:
- Manual:
  - `python scripts/smoke_self_host_runtime.py --base-url http://localhost:8000`
  - optionally `--api-token` for authenticated checks

Dependencies:
- PR 7 preferred (doc references).

---

## What To Defer (Explicitly Out of Sprint 2)

- Deep consolidation of the multiple onboarding implementations in `aragora/live/src/components/onboarding/` beyond what is required to make the wizard stable.
- Connector setup wizards (Slack/Gmail/Drive) beyond status visibility (those are Sprint 4 items).
- Full “usage dashboard / budgets” UI (Sprint 3).

