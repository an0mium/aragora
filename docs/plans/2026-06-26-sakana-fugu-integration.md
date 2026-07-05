# Sakana Fugu Integration Plan

**Date:** 2026-06-26
**Status:** Design / durable multi-session backlog
**Owner:** founder (Armand) + agent fleet
**Tracking epic:** _(filled in when the GitHub epic is created)_

## Thesis

Sakana Fugu (GA 2026-06-22) is **"multi-agent orchestration delivered as a single foundation
model"** — one OpenAI-compatible API that internally decides whether to answer directly or
delegate to a swappable pool of frontier models, then handles selection, verification, and
synthesis, and routes around degraded providers. It ships as `fugu` (balanced/low-latency) and
`fugu-ultra` (max accuracy, claimed to rival Fable 5). Grounded in two ICLR 2026 papers
(TRINITY: an evolved LLM coordinator; the Conductor: learning to orchestrate agents in natural
language). Technical report: arxiv 2606.21228. Subscription tiers $20 / $100 / $200; both models
in every tier. **Not available in the EU/EEA at launch** (GDPR/EU compliance pending).

**Strategic framing — Fugu is the *inverse* of Aragora's moat.** Aragora's value is
*transparent, auditable, receipt-backed heterogeneous consensus* — you can see exactly which
models said what and why. Fugu is an *opaque* orchestrator that hides its internal models. So
"incorporate everywhere useful" deliberately does NOT mean "everywhere." Fugu is excellent as a
**capable agent**, a **router option**, and a **resilience backstop**, and a valuable
**differentiation benchmark** — but it must never silently replace the transparent quorum that is
the product.

### Load-bearing decision (founder-approved 2026-06-26)

> **A Fugu vote is one opaque participant.** Fugu is a first-class agent, router option, and
> resilience fallback and may speak in debates — but a Fugu vote is flagged `opaque-orchestrated`
> in DecisionReceipts and **NEVER counts toward model-family diversity** in quorum/merge-evidence
> (`claude + fugu ≠ 2 families`). This protects the auditability guarantee.

## Where Fugu is useful (and where it is not)

| Surface | Use | Value | Phase |
|---|---|---|---|
| Transport (`api_agents/openai_compatible.py`) | `fugu`/`fugu-ultra` as OpenAI-compatible agents | Trivial, high | F1 |
| Agent registry (`agents/registry.py`, `types.py`, `spec.py`, `config_loader.py`) | First-class agent types | Easy | F1 |
| Provider readiness/health (`config/provider_readiness.py`, `cli/doctor.py`) | Readiness + `aragora doctor` check | Easy | F1 |
| Router (`aragora/routing/` Pareto optimizer) | `fugu`=balanced tier, `fugu-ultra`=hard-task tier | Strong (Aragora's analog to Factory Router) | F2 |
| Resilience (Airlock / OpenRouter fallback chain) | Fugu as HA fallback — it self-routes around degraded providers | Strong (resilience pillar) | F2 |
| EU/compliance guard (tenancy/compliance routing, agent construction, team selection, fallback execution) | **Exclude** Fugu for EU-tenant / EU-residency contexts | Required guardrail | F2 |
| Debate participant (`debate/orchestrator.py`, team selection) | Fugu Ultra as a strong single voice | Useful — with quorum caveat | F3 |
| Quorum/evidence (`swarm/quorum_evidence.py`, `merge_quorum_io.py`, `cli/commands/review_queue.py`) | Flag `opaque-orchestrated`; never counts as a family in debate receipts or PR merge evidence | Moat-protecting | F3 |
| Differentiation benchmark | Aragora transparent consensus vs Fugu black box, same decisions | High strategic / external proof | F4 |
| Research spike (TRINITY, Conductor, Fugu report) | Inform `team_selector`/router learned-orchestration | Longer horizon | F5 |

**Where Fugu is NOT used:** as a substitute for transparent quorum; in EU/compliance contexts;
anywhere a DecisionReceipt must attribute which model produced which claim (Fugu is opaque).

## Integration surfaces (verified in repo, 2026-06-26)

- `aragora/agents/api_agents/openai_compatible.py` — existing OpenAI-compatible transport (reuse).
- `aragora/agents/registry.py`, `types.py` (`ALLOWED_AGENT_TYPES`), `spec.py`, `config_loader.py` — agent registration.
- `aragora/config/provider_readiness.py`, `aragora/cli/doctor.py` — readiness/health.
- `aragora/routing/` — `provider_router.py`, `cost_quality_optimizer.py`, `provider_config.py`, `provider_metrics.py`, `lara_router.py`.
- `aragora/agents/fallback.py`, `aragora/agents/airlock.py` — resilience fallback chain.
- `aragora/swarm/quorum_evidence.py`, `merge_quorum_io.py`, `aragora/cli/commands/review_queue.py` (`evidence-lint`, `_normalize_model_family`, counted families) — quorum / model-family diversity and PR merge-evidence counting.
- `aragora/gauntlet/receipt_models.py` (`DecisionReceipt`), `aragora/gauntlet/odr_export.py`, `aragora/gauntlet/odr_schema.json` — receipt annotation and public audit export/schema compatibility.
- Secrets: `aragora/config/secrets.py` (AWS Secrets Manager) — add `FUGU_API_KEY` to both `MANAGED_SECRETS` and `CRITICAL_SECRETS` (no raw env key, per founder principle).

## Phased plan

Each phase is independently useful, flag-gated behind `enable_fugu` (default OFF until F1–F3 tests
pass). Confirm exact base URL, model IDs, auth header, streaming, and rate limits from
`console.sakana.ai` docs during F1 — the release page did not publish them.

### F1 — Transport & registry (the easy, high-value core)
- **F1.1** Add `fugu` / `fugu-ultra` via `openai_compatible.py` (exact API base URL TBD from Sakana console/docs, `FUGU_API_KEY` via Secrets Manager). Confirm endpoint/model-ids/auth from console docs before wiring a concrete base URL.
- **F1.2** Register as agent types (`registry.py`, `types.py`, `spec.py`, `config_loader.py`).
- **F1.3** Provider-readiness entry + `aragora doctor` check + credential validator.
- **F1.4** Pricing/cost metadata for cost-quality routing.
- **Acceptance:** `aragora debate` can include a `fugu`/`fugu-ultra` participant when `enable_fugu=1`; unit tests for adapter, registry, readiness; flag OFF by default.

### F2 — Routing & resilience
- **F2.1** Add Fugu to the Pareto router (`fugu`=balanced, `fugu-ultra`=hard-task) with cost/quality/latency metadata.
- **F2.2** Register Fugu as an HA fallback in the Airlock/OpenRouter fallback chain (leverages Fugu's own route-around-degradation).
- **F2.3** EU/compliance guard: router, agent construction, debate team selection, and fallback execution all exclude Fugu for EU-tenant / EU-residency contexts (wire into tenancy/compliance routing).
- **Acceptance:** router selects Fugu for appropriate cost/quality profiles; fallback engages on primary degradation; EU contexts provably never select Fugu through router, direct debate participant selection, or fallback execution (tests).

### F3 — Debate participation with auditability guards (moat-protecting)
- **F3.1** Fugu/Ultra selectable as a debate participant.
- **F3.2** Quorum/evidence: Fugu votes flagged `opaque-orchestrated`; **never** count toward model-family diversity in debate receipts or PR merge evidence (`swarm/quorum_evidence.py`, `merge_quorum_io.py`, `cli/commands/review_queue.py` evidence-lint/counting).
- **F3.3** `DecisionReceipt` annotation for opaque-orchestrated votes, plus ODR export/schema updates so public audit artifacts preserve the label without schema failure.
- **Acceptance:** test asserting `claude + fugu` does NOT satisfy a 2-family requirement; PR merge evidence-lint/counting also refuses Fugu as a canonical/countable review family; receipts and ODR exports label Fugu votes.

### F4 — Differentiation benchmark (external proof)
- **F4.1** Harness comparing Aragora transparent consensus vs Fugu black box on the same decisions (quality, auditability, cost, latency).
- **F4.2** Publishable artifact: "Aragora shows its work and produces receipts; Fugu doesn't." Ties to the external-proof / substrate-freeze principle.
- **Acceptance:** reproducible benchmark + a written comparison artifact.

### F5 — Research spike (longer horizon, non-binding)
- **F5.1** Study TRINITY (arxiv 2512.04695), Conductor (arxiv 2512.04388), and the Fugu technical report (arxiv 2606.21228); evaluate whether learned-orchestration improves `team_selector` / router vs hand-designed selection. Output: a findings memo, not necessarily code.

## Cross-cutting requirements
- **Feature flag** `enable_fugu` (default OFF) until F1–F3 test matrix passes.
- **Secrets**: `FUGU_API_KEY` via AWS Secrets Manager only — never a raw env key; add it to `MANAGED_SECRETS` and `CRITICAL_SECRETS` so hydration and strict-mode enforcement are explicit.
- **EU guard** is a hard requirement, not best-effort (compliance risk), and applies to router selection, direct agent construction, team selection, and fallback execution.
- **Auditability**: every Fugu use that reaches a receipt is labeled `opaque-orchestrated`.
- **No grinder coupling**: do not attach `boss-ready` to these issues (fleet is frozen).

## Test plan
- Adapter unit tests (auth, streaming, error/timeout, token accounting) against a mocked OpenAI-compatible endpoint.
- Registry/readiness tests (agent types resolve; doctor reports Fugu; missing key degrades gracefully).
- Router tests (Fugu selected on correct cost/quality profile; fallback engages on degradation).
- EU-guard tests (EU context never selects Fugu via router, direct debate participant/team selection, or fallback execution).
- Quorum/merge-evidence/ODR tests (Fugu vote flagged; `claude+fugu` ≠ 2 families; review-queue evidence-lint/counted-families refuse Fugu as canonical/countable; receipts and ODR exports annotated; `odr_schema.json` accepts the public label).
- Benchmark harness reproducibility test.

## Open items to confirm at implementation
1. Exact API base URL, model IDs, auth header, streaming semantics, rate limits (from `console.sakana.ai`).
2. Whether Fugu exposes any sub-model attribution (would relax — but not remove — the opaque-vote rule).
3. Token/cost accounting shape for the cost-quality optimizer.
4. Whether to expose OpenRouter's current `sakana/fugu-ultra` surface alongside the direct Sakana API path, and whether direct `fugu` should remain the default for the balanced tier.

## References
- Sakana Fugu release: https://sakana.ai/fugu-release/ and https://sakana.ai/fugu/
- OpenRouter Fugu Ultra listing: https://openrouter.ai/sakana/fugu-ultra
- Fugu technical report: https://arxiv.org/abs/2606.21228
- Conductor paper: https://arxiv.org/abs/2512.04388
- TRINITY paper: https://arxiv.org/abs/2512.04695
- Related Aragora work: `docs/superpowers/specs/2026-06-26-reconcile-lane-design.md` (reconcile lane) and PR #8655's native mission orchestrator spec (mission spine; lands separately).
