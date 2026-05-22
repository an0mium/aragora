# EU AI Act Compliance Artifact (REAL RECEIPT) — claude-C1CE7926

**Generated:** 2026-05-21T03:21:30Z
**Session:** claude-C1CE7926 (Phase 10/extension of substrate-soak)
**Bundle ID:** EUAIA-…  (see `conformity_report.json`)
**Integrity Hash:** `f5ba9081c8f7189f27af65c3d3ebad58663a174ea90b58b9ed5930e3df2b741f`

## Why this artifact is different from `EU_AI_ACT_ARTIFACT_claude-C1CE7926/`

The earlier P97 artifact (PR #7391) was generated against **synthetic demo data** and reported all 5 articles as PASS / bundle CONFORMANT. That is the artifact-pipeline smoke test.

**This** artifact was generated against a **real DecisionReceipt** produced by running `aragora demo` end-to-end. Result: a genuine product-quality assessment, not a smoke test.

## Generation chain

```bash
# Step 1: produce a real DecisionReceipt JSON
aragora demo \
  --topic "Should Aragora prioritize the EU AI Act compliance pipeline over new vertical onboarding for the August 2026 deadline?" \
  --receipt docs/compliance/EU_AI_ACT_REAL_claude-C1CE7926/real_receipt.json

# Step 2: feed it to the EU AI Act bundle generator
aragora compliance eu-ai-act generate ./real_receipt.json \
  --output . \
  --provider-name "Synaptent / Aragora" \
  --provider-contact "armand@synaptent.com" \
  --system-name "Aragora Decision Integrity Platform" \
  --system-version "2.9.0" \
  --format all
```

## Article compliance — REAL ASSESSMENT

| Article | Status | Description |
|---------|--------|-------------|
| **Article 9** | **PARTIAL** | Identify and analyze known and reasonably foreseeable risks |
| **Article 12** | **FAIL** | Automatic logging of events with traceability |
| **Article 13** | **PARTIAL** | Identify participating agents, their arguments, and interpretation |
| **Article 14** | **PARTIAL** | Enable human oversight, including ability to override and stop |
| **Article 15** | **FAIL** | Appropriate levels of accuracy and robustness; cybersecurity |

## Generator-emitted recommendations (verbatim)

1. **Ensure all decision events are logged in the provenance chain.**
2. **Include agent identities and reasoning in all receipts.**
3. **Integrate human-in-the-loop approval before critical decisions are finalized.**
4. **Improve robustness score and add cryptographic signing.**

## Source receipt summary

- **Receipt ID:** DR-20260521-b934a0
- **Verdict:** `needs_review` (no consensus, 49% confidence)
- **Agents:** Analyst, Critic, Synthesizer, Devil's Advocate
- **Rounds:** 2
- **Mode:** demo (offline)
- **Artifact hash:** `da72228e1425ac14376294b0a0514b5550dd193fc0d8bca93e6303dd41816361`
- **Signature algorithm:** SHA-256-content-hash (no asymmetric key)
- **Elapsed:** 0.0009 seconds (mock agents, not real LLMs)

## What this tells us about the product

The bundle generator correctly flags that demo-mode debates do **not** carry:

- A persistent provenance chain (Art. 12 FAIL)
- Asymmetric / signing infrastructure (Art. 15 FAIL)
- Explicit human-oversight checkpoints between rounds (Art. 14 PARTIAL)
- Full reasoning traces wired to identifiable model versions (Art. 13 PARTIAL)
- Comprehensive risk enumeration tied to Annex III categories (Art. 9 PARTIAL)

These are **real gaps** for the August 2, 2026 deadline. The earlier demo-data artifact (PR #7391) gave PASS across the board because the synthetic data was fabricated to satisfy each article's check. The live-receipt-fed generator gives an honest "no, this is the work that needs to happen."

## Bundle contents

| File | Bytes |
|------|-------|
| `real_receipt.json` | 2965 |
| `compliance_bundle.json` | (full bundle) |
| `conformity_report.{json,md}` | human + machine readable |
| `article_{9,12,13,14,15}_*.json` | per-article |
| `generate.log` | full CLI stdout |

## Substrate-freeze rationale

This is the most valuable external-proof artifact this soak produced. No new tooling — both `aragora demo` and `aragora compliance eu-ai-act generate` are CLIs already shipped on main. The novelty is feeding the output of one into the input of the other, which produces a genuine compliance-gap map instead of a fabricated CONFORMANT report.

**Recommend operator:** treat the four recommendations above as binding work items for the Aug 2 deadline. Each maps cleanly to existing aragora subsystems:
- Art. 12 logging → `aragora/observability/` + `aragora/audit/` (already strong, gap is in receipt embedding)
- Art. 15 signing → `aragora/policy/contract_signing.py` (just shipped in ADC v0.4 #7361, still draft)
- Art. 14 oversight → `aragora/inbox/`, `aragora/approvals/` (wedge already exists)
- Art. 13 transparency → `aragora/explainability/`, `aragora/introspection/` (live but not bound to receipt)
- Art. 9 risk → `aragora/compliance/framework.py` (already classifies; needs to bind to receipt risk fields)

## Notes

- The receipt was generated using mock agents (no API keys consumed).
- The bundle hash is deterministic given the same input — re-running with the same receipt JSON will produce byte-identical articles.
- No raw API keys touched; demo mode requires none.
