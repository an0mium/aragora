# EU AI Act Compliance -- Onboarding Guide

> You have an August 2, 2026 deadline. Start here.
>
> This guide walks you from zero to a compliance-ready posture in five steps.
> No legal background required. Each step includes the exact command to run,
> what you will see, and how long it takes.

---

## Before You Begin

**What you need:**
- Aragora installed (`pip install -e .` or your organization's package manager)
- A terminal (bash, zsh, PowerShell)
- 30 minutes for the initial walkthrough (demo mode, no API keys needed)
- For production use: at least one AI provider API key (Anthropic, OpenAI, Mistral, or Gemini)

---

## The Five Steps

### Step 1: Classify Your AI Systems

**Time:** 1 minute per use case.

```bash
aragora compliance classify \
  "AI-powered CV screening for automated hiring decisions"
```

**What you will see:**

```
Risk Level: HIGH
Annex III:  4. Employment and worker management
Obligations:
  - Establish and maintain a risk management system (Art. 9)
  - Implement automatic logging of events (Art. 12)
  - Ensure transparency for deployers (Art. 13)
  - Design for effective human oversight (Art. 14)
  - Achieve appropriate accuracy and robustness (Art. 15)
```

Run this for every AI system in your organization. Systems classified as HIGH
require full compliance before the deadline. LIMITED requires only transparency
measures. MINIMAL has no mandatory requirements. UNACCEPTABLE means the practice
is prohibited in the EU.

---

### Step 2: Configure Your Debate Settings

**Time:** 5 minutes (one-time setup).

```bash
aragora ask "Should we approve this loan application?" \
  --agents anthropic-api,openai-api,mistral \
  --rounds 3 \
  --decision-integrity \
  --require-approval
```

Or set defaults for all future debates:

```bash
aragora config set default-agents "anthropic-api,openai-api,mistral"
aragora config set default-rounds 3
aragora config set decision-integrity true
aragora config set require-approval true
```

**Why these settings matter:**
- **3+ agents from different providers** -- Art. 15 robustness through heterogeneous consensus
- **3 rounds** -- richer audit trails for Art. 12
- **Decision integrity mode** -- cryptographic receipts for Art. 9 and Art. 15
- **Require approval** -- human signs off on every decision, satisfying Art. 14

---

### Step 3: Run a Decision Through Debate

**Time:** 2-5 minutes per decision.

```bash
aragora ask "Evaluate the bias risk of our hiring algorithm for EU operations" \
  --agents anthropic-api,openai-api,mistral \
  --rounds 3 \
  --decision-integrity \
  --require-approval
```

Or use demo mode without API keys: `aragora compliance export --demo --output-dir ./demo-pack`

Agents propose, critique, and revise their positions. At the end you see a
verdict with confidence score, dissenting views, and a prompt for human approval.
Once approved, a decision receipt is generated. Save the receipt ID for Step 4.

---

### Step 4: Generate Your Compliance Bundle

**Time:** Under 30 seconds.

```bash
aragora compliance export \
  --debate-id <DEBATE_ID> \
  --output-dir ./compliance-pack
```

**What you will see:**

```
EU AI Act Compliance Bundle
=======================================================
  Compliance Score:  87/100 -- Substantially Conformant
  Risk Level:  HIGH

  Article 9   Risk management          [PASS]
  Article 12  Record-keeping           [PASS]
  Article 13  Transparency             [PASS]
  Article 14  Human oversight          [PASS]
  Article 15  Accuracy & robustness    [PASS]

  Output: ./compliance-pack/
    bundle.json, receipt.md, audit_trail.md,
    transparency_report.md, human_oversight.md, accuracy_report.md
```

For formal regulatory submissions, use the extended generator:

```bash
aragora compliance eu-ai-act generate receipt.json \
  --output ./compliance-bundle/ \
  --provider-name "Your Organization" \
  --provider-contact "compliance@your-org.com" \
  --eu-representative "Your EU Rep GmbH, Berlin" \
  --system-name "Your AI System" \
  --system-version "1.0.0"
```

---

### Step 5: Submit and Maintain

**Time:** 5 minutes setup, then automatic.

```bash
# Periodic conformity audits (run weekly or monthly)
aragora compliance audit receipt.json --format markdown --output report.md

# Check your overall compliance posture
aragora compliance status

# Run adversarial stress tests (recommended quarterly)
aragora gauntlet run --suite fairness
```

Every decision through Aragora automatically generates a receipt. Export bundles
periodically or on demand for decisions requiring regulatory scrutiny. See the
[Notified Body Submission Checklist](./EU_AI_ACT_NOTIFIED_BODY_CHECKLIST.md)
for full submission package requirements.

---

## Compliance Score Interpretation

| Score | Label | What It Means |
|-------|-------|---------------|
| 95-100 | Full Conformity | All article requirements satisfied. Ready for submission. |
| 75-94 | Substantially Conformant | Minor gaps. Review the recommendations in your bundle. |
| 40-74 | Partial Conformity | Material gaps. Address the failing articles before submission. |
| 0-39 | Not Ready | Significant gaps. Review your debate configuration and re-run. |

A score below 75 typically means: too few agents (add a third provider), too few
rounds (increase to 3+), decision integrity mode not enabled, or human approval
not configured.

---

## Frequently Asked Questions

**Does using Aragora guarantee compliance with the EU AI Act?**
No. Aragora generates the technical evidence layer -- audit trails, risk
assessments, transparency records, and oversight documentation. Full compliance
also requires organizational measures: quality management policies, personnel
assignments, training data governance from model providers, and a formal
conformity assessment. Aragora handles the hardest part (continuous evidence
generation), but it is one component of your compliance program.

**How long does this entire process take?**
Technical setup: 30 minutes. First compliance bundle: under 5 minutes.
Full organizational documentation (QMS, risk register, oversight procedures):
2-8 weeks depending on maturity. Notified body assessment: 4-8 additional weeks.

**What if a debate does not reach consensus?**
Non-consensus debates still produce valid compliance artifacts. The receipt
records the disagreement, showing the system surfaced genuine uncertainty. Dissent
is documented in the Art. 13 transparency artifact. A human reviewer makes the
final decision, which strengthens your Art. 14 (human oversight) evidence.

**Can I use the same compliance bundle for multiple years?**
No. The EU AI Act requires ongoing compliance. Article 12 requires continuous
logging; Article 9 requires the risk management system to be maintained. Generate
bundles continuously -- per-decision for high-stakes decisions, periodically for
routine ones. Compliance is an ongoing output, not a one-time project.

**What if my system is not high-risk?**
LIMITED systems need only transparency measures. MINIMAL systems have no mandatory
requirements. You can still use Aragora's compliance features voluntarily. Many
organizations generate bundles for all AI systems as a governance best practice.

**Who is responsible for Article 10 (data governance)?**
Model providers (Anthropic, OpenAI, Google, Mistral). Your organization
orchestrates models, not trains them. Request training data documentation from
each provider. Aragora mitigates single-provider bias through multi-model consensus.

**We have ISO 27001 / SOC 2. Does that help?**
Yes. Existing certifications cover parts of Art. 15 (cybersecurity) and reduce
work on QMS and security documentation. They do not substitute for EU AI Act
compliance. Reference them in your Annex IV technical file.

**Where do I store compliance bundles?**
In a system with access controls and tamper detection. Aragora's Knowledge Mound
provides built-in SHA-256 integrity. Article 26(6) requires deployers to retain
logs for a minimum of 6 months.

---

## What to Read Next

| Document | When to Read It |
|----------|----------------|
| [EU AI Act Compliance Guide](./EU_AI_ACT_GUIDE.md) | Full technical reference with API examples |
| [Notified Body Submission Checklist](./EU_AI_ACT_NOTIFIED_BODY_CHECKLIST.md) | When preparing for conformity assessment |
| [Annex IV Technical Documentation Template](./EU_AI_ACT_ANNEX_IV_TEMPLATE.md) | When building your formal technical file |
| [Compliance Checklist](./EU_AI_ACT_CHECKLIST.md) | Phase-by-phase compliance tracker |
| [Customer Playbook](./EU_AI_ACT_CUSTOMER_PLAYBOOK.md) | Detailed article-by-article coverage |

---

*This guide is based on the EU AI Act (Regulation (EU) 2024/1689) as published.
It is not legal advice. Consult qualified legal counsel for your organization's
specific compliance obligations.*
