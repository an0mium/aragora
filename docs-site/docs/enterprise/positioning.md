---
title: "Aragora: Commercial Positioning & Value Proposition"
description: "Control plane for multi-agent deliberation across org knowledge and channels"
---

# Aragora: Commercial Positioning & Value Proposition

**Control Plane for Multi-Agent Deliberation Across Org Knowledge and Channels**

*Version 2.0 | January 2026*

---

## Executive Summary

Aragora is the control plane for multi-agent deliberation—orchestrating 15+ AI models to debate your organization's knowledge (documents, databases, APIs) and deliver defensible decisions to any channel (Slack, Teams, Discord, voice). Unlike single-model chatbots, Aragora builds institutional memory with full audit trails.

**Key Value:** Replace "the AI said so" with decision assurance backed by multi-agent consensus, evidence chains, and compliance-ready artifacts. Decisions delivered wherever your team works.

---

## Product Positioning

### What Aragora Is

| Category | Description |
|----------|-------------|
| **Primary** | AI Decision Validation Platform |
| **Secondary** | Enterprise AI Governance / Adversarial Testing |
| **Technical** | Multi-Agent Debate Orchestration Engine |

### What Aragora Is NOT

- **Not a chatbot** - Produces structured decision records, not conversations
- **Not a code generator** - Validates and stress-tests, not creates
- **Not AutoGen/CrewAI** - Adversarial (debate), not cooperative (collaboration)
- **Not a single-model wrapper** - Requires heterogeneous AI providers

---

## Target Customers

### Primary Segments

| Segment | Pain Point | Aragora Solution |
|---------|------------|------------------|
| **Platform Engineering** | Architecture decisions lack rigor | Gauntlet stress-testing with audit trail |
| **Security Teams** | Manual security review bottlenecks | Automated adversarial security analysis |
| **Compliance/GRC** | AI decisions lack documentation | DecisionReceipts for audit evidence |
| **AI/ML Teams** | Model outputs are untrusted | Multi-model consensus validation |

### Ideal Customer Profile

- **Size:** 50-5,000 employees (mid-market to enterprise)
- **Industry:** Financial services, healthcare, legal, enterprise SaaS
- **Maturity:** Already using AI/LLMs in production
- **Trigger:** Regulatory pressure, security incident, scaling review bottleneck

---

## Competitive Differentiation

### vs. Cooperative Agent Frameworks (AutoGen, CrewAI)

| Dimension | Cooperative Frameworks | Aragora |
|-----------|----------------------|---------|
| Architecture | Agents collaborate on tasks | Agents debate and critique each other |
| Output | Task completion | Decision validation + dissent record |
| Use Case | Automation, workflows | Stress-testing, validation, governance |
| Blind Spots | Shared (single paradigm) | Diverse (heterogeneous models required) |

**Positioning:** "AutoGen helps agents work together. Aragora makes them argue. Use both."

### vs. Human Expert Review

| Dimension | Human Review | Aragora |
|-----------|--------------|---------|
| Cost | $15,000-$100,000 | $0.10-$2.00 per debate |
| Time | 2-6 weeks | 15-45 minutes |
| Consistency | Variable | Reproducible |
| Scalability | Limited | Unlimited |
| Audit Trail | Manual documentation | Automatic DecisionReceipts |

**Positioning:** "Expert review quality at API call prices."

### vs. Single AI Model

| Dimension | Single Model | Aragora |
|-----------|--------------|---------|
| Blind Spots | Correlated | Diverse (heterogeneous) |
| Self-Correction | Limited | Enforced through debate |
| Confidence | Uncertain | Consensus-based scoring |
| Dissent | Lost | Preserved and tracked |

**Positioning:** "One model can't catch its own mistakes. Three models can."

---

## Core Capabilities

### 1. Adversarial Debate Engine

**What:** Structured debate between heterogeneous AI models with role assignment (proposer, critic, synthesizer, judge).

**Why It Matters:** Different models have different blind spots. By requiring disagreement and synthesis, Aragora catches errors that single-model validation misses.

**Technical:** 9-round debate protocol, 8 consensus modes, convergence detection, early exit optimization.

### 2. Gauntlet Mode

**What:** Productized adversarial stress-testing with pre-configured attack personas.

| Persona | Focus |
|---------|-------|
| Security | Injection, auth, API vulnerabilities |
| Devil's Advocate | Logic flaws, hidden assumptions |
| Scaling Critic | Bottlenecks, SPOFs |
| Compliance | GDPR, HIPAA, SOC 2, AI Act |

**Why It Matters:** Go/no-go gate for CI/CD pipelines. Executive-ready output.

### 3. DecisionReceipts

**What:** Audit-ready decision records with risk heatmaps, dissent trails, and evidence chains.

**Why It Matters:** Compliance teams need documentation. DecisionReceipts provide the paper trail for SOC 2, AI Act, and internal governance.

**Format:** JSON (machine-readable) + Markdown (human-readable) + PDF (executive-ready).

### 4. Formal Verification

**What:** Z3 SMT solver integration for mathematical proof of claims.

**Why It Matters:** When models agree on a mathematical property, Z3 can prove it's correct. Moves from "likely true" to "provably true."

**Use Cases:** Invariant validation, boundary conditions, access control policies.

### 5. Multi-Provider Integration

**Supported Providers (14+):**
- **Direct APIs:** Anthropic, OpenAI, Google Gemini, Mistral, xAI Grok
- **OpenRouter:** DeepSeek, Qwen, Yi, Llama, Kimi
- **Local:** Ollama, LM Studio

**Why It Matters:** Heterogeneous models catch more errors than homogeneous ones. Different training data = different blind spots.

---

## Deployment Options

| Option | Description | Best For |
|--------|-------------|----------|
| **CLI** | Command-line tool for developers | Individual use, scripting |
| **API/SDK** | REST API + Python/Node SDKs | Application integration |
| **SaaS** | Hosted web application | Teams, dashboards |
| **On-Premise** | Self-hosted Docker/K8s | Regulated environments |

---

## Pricing Model

| Tier | Price | Debates/Month | Features |
|------|-------|---------------|----------|
| **Free** | $0/forever | 10 | CLI, 2 models |
| **Starter** | $49/mo | 100 | API, 4 models, basic personas |
| **Professional** | $199/mo | 500 | All models, Gauntlet, DecisionReceipts |
| **Enterprise** | Custom | Unlimited | On-premise, SSO, priority support |

**Unit Economics:**
- Cost per debate: $0.10-$2.00 (depending on models and rounds)
- Target gross margin: 70%+

---

## Integration Points

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Aragora Gauntlet
  uses: aragora/gauntlet-action@v1
  with:
    file: architecture.md
    profile: security
    fail-on: high-risk
```

### API Integration

```python
from aragora import Arena, DebateProtocol

arena = Arena(
    question="Should we use JWT or session tokens?",
    agents=["anthropic-api", "openai-api", "gemini"],
    protocol=DebateProtocol(rounds=5, consensus="judge"),
)
result = await arena.run()
print(result.decision_receipt)
```

### Workflow Integration

- Slack: `/aragora review <url>`
- GitHub: PR comment triggers
- Jira: Ticket automation
- Webhook: Custom integrations

---

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Time to decision | &lt;45 min | 15-30 min avg |
| Cost vs. human review | 99% reduction | $2 vs. $50,000 |
| Blind spot detection | >90% | Field testing |
| Customer retention | >90% | Pre-launch |

---

## Compliance & Security

| Standard | Status |
|----------|--------|
| SOC 2 Type II | 92% ready, audit Q1 2026 |
| GDPR | Compliant |
| HIPAA | BAA available |
| AI Act | Built-in compliance personas |

**Security Features:**
- API key isolation per tenant
- SSRF protection with domain allowlists
- Audit logging for all decisions
- Data encryption at rest and in transit

---

## Customer Use Cases

### 1. Architecture Review

**Before:** 2-week review cycle with senior architects
**After:** 30-minute Gauntlet run before PRD approval

### 2. Security Validation

**Before:** Manual security review backlog
**After:** Automated security stress-test in CI pipeline

### 3. Compliance Documentation

**Before:** Manual policy review and documentation
**After:** DecisionReceipts auto-generated for audit

### 4. AI Model Validation

**Before:** Single-model outputs shipped without validation
**After:** Multi-model consensus with dissent tracking

---

## Messaging Guidelines

### Tagline
**"AI decisions you can trust."**

### Elevator Pitch
**"Aragora is an AI red team for decisions. Multiple AI models debate your specs, policies, and code to catch blind spots before you ship. Output: an audit-ready decision receipt."**

### Value Props (By Audience)

| Audience | Message |
|----------|---------|
| **Developers** | "Catch bugs before they ship. 15-minute architecture review." |
| **Security** | "Automated adversarial testing with audit trail." |
| **Compliance** | "Decision documentation that satisfies auditors." |
| **Executives** | "De-risk AI decisions. Enterprise-grade governance." |

---

## Roadmap Highlights

| Quarter | Milestone |
|---------|-----------|
| Q1 2026 | SOC 2 Type II, first enterprise pilots |
| Q2 2026 | On-premise deployment, mobile SDK |
| Q3 2026 | Vertical-specific templates (healthcare, finance) |
| Q4 2026 | International expansion, EU data residency |

---

## Contact

- **Website:** aragora.ai
- **Documentation:** docs.aragora.ai
- **Demo:** demo.aragora.ai
- **Sales:** sales@aragora.ai
- **Support:** support@aragora.ai

---

*Document Version: 1.0 | Last Updated: January 2026*
