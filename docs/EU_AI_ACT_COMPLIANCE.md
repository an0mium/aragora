# Aragora and the EU AI Act: Compliance Mapping

> **TL;DR**: The EU AI Act (Regulation 2024/1689) requires audit trails, transparency,
> human oversight, and risk management for high-risk AI systems. Enforcement begins
> August 2, 2026. Aragora provides the open-source infrastructure to satisfy these
> requirements — self-hosted, vendor-neutral, and auditable.

## Regulatory Timeline

| Date | Milestone |
|------|-----------|
| August 1, 2024 | EU AI Act entered into force |
| February 2, 2025 | Prohibitions on unacceptable-risk AI apply |
| August 2, 2025 | Governance rules and obligations for general-purpose AI apply |
| **August 2, 2026** | **Full enforcement for high-risk AI systems** |
| August 2, 2027 | Obligations for high-risk AI in Annex I apply |

**The clock is ticking.** Organizations deploying AI for high-risk decisions have
until August 2, 2026 to implement compliant audit trails, transparency mechanisms,
and human oversight.

---

## High-Risk AI Systems (Article 6 + Annex III)

The EU AI Act classifies the following as **high-risk**, requiring full compliance:

| Domain | Examples | Aragora Relevance |
|--------|----------|------------------|
| Employment & HR | AI-assisted hiring, promotion, termination decisions | Debate engine for hiring decisions with dissent tracking |
| Credit & Finance | Creditworthiness assessment, risk scoring | Adversarial vetting of financial risk models |
| Critical Infrastructure | Energy, water, transport management | Decision receipts for infrastructure changes |
| Education | Student assessment, admission decisions | Transparent scoring with explainability |
| Law Enforcement | Predictive policing, evidence evaluation | Full audit trail with provenance |
| Migration & Border | Visa/asylum application processing | Human oversight with calibrated confidence |
| Justice & Democracy | Judicial decision support | Adversarial debate with belief network analysis |

---

## EU AI Act Requirements vs. Aragora Capabilities

### Article 9: Risk Management System

> *"A risk management system shall be established, implemented, documented and
> maintained in relation to high-risk AI systems."*

| Requirement | EU AI Act Reference | Aragora Implementation |
|-------------|-------------------|----------------------|
| Identify and analyze known/foreseeable risks | Art. 9(2)(a) | **Gauntlet runner**: 3-phase adversarial stress testing (red team attacks, capability probes, scenario matrix) identifies risks before deployment |
| Estimate and evaluate risks from intended use | Art. 9(2)(b) | **Decision receipts**: Document risk assessment with confidence intervals and dissent tracking |
| Evaluate risks from reasonably foreseeable misuse | Art. 9(2)(c) | **Red-team agents**: Devil's Advocate, Scaling Critic, and Compliance Auditor personas probe edge cases |
| Adopt risk management measures | Art. 9(4) | **Calibrated trust**: ELO rankings + Brier scores track which agents are reliable on which domains, enabling risk-proportionate agent selection |

### Article 11: Technical Documentation

> *"Technical documentation shall be drawn up before a high-risk AI system is
> placed on the market and shall be kept up to date."*

| Requirement | EU AI Act Reference | Aragora Implementation |
|-------------|-------------------|----------------------|
| General description of the AI system | Art. 11 + Annex IV | **Decision receipts** document the full system configuration: agents used, protocols, consensus thresholds |
| Description of development process | Annex IV, Section 2 | **Knowledge Mound** maintains institutional memory of how decisions evolve over time |
| Monitoring, functioning, and control | Annex IV, Section 3 | **Observability stack**: Prometheus metrics, OpenTelemetry tracing, SLO alerting |
| Detailed description of validation and testing | Annex IV, Section 6 | **Gauntlet reports**: Comprehensive adversarial test results with pass/fail criteria |

### Article 12: Record-Keeping (Logging)

> *"High-risk AI systems shall technically allow for the automatic recording
> of events ('logs') over the lifetime of the system."*

| Requirement | EU AI Act Reference | Aragora Implementation |
|-------------|-------------------|----------------------|
| Automatic event logging | Art. 12(1) | **Audit module** (`aragora/audit/`): Comprehensive event logging with tamper detection |
| Identify situations that may result in risk | Art. 12(2) | **Anomaly detection** (`aragora/security/anomaly_detection.py`): Identifies unusual patterns in agent behavior |
| Facilitate post-market monitoring | Art. 12(3) | **Continuum memory**: 4-tier memory (fast/medium/slow/glacial) preserves decision history for retrospective analysis |
| Traceability of system functioning | Art. 12(1) | **Decision receipts** with cryptographic signing (HMAC-SHA256, RSA-SHA256, Ed25519) provide tamper-evident audit trail |

### Article 13: Transparency and Information

> *"High-risk AI systems shall be designed and developed in such a way as to
> ensure that their operation is sufficiently transparent to enable deployers
> to interpret the system's output and use it appropriately."*

| Requirement | EU AI Act Reference | Aragora Implementation |
|-------------|-------------------|----------------------|
| Understand and interpret output | Art. 13(1) | **Explainability module**: Evidence chains, vote pivots, belief network analysis, counterfactual explanations |
| Appropriate level of accuracy | Art. 13(3)(b)(ii) | **Calibration tracking**: Brier scores measure actual accuracy vs. stated confidence. ELO rankings track domain-specific reliability |
| Known limitations | Art. 13(3)(b)(iv) | **Dissent records**: Decision receipts explicitly document where agents disagreed and why, surfacing uncertainty rather than hiding it |
| Foreseeable unintended outcomes | Art. 13(3)(b)(v) | **Gauntlet findings**: Red-team results document failure modes and edge cases discovered during adversarial testing |

### Article 14: Human Oversight

> *"High-risk AI systems shall be designed and developed in such a way [...]
> as to be effectively overseen by natural persons."*

| Requirement | EU AI Act Reference | Aragora Implementation |
|-------------|-------------------|----------------------|
| Humans can understand AI capabilities and limitations | Art. 14(4)(a) | **Calibrated confidence**: Every decision includes confidence intervals based on historical agent accuracy, not just model self-assessment |
| Humans can correctly interpret output | Art. 14(4)(b) | **Adversarial debate transcript**: Full reasoning chain with critiques, showing why consensus was (or wasn't) reached |
| Humans can decide not to use the system | Art. 14(4)(c) | **Approval gates**: Nomic Loop requires human approval for changes. Decision receipts are advisory, not autonomous |
| Humans can intervene or stop the system | Art. 14(4)(d) | **Circuit breakers**: Automatic halt on repeated failures. RBAC controls who can initiate debates and approve results |

### Article 15: Accuracy, Robustness, Cybersecurity

> *"High-risk AI systems shall be designed and developed in such a way that
> they achieve an appropriate level of accuracy, robustness and cybersecurity."*

| Requirement | EU AI Act Reference | Aragora Implementation |
|-------------|-------------------|----------------------|
| Appropriate level of accuracy | Art. 15(1) | **Heterogeneous consensus**: Using multiple model providers (Claude, GPT, Gemini, local) surfaces disagreement that single-model systems hide |
| Resilient to errors, faults, inconsistencies | Art. 15(4) | **Circuit breakers** + retry with exponential backoff + agent fallback (OpenRouter on quota errors) |
| Resilient to adversarial manipulation | Art. 15(5) | **Trickster** (hollow consensus detector) + SecurityBarrier (telemetry redaction) + rate limiting + SSRF protection |
| Cybersecurity measures | Art. 15(5) | AES-256-GCM encryption, key rotation, webhook signature verification, RBAC, MFA, API key management |

---

## How Decision Receipts Satisfy Audit Requirements

A decision receipt generated by Aragora contains:

```
Decision Receipt #DR-2026-0211-001
===================================
Question:    "Should we approve vendor X for SOC2 data processing?"
Protocol:    Adversarial debate (3 rounds, supermajority consensus)
Agents:      Claude-3.5, GPT-4o, Gemini-1.5, Mistral-Large
Timestamp:   2026-02-11T14:23:00Z

CONSENSUS: PARTIAL (3/4 agents agree)
  Agreed:   Vendor X meets 18/20 SOC2 controls
  Dissent:  Gemini-1.5 flagged gaps in access logging (control CC6.1)

Confidence: 0.82 (calibrated via Brier score, historical accuracy 0.85)

Evidence Chain:
  1. [Proposal] Claude-3.5: "Vendor X satisfies controls..." (evidence: SOC2 report)
  2. [Critique] GPT-4o: "Missing evidence for CC6.1 access logging"
  3. [Revision] Claude-3.5: "CC6.1 gap acknowledged, mitigation: ..."
  4. [Vote] 3/4 approve with condition on CC6.1 remediation

Signature: HMAC-SHA256:a3f8c2d1... (tamper-evident)
```

This receipt directly satisfies:
- **Art. 12** (record-keeping): Automatic, structured logging
- **Art. 13** (transparency): Interpretable reasoning chain
- **Art. 14** (human oversight): Humans review before acting
- **Art. 11** (documentation): Full system configuration captured

---

## Aragora vs. Proprietary Governance Tools

| Capability | Fiddler AI | OneTrust | Azure AI | **Aragora** |
|-----------|-----------|---------|---------|-------------|
| Audit trails | Yes | Yes | Yes | **Yes** (cryptographic receipts) |
| Bias detection | Yes | Yes | Yes | **Yes** (adversarial debate surfaces bias) |
| Explainability | LIME/SHAP | Limited | Built-in | **Yes** (evidence chains, belief networks) |
| Adversarial testing | No | No | Limited | **Yes** (Gauntlet: 3-phase red teaming) |
| Multi-model consensus | No | No | No | **Yes** (heterogeneous provider debate) |
| Confidence calibration | No | No | No | **Yes** (ELO + Brier scores) |
| Self-hosted | No | No | No | **Yes** (full control of data) |
| Open source | No | No | No | **Yes** (MIT license, auditable code) |
| Vendor lock-in | AWS/GCP | SaaS | Azure | **None** (any LLM provider) |
| Cost | $100K+/yr | $100K+/yr | Azure pricing | **Free** (self-hosted) |

---

## Implementation Checklist for Regulated Organizations

### Phase 1: Immediate (0-3 months)
- [ ] Deploy Aragora with decision receipt generation enabled
- [ ] Configure RBAC roles for decision approvers
- [ ] Enable audit logging with tamper-evident signatures
- [ ] Set up Gauntlet adversarial testing for high-risk decision categories

### Phase 2: Integration (3-6 months)
- [ ] Integrate with existing GRC tools via API
- [ ] Configure multi-model consensus (minimum 3 providers for high-risk)
- [ ] Enable calibration tracking to monitor confidence accuracy over time
- [ ] Set up Knowledge Mound for institutional memory

### Phase 3: Compliance Documentation (Before August 2, 2026)
- [ ] Generate technical documentation from decision receipt archives
- [ ] Compile Gauntlet test reports for risk management file
- [ ] Document human oversight procedures (approval gates, override protocols)
- [ ] Prepare calibration reports showing accuracy vs. confidence alignment

---

## Relevant EU AI Act Articles (Quick Reference)

| Article | Topic | Key Obligation |
|---------|-------|---------------|
| Art. 6 | Classification rules | Defines what qualifies as high-risk |
| Art. 9 | Risk management | Continuous risk identification and mitigation |
| Art. 10 | Data governance | Training data quality, bias testing |
| Art. 11 | Technical documentation | Pre-market documentation, kept up to date |
| Art. 12 | Record-keeping | Automatic event logging over system lifetime |
| Art. 13 | Transparency | Interpretable output, known limitations |
| Art. 14 | Human oversight | Effective human control and intervention |
| Art. 15 | Accuracy/robustness | Resilience to errors and adversarial attacks |
| Art. 16 | Provider obligations | Overall compliance responsibility |
| Art. 26 | Deployer obligations | Proper use, monitoring, record retention |
| Art. 72 | Penalties | Up to 35M EUR or 7% of global turnover |

---

## Contact and Resources

- **Documentation**: [docs/EXTENDED_README.md](EXTENDED_README.md)
- **Decision Receipt API**: [docs/api/API_REFERENCE.md](api/API_REFERENCE.md)
- **Gauntlet Testing Guide**: [aragora/gauntlet/README.md](../aragora/gauntlet/README.md)
- **RBAC Configuration**: [docs/reference/CONFIGURATION.md](reference/CONFIGURATION.md)
