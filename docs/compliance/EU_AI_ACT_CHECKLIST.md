# EU AI Act Compliance Checklist

> Track your organization's compliance progress for the EU AI Act (Regulation
> (EU) 2024/1689). Enforcement deadline for high-risk AI systems:
> **August 2, 2026**.
>
> Each item includes the relevant article reference and the Aragora feature
> that addresses it. Items marked with (P) are provider obligations; items
> marked with (D) are deployer obligations.

---

## Phase 0: Risk Assessment and Classification

- [ ] **Inventory all AI systems** used in your organization
- [ ] **Classify each system** by EU AI Act risk level
  - Use `aragora compliance classify "<description>"` for automated classification
  - Article 6 + Annex III define high-risk categories
- [ ] **Identify Annex III matches** -- systems in employment, credit, infrastructure, law enforcement, education, migration, justice, or biometrics
- [ ] **Flag any prohibited practices** (Art. 5) -- social scoring, subliminal manipulation, real-time biometric ID for law enforcement
- [ ] **Document classification decisions** with rationale for each system
  - Aragora: `RiskClassifier.classify()` returns `RiskClassification` with `rationale` and `matched_keywords`

---

## Phase 1: Risk Management System (Article 9)

- [ ] **(P) Establish a risk management system** for each high-risk AI system (Art. 9(1))
- [ ] **(P) Identify and analyze known and foreseeable risks** (Art. 9(2)(a))
  - Aragora: Gauntlet adversarial stress testing (`aragora gauntlet`)
  - Aragora: Risk summary in decision receipts (`risk_summary` field)
- [ ] **(P) Estimate and evaluate risks from intended use** (Art. 9(2)(b))
  - Aragora: Multi-agent debate surfaces risks from multiple perspectives
- [ ] **(P) Evaluate risks from reasonably foreseeable misuse** (Art. 9(2)(c))
  - Aragora: Red-team agent personas (Devil's Advocate, Scaling Critic)
- [ ] **(P) Adopt risk management measures** (Art. 9(4))
  - Aragora: Calibrated trust via ELO rankings + Brier scores
  - Aragora: Conformity report recommendations
- [ ] **(P) Document risk management process** and keep updated
  - Aragora: Conformity report `article_mappings` for Art. 9
- [ ] **(D) Use the system in accordance with instructions** (Art. 26(1))

---

## Phase 2: Data Governance (Article 10)

- [ ] **(P) Ensure training data quality** (Art. 10(2))
  - Note: Aragora does not train models; this obligation falls on model providers
- [ ] **(P) Examine training data for biases** (Art. 10(2)(f))
  - Aragora: Multi-model consensus reduces single-provider bias
  - Aragora: Gauntlet fairness probes test for demographic bias
- [ ] **(P) Identify data gaps and shortcomings** (Art. 10(2)(g))
  - Aragora: Dissent records surface uncertainty and disagreement

---

## Phase 3: Technical Documentation (Article 11)

- [ ] **(P) Prepare technical documentation before market placement** (Art. 11(1))
  - Aragora: Art. 12 artifact includes Annex IV technical documentation
- [ ] **(P) Document general system description** (Annex IV, Section 1)
  - Aragora: `technical_documentation.annex_iv_sec1_general` in Art. 12 artifact
- [ ] **(P) Document development process** (Annex IV, Section 2)
  - Aragora: `annex_iv_sec2_development` with architecture, consensus method, agents
- [ ] **(P) Document risk management measures** (Annex IV, Section 5)
  - Aragora: `annex_iv_sec5_risk_management` with adversarial debate, circuit breakers
- [ ] **(P) Keep documentation up to date** (Art. 11(1))

---

## Phase 4: Record-Keeping and Logging (Article 12)

- [ ] **(P) Implement automatic event logging** (Art. 12(1))
  - Aragora: Every decision receipt includes a `provenance_chain` of timestamped events
  - Aragora: Art. 12 artifact `event_log` with event IDs, types, timestamps, actors
- [ ] **(P) Enable identification of risk-generating situations** (Art. 12(2))
  - Aragora: Anomaly detection (`aragora/security/anomaly_detection.py`)
  - Aragora: Calibration drift alerts
- [ ] **(P) Facilitate post-market monitoring** (Art. 12(3))
  - Aragora: Continuum memory (4-tier) preserves decision history
  - Aragora: Knowledge Mound institutional memory
- [ ] **(P) Ensure log traceability** (Art. 12(1))
  - Aragora: SHA-256 hash chains in provenance, tamper-evident receipts
- [ ] **(D) Retain automatically generated logs for minimum 6 months** (Art. 26(6))
  - Aragora: Art. 12 artifact `retention_policy` documents this obligation

---

## Phase 5: Transparency (Article 13)

- [ ] **(P) Design system for transparent operation** (Art. 13(1))
- [ ] **(P) Provide instructions for use to deployers** (Art. 13(1))
  - Aragora: Art. 13 artifact `intended_purpose` with description and exclusions
- [ ] **(P) Document provider identity and contact** (Art. 13(3)(a))
  - Aragora: Art. 13 artifact `provider_identity` (name, contact, EU representative)
- [ ] **(P) Document accuracy and robustness metrics** (Art. 13(3)(b)(ii))
  - Aragora: Art. 13 artifact `accuracy_robustness` (confidence, robustness score, agreement ratio)
- [ ] **(P) Document known limitations** (Art. 13(3)(b)(iv))
  - Aragora: Decision receipts explicitly record dissenting views
  - Aragora: Art. 14 artifact `understanding_monitoring.limitations_documented`
- [ ] **(P) Document foreseeable unintended outcomes** (Art. 13(3)(b)(v))
  - Aragora: Art. 13 artifact `known_risks` (automation bias, hollow consensus, hallucination)
- [ ] **(P) Provide output interpretation guidance** (Art. 13(3)(b)(vi))
  - Aragora: Art. 13 artifact `output_interpretation` with confidence context
- [ ] **(P) Cross-reference human oversight measures** (Art. 13(3)(b)(vii))
  - Aragora: Art. 13 artifact `human_oversight_reference`

---

## Phase 6: Human Oversight (Article 14)

- [ ] **(P) Design for effective human oversight** (Art. 14(1))
  - Aragora: Art. 14 artifact `oversight_model` (HITL or HOTL)
- [ ] **(P) Enable humans to understand capabilities and limitations** (Art. 14(4)(a))
  - Aragora: Art. 14 artifact `understanding_monitoring` (capabilities, limitations, monitoring features)
  - Aragora: Calibrated confidence scores based on historical accuracy
- [ ] **(P) Enable humans to correctly interpret output** (Art. 14(4)(b))
  - Aragora: Art. 14 artifact `interpretation_features` (factor decomposition, counterfactuals, evidence chains)
  - Aragora: Dissent views prominently displayed alongside verdict
- [ ] **(P) Implement automation bias safeguards** (Art. 14(4)(b))
  - Aragora: Art. 14 artifact `automation_bias_safeguards` (warnings, dissent display, review prompts)
- [ ] **(P) Enable humans to decide not to use the system** (Art. 14(4)(c))
  - Aragora: Decision receipts are advisory; approval gates required
- [ ] **(P) Enable humans to override or stop the system** (Art. 14(4)(d))
  - Aragora: Art. 14 artifact `override_capability` (reject, override with reason, reverse)
  - Aragora: Art. 14 artifact `intervention_capability` (stop debate, cancel decision)
- [ ] **(D) Assign competent human oversight personnel** (Art. 26(2))
- [ ] **(D) Ensure human overseers understand the system** (Art. 26(5))

---

## Phase 7: Accuracy, Robustness and Cybersecurity (Article 15)

- [ ] **(P) Achieve appropriate accuracy levels** (Art. 15(1))
  - Aragora: Heterogeneous multi-model consensus
  - Aragora: ELO skill rankings track domain-specific reliability
- [ ] **(P) Declare accuracy levels in documentation** (Art. 15(2))
  - Aragora: Conformity report Art. 15 mapping with robustness score
- [ ] **(P) Ensure resilience to errors and faults** (Art. 15(4))
  - Aragora: Circuit breakers, retry with exponential backoff, agent fallback
- [ ] **(P) Ensure resilience to adversarial manipulation** (Art. 15(5))
  - Aragora: Trickster (hollow consensus detection), SecurityBarrier, rate limiting
- [ ] **(P) Implement cybersecurity measures** (Art. 15(5))
  - Aragora: AES-256-GCM encryption, key rotation, RBAC, MFA, API key management

---

## Phase 8: Conformity Assessment (Article 43)

- [ ] **(P) Undergo conformity assessment before market placement** (Art. 43)
  - Aragora: `aragora compliance eu-ai-act generate` produces supporting artifacts
- [ ] **(P) Generate compliance artifact bundle**
  - Aragora: CLI command produces Art. 12, 13, 14 artifacts with integrity hash
- [ ] **(P) Retain conformity assessment documentation**
- [ ] **(P) Register in EU database** (Art. 49) before market placement

---

## Phase 9: Post-Market Obligations

- [ ] **(P) Implement post-market monitoring system** (Art. 72)
  - Aragora: Observability stack (Prometheus, OpenTelemetry, SLO alerting)
- [ ] **(P) Report serious incidents to authorities** (Art. 73)
- [ ] **(D) Monitor AI system operation** (Art. 26(5))
  - Aragora: Real-time debate spectate, agent performance dashboard
- [ ] **(D) Inform provider of serious incidents** (Art. 26(5))
- [ ] **(D) Retain logs for minimum 6 months** (Art. 26(6))
  - Aragora: Art. 12 retention policy artifact
- [ ] **(D) Conduct data protection impact assessment** when applicable (Art. 26(9))

---

## Phase 10: Ongoing Compliance

- [ ] **Regularly re-classify systems** as use cases evolve
- [ ] **Update technical documentation** when system changes
- [ ] **Regenerate compliance artifact bundles** periodically
- [ ] **Review and update risk management measures** continuously
- [ ] **Monitor regulatory guidance** from the EU AI Office
- [ ] **Train staff** on EU AI Act obligations and Aragora's compliance features

---

## Quick Reference: Aragora Commands

| Task | Command |
|------|---------|
| Classify AI use case | `aragora compliance classify "description"` |
| Generate conformity report | `aragora compliance audit receipt.json` |
| Generate full artifact bundle | `aragora compliance eu-ai-act generate receipt.json -o ./bundle/` |
| Generate demo bundle | `aragora compliance eu-ai-act generate -o ./demo-bundle/` |
| Run adversarial stress test | `aragora gauntlet run document.md` |
| Run full decision pipeline | `aragora decide "task" --agents ...` |

---

## Penalty Reference (Article 72)

| Violation | Maximum Fine |
|-----------|-------------|
| Prohibited AI practices (Art. 5) | 35M EUR or 7% of global annual turnover |
| High-risk system non-compliance | 15M EUR or 3% of global annual turnover |
| Providing incorrect information to authorities | 7.5M EUR or 1% of global annual turnover |

---

*This checklist is a guide based on the EU AI Act (Regulation (EU) 2024/1689)
as published. It is not legal advice. Consult qualified legal counsel for your
organization's specific compliance obligations.*
