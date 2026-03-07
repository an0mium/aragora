# Notified Body Submission Checklist

> Preparing for a conformity assessment under Article 43 of the EU AI Act.
> This checklist maps each notified body requirement to the artifacts Aragora
> generates and the items your organization must provide separately.

---

## Do You Need a Notified Body?

Article 43 defines two conformity assessment paths. Your choice depends on
whether your AI system falls under a harmonized standard with full coverage.

| Path | When It Applies | Who Performs It |
|------|----------------|-----------------|
| **Internal control** (Annex VI) | Your high-risk system is covered by harmonized standards listed in Art. 40, AND you have applied them in full | Your own quality management team |
| **Third-party assessment** (Annex VII) | No harmonized standard covers your system, or you have not applied one in full, or your system falls under Annex III point 1 (biometrics) | A notified body designated under Art. 28 |

**Rule of thumb:** If your AI system is used for biometric identification (Annex III
category 1), you must use a notified body. For all other Annex III categories, you
may use internal control if harmonized standards exist and you fully comply with them.
As of March 2026, harmonized standards under the EU AI Act are still being finalized by
CEN/CENELEC, so most organizations should plan for third-party assessment.

### Finding a Notified Body

Use the **NANDO database** (New Approach Notified and Designated Organisations):

> [https://ec.europa.eu/growth/tools-databases/nando/](https://ec.europa.eu/growth/tools-databases/nando/)

Search for bodies designated under **Regulation (EU) 2024/1689** (EU AI Act).
Filter by your Annex III category and Member State. Contact bodies early --
assessment timelines are typically 4-8 weeks and demand will increase as the
August 2, 2026 deadline approaches.

---

## Submission Requirements

A notified body performing a conformity assessment under Annex VII will
evaluate your technical documentation (Annex IV), quality management system,
and evidence of compliance with Articles 8-15. The table below maps each
requirement to what Aragora provides and what your organization must supply.

### Technical Documentation (Annex IV)

| Requirement | Aragora Provides | Customer Provides | Notes |
|-------------|-----------------|-------------------|-------|
| **1. General description** -- system name, version, intended purpose, provider identity | Art. 12 artifact `annex_iv_sec1_general`; Art. 13 artifact `provider_identity`, `intended_purpose` | Organizational context, market placement details, product labeling | Generated via `aragora compliance eu-ai-act generate` with `--provider-name`, `--system-name` flags |
| **2. Development process** -- architecture, design decisions, data pipeline, algorithms | Art. 12 artifact `annex_iv_sec2_development` (architecture, consensus method, agents, protocol, rounds) | Training data governance docs from model providers (Art. 10); internal validation procedures | Aragora documents the decision layer; model providers document the training layer |
| **3. Monitoring and control** -- human oversight mechanisms, logging capabilities | Art. 14 artifact (oversight model, override, stop capabilities); Art. 12 artifact (event log, retention policy) | Internal oversight procedures, escalation policies, personnel assignments | Aragora provides technical capabilities; you document organizational procedures |
| **4. Risk management system** -- identified risks, mitigations, residual risk | Art. 12 artifact `annex_iv_sec5_risk_management`; Art. 9 receipt fields (risk summary, confidence, robustness) | Organization-specific risk register, risk appetite statement, residual risk acceptance | Combine Aragora's platform-level risks with your domain-specific risk analysis |
| **5. Lifecycle changes** -- version history, change log, re-assessment triggers | Decision receipts with timestamps, version fields, provenance chains | Change management procedures, release governance, re-assessment policy | Aragora tracks per-decision versioning; you track system-level versioning |
| **6. Standards applied** -- list of harmonized standards or common specifications | Not applicable (standards are organizational decisions) | List of harmonized standards applied (Art. 40), common specifications adopted (Art. 41), or justification for alternatives | As CEN/CENELEC standards are published, map your implementation to them |
| **7. EU declaration of conformity** -- formal declaration per Art. 47 | Template fields populated from conformity report | Signed declaration with legal representative, CE marking decision | Aragora's conformity report provides the evidence basis; the declaration is a legal act by your organization |

### Quality Management System (Art. 17)

| Requirement | Aragora Provides | Customer Provides | Notes |
|-------------|-----------------|-------------------|-------|
| Quality management policy | -- | Written QMS policy covering AI system lifecycle | Organizational document |
| Design and development procedures | Debate protocol configuration, consensus method documentation | Procedures for configuring and validating Aragora deployments | How your team sets up and validates debate configurations |
| Testing and validation | Gauntlet stress-test results, robustness scores, calibration metrics | Test plans, acceptance criteria, validation datasets | `aragora gauntlet run --suite fairness` generates adversarial test evidence |
| Data management | Multi-model consensus as bias mitigation evidence | Data processing agreements with model providers | Document which providers are used for which decision categories |
| Post-market monitoring | Observability stack (Prometheus, OpenTelemetry), SLO alerting, audit dashboards | Monitoring plan, incident response procedures, corrective action process | `aragora compliance audit` on a periodic schedule |
| Serious incident reporting | Incident detection and logging | Reporting procedures, authority contact details, escalation timelines | Art. 73 requires reporting within specific timeframes |
| Communication with authorities | Compliance bundle export capabilities | Designated contact person, response procedures | Keep bundles ready for regulatory inquiry |

### Evidence of Article Compliance

| Article | Requirement | Aragora Evidence | Customer Evidence |
|---------|-------------|-----------------|-------------------|
| **Art. 9** | Risk management system | Decision receipts with risk summaries, Gauntlet adversarial results, conformity report Art. 9 mapping | Organization-level risk register, risk management policy, residual risk acceptance |
| **Art. 10** | Data governance | Multi-model consensus documentation, Gauntlet fairness probes | Model provider DPAs, training data documentation from each provider |
| **Art. 11** | Technical documentation | Per-article artifacts (Art. 12, 13, 14), Annex IV sections 1, 2, 5 | Complete Annex IV file integrating Aragora artifacts with organizational context |
| **Art. 12** | Record-keeping | SHA-256 hash chain provenance, timestamped event logs, retention policy artifact | Log retention infrastructure, access controls on stored logs |
| **Art. 13** | Transparency | Agent identities, reasoning chains, dissent records, confidence interpretation, known risks | Instructions for use document, user-facing documentation |
| **Art. 14** | Human oversight | HITL/HOTL model documentation, override/stop mechanisms, automation bias safeguards | Oversight personnel assignments, competency requirements, training records |
| **Art. 15** | Accuracy and robustness | Confidence scores, robustness scores, integrity hashes, cryptographic signatures, multi-agent consensus | Cybersecurity measures documentation, penetration test results, security audit reports |

---

## Submission Package Assembly

Compile these items into your submission package:

1. **Aragora compliance bundle** -- `aragora compliance eu-ai-act generate receipt.json --output ./submission/`
2. **Annex IV technical file** -- Use the Annex IV Template (`EU_AI_ACT_ANNEX_IV_TEMPLATE.md`) to integrate Aragora artifacts with your organizational documentation
3. **Quality management system documentation** -- Your QMS policies and procedures
4. **Risk management documentation** -- Organization-level risk register merged with Aragora risk evidence
5. **EU declaration of conformity draft** -- Prepared per Art. 47 (to be signed after successful assessment)
6. **Sample decision receipts** -- 3-5 representative receipts showing the system in production operation

### Timeline

| Weeks Before Deadline | Action |
|----------------------|--------|
| **16-20 weeks** | Identify notified body via NANDO; make initial contact |
| **12-16 weeks** | Assemble submission package; fill organizational documentation gaps |
| **8-12 weeks** | Submit package to notified body; respond to preliminary questions |
| **4-8 weeks** | Assessment engagement (review, site visit if required, findings) |
| **2-4 weeks** | Address any findings; obtain certificate |
| **0-2 weeks** | Sign EU declaration of conformity; register in EU database (Art. 49) |

---

## Common Notified Body Questions

| Question They Will Ask | Where to Find the Answer |
|-----------------------|--------------------------|
| How does the system reach decisions? | Art. 12 `annex_iv_sec2_development` -- architecture, consensus method, agents |
| What risks have you identified? | Art. 9 receipt fields + your organization-level risk register |
| How do you prevent automation bias? | Art. 14 artifact `automation_bias_safeguards` |
| Can a human stop the system? | Art. 14 artifact `intervention_capability` |
| How do you detect adversarial manipulation? | Art. 12 `annex_iv_sec5_risk_management` -- Trickster, circuit breakers |
| How do you ensure log integrity? | Art. 12 `retention_policy` -- SHA-256 hash chain, retention period |
| What training data is used? | Redirect to model provider documentation (Art. 10 is provider obligation) |
| How do you handle model updates? | Change management procedures (customer-owned) + Aragora versioning |

---

## Generate Your Submission Artifacts

```bash
# Step 1: Generate the full artifact bundle
aragora compliance eu-ai-act generate receipt.json \
  --output ./submission/ \
  --provider-name "Your Organization" \
  --provider-contact "compliance@your-org.com" \
  --eu-representative "Your EU Rep, City, Country" \
  --system-name "Your AI System" \
  --system-version "1.0.0"

# Step 2: Generate a conformity report
aragora compliance audit receipt.json --format markdown --output ./submission/conformity_report.md

# Step 3: Review compliance score
aragora compliance status
```

---

*This checklist is a preparation guide based on the EU AI Act (Regulation (EU) 2024/1689).
It is not legal advice. Conformity assessment requirements may evolve as harmonized
standards and implementing acts are published. Consult qualified legal counsel for your
organization's specific obligations.*
