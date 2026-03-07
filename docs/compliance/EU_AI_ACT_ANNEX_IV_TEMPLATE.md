# Annex IV Technical Documentation Template

> Annex IV of the EU AI Act (Regulation (EU) 2024/1689) requires providers of
> high-risk AI systems to prepare technical documentation before market placement.
> This template maps each required section to what Aragora auto-generates and
> provides placeholder text for the items your organization must fill in.
>
> **How to use this template:** Copy this file, replace every `[PLACEHOLDER]`
> with your organization's information, and integrate the Aragora-generated
> artifacts referenced in each section. The result is your Annex IV technical file.

---

## Section 1: General Description of the AI System

**What the regulation requires:** A general description including the intended
purpose, the provider's identity, the system version, how the system interacts
with hardware or software that is not part of the system itself, and the versions
of relevant software or firmware.

**What Aragora auto-generates:**
- Provider identity fields (Art. 13 artifact `provider_identity`)
- System name and version (Art. 12 artifact `annex_iv_sec1_general`)
- Intended purpose and exclusions (Art. 13 artifact `intended_purpose`)

Generate with: `aragora compliance eu-ai-act generate receipt.json --output ./bundle/`

**Fill in the following:**

> **System Name:** [SYSTEM_NAME]
>
> **Version:** [VERSION_NUMBER]
>
> **Provider:** [ORGANIZATION_NAME]
>
> **Provider Contact:** [COMPLIANCE_EMAIL]
>
> **EU Authorized Representative:** [EU_REP_NAME_AND_ADDRESS] (required if provider is outside the EU)
>
> **Intended Purpose:** [DESCRIBE_THE_SPECIFIC_DECISIONS_THIS_SYSTEM_SUPPORTS --
> e.g., "Automated screening and ranking of job applications for open positions
> in the EU, with human review of all shortlisted candidates."]
>
> **Not Intended For:** [LIST_EXCLUDED_USES -- e.g., "Fully autonomous hiring
> decisions without human review; real-time biometric identification; social scoring."]
>
> **Interaction with Other Systems:** [DESCRIBE_INTEGRATIONS -- e.g., "Receives
> candidate data from ATS (Workday). Outputs recommendations via REST API to
> internal review dashboard. No direct access to production databases."]
>
> **Hardware/Software Dependencies:** [LIST_DEPENDENCIES -- e.g., "Runs on
> AWS eu-west-1. Requires Python 3.11+, PostgreSQL 15+. Communicates with
> Anthropic, OpenAI, and Mistral APIs via HTTPS."]
>
> **Date of Market Placement:** [DATE_OR_PLANNED_DATE]

---

## Section 2: Detailed Description of Elements and Development Process

**What the regulation requires:** A detailed description of the development
process including: design specifications, system architecture, the computational
and hardware resources used, the data requirements and data sheets, and a
description of how the system was built and operates.

**What Aragora auto-generates:**
- Architecture description (Art. 12 artifact `annex_iv_sec2_development`)
- Consensus method and protocol (weighted majority, adversarial, etc.)
- Agent roster with provider identities
- Round count and debate configuration

Generate with: `aragora compliance eu-ai-act generate receipt.json --output ./bundle/`

Reference: `article_12_record_keeping.json` field `technical_documentation.annex_iv_sec2_development`

**Fill in the following:**

> **Design Specifications:**
>
> Architecture: Aragora multi-agent debate engine with adversarial consensus.
> See generated artifact `annex_iv_sec2_development` for architecture details.
>
> [ADD_ANY_CUSTOM_MODULES_OR_EXTENSIONS_YOUR_ORGANIZATION_HAS_BUILT]
>
> **Computational Resources:** [DESCRIBE_INFRASTRUCTURE -- e.g., "2x c5.xlarge
> EC2 instances (eu-west-1), 8 vCPU / 16 GB RAM each. PostgreSQL RDS
> db.r6g.large. Average decision latency: 45 seconds for 3 agents, 3 rounds."]
>
> **AI Models Used:**
>
> | Provider | Model | Version | Role in System |
> |----------|-------|---------|----------------|
> | [PROVIDER_1] | [MODEL_NAME] | [VERSION] | [ROLE -- e.g., "Primary debater"] |
> | [PROVIDER_2] | [MODEL_NAME] | [VERSION] | [ROLE -- e.g., "Adversarial challenger"] |
> | [PROVIDER_3] | [MODEL_NAME] | [VERSION] | [ROLE -- e.g., "Consensus arbiter"] |
>
> **Data Requirements:** [DESCRIBE_INPUT_DATA -- e.g., "System receives structured
> decision prompts (text, max 4000 tokens). No PII is included in prompts.
> Organizational knowledge base (internal policies, compliance documents)
> is injected as context. No training data is used -- all models are pre-trained
> by their respective providers."]
>
> **Development Methodology:** [DESCRIBE_HOW_THE_SYSTEM_WAS_VALIDATED -- e.g.,
> "Debate configurations validated against 50 benchmark decisions across 5
> domains. Gauntlet adversarial stress tests run quarterly. ELO calibration
> tracks per-agent reliability over time."]

---

## Section 3: Monitoring, Functioning, and Control

**What the regulation requires:** A detailed description of how the system is
monitored, how it functions in operation, and what control mechanisms exist for
human oversight, including measures to facilitate interpretation of outputs.

**What Aragora auto-generates:**
- Human oversight model -- HITL or HOTL (Art. 14 artifact `oversight_model`)
- Override and intervention mechanisms (Art. 14 artifact `override_capability`, `intervention_capability`)
- Automation bias safeguards (Art. 14 artifact `automation_bias_safeguards`)
- Monitoring features (Art. 14 artifact `understanding_monitoring.monitoring_features`)
- Explainability features (Art. 14 artifact `interpretation_features`)

Reference: `article_14_human_oversight.json`

**Fill in the following:**

> **Oversight Model:** [HITL_OR_HOTL] -- [DESCRIBE_YOUR_OVERSIGHT_APPROACH --
> e.g., "Human-in-the-Loop. All AI recommendations require explicit approval by
> a qualified reviewer before any downstream action is taken."]
>
> **Oversight Personnel:**
>
> | Role | Responsibility | Required Competency |
> |------|---------------|---------------------|
> | [ROLE_1 -- e.g., "Decision Reviewer"] | [RESPONSIBILITY -- e.g., "Reviews and approves/rejects all AI recommendations"] | [COMPETENCY -- e.g., "Domain expertise in HR/recruitment, completion of AI oversight training module"] |
> | [ROLE_2 -- e.g., "Compliance Officer"] | [RESPONSIBILITY -- e.g., "Periodic audit of decision quality and compliance scores"] | [COMPETENCY -- e.g., "EU AI Act training, access to compliance dashboard"] |
>
> **Monitoring Procedures:** [DESCRIBE_HOW_THE_SYSTEM_IS_MONITORED -- e.g.,
> "Real-time debate spectate view for in-progress decisions. Weekly compliance
> audit reports via `aragora compliance audit`. Monthly review of agent
> calibration drift. Quarterly Gauntlet stress tests."]
>
> **Intervention Procedures:** [DESCRIBE_HOW_A_HUMAN_STOPS_THE_SYSTEM -- e.g.,
> "Any reviewer can reject a verdict, override it with documented rationale, or
> halt a debate mid-round. All interventions are logged in the provenance chain.
> The system reaches a safe state (no downstream actions) when halted."]
>
> **Training Program:** [DESCRIBE_OVERSIGHT_TRAINING -- e.g., "All decision
> reviewers complete a 2-hour AI oversight training module covering: system
> capabilities and limitations, confidence score interpretation, dissent
> significance, and override procedures. Refresher training annually."]

---

## Section 4: Risk Management System

**What the regulation requires:** A detailed description of the risk management
system established per Article 9, including identified risks, risk estimation,
evaluation, and mitigation measures.

**What Aragora auto-generates:**
- Platform-level risks (Art. 13 artifact `known_risks` -- automation bias, hollow consensus, hallucination)
- Risk mitigations (Trickster, circuit breakers, calibration monitoring)
- Per-decision risk assessment (decision receipt `risk_summary`, `confidence`, `robustness_score`)
- Adversarial stress-test results (Gauntlet reports)

Reference: `article_12_record_keeping.json` field `technical_documentation.annex_iv_sec5_risk_management`;
`bundle.json` field `risk_classification`

**Fill in the following:**

> **Risk Management Policy:** [REFERENCE_YOUR_ORGANIZATION_RISK_MANAGEMENT_POLICY]
>
> **Identified Risks:**
>
> | Risk | Likelihood | Severity | Mitigation | Residual Risk | Owner |
> |------|-----------|----------|------------|---------------|-------|
> | Automation bias -- users over-rely on AI output | [L/M/H] | [L/M/H] | Mandatory human review, dissent highlighting, confidence calibration | [DESCRIBE_RESIDUAL] | [ROLE] |
> | Hollow consensus -- agents agree without substantive reasoning | [L/M/H] | [L/M/H] | Trickster detection module, evidence grounding requirements | [DESCRIBE_RESIDUAL] | [ROLE] |
> | Model hallucination -- plausible but incorrect claims | [L/M/H] | [L/M/H] | Multi-agent challenge, calibration tracking, human review | [DESCRIBE_RESIDUAL] | [ROLE] |
> | [DOMAIN_SPECIFIC_RISK_1] | [L/M/H] | [L/M/H] | [MITIGATION] | [RESIDUAL] | [ROLE] |
> | [DOMAIN_SPECIFIC_RISK_2] | [L/M/H] | [L/M/H] | [MITIGATION] | [RESIDUAL] | [ROLE] |
>
> **Risk Appetite Statement:** [DESCRIBE_YOUR_ORGANIZATION_RISK_APPETITE --
> e.g., "We accept low residual risk for routine decisions. High-impact
> decisions (budget >EUR 100K, personnel actions, regulatory filings) require
> a compliance score of 85+ and unanimous human approval."]
>
> **Testing and Validation:** Aragora Gauntlet adversarial stress tests run
> [FREQUENCY -- e.g., "quarterly"] covering fairness probes, adversarial
> manipulation resistance, and edge-case handling.
>
> Generate Gauntlet evidence: `aragora gauntlet run --suite fairness`
>
> **Residual Risk Acceptance:** [DOCUMENT_WHO_ACCEPTS_RESIDUAL_RISK_AND_ON_WHAT_BASIS]

---

## Section 5: Changes Throughout the Lifecycle

**What the regulation requires:** A description of changes made to the system
throughout its lifecycle, including version history, nature of changes, and
whether changes trigger a new conformity assessment.

**What Aragora auto-generates:**
- Per-decision versioning in receipt fields (system version, agent versions)
- Timestamped provenance chains for every decision
- Configuration change tracking

**Fill in the following:**

> **Version History:**
>
> | Version | Date | Nature of Change | Re-assessment Required |
> |---------|------|-----------------|----------------------|
> | [VERSION] | [DATE] | [DESCRIPTION -- e.g., "Added Mistral agent to debate roster"] | [YES/NO -- significant changes to agent composition or decision logic may require re-assessment] |
> | [VERSION] | [DATE] | [DESCRIPTION] | [YES/NO] |
>
> **Change Management Policy:** [REFERENCE_YOUR_CHANGE_MANAGEMENT_PROCEDURES --
> e.g., "All changes to debate configuration, agent roster, or consensus
> thresholds require approval from the Compliance Officer. Changes that
> materially alter decision-making behavior trigger a compliance bundle
> re-generation and review."]
>
> **Substantial Modification Criteria:** A change is considered substantial
> (requiring new conformity assessment per Art. 43) if it:
> - Changes the intended purpose of the system
> - Materially alters the risk profile (e.g., adding a new Annex III use case)
> - Replaces the core decision-making method
> - [ADD_ORGANIZATION_SPECIFIC_CRITERIA]

---

## Section 6: Standards Applied

**What the regulation requires:** A list of the harmonized standards (Art. 40)
or common specifications (Art. 41) applied, and where they have not been applied,
a description of the alternative solutions adopted.

**What Aragora auto-generates:** This section is not auto-generated. Standards
are organizational decisions.

**Fill in the following:**

> **Harmonized Standards Applied:**
>
> | Standard | Title | Scope of Application |
> |----------|-------|---------------------|
> | [STANDARD_ID -- e.g., "EN XXXXX"] | [TITLE] | [HOW_APPLIED -- e.g., "Risk management system per Art. 9"] |
>
> *Note: As of March 2026, CEN/CENELEC harmonized standards under the EU AI Act
> are being finalized. Update this section as standards are published in the
> Official Journal. Monitor the [CEN/CENELEC JTC 21](https://www.cencenelec.eu/)
> work program for publication dates.*
>
> **Common Specifications Applied:** [LIST_ANY_COMMON_SPECIFICATIONS_ADOPTED_PER_ART_41]
>
> **Alternative Solutions:** [IF_NO_HARMONIZED_STANDARD_EXISTS_FOR_A_REQUIREMENT,
> DESCRIBE_THE_ALTERNATIVE_APPROACH -- e.g., "In the absence of a harmonized
> standard for AI decision audit trails, we apply SHA-256 cryptographic integrity
> verification per NIST FIPS 180-4, with tamper-evident provenance chains
> following the W3C PROV-DM data model."]

---

## Section 7: EU Declaration of Conformity

**What the regulation requires:** A copy of the EU declaration of conformity
issued under Article 47. The declaration must include the provider's identity,
the AI system's identity, a statement that the system complies with the
applicable requirements, and the date and signature of the responsible person.

**What Aragora auto-generates:**
- Conformity report with per-article compliance status (conformity report artifact)
- Risk classification (bundle `risk_classification`)
- Integrity hash for tamper detection

Generate with: `aragora compliance audit receipt.json --format markdown`

Reference: `conformity_report.md` and `conformity_report.json` in the generated bundle

**Fill in the following:**

> ## EU Declaration of Conformity
>
> Issued pursuant to Article 47 of Regulation (EU) 2024/1689
>
> **Provider:**
> - Name: [ORGANIZATION_NAME]
> - Address: [REGISTERED_ADDRESS]
> - Contact: [COMPLIANCE_EMAIL]
>
> **EU Authorized Representative** (if applicable):
> - Name: [EU_REP_NAME]
> - Address: [EU_REP_ADDRESS]
>
> **AI System:**
> - Name: [SYSTEM_NAME]
> - Version: [VERSION]
> - Unique identification: [PRODUCT_ID_OR_SERIAL]
>
> **Risk Classification:**
> - Risk level: [HIGH]
> - Annex III category: [CATEGORY_NUMBER_AND_NAME]
>
> **Declaration:**
> This AI system has been designed, developed, and assessed in conformity with
> the requirements set out in Chapter III, Section 2 of Regulation (EU) 2024/1689,
> including Articles 9, 10, 11, 12, 13, 14, and 15.
>
> **Harmonized standards or common specifications applied:** [LIST_OR_STATE_NONE]
>
> **Conformity assessment procedure:** [INTERNAL_CONTROL_ANNEX_VI / THIRD_PARTY_ANNEX_VII]
>
> **Notified body** (if applicable):
> - Name: [NOTIFIED_BODY_NAME]
> - Identification number: [NANDO_ID]
> - Certificate number: [CERTIFICATE_NUMBER]
>
> **Supporting evidence:** Aragora Compliance Bundle [BUNDLE_ID], integrity hash
> [INTEGRITY_HASH]. Full conformity report attached.
>
> **Signed on behalf of the provider:**
>
> Name: [SIGNATORY_NAME]
> Title: [SIGNATORY_TITLE]
> Date: [DATE]
> Signature: ____________________

---

## Assembly Instructions

1. Copy this template to your compliance documentation repository
2. Fill in all `[PLACEHOLDER]` fields with your organization's information
3. Generate Aragora artifacts: `aragora compliance eu-ai-act generate receipt.json --output ./bundle/`
4. Attach the generated artifact files alongside each section as supporting evidence
5. Have your legal team review the completed document
6. Update Section 5 (Changes) whenever the system is modified
7. Update Section 6 (Standards) as CEN/CENELEC publishes harmonized standards

For the full submission package requirements, see the
[Notified Body Submission Checklist](./EU_AI_ACT_NOTIFIED_BODY_CHECKLIST.md).

---

*This template is based on Annex IV of the EU AI Act (Regulation (EU) 2024/1689).
It is not legal advice. The final structure and content of your technical
documentation should be validated by qualified legal counsel and, where applicable,
by your notified body.*
