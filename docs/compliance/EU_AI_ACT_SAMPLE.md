# EU AI Act Compliance Artifacts -- Sample Output

> This document shows what Aragora's EU AI Act compliance artifacts look like
> when generated from a real decision receipt. Every field below is produced
> automatically by `ComplianceArtifactGenerator` -- no manual authoring required.

**Scenario:** A procurement team uses Aragora to vet whether to onboard
Luminos Analytics as the primary data warehouse vendor for EU operations.
Three agents debate for 2 rounds. Consensus is reached at 87% confidence
with one dissenting view. A human approver signs off before the decision
is finalized.

**Risk classification:** HIGH (Annex III Category 2 -- Critical infrastructure),
because the vendor will handle digital infrastructure for EU data operations.

---

## How to Generate

### CLI

```bash
# Step 1: Run the decision through Aragora's debate engine
aragora ask "Should we onboard Luminos Analytics as our primary data warehouse vendor for EU operations?" \
    --agents anthropic-api,openai-api,mistral \
    --rounds 2 \
    --decision-integrity \
    --require-approval

# Step 2: Generate the full EU AI Act artifact bundle from the receipt
aragora compliance eu-ai-act generate receipt.json \
    --output ./compliance-bundle/ \
    --provider-name "Meridian Procurement GmbH" \
    --provider-contact "compliance@meridian-procurement.eu" \
    --eu-representative "Meridian Procurement GmbH, Frankfurt, Germany" \
    --system-name "Meridian Vendor Assessment Platform" \
    --system-version "3.1.0"
```

### Python API

```python
from aragora.compliance.eu_ai_act import ComplianceArtifactGenerator

generator = ComplianceArtifactGenerator(
    provider_name="Meridian Procurement GmbH",
    provider_contact="compliance@meridian-procurement.eu",
    eu_representative="Meridian Procurement GmbH, Frankfurt, Germany",
    system_name="Meridian Vendor Assessment Platform",
    system_version="3.1.0",
)

bundle = generator.generate(receipt.to_dict())

# Write per-article artifacts
import json, pathlib

out = pathlib.Path("compliance-bundle")
out.mkdir(exist_ok=True)
(out / "compliance_bundle.json").write_text(bundle.to_json(indent=2))
(out / "article_12_record_keeping.json").write_text(
    json.dumps(bundle.article_12.to_dict(), indent=2)
)
(out / "article_13_transparency.json").write_text(
    json.dumps(bundle.article_13.to_dict(), indent=2)
)
(out / "article_14_human_oversight.json").write_text(
    json.dumps(bundle.article_14.to_dict(), indent=2)
)
(out / "conformity_report.md").write_text(
    bundle.conformity_report.to_markdown()
)
```

### Output Files

```
compliance-bundle/
  compliance_bundle.json            # Full artifact bundle (all articles)
  article_12_record_keeping.json    # Event log, tech docs, retention policy
  article_13_transparency.json      # Provider identity, risks, interpretation
  article_14_human_oversight.json   # Oversight model, override, stop mechanisms
  conformity_report.md              # Human-readable conformity assessment
  conformity_report.json            # Machine-readable conformity assessment
```

---

## Sample Article 12 Artifact -- Record-Keeping

Article 12 of the EU AI Act requires automatic recording of events ("logs")
with traceability over the system's lifetime. Aragora maps this to the
provenance chain embedded in every decision receipt.

```json
{
  "article": "Article 12",
  "title": "Record-Keeping",
  "receipt_id": "RCP-VENDOR-2026-0087",
  "generated_at": "2026-02-23T14:32:08.741000+00:00",
  "event_log": [
    {
      "event_id": "evt_0001",
      "event_type": "debate_started",
      "timestamp": "2026-02-23T14:20:01.112000+00:00",
      "actor": "system"
    },
    {
      "event_id": "evt_0002",
      "event_type": "proposal_submitted",
      "timestamp": "2026-02-23T14:20:18.445000+00:00",
      "actor": "anthropic-api"
    },
    {
      "event_id": "evt_0003",
      "event_type": "proposal_submitted",
      "timestamp": "2026-02-23T14:20:22.891000+00:00",
      "actor": "openai-api"
    },
    {
      "event_id": "evt_0004",
      "event_type": "proposal_submitted",
      "timestamp": "2026-02-23T14:20:25.003000+00:00",
      "actor": "mistral"
    },
    {
      "event_id": "evt_0005",
      "event_type": "critique_submitted",
      "timestamp": "2026-02-23T14:22:41.677000+00:00",
      "actor": "openai-api"
    },
    {
      "event_id": "evt_0006",
      "event_type": "critique_submitted",
      "timestamp": "2026-02-23T14:22:44.219000+00:00",
      "actor": "mistral"
    },
    {
      "event_id": "evt_0007",
      "event_type": "revision_submitted",
      "timestamp": "2026-02-23T14:24:11.530000+00:00",
      "actor": "anthropic-api"
    },
    {
      "event_id": "evt_0008",
      "event_type": "consensus_reached",
      "timestamp": "2026-02-23T14:25:03.882000+00:00",
      "actor": "system"
    },
    {
      "event_id": "evt_0009",
      "event_type": "human_approval",
      "timestamp": "2026-02-23T14:30:47.215000+00:00",
      "actor": "j.weber@meridian-procurement.eu"
    },
    {
      "event_id": "evt_0010",
      "event_type": "receipt_generated",
      "timestamp": "2026-02-23T14:30:48.001000+00:00",
      "actor": "system"
    }
  ],
  "reference_databases": [
    {
      "source": "meridian_vendor_registry",
      "type": "knowledge_base"
    },
    {
      "source": "eu_gdpr_compliance_corpus",
      "type": "knowledge_base"
    }
  ],
  "input_record": {
    "input_summary": "Should we onboard Luminos Analytics as our primary data warehouse vendor for EU operations?",
    "input_hash": "f95b734f8c3d788c068ffe3883b2e1929d91de31db14879c032eb087dc3fc1e9",
    "agents_participating": [
      "anthropic-api",
      "openai-api",
      "mistral"
    ]
  },
  "technical_documentation": {
    "annex_iv_sec1_general": {
      "system_name": "Meridian Vendor Assessment Platform",
      "version": "3.1.0",
      "provider": "Meridian Procurement GmbH",
      "intended_purpose": "Multi-agent adversarial vetting of decisions against organizational knowledge, delivering audit-ready decision receipts."
    },
    "annex_iv_sec2_development": {
      "architecture": "Multi-agent debate with adversarial consensus",
      "consensus_method": "weighted_majority",
      "agents": [
        "anthropic-api",
        "openai-api",
        "mistral"
      ],
      "protocol": "adversarial",
      "rounds": 2
    },
    "annex_iv_sec5_risk_management": {
      "adversarial_debate": "Multi-agent challenge reduces single-point-of-failure",
      "hollow_consensus_detection": "Trickster module",
      "circuit_breakers": "Per-agent failure isolation",
      "calibration_monitoring": "Continuous Brier score tracking"
    }
  },
  "retention_policy": {
    "minimum_months": 6,
    "basis": "Art. 26(6) — minimum 6 months for high-risk systems",
    "provenance_events": 10,
    "integrity_mechanism": "SHA-256 hash chain"
  }
}
```

**What an evaluator sees:** A complete, timestamped event log covering the
full decision lifecycle -- from debate start through agent proposals, critiques,
revision, consensus, human approval, and receipt generation. Each event is
tied to a specific actor. The Annex IV technical documentation sections map
directly to the regulation's structure. The retention policy cites the
specific article (Art. 26(6)) requiring minimum 6-month retention for
high-risk systems.

---

## Sample Article 13 Artifact -- Transparency

Article 13 requires providers to ensure their systems are sufficiently
transparent for deployers to interpret output and use it appropriately.
Aragora maps this to provider identity, known risk catalogs, confidence
interpretation, and dissent records.

```json
{
  "article": "Article 13",
  "title": "Transparency and Provision of Information to Deployers",
  "receipt_id": "RCP-VENDOR-2026-0087",
  "generated_at": "2026-02-23T14:32:08.741000+00:00",
  "provider_identity": {
    "name": "Meridian Procurement GmbH",
    "contact": "compliance@meridian-procurement.eu",
    "eu_representative": "Meridian Procurement GmbH, Frankfurt, Germany"
  },
  "intended_purpose": {
    "description": "Aragora orchestrates adversarial debate among heterogeneous AI models to vet decisions against organizational knowledge. It produces audit-ready decision receipts with cryptographic integrity for regulatory compliance.",
    "not_intended_for": [
      "Fully autonomous decision-making without human oversight",
      "Real-time biometric identification",
      "Social scoring or behavioral manipulation"
    ]
  },
  "accuracy_robustness": {
    "consensus_confidence": 0.87,
    "robustness_score": 0.74,
    "agents_participating": 3,
    "consensus_method": "weighted_majority",
    "agreement_ratio": 0.87,
    "integrity_hash_present": true,
    "signature_present": true
  },
  "known_risks": [
    {
      "risk": "Automation bias",
      "description": "Over-reliance on AI recommendations",
      "mitigation": "Mandatory human review, dissent highlighting",
      "article_ref": "Art. 14(4)(b)"
    },
    {
      "risk": "Hollow consensus",
      "description": "Surface-level agreement without substantive reasoning",
      "mitigation": "Trickster detection module, evidence grounding",
      "article_ref": "Art. 15(4)"
    },
    {
      "risk": "Model hallucination",
      "description": "Plausible but incorrect claims persisting through consensus",
      "mitigation": "Multi-agent challenge, calibration tracking",
      "article_ref": "Art. 15(1)"
    }
  ],
  "output_interpretation": {
    "verdict": "Approve onboarding with conditions: require SOC 2 Type II report, GDPR DPA, and quarterly access reviews.",
    "confidence": 0.87,
    "confidence_interpretation": "High confidence — strong agreement",
    "dissent_count": 1,
    "dissent_significance": "1 dissenting view(s) recorded. Review dissenting reasoning before finalizing."
  },
  "human_oversight_reference": {
    "human_approval_detected": true,
    "approval_config": {
      "require_approval": true,
      "approver_id": "j.weber@meridian-procurement.eu"
    }
  }
}
```

**What an evaluator sees:** The provider is clearly identified with EU contact
details. The system's intended purpose and explicit exclusions are documented.
Accuracy metrics include both confidence (0.87) and robustness (0.74) with
plain-language interpretation. Three known risks are cataloged with specific
mitigations and article cross-references. The single dissenting view is flagged
with a recommendation to review it before finalizing. Human approval is confirmed
with an auditable approver identity.

---

## Sample Article 14 Artifact -- Human Oversight

Article 14 requires high-risk AI systems to be designed and developed so that
they can be effectively overseen by natural persons. Aragora maps this to
oversight models, bias safeguards, override mechanisms, and stop capabilities.

```json
{
  "article": "Article 14",
  "title": "Human Oversight",
  "receipt_id": "RCP-VENDOR-2026-0087",
  "generated_at": "2026-02-23T14:32:08.741000+00:00",
  "oversight_model": {
    "primary": "Human-in-the-Loop (HITL)",
    "description": "All final decisions require explicit human approval.",
    "human_approval_detected": true
  },
  "understanding_monitoring": {
    "capabilities_documented": [
      "Multi-agent adversarial debate with consensus",
      "Tamper-evident decision receipts",
      "Calibration tracking per agent",
      "Dissent recording for minority opinions"
    ],
    "limitations_documented": [
      "Consensus does not guarantee correctness",
      "Confidence != probability of being correct",
      "Performance varies by domain complexity",
      "Underlying model knowledge cutoff dates apply"
    ],
    "monitoring_features": [
      "Real-time debate spectate view",
      "Agent performance dashboard",
      "Calibration drift alerts",
      "Anomaly detection"
    ]
  },
  "automation_bias_safeguards": {
    "warnings_present": true,
    "mechanisms": [
      "Dissent views prominently displayed alongside verdict",
      "Confidence scores presented with interpretation context",
      "Periodic independent evaluation prompts",
      "Mandatory review intervals"
    ]
  },
  "interpretation_features": {
    "explainability": [
      "Factor decomposition: contributing factors with weights",
      "Counterfactual analysis: what-if scenarios",
      "Evidence chain: claims linked to sources",
      "Vote pivot: which arguments changed outcomes"
    ]
  },
  "override_capability": {
    "override_available": true,
    "mechanisms": [
      {
        "action": "Reject verdict",
        "description": "Deployer rejects AI consensus and decides independently",
        "audit_logged": true
      },
      {
        "action": "Override with reason",
        "description": "Deployer overrides with documented rationale",
        "audit_logged": true
      },
      {
        "action": "Reverse prior decision",
        "description": "Previously accepted decisions can be reversed",
        "audit_logged": true
      }
    ]
  },
  "intervention_capability": {
    "stop_available": true,
    "mechanisms": [
      {
        "action": "Stop debate",
        "description": "Halts debate mid-round, partial results preserved",
        "safe_state": true
      },
      {
        "action": "Cancel decision",
        "description": "Cancels in-progress decision, no downstream actions",
        "safe_state": true
      }
    ]
  }
}
```

**What an evaluator sees:** The oversight model is explicitly identified as
Human-in-the-Loop (HITL), confirmed by actual detection in the receipt.
Capabilities and limitations are documented side by side -- the system does not
overstate its reliability. Four automation bias safeguards are in place, including
prominent dissent display. Three override mechanisms exist, each audit-logged.
Two intervention ("stop") mechanisms confirm the system reaches a safe state
when halted. This directly satisfies Art. 14(4)(a) through 14(4)(e).

---

## Sample Conformity Report

The conformity report maps each applicable article to a PASS/PARTIAL/FAIL
status with evidence. This is the human-readable markdown output from
`bundle.conformity_report.to_markdown()`.

```
# EU AI Act Conformity Report

**Report ID:** EUAIA-e4d21b08
**Receipt ID:** RCP-VENDOR-2026-0087
**Generated:** 2026-02-23T14:32:08.741000+00:00
**Integrity Hash:** `a014ba0955f1d388...`

---

## Risk Classification

**Risk Level:** HIGH
**Annex III Category:** 2. Critical infrastructure
**Rationale:** Use case falls under Annex III category 2: Critical infrastructure.
Safety components in critical digital infrastructure, road traffic, water, gas,
heating, electricity.

### Obligations

- Establish and maintain a risk management system (Art. 9).
- Use high-quality training, validation, and testing data (Art. 10).
- Maintain technical documentation (Art. 11).
- Implement automatic logging of events (Art. 12).
- Ensure transparency and provide instructions for deployers (Art. 13).
- Design for effective human oversight (Art. 14).
- Achieve appropriate accuracy, robustness, and cybersecurity (Art. 15).
- Register in the EU database before placing on market (Art. 49).
- Undergo conformity assessment (Art. 43).

---

## Article Compliance Assessment

**Overall Status:** CONFORMANT

| Article | Requirement | Status | Evidence |
|---------|-------------|--------|----------|
| Article 9 | Identify and analyze known and reasonably foreseeab... | PASS | Risk assessment performed: 3 risks identified (0 critical). Confidence: 87.0%. |
| Article 12 | Automatic logging of events with traceability | PASS | Provenance chain contains 10 events. |
| Article 13 | Identify participating agents, their arguments, and... | PASS | 3 agents participated. Verdict reasoning: Approve onboarding with conditions... |
| Article 14 | Enable human oversight, including ability to overrid... | PASS | Human approval/override mechanism detected in receipt configuration. |
| Article 15 | Appropriate levels of accuracy and robustness; resi... | PASS | Robustness score: 74.0%. Integrity hash: present. Cryptographic signature: present. |

---

## Summary

Conformity assessment for receipt RCP-VENDOR-2026-0087 against the EU AI Act.
Risk level: high. 5/5 applicable article requirements satisfied.
```

---

## Sample Bundle Envelope

The top-level `compliance_bundle.json` wraps all artifacts into a single
auditable package with a SHA-256 integrity hash computed from the bundle ID,
receipt ID, risk level, and conformity status.

```json
{
  "bundle_id": "EUAIA-7c3a9f12",
  "regulation": "EU AI Act (Regulation 2024/1689)",
  "compliance_deadline": "2026-08-02",
  "receipt_id": "RCP-VENDOR-2026-0087",
  "generated_at": "2026-02-23T14:32:08.741000+00:00",
  "risk_classification": {
    "risk_level": "high",
    "annex_iii_category": "Critical infrastructure",
    "annex_iii_number": 2,
    "rationale": "Use case falls under Annex III category 2: Critical infrastructure. Safety components in critical digital infrastructure, road traffic, water, gas, heating, electricity.",
    "matched_keywords": ["digital infrastructure"],
    "applicable_articles": [
      "Article 6 (Classification)",
      "Article 9 (Risk management)",
      "Article 13 (Transparency)",
      "Article 14 (Human oversight)",
      "Article 15 (Accuracy, robustness, cybersecurity)"
    ],
    "obligations": [
      "Establish and maintain a risk management system (Art. 9).",
      "Use high-quality training, validation, and testing data (Art. 10).",
      "Maintain technical documentation (Art. 11).",
      "Implement automatic logging of events (Art. 12).",
      "Ensure transparency and provide instructions for deployers (Art. 13).",
      "Design for effective human oversight (Art. 14).",
      "Achieve appropriate accuracy, robustness, and cybersecurity (Art. 15).",
      "Register in the EU database before placing on market (Art. 49).",
      "Undergo conformity assessment (Art. 43)."
    ]
  },
  "conformity_report": {
    "report_id": "EUAIA-e4d21b08",
    "receipt_id": "RCP-VENDOR-2026-0087",
    "generated_at": "2026-02-23T14:32:08.741000+00:00",
    "overall_status": "conformant",
    "article_mappings": [
      {
        "article": "Article 9",
        "article_title": "Risk management system",
        "requirement": "Identify and analyze known and reasonably foreseeable risks",
        "receipt_field": "risk_summary, confidence",
        "status": "satisfied",
        "evidence": "Risk assessment performed: 3 risks identified (0 critical). Confidence: 87.0%.",
        "recommendation": ""
      },
      {
        "article": "Article 12",
        "article_title": "Record-keeping",
        "requirement": "Automatic logging of events with traceability",
        "receipt_field": "provenance_chain",
        "status": "satisfied",
        "evidence": "Provenance chain contains 10 events.",
        "recommendation": ""
      },
      {
        "article": "Article 13",
        "article_title": "Transparency and provision of information to deployers",
        "requirement": "Identify participating agents, their arguments, and decision rationale",
        "receipt_field": "consensus_proof, verdict_reasoning, dissenting_views",
        "status": "satisfied",
        "evidence": "3 agents participated. Verdict reasoning: Approve onboarding with conditions: require SOC 2 Type II report, GDPR DPA, and quarterly access r.... 1 dissenting view(s) recorded.",
        "recommendation": ""
      },
      {
        "article": "Article 14",
        "article_title": "Human oversight",
        "requirement": "Enable human oversight, including ability to override or halt",
        "receipt_field": "config_used",
        "status": "satisfied",
        "evidence": "Human approval/override mechanism detected in receipt configuration.",
        "recommendation": ""
      },
      {
        "article": "Article 15",
        "article_title": "Accuracy, robustness and cybersecurity",
        "requirement": "Appropriate levels of accuracy and robustness; resilience to attacks",
        "receipt_field": "robustness_score, artifact_hash, signature",
        "status": "satisfied",
        "evidence": "Robustness score: 74.0%. Integrity hash: present. Cryptographic signature: present.",
        "recommendation": ""
      }
    ],
    "summary": "Conformity assessment for receipt RCP-VENDOR-2026-0087 against the EU AI Act. Risk level: high. 5/5 applicable article requirements satisfied.",
    "recommendations": [],
    "integrity_hash": "a014ba0955f1d3885a19f07fcfd861224838fc8c7126890203bae8ddbeaa2373"
  },
  "article_12_record_keeping": { "...": "see Article 12 artifact above" },
  "article_13_transparency": { "...": "see Article 13 artifact above" },
  "article_14_human_oversight": { "...": "see Article 14 artifact above" },
  "integrity_hash": "5a76139841419513566cb3c6f99b4673c8264924a5f9cc3b6ee9913ba9f7fbd2"
}
```

The `integrity_hash` at the bundle level is computed as:

```
SHA-256(json.dumps({
    "bundle_id": "EUAIA-7c3a9f12",
    "receipt_id": "RCP-VENDOR-2026-0087",
    "risk_level": "high",
    "conformity_status": "conformant"
}, sort_keys=True))
```

Any modification to the bundle ID, receipt ID, risk level, or conformity status
will produce a different hash, making tampering detectable.

---

## Verifying Integrity

To verify a bundle has not been tampered with:

```python
import hashlib, json

bundle = json.load(open("compliance-bundle/compliance_bundle.json"))

# Recompute and compare
content = json.dumps({
    "bundle_id": bundle["bundle_id"],
    "receipt_id": bundle["receipt_id"],
    "risk_level": bundle["risk_classification"]["risk_level"],
    "conformity_status": bundle["conformity_report"]["overall_status"],
}, sort_keys=True)

expected = hashlib.sha256(content.encode()).hexdigest()
assert expected == bundle["integrity_hash"], "Bundle integrity check failed!"
print(f"Integrity verified: {expected[:16]}...")
```

---

## Related Documentation

- [EU AI Act Compliance Guide](./EU_AI_ACT_GUIDE.md) -- Full reference with risk tiers, article explanations, CLI commands, and FAQ
- [Gauntlet Testing](../../aragora/gauntlet/README.md) -- Adversarial stress testing for robustness scores
- [Enterprise Compliance](../enterprise/COMPLIANCE.md) -- Operational controls and governance model
