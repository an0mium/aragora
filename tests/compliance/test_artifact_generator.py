"""
Tests for compliance artifact generator.

Covers:
- ReceiptComplianceGenerator: unified generation across all frameworks
- EUAIActArtifact: risk classification, transparency, human oversight, data governance
- SOC2Artifact: trust service criteria mapping, control evidence, exceptions
- HIPAAArtifact: PHI handling, access control, audit trail, minimum necessary, breach notification
- ComplianceArtifactResult: multi-framework generation, JSON serialization
- DecisionReceipt.generate_compliance_artifacts integration
- Framework selection and validation
"""

from __future__ import annotations

import json

import pytest

from aragora.compliance.artifact_generator import (
    ComplianceArtifactResult,
    EUAIActArtifact,
    HIPAAArtifact,
    ReceiptComplianceGenerator,
    SOC2Artifact,
    TransparencyChecklist,
    TrustServiceCriteria,
)
from aragora.gauntlet.receipt_models import (
    ConsensusProof,
    DecisionReceipt,
    ProvenanceRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def generator():
    return ReceiptComplianceGenerator()


@pytest.fixture
def sample_receipt_dict() -> dict:
    """A well-formed receipt dict with all fields populated."""
    return {
        "receipt_id": "test-receipt-001",
        "gauntlet_id": "gauntlet-001",
        "timestamp": "2026-02-22T00:00:00Z",
        "input_summary": "Evaluate patient treatment recommendation system",
        "input_hash": "abc123def456",
        "risk_summary": {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 1,
            "total": 4,
        },
        "attacks_attempted": 5,
        "attacks_successful": 1,
        "probes_run": 10,
        "vulnerabilities_found": 4,
        "verdict": "CONDITIONAL",
        "confidence": 0.75,
        "robustness_score": 0.8,
        "verdict_reasoning": "Treatment recommendation shows acceptable accuracy with minor bias concerns",
        "dissenting_views": ["Agent-B: potential bias in recommendation for elderly patients"],
        "consensus_proof": {
            "reached": True,
            "confidence": 0.75,
            "supporting_agents": ["Agent-A", "Agent-C"],
            "dissenting_agents": ["Agent-B"],
            "method": "adversarial_validation",
            "evidence_hash": "abc123",
        },
        "provenance_chain": [
            {
                "timestamp": "2026-02-22T00:00:01Z",
                "event_type": "attack",
                "agent": "Agent-A",
                "description": "[HIGH] Bias probe on age factor",
                "evidence_hash": "hash1",
            },
            {
                "timestamp": "2026-02-22T00:00:02Z",
                "event_type": "probe",
                "agent": "Agent-C",
                "description": "[MEDIUM] Data coverage check",
                "evidence_hash": "hash2",
            },
            {
                "timestamp": "2026-02-22T00:00:03Z",
                "event_type": "verdict",
                "agent": None,
                "description": "Verdict: CONDITIONAL (75.0% confidence)",
                "evidence_hash": "",
            },
        ],
        "explainability": None,
        "schema_version": "1.1",
        "artifact_hash": "deadbeef1234567890",
        "config_used": {
            "rounds": 3,
            "participants": ["Agent-A", "Agent-B", "Agent-C"],
            "human_approval": True,
        },
        "signature": "sig_abc123",
        "signature_algorithm": "HMAC-SHA256",
        "signature_key_id": "key-001",
        "signed_at": "2026-02-22T00:01:00Z",
    }


@pytest.fixture
def minimal_receipt_dict() -> dict:
    """A minimal receipt dict with sparse fields."""
    return {
        "receipt_id": "test-receipt-minimal",
        "gauntlet_id": "gauntlet-minimal",
        "timestamp": "2026-02-22T00:00:00Z",
        "input_summary": "Review internal process documentation",
        "input_hash": "",
        "risk_summary": {"critical": 0, "high": 0, "medium": 0, "low": 0, "total": 0},
        "attacks_attempted": 0,
        "attacks_successful": 0,
        "probes_run": 0,
        "vulnerabilities_found": 0,
        "verdict": "PASS",
        "confidence": 0.9,
        "robustness_score": 0.95,
        "verdict_reasoning": "Documentation review complete with no issues found",
        "dissenting_views": [],
        "consensus_proof": {
            "reached": True,
            "confidence": 0.9,
            "supporting_agents": ["Agent-A"],
            "dissenting_agents": [],
            "method": "majority",
            "evidence_hash": "",
        },
        "provenance_chain": [
            {
                "timestamp": "2026-02-22T00:00:01Z",
                "event_type": "verdict",
                "description": "Verdict: PASS",
            },
        ],
        "artifact_hash": "",
        "config_used": {},
    }


@pytest.fixture
def hiring_receipt_dict() -> dict:
    """Receipt about recruitment/hiring (EU AI Act high-risk)."""
    return {
        "receipt_id": "test-receipt-hiring",
        "gauntlet_id": "gauntlet-hiring",
        "timestamp": "2026-02-22T00:00:00Z",
        "input_summary": "CV screening algorithm for recruitment decisions",
        "input_hash": "hash456",
        "risk_summary": {"critical": 0, "high": 0, "medium": 1, "low": 0, "total": 1},
        "attacks_attempted": 3,
        "attacks_successful": 0,
        "probes_run": 5,
        "vulnerabilities_found": 1,
        "verdict": "PASS",
        "confidence": 0.85,
        "robustness_score": 0.9,
        "verdict_reasoning": "Recruitment screening passed adversarial validation",
        "dissenting_views": [],
        "consensus_proof": {
            "reached": True,
            "confidence": 0.85,
            "supporting_agents": ["Agent-A", "Agent-B"],
            "dissenting_agents": [],
            "method": "adversarial_validation",
        },
        "provenance_chain": [
            {"timestamp": "2026-02-22T00:00:01Z", "event_type": "attack"},
            {"timestamp": "2026-02-22T00:00:02Z", "event_type": "verdict"},
        ],
        "artifact_hash": "hash789",
        "config_used": {"require_approval": True},
    }


@pytest.fixture
def decision_receipt_object() -> DecisionReceipt:
    """A DecisionReceipt dataclass instance for integration testing."""
    return DecisionReceipt(
        receipt_id="test-receipt-obj-001",
        gauntlet_id="gauntlet-obj-001",
        timestamp="2026-02-22T00:00:00Z",
        input_summary="Evaluate loan application scoring model",
        input_hash="sha256_loan_hash",
        risk_summary={"critical": 0, "high": 0, "medium": 1, "low": 0, "total": 1},
        attacks_attempted=4,
        attacks_successful=0,
        probes_run=8,
        vulnerabilities_found=1,
        verdict="PASS",
        confidence=0.82,
        robustness_score=0.88,
        verdict_reasoning="Credit scoring model passed validation",
        dissenting_views=[],
        consensus_proof=ConsensusProof(
            reached=True,
            confidence=0.82,
            supporting_agents=["Agent-A", "Agent-B"],
            method="adversarial_validation",
        ),
        provenance_chain=[
            ProvenanceRecord(
                timestamp="2026-02-22T00:00:01Z",
                event_type="attack",
                agent="Agent-A",
                description="Bias test on income brackets",
            ),
            ProvenanceRecord(
                timestamp="2026-02-22T00:00:02Z",
                event_type="verdict",
                description="Verdict: PASS (82.0% confidence)",
            ),
        ],
        config_used={"rounds": 3, "human_approval": True},
    )


# ---------------------------------------------------------------------------
# TransparencyChecklist
# ---------------------------------------------------------------------------


class TestTransparencyChecklist:
    def test_all_satisfied_when_all_true(self):
        checklist = TransparencyChecklist(
            agents_identified=True,
            decision_rationale_provided=True,
            dissenting_views_recorded=True,
            confidence_score_disclosed=True,
            ai_system_disclosed=True,
            limitations_documented=True,
        )
        assert checklist.all_satisfied is True

    def test_not_all_satisfied_when_one_false(self):
        checklist = TransparencyChecklist(
            agents_identified=True,
            decision_rationale_provided=False,
            dissenting_views_recorded=True,
            confidence_score_disclosed=True,
            ai_system_disclosed=True,
            limitations_documented=True,
        )
        assert checklist.all_satisfied is False

    def test_to_dict(self):
        checklist = TransparencyChecklist(agents_identified=True)
        d = checklist.to_dict()
        assert isinstance(d, dict)
        assert d["agents_identified"] is True
        assert "decision_rationale_provided" in d


# ---------------------------------------------------------------------------
# EU AI Act Artifact Generator
# ---------------------------------------------------------------------------


class TestGenerateEUAIAct:
    def test_generates_artifact_with_required_fields(self, generator, sample_receipt_dict):
        artifact = generator.generate_eu_ai_act(sample_receipt_dict)

        assert isinstance(artifact, EUAIActArtifact)
        assert artifact.artifact_id.startswith("EUAIA-")
        assert artifact.receipt_id == "test-receipt-001"
        assert artifact.generated_at  # Not empty
        assert artifact.risk_classification in ("minimal", "limited", "high", "unacceptable")
        assert artifact.risk_rationale  # Not empty
        assert isinstance(artifact.transparency_checklist, TransparencyChecklist)
        assert isinstance(artifact.human_oversight, dict)
        assert isinstance(artifact.data_governance, dict)
        assert isinstance(artifact.technical_documentation, dict)
        assert artifact.integrity_hash  # Not empty

    def test_high_risk_classification_for_healthcare(self, generator, sample_receipt_dict):
        # "patient" in input_summary triggers healthcare/high-risk keywords
        # Actually "patient" maps to PHI but not directly to EU high-risk.
        # Let's use a hiring receipt which contains "recruitment"
        sample_receipt_dict["input_summary"] = "Recruitment CV screening algorithm"
        artifact = generator.generate_eu_ai_act(sample_receipt_dict)
        assert artifact.risk_classification == "high"
        assert "Article 6" in str(artifact.applicable_articles)

    def test_unacceptable_risk_for_social_scoring(self, generator, sample_receipt_dict):
        sample_receipt_dict["input_summary"] = "Implement social scoring system for citizens"
        artifact = generator.generate_eu_ai_act(sample_receipt_dict)
        assert artifact.risk_classification == "unacceptable"
        assert "Article 5" in str(artifact.applicable_articles)

    def test_limited_risk_for_chatbot(self, generator, sample_receipt_dict):
        sample_receipt_dict["input_summary"] = "Deploy customer service chatbot"
        sample_receipt_dict["verdict_reasoning"] = "Chatbot performs within expectations"
        artifact = generator.generate_eu_ai_act(sample_receipt_dict)
        assert artifact.risk_classification == "limited"

    def test_minimal_risk_for_generic_content(self, generator, minimal_receipt_dict):
        artifact = generator.generate_eu_ai_act(minimal_receipt_dict)
        assert artifact.risk_classification == "minimal"
        assert artifact.applicable_articles == []

    def test_transparency_checklist_populated(self, generator, sample_receipt_dict):
        artifact = generator.generate_eu_ai_act(sample_receipt_dict)
        checklist = artifact.transparency_checklist

        assert checklist.agents_identified is True  # Has supporting_agents
        assert checklist.decision_rationale_provided is True  # Has verdict_reasoning
        assert checklist.dissenting_views_recorded is True  # Has dissenting_views
        assert checklist.confidence_score_disclosed is True  # confidence > 0
        assert checklist.ai_system_disclosed is True  # Always true

    def test_human_oversight_detected(self, generator, sample_receipt_dict):
        # sample_receipt_dict has "human_approval": True in config_used
        artifact = generator.generate_eu_ai_act(sample_receipt_dict)
        assert artifact.human_oversight["human_in_the_loop"] is True
        assert artifact.human_oversight["oversight_model"] == "HITL"

    def test_human_oversight_not_detected(self, generator, minimal_receipt_dict):
        artifact = generator.generate_eu_ai_act(minimal_receipt_dict)
        assert artifact.human_oversight["human_in_the_loop"] is False
        assert artifact.human_oversight["oversight_model"] == "HOTL"

    def test_data_governance_with_signature(self, generator, sample_receipt_dict):
        artifact = generator.generate_eu_ai_act(sample_receipt_dict)
        dg = artifact.data_governance

        assert dg["input_integrity_verified"] is True
        assert dg["input_hash_algorithm"] == "SHA-256"
        assert dg["output_integrity_verified"] is True
        assert dg["cryptographic_signature"] is True

    def test_data_governance_without_signature(self, generator, minimal_receipt_dict):
        artifact = generator.generate_eu_ai_act(minimal_receipt_dict)
        dg = artifact.data_governance

        assert dg["input_integrity_verified"] is False
        assert dg["cryptographic_signature"] is False

    def test_technical_documentation_populated(self, generator, sample_receipt_dict):
        artifact = generator.generate_eu_ai_act(sample_receipt_dict)
        td = artifact.technical_documentation

        assert "system_description" in td
        assert "performance_metrics" in td
        assert "risk_assessment" in td
        assert "logging" in td
        assert td["performance_metrics"]["confidence"] == 0.75
        assert td["risk_assessment"]["risks_identified"] == 4

    def test_json_serialization(self, generator, sample_receipt_dict):
        artifact = generator.generate_eu_ai_act(sample_receipt_dict)
        json_str = artifact.to_json()
        parsed = json.loads(json_str)

        assert parsed["framework"] == "eu_ai_act"
        assert parsed["receipt_id"] == "test-receipt-001"
        assert "transparency_checklist" in parsed
        assert "human_oversight" in parsed

    def test_integrity_hash_is_deterministic(self, generator, sample_receipt_dict):
        # Two artifacts with the same ID/receipt should produce the same hash
        a1 = generator.generate_eu_ai_act(sample_receipt_dict)
        # Manually construct with same IDs
        a2 = EUAIActArtifact(
            artifact_id=a1.artifact_id,
            receipt_id=a1.receipt_id,
            generated_at=a1.generated_at,
            risk_classification=a1.risk_classification,
            risk_rationale=a1.risk_rationale,
            transparency_checklist=a1.transparency_checklist,
            human_oversight=a1.human_oversight,
            data_governance=a1.data_governance,
            technical_documentation=a1.technical_documentation,
        )
        assert a1.integrity_hash == a2.integrity_hash


# ---------------------------------------------------------------------------
# SOC 2 Artifact Generator
# ---------------------------------------------------------------------------


class TestGenerateSOC2:
    def test_generates_artifact_with_required_fields(self, generator, sample_receipt_dict):
        artifact = generator.generate_soc2(sample_receipt_dict)

        assert isinstance(artifact, SOC2Artifact)
        assert artifact.artifact_id.startswith("SOC2-")
        assert artifact.receipt_id == "test-receipt-001"
        assert artifact.generated_at  # Not empty
        assert isinstance(artifact.trust_service_criteria, TrustServiceCriteria)
        assert isinstance(artifact.control_evidence, list)
        assert isinstance(artifact.exceptions, list)
        assert isinstance(artifact.period_of_review, dict)
        assert artifact.integrity_hash  # Not empty

    def test_trust_service_criteria_all_present(self, generator, sample_receipt_dict):
        artifact = generator.generate_soc2(sample_receipt_dict)
        tsc = artifact.trust_service_criteria.to_dict()

        assert "security" in tsc
        assert "availability" in tsc
        assert "processing_integrity" in tsc
        assert "confidentiality" in tsc
        assert "privacy" in tsc

    def test_security_criteria_with_hashes(self, generator, sample_receipt_dict):
        artifact = generator.generate_soc2(sample_receipt_dict)
        security = artifact.trust_service_criteria.security

        assert security["status"] == "satisfied"
        assert security["controls"]["cryptographic_integrity"]["input_hash"] is True
        assert security["controls"]["cryptographic_integrity"]["artifact_hash"] is True
        assert security["controls"]["cryptographic_integrity"]["signature"] is True

    def test_security_criteria_partial_without_hashes(self, generator, minimal_receipt_dict):
        artifact = generator.generate_soc2(minimal_receipt_dict)
        security = artifact.trust_service_criteria.security

        assert security["status"] == "partial"

    def test_availability_with_agents(self, generator, sample_receipt_dict):
        artifact = generator.generate_soc2(sample_receipt_dict)
        availability = artifact.trust_service_criteria.availability

        assert availability["status"] == "satisfied"
        assert "3 agents" in availability["controls"]["multi_agent_redundancy"]

    def test_processing_integrity_with_consensus(self, generator, sample_receipt_dict):
        artifact = generator.generate_soc2(sample_receipt_dict)
        pi = artifact.trust_service_criteria.processing_integrity

        assert pi["status"] == "satisfied"
        assert pi["controls"]["consensus_validation"]["confidence"] == 0.75
        assert pi["controls"]["consensus_validation"]["consensus_reached"] is True

    def test_control_evidence_includes_hash(self, generator, sample_receipt_dict):
        artifact = generator.generate_soc2(sample_receipt_dict)

        hash_evidence = [
            e for e in artifact.control_evidence if e["evidence_type"] == "artifact_hash"
        ]
        assert len(hash_evidence) == 1
        assert hash_evidence[0]["control"] == "CC6.1 - Integrity Controls"

    def test_control_evidence_includes_signature(self, generator, sample_receipt_dict):
        artifact = generator.generate_soc2(sample_receipt_dict)

        sig_evidence = [
            e for e in artifact.control_evidence if e["evidence_type"] == "digital_signature"
        ]
        assert len(sig_evidence) == 1

    def test_control_evidence_includes_provenance(self, generator, sample_receipt_dict):
        artifact = generator.generate_soc2(sample_receipt_dict)

        prov_evidence = [
            e for e in artifact.control_evidence if e["evidence_type"] == "provenance_chain"
        ]
        assert len(prov_evidence) == 1
        assert prov_evidence[0]["event_count"] == 3

    def test_control_evidence_includes_consensus(self, generator, sample_receipt_dict):
        artifact = generator.generate_soc2(sample_receipt_dict)

        cons_evidence = [
            e for e in artifact.control_evidence if e["evidence_type"] == "consensus_proof"
        ]
        assert len(cons_evidence) == 1

    def test_exceptions_for_high_risk(self, generator, sample_receipt_dict):
        artifact = generator.generate_soc2(sample_receipt_dict)

        high_exceptions = [e for e in artifact.exceptions if e["severity"] == "high"]
        assert len(high_exceptions) == 1
        assert "1 high-severity" in high_exceptions[0]["description"]

    def test_exceptions_for_critical_risk(self, generator):
        receipt = {
            "receipt_id": "critical-receipt",
            "risk_summary": {"critical": 2, "high": 0, "medium": 0, "low": 0, "total": 2},
            "consensus_proof": {"supporting_agents": [], "dissenting_agents": []},
            "provenance_chain": [],
            "verdict": "FAIL",
            "confidence": 0.3,
            "robustness_score": 0.2,
            "artifact_hash": "hash",
            "input_hash": "hash",
        }
        artifact = generator.generate_soc2(receipt)

        critical_exceptions = [e for e in artifact.exceptions if e["severity"] == "critical"]
        assert len(critical_exceptions) == 1
        assert "2 critical" in critical_exceptions[0]["description"]

    def test_no_signature_exception(self, generator, minimal_receipt_dict):
        artifact = generator.generate_soc2(minimal_receipt_dict)

        sig_exceptions = [e for e in artifact.exceptions if "signed" in e["description"].lower()]
        assert len(sig_exceptions) == 1

    def test_period_of_review(self, generator, sample_receipt_dict):
        artifact = generator.generate_soc2(sample_receipt_dict)

        assert "start" in artifact.period_of_review
        assert "end" in artifact.period_of_review
        assert artifact.period_of_review["type"] == "point-in-time"
        assert artifact.period_of_review["start"] == "2026-02-22T00:00:00Z"

    def test_service_organization(self, generator, sample_receipt_dict):
        artifact = generator.generate_soc2(sample_receipt_dict)

        assert artifact.service_organization["name"] == "Aragora Inc."
        assert "system" in artifact.service_organization

    def test_json_serialization(self, generator, sample_receipt_dict):
        artifact = generator.generate_soc2(sample_receipt_dict)
        json_str = artifact.to_json()
        parsed = json.loads(json_str)

        assert parsed["framework"] == "soc2"
        assert parsed["standard"] == "SOC 2 Type II (AICPA Trust Services Criteria)"
        assert "trust_service_criteria" in parsed
        assert "control_evidence" in parsed
        assert "exceptions" in parsed


# ---------------------------------------------------------------------------
# HIPAA Artifact Generator
# ---------------------------------------------------------------------------


class TestGenerateHIPAA:
    def test_generates_artifact_with_required_fields(self, generator, sample_receipt_dict):
        artifact = generator.generate_hipaa(sample_receipt_dict)

        assert isinstance(artifact, HIPAAArtifact)
        assert artifact.artifact_id.startswith("HIPAA-")
        assert artifact.receipt_id == "test-receipt-001"
        assert artifact.generated_at  # Not empty
        assert isinstance(artifact.phi_handling, dict)
        assert isinstance(artifact.access_control, dict)
        assert isinstance(artifact.audit_trail, dict)
        assert isinstance(artifact.minimum_necessary, dict)
        assert isinstance(artifact.breach_notification, dict)
        assert isinstance(artifact.safeguards, dict)
        assert artifact.integrity_hash  # Not empty

    def test_phi_detected_in_healthcare_content(self, generator, sample_receipt_dict):
        # sample_receipt_dict has "patient" in input_summary
        artifact = generator.generate_hipaa(sample_receipt_dict)

        assert artifact.phi_handling["phi_detected_in_summary"] is True
        assert "PHI-related" in artifact.phi_handling["attestation"]

    def test_phi_not_detected_in_generic_content(self, generator, minimal_receipt_dict):
        artifact = generator.generate_hipaa(minimal_receipt_dict)

        assert artifact.phi_handling["phi_detected_in_summary"] is False
        assert "No PHI" in artifact.phi_handling["attestation"]

    def test_phi_handling_encryption_fields(self, generator, sample_receipt_dict):
        artifact = generator.generate_hipaa(sample_receipt_dict)
        phi = artifact.phi_handling

        assert phi["encryption_at_rest"] == "AES-256-GCM (configurable)"
        assert phi["encryption_in_transit"] == "TLS 1.2+"
        assert phi["input_hash_present"] is True

    def test_access_control_verification(self, generator, sample_receipt_dict):
        artifact = generator.generate_hipaa(sample_receipt_dict)
        ac = artifact.access_control

        assert ac["regulation_ref"] == "45 CFR 164.312(a) - Access Control"
        assert ac["status"] == "satisfied"
        assert ac["unique_user_identification"]["agents_identified"] is True
        assert len(ac["unique_user_identification"]["agent_names"]) == 3  # A, B, C

    def test_access_control_partial_without_hash(self, generator, minimal_receipt_dict):
        artifact = generator.generate_hipaa(minimal_receipt_dict)
        ac = artifact.access_control

        # minimal has input_hash="" but has agents
        assert ac["status"] == "partial"

    def test_audit_trail_reference(self, generator, sample_receipt_dict):
        artifact = generator.generate_hipaa(sample_receipt_dict)
        at = artifact.audit_trail

        assert at["regulation_ref"] == "45 CFR 164.312(b) - Audit Controls"
        assert at["status"] == "satisfied"  # 3 provenance events >= 2
        assert at["provenance_chain_length"] == 3
        assert at["tamper_evident"] is True
        assert at["cryptographically_signed"] is True
        assert len(at["events_logged"]) == 3

    def test_audit_trail_partial_with_few_events(self, generator, minimal_receipt_dict):
        artifact = generator.generate_hipaa(minimal_receipt_dict)
        at = artifact.audit_trail

        assert at["status"] == "partial"  # Only 1 event

    def test_minimum_necessary_standard(self, generator, sample_receipt_dict):
        artifact = generator.generate_hipaa(sample_receipt_dict)
        mn = artifact.minimum_necessary

        assert mn["regulation_ref"] == "45 CFR 164.502(b) - Minimum Necessary Standard"
        assert mn["status"] == "satisfied"
        assert "controls" in mn
        assert "data_minimization" in mn["controls"]
        assert "purpose_limitation" in mn["controls"]

    def test_breach_notification_readiness(self, generator, sample_receipt_dict):
        artifact = generator.generate_hipaa(sample_receipt_dict)
        bn = artifact.breach_notification

        assert bn["regulation_ref"] == "45 CFR 164.404-164.410 - Breach Notification Rule"
        assert bn["readiness_status"] == "prepared"
        assert bn["capabilities"]["detection"]["tamper_detection"] is True
        assert bn["capabilities"]["detection"]["signature_verification"] is True
        assert "notification_timeline" in bn["capabilities"]

    def test_breach_notification_without_signature(self, generator, minimal_receipt_dict):
        artifact = generator.generate_hipaa(minimal_receipt_dict)
        bn = artifact.breach_notification

        assert bn["capabilities"]["detection"]["signature_verification"] is False

    def test_safeguards_all_categories(self, generator, sample_receipt_dict):
        artifact = generator.generate_hipaa(sample_receipt_dict)
        sg = artifact.safeguards

        assert "administrative" in sg
        assert "physical" in sg
        assert "technical" in sg
        assert sg["administrative"]["risk_analysis"] is True
        assert sg["technical"]["integrity_controls"]["digital_signature"] is True

    def test_json_serialization(self, generator, sample_receipt_dict):
        artifact = generator.generate_hipaa(sample_receipt_dict)
        json_str = artifact.to_json()
        parsed = json.loads(json_str)

        assert parsed["framework"] == "hipaa"
        assert parsed["regulation"] == "HIPAA (Health Insurance Portability and Accountability Act)"
        assert "phi_handling" in parsed
        assert "access_control" in parsed
        assert "audit_trail" in parsed
        assert "minimum_necessary" in parsed
        assert "breach_notification" in parsed
        assert "safeguards" in parsed


# ---------------------------------------------------------------------------
# Unified Generation and Framework Selection
# ---------------------------------------------------------------------------


class TestUnifiedGeneration:
    def test_generate_all_frameworks(self, generator, sample_receipt_dict):
        result = generator.generate(sample_receipt_dict)

        assert isinstance(result, ComplianceArtifactResult)
        assert result.receipt_id == "test-receipt-001"
        assert set(result.frameworks_generated) == {"eu_ai_act", "soc2", "hipaa"}
        assert result.eu_ai_act is not None
        assert result.soc2 is not None
        assert result.hipaa is not None

    def test_generate_single_framework(self, generator, sample_receipt_dict):
        result = generator.generate(sample_receipt_dict, frameworks=["soc2"])

        assert result.frameworks_generated == ["soc2"]
        assert result.soc2 is not None
        assert result.eu_ai_act is None
        assert result.hipaa is None

    def test_generate_two_frameworks(self, generator, sample_receipt_dict):
        result = generator.generate(sample_receipt_dict, frameworks=["eu_ai_act", "hipaa"])

        assert set(result.frameworks_generated) == {"eu_ai_act", "hipaa"}
        assert result.eu_ai_act is not None
        assert result.hipaa is not None
        assert result.soc2 is None

    def test_unsupported_framework_raises_error(self, generator, sample_receipt_dict):
        with pytest.raises(ValueError, match="Unsupported framework"):
            generator.generate(sample_receipt_dict, frameworks=["pci_dss"])

    def test_mixed_valid_invalid_frameworks_raises_error(self, generator, sample_receipt_dict):
        with pytest.raises(ValueError, match="Unsupported framework"):
            generator.generate(sample_receipt_dict, frameworks=["soc2", "iso27001"])

    def test_result_to_dict(self, generator, sample_receipt_dict):
        result = generator.generate(sample_receipt_dict)
        d = result.to_dict()

        assert d["receipt_id"] == "test-receipt-001"
        assert "eu_ai_act" in d
        assert "soc2" in d
        assert "hipaa" in d
        assert d["eu_ai_act"]["framework"] == "eu_ai_act"
        assert d["soc2"]["framework"] == "soc2"
        assert d["hipaa"]["framework"] == "hipaa"

    def test_result_to_json(self, generator, sample_receipt_dict):
        result = generator.generate(sample_receipt_dict)
        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["receipt_id"] == "test-receipt-001"
        assert len(parsed["frameworks_generated"]) == 3

    def test_result_to_dict_omits_none_frameworks(self, generator, sample_receipt_dict):
        result = generator.generate(sample_receipt_dict, frameworks=["hipaa"])
        d = result.to_dict()

        assert "hipaa" in d
        assert "eu_ai_act" not in d
        assert "soc2" not in d

    def test_custom_organization_name(self, sample_receipt_dict):
        gen = ReceiptComplianceGenerator(organization_name="Acme Corp.")
        result = gen.generate(sample_receipt_dict, frameworks=["soc2"])

        assert result.soc2.service_organization["name"] == "Acme Corp."


# ---------------------------------------------------------------------------
# DecisionReceipt Integration
# ---------------------------------------------------------------------------


class TestDecisionReceiptIntegration:
    def test_generate_compliance_artifacts_method_exists(self, decision_receipt_object):
        assert hasattr(decision_receipt_object, "generate_compliance_artifacts")

    def test_generate_all_frameworks_via_receipt(self, decision_receipt_object):
        result = decision_receipt_object.generate_compliance_artifacts()

        assert isinstance(result, ComplianceArtifactResult)
        assert result.receipt_id == "test-receipt-obj-001"
        assert result.eu_ai_act is not None
        assert result.soc2 is not None
        assert result.hipaa is not None

    def test_generate_selected_frameworks_via_receipt(self, decision_receipt_object):
        result = decision_receipt_object.generate_compliance_artifacts(
            frameworks=["eu_ai_act", "soc2"]
        )

        assert result.eu_ai_act is not None
        assert result.soc2 is not None
        assert result.hipaa is None

    def test_receipt_eu_risk_classification(self, decision_receipt_object):
        # "loan application" + "credit scoring" => high risk
        result = decision_receipt_object.generate_compliance_artifacts(frameworks=["eu_ai_act"])
        # "credit scoring" is in the input_summary => high risk
        assert result.eu_ai_act.risk_classification == "high"

    def test_receipt_soc2_has_control_evidence(self, decision_receipt_object):
        result = decision_receipt_object.generate_compliance_artifacts(frameworks=["soc2"])
        assert len(result.soc2.control_evidence) > 0

    def test_receipt_hipaa_detects_no_phi(self, decision_receipt_object):
        # "loan application" doesn't contain PHI keywords
        result = decision_receipt_object.generate_compliance_artifacts(frameworks=["hipaa"])
        assert result.hipaa.phi_handling["phi_detected_in_summary"] is False

    def test_receipt_custom_org_name(self, decision_receipt_object):
        result = decision_receipt_object.generate_compliance_artifacts(
            frameworks=["soc2"],
            organization_name="Test Corp.",
        )
        assert result.soc2.service_organization["name"] == "Test Corp."


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_receipt(self, generator):
        empty = {
            "receipt_id": "",
            "consensus_proof": None,
            "provenance_chain": [],
            "input_summary": "",
            "input_hash": "",
            "verdict": "",
            "confidence": 0.0,
            "robustness_score": 0.0,
            "risk_summary": {},
            "artifact_hash": "",
            "config_used": {},
            "verdict_reasoning": "",
            "dissenting_views": [],
        }
        result = generator.generate(empty)

        assert result.eu_ai_act is not None
        assert result.soc2 is not None
        assert result.hipaa is not None
        # Should not crash, all artifacts should be generated

    def test_receipt_with_provenance_objects_not_dicts(self, generator):
        """Test that ProvenanceRecord objects in provenance chain are handled."""
        receipt = {
            "receipt_id": "test-prov-obj",
            "input_summary": "",
            "input_hash": "",
            "verdict": "PASS",
            "confidence": 0.5,
            "robustness_score": 0.5,
            "risk_summary": {"total": 0},
            "artifact_hash": "",
            "config_used": {},
            "verdict_reasoning": "",
            "dissenting_views": [],
            "consensus_proof": {"supporting_agents": [], "dissenting_agents": []},
            "provenance_chain": [
                ProvenanceRecord(
                    timestamp="2026-01-01T00:00:00Z",
                    event_type="verdict",
                    description="Test",
                ),
            ],
        }
        # Should handle ProvenanceRecord objects in HIPAA audit trail
        artifact = generator.generate_hipaa(receipt)
        assert artifact.audit_trail["provenance_chain_length"] == 1
        assert artifact.audit_trail["events_logged"][0]["event_type"] == "verdict"

    def test_supported_frameworks_constant(self):
        assert "eu_ai_act" in ReceiptComplianceGenerator.SUPPORTED_FRAMEWORKS
        assert "soc2" in ReceiptComplianceGenerator.SUPPORTED_FRAMEWORKS
        assert "hipaa" in ReceiptComplianceGenerator.SUPPORTED_FRAMEWORKS

    def test_human_oversight_from_provenance_event(self, generator):
        receipt = {
            "receipt_id": "test-human-prov",
            "input_summary": "",
            "input_hash": "",
            "verdict": "PASS",
            "confidence": 0.8,
            "robustness_score": 0.8,
            "risk_summary": {},
            "artifact_hash": "",
            "config_used": {},
            "verdict_reasoning": "Good",
            "dissenting_views": [],
            "consensus_proof": {"supporting_agents": ["A"], "dissenting_agents": []},
            "provenance_chain": [
                {"event_type": "human_approval", "timestamp": "2026-01-01T00:00:00Z"},
                {"event_type": "verdict", "timestamp": "2026-01-01T00:00:01Z"},
            ],
        }
        artifact = generator.generate_eu_ai_act(receipt)
        assert artifact.human_oversight["human_in_the_loop"] is True

    def test_all_artifact_integrity_hashes_are_unique(self, generator, sample_receipt_dict):
        result = generator.generate(sample_receipt_dict)
        hashes = {
            result.eu_ai_act.integrity_hash,
            result.soc2.integrity_hash,
            result.hipaa.integrity_hash,
        }
        assert len(hashes) == 3  # All different
