"""
Healthcare Vertical Playbook Template.

HIPAA-aware clinical decision template with:
- Compliance checks (HIPAA, HITECH, FDA 21 CFR Part 11 where applicable)
- Evidence-grading stages (systematic review style, GRADE framework)
- Patient safety guardrails (Do-No-Harm gate, adverse event screening)
- Clinical outcome tracking with decision receipt generation

This is a *playbook* template -- a complete decision workflow that wires
together debate, compliance, evidence review, and receipt generation for
clinical or healthcare-administrative decisions.
"""

from __future__ import annotations

from typing import Any

HEALTHCARE_CLINICAL_DECISION_TEMPLATE: dict[str, Any] = {
    "name": "Healthcare Clinical Decision Playbook",
    "description": (
        "End-to-end clinical decision workflow with HIPAA/HITECH compliance, "
        "evidence grading (GRADE framework), patient safety guardrails, "
        "and audit-ready decision receipt generation."
    ),
    "category": "healthcare",
    "version": "1.0",
    "tags": [
        "healthcare",
        "clinical",
        "hipaa",
        "hitech",
        "evidence-grading",
        "patient-safety",
        "decision-receipt",
        "playbook",
    ],
    "compliance_frameworks": ["HIPAA", "HITECH", "FDA_21_CFR_11"],
    "required_agent_types": [
        "clinical_reviewer",
        "compliance_officer",
        "hipaa_auditor",
        "research_analyst_clinical",
        "medical_coder",
    ],
    "output_format": {
        "decision_receipt": {
            "type": "gauntlet_receipt",
            "includes": [
                "decision_summary",
                "evidence_grade",
                "compliance_attestations",
                "patient_safety_assessment",
                "dissenting_opinions",
                "audit_trail",
            ],
        },
        "clinical_summary": {
            "type": "structured_report",
            "sections": [
                "clinical_question",
                "evidence_summary",
                "risk_benefit_analysis",
                "recommendation",
                "follow_up_plan",
            ],
        },
    },
    "steps": [
        # --- Phase 1: Intake and Compliance Gate ---
        {
            "id": "clinical_intake",
            "type": "task",
            "name": "Clinical Decision Intake",
            "description": (
                "Capture the clinical question, patient context (de-identified), "
                "and decision scope. Classify urgency level."
            ),
            "config": {
                "task_type": "validate",
                "validation_rules": [
                    "clinical_question_defined",
                    "patient_context_deidentified",
                    "urgency_level_set",
                    "decision_scope_bounded",
                ],
            },
        },
        {
            "id": "phi_screening",
            "type": "task",
            "name": "PHI Screening and Redaction",
            "description": (
                "Screen all inputs for Protected Health Information. "
                "Apply Safe Harbor de-identification before any agent processing."
            ),
            "config": {
                "task_type": "function",
                "function_name": "redact_phi_safe_harbor",
                "fail_on_phi_detected": True,
            },
        },
        {
            "id": "compliance_pre_check",
            "type": "debate",
            "name": "Compliance Pre-Check",
            "description": (
                "Verify HIPAA/HITECH compliance posture before proceeding. "
                "Check authorization, minimum necessary standard, and consent."
            ),
            "config": {
                "agents": ["hipaa_auditor", "compliance_officer"],
                "rounds": 1,
                "topic_template": (
                    "Verify compliance prerequisites for clinical decision: "
                    "{clinical_question}. Check HIPAA authorization, minimum "
                    "necessary, and patient consent status."
                ),
                "compliance_checks": [
                    "hipaa_authorization_valid",
                    "minimum_necessary_standard",
                    "patient_consent_documented",
                    "hitech_breach_notification_ready",
                ],
            },
        },
        {
            "id": "compliance_gate",
            "type": "decision",
            "name": "Compliance Gate",
            "description": "Block workflow if compliance prerequisites are not met.",
            "config": {
                "condition": "compliance_pre_check_passed",
                "true_target": "evidence_retrieval",
                "false_target": "compliance_remediation",
            },
        },
        {
            "id": "compliance_remediation",
            "type": "human_checkpoint",
            "name": "Compliance Remediation",
            "description": (
                "Route to compliance officer for manual remediation when automated checks fail."
            ),
            "config": {
                "approval_type": "remediation",
                "required_role": "compliance_officer",
                "checklist": [
                    "Resolve HIPAA authorization gaps",
                    "Obtain missing patient consent",
                    "Document minimum necessary justification",
                ],
            },
        },
        # --- Phase 2: Evidence Gathering and Grading ---
        {
            "id": "evidence_retrieval",
            "type": "memory_read",
            "name": "Evidence Retrieval",
            "description": (
                "Retrieve relevant clinical evidence from knowledge base: "
                "prior decisions, clinical guidelines, research findings."
            ),
            "config": {
                "query_template": (
                    "Clinical evidence for: {clinical_question}. "
                    "Include guidelines, systematic reviews, and prior decisions."
                ),
                "domains": [
                    "healthcare/clinical",
                    "healthcare/guidelines",
                    "healthcare/research",
                ],
            },
        },
        {
            "id": "evidence_grading",
            "type": "debate",
            "name": "Evidence Grading (GRADE Framework)",
            "description": (
                "Multi-agent evidence grading using the GRADE framework. "
                "Classify evidence quality: High, Moderate, Low, Very Low."
            ),
            "config": {
                "agents": [
                    "research_analyst_clinical",
                    "clinical_reviewer",
                    "medical_coder",
                ],
                "rounds": 2,
                "topic_template": (
                    "Grade the following evidence using the GRADE framework "
                    "for clinical question: {clinical_question}. "
                    "Evidence: {retrieved_evidence}. "
                    "Assess: study design, risk of bias, inconsistency, "
                    "indirectness, imprecision, publication bias."
                ),
                "consensus_threshold": 0.7,
                "evidence_levels": ["high", "moderate", "low", "very_low"],
            },
        },
        # --- Phase 3: Clinical Analysis and Safety ---
        {
            "id": "risk_benefit_analysis",
            "type": "debate",
            "name": "Risk-Benefit Analysis",
            "description": (
                "Multi-agent debate on risks vs. benefits of the clinical "
                "decision, incorporating graded evidence."
            ),
            "config": {
                "agents": [
                    "clinical_reviewer",
                    "compliance_officer",
                    "research_analyst_clinical",
                ],
                "rounds": 3,
                "topic_template": (
                    "Analyze risks and benefits for: {clinical_question}. "
                    "Evidence grade: {evidence_grade}. "
                    "Patient context: {deidentified_context}."
                ),
                "consensus_threshold": 0.75,
            },
        },
        {
            "id": "patient_safety_screening",
            "type": "task",
            "name": "Patient Safety Screening",
            "description": (
                "Automated screening for patient safety concerns: "
                "contraindications, adverse event signals, drug interactions."
            ),
            "config": {
                "task_type": "function",
                "function_name": "screen_patient_safety",
                "checks": [
                    "contraindication_screening",
                    "adverse_event_signal_detection",
                    "drug_interaction_check",
                    "dosage_range_validation",
                ],
            },
        },
        {
            "id": "do_no_harm_gate",
            "type": "decision",
            "name": "Do-No-Harm Safety Gate",
            "description": (
                "Mandatory safety gate. If patient safety concerns are "
                "flagged, escalate to attending physician before proceeding."
            ),
            "config": {
                "condition": "patient_safety_clear",
                "true_target": "clinical_recommendation",
                "false_target": "safety_escalation",
            },
        },
        {
            "id": "safety_escalation",
            "type": "human_checkpoint",
            "name": "Safety Escalation to Attending",
            "description": (
                "Escalate patient safety concerns to attending physician "
                "for manual review and override decision."
            ),
            "config": {
                "approval_type": "safety_review",
                "required_role": "attending_physician",
                "urgent": True,
                "checklist": [
                    "Review flagged safety concerns",
                    "Assess contraindication severity",
                    "Determine if decision should proceed",
                    "Document override rationale if applicable",
                ],
            },
        },
        # --- Phase 4: Recommendation and Approval ---
        {
            "id": "clinical_recommendation",
            "type": "debate",
            "name": "Clinical Recommendation Synthesis",
            "description": (
                "Synthesize evidence, risk-benefit analysis, and safety "
                "screening into a clinical recommendation."
            ),
            "config": {
                "agents": [
                    "clinical_reviewer",
                    "research_analyst_clinical",
                    "compliance_officer",
                ],
                "rounds": 2,
                "topic_template": (
                    "Synthesize recommendation for: {clinical_question}. "
                    "Evidence grade: {evidence_grade}. "
                    "Risk-benefit: {risk_benefit_summary}. "
                    "Safety status: {safety_status}."
                ),
                "consensus_threshold": 0.8,
            },
        },
        {
            "id": "clinical_outcome_tracking",
            "type": "task",
            "name": "Clinical Outcome Tracking Setup",
            "description": (
                "Configure outcome tracking metrics and follow-up schedule "
                "for the clinical decision."
            ),
            "config": {
                "task_type": "function",
                "function_name": "setup_outcome_tracking",
                "tracking_config": {
                    "outcome_metrics": [
                        "primary_clinical_endpoint",
                        "adverse_event_rate",
                        "patient_satisfaction",
                        "readmission_rate",
                    ],
                    "follow_up_intervals_days": [7, 30, 90],
                },
            },
        },
        {
            "id": "attending_review",
            "type": "human_checkpoint",
            "name": "Attending Physician Review",
            "description": ("Final clinical review and sign-off by attending physician."),
            "config": {
                "approval_type": "sign_off",
                "required_role": "attending_physician",
                "checklist": [
                    "Review evidence grade and sources",
                    "Confirm risk-benefit assessment",
                    "Verify patient safety clearance",
                    "Approve clinical recommendation",
                    "Confirm outcome tracking plan",
                ],
            },
        },
        # --- Phase 5: Receipt and Archive ---
        {
            "id": "generate_decision_receipt",
            "type": "task",
            "name": "Generate Decision Receipt",
            "description": (
                "Generate a cryptographic decision receipt (Gauntlet receipt) "
                "capturing the full audit trail, evidence, and compliance attestations."
            ),
            "config": {
                "task_type": "function",
                "function_name": "generate_gauntlet_receipt",
                "receipt_config": {
                    "include_evidence_grade": True,
                    "include_compliance_attestations": True,
                    "include_safety_assessment": True,
                    "include_dissenting_opinions": True,
                    "hash_algorithm": "sha256",
                },
            },
        },
        {
            "id": "generate_clinical_summary",
            "type": "task",
            "name": "Generate Clinical Summary",
            "description": "Generate structured clinical summary report.",
            "config": {
                "task_type": "transform",
                "template": "clinical_decision_summary",
            },
        },
        {
            "id": "archive",
            "type": "memory_write",
            "name": "Archive Clinical Decision",
            "description": (
                "Archive decision with all supporting evidence for HIPAA-compliant retention."
            ),
            "config": {
                "domain": "healthcare/clinical/decisions",
                "retention_years": 6,
                "compliance_tags": ["hipaa", "hitech"],
            },
        },
    ],
    "transitions": [
        {"from": "clinical_intake", "to": "phi_screening"},
        {"from": "phi_screening", "to": "compliance_pre_check"},
        {"from": "compliance_pre_check", "to": "compliance_gate"},
        {
            "from": "compliance_gate",
            "to": "evidence_retrieval",
            "condition": "compliant",
        },
        {
            "from": "compliance_gate",
            "to": "compliance_remediation",
            "condition": "non_compliant",
        },
        {"from": "compliance_remediation", "to": "compliance_pre_check"},
        {"from": "evidence_retrieval", "to": "evidence_grading"},
        {"from": "evidence_grading", "to": "risk_benefit_analysis"},
        {"from": "risk_benefit_analysis", "to": "patient_safety_screening"},
        {"from": "patient_safety_screening", "to": "do_no_harm_gate"},
        {
            "from": "do_no_harm_gate",
            "to": "clinical_recommendation",
            "condition": "safe",
        },
        {
            "from": "do_no_harm_gate",
            "to": "safety_escalation",
            "condition": "safety_concern",
        },
        {"from": "safety_escalation", "to": "clinical_recommendation"},
        {"from": "clinical_recommendation", "to": "clinical_outcome_tracking"},
        {"from": "clinical_outcome_tracking", "to": "attending_review"},
        {
            "from": "attending_review",
            "to": "generate_decision_receipt",
            "condition": "approved",
        },
        {
            "from": "attending_review",
            "to": "risk_benefit_analysis",
            "condition": "rejected",
        },
        {"from": "generate_decision_receipt", "to": "generate_clinical_summary"},
        {"from": "generate_clinical_summary", "to": "archive"},
    ],
    "metadata": {
        "author": "aragora",
        "version": "1.0.0",
        "min_agents": 3,
        "max_agents": 5,
        "estimated_duration_minutes": 15,
        "regulatory_note": (
            "This template enforces HIPAA Privacy and Security Rule "
            "requirements. All PHI is de-identified before agent processing. "
            "Decision receipts provide audit-ready compliance evidence."
        ),
    },
}
