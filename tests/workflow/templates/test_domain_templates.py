"""
Tests for Domain-Specific Workflow Templates.

Tests coverage for:
- DevOps templates (CI/CD, Incident Response, Infrastructure Audit, PagerDuty)
- Accounting templates (Financial Audit, SOX Compliance, Bank Reconciliation)
- Legal templates (Contract Review, Due Diligence, Compliance Audit)
- Healthcare templates (HIPAA Assessment, Clinical Review, PHI Audit)
- Email Management templates (Categorization, Follow-up, Snooze, Triage)
- Package system (create, register, retrieve, list)
- Enum values (TemplateStatus, TemplateCategory)
"""

from __future__ import annotations

import json

import pytest

from aragora.workflow.templates.package import (
    TemplateStatus,
    TemplateCategory,
    TemplateAuthor,
    TemplateDependency,
    TemplateMetadata,
    TemplatePackage,
    create_package,
    register_package,
    get_package,
    list_packages,
    _template_registry,
)


# ============================================================================
# Helper: validate common template structure
# ============================================================================


def _assert_template_structure(template: dict, label: str) -> None:
    """Assert a template dict has all required top-level fields."""
    assert "name" in template, f"{label} missing 'name'"
    assert "description" in template, f"{label} missing 'description'"
    assert "category" in template, f"{label} missing 'category'"
    assert "version" in template, f"{label} missing 'version'"
    assert "tags" in template, f"{label} missing 'tags'"
    assert isinstance(template["tags"], list), f"{label} tags should be a list"
    assert "steps" in template, f"{label} missing 'steps'"
    assert len(template["steps"]) > 0, f"{label} has no steps"
    assert "transitions" in template, f"{label} missing 'transitions'"
    assert len(template["transitions"]) > 0, f"{label} has no transitions"


def _assert_steps_valid(template: dict, label: str) -> None:
    """Assert all steps in a template have required fields."""
    for step in template["steps"]:
        # Parallel steps use "branches" instead of normal step fields
        if step.get("type") == "parallel":
            assert "branches" in step, f"{label} parallel step missing 'branches'"
            continue
        assert "id" in step, f"{label} step missing 'id'"
        assert "type" in step, f"{label} step missing 'type'"
        assert "name" in step, f"{label} step missing 'name'"


def _assert_transitions_reference_valid_steps(template: dict, label: str) -> None:
    """Assert all transition 'from'/'to' reference existing step IDs."""
    step_ids = set()
    for step in template["steps"]:
        if step.get("type") == "parallel":
            step_ids.add(step["id"])
            for branch in step.get("branches", []):
                for sub_step in branch.get("steps", []):
                    step_ids.add(sub_step["id"])
        else:
            step_ids.add(step["id"])

    for transition in template["transitions"]:
        assert transition["from"] in step_ids, (
            f"{label} transition references unknown step '{transition['from']}'"
        )
        assert transition["to"] in step_ids, (
            f"{label} transition references unknown step '{transition['to']}'"
        )


# ============================================================================
# DevOps Template Tests
# ============================================================================


class TestCICDPipelineReviewTemplate:
    """Tests for CI/CD Pipeline Review template."""

    def test_template_structure(self):
        from aragora.workflow.templates.devops import CICD_PIPELINE_REVIEW_TEMPLATE

        _assert_template_structure(CICD_PIPELINE_REVIEW_TEMPLATE, "CICD_PIPELINE_REVIEW")

    def test_template_identity(self):
        from aragora.workflow.templates.devops import CICD_PIPELINE_REVIEW_TEMPLATE

        assert CICD_PIPELINE_REVIEW_TEMPLATE["name"] == "CI/CD Pipeline Review"
        assert CICD_PIPELINE_REVIEW_TEMPLATE["category"] == "devops"

    def test_steps_valid(self):
        from aragora.workflow.templates.devops import CICD_PIPELINE_REVIEW_TEMPLATE

        _assert_steps_valid(CICD_PIPELINE_REVIEW_TEMPLATE, "CICD_PIPELINE_REVIEW")

    def test_transitions_reference_valid_steps(self):
        from aragora.workflow.templates.devops import CICD_PIPELINE_REVIEW_TEMPLATE

        _assert_transitions_reference_valid_steps(
            CICD_PIPELINE_REVIEW_TEMPLATE, "CICD_PIPELINE_REVIEW"
        )

    def test_has_devops_tags(self):
        from aragora.workflow.templates.devops import CICD_PIPELINE_REVIEW_TEMPLATE

        tags = CICD_PIPELINE_REVIEW_TEMPLATE["tags"]
        assert "devops" in tags
        assert "cicd" in tags

    def test_has_security_review_step(self):
        from aragora.workflow.templates.devops import CICD_PIPELINE_REVIEW_TEMPLATE

        step_ids = [s["id"] for s in CICD_PIPELINE_REVIEW_TEMPLATE["steps"]]
        assert "security_review" in step_ids

    def test_has_compliance_gate(self):
        from aragora.workflow.templates.devops import CICD_PIPELINE_REVIEW_TEMPLATE

        step_ids = [s["id"] for s in CICD_PIPELINE_REVIEW_TEMPLATE["steps"]]
        assert "compliance_gate" in step_ids

    def test_agents_configured(self):
        from aragora.workflow.templates.devops import CICD_PIPELINE_REVIEW_TEMPLATE

        debate_steps = [s for s in CICD_PIPELINE_REVIEW_TEMPLATE["steps"] if s["type"] == "debate"]
        for step in debate_steps:
            assert "agents" in step["config"], f"Debate step '{step['id']}' missing agents"
            assert len(step["config"]["agents"]) >= 2


class TestIncidentResponseTemplate:
    """Tests for Incident Response template."""

    def test_template_structure(self):
        from aragora.workflow.templates.devops import INCIDENT_RESPONSE_TEMPLATE

        _assert_template_structure(INCIDENT_RESPONSE_TEMPLATE, "INCIDENT_RESPONSE")

    def test_template_identity(self):
        from aragora.workflow.templates.devops import INCIDENT_RESPONSE_TEMPLATE

        assert INCIDENT_RESPONSE_TEMPLATE["name"] == "Incident Response Workflow"
        assert INCIDENT_RESPONSE_TEMPLATE["category"] == "devops"

    def test_steps_valid(self):
        from aragora.workflow.templates.devops import INCIDENT_RESPONSE_TEMPLATE

        _assert_steps_valid(INCIDENT_RESPONSE_TEMPLATE, "INCIDENT_RESPONSE")

    def test_transitions_reference_valid_steps(self):
        from aragora.workflow.templates.devops import INCIDENT_RESPONSE_TEMPLATE

        _assert_transitions_reference_valid_steps(INCIDENT_RESPONSE_TEMPLATE, "INCIDENT_RESPONSE")

    def test_has_incident_tags(self):
        from aragora.workflow.templates.devops import INCIDENT_RESPONSE_TEMPLATE

        tags = INCIDENT_RESPONSE_TEMPLATE["tags"]
        assert "incident" in tags
        assert "sre" in tags

    def test_has_severity_classification(self):
        from aragora.workflow.templates.devops import INCIDENT_RESPONSE_TEMPLATE

        step_ids = [s["id"] for s in INCIDENT_RESPONSE_TEMPLATE["steps"]]
        assert "severity_classification" in step_ids
        assert "severity_gate" in step_ids

    def test_has_postmortem_step(self):
        from aragora.workflow.templates.devops import INCIDENT_RESPONSE_TEMPLATE

        step_ids = [s["id"] for s in INCIDENT_RESPONSE_TEMPLATE["steps"]]
        assert "postmortem_prep" in step_ids

    def test_has_war_room(self):
        from aragora.workflow.templates.devops import INCIDENT_RESPONSE_TEMPLATE

        step_ids = [s["id"] for s in INCIDENT_RESPONSE_TEMPLATE["steps"]]
        assert "war_room" in step_ids


class TestInfrastructureAuditTemplate:
    """Tests for Infrastructure Audit template."""

    def test_template_structure(self):
        from aragora.workflow.templates.devops import INFRASTRUCTURE_AUDIT_TEMPLATE

        _assert_template_structure(INFRASTRUCTURE_AUDIT_TEMPLATE, "INFRASTRUCTURE_AUDIT")

    def test_template_identity(self):
        from aragora.workflow.templates.devops import INFRASTRUCTURE_AUDIT_TEMPLATE

        assert INFRASTRUCTURE_AUDIT_TEMPLATE["name"] == "Infrastructure Security Audit"
        assert INFRASTRUCTURE_AUDIT_TEMPLATE["category"] == "devops"

    def test_steps_valid(self):
        from aragora.workflow.templates.devops import INFRASTRUCTURE_AUDIT_TEMPLATE

        _assert_steps_valid(INFRASTRUCTURE_AUDIT_TEMPLATE, "INFRASTRUCTURE_AUDIT")

    def test_transitions_reference_valid_steps(self):
        from aragora.workflow.templates.devops import INFRASTRUCTURE_AUDIT_TEMPLATE

        _assert_transitions_reference_valid_steps(
            INFRASTRUCTURE_AUDIT_TEMPLATE, "INFRASTRUCTURE_AUDIT"
        )

    def test_has_infrastructure_tags(self):
        from aragora.workflow.templates.devops import INFRASTRUCTURE_AUDIT_TEMPLATE

        tags = INFRASTRUCTURE_AUDIT_TEMPLATE["tags"]
        assert "infrastructure" in tags
        assert "security" in tags
        assert "cloud" in tags

    def test_has_ciso_review(self):
        from aragora.workflow.templates.devops import INFRASTRUCTURE_AUDIT_TEMPLATE

        step_ids = [s["id"] for s in INFRASTRUCTURE_AUDIT_TEMPLATE["steps"]]
        assert "ciso_review" in step_ids


class TestPagerDutyIncidentTemplate:
    """Tests for PagerDuty Incident Management template."""

    def test_template_structure(self):
        from aragora.workflow.templates.devops import PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE

        _assert_template_structure(PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE, "PAGERDUTY_INCIDENT")

    def test_template_identity(self):
        from aragora.workflow.templates.devops import PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE

        assert PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE["name"] == "PagerDuty Incident Management"
        assert PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE["category"] == "devops"

    def test_steps_valid(self):
        from aragora.workflow.templates.devops import PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE

        _assert_steps_valid(PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE, "PAGERDUTY_INCIDENT")

    def test_transitions_reference_valid_steps(self):
        from aragora.workflow.templates.devops import PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE

        _assert_transitions_reference_valid_steps(
            PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE, "PAGERDUTY_INCIDENT"
        )

    def test_has_pagerduty_connector(self):
        from aragora.workflow.templates.devops import PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE

        assert "pagerduty" in PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE.get("connectors", [])

    def test_has_pagerduty_tags(self):
        from aragora.workflow.templates.devops import PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE

        tags = PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE["tags"]
        assert "pagerduty" in tags
        assert "on-call" in tags

    def test_has_urgency_routing(self):
        from aragora.workflow.templates.devops import PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE

        step_ids = [s["id"] for s in PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE["steps"]]
        assert "urgency_decision" in step_ids
        assert "create_high_urgency_incident" in step_ids
        assert "create_low_urgency_incident" in step_ids

    def test_has_postmortem_generation(self):
        from aragora.workflow.templates.devops import PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE

        step_ids = [s["id"] for s in PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE["steps"]]
        assert "postmortem_generation" in step_ids


# ============================================================================
# Accounting Template Tests
# ============================================================================


class TestFinancialAuditTemplate:
    """Tests for Financial Audit template."""

    def test_template_structure(self):
        from aragora.workflow.templates.accounting import FINANCIAL_AUDIT_TEMPLATE

        _assert_template_structure(FINANCIAL_AUDIT_TEMPLATE, "FINANCIAL_AUDIT")

    def test_template_identity(self):
        from aragora.workflow.templates.accounting import FINANCIAL_AUDIT_TEMPLATE

        assert FINANCIAL_AUDIT_TEMPLATE["name"] == "Financial Statement Audit"
        assert FINANCIAL_AUDIT_TEMPLATE["category"] == "accounting"

    def test_steps_valid(self):
        from aragora.workflow.templates.accounting import FINANCIAL_AUDIT_TEMPLATE

        _assert_steps_valid(FINANCIAL_AUDIT_TEMPLATE, "FINANCIAL_AUDIT")

    def test_transitions_reference_valid_steps(self):
        from aragora.workflow.templates.accounting import FINANCIAL_AUDIT_TEMPLATE

        _assert_transitions_reference_valid_steps(FINANCIAL_AUDIT_TEMPLATE, "FINANCIAL_AUDIT")

    def test_has_accounting_tags(self):
        from aragora.workflow.templates.accounting import FINANCIAL_AUDIT_TEMPLATE

        tags = FINANCIAL_AUDIT_TEMPLATE["tags"]
        assert "accounting" in tags
        assert "audit" in tags
        assert "gaap" in tags

    def test_has_parallel_substantive_testing(self):
        from aragora.workflow.templates.accounting import FINANCIAL_AUDIT_TEMPLATE

        parallel_steps = [
            s for s in FINANCIAL_AUDIT_TEMPLATE["steps"] if s.get("type") == "parallel"
        ]
        assert len(parallel_steps) == 1
        assert len(parallel_steps[0]["branches"]) == 4

    def test_has_partner_review(self):
        from aragora.workflow.templates.accounting import FINANCIAL_AUDIT_TEMPLATE

        step_ids = [s["id"] for s in FINANCIAL_AUDIT_TEMPLATE["steps"]]
        assert "partner_review" in step_ids

    def test_has_going_concern(self):
        from aragora.workflow.templates.accounting import FINANCIAL_AUDIT_TEMPLATE

        step_ids = [s["id"] for s in FINANCIAL_AUDIT_TEMPLATE["steps"]]
        assert "going_concern" in step_ids

    def test_archive_has_retention_years(self):
        from aragora.workflow.templates.accounting import FINANCIAL_AUDIT_TEMPLATE

        archive_step = next(s for s in FINANCIAL_AUDIT_TEMPLATE["steps"] if s["id"] == "archive")
        assert archive_step["config"]["retention_years"] == 7


class TestSOXComplianceTemplate:
    """Tests for SOX Compliance template."""

    def test_template_structure(self):
        from aragora.workflow.templates.accounting import SOX_COMPLIANCE_TEMPLATE

        _assert_template_structure(SOX_COMPLIANCE_TEMPLATE, "SOX_COMPLIANCE")

    def test_template_identity(self):
        from aragora.workflow.templates.accounting import SOX_COMPLIANCE_TEMPLATE

        assert SOX_COMPLIANCE_TEMPLATE["name"] == "SOX Compliance Assessment"
        assert SOX_COMPLIANCE_TEMPLATE["category"] == "accounting"

    def test_steps_valid(self):
        from aragora.workflow.templates.accounting import SOX_COMPLIANCE_TEMPLATE

        _assert_steps_valid(SOX_COMPLIANCE_TEMPLATE, "SOX_COMPLIANCE")

    def test_transitions_reference_valid_steps(self):
        from aragora.workflow.templates.accounting import SOX_COMPLIANCE_TEMPLATE

        _assert_transitions_reference_valid_steps(SOX_COMPLIANCE_TEMPLATE, "SOX_COMPLIANCE")

    def test_has_sox_tags(self):
        from aragora.workflow.templates.accounting import SOX_COMPLIANCE_TEMPLATE

        tags = SOX_COMPLIANCE_TEMPLATE["tags"]
        assert "sox" in tags
        assert "compliance" in tags

    def test_has_material_weakness_check(self):
        from aragora.workflow.templates.accounting import SOX_COMPLIANCE_TEMPLATE

        step_ids = [s["id"] for s in SOX_COMPLIANCE_TEMPLATE["steps"]]
        assert "material_weakness_check" in step_ids

    def test_has_itgc_testing(self):
        from aragora.workflow.templates.accounting import SOX_COMPLIANCE_TEMPLATE

        step_ids = [s["id"] for s in SOX_COMPLIANCE_TEMPLATE["steps"]]
        assert "itgc_testing" in step_ids


class TestBankReconciliationTemplate:
    """Tests for Bank Reconciliation template."""

    def test_template_structure(self):
        from aragora.workflow.templates.accounting import BANK_RECONCILIATION_TEMPLATE

        _assert_template_structure(BANK_RECONCILIATION_TEMPLATE, "BANK_RECONCILIATION")

    def test_template_identity(self):
        from aragora.workflow.templates.accounting import BANK_RECONCILIATION_TEMPLATE

        assert BANK_RECONCILIATION_TEMPLATE["name"] == "Bank Reconciliation"
        assert BANK_RECONCILIATION_TEMPLATE["category"] == "accounting"

    def test_steps_valid(self):
        from aragora.workflow.templates.accounting import BANK_RECONCILIATION_TEMPLATE

        _assert_steps_valid(BANK_RECONCILIATION_TEMPLATE, "BANK_RECONCILIATION")

    def test_transitions_reference_valid_steps(self):
        from aragora.workflow.templates.accounting import BANK_RECONCILIATION_TEMPLATE

        _assert_transitions_reference_valid_steps(
            BANK_RECONCILIATION_TEMPLATE, "BANK_RECONCILIATION"
        )

    def test_has_connectors(self):
        from aragora.workflow.templates.accounting import BANK_RECONCILIATION_TEMPLATE

        connectors = BANK_RECONCILIATION_TEMPLATE.get("connectors", [])
        assert "plaid" in connectors
        assert "qbo" in connectors

    def test_has_banking_tags(self):
        from aragora.workflow.templates.accounting import BANK_RECONCILIATION_TEMPLATE

        tags = BANK_RECONCILIATION_TEMPLATE["tags"]
        assert "reconciliation" in tags
        assert "plaid" in tags
        assert "banking" in tags

    def test_has_fraud_investigation(self):
        from aragora.workflow.templates.accounting import BANK_RECONCILIATION_TEMPLATE

        step_ids = [s["id"] for s in BANK_RECONCILIATION_TEMPLATE["steps"]]
        assert "fraud_investigation" in step_ids

    def test_has_auto_match(self):
        from aragora.workflow.templates.accounting import BANK_RECONCILIATION_TEMPLATE

        step_ids = [s["id"] for s in BANK_RECONCILIATION_TEMPLATE["steps"]]
        assert "auto_match" in step_ids


# ============================================================================
# Legal Template Tests
# ============================================================================


class TestContractReviewTemplate:
    """Tests for Contract Review template."""

    def test_template_structure(self):
        from aragora.workflow.templates.legal import CONTRACT_REVIEW_TEMPLATE

        _assert_template_structure(CONTRACT_REVIEW_TEMPLATE, "CONTRACT_REVIEW")

    def test_template_identity(self):
        from aragora.workflow.templates.legal import CONTRACT_REVIEW_TEMPLATE

        assert CONTRACT_REVIEW_TEMPLATE["name"] == "Contract Review"
        assert CONTRACT_REVIEW_TEMPLATE["category"] == "legal"

    def test_steps_valid(self):
        from aragora.workflow.templates.legal import CONTRACT_REVIEW_TEMPLATE

        _assert_steps_valid(CONTRACT_REVIEW_TEMPLATE, "CONTRACT_REVIEW")

    def test_transitions_reference_valid_steps(self):
        from aragora.workflow.templates.legal import CONTRACT_REVIEW_TEMPLATE

        _assert_transitions_reference_valid_steps(CONTRACT_REVIEW_TEMPLATE, "CONTRACT_REVIEW")

    def test_has_legal_tags(self):
        from aragora.workflow.templates.legal import CONTRACT_REVIEW_TEMPLATE

        tags = CONTRACT_REVIEW_TEMPLATE["tags"]
        assert "legal" in tags
        assert "contracts" in tags
        assert "risk-assessment" in tags

    def test_has_risk_debate(self):
        from aragora.workflow.templates.legal import CONTRACT_REVIEW_TEMPLATE

        step_ids = [s["id"] for s in CONTRACT_REVIEW_TEMPLATE["steps"]]
        assert "risk_debate" in step_ids

    def test_risk_debate_has_consensus_threshold(self):
        from aragora.workflow.templates.legal import CONTRACT_REVIEW_TEMPLATE

        risk_step = next(s for s in CONTRACT_REVIEW_TEMPLATE["steps"] if s["id"] == "risk_debate")
        assert risk_step["config"]["consensus_threshold"] == 0.7

    def test_has_human_review_step(self):
        from aragora.workflow.templates.legal import CONTRACT_REVIEW_TEMPLATE

        step_ids = [s["id"] for s in CONTRACT_REVIEW_TEMPLATE["steps"]]
        assert "human_review" in step_ids


class TestDueDiligenceTemplate:
    """Tests for Due Diligence template."""

    def test_template_structure(self):
        from aragora.workflow.templates.legal import DUE_DILIGENCE_TEMPLATE

        _assert_template_structure(DUE_DILIGENCE_TEMPLATE, "DUE_DILIGENCE")

    def test_template_identity(self):
        from aragora.workflow.templates.legal import DUE_DILIGENCE_TEMPLATE

        assert DUE_DILIGENCE_TEMPLATE["name"] == "Due Diligence Review"
        assert DUE_DILIGENCE_TEMPLATE["category"] == "legal"

    def test_steps_valid(self):
        from aragora.workflow.templates.legal import DUE_DILIGENCE_TEMPLATE

        _assert_steps_valid(DUE_DILIGENCE_TEMPLATE, "DUE_DILIGENCE")

    def test_transitions_reference_valid_steps(self):
        from aragora.workflow.templates.legal import DUE_DILIGENCE_TEMPLATE

        _assert_transitions_reference_valid_steps(DUE_DILIGENCE_TEMPLATE, "DUE_DILIGENCE")

    def test_has_ma_tags(self):
        from aragora.workflow.templates.legal import DUE_DILIGENCE_TEMPLATE

        tags = DUE_DILIGENCE_TEMPLATE["tags"]
        assert "m&a" in tags
        assert "due-diligence" in tags

    def test_has_parallel_review_branches(self):
        from aragora.workflow.templates.legal import DUE_DILIGENCE_TEMPLATE

        parallel_steps = [s for s in DUE_DILIGENCE_TEMPLATE["steps"] if s.get("type") == "parallel"]
        assert len(parallel_steps) == 1
        branches = parallel_steps[0]["branches"]
        assert len(branches) == 3
        branch_ids = [b["id"] for b in branches]
        assert "corporate_review" in branch_ids
        assert "contract_review" in branch_ids
        assert "ip_review" in branch_ids

    def test_has_partner_review(self):
        from aragora.workflow.templates.legal import DUE_DILIGENCE_TEMPLATE

        step_ids = [s["id"] for s in DUE_DILIGENCE_TEMPLATE["steps"]]
        assert "partner_review" in step_ids


class TestComplianceAuditTemplate:
    """Tests for Compliance Audit template."""

    def test_template_structure(self):
        from aragora.workflow.templates.legal import COMPLIANCE_AUDIT_TEMPLATE

        _assert_template_structure(COMPLIANCE_AUDIT_TEMPLATE, "COMPLIANCE_AUDIT")

    def test_template_identity(self):
        from aragora.workflow.templates.legal import COMPLIANCE_AUDIT_TEMPLATE

        assert COMPLIANCE_AUDIT_TEMPLATE["name"] == "Compliance Audit"
        assert COMPLIANCE_AUDIT_TEMPLATE["category"] == "legal"

    def test_steps_valid(self):
        from aragora.workflow.templates.legal import COMPLIANCE_AUDIT_TEMPLATE

        _assert_steps_valid(COMPLIANCE_AUDIT_TEMPLATE, "COMPLIANCE_AUDIT")

    def test_transitions_reference_valid_steps(self):
        from aragora.workflow.templates.legal import COMPLIANCE_AUDIT_TEMPLATE

        _assert_transitions_reference_valid_steps(COMPLIANCE_AUDIT_TEMPLATE, "COMPLIANCE_AUDIT")

    def test_has_compliance_tags(self):
        from aragora.workflow.templates.legal import COMPLIANCE_AUDIT_TEMPLATE

        tags = COMPLIANCE_AUDIT_TEMPLATE["tags"]
        assert "compliance" in tags
        assert "regulatory" in tags

    def test_has_gap_analysis_decision(self):
        from aragora.workflow.templates.legal import COMPLIANCE_AUDIT_TEMPLATE

        step_ids = [s["id"] for s in COMPLIANCE_AUDIT_TEMPLATE["steps"]]
        assert "gap_analysis" in step_ids

    def test_archive_retention(self):
        from aragora.workflow.templates.legal import COMPLIANCE_AUDIT_TEMPLATE

        archive_step = next(s for s in COMPLIANCE_AUDIT_TEMPLATE["steps"] if s["id"] == "archive")
        assert archive_step["config"]["retention_years"] == 7


# ============================================================================
# Healthcare Template Tests
# ============================================================================


class TestHIPAAAssessmentTemplate:
    """Tests for HIPAA Assessment template."""

    def test_template_structure(self):
        from aragora.workflow.templates.healthcare import HIPAA_ASSESSMENT_TEMPLATE

        _assert_template_structure(HIPAA_ASSESSMENT_TEMPLATE, "HIPAA_ASSESSMENT")

    def test_template_identity(self):
        from aragora.workflow.templates.healthcare import HIPAA_ASSESSMENT_TEMPLATE

        assert HIPAA_ASSESSMENT_TEMPLATE["name"] == "HIPAA Risk Assessment"
        assert HIPAA_ASSESSMENT_TEMPLATE["category"] == "healthcare"

    def test_steps_valid(self):
        from aragora.workflow.templates.healthcare import HIPAA_ASSESSMENT_TEMPLATE

        _assert_steps_valid(HIPAA_ASSESSMENT_TEMPLATE, "HIPAA_ASSESSMENT")

    def test_transitions_reference_valid_steps(self):
        from aragora.workflow.templates.healthcare import HIPAA_ASSESSMENT_TEMPLATE

        _assert_transitions_reference_valid_steps(HIPAA_ASSESSMENT_TEMPLATE, "HIPAA_ASSESSMENT")

    def test_has_hipaa_tags(self):
        from aragora.workflow.templates.healthcare import HIPAA_ASSESSMENT_TEMPLATE

        tags = HIPAA_ASSESSMENT_TEMPLATE["tags"]
        assert "hipaa" in tags
        assert "healthcare" in tags
        assert "compliance" in tags

    def test_has_privacy_and_security_assessment(self):
        from aragora.workflow.templates.healthcare import HIPAA_ASSESSMENT_TEMPLATE

        step_ids = [s["id"] for s in HIPAA_ASSESSMENT_TEMPLATE["steps"]]
        assert "privacy_assessment" in step_ids
        assert "security_assessment" in step_ids

    def test_has_breach_readiness(self):
        from aragora.workflow.templates.healthcare import HIPAA_ASSESSMENT_TEMPLATE

        step_ids = [s["id"] for s in HIPAA_ASSESSMENT_TEMPLATE["steps"]]
        assert "breach_readiness" in step_ids

    def test_archive_retention_six_years(self):
        from aragora.workflow.templates.healthcare import HIPAA_ASSESSMENT_TEMPLATE

        archive_step = next(s for s in HIPAA_ASSESSMENT_TEMPLATE["steps"] if s["id"] == "archive")
        assert archive_step["config"]["retention_years"] == 6


class TestClinicalReviewTemplate:
    """Tests for Clinical Review template."""

    def test_template_structure(self):
        from aragora.workflow.templates.healthcare import CLINICAL_REVIEW_TEMPLATE

        _assert_template_structure(CLINICAL_REVIEW_TEMPLATE, "CLINICAL_REVIEW")

    def test_template_identity(self):
        from aragora.workflow.templates.healthcare import CLINICAL_REVIEW_TEMPLATE

        assert CLINICAL_REVIEW_TEMPLATE["name"] == "Clinical Document Review"
        assert CLINICAL_REVIEW_TEMPLATE["category"] == "healthcare"

    def test_steps_valid(self):
        from aragora.workflow.templates.healthcare import CLINICAL_REVIEW_TEMPLATE

        _assert_steps_valid(CLINICAL_REVIEW_TEMPLATE, "CLINICAL_REVIEW")

    def test_transitions_reference_valid_steps(self):
        from aragora.workflow.templates.healthcare import CLINICAL_REVIEW_TEMPLATE

        _assert_transitions_reference_valid_steps(CLINICAL_REVIEW_TEMPLATE, "CLINICAL_REVIEW")

    def test_has_clinical_tags(self):
        from aragora.workflow.templates.healthcare import CLINICAL_REVIEW_TEMPLATE

        tags = CLINICAL_REVIEW_TEMPLATE["tags"]
        assert "clinical" in tags
        assert "documentation" in tags

    def test_has_phi_redaction(self):
        from aragora.workflow.templates.healthcare import CLINICAL_REVIEW_TEMPLATE

        step_ids = [s["id"] for s in CLINICAL_REVIEW_TEMPLATE["steps"]]
        assert "phi_redaction" in step_ids

    def test_has_quality_gate(self):
        from aragora.workflow.templates.healthcare import CLINICAL_REVIEW_TEMPLATE

        step_ids = [s["id"] for s in CLINICAL_REVIEW_TEMPLATE["steps"]]
        assert "quality_assessment" in step_ids


class TestPHIAuditTemplate:
    """Tests for PHI Audit template."""

    def test_template_structure(self):
        from aragora.workflow.templates.healthcare import PHI_AUDIT_TEMPLATE

        _assert_template_structure(PHI_AUDIT_TEMPLATE, "PHI_AUDIT")

    def test_template_identity(self):
        from aragora.workflow.templates.healthcare import PHI_AUDIT_TEMPLATE

        assert PHI_AUDIT_TEMPLATE["name"] == "PHI Access Audit"
        assert PHI_AUDIT_TEMPLATE["category"] == "healthcare"

    def test_steps_valid(self):
        from aragora.workflow.templates.healthcare import PHI_AUDIT_TEMPLATE

        _assert_steps_valid(PHI_AUDIT_TEMPLATE, "PHI_AUDIT")

    def test_transitions_reference_valid_steps(self):
        from aragora.workflow.templates.healthcare import PHI_AUDIT_TEMPLATE

        _assert_transitions_reference_valid_steps(PHI_AUDIT_TEMPLATE, "PHI_AUDIT")

    def test_has_phi_tags(self):
        from aragora.workflow.templates.healthcare import PHI_AUDIT_TEMPLATE

        tags = PHI_AUDIT_TEMPLATE["tags"]
        assert "phi" in tags
        assert "hipaa" in tags
        assert "audit" in tags

    def test_has_anomaly_detection(self):
        from aragora.workflow.templates.healthcare import PHI_AUDIT_TEMPLATE

        step_ids = [s["id"] for s in PHI_AUDIT_TEMPLATE["steps"]]
        assert "anomaly_detection" in step_ids

    def test_has_violation_assessment_decision(self):
        from aragora.workflow.templates.healthcare import PHI_AUDIT_TEMPLATE

        step_ids = [s["id"] for s in PHI_AUDIT_TEMPLATE["steps"]]
        assert "violation_assessment" in step_ids

    def test_archive_retention_six_years(self):
        from aragora.workflow.templates.healthcare import PHI_AUDIT_TEMPLATE

        archive_step = next(s for s in PHI_AUDIT_TEMPLATE["steps"] if s["id"] == "archive")
        assert archive_step["config"]["retention_years"] == 6


# ============================================================================
# Email Management Template Tests
# ============================================================================


class TestEmailCategorizationTemplate:
    """Tests for Email Categorization template."""

    def test_template_structure(self):
        from aragora.workflow.templates.email_management import EMAIL_CATEGORIZATION_TEMPLATE

        _assert_template_structure(EMAIL_CATEGORIZATION_TEMPLATE, "EMAIL_CATEGORIZATION")

    def test_template_identity(self):
        from aragora.workflow.templates.email_management import EMAIL_CATEGORIZATION_TEMPLATE

        assert EMAIL_CATEGORIZATION_TEMPLATE["name"] == "Email Categorization Workflow"
        assert EMAIL_CATEGORIZATION_TEMPLATE["category"] == "email"

    def test_steps_valid(self):
        from aragora.workflow.templates.email_management import EMAIL_CATEGORIZATION_TEMPLATE

        _assert_steps_valid(EMAIL_CATEGORIZATION_TEMPLATE, "EMAIL_CATEGORIZATION")

    def test_transitions_reference_valid_steps(self):
        from aragora.workflow.templates.email_management import EMAIL_CATEGORIZATION_TEMPLATE

        _assert_transitions_reference_valid_steps(
            EMAIL_CATEGORIZATION_TEMPLATE, "EMAIL_CATEGORIZATION"
        )

    def test_has_email_tags(self):
        from aragora.workflow.templates.email_management import EMAIL_CATEGORIZATION_TEMPLATE

        tags = EMAIL_CATEGORIZATION_TEMPLATE["tags"]
        assert "email" in tags
        assert "categorization" in tags

    def test_has_low_confidence_review(self):
        from aragora.workflow.templates.email_management import EMAIL_CATEGORIZATION_TEMPLATE

        step_ids = [s["id"] for s in EMAIL_CATEGORIZATION_TEMPLATE["steps"]]
        assert "review_low_confidence" in step_ids


class TestEmailManagementRegistry:
    """Tests for email management template registry functions."""

    def test_get_email_management_template_exists(self):
        from aragora.workflow.templates.email_management import get_email_management_template

        template = get_email_management_template("email_categorization")
        assert template["name"] == "Email Categorization Workflow"

    def test_get_email_management_template_all_names(self):
        from aragora.workflow.templates.email_management import get_email_management_template

        for name in [
            "email_categorization",
            "followup_tracking",
            "snooze_management",
            "inbox_triage",
        ]:
            template = get_email_management_template(name)
            assert "name" in template
            assert "steps" in template

    def test_get_email_management_template_not_found(self):
        from aragora.workflow.templates.email_management import get_email_management_template

        with pytest.raises(KeyError, match="Unknown email management template"):
            get_email_management_template("nonexistent")

    def test_list_email_management_templates(self):
        from aragora.workflow.templates.email_management import list_email_management_templates

        templates = list_email_management_templates()
        assert len(templates) == 4
        names = [t["name"] for t in templates]
        assert "email_categorization" in names
        assert "followup_tracking" in names
        assert "snooze_management" in names
        assert "inbox_triage" in names

    def test_list_email_management_templates_fields(self):
        from aragora.workflow.templates.email_management import list_email_management_templates

        templates = list_email_management_templates()
        for t in templates:
            assert "name" in t
            assert "display_name" in t
            assert "description" in t
            assert "tags" in t

    def test_email_management_templates_registry(self):
        from aragora.workflow.templates.email_management import EMAIL_MANAGEMENT_TEMPLATES

        assert len(EMAIL_MANAGEMENT_TEMPLATES) == 4
        for key, template in EMAIL_MANAGEMENT_TEMPLATES.items():
            assert "name" in template
            assert "steps" in template
            assert "transitions" in template


# ============================================================================
# Package System Tests
# ============================================================================


class TestTemplateStatusEnum:
    """Tests for TemplateStatus enum."""

    def test_draft_value(self):
        assert TemplateStatus.DRAFT.value == "draft"

    def test_stable_value(self):
        assert TemplateStatus.STABLE.value == "stable"

    def test_deprecated_value(self):
        assert TemplateStatus.DEPRECATED.value == "deprecated"

    def test_archived_value(self):
        assert TemplateStatus.ARCHIVED.value == "archived"

    def test_all_values(self):
        values = {s.value for s in TemplateStatus}
        assert values == {"draft", "stable", "deprecated", "archived"}

    def test_string_enum(self):
        assert isinstance(TemplateStatus.STABLE, str)
        assert TemplateStatus.STABLE == "stable"


class TestTemplateCategoryEnum:
    """Tests for TemplateCategory enum."""

    def test_core_categories(self):
        assert TemplateCategory.LEGAL.value == "legal"
        assert TemplateCategory.HEALTHCARE.value == "healthcare"
        assert TemplateCategory.CODE.value == "code"
        assert TemplateCategory.ACCOUNTING.value == "accounting"
        assert TemplateCategory.DEVOPS.value == "devops"
        assert TemplateCategory.GENERAL.value == "general"

    def test_sme_categories(self):
        assert TemplateCategory.SME.value == "sme"
        assert TemplateCategory.QUICKSTART.value == "quickstart"
        assert TemplateCategory.RETAIL.value == "retail"

    def test_additional_categories(self):
        assert TemplateCategory.AI_ML.value == "ai_ml"
        assert TemplateCategory.PRODUCT.value == "product"
        assert TemplateCategory.COMPLIANCE.value == "compliance"
        assert TemplateCategory.FINANCE.value == "finance"
        assert TemplateCategory.CUSTOM.value == "custom"

    def test_string_enum(self):
        assert isinstance(TemplateCategory.LEGAL, str)
        assert TemplateCategory.LEGAL == "legal"


class TestTemplateAuthor:
    """Tests for TemplateAuthor dataclass."""

    def test_create_minimal(self):
        author = TemplateAuthor(name="Test Author")
        assert author.name == "Test Author"
        assert author.email is None
        assert author.organization is None
        assert author.url is None

    def test_create_full(self):
        author = TemplateAuthor(
            name="Test Author",
            email="test@example.com",
            organization="Test Org",
            url="https://example.com",
        )
        assert author.name == "Test Author"
        assert author.email == "test@example.com"
        assert author.organization == "Test Org"
        assert author.url == "https://example.com"


class TestTemplateDependency:
    """Tests for TemplateDependency dataclass."""

    def test_create_required(self):
        dep = TemplateDependency(name="debate_step", type="step_type")
        assert dep.name == "debate_step"
        assert dep.type == "step_type"
        assert dep.required is True
        assert dep.version is None

    def test_create_optional(self):
        dep = TemplateDependency(name="claude", type="agent", required=False, version="3.5")
        assert dep.required is False
        assert dep.version == "3.5"


class TestTemplateMetadata:
    """Tests for TemplateMetadata dataclass."""

    def test_create_minimal(self):
        meta = TemplateMetadata(id="test/template", name="Test Template", version="1.0.0")
        assert meta.id == "test/template"
        assert meta.name == "Test Template"
        assert meta.version == "1.0.0"
        assert meta.category == TemplateCategory.GENERAL
        assert meta.status == TemplateStatus.STABLE

    def test_to_dict(self):
        meta = TemplateMetadata(
            id="test/template",
            name="Test Template",
            version="1.0.0",
            category=TemplateCategory.LEGAL,
            status=TemplateStatus.DRAFT,
        )
        d = meta.to_dict()
        assert d["id"] == "test/template"
        assert d["category"] == "legal"
        assert d["status"] == "draft"

    def test_from_dict(self):
        data = {
            "id": "test/template",
            "name": "Test Template",
            "version": "1.0.0",
            "category": "legal",
            "status": "stable",
            "tags": ["legal", "test"],
        }
        meta = TemplateMetadata.from_dict(data)
        assert meta.id == "test/template"
        assert meta.category == TemplateCategory.LEGAL
        assert meta.status == TemplateStatus.STABLE
        assert meta.tags == ["legal", "test"]

    def test_from_dict_with_author(self):
        data = {
            "id": "test/template",
            "name": "Test",
            "version": "1.0.0",
            "author": {"name": "Author", "email": "a@b.com"},
        }
        meta = TemplateMetadata.from_dict(data)
        assert meta.author.name == "Author"
        assert meta.author.email == "a@b.com"

    def test_from_dict_with_dependencies(self):
        data = {
            "id": "test/template",
            "name": "Test",
            "version": "1.0.0",
            "dependencies": [
                {"name": "debate_step", "type": "step_type", "required": True},
            ],
        }
        meta = TemplateMetadata.from_dict(data)
        assert len(meta.dependencies) == 1
        assert meta.dependencies[0].name == "debate_step"

    def test_roundtrip(self):
        meta = TemplateMetadata(
            id="test/roundtrip",
            name="Roundtrip",
            version="2.0.0",
            category=TemplateCategory.DEVOPS,
            tags=["devops", "test"],
        )
        d = meta.to_dict()
        restored = TemplateMetadata.from_dict(d)
        assert restored.id == meta.id
        assert restored.version == meta.version
        assert restored.category == meta.category
        assert restored.tags == meta.tags


class TestTemplatePackage:
    """Tests for TemplatePackage dataclass."""

    def _make_package(self, **overrides):
        meta = TemplateMetadata(
            id="test/pkg",
            name="Test Pkg",
            version="1.0.0",
            **overrides.pop("meta_kwargs", {}),
        )
        template = overrides.pop("template", {"id": "test/pkg", "name": "Test", "steps": []})
        return TemplatePackage(metadata=meta, template=template, **overrides)

    def test_create_package(self):
        pkg = self._make_package()
        assert pkg.id == "test/pkg"
        assert pkg.version == "1.0.0"
        assert pkg.checksum is not None

    def test_checksum_computed(self):
        pkg = self._make_package()
        assert len(pkg.checksum) == 16

    def test_verify_checksum_valid(self):
        pkg = self._make_package()
        assert pkg.verify_checksum() is True

    def test_verify_checksum_invalid(self):
        pkg = self._make_package()
        pkg.template["extra"] = "modified"
        assert pkg.verify_checksum() is False

    def test_is_stable(self):
        pkg = self._make_package(meta_kwargs={"status": TemplateStatus.STABLE})
        assert pkg.is_stable is True
        assert pkg.is_deprecated is False

    def test_is_deprecated(self):
        pkg = self._make_package(meta_kwargs={"status": TemplateStatus.DEPRECATED})
        assert pkg.is_deprecated is True
        assert pkg.is_stable is False

    def test_to_dict(self):
        pkg = self._make_package()
        d = pkg.to_dict()
        assert d["package_version"] == "1.0"
        assert "metadata" in d
        assert "template" in d
        assert "checksum" in d

    def test_to_json(self):
        pkg = self._make_package()
        j = pkg.to_json()
        parsed = json.loads(j)
        assert parsed["metadata"]["id"] == "test/pkg"

    def test_from_dict(self):
        pkg = self._make_package()
        d = pkg.to_dict()
        restored = TemplatePackage.from_dict(d)
        assert restored.id == pkg.id
        assert restored.version == pkg.version
        assert restored.checksum == pkg.checksum

    def test_from_json(self):
        pkg = self._make_package()
        j = pkg.to_json()
        restored = TemplatePackage.from_json(j)
        assert restored.id == pkg.id

    def test_save_and_load(self, tmp_path):
        pkg = self._make_package()
        file_path = tmp_path / "test_pkg.json"
        pkg.save(file_path)
        loaded = TemplatePackage.load(file_path)
        assert loaded.id == pkg.id
        assert loaded.verify_checksum() is True


class TestCreatePackage:
    """Tests for create_package() function."""

    def test_create_from_template_dict(self):
        template = {
            "id": "legal/contract-review",
            "name": "Contract Review",
            "description": "A test template",
            "category": "legal",
            "tags": ["legal", "contracts"],
            "steps": [],
        }
        pkg = create_package(template)
        assert pkg.id == "legal/contract-review"
        assert pkg.metadata.name == "Contract Review"
        assert pkg.version == "1.0.0"

    def test_create_with_author_string(self):
        template = {"id": "test", "name": "Test", "steps": []}
        pkg = create_package(template, author="Test Author")
        assert pkg.metadata.author.name == "Test Author"

    def test_create_with_author_object(self):
        template = {"id": "test", "name": "Test", "steps": []}
        author = TemplateAuthor(name="Test Author", email="test@example.com")
        pkg = create_package(template, author=author)
        assert pkg.metadata.author.email == "test@example.com"

    def test_create_with_category_string(self):
        template = {"id": "test", "name": "Test", "steps": []}
        pkg = create_package(template, category="devops")
        assert pkg.metadata.category == TemplateCategory.DEVOPS

    def test_create_with_category_enum(self):
        template = {"id": "test", "name": "Test", "steps": []}
        pkg = create_package(template, category=TemplateCategory.HEALTHCARE)
        assert pkg.metadata.category == TemplateCategory.HEALTHCARE

    def test_create_with_invalid_category_string(self):
        template = {"id": "test", "name": "Test", "steps": []}
        pkg = create_package(template, category="nonexistent_category")
        assert pkg.metadata.category == TemplateCategory.CUSTOM

    def test_create_infers_category_from_template(self):
        template = {"id": "test", "name": "Test", "category": "legal", "steps": []}
        pkg = create_package(template)
        assert pkg.metadata.category == TemplateCategory.LEGAL

    def test_create_with_version(self):
        template = {"id": "test", "name": "Test", "steps": []}
        pkg = create_package(template, version="2.5.0")
        assert pkg.version == "2.5.0"

    def test_create_with_tags(self):
        template = {"id": "test", "name": "Test", "steps": []}
        pkg = create_package(template, tags=["tag1", "tag2"])
        assert pkg.metadata.tags == ["tag1", "tag2"]

    def test_create_with_readme(self):
        template = {"id": "test", "name": "Test", "steps": []}
        pkg = create_package(template, readme="# My Template")
        assert pkg.readme == "# My Template"

    def test_create_generates_id_from_name(self):
        template = {"name": "My Cool Template", "steps": []}
        pkg = create_package(template)
        assert pkg.id == "my-cool-template"


class TestPackageRegistry:
    """Tests for package registry functions."""

    def setup_method(self):
        """Clear registry before each test."""
        _template_registry.clear()

    def test_register_and_get_package(self):
        meta = TemplateMetadata(id="test/registry", name="Test", version="1.0.0")
        pkg = TemplatePackage(metadata=meta, template={"steps": []})
        register_package(pkg)
        result = get_package("test/registry")
        assert result is not None
        assert result.id == "test/registry"

    def test_get_package_not_found(self):
        result = get_package("nonexistent/template")
        assert result is None

    def test_get_package_specific_version(self):
        for ver in ["1.0.0", "2.0.0", "3.0.0"]:
            meta = TemplateMetadata(id="test/versioned", name="Test", version=ver)
            pkg = TemplatePackage(metadata=meta, template={"steps": []})
            register_package(pkg)

        result = get_package("test/versioned", version="2.0.0")
        assert result is not None
        assert result.version == "2.0.0"

    def test_get_package_latest_stable(self):
        meta1 = TemplateMetadata(
            id="test/stable", name="Test", version="1.0.0", status=TemplateStatus.STABLE
        )
        meta2 = TemplateMetadata(
            id="test/stable", name="Test", version="2.0.0", status=TemplateStatus.DEPRECATED
        )
        meta3 = TemplateMetadata(
            id="test/stable", name="Test", version="3.0.0", status=TemplateStatus.STABLE
        )
        for meta in [meta1, meta2, meta3]:
            register_package(TemplatePackage(metadata=meta, template={"steps": []}))

        result = get_package("test/stable")
        assert result.is_stable is True

    def test_get_package_version_not_found(self):
        meta = TemplateMetadata(id="test/ver", name="Test", version="1.0.0")
        register_package(TemplatePackage(metadata=meta, template={"steps": []}))
        result = get_package("test/ver", version="9.9.9")
        assert result is None

    def test_list_packages_empty(self):
        result = list_packages()
        assert result == []

    def test_list_packages(self):
        for i in range(3):
            meta = TemplateMetadata(id=f"test/pkg{i}", name=f"Test {i}", version="1.0.0")
            register_package(TemplatePackage(metadata=meta, template={"steps": []}))
        result = list_packages()
        assert len(result) == 3

    def test_list_packages_filter_by_category(self):
        meta1 = TemplateMetadata(
            id="cat/legal", name="Legal", version="1.0.0", category=TemplateCategory.LEGAL
        )
        meta2 = TemplateMetadata(
            id="cat/devops", name="DevOps", version="1.0.0", category=TemplateCategory.DEVOPS
        )
        register_package(TemplatePackage(metadata=meta1, template={"steps": []}))
        register_package(TemplatePackage(metadata=meta2, template={"steps": []}))

        result = list_packages(category=TemplateCategory.LEGAL)
        assert len(result) == 1
        assert result[0].id == "cat/legal"

    def test_list_packages_filter_by_category_string(self):
        meta = TemplateMetadata(
            id="cat/devops2", name="DevOps", version="1.0.0", category=TemplateCategory.DEVOPS
        )
        register_package(TemplatePackage(metadata=meta, template={"steps": []}))

        result = list_packages(category="devops")
        assert len(result) == 1

    def test_list_packages_filter_by_tags(self):
        meta1 = TemplateMetadata(
            id="tag/a", name="A", version="1.0.0", tags=["legal", "compliance"]
        )
        meta2 = TemplateMetadata(id="tag/b", name="B", version="1.0.0", tags=["devops", "cicd"])
        register_package(TemplatePackage(metadata=meta1, template={"steps": []}))
        register_package(TemplatePackage(metadata=meta2, template={"steps": []}))

        result = list_packages(tags=["legal"])
        assert len(result) == 1
        assert result[0].id == "tag/a"

    def test_list_packages_excludes_deprecated(self):
        meta1 = TemplateMetadata(
            id="dep/active", name="Active", version="1.0.0", status=TemplateStatus.STABLE
        )
        meta2 = TemplateMetadata(
            id="dep/old", name="Old", version="1.0.0", status=TemplateStatus.DEPRECATED
        )
        register_package(TemplatePackage(metadata=meta1, template={"steps": []}))
        register_package(TemplatePackage(metadata=meta2, template={"steps": []}))

        result = list_packages()
        assert len(result) == 1
        assert result[0].id == "dep/active"

    def test_list_packages_include_deprecated(self):
        meta1 = TemplateMetadata(
            id="dep2/active", name="Active", version="1.0.0", status=TemplateStatus.STABLE
        )
        meta2 = TemplateMetadata(
            id="dep2/old", name="Old", version="1.0.0", status=TemplateStatus.DEPRECATED
        )
        register_package(TemplatePackage(metadata=meta1, template={"steps": []}))
        register_package(TemplatePackage(metadata=meta2, template={"steps": []}))

        result = list_packages(include_deprecated=True)
        assert len(result) == 2


# ============================================================================
# Cross-Domain Validation Tests
# ============================================================================


class TestAllDomainTemplatesStructure:
    """Cross-domain structural validation of all templates."""

    @pytest.fixture
    def all_templates(self):
        from aragora.workflow.templates.devops import (
            CICD_PIPELINE_REVIEW_TEMPLATE,
            INCIDENT_RESPONSE_TEMPLATE,
            INFRASTRUCTURE_AUDIT_TEMPLATE,
            PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE,
        )
        from aragora.workflow.templates.accounting import (
            FINANCIAL_AUDIT_TEMPLATE,
            SOX_COMPLIANCE_TEMPLATE,
            BANK_RECONCILIATION_TEMPLATE,
        )
        from aragora.workflow.templates.legal import (
            CONTRACT_REVIEW_TEMPLATE,
            DUE_DILIGENCE_TEMPLATE,
            COMPLIANCE_AUDIT_TEMPLATE,
        )
        from aragora.workflow.templates.healthcare import (
            HIPAA_ASSESSMENT_TEMPLATE,
            CLINICAL_REVIEW_TEMPLATE,
            PHI_AUDIT_TEMPLATE,
        )
        from aragora.workflow.templates.email_management import (
            EMAIL_CATEGORIZATION_TEMPLATE,
            FOLLOWUP_TRACKING_TEMPLATE,
            SNOOZE_MANAGEMENT_TEMPLATE,
            INBOX_TRIAGE_TEMPLATE,
        )

        return {
            "CICD_PIPELINE_REVIEW": CICD_PIPELINE_REVIEW_TEMPLATE,
            "INCIDENT_RESPONSE": INCIDENT_RESPONSE_TEMPLATE,
            "INFRASTRUCTURE_AUDIT": INFRASTRUCTURE_AUDIT_TEMPLATE,
            "PAGERDUTY_INCIDENT": PAGERDUTY_INCIDENT_WORKFLOW_TEMPLATE,
            "FINANCIAL_AUDIT": FINANCIAL_AUDIT_TEMPLATE,
            "SOX_COMPLIANCE": SOX_COMPLIANCE_TEMPLATE,
            "BANK_RECONCILIATION": BANK_RECONCILIATION_TEMPLATE,
            "CONTRACT_REVIEW": CONTRACT_REVIEW_TEMPLATE,
            "DUE_DILIGENCE": DUE_DILIGENCE_TEMPLATE,
            "COMPLIANCE_AUDIT": COMPLIANCE_AUDIT_TEMPLATE,
            "HIPAA_ASSESSMENT": HIPAA_ASSESSMENT_TEMPLATE,
            "CLINICAL_REVIEW": CLINICAL_REVIEW_TEMPLATE,
            "PHI_AUDIT": PHI_AUDIT_TEMPLATE,
            "EMAIL_CATEGORIZATION": EMAIL_CATEGORIZATION_TEMPLATE,
            "FOLLOWUP_TRACKING": FOLLOWUP_TRACKING_TEMPLATE,
            "SNOOZE_MANAGEMENT": SNOOZE_MANAGEMENT_TEMPLATE,
            "INBOX_TRIAGE": INBOX_TRIAGE_TEMPLATE,
        }

    def test_all_templates_have_required_fields(self, all_templates):
        for label, template in all_templates.items():
            _assert_template_structure(template, label)

    def test_all_templates_have_valid_steps(self, all_templates):
        for label, template in all_templates.items():
            _assert_steps_valid(template, label)

    def test_all_templates_have_valid_transitions(self, all_templates):
        for label, template in all_templates.items():
            _assert_transitions_reference_valid_steps(template, label)

    def test_all_templates_have_version(self, all_templates):
        for label, template in all_templates.items():
            assert template["version"] == "1.0", f"{label} version is not 1.0"

    def test_all_templates_have_nonempty_tags(self, all_templates):
        for label, template in all_templates.items():
            assert len(template["tags"]) >= 2, f"{label} should have at least 2 tags"

    def test_all_debate_steps_have_agents(self, all_templates):
        for label, template in all_templates.items():
            for step in template["steps"]:
                if step.get("type") == "debate":
                    assert "agents" in step.get("config", {}), (
                        f"{label} debate step '{step['id']}' missing agents"
                    )

    def test_no_duplicate_step_ids(self, all_templates):
        for label, template in all_templates.items():
            step_ids = []
            for step in template["steps"]:
                if step.get("type") == "parallel":
                    step_ids.append(step["id"])
                    for branch in step.get("branches", []):
                        for sub in branch.get("steps", []):
                            step_ids.append(sub["id"])
                else:
                    step_ids.append(step["id"])
            assert len(step_ids) == len(set(step_ids)), f"{label} has duplicate step IDs"
