"""
Tests for Vertical-Specific Playbook Templates.

Tests coverage for:
- Healthcare Clinical Decision Playbook (HIPAA/HITECH/FDA)
- Financial Regulatory Decision Playbook (SOX/Basel III/MiFID II)
- Legal Analysis Decision Playbook (privilege, precedent, jurisdictional)
- Cross-vertical structural validation
- Serialization/deserialization (JSON round-trip)
- Template registry integration (VERTICAL_TEMPLATES, WORKFLOW_TEMPLATES)
"""

from __future__ import annotations

import json

import pytest


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


def _assert_playbook_fields(template: dict, label: str) -> None:
    """Assert playbook-specific fields present on vertical templates."""
    assert "compliance_frameworks" in template, f"{label} missing 'compliance_frameworks'"
    assert isinstance(template["compliance_frameworks"], list)
    assert len(template["compliance_frameworks"]) > 0, f"{label} has no compliance_frameworks"

    assert "required_agent_types" in template, f"{label} missing 'required_agent_types'"
    assert isinstance(template["required_agent_types"], list)
    assert len(template["required_agent_types"]) >= 3, (
        f"{label} should have at least 3 required agent types"
    )

    assert "output_format" in template, f"{label} missing 'output_format'"
    assert "decision_receipt" in template["output_format"], (
        f"{label} output_format missing 'decision_receipt'"
    )

    assert "metadata" in template, f"{label} missing 'metadata'"
    assert "author" in template["metadata"]
    assert "version" in template["metadata"]


# ============================================================================
# Healthcare Clinical Decision Playbook Tests
# ============================================================================


class TestHealthcareClinicalDecisionTemplate:
    """Tests for Healthcare Clinical Decision Playbook template."""

    def test_template_structure(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        _assert_template_structure(HEALTHCARE_CLINICAL_DECISION_TEMPLATE, "HEALTHCARE_CLINICAL")

    def test_template_identity(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        assert HEALTHCARE_CLINICAL_DECISION_TEMPLATE["name"] == (
            "Healthcare Clinical Decision Playbook"
        )
        assert HEALTHCARE_CLINICAL_DECISION_TEMPLATE["category"] == "healthcare"

    def test_playbook_fields(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        _assert_playbook_fields(HEALTHCARE_CLINICAL_DECISION_TEMPLATE, "HEALTHCARE_CLINICAL")

    def test_steps_valid(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        _assert_steps_valid(HEALTHCARE_CLINICAL_DECISION_TEMPLATE, "HEALTHCARE_CLINICAL")

    def test_transitions_reference_valid_steps(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        _assert_transitions_reference_valid_steps(
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE, "HEALTHCARE_CLINICAL"
        )

    def test_compliance_frameworks(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        frameworks = HEALTHCARE_CLINICAL_DECISION_TEMPLATE["compliance_frameworks"]
        assert "HIPAA" in frameworks
        assert "HITECH" in frameworks
        assert "FDA_21_CFR_11" in frameworks

    def test_required_agent_types(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        agents = HEALTHCARE_CLINICAL_DECISION_TEMPLATE["required_agent_types"]
        assert "clinical_reviewer" in agents
        assert "compliance_officer" in agents
        assert "hipaa_auditor" in agents

    def test_has_healthcare_tags(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        tags = HEALTHCARE_CLINICAL_DECISION_TEMPLATE["tags"]
        assert "healthcare" in tags
        assert "hipaa" in tags
        assert "hitech" in tags
        assert "clinical" in tags
        assert "evidence-grading" in tags
        assert "patient-safety" in tags
        assert "playbook" in tags

    def test_has_phi_screening(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in HEALTHCARE_CLINICAL_DECISION_TEMPLATE["steps"]]
        assert "phi_screening" in step_ids

    def test_has_compliance_pre_check(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in HEALTHCARE_CLINICAL_DECISION_TEMPLATE["steps"]]
        assert "compliance_pre_check" in step_ids

    def test_compliance_pre_check_has_checks(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        step = next(
            s
            for s in HEALTHCARE_CLINICAL_DECISION_TEMPLATE["steps"]
            if s["id"] == "compliance_pre_check"
        )
        checks = step["config"]["compliance_checks"]
        assert "hipaa_authorization_valid" in checks
        assert "minimum_necessary_standard" in checks
        assert "patient_consent_documented" in checks

    def test_has_evidence_grading(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in HEALTHCARE_CLINICAL_DECISION_TEMPLATE["steps"]]
        assert "evidence_grading" in step_ids

    def test_evidence_grading_has_grade_levels(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        step = next(
            s
            for s in HEALTHCARE_CLINICAL_DECISION_TEMPLATE["steps"]
            if s["id"] == "evidence_grading"
        )
        levels = step["config"]["evidence_levels"]
        assert "high" in levels
        assert "moderate" in levels
        assert "low" in levels
        assert "very_low" in levels

    def test_has_patient_safety_screening(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in HEALTHCARE_CLINICAL_DECISION_TEMPLATE["steps"]]
        assert "patient_safety_screening" in step_ids

    def test_has_do_no_harm_gate(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in HEALTHCARE_CLINICAL_DECISION_TEMPLATE["steps"]]
        assert "do_no_harm_gate" in step_ids

    def test_has_clinical_outcome_tracking(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in HEALTHCARE_CLINICAL_DECISION_TEMPLATE["steps"]]
        assert "clinical_outcome_tracking" in step_ids

    def test_has_decision_receipt_generation(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in HEALTHCARE_CLINICAL_DECISION_TEMPLATE["steps"]]
        assert "generate_decision_receipt" in step_ids

    def test_archive_retention_six_years(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        archive_step = next(
            s for s in HEALTHCARE_CLINICAL_DECISION_TEMPLATE["steps"] if s["id"] == "archive"
        )
        assert archive_step["config"]["retention_years"] == 6

    def test_output_format_has_receipt_and_summary(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        output = HEALTHCARE_CLINICAL_DECISION_TEMPLATE["output_format"]
        assert "decision_receipt" in output
        assert "clinical_summary" in output
        assert output["decision_receipt"]["type"] == "gauntlet_receipt"

    def test_debate_steps_have_agents(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        for step in HEALTHCARE_CLINICAL_DECISION_TEMPLATE["steps"]:
            if step.get("type") == "debate":
                assert "agents" in step["config"], f"Debate step '{step['id']}' missing agents"
                assert len(step["config"]["agents"]) >= 2

    def test_no_duplicate_step_ids(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in HEALTHCARE_CLINICAL_DECISION_TEMPLATE["steps"]]
        assert len(step_ids) == len(set(step_ids))

    def test_json_serialization_roundtrip(self):
        from aragora.workflow.templates.catalog.verticals.healthcare import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
        )

        serialized = json.dumps(HEALTHCARE_CLINICAL_DECISION_TEMPLATE)
        deserialized = json.loads(serialized)
        assert deserialized["name"] == HEALTHCARE_CLINICAL_DECISION_TEMPLATE["name"]
        assert len(deserialized["steps"]) == len(HEALTHCARE_CLINICAL_DECISION_TEMPLATE["steps"])
        assert len(deserialized["transitions"]) == len(
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE["transitions"]
        )
        assert (
            deserialized["compliance_frameworks"]
            == (HEALTHCARE_CLINICAL_DECISION_TEMPLATE["compliance_frameworks"])
        )


# ============================================================================
# Financial Regulatory Decision Playbook Tests
# ============================================================================


class TestFinancialRegulatoryDecisionTemplate:
    """Tests for Financial Regulatory Decision Playbook template."""

    def test_template_structure(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        _assert_template_structure(FINANCIAL_REGULATORY_DECISION_TEMPLATE, "FINANCIAL_REGULATORY")

    def test_template_identity(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        assert FINANCIAL_REGULATORY_DECISION_TEMPLATE["name"] == (
            "Financial Regulatory Decision Playbook"
        )
        assert FINANCIAL_REGULATORY_DECISION_TEMPLATE["category"] == "finance"

    def test_playbook_fields(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        _assert_playbook_fields(FINANCIAL_REGULATORY_DECISION_TEMPLATE, "FINANCIAL_REGULATORY")

    def test_steps_valid(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        _assert_steps_valid(FINANCIAL_REGULATORY_DECISION_TEMPLATE, "FINANCIAL_REGULATORY")

    def test_transitions_reference_valid_steps(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        _assert_transitions_reference_valid_steps(
            FINANCIAL_REGULATORY_DECISION_TEMPLATE, "FINANCIAL_REGULATORY"
        )

    def test_compliance_frameworks(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        frameworks = FINANCIAL_REGULATORY_DECISION_TEMPLATE["compliance_frameworks"]
        assert "SOX" in frameworks
        assert "BASEL_III" in frameworks
        assert "MIFID_II" in frameworks
        assert "GAAP" in frameworks

    def test_required_agent_types(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        agents = FINANCIAL_REGULATORY_DECISION_TEMPLATE["required_agent_types"]
        assert "financial_auditor" in agents
        assert "compliance_officer" in agents
        assert "sox" in agents
        assert "risk_analyst" in agents

    def test_has_financial_tags(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        tags = FINANCIAL_REGULATORY_DECISION_TEMPLATE["tags"]
        assert "finance" in tags
        assert "sox" in tags
        assert "basel-iii" in tags
        assert "mifid-ii" in tags
        assert "segregation-of-duties" in tags
        assert "playbook" in tags

    def test_has_segregation_of_duties_check(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in FINANCIAL_REGULATORY_DECISION_TEMPLATE["steps"]]
        assert "segregation_of_duties_check" in step_ids

    def test_sod_check_has_rules(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        step = next(
            s
            for s in FINANCIAL_REGULATORY_DECISION_TEMPLATE["steps"]
            if s["id"] == "segregation_of_duties_check"
        )
        rules = step["config"]["sod_rules"]
        assert "proposer_not_approver" in rules
        assert "reviewer_not_proposer" in rules
        assert "approver_not_reviewer" in rules

    def test_has_regulatory_compliance_screening(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in FINANCIAL_REGULATORY_DECISION_TEMPLATE["steps"]]
        assert "regulatory_compliance_screening" in step_ids

    def test_regulatory_screening_has_compliance_checks(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        step = next(
            s
            for s in FINANCIAL_REGULATORY_DECISION_TEMPLATE["steps"]
            if s["id"] == "regulatory_compliance_screening"
        )
        checks = step["config"]["compliance_checks"]
        assert "sox_section_302_certification" in checks
        assert "sox_section_404_internal_controls" in checks
        assert "basel_iii_capital_adequacy" in checks
        assert "mifid_ii_best_execution" in checks

    def test_has_risk_assessment_stages(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in FINANCIAL_REGULATORY_DECISION_TEMPLATE["steps"]]
        assert "market_risk_assessment" in step_ids
        assert "credit_risk_assessment" in step_ids
        assert "operational_risk_assessment" in step_ids
        assert "risk_consolidation" in step_ids

    def test_market_risk_has_categories(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        step = next(
            s
            for s in FINANCIAL_REGULATORY_DECISION_TEMPLATE["steps"]
            if s["id"] == "market_risk_assessment"
        )
        categories = step["config"]["risk_categories"]
        assert "price_volatility" in categories
        assert "liquidity_risk" in categories

    def test_has_materiality_gate(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in FINANCIAL_REGULATORY_DECISION_TEMPLATE["steps"]]
        assert "materiality_gate" in step_ids

    def test_has_audit_trail_generation(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in FINANCIAL_REGULATORY_DECISION_TEMPLATE["steps"]]
        assert "generate_audit_trail" in step_ids

    def test_has_decision_receipt_generation(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in FINANCIAL_REGULATORY_DECISION_TEMPLATE["steps"]]
        assert "generate_decision_receipt" in step_ids

    def test_archive_retention_seven_years(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        archive_step = next(
            s for s in FINANCIAL_REGULATORY_DECISION_TEMPLATE["steps"] if s["id"] == "archive"
        )
        assert archive_step["config"]["retention_years"] == 7

    def test_output_format_has_receipt_and_risk_report(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        output = FINANCIAL_REGULATORY_DECISION_TEMPLATE["output_format"]
        assert "decision_receipt" in output
        assert "risk_report" in output
        assert output["decision_receipt"]["type"] == "gauntlet_receipt"

    def test_debate_steps_have_agents(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        for step in FINANCIAL_REGULATORY_DECISION_TEMPLATE["steps"]:
            if step.get("type") == "debate":
                assert "agents" in step["config"], f"Debate step '{step['id']}' missing agents"
                assert len(step["config"]["agents"]) >= 2

    def test_no_duplicate_step_ids(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in FINANCIAL_REGULATORY_DECISION_TEMPLATE["steps"]]
        assert len(step_ids) == len(set(step_ids))

    def test_json_serialization_roundtrip(self):
        from aragora.workflow.templates.catalog.verticals.financial import (
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
        )

        serialized = json.dumps(FINANCIAL_REGULATORY_DECISION_TEMPLATE)
        deserialized = json.loads(serialized)
        assert deserialized["name"] == (FINANCIAL_REGULATORY_DECISION_TEMPLATE["name"])
        assert len(deserialized["steps"]) == len(FINANCIAL_REGULATORY_DECISION_TEMPLATE["steps"])
        assert len(deserialized["transitions"]) == len(
            FINANCIAL_REGULATORY_DECISION_TEMPLATE["transitions"]
        )
        assert (
            deserialized["compliance_frameworks"]
            == (FINANCIAL_REGULATORY_DECISION_TEMPLATE["compliance_frameworks"])
        )


# ============================================================================
# Legal Analysis Decision Playbook Tests
# ============================================================================


class TestLegalAnalysisDecisionTemplate:
    """Tests for Legal Analysis Decision Playbook template."""

    def test_template_structure(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        _assert_template_structure(LEGAL_ANALYSIS_DECISION_TEMPLATE, "LEGAL_ANALYSIS")

    def test_template_identity(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        assert LEGAL_ANALYSIS_DECISION_TEMPLATE["name"] == ("Legal Analysis Decision Playbook")
        assert LEGAL_ANALYSIS_DECISION_TEMPLATE["category"] == "legal"

    def test_playbook_fields(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        _assert_playbook_fields(LEGAL_ANALYSIS_DECISION_TEMPLATE, "LEGAL_ANALYSIS")

    def test_steps_valid(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        _assert_steps_valid(LEGAL_ANALYSIS_DECISION_TEMPLATE, "LEGAL_ANALYSIS")

    def test_transitions_reference_valid_steps(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        _assert_transitions_reference_valid_steps(
            LEGAL_ANALYSIS_DECISION_TEMPLATE, "LEGAL_ANALYSIS"
        )

    def test_compliance_frameworks(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        frameworks = LEGAL_ANALYSIS_DECISION_TEMPLATE["compliance_frameworks"]
        assert "ABA_MODEL_RULES" in frameworks
        assert "ATTORNEY_CLIENT_PRIVILEGE" in frameworks
        assert "WORK_PRODUCT_DOCTRINE" in frameworks
        assert "CONFLICT_OF_INTEREST" in frameworks

    def test_required_agent_types(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        agents = LEGAL_ANALYSIS_DECISION_TEMPLATE["required_agent_types"]
        assert "contract_analyst" in agents
        assert "compliance_officer" in agents
        assert "litigation_support" in agents

    def test_has_legal_tags(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        tags = LEGAL_ANALYSIS_DECISION_TEMPLATE["tags"]
        assert "legal" in tags
        assert "precedent" in tags
        assert "jurisdictional" in tags
        assert "privilege" in tags
        assert "confidentiality" in tags
        assert "playbook" in tags

    def test_has_conflict_of_interest_check(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"]]
        assert "conflict_of_interest_check" in step_ids

    def test_conflict_check_has_rules(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        step = next(
            s
            for s in LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"]
            if s["id"] == "conflict_of_interest_check"
        )
        rules = step["config"]["conflict_rules"]
        assert "current_client_conflict_1_7" in rules
        assert "former_client_conflict_1_9" in rules

    def test_has_privilege_classification(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"]]
        assert "privilege_classification" in step_ids

    def test_privilege_classification_has_tiers(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        step = next(
            s
            for s in LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"]
            if s["id"] == "privilege_classification"
        )
        tiers = step["config"]["classification_tiers"]
        assert "public" in tiers
        assert "confidential" in tiers
        assert "highly_confidential_aeo" in tiers

    def test_privilege_classification_has_privilege_categories(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        step = next(
            s
            for s in LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"]
            if s["id"] == "privilege_classification"
        )
        categories = step["config"]["privilege_categories"]
        assert "attorney_client_privilege" in categories
        assert "work_product_doctrine" in categories

    def test_has_precedent_analysis(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"]]
        assert "precedent_retrieval" in step_ids
        assert "precedent_analysis" in step_ids

    def test_has_jurisdictional_analysis(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"]]
        assert "jurisdictional_analysis" in step_ids

    def test_jurisdictional_analysis_has_dimensions(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        step = next(
            s
            for s in LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"]
            if s["id"] == "jurisdictional_analysis"
        )
        dimensions = step["config"]["analysis_dimensions"]
        assert "governing_law" in dimensions
        assert "choice_of_law" in dimensions
        assert "forum_selection" in dimensions

    def test_has_liability_assessment(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"]]
        assert "liability_assessment" in step_ids

    def test_liability_assessment_has_risk_dimensions(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        step = next(
            s
            for s in LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"]
            if s["id"] == "liability_assessment"
        )
        dims = step["config"]["risk_dimensions"]
        assert "probability_of_adverse_outcome" in dims
        assert "magnitude_of_exposure" in dims
        assert "reputational_risk" in dims

    def test_has_privilege_review(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"]]
        assert "privilege_review" in step_ids

    def test_has_partner_sign_off(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"]]
        assert "partner_sign_off" in step_ids

    def test_has_decision_receipt_generation(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"]]
        assert "generate_decision_receipt" in step_ids

    def test_has_legal_memorandum_generation(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"]]
        assert "generate_legal_memorandum" in step_ids

    def test_archive_retention_ten_years(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        archive_step = next(
            s for s in LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"] if s["id"] == "archive"
        )
        assert archive_step["config"]["retention_years"] == 10

    def test_archive_has_access_control(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        archive_step = next(
            s for s in LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"] if s["id"] == "archive"
        )
        assert archive_step["config"]["access_control"] == "attorney_eyes_only"

    def test_output_format_has_receipt_and_memorandum(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        output = LEGAL_ANALYSIS_DECISION_TEMPLATE["output_format"]
        assert "decision_receipt" in output
        assert "legal_memorandum" in output
        assert output["decision_receipt"]["type"] == "gauntlet_receipt"

    def test_legal_memorandum_has_standard_sections(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        sections = LEGAL_ANALYSIS_DECISION_TEMPLATE["output_format"]["legal_memorandum"]["sections"]
        assert "question_presented" in sections
        assert "brief_answer" in sections
        assert "statement_of_facts" in sections
        assert "applicable_law" in sections
        assert "analysis" in sections
        assert "conclusion" in sections

    def test_debate_steps_have_agents(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        for step in LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"]:
            if step.get("type") == "debate":
                assert "agents" in step["config"], f"Debate step '{step['id']}' missing agents"
                assert len(step["config"]["agents"]) >= 2

    def test_no_duplicate_step_ids(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        step_ids = [s["id"] for s in LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"]]
        assert len(step_ids) == len(set(step_ids))

    def test_json_serialization_roundtrip(self):
        from aragora.workflow.templates.catalog.verticals.legal import (
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
        )

        serialized = json.dumps(LEGAL_ANALYSIS_DECISION_TEMPLATE)
        deserialized = json.loads(serialized)
        assert deserialized["name"] == LEGAL_ANALYSIS_DECISION_TEMPLATE["name"]
        assert len(deserialized["steps"]) == len(LEGAL_ANALYSIS_DECISION_TEMPLATE["steps"])
        assert len(deserialized["transitions"]) == len(
            LEGAL_ANALYSIS_DECISION_TEMPLATE["transitions"]
        )
        assert (
            deserialized["compliance_frameworks"]
            == (LEGAL_ANALYSIS_DECISION_TEMPLATE["compliance_frameworks"])
        )


# ============================================================================
# Cross-Vertical Validation Tests
# ============================================================================


class TestVerticalTemplateRegistry:
    """Tests for vertical template registry integration."""

    def test_vertical_templates_dict_has_all_three(self):
        from aragora.workflow.templates.catalog.verticals import VERTICAL_TEMPLATES

        assert "vertical/healthcare-clinical-decision" in VERTICAL_TEMPLATES
        assert "vertical/financial-regulatory-decision" in VERTICAL_TEMPLATES
        assert "vertical/legal-analysis-decision" in VERTICAL_TEMPLATES
        assert len(VERTICAL_TEMPLATES) == 3

    def test_vertical_templates_registered_in_workflow_templates(self):
        from aragora.workflow.templates import WORKFLOW_TEMPLATES

        assert "vertical/healthcare-clinical-decision" in WORKFLOW_TEMPLATES
        assert "vertical/financial-regulatory-decision" in WORKFLOW_TEMPLATES
        assert "vertical/legal-analysis-decision" in WORKFLOW_TEMPLATES

    def test_get_template_returns_verticals(self):
        from aragora.workflow.templates import get_template

        healthcare = get_template("vertical/healthcare-clinical-decision")
        assert healthcare is not None
        assert healthcare["name"] == "Healthcare Clinical Decision Playbook"

        financial = get_template("vertical/financial-regulatory-decision")
        assert financial is not None
        assert financial["name"] == "Financial Regulatory Decision Playbook"

        legal = get_template("vertical/legal-analysis-decision")
        assert legal is not None
        assert legal["name"] == "Legal Analysis Decision Playbook"

    def test_list_templates_includes_verticals(self):
        from aragora.workflow.templates import list_templates

        templates = list_templates(category="vertical")
        template_ids = [t["id"] for t in templates]
        assert "vertical/healthcare-clinical-decision" in template_ids
        assert "vertical/financial-regulatory-decision" in template_ids
        assert "vertical/legal-analysis-decision" in template_ids

    def test_catalog_init_exports(self):
        from aragora.workflow.templates.catalog import (
            HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
            FINANCIAL_REGULATORY_DECISION_TEMPLATE,
            LEGAL_ANALYSIS_DECISION_TEMPLATE,
            VERTICAL_TEMPLATES,
        )

        assert HEALTHCARE_CLINICAL_DECISION_TEMPLATE is not None
        assert FINANCIAL_REGULATORY_DECISION_TEMPLATE is not None
        assert LEGAL_ANALYSIS_DECISION_TEMPLATE is not None
        assert len(VERTICAL_TEMPLATES) == 3


class TestAllVerticalTemplatesStructure:
    """Cross-vertical structural validation."""

    @pytest.fixture
    def all_vertical_templates(self):
        from aragora.workflow.templates.catalog.verticals import VERTICAL_TEMPLATES

        return VERTICAL_TEMPLATES

    def test_all_have_required_fields(self, all_vertical_templates):
        for key, template in all_vertical_templates.items():
            _assert_template_structure(template, key)

    def test_all_have_playbook_fields(self, all_vertical_templates):
        for key, template in all_vertical_templates.items():
            _assert_playbook_fields(template, key)

    def test_all_have_valid_steps(self, all_vertical_templates):
        for key, template in all_vertical_templates.items():
            _assert_steps_valid(template, key)

    def test_all_have_valid_transitions(self, all_vertical_templates):
        for key, template in all_vertical_templates.items():
            _assert_transitions_reference_valid_steps(template, key)

    def test_all_have_version_1_0(self, all_vertical_templates):
        for key, template in all_vertical_templates.items():
            assert template["version"] == "1.0", f"{key} version is not 1.0"

    def test_all_have_playbook_tag(self, all_vertical_templates):
        for key, template in all_vertical_templates.items():
            assert "playbook" in template["tags"], f"{key} missing 'playbook' tag"

    def test_all_have_decision_receipt_tag(self, all_vertical_templates):
        for key, template in all_vertical_templates.items():
            assert "decision-receipt" in template["tags"], f"{key} missing 'decision-receipt' tag"

    def test_all_have_at_least_five_tags(self, all_vertical_templates):
        for key, template in all_vertical_templates.items():
            assert len(template["tags"]) >= 5, f"{key} should have at least 5 tags"

    def test_all_have_compliance_frameworks(self, all_vertical_templates):
        for key, template in all_vertical_templates.items():
            assert len(template["compliance_frameworks"]) >= 2, (
                f"{key} should have at least 2 compliance frameworks"
            )

    def test_all_have_decision_receipt_step(self, all_vertical_templates):
        for key, template in all_vertical_templates.items():
            step_ids = [s["id"] for s in template["steps"]]
            assert "generate_decision_receipt" in step_ids, (
                f"{key} missing 'generate_decision_receipt' step"
            )

    def test_all_have_archive_step(self, all_vertical_templates):
        for key, template in all_vertical_templates.items():
            step_ids = [s["id"] for s in template["steps"]]
            assert "archive" in step_ids, f"{key} missing 'archive' step"

    def test_all_archive_steps_have_retention_years(self, all_vertical_templates):
        for key, template in all_vertical_templates.items():
            archive_step = next(s for s in template["steps"] if s["id"] == "archive")
            assert "retention_years" in archive_step["config"], (
                f"{key} archive step missing retention_years"
            )
            assert archive_step["config"]["retention_years"] >= 6

    def test_all_debate_steps_have_agents(self, all_vertical_templates):
        for key, template in all_vertical_templates.items():
            for step in template["steps"]:
                if step.get("type") == "debate":
                    assert "agents" in step.get("config", {}), (
                        f"{key} debate step '{step['id']}' missing agents"
                    )

    def test_no_duplicate_step_ids_in_any(self, all_vertical_templates):
        for key, template in all_vertical_templates.items():
            step_ids = [s["id"] for s in template["steps"]]
            assert len(step_ids) == len(set(step_ids)), f"{key} has duplicate step IDs"

    def test_all_have_gauntlet_receipt_output(self, all_vertical_templates):
        for key, template in all_vertical_templates.items():
            receipt = template["output_format"]["decision_receipt"]
            assert receipt["type"] == "gauntlet_receipt", (
                f"{key} receipt type should be gauntlet_receipt"
            )

    def test_all_json_serializable(self, all_vertical_templates):
        for key, template in all_vertical_templates.items():
            serialized = json.dumps(template)
            deserialized = json.loads(serialized)
            assert deserialized["name"] == template["name"]
            assert len(deserialized["steps"]) == len(template["steps"])
