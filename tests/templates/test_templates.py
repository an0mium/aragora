"""
Tests for individual template validation.

Tests cover:
- Template structure validation
- Role configuration validation
- Phase configuration validation
- Output format validation
- Rubric validation
- Template-specific requirements
"""

import re

import pytest

from aragora.templates import (
    CODE_REVIEW_TEMPLATE,
    DESIGN_DOC_TEMPLATE,
    INCIDENT_RESPONSE_TEMPLATE,
    RESEARCH_SYNTHESIS_TEMPLATE,
    SECURITY_AUDIT_TEMPLATE,
    ARCHITECTURE_REVIEW_TEMPLATE,
    HEALTHCARE_COMPLIANCE_TEMPLATE,
    FINANCIAL_RISK_TEMPLATE,
    TEMPLATES,
    DebateTemplate,
    TemplateType,
)


class TestTemplateStructureValidation:
    """Tests for template structure validation."""

    @pytest.fixture(params=list(TEMPLATES.values()), ids=lambda t: t.template_id)
    def template(self, request):
        """Parametrize tests with all templates."""
        return request.param

    def test_template_has_unique_id(self, template):
        """Test that template has a unique ID."""
        assert template.template_id
        assert len(template.template_id) > 0

    def test_template_has_name(self, template):
        """Test that template has a name."""
        assert template.name
        assert len(template.name) > 0

    def test_template_has_description(self, template):
        """Test that template has a description."""
        assert template.description
        assert len(template.description) > 0

    def test_template_has_roles(self, template):
        """Test that template has at least one role."""
        assert len(template.roles) > 0

    def test_template_has_phases(self, template):
        """Test that template has at least one phase."""
        assert len(template.phases) > 0

    def test_template_has_positive_recommended_agents(self, template):
        """Test that recommended_agents is positive."""
        assert template.recommended_agents > 0

    def test_template_has_positive_max_rounds(self, template):
        """Test that max_rounds is positive."""
        assert template.max_rounds > 0

    def test_template_has_valid_consensus_threshold(self, template):
        """Test that consensus_threshold is between 0 and 1."""
        assert 0.0 <= template.consensus_threshold <= 1.0

    def test_template_has_domain(self, template):
        """Test that template has a domain."""
        assert template.domain
        assert len(template.domain) > 0

    def test_template_has_valid_difficulty(self, template):
        """Test that difficulty is between 0 and 1."""
        assert 0.0 <= template.difficulty <= 1.0


class TestRoleValidation:
    """Tests for role configuration validation."""

    @pytest.fixture(params=list(TEMPLATES.values()), ids=lambda t: t.template_id)
    def template(self, request):
        """Parametrize tests with all templates."""
        return request.param

    def test_all_roles_have_names(self, template):
        """Test that all roles have names."""
        for role in template.roles:
            assert role.name
            assert len(role.name) > 0

    def test_all_roles_have_descriptions(self, template):
        """Test that all roles have descriptions."""
        for role in template.roles:
            assert role.description
            assert len(role.description) > 0

    def test_all_roles_have_objectives(self, template):
        """Test that all roles have at least one objective."""
        for role in template.roles:
            assert len(role.objectives) > 0

    def test_all_roles_have_evaluation_criteria(self, template):
        """Test that all roles have at least one evaluation criterion."""
        for role in template.roles:
            assert len(role.evaluation_criteria) > 0

    def test_role_names_are_unique(self, template):
        """Test that role names are unique within a template."""
        role_names = [role.name for role in template.roles]
        assert len(role_names) == len(set(role_names))


class TestPhaseValidation:
    """Tests for phase configuration validation."""

    @pytest.fixture(params=list(TEMPLATES.values()), ids=lambda t: t.template_id)
    def template(self, request):
        """Parametrize tests with all templates."""
        return request.param

    def test_all_phases_have_names(self, template):
        """Test that all phases have names."""
        for phase in template.phases:
            assert phase.name
            assert len(phase.name) > 0

    def test_all_phases_have_descriptions(self, template):
        """Test that all phases have descriptions."""
        for phase in template.phases:
            assert phase.description
            assert len(phase.description) > 0

    def test_all_phases_have_positive_duration(self, template):
        """Test that all phases have positive duration."""
        for phase in template.phases:
            assert phase.duration_rounds > 0

    def test_all_phases_have_active_roles(self, template):
        """Test that all phases have at least one active role."""
        for phase in template.phases:
            assert len(phase.roles_active) > 0

    def test_all_phases_have_objectives(self, template):
        """Test that all phases have at least one objective."""
        for phase in template.phases:
            assert len(phase.objectives) > 0

    def test_all_phases_have_outputs(self, template):
        """Test that all phases have at least one output."""
        for phase in template.phases:
            assert len(phase.outputs) > 0

    def test_phase_names_are_unique(self, template):
        """Test that phase names are unique within a template."""
        phase_names = [phase.name for phase in template.phases]
        assert len(phase_names) == len(set(phase_names))

    def test_phase_roles_reference_defined_roles(self, template):
        """Test that phase roles reference roles defined in template."""
        role_names = {role.name for role in template.roles}

        for phase in template.phases:
            for active_role in phase.roles_active:
                assert active_role in role_names, (
                    f"Phase '{phase.name}' references undefined role '{active_role}'"
                )


class TestOutputFormatValidation:
    """Tests for output format validation."""

    @pytest.fixture(params=list(TEMPLATES.values()), ids=lambda t: t.template_id)
    def template(self, request):
        """Parametrize tests with all templates."""
        return request.param

    def test_output_format_not_empty(self, template):
        """Test that output format is not empty."""
        assert template.output_format
        assert len(template.output_format) > 0

    def test_output_format_has_placeholders(self, template):
        """Test that output format contains placeholders."""
        # Look for {placeholder} pattern
        placeholders = re.findall(r"\{(\w+)\}", template.output_format)
        assert len(placeholders) > 0, (
            f"Template '{template.template_id}' output format has no placeholders"
        )

    def test_output_format_uses_markdown(self, template):
        """Test that output format uses markdown headers."""
        # Should have at least one markdown header
        assert "#" in template.output_format


class TestRubricValidation:
    """Tests for rubric validation."""

    @pytest.fixture(params=list(TEMPLATES.values()), ids=lambda t: t.template_id)
    def template(self, request):
        """Parametrize tests with all templates."""
        return request.param

    def test_rubric_not_empty(self, template):
        """Test that rubric is not empty."""
        assert len(template.rubric) > 0

    def test_rubric_weights_are_positive(self, template):
        """Test that all rubric weights are positive."""
        for category, weight in template.rubric.items():
            assert weight > 0, f"Rubric weight for '{category}' is not positive"

    def test_rubric_weights_sum_to_one(self, template):
        """Test that rubric weights sum to approximately 1.0."""
        total = sum(template.rubric.values())
        assert abs(total - 1.0) < 0.01, (
            f"Template '{template.template_id}' rubric weights sum to {total}, not 1.0"
        )


class TestCodeReviewTemplate:
    """Tests specific to the CODE_REVIEW template."""

    def test_has_security_critic_role(self):
        """Test that code review has security critic role."""
        role_names = [r.name for r in CODE_REVIEW_TEMPLATE.roles]
        assert "security_critic" in role_names

    def test_has_performance_critic_role(self):
        """Test that code review has performance critic role."""
        role_names = [r.name for r in CODE_REVIEW_TEMPLATE.roles]
        assert "performance_critic" in role_names

    def test_has_maintainability_critic_role(self):
        """Test that code review has maintainability critic role."""
        role_names = [r.name for r in CODE_REVIEW_TEMPLATE.roles]
        assert "maintainability_critic" in role_names

    def test_has_author_role(self):
        """Test that code review has author role."""
        role_names = [r.name for r in CODE_REVIEW_TEMPLATE.roles]
        assert "author" in role_names

    def test_has_synthesizer_role(self):
        """Test that code review has synthesizer role."""
        role_names = [r.name for r in CODE_REVIEW_TEMPLATE.roles]
        assert "synthesizer" in role_names

    def test_rubric_includes_security(self):
        """Test that rubric includes security category."""
        assert "security_coverage" in CODE_REVIEW_TEMPLATE.rubric

    def test_rubric_includes_performance(self):
        """Test that rubric includes performance category."""
        assert "performance_impact" in CODE_REVIEW_TEMPLATE.rubric

    def test_output_format_includes_risk_score(self):
        """Test that output includes risk score."""
        assert "{risk_score}" in CODE_REVIEW_TEMPLATE.output_format

    def test_domain_is_software_engineering(self):
        """Test that domain is software_engineering."""
        assert CODE_REVIEW_TEMPLATE.domain == "software_engineering"

    def test_has_code_tag(self):
        """Test that template has 'code' tag."""
        assert "code" in CODE_REVIEW_TEMPLATE.tags


class TestSecurityAuditTemplate:
    """Tests specific to the SECURITY_AUDIT template."""

    def test_has_threat_modeler_role(self):
        """Test that security audit has threat modeler role."""
        role_names = [r.name for r in SECURITY_AUDIT_TEMPLATE.roles]
        assert "threat_modeler" in role_names

    def test_has_red_team_role(self):
        """Test that security audit has red team role."""
        role_names = [r.name for r in SECURITY_AUDIT_TEMPLATE.roles]
        assert "red_team" in role_names

    def test_has_blue_team_role(self):
        """Test that security audit has blue team role."""
        role_names = [r.name for r in SECURITY_AUDIT_TEMPLATE.roles]
        assert "blue_team" in role_names

    def test_high_consensus_threshold(self):
        """Test that security audit has high consensus threshold."""
        assert SECURITY_AUDIT_TEMPLATE.consensus_threshold >= 0.8

    def test_high_difficulty(self):
        """Test that security audit has high difficulty."""
        assert SECURITY_AUDIT_TEMPLATE.difficulty >= 0.8

    def test_output_includes_vulnerabilities(self):
        """Test that output includes vulnerability sections."""
        assert "{critical_vulnerabilities}" in SECURITY_AUDIT_TEMPLATE.output_format

    def test_domain_is_security(self):
        """Test that domain is security."""
        assert SECURITY_AUDIT_TEMPLATE.domain == "security"


class TestIncidentResponseTemplate:
    """Tests specific to the INCIDENT_RESPONSE template."""

    def test_has_investigator_role(self):
        """Test that incident response has investigator role."""
        role_names = [r.name for r in INCIDENT_RESPONSE_TEMPLATE.roles]
        assert "investigator" in role_names

    def test_has_responder_role(self):
        """Test that incident response has responder role."""
        role_names = [r.name for r in INCIDENT_RESPONSE_TEMPLATE.roles]
        assert "responder" in role_names

    def test_has_prevention_role(self):
        """Test that incident response has prevention role."""
        role_names = [r.name for r in INCIDENT_RESPONSE_TEMPLATE.roles]
        assert "prevention" in role_names

    def test_has_timeline_phase(self):
        """Test that incident response has timeline phase."""
        phase_names = [p.name for p in INCIDENT_RESPONSE_TEMPLATE.phases]
        assert "timeline" in phase_names

    def test_has_root_cause_phase(self):
        """Test that incident response has root cause phase."""
        phase_names = [p.name for p in INCIDENT_RESPONSE_TEMPLATE.phases]
        assert "root_cause" in phase_names

    def test_output_includes_timeline(self):
        """Test that output includes timeline section."""
        assert "{timeline}" in INCIDENT_RESPONSE_TEMPLATE.output_format

    def test_output_includes_action_items(self):
        """Test that output includes action items."""
        assert "{action_items}" in INCIDENT_RESPONSE_TEMPLATE.output_format


class TestHealthcareComplianceTemplate:
    """Tests specific to the HEALTHCARE_COMPLIANCE template."""

    def test_has_privacy_officer_role(self):
        """Test that healthcare compliance has privacy officer role."""
        role_names = [r.name for r in HEALTHCARE_COMPLIANCE_TEMPLATE.roles]
        assert "privacy_officer" in role_names

    def test_has_compliance_auditor_role(self):
        """Test that healthcare compliance has compliance auditor role."""
        role_names = [r.name for r in HEALTHCARE_COMPLIANCE_TEMPLATE.roles]
        assert "compliance_auditor" in role_names

    def test_has_breach_analyst_role(self):
        """Test that healthcare compliance has breach analyst role."""
        role_names = [r.name for r in HEALTHCARE_COMPLIANCE_TEMPLATE.roles]
        assert "breach_analyst" in role_names

    def test_high_consensus_threshold(self):
        """Test that healthcare compliance has high consensus threshold."""
        assert HEALTHCARE_COMPLIANCE_TEMPLATE.consensus_threshold >= 0.8

    def test_output_includes_phi_inventory(self):
        """Test that output includes PHI inventory."""
        assert "{phi_inventory}" in HEALTHCARE_COMPLIANCE_TEMPLATE.output_format

    def test_output_includes_safeguards(self):
        """Test that output includes safeguards sections."""
        assert "{administrative_safeguards}" in HEALTHCARE_COMPLIANCE_TEMPLATE.output_format
        assert "{technical_safeguards}" in HEALTHCARE_COMPLIANCE_TEMPLATE.output_format

    def test_domain_is_healthcare(self):
        """Test that domain is healthcare."""
        assert HEALTHCARE_COMPLIANCE_TEMPLATE.domain == "healthcare"

    def test_has_hipaa_tag(self):
        """Test that template has 'hipaa' tag."""
        assert "hipaa" in HEALTHCARE_COMPLIANCE_TEMPLATE.tags


class TestFinancialRiskTemplate:
    """Tests specific to the FINANCIAL_RISK template."""

    def test_has_quant_analyst_role(self):
        """Test that financial risk has quant analyst role."""
        role_names = [r.name for r in FINANCIAL_RISK_TEMPLATE.roles]
        assert "quant_analyst" in role_names

    def test_has_risk_manager_role(self):
        """Test that financial risk has risk manager role."""
        role_names = [r.name for r in FINANCIAL_RISK_TEMPLATE.roles]
        assert "risk_manager" in role_names

    def test_has_market_skeptic_role(self):
        """Test that financial risk has market skeptic role."""
        role_names = [r.name for r in FINANCIAL_RISK_TEMPLATE.roles]
        assert "market_skeptic" in role_names

    def test_has_stress_testing_phase(self):
        """Test that financial risk has stress testing phase."""
        phase_names = [p.name for p in FINANCIAL_RISK_TEMPLATE.phases]
        assert "stress_testing" in phase_names

    def test_output_includes_stress_test_results(self):
        """Test that output includes stress test results."""
        assert "{historical_scenarios}" in FINANCIAL_RISK_TEMPLATE.output_format

    def test_output_includes_risk_limits(self):
        """Test that output includes risk limits."""
        assert "{risk_limits}" in FINANCIAL_RISK_TEMPLATE.output_format

    def test_domain_is_finance(self):
        """Test that domain is finance."""
        assert FINANCIAL_RISK_TEMPLATE.domain == "finance"


class TestResearchSynthesisTemplate:
    """Tests specific to the RESEARCH_SYNTHESIS template."""

    def test_has_extractor_role(self):
        """Test that research synthesis has extractor role."""
        role_names = [r.name for r in RESEARCH_SYNTHESIS_TEMPLATE.roles]
        assert "extractor" in role_names

    def test_has_validator_role(self):
        """Test that research synthesis has validator role."""
        role_names = [r.name for r in RESEARCH_SYNTHESIS_TEMPLATE.roles]
        assert "validator" in role_names

    def test_has_synthesizer_role(self):
        """Test that research synthesis has synthesizer role."""
        role_names = [r.name for r in RESEARCH_SYNTHESIS_TEMPLATE.roles]
        assert "synthesizer" in role_names

    def test_has_extraction_phase(self):
        """Test that research synthesis has extraction phase."""
        phase_names = [p.name for p in RESEARCH_SYNTHESIS_TEMPLATE.phases]
        assert "extraction" in phase_names

    def test_has_validation_phase(self):
        """Test that research synthesis has validation phase."""
        phase_names = [p.name for p in RESEARCH_SYNTHESIS_TEMPLATE.phases]
        assert "validation" in phase_names

    def test_output_includes_sources(self):
        """Test that output includes sources."""
        assert "{sources}" in RESEARCH_SYNTHESIS_TEMPLATE.output_format

    def test_output_includes_consensus_claims(self):
        """Test that output includes consensus claims."""
        assert "{consensus}" in RESEARCH_SYNTHESIS_TEMPLATE.output_format


class TestArchitectureReviewTemplate:
    """Tests specific to the ARCHITECTURE_REVIEW template."""

    def test_has_architect_role(self):
        """Test that architecture review has architect role."""
        role_names = [r.name for r in ARCHITECTURE_REVIEW_TEMPLATE.roles]
        assert "architect" in role_names

    def test_has_scalability_reviewer_role(self):
        """Test that architecture review has scalability reviewer role."""
        role_names = [r.name for r in ARCHITECTURE_REVIEW_TEMPLATE.roles]
        assert "scalability_reviewer" in role_names

    def test_has_reliability_reviewer_role(self):
        """Test that architecture review has reliability reviewer role."""
        role_names = [r.name for r in ARCHITECTURE_REVIEW_TEMPLATE.roles]
        assert "reliability_reviewer" in role_names

    def test_has_operations_reviewer_role(self):
        """Test that architecture review has operations reviewer role."""
        role_names = [r.name for r in ARCHITECTURE_REVIEW_TEMPLATE.roles]
        assert "operations_reviewer" in role_names

    def test_output_includes_scores(self):
        """Test that output includes domain scores."""
        assert "{scalability_score}" in ARCHITECTURE_REVIEW_TEMPLATE.output_format
        assert "{reliability_score}" in ARCHITECTURE_REVIEW_TEMPLATE.output_format
        assert "{security_score}" in ARCHITECTURE_REVIEW_TEMPLATE.output_format

    def test_domain_is_architecture(self):
        """Test that domain is architecture."""
        assert ARCHITECTURE_REVIEW_TEMPLATE.domain == "architecture"


class TestDesignDocTemplate:
    """Tests specific to the DESIGN_DOC template."""

    def test_has_devils_advocate_role(self):
        """Test that design doc has devil's advocate role."""
        role_names = [r.name for r in DESIGN_DOC_TEMPLATE.roles]
        assert "devils_advocate" in role_names

    def test_has_stakeholder_role(self):
        """Test that design doc has stakeholder role."""
        role_names = [r.name for r in DESIGN_DOC_TEMPLATE.roles]
        assert "stakeholder" in role_names

    def test_has_implementer_role(self):
        """Test that design doc has implementer role."""
        role_names = [r.name for r in DESIGN_DOC_TEMPLATE.roles]
        assert "implementer" in role_names

    def test_has_presentation_phase(self):
        """Test that design doc has presentation phase."""
        phase_names = [p.name for p in DESIGN_DOC_TEMPLATE.phases]
        assert "presentation" in phase_names

    def test_output_includes_decision(self):
        """Test that output includes decision."""
        assert "{decision}" in DESIGN_DOC_TEMPLATE.output_format

    def test_output_includes_risks(self):
        """Test that output includes risks."""
        assert "{risks}" in DESIGN_DOC_TEMPLATE.output_format

    def test_domain_is_architecture(self):
        """Test that domain is architecture."""
        assert DESIGN_DOC_TEMPLATE.domain == "architecture"


class TestTemplateIdUniqueness:
    """Tests for template ID uniqueness across all templates."""

    def test_all_template_ids_unique(self):
        """Test that all template IDs are unique."""
        ids = [t.template_id for t in TEMPLATES.values()]
        assert len(ids) == len(set(ids))

    def test_template_ids_follow_naming_convention(self):
        """Test that template IDs follow naming convention."""
        for template in TEMPLATES.values():
            # Should be lowercase with hyphens
            assert template.template_id == template.template_id.lower()
            # Should contain version suffix
            assert "-v" in template.template_id


class TestTemplateTagConsistency:
    """Tests for template tag consistency."""

    @pytest.fixture(params=list(TEMPLATES.values()), ids=lambda t: t.template_id)
    def template(self, request):
        """Parametrize tests with all templates."""
        return request.param

    def test_tags_are_lowercase(self, template):
        """Test that all tags are lowercase."""
        for tag in template.tags:
            assert tag == tag.lower()

    def test_tags_have_no_spaces(self, template):
        """Test that tags have no spaces."""
        for tag in template.tags:
            assert " " not in tag
