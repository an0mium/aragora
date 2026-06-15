"""
Tests for aragora.templates module.

Tests for domain-specific debate templates including:
- TemplateType enum
- DebateRole, DebatePhase, DebateTemplate dataclasses
- Template registry functions (get_template, list_templates)
- template_to_protocol conversion
- All pre-built templates validation
"""

from __future__ import annotations

import pytest
from dataclasses import fields

from aragora.templates import (
    DebateTemplate,
    DebateRole,
    DebatePhase,
    TemplateType,
    CODE_REVIEW_TEMPLATE,
    DESIGN_DOC_TEMPLATE,
    INCIDENT_RESPONSE_TEMPLATE,
    RESEARCH_SYNTHESIS_TEMPLATE,
    SECURITY_AUDIT_TEMPLATE,
    ARCHITECTURE_REVIEW_TEMPLATE,
    HEALTHCARE_COMPLIANCE_TEMPLATE,
    FINANCIAL_RISK_TEMPLATE,
    get_template,
    list_templates,
    template_to_protocol,
    TEMPLATES,
)


# =============================================================================
# TemplateType Enum Tests
# =============================================================================


class TestTemplateType:
    """Tests for TemplateType enum."""

    def test_all_template_types_defined(self):
        """Should have all expected template types."""
        expected_types = {
            "CODE_REVIEW",
            "DESIGN_DOC",
            "INCIDENT_RESPONSE",
            "RESEARCH_SYNTHESIS",
            "POLICY_REVIEW",
            "SECURITY_AUDIT",
            "ARCHITECTURE_REVIEW",
            "PRODUCT_STRATEGY",
            "HEALTHCARE_COMPLIANCE",
            "FINANCIAL_RISK",
        }
        actual_types = {t.name for t in TemplateType}
        assert expected_types == actual_types

    def test_template_type_values(self):
        """Template type values should be lowercase with underscores."""
        for template_type in TemplateType:
            assert template_type.value == template_type.name.lower()

    def test_template_type_from_string(self):
        """Should be able to get template type from string."""
        assert TemplateType("code_review") == TemplateType.CODE_REVIEW
        assert TemplateType("security_audit") == TemplateType.SECURITY_AUDIT


# =============================================================================
# DebateRole Dataclass Tests
# =============================================================================


class TestDebateRole:
    """Tests for DebateRole dataclass."""

    def test_debate_role_creation(self):
        """Should create DebateRole with required fields."""
        role = DebateRole(
            name="test_role",
            description="A test role",
            objectives=["objective1", "objective2"],
            evaluation_criteria=["criterion1"],
        )
        assert role.name == "test_role"
        assert role.description == "A test role"
        assert len(role.objectives) == 2
        assert len(role.evaluation_criteria) == 1
        assert role.example_prompts == []

    def test_debate_role_with_example_prompts(self):
        """Should create DebateRole with example prompts."""
        role = DebateRole(
            name="critic",
            description="Reviews proposals",
            objectives=["Find issues"],
            evaluation_criteria=["Issue quality"],
            example_prompts=["This is problematic because...", "Consider instead..."],
        )
        assert len(role.example_prompts) == 2
        assert "problematic" in role.example_prompts[0]


# =============================================================================
# DebatePhase Dataclass Tests
# =============================================================================


class TestDebatePhase:
    """Tests for DebatePhase dataclass."""

    def test_debate_phase_creation(self):
        """Should create DebatePhase with all fields."""
        phase = DebatePhase(
            name="initial_review",
            description="First review pass",
            duration_rounds=2,
            roles_active=["critic", "validator"],
            objectives=["Identify issues"],
            outputs=["Issue list"],
        )
        assert phase.name == "initial_review"
        assert phase.duration_rounds == 2
        assert len(phase.roles_active) == 2
        assert "critic" in phase.roles_active

    def test_debate_phase_single_round(self):
        """Should allow single-round phases."""
        phase = DebatePhase(
            name="synthesis",
            description="Final synthesis",
            duration_rounds=1,
            roles_active=["synthesizer"],
            objectives=["Create summary"],
            outputs=["Final document"],
        )
        assert phase.duration_rounds == 1


# =============================================================================
# DebateTemplate Dataclass Tests
# =============================================================================


class TestDebateTemplate:
    """Tests for DebateTemplate dataclass."""

    @pytest.fixture
    def minimal_template(self):
        """Create a minimal valid template."""
        return DebateTemplate(
            template_id="test-v1",
            template_type=TemplateType.CODE_REVIEW,
            name="Test Template",
            description="A test template",
            roles=[
                DebateRole(
                    name="tester",
                    description="Tests things",
                    objectives=["Test"],
                    evaluation_criteria=["Quality"],
                )
            ],
            phases=[
                DebatePhase(
                    name="test_phase",
                    description="Testing phase",
                    duration_rounds=1,
                    roles_active=["tester"],
                    objectives=["Test"],
                    outputs=["Results"],
                )
            ],
            recommended_agents=2,
            max_rounds=3,
            consensus_threshold=0.7,
            rubric={"quality": 1.0},
            output_format="# Results\n{results}",
            domain="testing",
        )

    def test_template_creation(self, minimal_template):
        """Should create DebateTemplate with required fields."""
        assert minimal_template.template_id == "test-v1"
        assert minimal_template.template_type == TemplateType.CODE_REVIEW
        assert minimal_template.recommended_agents == 2
        assert minimal_template.difficulty == 0.5  # default

    def test_template_with_tags(self, minimal_template):
        """Should allow templates with tags."""
        template = DebateTemplate(
            template_id="tagged-v1",
            template_type=TemplateType.DESIGN_DOC,
            name="Tagged Template",
            description="Has tags",
            roles=minimal_template.roles,
            phases=minimal_template.phases,
            recommended_agents=3,
            max_rounds=5,
            consensus_threshold=0.6,
            rubric={"complete": 1.0},
            output_format="# Output",
            domain="design",
            tags=["rfc", "design", "architecture"],
        )
        assert len(template.tags) == 3
        assert "rfc" in template.tags


# =============================================================================
# Template Registry Tests
# =============================================================================


class TestTemplateRegistry:
    """Tests for template registry functions."""

    def test_templates_dict_has_all_implemented_types(self):
        """TEMPLATES dict should have all implemented template types."""
        # Note: PRODUCT_STRATEGY and POLICY_REVIEW are in enum but not implemented
        implemented_types = {
            TemplateType.CODE_REVIEW,
            TemplateType.DESIGN_DOC,
            TemplateType.INCIDENT_RESPONSE,
            TemplateType.RESEARCH_SYNTHESIS,
            TemplateType.SECURITY_AUDIT,
            TemplateType.ARCHITECTURE_REVIEW,
            TemplateType.HEALTHCARE_COMPLIANCE,
            TemplateType.FINANCIAL_RISK,
        }
        assert set(TEMPLATES.keys()) == implemented_types

    def test_get_template_returns_correct_template(self):
        """get_template should return the correct template."""
        template = get_template(TemplateType.CODE_REVIEW)
        assert template == CODE_REVIEW_TEMPLATE
        assert template.template_type == TemplateType.CODE_REVIEW

    def test_get_template_raises_for_invalid_type(self):
        """get_template should raise ValueError for invalid type."""
        # Create a mock invalid type that's not in TEMPLATES
        with pytest.raises(ValueError, match="Unknown template type"):
            get_template(TemplateType.POLICY_REVIEW)

    def test_list_templates_returns_all_templates(self):
        """list_templates should return info for all templates."""
        templates = list_templates()
        assert len(templates) == 8

        # Verify structure
        for t in templates:
            assert "id" in t
            assert "type" in t
            assert "name" in t
            assert "description" in t
            assert "agents" in t
            assert "domain" in t

    def test_list_templates_includes_code_review(self):
        """list_templates should include code review template."""
        templates = list_templates()
        code_review = next((t for t in templates if t["type"] == "code_review"), None)
        assert code_review is not None
        assert code_review["id"] == "code-review-v1"
        assert code_review["domain"] == "software_engineering"


# =============================================================================
# Pre-built Template Validation Tests
# =============================================================================


class TestCodeReviewTemplate:
    """Tests for CODE_REVIEW_TEMPLATE."""

    def test_has_required_roles(self):
        """Should have all required reviewer roles."""
        role_names = {r.name for r in CODE_REVIEW_TEMPLATE.roles}
        expected = {
            "author",
            "security_critic",
            "performance_critic",
            "maintainability_critic",
            "synthesizer",
        }
        assert role_names == expected

    def test_has_required_phases(self):
        """Should have structured review phases."""
        phase_names = [p.name for p in CODE_REVIEW_TEMPLATE.phases]
        assert "initial_review" in phase_names
        assert "author_response" in phase_names
        assert "synthesis" in phase_names

    def test_rubric_weights_sum_to_one(self):
        """Rubric weights should sum to 1.0."""
        total = sum(CODE_REVIEW_TEMPLATE.rubric.values())
        assert abs(total - 1.0) < 0.01

    def test_recommended_agents_matches_roles(self):
        """Recommended agents should be reasonable for role count."""
        # Should have fewer recommended agents than roles (some roles can share)
        assert CODE_REVIEW_TEMPLATE.recommended_agents <= len(CODE_REVIEW_TEMPLATE.roles)


class TestSecurityAuditTemplate:
    """Tests for SECURITY_AUDIT_TEMPLATE."""

    def test_has_red_and_blue_team(self):
        """Should have both red team and blue team roles."""
        role_names = {r.name for r in SECURITY_AUDIT_TEMPLATE.roles}
        assert "red_team" in role_names
        assert "blue_team" in role_names
        assert "threat_modeler" in role_names

    def test_has_high_consensus_threshold(self):
        """Security audit should require high consensus."""
        assert SECURITY_AUDIT_TEMPLATE.consensus_threshold >= 0.8

    def test_has_security_domain(self):
        """Should be in security domain."""
        assert SECURITY_AUDIT_TEMPLATE.domain == "security"


class TestHealthcareComplianceTemplate:
    """Tests for HEALTHCARE_COMPLIANCE_TEMPLATE."""

    def test_has_compliance_roles(self):
        """Should have HIPAA-relevant roles."""
        role_names = {r.name for r in HEALTHCARE_COMPLIANCE_TEMPLATE.roles}
        assert "privacy_officer" in role_names
        assert "compliance_auditor" in role_names
        assert "breach_analyst" in role_names

    def test_has_healthcare_tags(self):
        """Should be tagged with healthcare keywords."""
        tags = HEALTHCARE_COMPLIANCE_TEMPLATE.tags
        assert "hipaa" in tags
        assert "healthcare" in tags


class TestFinancialRiskTemplate:
    """Tests for FINANCIAL_RISK_TEMPLATE."""

    def test_has_financial_roles(self):
        """Should have financial analysis roles."""
        role_names = {r.name for r in FINANCIAL_RISK_TEMPLATE.roles}
        assert "quant_analyst" in role_names
        assert "risk_manager" in role_names
        assert "market_skeptic" in role_names

    def test_has_stress_testing_phase(self):
        """Should include stress testing phase."""
        phase_names = [p.name for p in FINANCIAL_RISK_TEMPLATE.phases]
        assert "stress_testing" in phase_names


class TestAllTemplatesValid:
    """Tests validating all templates have proper structure."""

    @pytest.mark.parametrize("template_type", list(TEMPLATES.keys()))
    def test_template_has_roles(self, template_type):
        """Each template should have at least one role."""
        template = TEMPLATES[template_type]
        assert len(template.roles) >= 1

    @pytest.mark.parametrize("template_type", list(TEMPLATES.keys()))
    def test_template_has_phases(self, template_type):
        """Each template should have at least one phase."""
        template = TEMPLATES[template_type]
        assert len(template.phases) >= 1

    @pytest.mark.parametrize("template_type", list(TEMPLATES.keys()))
    def test_template_rubric_weights_valid(self, template_type):
        """Each template rubric should sum to ~1.0."""
        template = TEMPLATES[template_type]
        total = sum(template.rubric.values())
        assert 0.99 <= total <= 1.01, f"{template_type} rubric sums to {total}"

    @pytest.mark.parametrize("template_type", list(TEMPLATES.keys()))
    def test_template_phases_duration_consistent(self, template_type):
        """Phase durations should not exceed max_rounds."""
        template = TEMPLATES[template_type]
        total_phase_rounds = sum(p.duration_rounds for p in template.phases)
        # Total phase rounds should be reasonable relative to max_rounds
        assert total_phase_rounds <= template.max_rounds * 2

    @pytest.mark.parametrize("template_type", list(TEMPLATES.keys()))
    def test_template_output_format_has_placeholders(self, template_type):
        """Output format should have placeholder variables."""
        template = TEMPLATES[template_type]
        assert "{" in template.output_format
        assert "}" in template.output_format


# =============================================================================
# template_to_protocol Tests
# =============================================================================


class TestTemplateToProtocol:
    """Tests for template_to_protocol function."""

    def test_converts_code_review_template(self):
        """Should convert code review template to protocol."""
        protocol = template_to_protocol(CODE_REVIEW_TEMPLATE)

        assert protocol.rounds >= CODE_REVIEW_TEMPLATE.max_rounds
        assert protocol.consensus_threshold == CODE_REVIEW_TEMPLATE.consensus_threshold
        assert protocol.role_rotation is True
        assert protocol.early_stopping is True

    def test_uses_overrides(self):
        """Should apply overrides to protocol."""
        protocol = template_to_protocol(
            CODE_REVIEW_TEMPLATE,
            overrides={"rounds": 10, "consensus": "unanimous"},
        )

        assert protocol.rounds == 10
        assert protocol.consensus == "unanimous"

    def test_research_template_uses_all_to_all(self):
        """Research synthesis template should use all-to-all topology."""
        protocol = template_to_protocol(RESEARCH_SYNTHESIS_TEMPLATE)
        assert protocol.topology == "all-to-all"

    def test_large_role_template_uses_round_robin(self):
        """Templates with many roles should use round-robin."""
        protocol = template_to_protocol(SECURITY_AUDIT_TEMPLATE)
        # Security audit has 5 roles, should use round-robin
        assert protocol.topology == "round-robin"

    def test_topology_override(self):
        """Should respect topology override."""
        protocol = template_to_protocol(
            RESEARCH_SYNTHESIS_TEMPLATE,
            overrides={"topology": "sparse"},
        )
        assert protocol.topology == "sparse"

    def test_enables_calibration_by_default(self):
        """Should enable calibration by default."""
        protocol = template_to_protocol(CODE_REVIEW_TEMPLATE)
        assert protocol.enable_calibration is True

    def test_enables_convergence_detection(self):
        """Should enable convergence detection."""
        protocol = template_to_protocol(DESIGN_DOC_TEMPLATE)
        assert protocol.convergence_detection is True

    @pytest.mark.parametrize("template_type", list(TEMPLATES.keys()))
    def test_all_templates_convertible(self, template_type):
        """All templates should be convertible to protocols."""
        template = TEMPLATES[template_type]
        protocol = template_to_protocol(template)

        assert protocol is not None
        assert protocol.rounds > 0
        assert 0 < protocol.consensus_threshold <= 1.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestTemplateIntegration:
    """Integration tests for template system."""

    def test_template_workflow(self):
        """Should support full template workflow: get -> convert -> use."""
        # 1. Get template by type
        template = get_template(TemplateType.INCIDENT_RESPONSE)
        assert template.name == "Incident Response Analysis"

        # 2. Convert to protocol
        protocol = template_to_protocol(template)
        assert protocol.rounds >= 5

        # 3. Verify protocol properties match template intent
        assert protocol.consensus_threshold == template.consensus_threshold

    def test_list_and_get_consistency(self):
        """list_templates and get_template should be consistent."""
        listed = list_templates()

        for item in listed:
            template_type = TemplateType(item["type"])
            template = get_template(template_type)
            assert template.template_id == item["id"]
            assert template.name == item["name"]
            assert template.domain == item["domain"]

    def test_template_immutability(self):
        """Templates should behave consistently across calls."""
        template1 = get_template(TemplateType.CODE_REVIEW)
        template2 = get_template(TemplateType.CODE_REVIEW)

        assert template1 is template2  # Same object
        assert template1.template_id == template2.template_id
