"""
Tests for the template registry functions.

Tests cover:
- get_template function
- list_templates function
- TEMPLATES dictionary
- Error handling for invalid template types
"""

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
    get_template,
    list_templates,
)


class TestGetTemplate:
    """Tests for the get_template function."""

    def test_get_code_review_template(self):
        """Test getting the CODE_REVIEW template."""
        template = get_template(TemplateType.CODE_REVIEW)

        assert template is CODE_REVIEW_TEMPLATE
        assert template.template_type == TemplateType.CODE_REVIEW
        assert template.template_id == "code-review-v1"

    def test_get_design_doc_template(self):
        """Test getting the DESIGN_DOC template."""
        template = get_template(TemplateType.DESIGN_DOC)

        assert template is DESIGN_DOC_TEMPLATE
        assert template.template_type == TemplateType.DESIGN_DOC
        assert template.template_id == "design-doc-v1"

    def test_get_incident_response_template(self):
        """Test getting the INCIDENT_RESPONSE template."""
        template = get_template(TemplateType.INCIDENT_RESPONSE)

        assert template is INCIDENT_RESPONSE_TEMPLATE
        assert template.template_type == TemplateType.INCIDENT_RESPONSE
        assert template.template_id == "incident-response-v1"

    def test_get_research_synthesis_template(self):
        """Test getting the RESEARCH_SYNTHESIS template."""
        template = get_template(TemplateType.RESEARCH_SYNTHESIS)

        assert template is RESEARCH_SYNTHESIS_TEMPLATE
        assert template.template_type == TemplateType.RESEARCH_SYNTHESIS
        assert template.template_id == "research-synthesis-v1"

    def test_get_security_audit_template(self):
        """Test getting the SECURITY_AUDIT template."""
        template = get_template(TemplateType.SECURITY_AUDIT)

        assert template is SECURITY_AUDIT_TEMPLATE
        assert template.template_type == TemplateType.SECURITY_AUDIT
        assert template.template_id == "security-audit-v1"

    def test_get_architecture_review_template(self):
        """Test getting the ARCHITECTURE_REVIEW template."""
        template = get_template(TemplateType.ARCHITECTURE_REVIEW)

        assert template is ARCHITECTURE_REVIEW_TEMPLATE
        assert template.template_type == TemplateType.ARCHITECTURE_REVIEW
        assert template.template_id == "architecture-review-v1"

    def test_get_healthcare_compliance_template(self):
        """Test getting the HEALTHCARE_COMPLIANCE template."""
        template = get_template(TemplateType.HEALTHCARE_COMPLIANCE)

        assert template is HEALTHCARE_COMPLIANCE_TEMPLATE
        assert template.template_type == TemplateType.HEALTHCARE_COMPLIANCE
        assert template.template_id == "healthcare-compliance-v1"

    def test_get_financial_risk_template(self):
        """Test getting the FINANCIAL_RISK template."""
        template = get_template(TemplateType.FINANCIAL_RISK)

        assert template is FINANCIAL_RISK_TEMPLATE
        assert template.template_type == TemplateType.FINANCIAL_RISK
        assert template.template_id == "financial-risk-v1"

    def test_get_template_returns_debate_template(self):
        """Test that get_template returns DebateTemplate instances."""
        for template_type in TemplateType:
            if template_type in TEMPLATES:
                template = get_template(template_type)
                assert isinstance(template, DebateTemplate)

    def test_get_template_invalid_type_raises_error(self):
        """Test that get_template raises ValueError for invalid types."""
        # Create a mock invalid type - using a value not in TEMPLATES
        # Note: TemplateType.POLICY_REVIEW and PRODUCT_STRATEGY are defined
        # but may not have templates in the registry
        for template_type in TemplateType:
            if template_type not in TEMPLATES:
                with pytest.raises(ValueError, match="Unknown template type"):
                    get_template(template_type)


class TestListTemplates:
    """Tests for the list_templates function."""

    def test_list_templates_returns_list(self):
        """Test that list_templates returns a list."""
        templates = list_templates()
        assert isinstance(templates, list)

    def test_list_templates_not_empty(self):
        """Test that list_templates returns non-empty list."""
        templates = list_templates()
        assert len(templates) > 0

    def test_list_templates_count_matches_registry(self):
        """Test that list_templates count matches TEMPLATES."""
        templates = list_templates()
        assert len(templates) == len(TEMPLATES)

    def test_list_templates_entry_structure(self):
        """Test that list_templates entries have expected structure."""
        templates = list_templates()

        for entry in templates:
            assert isinstance(entry, dict)
            assert "id" in entry
            assert "type" in entry
            assert "name" in entry
            assert "description" in entry
            assert "agents" in entry
            assert "domain" in entry

    def test_list_templates_id_format(self):
        """Test that template IDs have expected format."""
        templates = list_templates()

        for entry in templates:
            # IDs should be lowercase with hyphens
            assert (
                entry["id"].replace("-", "").replace("v", "").replace("1", "").isalpha()
                or "-v" in entry["id"]
            )

    def test_list_templates_type_matches_value(self):
        """Test that type field uses enum value."""
        templates = list_templates()

        for entry in templates:
            # Type should be the enum value (lowercase with underscores)
            assert "_" in entry["type"] or entry["type"].islower()

    def test_list_templates_agents_is_positive(self):
        """Test that recommended agents count is positive."""
        templates = list_templates()

        for entry in templates:
            assert entry["agents"] > 0

    def test_list_templates_has_code_review(self):
        """Test that code_review template is listed."""
        templates = list_templates()
        ids = [t["id"] for t in templates]

        assert "code-review-v1" in ids

    def test_list_templates_has_all_registered(self):
        """Test that all registered templates are listed."""
        templates = list_templates()
        ids = set(t["id"] for t in templates)

        for template in TEMPLATES.values():
            assert template.template_id in ids


class TestTemplatesRegistry:
    """Tests for the TEMPLATES dictionary."""

    def test_templates_is_dict(self):
        """Test that TEMPLATES is a dictionary."""
        assert isinstance(TEMPLATES, dict)

    def test_templates_not_empty(self):
        """Test that TEMPLATES is not empty."""
        assert len(TEMPLATES) > 0

    def test_templates_keys_are_template_types(self):
        """Test that TEMPLATES keys are TemplateType enums."""
        for key in TEMPLATES.keys():
            assert isinstance(key, TemplateType)

    def test_templates_values_are_debate_templates(self):
        """Test that TEMPLATES values are DebateTemplate instances."""
        for value in TEMPLATES.values():
            assert isinstance(value, DebateTemplate)

    def test_templates_key_matches_value_type(self):
        """Test that each key matches its template's type."""
        for template_type, template in TEMPLATES.items():
            assert template.template_type == template_type

    def test_templates_contains_code_review(self):
        """Test that TEMPLATES contains CODE_REVIEW."""
        assert TemplateType.CODE_REVIEW in TEMPLATES

    def test_templates_contains_design_doc(self):
        """Test that TEMPLATES contains DESIGN_DOC."""
        assert TemplateType.DESIGN_DOC in TEMPLATES

    def test_templates_contains_incident_response(self):
        """Test that TEMPLATES contains INCIDENT_RESPONSE."""
        assert TemplateType.INCIDENT_RESPONSE in TEMPLATES

    def test_templates_contains_research_synthesis(self):
        """Test that TEMPLATES contains RESEARCH_SYNTHESIS."""
        assert TemplateType.RESEARCH_SYNTHESIS in TEMPLATES

    def test_templates_contains_security_audit(self):
        """Test that TEMPLATES contains SECURITY_AUDIT."""
        assert TemplateType.SECURITY_AUDIT in TEMPLATES

    def test_templates_contains_architecture_review(self):
        """Test that TEMPLATES contains ARCHITECTURE_REVIEW."""
        assert TemplateType.ARCHITECTURE_REVIEW in TEMPLATES

    def test_templates_contains_healthcare_compliance(self):
        """Test that TEMPLATES contains HEALTHCARE_COMPLIANCE."""
        assert TemplateType.HEALTHCARE_COMPLIANCE in TEMPLATES

    def test_templates_contains_financial_risk(self):
        """Test that TEMPLATES contains FINANCIAL_RISK."""
        assert TemplateType.FINANCIAL_RISK in TEMPLATES


class TestTemplateExports:
    """Tests for module-level template exports."""

    def test_code_review_template_exported(self):
        """Test that CODE_REVIEW_TEMPLATE is exported."""
        assert CODE_REVIEW_TEMPLATE is not None
        assert CODE_REVIEW_TEMPLATE.template_type == TemplateType.CODE_REVIEW

    def test_design_doc_template_exported(self):
        """Test that DESIGN_DOC_TEMPLATE is exported."""
        assert DESIGN_DOC_TEMPLATE is not None
        assert DESIGN_DOC_TEMPLATE.template_type == TemplateType.DESIGN_DOC

    def test_incident_response_template_exported(self):
        """Test that INCIDENT_RESPONSE_TEMPLATE is exported."""
        assert INCIDENT_RESPONSE_TEMPLATE is not None
        assert INCIDENT_RESPONSE_TEMPLATE.template_type == TemplateType.INCIDENT_RESPONSE

    def test_research_synthesis_template_exported(self):
        """Test that RESEARCH_SYNTHESIS_TEMPLATE is exported."""
        assert RESEARCH_SYNTHESIS_TEMPLATE is not None
        assert RESEARCH_SYNTHESIS_TEMPLATE.template_type == TemplateType.RESEARCH_SYNTHESIS

    def test_security_audit_template_exported(self):
        """Test that SECURITY_AUDIT_TEMPLATE is exported."""
        assert SECURITY_AUDIT_TEMPLATE is not None
        assert SECURITY_AUDIT_TEMPLATE.template_type == TemplateType.SECURITY_AUDIT

    def test_architecture_review_template_exported(self):
        """Test that ARCHITECTURE_REVIEW_TEMPLATE is exported."""
        assert ARCHITECTURE_REVIEW_TEMPLATE is not None
        assert ARCHITECTURE_REVIEW_TEMPLATE.template_type == TemplateType.ARCHITECTURE_REVIEW

    def test_healthcare_compliance_template_exported(self):
        """Test that HEALTHCARE_COMPLIANCE_TEMPLATE is exported."""
        assert HEALTHCARE_COMPLIANCE_TEMPLATE is not None
        assert HEALTHCARE_COMPLIANCE_TEMPLATE.template_type == TemplateType.HEALTHCARE_COMPLIANCE

    def test_financial_risk_template_exported(self):
        """Test that FINANCIAL_RISK_TEMPLATE is exported."""
        assert FINANCIAL_RISK_TEMPLATE is not None
        assert FINANCIAL_RISK_TEMPLATE.template_type == TemplateType.FINANCIAL_RISK

    def test_exported_templates_match_registry(self):
        """Test that exported templates match registry entries."""
        assert TEMPLATES[TemplateType.CODE_REVIEW] is CODE_REVIEW_TEMPLATE
        assert TEMPLATES[TemplateType.DESIGN_DOC] is DESIGN_DOC_TEMPLATE
        assert TEMPLATES[TemplateType.INCIDENT_RESPONSE] is INCIDENT_RESPONSE_TEMPLATE
        assert TEMPLATES[TemplateType.RESEARCH_SYNTHESIS] is RESEARCH_SYNTHESIS_TEMPLATE
        assert TEMPLATES[TemplateType.SECURITY_AUDIT] is SECURITY_AUDIT_TEMPLATE
        assert TEMPLATES[TemplateType.ARCHITECTURE_REVIEW] is ARCHITECTURE_REVIEW_TEMPLATE
        assert TEMPLATES[TemplateType.HEALTHCARE_COMPLIANCE] is HEALTHCARE_COMPLIANCE_TEMPLATE
        assert TEMPLATES[TemplateType.FINANCIAL_RISK] is FINANCIAL_RISK_TEMPLATE
