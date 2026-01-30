"""
Tests for the TemplateType enum.

Tests cover:
- Enum values
- Enum member access
- Enum iteration
- String representation
"""

import pytest

from aragora.templates import TemplateType


class TestTemplateTypeValues:
    """Tests for TemplateType enum values."""

    def test_code_review_value(self):
        """Test CODE_REVIEW enum value."""
        assert TemplateType.CODE_REVIEW.value == "code_review"

    def test_design_doc_value(self):
        """Test DESIGN_DOC enum value."""
        assert TemplateType.DESIGN_DOC.value == "design_doc"

    def test_incident_response_value(self):
        """Test INCIDENT_RESPONSE enum value."""
        assert TemplateType.INCIDENT_RESPONSE.value == "incident_response"

    def test_research_synthesis_value(self):
        """Test RESEARCH_SYNTHESIS enum value."""
        assert TemplateType.RESEARCH_SYNTHESIS.value == "research_synthesis"

    def test_policy_review_value(self):
        """Test POLICY_REVIEW enum value."""
        assert TemplateType.POLICY_REVIEW.value == "policy_review"

    def test_security_audit_value(self):
        """Test SECURITY_AUDIT enum value."""
        assert TemplateType.SECURITY_AUDIT.value == "security_audit"

    def test_architecture_review_value(self):
        """Test ARCHITECTURE_REVIEW enum value."""
        assert TemplateType.ARCHITECTURE_REVIEW.value == "architecture_review"

    def test_product_strategy_value(self):
        """Test PRODUCT_STRATEGY enum value."""
        assert TemplateType.PRODUCT_STRATEGY.value == "product_strategy"

    def test_healthcare_compliance_value(self):
        """Test HEALTHCARE_COMPLIANCE enum value."""
        assert TemplateType.HEALTHCARE_COMPLIANCE.value == "healthcare_compliance"

    def test_financial_risk_value(self):
        """Test FINANCIAL_RISK enum value."""
        assert TemplateType.FINANCIAL_RISK.value == "financial_risk"


class TestTemplateTypeEnumeration:
    """Tests for TemplateType enum enumeration."""

    def test_total_template_types(self):
        """Test the total number of template types."""
        template_types = list(TemplateType)
        assert len(template_types) == 10

    def test_iteration_yields_all_types(self):
        """Test that iteration yields all template types."""
        types_list = list(TemplateType)

        assert TemplateType.CODE_REVIEW in types_list
        assert TemplateType.DESIGN_DOC in types_list
        assert TemplateType.INCIDENT_RESPONSE in types_list
        assert TemplateType.RESEARCH_SYNTHESIS in types_list
        assert TemplateType.POLICY_REVIEW in types_list
        assert TemplateType.SECURITY_AUDIT in types_list
        assert TemplateType.ARCHITECTURE_REVIEW in types_list
        assert TemplateType.PRODUCT_STRATEGY in types_list
        assert TemplateType.HEALTHCARE_COMPLIANCE in types_list
        assert TemplateType.FINANCIAL_RISK in types_list


class TestTemplateTypeAccess:
    """Tests for TemplateType enum member access."""

    def test_access_by_name(self):
        """Test accessing enum by name."""
        assert TemplateType["CODE_REVIEW"] == TemplateType.CODE_REVIEW
        assert TemplateType["DESIGN_DOC"] == TemplateType.DESIGN_DOC

    def test_access_by_value(self):
        """Test accessing enum by value."""
        assert TemplateType("code_review") == TemplateType.CODE_REVIEW
        assert TemplateType("design_doc") == TemplateType.DESIGN_DOC

    def test_invalid_name_raises_error(self):
        """Test that invalid name raises KeyError."""
        with pytest.raises(KeyError):
            TemplateType["INVALID_TYPE"]

    def test_invalid_value_raises_error(self):
        """Test that invalid value raises ValueError."""
        with pytest.raises(ValueError):
            TemplateType("invalid_type")


class TestTemplateTypeRepresentation:
    """Tests for TemplateType enum string representation."""

    def test_str_representation(self):
        """Test string representation of enum."""
        assert str(TemplateType.CODE_REVIEW) == "TemplateType.CODE_REVIEW"

    def test_name_property(self):
        """Test name property of enum members."""
        assert TemplateType.CODE_REVIEW.name == "CODE_REVIEW"
        assert TemplateType.DESIGN_DOC.name == "DESIGN_DOC"
        assert TemplateType.INCIDENT_RESPONSE.name == "INCIDENT_RESPONSE"

    def test_value_property(self):
        """Test value property of enum members."""
        assert TemplateType.CODE_REVIEW.value == "code_review"
        assert TemplateType.DESIGN_DOC.value == "design_doc"


class TestTemplateTypeComparison:
    """Tests for TemplateType enum comparison."""

    def test_equality(self):
        """Test enum equality."""
        assert TemplateType.CODE_REVIEW == TemplateType.CODE_REVIEW
        assert TemplateType.DESIGN_DOC == TemplateType.DESIGN_DOC

    def test_inequality(self):
        """Test enum inequality."""
        assert TemplateType.CODE_REVIEW != TemplateType.DESIGN_DOC
        assert TemplateType.SECURITY_AUDIT != TemplateType.POLICY_REVIEW

    def test_identity(self):
        """Test enum identity."""
        assert TemplateType.CODE_REVIEW is TemplateType.CODE_REVIEW
        assert TemplateType.DESIGN_DOC is TemplateType.DESIGN_DOC


class TestTemplateTypeCategories:
    """Tests for logical categorization of template types."""

    def test_software_engineering_types(self):
        """Test that software engineering types exist."""
        software_types = {
            TemplateType.CODE_REVIEW,
            TemplateType.DESIGN_DOC,
            TemplateType.ARCHITECTURE_REVIEW,
        }

        for t in software_types:
            assert isinstance(t, TemplateType)

    def test_operations_types(self):
        """Test that operations types exist."""
        ops_types = {
            TemplateType.INCIDENT_RESPONSE,
            TemplateType.SECURITY_AUDIT,
        }

        for t in ops_types:
            assert isinstance(t, TemplateType)

    def test_business_types(self):
        """Test that business/strategy types exist."""
        business_types = {
            TemplateType.PRODUCT_STRATEGY,
            TemplateType.POLICY_REVIEW,
        }

        for t in business_types:
            assert isinstance(t, TemplateType)

    def test_compliance_types(self):
        """Test that compliance types exist."""
        compliance_types = {
            TemplateType.HEALTHCARE_COMPLIANCE,
            TemplateType.FINANCIAL_RISK,
        }

        for t in compliance_types:
            assert isinstance(t, TemplateType)

    def test_research_types(self):
        """Test that research types exist."""
        research_types = {
            TemplateType.RESEARCH_SYNTHESIS,
        }

        for t in research_types:
            assert isinstance(t, TemplateType)


class TestTemplateTypeHashability:
    """Tests for TemplateType enum hashability."""

    def test_is_hashable(self):
        """Test that enum members are hashable."""
        # Should not raise
        hash(TemplateType.CODE_REVIEW)
        hash(TemplateType.DESIGN_DOC)

    def test_usable_as_dict_key(self):
        """Test that enum members can be used as dictionary keys."""
        template_dict = {
            TemplateType.CODE_REVIEW: "code review template",
            TemplateType.DESIGN_DOC: "design doc template",
        }

        assert template_dict[TemplateType.CODE_REVIEW] == "code review template"
        assert template_dict[TemplateType.DESIGN_DOC] == "design doc template"

    def test_usable_in_set(self):
        """Test that enum members can be used in sets."""
        template_set = {
            TemplateType.CODE_REVIEW,
            TemplateType.DESIGN_DOC,
            TemplateType.CODE_REVIEW,  # Duplicate
        }

        assert len(template_set) == 2
        assert TemplateType.CODE_REVIEW in template_set
        assert TemplateType.DESIGN_DOC in template_set
