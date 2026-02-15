"""
Tests for SME business decision templates.
"""

import pytest

from aragora.deliberation.templates.base import (
    DeliberationTemplate,
    OutputFormat,
    TemplateCategory,
)
from aragora.deliberation.templates.builtins import (
    BUDGET_ALLOCATION,
    BUILTIN_TEMPLATES,
    HIRING_DECISION,
    PERFORMANCE_REVIEW,
    STRATEGIC_PLANNING,
    TOOL_SELECTION,
    VENDOR_EVALUATION,
)
from aragora.deliberation.templates.registry import TemplateRegistry, get_template


BUSINESS_TEMPLATE_NAMES = [
    "hiring_decision",
    "vendor_evaluation",
    "budget_allocation",
    "tool_selection",
    "performance_review",
    "strategic_planning",
]

BUSINESS_TEMPLATES = [
    HIRING_DECISION,
    VENDOR_EVALUATION,
    BUDGET_ALLOCATION,
    TOOL_SELECTION,
    PERFORMANCE_REVIEW,
    STRATEGIC_PLANNING,
]


class TestBusinessCategory:
    """Tests for the BUSINESS template category."""

    def test_business_category_exists(self):
        """BUSINESS is a valid TemplateCategory."""
        assert TemplateCategory.BUSINESS.value == "business"

    def test_business_category_from_string(self):
        """Can construct BUSINESS category from string."""
        cat = TemplateCategory("business")
        assert cat == TemplateCategory.BUSINESS


class TestBusinessTemplatesRegistered:
    """Tests that all business templates are registered in BUILTIN_TEMPLATES."""

    @pytest.mark.parametrize("name", BUSINESS_TEMPLATE_NAMES)
    def test_template_in_builtins(self, name):
        """Template is present in BUILTIN_TEMPLATES dict."""
        assert name in BUILTIN_TEMPLATES

    @pytest.mark.parametrize("name", BUSINESS_TEMPLATE_NAMES)
    def test_template_accessible_via_registry(self, name):
        """Template is accessible via the global registry."""
        template = get_template(name)
        assert template is not None
        assert template.name == name

    def test_registry_filter_by_business_category(self):
        """Registry can filter templates by BUSINESS category."""
        registry = TemplateRegistry()
        business = registry.list(category=TemplateCategory.BUSINESS)
        names = [t.name for t in business]
        for name in BUSINESS_TEMPLATE_NAMES:
            assert name in names


class TestBusinessTemplateCategory:
    """Tests that all business templates have the correct category."""

    @pytest.mark.parametrize(
        "template", BUSINESS_TEMPLATES, ids=[t.name for t in BUSINESS_TEMPLATES]
    )
    def test_has_business_category(self, template):
        """Each business template has category BUSINESS."""
        assert template.category == TemplateCategory.BUSINESS


class TestBusinessTemplateFields:
    """Tests for field correctness on each business template."""

    @pytest.mark.parametrize(
        "template", BUSINESS_TEMPLATES, ids=[t.name for t in BUSINESS_TEMPLATES]
    )
    def test_has_default_agents(self, template):
        """Each business template has at least one default agent."""
        assert len(template.default_agents) >= 1

    @pytest.mark.parametrize(
        "template", BUSINESS_TEMPLATES, ids=[t.name for t in BUSINESS_TEMPLATES]
    )
    def test_has_personas(self, template):
        """Each business template defines personas."""
        assert len(template.personas) >= 2

    @pytest.mark.parametrize(
        "template", BUSINESS_TEMPLATES, ids=[t.name for t in BUSINESS_TEMPLATES]
    )
    def test_has_business_tag(self, template):
        """Each business template is tagged 'business'."""
        assert "business" in template.tags

    @pytest.mark.parametrize(
        "template", BUSINESS_TEMPLATES, ids=[t.name for t in BUSINESS_TEMPLATES]
    )
    def test_has_description(self, template):
        """Each business template has a non-empty description."""
        assert len(template.description) > 10

    def test_hiring_decision_specifics(self):
        """Hiring template has expected configuration."""
        assert "talent" in HIRING_DECISION.personas
        assert "hiring" in HIRING_DECISION.tags
        assert HIRING_DECISION.output_format == OutputFormat.DECISION_RECEIPT

    def test_vendor_evaluation_specifics(self):
        """Vendor template focuses on comparison criteria."""
        assert "cost" in VENDOR_EVALUATION.personas
        assert "vendor" in VENDOR_EVALUATION.tags

    def test_budget_allocation_specifics(self):
        """Budget template allows enough rounds for debate."""
        assert BUDGET_ALLOCATION.max_rounds >= 4
        assert "finance" in BUDGET_ALLOCATION.personas

    def test_tool_selection_specifics(self):
        """Tool selection template is fast and summary-oriented."""
        assert TOOL_SELECTION.output_format == OutputFormat.SUMMARY
        assert TOOL_SELECTION.max_rounds <= 4

    def test_strategic_planning_specifics(self):
        """Strategic planning template uses diverse agents."""
        assert len(STRATEGIC_PLANNING.default_agents) >= 3
        assert "strategy" in STRATEGIC_PLANNING.personas


class TestBusinessTemplateSerialization:
    """Tests for to_dict/from_dict roundtrip."""

    @pytest.mark.parametrize(
        "template", BUSINESS_TEMPLATES, ids=[t.name for t in BUSINESS_TEMPLATES]
    )
    def test_to_dict_roundtrip(self, template):
        """Template survives to_dict -> from_dict roundtrip."""
        data = template.to_dict()
        restored = DeliberationTemplate.from_dict(data)
        assert restored.name == template.name
        assert restored.description == template.description
        assert restored.category == template.category
        assert restored.consensus_threshold == template.consensus_threshold
        assert restored.max_rounds == template.max_rounds
        assert restored.tags == template.tags

    @pytest.mark.parametrize(
        "template", BUSINESS_TEMPLATES, ids=[t.name for t in BUSINESS_TEMPLATES]
    )
    def test_to_dict_category_value(self, template):
        """to_dict produces category as string 'business'."""
        data = template.to_dict()
        assert data["category"] == "business"

    @pytest.mark.parametrize(
        "template", BUSINESS_TEMPLATES, ids=[t.name for t in BUSINESS_TEMPLATES]
    )
    def test_to_dict_has_required_keys(self, template):
        """to_dict output has all required keys."""
        data = template.to_dict()
        required_keys = [
            "name",
            "description",
            "category",
            "default_agents",
            "team_strategy",
            "consensus_threshold",
            "max_rounds",
            "output_format",
            "personas",
            "tags",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"
