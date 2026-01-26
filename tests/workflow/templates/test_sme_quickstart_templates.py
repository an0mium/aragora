"""
Tests for SME and Quickstart Workflow Templates.

Tests coverage for:
- SME decision templates (vendor, hiring, budget, business)
- Quickstart templates (yes/no, pros-cons, risk, brainstorm)
- Template factory functions
- Template category enums
"""

from __future__ import annotations

import pytest

from aragora.workflow.types import WorkflowCategory, WorkflowDefinition


class TestSMEDecisionTemplates:
    """Tests for SME decision workflow templates."""

    def test_create_vendor_evaluation_workflow(self):
        """Vendor evaluation workflow creates correctly."""
        from aragora.workflow.templates.sme import create_vendor_evaluation_workflow

        workflow = create_vendor_evaluation_workflow(
            vendor_name="Acme Corp",
            evaluation_criteria=["price", "support", "integration"],
            budget_range="$10k-$50k",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "vendor_eval_acme_corp" in workflow.id
        assert "Acme Corp" in workflow.name
        assert workflow.category == WorkflowCategory.GENERAL
        assert "vendor" in workflow.tags
        assert "sme" in workflow.tags

    def test_vendor_evaluation_default_criteria(self):
        """Vendor evaluation uses default criteria if none provided."""
        from aragora.workflow.templates.sme import create_vendor_evaluation_workflow

        workflow = create_vendor_evaluation_workflow(vendor_name="TestVendor")

        assert "pricing" in workflow.inputs["criteria"]
        assert "reliability" in workflow.inputs["criteria"]

    def test_create_hiring_decision_workflow(self):
        """Hiring decision workflow creates correctly."""
        from aragora.workflow.templates.sme import create_hiring_decision_workflow

        workflow = create_hiring_decision_workflow(
            position="Senior Developer",
            candidate_name="Jane Doe",
            interview_notes="Strong Python skills",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "hire_senior_developer" in workflow.id
        assert "Jane Doe" in workflow.name
        assert "hiring" in workflow.tags
        assert workflow.inputs["position"] == "Senior Developer"

    def test_create_budget_allocation_workflow(self):
        """Budget allocation workflow creates correctly."""
        from aragora.workflow.templates.sme import create_budget_allocation_workflow

        workflow = create_budget_allocation_workflow(
            department="Engineering",
            total_budget=500000,
            categories=["infrastructure", "tools", "training"],
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "budget_engineering" in workflow.id
        assert "$500,000" in workflow.description
        assert workflow.category == WorkflowCategory.ACCOUNTING
        assert "budget" in workflow.tags

    def test_budget_allocation_default_categories(self):
        """Budget allocation uses default categories if none provided."""
        from aragora.workflow.templates.sme import create_budget_allocation_workflow

        workflow = create_budget_allocation_workflow(
            department="Marketing",
            total_budget=100000,
        )

        categories = workflow.inputs["categories"]
        assert "operations" in categories
        assert "growth" in categories

    def test_create_business_decision_workflow(self):
        """Business decision workflow creates correctly."""
        from aragora.workflow.templates.sme import create_business_decision_workflow

        workflow = create_business_decision_workflow(
            decision_topic="Should we expand to Europe?",
            context="50% US market share",
            impact_level="high",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "decision_" in workflow.id
        assert "expand" in workflow.name.lower() or "business" in workflow.name.lower()
        assert "strategy" in workflow.tags

    def test_business_decision_impact_levels(self):
        """Business decision respects impact level for approval step."""
        from aragora.workflow.templates.sme import create_business_decision_workflow

        high_impact = create_business_decision_workflow(
            decision_topic="Major acquisition",
            impact_level="high",
        )
        low_impact = create_business_decision_workflow(
            decision_topic="Office supplies",
            impact_level="low",
        )

        # High impact should have approval step in workflow
        step_ids = [s.id for s in high_impact.steps]
        assert "approve" in step_ids

        # Low impact should skip approval
        low_step_ids = [s.id for s in low_impact.steps]
        assert "approve" not in low_step_ids or low_impact.inputs["impact_level"] == "low"


class TestQuickstartTemplates:
    """Tests for quickstart workflow templates."""

    def test_create_yes_no_workflow(self):
        """Yes/No workflow creates correctly."""
        from aragora.workflow.templates.quickstart import create_yes_no_workflow

        workflow = create_yes_no_workflow(
            question="Should we launch this week?",
            context="Testing is 95% complete",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "yesno_" in workflow.id
        assert "Quick Decision" in workflow.name
        assert "yes-no" in workflow.tags
        assert "quickstart" in workflow.tags

    def test_yes_no_minimal_steps(self):
        """Yes/No workflow has minimal steps for speed."""
        from aragora.workflow.templates.quickstart import create_yes_no_workflow

        workflow = create_yes_no_workflow(question="Test?")

        # Should have 3 or fewer main steps
        assert len(workflow.steps) <= 3

    def test_create_pros_cons_workflow(self):
        """Pros & Cons workflow creates correctly."""
        from aragora.workflow.templates.quickstart import create_pros_cons_workflow

        workflow = create_pros_cons_workflow(
            topic="Remote work policy",
            weighted=True,
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "proscons_" in workflow.id
        assert "Pros & Cons" in workflow.name
        assert "pros-cons" in workflow.tags

    def test_pros_cons_weighting(self):
        """Pros & Cons includes weight step when requested."""
        from aragora.workflow.templates.quickstart import create_pros_cons_workflow

        weighted = create_pros_cons_workflow(topic="Test", weighted=True)
        unweighted = create_pros_cons_workflow(topic="Test", weighted=False)

        weighted_step_ids = [s.id for s in weighted.steps]
        unweighted_step_ids = [s.id for s in unweighted.steps]

        assert "weight" in weighted_step_ids
        assert "weight" not in unweighted_step_ids

    def test_create_risk_assessment_workflow(self):
        """Risk assessment workflow creates correctly."""
        from aragora.workflow.templates.quickstart import create_risk_assessment_workflow

        workflow = create_risk_assessment_workflow(
            scenario="Launch new product",
            risk_categories=["market", "technical"],
            include_mitigation=True,
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "risk_" in workflow.id
        assert "Risk Assessment" in workflow.name
        assert "risk" in workflow.tags

    def test_risk_assessment_mitigation(self):
        """Risk assessment includes mitigation when requested."""
        from aragora.workflow.templates.quickstart import create_risk_assessment_workflow

        with_mitigation = create_risk_assessment_workflow(
            scenario="Test",
            include_mitigation=True,
        )
        without_mitigation = create_risk_assessment_workflow(
            scenario="Test",
            include_mitigation=False,
        )

        with_steps = [s.id for s in with_mitigation.steps]
        without_steps = [s.id for s in without_mitigation.steps]

        assert "mitigate" in with_steps
        assert "mitigate" not in without_steps

    def test_create_brainstorm_workflow(self):
        """Brainstorm workflow creates correctly."""
        from aragora.workflow.templates.quickstart import create_brainstorm_workflow

        workflow = create_brainstorm_workflow(
            topic="Improve customer retention",
            goal="Reduce churn by 20%",
            num_ideas=15,
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "brainstorm_" in workflow.id
        assert "Brainstorm" in workflow.name
        assert "brainstorm" in workflow.tags
        assert workflow.inputs["num_ideas"] == 15


class TestQuickstartConvenienceFunctions:
    """Tests for quickstart convenience functions."""

    def test_quick_decision(self):
        """quick_decision creates minimal yes/no workflow."""
        from aragora.workflow.templates.quickstart import quick_decision

        workflow = quick_decision("Should we proceed?")

        assert isinstance(workflow, WorkflowDefinition)
        assert "yesno_" in workflow.id

    def test_quick_analysis(self):
        """quick_analysis creates streamlined pros/cons workflow."""
        from aragora.workflow.templates.quickstart import quick_analysis

        workflow = quick_analysis("New feature idea")

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.inputs["max_items"] == 3
        assert workflow.inputs["weighted"] is False

    def test_quick_risks(self):
        """quick_risks creates minimal risk workflow."""
        from aragora.workflow.templates.quickstart import quick_risks

        workflow = quick_risks("Launch scenario")

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.inputs["include_mitigation"] is False

    def test_quick_ideas(self):
        """quick_ideas creates fast brainstorm workflow."""
        from aragora.workflow.templates.quickstart import quick_ideas

        workflow = quick_ideas("Marketing campaign", count=5)

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.inputs["num_ideas"] == 5
        assert workflow.inputs["prioritize"] is False


class TestTemplateCategoryEnums:
    """Tests for template category enums."""

    def test_sme_category_exists(self):
        """SME category exists in TemplateCategory."""
        from aragora.workflow.templates.package import TemplateCategory

        assert hasattr(TemplateCategory, "SME")
        assert TemplateCategory.SME.value == "sme"

    def test_quickstart_category_exists(self):
        """QUICKSTART category exists in TemplateCategory."""
        from aragora.workflow.templates.package import TemplateCategory

        assert hasattr(TemplateCategory, "QUICKSTART")
        assert TemplateCategory.QUICKSTART.value == "quickstart"

    def test_retail_category_exists(self):
        """RETAIL category exists in TemplateCategory."""
        from aragora.workflow.templates.package import TemplateCategory

        assert hasattr(TemplateCategory, "RETAIL")
        assert TemplateCategory.RETAIL.value == "retail"


class TestTemplateExports:
    """Tests for template module exports."""

    def test_sme_functions_exported(self):
        """SME functions are exported from templates module."""
        from aragora.workflow.templates import (
            create_vendor_evaluation_workflow,
            create_hiring_decision_workflow,
            create_budget_allocation_workflow,
            create_business_decision_workflow,
        )

        assert callable(create_vendor_evaluation_workflow)
        assert callable(create_hiring_decision_workflow)
        assert callable(create_budget_allocation_workflow)
        assert callable(create_business_decision_workflow)

    def test_quickstart_functions_exported(self):
        """Quickstart functions are exported from templates module."""
        from aragora.workflow.templates import (
            create_yes_no_workflow,
            create_pros_cons_workflow,
            create_risk_assessment_workflow,
            create_brainstorm_workflow,
            quick_decision,
            quick_analysis,
            quick_risks,
            quick_ideas,
        )

        assert callable(create_yes_no_workflow)
        assert callable(create_pros_cons_workflow)
        assert callable(create_risk_assessment_workflow)
        assert callable(create_brainstorm_workflow)
        assert callable(quick_decision)
        assert callable(quick_analysis)
        assert callable(quick_risks)
        assert callable(quick_ideas)


class TestMarketplaceTemplates:
    """Tests for marketplace template seeding."""

    def test_sme_templates_in_marketplace(self):
        """SME templates are seeded in marketplace."""
        from aragora.server.handlers.template_marketplace import (
            _seed_marketplace_templates,
            _marketplace_templates,
        )

        # Clear and re-seed
        _marketplace_templates.clear()
        _seed_marketplace_templates()

        sme_templates = [t for t in _marketplace_templates.values() if t.category == "sme"]

        assert len(sme_templates) >= 4
        template_ids = [t.id for t in sme_templates]
        assert "sme/vendor-evaluation" in template_ids
        assert "sme/hiring-decision" in template_ids
        assert "sme/budget-allocation" in template_ids
        assert "sme/business-decision" in template_ids

    def test_quickstart_templates_in_marketplace(self):
        """Quickstart templates are seeded in marketplace."""
        from aragora.server.handlers.template_marketplace import (
            _seed_marketplace_templates,
            _marketplace_templates,
        )

        # Clear and re-seed
        _marketplace_templates.clear()
        _seed_marketplace_templates()

        quickstart_templates = [
            t for t in _marketplace_templates.values() if t.category == "quickstart"
        ]

        assert len(quickstart_templates) >= 4
        template_ids = [t.id for t in quickstart_templates]
        assert "quickstart/yes-no" in template_ids
        assert "quickstart/pros-cons" in template_ids
        assert "quickstart/risk-assessment" in template_ids
        assert "quickstart/brainstorm" in template_ids

    def test_sme_templates_are_verified(self):
        """SME templates are marked as verified."""
        from aragora.server.handlers.template_marketplace import (
            _seed_marketplace_templates,
            _marketplace_templates,
        )

        _marketplace_templates.clear()
        _seed_marketplace_templates()

        sme_templates = [t for t in _marketplace_templates.values() if t.category == "sme"]

        for template in sme_templates:
            assert template.is_verified is True

    def test_featured_templates_include_sme(self):
        """Some SME/quickstart templates are featured."""
        from aragora.server.handlers.template_marketplace import (
            _seed_marketplace_templates,
            _marketplace_templates,
        )

        _marketplace_templates.clear()
        _seed_marketplace_templates()

        featured = [t for t in _marketplace_templates.values() if t.is_featured]
        featured_categories = {t.category for t in featured}

        assert "sme" in featured_categories or "quickstart" in featured_categories
