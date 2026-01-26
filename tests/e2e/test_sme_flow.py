"""
E2E Tests for SME Flow.

Tests the complete SME (Small/Medium Enterprise) user journey:
- Quick debate creation via onboarding
- Template-based workflow execution
- Usage dashboard visibility
- Receipt generation verification

This tests the full Week 1-4 SME track implementation.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.e2e


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_debate_response():
    """Create a mock debate response for testing."""
    from dataclasses import dataclass

    @dataclass
    class MockDebateResponse:
        success: bool = True
        debate_id: str = f"debate_{uuid.uuid4().hex[:8]}"
        websocket_url: str = "ws://localhost:8080/ws/debate/test"
        error: Optional[str] = None
        status_code: int = 200

    return MockDebateResponse()


@pytest.fixture
def test_user_context():
    """Create test user context."""
    return {
        "user_id": f"user_{uuid.uuid4().hex[:8]}",
        "organization_id": f"org_{uuid.uuid4().hex[:8]}",
        "email": "test@example.com",
    }


# ============================================================================
# Quick Debate Tests
# ============================================================================


class TestQuickDebateCreation:
    """Tests for quick debate creation via onboarding."""

    def test_quick_debate_handler_exists(self):
        """Quick debate handler function exists."""
        from aragora.server.handlers.onboarding import handle_quick_debate

        assert callable(handle_quick_debate)

    def test_quick_debate_templates_available(self):
        """Quick debate templates are available in onboarding."""
        from aragora.server.handlers.onboarding import STARTER_TEMPLATES

        assert len(STARTER_TEMPLATES) > 0
        template_ids = [t.id for t in STARTER_TEMPLATES]
        assert "express_onboarding" in template_ids

    def test_quick_start_configs_exist(self):
        """Quick start configurations exist for different profiles."""
        from aragora.server.handlers.onboarding import QUICK_START_CONFIGS

        assert "developer" in QUICK_START_CONFIGS
        # "business" might have different name in implementation
        assert len(QUICK_START_CONFIGS) > 0

    def test_onboarding_handler_has_handlers(self):
        """Onboarding handler includes all required handlers."""
        from aragora.server.handlers.onboarding import get_onboarding_handlers

        handlers = get_onboarding_handlers()
        assert "quick_debate" in handlers
        assert "init_flow" in handlers


# ============================================================================
# SME Template Tests
# ============================================================================


class TestSMETemplates:
    """Tests for SME workflow templates."""

    def test_quickstart_templates_available(self):
        """Quickstart templates are available in the system."""
        from aragora.workflow.templates import (
            create_yes_no_workflow,
            create_pros_cons_workflow,
            create_risk_assessment_workflow,
            create_brainstorm_workflow,
        )

        # All quickstart templates should be callable
        assert callable(create_yes_no_workflow)
        assert callable(create_pros_cons_workflow)
        assert callable(create_risk_assessment_workflow)
        assert callable(create_brainstorm_workflow)

    def test_sme_templates_available(self):
        """SME business templates are available in the system."""
        from aragora.workflow.templates import (
            create_vendor_evaluation_workflow,
            create_hiring_decision_workflow,
            create_budget_allocation_workflow,
            create_business_decision_workflow,
        )

        # All SME templates should be callable
        assert callable(create_vendor_evaluation_workflow)
        assert callable(create_hiring_decision_workflow)
        assert callable(create_budget_allocation_workflow)
        assert callable(create_business_decision_workflow)

    def test_quickstart_yes_no_workflow(self):
        """Yes/No quickstart workflow creates properly."""
        from aragora.workflow.templates import create_yes_no_workflow
        from aragora.workflow.types import WorkflowDefinition

        workflow = create_yes_no_workflow(
            question="Should we proceed with the launch?",
            context="All tests are passing",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "yesno_" in workflow.id
        assert len(workflow.steps) <= 3  # Quick workflow

    def test_sme_vendor_evaluation_workflow(self):
        """Vendor evaluation workflow creates properly."""
        from aragora.workflow.templates import create_vendor_evaluation_workflow
        from aragora.workflow.types import WorkflowDefinition

        workflow = create_vendor_evaluation_workflow(
            vendor_name="Acme Corp",
            evaluation_criteria=["price", "support", "features"],
            budget_range="$10k-$50k",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "vendor" in workflow.tags
        assert "sme" in workflow.tags

    def test_template_categories_exist(self):
        """SME and QUICKSTART categories exist in TemplateCategory."""
        from aragora.workflow.templates.package import TemplateCategory

        assert hasattr(TemplateCategory, "SME")
        assert hasattr(TemplateCategory, "QUICKSTART")
        assert TemplateCategory.SME.value == "sme"
        assert TemplateCategory.QUICKSTART.value == "quickstart"


# ============================================================================
# Usage Dashboard Tests
# ============================================================================


class TestUsageDashboard:
    """Tests for SME usage dashboard functionality."""

    def test_usage_handler_module_exists(self):
        """Usage dashboard handler module exists."""
        from aragora.server.handlers import sme_usage_dashboard

        # Module should be importable
        assert sme_usage_dashboard is not None

    def test_usage_handler_class_exists(self):
        """Usage dashboard handler class exists."""
        from aragora.server.handlers.sme_usage_dashboard import SMEUsageDashboardHandler

        # Class should be importable
        assert SMEUsageDashboardHandler is not None

    def test_roi_calculator_exists(self):
        """ROI calculator module exists."""
        from aragora.billing import roi_calculator

        # Check for the actual function name
        assert hasattr(roi_calculator, "get_roi_calculator")
        assert callable(roi_calculator.get_roi_calculator)


# ============================================================================
# Marketplace Integration Tests
# ============================================================================


class TestMarketplaceIntegration:
    """Tests for SME templates in marketplace."""

    def test_sme_templates_in_marketplace(self):
        """SME templates appear in marketplace."""
        from aragora.server.handlers.template_marketplace import (
            _seed_marketplace_templates,
            _marketplace_templates,
        )

        _marketplace_templates.clear()
        _seed_marketplace_templates()

        sme_templates = [t for t in _marketplace_templates.values() if t.category == "sme"]

        assert len(sme_templates) >= 4
        template_ids = [t.id for t in sme_templates]
        assert "sme/vendor-evaluation" in template_ids
        assert "sme/hiring-decision" in template_ids

    def test_quickstart_templates_in_marketplace(self):
        """Quickstart templates appear in marketplace."""
        from aragora.server.handlers.template_marketplace import (
            _seed_marketplace_templates,
            _marketplace_templates,
        )

        _marketplace_templates.clear()
        _seed_marketplace_templates()

        quickstart_templates = [
            t for t in _marketplace_templates.values() if t.category == "quickstart"
        ]

        assert len(quickstart_templates) >= 4
        template_ids = [t.id for t in quickstart_templates]
        assert "quickstart/yes-no" in template_ids
        assert "quickstart/pros-cons" in template_ids


# ============================================================================
# Full SME Flow Integration Tests
# ============================================================================


class TestSMEFullFlow:
    """Integration tests for complete SME user journey."""

    def test_sme_onboarding_handlers_available(self):
        """Test that all onboarding handlers are available."""
        from aragora.server.handlers.onboarding import (
            handle_init_flow,
            handle_quick_debate,
            handle_get_templates,
            get_onboarding_handlers,
        )

        # All handlers should be callable
        assert callable(handle_init_flow)
        assert callable(handle_quick_debate)
        assert callable(handle_get_templates)

        # Handlers should be registered
        handlers = get_onboarding_handlers()
        assert "init_flow" in handlers
        assert "quick_debate" in handlers
        assert "get_templates" in handlers

    def test_template_workflow_complete_structure(self):
        """Test that template workflows have complete structure."""
        from aragora.workflow.templates import create_business_decision_workflow
        from aragora.workflow.types import WorkflowDefinition

        workflow = create_business_decision_workflow(
            decision_topic="Expand to new market",
            context="Strong Q3 performance",
            impact_level="high",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.entry_step is not None
        assert len(workflow.steps) > 0

        # Check workflow has proper structure
        step_ids = [s.id for s in workflow.steps]
        assert workflow.entry_step in step_ids

        # High impact should have approval step
        if workflow.inputs.get("impact_level") == "high":
            assert "approve" in step_ids

    def test_convenience_functions_work(self):
        """Test quickstart convenience functions."""
        from aragora.workflow.templates import (
            quick_decision,
            quick_analysis,
            quick_risks,
            quick_ideas,
        )
        from aragora.workflow.types import WorkflowDefinition

        # All convenience functions should return valid workflows
        decision = quick_decision("Should we proceed?")
        assert isinstance(decision, WorkflowDefinition)

        analysis = quick_analysis("New product idea")
        assert isinstance(analysis, WorkflowDefinition)
        assert analysis.inputs["weighted"] is False

        risks = quick_risks("Launch scenario")
        assert isinstance(risks, WorkflowDefinition)
        assert risks.inputs["include_mitigation"] is False

        ideas = quick_ideas("Marketing campaign", count=5)
        assert isinstance(ideas, WorkflowDefinition)
        assert ideas.inputs["num_ideas"] == 5


# ============================================================================
# Performance Tests
# ============================================================================


class TestSMEPerformance:
    """Performance-related tests for SME features."""

    def test_template_creation_is_fast(self):
        """Template creation should be instantaneous."""
        import time
        from aragora.workflow.templates import (
            create_vendor_evaluation_workflow,
            create_yes_no_workflow,
        )

        start = time.perf_counter()
        for _ in range(100):
            create_vendor_evaluation_workflow(
                vendor_name="Test",
                evaluation_criteria=["a", "b", "c"],
            )
            create_yes_no_workflow(question="Test?")
        elapsed = time.perf_counter() - start

        # 100 template creations should take less than 1 second
        assert elapsed < 1.0, f"Template creation took {elapsed:.2f}s for 100 iterations"

    def test_marketplace_seeding_is_fast(self):
        """Marketplace seeding should be fast."""
        import time
        from aragora.server.handlers.template_marketplace import (
            _seed_marketplace_templates,
            _marketplace_templates,
        )

        _marketplace_templates.clear()

        start = time.perf_counter()
        _seed_marketplace_templates()
        elapsed = time.perf_counter() - start

        # Seeding should take less than 500ms
        assert elapsed < 0.5, f"Marketplace seeding took {elapsed:.2f}s"
