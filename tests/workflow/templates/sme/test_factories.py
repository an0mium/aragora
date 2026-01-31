"""
Comprehensive Tests for SME Workflow Factories Module.

Tests coverage for:
- Factory method registration and discovery
- Template creation for different SME use cases
- Parameter validation and defaults
- Workflow node configuration
- Integration between factories and templates
- Error handling for invalid configurations
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.workflow.types import (
    StepDefinition,
    WorkflowCategory,
    WorkflowDefinition,
)

# Import all factory functions
from aragora.workflow.templates.sme._factories import (
    # Main factory functions
    create_invoice_workflow,
    create_followup_workflow,
    create_inventory_alert_workflow,
    create_report_workflow,
    # SME Decision templates
    create_vendor_evaluation_workflow,
    create_hiring_decision_workflow,
    create_budget_allocation_workflow,
    create_business_decision_workflow,
    create_performance_review_workflow,
    create_feature_prioritization_workflow,
    create_sprint_planning_workflow,
    create_tool_selection_workflow,
    create_contract_review_workflow,
    create_remote_work_policy_workflow,
    # Quick convenience functions
    quick_invoice,
    weekly_sales_report,
    daily_inventory_check,
    renewal_followup_campaign,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_items():
    """Sample invoice items for testing."""
    return [
        {"name": "Consulting", "quantity": 10, "unit_price": 150.00},
        {"name": "Support", "quantity": 1, "unit_price": 500.00},
    ]


@pytest.fixture
def sample_features():
    """Sample features list for testing."""
    return ["Dark mode", "API v2", "Mobile app", "Export to PDF"]


@pytest.fixture
def sample_backlog():
    """Sample backlog items for testing."""
    return ["User authentication", "Dashboard redesign", "API refactor", "Bug fixes"]


@pytest.fixture
def sample_candidates():
    """Sample tool candidates for testing."""
    return ["Jira", "Linear", "Asana", "Monday"]


# =============================================================================
# Invoice Workflow Tests
# =============================================================================


class TestCreateInvoiceWorkflow:
    """Tests for create_invoice_workflow factory."""

    def test_creates_valid_workflow_definition(self, sample_items):
        """Invoice workflow creates a valid WorkflowDefinition."""
        workflow = create_invoice_workflow(
            customer_id="cust_123",
            items=sample_items,
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.id is not None
        assert "invoice" in workflow.id
        assert "cust_123" in workflow.id

    def test_workflow_has_correct_name(self, sample_items):
        """Invoice workflow has correct name format."""
        workflow = create_invoice_workflow(
            customer_id="cust_abc",
            items=sample_items,
        )

        assert "Invoice for cust_abc" in workflow.name

    def test_workflow_has_accounting_category(self, sample_items):
        """Invoice workflow has ACCOUNTING category."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=sample_items,
        )

        assert workflow.category == WorkflowCategory.ACCOUNTING

    def test_workflow_has_correct_tags(self, sample_items):
        """Invoice workflow has expected tags."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=sample_items,
        )

        assert "sme" in workflow.tags
        assert "invoice" in workflow.tags
        assert "billing" in workflow.tags

    def test_default_tax_rate_is_zero(self, sample_items):
        """Invoice workflow defaults to 0% tax rate."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=sample_items,
        )

        assert workflow.inputs["tax_rate"] == 0.0

    def test_default_due_days_is_thirty(self, sample_items):
        """Invoice workflow defaults to 30 days due."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=sample_items,
        )

        assert workflow.inputs["due_days"] == 30

    def test_send_email_false_by_default(self, sample_items):
        """Invoice workflow defaults to not sending email."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=sample_items,
        )

        assert workflow.inputs["send_email"] is False

    def test_custom_tax_rate_applied(self, sample_items):
        """Invoice workflow applies custom tax rate."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=sample_items,
            tax_rate=0.08,
        )

        assert workflow.inputs["tax_rate"] == 0.08

    def test_custom_due_days_applied(self, sample_items):
        """Invoice workflow applies custom due days."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=sample_items,
            due_days=45,
        )

        assert workflow.inputs["due_days"] == 45

    def test_items_stored_in_inputs(self, sample_items):
        """Invoice workflow stores items in inputs."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=sample_items,
        )

        assert workflow.inputs["items"] == sample_items

    def test_notes_stored_when_provided(self, sample_items):
        """Invoice workflow stores notes when provided."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=sample_items,
            notes="Payment terms: Net 30",
        )

        assert workflow.inputs["notes"] == "Payment terms: Net 30"

    def test_has_validate_step(self, sample_items):
        """Invoice workflow has validation step."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=sample_items,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "validate" in step_ids

    def test_has_calculate_step(self, sample_items):
        """Invoice workflow has calculate totals step."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=sample_items,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "calculate" in step_ids

    def test_has_generate_step(self, sample_items):
        """Invoice workflow has generate step."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=sample_items,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "generate" in step_ids

    def test_has_store_step(self, sample_items):
        """Invoice workflow has store step."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=sample_items,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "store" in step_ids

    def test_entry_step_is_validate(self, sample_items):
        """Invoice workflow starts with validate step."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=sample_items,
        )

        assert workflow.entry_step == "validate"

    def test_send_email_adds_deliver_step(self, sample_items):
        """Invoice workflow adds deliver step when send_email is True."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=sample_items,
            send_email=True,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "deliver" in step_ids

    def test_generate_step_routes_to_deliver_when_sending(self, sample_items):
        """Generate step routes to deliver when send_email is True."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=sample_items,
            send_email=True,
        )

        generate_step = workflow.get_step("generate")
        assert generate_step is not None
        assert "deliver" in generate_step.next_steps

    def test_generate_step_routes_to_store_when_not_sending(self, sample_items):
        """Generate step routes to store when send_email is False."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=sample_items,
            send_email=False,
        )

        generate_step = workflow.get_step("generate")
        assert generate_step is not None
        assert "store" in generate_step.next_steps


# =============================================================================
# Follow-up Workflow Tests
# =============================================================================


class TestCreateFollowupWorkflow:
    """Tests for create_followup_workflow factory."""

    def test_creates_valid_workflow_definition(self):
        """Followup workflow creates a valid WorkflowDefinition."""
        workflow = create_followup_workflow()

        assert isinstance(workflow, WorkflowDefinition)
        assert "followup" in workflow.id

    def test_default_followup_type_is_check_in(self):
        """Followup workflow defaults to check_in type."""
        workflow = create_followup_workflow()

        assert workflow.inputs["followup_type"] == "check_in"

    def test_default_days_since_contact_is_thirty(self):
        """Followup workflow defaults to 30 days since contact."""
        workflow = create_followup_workflow()

        assert workflow.inputs["days_since_contact"] == 30

    def test_default_channel_is_email(self):
        """Followup workflow defaults to email channel."""
        workflow = create_followup_workflow()

        assert workflow.inputs["channel"] == "email"

    def test_default_auto_send_is_false(self):
        """Followup workflow defaults to not auto-sending."""
        workflow = create_followup_workflow()

        assert workflow.inputs["auto_send"] is False

    def test_custom_followup_type_in_name(self):
        """Followup workflow includes type in name."""
        workflow = create_followup_workflow(followup_type="renewal")

        assert "renewal" in workflow.name

    def test_has_identify_step(self):
        """Followup workflow has identify customers step."""
        workflow = create_followup_workflow()

        step_ids = [s.id for s in workflow.steps]
        assert "identify" in step_ids

    def test_has_analyze_step(self):
        """Followup workflow has analyze context step."""
        workflow = create_followup_workflow()

        step_ids = [s.id for s in workflow.steps]
        assert "analyze" in step_ids

    def test_has_draft_step(self):
        """Followup workflow has draft messages step."""
        workflow = create_followup_workflow()

        step_ids = [s.id for s in workflow.steps]
        assert "draft" in step_ids

    def test_auto_send_false_includes_review_step(self):
        """Followup workflow includes review step when auto_send is False."""
        workflow = create_followup_workflow(auto_send=False)

        step_ids = [s.id for s in workflow.steps]
        assert "review" in step_ids

    def test_auto_send_true_skips_review_step(self):
        """Followup workflow draft step routes to send when auto_send is True."""
        workflow = create_followup_workflow(auto_send=True)

        draft_step = workflow.get_step("draft")
        assert draft_step is not None
        assert "send" in draft_step.next_steps

    def test_entry_step_is_identify(self):
        """Followup workflow starts with identify step."""
        workflow = create_followup_workflow()

        assert workflow.entry_step == "identify"

    def test_customer_id_filter_stored(self):
        """Followup workflow stores customer_id filter when provided."""
        workflow = create_followup_workflow(customer_id="cust_specific")

        assert workflow.inputs["customer_id"] == "cust_specific"

    def test_has_crm_tag(self):
        """Followup workflow has CRM tag."""
        workflow = create_followup_workflow()

        assert "crm" in workflow.tags


# =============================================================================
# Inventory Alert Workflow Tests
# =============================================================================


class TestCreateInventoryAlertWorkflow:
    """Tests for create_inventory_alert_workflow factory."""

    def test_creates_valid_workflow_definition(self):
        """Inventory alert workflow creates a valid WorkflowDefinition."""
        workflow = create_inventory_alert_workflow()

        assert isinstance(workflow, WorkflowDefinition)
        assert "inventory_alert" in workflow.id

    def test_default_alert_threshold_is_twenty(self):
        """Inventory alert workflow defaults to 20% threshold."""
        workflow = create_inventory_alert_workflow()

        assert workflow.inputs["alert_threshold"] == 20

    def test_default_auto_reorder_is_false(self):
        """Inventory alert workflow defaults to not auto-reordering."""
        workflow = create_inventory_alert_workflow()

        assert workflow.inputs["auto_reorder"] is False

    def test_default_notification_channels_is_email(self):
        """Inventory alert workflow defaults to email notification."""
        workflow = create_inventory_alert_workflow()

        assert "email" in workflow.inputs["notification_channels"]

    def test_custom_notification_channels(self):
        """Inventory alert workflow accepts custom notification channels."""
        workflow = create_inventory_alert_workflow(notification_channels=["email", "slack", "sms"])

        channels = workflow.inputs["notification_channels"]
        assert "email" in channels
        assert "slack" in channels
        assert "sms" in channels

    def test_categories_filter_stored(self):
        """Inventory alert workflow stores categories filter."""
        workflow = create_inventory_alert_workflow(categories=["electronics", "clothing"])

        assert workflow.inputs["categories"] == ["electronics", "clothing"]

    def test_has_fetch_step(self):
        """Inventory alert workflow has fetch inventory step."""
        workflow = create_inventory_alert_workflow()

        step_ids = [s.id for s in workflow.steps]
        assert "fetch" in step_ids

    def test_has_analyze_step(self):
        """Inventory alert workflow has analyze levels step."""
        workflow = create_inventory_alert_workflow()

        step_ids = [s.id for s in workflow.steps]
        assert "analyze" in step_ids

    def test_has_alert_step(self):
        """Inventory alert workflow has send alerts step."""
        workflow = create_inventory_alert_workflow()

        step_ids = [s.id for s in workflow.steps]
        assert "alert" in step_ids

    def test_auto_reorder_includes_reorder_step(self):
        """Inventory alert workflow includes reorder step when auto_reorder is True."""
        workflow = create_inventory_alert_workflow(auto_reorder=True)

        step_ids = [s.id for s in workflow.steps]
        assert "reorder" in step_ids
        assert "submit" in step_ids

    def test_alert_routes_to_reorder_when_auto_reorder(self):
        """Alert step routes to reorder when auto_reorder is True."""
        workflow = create_inventory_alert_workflow(auto_reorder=True)

        alert_step = workflow.get_step("alert")
        assert alert_step is not None
        assert "reorder" in alert_step.next_steps

    def test_alert_routes_to_store_when_not_auto_reorder(self):
        """Alert step routes to store when auto_reorder is False."""
        workflow = create_inventory_alert_workflow(auto_reorder=False)

        alert_step = workflow.get_step("alert")
        assert alert_step is not None
        assert "store" in alert_step.next_steps

    def test_entry_step_is_fetch(self):
        """Inventory alert workflow starts with fetch step."""
        workflow = create_inventory_alert_workflow()

        assert workflow.entry_step == "fetch"

    def test_has_supply_chain_tag(self):
        """Inventory alert workflow has supply-chain tag."""
        workflow = create_inventory_alert_workflow()

        assert "supply-chain" in workflow.tags


# =============================================================================
# Report Workflow Tests
# =============================================================================


class TestCreateReportWorkflow:
    """Tests for create_report_workflow factory."""

    def test_creates_valid_workflow_definition(self):
        """Report workflow creates a valid WorkflowDefinition."""
        workflow = create_report_workflow(report_type="sales")

        assert isinstance(workflow, WorkflowDefinition)
        assert "report_sales" in workflow.id

    def test_report_type_in_name(self):
        """Report workflow includes report type in name."""
        workflow = create_report_workflow(report_type="financial")

        assert "Financial" in workflow.name

    def test_frequency_in_name(self):
        """Report workflow includes frequency in name."""
        workflow = create_report_workflow(report_type="sales", frequency="monthly")

        assert "monthly" in workflow.name

    def test_default_frequency_is_weekly(self):
        """Report workflow defaults to weekly frequency."""
        workflow = create_report_workflow(report_type="sales")

        assert workflow.inputs["frequency"] == "weekly"

    def test_default_date_range_is_last_week(self):
        """Report workflow defaults to last_week date range."""
        workflow = create_report_workflow(report_type="sales")

        assert workflow.inputs["date_range"] == "last_week"

    def test_default_format_is_pdf(self):
        """Report workflow defaults to PDF format."""
        workflow = create_report_workflow(report_type="sales")

        assert workflow.inputs["format"] == "pdf"

    def test_default_include_charts_is_true(self):
        """Report workflow defaults to including charts."""
        workflow = create_report_workflow(report_type="sales")

        assert workflow.inputs["include_charts"] is True

    def test_custom_recipients_stored(self):
        """Report workflow stores custom recipients."""
        workflow = create_report_workflow(
            report_type="sales",
            recipients=["ceo@company.com", "cfo@company.com"],
        )

        assert "ceo@company.com" in workflow.inputs["recipients"]
        assert "cfo@company.com" in workflow.inputs["recipients"]

    def test_has_fetch_step(self):
        """Report workflow has fetch data step."""
        workflow = create_report_workflow(report_type="sales")

        step_ids = [s.id for s in workflow.steps]
        assert "fetch" in step_ids

    def test_has_analyze_step(self):
        """Report workflow has analyze data step."""
        workflow = create_report_workflow(report_type="sales")

        step_ids = [s.id for s in workflow.steps]
        assert "analyze" in step_ids

    def test_include_charts_adds_charts_steps(self):
        """Report workflow includes charts steps when include_charts is True."""
        workflow = create_report_workflow(report_type="sales", include_charts=True)

        step_ids = [s.id for s in workflow.steps]
        assert "charts" in step_ids
        assert "render" in step_ids

    def test_analyze_routes_to_charts_when_include_charts(self):
        """Analyze step routes to charts when include_charts is True."""
        workflow = create_report_workflow(report_type="sales", include_charts=True)

        analyze_step = workflow.get_step("analyze")
        assert analyze_step is not None
        assert "charts" in analyze_step.next_steps

    def test_analyze_routes_to_format_when_no_charts(self):
        """Analyze step routes to format when include_charts is False."""
        workflow = create_report_workflow(report_type="sales", include_charts=False)

        analyze_step = workflow.get_step("analyze")
        assert analyze_step is not None
        assert "format" in analyze_step.next_steps

    def test_entry_step_is_fetch(self):
        """Report workflow starts with fetch step."""
        workflow = create_report_workflow(report_type="sales")

        assert workflow.entry_step == "fetch"

    def test_report_type_in_tags(self):
        """Report workflow includes report type in tags."""
        workflow = create_report_workflow(report_type="inventory")

        assert "inventory" in workflow.tags


# =============================================================================
# Vendor Evaluation Workflow Tests
# =============================================================================


class TestCreateVendorEvaluationWorkflow:
    """Tests for create_vendor_evaluation_workflow factory."""

    def test_creates_valid_workflow_definition(self):
        """Vendor evaluation workflow creates a valid WorkflowDefinition."""
        workflow = create_vendor_evaluation_workflow(vendor_name="Acme Corp")

        assert isinstance(workflow, WorkflowDefinition)
        assert "vendor_eval_acme_corp" in workflow.id

    def test_vendor_name_in_workflow_name(self):
        """Vendor evaluation workflow includes vendor name."""
        workflow = create_vendor_evaluation_workflow(vendor_name="Test Vendor")

        assert "Test Vendor" in workflow.name

    def test_default_evaluation_criteria(self):
        """Vendor evaluation workflow uses default criteria."""
        workflow = create_vendor_evaluation_workflow(vendor_name="Test")

        criteria = workflow.inputs["criteria"]
        assert "pricing" in criteria
        assert "reliability" in criteria
        assert "support" in criteria
        assert "integration" in criteria
        assert "scalability" in criteria

    def test_custom_evaluation_criteria(self):
        """Vendor evaluation workflow accepts custom criteria."""
        workflow = create_vendor_evaluation_workflow(
            vendor_name="Test",
            evaluation_criteria=["price", "quality", "delivery"],
        )

        assert workflow.inputs["criteria"] == ["price", "quality", "delivery"]

    def test_budget_range_stored(self):
        """Vendor evaluation workflow stores budget range."""
        workflow = create_vendor_evaluation_workflow(
            vendor_name="Test",
            budget_range="$10k-$50k",
        )

        assert workflow.inputs["budget_range"] == "$10k-$50k"

    def test_has_debate_step(self):
        """Vendor evaluation workflow has debate step."""
        workflow = create_vendor_evaluation_workflow(vendor_name="Test")

        step_ids = [s.id for s in workflow.steps]
        assert "debate" in step_ids

    def test_debate_step_has_agents_config(self):
        """Vendor evaluation debate step has agents configured."""
        workflow = create_vendor_evaluation_workflow(vendor_name="Test")

        debate_step = workflow.get_step("debate")
        assert debate_step is not None
        assert "agents" in debate_step.config

    def test_require_approval_includes_review_step(self):
        """Vendor evaluation includes review step when require_approval is True."""
        workflow = create_vendor_evaluation_workflow(
            vendor_name="Test",
            require_approval=True,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "review" in step_ids

    def test_require_approval_false_skips_review(self):
        """Vendor evaluation recommend routes to store when require_approval is False."""
        workflow = create_vendor_evaluation_workflow(
            vendor_name="Test",
            require_approval=False,
        )

        recommend_step = workflow.get_step("recommend")
        assert recommend_step is not None
        assert "store" in recommend_step.next_steps


# =============================================================================
# Hiring Decision Workflow Tests
# =============================================================================


class TestCreateHiringDecisionWorkflow:
    """Tests for create_hiring_decision_workflow factory."""

    def test_creates_valid_workflow_definition(self):
        """Hiring decision workflow creates a valid WorkflowDefinition."""
        workflow = create_hiring_decision_workflow(
            position="Developer",
            candidate_name="Jane Doe",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "hire_developer" in workflow.id

    def test_candidate_name_in_workflow_name(self):
        """Hiring decision workflow includes candidate name."""
        workflow = create_hiring_decision_workflow(
            position="Developer",
            candidate_name="John Smith",
        )

        assert "John Smith" in workflow.name
        assert "Developer" in workflow.name

    def test_interview_notes_stored(self):
        """Hiring decision workflow stores interview notes."""
        workflow = create_hiring_decision_workflow(
            position="Developer",
            candidate_name="Test",
            interview_notes="Strong Python skills",
        )

        assert workflow.inputs["interview_notes"] == "Strong Python skills"

    def test_has_analyze_step(self):
        """Hiring decision workflow has analyze step."""
        workflow = create_hiring_decision_workflow(
            position="Developer",
            candidate_name="Test",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "analyze" in step_ids

    def test_has_debate_step(self):
        """Hiring decision workflow has debate step."""
        workflow = create_hiring_decision_workflow(
            position="Developer",
            candidate_name="Test",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "debate" in step_ids

    def test_has_risks_step(self):
        """Hiring decision workflow has risk assessment step."""
        workflow = create_hiring_decision_workflow(
            position="Developer",
            candidate_name="Test",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "risks" in step_ids

    def test_has_approval_step(self):
        """Hiring decision workflow has approval step."""
        workflow = create_hiring_decision_workflow(
            position="Developer",
            candidate_name="Test",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "approval" in step_ids

    def test_hiring_tag_present(self):
        """Hiring decision workflow has hiring tag."""
        workflow = create_hiring_decision_workflow(
            position="Developer",
            candidate_name="Test",
        )

        assert "hiring" in workflow.tags
        assert "hr" in workflow.tags


# =============================================================================
# Budget Allocation Workflow Tests
# =============================================================================


class TestCreateBudgetAllocationWorkflow:
    """Tests for create_budget_allocation_workflow factory."""

    def test_creates_valid_workflow_definition(self):
        """Budget allocation workflow creates a valid WorkflowDefinition."""
        workflow = create_budget_allocation_workflow(
            department="Engineering",
            total_budget=500000,
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "budget_engineering" in workflow.id

    def test_department_in_workflow_name(self):
        """Budget allocation workflow includes department in name."""
        workflow = create_budget_allocation_workflow(
            department="Marketing",
            total_budget=100000,
        )

        assert "Marketing" in workflow.name

    def test_budget_amount_in_description(self):
        """Budget allocation workflow includes budget in description."""
        workflow = create_budget_allocation_workflow(
            department="HR",
            total_budget=250000,
        )

        assert "$250,000" in workflow.description

    def test_has_accounting_category(self):
        """Budget allocation workflow has ACCOUNTING category."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        assert workflow.category == WorkflowCategory.ACCOUNTING

    def test_default_categories(self):
        """Budget allocation workflow uses default categories."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        categories = workflow.inputs["categories"]
        assert "operations" in categories
        assert "growth" in categories
        assert "maintenance" in categories
        assert "innovation" in categories

    def test_custom_categories(self):
        """Budget allocation workflow accepts custom categories."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
            categories=["infrastructure", "training"],
        )

        assert workflow.inputs["categories"] == ["infrastructure", "training"]

    def test_has_debate_step(self):
        """Budget allocation workflow has debate step."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "debate" in step_ids

    def test_has_cfo_review_step(self):
        """Budget allocation workflow has CFO review step."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        review_step = workflow.get_step("review")
        assert review_step is not None
        assert "CFO" in review_step.name


# =============================================================================
# Performance Review Workflow Tests
# =============================================================================


class TestCreatePerformanceReviewWorkflow:
    """Tests for create_performance_review_workflow factory."""

    def test_creates_valid_workflow_definition(self):
        """Performance review workflow creates a valid WorkflowDefinition."""
        workflow = create_performance_review_workflow(
            employee_name="Jane Doe",
            role="Senior Developer",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "perf_review" in workflow.id

    def test_employee_name_in_workflow_name(self):
        """Performance review workflow includes employee name."""
        workflow = create_performance_review_workflow(
            employee_name="John Smith",
            role="Manager",
        )

        assert "John Smith" in workflow.name

    def test_default_review_period(self):
        """Performance review workflow has default review period."""
        workflow = create_performance_review_workflow(
            employee_name="Test",
            role="Developer",
        )

        assert workflow.inputs["review_period"] == "Q4 2024"

    def test_self_assessment_stored(self):
        """Performance review workflow stores self assessment."""
        workflow = create_performance_review_workflow(
            employee_name="Test",
            role="Developer",
            self_assessment="Exceeded goals",
        )

        assert workflow.inputs["self_assessment"] == "Exceeded goals"

    def test_has_multi_perspective_debate(self):
        """Performance review workflow has debate step."""
        workflow = create_performance_review_workflow(
            employee_name="Test",
            role="Developer",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "debate" in step_ids

    def test_has_strengths_step(self):
        """Performance review workflow has strengths identification step."""
        workflow = create_performance_review_workflow(
            employee_name="Test",
            role="Developer",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "strengths" in step_ids

    def test_has_development_step(self):
        """Performance review workflow has development areas step."""
        workflow = create_performance_review_workflow(
            employee_name="Test",
            role="Developer",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "development" in step_ids

    def test_has_performance_tag(self):
        """Performance review workflow has performance tag."""
        workflow = create_performance_review_workflow(
            employee_name="Test",
            role="Developer",
        )

        assert "performance" in workflow.tags


# =============================================================================
# Feature Prioritization Workflow Tests
# =============================================================================


class TestCreateFeaturePrioritizationWorkflow:
    """Tests for create_feature_prioritization_workflow factory."""

    def test_creates_valid_workflow_definition(self, sample_features):
        """Feature prioritization workflow creates a valid WorkflowDefinition."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        assert isinstance(workflow, WorkflowDefinition)
        assert "feature_priority" in workflow.id

    def test_feature_count_in_name(self, sample_features):
        """Feature prioritization workflow includes feature count in name."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        assert str(len(sample_features)) in workflow.name

    def test_default_scoring_criteria(self, sample_features):
        """Feature prioritization workflow uses default scoring criteria."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        criteria = workflow.inputs["criteria"]
        assert "impact" in criteria
        assert "effort" in criteria
        assert "urgency" in criteria
        assert "dependencies" in criteria

    def test_custom_scoring_criteria(self, sample_features):
        """Feature prioritization workflow accepts custom scoring criteria."""
        workflow = create_feature_prioritization_workflow(
            features=sample_features,
            scoring_criteria=["value", "risk", "complexity"],
        )

        assert workflow.inputs["criteria"] == ["value", "risk", "complexity"]

    def test_constraints_stored(self, sample_features):
        """Feature prioritization workflow stores constraints."""
        workflow = create_feature_prioritization_workflow(
            features=sample_features,
            constraints=["2 developers available", "Q2 deadline"],
        )

        assert "2 developers available" in workflow.inputs["constraints"]
        assert "Q2 deadline" in workflow.inputs["constraints"]

    def test_has_score_step(self, sample_features):
        """Feature prioritization workflow has scoring step."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        step_ids = [s.id for s in workflow.steps]
        assert "score" in step_ids

    def test_has_rank_step(self, sample_features):
        """Feature prioritization workflow has ranking step."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        step_ids = [s.id for s in workflow.steps]
        assert "rank" in step_ids

    def test_has_prioritization_tag(self, sample_features):
        """Feature prioritization workflow has prioritization tag."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        assert "prioritization" in workflow.tags


# =============================================================================
# Sprint Planning Workflow Tests
# =============================================================================


class TestCreateSprintPlanningWorkflow:
    """Tests for create_sprint_planning_workflow factory."""

    def test_creates_valid_workflow_definition(self, sample_backlog):
        """Sprint planning workflow creates a valid WorkflowDefinition."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 24",
            backlog_items=sample_backlog,
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "sprint_sprint_24" in workflow.id

    def test_sprint_name_in_workflow_name(self, sample_backlog):
        """Sprint planning workflow includes sprint name."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 25",
            backlog_items=sample_backlog,
        )

        assert "Sprint 25" in workflow.name

    def test_default_team_size(self, sample_backlog):
        """Sprint planning workflow defaults to team size of 5."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
        )

        assert workflow.inputs["team_size"] == 5

    def test_velocity_stored_when_provided(self, sample_backlog):
        """Sprint planning workflow stores velocity when provided."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
            velocity=32,
        )

        assert workflow.inputs["velocity"] == 32

    def test_has_capacity_step(self, sample_backlog):
        """Sprint planning workflow has capacity calculation step."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "capacity" in step_ids

    def test_has_estimate_step(self, sample_backlog):
        """Sprint planning workflow has estimate step."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "estimate" in step_ids

    def test_has_agile_tag(self, sample_backlog):
        """Sprint planning workflow has agile tag."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
        )

        assert "agile" in workflow.tags
        assert "sprint" in workflow.tags


# =============================================================================
# Tool Selection Workflow Tests
# =============================================================================


class TestCreateToolSelectionWorkflow:
    """Tests for create_tool_selection_workflow factory."""

    def test_creates_valid_workflow_definition(self, sample_candidates):
        """Tool selection workflow creates a valid WorkflowDefinition."""
        workflow = create_tool_selection_workflow(
            category="Project Management",
            candidates=sample_candidates,
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "tool_select_project_management" in workflow.id

    def test_category_in_workflow_name(self, sample_candidates):
        """Tool selection workflow includes category in name."""
        workflow = create_tool_selection_workflow(
            category="CRM",
            candidates=sample_candidates,
        )

        assert "CRM" in workflow.name

    def test_candidate_count_in_description(self, sample_candidates):
        """Tool selection workflow includes candidate count in description."""
        workflow = create_tool_selection_workflow(
            category="Test",
            candidates=sample_candidates,
        )

        assert str(len(sample_candidates)) in workflow.description

    def test_has_research_step(self, sample_candidates):
        """Tool selection workflow has research step."""
        workflow = create_tool_selection_workflow(
            category="Test",
            candidates=sample_candidates,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "research" in step_ids

    def test_has_cost_analysis_step(self, sample_candidates):
        """Tool selection workflow has cost analysis step."""
        workflow = create_tool_selection_workflow(
            category="Test",
            candidates=sample_candidates,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "cost" in step_ids

    def test_has_score_matrix_step(self, sample_candidates):
        """Tool selection workflow has score matrix step."""
        workflow = create_tool_selection_workflow(
            category="Test",
            candidates=sample_candidates,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "score" in step_ids

    def test_requirements_stored(self, sample_candidates):
        """Tool selection workflow stores requirements."""
        workflow = create_tool_selection_workflow(
            category="Test",
            candidates=sample_candidates,
            requirements=["API access", "SSO support"],
        )

        assert "API access" in workflow.inputs["requirements"]


# =============================================================================
# Contract Review Workflow Tests
# =============================================================================


class TestCreateContractReviewWorkflow:
    """Tests for create_contract_review_workflow factory."""

    def test_creates_valid_workflow_definition(self):
        """Contract review workflow creates a valid WorkflowDefinition."""
        workflow = create_contract_review_workflow(
            contract_type="SaaS Agreement",
            counterparty="Vendor Corp",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "contract_saas_agreement" in workflow.id

    def test_counterparty_in_workflow_name(self):
        """Contract review workflow includes counterparty in name."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Partner Inc",
        )

        assert "Partner Inc" in workflow.name

    def test_default_key_terms(self):
        """Contract review workflow uses default key terms."""
        workflow = create_contract_review_workflow(
            contract_type="Agreement",
            counterparty="Test",
        )

        terms = workflow.inputs["key_terms"]
        assert "liability" in terms
        assert "termination" in terms
        assert "IP" in terms
        assert "confidentiality" in terms

    def test_custom_key_terms(self):
        """Contract review workflow accepts custom key terms."""
        workflow = create_contract_review_workflow(
            contract_type="Agreement",
            counterparty="Test",
            key_terms=["SLA", "data ownership"],
        )

        assert workflow.inputs["key_terms"] == ["SLA", "data ownership"]

    def test_has_parse_step(self):
        """Contract review workflow has parse step."""
        workflow = create_contract_review_workflow(
            contract_type="Agreement",
            counterparty="Test",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "parse" in step_ids

    def test_has_risks_step(self):
        """Contract review workflow has risk identification step."""
        workflow = create_contract_review_workflow(
            contract_type="Agreement",
            counterparty="Test",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "risks" in step_ids

    def test_has_negotiate_step(self):
        """Contract review workflow has negotiation points step."""
        workflow = create_contract_review_workflow(
            contract_type="Agreement",
            counterparty="Test",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "negotiate" in step_ids

    def test_has_legal_tag(self):
        """Contract review workflow has legal tag."""
        workflow = create_contract_review_workflow(
            contract_type="Agreement",
            counterparty="Test",
        )

        assert "legal" in workflow.tags
        assert "contract" in workflow.tags


# =============================================================================
# Remote Work Policy Workflow Tests
# =============================================================================


class TestCreateRemoteWorkPolicyWorkflow:
    """Tests for create_remote_work_policy_workflow factory."""

    def test_creates_valid_workflow_definition(self):
        """Remote work policy workflow creates a valid WorkflowDefinition."""
        workflow = create_remote_work_policy_workflow()

        assert isinstance(workflow, WorkflowDefinition)
        assert "remote_policy" in workflow.id

    def test_company_size_in_description(self):
        """Remote work policy workflow includes company size in description."""
        workflow = create_remote_work_policy_workflow(company_size=100)

        assert "100-person" in workflow.description

    def test_industry_in_description(self):
        """Remote work policy workflow includes industry in description."""
        workflow = create_remote_work_policy_workflow(industry="fintech")

        assert "fintech" in workflow.description

    def test_default_company_size(self):
        """Remote work policy workflow defaults to 50 employees."""
        workflow = create_remote_work_policy_workflow()

        assert workflow.inputs["company_size"] == 50

    def test_default_industry(self):
        """Remote work policy workflow defaults to tech industry."""
        workflow = create_remote_work_policy_workflow()

        assert workflow.inputs["industry"] == "tech"

    def test_has_benchmark_step(self):
        """Remote work policy workflow has benchmark step."""
        workflow = create_remote_work_policy_workflow()

        step_ids = [s.id for s in workflow.steps]
        assert "benchmark" in step_ids

    def test_has_legal_step(self):
        """Remote work policy workflow has legal considerations step."""
        workflow = create_remote_work_policy_workflow()

        step_ids = [s.id for s in workflow.steps]
        assert "legal" in step_ids

    def test_has_draft_step(self):
        """Remote work policy workflow has draft policy step."""
        workflow = create_remote_work_policy_workflow()

        step_ids = [s.id for s in workflow.steps]
        assert "draft" in step_ids

    def test_has_remote_tag(self):
        """Remote work policy workflow has remote tag."""
        workflow = create_remote_work_policy_workflow()

        assert "remote" in workflow.tags
        assert "policy" in workflow.tags


# =============================================================================
# Business Decision Workflow Tests
# =============================================================================


class TestCreateBusinessDecisionWorkflow:
    """Tests for create_business_decision_workflow factory."""

    def test_creates_valid_workflow_definition(self):
        """Business decision workflow creates a valid WorkflowDefinition."""
        workflow = create_business_decision_workflow(
            decision_topic="Should we expand to Europe?",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "decision_" in workflow.id

    def test_topic_truncated_in_name(self):
        """Business decision workflow truncates long topic in name."""
        long_topic = "A" * 100
        workflow = create_business_decision_workflow(decision_topic=long_topic)

        # Name should be truncated to 50 chars
        assert len(workflow.name) < 100

    def test_default_urgency_is_normal(self):
        """Business decision workflow defaults to normal urgency."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        assert workflow.inputs["urgency"] == "normal"

    def test_default_impact_is_medium(self):
        """Business decision workflow defaults to medium impact."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        assert workflow.inputs["impact_level"] == "medium"

    def test_high_impact_includes_approval_step(self):
        """Business decision includes approval for high impact decisions."""
        workflow = create_business_decision_workflow(
            decision_topic="Test",
            impact_level="high",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "approve" in step_ids

    def test_low_impact_skips_approval(self):
        """Business decision recommend routes to store for low impact."""
        workflow = create_business_decision_workflow(
            decision_topic="Test",
            impact_level="low",
        )

        recommend_step = workflow.get_step("recommend")
        assert recommend_step is not None
        assert "store" in recommend_step.next_steps

    def test_has_frame_step(self):
        """Business decision workflow has frame decision step."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        step_ids = [s.id for s in workflow.steps]
        assert "frame" in step_ids

    def test_has_research_step(self):
        """Business decision workflow has research step."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        step_ids = [s.id for s in workflow.steps]
        assert "research" in step_ids

    def test_has_options_step(self):
        """Business decision workflow has options generation step."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        step_ids = [s.id for s in workflow.steps]
        assert "options" in step_ids

    def test_has_strategy_tag(self):
        """Business decision workflow has strategy tag."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        assert "strategy" in workflow.tags


# =============================================================================
# Quick Convenience Function Tests
# =============================================================================


class TestQuickInvoice:
    """Tests for quick_invoice convenience function."""

    def test_creates_valid_workflow(self):
        """quick_invoice creates a valid workflow."""
        workflow = quick_invoice(
            customer="ACME Corp",
            amount=1500.00,
            description="Consulting services",
        )

        assert isinstance(workflow, WorkflowDefinition)

    def test_creates_single_item_invoice(self):
        """quick_invoice creates single item invoice."""
        workflow = quick_invoice(
            customer="Test",
            amount=1000.00,
            description="Service",
        )

        items = workflow.inputs["items"]
        assert len(items) == 1
        assert items[0]["quantity"] == 1
        assert items[0]["unit_price"] == 1000.00

    def test_send_email_default_true(self):
        """quick_invoice defaults to sending email."""
        workflow = quick_invoice(
            customer="Test",
            amount=1000.00,
            description="Service",
            send=True,
        )

        assert workflow.inputs["send_email"] is True

    def test_send_can_be_disabled(self):
        """quick_invoice can disable email sending."""
        workflow = quick_invoice(
            customer="Test",
            amount=1000.00,
            description="Service",
            send=False,
        )

        assert workflow.inputs["send_email"] is False


class TestWeeklySalesReport:
    """Tests for weekly_sales_report convenience function."""

    def test_creates_valid_workflow(self):
        """weekly_sales_report creates a valid workflow."""
        workflow = weekly_sales_report(recipients=["sales@company.com"])

        assert isinstance(workflow, WorkflowDefinition)

    def test_report_type_is_sales(self):
        """weekly_sales_report creates sales report."""
        workflow = weekly_sales_report(recipients=["test@test.com"])

        assert workflow.inputs["report_type"] == "sales"

    def test_frequency_is_weekly(self):
        """weekly_sales_report is weekly frequency."""
        workflow = weekly_sales_report(recipients=["test@test.com"])

        assert workflow.inputs["frequency"] == "weekly"

    def test_format_is_pdf(self):
        """weekly_sales_report uses PDF format."""
        workflow = weekly_sales_report(recipients=["test@test.com"])

        assert workflow.inputs["format"] == "pdf"

    def test_recipients_stored(self):
        """weekly_sales_report stores recipients."""
        workflow = weekly_sales_report(recipients=["a@a.com", "b@b.com"])

        assert "a@a.com" in workflow.inputs["recipients"]
        assert "b@b.com" in workflow.inputs["recipients"]


class TestDailyInventoryCheck:
    """Tests for daily_inventory_check convenience function."""

    def test_creates_valid_workflow(self):
        """daily_inventory_check creates a valid workflow."""
        workflow = daily_inventory_check()

        assert isinstance(workflow, WorkflowDefinition)

    def test_default_channels_is_email(self):
        """daily_inventory_check defaults to email channel."""
        workflow = daily_inventory_check()

        channels = workflow.inputs["notification_channels"]
        assert "email" in channels

    def test_slack_channel_adds_slack(self):
        """daily_inventory_check adds slack when channel provided."""
        workflow = daily_inventory_check(slack_channel="#inventory-alerts")

        channels = workflow.inputs["notification_channels"]
        assert "email" in channels
        assert "slack" in channels

    def test_auto_reorder_is_false(self):
        """daily_inventory_check does not auto-reorder."""
        workflow = daily_inventory_check()

        assert workflow.inputs["auto_reorder"] is False

    def test_threshold_is_twenty(self):
        """daily_inventory_check uses 20% threshold."""
        workflow = daily_inventory_check()

        assert workflow.inputs["alert_threshold"] == 20


class TestRenewalFollowupCampaign:
    """Tests for renewal_followup_campaign convenience function."""

    def test_creates_valid_workflow(self):
        """renewal_followup_campaign creates a valid workflow."""
        workflow = renewal_followup_campaign()

        assert isinstance(workflow, WorkflowDefinition)

    def test_followup_type_is_renewal(self):
        """renewal_followup_campaign uses renewal type."""
        workflow = renewal_followup_campaign()

        assert workflow.inputs["followup_type"] == "renewal"

    def test_days_since_contact_is_sixty(self):
        """renewal_followup_campaign uses 60 days threshold."""
        workflow = renewal_followup_campaign()

        assert workflow.inputs["days_since_contact"] == 60

    def test_channel_is_email(self):
        """renewal_followup_campaign uses email channel."""
        workflow = renewal_followup_campaign()

        assert workflow.inputs["channel"] == "email"

    def test_auto_send_is_false(self):
        """renewal_followup_campaign does not auto-send."""
        workflow = renewal_followup_campaign()

        assert workflow.inputs["auto_send"] is False


# =============================================================================
# Workflow Validation Tests
# =============================================================================


class TestWorkflowValidation:
    """Tests for workflow validation."""

    def test_all_workflows_have_unique_step_ids(self, sample_items, sample_features):
        """All factory workflows have unique step IDs."""
        workflows = [
            create_invoice_workflow("test", sample_items),
            create_followup_workflow(),
            create_inventory_alert_workflow(),
            create_report_workflow("sales"),
            create_vendor_evaluation_workflow("Test"),
            create_hiring_decision_workflow("Dev", "Jane"),
            create_budget_allocation_workflow("HR", 100000),
            create_performance_review_workflow("Jane", "Dev"),
            create_feature_prioritization_workflow(sample_features),
            create_sprint_planning_workflow("Sprint 1", ["Task"]),
            create_tool_selection_workflow("Test", ["A", "B"]),
            create_contract_review_workflow("NDA", "Partner"),
            create_remote_work_policy_workflow(),
            create_business_decision_workflow("Expand?"),
        ]

        for workflow in workflows:
            step_ids = [s.id for s in workflow.steps]
            assert len(step_ids) == len(set(step_ids)), (
                f"Duplicate step IDs in {workflow.id}: {step_ids}"
            )

    def test_all_workflows_have_valid_entry_step(self, sample_items, sample_features):
        """All factory workflows have entry step that exists in steps."""
        workflows = [
            create_invoice_workflow("test", sample_items),
            create_followup_workflow(),
            create_inventory_alert_workflow(),
            create_report_workflow("sales"),
            create_vendor_evaluation_workflow("Test"),
            create_hiring_decision_workflow("Dev", "Jane"),
            create_budget_allocation_workflow("HR", 100000),
            create_performance_review_workflow("Jane", "Dev"),
            create_feature_prioritization_workflow(sample_features),
            create_sprint_planning_workflow("Sprint 1", ["Task"]),
            create_tool_selection_workflow("Test", ["A", "B"]),
            create_contract_review_workflow("NDA", "Partner"),
            create_remote_work_policy_workflow(),
            create_business_decision_workflow("Expand?"),
        ]

        for workflow in workflows:
            step_ids = [s.id for s in workflow.steps]
            assert workflow.entry_step in step_ids, (
                f"Entry step '{workflow.entry_step}' not in steps for {workflow.id}"
            )

    def test_all_next_steps_reference_valid_steps(self, sample_items):
        """All next_steps references point to valid step IDs."""
        workflow = create_invoice_workflow("test", sample_items, send_email=True)
        step_ids = {s.id for s in workflow.steps}

        for step in workflow.steps:
            for next_step in step.next_steps:
                assert next_step in step_ids, f"Invalid next_step '{next_step}' in step '{step.id}'"

    def test_all_workflows_have_terminal_step(self, sample_items):
        """All workflows have at least one terminal step (no next_steps)."""
        workflow = create_invoice_workflow("test", sample_items)

        terminal_steps = [s for s in workflow.steps if not s.next_steps]
        assert len(terminal_steps) >= 1, "Workflow should have at least one terminal step"


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_main_factories_exported(self):
        """All main factory functions are exported."""
        from aragora.workflow.templates.sme._factories import __all__

        assert "create_invoice_workflow" in __all__
        assert "create_followup_workflow" in __all__
        assert "create_inventory_alert_workflow" in __all__
        assert "create_report_workflow" in __all__

    def test_all_sme_decision_templates_exported(self):
        """All SME decision template factories are exported."""
        from aragora.workflow.templates.sme._factories import __all__

        assert "create_vendor_evaluation_workflow" in __all__
        assert "create_hiring_decision_workflow" in __all__
        assert "create_budget_allocation_workflow" in __all__
        assert "create_business_decision_workflow" in __all__
        assert "create_performance_review_workflow" in __all__
        assert "create_feature_prioritization_workflow" in __all__
        assert "create_sprint_planning_workflow" in __all__
        assert "create_tool_selection_workflow" in __all__
        assert "create_contract_review_workflow" in __all__
        assert "create_remote_work_policy_workflow" in __all__

    def test_all_convenience_functions_exported(self):
        """All convenience functions are exported."""
        from aragora.workflow.templates.sme._factories import __all__

        assert "quick_invoice" in __all__
        assert "weekly_sales_report" in __all__
        assert "daily_inventory_check" in __all__
        assert "renewal_followup_campaign" in __all__


# =============================================================================
# Step Configuration Tests
# =============================================================================


class TestStepConfiguration:
    """Tests for step configuration details."""

    def test_agent_steps_have_agent_type(self, sample_items):
        """Agent steps have agent_type in config."""
        workflow = create_invoice_workflow("test", sample_items)

        agent_steps = [s for s in workflow.steps if s.step_type == "agent"]
        for step in agent_steps:
            assert "agent_type" in step.config, f"Step {step.id} missing agent_type"

    def test_agent_steps_have_prompt_template(self, sample_items):
        """Agent steps have prompt_template in config."""
        workflow = create_invoice_workflow("test", sample_items)

        agent_steps = [s for s in workflow.steps if s.step_type == "agent"]
        for step in agent_steps:
            assert "prompt_template" in step.config, f"Step {step.id} missing prompt_template"

    def test_memory_write_steps_have_collection(self, sample_items):
        """Memory write steps have collection in config."""
        workflow = create_invoice_workflow("test", sample_items)

        memory_steps = [s for s in workflow.steps if s.step_type == "memory_write"]
        for step in memory_steps:
            assert "collection" in step.config, f"Step {step.id} missing collection"

    def test_debate_steps_have_agents_config(self):
        """Debate steps have agents configured."""
        workflow = create_vendor_evaluation_workflow("Test Vendor")

        debate_steps = [s for s in workflow.steps if s.step_type == "debate"]
        for step in debate_steps:
            assert "agents" in step.config, f"Step {step.id} missing agents"

    def test_debate_steps_have_rounds(self):
        """Debate steps have rounds configured."""
        workflow = create_vendor_evaluation_workflow("Test Vendor")

        debate_steps = [s for s in workflow.steps if s.step_type == "debate"]
        for step in debate_steps:
            assert "rounds" in step.config, f"Step {step.id} missing rounds"

    def test_human_checkpoint_steps_have_checkpoint_type(self):
        """Human checkpoint steps have checkpoint_type configured."""
        workflow = create_followup_workflow(auto_send=False)

        checkpoint_steps = [s for s in workflow.steps if s.step_type == "human_checkpoint"]
        for step in checkpoint_steps:
            assert "checkpoint_type" in step.config, f"Step {step.id} missing checkpoint_type"

    def test_task_steps_have_handler(self, sample_items):
        """Task steps have handler in config."""
        workflow = create_invoice_workflow("test", sample_items)

        task_steps = [s for s in workflow.steps if s.step_type == "task"]
        for step in task_steps:
            assert "handler" in step.config, f"Step {step.id} missing handler"


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_items_list(self):
        """Invoice workflow handles empty items list."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=[],
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.inputs["items"] == []

    def test_empty_features_list(self):
        """Feature prioritization handles empty features list."""
        workflow = create_feature_prioritization_workflow(features=[])

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.inputs["features"] == []

    def test_single_feature(self):
        """Feature prioritization handles single feature."""
        workflow = create_feature_prioritization_workflow(features=["Only Feature"])

        assert isinstance(workflow, WorkflowDefinition)
        assert "1 features" in workflow.name

    def test_very_long_customer_id(self):
        """Invoice workflow handles very long customer ID."""
        long_id = "customer_" + "x" * 200
        workflow = create_invoice_workflow(
            customer_id=long_id,
            items=[{"name": "Test", "quantity": 1, "unit_price": 100}],
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert long_id in workflow.inputs["customer_id"]

    def test_special_characters_in_vendor_name(self):
        """Vendor evaluation handles special characters in name."""
        workflow = create_vendor_evaluation_workflow(vendor_name="Test & Company (LLC)")

        assert isinstance(workflow, WorkflowDefinition)
        # ID should be sanitized
        assert "vendor_eval_" in workflow.id

    def test_zero_budget(self):
        """Budget allocation handles zero budget."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=0,
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.inputs["total_budget"] == 0

    def test_negative_budget(self):
        """Budget allocation accepts negative budget (for testing)."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=-1000,
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.inputs["total_budget"] == -1000

    def test_very_high_tax_rate(self):
        """Invoice workflow accepts high tax rate."""
        workflow = create_invoice_workflow(
            customer_id="test",
            items=[{"name": "Test", "quantity": 1, "unit_price": 100}],
            tax_rate=1.0,  # 100% tax
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.inputs["tax_rate"] == 1.0

    def test_zero_team_size(self):
        """Sprint planning handles zero team size."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=["Task"],
            team_size=0,
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.inputs["team_size"] == 0

    def test_unicode_in_workflow_inputs(self):
        """Workflows handle unicode characters."""
        workflow = create_hiring_decision_workflow(
            position="Developpeur Senior",
            candidate_name="Jean-Pierre Francois",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.inputs["candidate_name"] == "Jean-Pierre Francois"


# =============================================================================
# Workflow ID Format and Structure Tests
# =============================================================================


class TestWorkflowIdFormat:
    """Tests for workflow ID format and structure."""

    def test_invoice_workflow_id_has_timestamp(self, sample_items):
        """Invoice workflow ID contains timestamp component."""
        workflow = create_invoice_workflow("cust1", sample_items)

        # ID should have format: invoice_cust1_YYYYMMDD_HHMMSS
        assert workflow.id.startswith("invoice_cust1_")
        # Timestamp portion should be numeric (date and time)
        parts = workflow.id.split("_")
        assert len(parts) >= 4
        assert parts[2].isdigit()  # Date part
        assert parts[3].isdigit()  # Time part

    def test_followup_workflow_id_format(self):
        """Followup workflow ID has correct format."""
        workflow = create_followup_workflow(followup_type="renewal")

        # ID should have format: followup_renewal_YYYYMMDD_HHMMSS
        assert workflow.id.startswith("followup_renewal_")
        parts = workflow.id.split("_")
        assert len(parts) >= 4

    def test_report_workflow_id_format(self):
        """Report workflow ID has correct format."""
        workflow = create_report_workflow("financial")

        # ID should have format: report_financial_YYYYMMDD_HHMMSS
        assert workflow.id.startswith("report_financial_")

    def test_different_customers_have_different_ids(self, sample_items):
        """Different customer IDs produce different workflow IDs."""
        workflow1 = create_invoice_workflow("customer_a", sample_items)
        workflow2 = create_invoice_workflow("customer_b", sample_items)

        assert workflow1.id != workflow2.id
        assert "customer_a" in workflow1.id
        assert "customer_b" in workflow2.id

    def test_different_report_types_have_different_ids(self):
        """Different report types produce different workflow IDs."""
        workflow1 = create_report_workflow("sales")
        workflow2 = create_report_workflow("inventory")

        assert "sales" in workflow1.id
        assert "inventory" in workflow2.id

    def test_vendor_evaluation_id_sanitizes_spaces(self):
        """Vendor evaluation workflow ID sanitizes spaces in vendor name."""
        workflow = create_vendor_evaluation_workflow(vendor_name="Acme Corp Inc")

        # Spaces should be replaced with underscores
        assert "acme_corp_inc" in workflow.id.lower()
        assert " " not in workflow.id
