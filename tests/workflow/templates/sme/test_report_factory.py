"""
Comprehensive Tests for SME Report and Business Decision Workflow Factories.

Tests coverage for:
- Report workflow factory (create_report_workflow)
- Budget allocation workflow (create_budget_allocation_workflow)
- Feature prioritization workflow (create_feature_prioritization_workflow)
- Sprint planning workflow (create_sprint_planning_workflow)
- Contract review workflow (create_contract_review_workflow)
- Business decision workflow (create_business_decision_workflow)
- Weekly sales report convenience function
- Step validation and linking
- Parameter variations and edge cases
- Workflow serialization/deserialization
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytest

from aragora.workflow.templates.sme.report_factory import (
    create_budget_allocation_workflow,
    create_business_decision_workflow,
    create_contract_review_workflow,
    create_feature_prioritization_workflow,
    create_report_workflow,
    create_sprint_planning_workflow,
    weekly_sales_report,
)
from aragora.workflow.types import (
    StepDefinition,
    WorkflowCategory,
    WorkflowDefinition,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_features() -> list[str]:
    """Sample features list for testing."""
    return ["Dark mode", "API v2", "Mobile app", "Export to PDF"]


@pytest.fixture
def sample_backlog() -> list[str]:
    """Sample backlog items for sprint planning."""
    return ["User authentication", "Dashboard redesign", "API refactor", "Bug fixes"]


@pytest.fixture
def sample_recipients() -> list[str]:
    """Sample email recipients for reports."""
    return ["ceo@company.com", "cfo@company.com", "sales@company.com"]


@pytest.fixture
def sample_constraints() -> list[str]:
    """Sample constraints for planning workflows."""
    return ["2 developers available", "Q2 deadline", "Budget limit $50k"]


@pytest.fixture
def sample_key_terms() -> list[str]:
    """Sample key terms for contract review."""
    return ["SLA", "data ownership", "termination clause", "liability"]


# =============================================================================
# Report Workflow Tests
# =============================================================================


class TestCreateReportWorkflow:
    """Tests for create_report_workflow factory."""

    def test_creates_valid_workflow_definition(self) -> None:
        """Report workflow creates a valid WorkflowDefinition."""
        workflow = create_report_workflow(report_type="sales")

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.id is not None
        assert "report_sales" in workflow.id

    def test_report_type_in_workflow_name(self) -> None:
        """Report workflow includes report type in name."""
        workflow = create_report_workflow(report_type="financial")

        assert "Financial" in workflow.name

    def test_frequency_in_workflow_name(self) -> None:
        """Report workflow includes frequency in name."""
        workflow = create_report_workflow(report_type="sales", frequency="monthly")

        assert "monthly" in workflow.name

    def test_workflow_has_general_category(self) -> None:
        """Report workflow has GENERAL category."""
        workflow = create_report_workflow(report_type="sales")

        assert workflow.category == WorkflowCategory.GENERAL

    def test_workflow_has_correct_tags(self) -> None:
        """Report workflow has expected tags."""
        workflow = create_report_workflow(report_type="inventory")

        assert "sme" in workflow.tags
        assert "reports" in workflow.tags
        assert "analytics" in workflow.tags
        assert "inventory" in workflow.tags

    def test_default_frequency_is_weekly(self) -> None:
        """Report workflow defaults to weekly frequency."""
        workflow = create_report_workflow(report_type="sales")

        assert workflow.inputs["frequency"] == "weekly"

    def test_default_date_range_is_last_week(self) -> None:
        """Report workflow defaults to last_week date range."""
        workflow = create_report_workflow(report_type="sales")

        assert workflow.inputs["date_range"] == "last_week"

    def test_default_format_is_pdf(self) -> None:
        """Report workflow defaults to PDF format."""
        workflow = create_report_workflow(report_type="sales")

        assert workflow.inputs["format"] == "pdf"

    def test_default_include_charts_is_true(self) -> None:
        """Report workflow defaults to including charts."""
        workflow = create_report_workflow(report_type="sales")

        assert workflow.inputs["include_charts"] is True

    def test_default_include_comparison_is_true(self) -> None:
        """Report workflow defaults to including comparison."""
        workflow = create_report_workflow(report_type="sales")

        assert workflow.inputs["comparison"] is True

    def test_custom_frequency_applied(self) -> None:
        """Report workflow applies custom frequency."""
        workflow = create_report_workflow(report_type="sales", frequency="daily")

        assert workflow.inputs["frequency"] == "daily"

    def test_custom_date_range_applied(self) -> None:
        """Report workflow applies custom date range."""
        workflow = create_report_workflow(report_type="sales", date_range="last_month")

        assert workflow.inputs["date_range"] == "last_month"

    def test_custom_format_applied(self) -> None:
        """Report workflow applies custom format."""
        workflow = create_report_workflow(report_type="sales", format="excel")

        assert workflow.inputs["format"] == "excel"

    def test_custom_recipients_stored(self, sample_recipients: list[str]) -> None:
        """Report workflow stores custom recipients."""
        workflow = create_report_workflow(report_type="sales", recipients=sample_recipients)

        assert workflow.inputs["recipients"] == sample_recipients

    def test_none_recipients_defaults_to_empty_list(self) -> None:
        """Report workflow defaults to empty recipients list when None."""
        workflow = create_report_workflow(report_type="sales", recipients=None)

        assert workflow.inputs["recipients"] == []

    def test_include_charts_false_applied(self) -> None:
        """Report workflow applies include_charts=False."""
        workflow = create_report_workflow(report_type="sales", include_charts=False)

        assert workflow.inputs["include_charts"] is False

    def test_include_comparison_false_applied(self) -> None:
        """Report workflow applies include_comparison=False."""
        workflow = create_report_workflow(report_type="sales", include_comparison=False)

        assert workflow.inputs["comparison"] is False

    def test_has_fetch_step(self) -> None:
        """Report workflow has fetch data step."""
        workflow = create_report_workflow(report_type="sales")

        step_ids = [s.id for s in workflow.steps]
        assert "fetch" in step_ids

    def test_has_analyze_step(self) -> None:
        """Report workflow has analyze data step."""
        workflow = create_report_workflow(report_type="sales")

        step_ids = [s.id for s in workflow.steps]
        assert "analyze" in step_ids

    def test_has_format_step(self) -> None:
        """Report workflow has format report step."""
        workflow = create_report_workflow(report_type="sales")

        step_ids = [s.id for s in workflow.steps]
        assert "format" in step_ids

    def test_has_generate_step(self) -> None:
        """Report workflow has generate file step."""
        workflow = create_report_workflow(report_type="sales")

        step_ids = [s.id for s in workflow.steps]
        assert "generate" in step_ids

    def test_has_deliver_step(self) -> None:
        """Report workflow has deliver step."""
        workflow = create_report_workflow(report_type="sales")

        step_ids = [s.id for s in workflow.steps]
        assert "deliver" in step_ids

    def test_has_log_step(self) -> None:
        """Report workflow has log completion step."""
        workflow = create_report_workflow(report_type="sales")

        step_ids = [s.id for s in workflow.steps]
        assert "log" in step_ids

    def test_include_charts_adds_charts_steps(self) -> None:
        """Report workflow includes charts steps when include_charts is True."""
        workflow = create_report_workflow(report_type="sales", include_charts=True)

        step_ids = [s.id for s in workflow.steps]
        assert "charts" in step_ids
        assert "render" in step_ids

    def test_include_charts_false_skips_charts_steps(self) -> None:
        """Report workflow skips charts steps when include_charts is False."""
        workflow = create_report_workflow(report_type="sales", include_charts=False)

        step_ids = [s.id for s in workflow.steps]
        # Charts step still exists but analyze routes directly to format
        analyze_step = workflow.get_step("analyze")
        assert analyze_step is not None
        assert "format" in analyze_step.next_steps

    def test_analyze_routes_to_charts_when_include_charts(self) -> None:
        """Analyze step routes to charts when include_charts is True."""
        workflow = create_report_workflow(report_type="sales", include_charts=True)

        analyze_step = workflow.get_step("analyze")
        assert analyze_step is not None
        assert "charts" in analyze_step.next_steps

    def test_analyze_routes_to_format_when_no_charts(self) -> None:
        """Analyze step routes to format when include_charts is False."""
        workflow = create_report_workflow(report_type="sales", include_charts=False)

        analyze_step = workflow.get_step("analyze")
        assert analyze_step is not None
        assert "format" in analyze_step.next_steps

    def test_entry_step_is_fetch(self) -> None:
        """Report workflow starts with fetch step."""
        workflow = create_report_workflow(report_type="sales")

        assert workflow.entry_step == "fetch"

    def test_fetch_step_is_parallel_type(self) -> None:
        """Fetch step is parallel type."""
        workflow = create_report_workflow(report_type="sales")

        fetch_step = workflow.get_step("fetch")
        assert fetch_step is not None
        assert fetch_step.step_type == "parallel"
        assert "sub_steps" in fetch_step.config

    def test_deliver_step_is_parallel_type(self) -> None:
        """Deliver step is parallel type."""
        workflow = create_report_workflow(report_type="sales")

        deliver_step = workflow.get_step("deliver")
        assert deliver_step is not None
        assert deliver_step.step_type == "parallel"

    def test_log_step_is_memory_write_type(self) -> None:
        """Log step is memory_write type."""
        workflow = create_report_workflow(report_type="sales")

        log_step = workflow.get_step("log")
        assert log_step is not None
        assert log_step.step_type == "memory_write"
        assert log_step.config["collection"] == "report_logs"

    def test_generate_step_has_format_config(self) -> None:
        """Generate step has format in config."""
        workflow = create_report_workflow(report_type="sales", format="excel")

        generate_step = workflow.get_step("generate")
        assert generate_step is not None
        assert generate_step.config["format"] == "excel"


# =============================================================================
# Weekly Sales Report Tests
# =============================================================================


class TestWeeklySalesReport:
    """Tests for weekly_sales_report convenience function."""

    def test_creates_valid_workflow(self, sample_recipients: list[str]) -> None:
        """weekly_sales_report creates a valid workflow."""
        workflow = weekly_sales_report(recipients=sample_recipients)

        assert isinstance(workflow, WorkflowDefinition)

    def test_report_type_is_sales(self, sample_recipients: list[str]) -> None:
        """weekly_sales_report creates sales report."""
        workflow = weekly_sales_report(recipients=sample_recipients)

        assert workflow.inputs["report_type"] == "sales"

    def test_frequency_is_weekly(self, sample_recipients: list[str]) -> None:
        """weekly_sales_report is weekly frequency."""
        workflow = weekly_sales_report(recipients=sample_recipients)

        assert workflow.inputs["frequency"] == "weekly"

    def test_date_range_is_last_week(self, sample_recipients: list[str]) -> None:
        """weekly_sales_report uses last_week date range."""
        workflow = weekly_sales_report(recipients=sample_recipients)

        assert workflow.inputs["date_range"] == "last_week"

    def test_format_is_pdf(self, sample_recipients: list[str]) -> None:
        """weekly_sales_report uses PDF format."""
        workflow = weekly_sales_report(recipients=sample_recipients)

        assert workflow.inputs["format"] == "pdf"

    def test_recipients_stored(self, sample_recipients: list[str]) -> None:
        """weekly_sales_report stores recipients."""
        workflow = weekly_sales_report(recipients=sample_recipients)

        assert workflow.inputs["recipients"] == sample_recipients

    def test_empty_recipients_list(self) -> None:
        """weekly_sales_report handles empty recipients list."""
        workflow = weekly_sales_report(recipients=[])

        assert workflow.inputs["recipients"] == []


# =============================================================================
# Budget Allocation Workflow Tests
# =============================================================================


class TestCreateBudgetAllocationWorkflow:
    """Tests for create_budget_allocation_workflow factory."""

    def test_creates_valid_workflow_definition(self) -> None:
        """Budget allocation workflow creates a valid WorkflowDefinition."""
        workflow = create_budget_allocation_workflow(
            department="Engineering",
            total_budget=500000,
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "budget_engineering" in workflow.id

    def test_department_in_workflow_name(self) -> None:
        """Budget allocation workflow includes department in name."""
        workflow = create_budget_allocation_workflow(
            department="Marketing",
            total_budget=100000,
        )

        assert "Marketing" in workflow.name

    def test_fiscal_year_in_workflow_name(self) -> None:
        """Budget allocation workflow includes fiscal year in name."""
        workflow = create_budget_allocation_workflow(
            department="HR",
            total_budget=100000,
            fiscal_year="2025",
        )

        assert "FY2025" in workflow.name

    def test_default_fiscal_year_is_current_year(self) -> None:
        """Budget allocation defaults to current fiscal year."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        current_year = datetime.now().strftime("%Y")
        assert workflow.inputs["fiscal_year"] == current_year

    def test_budget_amount_in_description(self) -> None:
        """Budget allocation workflow includes budget in description."""
        workflow = create_budget_allocation_workflow(
            department="HR",
            total_budget=250000,
        )

        assert "$250,000" in workflow.description

    def test_has_accounting_category(self) -> None:
        """Budget allocation workflow has ACCOUNTING category."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        assert workflow.category == WorkflowCategory.ACCOUNTING

    def test_workflow_has_correct_tags(self) -> None:
        """Budget allocation workflow has expected tags."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        assert "sme" in workflow.tags
        assert "budget" in workflow.tags
        assert "finance" in workflow.tags
        assert "decision" in workflow.tags

    def test_default_categories(self) -> None:
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

    def test_custom_categories(self) -> None:
        """Budget allocation workflow accepts custom categories."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
            categories=["infrastructure", "training", "tools"],
        )

        assert workflow.inputs["categories"] == ["infrastructure", "training", "tools"]

    def test_constraints_stored(self, sample_constraints: list[str]) -> None:
        """Budget allocation workflow stores constraints."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
            constraints=sample_constraints,
        )

        assert workflow.inputs["constraints"] == sample_constraints

    def test_none_constraints_defaults_to_empty_list(self) -> None:
        """Budget allocation defaults to empty constraints list when None."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
            constraints=None,
        )

        assert workflow.inputs["constraints"] == []

    def test_has_analyze_step(self) -> None:
        """Budget allocation workflow has analyze needs step."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "analyze" in step_ids

    def test_has_historical_step(self) -> None:
        """Budget allocation workflow has historical analysis step."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "historical" in step_ids

    def test_has_debate_step(self) -> None:
        """Budget allocation workflow has debate step."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "debate" in step_ids

    def test_has_propose_step(self) -> None:
        """Budget allocation workflow has propose step."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "propose" in step_ids

    def test_has_review_step(self) -> None:
        """Budget allocation workflow has CFO review step."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        review_step = workflow.get_step("review")
        assert review_step is not None
        assert "CFO" in review_step.name

    def test_has_finalize_step(self) -> None:
        """Budget allocation workflow has finalize step."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "finalize" in step_ids

    def test_has_store_step(self) -> None:
        """Budget allocation workflow has store step."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "store" in step_ids

    def test_entry_step_is_analyze(self) -> None:
        """Budget allocation workflow starts with analyze step."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        assert workflow.entry_step == "analyze"

    def test_debate_step_has_agents_config(self) -> None:
        """Budget debate step has agents configured."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        debate_step = workflow.get_step("debate")
        assert debate_step is not None
        assert "agents" in debate_step.config
        assert len(debate_step.config["agents"]) >= 2

    def test_debate_step_has_rounds_config(self) -> None:
        """Budget debate step has rounds configured."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        debate_step = workflow.get_step("debate")
        assert debate_step is not None
        assert "rounds" in debate_step.config
        assert debate_step.config["rounds"] == 3

    def test_debate_step_has_categories_in_config(self) -> None:
        """Budget debate step has categories in config."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
            categories=["A", "B", "C"],
        )

        debate_step = workflow.get_step("debate")
        assert debate_step is not None
        assert debate_step.config["categories"] == ["A", "B", "C"]

    def test_review_step_is_human_checkpoint(self) -> None:
        """Review step is human_checkpoint type."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        review_step = workflow.get_step("review")
        assert review_step is not None
        assert review_step.step_type == "human_checkpoint"
        assert review_step.config["checkpoint_type"] == "selection"

    def test_store_step_is_memory_write(self) -> None:
        """Store step is memory_write type."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=100000,
        )

        store_step = workflow.get_step("store")
        assert store_step is not None
        assert store_step.step_type == "memory_write"
        assert store_step.config["collection"] == "budget_allocations"


# =============================================================================
# Feature Prioritization Workflow Tests
# =============================================================================


class TestCreateFeaturePrioritizationWorkflow:
    """Tests for create_feature_prioritization_workflow factory."""

    def test_creates_valid_workflow_definition(self, sample_features: list[str]) -> None:
        """Feature prioritization workflow creates a valid WorkflowDefinition."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        assert isinstance(workflow, WorkflowDefinition)
        assert "feature_priority" in workflow.id

    def test_feature_count_in_name(self, sample_features: list[str]) -> None:
        """Feature prioritization workflow includes feature count in name."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        assert str(len(sample_features)) in workflow.name

    def test_timeline_in_description(self, sample_features: list[str]) -> None:
        """Feature prioritization workflow includes timeline in description."""
        workflow = create_feature_prioritization_workflow(
            features=sample_features,
            timeline="Q3 2025",
        )

        assert "Q3 2025" in workflow.description

    def test_workflow_has_general_category(self, sample_features: list[str]) -> None:
        """Feature prioritization workflow has GENERAL category."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        assert workflow.category == WorkflowCategory.GENERAL

    def test_workflow_has_correct_tags(self, sample_features: list[str]) -> None:
        """Feature prioritization workflow has expected tags."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        assert "sme" in workflow.tags
        assert "product" in workflow.tags
        assert "prioritization" in workflow.tags
        assert "planning" in workflow.tags

    def test_default_scoring_criteria(self, sample_features: list[str]) -> None:
        """Feature prioritization workflow uses default scoring criteria."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        criteria = workflow.inputs["criteria"]
        assert "impact" in criteria
        assert "effort" in criteria
        assert "urgency" in criteria
        assert "dependencies" in criteria

    def test_custom_scoring_criteria(self, sample_features: list[str]) -> None:
        """Feature prioritization workflow accepts custom scoring criteria."""
        workflow = create_feature_prioritization_workflow(
            features=sample_features,
            scoring_criteria=["value", "risk", "complexity"],
        )

        assert workflow.inputs["criteria"] == ["value", "risk", "complexity"]

    def test_default_timeline_is_next_quarter(self, sample_features: list[str]) -> None:
        """Feature prioritization workflow defaults to next quarter timeline."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        assert workflow.inputs["timeline"] == "next quarter"

    def test_constraints_stored(
        self, sample_features: list[str], sample_constraints: list[str]
    ) -> None:
        """Feature prioritization workflow stores constraints."""
        workflow = create_feature_prioritization_workflow(
            features=sample_features,
            constraints=sample_constraints,
        )

        assert workflow.inputs["constraints"] == sample_constraints

    def test_none_constraints_defaults_to_empty_list(self, sample_features: list[str]) -> None:
        """Feature prioritization defaults to empty constraints list when None."""
        workflow = create_feature_prioritization_workflow(
            features=sample_features,
            constraints=None,
        )

        assert workflow.inputs["constraints"] == []

    def test_team_capacity_stored(self, sample_features: list[str]) -> None:
        """Feature prioritization workflow stores team capacity."""
        workflow = create_feature_prioritization_workflow(
            features=sample_features,
            team_capacity="3 developers, 1 designer",
        )

        assert workflow.inputs["team_capacity"] == "3 developers, 1 designer"

    def test_has_analyze_step(self, sample_features: list[str]) -> None:
        """Feature prioritization workflow has analyze step."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        step_ids = [s.id for s in workflow.steps]
        assert "analyze" in step_ids

    def test_has_debate_step(self, sample_features: list[str]) -> None:
        """Feature prioritization workflow has debate step."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        step_ids = [s.id for s in workflow.steps]
        assert "debate" in step_ids

    def test_has_score_step(self, sample_features: list[str]) -> None:
        """Feature prioritization workflow has scoring step."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        step_ids = [s.id for s in workflow.steps]
        assert "score" in step_ids

    def test_has_rank_step(self, sample_features: list[str]) -> None:
        """Feature prioritization workflow has ranking step."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        step_ids = [s.id for s in workflow.steps]
        assert "rank" in step_ids

    def test_has_review_step(self, sample_features: list[str]) -> None:
        """Feature prioritization workflow has review step."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        step_ids = [s.id for s in workflow.steps]
        assert "review" in step_ids

    def test_has_store_step(self, sample_features: list[str]) -> None:
        """Feature prioritization workflow has store step."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        step_ids = [s.id for s in workflow.steps]
        assert "store" in step_ids

    def test_entry_step_is_analyze(self, sample_features: list[str]) -> None:
        """Feature prioritization workflow starts with analyze step."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        assert workflow.entry_step == "analyze"

    def test_debate_step_has_agents_config(self, sample_features: list[str]) -> None:
        """Feature debate step has agents configured."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        debate_step = workflow.get_step("debate")
        assert debate_step is not None
        assert "agents" in debate_step.config

    def test_debate_step_has_criteria_config(self, sample_features: list[str]) -> None:
        """Feature debate step has criteria in config."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        debate_step = workflow.get_step("debate")
        assert debate_step is not None
        assert "criteria" in debate_step.config

    def test_review_step_is_human_checkpoint(self, sample_features: list[str]) -> None:
        """Review step is human_checkpoint type with approval."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        review_step = workflow.get_step("review")
        assert review_step is not None
        assert review_step.step_type == "human_checkpoint"
        assert review_step.config["checkpoint_type"] == "approval"

    def test_store_step_collection(self, sample_features: list[str]) -> None:
        """Store step uses feature_prioritizations collection."""
        workflow = create_feature_prioritization_workflow(features=sample_features)

        store_step = workflow.get_step("store")
        assert store_step is not None
        assert store_step.config["collection"] == "feature_prioritizations"


# =============================================================================
# Sprint Planning Workflow Tests
# =============================================================================


class TestCreateSprintPlanningWorkflow:
    """Tests for create_sprint_planning_workflow factory."""

    def test_creates_valid_workflow_definition(self, sample_backlog: list[str]) -> None:
        """Sprint planning workflow creates a valid WorkflowDefinition."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 24",
            backlog_items=sample_backlog,
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "sprint_sprint_24" in workflow.id

    def test_sprint_name_in_workflow_name(self, sample_backlog: list[str]) -> None:
        """Sprint planning workflow includes sprint name."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 25",
            backlog_items=sample_backlog,
        )

        assert "Sprint 25" in workflow.name

    def test_team_size_in_description(self, sample_backlog: list[str]) -> None:
        """Sprint planning workflow includes team size in description."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
            team_size=8,
        )

        assert "8 team members" in workflow.description

    def test_workflow_has_general_category(self, sample_backlog: list[str]) -> None:
        """Sprint planning workflow has GENERAL category."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
        )

        assert workflow.category == WorkflowCategory.GENERAL

    def test_workflow_has_correct_tags(self, sample_backlog: list[str]) -> None:
        """Sprint planning workflow has expected tags."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
        )

        assert "sme" in workflow.tags
        assert "agile" in workflow.tags
        assert "sprint" in workflow.tags
        assert "planning" in workflow.tags

    def test_default_team_size_is_five(self, sample_backlog: list[str]) -> None:
        """Sprint planning workflow defaults to team size of 5."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
        )

        assert workflow.inputs["team_size"] == 5

    def test_default_sprint_duration_is_two_weeks(self, sample_backlog: list[str]) -> None:
        """Sprint planning workflow defaults to 2 weeks duration."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
        )

        assert workflow.inputs["sprint_duration"] == "2 weeks"

    def test_custom_team_size_applied(self, sample_backlog: list[str]) -> None:
        """Sprint planning workflow applies custom team size."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
            team_size=10,
        )

        assert workflow.inputs["team_size"] == 10

    def test_custom_sprint_duration_applied(self, sample_backlog: list[str]) -> None:
        """Sprint planning workflow applies custom sprint duration."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
            sprint_duration="3 weeks",
        )

        assert workflow.inputs["sprint_duration"] == "3 weeks"

    def test_velocity_stored_when_provided(self, sample_backlog: list[str]) -> None:
        """Sprint planning workflow stores velocity when provided."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
            velocity=32,
        )

        assert workflow.inputs["velocity"] == 32

    def test_velocity_none_when_not_provided(self, sample_backlog: list[str]) -> None:
        """Sprint planning workflow has None velocity when not provided."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
        )

        assert workflow.inputs["velocity"] is None

    def test_has_capacity_step(self, sample_backlog: list[str]) -> None:
        """Sprint planning workflow has capacity calculation step."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "capacity" in step_ids

    def test_has_debate_step(self, sample_backlog: list[str]) -> None:
        """Sprint planning workflow has debate step."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "debate" in step_ids

    def test_has_estimate_step(self, sample_backlog: list[str]) -> None:
        """Sprint planning workflow has estimate step."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "estimate" in step_ids

    def test_has_finalize_step(self, sample_backlog: list[str]) -> None:
        """Sprint planning workflow has finalize step."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "finalize" in step_ids

    def test_has_approval_step(self, sample_backlog: list[str]) -> None:
        """Sprint planning workflow has approval step."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "approval" in step_ids

    def test_has_store_step(self, sample_backlog: list[str]) -> None:
        """Sprint planning workflow has store step."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
        )

        step_ids = [s.id for s in workflow.steps]
        assert "store" in step_ids

    def test_entry_step_is_capacity(self, sample_backlog: list[str]) -> None:
        """Sprint planning workflow starts with capacity step."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
        )

        assert workflow.entry_step == "capacity"

    def test_debate_step_has_backlog_config(self, sample_backlog: list[str]) -> None:
        """Sprint debate step has backlog items in config."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
        )

        debate_step = workflow.get_step("debate")
        assert debate_step is not None
        assert debate_step.config["backlog"] == sample_backlog

    def test_approval_step_has_sprint_name_in_title(self, sample_backlog: list[str]) -> None:
        """Approval step title includes sprint name."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 42",
            backlog_items=sample_backlog,
        )

        approval_step = workflow.get_step("approval")
        assert approval_step is not None
        assert "Sprint 42" in approval_step.config["title"]

    def test_store_step_collection(self, sample_backlog: list[str]) -> None:
        """Store step uses sprint_plans collection."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=sample_backlog,
        )

        store_step = workflow.get_step("store")
        assert store_step is not None
        assert store_step.config["collection"] == "sprint_plans"


# =============================================================================
# Contract Review Workflow Tests
# =============================================================================


class TestCreateContractReviewWorkflow:
    """Tests for create_contract_review_workflow factory."""

    def test_creates_valid_workflow_definition(self) -> None:
        """Contract review workflow creates a valid WorkflowDefinition."""
        workflow = create_contract_review_workflow(
            contract_type="SaaS Agreement",
            counterparty="Vendor Corp",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "contract_saas_agreement" in workflow.id

    def test_counterparty_in_workflow_name(self) -> None:
        """Contract review workflow includes counterparty in name."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Partner Inc",
        )

        assert "Partner Inc" in workflow.name

    def test_contract_type_in_description(self) -> None:
        """Contract review workflow includes contract type in description."""
        workflow = create_contract_review_workflow(
            contract_type="Employment Agreement",
            counterparty="Test",
        )

        assert "Employment Agreement" in workflow.description

    def test_workflow_has_general_category(self) -> None:
        """Contract review workflow has GENERAL category."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Test",
        )

        assert workflow.category == WorkflowCategory.GENERAL

    def test_workflow_has_correct_tags(self) -> None:
        """Contract review workflow has expected tags."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Test",
        )

        assert "sme" in workflow.tags
        assert "legal" in workflow.tags
        assert "contract" in workflow.tags
        assert "review" in workflow.tags

    def test_default_key_terms(self) -> None:
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

    def test_custom_key_terms(self, sample_key_terms: list[str]) -> None:
        """Contract review workflow accepts custom key terms."""
        workflow = create_contract_review_workflow(
            contract_type="Agreement",
            counterparty="Test",
            key_terms=sample_key_terms,
        )

        assert workflow.inputs["key_terms"] == sample_key_terms

    def test_contract_value_stored(self) -> None:
        """Contract review workflow stores contract value."""
        workflow = create_contract_review_workflow(
            contract_type="SaaS",
            counterparty="Test",
            contract_value="$120k/year",
        )

        assert workflow.inputs["contract_value"] == "$120k/year"

    def test_contract_value_none_when_not_provided(self) -> None:
        """Contract review workflow has None contract value when not provided."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Test",
        )

        assert workflow.inputs["contract_value"] is None

    def test_concerns_stored(self) -> None:
        """Contract review workflow stores concerns."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Test",
            concerns=["Non-compete clause", "Jurisdiction"],
        )

        assert workflow.inputs["concerns"] == ["Non-compete clause", "Jurisdiction"]

    def test_none_concerns_defaults_to_empty_list(self) -> None:
        """Contract review defaults to empty concerns list when None."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Test",
            concerns=None,
        )

        assert workflow.inputs["concerns"] == []

    def test_has_parse_step(self) -> None:
        """Contract review workflow has parse step."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Test",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "parse" in step_ids

    def test_has_debate_step(self) -> None:
        """Contract review workflow has debate step."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Test",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "debate" in step_ids

    def test_has_risks_step(self) -> None:
        """Contract review workflow has risk identification step."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Test",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "risks" in step_ids

    def test_has_negotiate_step(self) -> None:
        """Contract review workflow has negotiation points step."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Test",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "negotiate" in step_ids

    def test_has_summary_step(self) -> None:
        """Contract review workflow has executive summary step."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Test",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "summary" in step_ids

    def test_has_review_step(self) -> None:
        """Contract review workflow has legal review step."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Test",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "review" in step_ids

    def test_has_store_step(self) -> None:
        """Contract review workflow has store step."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Test",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "store" in step_ids

    def test_entry_step_is_parse(self) -> None:
        """Contract review workflow starts with parse step."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Test",
        )

        assert workflow.entry_step == "parse"

    def test_debate_step_has_focus_terms_config(self) -> None:
        """Contract debate step has focus terms in config."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Test",
            key_terms=["SLA", "data ownership"],
        )

        debate_step = workflow.get_step("debate")
        assert debate_step is not None
        assert debate_step.config["focus_terms"] == ["SLA", "data ownership"]

    def test_review_step_has_counterparty_in_title(self) -> None:
        """Review step title includes counterparty name."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Big Corp",
        )

        review_step = workflow.get_step("review")
        assert review_step is not None
        assert "Big Corp" in review_step.config["title"]

    def test_store_step_collection(self) -> None:
        """Store step uses contract_reviews collection."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Test",
        )

        store_step = workflow.get_step("store")
        assert store_step is not None
        assert store_step.config["collection"] == "contract_reviews"


# =============================================================================
# Business Decision Workflow Tests
# =============================================================================


class TestCreateBusinessDecisionWorkflow:
    """Tests for create_business_decision_workflow factory."""

    def test_creates_valid_workflow_definition(self) -> None:
        """Business decision workflow creates a valid WorkflowDefinition."""
        workflow = create_business_decision_workflow(
            decision_topic="Should we expand to Europe?",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "decision_" in workflow.id

    def test_topic_in_workflow_name(self) -> None:
        """Business decision workflow includes topic in name."""
        workflow = create_business_decision_workflow(
            decision_topic="Launch new product line",
        )

        assert "Launch new product line" in workflow.name

    def test_topic_truncated_in_name(self) -> None:
        """Business decision workflow truncates long topic in name."""
        long_topic = "A" * 100
        workflow = create_business_decision_workflow(decision_topic=long_topic)

        # Name should be truncated to 50 chars of the topic
        assert len(workflow.name) < 100
        assert workflow.name.startswith("Business Decision: ")

    def test_topic_in_description(self) -> None:
        """Business decision workflow includes full topic in description."""
        workflow = create_business_decision_workflow(
            decision_topic="Open a new office",
        )

        assert "Open a new office" in workflow.description

    def test_workflow_has_general_category(self) -> None:
        """Business decision workflow has GENERAL category."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        assert workflow.category == WorkflowCategory.GENERAL

    def test_workflow_has_correct_tags(self) -> None:
        """Business decision workflow has expected tags."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        assert "sme" in workflow.tags
        assert "business" in workflow.tags
        assert "strategy" in workflow.tags
        assert "decision" in workflow.tags

    def test_default_urgency_is_normal(self) -> None:
        """Business decision workflow defaults to normal urgency."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        assert workflow.inputs["urgency"] == "normal"

    def test_default_impact_level_is_medium(self) -> None:
        """Business decision workflow defaults to medium impact."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        assert workflow.inputs["impact_level"] == "medium"

    def test_custom_urgency_applied(self) -> None:
        """Business decision workflow applies custom urgency."""
        workflow = create_business_decision_workflow(
            decision_topic="Test",
            urgency="critical",
        )

        assert workflow.inputs["urgency"] == "critical"

    def test_custom_impact_level_applied(self) -> None:
        """Business decision workflow applies custom impact level."""
        workflow = create_business_decision_workflow(
            decision_topic="Test",
            impact_level="high",
        )

        assert workflow.inputs["impact_level"] == "high"

    def test_context_stored(self) -> None:
        """Business decision workflow stores context."""
        workflow = create_business_decision_workflow(
            decision_topic="Test",
            context="We have 50% market share in the US",
        )

        assert workflow.inputs["context"] == "We have 50% market share in the US"

    def test_stakeholders_stored(self) -> None:
        """Business decision workflow stores stakeholders."""
        workflow = create_business_decision_workflow(
            decision_topic="Test",
            stakeholders=["CEO", "CFO", "CTO"],
        )

        assert workflow.inputs["stakeholders"] == ["CEO", "CFO", "CTO"]

    def test_none_stakeholders_defaults_to_empty_list(self) -> None:
        """Business decision defaults to empty stakeholders list when None."""
        workflow = create_business_decision_workflow(
            decision_topic="Test",
            stakeholders=None,
        )

        assert workflow.inputs["stakeholders"] == []

    def test_has_frame_step(self) -> None:
        """Business decision workflow has frame decision step."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        step_ids = [s.id for s in workflow.steps]
        assert "frame" in step_ids

    def test_has_research_step(self) -> None:
        """Business decision workflow has research step."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        step_ids = [s.id for s in workflow.steps]
        assert "research" in step_ids

    def test_has_debate_step(self) -> None:
        """Business decision workflow has debate step."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        step_ids = [s.id for s in workflow.steps]
        assert "debate" in step_ids

    def test_has_options_step(self) -> None:
        """Business decision workflow has options generation step."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        step_ids = [s.id for s in workflow.steps]
        assert "options" in step_ids

    def test_has_recommend_step(self) -> None:
        """Business decision workflow has recommendation step."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        step_ids = [s.id for s in workflow.steps]
        assert "recommend" in step_ids

    def test_has_store_step(self) -> None:
        """Business decision workflow has store step."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        step_ids = [s.id for s in workflow.steps]
        assert "store" in step_ids

    def test_high_impact_includes_approval_step(self) -> None:
        """Business decision includes approval for high impact decisions."""
        workflow = create_business_decision_workflow(
            decision_topic="Test",
            impact_level="high",
        )

        step_ids = [s.id for s in workflow.steps]
        assert "approve" in step_ids

    def test_high_impact_recommend_routes_to_approve(self) -> None:
        """High impact recommend step routes to approve."""
        workflow = create_business_decision_workflow(
            decision_topic="Test",
            impact_level="high",
        )

        recommend_step = workflow.get_step("recommend")
        assert recommend_step is not None
        assert "approve" in recommend_step.next_steps

    def test_medium_impact_recommend_routes_to_store(self) -> None:
        """Medium impact recommend step routes to store."""
        workflow = create_business_decision_workflow(
            decision_topic="Test",
            impact_level="medium",
        )

        recommend_step = workflow.get_step("recommend")
        assert recommend_step is not None
        assert "store" in recommend_step.next_steps

    def test_low_impact_recommend_routes_to_store(self) -> None:
        """Low impact recommend step routes to store."""
        workflow = create_business_decision_workflow(
            decision_topic="Test",
            impact_level="low",
        )

        recommend_step = workflow.get_step("recommend")
        assert recommend_step is not None
        assert "store" in recommend_step.next_steps

    def test_entry_step_is_frame(self) -> None:
        """Business decision workflow starts with frame step."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        assert workflow.entry_step == "frame"

    def test_research_step_is_parallel(self) -> None:
        """Research step is parallel type."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        research_step = workflow.get_step("research")
        assert research_step is not None
        assert research_step.step_type == "parallel"
        assert "sub_steps" in research_step.config

    def test_debate_step_has_perspectives_config(self) -> None:
        """Business debate step has perspectives configured."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        debate_step = workflow.get_step("debate")
        assert debate_step is not None
        assert "perspectives" in debate_step.config
        assert "optimist" in debate_step.config["perspectives"]
        assert "skeptic" in debate_step.config["perspectives"]
        assert "pragmatist" in debate_step.config["perspectives"]

    def test_debate_step_has_four_rounds(self) -> None:
        """Business debate step has 4 rounds."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        debate_step = workflow.get_step("debate")
        assert debate_step is not None
        assert debate_step.config["rounds"] == 4

    def test_approve_step_title_includes_topic(self) -> None:
        """Approve step title includes truncated topic."""
        workflow = create_business_decision_workflow(
            decision_topic="Should we expand to Europe and Asia?",
            impact_level="high",
        )

        approve_step = workflow.get_step("approve")
        assert approve_step is not None
        # Topic truncated to 30 chars in approve title
        assert "Approve:" in approve_step.config["title"]

    def test_store_step_collection(self) -> None:
        """Store step uses business_decisions collection."""
        workflow = create_business_decision_workflow(decision_topic="Test")

        store_step = workflow.get_step("store")
        assert store_step is not None
        assert store_step.config["collection"] == "business_decisions"


# =============================================================================
# Step Validation Tests
# =============================================================================


class TestStepValidation:
    """Tests for step validation across all workflows."""

    def test_all_workflows_have_unique_step_ids(
        self, sample_features: list[str], sample_backlog: list[str]
    ) -> None:
        """All factory workflows have unique step IDs."""
        workflows = [
            create_report_workflow("sales"),
            create_budget_allocation_workflow("HR", 100000),
            create_feature_prioritization_workflow(sample_features),
            create_sprint_planning_workflow("Sprint 1", sample_backlog),
            create_contract_review_workflow("NDA", "Partner"),
            create_business_decision_workflow("Expand?"),
            weekly_sales_report(["test@test.com"]),
        ]

        for workflow in workflows:
            step_ids = [s.id for s in workflow.steps]
            assert len(step_ids) == len(set(step_ids)), (
                f"Duplicate step IDs in {workflow.id}: {step_ids}"
            )

    def test_all_workflows_have_valid_entry_step(
        self, sample_features: list[str], sample_backlog: list[str]
    ) -> None:
        """All factory workflows have entry step that exists in steps."""
        workflows = [
            create_report_workflow("sales"),
            create_budget_allocation_workflow("HR", 100000),
            create_feature_prioritization_workflow(sample_features),
            create_sprint_planning_workflow("Sprint 1", sample_backlog),
            create_contract_review_workflow("NDA", "Partner"),
            create_business_decision_workflow("Expand?"),
        ]

        for workflow in workflows:
            step_ids = [s.id for s in workflow.steps]
            assert workflow.entry_step in step_ids, (
                f"Entry step '{workflow.entry_step}' not in steps for {workflow.id}"
            )

    def test_all_next_steps_reference_valid_steps(
        self, sample_features: list[str], sample_backlog: list[str]
    ) -> None:
        """All next_steps references point to valid step IDs."""
        workflows = [
            create_report_workflow("sales"),
            create_budget_allocation_workflow("HR", 100000),
            create_feature_prioritization_workflow(sample_features),
            create_sprint_planning_workflow("Sprint 1", sample_backlog),
            create_contract_review_workflow("NDA", "Partner"),
            create_business_decision_workflow("Expand?", impact_level="high"),
        ]

        for workflow in workflows:
            step_ids = {s.id for s in workflow.steps}
            for step in workflow.steps:
                for next_step in step.next_steps:
                    assert next_step in step_ids, (
                        f"Invalid next_step '{next_step}' in step '{step.id}' "
                        f"of workflow {workflow.id}"
                    )

    def test_all_workflows_have_terminal_step(
        self, sample_features: list[str], sample_backlog: list[str]
    ) -> None:
        """All workflows have at least one terminal step (no next_steps)."""
        workflows = [
            create_report_workflow("sales"),
            create_budget_allocation_workflow("HR", 100000),
            create_feature_prioritization_workflow(sample_features),
            create_sprint_planning_workflow("Sprint 1", sample_backlog),
            create_contract_review_workflow("NDA", "Partner"),
            create_business_decision_workflow("Expand?"),
        ]

        for workflow in workflows:
            terminal_steps = [s for s in workflow.steps if not s.next_steps]
            assert len(terminal_steps) >= 1, (
                f"Workflow {workflow.id} should have at least one terminal step"
            )

    def test_workflows_validate_successfully(
        self, sample_features: list[str], sample_backlog: list[str]
    ) -> None:
        """All workflows pass validation."""
        workflows = [
            create_report_workflow("sales"),
            create_budget_allocation_workflow("HR", 100000),
            create_feature_prioritization_workflow(sample_features),
            create_sprint_planning_workflow("Sprint 1", sample_backlog),
            create_contract_review_workflow("NDA", "Partner"),
            create_business_decision_workflow("Expand?"),
        ]

        for workflow in workflows:
            is_valid, errors = workflow.validate()
            assert is_valid, f"Workflow {workflow.id} validation failed: {errors}"


# =============================================================================
# Step Configuration Tests
# =============================================================================


class TestStepConfiguration:
    """Tests for step configuration details."""

    def test_agent_steps_have_agent_type(self) -> None:
        """Agent steps have agent_type in config."""
        workflow = create_report_workflow("sales")

        agent_steps = [s for s in workflow.steps if s.step_type == "agent"]
        for step in agent_steps:
            assert "agent_type" in step.config, f"Step {step.id} missing agent_type"

    def test_agent_steps_have_prompt_template(self) -> None:
        """Agent steps have prompt_template in config."""
        workflow = create_report_workflow("sales")

        agent_steps = [s for s in workflow.steps if s.step_type == "agent"]
        for step in agent_steps:
            assert "prompt_template" in step.config, f"Step {step.id} missing prompt_template"

    def test_memory_write_steps_have_collection(self) -> None:
        """Memory write steps have collection in config."""
        workflow = create_budget_allocation_workflow("Test", 100000)

        memory_steps = [s for s in workflow.steps if s.step_type == "memory_write"]
        for step in memory_steps:
            assert "collection" in step.config, f"Step {step.id} missing collection"

    def test_debate_steps_have_agents_config(self) -> None:
        """Debate steps have agents configured."""
        workflow = create_contract_review_workflow("NDA", "Test")

        debate_steps = [s for s in workflow.steps if s.step_type == "debate"]
        for step in debate_steps:
            assert "agents" in step.config, f"Step {step.id} missing agents"

    def test_debate_steps_have_rounds(self) -> None:
        """Debate steps have rounds configured."""
        workflow = create_feature_prioritization_workflow(["Feature 1"])

        debate_steps = [s for s in workflow.steps if s.step_type == "debate"]
        for step in debate_steps:
            assert "rounds" in step.config, f"Step {step.id} missing rounds"

    def test_human_checkpoint_steps_have_checkpoint_type(self) -> None:
        """Human checkpoint steps have checkpoint_type configured."""
        workflow = create_sprint_planning_workflow("Sprint 1", ["Task 1"])

        checkpoint_steps = [s for s in workflow.steps if s.step_type == "human_checkpoint"]
        for step in checkpoint_steps:
            assert "checkpoint_type" in step.config, f"Step {step.id} missing checkpoint_type"

    def test_parallel_steps_have_sub_steps(self) -> None:
        """Parallel steps have sub_steps in config."""
        workflow = create_report_workflow("sales")

        parallel_steps = [s for s in workflow.steps if s.step_type == "parallel"]
        for step in parallel_steps:
            assert "sub_steps" in step.config, f"Step {step.id} missing sub_steps"

    def test_task_steps_have_handler(self) -> None:
        """Task steps have handler in config."""
        workflow = create_report_workflow("sales")

        task_steps = [s for s in workflow.steps if s.step_type == "task"]
        for step in task_steps:
            assert "handler" in step.config, f"Step {step.id} missing handler"


# =============================================================================
# Workflow Serialization Tests
# =============================================================================


class TestWorkflowSerialization:
    """Tests for workflow serialization and deserialization."""

    def test_report_workflow_to_dict(self) -> None:
        """Report workflow can be converted to dict."""
        workflow = create_report_workflow("sales")
        data = workflow.to_dict()

        assert isinstance(data, dict)
        assert data["id"] == workflow.id
        assert data["name"] == workflow.name
        assert len(data["steps"]) == len(workflow.steps)

    def test_report_workflow_from_dict_roundtrip(self) -> None:
        """Report workflow survives dict roundtrip."""
        workflow = create_report_workflow("sales", recipients=["test@test.com"])
        data = workflow.to_dict()
        restored = WorkflowDefinition.from_dict(data)

        assert restored.id == workflow.id
        assert restored.name == workflow.name
        assert len(restored.steps) == len(workflow.steps)
        assert restored.inputs == workflow.inputs

    def test_budget_workflow_to_yaml(self) -> None:
        """Budget workflow can be serialized to YAML."""
        workflow = create_budget_allocation_workflow("Engineering", 500000)
        yaml_str = workflow.to_yaml()

        assert isinstance(yaml_str, str)
        assert "Engineering" in yaml_str
        assert "500000" in yaml_str

    def test_budget_workflow_yaml_roundtrip(self) -> None:
        """Budget workflow survives YAML roundtrip."""
        workflow = create_budget_allocation_workflow("HR", 100000)
        yaml_str = workflow.to_yaml()
        restored = WorkflowDefinition.from_yaml(yaml_str)

        assert restored.id == workflow.id
        assert restored.name == workflow.name
        assert len(restored.steps) == len(workflow.steps)

    def test_workflow_to_json(self) -> None:
        """Workflow can be serialized to JSON."""
        workflow = create_feature_prioritization_workflow(["Feature 1", "Feature 2"])
        data = workflow.to_dict()
        json_str = json.dumps(data)

        assert isinstance(json_str, str)
        restored_data = json.loads(json_str)
        assert restored_data["id"] == workflow.id

    def test_step_definitions_serialize(self) -> None:
        """Individual step definitions serialize correctly."""
        workflow = create_sprint_planning_workflow("Sprint 1", ["Task"])
        step = workflow.get_step("debate")
        assert step is not None

        step_data = step.to_dict()
        assert step_data["id"] == "debate"
        assert step_data["step_type"] == "debate"
        assert "config" in step_data


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_features_list(self) -> None:
        """Feature prioritization handles empty features list."""
        workflow = create_feature_prioritization_workflow(features=[])

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.inputs["features"] == []

    def test_single_feature(self) -> None:
        """Feature prioritization handles single feature."""
        workflow = create_feature_prioritization_workflow(features=["Only Feature"])

        assert isinstance(workflow, WorkflowDefinition)
        assert "1 features" in workflow.name

    def test_many_features(self) -> None:
        """Feature prioritization handles many features."""
        features = [f"Feature {i}" for i in range(100)]
        workflow = create_feature_prioritization_workflow(features=features)

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.inputs["features"] == features

    def test_empty_backlog_list(self) -> None:
        """Sprint planning handles empty backlog list."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=[],
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.inputs["backlog_items"] == []

    def test_zero_budget(self) -> None:
        """Budget allocation handles zero budget."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=0,
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.inputs["total_budget"] == 0

    def test_very_large_budget(self) -> None:
        """Budget allocation handles very large budget."""
        workflow = create_budget_allocation_workflow(
            department="Test",
            total_budget=10_000_000_000,  # 10 billion
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "$10,000,000,000" in workflow.description

    def test_zero_team_size(self) -> None:
        """Sprint planning handles zero team size."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 1",
            backlog_items=["Task"],
            team_size=0,
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.inputs["team_size"] == 0

    def test_special_characters_in_department_name(self) -> None:
        """Budget allocation handles special characters in department name."""
        workflow = create_budget_allocation_workflow(
            department="R&D / Innovation",
            total_budget=100000,
        )

        assert isinstance(workflow, WorkflowDefinition)
        # ID should be sanitized
        assert " " not in workflow.id

    def test_special_characters_in_counterparty(self) -> None:
        """Contract review handles special characters in counterparty."""
        workflow = create_contract_review_workflow(
            contract_type="NDA",
            counterparty="Acme Corp (LLC) & Partners",
        )

        assert isinstance(workflow, WorkflowDefinition)

    def test_unicode_in_topic(self) -> None:
        """Business decision handles unicode in topic."""
        workflow = create_business_decision_workflow(
            decision_topic="Expansion internationale - Europe et Asie",
        )

        assert isinstance(workflow, WorkflowDefinition)
        assert "Europe" in workflow.inputs["topic"]

    def test_very_long_topic(self) -> None:
        """Business decision handles very long topic."""
        long_topic = "Should we " + "expand " * 100 + "?"
        workflow = create_business_decision_workflow(decision_topic=long_topic)

        assert isinstance(workflow, WorkflowDefinition)
        # Full topic in inputs
        assert workflow.inputs["topic"] == long_topic
        # Truncated in name
        assert len(workflow.name) < len(long_topic)

    def test_empty_recipients_list(self) -> None:
        """Report workflow handles empty recipients list."""
        workflow = create_report_workflow(report_type="sales", recipients=[])

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.inputs["recipients"] == []


# =============================================================================
# Workflow ID Format Tests
# =============================================================================


class TestWorkflowIdFormat:
    """Tests for workflow ID format and structure."""

    def test_report_workflow_id_has_timestamp(self) -> None:
        """Report workflow ID contains timestamp component."""
        workflow = create_report_workflow(report_type="financial")

        assert workflow.id.startswith("report_financial_")
        parts = workflow.id.split("_")
        assert len(parts) >= 4
        # Date and time parts should be numeric
        assert parts[2].isdigit()
        assert parts[3].isdigit()

    def test_budget_workflow_id_format(self) -> None:
        """Budget workflow ID has correct format."""
        workflow = create_budget_allocation_workflow(
            department="Engineering",
            total_budget=100000,
        )

        assert workflow.id.startswith("budget_engineering_")
        # Should have date part
        parts = workflow.id.split("_")
        assert len(parts) >= 3
        assert parts[2].isdigit()

    def test_feature_workflow_id_format(self) -> None:
        """Feature prioritization workflow ID has correct format."""
        workflow = create_feature_prioritization_workflow(features=["A", "B"])

        assert workflow.id.startswith("feature_priority_")
        parts = workflow.id.split("_")
        assert len(parts) >= 4

    def test_sprint_workflow_id_sanitizes_spaces(self) -> None:
        """Sprint workflow ID sanitizes spaces in sprint name."""
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 24",
            backlog_items=["Task"],
        )

        assert " " not in workflow.id
        assert "sprint_24" in workflow.id.lower()

    def test_contract_workflow_id_sanitizes_spaces(self) -> None:
        """Contract workflow ID sanitizes spaces in contract type."""
        workflow = create_contract_review_workflow(
            contract_type="SaaS Agreement",
            counterparty="Test",
        )

        assert " " not in workflow.id
        assert "saas_agreement" in workflow.id.lower()

    def test_different_report_types_have_different_ids(self) -> None:
        """Different report types produce different workflow IDs."""
        workflow1 = create_report_workflow("sales")
        workflow2 = create_report_workflow("inventory")

        assert "sales" in workflow1.id
        assert "inventory" in workflow2.id
        assert workflow1.id != workflow2.id


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_factory_functions_exported(self) -> None:
        """All factory functions are exported."""
        from aragora.workflow.templates.sme.report_factory import __all__

        expected_exports = [
            "create_report_workflow",
            "weekly_sales_report",
            "create_budget_allocation_workflow",
            "create_feature_prioritization_workflow",
            "create_sprint_planning_workflow",
            "create_contract_review_workflow",
            "create_business_decision_workflow",
        ]

        for export in expected_exports:
            assert export in __all__, f"{export} not in __all__"

    def test_imports_work_correctly(self) -> None:
        """All exports can be imported."""
        from aragora.workflow.templates.sme.report_factory import (
            create_budget_allocation_workflow,
            create_business_decision_workflow,
            create_contract_review_workflow,
            create_feature_prioritization_workflow,
            create_report_workflow,
            create_sprint_planning_workflow,
            weekly_sales_report,
        )

        # All should be callable
        assert callable(create_report_workflow)
        assert callable(weekly_sales_report)
        assert callable(create_budget_allocation_workflow)
        assert callable(create_feature_prioritization_workflow)
        assert callable(create_sprint_planning_workflow)
        assert callable(create_contract_review_workflow)
        assert callable(create_business_decision_workflow)


# =============================================================================
# Parameter Variation Tests
# =============================================================================


class TestParameterVariations:
    """Tests for parameter variations producing different workflows."""

    def test_different_frequencies_produce_different_workflows(self) -> None:
        """Different frequencies produce workflows with different configurations."""
        workflow_daily = create_report_workflow("sales", frequency="daily")
        workflow_weekly = create_report_workflow("sales", frequency="weekly")
        workflow_monthly = create_report_workflow("sales", frequency="monthly")

        assert workflow_daily.inputs["frequency"] != workflow_weekly.inputs["frequency"]
        assert workflow_weekly.inputs["frequency"] != workflow_monthly.inputs["frequency"]
        assert "daily" in workflow_daily.name
        assert "weekly" in workflow_weekly.name
        assert "monthly" in workflow_monthly.name

    def test_different_formats_affect_generate_step(self) -> None:
        """Different formats configure generate step differently."""
        workflow_pdf = create_report_workflow("sales", format="pdf")
        workflow_excel = create_report_workflow("sales", format="excel")
        workflow_html = create_report_workflow("sales", format="html")

        pdf_step = workflow_pdf.get_step("generate")
        excel_step = workflow_excel.get_step("generate")
        html_step = workflow_html.get_step("generate")

        assert pdf_step and pdf_step.config["format"] == "pdf"
        assert excel_step and excel_step.config["format"] == "excel"
        assert html_step and html_step.config["format"] == "html"

    def test_include_charts_affects_step_routing(self) -> None:
        """include_charts parameter affects step routing."""
        workflow_with_charts = create_report_workflow("sales", include_charts=True)
        workflow_no_charts = create_report_workflow("sales", include_charts=False)

        analyze_with = workflow_with_charts.get_step("analyze")
        analyze_without = workflow_no_charts.get_step("analyze")

        assert analyze_with and "charts" in analyze_with.next_steps
        assert analyze_without and "format" in analyze_without.next_steps

    def test_impact_level_affects_approval_step(self) -> None:
        """impact_level parameter affects approval step presence."""
        workflow_high = create_business_decision_workflow("Test", impact_level="high")
        workflow_medium = create_business_decision_workflow("Test", impact_level="medium")
        workflow_low = create_business_decision_workflow("Test", impact_level="low")

        high_steps = [s.id for s in workflow_high.steps]
        medium_steps = [s.id for s in workflow_medium.steps]
        low_steps = [s.id for s in workflow_low.steps]

        assert "approve" in high_steps
        # Medium and low should not have approve step in routing
        recommend_medium = workflow_medium.get_step("recommend")
        recommend_low = workflow_low.get_step("recommend")
        assert recommend_medium and "store" in recommend_medium.next_steps
        assert recommend_low and "store" in recommend_low.next_steps

    def test_categories_affect_debate_config(self) -> None:
        """Custom categories affect debate step configuration."""
        workflow_default = create_budget_allocation_workflow("Test", 100000)
        workflow_custom = create_budget_allocation_workflow(
            "Test",
            100000,
            categories=["A", "B", "C"],
        )

        debate_default = workflow_default.get_step("debate")
        debate_custom = workflow_custom.get_step("debate")

        assert debate_default and len(debate_default.config["categories"]) == 4
        assert debate_custom and debate_custom.config["categories"] == ["A", "B", "C"]

    def test_velocity_stored_in_inputs(self) -> None:
        """Velocity parameter is stored in inputs."""
        workflow_no_velocity = create_sprint_planning_workflow(
            "Sprint 1",
            ["Task"],
        )
        workflow_with_velocity = create_sprint_planning_workflow(
            "Sprint 1",
            ["Task"],
            velocity=42,
        )

        assert workflow_no_velocity.inputs["velocity"] is None
        assert workflow_with_velocity.inputs["velocity"] == 42
