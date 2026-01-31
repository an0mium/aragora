"""
Tests for Marketing and Advertising Workflow Templates.

Tests coverage for:
- Ad Performance Review template (multi-platform analytics)
- Lead-to-CRM Sync template (lead enrichment and qualification)
- Cross-Platform Analytics template (unified analytics)
- Support Ticket Triage template (AI-powered routing)
- E-commerce Order Sync template (order reconciliation)
- Template registry (MARKETING_TEMPLATES)
- Parameter validation
- Step configuration and transitions
"""

from __future__ import annotations

import pytest

from aragora.workflow.templates.marketing import (
    AD_PERFORMANCE_REVIEW_TEMPLATE,
    LEAD_TO_CRM_SYNC_TEMPLATE,
    CROSS_PLATFORM_ANALYTICS_TEMPLATE,
    SUPPORT_TICKET_TRIAGE_TEMPLATE,
    ECOMMERCE_ORDER_SYNC_TEMPLATE,
    MARKETING_TEMPLATES,
)


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
        # Parallel steps use "branches" instead of normal step fields
        if step.get("type") == "parallel":
            assert "branches" in step, f"{label} parallel step missing 'branches'"
            continue
        # Conditional steps may have then_steps/else_steps
        if step.get("type") == "conditional":
            assert "condition" in step, f"{label} conditional step missing 'condition'"
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
                step_ids.add(branch["id"])
                for sub_step in branch.get("steps", []):
                    step_ids.add(sub_step["id"])
        elif step.get("type") == "conditional":
            step_ids.add(step["id"])
            for then_step in step.get("then_steps", []):
                step_ids.add(then_step["id"])
            for else_step in step.get("else_steps", []):
                step_ids.add(else_step["id"])
        else:
            step_ids.add(step["id"])

    for transition in template["transitions"]:
        assert transition["from"] in step_ids, (
            f"{label} transition references unknown step '{transition['from']}'"
        )
        assert transition["to"] in step_ids, (
            f"{label} transition references unknown step '{transition['to']}'"
        )


def _get_step_ids(template: dict) -> list[str]:
    """Extract all step IDs from a template."""
    step_ids = []
    for step in template["steps"]:
        if step.get("type") == "parallel":
            step_ids.append(step["id"])
            for branch in step.get("branches", []):
                step_ids.append(branch["id"])
                for sub_step in branch.get("steps", []):
                    step_ids.append(sub_step["id"])
        elif step.get("type") == "conditional":
            step_ids.append(step["id"])
            for then_step in step.get("then_steps", []):
                step_ids.append(then_step["id"])
        else:
            step_ids.append(step["id"])
    return step_ids


def _get_debate_steps(template: dict) -> list[dict]:
    """Extract all debate steps from a template."""
    return [s for s in template["steps"] if s.get("type") == "debate"]


def _get_connector_steps(template: dict) -> list[dict]:
    """Extract all connector steps from a template."""
    steps = []
    for step in template["steps"]:
        if step.get("type") == "connector":
            steps.append(step)
        elif step.get("type") == "parallel":
            for branch in step.get("branches", []):
                for sub_step in branch.get("steps", []):
                    if sub_step.get("type") == "connector":
                        steps.append(sub_step)
    return steps


# ============================================================================
# Ad Performance Review Template Tests
# ============================================================================


class TestAdPerformanceReviewTemplate:
    """Tests for Ad Performance Review template."""

    def test_template_structure(self):
        """Test template has all required fields."""
        _assert_template_structure(AD_PERFORMANCE_REVIEW_TEMPLATE, "AD_PERFORMANCE_REVIEW")

    def test_template_identity(self):
        """Test template identity and metadata."""
        assert AD_PERFORMANCE_REVIEW_TEMPLATE["name"] == "Ad Performance Review"
        assert AD_PERFORMANCE_REVIEW_TEMPLATE["category"] == "marketing"
        assert AD_PERFORMANCE_REVIEW_TEMPLATE["version"] == "1.0"

    def test_steps_valid(self):
        """Test all steps have required fields."""
        _assert_steps_valid(AD_PERFORMANCE_REVIEW_TEMPLATE, "AD_PERFORMANCE_REVIEW")

    def test_transitions_reference_valid_steps(self):
        """Test transitions reference valid steps."""
        _assert_transitions_reference_valid_steps(
            AD_PERFORMANCE_REVIEW_TEMPLATE, "AD_PERFORMANCE_REVIEW"
        )

    def test_has_marketing_tags(self):
        """Test template has marketing-related tags."""
        tags = AD_PERFORMANCE_REVIEW_TEMPLATE["tags"]
        assert "advertising" in tags
        assert "marketing" in tags
        assert "analytics" in tags
        assert "optimization" in tags

    def test_has_parallel_data_collection(self):
        """Test template has parallel data collection step."""
        step_ids = _get_step_ids(AD_PERFORMANCE_REVIEW_TEMPLATE)
        assert "data_collection" in step_ids

        parallel_step = next(
            s for s in AD_PERFORMANCE_REVIEW_TEMPLATE["steps"] if s.get("id") == "data_collection"
        )
        assert parallel_step["type"] == "parallel"
        assert len(parallel_step["branches"]) == 4

    def test_parallel_branches_for_ad_platforms(self):
        """Test parallel branches cover all major ad platforms."""
        parallel_step = next(
            s for s in AD_PERFORMANCE_REVIEW_TEMPLATE["steps"] if s.get("id") == "data_collection"
        )
        branch_ids = [b["id"] for b in parallel_step["branches"]]
        assert "google_ads_data" in branch_ids
        assert "meta_ads_data" in branch_ids
        assert "linkedin_ads_data" in branch_ids
        assert "microsoft_ads_data" in branch_ids

    def test_has_performance_analysis_debate(self):
        """Test template has performance analysis debate step."""
        step_ids = _get_step_ids(AD_PERFORMANCE_REVIEW_TEMPLATE)
        assert "performance_analysis" in step_ids

        perf_step = next(
            s
            for s in AD_PERFORMANCE_REVIEW_TEMPLATE["steps"]
            if s.get("id") == "performance_analysis"
        )
        assert perf_step["type"] == "debate"

    def test_performance_analysis_agents(self):
        """Test performance analysis has correct agents."""
        perf_step = next(
            s
            for s in AD_PERFORMANCE_REVIEW_TEMPLATE["steps"]
            if s.get("id") == "performance_analysis"
        )
        agents = perf_step["config"]["agents"]
        assert "marketing_analyst" in agents
        assert "data_scientist" in agents
        assert "cfo" in agents

    def test_has_audience_analysis_debate(self):
        """Test template has audience analysis debate step."""
        step_ids = _get_step_ids(AD_PERFORMANCE_REVIEW_TEMPLATE)
        assert "audience_analysis" in step_ids

    def test_has_budget_recommendations_debate(self):
        """Test template has budget recommendations debate step."""
        step_ids = _get_step_ids(AD_PERFORMANCE_REVIEW_TEMPLATE)
        assert "budget_recommendations" in step_ids

    def test_has_creative_recommendations_debate(self):
        """Test template has creative recommendations debate step."""
        step_ids = _get_step_ids(AD_PERFORMANCE_REVIEW_TEMPLATE)
        assert "creative_recommendations" in step_ids

    def test_has_human_review_checkpoint(self):
        """Test template has human review checkpoint."""
        step_ids = _get_step_ids(AD_PERFORMANCE_REVIEW_TEMPLATE)
        assert "human_review" in step_ids

        human_step = next(
            s for s in AD_PERFORMANCE_REVIEW_TEMPLATE["steps"] if s.get("id") == "human_review"
        )
        assert human_step["type"] == "human_checkpoint"
        assert human_step["config"]["required_role"] == "marketing_manager"

    def test_human_review_has_checklist(self):
        """Test human review has approval checklist."""
        human_step = next(
            s for s in AD_PERFORMANCE_REVIEW_TEMPLATE["steps"] if s.get("id") == "human_review"
        )
        checklist = human_step["config"]["checklist"]
        assert len(checklist) == 3

    def test_has_memory_read_step(self):
        """Test template has memory read step for historical data."""
        step_ids = _get_step_ids(AD_PERFORMANCE_REVIEW_TEMPLATE)
        assert "memory_read_historical" in step_ids

    def test_has_memory_write_step(self):
        """Test template has memory write step for findings."""
        step_ids = _get_step_ids(AD_PERFORMANCE_REVIEW_TEMPLATE)
        assert "memory_write_findings" in step_ids

    def test_has_report_generation_step(self):
        """Test template has report generation step."""
        step_ids = _get_step_ids(AD_PERFORMANCE_REVIEW_TEMPLATE)
        assert "generate_report" in step_ids

    def test_inputs_configuration(self):
        """Test template has correct input configuration."""
        inputs = AD_PERFORMANCE_REVIEW_TEMPLATE["inputs"]
        assert "platforms" in inputs
        assert "date_range_days" in inputs
        assert "budget" in inputs
        assert inputs["date_range_days"]["default"] == 30

    def test_outputs_configuration(self):
        """Test template has correct output configuration."""
        outputs = AD_PERFORMANCE_REVIEW_TEMPLATE["outputs"]
        assert "report" in outputs
        assert "recommendations" in outputs
        assert "budget_allocation" in outputs

    def test_all_debate_steps_have_agents(self):
        """Test all debate steps have agents configured."""
        debate_steps = _get_debate_steps(AD_PERFORMANCE_REVIEW_TEMPLATE)
        assert len(debate_steps) == 4
        for step in debate_steps:
            assert "agents" in step["config"], f"Debate step '{step['id']}' missing agents"
            assert len(step["config"]["agents"]) >= 2

    def test_transitions_include_conditional_paths(self):
        """Test transitions include conditional approval/rejection paths."""
        transitions = AD_PERFORMANCE_REVIEW_TEMPLATE["transitions"]
        conditional_transitions = [t for t in transitions if "condition" in t]
        assert len(conditional_transitions) == 2

        conditions = {t["condition"] for t in conditional_transitions}
        assert "approved" in conditions
        assert "rejected" in conditions


# ============================================================================
# Lead-to-CRM Sync Template Tests
# ============================================================================


class TestLeadToCrmSyncTemplate:
    """Tests for Lead-to-CRM Sync template."""

    def test_template_structure(self):
        """Test template has all required fields."""
        _assert_template_structure(LEAD_TO_CRM_SYNC_TEMPLATE, "LEAD_TO_CRM_SYNC")

    def test_template_identity(self):
        """Test template identity and metadata."""
        assert LEAD_TO_CRM_SYNC_TEMPLATE["name"] == "Lead-to-CRM Sync"
        assert LEAD_TO_CRM_SYNC_TEMPLATE["category"] == "marketing"
        assert LEAD_TO_CRM_SYNC_TEMPLATE["version"] == "1.0"

    def test_steps_valid(self):
        """Test all steps have required fields."""
        _assert_steps_valid(LEAD_TO_CRM_SYNC_TEMPLATE, "LEAD_TO_CRM_SYNC")

    def test_transitions_reference_valid_steps(self):
        """Test transitions reference valid steps."""
        _assert_transitions_reference_valid_steps(LEAD_TO_CRM_SYNC_TEMPLATE, "LEAD_TO_CRM_SYNC")

    def test_has_crm_tags(self):
        """Test template has CRM-related tags."""
        tags = LEAD_TO_CRM_SYNC_TEMPLATE["tags"]
        assert "crm" in tags
        assert "leads" in tags
        assert "advertising" in tags
        assert "sync" in tags

    def test_has_fetch_leads_step(self):
        """Test template has lead fetching step."""
        step_ids = _get_step_ids(LEAD_TO_CRM_SYNC_TEMPLATE)
        assert "fetch_leads" in step_ids

    def test_has_deduplicate_step(self):
        """Test template has deduplication step."""
        step_ids = _get_step_ids(LEAD_TO_CRM_SYNC_TEMPLATE)
        assert "deduplicate" in step_ids

    def test_has_conditional_enrichment(self):
        """Test template has conditional enrichment step."""
        step_ids = _get_step_ids(LEAD_TO_CRM_SYNC_TEMPLATE)
        assert "enrich_leads" in step_ids

        enrich_step = next(
            s for s in LEAD_TO_CRM_SYNC_TEMPLATE["steps"] if s.get("id") == "enrich_leads"
        )
        assert enrich_step["type"] == "conditional"
        assert enrich_step["condition"] == "{enrich_data}"

    def test_has_lead_qualification_debate(self):
        """Test template has lead qualification debate step."""
        step_ids = _get_step_ids(LEAD_TO_CRM_SYNC_TEMPLATE)
        assert "qualify_leads" in step_ids

        qualify_step = next(
            s for s in LEAD_TO_CRM_SYNC_TEMPLATE["steps"] if s.get("id") == "qualify_leads"
        )
        assert qualify_step["type"] == "debate"

    def test_lead_qualification_agents(self):
        """Test lead qualification has correct agents."""
        qualify_step = next(
            s for s in LEAD_TO_CRM_SYNC_TEMPLATE["steps"] if s.get("id") == "qualify_leads"
        )
        agents = qualify_step["config"]["agents"]
        assert "sales_analyst" in agents
        assert "marketing_analyst" in agents

    def test_has_create_contacts_step(self):
        """Test template has contact creation step."""
        step_ids = _get_step_ids(LEAD_TO_CRM_SYNC_TEMPLATE)
        assert "create_contacts" in step_ids

    def test_has_lead_assignment_step(self):
        """Test template has lead assignment step."""
        step_ids = _get_step_ids(LEAD_TO_CRM_SYNC_TEMPLATE)
        assert "assign_leads" in step_ids

    def test_lead_assignment_routing_rules(self):
        """Test lead assignment has routing rules."""
        assign_step = next(
            s for s in LEAD_TO_CRM_SYNC_TEMPLATE["steps"] if s.get("id") == "assign_leads"
        )
        routing_rules = assign_step["config"]["routing_rules"]
        assert len(routing_rules) == 3

    def test_has_notification_step(self):
        """Test template has notification step."""
        step_ids = _get_step_ids(LEAD_TO_CRM_SYNC_TEMPLATE)
        assert "notify" in step_ids

    def test_inputs_configuration(self):
        """Test template has correct input configuration."""
        inputs = LEAD_TO_CRM_SYNC_TEMPLATE["inputs"]
        assert "source_platform" in inputs
        assert "target_crm" in inputs
        assert "enrich_data" in inputs
        assert inputs["target_crm"]["default"] == "hubspot"
        assert inputs["enrich_data"]["default"] is True

    def test_source_platform_enum(self):
        """Test source platform has valid enum values."""
        inputs = LEAD_TO_CRM_SYNC_TEMPLATE["inputs"]
        enum_values = inputs["source_platform"]["enum"]
        assert "linkedin_ads" in enum_values
        assert "meta_ads" in enum_values
        assert "google_ads" in enum_values

    def test_outputs_configuration(self):
        """Test template has correct output configuration."""
        outputs = LEAD_TO_CRM_SYNC_TEMPLATE["outputs"]
        assert "leads_synced" in outputs
        assert "leads_qualified" in outputs
        assert "assignments" in outputs


# ============================================================================
# Cross-Platform Analytics Template Tests
# ============================================================================


class TestCrossPlatformAnalyticsTemplate:
    """Tests for Cross-Platform Analytics template."""

    def test_template_structure(self):
        """Test template has all required fields."""
        _assert_template_structure(CROSS_PLATFORM_ANALYTICS_TEMPLATE, "CROSS_PLATFORM_ANALYTICS")

    def test_template_identity(self):
        """Test template identity and metadata."""
        assert CROSS_PLATFORM_ANALYTICS_TEMPLATE["name"] == "Cross-Platform Analytics"
        assert CROSS_PLATFORM_ANALYTICS_TEMPLATE["category"] == "analytics"
        assert CROSS_PLATFORM_ANALYTICS_TEMPLATE["version"] == "1.0"

    def test_steps_valid(self):
        """Test all steps have required fields."""
        _assert_steps_valid(CROSS_PLATFORM_ANALYTICS_TEMPLATE, "CROSS_PLATFORM_ANALYTICS")

    def test_transitions_reference_valid_steps(self):
        """Test transitions reference valid steps."""
        _assert_transitions_reference_valid_steps(
            CROSS_PLATFORM_ANALYTICS_TEMPLATE, "CROSS_PLATFORM_ANALYTICS"
        )

    def test_has_analytics_tags(self):
        """Test template has analytics-related tags."""
        tags = CROSS_PLATFORM_ANALYTICS_TEMPLATE["tags"]
        assert "analytics" in tags
        assert "reporting" in tags
        assert "marketing" in tags
        assert "bi" in tags

    def test_has_parallel_analytics_collection(self):
        """Test template has parallel analytics collection step."""
        step_ids = _get_step_ids(CROSS_PLATFORM_ANALYTICS_TEMPLATE)
        assert "collect_analytics" in step_ids

        parallel_step = next(
            s
            for s in CROSS_PLATFORM_ANALYTICS_TEMPLATE["steps"]
            if s.get("id") == "collect_analytics"
        )
        assert parallel_step["type"] == "parallel"

    def test_parallel_branches_for_analytics_sources(self):
        """Test parallel branches cover analytics sources."""
        parallel_step = next(
            s
            for s in CROSS_PLATFORM_ANALYTICS_TEMPLATE["steps"]
            if s.get("id") == "collect_analytics"
        )
        branch_ids = [b["id"] for b in parallel_step["branches"]]
        assert "web_analytics" in branch_ids
        assert "product_analytics" in branch_ids
        assert "ad_performance" in branch_ids

    def test_has_attribution_analysis_debate(self):
        """Test template has attribution analysis debate step."""
        step_ids = _get_step_ids(CROSS_PLATFORM_ANALYTICS_TEMPLATE)
        assert "attribution_analysis" in step_ids

    def test_has_funnel_analysis_debate(self):
        """Test template has funnel analysis debate step."""
        step_ids = _get_step_ids(CROSS_PLATFORM_ANALYTICS_TEMPLATE)
        assert "funnel_analysis" in step_ids

    def test_has_roi_calculation_step(self):
        """Test template has ROI calculation step."""
        step_ids = _get_step_ids(CROSS_PLATFORM_ANALYTICS_TEMPLATE)
        assert "roi_calculation" in step_ids

    def test_roi_calculation_formulas(self):
        """Test ROI calculation has proper formulas."""
        roi_step = next(
            s
            for s in CROSS_PLATFORM_ANALYTICS_TEMPLATE["steps"]
            if s.get("id") == "roi_calculation"
        )
        calculations = roi_step["config"]["calculations"]
        calc_names = [c["name"] for c in calculations]
        assert "channel_roi" in calc_names
        assert "cac" in calc_names
        assert "ltv_cac_ratio" in calc_names

    def test_has_dashboard_generation_step(self):
        """Test template has dashboard generation step."""
        step_ids = _get_step_ids(CROSS_PLATFORM_ANALYTICS_TEMPLATE)
        assert "generate_dashboard" in step_ids

    def test_dashboard_widgets(self):
        """Test dashboard has multiple widget types."""
        dashboard_step = next(
            s
            for s in CROSS_PLATFORM_ANALYTICS_TEMPLATE["steps"]
            if s.get("id") == "generate_dashboard"
        )
        widgets = dashboard_step["config"]["widgets"]
        widget_types = [w["type"] for w in widgets]
        assert "kpi" in widget_types
        assert "chart" in widget_types
        assert "table" in widget_types
        assert "funnel" in widget_types

    def test_inputs_configuration(self):
        """Test template has correct input configuration."""
        inputs = CROSS_PLATFORM_ANALYTICS_TEMPLATE["inputs"]
        assert "analytics_platforms" in inputs
        assert "advertising_platforms" in inputs
        assert "date_range_days" in inputs

    def test_outputs_configuration(self):
        """Test template has correct output configuration."""
        outputs = CROSS_PLATFORM_ANALYTICS_TEMPLATE["outputs"]
        assert "dashboard" in outputs
        assert "attribution_report" in outputs
        assert "roi_analysis" in outputs


# ============================================================================
# Support Ticket Triage Template Tests
# ============================================================================


class TestSupportTicketTriageTemplate:
    """Tests for Support Ticket Triage template."""

    def test_template_structure(self):
        """Test template has all required fields."""
        _assert_template_structure(SUPPORT_TICKET_TRIAGE_TEMPLATE, "SUPPORT_TICKET_TRIAGE")

    def test_template_identity(self):
        """Test template identity and metadata."""
        assert SUPPORT_TICKET_TRIAGE_TEMPLATE["name"] == "Support Ticket Triage"
        assert SUPPORT_TICKET_TRIAGE_TEMPLATE["category"] == "support"
        assert SUPPORT_TICKET_TRIAGE_TEMPLATE["version"] == "1.0"

    def test_steps_valid(self):
        """Test all steps have required fields."""
        _assert_steps_valid(SUPPORT_TICKET_TRIAGE_TEMPLATE, "SUPPORT_TICKET_TRIAGE")

    def test_transitions_reference_valid_steps(self):
        """Test transitions reference valid steps."""
        _assert_transitions_reference_valid_steps(
            SUPPORT_TICKET_TRIAGE_TEMPLATE, "SUPPORT_TICKET_TRIAGE"
        )

    def test_has_support_tags(self):
        """Test template has support-related tags."""
        tags = SUPPORT_TICKET_TRIAGE_TEMPLATE["tags"]
        assert "support" in tags
        assert "triage" in tags
        assert "customer-service" in tags
        assert "automation" in tags

    def test_has_fetch_tickets_step(self):
        """Test template has ticket fetching step."""
        step_ids = _get_step_ids(SUPPORT_TICKET_TRIAGE_TEMPLATE)
        assert "fetch_tickets" in step_ids

    def test_has_categorize_tickets_debate(self):
        """Test template has ticket categorization debate step."""
        step_ids = _get_step_ids(SUPPORT_TICKET_TRIAGE_TEMPLATE)
        assert "categorize_tickets" in step_ids

    def test_categorize_tickets_has_categories(self):
        """Test ticket categorization includes category definitions."""
        categorize_step = next(
            s
            for s in SUPPORT_TICKET_TRIAGE_TEMPLATE["steps"]
            if s.get("id") == "categorize_tickets"
        )
        topic_template = categorize_step["config"]["topic_template"]
        assert "billing" in topic_template
        assert "technical" in topic_template
        assert "account" in topic_template

    def test_has_prioritize_tickets_debate(self):
        """Test template has ticket prioritization debate step."""
        step_ids = _get_step_ids(SUPPORT_TICKET_TRIAGE_TEMPLATE)
        assert "prioritize_tickets" in step_ids

    def test_has_conditional_auto_respond(self):
        """Test template has conditional auto-respond step."""
        step_ids = _get_step_ids(SUPPORT_TICKET_TRIAGE_TEMPLATE)
        assert "suggest_responses" in step_ids

        respond_step = next(
            s for s in SUPPORT_TICKET_TRIAGE_TEMPLATE["steps"] if s.get("id") == "suggest_responses"
        )
        assert respond_step["type"] == "conditional"
        assert respond_step["condition"] == "{auto_respond}"

    def test_has_route_tickets_step(self):
        """Test template has ticket routing step."""
        step_ids = _get_step_ids(SUPPORT_TICKET_TRIAGE_TEMPLATE)
        assert "route_tickets" in step_ids

    def test_routing_rules_configuration(self):
        """Test ticket routing has proper rules."""
        route_step = next(
            s for s in SUPPORT_TICKET_TRIAGE_TEMPLATE["steps"] if s.get("id") == "route_tickets"
        )
        routing_rules = route_step["config"]["routing_rules"]
        assert len(routing_rules) == 4

    def test_has_update_tickets_step(self):
        """Test template has ticket update step."""
        step_ids = _get_step_ids(SUPPORT_TICKET_TRIAGE_TEMPLATE)
        assert "update_tickets" in step_ids

    def test_has_notify_teams_step(self):
        """Test template has team notification step."""
        step_ids = _get_step_ids(SUPPORT_TICKET_TRIAGE_TEMPLATE)
        assert "notify_teams" in step_ids

    def test_inputs_configuration(self):
        """Test template has correct input configuration."""
        inputs = SUPPORT_TICKET_TRIAGE_TEMPLATE["inputs"]
        assert "platforms" in inputs
        assert "auto_respond" in inputs
        assert inputs["auto_respond"]["default"] is False

    def test_outputs_configuration(self):
        """Test template has correct output configuration."""
        outputs = SUPPORT_TICKET_TRIAGE_TEMPLATE["outputs"]
        assert "tickets_triaged" in outputs
        assert "priority_breakdown" in outputs
        assert "routing_summary" in outputs


# ============================================================================
# E-commerce Order Sync Template Tests
# ============================================================================


class TestEcommerceOrderSyncTemplate:
    """Tests for E-commerce Order Sync template."""

    def test_template_structure(self):
        """Test template has all required fields."""
        _assert_template_structure(ECOMMERCE_ORDER_SYNC_TEMPLATE, "ECOMMERCE_ORDER_SYNC")

    def test_template_identity(self):
        """Test template identity and metadata."""
        assert ECOMMERCE_ORDER_SYNC_TEMPLATE["name"] == "E-commerce Order Sync"
        assert ECOMMERCE_ORDER_SYNC_TEMPLATE["category"] == "ecommerce"
        assert ECOMMERCE_ORDER_SYNC_TEMPLATE["version"] == "1.0"

    def test_steps_valid(self):
        """Test all steps have required fields."""
        _assert_steps_valid(ECOMMERCE_ORDER_SYNC_TEMPLATE, "ECOMMERCE_ORDER_SYNC")

    def test_transitions_reference_valid_steps(self):
        """Test transitions reference valid steps."""
        _assert_transitions_reference_valid_steps(
            ECOMMERCE_ORDER_SYNC_TEMPLATE, "ECOMMERCE_ORDER_SYNC"
        )

    def test_has_ecommerce_tags(self):
        """Test template has e-commerce-related tags."""
        tags = ECOMMERCE_ORDER_SYNC_TEMPLATE["tags"]
        assert "ecommerce" in tags
        assert "orders" in tags
        assert "accounting" in tags
        assert "sync" in tags

    def test_has_fetch_orders_step(self):
        """Test template has order fetching step."""
        step_ids = _get_step_ids(ECOMMERCE_ORDER_SYNC_TEMPLATE)
        assert "fetch_orders" in step_ids

    def test_has_validate_orders_step(self):
        """Test template has order validation step."""
        step_ids = _get_step_ids(ECOMMERCE_ORDER_SYNC_TEMPLATE)
        assert "validate_orders" in step_ids

    def test_validation_rules(self):
        """Test order validation has proper rules."""
        validate_step = next(
            s for s in ECOMMERCE_ORDER_SYNC_TEMPLATE["steps"] if s.get("id") == "validate_orders"
        )
        rules = validate_step["config"]["validation_rules"]
        assert "customer_email_present" in rules
        assert "shipping_address_complete" in rules
        assert "line_items_present" in rules

    def test_has_map_to_accounting_step(self):
        """Test template has accounting mapping step."""
        step_ids = _get_step_ids(ECOMMERCE_ORDER_SYNC_TEMPLATE)
        assert "map_to_accounting" in step_ids

    def test_accounting_mappings(self):
        """Test accounting mapping configuration."""
        map_step = next(
            s for s in ECOMMERCE_ORDER_SYNC_TEMPLATE["steps"] if s.get("id") == "map_to_accounting"
        )
        mappings = map_step["config"]["mappings"]
        assert "customer" in mappings
        assert "line_items" in mappings
        assert "total" in mappings

    def test_has_conditional_invoice_creation(self):
        """Test template has conditional invoice creation step."""
        step_ids = _get_step_ids(ECOMMERCE_ORDER_SYNC_TEMPLATE)
        assert "create_invoices" in step_ids

        invoice_step = next(
            s for s in ECOMMERCE_ORDER_SYNC_TEMPLATE["steps"] if s.get("id") == "create_invoices"
        )
        assert invoice_step["type"] == "conditional"
        assert invoice_step["condition"] == "{create_invoices}"

    def test_has_reconcile_debate(self):
        """Test template has reconciliation debate step."""
        step_ids = _get_step_ids(ECOMMERCE_ORDER_SYNC_TEMPLATE)
        assert "reconcile" in step_ids

    def test_reconcile_agents(self):
        """Test reconciliation has correct agents."""
        reconcile_step = next(
            s for s in ECOMMERCE_ORDER_SYNC_TEMPLATE["steps"] if s.get("id") == "reconcile"
        )
        agents = reconcile_step["config"]["agents"]
        assert "accountant" in agents
        assert "operations" in agents

    def test_has_mark_synced_step(self):
        """Test template has sync marking step."""
        step_ids = _get_step_ids(ECOMMERCE_ORDER_SYNC_TEMPLATE)
        assert "mark_synced" in step_ids

    def test_inputs_configuration(self):
        """Test template has correct input configuration."""
        inputs = ECOMMERCE_ORDER_SYNC_TEMPLATE["inputs"]
        assert "source_platforms" in inputs
        assert "accounting_platform" in inputs
        assert "create_invoices" in inputs
        assert inputs["accounting_platform"]["default"] == "xero"
        assert inputs["create_invoices"]["default"] is True

    def test_outputs_configuration(self):
        """Test template has correct output configuration."""
        outputs = ECOMMERCE_ORDER_SYNC_TEMPLATE["outputs"]
        assert "orders_synced" in outputs
        assert "invoices_created" in outputs
        assert "reconciliation_status" in outputs


# ============================================================================
# Marketing Templates Registry Tests
# ============================================================================


class TestMarketingTemplatesRegistry:
    """Tests for MARKETING_TEMPLATES registry."""

    def test_all_templates_in_registry(self):
        """Test all marketing templates are in registry."""
        assert "ad_performance_review" in MARKETING_TEMPLATES
        assert "lead_to_crm_sync" in MARKETING_TEMPLATES
        assert "cross_platform_analytics" in MARKETING_TEMPLATES
        assert "support_ticket_triage" in MARKETING_TEMPLATES
        assert "ecommerce_order_sync" in MARKETING_TEMPLATES

    def test_registry_count(self):
        """Test registry has expected number of templates."""
        assert len(MARKETING_TEMPLATES) == 5

    def test_registry_templates_are_valid(self):
        """Test all templates in registry are valid."""
        for key, template in MARKETING_TEMPLATES.items():
            _assert_template_structure(template, key)

    def test_registry_templates_have_valid_steps(self):
        """Test all templates in registry have valid steps."""
        for key, template in MARKETING_TEMPLATES.items():
            _assert_steps_valid(template, key)

    def test_registry_templates_have_valid_transitions(self):
        """Test all templates in registry have valid transitions."""
        for key, template in MARKETING_TEMPLATES.items():
            _assert_transitions_reference_valid_steps(template, key)


# ============================================================================
# Cross-Template Validation Tests
# ============================================================================


class TestAllMarketingTemplatesStructure:
    """Cross-template structural validation of all marketing templates."""

    @pytest.fixture
    def all_templates(self):
        """Fixture providing all marketing templates."""
        return {
            "AD_PERFORMANCE_REVIEW": AD_PERFORMANCE_REVIEW_TEMPLATE,
            "LEAD_TO_CRM_SYNC": LEAD_TO_CRM_SYNC_TEMPLATE,
            "CROSS_PLATFORM_ANALYTICS": CROSS_PLATFORM_ANALYTICS_TEMPLATE,
            "SUPPORT_TICKET_TRIAGE": SUPPORT_TICKET_TRIAGE_TEMPLATE,
            "ECOMMERCE_ORDER_SYNC": ECOMMERCE_ORDER_SYNC_TEMPLATE,
        }

    def test_all_templates_have_required_fields(self, all_templates):
        """Test all templates have required top-level fields."""
        for label, template in all_templates.items():
            _assert_template_structure(template, label)

    def test_all_templates_have_valid_steps(self, all_templates):
        """Test all templates have valid steps."""
        for label, template in all_templates.items():
            _assert_steps_valid(template, label)

    def test_all_templates_have_valid_transitions(self, all_templates):
        """Test all templates have valid transitions."""
        for label, template in all_templates.items():
            _assert_transitions_reference_valid_steps(template, label)

    def test_all_templates_have_version(self, all_templates):
        """Test all templates have version 1.0."""
        for label, template in all_templates.items():
            assert template["version"] == "1.0", f"{label} version is not 1.0"

    def test_all_templates_have_nonempty_tags(self, all_templates):
        """Test all templates have at least 2 tags."""
        for label, template in all_templates.items():
            assert len(template["tags"]) >= 2, f"{label} should have at least 2 tags"

    def test_all_debate_steps_have_agents(self, all_templates):
        """Test all debate steps have agents configured."""
        for label, template in all_templates.items():
            debate_steps = _get_debate_steps(template)
            for step in debate_steps:
                assert "agents" in step.get("config", {}), (
                    f"{label} debate step '{step['id']}' missing agents"
                )

    def test_no_duplicate_step_ids(self, all_templates):
        """Test no duplicate step IDs in any template."""
        for label, template in all_templates.items():
            step_ids = _get_step_ids(template)
            assert len(step_ids) == len(set(step_ids)), f"{label} has duplicate step IDs"

    def test_all_templates_have_inputs(self, all_templates):
        """Test all templates have inputs configuration."""
        for label, template in all_templates.items():
            assert "inputs" in template, f"{label} missing 'inputs'"
            assert len(template["inputs"]) > 0, f"{label} has no inputs"

    def test_all_templates_have_outputs(self, all_templates):
        """Test all templates have outputs configuration."""
        for label, template in all_templates.items():
            assert "outputs" in template, f"{label} missing 'outputs'"
            assert len(template["outputs"]) > 0, f"{label} has no outputs"

    def test_all_templates_have_human_review(self, all_templates):
        """Test all templates have a human review checkpoint."""
        for label, template in all_templates.items():
            step_ids = _get_step_ids(template)
            assert "human_review" in step_ids, f"{label} missing 'human_review' checkpoint"

    def test_all_templates_have_memory_steps(self, all_templates):
        """Test all templates have memory read and write steps."""
        for label, template in all_templates.items():
            step_ids = _get_step_ids(template)
            memory_read_steps = [s for s in step_ids if s.startswith("memory_read")]
            memory_write_steps = [s for s in step_ids if s.startswith("memory_write")]
            assert len(memory_read_steps) >= 1, f"{label} missing memory read step"
            assert len(memory_write_steps) >= 1, f"{label} missing memory write step"


# ============================================================================
# Module Exports Tests
# ============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_ad_performance_review_exported(self):
        """Test AD_PERFORMANCE_REVIEW_TEMPLATE is exported."""
        from aragora.workflow.templates.marketing import AD_PERFORMANCE_REVIEW_TEMPLATE

        assert AD_PERFORMANCE_REVIEW_TEMPLATE is not None

    def test_lead_to_crm_sync_exported(self):
        """Test LEAD_TO_CRM_SYNC_TEMPLATE is exported."""
        from aragora.workflow.templates.marketing import LEAD_TO_CRM_SYNC_TEMPLATE

        assert LEAD_TO_CRM_SYNC_TEMPLATE is not None

    def test_cross_platform_analytics_exported(self):
        """Test CROSS_PLATFORM_ANALYTICS_TEMPLATE is exported."""
        from aragora.workflow.templates.marketing import CROSS_PLATFORM_ANALYTICS_TEMPLATE

        assert CROSS_PLATFORM_ANALYTICS_TEMPLATE is not None

    def test_support_ticket_triage_exported(self):
        """Test SUPPORT_TICKET_TRIAGE_TEMPLATE is exported."""
        from aragora.workflow.templates.marketing import SUPPORT_TICKET_TRIAGE_TEMPLATE

        assert SUPPORT_TICKET_TRIAGE_TEMPLATE is not None

    def test_ecommerce_order_sync_exported(self):
        """Test ECOMMERCE_ORDER_SYNC_TEMPLATE is exported."""
        from aragora.workflow.templates.marketing import ECOMMERCE_ORDER_SYNC_TEMPLATE

        assert ECOMMERCE_ORDER_SYNC_TEMPLATE is not None

    def test_marketing_templates_exported(self):
        """Test MARKETING_TEMPLATES is exported."""
        from aragora.workflow.templates.marketing import MARKETING_TEMPLATES

        assert MARKETING_TEMPLATES is not None
        assert isinstance(MARKETING_TEMPLATES, dict)
