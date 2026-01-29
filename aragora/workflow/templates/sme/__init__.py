"""
SME Workflow Templates and Quick-Start Bundles.

Provides pre-built workflow automations for common SME business processes:
- Invoice generation and billing
- Customer follow-up and CRM
- Inventory monitoring and alerts
- Report scheduling and delivery
- Decision-making workflows (hiring, vendors, budgets, etc.)

Quick-Start Bundles:
- month_end_close: Monthly financial close process
- hiring_sprint: Rapid hiring decision workflow
- product_planning: Product/sprint planning cycle
- vendor_refresh: Vendor evaluation and transition
- q_planning: Quarterly business planning

Usage:
    # Factory functions
    from aragora.workflow.templates.sme import create_vendor_evaluation_workflow

    workflow = create_vendor_evaluation_workflow(
        vendor_name="Acme Corp",
        evaluation_criteria=["price", "support"],
    )

    # Quick-start bundles
    from aragora.workflow.templates.sme.bundles import get_bundle, list_bundles

    bundles = list_bundles()
    month_end = get_bundle("month_end_close")
"""

from __future__ import annotations

# Re-export all factory functions from _factories
from aragora.workflow.templates.sme._factories import (
    # Main factory functions
    create_invoice_workflow,
    create_followup_workflow,
    create_inventory_alert_workflow,
    create_report_workflow,
    # SME Decision templates (original)
    create_vendor_evaluation_workflow,
    create_hiring_decision_workflow,
    create_budget_allocation_workflow,
    create_business_decision_workflow,
    # SME Decision templates (new)
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

__all__ = [
    # Main factory functions
    "create_invoice_workflow",
    "create_followup_workflow",
    "create_inventory_alert_workflow",
    "create_report_workflow",
    # SME Decision templates (original)
    "create_vendor_evaluation_workflow",
    "create_hiring_decision_workflow",
    "create_budget_allocation_workflow",
    "create_business_decision_workflow",
    # SME Decision templates (new)
    "create_performance_review_workflow",
    "create_feature_prioritization_workflow",
    "create_sprint_planning_workflow",
    "create_tool_selection_workflow",
    "create_contract_review_workflow",
    "create_remote_work_policy_workflow",
    # Quick convenience functions
    "quick_invoice",
    "weekly_sales_report",
    "daily_inventory_check",
    "renewal_followup_campaign",
]
