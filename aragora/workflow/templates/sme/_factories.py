"""SME Workflow Templates and Factory Functions.

Provides pre-built workflow automations for common SME business processes:
- Invoice generation and billing
- Customer follow-up and CRM
- Inventory monitoring and alerts
- Report scheduling and delivery

These templates are designed for small and medium enterprises with:
- Minimal configuration required
- Smart defaults for common use cases
- Integration with existing Aragora subsystems

This module re-exports all factory functions from specialized submodules
for backwards compatibility.
"""

from __future__ import annotations

from aragora.workflow.templates.sme.followup_factory import (
    create_followup_workflow,
    create_hiring_decision_workflow,
    create_performance_review_workflow,
    create_remote_work_policy_workflow,
    renewal_followup_campaign,
)
from aragora.workflow.templates.sme.inventory_factory import (
    create_inventory_alert_workflow,
    create_tool_selection_workflow,
    create_vendor_evaluation_workflow,
    daily_inventory_check,
)
from aragora.workflow.templates.sme.invoice_factory import (
    create_invoice_workflow,
    quick_invoice,
)
from aragora.workflow.templates.sme.report_factory import (
    create_budget_allocation_workflow,
    create_business_decision_workflow,
    create_contract_review_workflow,
    create_feature_prioritization_workflow,
    create_report_workflow,
    create_sprint_planning_workflow,
    weekly_sales_report,
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
