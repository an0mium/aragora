"""
SME Workflow Templates and Factory Functions.

Provides pre-built workflow automations for common SME business processes:
- Invoice generation and billing
- Customer follow-up and CRM
- Inventory monitoring and alerts
- Report scheduling and delivery

These templates are designed for small and medium enterprises with:
- Minimal configuration required
- Smart defaults for common use cases
- Integration with existing Aragora subsystems
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from aragora.workflow.types import (
    StepDefinition,
    WorkflowCategory,
    WorkflowDefinition,
)


def create_invoice_workflow(
    customer_id: str,
    items: List[Dict[str, Any]],
    tax_rate: float = 0.0,
    due_days: int = 30,
    send_email: bool = False,
    notes: str = "",
) -> WorkflowDefinition:
    """Create an invoice generation workflow.

    Args:
        customer_id: Customer identifier
        items: List of items with name, quantity, unit_price
        tax_rate: Tax rate as decimal (e.g., 0.1 for 10%)
        due_days: Days until payment is due
        send_email: Whether to email invoice to customer
        notes: Additional notes for invoice

    Returns:
        WorkflowDefinition ready for execution

    Example:
        workflow = create_invoice_workflow(
            customer_id="cust_123",
            items=[
                {"name": "Consulting", "quantity": 10, "unit_price": 150.00},
                {"name": "Support", "quantity": 1, "unit_price": 500.00},
            ],
            tax_rate=0.08,
            send_email=True,
        )
        result = await engine.execute(workflow)
    """
    workflow_id = f"invoice_{customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return WorkflowDefinition(
        id=workflow_id,
        name=f"Invoice for {customer_id}",
        description="Generate and optionally send invoice",
        category=WorkflowCategory.ACCOUNTING,
        tags=["sme", "invoice", "billing"],
        inputs={
            "customer_id": customer_id,
            "items": items,
            "tax_rate": tax_rate,
            "due_days": due_days,
            "send_email": send_email,
            "notes": notes,
        },
        steps=[
            StepDefinition(
                id="validate",
                name="Validate Customer",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": f"Validate customer {customer_id} and retrieve billing details",
                },
                next_steps=["calculate"],
            ),
            StepDefinition(
                id="calculate",
                name="Calculate Totals",
                step_type="task",
                config={
                    "handler": "calculate_invoice_totals",
                    "items": items,
                    "tax_rate": tax_rate,
                },
                next_steps=["generate"],
            ),
            StepDefinition(
                id="generate",
                name="Generate Invoice",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Generate professional invoice document",
                },
                next_steps=["deliver"] if send_email else ["store"],
            ),
            StepDefinition(
                id="deliver",
                name="Send Email",
                step_type="task",
                config={
                    "handler": "send_email",
                    "template": "invoice",
                },
                next_steps=["store"],
            ),
            StepDefinition(
                id="store",
                name="Store Record",
                step_type="memory_write",
                config={
                    "collection": "invoices",
                },
                next_steps=[],
            ),
        ],
        entry_step="validate",
    )


def create_followup_workflow(
    followup_type: str = "check_in",
    days_since_contact: int = 30,
    channel: str = "email",
    auto_send: bool = False,
    customer_id: Optional[str] = None,
) -> WorkflowDefinition:
    """Create a customer follow-up workflow.

    Args:
        followup_type: Type of follow-up (post_sale, check_in, renewal, feedback)
        days_since_contact: Filter for customers not contacted in N days
        channel: Communication channel (email, sms, call_scheduled)
        auto_send: Whether to auto-send or queue for review
        customer_id: Specific customer to follow up (optional)

    Returns:
        WorkflowDefinition ready for execution

    Example:
        workflow = create_followup_workflow(
            followup_type="renewal",
            days_since_contact=60,
            channel="email",
        )
        result = await engine.execute(workflow)
    """
    workflow_id = f"followup_{followup_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return WorkflowDefinition(
        id=workflow_id,
        name=f"Customer Follow-up: {followup_type}",
        description=f"Follow up with customers ({followup_type})",
        category=WorkflowCategory.GENERAL,
        tags=["sme", "customer", "crm", "followup"],
        inputs={
            "followup_type": followup_type,
            "days_since_contact": days_since_contact,
            "channel": channel,
            "auto_send": auto_send,
            "customer_id": customer_id,
        },
        steps=[
            StepDefinition(
                id="identify",
                name="Identify Customers",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": f"Find customers for {followup_type} follow-up",
                },
                next_steps=["analyze"],
            ),
            StepDefinition(
                id="analyze",
                name="Analyze Context",
                step_type="parallel",
                config={
                    "sub_steps": ["sentiment", "opportunities"],
                },
                next_steps=["draft"],
            ),
            StepDefinition(
                id="draft",
                name="Draft Messages",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Draft personalized follow-up messages",
                },
                next_steps=["review"] if not auto_send else ["send"],
            ),
            StepDefinition(
                id="review",
                name="Human Review",
                step_type="human_checkpoint",
                config={
                    "checkpoint_type": "approval",
                    "title": "Review Follow-up Messages",
                },
                next_steps=["send"],
            ),
            StepDefinition(
                id="send",
                name="Send Messages",
                step_type="task",
                config={
                    "handler": "send_bulk_messages",
                    "channel": channel,
                },
                next_steps=["schedule"],
            ),
            StepDefinition(
                id="schedule",
                name="Schedule Next",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Schedule next follow-up dates",
                },
                next_steps=["store"],
            ),
            StepDefinition(
                id="store",
                name="Store Records",
                step_type="memory_write",
                config={
                    "collection": "customer_followups",
                },
                next_steps=[],
            ),
        ],
        entry_step="identify",
    )


def create_inventory_alert_workflow(
    alert_threshold: int = 20,
    auto_reorder: bool = False,
    notification_channels: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> WorkflowDefinition:
    """Create an inventory monitoring and alert workflow.

    Args:
        alert_threshold: Percentage of safety stock to trigger alert
        auto_reorder: Whether to auto-create purchase orders
        notification_channels: Channels for alerts (email, slack, sms)
        categories: Product categories to monitor (all if None)

    Returns:
        WorkflowDefinition ready for execution

    Example:
        workflow = create_inventory_alert_workflow(
            alert_threshold=25,
            auto_reorder=True,
            notification_channels=["email", "slack"],
        )
        result = await engine.execute(workflow)
    """
    workflow_id = f"inventory_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    channels = notification_channels or ["email"]

    return WorkflowDefinition(
        id=workflow_id,
        name="Inventory Alert Check",
        description="Monitor inventory levels and send alerts",
        category=WorkflowCategory.GENERAL,
        tags=["sme", "inventory", "alerts", "supply-chain"],
        inputs={
            "alert_threshold": alert_threshold,
            "auto_reorder": auto_reorder,
            "notification_channels": channels,
            "categories": categories,
        },
        steps=[
            StepDefinition(
                id="fetch",
                name="Fetch Inventory",
                step_type="task",
                config={
                    "handler": "fetch_inventory_data",
                    "categories": categories,
                },
                next_steps=["analyze"],
            ),
            StepDefinition(
                id="analyze",
                name="Analyze Levels",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": f"Analyze inventory with {alert_threshold}% threshold",
                },
                next_steps=["calculate"],
            ),
            StepDefinition(
                id="calculate",
                name="Calculate Reorder",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Calculate optimal reorder quantities",
                },
                next_steps=["alert"],
            ),
            StepDefinition(
                id="alert",
                name="Send Alerts",
                step_type="parallel",
                config={
                    "channels": channels,
                },
                next_steps=["reorder"] if auto_reorder else ["store"],
            ),
            StepDefinition(
                id="reorder",
                name="Create Orders",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Generate purchase orders",
                },
                next_steps=["submit"],
            ),
            StepDefinition(
                id="submit",
                name="Submit Orders",
                step_type="task",
                config={
                    "handler": "submit_purchase_orders",
                },
                next_steps=["store"],
            ),
            StepDefinition(
                id="store",
                name="Store Results",
                step_type="memory_write",
                config={
                    "collection": "inventory_checks",
                },
                next_steps=[],
            ),
        ],
        entry_step="fetch",
    )


def create_report_workflow(
    report_type: str,
    frequency: str = "weekly",
    date_range: str = "last_week",
    format: str = "pdf",
    recipients: Optional[List[str]] = None,
    include_charts: bool = True,
    include_comparison: bool = True,
) -> WorkflowDefinition:
    """Create a report generation and scheduling workflow.

    Args:
        report_type: Type of report (sales, financial, inventory, customer)
        frequency: Report frequency (daily, weekly, monthly, quarterly)
        date_range: Date range for data (last_day, last_week, last_month)
        format: Output format (pdf, excel, html)
        recipients: List of email recipients
        include_charts: Whether to include visualizations
        include_comparison: Whether to include period comparison

    Returns:
        WorkflowDefinition ready for execution

    Example:
        workflow = create_report_workflow(
            report_type="sales",
            frequency="weekly",
            format="pdf",
            recipients=["sales@company.com", "ceo@company.com"],
        )
        result = await engine.execute(workflow)
    """
    workflow_id = f"report_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return WorkflowDefinition(
        id=workflow_id,
        name=f"{report_type.title()} Report ({frequency})",
        description=f"Generate {frequency} {report_type} report",
        category=WorkflowCategory.GENERAL,
        tags=["sme", "reports", "analytics", report_type],
        inputs={
            "report_type": report_type,
            "frequency": frequency,
            "date_range": date_range,
            "format": format,
            "recipients": recipients or [],
            "include_charts": include_charts,
            "comparison": include_comparison,
        },
        steps=[
            StepDefinition(
                id="fetch",
                name="Fetch Data",
                step_type="parallel",
                config={
                    "sub_steps": ["primary", "comparison", "benchmarks"],
                },
                next_steps=["analyze"],
            ),
            StepDefinition(
                id="analyze",
                name="Analyze Data",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": f"Analyze {report_type} data and extract insights",
                },
                next_steps=["charts"] if include_charts else ["format"],
            ),
            StepDefinition(
                id="charts",
                name="Generate Charts",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Generate chart configurations",
                },
                next_steps=["render"],
            ),
            StepDefinition(
                id="render",
                name="Render Charts",
                step_type="task",
                config={
                    "handler": "render_charts",
                },
                next_steps=["format"],
            ),
            StepDefinition(
                id="format",
                name="Format Report",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": f"Create professional {report_type} report",
                },
                next_steps=["generate"],
            ),
            StepDefinition(
                id="generate",
                name="Generate File",
                step_type="task",
                config={
                    "handler": "generate_report_file",
                    "format": format,
                },
                next_steps=["deliver"],
            ),
            StepDefinition(
                id="deliver",
                name="Deliver Report",
                step_type="parallel",
                config={
                    "sub_steps": ["email", "store"],
                },
                next_steps=["log"],
            ),
            StepDefinition(
                id="log",
                name="Log Completion",
                step_type="memory_write",
                config={
                    "collection": "report_logs",
                },
                next_steps=[],
            ),
        ],
        entry_step="fetch",
    )


# Convenience functions for common patterns


def quick_invoice(
    customer: str,
    amount: float,
    description: str,
    send: bool = True,
) -> WorkflowDefinition:
    """Create a quick single-item invoice.

    Args:
        customer: Customer name or ID
        amount: Total amount
        description: Invoice description
        send: Whether to send via email

    Returns:
        WorkflowDefinition for simple invoice
    """
    return create_invoice_workflow(
        customer_id=customer,
        items=[{"name": description, "quantity": 1, "unit_price": amount}],
        send_email=send,
    )


def weekly_sales_report(recipients: List[str]) -> WorkflowDefinition:
    """Create a weekly sales report workflow.

    Args:
        recipients: Email addresses to receive report

    Returns:
        WorkflowDefinition for weekly sales report
    """
    return create_report_workflow(
        report_type="sales",
        frequency="weekly",
        date_range="last_week",
        format="pdf",
        recipients=recipients,
    )


def daily_inventory_check(
    slack_channel: Optional[str] = None,
) -> WorkflowDefinition:
    """Create a daily inventory check workflow.

    Args:
        slack_channel: Optional Slack channel for alerts

    Returns:
        WorkflowDefinition for daily inventory monitoring
    """
    channels = ["email"]
    if slack_channel:
        channels.append("slack")

    return create_inventory_alert_workflow(
        alert_threshold=20,
        auto_reorder=False,
        notification_channels=channels,
    )


def renewal_followup_campaign() -> WorkflowDefinition:
    """Create a renewal follow-up campaign workflow.

    Returns:
        WorkflowDefinition for renewal follow-ups
    """
    return create_followup_workflow(
        followup_type="renewal",
        days_since_contact=60,
        channel="email",
        auto_send=False,
    )


__all__ = [
    # Main factory functions
    "create_invoice_workflow",
    "create_followup_workflow",
    "create_inventory_alert_workflow",
    "create_report_workflow",
    # Quick convenience functions
    "quick_invoice",
    "weekly_sales_report",
    "daily_inventory_check",
    "renewal_followup_campaign",
]
