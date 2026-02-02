"""Invoice workflow factory functions.

Provides workflow templates for invoice generation and billing processes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from aragora.workflow.types import (
    StepDefinition,
    WorkflowCategory,
    WorkflowDefinition,
)


def create_invoice_workflow(
    customer_id: str,
    items: list[dict[str, Any]],
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


__all__ = [
    "create_invoice_workflow",
    "quick_invoice",
]
