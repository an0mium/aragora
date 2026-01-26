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


# =============================================================================
# SME Decision Templates
# =============================================================================


def create_vendor_evaluation_workflow(
    vendor_name: str,
    evaluation_criteria: Optional[List[str]] = None,
    budget_range: Optional[str] = None,
    timeline: str = "30 days",
    require_approval: bool = True,
) -> WorkflowDefinition:
    """Create a vendor evaluation workflow.

    Evaluates potential vendors based on criteria like cost, reliability,
    quality, and fit with business needs.

    Args:
        vendor_name: Name of vendor to evaluate
        evaluation_criteria: Custom criteria (defaults to standard set)
        budget_range: Budget constraints (e.g., "$10k-$50k")
        timeline: Decision timeline
        require_approval: Whether to require human approval

    Returns:
        WorkflowDefinition for vendor evaluation

    Example:
        workflow = create_vendor_evaluation_workflow(
            vendor_name="Acme Corp",
            evaluation_criteria=["price", "support", "integration"],
            budget_range="$20k-$40k",
        )
    """
    workflow_id = (
        f"vendor_eval_{vendor_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
    )
    criteria = evaluation_criteria or [
        "pricing",
        "reliability",
        "support",
        "integration",
        "scalability",
    ]

    return WorkflowDefinition(
        id=workflow_id,
        name=f"Vendor Evaluation: {vendor_name}",
        description=f"Evaluate {vendor_name} as a potential vendor",
        category=WorkflowCategory.GENERAL,
        tags=["sme", "vendor", "evaluation", "decision"],
        inputs={
            "vendor_name": vendor_name,
            "criteria": criteria,
            "budget_range": budget_range,
            "timeline": timeline,
        },
        steps=[
            StepDefinition(
                id="gather",
                name="Gather Information",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": f"Research {vendor_name}: pricing, reviews, capabilities",
                },
                next_steps=["debate"],
            ),
            StepDefinition(
                id="debate",
                name="Multi-Agent Evaluation",
                step_type="debate",
                config={
                    "agents": ["claude", "gpt-4", "gemini"],
                    "topic": f"Should we proceed with {vendor_name}?",
                    "rounds": 3,
                    "criteria": criteria,
                },
                next_steps=["score"],
            ),
            StepDefinition(
                id="score",
                name="Score Criteria",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Score vendor on each criterion (1-10)",
                },
                next_steps=["recommend"],
            ),
            StepDefinition(
                id="recommend",
                name="Generate Recommendation",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Synthesize debate and generate recommendation",
                },
                next_steps=["review"] if require_approval else ["store"],
            ),
            StepDefinition(
                id="review",
                name="Human Review",
                step_type="human_checkpoint",
                config={
                    "checkpoint_type": "approval",
                    "title": f"Review {vendor_name} Evaluation",
                },
                next_steps=["store"],
            ),
            StepDefinition(
                id="store",
                name="Store Decision",
                step_type="memory_write",
                config={
                    "collection": "vendor_evaluations",
                },
                next_steps=[],
            ),
        ],
        entry_step="gather",
    )


def create_hiring_decision_workflow(
    position: str,
    candidate_name: str,
    interview_notes: Optional[str] = None,
    team_size: int = 5,
    urgency: str = "normal",
) -> WorkflowDefinition:
    """Create a hiring decision workflow.

    Evaluates a candidate for a position using multi-agent debate
    to assess fit, skills, and potential.

    Args:
        position: Job position/title
        candidate_name: Candidate's name
        interview_notes: Interview notes or resume summary
        team_size: Current team size (for fit assessment)
        urgency: Urgency level (low, normal, high)

    Returns:
        WorkflowDefinition for hiring decision

    Example:
        workflow = create_hiring_decision_workflow(
            position="Senior Developer",
            candidate_name="Jane Doe",
            interview_notes="Strong Python skills, 5 years experience...",
        )
    """
    workflow_id = f"hire_{position.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"

    return WorkflowDefinition(
        id=workflow_id,
        name=f"Hiring Decision: {candidate_name} for {position}",
        description=f"Evaluate {candidate_name} for {position}",
        category=WorkflowCategory.GENERAL,
        tags=["sme", "hiring", "hr", "decision"],
        inputs={
            "position": position,
            "candidate_name": candidate_name,
            "interview_notes": interview_notes,
            "team_size": team_size,
            "urgency": urgency,
        },
        steps=[
            StepDefinition(
                id="analyze",
                name="Analyze Candidate",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": f"Analyze {candidate_name}'s qualifications for {position}",
                },
                next_steps=["debate"],
            ),
            StepDefinition(
                id="debate",
                name="Team Fit Debate",
                step_type="debate",
                config={
                    "agents": ["claude", "gpt-4"],
                    "topic": f"Is {candidate_name} a good fit for {position}?",
                    "rounds": 3,
                    "focus_areas": ["skills", "culture", "growth_potential"],
                },
                next_steps=["risks"],
            ),
            StepDefinition(
                id="risks",
                name="Risk Assessment",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Identify hiring risks and mitigation strategies",
                },
                next_steps=["recommend"],
            ),
            StepDefinition(
                id="recommend",
                name="Final Recommendation",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Generate hiring recommendation with rationale",
                },
                next_steps=["approval"],
            ),
            StepDefinition(
                id="approval",
                name="Manager Approval",
                step_type="human_checkpoint",
                config={
                    "checkpoint_type": "approval",
                    "title": f"Approve hiring {candidate_name}?",
                },
                next_steps=["store"],
            ),
            StepDefinition(
                id="store",
                name="Record Decision",
                step_type="memory_write",
                config={
                    "collection": "hiring_decisions",
                },
                next_steps=[],
            ),
        ],
        entry_step="analyze",
    )


def create_budget_allocation_workflow(
    department: str,
    total_budget: float,
    categories: Optional[List[str]] = None,
    fiscal_year: Optional[str] = None,
    constraints: Optional[List[str]] = None,
) -> WorkflowDefinition:
    """Create a budget allocation workflow.

    Uses multi-agent debate to determine optimal budget allocation
    across categories or projects.

    Args:
        department: Department name
        total_budget: Total budget amount
        categories: Budget categories to allocate
        fiscal_year: Fiscal year for allocation
        constraints: Budget constraints or requirements

    Returns:
        WorkflowDefinition for budget allocation

    Example:
        workflow = create_budget_allocation_workflow(
            department="Engineering",
            total_budget=500000,
            categories=["infrastructure", "tools", "training", "contractors"],
        )
    """
    workflow_id = (
        f"budget_{department.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
    )
    year = fiscal_year or datetime.now().strftime("%Y")
    budget_categories = categories or ["operations", "growth", "maintenance", "innovation"]

    return WorkflowDefinition(
        id=workflow_id,
        name=f"Budget Allocation: {department} FY{year}",
        description=f"Allocate ${total_budget:,.0f} budget for {department}",
        category=WorkflowCategory.ACCOUNTING,
        tags=["sme", "budget", "finance", "decision"],
        inputs={
            "department": department,
            "total_budget": total_budget,
            "categories": budget_categories,
            "fiscal_year": year,
            "constraints": constraints or [],
        },
        steps=[
            StepDefinition(
                id="analyze",
                name="Analyze Needs",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": f"Analyze {department}'s budget needs for FY{year}",
                },
                next_steps=["historical"],
            ),
            StepDefinition(
                id="historical",
                name="Historical Analysis",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Analyze historical spending patterns and trends",
                },
                next_steps=["debate"],
            ),
            StepDefinition(
                id="debate",
                name="Allocation Debate",
                step_type="debate",
                config={
                    "agents": ["claude", "gpt-4", "gemini"],
                    "topic": f"How to allocate ${total_budget:,.0f} across {len(budget_categories)} categories?",
                    "rounds": 3,
                    "categories": budget_categories,
                },
                next_steps=["propose"],
            ),
            StepDefinition(
                id="propose",
                name="Generate Proposals",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Generate 3 budget allocation proposals",
                },
                next_steps=["review"],
            ),
            StepDefinition(
                id="review",
                name="CFO Review",
                step_type="human_checkpoint",
                config={
                    "checkpoint_type": "selection",
                    "title": "Select budget allocation proposal",
                },
                next_steps=["finalize"],
            ),
            StepDefinition(
                id="finalize",
                name="Finalize Budget",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Finalize budget with selected allocation",
                },
                next_steps=["store"],
            ),
            StepDefinition(
                id="store",
                name="Record Allocation",
                step_type="memory_write",
                config={
                    "collection": "budget_allocations",
                },
                next_steps=[],
            ),
        ],
        entry_step="analyze",
    )


def create_business_decision_workflow(
    decision_topic: str,
    context: Optional[str] = None,
    stakeholders: Optional[List[str]] = None,
    urgency: str = "normal",
    impact_level: str = "medium",
) -> WorkflowDefinition:
    """Create a general business decision workflow.

    Flexible workflow for any strategic business decision using
    multi-agent debate for thorough analysis.

    Args:
        decision_topic: Topic or question to decide
        context: Background context for the decision
        stakeholders: List of stakeholders affected
        urgency: Decision urgency (low, normal, high, critical)
        impact_level: Impact level (low, medium, high)

    Returns:
        WorkflowDefinition for business decision

    Example:
        workflow = create_business_decision_workflow(
            decision_topic="Should we expand to the European market?",
            context="We have 50% US market share...",
            impact_level="high",
        )
    """
    workflow_id = f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return WorkflowDefinition(
        id=workflow_id,
        name=f"Business Decision: {decision_topic[:50]}",
        description=f"Analyze and decide: {decision_topic}",
        category=WorkflowCategory.GENERAL,
        tags=["sme", "business", "strategy", "decision"],
        inputs={
            "topic": decision_topic,
            "context": context,
            "stakeholders": stakeholders or [],
            "urgency": urgency,
            "impact_level": impact_level,
        },
        steps=[
            StepDefinition(
                id="frame",
                name="Frame Decision",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": f"Frame the decision: {decision_topic}",
                },
                next_steps=["research"],
            ),
            StepDefinition(
                id="research",
                name="Research & Analysis",
                step_type="parallel",
                config={
                    "sub_steps": ["market", "internal", "risks"],
                },
                next_steps=["debate"],
            ),
            StepDefinition(
                id="debate",
                name="Strategic Debate",
                step_type="debate",
                config={
                    "agents": ["claude", "gpt-4", "gemini"],
                    "topic": decision_topic,
                    "rounds": 4,
                    "perspectives": ["optimist", "skeptic", "pragmatist"],
                },
                next_steps=["options"],
            ),
            StepDefinition(
                id="options",
                name="Generate Options",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Generate 3-5 decision options with pros/cons",
                },
                next_steps=["recommend"],
            ),
            StepDefinition(
                id="recommend",
                name="Final Recommendation",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Synthesize debate into recommendation",
                },
                next_steps=["approve"] if impact_level == "high" else ["store"],
            ),
            StepDefinition(
                id="approve",
                name="Executive Approval",
                step_type="human_checkpoint",
                config={
                    "checkpoint_type": "approval",
                    "title": f"Approve: {decision_topic[:30]}",
                },
                next_steps=["store"],
            ),
            StepDefinition(
                id="store",
                name="Record Decision",
                step_type="memory_write",
                config={
                    "collection": "business_decisions",
                },
                next_steps=[],
            ),
        ],
        entry_step="frame",
    )


__all__ = [
    # Main factory functions
    "create_invoice_workflow",
    "create_followup_workflow",
    "create_inventory_alert_workflow",
    "create_report_workflow",
    # SME Decision templates
    "create_vendor_evaluation_workflow",
    "create_hiring_decision_workflow",
    "create_budget_allocation_workflow",
    "create_business_decision_workflow",
    # Quick convenience functions
    "quick_invoice",
    "weekly_sales_report",
    "daily_inventory_check",
    "renewal_followup_campaign",
]
