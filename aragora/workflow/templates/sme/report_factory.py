"""Report and business decision workflow factory functions.

Provides workflow templates for report generation, budget allocation,
feature prioritization, sprint planning, contract review, and
general business decision workflows.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from aragora.workflow.types import (
    StepDefinition,
    WorkflowCategory,
    WorkflowDefinition,
)


def create_report_workflow(
    report_type: str,
    frequency: str = "weekly",
    date_range: str = "last_week",
    format: str = "pdf",
    recipients: list[str] | None = None,
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


def weekly_sales_report(recipients: list[str]) -> WorkflowDefinition:
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


def create_budget_allocation_workflow(
    department: str,
    total_budget: float,
    categories: list[str] | None = None,
    fiscal_year: str | None = None,
    constraints: list[str] | None = None,
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


def create_feature_prioritization_workflow(
    features: list[str],
    constraints: list[str] | None = None,
    team_capacity: str | None = None,
    timeline: str = "next quarter",
    scoring_criteria: list[str] | None = None,
) -> WorkflowDefinition:
    """Create a feature prioritization workflow.

    Uses multi-agent debate to prioritize features based on impact,
    effort, dependencies, and strategic alignment.

    Args:
        features: List of features to prioritize
        constraints: Resource or technical constraints
        team_capacity: Available team capacity description
        timeline: Planning timeline
        scoring_criteria: Custom scoring criteria

    Returns:
        WorkflowDefinition for feature prioritization

    Example:
        workflow = create_feature_prioritization_workflow(
            features=["Dark mode", "Export to PDF", "API v2", "Mobile app"],
            constraints=["2 developers available", "Must ship before Q2"],
            timeline="next quarter",
        )
    """
    workflow_id = f"feature_priority_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    criteria = scoring_criteria or ["impact", "effort", "urgency", "dependencies"]

    return WorkflowDefinition(
        id=workflow_id,
        name=f"Feature Prioritization ({len(features)} features)",
        description=f"Prioritize {len(features)} features for {timeline}",
        category=WorkflowCategory.GENERAL,
        tags=["sme", "product", "prioritization", "planning"],
        inputs={
            "features": features,
            "constraints": constraints or [],
            "team_capacity": team_capacity,
            "timeline": timeline,
            "criteria": criteria,
        },
        steps=[
            StepDefinition(
                id="analyze",
                name="Analyze Features",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": f"Analyze {len(features)} features for prioritization",
                },
                next_steps=["debate"],
            ),
            StepDefinition(
                id="debate",
                name="Prioritization Debate",
                step_type="debate",
                config={
                    "agents": ["claude", "gpt-4", "mistral"],
                    "topic": f"How should we prioritize: {', '.join(features[:3])}...?",
                    "rounds": 3,
                    "criteria": criteria,
                },
                next_steps=["score"],
            ),
            StepDefinition(
                id="score",
                name="Score Features",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Score each feature on criteria (1-10)",
                },
                next_steps=["rank"],
            ),
            StepDefinition(
                id="rank",
                name="Create Ranking",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Generate final priority ranking with rationale",
                },
                next_steps=["review"],
            ),
            StepDefinition(
                id="review",
                name="Team Review",
                step_type="human_checkpoint",
                config={
                    "checkpoint_type": "approval",
                    "title": "Review feature prioritization",
                },
                next_steps=["store"],
            ),
            StepDefinition(
                id="store",
                name="Record Priorities",
                step_type="memory_write",
                config={
                    "collection": "feature_prioritizations",
                },
                next_steps=[],
            ),
        ],
        entry_step="analyze",
    )


def create_sprint_planning_workflow(
    sprint_name: str,
    backlog_items: list[str],
    team_size: int = 5,
    sprint_duration: str = "2 weeks",
    velocity: int | None = None,
) -> WorkflowDefinition:
    """Create a sprint planning workflow.

    Uses multi-agent debate to select and scope sprint work
    based on team capacity and priorities.

    Args:
        sprint_name: Sprint identifier (e.g., "Sprint 24")
        backlog_items: Available backlog items to consider
        team_size: Number of team members
        sprint_duration: Sprint length
        velocity: Historical team velocity (story points)

    Returns:
        WorkflowDefinition for sprint planning

    Example:
        workflow = create_sprint_planning_workflow(
            sprint_name="Sprint 24",
            backlog_items=["User auth", "Dashboard", "API refactor"],
            team_size=4,
            velocity=32,
        )
    """
    workflow_id = (
        f"sprint_{sprint_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
    )

    return WorkflowDefinition(
        id=workflow_id,
        name=f"Sprint Planning: {sprint_name}",
        description=f"Plan {sprint_name} for {team_size} team members",
        category=WorkflowCategory.GENERAL,
        tags=["sme", "agile", "sprint", "planning"],
        inputs={
            "sprint_name": sprint_name,
            "backlog_items": backlog_items,
            "team_size": team_size,
            "sprint_duration": sprint_duration,
            "velocity": velocity,
        },
        steps=[
            StepDefinition(
                id="capacity",
                name="Calculate Capacity",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": f"Calculate sprint capacity for {team_size} members over {sprint_duration}",
                },
                next_steps=["debate"],
            ),
            StepDefinition(
                id="debate",
                name="Sprint Scope Debate",
                step_type="debate",
                config={
                    "agents": ["claude", "gpt-4"],
                    "topic": f"What should we commit to in {sprint_name}?",
                    "rounds": 2,
                    "backlog": backlog_items,
                },
                next_steps=["estimate"],
            ),
            StepDefinition(
                id="estimate",
                name="Estimate Items",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Estimate selected items and validate against capacity",
                },
                next_steps=["finalize"],
            ),
            StepDefinition(
                id="finalize",
                name="Finalize Sprint",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Generate sprint commitment with goals and risks",
                },
                next_steps=["approval"],
            ),
            StepDefinition(
                id="approval",
                name="Team Approval",
                step_type="human_checkpoint",
                config={
                    "checkpoint_type": "approval",
                    "title": f"Approve {sprint_name} commitment",
                },
                next_steps=["store"],
            ),
            StepDefinition(
                id="store",
                name="Record Sprint",
                step_type="memory_write",
                config={
                    "collection": "sprint_plans",
                },
                next_steps=[],
            ),
        ],
        entry_step="capacity",
    )


def create_contract_review_workflow(
    contract_type: str,
    counterparty: str,
    contract_value: str | None = None,
    key_terms: list[str] | None = None,
    concerns: list[str] | None = None,
) -> WorkflowDefinition:
    """Create a contract review workflow.

    Uses multi-agent debate to identify risks, negotiate points,
    and ensure favorable terms.

    Args:
        contract_type: Type of contract (e.g., "SaaS", "NDA", "Employment")
        counterparty: Other party name
        contract_value: Contract value if applicable
        key_terms: Important terms to focus on
        concerns: Specific concerns to address

    Returns:
        WorkflowDefinition for contract review

    Example:
        workflow = create_contract_review_workflow(
            contract_type="SaaS Agreement",
            counterparty="Vendor Corp",
            contract_value="$120k/year",
            key_terms=["SLA", "data ownership", "termination"],
        )
    """
    workflow_id = (
        f"contract_{contract_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
    )
    terms_focus = key_terms or ["liability", "termination", "IP", "confidentiality"]

    return WorkflowDefinition(
        id=workflow_id,
        name=f"Contract Review: {counterparty}",
        description=f"Review {contract_type} with {counterparty}",
        category=WorkflowCategory.GENERAL,
        tags=["sme", "legal", "contract", "review"],
        inputs={
            "contract_type": contract_type,
            "counterparty": counterparty,
            "contract_value": contract_value,
            "key_terms": terms_focus,
            "concerns": concerns or [],
        },
        steps=[
            StepDefinition(
                id="parse",
                name="Parse Contract",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": f"Extract key clauses from {contract_type}",
                },
                next_steps=["debate"],
            ),
            StepDefinition(
                id="debate",
                name="Risk Assessment Debate",
                step_type="debate",
                config={
                    "agents": ["claude", "gpt-4"],
                    "topic": f"What are the risks in this {contract_type}?",
                    "rounds": 3,
                    "focus_terms": terms_focus,
                },
                next_steps=["risks"],
            ),
            StepDefinition(
                id="risks",
                name="Identify Red Flags",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "List potential risks and unfavorable terms",
                },
                next_steps=["negotiate"],
            ),
            StepDefinition(
                id="negotiate",
                name="Negotiation Points",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Generate negotiation points and alternative language",
                },
                next_steps=["summary"],
            ),
            StepDefinition(
                id="summary",
                name="Executive Summary",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Create executive summary with recommendation",
                },
                next_steps=["review"],
            ),
            StepDefinition(
                id="review",
                name="Legal Review",
                step_type="human_checkpoint",
                config={
                    "checkpoint_type": "approval",
                    "title": f"Approve {counterparty} contract analysis",
                },
                next_steps=["store"],
            ),
            StepDefinition(
                id="store",
                name="Record Review",
                step_type="memory_write",
                config={
                    "collection": "contract_reviews",
                },
                next_steps=[],
            ),
        ],
        entry_step="parse",
    )


def create_business_decision_workflow(
    decision_topic: str,
    context: str | None = None,
    stakeholders: list[str] | None = None,
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
    "create_report_workflow",
    "weekly_sales_report",
    "create_budget_allocation_workflow",
    "create_feature_prioritization_workflow",
    "create_sprint_planning_workflow",
    "create_contract_review_workflow",
    "create_business_decision_workflow",
]
