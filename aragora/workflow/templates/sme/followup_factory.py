"""Customer follow-up and HR workflow factory functions.

Provides workflow templates for customer relationship management,
hiring decisions, performance reviews, and HR policy workflows.
"""

from __future__ import annotations

from datetime import datetime

from aragora.workflow.types import (
    StepDefinition,
    WorkflowCategory,
    WorkflowDefinition,
)


def create_followup_workflow(
    followup_type: str = "check_in",
    days_since_contact: int = 30,
    channel: str = "email",
    auto_send: bool = False,
    customer_id: str | None = None,
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


def create_hiring_decision_workflow(
    position: str,
    candidate_name: str,
    interview_notes: str | None = None,
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


def create_performance_review_workflow(
    employee_name: str,
    role: str,
    review_period: str = "Q4 2024",
    self_assessment: str | None = None,
    manager_notes: str | None = None,
    goals_met: list[str] | None = None,
) -> WorkflowDefinition:
    """Create a performance review workflow.

    Uses multi-agent debate to ensure balanced, fair performance
    assessment with multiple perspectives.

    Args:
        employee_name: Employee being reviewed
        role: Employee's role/title
        review_period: Review period (e.g., "Q4 2024", "Annual 2024")
        self_assessment: Employee's self-assessment summary
        manager_notes: Manager's preliminary notes
        goals_met: List of goals achieved/not achieved

    Returns:
        WorkflowDefinition for performance review

    Example:
        workflow = create_performance_review_workflow(
            employee_name="Jane Doe",
            role="Senior Developer",
            review_period="Q4 2024",
            self_assessment="Exceeded sprint goals, led migration project...",
        )
    """
    workflow_id = (
        f"perf_review_{role.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
    )

    return WorkflowDefinition(
        id=workflow_id,
        name=f"Performance Review: {employee_name}",
        description=f"Review {employee_name}'s performance for {review_period}",
        category=WorkflowCategory.GENERAL,
        tags=["sme", "hr", "performance", "review"],
        inputs={
            "employee_name": employee_name,
            "role": role,
            "review_period": review_period,
            "self_assessment": self_assessment,
            "manager_notes": manager_notes,
            "goals_met": goals_met or [],
        },
        steps=[
            StepDefinition(
                id="context",
                name="Gather Context",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": f"Summarize {employee_name}'s role expectations and review context",
                },
                next_steps=["debate"],
            ),
            StepDefinition(
                id="debate",
                name="Multi-Perspective Review",
                step_type="debate",
                config={
                    "agents": ["claude", "gemini"],
                    "topic": f"Evaluate {employee_name}'s performance in {role}",
                    "rounds": 2,
                    "focus_areas": ["achievements", "growth_areas", "team_contribution"],
                },
                next_steps=["strengths"],
            ),
            StepDefinition(
                id="strengths",
                name="Identify Strengths",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Identify key strengths and accomplishments",
                },
                next_steps=["development"],
            ),
            StepDefinition(
                id="development",
                name="Development Areas",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Identify growth opportunities and development areas",
                },
                next_steps=["recommend"],
            ),
            StepDefinition(
                id="recommend",
                name="Generate Recommendation",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Generate balanced performance review with rating recommendation",
                },
                next_steps=["review"],
            ),
            StepDefinition(
                id="review",
                name="Manager Review",
                step_type="human_checkpoint",
                config={
                    "checkpoint_type": "approval",
                    "title": f"Review {employee_name}'s performance assessment",
                },
                next_steps=["store"],
            ),
            StepDefinition(
                id="store",
                name="Record Review",
                step_type="memory_write",
                config={
                    "collection": "performance_reviews",
                },
                next_steps=[],
            ),
        ],
        entry_step="context",
    )


def create_remote_work_policy_workflow(
    company_size: int = 50,
    current_policy: str | None = None,
    concerns: list[str] | None = None,
    industry: str = "tech",
) -> WorkflowDefinition:
    """Create a remote work policy review workflow.

    Uses multi-agent debate to design or refine remote work
    policies balancing flexibility and productivity.

    Args:
        company_size: Number of employees
        current_policy: Existing policy summary if any
        concerns: Specific concerns to address
        industry: Company industry

    Returns:
        WorkflowDefinition for remote work policy

    Example:
        workflow = create_remote_work_policy_workflow(
            company_size=75,
            current_policy="3 days in office required",
            concerns=["collaboration", "timezone coverage"],
            industry="fintech",
        )
    """
    workflow_id = f"remote_policy_{datetime.now().strftime('%Y%m%d')}"

    return WorkflowDefinition(
        id=workflow_id,
        name="Remote Work Policy Review",
        description=f"Design remote work policy for {company_size}-person {industry} company",
        category=WorkflowCategory.GENERAL,
        tags=["sme", "hr", "policy", "remote"],
        inputs={
            "company_size": company_size,
            "current_policy": current_policy,
            "concerns": concerns or [],
            "industry": industry,
        },
        steps=[
            StepDefinition(
                id="benchmark",
                name="Industry Benchmark",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": f"Research remote work best practices for {industry}",
                },
                next_steps=["debate"],
            ),
            StepDefinition(
                id="debate",
                name="Policy Debate",
                step_type="debate",
                config={
                    "agents": ["claude", "gpt-4", "gemini"],
                    "topic": "What remote work policy best balances flexibility and productivity?",
                    "rounds": 3,
                    "perspectives": ["employee", "manager", "executive"],
                },
                next_steps=["draft"],
            ),
            StepDefinition(
                id="draft",
                name="Draft Policy",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Draft comprehensive remote work policy",
                },
                next_steps=["legal"],
            ),
            StepDefinition(
                id="legal",
                name="Legal Considerations",
                step_type="agent",
                config={
                    "agent_type": "claude",
                    "prompt_template": "Identify legal and compliance considerations",
                },
                next_steps=["review"],
            ),
            StepDefinition(
                id="review",
                name="HR Review",
                step_type="human_checkpoint",
                config={
                    "checkpoint_type": "approval",
                    "title": "Review remote work policy draft",
                },
                next_steps=["store"],
            ),
            StepDefinition(
                id="store",
                name="Record Policy",
                step_type="memory_write",
                config={
                    "collection": "hr_policies",
                },
                next_steps=[],
            ),
        ],
        entry_step="benchmark",
    )


__all__ = [
    "create_followup_workflow",
    "renewal_followup_campaign",
    "create_hiring_decision_workflow",
    "create_performance_review_workflow",
    "create_remote_work_policy_workflow",
]
