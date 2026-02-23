"""
Pipeline Template Library.

Pre-built pipeline templates for common decision workflows.
Each template pre-populates Stage 1 ideas, configures agents,
and returns a PipelineConfig with the right settings pre-filled.

Usage:
    from aragora.pipeline.templates import get_template, list_templates

    templates = list_templates()
    template = get_template("strategic_review")
    config = template.to_pipeline_config()
    ideas = template.seed_ideas

    # Or use the convenience function:
    config, ideas = get_template_config("product_launch")
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PipelineTemplate:
    """A pre-built pipeline template for a common decision workflow.

    Combines seed ideas with a pre-configured PipelineConfig so SMEs
    can start a pipeline without understanding every knob.
    """

    name: str
    display_name: str
    description: str
    category: str
    stage_1_ideas: list[str] = field(default_factory=list)
    agent_config: dict[str, Any] = field(default_factory=dict)
    vertical_profile: str | None = None
    goal_extraction_hints: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    # PipelineConfig settings
    workflow_mode: str = "quick"  # "quick" or "debate"
    enable_smart_goals: bool = True
    enable_elo_assignment: bool = True
    enable_km_precedents: bool = True
    human_approval_required: bool = False
    dry_run: bool = False

    @property
    def seed_ideas(self) -> list[str]:
        """Alias for stage_1_ideas for clearer API."""
        return self.stage_1_ideas

    def to_pipeline_config(self) -> Any:
        """Return a PipelineConfig pre-filled with this template's settings.

        Returns:
            PipelineConfig configured for this template's use case.
        """
        from aragora.pipeline.idea_to_execution import PipelineConfig

        return PipelineConfig(
            workflow_mode=self.workflow_mode,
            enable_smart_goals=self.enable_smart_goals,
            enable_elo_assignment=self.enable_elo_assignment,
            enable_km_precedents=self.enable_km_precedents,
            dry_run=self.dry_run,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category,
            "stage_1_ideas": self.stage_1_ideas,
            "vertical_profile": self.vertical_profile,
            "tags": self.tags,
            "idea_count": len(self.stage_1_ideas),
            "workflow_mode": self.workflow_mode,
            "enable_smart_goals": self.enable_smart_goals,
            "enable_elo_assignment": self.enable_elo_assignment,
            "enable_km_precedents": self.enable_km_precedents,
            "human_approval_required": self.human_approval_required,
        }

    def create_pipeline(self) -> Any:
        """Create an IdeaToExecutionPipeline result from this template.

        Returns a PipelineResult with Stage 1 pre-populated from the
        template's ideas, ready for the user to review and advance.
        """
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_ideas(
            self.stage_1_ideas,
            auto_advance=False,
            pipeline_id=f"pipe-{self.name}-{uuid.uuid4().hex[:8]}",
        )
        # Attach template metadata to the result
        result.stage_status["_template"] = self.name
        return result


# =============================================================================
# Template Definitions
# =============================================================================

TEMPLATE_REGISTRY: dict[str, PipelineTemplate] = {}


def _register(template: PipelineTemplate) -> PipelineTemplate:
    """Register a template in the global registry."""
    TEMPLATE_REGISTRY[template.name] = template
    return template


# ---------------------------------------------------------------------------
# SME Pipeline Templates
# ---------------------------------------------------------------------------

_register(
    PipelineTemplate(
        name="product_launch",
        display_name="Product Launch",
        description=(
            "Ideas for new product features or products flow through goal-setting, "
            "action planning, and execution. Smart goals are enabled so each idea "
            "becomes measurable, and human approval gates ensure nothing ships "
            "without sign-off."
        ),
        category="product",
        workflow_mode="quick",
        enable_smart_goals=True,
        enable_elo_assignment=True,
        enable_km_precedents=True,
        human_approval_required=True,
        stage_1_ideas=[
            "Define target customer persona and validate with user interviews",
            "Identify top 3 competitor weaknesses to exploit at launch",
            "Scope the minimum viable feature set for first release",
            "Draft go-to-market messaging and positioning statement",
            "Set pricing tiers based on willingness-to-pay research",
            "Plan launch-day marketing blitz across email, social, and PR",
            "Establish post-launch success metrics and review cadence",
        ],
        agent_config={
            "min_agents": 4,
            "preferred_archetypes": ["strategist", "analyst", "critic", "implementer"],
        },
        vertical_profile=None,
        goal_extraction_hints={
            "prioritize": ["market_validation", "technical_readiness", "revenue_model"],
        },
        tags=["product", "launch", "gtm", "strategy"],
    )
)

_register(
    PipelineTemplate(
        name="bug_triage",
        display_name="Bug Triage",
        description=(
            "Bug reports are prioritized by severity and impact, then routed to "
            "the best debugging agents via ELO assignment. Quick mode keeps "
            "turnaround fast; no human gate so fixes can flow straight to PR."
        ),
        category="engineering",
        workflow_mode="quick",
        enable_smart_goals=False,
        enable_elo_assignment=True,
        enable_km_precedents=False,
        human_approval_required=False,
        stage_1_ideas=[
            "Reproduce the bug with a minimal test case",
            "Classify severity: P0 (outage), P1 (degraded), P2 (cosmetic)",
            "Identify root cause via stack trace and log analysis",
            "Check for related regressions in recent commits",
            "Draft a fix with unit test covering the failure mode",
            "Verify the fix does not break adjacent functionality",
        ],
        agent_config={
            "min_agents": 2,
            "preferred_archetypes": ["implementer", "tester"],
        },
        vertical_profile=None,
        goal_extraction_hints={
            "prioritize": ["severity_classification", "root_cause", "regression_check"],
        },
        tags=["engineering", "bugs", "triage", "debugging"],
    )
)

_register(
    PipelineTemplate(
        name="content_calendar",
        display_name="Content Calendar",
        description=(
            "Content ideas are organized into themed goals with SMART criteria, "
            "then scheduled into a publishing calendar with assigned owners and "
            "deadlines."
        ),
        category="marketing",
        workflow_mode="quick",
        enable_smart_goals=True,
        enable_elo_assignment=False,
        enable_km_precedents=True,
        human_approval_required=False,
        stage_1_ideas=[
            "Audit existing content for gaps in the customer journey",
            "Research trending topics in our industry vertical",
            "Plan a monthly theme that ties individual pieces together",
            "Assign content types: blog posts, videos, social threads, whitepapers",
            "Set publication dates and responsible authors for each piece",
            "Define promotion strategy per channel (email, LinkedIn, Twitter)",
            "Establish engagement KPIs: views, shares, leads generated",
        ],
        agent_config={
            "min_agents": 3,
            "preferred_archetypes": ["strategist", "analyst", "implementer"],
        },
        vertical_profile=None,
        goal_extraction_hints={
            "prioritize": ["content_gaps", "audience_alignment", "publishing_cadence"],
        },
        tags=["marketing", "content", "calendar", "publishing"],
    )
)

_register(
    PipelineTemplate(
        name="strategic_review",
        display_name="Strategic Review",
        description=(
            "Strategy proposals are debated by diverse agents to surface conflicts "
            "and blind spots. KM precedents enrich evaluation with historical "
            "outcomes. Human approval gates ensure executive sign-off before "
            "execution begins."
        ),
        category="strategy",
        workflow_mode="debate",
        enable_smart_goals=True,
        enable_elo_assignment=True,
        enable_km_precedents=True,
        human_approval_required=True,
        stage_1_ideas=[
            "Articulate the strategic objective and desired end state",
            "Map current competitive position and market dynamics",
            "Identify resource constraints and organizational readiness",
            "Surface conflicting priorities across business units",
            "Evaluate risk scenarios: best case, worst case, most likely",
            "Define measurable milestones for the first 90 days",
            "Establish a review cadence with clear escalation triggers",
        ],
        agent_config={
            "min_agents": 5,
            "preferred_archetypes": ["strategist", "analyst", "critic", "advocate", "finance"],
        },
        vertical_profile=None,
        goal_extraction_hints={
            "prioritize": ["conflict_detection", "resource_alignment", "risk_assessment"],
        },
        tags=["strategy", "review", "planning", "executive"],
    )
)

_register(
    PipelineTemplate(
        name="hiring_pipeline",
        display_name="Hiring Pipeline",
        description=(
            "Candidate requirements flow into structured role goals, then into "
            "an interview process with evaluation rubrics. Human approval is "
            "required at every stage to ensure fair, compliant hiring."
        ),
        category="people",
        workflow_mode="quick",
        enable_smart_goals=True,
        enable_elo_assignment=False,
        enable_km_precedents=True,
        human_approval_required=True,
        stage_1_ideas=[
            "Define the role: responsibilities, required skills, and seniority",
            "Write a job description that attracts diverse qualified candidates",
            "Design a structured interview rubric with scoring criteria",
            "Plan interview stages: screen, technical, culture, final",
            "Set compensation range based on market data and internal equity",
            "Prepare onboarding checklist for the first 30/60/90 days",
        ],
        agent_config={
            "min_agents": 3,
            "preferred_archetypes": ["analyst", "advocate", "critic"],
        },
        vertical_profile=None,
        goal_extraction_hints={
            "prioritize": ["role_definition", "interview_design", "compensation_equity"],
        },
        tags=["hr", "hiring", "recruiting", "people"],
    )
)

_register(
    PipelineTemplate(
        name="compliance_audit",
        display_name="Compliance Audit",
        description=(
            "Compliance gaps are surfaced through adversarial debate, then "
            "converted into remediation goals with deadlines and owners. Human "
            "approval gates ensure legal and executive sign-off on every "
            "remediation action."
        ),
        category="compliance",
        workflow_mode="debate",
        enable_smart_goals=True,
        enable_elo_assignment=True,
        enable_km_precedents=True,
        human_approval_required=True,
        stage_1_ideas=[
            "Identify applicable regulations and compliance frameworks",
            "Map current controls to regulatory requirements",
            "Perform gap analysis against compliance standards",
            "Assess risk severity for each identified gap",
            "Develop remediation plan with timelines and owners",
            "Prepare audit-ready evidence and documentation",
        ],
        agent_config={
            "min_agents": 3,
            "preferred_archetypes": ["analyst", "auditor", "implementer"],
        },
        vertical_profile="compliance_sox",
        goal_extraction_hints={
            "prioritize": ["gap_analysis", "risk_severity", "remediation"],
        },
        tags=["compliance", "audit", "regulation", "sox", "controls"],
    )
)


# ---------------------------------------------------------------------------
# Legacy templates (preserved for backward compatibility)
# ---------------------------------------------------------------------------

_register(
    PipelineTemplate(
        name="hiring_decision",
        display_name="Hiring Decision",
        description=(
            "Evaluate candidates with diverse agent perspectives covering "
            "technical skills, culture fit, growth potential, and risk factors."
        ),
        category="evaluation",
        workflow_mode="quick",
        enable_smart_goals=True,
        enable_elo_assignment=False,
        enable_km_precedents=False,
        human_approval_required=True,
        stage_1_ideas=[
            "Define role requirements and must-have qualifications",
            "Evaluate candidate technical skills against job requirements",
            "Assess culture fit and team dynamics compatibility",
            "Analyze growth potential and career trajectory alignment",
            "Identify risk factors and reference check findings",
            "Compare compensation expectations with budget and market rates",
        ],
        agent_config={
            "min_agents": 3,
            "preferred_archetypes": ["analyst", "advocate", "critic"],
        },
        vertical_profile=None,
        goal_extraction_hints={
            "prioritize": ["qualification_match", "culture_fit", "risk_assessment"],
        },
        tags=["hr", "hiring", "people", "evaluation"],
    )
)

_register(
    PipelineTemplate(
        name="market_entry",
        display_name="Market Entry Strategy",
        description=(
            "Evaluate new market opportunities: competitive analysis, "
            "entry strategy selection, resource planning, and execution roadmap."
        ),
        category="expansion",
        workflow_mode="debate",
        enable_smart_goals=True,
        enable_elo_assignment=True,
        enable_km_precedents=True,
        human_approval_required=True,
        stage_1_ideas=[
            "Analyze target market dynamics and growth potential",
            "Profile key competitors and their market positions",
            "Evaluate entry barriers and regulatory requirements",
            "Assess internal capabilities and resource gaps",
            "Compare entry strategies: organic, acquisition, partnership",
            "Model financial scenarios and investment requirements",
            "Define success metrics and decision checkpoints",
        ],
        agent_config={
            "min_agents": 4,
            "preferred_archetypes": ["strategist", "analyst", "critic", "finance"],
        },
        vertical_profile=None,
        goal_extraction_hints={
            "prioritize": ["market_size", "competitive_advantage", "resource_requirements"],
        },
        tags=["strategy", "market", "expansion", "growth"],
    )
)

_register(
    PipelineTemplate(
        name="vendor_selection",
        display_name="Vendor Selection",
        description=(
            "Systematic vendor evaluation: requirements definition, "
            "scoring criteria, comparative analysis, and selection decision."
        ),
        category="procurement",
        workflow_mode="quick",
        enable_smart_goals=True,
        enable_elo_assignment=False,
        enable_km_precedents=True,
        human_approval_required=True,
        stage_1_ideas=[
            "Define functional and non-functional requirements",
            "Establish evaluation criteria and scoring weights",
            "Shortlist vendors based on initial screening",
            "Conduct detailed vendor capability assessment",
            "Evaluate total cost of ownership and pricing models",
            "Assess vendor reliability, support, and track record",
            "Perform security and compliance due diligence",
        ],
        agent_config={
            "min_agents": 3,
            "preferred_archetypes": ["analyst", "critic", "finance"],
        },
        vertical_profile=None,
        goal_extraction_hints={
            "prioritize": ["requirements_match", "cost_analysis", "risk_assessment"],
        },
        tags=["procurement", "vendor", "evaluation", "selection"],
    )
)


# =============================================================================
# Public API
# =============================================================================


def get_template(name: str) -> PipelineTemplate | None:
    """Get a pipeline template by name.

    Args:
        name: Template name (e.g., "strategic_review")

    Returns:
        PipelineTemplate if found, None otherwise
    """
    return TEMPLATE_REGISTRY.get(name)


def list_templates(category: str | None = None) -> list[PipelineTemplate]:
    """List all available pipeline templates.

    Args:
        category: Optional category filter

    Returns:
        List of PipelineTemplate objects
    """
    templates = list(TEMPLATE_REGISTRY.values())
    if category:
        templates = [t for t in templates if t.category == category]
    return templates


def get_template_config(name: str) -> tuple[Any, list[str]] | None:
    """Convenience: get a PipelineConfig and seed ideas in one call.

    Args:
        name: Template name (e.g., "product_launch")

    Returns:
        Tuple of (PipelineConfig, seed_ideas) if found, None otherwise.
    """
    template = get_template(name)
    if template is None:
        return None
    return template.to_pipeline_config(), template.seed_ideas
