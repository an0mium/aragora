"""
Pipeline Template Library.

Pre-built pipeline templates for common decision workflows.
Each template pre-populates Stage 1 ideas, configures agents,
and selects appropriate vertical evaluation profiles.

Usage:
    from aragora.pipeline.templates import get_template, list_templates

    templates = list_templates()
    template = get_template("hiring_decision")
    pipeline_result = template.create_pipeline()
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PipelineTemplate:
    """A pre-built pipeline template for a common decision workflow."""

    name: str
    display_name: str
    description: str
    category: str
    stage_1_ideas: list[str] = field(default_factory=list)
    agent_config: dict[str, Any] = field(default_factory=dict)
    vertical_profile: str | None = None
    goal_extraction_hints: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

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


_register(PipelineTemplate(
    name="hiring_decision",
    display_name="Hiring Decision",
    description=(
        "Evaluate candidates with diverse agent perspectives covering "
        "technical skills, culture fit, growth potential, and risk factors."
    ),
    category="people",
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
))

_register(PipelineTemplate(
    name="product_launch",
    display_name="Product Launch",
    description=(
        "End-to-end product launch pipeline: market analysis, competitive "
        "positioning, go/no-go decision, and launch execution plan."
    ),
    category="product",
    stage_1_ideas=[
        "Analyze target market size and customer segments",
        "Map competitive landscape and identify differentiation",
        "Define minimum viable feature set for launch",
        "Assess technical readiness and infrastructure requirements",
        "Develop pricing strategy and revenue model",
        "Plan marketing channels and go-to-market timeline",
        "Identify launch risks and mitigation strategies",
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
))

_register(PipelineTemplate(
    name="compliance_audit",
    display_name="Compliance Audit",
    description=(
        "Structured compliance review: regulation mapping, gap analysis, "
        "remediation planning, and audit-ready documentation."
    ),
    category="compliance",
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
))

_register(PipelineTemplate(
    name="market_entry",
    display_name="Market Entry Strategy",
    description=(
        "Evaluate new market opportunities: competitive analysis, "
        "entry strategy selection, resource planning, and execution roadmap."
    ),
    category="strategy",
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
))

_register(PipelineTemplate(
    name="vendor_selection",
    display_name="Vendor Selection",
    description=(
        "Systematic vendor evaluation: requirements definition, "
        "scoring criteria, comparative analysis, and selection decision."
    ),
    category="procurement",
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
))


# =============================================================================
# Public API
# =============================================================================


def get_template(name: str) -> PipelineTemplate | None:
    """Get a pipeline template by name.

    Args:
        name: Template name (e.g., "hiring_decision")

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
