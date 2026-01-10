"""
Domain-Specific Debate Templates.

Pre-built debate structures for high-value use cases:
- Code Review
- RFC/Design Doc Review
- Incident Response
- Research Synthesis
- Policy Review
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from enum import Enum


class TemplateType(Enum):
    """Types of debate templates."""

    CODE_REVIEW = "code_review"
    DESIGN_DOC = "design_doc"
    INCIDENT_RESPONSE = "incident_response"
    RESEARCH_SYNTHESIS = "research_synthesis"
    POLICY_REVIEW = "policy_review"
    SECURITY_AUDIT = "security_audit"
    ARCHITECTURE_REVIEW = "architecture_review"
    PRODUCT_STRATEGY = "product_strategy"


@dataclass
class DebateRole:
    """A role in a debate template."""

    name: str
    description: str
    objectives: list[str]
    evaluation_criteria: list[str]
    example_prompts: list[str] = field(default_factory=list)


@dataclass
class DebatePhase:
    """A phase in a structured debate."""

    name: str
    description: str
    duration_rounds: int
    roles_active: list[str]
    objectives: list[str]
    outputs: list[str]


@dataclass
class DebateTemplate:
    """A complete debate template."""

    template_id: str
    template_type: TemplateType
    name: str
    description: str

    # Structure
    roles: list[DebateRole]
    phases: list[DebatePhase]

    # Configuration
    recommended_agents: int
    max_rounds: int
    consensus_threshold: float

    # Evaluation
    rubric: dict[str, float]  # Criteria -> weight
    output_format: str  # Markdown template for output

    # Metadata
    domain: str
    difficulty: float = 0.5
    tags: list[str] = field(default_factory=list)


# ============================================================================
# Code Review Template
# ============================================================================

CODE_REVIEW_TEMPLATE = DebateTemplate(
    template_id="code-review-v1",
    template_type=TemplateType.CODE_REVIEW,
    name="Multi-Agent Code Review",
    description="Structured code review with security, performance, and maintainability critics",

    roles=[
        DebateRole(
            name="author",
            description="Presents and defends the code",
            objectives=["Explain design decisions", "Address concerns", "Accept valid feedback"],
            evaluation_criteria=["Responsiveness", "Openness to feedback"],
            example_prompts=[
                "This implementation uses X because...",
                "I chose this approach to optimize for...",
            ],
        ),
        DebateRole(
            name="security_critic",
            description="Reviews for security vulnerabilities",
            objectives=["Find injection risks", "Check auth/authz", "Identify data exposure"],
            evaluation_criteria=["OWASP coverage", "Severity accuracy"],
            example_prompts=[
                "This input is not sanitized, allowing SQL injection via...",
                "The authentication check can be bypassed by...",
            ],
        ),
        DebateRole(
            name="performance_critic",
            description="Reviews for performance issues",
            objectives=["Find N+1 queries", "Identify bottlenecks", "Check resource usage"],
            evaluation_criteria=["Impact accuracy", "Scalability awareness"],
            example_prompts=[
                "This loop has O(n²) complexity because...",
                "This query will cause N+1 issues when...",
            ],
        ),
        DebateRole(
            name="maintainability_critic",
            description="Reviews for code quality and maintainability",
            objectives=["Check readability", "Identify code smells", "Suggest refactoring"],
            evaluation_criteria=["Actionable suggestions", "Balance of concerns"],
            example_prompts=[
                "This function does too many things. Consider splitting...",
                "The naming convention is inconsistent with...",
            ],
        ),
        DebateRole(
            name="synthesizer",
            description="Synthesizes feedback into actionable review",
            objectives=["Prioritize issues", "Resolve conflicts", "Create action items"],
            evaluation_criteria=["Clarity", "Prioritization"],
            example_prompts=[
                "The critical issues are: 1) Security: X, 2) Performance: Y",
                "Recommended changes in order of priority...",
            ],
        ),
    ],

    phases=[
        DebatePhase(
            name="initial_review",
            description="Each critic reviews independently",
            duration_rounds=1,
            roles_active=["security_critic", "performance_critic", "maintainability_critic"],
            objectives=["Identify issues in respective domains"],
            outputs=["Issue list per domain"],
        ),
        DebatePhase(
            name="author_response",
            description="Author responds to critiques",
            duration_rounds=1,
            roles_active=["author"],
            objectives=["Address concerns", "Explain decisions", "Accept valid points"],
            outputs=["Response to each critique"],
        ),
        DebatePhase(
            name="debate",
            description="Critics and author discuss unresolved issues",
            duration_rounds=2,
            roles_active=["author", "security_critic", "performance_critic", "maintainability_critic"],
            objectives=["Resolve disagreements", "Prioritize issues"],
            outputs=["Consensus on critical issues"],
        ),
        DebatePhase(
            name="synthesis",
            description="Create final review document",
            duration_rounds=1,
            roles_active=["synthesizer"],
            objectives=["Summarize findings", "Create action items"],
            outputs=["Final review with risk score"],
        ),
    ],

    recommended_agents=4,
    max_rounds=5,
    consensus_threshold=0.7,

    rubric={
        "security_coverage": 0.3,
        "performance_impact": 0.2,
        "maintainability": 0.2,
        "actionability": 0.2,
        "consensus": 0.1,
    },

    output_format="""
# Code Review Summary

## Risk Score: {risk_score}/10

## Critical Issues
{critical_issues}

## Security ({security_score})
{security_findings}

## Performance ({performance_score})
{performance_findings}

## Maintainability ({maintainability_score})
{maintainability_findings}

## Action Items
{action_items}

## Consensus Notes
{consensus_notes}
""",

    domain="software_engineering",
    difficulty=0.6,
    tags=["code", "review", "security", "performance"],
)


# ============================================================================
# Design Doc / RFC Template
# ============================================================================

DESIGN_DOC_TEMPLATE = DebateTemplate(
    template_id="design-doc-v1",
    template_type=TemplateType.DESIGN_DOC,
    name="RFC/Design Doc Review",
    description="Structured review for technical design documents and RFCs",

    roles=[
        DebateRole(
            name="author",
            description="Presents the design",
            objectives=["Explain architecture", "Justify tradeoffs", "Address concerns"],
            evaluation_criteria=["Clarity", "Completeness"],
        ),
        DebateRole(
            name="devils_advocate",
            description="Challenges assumptions and proposes alternatives",
            objectives=["Find unstated assumptions", "Propose alternatives", "Stress-test scalability"],
            evaluation_criteria=["Quality of alternatives", "Assumption coverage"],
        ),
        DebateRole(
            name="stakeholder",
            description="Represents user/business requirements",
            objectives=["Ensure requirements met", "Identify gaps", "Validate priorities"],
            evaluation_criteria=["Requirements coverage", "Priority alignment"],
        ),
        DebateRole(
            name="implementer",
            description="Evaluates implementation feasibility",
            objectives=["Assess complexity", "Identify blockers", "Estimate effort"],
            evaluation_criteria=["Realism", "Risk identification"],
        ),
    ],

    phases=[
        DebatePhase(
            name="presentation",
            description="Author presents the design",
            duration_rounds=1,
            roles_active=["author"],
            objectives=["Present problem and solution"],
            outputs=["Design overview"],
        ),
        DebatePhase(
            name="critique",
            description="Each role critiques from their perspective",
            duration_rounds=1,
            roles_active=["devils_advocate", "stakeholder", "implementer"],
            objectives=["Identify concerns"],
            outputs=["Critique list"],
        ),
        DebatePhase(
            name="debate",
            description="Resolve disagreements through structured debate",
            duration_rounds=3,
            roles_active=["author", "devils_advocate", "stakeholder", "implementer"],
            objectives=["Reach consensus on key decisions"],
            outputs=["Decision record"],
        ),
    ],

    recommended_agents=4,
    max_rounds=5,
    consensus_threshold=0.6,

    rubric={
        "technical_soundness": 0.25,
        "requirement_coverage": 0.25,
        "feasibility": 0.2,
        "risk_assessment": 0.15,
        "alternatives_considered": 0.15,
    },

    output_format="""
# Design Review: {title}

## Decision: {decision} (Confidence: {confidence})

## Summary
{summary}

## Key Decisions
{decisions}

## Risks & Mitigations
{risks}

## Open Questions
{questions}

## Next Steps
{next_steps}
""",

    domain="architecture",
    difficulty=0.7,
    tags=["design", "rfc", "architecture", "review"],
)


# ============================================================================
# Incident Response Template
# ============================================================================

INCIDENT_RESPONSE_TEMPLATE = DebateTemplate(
    template_id="incident-response-v1",
    template_type=TemplateType.INCIDENT_RESPONSE,
    name="Incident Response Analysis",
    description="Structured post-incident analysis and remediation planning",

    roles=[
        DebateRole(
            name="investigator",
            description="Leads root cause analysis",
            objectives=["Establish timeline", "Identify root cause", "Document evidence"],
            evaluation_criteria=["Thoroughness", "Evidence quality"],
        ),
        DebateRole(
            name="responder",
            description="Evaluates response effectiveness",
            objectives=["Assess response time", "Identify gaps", "Evaluate communication"],
            evaluation_criteria=["Response analysis", "Gap identification"],
        ),
        DebateRole(
            name="prevention",
            description="Proposes preventive measures",
            objectives=["Suggest monitoring", "Propose safeguards", "Recommend processes"],
            evaluation_criteria=["Actionability", "Coverage"],
        ),
        DebateRole(
            name="challenger",
            description="Challenges proposed solutions",
            objectives=["Find gaps in solutions", "Stress-test proposals", "Consider edge cases"],
            evaluation_criteria=["Critical thinking", "Scenario coverage"],
        ),
    ],

    phases=[
        DebatePhase(
            name="timeline",
            description="Establish incident timeline",
            duration_rounds=1,
            roles_active=["investigator"],
            objectives=["Document what happened when"],
            outputs=["Timeline"],
        ),
        DebatePhase(
            name="root_cause",
            description="Debate root cause",
            duration_rounds=2,
            roles_active=["investigator", "responder", "challenger"],
            objectives=["Agree on root cause(s)"],
            outputs=["Root cause analysis"],
        ),
        DebatePhase(
            name="remediation",
            description="Plan remediation",
            duration_rounds=2,
            roles_active=["prevention", "challenger", "responder"],
            objectives=["Agree on action items"],
            outputs=["Remediation plan"],
        ),
    ],

    recommended_agents=4,
    max_rounds=5,
    consensus_threshold=0.7,

    rubric={
        "root_cause_accuracy": 0.3,
        "timeline_completeness": 0.2,
        "remediation_quality": 0.3,
        "prevention_coverage": 0.2,
    },

    output_format="""
# Incident Post-Mortem: {incident_id}

## Severity: {severity}
## Duration: {duration}
## Impact: {impact}

## Timeline
{timeline}

## Root Cause
{root_cause}

## What Went Well
{went_well}

## What Went Poorly
{went_poorly}

## Action Items
{action_items}

## Prevention Measures
{prevention}
""",

    domain="operations",
    difficulty=0.8,
    tags=["incident", "postmortem", "sre", "reliability"],
)


# ============================================================================
# Research Synthesis Template
# ============================================================================

RESEARCH_SYNTHESIS_TEMPLATE = DebateTemplate(
    template_id="research-synthesis-v1",
    template_type=TemplateType.RESEARCH_SYNTHESIS,
    name="Research Synthesis",
    description="Synthesize findings from multiple research sources",

    roles=[
        DebateRole(
            name="extractor",
            description="Extracts key claims from sources",
            objectives=["Identify key findings", "Extract methodology", "Note limitations"],
            evaluation_criteria=["Accuracy", "Completeness"],
        ),
        DebateRole(
            name="validator",
            description="Cross-validates claims across sources",
            objectives=["Find agreements", "Identify conflicts", "Assess evidence quality"],
            evaluation_criteria=["Rigor", "Citation accuracy"],
        ),
        DebateRole(
            name="synthesizer",
            description="Creates unified synthesis",
            objectives=["Integrate findings", "Identify gaps", "Propose conclusions"],
            evaluation_criteria=["Coherence", "Insight quality"],
        ),
        DebateRole(
            name="critic",
            description="Challenges synthesis quality",
            objectives=["Check for bias", "Identify missing perspectives", "Assess confidence"],
            evaluation_criteria=["Critical thinking", "Balance"],
        ),
    ],

    phases=[
        DebatePhase(
            name="extraction",
            description="Extract claims from sources",
            duration_rounds=1,
            roles_active=["extractor"],
            objectives=["Document all key claims"],
            outputs=["Claim matrix"],
        ),
        DebatePhase(
            name="validation",
            description="Cross-validate claims",
            duration_rounds=2,
            roles_active=["validator", "critic"],
            objectives=["Assess claim validity"],
            outputs=["Validated claims with confidence"],
        ),
        DebatePhase(
            name="synthesis",
            description="Create integrated synthesis",
            duration_rounds=2,
            roles_active=["synthesizer", "critic"],
            objectives=["Produce coherent synthesis"],
            outputs=["Research synthesis"],
        ),
    ],

    recommended_agents=4,
    max_rounds=5,
    consensus_threshold=0.6,

    rubric={
        "claim_accuracy": 0.25,
        "source_coverage": 0.2,
        "synthesis_quality": 0.25,
        "gap_identification": 0.15,
        "confidence_calibration": 0.15,
    },

    output_format="""
# Research Synthesis: {topic}

## Key Findings
{findings}

## Consensus Claims (High Confidence)
{consensus}

## Contested Claims
{contested}

## Evidence Gaps
{gaps}

## Methodology Notes
{methodology}

## Recommendations
{recommendations}

## Sources
{sources}
""",

    domain="research",
    difficulty=0.7,
    tags=["research", "synthesis", "literature", "analysis"],
)


# ============================================================================
# Template Registry
# ============================================================================

TEMPLATES = {
    TemplateType.CODE_REVIEW: CODE_REVIEW_TEMPLATE,
    TemplateType.DESIGN_DOC: DESIGN_DOC_TEMPLATE,
    TemplateType.INCIDENT_RESPONSE: INCIDENT_RESPONSE_TEMPLATE,
    TemplateType.RESEARCH_SYNTHESIS: RESEARCH_SYNTHESIS_TEMPLATE,
}


def get_template(template_type: TemplateType) -> DebateTemplate:
    """Get a debate template by type."""
    if template_type not in TEMPLATES:
        raise ValueError(f"Unknown template type: {template_type}")
    return TEMPLATES[template_type]


def list_templates() -> list[dict]:
    """List all available templates."""
    return [
        {
            "id": t.template_id,
            "type": t.template_type.value,
            "name": t.name,
            "description": t.description,
            "agents": t.recommended_agents,
            "domain": t.domain,
        }
        for t in TEMPLATES.values()
    ]


def template_to_protocol(
    template: DebateTemplate,
    overrides: Optional[dict] = None,
) -> "DebateProtocol":
    """Convert a DebateTemplate to a DebateProtocol.

    Maps template structure to protocol mechanics:
    - template.max_rounds -> protocol.rounds
    - template.consensus_threshold -> protocol.consensus_threshold
    - template.phases -> total rounds from phase durations
    - Multiple roles -> round-robin topology for structured exchanges

    Note: Template roles (e.g., "author", "security_critic") are domain-specific
    and don't directly map to CognitiveRole. Role rotation uses default cognitive
    roles (analyst, skeptic, lateral_thinker, synthesizer) which complement
    the template structure.

    Args:
        template: The DebateTemplate to convert
        overrides: Optional dict of protocol fields to override

    Returns:
        Configured DebateProtocol matching the template structure
    """
    from aragora.debate.protocol import DebateProtocol

    overrides = overrides or {}

    # Calculate total rounds from phases
    total_rounds = sum(phase.duration_rounds for phase in template.phases)
    rounds = overrides.get("rounds", max(total_rounds, template.max_rounds))

    # Determine topology based on template structure
    # Templates with distinct critic roles benefit from round-robin
    # Research templates benefit from all-to-all for synthesis
    topology = overrides.get("topology")
    if topology is None:
        if template.template_type == TemplateType.RESEARCH_SYNTHESIS:
            topology = "all-to-all"
        elif len(template.roles) >= 4:
            topology = "round-robin"
        else:
            topology = "all-to-all"

    return DebateProtocol(
        rounds=rounds,
        topology=topology,
        consensus=overrides.get("consensus", "majority"),
        consensus_threshold=overrides.get(
            "consensus_threshold", template.consensus_threshold
        ),
        role_rotation=overrides.get("role_rotation", True),
        require_reasoning=True,
        early_stopping=overrides.get("early_stopping", True),
        convergence_detection=overrides.get("convergence_detection", True),
        enable_calibration=overrides.get("enable_calibration", True),
    )


__all__ = [
    "DebateTemplate",
    "DebateRole",
    "DebatePhase",
    "TemplateType",
    "CODE_REVIEW_TEMPLATE",
    "DESIGN_DOC_TEMPLATE",
    "INCIDENT_RESPONSE_TEMPLATE",
    "RESEARCH_SYNTHESIS_TEMPLATE",
    "get_template",
    "list_templates",
    "template_to_protocol",
    "TEMPLATES",
]
