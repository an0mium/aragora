"""
Pre-built Gauntlet templates for common validation scenarios.

Templates provide ready-to-use configurations for different types of
adversarial validation, optimized for specific domains and use cases.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from .config import (
    GauntletConfig,
    PassFailCriteria,
    AttackCategory,
)


class GauntletTemplate(Enum):
    """Pre-built Gauntlet template identifiers."""
    # Core templates
    API_ROBUSTNESS = "api_robustness"
    DECISION_QUALITY = "decision_quality"
    COMPLIANCE_AUDIT = "compliance_audit"
    SECURITY_ASSESSMENT = "security_assessment"
    ARCHITECTURE_REVIEW = "architecture_review"

    # Specialized templates
    PROMPT_INJECTION = "prompt_injection"
    FINANCIAL_RISK = "financial_risk"
    GDPR_COMPLIANCE = "gdpr_compliance"
    AI_ACT_COMPLIANCE = "ai_act_compliance"
    CODE_REVIEW = "code_review"

    # Quick templates
    QUICK_SANITY = "quick_sanity"
    COMPREHENSIVE = "comprehensive"


# Template definitions
_TEMPLATES: dict[GauntletTemplate, GauntletConfig] = {
    GauntletTemplate.API_ROBUSTNESS: GauntletConfig(
        name="API Robustness Gauntlet",
        description="Stress-test API designs for edge cases, scalability, and failure modes",
        template_id="api_robustness",
        input_type="architecture",
        domain="api",
        agents=["claude", "gpt4", "gemini"],
        enable_scenario_analysis=True,
        enable_adversarial_probing=True,
        enable_formal_verification=True,
        enable_deep_audit=False,
        scenario_presets=["scale", "stakeholder"],
        attack_categories=[
            AttackCategory.EDGE_CASE,
            AttackCategory.SCALABILITY,
            AttackCategory.RESOURCE_EXHAUSTION,
            AttackCategory.RACE_CONDITION,
            AttackCategory.ADVERSARIAL_INPUT,
            AttackCategory.DEPENDENCY_FAILURE,
        ],
        probes_per_category=3,
        max_total_probes=15,
        timeout_seconds=180,
        criteria=PassFailCriteria(
            max_critical_findings=0,
            max_high_findings=2,
            min_robustness_score=0.75,
            require_formal_verification=False,
            require_consensus=True,
            min_confidence=0.7,
        ),
        tags=["api", "robustness", "scalability"],
    ),

    GauntletTemplate.DECISION_QUALITY: GauntletConfig(
        name="Decision Quality Gauntlet",
        description="Validate strategic decisions by attacking assumptions and exploring alternatives",
        template_id="decision_quality",
        input_type="text",
        domain="strategy",
        agents=["claude", "gpt4", "gemini", "deepseek"],
        enable_scenario_analysis=True,
        enable_adversarial_probing=True,
        enable_formal_verification=False,
        enable_deep_audit=True,
        scenario_presets=["comprehensive"],
        attack_categories=[
            AttackCategory.LOGICAL_FALLACY,
            AttackCategory.UNSTATED_ASSUMPTION,
            AttackCategory.COUNTEREXAMPLE,
            AttackCategory.STAKEHOLDER_CONFLICT,
        ],
        probes_per_category=4,
        max_total_probes=16,
        timeout_seconds=300,
        criteria=PassFailCriteria(
            max_critical_findings=0,
            max_high_findings=3,
            min_robustness_score=0.65,
            require_formal_verification=False,
            require_consensus=True,
            min_confidence=0.6,
        ),
        tags=["strategy", "decision", "assumptions"],
    ),

    GauntletTemplate.COMPLIANCE_AUDIT: GauntletConfig(
        name="Compliance Audit Gauntlet",
        description="Audit decisions and systems for regulatory compliance violations",
        template_id="compliance_audit",
        input_type="policy",
        domain="compliance",
        agents=["claude", "gpt4", "mistral"],
        enable_scenario_analysis=True,
        enable_adversarial_probing=True,
        enable_formal_verification=True,
        enable_deep_audit=True,
        scenario_presets=["regulatory"],
        attack_categories=[
            AttackCategory.REGULATORY_VIOLATION,
            AttackCategory.EDGE_CASE,
            AttackCategory.UNSTATED_ASSUMPTION,
        ],
        probes_per_category=5,
        max_total_probes=15,
        timeout_seconds=360,
        criteria=PassFailCriteria.strict(),
        tags=["compliance", "regulatory", "audit"],
    ),

    GauntletTemplate.SECURITY_ASSESSMENT: GauntletConfig(
        name="Security Assessment Gauntlet",
        description="Red-team security analysis with adversarial attack simulation",
        template_id="security_assessment",
        input_type="code",
        domain="security",
        agents=["claude", "gpt4", "deepseek"],
        enable_scenario_analysis=False,
        enable_adversarial_probing=True,
        enable_formal_verification=True,
        enable_deep_audit=False,
        scenario_presets=[],
        attack_categories=[
            AttackCategory.SECURITY,
            AttackCategory.ADVERSARIAL_INPUT,
            AttackCategory.EDGE_CASE,
            AttackCategory.RACE_CONDITION,
            AttackCategory.RESOURCE_EXHAUSTION,
        ],
        probes_per_category=4,
        max_total_probes=20,
        timeout_seconds=240,
        criteria=PassFailCriteria(
            max_critical_findings=0,
            max_high_findings=0,
            min_robustness_score=0.85,
            require_formal_verification=True,
            require_consensus=True,
            min_confidence=0.8,
        ),
        tags=["security", "red-team", "vulnerability"],
    ),

    GauntletTemplate.ARCHITECTURE_REVIEW: GauntletConfig(
        name="Architecture Review Gauntlet",
        description="Comprehensive review of system architecture for scalability and resilience",
        template_id="architecture_review",
        input_type="architecture",
        domain="architecture",
        agents=["claude", "gpt4", "gemini", "deepseek"],
        enable_scenario_analysis=True,
        enable_adversarial_probing=True,
        enable_formal_verification=True,
        enable_deep_audit=True,
        scenario_presets=["scale", "time_horizon", "tech_stack"],
        attack_categories=[
            AttackCategory.SCALABILITY,
            AttackCategory.DEPENDENCY_FAILURE,
            AttackCategory.RACE_CONDITION,
            AttackCategory.RESOURCE_EXHAUSTION,
            AttackCategory.EDGE_CASE,
        ],
        probes_per_category=3,
        max_total_probes=15,
        timeout_seconds=360,
        criteria=PassFailCriteria(
            max_critical_findings=0,
            max_high_findings=2,
            min_robustness_score=0.7,
            require_formal_verification=False,
            require_consensus=True,
            min_confidence=0.65,
        ),
        tags=["architecture", "scalability", "resilience"],
    ),

    GauntletTemplate.PROMPT_INJECTION: GauntletConfig(
        name="Prompt Injection Gauntlet",
        description="Test LLM prompts and systems for injection vulnerabilities",
        template_id="prompt_injection",
        input_type="prompt",
        domain="security",
        agents=["claude", "gpt4"],
        enable_scenario_analysis=False,
        enable_adversarial_probing=True,
        enable_formal_verification=False,
        enable_deep_audit=False,
        scenario_presets=[],
        attack_categories=[
            AttackCategory.ADVERSARIAL_INPUT,
            AttackCategory.SECURITY,
        ],
        probes_per_category=10,
        max_total_probes=20,
        timeout_seconds=120,
        criteria=PassFailCriteria(
            max_critical_findings=0,
            max_high_findings=0,
            min_robustness_score=0.9,
            require_formal_verification=False,
            require_consensus=False,
            min_confidence=0.7,
        ),
        tags=["prompt", "injection", "llm-security"],
    ),

    GauntletTemplate.FINANCIAL_RISK: GauntletConfig(
        name="Financial Risk Gauntlet",
        description="Assess financial decisions and models for risk exposure",
        template_id="financial_risk",
        input_type="text",
        domain="finance",
        agents=["claude", "gpt4", "deepseek"],
        enable_scenario_analysis=True,
        enable_adversarial_probing=True,
        enable_formal_verification=True,
        enable_deep_audit=True,
        scenario_presets=["risk", "scale", "time_horizon"],
        attack_categories=[
            AttackCategory.EDGE_CASE,
            AttackCategory.UNSTATED_ASSUMPTION,
            AttackCategory.COUNTEREXAMPLE,
            AttackCategory.SCALABILITY,
        ],
        probes_per_category=4,
        max_total_probes=16,
        timeout_seconds=300,
        criteria=PassFailCriteria.strict(),
        tags=["finance", "risk", "quantitative"],
    ),

    GauntletTemplate.GDPR_COMPLIANCE: GauntletConfig(
        name="GDPR Compliance Gauntlet",
        description="Validate data processing for GDPR compliance requirements",
        template_id="gdpr_compliance",
        input_type="policy",
        domain="compliance",
        agents=["claude", "mistral", "gpt4"],
        enable_scenario_analysis=True,
        enable_adversarial_probing=True,
        enable_formal_verification=True,
        enable_deep_audit=True,
        scenario_presets=["regulatory"],
        attack_categories=[
            AttackCategory.REGULATORY_VIOLATION,
            AttackCategory.EDGE_CASE,
            AttackCategory.STAKEHOLDER_CONFLICT,
        ],
        probes_per_category=5,
        max_total_probes=15,
        timeout_seconds=300,
        criteria=PassFailCriteria.strict(),
        tags=["gdpr", "privacy", "eu", "compliance"],
    ),

    GauntletTemplate.AI_ACT_COMPLIANCE: GauntletConfig(
        name="EU AI Act Compliance Gauntlet",
        description="Assess AI systems against EU AI Act requirements",
        template_id="ai_act_compliance",
        input_type="architecture",
        domain="compliance",
        agents=["claude", "mistral", "gpt4"],
        enable_scenario_analysis=True,
        enable_adversarial_probing=True,
        enable_formal_verification=True,
        enable_deep_audit=True,
        scenario_presets=["regulatory", "risk"],
        attack_categories=[
            AttackCategory.REGULATORY_VIOLATION,
            AttackCategory.EDGE_CASE,
            AttackCategory.UNSTATED_ASSUMPTION,
            AttackCategory.STAKEHOLDER_CONFLICT,
        ],
        probes_per_category=4,
        max_total_probes=16,
        timeout_seconds=360,
        criteria=PassFailCriteria.strict(),
        tags=["ai-act", "eu", "compliance", "ai-governance"],
    ),

    GauntletTemplate.CODE_REVIEW: GauntletConfig(
        name="Code Review Gauntlet",
        description="Adversarial code review for bugs, security issues, and design flaws",
        template_id="code_review",
        input_type="code",
        domain="code",
        agents=["claude", "gpt4", "deepseek"],
        enable_scenario_analysis=False,
        enable_adversarial_probing=True,
        enable_formal_verification=True,
        enable_deep_audit=False,
        scenario_presets=[],
        attack_categories=[
            AttackCategory.SECURITY,
            AttackCategory.EDGE_CASE,
            AttackCategory.RACE_CONDITION,
            AttackCategory.RESOURCE_EXHAUSTION,
            AttackCategory.LOGICAL_FALLACY,
        ],
        probes_per_category=3,
        max_total_probes=15,
        timeout_seconds=180,
        criteria=PassFailCriteria(
            max_critical_findings=0,
            max_high_findings=2,
            min_robustness_score=0.7,
            require_formal_verification=False,
            require_consensus=True,
            min_confidence=0.65,
        ),
        tags=["code", "review", "bugs", "security"],
    ),

    GauntletTemplate.QUICK_SANITY: GauntletConfig(
        name="Quick Sanity Check",
        description="Fast validation for obvious issues and red flags",
        template_id="quick_sanity",
        input_type="text",
        domain="general",
        agents=["claude", "gpt4"],
        enable_scenario_analysis=False,
        enable_adversarial_probing=True,
        enable_formal_verification=False,
        enable_deep_audit=False,
        scenario_presets=[],
        attack_categories=[
            AttackCategory.LOGICAL_FALLACY,
            AttackCategory.UNSTATED_ASSUMPTION,
            AttackCategory.COUNTEREXAMPLE,
        ],
        probes_per_category=2,
        max_total_probes=6,
        timeout_seconds=60,
        criteria=PassFailCriteria.lenient(),
        tags=["quick", "sanity", "basic"],
    ),

    GauntletTemplate.COMPREHENSIVE: GauntletConfig(
        name="Comprehensive Gauntlet",
        description="Full adversarial validation with all attack vectors enabled",
        template_id="comprehensive",
        input_type="text",
        domain="general",
        agents=["claude", "gpt4", "gemini", "deepseek", "mistral"],
        enable_scenario_analysis=True,
        enable_adversarial_probing=True,
        enable_formal_verification=True,
        enable_deep_audit=True,
        scenario_presets=["comprehensive"],
        attack_categories=list(AttackCategory),
        probes_per_category=3,
        max_total_probes=30,
        timeout_seconds=600,
        criteria=PassFailCriteria(
            max_critical_findings=0,
            max_high_findings=3,
            min_robustness_score=0.7,
            require_formal_verification=True,
            require_consensus=True,
            min_confidence=0.7,
        ),
        tags=["comprehensive", "full", "all-attacks"],
    ),
}


def get_template(template: GauntletTemplate | str) -> GauntletConfig:
    """Get a Gauntlet configuration from a template.

    Args:
        template: Template enum or string identifier

    Returns:
        GauntletConfig configured for the template

    Raises:
        ValueError: If template not found
    """
    if isinstance(template, str):
        try:
            template = GauntletTemplate(template)
        except ValueError:
            raise ValueError(
                f"Unknown template: {template}. "
                f"Available: {[t.value for t in GauntletTemplate]}"
            )

    if template not in _TEMPLATES:
        raise ValueError(f"Template {template.value} not configured")

    # Return a copy to avoid mutation
    config = _TEMPLATES[template]
    return GauntletConfig.from_dict(config.to_dict())


def list_templates() -> list[dict]:
    """List all available Gauntlet templates.

    Returns:
        List of template summaries with id, name, description, tags
    """
    return [
        {
            "id": template.value,
            "name": config.name,
            "description": config.description,
            "domain": config.domain,
            "input_type": config.input_type,
            "tags": config.tags,
            "estimated_duration_seconds": config.timeout_seconds,
            "criteria_level": (
                "strict" if config.criteria.max_critical_findings == 0 and
                config.criteria.max_high_findings == 0 else
                "standard" if config.criteria.max_high_findings <= 3 else
                "lenient"
            ),
        }
        for template, config in _TEMPLATES.items()
    ]


def get_template_by_domain(domain: str) -> list[GauntletTemplate]:
    """Get templates matching a specific domain.

    Args:
        domain: Domain to filter by (api, security, compliance, etc.)

    Returns:
        List of matching template enums
    """
    return [
        template
        for template, config in _TEMPLATES.items()
        if config.domain == domain
    ]


def get_template_by_tags(tags: list[str]) -> list[GauntletTemplate]:
    """Get templates matching any of the given tags.

    Args:
        tags: Tags to match

    Returns:
        List of matching template enums
    """
    tag_set = set(tags)
    return [
        template
        for template, config in _TEMPLATES.items()
        if tag_set & set(config.tags)
    ]
