"""User Type Presets.

Pre-configured pipeline settings for different user personas.
Each preset adjusts interrogation depth, autonomy level,
output format, and domain focus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserPreset:
    """Pipeline configuration preset for a user type."""

    name: str
    label: str  # Human-readable display name
    description: str

    # Interrogation
    interrogation_depth: str = "standard"  # "minimal", "standard", "deep"
    max_questions: int = 5
    skip_obvious_questions: bool = True
    explain_questions: bool = True

    # Autonomy
    autonomy_level: str = "propose_and_approve"
    auto_execute_low_risk: bool = False

    # Output
    output_format: str = "structured"  # "minimal", "structured", "detailed"
    include_rationale: bool = True
    include_alternatives: bool = False
    include_risk_analysis: bool = False

    # Domain
    default_domains: list[str] = field(default_factory=list)
    agent_count: int = 5

    # Debate
    debate_rounds: int = 3
    consensus_threshold: float = 0.6

    def to_pipeline_config(self) -> dict[str, Any]:
        """Convert preset to pipeline configuration dict."""
        return {
            "interrogation": {
                "depth": self.interrogation_depth,
                "max_questions": self.max_questions,
                "skip_obvious": self.skip_obvious_questions,
                "explain": self.explain_questions,
            },
            "autonomy": {
                "level": self.autonomy_level,
                "auto_execute_low_risk": self.auto_execute_low_risk,
            },
            "output": {
                "format": self.output_format,
                "rationale": self.include_rationale,
                "alternatives": self.include_alternatives,
                "risk_analysis": self.include_risk_analysis,
            },
            "debate": {
                "agent_count": self.agent_count,
                "rounds": self.debate_rounds,
                "consensus_threshold": self.consensus_threshold,
            },
            "domains": self.default_domains,
        }


# Pre-defined presets
FOUNDER_PRESET = UserPreset(
    name="founder",
    label="Founder / Solo Builder",
    description="Fast decisions, minimal overhead. Get from idea to execution quickly.",
    interrogation_depth="minimal",
    max_questions=3,
    skip_obvious_questions=True,
    explain_questions=False,
    autonomy_level="metrics_driven",
    auto_execute_low_risk=True,
    output_format="minimal",
    include_rationale=False,
    include_alternatives=False,
    agent_count=3,
    debate_rounds=2,
    consensus_threshold=0.5,
    default_domains=["technical", "product"],
)

CTO_PRESET = UserPreset(
    name="cto",
    label="CTO / Technical Lead",
    description="Balanced depth. See trade-offs, alternatives, and technical risks.",
    interrogation_depth="standard",
    max_questions=5,
    skip_obvious_questions=True,
    explain_questions=True,
    autonomy_level="propose_and_approve",
    output_format="structured",
    include_rationale=True,
    include_alternatives=True,
    include_risk_analysis=True,
    agent_count=5,
    debate_rounds=3,
    consensus_threshold=0.6,
    default_domains=["technical", "architecture", "security"],
)

TEAM_PRESET = UserPreset(
    name="team",
    label="Team / Collaborative",
    description="Full transparency. Every decision documented for team review.",
    interrogation_depth="deep",
    max_questions=8,
    skip_obvious_questions=False,
    explain_questions=True,
    autonomy_level="human_guided",
    output_format="detailed",
    include_rationale=True,
    include_alternatives=True,
    include_risk_analysis=True,
    agent_count=5,
    debate_rounds=3,
    consensus_threshold=0.7,
    default_domains=["technical", "product", "compliance"],
)

NON_TECHNICAL_PRESET = UserPreset(
    name="non_technical",
    label="Non-Technical Decision Maker",
    description="Clear explanations, no jargon. Focus on business impact.",
    interrogation_depth="standard",
    max_questions=4,
    skip_obvious_questions=True,
    explain_questions=True,
    autonomy_level="propose_and_approve",
    output_format="structured",
    include_rationale=True,
    include_alternatives=True,
    include_risk_analysis=False,
    agent_count=5,
    debate_rounds=2,
    consensus_threshold=0.6,
    default_domains=["business", "product", "strategy"],
)

# Registry
PRESETS: dict[str, UserPreset] = {
    "founder": FOUNDER_PRESET,
    "cto": CTO_PRESET,
    "team": TEAM_PRESET,
    "non_technical": NON_TECHNICAL_PRESET,
}


def get_preset(name: str) -> UserPreset:
    """Get a preset by name.

    Raises:
        ValueError: If preset name is unknown.
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {', '.join(PRESETS.keys())}")
    return PRESETS[name]


def list_presets() -> list[UserPreset]:
    """List all available presets."""
    return list(PRESETS.values())


def create_custom_preset(name: str, base: str = "cto", **overrides: Any) -> UserPreset:
    """Create a custom preset based on an existing one."""
    base_preset = get_preset(base)
    config = {k: v for k, v in base_preset.__dict__.items() if not k.startswith("_")}
    config["name"] = name
    config["label"] = f"Custom: {name}"
    config.update(overrides)
    return UserPreset(**config)
