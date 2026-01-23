"""
Base classes for deliberation templates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TeamStrategy(Enum):
    """Strategy for selecting the agent team."""

    SPECIFIED = "specified"  # Use explicitly provided agents
    BEST_FOR_DOMAIN = "best_for_domain"  # Auto-select based on domain
    DIVERSE = "diverse"  # Maximize agent diversity
    FAST = "fast"  # Optimize for speed
    RANDOM = "random"  # Random selection


class OutputFormat(Enum):
    """Format for deliberation output."""

    STANDARD = "standard"  # Default JSON response
    DECISION_RECEIPT = "decision_receipt"  # Full audit receipt
    SUMMARY = "summary"  # Condensed summary
    GITHUB_REVIEW = "github_review"  # GitHub PR review format
    SLACK_MESSAGE = "slack_message"  # Slack-formatted message
    JIRA_COMMENT = "jira_comment"  # Jira comment format
    CONFLUENCE_PAGE = "confluence_page"  # Confluence page format
    EMAIL = "email"  # Email format
    COMPLIANCE_REPORT = "compliance_report"  # Compliance audit report


class TemplateCategory(Enum):
    """Category for organizing templates."""

    CODE = "code"
    LEGAL = "legal"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    COMPLIANCE = "compliance"
    ACADEMIC = "academic"
    GENERAL = "general"


@dataclass
class DeliberationTemplate:
    """
    Pre-built deliberation pattern for common use cases.

    Templates provide sensible defaults for:
    - Agent selection strategy and default agents
    - Knowledge sources to query
    - Output format and delivery channels
    - Consensus requirements
    - Persona configurations for specialized analysis

    Example usage:
        template = get_template("code_review")
        request = OrchestrationRequest(
            question="Review this PR",
            template="code_review",
            knowledge_sources=["github:owner/repo/pr/123"],
        )
    """

    name: str
    description: str
    category: TemplateCategory = TemplateCategory.GENERAL

    # Agent configuration
    default_agents: List[str] = field(default_factory=list)
    team_strategy: TeamStrategy = TeamStrategy.BEST_FOR_DOMAIN
    min_agents: int = 2
    max_agents: int = 5

    # Knowledge configuration
    default_knowledge_sources: List[str] = field(default_factory=list)
    required_knowledge_types: List[str] = field(default_factory=list)

    # Output configuration
    output_format: OutputFormat = OutputFormat.STANDARD
    default_output_channels: List[str] = field(default_factory=list)

    # Deliberation configuration
    consensus_threshold: float = 0.7
    max_rounds: int = 5
    require_consensus: bool = True
    timeout_seconds: float = 300.0

    # Persona configuration for specialized analysis
    personas: List[str] = field(default_factory=list)

    # Tags for filtering and discovery
    tags: List[str] = field(default_factory=list)

    # Version for template evolution
    version: str = "1.0.0"

    # Optional system prompt additions
    system_prompt_additions: Optional[str] = None

    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "default_agents": self.default_agents,
            "team_strategy": self.team_strategy.value,
            "min_agents": self.min_agents,
            "max_agents": self.max_agents,
            "default_knowledge_sources": self.default_knowledge_sources,
            "required_knowledge_types": self.required_knowledge_types,
            "output_format": self.output_format.value,
            "default_output_channels": self.default_output_channels,
            "consensus_threshold": self.consensus_threshold,
            "max_rounds": self.max_rounds,
            "require_consensus": self.require_consensus,
            "timeout_seconds": self.timeout_seconds,
            "personas": self.personas,
            "tags": self.tags,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeliberationTemplate":
        """Create from dictionary (e.g., YAML loaded data)."""
        # Parse enums
        category = TemplateCategory.GENERAL
        if "category" in data:
            try:
                category = TemplateCategory(data["category"])
            except ValueError:
                pass

        team_strategy = TeamStrategy.BEST_FOR_DOMAIN
        if "team_strategy" in data:
            try:
                team_strategy = TeamStrategy(data["team_strategy"])
            except ValueError:
                pass

        output_format = OutputFormat.STANDARD
        if "output_format" in data:
            try:
                output_format = OutputFormat(data["output_format"])
            except ValueError:
                pass

        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            category=category,
            default_agents=data.get("default_agents", []),
            team_strategy=team_strategy,
            min_agents=data.get("min_agents", 2),
            max_agents=data.get("max_agents", 5),
            default_knowledge_sources=data.get("default_knowledge_sources", []),
            required_knowledge_types=data.get("required_knowledge_types", []),
            output_format=output_format,
            default_output_channels=data.get("default_output_channels", []),
            consensus_threshold=data.get("consensus_threshold", 0.7),
            max_rounds=data.get("max_rounds", 5),
            require_consensus=data.get("require_consensus", True),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            personas=data.get("personas", []),
            tags=data.get("tags", []),
            version=data.get("version", "1.0.0"),
            system_prompt_additions=data.get("system_prompt_additions"),
            metadata=data.get("metadata", {}),
        )

    def merge_with_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge template defaults with request data.

        Request values override template defaults.
        """
        merged = {
            "agents": request_data.get("agents") or self.default_agents,
            "team_strategy": request_data.get("team_strategy", self.team_strategy.value),
            "knowledge_sources": request_data.get("knowledge_sources")
            or self.default_knowledge_sources,
            "output_format": request_data.get("output_format", self.output_format.value),
            "output_channels": request_data.get("output_channels") or self.default_output_channels,
            "consensus_threshold": request_data.get(
                "consensus_threshold", self.consensus_threshold
            ),
            "max_rounds": request_data.get("max_rounds", self.max_rounds),
            "require_consensus": request_data.get("require_consensus", self.require_consensus),
            "timeout_seconds": request_data.get("timeout_seconds", self.timeout_seconds),
        }

        # Merge other request data
        for key, value in request_data.items():
            if key not in merged:
                merged[key] = value

        return merged
