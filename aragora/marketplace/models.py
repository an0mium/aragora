"""
Marketplace data models.

Defines templates for agents, debates, and workflows that can be
shared and reused across the Aragora ecosystem.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import hashlib
import json


class TemplateCategory(Enum):
    """Categories for marketplace templates."""

    ANALYSIS = "analysis"
    CODING = "coding"
    CREATIVE = "creative"
    DEBATE = "debate"
    RESEARCH = "research"
    DECISION = "decision"
    BRAINSTORM = "brainstorm"
    REVIEW = "review"
    PLANNING = "planning"
    CUSTOM = "custom"


@dataclass
class TemplateMetadata:
    """Metadata for a marketplace template."""

    id: str
    name: str
    description: str
    version: str
    author: str
    category: TemplateCategory
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    downloads: int = 0
    stars: int = 0
    license: str = "MIT"
    repository_url: Optional[str] = None
    documentation_url: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "category": self.category.value,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "downloads": self.downloads,
            "stars": self.stars,
            "license": self.license,
            "repository_url": self.repository_url,
            "documentation_url": self.documentation_url,
        }


@dataclass
class TemplateRating:
    """User rating for a template."""

    user_id: str
    template_id: str
    score: int  # 1-5
    review: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if not 1 <= self.score <= 5:
            raise ValueError("Score must be between 1 and 5")


@dataclass
class AgentTemplate:
    """Template for creating an agent configuration."""

    metadata: TemplateMetadata
    agent_type: str  # e.g., "claude", "gpt4", "custom"
    system_prompt: str
    model_config: dict[str, Any] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    examples: list[dict[str, str]] = field(default_factory=list)

    def content_hash(self) -> str:
        """Generate a hash of the template content."""
        content = json.dumps(
            {
                "agent_type": self.agent_type,
                "system_prompt": self.system_prompt,
                "model_config": self.model_config,
                "capabilities": self.capabilities,
                "constraints": self.constraints,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "agent_type": self.agent_type,
            "system_prompt": self.system_prompt,
            "model_config": self.model_config,
            "capabilities": self.capabilities,
            "constraints": self.constraints,
            "examples": self.examples,
            "content_hash": self.content_hash(),
        }


@dataclass
class DebateTemplate:
    """Template for creating a debate configuration."""

    metadata: TemplateMetadata
    task_template: str
    agent_roles: list[dict[str, Any]]
    protocol: dict[str, Any]
    evaluation_criteria: list[str] = field(default_factory=list)
    success_metrics: dict[str, float] = field(default_factory=dict)

    def content_hash(self) -> str:
        """Generate a hash of the template content."""
        content = json.dumps(
            {
                "task_template": self.task_template,
                "agent_roles": self.agent_roles,
                "protocol": self.protocol,
                "evaluation_criteria": self.evaluation_criteria,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "task_template": self.task_template,
            "agent_roles": self.agent_roles,
            "protocol": self.protocol,
            "evaluation_criteria": self.evaluation_criteria,
            "success_metrics": self.success_metrics,
            "content_hash": self.content_hash(),
        }


@dataclass
class WorkflowTemplate:
    """Template for creating a workflow DAG."""

    metadata: TemplateMetadata
    nodes: list[dict[str, Any]]
    edges: list[dict[str, str]]
    inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        """Generate a hash of the template content."""
        content = json.dumps(
            {
                "nodes": self.nodes,
                "edges": self.edges,
                "inputs": self.inputs,
                "outputs": self.outputs,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "nodes": self.nodes,
            "edges": self.edges,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "variables": self.variables,
            "content_hash": self.content_hash(),
        }


# Pre-built templates
BUILTIN_AGENT_TEMPLATES: list[AgentTemplate] = [
    AgentTemplate(
        metadata=TemplateMetadata(
            id="devil-advocate",
            name="Devil's Advocate",
            description="An agent that challenges assumptions and presents counterarguments",
            version="1.0.0",
            author="aragora",
            category=TemplateCategory.DEBATE,
            tags=["debate", "critical-thinking", "counterargument"],
        ),
        agent_type="claude",
        system_prompt="""You are a Devil's Advocate. Your role is to:
1. Challenge every assumption presented
2. Present strong counterarguments
3. Identify logical fallacies
4. Play the skeptic role constructively
5. Push others to strengthen their arguments

Always be respectful but persistent in your challenges.""",
        capabilities=["counterargument", "critical_analysis", "fallacy_detection"],
        constraints=["must_provide_reasoning", "no_personal_attacks"],
    ),
    AgentTemplate(
        metadata=TemplateMetadata(
            id="code-reviewer",
            name="Code Reviewer",
            description="An agent specialized in reviewing code for quality and best practices",
            version="1.0.0",
            author="aragora",
            category=TemplateCategory.CODING,
            tags=["code-review", "best-practices", "security"],
        ),
        agent_type="claude",
        system_prompt="""You are an expert Code Reviewer. Your role is to:
1. Review code for correctness and bugs
2. Check for security vulnerabilities
3. Evaluate code style and maintainability
4. Suggest performance improvements
5. Ensure best practices are followed

Provide specific, actionable feedback with examples.""",
        capabilities=["code_analysis", "security_review", "performance_analysis"],
        constraints=["must_cite_line_numbers", "provide_examples"],
    ),
    AgentTemplate(
        metadata=TemplateMetadata(
            id="research-analyst",
            name="Research Analyst",
            description="An agent that conducts thorough research and synthesizes information",
            version="1.0.0",
            author="aragora",
            category=TemplateCategory.RESEARCH,
            tags=["research", "analysis", "synthesis"],
        ),
        agent_type="claude",
        system_prompt="""You are a Research Analyst. Your role is to:
1. Gather and synthesize information
2. Identify key trends and patterns
3. Evaluate source credibility
4. Present balanced perspectives
5. Draw evidence-based conclusions

Always cite your sources and acknowledge uncertainty.""",
        capabilities=["research", "synthesis", "trend_analysis"],
        constraints=["must_cite_sources", "acknowledge_uncertainty"],
    ),
]


BUILTIN_DEBATE_TEMPLATES: list[DebateTemplate] = [
    DebateTemplate(
        metadata=TemplateMetadata(
            id="oxford-style",
            name="Oxford-Style Debate",
            description="Formal debate format with proposing and opposing teams",
            version="1.0.0",
            author="aragora",
            category=TemplateCategory.DEBATE,
            tags=["formal", "competitive", "structured"],
        ),
        task_template="Resolved: {motion}",
        agent_roles=[
            {"role": "proposition_lead", "team": "proposition", "speaks": 1},
            {"role": "opposition_lead", "team": "opposition", "speaks": 2},
            {"role": "proposition_second", "team": "proposition", "speaks": 3},
            {"role": "opposition_second", "team": "opposition", "speaks": 4},
        ],
        protocol={
            "rounds": 4,
            "speaking_time": 300,
            "consensus_mode": "vote",
            "allow_rebuttals": True,
        },
        evaluation_criteria=["logic", "evidence", "rhetoric", "rebuttals"],
    ),
    DebateTemplate(
        metadata=TemplateMetadata(
            id="brainstorm-session",
            name="Brainstorm Session",
            description="Collaborative ideation format for generating creative solutions",
            version="1.0.0",
            author="aragora",
            category=TemplateCategory.BRAINSTORM,
            tags=["creative", "collaborative", "ideation"],
        ),
        task_template="Generate solutions for: {problem}",
        agent_roles=[
            {"role": "facilitator", "guides": True},
            {"role": "ideator", "count": 3},
            {"role": "synthesizer", "summarizes": True},
        ],
        protocol={
            "rounds": 3,
            "consensus_mode": "synthesis",
            "allow_building": True,
            "no_criticism_phase": True,
        },
        evaluation_criteria=["creativity", "feasibility", "novelty"],
    ),
    DebateTemplate(
        metadata=TemplateMetadata(
            id="code-review-session",
            name="Code Review Session",
            description="Multi-agent code review with different perspectives",
            version="1.0.0",
            author="aragora",
            category=TemplateCategory.REVIEW,
            tags=["code", "review", "quality"],
        ),
        task_template="Review this code:\n```\n{code}\n```",
        agent_roles=[
            {"role": "security_reviewer", "focus": "security"},
            {"role": "performance_reviewer", "focus": "performance"},
            {"role": "maintainability_reviewer", "focus": "maintainability"},
            {"role": "synthesizer", "aggregates": True},
        ],
        protocol={
            "rounds": 2,
            "consensus_mode": "synthesis",
            "require_specific_feedback": True,
        },
        evaluation_criteria=["thoroughness", "specificity", "actionability"],
    ),
]
