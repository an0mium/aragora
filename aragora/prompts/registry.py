"""Prompt registry — version-controlled prompt templates with A/B metrics.

Provides a centralized registry for all prompt templates used across the
pipeline, debate engine, and agent interactions. Each template is versioned,
tracks performance metrics, and supports A/B testing.

Usage::

    registry = PromptRegistry()

    # Register a template
    registry.register(
        PromptTemplate(
            id="goal_extraction_v1",
            domain="pipeline",
            stage="ideas_to_goals",
            template="Extract goals from: {ideas}",
            version=1,
        )
    )

    # Get best template for a domain
    template = registry.get_best("pipeline", "ideas_to_goals")
    prompt = template.render(ideas="improve UX, add caching")

    # Record outcome for optimization
    registry.record_outcome(template.id, success=True, score=0.85)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PromptMetrics:
    """Performance metrics for a prompt template."""

    uses: int = 0
    successes: int = 0
    failures: int = 0
    total_score: float = 0.0
    scores: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.uses == 0:
            return 0.0
        return self.successes / self.uses

    @property
    def avg_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uses": self.uses,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": self.success_rate,
            "avg_score": self.avg_score,
        }


@dataclass
class PromptTemplate:
    """A versioned prompt template with performance tracking."""

    id: str
    domain: str  # pipeline | debate | agent | verification
    stage: str  # e.g., ideas_to_goals, goal_extraction, critique
    template: str
    version: int = 1
    description: str = ""
    active: bool = True
    created_at: str = ""
    variables: list[str] = field(default_factory=list)
    metrics: PromptMetrics = field(default_factory=PromptMetrics)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.variables:
            # Auto-detect variables from template
            import re

            self.variables = re.findall(r"\{(\w+)\}", self.template)

    def render(self, **kwargs: Any) -> str:
        """Render the template with the given variables.

        Missing variables are left as placeholders.
        """
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of the template content."""
        return hashlib.sha256(self.template.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "stage": self.stage,
            "version": self.version,
            "description": self.description,
            "active": self.active,
            "content_hash": self.content_hash,
            "variables": self.variables,
            "metrics": self.metrics.to_dict(),
        }


class PromptRegistry:
    """Registry for version-controlled prompt templates.

    Supports:
    - Template registration and versioning
    - A/B testing with performance-based selection
    - Outcome tracking for optimization
    - Domain and stage filtering

    The registry uses a Thompson Sampling-inspired selection strategy:
    templates with better performance metrics are selected more often,
    but new or under-tested templates still get sampled.
    """

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        self._domain_index: dict[str, dict[str, list[str]]] = {}

    def register(self, template: PromptTemplate) -> None:
        """Register a prompt template.

        If a template with the same ID exists, it's replaced.
        Automatically indexes by domain and stage.
        """
        self._templates[template.id] = template

        # Update domain index
        domain_stages = self._domain_index.setdefault(template.domain, {})
        stage_ids = domain_stages.setdefault(template.stage, [])
        if template.id not in stage_ids:
            stage_ids.append(template.id)

        logger.debug(
            "Registered prompt template: %s (domain=%s, stage=%s, v%d)",
            template.id,
            template.domain,
            template.stage,
            template.version,
        )

    def get(self, template_id: str) -> PromptTemplate | None:
        """Get a template by ID."""
        return self._templates.get(template_id)

    def get_best(self, domain: str, stage: str) -> PromptTemplate | None:
        """Get the best-performing active template for a domain/stage.

        Selection strategy:
        1. Filter to active templates for the domain/stage
        2. If only one template, return it
        3. If templates have metrics, return highest avg_score
        4. If no metrics, return the latest version
        """
        candidates = self._get_candidates(domain, stage)
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Sort by avg_score (desc), then by version (desc)
        candidates.sort(
            key=lambda t: (t.metrics.avg_score, t.version),
            reverse=True,
        )

        # If top candidate has enough data, use it
        if candidates[0].metrics.uses >= 5:
            return candidates[0]

        # Otherwise explore: return least-used template
        candidates.sort(key=lambda t: t.metrics.uses)
        return candidates[0]

    def get_for_ab_test(
        self, domain: str, stage: str
    ) -> tuple[PromptTemplate, PromptTemplate] | None:
        """Get two templates for A/B testing.

        Returns the current best and the challenger (next best or newest).
        Returns None if fewer than 2 templates are available.
        """
        candidates = self._get_candidates(domain, stage)
        if len(candidates) < 2:
            return None

        # Sort by avg_score desc
        candidates.sort(
            key=lambda t: (t.metrics.avg_score, t.version),
            reverse=True,
        )

        return candidates[0], candidates[1]

    def record_outcome(
        self,
        template_id: str,
        success: bool,
        score: float = 0.0,
    ) -> None:
        """Record the outcome of using a template.

        Args:
            template_id: Template that was used.
            success: Whether the outcome was successful.
            score: Quality score (0.0-1.0) for the output.
        """
        template = self._templates.get(template_id)
        if not template:
            logger.warning("Cannot record outcome for unknown template: %s", template_id)
            return

        template.metrics.uses += 1
        if success:
            template.metrics.successes += 1
        else:
            template.metrics.failures += 1

        if score > 0:
            template.metrics.total_score += score
            # Keep last 100 scores for windowed average
            template.metrics.scores.append(score)
            if len(template.metrics.scores) > 100:
                template.metrics.scores = template.metrics.scores[-100:]

        logger.debug(
            "Recorded outcome for %s: success=%s score=%.2f (avg=%.2f over %d uses)",
            template_id,
            success,
            score,
            template.metrics.avg_score,
            template.metrics.uses,
        )

    def list_templates(
        self,
        domain: str | None = None,
        stage: str | None = None,
        active_only: bool = True,
    ) -> list[PromptTemplate]:
        """List templates with optional filtering."""
        templates = list(self._templates.values())

        if domain:
            templates = [t for t in templates if t.domain == domain]
        if stage:
            templates = [t for t in templates if t.stage == stage]
        if active_only:
            templates = [t for t in templates if t.active]

        return sorted(templates, key=lambda t: (t.domain, t.stage, -t.version))

    def deactivate(self, template_id: str) -> bool:
        """Deactivate a template (soft delete)."""
        template = self._templates.get(template_id)
        if template:
            template.active = False
            return True
        return False

    def get_metrics_report(self) -> dict[str, Any]:
        """Get a summary report of all template metrics."""
        report: dict[str, Any] = {
            "total_templates": len(self._templates),
            "active_templates": sum(1 for t in self._templates.values() if t.active),
            "domains": {},
        }

        for domain, stages in self._domain_index.items():
            domain_report: dict[str, Any] = {"stages": {}}
            for stage, ids in stages.items():
                templates = [self._templates[tid] for tid in ids if tid in self._templates]
                domain_report["stages"][stage] = {
                    "templates": len(templates),
                    "total_uses": sum(t.metrics.uses for t in templates),
                    "best_score": max(
                        (t.metrics.avg_score for t in templates),
                        default=0.0,
                    ),
                }
            report["domains"][domain] = domain_report

        return report

    def _get_candidates(self, domain: str, stage: str) -> list[PromptTemplate]:
        """Get active template candidates for a domain/stage."""
        ids = self._domain_index.get(domain, {}).get(stage, [])
        return [
            self._templates[tid]
            for tid in ids
            if tid in self._templates and self._templates[tid].active
        ]


# -- Singleton instance --------------------------------------------------

_global_registry: PromptRegistry | None = None


def get_prompt_registry() -> PromptRegistry:
    """Get the global prompt registry singleton."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PromptRegistry()
        _register_builtin_templates(_global_registry)
    return _global_registry


def _register_builtin_templates(registry: PromptRegistry) -> None:
    """Register built-in prompt templates."""
    # Pipeline stage transitions
    registry.register(
        PromptTemplate(
            id="pipeline_ideas_to_goals_v1",
            domain="pipeline",
            stage="ideas_to_goals",
            template=(
                "Given these ideas:\n{ideas}\n\n"
                "Extract SMART goals that capture the intent.\n"
                "For each goal, specify:\n"
                "- Title (concise, actionable)\n"
                "- Priority (high/medium/low)\n"
                "- Success criteria (measurable)\n"
                "- Dependencies (other goals that must complete first)"
            ),
            version=1,
            description="Standard goal extraction from ideas",
        )
    )

    registry.register(
        PromptTemplate(
            id="pipeline_goals_to_actions_v1",
            domain="pipeline",
            stage="goals_to_actions",
            template=(
                "Given these goals:\n{goals}\n\n"
                "Decompose each goal into concrete action steps.\n"
                "For each action:\n"
                "- Name (verb + object)\n"
                "- Type (research/design/implement/test/review)\n"
                "- Estimated effort (hours)\n"
                "- Parallelizable (yes/no)"
            ),
            version=1,
            description="Standard action decomposition from goals",
        )
    )

    # Debate prompts
    registry.register(
        PromptTemplate(
            id="debate_transition_review_v1",
            domain="debate",
            stage="transition_review",
            template=(
                "Review this pipeline stage transition from {from_stage} to {to_stage}.\n\n"
                "## Input\n{input_data}\n\n"
                "## Output\n{output_data}\n\n"
                "Evaluate whether the output correctly captures the input intent.\n"
                "End with: VERDICT: PROCEED | REVISE | BLOCK"
            ),
            version=1,
            description="Stage transition review prompt for mini-debates",
        )
    )

    # Agent system prompts
    registry.register(
        PromptTemplate(
            id="agent_proposer_v1",
            domain="agent",
            stage="proposal",
            template=(
                "You are a {role} participating in a structured debate.\n\n"
                "Task: {task}\n\n"
                "Provide a well-reasoned proposal addressing this task.\n"
                "Support your reasoning with evidence and consider trade-offs."
            ),
            version=1,
            description="Standard proposer system prompt",
        )
    )

    registry.register(
        PromptTemplate(
            id="agent_critic_v1",
            domain="agent",
            stage="critique",
            template=(
                "You are a critical reviewer in a structured debate.\n\n"
                "Task: {task}\n\n"
                "Proposal to critique:\n{proposal}\n\n"
                "Identify weaknesses, gaps, and risks in this proposal.\n"
                "Be constructive — suggest improvements for each critique."
            ),
            version=1,
            description="Standard critic system prompt",
        )
    )

    logger.debug("Registered %d built-in prompt templates", len(registry._templates))
