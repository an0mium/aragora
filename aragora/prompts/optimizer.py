"""Prompt optimizer — DSPy-inspired programmatic prompt improvement.

Uses outcome data from the PromptRegistry to iteratively improve prompt
templates. The optimizer works in cycles:

1. Select underperforming prompts (low avg_score or high variance)
2. Generate candidate variants using a meta-agent
3. A/B test candidates against the current best
4. Promote winners, retire losers

Usage::

    optimizer = PromptOptimizer(registry)

    # Generate a variant of an underperforming prompt
    variant = optimizer.generate_variant("pipeline_ideas_to_goals_v1")

    # Run an optimization cycle
    report = optimizer.run_cycle("pipeline", "ideas_to_goals")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from aragora.prompts.registry import PromptRegistry, PromptTemplate

logger = logging.getLogger(__name__)


@dataclass
class OptimizationCandidate:
    """A candidate prompt variant to test."""

    template: PromptTemplate
    source_id: str  # Original template this was derived from
    mutation_type: str  # rewrite | restructure | simplify | domain_tune
    rationale: str = ""


@dataclass
class OptimizationResult:
    """Result of an optimization cycle."""

    domain: str
    stage: str
    candidates_generated: int = 0
    candidates_promoted: int = 0
    candidates_retired: int = 0
    best_template_id: str = ""
    best_score: float = 0.0
    improvement: float = 0.0  # Score delta vs previous best
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "stage": self.stage,
            "candidates_generated": self.candidates_generated,
            "candidates_promoted": self.candidates_promoted,
            "candidates_retired": self.candidates_retired,
            "best_template_id": self.best_template_id,
            "best_score": self.best_score,
            "improvement": self.improvement,
            "timestamp": self.timestamp,
        }


# -- Mutation strategies ---------------------------------------------------

MUTATION_STRATEGIES = {
    "restructure": (
        "Restructure the prompt for clarity:\n"
        "- Use numbered steps instead of prose\n"
        "- Put the most important instruction first\n"
        "- Add explicit output format requirements\n"
    ),
    "simplify": (
        "Simplify the prompt:\n"
        "- Remove redundant instructions\n"
        "- Use shorter, clearer sentences\n"
        "- Focus on the single most important outcome\n"
    ),
    "add_examples": (
        "Add concrete examples to the prompt:\n"
        "- Include one good example of expected output\n"
        "- Show the format you want responses in\n"
        "- Keep examples concise but illustrative\n"
    ),
    "add_constraints": (
        "Add quality constraints to the prompt:\n"
        "- Specify what NOT to include\n"
        "- Add measurability criteria\n"
        "- Include validation checks for the output\n"
    ),
}


class PromptOptimizer:
    """Optimizes prompt templates based on performance data.

    The optimizer uses a lightweight evolutionary approach:
    - Templates with enough data (>= min_uses) are evaluated
    - Underperformers (< score_threshold) get mutation candidates
    - Candidates are registered for A/B testing
    - After enough data, winners are promoted

    This is intentionally simpler than full DSPy — it operates on
    template strings rather than full program graphs, and uses
    structural mutations rather than gradient-based optimization.
    """

    def __init__(
        self,
        registry: PromptRegistry,
        min_uses: int = 10,
        score_threshold: float = 0.7,
        max_candidates: int = 3,
    ) -> None:
        self.registry = registry
        self.min_uses = min_uses
        self.score_threshold = score_threshold
        self.max_candidates = max_candidates
        self._optimization_history: list[OptimizationResult] = []

    def identify_underperformers(
        self,
        domain: str | None = None,
        stage: str | None = None,
    ) -> list[PromptTemplate]:
        """Find templates that have enough data but underperform.

        Returns templates with:
        - At least `min_uses` recorded outcomes
        - Average score below `score_threshold`
        """
        templates = self.registry.list_templates(domain=domain, stage=stage, active_only=True)

        underperformers = []
        for t in templates:
            if t.metrics.uses >= self.min_uses and t.metrics.avg_score < self.score_threshold:
                underperformers.append(t)

        underperformers.sort(key=lambda t: t.metrics.avg_score)
        return underperformers

    def generate_variant(
        self,
        template_id: str,
        mutation_type: str | None = None,
    ) -> OptimizationCandidate | None:
        """Generate a variant of an existing template using structural mutation.

        If no mutation_type is specified, one is chosen based on the template's
        weaknesses (low score → restructure, high variance → simplify, etc.).
        """
        source = self.registry.get(template_id)
        if source is None:
            logger.warning("Cannot generate variant for unknown template: %s", template_id)
            return None

        if mutation_type is None:
            mutation_type = self._select_mutation_type(source)

        mutated_template = self._apply_mutation(source, mutation_type)
        if mutated_template is None:
            return None

        # Create a new template with incremented version
        existing = self.registry.list_templates(
            domain=source.domain, stage=source.stage, active_only=False
        )
        next_version = max((t.version for t in existing), default=0) + 1

        new_id = f"{source.domain}_{source.stage}_v{next_version}"
        variant = PromptTemplate(
            id=new_id,
            domain=source.domain,
            stage=source.stage,
            template=mutated_template,
            version=next_version,
            description=f"Auto-generated variant of {template_id} ({mutation_type})",
            metadata={
                "source_id": template_id,
                "mutation_type": mutation_type,
                "auto_generated": True,
            },
        )

        return OptimizationCandidate(
            template=variant,
            source_id=template_id,
            mutation_type=mutation_type,
            rationale=f"Template {template_id} has avg_score={source.metrics.avg_score:.2f}, "
            f"applying {mutation_type} mutation",
        )

    def promote_candidate(self, candidate: OptimizationCandidate) -> str:
        """Register a candidate variant in the registry for A/B testing.

        Returns the template ID of the registered variant.
        """
        self.registry.register(candidate.template)
        logger.info(
            "Promoted candidate %s (from %s, mutation=%s)",
            candidate.template.id,
            candidate.source_id,
            candidate.mutation_type,
        )
        return candidate.template.id

    def retire_underperformers(
        self,
        domain: str,
        stage: str,
        keep_best: int = 2,
    ) -> list[str]:
        """Deactivate the worst-performing templates, keeping the top N.

        Only retires templates with enough data to make a fair comparison.
        Returns IDs of retired templates.
        """
        templates = self.registry.list_templates(domain=domain, stage=stage, active_only=True)

        # Only consider templates with enough data
        evaluated = [t for t in templates if t.metrics.uses >= self.min_uses]
        if len(evaluated) <= keep_best:
            return []

        # Sort by avg_score descending
        evaluated.sort(key=lambda t: t.metrics.avg_score, reverse=True)

        retired = []
        for t in evaluated[keep_best:]:
            self.registry.deactivate(t.id)
            retired.append(t.id)
            logger.info(
                "Retired template %s (avg_score=%.2f, uses=%d)",
                t.id,
                t.metrics.avg_score,
                t.metrics.uses,
            )

        return retired

    def run_cycle(
        self,
        domain: str,
        stage: str,
    ) -> OptimizationResult:
        """Run a full optimization cycle for a domain/stage.

        Steps:
        1. Identify underperformers
        2. Generate mutation candidates
        3. Register candidates for A/B testing
        4. Retire consistently bad templates
        5. Return cycle report
        """
        result = OptimizationResult(domain=domain, stage=stage)

        # Step 1: Find underperformers
        underperformers = self.identify_underperformers(domain=domain, stage=stage)

        # Step 2-3: Generate and register candidates
        for source in underperformers[: self.max_candidates]:
            candidate = self.generate_variant(source.id)
            if candidate:
                self.promote_candidate(candidate)
                result.candidates_generated += 1
                result.candidates_promoted += 1

        # Step 4: Retire consistently bad templates
        retired = self.retire_underperformers(domain, stage)
        result.candidates_retired = len(retired)

        # Step 5: Report on current best
        best = self.registry.get_best(domain, stage)
        if best:
            result.best_template_id = best.id
            result.best_score = best.metrics.avg_score

        self._optimization_history.append(result)

        logger.info(
            "Optimization cycle for %s/%s: generated=%d, retired=%d, best=%s (%.2f)",
            domain,
            stage,
            result.candidates_generated,
            result.candidates_retired,
            result.best_template_id,
            result.best_score,
        )

        return result

    def get_optimization_history(self) -> list[dict[str, Any]]:
        """Get history of all optimization cycles."""
        return [r.to_dict() for r in self._optimization_history]

    def suggest_feedback_prompts(
        self,
        domain: str,
        stage: str,
    ) -> list[str]:
        """Generate user-facing feedback questions for a domain/stage.

        Returns questions that, when answered, improve prompt optimization.
        """
        templates = self.registry.list_templates(domain=domain, stage=stage, active_only=True)

        questions = []
        if not templates:
            return questions

        best = self.registry.get_best(domain, stage)
        if best and best.metrics.uses < self.min_uses:
            questions.append(
                f"Rate the quality of this {stage} output (1-5): "
                "How well does it capture your intent?"
            )

        if len(templates) > 1:
            questions.append(
                f"Which {stage} format do you prefer? We're testing different approaches."
            )

        if any(t.metrics.avg_score < self.score_threshold for t in templates if t.metrics.uses > 0):
            questions.append(
                f"What's missing from the {stage} output? What would make it more useful?"
            )

        return questions

    # -- Internal methods ---------------------------------------------------

    def _select_mutation_type(self, template: PromptTemplate) -> str:
        """Choose the best mutation type based on template characteristics."""
        # Low score with enough data → restructure
        if template.metrics.uses >= self.min_uses and template.metrics.avg_score < 0.5:
            return "restructure"

        # High variance in scores → add constraints
        if template.metrics.scores:
            score_variance = _variance(template.metrics.scores)
            if score_variance > 0.04:  # stddev > 0.2
                return "add_constraints"

        # Short template → add examples
        if len(template.template) < 200:
            return "add_examples"

        # Default → simplify
        return "simplify"

    def _apply_mutation(self, source: PromptTemplate, mutation_type: str) -> str | None:
        """Apply a structural mutation to a template string.

        Returns the mutated template string, or None if mutation fails.
        """
        template = source.template

        if mutation_type == "restructure":
            return self._restructure(template)
        elif mutation_type == "simplify":
            return self._simplify(template)
        elif mutation_type == "add_examples":
            return self._add_examples(template, source.stage)
        elif mutation_type == "add_constraints":
            return self._add_constraints(template)
        else:
            logger.warning("Unknown mutation type: %s", mutation_type)
            return None

    def _restructure(self, template: str) -> str:
        """Restructure template into numbered steps."""
        lines = [line.strip() for line in template.split("\n") if line.strip()]

        if not lines:
            return template

        # Keep first line as instruction, restructure rest as numbered steps
        result_lines = [lines[0], ""]
        step_num = 1

        for line in lines[1:]:
            # Convert bullet points or free-text to numbered steps
            cleaned = re.sub(r"^[-*•]\s*", "", line)
            if cleaned and not cleaned.startswith("{"):
                result_lines.append(f"{step_num}. {cleaned}")
                step_num += 1
            else:
                result_lines.append(cleaned)

        return "\n".join(result_lines)

    def _simplify(self, template: str) -> str:
        """Remove redundant instructions and shorten."""
        lines = template.split("\n")

        # Remove empty lines and deduplicate similar instructions
        seen_patterns: set[str] = set()
        simplified = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if simplified and simplified[-1]:  # Keep single blank lines
                    simplified.append("")
                continue

            # Normalize for dedup check
            pattern = re.sub(r"\{[^}]+\}", "VAR", stripped.lower())
            if pattern not in seen_patterns:
                seen_patterns.add(pattern)
                simplified.append(line)

        return "\n".join(simplified).strip()

    def _add_examples(self, template: str, stage: str) -> str:
        """Add an example output section to the template."""
        example_section = (
            f"\n\nExample output format for {stage}:\n"
            "- Use clear, actionable language\n"
            "- Each item should be independently verifiable\n"
            "- Include reasoning for each recommendation"
        )
        return template + example_section

    def _add_constraints(self, template: str) -> str:
        """Add quality constraints to the template."""
        constraints = (
            "\n\nQuality constraints:\n"
            "- Do NOT include vague or unmeasurable goals\n"
            "- Every recommendation must have a clear success criterion\n"
            "- Limit output to the most impactful items (max 5)\n"
            "- Flag any assumptions you are making"
        )
        return template + constraints


def _variance(values: list[float]) -> float:
    """Calculate variance of a list of values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)
