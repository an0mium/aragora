"""Crystallize interrogation results into a structured spec."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from aragora.interrogation.decomposer import DecompositionResult
from aragora.interrogation.questioner import QuestionSet
from aragora.interrogation.researcher import ResearchResult


class RequirementLevel(Enum):
    MUST = "must"
    SHOULD = "should"
    COULD = "could"


@dataclass
class Requirement:
    """A single requirement in the spec."""

    description: str
    level: RequirementLevel
    dimension: str
    source: str = ""


@dataclass
class Spec:
    """Structured specification produced by the Interrogation Engine."""

    problem_statement: str
    requirements: list[Requirement] = field(default_factory=list)
    non_requirements: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    context_summary: str = ""

    def to_goal_text(self) -> str:
        """Convert spec to a goal string for HardenedOrchestrator."""
        musts = [r.description for r in self.requirements if r.level == RequirementLevel.MUST]
        parts = [self.problem_statement]
        if musts:
            parts.append("Requirements: " + "; ".join(musts))
        if self.success_criteria:
            parts.append("Success criteria: " + "; ".join(self.success_criteria))
        return "\n".join(parts)


class Crystallizer:
    """Combines decomposition, questions, answers, and research into a spec."""

    def crystallize(
        self,
        decomposition: DecompositionResult,
        questions: QuestionSet,
        user_answers: dict[str, str],
        research: ResearchResult,
    ) -> Spec:
        problem = f"Original request: {decomposition.original_prompt}"

        requirements: list[Requirement] = []
        success_criteria: list[str] = []

        for q in questions.questions:
            answer = user_answers.get(q.text, "")
            if answer:
                requirements.append(
                    Requirement(
                        description=answer,
                        level=RequirementLevel.MUST
                        if q.priority > 0.6
                        else RequirementLevel.SHOULD,
                        dimension=q.dimension_name,
                        source="user_answer",
                    )
                )
                success_criteria.append(f"{q.dimension_name}: {answer[:100]} — verified")

        for dim in decomposition.dimensions:
            already_covered = any(r.dimension == dim.name for r in requirements)
            if not already_covered:
                requirements.append(
                    Requirement(
                        description=dim.description,
                        level=RequirementLevel.COULD,
                        dimension=dim.name,
                        source="decomposition",
                    )
                )

        context_parts = []
        for dim_name, findings in research.findings.items():
            if findings:
                context_parts.append(f"{dim_name}: {len(findings)} prior findings")

        return Spec(
            problem_statement=problem,
            requirements=requirements,
            success_criteria=success_criteria
            or [f"All {len(requirements)} requirements addressed"],
            risks=[],
            context_summary="; ".join(context_parts) if context_parts else "No prior context found",
        )
