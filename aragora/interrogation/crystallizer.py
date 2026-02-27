"""Crystallizer: Transforms debate output + user answers into MoSCoW specs.

Takes unstructured research, debate conclusions, and user answers, then
produces a structured specification with Must/Should/Could/Won't priorities,
explicit non-requirements, measurable success criteria, risk register,
and implications the user didn't state but would want.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class CrystallizerAgent(Protocol):
    """Agent that can generate crystallization output."""

    async def generate(self, prompt: str) -> str: ...


@dataclass
class MoSCoWItem:
    """A single requirement with MoSCoW priority."""

    description: str
    priority: str  # "must", "should", "could", "wont"
    rationale: str = ""


@dataclass
class CrystallizedSpec:
    """Structured specification output from crystallization."""

    title: str
    problem_statement: str
    requirements: list[MoSCoWItem] = field(default_factory=list)
    non_requirements: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    risks: list[dict[str, str]] = field(default_factory=list)
    implications: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    prior_art: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def musts(self) -> list[MoSCoWItem]:
        return [r for r in self.requirements if r.priority == "must"]

    @property
    def shoulds(self) -> list[MoSCoWItem]:
        return [r for r in self.requirements if r.priority == "should"]

    @property
    def coulds(self) -> list[MoSCoWItem]:
        return [r for r in self.requirements if r.priority == "could"]

    @property
    def wonts(self) -> list[MoSCoWItem]:
        return [r for r in self.requirements if r.priority == "wont"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "problem_statement": self.problem_statement,
            "requirements": [
                {"description": r.description, "priority": r.priority, "rationale": r.rationale}
                for r in self.requirements
            ],
            "non_requirements": self.non_requirements,
            "success_criteria": self.success_criteria,
            "risks": self.risks,
            "implications": self.implications,
            "constraints": self.constraints,
            "prior_art": self.prior_art,
        }

    def to_legacy_spec(self) -> Spec:
        requirements: list[Requirement] = []
        for item in self.requirements:
            if item.priority == "must":
                level = RequirementLevel.MUST
            elif item.priority == "should":
                level = RequirementLevel.SHOULD
            elif item.priority == "could":
                level = RequirementLevel.COULD
            else:
                level = RequirementLevel.WONT

            requirements.append(
                Requirement(
                    description=item.description,
                    level=level,
                    dimension="general",
                )
            )

        risks = [
            f"{risk.get('risk', '').strip()}: {risk.get('mitigation', '').strip()}".strip(": ")
            for risk in self.risks
            if risk.get("risk") or risk.get("mitigation")
        ]
        return Spec(
            problem_statement=self.problem_statement,
            requirements=requirements,
            non_requirements=list(self.non_requirements),
            success_criteria=list(self.success_criteria),
            risks=risks,
            context_summary="Derived from crystallized MoSCoW output.",
        )


class RequirementLevel(str, Enum):
    MUST = "must"
    SHOULD = "should"
    COULD = "could"
    WONT = "wont"


@dataclass
class Requirement:
    description: str
    level: RequirementLevel = RequirementLevel.MUST
    dimension: str = "general"


@dataclass
class Spec:
    problem_statement: str
    requirements: list[Requirement] = field(default_factory=list)
    non_requirements: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    context_summary: str = ""

    def to_goal_text(self) -> str:
        if not self.requirements:
            return self.problem_statement
        requirement_lines = "\n".join(f"- {req.description}" for req in self.requirements)
        return f"{self.problem_statement}\n\nRequirements:\n{requirement_lines}"


CRYSTALLIZE_PROMPT = """You are a specification crystallizer. Given research findings, debate conclusions,
and user answers, produce a structured MoSCoW specification.

ORIGINAL PROMPT: {prompt}

RESEARCH CONTEXT:
{research_context}

DEBATE CONCLUSIONS:
{debate_conclusions}

USER ANSWERS:
{user_answers}

Produce a structured specification with the following sections:

TITLE: [concise project title]

PROBLEM_STATEMENT: [1-3 sentences defining the core problem]

MUST:
- [requirement]: [rationale]
...

SHOULD:
- [requirement]: [rationale]
...

COULD:
- [requirement]: [rationale]
...

WONT:
- [requirement]: [rationale]
...

NON_REQUIREMENTS:
- [things explicitly out of scope]
...

SUCCESS_CRITERIA:
- [measurable criterion]
...

RISKS:
- [risk]: [mitigation]
...

IMPLICATIONS:
- [things the user didn't state but would want to know]
...

CONSTRAINTS:
- [technical or business constraints]
...

PRIOR_ART:
- [existing solutions or approaches to consider]
...

Be precise. Every requirement should be testable. Every success criterion should be measurable."""


class Crystallizer:
    """Transforms debate output into structured MoSCoW specifications.

    The crystallizer takes the messy output of research, debate, and user
    interaction and produces a clean, actionable specification that can
    be handed to an execution pipeline.

    Args:
        agent: LLM agent for crystallization
    """

    def __init__(self, agent: CrystallizerAgent | None = None):
        self.agent = agent

    def crystallize(self, *args: Any, **kwargs: Any) -> Any:
        """Compatibility facade for modern async and legacy sync APIs."""
        if args and hasattr(args[0], "original_prompt") and hasattr(args[0], "dimensions"):
            decomposition = args[0]
            answered_questions = args[1] if len(args) > 1 else None
            user_answers = args[2] if len(args) > 2 else {}
            research = args[3] if len(args) > 3 else None
            return self._crystallize_legacy(
                decomposition=decomposition,
                answered_questions=answered_questions,
                user_answers=user_answers,
                research=research,
            )

        prompt = kwargs.get("prompt", args[0] if args else "")
        research_context = kwargs.get("research_context", "")
        debate_conclusions = kwargs.get("debate_conclusions", "")
        user_answers = kwargs.get("user_answers", "")
        return self._crystallize_modern(
            prompt=prompt,
            research_context=research_context,
            debate_conclusions=debate_conclusions,
            user_answers=user_answers,
        )

    async def _crystallize_modern(
        self,
        prompt: str,
        research_context: str = "",
        debate_conclusions: str = "",
        user_answers: str = "",
    ) -> CrystallizedSpec:
        """Modern async crystallization path used by the new engine."""
        full_prompt = CRYSTALLIZE_PROMPT.format(
            prompt=prompt,
            research_context=research_context or "No research available.",
            debate_conclusions=debate_conclusions or "No debate conclusions.",
            user_answers=user_answers or "No user answers provided.",
        )

        if self.agent is None:
            return CrystallizedSpec(
                title=prompt[:80] or "Crystallized Spec",
                problem_statement=prompt,
                requirements=[
                    MoSCoWItem(
                        description="Clarify concrete implementation requirements.",
                        priority="must",
                        rationale="No crystallizer agent was configured.",
                    )
                ],
                success_criteria=["Stakeholders approve the clarified requirements."],
                risks=["Missing LLM crystallizer may reduce requirement quality."],
            )

        response = await self.agent.generate(full_prompt)
        return self._parse_spec(response, prompt)

    def _crystallize_legacy(
        self,
        decomposition: Any,
        answered_questions: Any,
        user_answers: dict[str, str] | None,
        research: Any,
    ) -> Spec:
        """Legacy sync crystallization path expected by older tests/callers."""
        answers = user_answers or {}
        requirements: list[Requirement] = []

        questions = getattr(answered_questions, "questions", []) if answered_questions else []
        for question in questions:
            question_text = getattr(question, "text", "").strip()
            if not question_text:
                continue

            answer_text = answers.get(question_text, "").strip()
            if not answer_text:
                continue

            priority = float(getattr(question, "priority", 0.5) or 0.5)
            if priority >= 0.75:
                level = RequirementLevel.MUST
            elif priority >= 0.5:
                level = RequirementLevel.SHOULD
            else:
                level = RequirementLevel.COULD

            requirements.append(
                Requirement(
                    description=f"{question_text} {answer_text}",
                    level=level,
                    dimension=getattr(question, "dimension_name", "general"),
                )
            )

        if not requirements:
            requirements.append(
                Requirement(
                    description="Capture at least one explicit requirement before execution.",
                    level=RequirementLevel.MUST,
                    dimension="scope",
                )
            )

        success_criteria = [
            f"Requirement validated: {req.description[:140]}" for req in requirements[:3]
        ]
        if not success_criteria:
            success_criteria = ["At least one requirement is validated with the user."]

        findings = getattr(research, "findings", {}) if research is not None else {}
        context_summary = (
            f"Research dimensions referenced: {', '.join(sorted(findings.keys()))}"
            if findings
            else "No prior research context available."
        )

        return Spec(
            problem_statement=getattr(decomposition, "original_prompt", "").strip()
            or "Unknown prompt",
            requirements=requirements,
            non_requirements=["Out-of-scope enhancements without explicit user confirmation."],
            success_criteria=success_criteria,
            risks=["Ambiguity can still create implementation drift."],
            context_summary=context_summary,
        )

    def _parse_spec(self, text: str, original_prompt: str) -> CrystallizedSpec:
        """Parse crystallizer LLM output into structured spec."""
        title = self._extract_field(text, "TITLE") or original_prompt[:80]
        problem = self._extract_field(text, "PROBLEM_STATEMENT") or ""

        requirements: list[MoSCoWItem] = []
        for priority in ("MUST", "SHOULD", "COULD", "WONT"):
            items = self._extract_list_with_rationale(text, priority)
            moscow = priority.lower()
            for desc, rationale in items:
                requirements.append(
                    MoSCoWItem(description=desc, priority=moscow, rationale=rationale)
                )

        non_requirements = self._extract_list(text, "NON_REQUIREMENTS")
        success_criteria = self._extract_list(text, "SUCCESS_CRITERIA")
        risks = self._extract_risk_list(text)
        implications = self._extract_list(text, "IMPLICATIONS")
        constraints = self._extract_list(text, "CONSTRAINTS")
        prior_art = self._extract_list(text, "PRIOR_ART")

        return CrystallizedSpec(
            title=title,
            problem_statement=problem,
            requirements=requirements,
            non_requirements=non_requirements,
            success_criteria=success_criteria,
            risks=risks,
            implications=implications,
            constraints=constraints,
            prior_art=prior_art,
        )

    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract a single-line field value."""
        match = re.search(
            rf"{field_name}:\s*(.+?)(?:\n\n|\n[A-Z_]+:)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        return match.group(1).strip() if match else ""

    def _extract_list(self, text: str, section_name: str) -> list[str]:
        """Extract a bulleted list from a section."""
        pattern = rf"{section_name}:\s*\n((?:\s*-\s*.+\n?)*)"
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return []
        raw = match.group(1)
        return [
            line.strip().lstrip("- ").strip()
            for line in raw.split("\n")
            if line.strip().startswith("-")
        ]

    def _extract_list_with_rationale(self, text: str, section_name: str) -> list[tuple[str, str]]:
        """Extract list items with optional rationale after colon."""
        items = self._extract_list(text, section_name)
        result: list[tuple[str, str]] = []
        for item in items:
            if ":" in item:
                parts = item.split(":", 1)
                result.append((parts[0].strip(), parts[1].strip()))
            else:
                result.append((item, ""))
        return result

    def _extract_risk_list(self, text: str) -> list[dict[str, str]]:
        """Extract risks with mitigations."""
        items = self._extract_list_with_rationale(text, "RISKS")
        return [{"risk": desc, "mitigation": rationale} for desc, rationale in items]
