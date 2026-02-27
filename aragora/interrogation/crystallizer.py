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


class RequirementLevel(str, Enum):
    """Legacy requirement priority levels."""

    MUST = "must"
    SHOULD = "should"
    COULD = "could"
    WONT = "wont"


@dataclass
class Requirement:
    """Legacy requirement shape used by interrogation tests/executor."""

    description: str
    level: RequirementLevel
    dimension: str


@dataclass
class Spec:
    """Legacy spec shape used by interrogation tests/executor."""

    problem_statement: str
    requirements: list[Requirement] = field(default_factory=list)
    non_requirements: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    context_summary: str = ""

    def to_goal_text(self) -> str:
        """Render a deterministic goal string for execution."""
        lines = [self.problem_statement.strip()]
        if self.requirements:
            lines.append("\nRequirements:")
            for req in self.requirements:
                lines.append(f"- [{req.level.value}] {req.description}")
        if self.success_criteria:
            lines.append("\nSuccess Criteria:")
            for criterion in self.success_criteria:
                lines.append(f"- {criterion}")
        return "\n".join(line for line in lines if line).strip()


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

    Also preserves the legacy, synchronous crystallize API used by
    interrogation unit tests and executor integration.
    """

    def __init__(self, agent: CrystallizerAgent | None = None):
        self.agent = agent

    def crystallize(self, *args: Any, **kwargs: Any) -> Any:
        """Dual-mode API.

        Legacy mode (sync):
          crystallize(decomposition, question_set, user_answers, research_result) -> Spec

        Modern mode (async):
          await crystallize(prompt=..., research_context=..., debate_conclusions=..., user_answers=...)
            -> CrystallizedSpec
        """
        if args and not kwargs and len(args) >= 3 and hasattr(args[0], "dimensions"):
            decomposition = args[0]
            question_set = args[1]
            user_answers = args[2]
            research_result = args[3] if len(args) > 3 else None
            return self._legacy_crystallize(
                decomposition=decomposition,
                question_set=question_set,
                user_answers=user_answers,
                research_result=research_result,
            )

        if args and isinstance(args[0], str) and "prompt" not in kwargs:
            kwargs["prompt"] = args[0]

        if "prompt" not in kwargs:
            raise TypeError("crystallize() requires either legacy args or prompt=... kwargs")

        return self._crystallize_async(
            prompt=kwargs.get("prompt", ""),
            research_context=kwargs.get("research_context", ""),
            debate_conclusions=kwargs.get("debate_conclusions", ""),
            user_answers=kwargs.get("user_answers", ""),
        )

    async def _crystallize_async(
        self,
        prompt: str,
        research_context: str = "",
        debate_conclusions: str = "",
        user_answers: str = "",
    ) -> CrystallizedSpec:
        """Modern async crystallization path."""
        if self.agent is None:
            fallback_req = user_answers.splitlines()[0].strip() if user_answers.strip() else prompt
            if not fallback_req:
                fallback_req = "Define a concrete implementation outcome"
            return CrystallizedSpec(
                title=prompt[:80] or "Interrogation Spec",
                problem_statement=prompt or "Clarify objective",
                requirements=[MoSCoWItem(description=fallback_req, priority="must")],
                non_requirements=[],
                success_criteria=["Objective is specific and testable"],
                risks=[],
                implications=[],
                constraints=[],
                prior_art=[],
            )

        full_prompt = CRYSTALLIZE_PROMPT.format(
            prompt=prompt,
            research_context=research_context or "No research available.",
            debate_conclusions=debate_conclusions or "No debate conclusions.",
            user_answers=user_answers or "No user answers provided.",
        )

        response = await self.agent.generate(full_prompt)
        return self._parse_spec(response, prompt)

    def _legacy_crystallize(
        self,
        decomposition: Any,
        question_set: Any,
        user_answers: dict[str, str],
        research_result: Any | None = None,
    ) -> Spec:
        """Legacy sync API used by tests and executor wiring."""
        requirements: list[Requirement] = []

        for question in getattr(question_set, "questions", []):
            text = getattr(question, "text", "").strip()
            if not text:
                continue
            answer = user_answers.get(text, "").strip()
            if not answer:
                continue
            requirements.append(
                Requirement(
                    description=f"{text}: {answer}",
                    level=RequirementLevel.MUST,
                    dimension=getattr(question, "dimension_name", "interrogation"),
                )
            )

        if not requirements:
            requirements.append(
                Requirement(
                    description="Define a measurable implementation target",
                    level=RequirementLevel.MUST,
                    dimension="interrogation",
                )
            )

        success_criteria = [
            f"Requirement satisfied: {req.description}" for req in requirements[:3]
        ] or ["User confirms the spec is actionable"]

        summary = ""
        if research_result is not None:
            summary = getattr(research_result, "summary_text", "") or ""
            if not summary and hasattr(research_result, "summary"):
                try:
                    summary = str(research_result.summary())
                except Exception:  # pragma: no cover - defensive
                    summary = ""

        return Spec(
            problem_statement=getattr(decomposition, "original_prompt", "") or "Clarify objective",
            requirements=requirements,
            non_requirements=["Unspecified enhancements are out of scope"],
            success_criteria=success_criteria,
            risks=["Ambiguous answers can lead to rework"],
            context_summary=summary,
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
