"""Interrogation Engine: Debate-driven prompt clarification and spec generation.

The engine coordinates:
1. Decomposition: vague prompt -> concrete dimensions
2. Research: gather context from KM, Obsidian, codebase, web
3. Question prioritization: agents debate which questions matter most
4. Crystallization: answers + research -> structured specification
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Any, Protocol

from aragora.interrogation.crystallizer import (
    CrystallizedSpec,
    Crystallizer,
    Requirement,
    RequirementLevel,
    Spec,
)

logger = logging.getLogger(__name__)


class InterrogationAgent(Protocol):
    """Agent interface for interrogation stages."""

    async def generate(self, prompt: str) -> str: ...


@dataclass
class PrioritizedQuestion:
    """A question with debate-assigned priority and context."""

    question: str
    why_it_matters: str
    priority_score: float  # 0.0-1.0, from debate ranking
    hidden_assumption: str = ""
    options: list[str] = field(default_factory=list)
    answer: str = ""
    category: str = ""


@dataclass
class InterrogationDimension:
    """Legacy dimension shape used by start() API."""

    name: str
    description: str
    vagueness_score: float


@dataclass
class InterrogationQuestion:
    """Legacy question shape used by start()/answer() APIs."""

    text: str
    why: str
    options: list[str] = field(default_factory=list)
    context: str = ""
    priority: int = 1
    dimension_name: str = "interrogation"


@dataclass
class InterrogationState:
    """Legacy mutable interrogation state used by handler/tests."""

    prompt: str
    dimensions: list[InterrogationDimension]
    questions: list[InterrogationQuestion]
    answers: dict[str, str] = field(default_factory=dict)

    @property
    def unanswered(self) -> list[InterrogationQuestion]:
        return [q for q in self.questions if q.text not in self.answers]

    @property
    def is_complete(self) -> bool:
        return len(self.unanswered) == 0


@dataclass
class InterrogationConfig:
    """Configuration for the interrogation engine."""

    max_questions: int = 7
    min_priority: float = 0.3
    skip_research: bool = False
    skip_debate: bool = False
    debate_rounds: int = 2
    autonomy: str = "propose_and_approve"


@dataclass
class InterrogationResult:
    """Full result of an interrogation session."""

    original_prompt: str
    dimensions: list[str]
    research_summary: str
    prioritized_questions: list[PrioritizedQuestion]
    crystallized_spec: CrystallizedSpec | Spec | None = None
    debate_reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self) -> Spec:
        """Legacy accessor expected by interrogation tests."""
        if isinstance(self.crystallized_spec, Spec):
            return self.crystallized_spec

        if isinstance(self.crystallized_spec, CrystallizedSpec):
            requirements = [
                Requirement(
                    description=item.description,
                    level=_priority_to_level(item.priority),
                    dimension="interrogation",
                )
                for item in self.crystallized_spec.requirements
            ]
            if not requirements:
                requirements.append(
                    Requirement(
                        description="Define measurable implementation steps",
                        level=RequirementLevel.MUST,
                        dimension="interrogation",
                    )
                )
            return Spec(
                problem_statement=self.crystallized_spec.problem_statement or self.original_prompt,
                requirements=requirements,
                non_requirements=self.crystallized_spec.non_requirements,
                success_criteria=self.crystallized_spec.success_criteria
                or ["Specification is actionable"],
                risks=[risk.get("risk", "") for risk in self.crystallized_spec.risks],
                context_summary=self.research_summary,
            )

        return Spec(
            problem_statement=self.original_prompt,
            requirements=[
                Requirement(
                    description="Define measurable implementation steps",
                    level=RequirementLevel.MUST,
                    dimension="interrogation",
                )
            ],
            success_criteria=["Specification is actionable"],
        )


def _priority_to_level(priority: str) -> RequirementLevel:
    normalized = (priority or "must").strip().lower()
    if normalized == "should":
        return RequirementLevel.SHOULD
    if normalized == "could":
        return RequirementLevel.COULD
    if normalized in {"wont", "won't"}:
        return RequirementLevel.WONT
    return RequirementLevel.MUST


QuestionCallback = Callable[[list[PrioritizedQuestion]], Awaitable[list[PrioritizedQuestion]]]


DECOMPOSE_PROMPT = """Analyze this prompt and identify 3-7 concrete dimensions that need clarification.

PROMPT: {prompt}

For each dimension, explain:
DIMENSION: [name]
DESCRIPTION: [what needs clarification]
CATEGORY: [scope | technical | business | risk]

Focus on dimensions where the user's intent is ambiguous or where
different interpretations lead to very different outcomes."""


QUESTION_GEN_PROMPT = """Given these dimensions of an ambiguous prompt, generate clarifying questions.

ORIGINAL PROMPT: {prompt}

DIMENSIONS:
{dimensions}

RESEARCH CONTEXT:
{research_context}

For each important dimension, generate a clarifying question:
QUESTION: [the question to ask]
WHY: [why this question matters for the outcome]
ASSUMPTION: [hidden assumption this question reveals]
OPTIONS: [2-4 suggested answers, comma-separated]
CATEGORY: [scope | technical | business | risk]

Generate the most impactful questions first. Each question should change
the implementation direction if answered differently."""


DEBATE_PRIORITY_PROMPT = """You are debating which clarifying questions are most important to ask a user.

The user's prompt is: {prompt}

Here are the candidate questions:
{questions_text}

Your position: Rank these questions by how much the answer would change
the implementation. A question is HIGH priority if:
- Different answers lead to fundamentally different architectures
- The default assumption is likely wrong
- Getting this wrong would waste significant effort

A question is LOW priority if:
- The answer is obvious from context
- It's a nice-to-have clarification but won't change the approach
- It can be decided later without cost

Respond with your ranking:
RANK 1: [question number] - [reasoning]
RANK 2: [question number] - [reasoning]
...

Then provide your overall assessment:
REASONING: [why you ranked them this way]"""


RESEARCH_PROMPT = """Analyze the following prompt and provide research context.

PROMPT: {prompt}

Consider:
1. What existing patterns or solutions address this?
2. What constraints or dependencies exist?
3. What are common pitfalls with this type of work?
4. What prior decisions are relevant?

Provide a concise research summary:
SUMMARY: [2-3 paragraphs of relevant context]"""


class InterrogationEngine:
    """Orchestrates debate-driven prompt interrogation."""

    def __init__(
        self,
        agents: list[InterrogationAgent] | None = None,
        crystallizer_agent: InterrogationAgent | None = None,
        config: InterrogationConfig | None = None,
        on_questions: QuestionCallback | None = None,
        knowledge_mound: Any | None = None,
        **_unused: Any,
    ):
        self.agents = agents or []
        self.crystallizer = Crystallizer(crystallizer_agent) if crystallizer_agent else None
        self.config = config or InterrogationConfig()
        self.on_questions = on_questions
        self.knowledge_mound = knowledge_mound

    async def start(self, prompt: str, sources: list[str] | None = None) -> InterrogationState:
        """Legacy entrypoint preserved for handlers/tests."""
        _ = sources  # reserved for future source-aware state initialization

        if self.agents:
            result = await self.interrogate(prompt)
            dimensions = self._legacy_dimensions_from_result(result)
            questions = self._legacy_questions_from_result(result)
            if dimensions and questions:
                return InterrogationState(prompt=prompt, dimensions=dimensions, questions=questions)

        return self._fallback_state(prompt)

    def answer(self, state: InterrogationState, question: str, answer: str) -> InterrogationState:
        """Legacy answer mutator preserved for handlers/tests."""
        state.answers[question] = answer
        return state

    async def crystallize(self, state: InterrogationState) -> InterrogationResult:
        """Legacy crystallization API preserved for handlers/tests."""
        crystallized: CrystallizedSpec | Spec | None = None

        if self.crystallizer and state.answers:
            user_answers = "\n".join(f"Q: {q}\nA: {a}" for q, a in state.answers.items())
            maybe_spec = self.crystallizer.crystallize(
                prompt=state.prompt,
                research_context="",
                debate_conclusions="",
                user_answers=user_answers,
            )
            crystallized = await maybe_spec if isawaitable(maybe_spec) else maybe_spec

        if crystallized is None:
            crystallized = self._fallback_spec_from_state(state)

        prioritized = [
            PrioritizedQuestion(
                question=q.text,
                why_it_matters=q.why,
                priority_score=max(0.0, min(1.0, q.priority / 10.0)),
                hidden_assumption=q.context,
                options=q.options,
                answer=state.answers.get(q.text, ""),
                category=q.dimension_name,
            )
            for q in state.questions
        ]

        return InterrogationResult(
            original_prompt=state.prompt,
            dimensions=[f"{d.name}: {d.description}" for d in state.dimensions],
            research_summary="",
            prioritized_questions=prioritized,
            crystallized_spec=crystallized,
            metadata={"legacy_api": True, "answered": len(state.answers)},
        )

    async def interrogate(self, prompt: str) -> InterrogationResult:
        """Run the full interrogation pipeline on a prompt."""
        logger.info("Starting interrogation for: %s", prompt[:100])

        if not self.agents:
            return InterrogationResult(
                original_prompt=prompt,
                dimensions=[],
                research_summary="",
                prioritized_questions=[],
                metadata={"error": "No agents configured"},
            )

        primary_agent = self.agents[0]

        dimensions = await self._decompose(prompt, primary_agent)
        logger.info("Decomposed into %d dimensions", len(dimensions))

        research_summary = ""
        if not self.config.skip_research:
            research_summary = await self._research(prompt, primary_agent)

        questions = await self._generate_questions(
            prompt, dimensions, research_summary, primary_agent
        )
        logger.info("Generated %d candidate questions", len(questions))

        debate_reasoning = ""
        if len(self.agents) >= 2 and not self.config.skip_debate:
            questions, debate_reasoning = await self._debate_priorities(prompt, questions)
            logger.info("Debate-ranked %d questions", len(questions))

        questions = [q for q in questions if q.priority_score >= self.config.min_priority]
        questions = questions[: self.config.max_questions]

        if self.on_questions and questions:
            questions = await self.on_questions(questions)

        spec: CrystallizedSpec | Spec | None = None
        if self.crystallizer and any(q.answer for q in questions):
            user_answers = "\n".join(
                f"Q: {q.question}\nA: {q.answer}" for q in questions if q.answer
            )
            maybe_spec = self.crystallizer.crystallize(
                prompt=prompt,
                research_context=research_summary,
                debate_conclusions=debate_reasoning,
                user_answers=user_answers,
            )
            spec = await maybe_spec if isawaitable(maybe_spec) else maybe_spec

        result = InterrogationResult(
            original_prompt=prompt,
            dimensions=dimensions,
            research_summary=research_summary,
            prioritized_questions=questions,
            crystallized_spec=spec,
            debate_reasoning=debate_reasoning,
            metadata={
                "agent_count": len(self.agents),
                "debate_used": len(self.agents) >= 2 and not self.config.skip_debate,
                "total_candidates": len(questions),
            },
        )

        logger.info(
            "Interrogation complete: %d questions, spec=%s",
            len(questions),
            spec is not None,
        )
        return result

    def _fallback_state(self, prompt: str) -> InterrogationState:
        text = prompt.lower()
        if any(word in text for word in ("fast", "latency", "performance", "slow")):
            name = "performance"
            desc = "Clarify performance targets and tradeoffs"
            question = "What performance metric matters most (latency, throughput, or both)?"
        elif any(word in text for word in ("security", "auth", "safe")):
            name = "security"
            desc = "Clarify security requirements and constraints"
            question = "Which security requirements are mandatory for this change?"
        else:
            name = "scope"
            desc = "Clarify concrete outcome and boundaries"
            question = "What concrete outcome should this deliver first?"

        return InterrogationState(
            prompt=prompt,
            dimensions=[InterrogationDimension(name=name, description=desc, vagueness_score=0.7)],
            questions=[
                InterrogationQuestion(
                    text=question,
                    why="Different answers lead to different implementation plans.",
                    options=["Minimum viable", "Balanced", "Comprehensive"],
                    context="Assumes a technical implementation outcome.",
                    priority=1,
                    dimension_name=name,
                )
            ],
        )

    def _legacy_dimensions_from_result(
        self,
        result: InterrogationResult,
    ) -> list[InterrogationDimension]:
        dimensions: list[InterrogationDimension] = []
        for item in result.dimensions:
            if ":" in item:
                name, desc = item.split(":", 1)
                dimensions.append(
                    InterrogationDimension(
                        name=name.strip() or "scope",
                        description=desc.strip() or item.strip(),
                        vagueness_score=0.5,
                    )
                )
            else:
                dimensions.append(
                    InterrogationDimension(
                        name=item.strip() or "scope",
                        description=item.strip() or "Clarify scope",
                        vagueness_score=0.5,
                    )
                )
        return dimensions

    def _legacy_questions_from_result(
        self,
        result: InterrogationResult,
    ) -> list[InterrogationQuestion]:
        questions: list[InterrogationQuestion] = []
        for q in result.prioritized_questions:
            questions.append(
                InterrogationQuestion(
                    text=q.question,
                    why=q.why_it_matters or "Clarifies implementation direction.",
                    options=q.options or ["Yes", "No"],
                    context=q.hidden_assumption,
                    priority=max(1, min(10, int(round(q.priority_score * 10)))),
                    dimension_name=q.category or "interrogation",
                )
            )
        return questions

    def _fallback_spec_from_state(self, state: InterrogationState) -> Spec:
        requirements = [
            Requirement(
                description=f"{question}: {answer}",
                level=RequirementLevel.MUST,
                dimension="interrogation",
            )
            for question, answer in state.answers.items()
        ]
        if not requirements:
            requirements.append(
                Requirement(
                    description="Define a measurable success outcome before execution",
                    level=RequirementLevel.MUST,
                    dimension="scope",
                )
            )

        return Spec(
            problem_statement=state.prompt,
            requirements=requirements,
            non_requirements=["Unscoped enhancements without explicit requirement"],
            success_criteria=["User confirms the crystallized objective is accurate"],
            risks=["Ambiguity in requirements can lead to rework"],
            context_summary="Generated from interrogation answers.",
        )

    async def _decompose(self, prompt: str, agent: InterrogationAgent) -> list[str]:
        response = await agent.generate(DECOMPOSE_PROMPT.format(prompt=prompt))
        return self._parse_dimensions(response)

    async def _research(self, prompt: str, agent: InterrogationAgent) -> str:
        response = await agent.generate(RESEARCH_PROMPT.format(prompt=prompt))
        import re

        match = re.search(r"SUMMARY:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else response.strip()

    async def _generate_questions(
        self,
        prompt: str,
        dimensions: list[str],
        research_context: str,
        agent: InterrogationAgent,
    ) -> list[PrioritizedQuestion]:
        dimensions_text = "\n".join(f"- {d}" for d in dimensions)
        response = await agent.generate(
            QUESTION_GEN_PROMPT.format(
                prompt=prompt,
                dimensions=dimensions_text,
                research_context=research_context or "No research available.",
            )
        )
        return self._parse_questions(response)

    async def _debate_priorities(
        self,
        prompt: str,
        questions: list[PrioritizedQuestion],
    ) -> tuple[list[PrioritizedQuestion], str]:
        questions_text = "\n".join(
            f"{i + 1}. {q.question} (why: {q.why_it_matters})" for i, q in enumerate(questions)
        )

        all_rankings: list[list[int]] = []
        all_reasoning: list[str] = []

        for agent in self.agents:
            response = await agent.generate(
                DEBATE_PRIORITY_PROMPT.format(prompt=prompt, questions_text=questions_text)
            )
            ranking = self._parse_ranking(response, len(questions))
            all_rankings.append(ranking)

            import re

            reasoning_match = re.search(r"REASONING:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                all_reasoning.append(reasoning_match.group(1).strip())

        if all_rankings and questions:
            n = len(questions)
            for i, q in enumerate(questions):
                ranks = [r[i] if i < len(r) else n for r in all_rankings]
                avg_rank = sum(ranks) / len(ranks)
                q.priority_score = max(0.0, 1.0 - (avg_rank - 1) / max(n - 1, 1))

            questions.sort(key=lambda question: question.priority_score, reverse=True)

        debate_reasoning = " | ".join(all_reasoning) if all_reasoning else ""
        return questions, debate_reasoning

    def _parse_dimensions(self, text: str) -> list[str]:
        import re

        dimensions = []
        pattern = re.compile(
            r"DIMENSION:\s*(.+?)\n.*?DESCRIPTION:\s*(.+?)(?:\n|$)",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            name = match.group(1).strip()
            desc = match.group(2).strip()
            dimensions.append(f"{name}: {desc}")
        return dimensions

    def _parse_questions(self, text: str) -> list[PrioritizedQuestion]:
        import re

        questions = []
        blocks = re.split(r"(?=QUESTION:\s)", text, flags=re.IGNORECASE)

        for block in blocks:
            q_match = re.search(r"QUESTION:\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
            if not q_match:
                continue

            question_text = q_match.group(1).strip()
            why = ""
            assumption = ""
            options: list[str] = []
            category = ""

            why_match = re.search(r"WHY:\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
            if why_match:
                why = why_match.group(1).strip()

            assumption_match = re.search(r"ASSUMPTION:\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
            if assumption_match:
                assumption = assumption_match.group(1).strip()

            options_match = re.search(r"OPTIONS:\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
            if options_match:
                options = [
                    option.strip() for option in options_match.group(1).split(",") if option.strip()
                ]

            category_match = re.search(r"CATEGORY:\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
            if category_match:
                category = category_match.group(1).strip().lower()

            questions.append(
                PrioritizedQuestion(
                    question=question_text,
                    why_it_matters=why,
                    priority_score=0.5,
                    hidden_assumption=assumption,
                    options=options,
                    category=category,
                )
            )

        return questions

    def _parse_ranking(self, text: str, num_questions: int) -> list[int]:
        import re

        ranks = [num_questions] * num_questions

        pattern = re.compile(r"RANK\s+(\d+):\s*(?:question\s+)?(\d+)", re.IGNORECASE)
        for match in pattern.finditer(text):
            rank = int(match.group(1))
            question_number = int(match.group(2))
            if 1 <= question_number <= num_questions:
                ranks[question_number - 1] = rank

        return ranks
