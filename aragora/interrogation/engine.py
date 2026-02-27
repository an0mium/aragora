"""Interrogation Engine: Debate-driven prompt clarification and spec generation.

The engine coordinates:
1. Decomposition: vague prompt → concrete dimensions
2. Research: gather context from KM, Obsidian, codebase, web
3. Question prioritization: agents DEBATE which questions matter most
4. Crystallization: answers + research → MoSCoW specification

The key innovation is step 3: instead of a single LLM generating questions,
multiple agents argue about which questions are most important. This surfaces
blind spots that a single model would miss.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from aragora.interrogation.crystallizer import Crystallizer, CrystallizedSpec

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
    hidden_assumption: str = ""  # assumption revealed by the question
    options: list[str] = field(default_factory=list)  # suggested answers
    answer: str = ""  # user's answer (filled after asking)
    category: str = ""  # "scope", "technical", "business", "risk"


@dataclass
class InterrogationConfig:
    """Configuration for the interrogation engine."""

    max_questions: int = 7
    min_priority: float = 0.3  # questions below this threshold are dropped
    skip_research: bool = False
    skip_debate: bool = False  # if True, use single-agent question generation
    debate_rounds: int = 2  # rounds of debate about question priority
    autonomy: str = "propose_and_approve"  # or "fully_autonomous", "human_guided"


@dataclass
class InterrogationResult:
    """Full result of an interrogation session."""

    original_prompt: str
    dimensions: list[str]  # decomposed dimensions of the prompt
    research_summary: str
    prioritized_questions: list[PrioritizedQuestion]
    crystallized_spec: CrystallizedSpec | None = None
    debate_reasoning: str = ""  # why questions were ranked this way
    metadata: dict[str, Any] = field(default_factory=dict)


# Type alias for user interaction callback
QuestionCallback = Callable[[list[PrioritizedQuestion]], Awaitable[list[PrioritizedQuestion]]]


# ── Prompt Templates ──────────────────────────────────────────────


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
    """Orchestrates debate-driven prompt interrogation.

    The engine's unique value is using multi-agent debate to prioritize
    which clarifying questions matter most. This catches blind spots
    that single-model question generation misses.

    Args:
        agents: List of agents for question generation and debate (2+ recommended)
        crystallizer_agent: Agent for spec crystallization
        config: Interrogation configuration
        on_questions: Callback for user to answer questions
    """

    def __init__(
        self,
        agents: list[InterrogationAgent] | None = None,
        crystallizer_agent: InterrogationAgent | None = None,
        config: InterrogationConfig | None = None,
        on_questions: QuestionCallback | None = None,
    ):
        self.agents = agents or []
        self.crystallizer = Crystallizer(crystallizer_agent) if crystallizer_agent else None
        self.config = config or InterrogationConfig()
        self.on_questions = on_questions

    async def interrogate(self, prompt: str) -> InterrogationResult:
        """Run the full interrogation pipeline on a prompt.

        Args:
            prompt: The user's vague or broad prompt

        Returns:
            InterrogationResult with prioritized questions and optional spec
        """
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

        # Stage 1: Decompose prompt into dimensions
        dimensions = await self._decompose(prompt, primary_agent)
        logger.info("Decomposed into %d dimensions", len(dimensions))

        # Stage 2: Research context
        research_summary = ""
        if not self.config.skip_research:
            research_summary = await self._research(prompt, primary_agent)

        # Stage 3: Generate candidate questions
        questions = await self._generate_questions(
            prompt, dimensions, research_summary, primary_agent
        )
        logger.info("Generated %d candidate questions", len(questions))

        # Stage 4: Debate-prioritize questions (if multiple agents)
        debate_reasoning = ""
        if len(self.agents) >= 2 and not self.config.skip_debate:
            questions, debate_reasoning = await self._debate_priorities(prompt, questions)
            logger.info("Debate-ranked %d questions", len(questions))

        # Filter by minimum priority
        questions = [q for q in questions if q.priority_score >= self.config.min_priority]
        questions = questions[: self.config.max_questions]

        # Stage 5: Ask user (if callback provided)
        if self.on_questions and questions:
            questions = await self.on_questions(questions)

        # Stage 6: Crystallize spec (if crystallizer available and answers exist)
        spec = None
        if self.crystallizer and any(q.answer for q in questions):
            user_answers = "\n".join(
                f"Q: {q.question}\nA: {q.answer}" for q in questions if q.answer
            )
            spec = await self.crystallizer.crystallize(
                prompt=prompt,
                research_context=research_summary,
                debate_conclusions=debate_reasoning,
                user_answers=user_answers,
            )

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

    # ── Internal Stages ───────────────────────────────────────────

    async def _decompose(self, prompt: str, agent: InterrogationAgent) -> list[str]:
        """Decompose prompt into concrete dimensions."""
        response = await agent.generate(DECOMPOSE_PROMPT.format(prompt=prompt))
        return self._parse_dimensions(response)

    async def _research(self, prompt: str, agent: InterrogationAgent) -> str:
        """Gather research context for the prompt."""
        response = await agent.generate(RESEARCH_PROMPT.format(prompt=prompt))
        # Extract summary if structured, otherwise use raw response
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
        """Generate candidate clarifying questions."""
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
        """Use multi-agent debate to prioritize questions.

        Each agent ranks the questions by importance. The final priority
        is the average rank across all agents, normalized to 0-1.
        """
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

        # Average rankings across agents → priority scores
        if all_rankings and questions:
            n = len(questions)
            for i, q in enumerate(questions):
                ranks = [r[i] if i < len(r) else n for r in all_rankings]
                avg_rank = sum(ranks) / len(ranks)
                # Convert rank to score (rank 1 → score 1.0, rank n → score ~0)
                q.priority_score = max(0.0, 1.0 - (avg_rank - 1) / max(n - 1, 1))

            # Sort by priority
            questions.sort(key=lambda q: q.priority_score, reverse=True)

        debate_reasoning = " | ".join(all_reasoning) if all_reasoning else ""
        return questions, debate_reasoning

    # ── Parsing ───────────────────────────────────────────────────

    def _parse_dimensions(self, text: str) -> list[str]:
        """Parse decomposition output into dimension list."""
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
        """Parse question generation output."""
        import re

        questions = []
        # Split by QUESTION: markers
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
                options = [o.strip() for o in options_match.group(1).split(",") if o.strip()]

            cat_match = re.search(r"CATEGORY:\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
            if cat_match:
                category = cat_match.group(1).strip().lower()

            questions.append(
                PrioritizedQuestion(
                    question=question_text,
                    why_it_matters=why,
                    priority_score=0.5,  # default, updated by debate
                    hidden_assumption=assumption,
                    options=options,
                    category=category,
                )
            )

        return questions

    def _parse_ranking(self, text: str, num_questions: int) -> list[int]:
        """Parse debate ranking output into ordered list of question indices.

        Returns a list where index i contains the rank assigned to question i+1.
        """
        import re

        ranks = [num_questions] * num_questions  # default: lowest rank

        pattern = re.compile(r"RANK\s+(\d+):\s*(?:question\s+)?(\d+)", re.IGNORECASE)
        for match in pattern.finditer(text):
            rank = int(match.group(1))
            q_num = int(match.group(2))
            if 1 <= q_num <= num_questions:
                ranks[q_num - 1] = rank

        return ranks
