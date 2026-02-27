"""Interrogation Engine facade -- orchestrates decomposer, researcher, questioner, crystallizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aragora.interrogation.crystallizer import Crystallizer, Spec
from aragora.interrogation.decomposer import DecompositionResult, Dimension, InterrogationDecomposer
from aragora.interrogation.questioner import InterrogationQuestioner, Question, QuestionSet
from aragora.interrogation.researcher import InterrogationResearcher, ResearchResult


@dataclass
class InterrogationState:
    """Mutable state of an in-progress interrogation."""

    prompt: str
    dimensions: list[Dimension]
    research: ResearchResult
    questions: list[Question]
    answers: dict[str, str] = field(default_factory=dict)
    decomposition: DecompositionResult | None = None
    _question_set: QuestionSet | None = field(default=None, repr=False)

    @property
    def unanswered(self) -> list[Question]:
        return [q for q in self.questions if q.text not in self.answers]

    @property
    def is_complete(self) -> bool:
        return len(self.unanswered) == 0


@dataclass
class InterrogationResult:
    """Final output of the interrogation process."""

    spec: Spec
    state: InterrogationState


class InterrogationEngine:
    """Main facade for the Interrogation Engine.

    Usage:
        engine = InterrogationEngine()
        state = await engine.start("Make aragora more powerful")
        state = engine.answer(state, question_text, user_answer)
        result = await engine.crystallize(state)
        # result.spec is the structured specification
    """

    def __init__(
        self,
        knowledge_mound: Any | None = None,
        obsidian: Any | None = None,
    ) -> None:
        self._decomposer = InterrogationDecomposer()
        self._researcher = InterrogationResearcher(
            knowledge_mound=knowledge_mound,
            obsidian=obsidian,
        )
        self._questioner = InterrogationQuestioner()
        self._crystallizer = Crystallizer()

    async def start(
        self,
        prompt: str,
        sources: list[str] | None = None,
    ) -> InterrogationState:
        """Decompose a prompt, research dimensions, and generate questions.

        Args:
            prompt: The user's vague or specific goal text.
            sources: Optional list of source names for the researcher
                     (e.g. ["knowledge_mound", "obsidian"]).

        Returns:
            InterrogationState ready for answering questions.
        """
        decomposition = self._decomposer.decompose(prompt)
        research = await self._researcher.research(
            decomposition.dimensions,
            sources=sources or [],
        )
        question_set = self._questioner.generate(decomposition.dimensions, research)

        return InterrogationState(
            prompt=prompt,
            dimensions=decomposition.dimensions,
            research=research,
            questions=question_set.questions,
            decomposition=decomposition,
            _question_set=question_set,
        )

    def answer(
        self,
        state: InterrogationState,
        question_text: str,
        answer: str,
    ) -> InterrogationState:
        """Record a user's answer to a question.

        Args:
            state: The current interrogation state.
            question_text: The question being answered (must match Question.text).
            answer: The user's answer text.

        Returns:
            The updated state (same object, mutated in place).
        """
        state.answers[question_text] = answer
        return state

    async def crystallize(self, state: InterrogationState) -> InterrogationResult:
        """Crystallize the interrogation state into a structured spec.

        Args:
            state: The interrogation state with answers filled in.

        Returns:
            InterrogationResult containing the spec and final state.
        """
        question_set = state._question_set or QuestionSet(questions=state.questions)
        decomposition = state.decomposition or DecompositionResult(
            original_prompt=state.prompt,
            dimensions=state.dimensions,
            overall_vagueness=0.5,
        )

        spec = self._crystallizer.crystallize(
            decomposition=decomposition,
            questions=question_set,
            user_answers=state.answers,
            research=state.research,
        )

        return InterrogationResult(spec=spec, state=state)
