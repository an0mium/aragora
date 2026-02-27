"""Prompt Interrogator - Generates clarifying questions from ambiguities.

Takes a PromptIntent and produces targeted ClarifyingQuestions that resolve
ambiguities before building a specification.

Usage:
    interrogator = PromptInterrogator()
    questions = await interrogator.interrogate(intent, depth="thorough")
"""

from __future__ import annotations

import json
import logging
from typing import Any

from aragora.prompt_engine.types import (
    ClarifyingQuestion,
    InterrogationDepth,
    PromptIntent,
    QuestionOption,
)

logger = logging.getLogger(__name__)

_DEPTH_LIMITS = {
    InterrogationDepth.QUICK: (3, 5),
    InterrogationDepth.THOROUGH: (10, 15),
    InterrogationDepth.EXHAUSTIVE: (20, 30),
}

_INTERROGATION_PROMPT = """\
You are a senior product analyst. Given a user's intent, generate clarifying
questions that MUST be answered before building an implementation specification.

Focus on questions that change the architecture, scope, or approach. Skip
trivial clarifications.

For each question provide:
- question: The question text
- why_it_matters: Why this affects the implementation
- options: 2-4 suggested answers, each with label, description, and tradeoffs
- default: Your recommended answer (or null if genuinely ambiguous)

Intent summary: {summary}
Intent type: {intent_type}
Domains: {domains}
Scope: {scope}

Known ambiguities:
{ambiguities}

Known assumptions:
{assumptions}

Generate {min_q} to {max_q} questions. Respond with valid JSON only:
{{"questions": [...]}}
"""


class PromptInterrogator:
    """Generates clarifying questions from a PromptIntent."""

    def __init__(self, agent: Any | None = None) -> None:
        self._agent = agent

    async def _get_agent(self) -> Any:
        """Lazy-load the default agent."""
        if self._agent is not None:
            return self._agent

        try:
            from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

            self._agent = AnthropicAPIAgent(
                name="interrogator",
                model="claude-sonnet-4-6",
                thinking_budget=4000,
            )
            return self._agent
        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning("Could not create default agent: %s", e)
            raise RuntimeError("No agent available for interrogation") from e

    async def interrogate(
        self,
        intent: PromptIntent,
        depth: InterrogationDepth | str = InterrogationDepth.THOROUGH,
        context: dict[str, Any] | None = None,
    ) -> list[ClarifyingQuestion]:
        """Generate clarifying questions for a prompt intent.

        Args:
            intent: The decomposed prompt intent
            depth: How many questions to generate
            context: Optional additional context

        Returns:
            List of ClarifyingQuestion objects
        """
        if isinstance(depth, str):
            try:
                depth = InterrogationDepth(depth)
            except ValueError:
                depth = InterrogationDepth.THOROUGH

        min_q, max_q = _DEPTH_LIMITS.get(depth, (10, 15))

        ambiguity_text = (
            "\n".join(f"- {a.description} (impact: {a.impact})" for a in intent.ambiguities)
            or "None detected"
        )

        assumption_text = (
            "\n".join(f"- {a.description} (confidence: {a.confidence})" for a in intent.assumptions)
            or "None detected"
        )

        prompt = _INTERROGATION_PROMPT.format(
            summary=intent.summary,
            intent_type=intent.intent_type.value,
            domains=", ".join(intent.domains),
            scope=intent.scope_estimate.value,
            ambiguities=ambiguity_text,
            assumptions=assumption_text,
            min_q=min_q,
            max_q=max_q,
        )

        if context:
            prompt += f"\n\nAdditional context:\n{json.dumps(context, indent=2)}"

        agent = await self._get_agent()
        response = await agent.generate(prompt)
        return self._parse_questions(response, intent)

    def _parse_questions(self, response: str, intent: PromptIntent) -> list[ClarifyingQuestion]:
        """Parse LLM response into ClarifyingQuestion objects."""
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    logger.warning("Could not parse interrogation response")
                    return self._fallback_questions(intent)
            else:
                return self._fallback_questions(intent)

        questions_data = data.get("questions", [])
        if not questions_data:
            return self._fallback_questions(intent)

        questions = []
        for i, q in enumerate(questions_data):
            try:
                options = [
                    QuestionOption(
                        label=opt.get("label", f"Option {j + 1}"),
                        description=opt.get("description", ""),
                        tradeoffs=opt.get("tradeoffs", ""),
                    )
                    for j, opt in enumerate(q.get("options", []))
                ]

                ambiguity_ref = None
                if i < len(intent.ambiguities):
                    ambiguity_ref = intent.ambiguities[i]

                questions.append(
                    ClarifyingQuestion(
                        question=q.get("question", ""),
                        why_it_matters=q.get("why_it_matters", ""),
                        options=options,
                        default=q.get("default"),
                        ambiguity_ref=ambiguity_ref,
                    )
                )
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                logger.debug("Skipping malformed question: %s", e)

        return questions or self._fallback_questions(intent)

    @staticmethod
    def _fallback_questions(intent: PromptIntent) -> list[ClarifyingQuestion]:
        """Generate minimal questions from ambiguities when parsing fails."""
        questions = []
        for amb in intent.ambiguities:
            options = [QuestionOption(label=opt, description=opt) for opt in amb.options[:4]]
            questions.append(
                ClarifyingQuestion(
                    question=amb.description,
                    why_it_matters=amb.impact,
                    options=options,
                    default=amb.recommended,
                    ambiguity_ref=amb,
                )
            )
        return questions or [
            ClarifyingQuestion(
                question="What is the primary goal of this request?",
                why_it_matters="Determines scope and approach",
                options=[
                    QuestionOption(
                        label="Quick fix",
                        description="Minimal change to address the immediate need",
                    ),
                    QuestionOption(
                        label="Comprehensive solution",
                        description="Full implementation with tests and documentation",
                    ),
                ],
            )
        ]
