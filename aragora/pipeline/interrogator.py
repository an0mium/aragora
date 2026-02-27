"""Pipeline interrogator — gather clarifying questions before spec generation.

Wraps ``SwarmInterrogator`` with pipeline-specific context: parsed ideas,
detected themes, and autonomy-aware question depth.  At higher autonomy
levels fewer questions are asked (or none at all).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PipelineInterrogator:
    """Gather clarifying information via conversational Q&A.

    Uses ``SwarmInterrogator`` under the hood but enriches the context with
    previously-parsed ideas and themes so the LLM asks smarter questions.

    Parameters
    ----------
    max_turns:
        Maximum number of Q&A turns before synthesising a spec.
    model:
        LLM model to use for interrogation.
    """

    def __init__(
        self,
        max_turns: int = 5,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self.max_turns = max_turns
        self.model = model

    async def interrogate(
        self,
        initial_goal: str,
        ideas: list[str] | None = None,
        input_fn: Any | None = None,
        print_fn: Any | None = None,
    ) -> Any:
        """Run the interrogation loop and return a ``SwarmSpec``.

        Parameters
        ----------
        initial_goal:
            The user's raw prompt / goal string.
        ideas:
            Pre-parsed ideas from ``BrainDumpParser`` (used to enrich context).
        input_fn:
            Callable ``(prompt: str) -> str`` for getting user answers.
            For web UIs this should be an async bridge to the frontend.
        print_fn:
            Callable ``(text: str) -> None`` for displaying questions.
        """
        # Build enriched context for the interrogator
        context_parts = [initial_goal]
        if ideas:
            context_parts.append(
                f"\n\nPre-parsed ideas ({len(ideas)}):\n"
                + "\n".join(f"- {idea}" for idea in ideas[:20])
            )

        enriched_goal = "\n".join(context_parts)

        try:
            from aragora.swarm.interrogator import SwarmInterrogator
            from aragora.swarm.config import InterrogatorConfig

            config = InterrogatorConfig(
                max_turns=self.max_turns,
                model=self.model,
            )
            interrogator = SwarmInterrogator(config=config)
            spec = await interrogator.interrogate(
                initial_goal=enriched_goal,
                input_fn=input_fn,
                print_fn=print_fn,
            )
            return spec
        except ImportError:
            logger.warning("SwarmInterrogator not available, using fallback")
            return await self._fallback_interrogation(initial_goal, ideas, input_fn, print_fn)
        except Exception:
            logger.exception("Interrogation failed, using fallback")
            return await self._fallback_interrogation(initial_goal, ideas, input_fn, print_fn)

    async def _fallback_interrogation(
        self,
        goal: str,
        ideas: list[str] | None,
        input_fn: Any | None,
        print_fn: Any | None,
    ) -> Any:
        """Minimal fallback when SwarmInterrogator is unavailable."""
        from aragora.swarm.spec import SwarmSpec

        spec = SwarmSpec(
            raw_goal=goal,
            refined_goal=goal,
            acceptance_criteria=[],
            constraints=[],
            interrogation_turns=0,
        )

        if input_fn is None or print_fn is None:
            # Non-interactive mode — return spec as-is
            return spec

        questions = [
            "What specific outcome would tell you this succeeded?",
            "Are there any constraints or things that should NOT change?",
            "Who is the primary audience or user for this?",
        ]

        answers = []
        for q in questions:
            print_fn(q)
            answer = input_fn("> ")
            if answer and answer.strip():
                answers.append(answer.strip())

        if answers:
            spec.acceptance_criteria = answers[:1]
            spec.constraints = answers[1:2]
            spec.interrogation_turns = len(answers)

        return spec
