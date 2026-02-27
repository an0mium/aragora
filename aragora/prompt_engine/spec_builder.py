"""Spec Builder - Synthesizes intent, answers, and research into a Specification.

Takes all gathered information and produces a formal implementation
specification with file changes, risks, success criteria, and provenance.

Usage:
    builder = SpecBuilder()
    spec = await builder.build(intent, questions, research)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from aragora.prompt_engine.decomposer import PromptDecomposer
from aragora.prompt_engine.types import (
    ClarifyingQuestion,
    PromptIntent,
    ResearchReport,
    SpecFile,
    SpecProvenance,
    SpecRisk,
    Specification,
    SuccessCriterion,
)

logger = logging.getLogger(__name__)

_SPEC_PROMPT = """\
You are a senior software architect. Produce a detailed implementation
specification from the gathered information.

The specification must include:
1. **title**: Short descriptive title
2. **problem_statement**: What problem this solves
3. **proposed_solution**: How to solve it
4. **alternatives_considered**: Other approaches and why they were rejected
5. **file_changes**: List of files to create/modify/delete with descriptions
6. **dependencies**: External dependencies needed
7. **risks**: Technical and business risks with likelihood, impact, and mitigation
8. **success_criteria**: Measurable criteria for success
9. **estimated_effort**: small/medium/large/epic
10. **confidence**: 0-1, how confident you are in this specification

Intent: {summary}
Type: {intent_type}
Domains: {domains}
Scope: {scope}

{clarifications}

{research_summary}

Respond with valid JSON only. No markdown formatting.
"""


class SpecBuilder:
    """Builds a Specification from gathered information."""

    def __init__(self, agent: Any | None = None) -> None:
        self._agent = agent

    async def _get_agent(self) -> Any:
        """Lazy-load the default agent."""
        if self._agent is not None:
            return self._agent

        try:
            from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

            self._agent = AnthropicAPIAgent(
                name="spec_builder",
                model="claude-sonnet-4-6",
                thinking_budget=12000,
            )
            return self._agent
        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning("Could not create default agent: %s", e)
            raise RuntimeError("No agent available for spec building") from e

    async def build(
        self,
        intent: PromptIntent,
        answered_questions: list[ClarifyingQuestion] | None = None,
        research: ResearchReport | None = None,
        context: dict[str, Any] | None = None,
    ) -> Specification:
        """Build a specification from gathered information.

        Args:
            intent: The decomposed prompt intent
            answered_questions: Answered clarifying questions
            research: Research findings
            context: Optional additional context

        Returns:
            Formal Specification
        """
        agent = await self._get_agent()

        clarification_text = ""
        if answered_questions:
            lines = []
            for q in answered_questions:
                if q.is_answered:
                    lines.append(f"Q: {q.question}\nA: {q.answer}")
            clarification_text = "Clarifications:\n" + "\n\n".join(lines) if lines else ""

        research_text = ""
        if research:
            research_text = (
                f"Research findings:\n"
                f"Current state: {research.current_state[:500]}\n"
                f"Recommendations: {', '.join(research.recommendations[:5])}\n"
                f"Competitive: {research.competitive_analysis[:300]}"
            )

        prompt = _SPEC_PROMPT.format(
            summary=intent.summary,
            intent_type=intent.intent_type.value,
            domains=", ".join(intent.domains),
            scope=intent.scope_estimate.value,
            clarifications=clarification_text,
            research_summary=research_text,
        )

        if context:
            prompt += f"\n\nAdditional context:\n{json.dumps(context, indent=2)}"

        response = await agent.generate(prompt)
        spec = self._parse_spec(response)

        # Attach provenance chain
        spec.provenance = SpecProvenance(
            original_prompt=intent.raw_prompt,
            intent=intent,
            questions_asked=answered_questions or [],
            research=research,
            prompt_hash=PromptDecomposer.prompt_hash(intent.raw_prompt),
        )

        return spec

    def _parse_spec(self, response: str) -> Specification:
        """Parse LLM response into a Specification."""
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
                    logger.warning("Could not parse spec response")
                    return self._fallback_spec(text)
            else:
                return self._fallback_spec(text)

        try:
            file_changes = [
                SpecFile(
                    path=f.get("path", ""),
                    action=f.get("action", "modify"),
                    description=f.get("description", ""),
                    estimated_lines=int(f.get("estimated_lines", 0)),
                )
                for f in data.get("file_changes", [])
            ]

            risks = [
                SpecRisk(
                    description=r.get("description", ""),
                    likelihood=r.get("likelihood", "medium"),
                    impact=r.get("impact", "medium"),
                    mitigation=r.get("mitigation", ""),
                )
                for r in data.get("risks", [])
            ]

            success_criteria = [
                SuccessCriterion(
                    description=s.get("description", ""),
                    measurement=s.get("measurement", ""),
                    target=s.get("target", ""),
                )
                for s in data.get("success_criteria", [])
            ]

            return Specification(
                title=data.get("title", "Untitled Specification"),
                problem_statement=data.get("problem_statement", ""),
                proposed_solution=data.get("proposed_solution", ""),
                alternatives_considered=data.get("alternatives_considered", []),
                file_changes=file_changes,
                dependencies=data.get("dependencies", []),
                risks=risks,
                success_criteria=success_criteria,
                estimated_effort=data.get("estimated_effort", "medium"),
                confidence=float(data.get("confidence", 0.5)),
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Error building Specification: %s", e)
            return self._fallback_spec(text)

    @staticmethod
    def _fallback_spec(raw_text: str) -> Specification:
        """Create a minimal spec when parsing fails."""
        return Specification(
            title="Specification (requires manual review)",
            problem_statement=raw_text[:500],
            proposed_solution="Manual specification required",
            confidence=0.1,
        )
