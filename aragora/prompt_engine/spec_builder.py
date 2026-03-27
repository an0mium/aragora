"""Spec Builder - Synthesizes intent, answers, and research into a Specification.

Takes all gathered information and produces a formal implementation
specification with file changes, risks, success criteria, and provenance.

Usage:
    builder = SpecBuilder()
    spec = await builder.build(intent, questions, research)
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.prompt_engine.decomposer import PromptDecomposer
from aragora.prompt_engine.llm_utils import (
    append_json_context,
    format_answered_questions,
    parse_json_mapping,
    repair_truncated_json,
)
from aragora.prompt_engine.timing import OperationTiming, append_timing, format_timings, start_timer
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
        self._last_operation_timings: list[OperationTiming] = []

    @property
    def last_operation_timings(self) -> list[OperationTiming]:
        """Timing records from the most recent specification build."""
        return list(self._last_operation_timings)

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
        timings: list[OperationTiming] = []

        timer = start_timer()
        agent = await self._get_agent()
        append_timing(timings, "specify.get_agent", timer, category="setup")

        clarification_text = format_answered_questions(
            answered_questions,
            header="Clarifications:",
        )

        research_text = ""
        if research:
            current = str(research.current_state)[:500] if research.current_state else ""
            recs = ", ".join(str(r) for r in (research.recommendations or [])[:5])
            competitive = (
                str(research.competitive_analysis)[:300] if research.competitive_analysis else ""
            )
            research_text = (
                f"Research findings:\n"
                f"Current state: {current}\n"
                f"Recommendations: {recs}\n"
                f"Competitive: {competitive}"
            )

        prompt = _SPEC_PROMPT.format(
            summary=intent.summary,
            intent_type=intent.intent_type.value,
            domains=", ".join(intent.domains),
            scope=intent.scope_estimate.value,
            clarifications=clarification_text,
            research_summary=research_text,
        )

        prompt = append_json_context(prompt, context)

        timer = start_timer()
        response = await agent.generate(prompt)
        append_timing(
            timings,
            "specify.agent_generate",
            timer,
            category="llm",
            prompt_chars=len(prompt),
            response_chars=len(response),
        )
        timer = start_timer()
        spec = self._parse_spec(response)
        append_timing(
            timings,
            "specify.parse_spec",
            timer,
            category="compute",
            file_change_count=len(spec.file_changes),
            risk_count=len(spec.risks),
        )

        # Attach provenance chain
        spec.provenance = SpecProvenance(
            original_prompt=intent.raw_prompt,
            intent=intent,
            questions_asked=answered_questions or [],
            research=research,
            prompt_hash=PromptDecomposer.prompt_hash(intent.raw_prompt),
        )

        self._last_operation_timings = timings
        logger.debug("SpecBuilder timings: %s", format_timings(timings))

        return spec

    def _parse_spec(self, response: str) -> Specification:
        """Parse LLM response into a Specification."""
        text = response.strip()

        data = self._extract_json(text)
        if data is None:
            return self._fallback_spec(text)

        try:
            file_changes = []
            for f in data.get("file_changes", []):
                if isinstance(f, str):
                    file_changes.append(SpecFile(path=f, action="modify"))
                elif isinstance(f, dict):
                    file_changes.append(
                        SpecFile(
                            path=f.get("path", ""),
                            action=f.get("action", "modify"),
                            description=f.get("description", ""),
                            estimated_lines=int(f.get("estimated_lines", 0)),
                        )
                    )

            risks = []
            for r in data.get("risks", []):
                if isinstance(r, str):
                    risks.append(SpecRisk(description=r))
                elif isinstance(r, dict):
                    risks.append(
                        SpecRisk(
                            description=r.get("description", ""),
                            likelihood=r.get("likelihood", "medium"),
                            impact=r.get("impact", "medium"),
                            mitigation=r.get("mitigation", ""),
                        )
                    )

            success_criteria = []
            for s in data.get("success_criteria", []):
                if isinstance(s, str):
                    success_criteria.append(SuccessCriterion(description=s))
                elif isinstance(s, dict):
                    success_criteria.append(
                        SuccessCriterion(
                            description=s.get("description", ""),
                            measurement=s.get("measurement", ""),
                            target=s.get("target", ""),
                        )
                    )

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
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Try multiple strategies to extract JSON from LLM response."""
        return parse_json_mapping(
            text,
            unwrap_single_nested=True,
            repair_truncated=True,
        )

    @staticmethod
    def _unwrap_nested(data: dict[str, Any]) -> dict[str, Any]:
        """Unwrap LLM responses that nest the spec inside a wrapper key."""
        if not isinstance(data, dict):
            return data
        # If the dict has a single key containing a nested dict with spec fields, unwrap
        if len(data) == 1:
            key = next(iter(data))
            inner = data[key]
            if isinstance(inner, dict) and ("title" in inner or "problem_statement" in inner):
                return inner
        return data

    @staticmethod
    def _repair_truncated_json(text: str) -> dict[str, Any] | None:
        """Attempt to repair truncated JSON by closing open structures."""
        return repair_truncated_json(text)

    @staticmethod
    def _fallback_spec(raw_text: str) -> Specification:
        """Create a minimal spec when parsing fails."""
        return Specification(
            title="Specification (requires manual review)",
            problem_statement=raw_text[:500],
            proposed_solution="Manual specification required",
            confidence=0.1,
        )
