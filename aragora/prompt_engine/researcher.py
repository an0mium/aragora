"""Prompt Researcher - Investigates context around a PromptIntent.

Queries Knowledge Mound, codebase structure, and optionally web sources
to build a ResearchReport with evidence, current state analysis, and
recommendations.

Usage:
    researcher = PromptResearcher(knowledge_mound=km)
    report = await researcher.research(intent)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from aragora.prompt_engine.types import (
    ClarifyingQuestion,
    EvidenceLink,
    PromptIntent,
    ResearchReport,
)

logger = logging.getLogger(__name__)

_RESEARCH_PROMPT = """\
You are a senior technical researcher. Analyze the user's intent and produce
a research report covering:

1. **current_state**: What exists now relevant to this intent
2. **related_decisions**: Past decisions that affect this work (as list of dicts with title/summary/relevance)
3. **competitive_analysis**: How others solve this (if applicable)
4. **recommendations**: Ordered list of actionable recommendations

Intent summary: {summary}
Intent type: {intent_type}
Domains: {domains}
Scope: {scope}

{clarifications}

{knowledge_context}

Respond with valid JSON only. No markdown formatting.
"""


class PromptResearcher:
    """Researches context around a PromptIntent."""

    def __init__(
        self,
        agent: Any | None = None,
        knowledge_mound: Any | None = None,
    ) -> None:
        self._agent = agent
        self._km = knowledge_mound

    async def _get_agent(self) -> Any:
        """Lazy-load the default agent."""
        if self._agent is not None:
            return self._agent

        try:
            from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

            self._agent = AnthropicAPIAgent(
                name="researcher",
                model="claude-sonnet-4-6",
                thinking_budget=8000,
            )
            return self._agent
        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning("Could not create default agent: %s", e)
            raise RuntimeError("No agent available for research") from e

    async def research(
        self,
        intent: PromptIntent,
        answered_questions: list[ClarifyingQuestion] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ResearchReport:
        """Research context for a prompt intent.

        Args:
            intent: The decomposed prompt intent
            answered_questions: Questions that have been answered
            context: Optional additional context

        Returns:
            ResearchReport with findings
        """
        agent = await self._get_agent()

        clarification_text = ""
        if answered_questions:
            lines = []
            for q in answered_questions:
                if q.is_answered:
                    lines.append(f"Q: {q.question}\nA: {q.answer}")
            if lines:
                clarification_text = "Clarifications received:\n" + "\n\n".join(lines)

        km_context = await self._get_km_context(intent)
        knowledge_text = (
            f"Relevant knowledge from Knowledge Mound:\n{km_context}"
            if km_context
            else "No prior knowledge found."
        )

        prompt = _RESEARCH_PROMPT.format(
            summary=intent.summary,
            intent_type=intent.intent_type.value,
            domains=", ".join(intent.domains),
            scope=intent.scope_estimate.value,
            clarifications=clarification_text,
            knowledge_context=knowledge_text,
        )

        if context:
            prompt += f"\n\nAdditional context:\n{json.dumps(context, indent=2)}"

        response = await agent.generate(prompt)
        return self._parse_report(response, km_context)

    async def _get_km_context(self, intent: PromptIntent) -> str:
        """Query Knowledge Mound for relevant context."""
        if self._km is None:
            return ""

        try:
            results = await self._km.query(
                query=intent.summary,
                limit=10,
            )
            if not isinstance(results, list) or not results:
                return ""

            lines = []
            for item in results[:10]:
                title = item.get("title", item.get("document_id", "Unknown"))
                content = item.get("content", "")[:300]
                source = item.get("metadata", {}).get("source", "km")
                lines.append(f"- [{source}] {title}: {content}")
            return "\n".join(lines)
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.debug("KM research query failed: %s", e)
            return ""

    def _parse_report(self, response: str, km_context: str) -> ResearchReport:
        """Parse LLM response into a ResearchReport."""
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
                    logger.warning("Could not parse research response")
                    return self._fallback_report(text)
            else:
                return self._fallback_report(text)

        try:
            evidence = []
            if km_context:
                evidence.append(
                    EvidenceLink(
                        source="km",
                        title="Knowledge Mound context",
                        snippet=km_context[:500],
                    )
                )

            for ev in data.get("evidence", []):
                evidence.append(
                    EvidenceLink(
                        source=ev.get("source", "unknown"),
                        title=ev.get("title", ""),
                        url=ev.get("url"),
                        relevance=float(ev.get("relevance", 1.0)),
                        snippet=ev.get("snippet", ""),
                    )
                )

            return ResearchReport(
                summary=data.get("summary", "Research complete"),
                current_state=data.get("current_state", ""),
                related_decisions=data.get("related_decisions", []),
                evidence=evidence,
                competitive_analysis=data.get("competitive_analysis", ""),
                recommendations=data.get("recommendations", []),
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Error building ResearchReport: %s", e)
            return self._fallback_report(text)

    @staticmethod
    def _fallback_report(raw_text: str) -> ResearchReport:
        """Create a minimal report when parsing fails."""
        return ResearchReport(
            summary="Research completed (unstructured)",
            current_state=raw_text[:1000],
        )
