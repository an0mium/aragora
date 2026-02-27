"""Prompt Decomposer - Transforms vague user prompts into structured intents.

Takes any natural language input and produces a PromptIntent with classified
intent type, affected domains, detected ambiguities, and implicit assumptions.

Usage:
    decomposer = PromptDecomposer()
    intent = await decomposer.decompose("I want to improve the onboarding flow")
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from aragora.prompt_engine.types import (
    Ambiguity,
    Assumption,
    IntentType,
    PromptIntent,
    ScopeEstimate,
)

logger = logging.getLogger(__name__)

_DECOMPOSITION_PROMPT = """\
You are an expert product analyst. Your job is to decompose a vague user prompt
into a structured intent. Analyze the prompt carefully and extract:

1. **intent_type**: One of: feature, improvement, investigation, fix, strategic
2. **summary**: A single clear sentence describing what the user wants
3. **domains**: List of affected areas (e.g., "frontend", "auth", "database", "api")
4. **ambiguities**: Things that need clarification before proceeding. Each has:
   - description: What's unclear
   - impact: What changes depending on the answer
   - options: Possible interpretations (2-4)
   - recommended: Your best guess (or null if truly ambiguous)
5. **assumptions**: Implicit assumptions in the prompt. Each has:
   - description: The assumption
   - confidence: 0-1 how likely it's correct
   - alternative: What if it's wrong
6. **scope_estimate**: One of: small, medium, large, epic

Respond with valid JSON only. No markdown formatting.
"""


class PromptDecomposer:
    """Decomposes vague user prompts into structured PromptIntents."""

    def __init__(
        self,
        agent: Any | None = None,
        knowledge_mound: Any | None = None,
    ) -> None:
        self._agent = agent
        self._km = knowledge_mound
        self._km_results: list[dict[str, Any]] = []

    async def _get_agent(self) -> Any:
        """Lazy-load the default agent."""
        if self._agent is not None:
            return self._agent

        try:
            from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

            self._agent = AnthropicAPIAgent(
                name="decomposer",
                model="claude-sonnet-4-6",
                thinking_budget=8000,
            )
            return self._agent
        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning("Could not create default agent: %s", e)
            raise RuntimeError("No agent available for prompt decomposition") from e

    async def decompose(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> PromptIntent:
        """Decompose a vague prompt into a structured intent.

        Args:
            prompt: Raw user input (any length, any vagueness)
            context: Optional additional context

        Returns:
            Structured PromptIntent with classifications and ambiguities
        """
        agent = await self._get_agent()

        user_prompt = f"User prompt:\n{prompt}"
        if context:
            user_prompt += f"\n\nAdditional context:\n{json.dumps(context, indent=2)}"

        km_context = await self._get_km_context(prompt)
        if km_context:
            user_prompt += f"\n\nRelevant knowledge:\n{km_context}"

        full_prompt = f"{_DECOMPOSITION_PROMPT}\n\n{user_prompt}"
        response = await agent.generate(full_prompt)
        parsed = self._parse_response(response, prompt)

        if self._km_results:
            parsed.related_knowledge = self._km_results

        return parsed

    async def _get_km_context(self, prompt: str) -> str:
        """Query Knowledge Mound for context relevant to the prompt."""
        if self._km is None:
            return ""

        try:
            results = await self._km.query(query=prompt, limit=5)
            self._km_results = results if isinstance(results, list) else []
            if not self._km_results:
                return ""

            lines = []
            for item in self._km_results[:5]:
                title = item.get("title", item.get("document_id", "Unknown"))
                content = item.get("content", "")[:200]
                lines.append(f"- {title}: {content}")
            return "\n".join(lines)
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.debug("KM query failed: %s", e)
            self._km_results = []
            return ""

    def _parse_response(self, response: str, original_prompt: str) -> PromptIntent:
        """Parse the LLM response into a PromptIntent."""
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
                    logger.warning("Could not parse decomposition response")
                    return self._fallback_intent(original_prompt)
            else:
                return self._fallback_intent(original_prompt)

        try:
            ambiguities = [
                Ambiguity(
                    description=a.get("description", ""),
                    impact=a.get("impact", ""),
                    options=a.get("options", []),
                    recommended=a.get("recommended"),
                )
                for a in data.get("ambiguities", [])
            ]

            assumptions = [
                Assumption(
                    description=a.get("description", ""),
                    confidence=float(a.get("confidence", 0.5)),
                    alternative=a.get("alternative"),
                )
                for a in data.get("assumptions", [])
            ]

            intent_type_str = data.get("intent_type", "improvement")
            try:
                intent_type = IntentType(intent_type_str)
            except ValueError:
                intent_type = IntentType.IMPROVEMENT

            scope_str = data.get("scope_estimate", "medium")
            try:
                scope = ScopeEstimate(scope_str)
            except ValueError:
                scope = ScopeEstimate.MEDIUM

            return PromptIntent(
                raw_prompt=original_prompt,
                intent_type=intent_type,
                summary=data.get("summary", original_prompt),
                domains=data.get("domains", []),
                ambiguities=ambiguities,
                assumptions=assumptions,
                scope_estimate=scope,
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Error building PromptIntent: %s", e)
            return self._fallback_intent(original_prompt)

    @staticmethod
    def _fallback_intent(prompt: str) -> PromptIntent:
        """Create a minimal intent when parsing fails."""
        return PromptIntent(
            raw_prompt=prompt,
            intent_type=IntentType.IMPROVEMENT,
            summary=prompt[:200],
            domains=["unknown"],
            ambiguities=[
                Ambiguity(
                    description="Could not parse prompt automatically",
                    impact="Manual clarification needed",
                    options=[],
                    recommended=None,
                )
            ],
            scope_estimate=ScopeEstimate.MEDIUM,
        )

    @staticmethod
    def prompt_hash(prompt: str) -> str:
        """Generate a stable hash for a prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
