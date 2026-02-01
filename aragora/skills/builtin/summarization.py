"""
Summarization Skill.

Provides text summarization capabilities using LLM inference.
Supports various summarization modes including extractive and abstractive summaries.
"""

from __future__ import annotations

import logging
from typing import Any

from ..base import (
    Skill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)


class SummarizationSkill(Skill):
    """
    Skill for summarizing long text content.

    Supports:
    - Abstractive summarization (generates new text)
    - Extractive summarization (selects key sentences)
    - Bullet point summaries
    - Custom length control
    - Multi-document summarization
    """

    def __init__(
        self,
        default_max_length: int = 150,
        default_style: str = "abstractive",
    ):
        """
        Initialize summarization skill.

        Args:
            default_max_length: Default maximum summary length in words
            default_style: Default summarization style (abstractive, extractive, bullets)
        """
        self._default_max_length = default_max_length
        self._default_style = default_style

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="summarization",
            version="1.0.0",
            description="Summarize long text content",
            capabilities=[
                SkillCapability.LLM_INFERENCE,
            ],
            input_schema={
                "text": {
                    "type": "string",
                    "description": "Text to summarize",
                    "required": True,
                },
                "max_length": {
                    "type": "number",
                    "description": "Maximum summary length in words",
                    "default": 150,
                },
                "style": {
                    "type": "string",
                    "description": "Summary style: abstractive, extractive, bullets, tldr",
                    "default": "abstractive",
                },
                "focus": {
                    "type": "string",
                    "description": "Optional focus area for the summary",
                },
                "preserve_key_points": {
                    "type": "boolean",
                    "description": "Whether to ensure key points are preserved",
                    "default": True,
                },
            },
            tags=["summarization", "nlp", "text"],
            debate_compatible=True,
            max_execution_time_seconds=60.0,
            rate_limit_per_minute=30,
        )

    async def execute(
        self,
        input_data: dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """Execute text summarization."""
        text = input_data.get("text", "")
        if not text:
            return SkillResult.create_failure(
                "Text is required",
                error_code="missing_text",
            )

        # Validate text length
        if len(text) < 50:
            return SkillResult.create_failure(
                "Text is too short to summarize (minimum 50 characters)",
                error_code="text_too_short",
            )

        max_length = input_data.get("max_length", self._default_max_length)
        style = input_data.get("style", self._default_style)
        focus = input_data.get("focus")
        preserve_key_points = input_data.get("preserve_key_points", True)

        try:
            if style == "extractive":
                summary = await self._extractive_summarize(text, max_length)
            elif style == "bullets":
                summary = await self._bullet_summarize(text, max_length, focus)
            elif style == "tldr":
                summary = await self._tldr_summarize(text)
            else:  # abstractive (default)
                summary = await self._abstractive_summarize(
                    text, max_length, focus, preserve_key_points
                )

            # Calculate compression ratio
            original_words = len(text.split())
            summary_words = len(summary.split())
            compression_ratio = (
                round(1 - (summary_words / original_words), 2) if original_words > 0 else 0
            )

            return SkillResult.create_success(
                {
                    "summary": summary,
                    "style": style,
                    "original_length": len(text),
                    "summary_length": len(summary),
                    "original_words": original_words,
                    "summary_words": summary_words,
                    "compression_ratio": compression_ratio,
                },
                style=style,
            )

        except Exception as e:
            logger.exception(f"Summarization failed: {e}")
            return SkillResult.create_failure(f"Summarization failed: {e}")

    async def _abstractive_summarize(
        self,
        text: str,
        max_length: int,
        focus: str | None,
        preserve_key_points: bool,
    ) -> str:
        """Generate an abstractive summary using LLM."""
        # Try to use LLM for abstractive summarization
        llm = await self._get_llm()
        if llm:
            prompt = self._build_abstractive_prompt(text, max_length, focus, preserve_key_points)
            return await self._call_llm(llm, prompt)

        # Fallback to extractive if no LLM available
        logger.info("No LLM available, falling back to extractive summarization")
        return await self._extractive_summarize(text, max_length)

    async def _extractive_summarize(
        self,
        text: str,
        max_length: int,
    ) -> str:
        """Generate an extractive summary by selecting key sentences."""
        import re

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if not sentences:
            return text[:500]

        # Score sentences based on position, length, and keyword density
        scored_sentences = []
        total_sentences = len(sentences)

        for i, sentence in enumerate(sentences):
            score = 0.0

            # Position score (first and last sentences are important)
            if i == 0:
                score += 2.0
            elif i == total_sentences - 1:
                score += 1.0
            elif i < total_sentences * 0.2:
                score += 1.5

            # Length score (prefer medium-length sentences)
            words = len(sentence.split())
            if 10 <= words <= 30:
                score += 1.0
            elif words > 30:
                score += 0.5

            # Keyword density (simple heuristic)
            important_words = [
                "important",
                "key",
                "main",
                "significant",
                "conclusion",
                "result",
                "therefore",
                "however",
                "because",
                "first",
                "second",
                "finally",
            ]
            for word in important_words:
                if word.lower() in sentence.lower():
                    score += 0.3

            scored_sentences.append((score, i, sentence))

        # Sort by score and select top sentences
        scored_sentences.sort(reverse=True)

        # Select sentences until we reach max_length words
        selected = []
        current_words = 0
        for score, idx, sentence in scored_sentences:
            sentence_words = len(sentence.split())
            if current_words + sentence_words <= max_length:
                selected.append((idx, sentence))
                current_words += sentence_words

        # Sort by original position for coherence
        selected.sort(key=lambda x: x[0])

        return " ".join(sentence for _, sentence in selected)

    async def _bullet_summarize(
        self,
        text: str,
        max_length: int,
        focus: str | None,
    ) -> str:
        """Generate a bullet-point summary."""
        llm = await self._get_llm()
        if llm:
            prompt = f"""Summarize the following text as bullet points (maximum {max_length} words total).
{f"Focus on: {focus}" if focus else ""}

Text:
{text}

Provide 3-7 bullet points capturing the key information:"""
            result = await self._call_llm(llm, prompt)
            return result

        # Fallback: extract key sentences as bullets
        extractive = await self._extractive_summarize(text, max_length)
        import re

        sentences = re.split(r"(?<=[.!?])\s+", extractive)
        bullets = [f"- {s.strip()}" for s in sentences if s.strip()]
        return "\n".join(bullets[:7])

    async def _tldr_summarize(self, text: str) -> str:
        """Generate a very brief TL;DR summary."""
        llm = await self._get_llm()
        if llm:
            prompt = f"""Provide a TL;DR (one or two sentences maximum) for the following text:

{text}

TL;DR:"""
            return await self._call_llm(llm, prompt)

        # Fallback: first sentence + key conclusion
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        if sentences:
            return sentences[0]
        return text[:200]

    def _build_abstractive_prompt(
        self,
        text: str,
        max_length: int,
        focus: str | None,
        preserve_key_points: bool,
    ) -> str:
        """Build prompt for abstractive summarization."""
        prompt_parts = [
            f"Summarize the following text in approximately {max_length} words.",
        ]

        if focus:
            prompt_parts.append(f"Focus particularly on aspects related to: {focus}")

        if preserve_key_points:
            prompt_parts.append("Ensure all key points and important details are preserved.")

        prompt_parts.append(f"\nText to summarize:\n{text}\n\nSummary:")

        return " ".join(prompt_parts)

    async def _get_llm(self) -> Any | None:
        """Get an LLM instance for summarization."""
        try:
            # Try to get a lightweight LLM from the agent system
            from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

            return AnthropicAPIAgent(model="claude-3-haiku-20240307")
        except ImportError:
            pass

        try:
            from aragora.agents.api_agents.openai import OpenAIAPIAgent

            return OpenAIAPIAgent(model="gpt-4o-mini")
        except ImportError:
            pass

        return None

    async def _call_llm(self, llm: Any, prompt: str) -> str:
        """Call LLM for text generation."""
        try:
            if hasattr(llm, "generate"):
                response = await llm.generate(prompt, max_tokens=500)
                if hasattr(response, "text"):
                    return response.text
                return str(response)
            elif hasattr(llm, "complete"):
                response = await llm.complete(prompt, max_tokens=500)
                return str(response)
            else:
                # Fallback: try direct call
                response = await llm(prompt)
                return str(response)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            raise


# Skill instance for registration
SKILLS = [SummarizationSkill()]
