"""EssayRefinementPipeline -- end-to-end multi-model essay orchestration."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from aragora.agents.base import create_agent
from aragora.essay.prompts import (
    build_drafting_prompt,
    build_extraction_prompt,
    build_polish_prompt,
)
from aragora.essay.rubric import EssayScore, evaluate_essay, load_rubric
from aragora.essay.synthesizer import EssaySynthesizer

logger = logging.getLogger(__name__)


@dataclass
class EssayRefinementPipeline:
    """Orchestrate extraction, parallel drafting, evaluation, synthesis, and polish.

    Parameters
    ----------
    models:
        Agent types used for parallel drafting (one draft per model).
    target_words:
        Approximate word count for the final essay.
    max_rounds:
        Maximum refinement iterations.
    quality_threshold:
        Overall score (0-1) at which refinement stops early.
    voice_notes:
        Optional stylistic guidance forwarded to drafting / synthesis prompts.
    rubric_path:
        Path to a YAML rubric file.  ``None`` uses the built-in default.
    """

    models: list[str] = field(default_factory=lambda: ["anthropic-api", "openai-api", "gemini"])
    target_words: int = 1200
    max_rounds: int = 3
    quality_threshold: float = 0.8
    voice_notes: str = ""
    rubric_path: str | None = None

    # ── Public API ────────────────────────────────────────────────────────

    async def run(
        self,
        raw_ideas: str,
        *,
        dry_run: bool = False,
        max_rounds: int | None = None,
    ) -> dict[str, Any]:
        """Execute the full essay refinement pipeline.

        Parameters
        ----------
        raw_ideas:
            Unstructured notes or brainstorm text.
        dry_run:
            When ``True``, stop after extraction (no drafting/scoring).
        max_rounds:
            Override ``self.max_rounds`` for this invocation.

        Returns
        -------
        dict
            Keys depend on *dry_run*:
            - dry_run: ``thesis``, ``outline``, ``raw_extraction``
            - full: ``final_essay``, ``final_score``, ``thesis``, ``outline``,
              ``rounds_used``, ``critique_history``
        """
        rounds = max_rounds if max_rounds is not None else self.max_rounds
        rubric = load_rubric(self.rubric_path)

        # Phase 1 -- extraction
        extraction = await self._extract_ideas(raw_ideas)
        thesis = extraction["thesis"]
        outline = extraction["outline"]

        if dry_run:
            return extraction

        # Phase 2 -- parallel drafting
        drafts = await self._parallel_draft(thesis, outline)

        # Phase 3+4 -- evaluate / synthesize loop
        critique_history: list[str] = []
        current_draft = ""
        rounds_used = 1

        for round_idx in range(1, rounds + 1):
            if round_idx == 1:
                # First iteration: evaluate all drafts, synthesize best
                scores, critiques = await self._evaluate_drafts(drafts, rubric)
                critique_history.extend(critiques)
                current_draft = await self._synthesize(drafts, scores, critiques)
            else:
                # Subsequent: evaluate current draft, check threshold
                scores, critiques = await self._evaluate_drafts([current_draft], rubric)
                critique_history.extend(critiques)

                if scores and scores[0].overall >= self.quality_threshold:
                    break

                current_draft = await self._synthesize([current_draft], scores, critiques)

            rounds_used = round_idx

        # Phase 5 -- polish
        final_essay = await self._polish(current_draft)

        # Phase 6 -- final score (different model to avoid self-evaluation bias)
        final_score = await self._final_score(final_essay, rubric)

        return {
            "final_essay": final_essay,
            "final_score": final_score,
            "thesis": thesis,
            "outline": outline,
            "rounds_used": rounds_used,
            "critique_history": critique_history,
        }

    # ── Private helpers ───────────────────────────────────────────────────

    async def _extract_ideas(self, raw_ideas: str) -> dict[str, Any]:
        """Use an LLM to extract a thesis and outline from *raw_ideas*."""
        agent = create_agent(self.models[0], name="extractor", role="proposer")
        prompt = build_extraction_prompt(raw_ideas, target_words=self.target_words)
        response = await agent.generate(prompt)

        thesis = ""
        outline = ""

        # Parse "THESIS:" line
        thesis_match = re.search(r"(?i)\*?\*?thesis\*?\*?[:\s]+(.+)", response)
        if thesis_match:
            thesis = thesis_match.group(1).strip()

        # Parse outline: everything after "OUTLINE:" header
        outline_match = re.search(r"(?i)\*?\*?outline\*?\*?[:\s]*\n([\s\S]+)", response)
        if outline_match:
            outline = outline_match.group(1).strip()

        return {
            "thesis": thesis,
            "outline": outline,
            "raw_extraction": response,
        }

    async def _parallel_draft(self, thesis: str, outline: str) -> list[str]:
        """Create one draft per model in parallel."""

        async def _draft_one(model_type: str, idx: int) -> str:
            agent = create_agent(model_type, name=f"drafter-{idx}", role="proposer")
            prompt = build_drafting_prompt(
                thesis,
                outline,
                target_words=self.target_words,
                voice_notes=self.voice_notes,
            )
            return await agent.generate(prompt)

        tasks = [_draft_one(m, i) for i, m in enumerate(self.models)]
        return list(await asyncio.gather(*tasks))

    async def _evaluate_drafts(
        self,
        drafts: list[str],
        rubric: dict[str, Any],
    ) -> tuple[list[EssayScore], list[str]]:
        """Evaluate each draft and return (scores, critiques)."""
        # Use a different model from the drafters for evaluation
        judge_model = self.models[-1] if len(self.models) > 1 else self.models[0]
        judge = create_agent(judge_model, name="judge", role="critic")

        scores: list[EssayScore] = []
        critiques: list[str] = []

        for draft in drafts:
            score = await evaluate_essay(draft, judge, rubric=rubric)
            scores.append(score)
            # Build critique string from score feedback
            parts: list[str] = []
            if score.severity_notes:
                parts.append("Issues: " + "; ".join(score.severity_notes))
            if score.suggestions:
                parts.append("Suggestions: " + "; ".join(score.suggestions))
            if score.weakest_paragraph:
                parts.append(f"Weakest paragraph: {score.weakest_paragraph}")
            critiques.append(" | ".join(parts) if parts else "No specific critique.")

        return scores, critiques

    async def _synthesize(
        self,
        drafts: list[str],
        scores: list[EssayScore],
        critiques: list[str],
    ) -> str:
        """Merge drafts into a single improved essay via EssaySynthesizer."""
        synth_model = self.models[0]
        agent = create_agent(synth_model, name="synthesizer", role="synthesizer")
        synthesizer = EssaySynthesizer(agent)
        return await synthesizer.synthesize(
            drafts,
            scores,
            critiques,
            target_words=self.target_words,
            voice_notes=self.voice_notes,
        )

    async def _polish(self, draft: str) -> str:
        """Final style polish pass."""
        agent = create_agent(self.models[0], name="polisher", role="proposer")
        prompt = build_polish_prompt(
            draft,
            target_words=self.target_words,
            voice_notes=self.voice_notes,
        )
        return await agent.generate(prompt)

    async def _final_score(
        self,
        essay: str,
        rubric: dict[str, Any],
    ) -> EssayScore:
        """Score the final essay using a DIFFERENT model to avoid self-eval bias."""
        # Pick the last model (different from synthesizer which uses models[0])
        judge_model = self.models[-1] if len(self.models) > 1 else self.models[0]
        judge = create_agent(judge_model, name="final-judge", role="critic")
        return await evaluate_essay(essay, judge, rubric=rubric)
