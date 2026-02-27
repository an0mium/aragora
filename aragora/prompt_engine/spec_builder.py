"""SpecBuilder: assembles RefinedIntent + ResearchReport into SwarmSpec."""

from __future__ import annotations

import logging
from typing import Any

from aragora.prompt_engine.types import (
    PROFILE_DEFAULTS,
    RefinedIntent,
    ResearchReport,
    UserProfile,
)

logger = logging.getLogger(__name__)


class SpecBuilder:
    """Takes RefinedIntent + ResearchReport and produces a SwarmSpec."""

    async def build(
        self,
        refined_intent: RefinedIntent,
        research: ResearchReport,
        profile: UserProfile,
    ) -> Any:
        from aragora.swarm.spec import SwarmSpec

        result = await self._call_llm(
            refined_intent=refined_intent,
            research=research,
            profile=profile,
        )

        profile_defaults = PROFILE_DEFAULTS[profile.value]
        spec = SwarmSpec(
            raw_goal=refined_intent.intent.raw_prompt,
            refined_goal=result.get("refined_goal", refined_intent.intent.raw_prompt),
            acceptance_criteria=result.get("acceptance_criteria", []),
            constraints=result.get("constraints", []),
            track_hints=result.get("track_hints", []),
            estimated_complexity=result.get("estimated_complexity", "medium"),
            requires_approval=profile_defaults["require_approval"],
            interrogation_turns=len(refined_intent.answers),
        )
        return spec

    async def _call_llm(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Subclass or mock _call_llm for LLM integration")
