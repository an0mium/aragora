"""
Synthesis generation for consensus phase.

This module extracts the mandatory final synthesis logic from ConsensusPhase,
providing:
- LLM-based synthesis generation (Opus 4.5 with Sonnet fallback)
- Proposal combination fallback
- Synthesis prompt building
- Export link generation
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from aragora.server.stream.arena_hooks import streaming_task_context

if TYPE_CHECKING:
    from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)


class SynthesisGenerator:
    """Generates final synthesis for debates.

    Ensures every debate ends with a clear, synthesized conclusion
    using Claude Opus 4.5 (with Sonnet fallback).

    Usage:
        generator = SynthesisGenerator(
            protocol=protocol,
            hooks=hooks,
            notify_spectator=notify_spectator,
        )

        success = await generator.generate_mandatory_synthesis(ctx)
    """

    def __init__(
        self,
        *,
        protocol: Any = None,
        hooks: Optional[dict] = None,
        notify_spectator: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Initialize the synthesis generator.

        Args:
            protocol: Debate protocol for configuration
            hooks: Event hooks dict
            notify_spectator: Spectator notification callback
        """
        self.protocol = protocol
        self.hooks = hooks or {}
        self._notify_spectator = notify_spectator

    async def generate_mandatory_synthesis(self, ctx: "DebateContext") -> bool:
        """Generate mandatory final synthesis using Claude Opus 4.5.

        This runs after consensus is determined (by any mode) to ensure
        every debate ends with a clear, synthesized conclusion.

        Args:
            ctx: The DebateContext with proposals and consensus result

        Returns:
            bool: True if synthesis was successfully generated and emitted
        """
        # Skip if no proposals to synthesize
        if not ctx.proposals:
            logger.warning("synthesis_skipped reason=no_proposals")
            return False

        logger.info("synthesis_generation_start")

        synthesis = None
        synthesis_source = "opus"

        # Try 1: Claude Opus 4.5
        try:
            from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

            # Create dedicated synthesizer (always Opus 4.5)
            synthesizer = AnthropicAPIAgent(
                name="synthesis-agent",
                model="claude-opus-4-5-20251101",
            )

            # Build synthesis prompt
            synthesis_prompt = self._build_synthesis_prompt(ctx)

            # Generate synthesis with timeout (60s to fit within phase budget)
            with streaming_task_context("synthesis-agent:opus_synthesis"):
                synthesis = await asyncio.wait_for(
                    synthesizer.generate(synthesis_prompt, ctx.context_messages),
                    timeout=60.0,
                )
            logger.info(f"synthesis_generated_opus chars={len(synthesis)}")

        except asyncio.TimeoutError:
            logger.warning("synthesis_opus_timeout timeout=60s, trying sonnet fallback")
            synthesis_source = "sonnet"
        except ImportError as e:
            logger.warning(f"synthesis_import_error: {e}, trying sonnet fallback")
            synthesis_source = "sonnet"
        except Exception as e:
            logger.warning(f"synthesis_opus_failed error={e}, trying sonnet fallback")
            synthesis_source = "sonnet"

        # Try 2: Claude Sonnet fallback
        if not synthesis:
            try:
                from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

                synthesizer = AnthropicAPIAgent(
                    name="synthesis-agent-fallback",
                    model="claude-sonnet-4-20250514",
                )
                synthesis_prompt = self._build_synthesis_prompt(ctx)
                with streaming_task_context("synthesis-agent-fallback:sonnet_synthesis"):
                    synthesis = await asyncio.wait_for(
                        synthesizer.generate(synthesis_prompt, ctx.context_messages),
                        timeout=30.0,
                    )
                logger.info(f"synthesis_generated_sonnet chars={len(synthesis)}")
            except Exception as e:
                logger.warning(f"synthesis_sonnet_failed error={e}, using proposal combination")
                synthesis_source = "combined"

        # Try 3: Combine proposals as final fallback (always succeeds)
        if not synthesis:
            synthesis = self._combine_proposals_as_synthesis(ctx)
            logger.info(f"synthesis_generated_combined chars={len(synthesis)}")

        # Store synthesis in result
        ctx.result.synthesis = synthesis
        ctx.result.final_answer = synthesis

        # Emit explicit synthesis event (guaranteed delivery)
        self._emit_synthesis_events(ctx, synthesis, synthesis_source)

        # Generate export download links for aragora.ai debates
        self._generate_export_links(ctx)

        return True

    def _emit_synthesis_events(
        self,
        ctx: "DebateContext",
        synthesis: str,
        synthesis_source: str,
    ) -> None:
        """Emit synthesis-related events.

        Args:
            ctx: Debate context
            synthesis: Generated synthesis text
            synthesis_source: Source of synthesis (opus/sonnet/combined)
        """
        # Emit explicit synthesis event
        try:
            if self.hooks and "on_synthesis" in self.hooks:
                self.hooks["on_synthesis"](
                    content=synthesis,
                    confidence=ctx.result.confidence if ctx.result else 0.0,
                )
        except Exception as e:
            logger.warning(f"on_synthesis hook failed: {e}")

        # Also emit as agent_message for backwards compatibility
        try:
            if self.hooks and "on_message" in self.hooks:
                rounds = self.protocol.rounds if self.protocol else 3
                self.hooks["on_message"](
                    agent="synthesis-agent",
                    content=synthesis,
                    role="synthesis",  # Special role for frontend styling
                    round_num=rounds + 1,
                )
        except Exception as e:
            logger.warning(f"on_message hook failed: {e}")

        # Notify spectator
        try:
            if self._notify_spectator:
                self._notify_spectator(
                    "synthesis",
                    agent="synthesis-agent",
                    details=f"Final synthesis ({len(synthesis)} chars, source={synthesis_source})",
                    metric=ctx.result.confidence if ctx.result else 0.0,
                )
        except Exception as e:
            logger.warning(f"notify_spectator failed: {e}")

    def _generate_export_links(self, ctx: "DebateContext") -> None:
        """Generate export download links for the debate.

        Args:
            ctx: Debate context
        """
        debate_id = getattr(ctx, "debate_id", None) or getattr(ctx.result, "debate_id", None)
        if not debate_id:
            return

        ctx.result.export_links = {
            "json": f"/api/debates/{debate_id}/export/json",
            "markdown": f"/api/debates/{debate_id}/export/md",
            "html": f"/api/debates/{debate_id}/export/html",
            "txt": f"/api/debates/{debate_id}/export/txt",
            "csv_summary": f"/api/debates/{debate_id}/export/csv?table=summary",
            "csv_messages": f"/api/debates/{debate_id}/export/csv?table=messages",
        }
        logger.info(f"export_links_generated debate_id={debate_id}")

        # Emit export ready event
        try:
            if self.hooks and "on_export_ready" in self.hooks:
                self.hooks["on_export_ready"](
                    debate_id=debate_id,
                    links=ctx.result.export_links,
                )
        except Exception as e:
            logger.warning(f"on_export_ready hook failed: {e}")

    def _combine_proposals_as_synthesis(self, ctx: "DebateContext") -> str:
        """Combine proposals into a synthesis when LLM generation fails.

        This is a guaranteed fallback that always produces output.

        Args:
            ctx: The DebateContext with proposals

        Returns:
            Combined synthesis string
        """
        task = ctx.env.task if ctx.env else "the debate topic"
        proposals = ctx.proposals

        # If we have a winner, prioritize their proposal
        winner = ctx.result.winner if ctx.result else None
        if winner and winner in proposals:
            winner_proposal = proposals[winner]
            other_proposals = {k: v for k, v in proposals.items() if k != winner}

            synthesis = f"""## Final Synthesis

**Question:** {task}

### Winning Position ({winner})

{winner_proposal[:2000]}

### Other Perspectives

"""
            for agent, prop in list(other_proposals.items())[:3]:
                synthesis += f"**{agent}:** {prop[:500]}...\n\n"

            return synthesis

        # No winner - combine all proposals
        synthesis = f"""## Final Synthesis

**Question:** {task}

### Combined Perspectives

"""
        for agent, prop in list(proposals.items())[:5]:
            synthesis += f"**{agent}:**\n{prop[:800]}\n\n---\n\n"

        synthesis += "\n*Note: This synthesis was automatically generated from agent proposals.*"
        return synthesis

    def _build_synthesis_prompt(self, ctx: "DebateContext") -> str:
        """Build prompt for final synthesis generation.

        Args:
            ctx: The DebateContext with proposals, critiques, and task

        Returns:
            Formatted synthesis prompt string
        """
        proposals = ctx.proposals
        critiques = getattr(ctx, "critiques", []) or []
        task = ctx.env.task if ctx.env else "Unknown task"

        # Format proposals
        proposals_text = "\n\n---\n\n".join(
            f"**{agent}**:\n{prop[:1500]}" for agent, prop in proposals.items()
        )

        # Format critiques (if any)
        critiques_text = ""
        if critiques:
            critique_items = []
            for c in critiques[:5]:
                if hasattr(c, "agent") and hasattr(c, "target"):
                    summary = getattr(c, "summary", "")[:200] if hasattr(c, "summary") else ""
                    critique_items.append(f"- {c.agent} on {c.target}: {summary}")
            critiques_text = "\n".join(critique_items)

        return f"""You are Claude Opus 4.5, tasked with creating the DEFINITIVE synthesis of this multi-agent AI debate.

## ORIGINAL QUESTION
{task}

## AGENT FINAL PROPOSALS
{proposals_text}

## KEY CRITIQUES
{critiques_text if critiques_text else "No critiques recorded."}

## YOUR TASK
Create a comprehensive synthesis of **approximately 1200 words** (minimum 1000, maximum 1400) that includes:

1. **DEFINITIVE ANSWER** (2-3 sentences): State the conclusion clearly and authoritatively

2. **REASONING SUMMARY** (~300 words): Present the key arguments and evidence that emerged from the debate. Identify the strongest reasoning chains.

3. **CONSENSUS ANALYSIS** (~200 words): Detail where agents agreed and areas of genuine disagreement. Note which disagreements were resolved and which remain.

4. **SYNTHESIS OF PERSPECTIVES** (~300 words): Integrate the strongest points from each agent's position. Show how different viewpoints complement or challenge each other.

5. **ACTIONABLE RECOMMENDATIONS** (~200 words): Provide concrete, practical takeaways. What should someone do with this conclusion?

6. **REMAINING QUESTIONS** (~100 words): Note any unresolved issues, edge cases, or areas that merit further exploration.

Write authoritatively. This is the FINAL WORD on this debate.
Your response MUST be approximately 1200 words to provide comprehensive coverage."""


__all__ = ["SynthesisGenerator"]
