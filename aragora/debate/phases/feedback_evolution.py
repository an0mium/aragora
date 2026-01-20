"""
Evolution feedback methods for FeedbackPhase.

Extracted from feedback_phase.py for maintainability.
Handles genome fitness updates, population evolution, and pattern extraction.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.core import Agent
    from aragora.debate.context import DebateContext
    from aragora.type_protocols import (
        EventEmitterProtocol,
        PopulationManagerProtocol,
        PromptEvolverProtocol,
    )

logger = logging.getLogger(__name__)


class EvolutionFeedback:
    """Handles Genesis/evolution-related feedback operations."""

    def __init__(
        self,
        population_manager: Optional["PopulationManagerProtocol"] = None,
        prompt_evolver: Optional["PromptEvolverProtocol"] = None,
        event_emitter: Optional["EventEmitterProtocol"] = None,
        loop_id: Optional[str] = None,
        auto_evolve: bool = True,
        breeding_threshold: float = 0.8,
    ):
        self.population_manager = population_manager
        self.prompt_evolver = prompt_evolver
        self.event_emitter = event_emitter
        self.loop_id = loop_id
        self.auto_evolve = auto_evolve
        self.breeding_threshold = breeding_threshold

    def update_genome_fitness(self, ctx: "DebateContext") -> None:
        """Update genome fitness scores based on debate outcome.

        For agents with genome_id attributes (evolved via Genesis),
        update their fitness scores based on debate performance.
        """
        if not self.population_manager:
            return

        result = ctx.result
        if not result:
            return

        winner_agent = getattr(result, "winner", None)

        for agent in ctx.agents:
            genome_id = getattr(agent, "genome_id", None)
            if not genome_id:
                continue

            try:
                # Determine if this agent won
                consensus_win = agent.name == winner_agent

                # Check if agent's prediction was correct
                prediction_correct = self._check_agent_prediction(agent, ctx)

                # Update fitness in population manager
                self.population_manager.update_fitness(
                    genome_id,
                    consensus_win=consensus_win,
                    prediction_correct=prediction_correct,
                )

                logger.debug(
                    "[genesis] Updated fitness for genome %s: win=%s pred=%s",
                    genome_id[:8],
                    consensus_win,
                    prediction_correct,
                )
            except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
                logger.debug("Genome fitness update failed for %s: %s", agent.name, e)

    def _check_agent_prediction(
        self,
        agent: "Agent",
        ctx: "DebateContext",
    ) -> bool:
        """Check if an agent correctly predicted the debate outcome.

        Returns True if the agent's vote matched the final winner.
        """
        result = ctx.result
        if not result or not result.votes:
            return False

        winner = getattr(result, "winner", None)
        if not winner:
            return False

        for vote in result.votes:
            if vote.agent == agent.name:
                # Check if the agent's choice matches the winner
                canonical = ctx.choice_mapping.get(vote.choice, vote.choice)
                return canonical == winner

        return False

    async def maybe_evolve_population(self, ctx: "DebateContext") -> None:
        """Trigger population evolution after high-quality debates.

        Evolution is triggered when:
        1. auto_evolve is True
        2. Debate confidence >= breeding_threshold
        3. Population has accumulated enough debate history
        """
        if not self.population_manager or not self.auto_evolve:
            return

        result = ctx.result
        if not result:
            return

        # Only evolve after high-confidence debates
        if result.confidence < self.breeding_threshold:
            return

        try:
            # Get the population for these agents
            agent_names = [a.name for a in ctx.agents]
            population = self.population_manager.get_or_create_population(agent_names)

            if not population:
                return

            # Track debate in population history
            history = getattr(population, "debate_history", []) or []
            history.append(ctx.debate_id)

            # Evolve every 5 debates
            if len(history) % 5 == 0:
                # Fire-and-forget evolution
                asyncio.create_task(self._evolve_async(population))
                logger.info(
                    "[genesis] Triggered evolution after %d debates (confidence=%.2f)",
                    len(history),
                    result.confidence,
                )

        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.debug("Evolution check failed: %s", e)

    async def _evolve_async(self, population: Any) -> None:
        """Run population evolution asynchronously.

        This is a fire-and-forget task so it doesn't block debate completion.
        """
        try:
            evolved = self.population_manager.evolve_population(population)
            logger.info(
                "[genesis] Population evolved to generation %d with %d genomes",
                evolved.generation,
                len(evolved.genomes),
            )

            # Emit event if event_emitter available
            if self.event_emitter:
                from aragora.server.stream import StreamEvent, StreamEventType

                self.event_emitter.emit(
                    StreamEvent(
                        type=StreamEventType.GENESIS_EVOLUTION,
                        loop_id=self.loop_id,
                        data={
                            "generation": evolved.generation,
                            "genome_count": len(evolved.genomes),
                            "population_id": getattr(population, "id", ""),
                            "top_fitness": getattr(evolved, "top_fitness", 0.0),
                        },
                    )
                )

        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.warning("[genesis] Evolution failed: %s", e)

    def record_evolution_patterns(self, ctx: "DebateContext") -> None:
        """Extract winning patterns from high-confidence debates for prompt evolution.

        When enabled via protocol.enable_evolution, this method:
        1. Extracts patterns from successful debates (high confidence)
        2. Stores patterns in the PromptEvolver database
        3. Updates performance metrics for agent prompts

        Only runs for debates with confidence >= 0.7 to ensure quality patterns.
        """
        if not self.prompt_evolver:
            return

        result = ctx.result
        if not result:
            return

        # Only extract patterns from high-confidence debates
        if result.confidence < 0.7:
            return

        try:
            # Build a minimal DebateResult-like object for the evolver
            # The evolver expects objects with specific attributes
            class DebateResultProxy:
                def __init__(self, ctx_result, ctx_obj):
                    self.id = ctx_obj.debate_id
                    self.consensus_reached = ctx_result.consensus_reached
                    self.confidence = ctx_result.confidence
                    self.final_answer = ctx_result.final_answer or ""
                    self.critiques = []

                    # Extract critiques from messages if available
                    if ctx_result.messages:
                        for msg in ctx_result.messages:
                            if getattr(msg, "role", "") == "critic":
                                # Create a critique-like object
                                class CritiqueProxy:
                                    def __init__(self, m):
                                        self.severity = getattr(m, "severity", 0.5)
                                        self.issues = getattr(m, "issues", [])
                                        self.suggestions = getattr(m, "suggestions", [])

                                self.critiques.append(CritiqueProxy(msg))

            proxy = DebateResultProxy(result, ctx)

            # Extract patterns from this debate
            patterns = self.prompt_evolver.extract_winning_patterns([proxy])
            if patterns:
                self.prompt_evolver.store_patterns(patterns)
                logger.info(
                    "[evolution] Extracted %d patterns from debate %s (confidence=%.2f)",
                    len(patterns),
                    ctx.debate_id,
                    result.confidence,
                )

            # Update performance for each agent's current prompt version
            for agent in ctx.agents:
                prompt_version = getattr(agent, "prompt_version", None)
                if prompt_version is not None:
                    self.prompt_evolver.update_performance(
                        agent_name=agent.name,
                        version=prompt_version,
                        debate_result=proxy,
                    )

        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.debug("[evolution] Pattern extraction failed: %s", e)


__all__ = ["EvolutionFeedback"]
