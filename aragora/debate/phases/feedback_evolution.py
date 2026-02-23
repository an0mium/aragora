"""
Evolution feedback methods for FeedbackPhase.

Extracted from feedback_phase.py for maintainability.
Handles genome fitness updates, population evolution, and pattern extraction.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.core import Agent
    from aragora.debate.context import DebateContext
    from aragora.type_protocols import (
        EloSystemProtocol,
        EventEmitterProtocol,
        PopulationManagerProtocol,
        PromptEvolverProtocol,
    )

logger = logging.getLogger(__name__)


class EvolutionFeedback:
    """Handles Genesis/evolution-related feedback operations."""

    # Fitness deltas applied to genomes based on debate outcomes.
    # These are additive adjustments on top of the rate-based fitness
    # calculated by AgentGenome.update_fitness(), ensuring that debate
    # wins/losses create strong selection pressure for breeding.
    WIN_FITNESS_DELTA: float = 0.10
    LOSS_FITNESS_DELTA: float = -0.05

    def __init__(
        self,
        population_manager: PopulationManagerProtocol | None = None,
        prompt_evolver: PromptEvolverProtocol | None = None,
        event_emitter: EventEmitterProtocol | None = None,
        elo_system: EloSystemProtocol | None = None,
        loop_id: str | None = None,
        auto_evolve: bool = True,
        breeding_threshold: float = 0.8,
        elo_baseline: float = 1500.0,
    ):
        self.population_manager = population_manager
        self.prompt_evolver = prompt_evolver
        self.event_emitter = event_emitter
        self.elo_system = elo_system
        self.loop_id = loop_id
        self.auto_evolve = auto_evolve
        self.breeding_threshold = breeding_threshold
        self.elo_baseline = elo_baseline

    def update_genome_fitness(self, ctx: DebateContext) -> None:
        """Update genome fitness scores based on debate outcome.

        For agents with genome_id attributes (evolved via Genesis),
        update their fitness scores based on debate performance:

        1. ``consensus_win`` / ``prediction_correct`` / ``critique_accepted``
           feed into the rate-based fitness in AgentGenome.update_fitness().
        2. An explicit win/loss ``fitness_delta`` creates direct selection
           pressure so that winning agents are more likely to be chosen as
           parents in the next breeding round.
        3. ELO performance is layered on top as an additional delta.
        """
        if not self.population_manager:
            return

        result = ctx.result
        if not result:
            return

        winner_agent = getattr(result, "winner", None)

        # Pre-compute which agents had their critiques accepted so
        # we can pass this signal into the genome fitness update.
        accepted_critics = self._compute_accepted_critiques(ctx)

        for agent in ctx.agents:
            genome_id = getattr(agent, "genome_id", None)
            if not genome_id:
                continue

            try:
                # Determine if this agent won
                consensus_win = agent.name == winner_agent

                # Check if agent's prediction was correct
                prediction_correct = self._check_agent_prediction(agent, ctx)

                # Check if any of this agent's critiques were accepted
                critique_accepted = agent.name in accepted_critics

                # Update fitness in population manager (rate-based)
                self.population_manager.update_fitness(
                    genome_id,
                    consensus_win=consensus_win,
                    critique_accepted=critique_accepted,
                    prediction_correct=prediction_correct,
                )

                # Apply direct win/loss fitness delta for breeding selection
                outcome_delta = (
                    self.WIN_FITNESS_DELTA if consensus_win
                    else self.LOSS_FITNESS_DELTA
                )
                if winner_agent is not None and outcome_delta != 0.0:
                    self.population_manager.update_fitness(
                        genome_id, fitness_delta=outcome_delta,
                    )

                # Factor ELO performance into genome fitness
                elo_adj = self._compute_elo_fitness_adjustment(agent.name, ctx.domain)
                if elo_adj != 0.0:
                    self.population_manager.update_fitness(
                        genome_id, fitness_delta=elo_adj,
                    )

                logger.debug(
                    "[genesis] Updated fitness for genome %s: win=%s crit=%s pred=%s "
                    "outcome_delta=%+.2f elo_adj=%+.4f",
                    genome_id[:8],
                    consensus_win,
                    critique_accepted,
                    prediction_correct,
                    outcome_delta if winner_agent is not None else 0.0,
                    elo_adj,
                )
            except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
                logger.debug("Genome fitness update failed for %s: %s", agent.name, e)

    def _check_agent_prediction(
        self,
        agent: Agent,
        ctx: DebateContext,
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

    def _compute_accepted_critiques(self, ctx: DebateContext) -> set[str]:
        """Determine which agents had their critiques accepted.

        A critique is considered "accepted" when the critiquing agent
        targeted a non-winning agent (i.e., the critique correctly
        identified weaknesses that prevented the target from winning).
        This rewards agents whose critical judgment aligns with the
        final debate outcome.

        Returns:
            Set of agent names whose critiques were vindicated by the
            debate outcome.
        """
        result = ctx.result
        if not result:
            return set()

        winner = getattr(result, "winner", None)
        if not winner:
            return set()

        critiques = getattr(result, "critiques", None)
        if not critiques:
            return set()

        accepted: set[str] = set()
        for critique in critiques:
            critic_name = getattr(critique, "agent", None)
            target_name = getattr(critique, "target_agent", None) or getattr(
                critique, "target", None
            )
            if not critic_name or not target_name:
                continue
            # A critique is accepted when it targeted a non-winner,
            # meaning the critic correctly identified flaws in a
            # losing proposal.
            if target_name != winner:
                accepted.add(critic_name)

        return accepted

    def _compute_elo_fitness_adjustment(
        self,
        agent_name: str,
        domain: str | None = None,
    ) -> float:
        """Compute a genome fitness adjustment from the agent's ELO rating.

        Queries the ELO system for the agent's current rating, computes the
        delta from the baseline (default 1500), and converts it to a small
        fitness adjustment via ``fitness_from_elo``.

        Returns 0.0 when the ELO system is unavailable or on any error.
        """
        if not self.elo_system:
            return 0.0

        try:
            from aragora.genesis.breeding import fitness_from_elo

            elo_rating = self.elo_system.get_rating(agent_name, domain or "")
            delta = elo_rating - self.elo_baseline
            return fitness_from_elo(delta)
        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            logger.debug("[genesis] ELO fitness lookup failed for %s: %s", agent_name, e)
            return 0.0

    async def maybe_evolve_population(self, ctx: DebateContext) -> None:
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
                _evolve_task = asyncio.create_task(self._evolve_async(population))
                _evolve_task.add_done_callback(
                    lambda t: logger.warning("[genesis] Evolution failed: %s", t.exception())
                    if not t.cancelled() and t.exception()
                    else None
                )
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
                from aragora.events.types import StreamEvent, StreamEventType

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

    def record_evolution_patterns(self, ctx: DebateContext) -> None:
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
                def __init__(self, ctx_result: Any, ctx_obj: Any) -> None:
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
                                    def __init__(self, m: Any) -> None:
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
