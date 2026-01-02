"""
Fractal Orchestrator - Recursive sub-debate spawning for deep problem-solving.

When debates hit unresolved tensions, spawns recursive sub-debates with
evolved specialist agents to resolve specific points, then synthesizes
results back into the parent debate.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from aragora.core import Agent, DebateResult, Environment
from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.debate.consensus import UnresolvedTension, ConsensusProof
from aragora.genesis.genome import AgentGenome, GenomeStore
from aragora.genesis.breeding import GenomeBreeder, PopulationManager, Population


@dataclass
class SubDebateResult:
    """Result from a sub-debate spawned to resolve a tension."""

    debate_id: str
    parent_debate_id: str
    tension: UnresolvedTension
    result: DebateResult
    specialist_genomes: list[AgentGenome]
    depth: int
    resolution: str  # Summary of how the tension was resolved
    success: bool  # Whether the sub-debate reached consensus


@dataclass
class FractalResult:
    """
    Complete result from a fractal debate including all sub-debates.

    Contains the main debate result plus a tree of sub-debate results
    for tensions that required deeper exploration.
    """

    root_debate_id: str
    main_result: DebateResult
    sub_debates: list[SubDebateResult] = field(default_factory=list)
    evolved_genomes: list[AgentGenome] = field(default_factory=list)
    total_depth: int = 0
    tensions_resolved: int = 0
    tensions_unresolved: int = 0

    @property
    def debate_tree(self) -> dict:
        """Get the debate tree structure."""
        def build_tree(debate_id: str) -> dict:
            children = [sd for sd in self.sub_debates if sd.parent_debate_id == debate_id]
            return {
                "debate_id": debate_id,
                "children": [
                    {
                        "debate_id": child.debate_id,
                        "tension": child.tension.description,
                        "success": child.success,
                        "depth": child.depth,
                        "children": build_tree(child.debate_id)["children"]
                    }
                    for child in children
                ]
            }
        return build_tree(self.root_debate_id)

    def get_all_debate_ids(self) -> list[str]:
        """Get all debate IDs in the tree."""
        return [self.root_debate_id] + [sd.debate_id for sd in self.sub_debates]


class FractalOrchestrator:
    """
    Orchestrates fractal debates with recursive sub-debate spawning.

    When a debate produces unresolved tensions above the threshold,
    spawns sub-debates with specialist agents to resolve them,
    then synthesizes results back into the parent context.
    """

    def __init__(
        self,
        max_depth: int = 3,
        tension_threshold: float = 0.7,
        timeout_inheritance: float = 0.5,
        evolve_agents: bool = True,
        population_manager: Optional[PopulationManager] = None,
        event_hooks: dict = None,
    ):
        """
        Args:
            max_depth: Maximum recursion depth for sub-debates
            tension_threshold: Minimum tension severity to spawn sub-debate (0-1)
            timeout_inheritance: Fraction of parent's remaining time for sub-debates
            evolve_agents: Whether to spawn evolved specialist agents
            population_manager: Manager for agent populations (created if not provided)
            event_hooks: Optional hooks for streaming events
        """
        self.max_depth = max_depth
        self.tension_threshold = tension_threshold
        self.timeout_inheritance = timeout_inheritance
        self.evolve_agents = evolve_agents
        self.population_manager = population_manager or PopulationManager()
        self.hooks = event_hooks or {}
        self.breeder = GenomeBreeder()
        self.genome_store = GenomeStore()

        # Track all sub-debates
        self._sub_debates: list[SubDebateResult] = []
        self._evolved_genomes: list[AgentGenome] = []

    async def run(
        self,
        task: str,
        agents: list[Agent],
        population: Optional[Population] = None,
        depth: int = 0,
        parent_debate_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> FractalResult:
        """
        Run a fractal debate, spawning sub-debates as needed.

        Args:
            task: The debate task/question
            agents: Agents to participate in the debate
            population: Population to draw specialist agents from
            depth: Current recursion depth
            parent_debate_id: ID of parent debate (if sub-debate)
            timeout: Maximum time for this debate (seconds)
        """
        debate_id = str(uuid.uuid4())[:8]

        # Require at least 2 agents
        if len(agents) < 2:
            raise ValueError(f"Fractal debate requires at least 2 agents, got {len(agents)}")

        # Emit fractal start event
        if "on_fractal_start" in self.hooks:
            self.hooks["on_fractal_start"](
                debate_id=debate_id,
                task=task,
                depth=depth,
                parent_id=parent_debate_id,
            )

        # Get or create population
        if population is None:
            base_agent_names = [a.name.split("_")[0] for a in agents]
            population = self.population_manager.get_or_create_population(base_agent_names)

        # Create environment and arena
        env = Environment(task=task)
        protocol = DebateProtocol(rounds=2, consensus="majority")

        arena = Arena(
            environment=env,
            agents=agents,
            protocol=protocol,
            event_hooks=self.hooks,
        )

        # Run the main debate
        result = await arena.run()

        # Record debate participation
        self.population_manager.record_debate(population, debate_id)

        # Check for unresolved tensions
        tensions = self._extract_tensions(result)
        high_priority_tensions = [
            t for t in tensions
            if self._tension_severity(t) >= self.tension_threshold
        ]

        # Spawn sub-debates for high-priority tensions (if under depth limit)
        sub_debate_results = []
        if depth < self.max_depth and high_priority_tensions:
            sub_timeout = (timeout * self.timeout_inheritance) if timeout else None

            for tension in high_priority_tensions[:2]:  # Max 2 sub-debates per level
                sub_result = await self._spawn_sub_debate(
                    tension=tension,
                    parent_debate_id=debate_id,
                    population=population,
                    depth=depth + 1,
                    timeout=sub_timeout,
                )
                sub_debate_results.append(sub_result)
                self._sub_debates.append(sub_result)

        # Synthesize sub-debate results back into main result
        if sub_debate_results:
            result = self._synthesize_results(result, sub_debate_results)

        # Update fitness for participating genomes
        if self.evolve_agents:
            self._update_genome_fitness(population, result)

        # Emit fractal complete event
        if "on_fractal_complete" in self.hooks:
            self.hooks["on_fractal_complete"](
                debate_id=debate_id,
                depth=depth,
                sub_debates=len(sub_debate_results),
                consensus_reached=result.consensus_reached,
            )

        # Build final result
        return FractalResult(
            root_debate_id=debate_id if depth == 0 else parent_debate_id or debate_id,
            main_result=result,
            sub_debates=self._sub_debates if depth == 0 else [],
            evolved_genomes=self._evolved_genomes if depth == 0 else [],
            total_depth=max(
                depth,
                max((sd.depth for sd in self._sub_debates), default=0) if self._sub_debates else 0
            ),
            tensions_resolved=sum(1 for sd in sub_debate_results if sd.success),
            tensions_unresolved=len(high_priority_tensions) - sum(1 for sd in sub_debate_results if sd.success),
        )

    async def _spawn_sub_debate(
        self,
        tension: UnresolvedTension,
        parent_debate_id: str,
        population: Population,
        depth: int,
        timeout: Optional[float],
    ) -> SubDebateResult:
        """Spawn a sub-debate to resolve a specific tension."""
        sub_debate_id = f"{parent_debate_id}-sub-{str(uuid.uuid4())[:4]}"

        # Emit sub-debate spawn event
        if "on_fractal_spawn" in self.hooks:
            self.hooks["on_fractal_spawn"](
                debate_id=sub_debate_id,
                parent_id=parent_debate_id,
                tension=tension.description,
                depth=depth,
            )

        # Determine domain from tension
        domain = self._infer_domain(tension)

        # Spawn specialist agents
        specialists = []
        if self.evolve_agents and population.genomes:
            # Create specialists for this domain
            for _ in range(min(2, len(population.genomes))):
                specialist = self.breeder.spawn_specialist(
                    domain=domain,
                    parent_pool=population.genomes,
                    debate_id=sub_debate_id,
                )
                self.genome_store.save(specialist)
                specialists.append(specialist)
                self._evolved_genomes.append(specialist)

        # Convert genomes to agents
        from aragora.agents.base import create_agent
        specialist_agents = []
        for genome in specialists:
            try:
                agent = create_agent(
                    agent_type=genome.model_preference,
                    name=genome.name,
                    role="proposer",
                )
                # Inject genome context into agent's system prompt
                context = genome.to_persona().to_prompt_context()
                if hasattr(agent, 'system_prompt'):
                    agent.system_prompt = f"{agent.system_prompt}\n\n{context}"
                specialist_agents.append(agent)
            except Exception:
                # Fall back to generic agents if creation fails
                pass

        # If no specialists, use agents from population
        if not specialist_agents:
            best_genomes = self.population_manager.get_best_for_domain(domain, n=2)
            for genome in best_genomes:
                try:
                    agent = create_agent(
                        agent_type=genome.model_preference,
                        name=genome.name,
                        role="proposer",
                    )
                    specialist_agents.append(agent)
                except Exception:
                    pass

        # Build focused task from tension
        focused_task = self._build_sub_task(tension)

        # Require at least 2 agents for sub-debate
        if len(specialist_agents) < 2:
            # Not enough specialists, skip sub-debate
            if "on_fractal_merge" in self.hooks:
                self.hooks["on_fractal_merge"](
                    debate_id=sub_debate_id,
                    parent_id=parent_debate_id,
                    success=False,
                    resolution="Not enough agents available for sub-debate",
                )
            return SubDebateResult(
                debate_id=sub_debate_id,
                parent_debate_id=parent_debate_id,
                tension=tension,
                result=None,
                specialist_genomes=specialists,
                depth=depth,
                resolution="Not enough agents available for sub-debate",
                success=False,
            )

        # Run sub-debate recursively
        sub_result = await self.run(
            task=focused_task,
            agents=specialist_agents,
            population=population,
            depth=depth,
            parent_debate_id=parent_debate_id,
            timeout=timeout,
        )

        # Emit sub-debate complete event
        if "on_fractal_merge" in self.hooks:
            self.hooks["on_fractal_merge"](
                debate_id=sub_debate_id,
                parent_id=parent_debate_id,
                success=sub_result.main_result.consensus_reached,
            )

        return SubDebateResult(
            debate_id=sub_debate_id,
            parent_debate_id=parent_debate_id,
            tension=tension,
            result=sub_result.main_result,
            specialist_genomes=specialists,
            depth=depth,
            resolution=sub_result.main_result.final_answer or "",
            success=sub_result.main_result.consensus_reached,
        )

    def _extract_tensions(self, result: DebateResult) -> list[UnresolvedTension]:
        """Extract unresolved tensions from debate result."""
        tensions = []

        # Look for explicit tensions in dissenting views
        for i, dissent in enumerate(result.dissenting_views):
            if dissent and len(dissent) > 50:  # Non-trivial dissent
                tensions.append(UnresolvedTension(
                    tension_id=f"tension-{i}",
                    description=dissent[:200],
                    agents_involved=[],  # Would need to track this
                    options=[],
                    impact="May affect consensus quality",
                ))

        # Look for disagreement patterns in critiques
        high_severity_critiques = [c for c in result.critiques if c.severity > 0.7]
        if high_severity_critiques:
            # Group by similar issues
            issue_groups = {}
            for critique in high_severity_critiques:
                for issue in critique.issues:
                    key = issue[:50]  # Group by first 50 chars
                    if key not in issue_groups:
                        issue_groups[key] = []
                    issue_groups[key].append(critique)

            for key, critiques in issue_groups.items():
                if len(critiques) >= 2:  # Multiple agents raised this
                    tensions.append(UnresolvedTension(
                        tension_id=f"critique-tension-{len(tensions)}",
                        description=key,
                        agents_involved=[c.agent for c in critiques],
                        options=[s for c in critiques for s in c.suggestions[:1]],
                        impact="Multiple agents raised concerns",
                    ))

        return tensions

    def _tension_severity(self, tension: UnresolvedTension) -> float:
        """Calculate severity of a tension (0-1)."""
        score = 0.5  # Base score

        # More agents involved = higher severity
        if len(tension.agents_involved) >= 3:
            score += 0.2
        elif len(tension.agents_involved) >= 2:
            score += 0.1

        # Has multiple options = more complex
        if len(tension.options) >= 2:
            score += 0.1

        # Has explicit impact = higher severity
        if tension.impact and len(tension.impact) > 20:
            score += 0.1

        return min(1.0, score)

    def _infer_domain(self, tension: UnresolvedTension) -> str:
        """Infer expertise domain from tension description."""
        text = tension.description.lower()

        domain_keywords = {
            "security": ["security", "auth", "encrypt", "vulnerab", "inject", "xss"],
            "performance": ["performance", "speed", "latency", "optim", "cache", "fast"],
            "architecture": ["architecture", "design", "pattern", "structure", "module"],
            "testing": ["test", "coverage", "unit", "integrat", "mock"],
            "concurrency": ["concurrent", "async", "parallel", "thread", "race"],
            "database": ["database", "query", "sql", "schema", "index"],
            "api_design": ["api", "endpoint", "rest", "interface", "contract"],
            "error_handling": ["error", "exception", "fail", "retry", "recover"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in text for kw in keywords):
                return domain

        return "architecture"  # Default domain

    def _build_sub_task(self, tension: UnresolvedTension) -> str:
        """Build a focused task from a tension."""
        options_str = ""
        if tension.options:
            options_str = "\n\nOptions to consider:\n" + "\n".join(f"- {o}" for o in tension.options)

        return f"""Resolve this specific technical tension:

{tension.description}

Impact: {tension.impact}
{options_str}

Provide a clear recommendation with justification. Focus on this specific point only."""

    def _synthesize_results(
        self,
        parent_result: DebateResult,
        sub_results: list[SubDebateResult]
    ) -> DebateResult:
        """Merge sub-debate conclusions back into parent result."""
        # Build synthesis from sub-debate resolutions
        synthesis_parts = []
        for sub in sub_results:
            if sub.success and sub.resolution:
                synthesis_parts.append(
                    f"[Sub-debate on '{sub.tension.description[:50]}...']: {sub.resolution[:200]}"
                )

        if synthesis_parts:
            synthesis = "\n\n".join(synthesis_parts)
            if parent_result.final_answer:
                parent_result.final_answer += f"\n\n=== Resolved Tensions ===\n{synthesis}"
            else:
                parent_result.final_answer = synthesis

        return parent_result

    def _update_genome_fitness(self, population: Population, result: DebateResult) -> None:
        """Update fitness scores for genomes based on debate outcome."""
        consensus_reached = result.consensus_reached

        for genome in population.genomes:
            # Check if this genome's agent contributed to consensus
            agent_name = genome.name
            contributed = any(
                m.agent == agent_name or agent_name in m.agent
                for m in result.messages
            )

            if contributed:
                self.population_manager.update_fitness(
                    genome.genome_id,
                    consensus_win=consensus_reached,
                    critique_accepted=any(
                        c.agent == agent_name and c.severity < 0.5
                        for c in result.critiques
                    ),
                )

    def should_spawn(self, tensions: list[UnresolvedTension]) -> bool:
        """Check if sub-debate is warranted based on tensions."""
        high_priority = [
            t for t in tensions
            if self._tension_severity(t) >= self.tension_threshold
        ]
        return len(high_priority) > 0
