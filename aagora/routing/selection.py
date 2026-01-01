"""
Adaptive Agent Selection using ELO and Personas.

Routes tasks to best-fit agents by:
- Matching task domain to agent expertise
- Using ELO ratings for quality ranking
- Forming optimal teams for debates
- Maintaining a "bench" system with promotion/demotion
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
import random
import math


@dataclass
class AgentProfile:
    """Profile of an available agent."""

    name: str
    agent_type: str  # "claude", "codex", "gemini", etc.
    elo_rating: float = 1500
    domain_ratings: dict[str, float] = field(default_factory=dict)
    expertise: dict[str, float] = field(default_factory=dict)  # domain -> 0-1
    traits: list[str] = field(default_factory=list)
    availability: float = 1.0  # 0-1, how available is this agent
    cost_factor: float = 1.0  # Relative cost multiplier
    latency_ms: float = 1000  # Average response latency
    success_rate: float = 0.8  # Historical success rate

    @property
    def overall_score(self) -> float:
        """Calculate overall agent quality score."""
        return (
            self.elo_rating / 2000 * 0.4 +
            self.success_rate * 0.3 +
            (1 - min(self.latency_ms, 5000) / 5000) * 0.15 +
            (1 - min(self.cost_factor, 3) / 3) * 0.15
        )


@dataclass
class TaskRequirements:
    """Requirements for a task."""

    task_id: str
    description: str
    primary_domain: str
    secondary_domains: list[str] = field(default_factory=list)
    required_traits: list[str] = field(default_factory=list)
    min_agents: int = 2
    max_agents: int = 5
    quality_priority: float = 0.5  # 0 = speed/cost, 1 = quality
    diversity_preference: float = 0.5  # 0 = homogeneous, 1 = diverse


@dataclass
class TeamComposition:
    """A selected team of agents."""

    team_id: str
    task_id: str
    agents: list[AgentProfile]
    roles: dict[str, str]  # agent_name -> role
    expected_quality: float
    expected_cost: float
    diversity_score: float
    rationale: str


class AgentSelector:
    """
    Selects optimal agents for tasks based on ELO, expertise, and team dynamics.

    Features:
    - Domain-aware selection
    - Team diversity optimization
    - Cost/quality tradeoffs
    - Bench system for testing new agents
    """

    def __init__(
        self,
        elo_system: Optional[Any] = None,
        persona_manager: Optional[Any] = None,
    ):
        self.elo_system = elo_system
        self.persona_manager = persona_manager
        self.agent_pool: dict[str, AgentProfile] = {}
        self.bench: list[str] = []  # Agents on the bench (probation/testing)
        self._selection_history: list[dict] = []

    def register_agent(self, profile: AgentProfile):
        """Register an agent in the pool."""
        self.agent_pool[profile.name] = profile

    def remove_agent(self, name: str):
        """Remove an agent from the pool."""
        if name in self.agent_pool:
            del self.agent_pool[name]
        if name in self.bench:
            self.bench.remove(name)

    def move_to_bench(self, name: str):
        """Move an agent to the bench (probation)."""
        if name in self.agent_pool and name not in self.bench:
            self.bench.append(name)

    def promote_from_bench(self, name: str):
        """Promote an agent from bench to active pool."""
        if name in self.bench:
            self.bench.remove(name)

    def select_team(
        self,
        requirements: TaskRequirements,
        exclude: Optional[list[str]] = None,
    ) -> TeamComposition:
        """
        Select an optimal team for the task.

        Args:
            requirements: Task requirements
            exclude: Agent names to exclude

        Returns:
            TeamComposition with selected agents
        """
        exclude = exclude or []
        candidates = [
            a for a in self.agent_pool.values()
            if a.name not in exclude and a.name not in self.bench
        ]

        if not candidates:
            raise ValueError("No available agents in pool")

        # Score candidates for this task
        scored = [(a, self._score_for_task(a, requirements)) for a in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select team considering diversity
        team = self._select_diverse_team(
            scored,
            requirements.min_agents,
            requirements.max_agents,
            requirements.diversity_preference,
        )

        # Assign roles
        roles = self._assign_roles(team, requirements)

        # Calculate expected quality
        expected_quality = sum(
            self._score_for_task(a, requirements) for a in team
        ) / len(team)

        # Calculate cost
        expected_cost = sum(a.cost_factor for a in team)

        # Calculate diversity
        diversity_score = self._calculate_diversity(team)

        # Generate rationale
        rationale = self._generate_rationale(team, requirements, scored)

        # Record selection
        self._selection_history.append({
            "task_id": requirements.task_id,
            "selected": [a.name for a in team],
            "timestamp": datetime.now().isoformat(),
        })

        return TeamComposition(
            team_id=f"team-{requirements.task_id}",
            task_id=requirements.task_id,
            agents=team,
            roles=roles,
            expected_quality=expected_quality,
            expected_cost=expected_cost,
            diversity_score=diversity_score,
            rationale=rationale,
        )

    def _score_for_task(
        self,
        agent: AgentProfile,
        requirements: TaskRequirements,
    ) -> float:
        """Score an agent for a specific task."""
        score = 0.0

        # Base ELO contribution
        elo_score = (agent.elo_rating - 1000) / 1000  # Normalize around 0
        score += elo_score * 0.3

        # Domain expertise
        primary_exp = agent.expertise.get(requirements.primary_domain, 0.5)
        score += primary_exp * 0.3

        # Domain-specific ELO
        domain_elo = agent.domain_ratings.get(requirements.primary_domain, 1500)
        score += (domain_elo - 1000) / 1000 * 0.2

        # Secondary domains
        if requirements.secondary_domains:
            secondary_score = sum(
                agent.expertise.get(d, 0.3)
                for d in requirements.secondary_domains
            ) / len(requirements.secondary_domains)
            score += secondary_score * 0.1

        # Trait matching
        if requirements.required_traits:
            matching_traits = sum(
                1 for t in requirements.required_traits
                if t in agent.traits
            )
            score += matching_traits / len(requirements.required_traits) * 0.1

        # Adjust for quality priority
        if requirements.quality_priority > 0.5:
            # Prefer quality: weight success rate more
            score = score * 0.7 + agent.success_rate * 0.3
        else:
            # Prefer speed/cost: weight latency/cost more
            speed_score = 1 - min(agent.latency_ms, 5000) / 5000
            cost_score = 1 - min(agent.cost_factor, 3) / 3
            score = score * 0.6 + speed_score * 0.2 + cost_score * 0.2

        return max(0, min(1, score))

    def _select_diverse_team(
        self,
        scored: list[tuple[AgentProfile, float]],
        min_size: int,
        max_size: int,
        diversity_pref: float,
    ) -> list[AgentProfile]:
        """Select a diverse team from scored candidates."""
        if len(scored) <= min_size:
            return [a for a, _ in scored]

        team = []
        remaining = list(scored)

        while len(team) < max_size and remaining:
            if len(team) < min_size or random.random() > diversity_pref:
                # Greedy: pick highest scored
                agent, score = remaining[0]
                team.append(agent)
                remaining = remaining[1:]
            else:
                # Diversity: pick someone different
                team_types = set(a.agent_type for a in team)
                team_traits = set()
                for a in team:
                    team_traits.update(a.traits)

                # Find most different agent
                best_diff = None
                best_diff_score = -1

                for agent, score in remaining:
                    diff_score = 0
                    if agent.agent_type not in team_types:
                        diff_score += 0.5
                    new_traits = set(agent.traits) - team_traits
                    diff_score += len(new_traits) * 0.1
                    diff_score += score * 0.4  # Still consider quality

                    if diff_score > best_diff_score:
                        best_diff = agent
                        best_diff_score = diff_score

                if best_diff:
                    team.append(best_diff)
                    remaining = [(a, s) for a, s in remaining if a.name != best_diff.name]
                else:
                    break

        return team

    def _assign_roles(
        self,
        team: list[AgentProfile],
        requirements: TaskRequirements,
    ) -> dict[str, str]:
        """Assign debate roles to team members."""
        roles = {}

        if not team:
            return roles

        # Sort by domain expertise for proposer selection
        by_expertise = sorted(
            team,
            key=lambda a: a.expertise.get(requirements.primary_domain, 0),
            reverse=True,
        )

        # Assign proposer to highest domain expert
        roles[by_expertise[0].name] = "proposer"

        # Assign synthesizer to most balanced agent
        if len(team) > 1:
            remaining = by_expertise[1:]
            balanced = min(
                remaining,
                key=lambda a: abs(a.overall_score - 0.5),
            )
            roles[balanced.name] = "synthesizer"

        # Assign critics to rest
        for agent in team:
            if agent.name not in roles:
                # Try to match critic type to traits
                if "thorough" in agent.traits or "security" in agent.traits:
                    roles[agent.name] = "security_critic"
                elif "pragmatic" in agent.traits or "performance" in agent.traits:
                    roles[agent.name] = "performance_critic"
                else:
                    roles[agent.name] = "critic"

        return roles

    def _calculate_diversity(self, team: list[AgentProfile]) -> float:
        """Calculate team diversity score."""
        if len(team) <= 1:
            return 0.0

        # Type diversity
        types = set(a.agent_type for a in team)
        type_div = len(types) / len(team)

        # Trait diversity
        all_traits = set()
        for a in team:
            all_traits.update(a.traits)
        trait_div = len(all_traits) / (len(team) * 3)  # Assume avg 3 traits

        # ELO diversity (mix of skill levels)
        elos = [a.elo_rating for a in team]
        elo_range = max(elos) - min(elos) if len(elos) > 1 else 0
        elo_div = min(elo_range / 500, 1.0)  # Normalize to 500 range

        return (type_div * 0.4 + trait_div * 0.3 + elo_div * 0.3)

    def _generate_rationale(
        self,
        team: list[AgentProfile],
        requirements: TaskRequirements,
        all_scored: list[tuple[AgentProfile, float]],
    ) -> str:
        """Generate human-readable selection rationale."""
        lines = [
            f"Selected {len(team)} agents for task '{requirements.task_id}':",
            f"Primary domain: {requirements.primary_domain}",
            "",
            "Team composition:",
        ]

        for agent in team:
            expertise = agent.expertise.get(requirements.primary_domain, 0)
            domain_elo = agent.domain_ratings.get(requirements.primary_domain, agent.elo_rating)
            lines.append(
                f"- {agent.name} ({agent.agent_type}): "
                f"ELO {agent.elo_rating:.0f}, "
                f"Domain ELO {domain_elo:.0f}, "
                f"Expertise {expertise:.0%}"
            )

        if len(all_scored) > len(team):
            lines.append("")
            lines.append(f"Considered {len(all_scored)} candidates, selected top {len(team)}")

        return "\n".join(lines)

    def update_from_result(
        self,
        team: TeamComposition,
        result: Any,
    ):
        """Update agent profiles based on debate result."""
        if not hasattr(result, "scores") or not result.scores:
            return

        # Update success rates and ELOs
        for agent in team.agents:
            if agent.name in result.scores:
                score = result.scores[agent.name]

                # Update success rate with exponential moving average
                alpha = 0.1
                success = 1.0 if score > 0.5 else 0.5 if score > 0.3 else 0.0
                agent.success_rate = alpha * success + (1 - alpha) * agent.success_rate

        # Record to history
        self._selection_history.append({
            "task_id": team.task_id,
            "result": "success" if getattr(result, "consensus_reached", False) else "no_consensus",
            "confidence": getattr(result, "confidence", 0),
        })

    def get_leaderboard(self, domain: Optional[str] = None, limit: int = 10) -> list[dict]:
        """Get agent leaderboard."""
        agents = list(self.agent_pool.values())

        if domain:
            # Sort by domain-specific rating
            agents.sort(
                key=lambda a: a.domain_ratings.get(domain, a.elo_rating),
                reverse=True,
            )
        else:
            # Sort by overall score
            agents.sort(key=lambda a: a.overall_score, reverse=True)

        return [
            {
                "name": a.name,
                "type": a.agent_type,
                "elo": a.elo_rating,
                "domain_elo": a.domain_ratings.get(domain, a.elo_rating) if domain else None,
                "success_rate": a.success_rate,
                "overall_score": a.overall_score,
                "on_bench": a.name in self.bench,
            }
            for a in agents[:limit]
        ]

    def get_recommendations(
        self,
        requirements: TaskRequirements,
        limit: int = 5,
    ) -> list[dict]:
        """Get agent recommendations for a task."""
        candidates = list(self.agent_pool.values())
        scored = [(a, self._score_for_task(a, requirements)) for a in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "name": a.name,
                "type": a.agent_type,
                "match_score": score,
                "domain_expertise": a.expertise.get(requirements.primary_domain, 0),
                "reasoning": self._explain_match(a, requirements),
            }
            for a, score in scored[:limit]
        ]

    def _explain_match(self, agent: AgentProfile, requirements: TaskRequirements) -> str:
        """Explain why an agent matches requirements."""
        reasons = []

        exp = agent.expertise.get(requirements.primary_domain, 0)
        if exp > 0.7:
            reasons.append(f"Strong {requirements.primary_domain} expertise ({exp:.0%})")
        elif exp > 0.4:
            reasons.append(f"Moderate {requirements.primary_domain} expertise")

        if agent.elo_rating > 1600:
            reasons.append("High overall rating")

        matching_traits = [t for t in requirements.required_traits if t in agent.traits]
        if matching_traits:
            reasons.append(f"Has traits: {', '.join(matching_traits)}")

        if agent.success_rate > 0.8:
            reasons.append("Excellent success rate")

        return "; ".join(reasons) if reasons else "General purpose agent"
