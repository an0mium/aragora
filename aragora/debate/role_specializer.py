"""
A-HMAD: Adaptive Hierarchical Multi-Agent Debate Role Specialization.

Based on: A-HMAD research on dynamic role assignment in multi-agent systems.

Replaces static domain mapping with learned dynamic role specialization
that adapts to specific debate topics and agent performance history.

Key features:
- Topic analysis to determine required roles
- Agent capability profiling from ELO/calibration history
- Dynamic agent-role matching with diversity enforcement
- Fallback to static mapping when insufficient data
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Protocol
from enum import Enum
import logging
import time
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class RoleType(Enum):
    """Types of roles agents can play in a debate."""

    PROPOSER = "proposer"  # Generates initial proposals
    CRITIC = "critic"  # Critiques and finds weaknesses
    SYNTHESIZER = "synthesizer"  # Combines viewpoints
    FACT_CHECKER = "fact_checker"  # Verifies claims
    DEVIL_ADVOCATE = "devil_advocate"  # Argues against consensus
    DOMAIN_EXPERT = "domain_expert"  # Deep domain knowledge
    GENERALIST = "generalist"  # Broad knowledge
    MEDIATOR = "mediator"  # Resolves conflicts


@dataclass
class AgentCapability:
    """Capability profile for an agent."""

    agent_id: str
    elo_rating: float = 1500.0
    brier_score: float = 0.25  # Lower is better
    domain_scores: dict[str, float] = field(default_factory=dict)
    role_performance: dict[str, float] = field(default_factory=dict)
    total_debates: int = 0
    recent_accuracy: float = 0.5


@dataclass
class RoleRequirement:
    """A required role for a debate."""

    role_type: RoleType
    importance: float  # 0.0-1.0
    domain_preference: Optional[str] = None
    min_agents: int = 1
    max_agents: int = 1


@dataclass
class RoleAssignment:
    """Assignment of an agent to a role."""

    agent_id: str
    role: RoleType
    confidence: float
    reasoning: str
    is_fallback: bool = False


@dataclass
class TeamComposition:
    """Complete team composition with role assignments."""

    assignments: list[RoleAssignment]
    diversity_score: float
    coverage_score: float  # How well roles cover requirements
    total_capability_score: float
    assignment_time_ms: float = 0.0


@dataclass
class AHMADConfig:
    """Configuration for A-HMAD role specialization."""

    # Role weights
    elo_weight: float = 0.3
    calibration_weight: float = 0.25
    domain_weight: float = 0.25
    role_history_weight: float = 0.2

    # Diversity enforcement
    min_diversity_score: float = 0.6
    diversity_penalty: float = 0.3  # Penalty for duplicate agents

    # Role analysis
    default_roles: list[RoleType] = field(
        default_factory=lambda: [
            RoleType.PROPOSER,
            RoleType.CRITIC,
            RoleType.SYNTHESIZER,
        ]
    )

    # Topic keywords for role determination
    topic_role_triggers: dict[str, list[RoleType]] = field(
        default_factory=lambda: {
            "fact": [RoleType.FACT_CHECKER],
            "verify": [RoleType.FACT_CHECKER],
            "compare": [RoleType.CRITIC, RoleType.SYNTHESIZER],
            "debate": [RoleType.DEVIL_ADVOCATE],
            "controversial": [RoleType.DEVIL_ADVOCATE],
            "technical": [RoleType.DOMAIN_EXPERT],
            "expert": [RoleType.DOMAIN_EXPERT],
            "combine": [RoleType.SYNTHESIZER],
            "resolve": [RoleType.MEDIATOR],
            "conflict": [RoleType.MEDIATOR],
        }
    )

    # Fallback mapping (used when no historical data)
    static_fallback: dict[str, list[str]] = field(
        default_factory=lambda: {
            "proposer": ["claude", "gpt", "gemini"],
            "critic": ["claude", "deepseek-r1", "gpt"],
            "synthesizer": ["claude", "gpt", "gemini"],
            "fact_checker": ["claude", "deepseek-r1"],
            "devil_advocate": ["gpt", "llama", "mistral"],
            "domain_expert": ["claude", "deepseek-r1", "gemini"],
            "generalist": ["claude", "gpt", "gemini", "llama"],
            "mediator": ["claude", "gpt"],
        }
    )


class CalibrationProtocol(Protocol):
    """Protocol for calibration data access."""

    def get_brier_score(self, agent_id: str, domain: Optional[str] = None) -> float:
        """Get agent's Brier score."""
        ...


class EloProtocol(Protocol):
    """Protocol for ELO rating access."""

    def get_rating(self, agent_id: str) -> float:
        """Get agent's ELO rating."""
        ...


class AHMADRoleSpecializer:
    """
    Dynamic role specialization for multi-agent debates.

    Analyzes debate topic to determine required roles, then matches
    available agents to roles based on historical performance and
    capability profiles.

    Example:
        specializer = AHMADRoleSpecializer()

        # Analyze topic and get required roles
        roles = specializer.analyze_topic(
            topic="Compare the economic impacts of renewable vs fossil fuels",
            domain="economics",
        )

        # Get team composition
        team = specializer.assign_roles(
            roles=roles,
            available_agents=["claude", "gpt-4", "gemini", "deepseek-r1"],
            elo_scores={"claude": 1600, "gpt-4": 1580, ...},
            calibration_scores={"claude": 0.15, "gpt-4": 0.18, ...},
        )

        for assignment in team.assignments:
            print(f"{assignment.agent_id} -> {assignment.role.value}")
    """

    def __init__(self, config: Optional[AHMADConfig] = None):
        """Initialize the specializer.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or AHMADConfig()
        self._assignment_history: list[TeamComposition] = []
        self._agent_role_history: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def analyze_topic(
        self,
        topic: str,
        domain: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> list[RoleRequirement]:
        """
        Analyze a debate topic to determine required roles.

        Args:
            topic: The debate topic/question
            domain: Optional domain classification
            context: Optional additional context

        Returns:
            List of RoleRequirements for the debate
        """
        topic_lower = topic.lower()
        requirements: list[RoleRequirement] = []
        added_roles: set[RoleType] = set()

        # Check topic for role triggers
        for keyword, roles in self.config.topic_role_triggers.items():
            if keyword in topic_lower:
                for role in roles:
                    if role not in added_roles:
                        importance = 0.8 if keyword in topic_lower[:50] else 0.6
                        requirements.append(
                            RoleRequirement(
                                role_type=role,
                                importance=importance,
                                domain_preference=domain,
                            )
                        )
                        added_roles.add(role)

        # Always include default roles
        for role in self.config.default_roles:
            if role not in added_roles:
                requirements.append(
                    RoleRequirement(
                        role_type=role,
                        importance=0.7,
                        domain_preference=domain,
                    )
                )
                added_roles.add(role)

        # Sort by importance
        requirements.sort(key=lambda r: r.importance, reverse=True)

        logger.debug(
            "topic_analyzed topic_len=%d roles=%d domain=%s",
            len(topic),
            len(requirements),
            domain,
        )

        return requirements

    def assign_roles(
        self,
        roles: list[RoleRequirement],
        available_agents: list[str],
        elo_scores: Optional[dict[str, float]] = None,
        calibration_scores: Optional[dict[str, float]] = None,
        domain_scores: Optional[dict[str, dict[str, float]]] = None,
    ) -> TeamComposition:
        """
        Assign agents to roles based on capabilities.

        Args:
            roles: Required roles from analyze_topic
            available_agents: List of available agent IDs
            elo_scores: Optional ELO ratings per agent
            calibration_scores: Optional Brier scores per agent
            domain_scores: Optional domain-specific scores

        Returns:
            TeamComposition with role assignments
        """
        start_time = time.time()

        if not available_agents:
            return TeamComposition(
                assignments=[],
                diversity_score=0.0,
                coverage_score=0.0,
                total_capability_score=0.0,
                assignment_time_ms=0.0,
            )

        elo_scores = elo_scores or {}
        calibration_scores = calibration_scores or {}
        domain_scores = domain_scores or {}

        # Build capability profiles
        capabilities = self._build_capabilities(
            available_agents,
            elo_scores,
            calibration_scores,
            domain_scores,
        )

        # Assign agents to roles
        assignments: list[RoleAssignment] = []
        assigned_agents: set[str] = set()

        for role_req in roles:
            # Score agents for this role
            agent_scores = self._score_agents_for_role(
                role_req=role_req,
                capabilities=capabilities,
                assigned_agents=assigned_agents,
            )

            # Select best agent(s)
            for _ in range(role_req.min_agents):
                if not agent_scores:
                    break

                # Get best scoring agent
                best_agent, best_score = max(agent_scores.items(), key=lambda x: x[1])

                # Check if we should use fallback
                is_fallback = best_score < 0.3

                assignments.append(
                    RoleAssignment(
                        agent_id=best_agent,
                        role=role_req.role_type,
                        confidence=best_score,
                        reasoning=self._generate_reasoning(
                            best_agent, role_req.role_type, best_score
                        ),
                        is_fallback=is_fallback,
                    )
                )

                assigned_agents.add(best_agent)
                del agent_scores[best_agent]

        # Calculate team metrics
        diversity_score = self._calculate_diversity(assignments)
        coverage_score = self._calculate_coverage(assignments, roles)
        capability_score = self._calculate_total_capability(assignments)

        assignment_time_ms = (time.time() - start_time) * 1000

        composition = TeamComposition(
            assignments=assignments,
            diversity_score=diversity_score,
            coverage_score=coverage_score,
            total_capability_score=capability_score,
            assignment_time_ms=assignment_time_ms,
        )

        self._assignment_history.append(composition)

        logger.info(
            "roles_assigned agents=%d roles=%d diversity=%.2f coverage=%.2f time_ms=%.1f",
            len(assignments),
            len(roles),
            diversity_score,
            coverage_score,
            assignment_time_ms,
        )

        return composition

    def _build_capabilities(
        self,
        agents: list[str],
        elo_scores: dict[str, float],
        calibration_scores: dict[str, float],
        domain_scores: dict[str, dict[str, float]],
    ) -> dict[str, AgentCapability]:
        """Build capability profiles for agents."""
        capabilities = {}

        for agent_id in agents:
            # Get historical role performance
            role_perf = {}
            for role, scores in self._agent_role_history[agent_id].items():
                if scores:
                    role_perf[role] = np.mean(scores[-10:])  # Last 10 performances

            capabilities[agent_id] = AgentCapability(
                agent_id=agent_id,
                elo_rating=elo_scores.get(agent_id, 1500.0),
                brier_score=calibration_scores.get(agent_id, 0.25),
                domain_scores=domain_scores.get(agent_id, {}),
                role_performance=role_perf,
            )

        return capabilities

    def _score_agents_for_role(
        self,
        role_req: RoleRequirement,
        capabilities: dict[str, AgentCapability],
        assigned_agents: set[str],
    ) -> dict[str, float]:
        """Score all agents for a specific role."""
        scores = {}
        role_name = role_req.role_type.value

        for agent_id, cap in capabilities.items():
            score = 0.0

            # ELO component (normalized to 0-1)
            elo_norm = (cap.elo_rating - 1000) / 1000  # Assume range 1000-2000
            elo_norm = max(0.0, min(1.0, elo_norm))
            score += self.config.elo_weight * elo_norm

            # Calibration component (inverted, lower is better)
            cal_score = 1.0 - min(cap.brier_score, 0.5) * 2  # 0 brier = 1.0, 0.5 brier = 0.0
            score += self.config.calibration_weight * cal_score

            # Domain expertise component
            if role_req.domain_preference and cap.domain_scores:
                domain_score = cap.domain_scores.get(role_req.domain_preference, 0.5)
                score += self.config.domain_weight * domain_score
            else:
                score += self.config.domain_weight * 0.5  # Neutral if no domain

            # Role history component
            if role_name in cap.role_performance:
                role_score = cap.role_performance[role_name]
                score += self.config.role_history_weight * role_score
            else:
                # Check static fallback
                fallback_agents = self.config.static_fallback.get(role_name, [])
                if any(fa in agent_id.lower() for fa in fallback_agents):
                    score += self.config.role_history_weight * 0.7
                else:
                    score += self.config.role_history_weight * 0.4

            # Diversity penalty if already assigned
            if agent_id in assigned_agents:
                score *= 1.0 - self.config.diversity_penalty

            scores[agent_id] = score

        return scores

    def _generate_reasoning(
        self,
        agent_id: str,
        role: RoleType,
        score: float,
    ) -> str:
        """Generate reasoning for assignment."""
        if score >= 0.7:
            return f"Strong match: {agent_id} has excellent performance history for {role.value}"
        elif score >= 0.5:
            return f"Good match: {agent_id} shows solid capability for {role.value}"
        elif score >= 0.3:
            return f"Adequate match: {agent_id} can fulfill {role.value} with some limitations"
        else:
            return f"Fallback: {agent_id} assigned to {role.value} due to limited options"

    def _calculate_diversity(self, assignments: list[RoleAssignment]) -> float:
        """Calculate team diversity score."""
        if not assignments:
            return 0.0

        unique_agents = len(set(a.agent_id for a in assignments))
        unique_roles = len(set(a.role for a in assignments))

        agent_diversity = unique_agents / len(assignments)
        role_diversity = unique_roles / len(RoleType)

        return 0.6 * agent_diversity + 0.4 * role_diversity

    def _calculate_coverage(
        self,
        assignments: list[RoleAssignment],
        requirements: list[RoleRequirement],
    ) -> float:
        """Calculate how well assignments cover requirements."""
        if not requirements:
            return 1.0

        assigned_roles = set(a.role for a in assignments)
        required_roles = set(r.role_type for r in requirements)

        covered = len(assigned_roles & required_roles)
        total = len(required_roles)

        return covered / total if total > 0 else 0.0

    def _calculate_total_capability(
        self,
        assignments: list[RoleAssignment],
    ) -> float:
        """Calculate aggregate capability score."""
        if not assignments:
            return 0.0

        return np.mean([a.confidence for a in assignments])

    def record_performance(
        self,
        agent_id: str,
        role: RoleType,
        performance_score: float,
    ) -> None:
        """
        Record agent performance in a role for future assignments.

        Args:
            agent_id: The agent ID
            role: The role they played
            performance_score: Score from 0.0-1.0
        """
        self._agent_role_history[agent_id][role.value].append(performance_score)

        # Keep only recent history
        if len(self._agent_role_history[agent_id][role.value]) > 50:
            self._agent_role_history[agent_id][role.value] = self._agent_role_history[agent_id][
                role.value
            ][-50:]

    def reset(self) -> None:
        """Reset specializer state."""
        self._assignment_history.clear()
        self._agent_role_history.clear()

    def get_metrics(self) -> dict[str, Any]:
        """Get specializer metrics for telemetry."""
        if not self._assignment_history:
            return {
                "total_assignments": 0,
                "avg_diversity": 0.0,
                "avg_coverage": 0.0,
                "avg_capability": 0.0,
            }

        total = len(self._assignment_history)

        return {
            "total_assignments": total,
            "avg_diversity": np.mean([c.diversity_score for c in self._assignment_history]),
            "avg_coverage": np.mean([c.coverage_score for c in self._assignment_history]),
            "avg_capability": np.mean([c.total_capability_score for c in self._assignment_history]),
            "avg_time_ms": np.mean([c.assignment_time_ms for c in self._assignment_history]),
            "agents_tracked": len(self._agent_role_history),
        }


# Convenience functions


def create_role_specializer(
    min_diversity: float = 0.6,
    **kwargs: Any,
) -> AHMADRoleSpecializer:
    """Create an A-HMAD role specializer with common configuration.

    Args:
        min_diversity: Minimum diversity score to enforce
        **kwargs: Additional config options

    Returns:
        Configured AHMADRoleSpecializer
    """
    config = AHMADConfig(min_diversity_score=min_diversity, **kwargs)
    return AHMADRoleSpecializer(config)


def quick_assign_roles(
    topic: str,
    available_agents: list[str],
    elo_scores: Optional[dict[str, float]] = None,
) -> list[tuple[str, str]]:
    """
    Quick role assignment without full configuration.

    Args:
        topic: Debate topic
        available_agents: Available agent IDs
        elo_scores: Optional ELO ratings

    Returns:
        List of (agent_id, role_name) tuples
    """
    specializer = AHMADRoleSpecializer()
    roles = specializer.analyze_topic(topic)
    team = specializer.assign_roles(
        roles=roles,
        available_agents=available_agents,
        elo_scores=elo_scores,
    )
    return [(a.agent_id, a.role.value) for a in team.assignments]
