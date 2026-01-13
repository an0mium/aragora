"""
Agent Pool for Aragora Debates.

Manages agent lifecycle, selection, and performance-based team composition.
Extracted from Arena to enable cleaner agent management and testing.

Usage:
    from aragora.debate.agent_pool import AgentPool, AgentPoolConfig

    # Create pool with agents
    config = AgentPoolConfig(
        elo_system=elo,
        calibration_tracker=calibration,
        use_performance_selection=True,
    )
    pool = AgentPool(agents, config)

    # Select team for debate
    team = pool.select_team(domain="software", team_size=3)

    # Select critics for a proposal
    critics = pool.select_critics(proposer, all_agents)
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class AgentPoolConfig:
    """Configuration for agent pool behavior."""

    # Performance tracking
    elo_system: Optional[Any] = None  # EloSystem
    calibration_tracker: Optional[Any] = None  # CalibrationTracker
    circuit_breaker: Optional[Any] = None  # CircuitBreaker

    # Selection behavior
    use_performance_selection: bool = False
    performance_weight: float = 0.7  # Weight for ELO vs calibration
    min_team_size: int = 2
    max_team_size: int = 10

    # Topology for critic selection
    topology: str = "full_mesh"  # full_mesh, ring, star
    critic_count: int = 2  # Default critics per proposal


@dataclass
class AgentMetrics:
    """Metrics for a single agent."""

    name: str
    elo_rating: float = 1000.0
    calibration_score: float = 0.5
    debates_participated: int = 0
    win_rate: float = 0.5
    is_available: bool = True


class AgentPool:
    """
    Manages agent lifecycle and team selection for debates.

    Features:
    - Performance-based team selection (ELO + calibration)
    - Topology-based critic selection
    - Circuit breaker integration for fault tolerance
    - Agent metrics tracking
    """

    def __init__(
        self,
        agents: List[Any],  # List[Agent]
        config: Optional[AgentPoolConfig] = None,
    ):
        """
        Initialize the agent pool.

        Args:
            agents: List of available agents
            config: Optional configuration
        """
        self._agents = list(agents)
        self._config = config or AgentPoolConfig()

        # Agent name to agent mapping
        self._agent_map: Dict[str, Any] = {
            getattr(a, "name", str(a)): a for a in self._agents
        }

        # Metrics cache
        self._metrics: Dict[str, AgentMetrics] = {}
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize metrics for all agents."""
        for agent in self._agents:
            name = getattr(agent, "name", str(agent))
            self._metrics[name] = AgentMetrics(name=name)

    # =========================================================================
    # Agent Access
    # =========================================================================

    @property
    def agents(self) -> List[Any]:
        """Get all agents in the pool."""
        return self._agents.copy()

    @property
    def available_agents(self) -> List[Any]:
        """Get all available (non-circuit-broken) agents."""
        if self._config.circuit_breaker is None:
            return self._agents.copy()

        return [
            a for a in self._agents
            if not self._is_circuit_broken(getattr(a, "name", str(a)))
        ]

    def get_agent(self, name: str) -> Optional[Any]:
        """Get an agent by name."""
        return self._agent_map.get(name)

    def require_agents(self, min_count: int = 1) -> List[Any]:
        """
        Get available agents, raising if insufficient.

        Extracted from Arena._require_agents().

        Args:
            min_count: Minimum number of agents required

        Returns:
            List of available agents

        Raises:
            ValueError: If insufficient agents available
        """
        available = self.available_agents
        if len(available) < min_count:
            raise ValueError(
                f"Insufficient agents: need {min_count}, have {len(available)}. "
                f"Check circuit breaker status or add more agents."
            )
        return available

    def _is_circuit_broken(self, agent_name: str) -> bool:
        """Check if an agent's circuit breaker is open."""
        if self._config.circuit_breaker is None:
            return False

        try:
            return self._config.circuit_breaker.is_open(agent_name)
        except (KeyError, AttributeError, TypeError) as e:
            logger.debug(f"Circuit breaker check failed for {agent_name}: {e}")
            return False

    # =========================================================================
    # Team Selection
    # =========================================================================

    def select_team(
        self,
        domain: str = "",
        team_size: Optional[int] = None,
        exclude: Optional[Set[str]] = None,
    ) -> List[Any]:
        """
        Select a team of agents for a debate.

        Extracted from Arena._select_debate_team().

        Uses performance-based selection when enabled:
        - Combines ELO rating and calibration score
        - Weights by domain expertise if available
        - Falls back to random selection

        Args:
            domain: Optional domain for expertise weighting
            team_size: Number of agents to select (uses config default)
            exclude: Agent names to exclude from selection

        Returns:
            List of selected agents
        """
        exclude = exclude or set()
        available = [
            a for a in self.available_agents
            if getattr(a, "name", str(a)) not in exclude
        ]

        if not available:
            logger.warning("No available agents for team selection")
            return []

        # Determine team size
        size = team_size or min(
            len(available),
            self._config.max_team_size,
        )
        size = max(size, self._config.min_team_size)
        size = min(size, len(available))

        # Performance-based selection
        if self._config.use_performance_selection:
            return self._select_by_performance(available, size, domain)

        # Random selection fallback
        return random.sample(available, size)

    def _select_by_performance(
        self,
        agents: List[Any],
        count: int,
        domain: str = "",
    ) -> List[Any]:
        """Select agents based on composite performance score."""
        scores: List[tuple[Any, float]] = []

        for agent in agents:
            name = getattr(agent, "name", str(agent))
            score = self._compute_composite_score(name, domain)
            scores.append((agent, score))

        # Sort by score descending and take top N
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [agent for agent, _ in scores[:count]]

        logger.debug(
            f"Selected team by performance: {[getattr(a, 'name', str(a)) for a in selected]}"
        )
        return selected

    def _compute_composite_score(
        self,
        agent_name: str,
        domain: str = "",
    ) -> float:
        """
        Compute composite score for agent selection.

        Extracted from Arena._compute_composite_judge_score().

        Args:
            agent_name: Name of the agent
            domain: Optional domain for expertise weighting

        Returns:
            Composite score (0-2000+ range)
        """
        # Base ELO score
        elo_score = 1000.0
        if self._config.elo_system is not None:
            try:
                elo_score = self._config.elo_system.get_rating(agent_name)
            except (KeyError, AttributeError, TypeError, ValueError) as e:
                logger.debug(f"ELO rating lookup failed for {agent_name}: {e}")

        # Calibration weight
        calibration_weight = self._get_calibration_weight(agent_name)

        # Combine: ELO * calibration_weight
        composite = elo_score * calibration_weight

        # Domain expertise bonus (if available)
        if domain and self._config.elo_system is not None:
            try:
                domain_rating = self._config.elo_system.get_domain_rating(
                    agent_name, domain
                )
                if domain_rating:
                    composite = (composite + domain_rating) / 2
            except (KeyError, AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Domain rating lookup failed for {agent_name}/{domain}: {e}")

        return composite

    def _get_calibration_weight(self, agent_name: str) -> float:
        """
        Get calibration weight for an agent.

        Extracted from Arena._get_calibration_weight().

        Args:
            agent_name: Name of the agent

        Returns:
            Calibration weight (0.5 - 1.5 range)
        """
        if self._config.calibration_tracker is None:
            return 1.0

        try:
            calibration = self._config.calibration_tracker.get_calibration(agent_name)
            if calibration is None:
                return 1.0
            # Map calibration (0-1) to weight (0.5-1.5)
            return 0.5 + calibration
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Calibration lookup failed for {agent_name}: {e}")
            return 1.0

    # =========================================================================
    # Critic Selection
    # =========================================================================

    def select_critics(
        self,
        proposer: Any,
        candidates: Optional[List[Any]] = None,
        count: Optional[int] = None,
    ) -> List[Any]:
        """
        Select critics for a proposal based on topology.

        Extracted from Arena._select_critics_for_proposal().

        Args:
            proposer: The agent who made the proposal
            candidates: Optional list of candidate critics (defaults to all)
            count: Number of critics to select (uses config default)

        Returns:
            List of selected critic agents
        """
        proposer_name = getattr(proposer, "name", str(proposer))
        candidates = candidates or self.available_agents
        count = count or self._config.critic_count

        # Exclude proposer from critics
        critics_pool = [
            a for a in candidates
            if getattr(a, "name", str(a)) != proposer_name
        ]

        if not critics_pool:
            return []

        # Apply topology-based selection
        if self._config.topology == "ring":
            return self._select_ring_critics(proposer_name, critics_pool, count)
        elif self._config.topology == "star":
            return self._select_star_critics(proposer_name, critics_pool, count)
        else:
            # Full mesh: any agent can critique any other
            return self._select_mesh_critics(critics_pool, count)

    def _select_mesh_critics(
        self,
        pool: List[Any],
        count: int,
    ) -> List[Any]:
        """Select critics using full mesh topology (random selection)."""
        count = min(count, len(pool))
        return random.sample(pool, count)

    def _select_ring_critics(
        self,
        proposer_name: str,
        pool: List[Any],
        count: int,
    ) -> List[Any]:
        """Select critics using ring topology (neighbors only)."""
        # Find proposer index in original agent list
        agent_names = [getattr(a, "name", str(a)) for a in self._agents]
        try:
            proposer_idx = agent_names.index(proposer_name)
        except ValueError:
            return self._select_mesh_critics(pool, count)

        # Get neighbors in ring
        n = len(self._agents)
        neighbor_indices = [
            (proposer_idx - 1) % n,
            (proposer_idx + 1) % n,
        ]

        neighbors = []
        for idx in neighbor_indices:
            agent = self._agents[idx]
            if agent in pool:
                neighbors.append(agent)

        # If not enough neighbors, fall back to mesh
        if len(neighbors) < count:
            remaining = [a for a in pool if a not in neighbors]
            neighbors.extend(
                random.sample(remaining, min(count - len(neighbors), len(remaining)))
            )

        return neighbors[:count]

    def _select_star_critics(
        self,
        proposer_name: str,
        pool: List[Any],
        count: int,
    ) -> List[Any]:
        """Select critics using star topology (hub critiques all)."""
        # First agent is the hub
        if not self._agents:
            return []

        hub = self._agents[0]
        hub_name = getattr(hub, "name", str(hub))

        if proposer_name == hub_name:
            # Hub proposed, any can critique
            return self._select_mesh_critics(pool, count)
        else:
            # Non-hub proposed, hub must critique
            hub_in_pool = [a for a in pool if getattr(a, "name", str(a)) == hub_name]
            if hub_in_pool:
                return hub_in_pool[:1]
            return self._select_mesh_critics(pool, count)

    # =========================================================================
    # Metrics & Status
    # =========================================================================

    def update_metrics(
        self,
        agent_name: str,
        elo_rating: Optional[float] = None,
        calibration_score: Optional[float] = None,
        debate_participated: bool = False,
        won: bool = False,
    ) -> None:
        """
        Update metrics for an agent.

        Args:
            agent_name: Name of the agent
            elo_rating: New ELO rating
            calibration_score: New calibration score
            debate_participated: Whether agent participated in a debate
            won: Whether agent won the debate
        """
        if agent_name not in self._metrics:
            self._metrics[agent_name] = AgentMetrics(name=agent_name)

        metrics = self._metrics[agent_name]
        if elo_rating is not None:
            metrics.elo_rating = elo_rating
        if calibration_score is not None:
            metrics.calibration_score = calibration_score
        if debate_participated:
            metrics.debates_participated += 1
            if won:
                # Update win rate
                total = metrics.debates_participated
                wins = metrics.win_rate * (total - 1) + (1 if won else 0)
                metrics.win_rate = wins / total

    def get_agent_metrics(self, agent_name: str) -> Optional[AgentMetrics]:
        """Get metrics for an agent."""
        return self._metrics.get(agent_name)

    def set_scoring_systems(
        self,
        elo_system: Optional[Any] = None,
        calibration_tracker: Optional[Any] = None,
    ) -> None:
        """
        Update scoring systems for performance-based selection.

        Called after initialization when ELO/calibration systems become available.

        Args:
            elo_system: Optional EloSystem for agent ratings
            calibration_tracker: Optional CalibrationTracker for prediction accuracy
        """
        if elo_system:
            self._config.elo_system = elo_system
            self._config.use_performance_selection = True
        if calibration_tracker:
            self._config.calibration_tracker = calibration_tracker

        # Refresh metrics with new scoring systems
        if elo_system or calibration_tracker:
            self._refresh_metrics_from_systems()

    def _refresh_metrics_from_systems(self) -> None:
        """Refresh agent metrics from ELO/calibration systems."""
        for agent in self._agents:
            name = getattr(agent, "name", str(agent))
            metrics = self._metrics.get(name)
            if not metrics:
                continue

            # Update ELO rating
            if self._config.elo_system:
                try:
                    rating = self._config.elo_system.get_rating(name)
                    metrics.elo_rating = rating.elo
                except (KeyError, AttributeError):
                    pass

            # Update calibration score
            if self._config.calibration_tracker:
                try:
                    cal = self._config.calibration_tracker.get_calibration(name)
                    if hasattr(cal, "brier_score"):
                        metrics.calibration_score = 1.0 - min(cal.brier_score, 1.0)
                except (KeyError, AttributeError):
                    pass

    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get overall pool status.

        Returns:
            Dictionary with pool statistics
        """
        available = self.available_agents
        return {
            "total_agents": len(self._agents),
            "available_agents": len(available),
            "circuit_broken": len(self._agents) - len(available),
            "topology": self._config.topology,
            "performance_selection": self._config.use_performance_selection,
            "agents": [
                {
                    "name": getattr(a, "name", str(a)),
                    "available": a in available,
                    "metrics": self._metrics.get(getattr(a, "name", str(a))),
                }
                for a in self._agents
            ],
        }


__all__ = [
    "AgentPool",
    "AgentPoolConfig",
    "AgentMetrics",
]
