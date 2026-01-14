"""
Selection Plugin Protocols - Interfaces for custom selection algorithms.

Protocols define the contracts that selection plugins must implement.
Uses Python's Protocol (structural subtyping) for flexibility.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aragora.routing.selection import AgentProfile, TaskRequirements, TeamComposition


@dataclass
class SelectionContext:
    """
    Context passed to selection plugins during team selection.

    Contains all information a plugin might need to make decisions:
    - Current agent pool and their profiles
    - Historical performance data
    - System integrations (ELO, calibration, probes)
    """

    # Core data
    agent_pool: dict[str, "AgentProfile"] = field(default_factory=dict)
    bench: list[str] = field(default_factory=list)

    # System integrations (optional)
    elo_system: Optional[Any] = None
    calibration_tracker: Optional[Any] = None
    probe_filter: Optional[Any] = None
    persona_manager: Optional[Any] = None
    performance_monitor: Optional[Any] = None

    # Cached insights
    performance_insights: dict = field(default_factory=dict)

    # Selection history for meta-learning
    selection_history: list[dict] = field(default_factory=list)

    # Plugin-specific config
    config: dict = field(default_factory=dict)


@runtime_checkable
class ScorerProtocol(Protocol):
    """
    Protocol for agent scoring algorithms.

    Scorers take an agent and task requirements and return a score
    indicating how well the agent matches the task.

    The default implementation uses ELO + domain expertise + calibration.
    Custom implementations can use ML models, rule-based systems, etc.
    """

    def score_agent(
        self,
        agent: "AgentProfile",
        requirements: "TaskRequirements",
        context: SelectionContext,
    ) -> float:
        """
        Score an agent for a specific task.

        Args:
            agent: The agent profile to score
            requirements: Task requirements (domain, traits, etc.)
            context: Selection context with system state

        Returns:
            Score between 0.0 and 1.0 (higher is better)
        """
        ...

    @property
    def name(self) -> str:
        """Unique name for this scorer."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of the scoring algorithm."""
        ...


@runtime_checkable
class TeamSelectorProtocol(Protocol):
    """
    Protocol for team composition algorithms.

    TeamSelectors take a list of scored agents and select the optimal
    team composition based on requirements and constraints.

    The default implementation balances quality with diversity.
    Custom implementations can optimize for specific objectives.
    """

    def select_team(
        self,
        scored_agents: list[tuple["AgentProfile", float]],
        requirements: "TaskRequirements",
        context: SelectionContext,
    ) -> list["AgentProfile"]:
        """
        Select a team from scored candidates.

        Args:
            scored_agents: List of (agent, score) tuples, sorted by score desc
            requirements: Task requirements including min/max team size
            context: Selection context with system state

        Returns:
            List of selected agent profiles
        """
        ...

    @property
    def name(self) -> str:
        """Unique name for this team selector."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of the selection algorithm."""
        ...


@runtime_checkable
class RoleAssignerProtocol(Protocol):
    """
    Protocol for role assignment algorithms.

    RoleAssigners take a selected team and assign debate roles
    (proposer, critic, synthesizer, etc.) based on agent capabilities.

    The default implementation uses domain expertise for role matching.
    Custom implementations can use different assignment strategies.
    """

    def assign_roles(
        self,
        team: list["AgentProfile"],
        requirements: "TaskRequirements",
        context: SelectionContext,
        phase: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Assign roles to team members.

        Args:
            team: List of selected agent profiles
            requirements: Task requirements
            context: Selection context with system state
            phase: Optional phase name for phase-specific roles

        Returns:
            Dict mapping agent names to role strings
        """
        ...

    @property
    def name(self) -> str:
        """Unique name for this role assigner."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of the role assignment algorithm."""
        ...


@runtime_checkable
class SelectionPipelineProtocol(Protocol):
    """
    Protocol for complete selection pipelines.

    Combines scoring, team selection, and role assignment into
    a single composable unit. Useful for experimental pipelines.
    """

    def select(
        self,
        requirements: "TaskRequirements",
        context: SelectionContext,
        exclude: Optional[list[str]] = None,
    ) -> "TeamComposition":
        """
        Run the complete selection pipeline.

        Args:
            requirements: Task requirements
            context: Selection context
            exclude: Agent names to exclude

        Returns:
            Complete TeamComposition with agents and roles
        """
        ...

    @property
    def name(self) -> str:
        """Unique name for this pipeline."""
        ...

    @property
    def scorer(self) -> ScorerProtocol:
        """The scorer used by this pipeline."""
        ...

    @property
    def team_selector(self) -> TeamSelectorProtocol:
        """The team selector used by this pipeline."""
        ...

    @property
    def role_assigner(self) -> RoleAssignerProtocol:
        """The role assigner used by this pipeline."""
        ...
