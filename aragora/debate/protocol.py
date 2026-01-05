"""
Debate protocol configuration and circuit breaker.

Contains:
- CircuitBreaker: Failure handling pattern for agents
- DebateProtocol: Configuration for debate execution
- user_vote_multiplier: Conviction-weighted voting calculation
"""

import logging
import time
from dataclasses import dataclass
from typing import Literal, Optional

from aragora.debate.roles import RoleRotationConfig

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for graceful agent failure handling.

    Tracks consecutive failures per agent and temporarily removes
    agents that exceed the failure threshold from debate participation.
    """

    failure_threshold: int = 3  # Consecutive failures before opening circuit
    cooldown_seconds: float = 300.0  # 5 minutes before attempting recovery
    half_open_success_threshold: int = 2  # Successes needed to fully close

    def __post_init__(self):
        self._failures: dict[str, int] = {}  # agent_name -> consecutive failures
        self._circuit_open_at: dict[str, float] = {}  # agent_name -> timestamp
        self._half_open_successes: dict[str, int] = {}  # agent_name -> recovery successes

    def record_failure(self, agent_name: str) -> bool:
        """Record an agent failure. Returns True if circuit just opened."""
        self._failures[agent_name] = self._failures.get(agent_name, 0) + 1
        self._half_open_successes[agent_name] = 0  # Reset recovery progress

        if self._failures[agent_name] >= self.failure_threshold:
            if agent_name not in self._circuit_open_at:
                self._circuit_open_at[agent_name] = time.time()
                logger.warning(
                    f"[circuit_breaker] Circuit OPEN for {agent_name} "
                    f"after {self._failures[agent_name]} failures"
                )
                return True
        return False

    def record_success(self, agent_name: str) -> None:
        """Record an agent success. May close an open circuit."""
        if agent_name in self._circuit_open_at:
            # In half-open state, track recovery successes
            self._half_open_successes[agent_name] = (
                self._half_open_successes.get(agent_name, 0) + 1
            )
            if self._half_open_successes[agent_name] >= self.half_open_success_threshold:
                # Fully close the circuit
                del self._circuit_open_at[agent_name]
                self._failures[agent_name] = 0
                self._half_open_successes[agent_name] = 0
                logger.info(f"[circuit_breaker] Circuit CLOSED for {agent_name}")
        else:
            # Reset failure count on success
            self._failures[agent_name] = 0

    def is_available(self, agent_name: str) -> bool:
        """Check if an agent is available for use."""
        if agent_name not in self._circuit_open_at:
            return True

        # Check if cooldown has passed (half-open state)
        elapsed = time.time() - self._circuit_open_at[agent_name]
        if elapsed >= self.cooldown_seconds:
            return True  # Allow trial request in half-open state

        return False

    def get_status(self, agent_name: str) -> str:
        """Get circuit status: 'closed', 'open', or 'half-open'."""
        if agent_name not in self._circuit_open_at:
            return "closed"
        elapsed = time.time() - self._circuit_open_at[agent_name]
        if elapsed >= self.cooldown_seconds:
            return "half-open"
        return "open"

    def filter_available_agents(self, agents: list) -> list:
        """Return only agents with closed or half-open circuits."""
        return [a for a in agents if self.is_available(a.name)]

    def to_dict(self) -> dict:
        """Serialize to dict for persistence."""
        return {
            "failures": self._failures.copy(),
            "cooldowns": {
                name: time.time() - ts
                for name, ts in self._circuit_open_at.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> "CircuitBreaker":
        """Load from persisted dict."""
        cb = cls(**kwargs)
        cb._failures = data.get("failures", {})
        # Restore open circuits with remaining cooldown
        for name, elapsed in data.get("cooldowns", {}).items():
            if elapsed < cb.cooldown_seconds:
                cb._circuit_open_at[name] = time.time() - elapsed
        return cb


@dataclass
class DebateProtocol:
    """Configuration for how debates are conducted."""

    topology: Literal["all-to-all", "sparse", "round-robin", "ring", "star", "random-graph"] = "round-robin"
    topology_sparsity: float = 0.5  # fraction of possible critique connections (for sparse/random-graph)
    topology_hub_agent: Optional[str] = None  # for star topology, which agent is the hub
    rounds: int = 5  # Increased from 3 for more thorough debates
    consensus: Literal["majority", "unanimous", "judge", "none"] = "majority"
    consensus_threshold: float = 0.6  # fraction needed for majority
    allow_abstain: bool = True
    require_reasoning: bool = True

    # Role assignments
    proposer_count: int = 1  # how many agents propose initially
    critic_count: int = -1  # -1 means all non-proposers critique

    # Judge selection (for consensus="judge" mode)
    # "elo_ranked" selects highest ELO-rated agent from EloSystem (requires elo_system param)
    judge_selection: Literal["random", "voted", "last", "elo_ranked"] = "random"

    # Agreement intensity (0-10): Controls how much agents agree vs disagree
    # 0 = strongly disagree 100% of the time (adversarial)
    # 5 = balanced (agree/disagree based on argument quality)
    # 10 = fully incorporate others' opinions (collaborative)
    # Research shows intensity=2 (slight disagreement bias) often improves accuracy
    agreement_intensity: int = 5

    # Early stopping: End debate when agents agree further rounds won't help
    # Based on ai-counsel pattern - can save 40-70% API costs
    early_stopping: bool = True
    early_stop_threshold: float = 0.66  # fraction of agents saying stop to trigger
    min_rounds_before_early_stop: int = 2  # minimum rounds before allowing early exit

    # Asymmetric debate roles: Assign affirmative/negative/neutral stances
    # Forces perspective diversity, prevents premature consensus
    asymmetric_stances: bool = False  # Enable asymmetric stance assignment
    rotate_stances: bool = True  # Rotate stances between rounds

    # Semantic convergence detection
    # Auto-detect consensus without explicit voting
    convergence_detection: bool = True
    convergence_threshold: float = 0.85  # Similarity for convergence
    divergence_threshold: float = 0.40  # Below this is diverging

    # Vote option grouping: Merge semantically similar vote choices
    # Prevents artificial disagreement from wording variations
    vote_grouping: bool = True
    vote_grouping_threshold: float = 0.85  # Similarity to merge options

    # Judge-based termination: Single judge decides when debate is conclusive
    # Different from early_stopping (agent votes) - uses a designated judge
    judge_termination: bool = False
    min_rounds_before_judge_check: int = 2  # Check only after this many rounds

    # Human participation settings
    user_vote_weight: float = 0.5  # Weight of user votes relative to agent votes (0.5 = half weight)

    # Conviction-weighted voting (intensity 1-10 scale)
    # User votes with high conviction (8-10) count more than low conviction (1-3)
    user_vote_intensity_scale: int = 10  # Max intensity value
    user_vote_intensity_neutral: int = 5  # Neutral intensity (multiplier = 1.0)
    user_vote_intensity_min_multiplier: float = 0.5  # Multiplier at intensity=1
    user_vote_intensity_max_multiplier: float = 2.0  # Multiplier at intensity=10

    # Audience suggestion injection
    audience_injection: Literal["off", "summary", "inject"] = "off"

    # Pre-debate web research
    enable_research: bool = True  # Enable web research before debates

    # Cognitive role rotation (Heavy3-inspired)
    # Assigns different cognitive roles (Analyst, Skeptic, Lateral Thinker, Synthesizer)
    # to each agent per round, ensuring diverse perspectives
    role_rotation: bool = True  # Enable role rotation (cognitive diversity)
    role_rotation_config: Optional[RoleRotationConfig] = None  # Custom role config

    # Debate timeout (seconds) - prevents runaway debates
    # 0 = no timeout (default for backward compatibility)
    timeout_seconds: int = 0  # Max time for entire debate (0 = unlimited)
    round_timeout_seconds: int = 120  # Max time per round (2 minutes per round)


def user_vote_multiplier(intensity: int, protocol: DebateProtocol) -> float:
    """
    Calculate conviction-weighted vote multiplier based on intensity.

    Args:
        intensity: User's conviction level (1-10)
        protocol: DebateProtocol with intensity scaling parameters

    Returns:
        Bounded weight multiplier between min_multiplier and max_multiplier
    """
    # Normalize intensity to 1-10 range
    intensity = max(1, min(protocol.user_vote_intensity_scale, intensity))

    # Calculate position relative to neutral (0 = neutral, negative = low, positive = high)
    neutral = protocol.user_vote_intensity_neutral
    scale = protocol.user_vote_intensity_scale

    if intensity == neutral:
        return 1.0

    if intensity < neutral:
        # Below neutral: interpolate between min_multiplier and 1.0
        # intensity=1 -> min_multiplier, intensity=neutral -> 1.0
        ratio = (intensity - 1) / (neutral - 1) if neutral > 1 else 0
        return protocol.user_vote_intensity_min_multiplier + (1.0 - protocol.user_vote_intensity_min_multiplier) * ratio
    else:
        # Above neutral: interpolate between 1.0 and max_multiplier
        # intensity=neutral -> 1.0, intensity=scale -> max_multiplier
        ratio = (intensity - neutral) / (scale - neutral) if scale > neutral else 0
        return 1.0 + (protocol.user_vote_intensity_max_multiplier - 1.0) * ratio
