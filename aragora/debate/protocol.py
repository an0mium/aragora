"""
Debate protocol configuration.

Contains:
- DebateProtocol: Configuration for debate execution
- user_vote_multiplier: Conviction-weighted voting calculation

Note: CircuitBreaker is now in aragora.resilience module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

from aragora.config import DEBATE_TIMEOUT_SECONDS, AGENT_TIMEOUT_SECONDS
from aragora.debate.roles import RoleRotationConfig
from aragora.debate.role_matcher import RoleMatchingConfig
from aragora.resilience import CircuitBreaker  # Re-export for backwards compatibility

if TYPE_CHECKING:
    from aragora.debate.breakpoints import BreakpointConfig

logger = logging.getLogger(__name__)

# Re-export CircuitBreaker for backwards compatibility
__all__ = ["CircuitBreaker", "DebateProtocol", "user_vote_multiplier"]


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

    # Dynamic role matching (calibration-based)
    # Uses agent calibration scores and expertise to assign optimal roles
    # Overrides simple rotation when enabled
    role_matching: bool = False  # Enable calibration-based role matching
    role_matching_config: Optional[RoleMatchingConfig] = None  # Custom matching config

    # Debate timeout (seconds) - prevents runaway debates
    # Uses ARAGORA_DEBATE_TIMEOUT env var (default 900s = 15 min)
    # Set to 0 for unlimited (not recommended for production)
    timeout_seconds: int = DEBATE_TIMEOUT_SECONDS  # Max time for entire debate

    # Round timeout should exceed agent timeout (AGENT_TIMEOUT_SECONDS)
    # to allow at least one agent to complete. Default allows 1-2 agents serially
    # or many agents in parallel. Uses AGENT_TIMEOUT + 60s margin.
    round_timeout_seconds: int = AGENT_TIMEOUT_SECONDS + 60  # Per-round timeout

    # Breakpoints: Human-in-the-loop intervention points
    # When enabled, debates can pause at critical moments for human guidance
    enable_breakpoints: bool = False  # Enable breakpoint detection
    breakpoint_config: Optional["BreakpointConfig"] = None  # Custom breakpoint thresholds

    # Calibration tracking: Record prediction accuracy for calibration curves
    # When enabled, agent prediction confidence is tracked against outcomes
    enable_calibration: bool = False  # Enable calibration tracking

    # Rhetorical observer: Passive commentary on debate dynamics
    # Detects patterns like concession, rebuttal, synthesis for audience engagement
    enable_rhetorical_observer: bool = False  # Enable rhetorical pattern detection

    # Trickster for hollow consensus detection
    # Challenges convergence that lacks evidence quality
    enable_trickster: bool = False  # Enable hollow consensus detection
    trickster_sensitivity: float = 0.7  # Threshold for triggering challenges

    # Prompt evolution: Learn from debate outcomes to improve agent prompts
    # When enabled, PromptEvolver extracts winning patterns and updates prompts
    enable_evolution: bool = False  # Enable prompt evolution from debate outcomes

    # Formal verification during consensus: Optional claim verification
    # When enabled, claims in proposals are verified using Z3/Lean backends
    # during vote weighting. Verified claims get a weight bonus.
    verify_claims_during_consensus: bool = False  # Enable claim verification
    verification_weight_bonus: float = 0.2  # Boost for verified claims (0.0-1.0)
    verification_timeout_seconds: float = 5.0  # Quick timeout per verification


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
