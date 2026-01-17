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

from aragora.config import AGENT_TIMEOUT_SECONDS, DEBATE_TIMEOUT_SECONDS
from aragora.debate.role_matcher import RoleMatchingConfig
from aragora.debate.roles import RoleRotationConfig
from aragora.resilience import CircuitBreaker  # Re-export for backwards compatibility

if TYPE_CHECKING:
    from aragora.debate.breakpoints import BreakpointConfig

logger = logging.getLogger(__name__)

# Re-export CircuitBreaker for backwards compatibility
__all__ = [
    "ARAGORA_AI_LIGHT_PROTOCOL",
    "ARAGORA_AI_PROTOCOL",
    "CircuitBreaker",
    "DebateProtocol",
    "RoundPhase",
    "STRUCTURED_LIGHT_ROUND_PHASES",
    "STRUCTURED_ROUND_PHASES",
    "user_vote_multiplier",
]


@dataclass
class RoundPhase:
    """Configuration for a structured debate round phase."""

    number: int  # Round number (1-indexed)
    name: str  # Phase name (e.g., "Initial Analysis")
    description: str  # What this phase accomplishes
    focus: str  # Key focus area for agents
    cognitive_mode: str  # Analyst, Skeptic, Lateral, Synthesizer, etc.


# Default 9-round structured debate format for aragora.ai
# Round 0 (Context Gathering) runs parallel with Round 1
# Rounds 1-7 are the core debate cycle
# Round 8 (Adjudication) handles voting, judge verdict, and final synthesis
STRUCTURED_ROUND_PHASES: list[RoundPhase] = [
    RoundPhase(
        number=0,
        name="Context Gathering",
        description="Gather background information, evidence, and assign personas",
        focus="Research, evidence collection, historical context, persona assignment",
        cognitive_mode="Researcher",
    ),
    RoundPhase(
        number=1,
        name="Initial Analysis",
        description="Establish foundational understanding and key considerations",
        focus="Core facts, assumptions, and initial positions",
        cognitive_mode="Analyst",
    ),
    RoundPhase(
        number=2,
        name="Skeptical Review",
        description="Challenge assumptions and identify weaknesses",
        focus="Logical flaws, unsupported claims, edge cases",
        cognitive_mode="Skeptic",
    ),
    RoundPhase(
        number=3,
        name="Lateral Exploration",
        description="Explore alternative perspectives and creative solutions",
        focus="Novel approaches, analogies, unconventional ideas",
        cognitive_mode="Lateral Thinker",
    ),
    RoundPhase(
        number=4,
        name="Devil's Advocacy",
        description="Argue the strongest opposing viewpoint",
        focus="Counter-arguments, risks, unintended consequences",
        cognitive_mode="Devil's Advocate",
    ),
    RoundPhase(
        number=5,
        name="Integration",
        description="Connect insights across perspectives, identify patterns",
        focus="Emerging patterns, bridging views, key trade-offs, areas of agreement",
        cognitive_mode="Integrator",
    ),
    RoundPhase(
        number=6,
        name="Cross-Examination",
        description="Direct questioning between agents on remaining disputes",
        focus="Clarifying disagreements, testing convictions",
        cognitive_mode="Examiner",
    ),
    RoundPhase(
        number=7,
        name="Final Synthesis",
        description="Each agent synthesizes discussion and revises proposal to final form",
        focus="Polished final positions, integrated insights, honest uncertainty",
        cognitive_mode="Synthesizer",
    ),
    RoundPhase(
        number=8,
        name="Final Adjudication",
        description="Voting, judge verdict, Opus 4.5 synthesis, download links",
        focus="Final votes, judge selection, 1200-word conclusion, export formats",
        cognitive_mode="Adjudicator",
    ),
]


@dataclass
class DebateProtocol:
    """Configuration for how debates are conducted."""

    topology: Literal["all-to-all", "sparse", "round-robin", "ring", "star", "random-graph"] = (
        "all-to-all"
    )
    topology_sparsity: float = (
        0.5  # fraction of possible critique connections (for sparse/random-graph)
    )
    topology_hub_agent: Optional[str] = None  # for star topology, which agent is the hub
    rounds: int = 8  # Structured 8-round format (0-7), Round 8 is adjudication

    # Structured round phases: Use predefined phase structure for each round
    # When enabled, each round has a specific focus (Analysis, Skeptic, Lateral, etc.)
    use_structured_phases: bool = True  # Enable structured 7-round format
    round_phases: Optional[list[RoundPhase]] = (
        None  # Custom phases (uses STRUCTURED_ROUND_PHASES if None)
    )

    consensus: Literal[
        "majority",
        "unanimous",
        "judge",
        "none",
        "weighted",
        "supermajority",
        "any",
        "byzantine",
    ] = "judge"
    consensus_threshold: float = 0.6  # fraction needed for majority
    allow_abstain: bool = True
    require_reasoning: bool = True

    # Role assignments
    proposer_count: int = -1  # -1 means all agents propose (default for 9-round format)
    critic_count: int = -1  # -1 means all agents critique
    critique_required: bool = True  # require critiques before consensus

    # Judge selection (for consensus="judge" mode)
    # "elo_ranked" selects highest ELO-rated agent from EloSystem (requires elo_system param)
    judge_selection: Literal[
        "random",
        "voted",
        "last",
        "elo_ranked",
        "calibrated",
        "crux_aware",
    ] = "random"

    # Agreement intensity (0-10): Controls how much agents agree vs disagree
    # 0 = strongly disagree 100% of the time (adversarial)
    # 5 = balanced (agree/disagree based on argument quality)
    # 10 = fully incorporate others' opinions (collaborative)
    # Research shows intensity=2 (slight disagreement bias) often improves accuracy
    agreement_intensity: int = 5

    # Early stopping: End debate when agents agree further rounds won't help
    # Based on ai-counsel pattern - can save 40-70% API costs
    # Set to near-impossible thresholds to ensure full 9-round debates
    early_stopping: bool = True
    early_stop_threshold: float = (
        0.95  # fraction of agents saying stop to trigger (near-impossible)
    )
    min_rounds_before_early_stop: int = 7  # minimum rounds before allowing early exit

    # Asymmetric debate roles: Assign affirmative/negative/neutral stances
    # Forces perspective diversity, prevents premature consensus
    asymmetric_stances: bool = False  # Enable asymmetric stance assignment
    rotate_stances: bool = True  # Rotate stances between rounds

    # Semantic convergence detection
    # Auto-detect consensus without explicit voting
    convergence_detection: bool = True
    convergence_threshold: float = (
        0.95  # Similarity for convergence (high bar to prevent premature consensus)
    )
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
    user_vote_weight: float = (
        0.5  # Weight of user votes relative to agent votes (0.5 = half weight)
    )

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
    role_matching: bool = True  # Enable calibration-based role matching
    role_matching_config: Optional[RoleMatchingConfig] = None  # Custom matching config

    # Debate timeout (seconds) - prevents runaway debates
    # Uses ARAGORA_DEBATE_TIMEOUT env var (default 900s = 15 min)
    # Set to 0 for unlimited (not recommended for production)
    # Increased to 1200s (20 min) to allow 6+ min for 7-round debates
    timeout_seconds: int = max(1200, DEBATE_TIMEOUT_SECONDS)  # Max time for entire debate

    # Round timeout should exceed agent timeout (AGENT_TIMEOUT_SECONDS)
    # to allow all parallel agents to complete. Minimum 90s per round for thorough analysis.
    round_timeout_seconds: int = max(90, AGENT_TIMEOUT_SECONDS + 60)  # Per-round timeout

    # Debate rounds phase timeout - at least 6 minutes (360s) for all rounds
    debate_rounds_timeout_seconds: int = 420  # 7 minutes for debate_rounds phase

    # Breakpoints: Human-in-the-loop intervention points
    # When enabled, debates can pause at critical moments for human guidance
    enable_breakpoints: bool = True  # Enable breakpoint detection
    breakpoint_config: Optional["BreakpointConfig"] = None  # Custom breakpoint thresholds

    # Calibration tracking: Record prediction accuracy for calibration curves
    # When enabled, agent prediction confidence is tracked against outcomes
    enable_calibration: bool = True  # Enable calibration tracking by default

    # Rhetorical observer: Passive commentary on debate dynamics
    # Detects patterns like concession, rebuttal, synthesis for audience engagement
    enable_rhetorical_observer: bool = True  # Enable rhetorical pattern detection

    # Trickster for hollow consensus detection
    # Challenges convergence that lacks evidence quality
    enable_trickster: bool = True  # Enable hollow consensus detection by default
    trickster_sensitivity: float = 0.7  # Threshold for triggering challenges

    # Prompt evolution: Learn from debate outcomes to improve agent prompts
    # When enabled, PromptEvolver extracts winning patterns and updates prompts
    enable_evolution: bool = True  # Enable prompt evolution from debate outcomes

    # Formal verification during consensus: Claim verification for quality
    # When enabled, claims in proposals are verified using pattern matching
    # during vote weighting. Verified claims get a weight bonus.
    # Enabled by default to improve debate quality feedback loop.
    verify_claims_during_consensus: bool = True  # Enable claim verification
    verification_weight_bonus: float = 0.2  # Boost for verified claims (0.0-1.0)
    verification_timeout_seconds: float = 5.0  # Quick timeout per verification

    # Evidence citation weighting: Reward votes that cite evidence
    # When enabled, votes that properly reference evidence from evidence_pack
    # receive a weight bonus during consensus. Encourages factual grounding.
    # Detects EVID-xxx patterns in vote reasoning.
    enable_evidence_weighting: bool = True  # Enable evidence citation bonuses
    evidence_citation_bonus: float = 0.15  # Bonus per evidence citation (0.0-1.0)

    # Formal proof verification: Use Lean4/Z3 to verify consensus claims
    # When enabled, attempts machine-checkable proof of final consensus
    # Requires formal verification backends to be installed (z3-solver, etc.)
    formal_verification_enabled: bool = False  # Enable formal proof verification
    formal_verification_languages: list[str] = field(
        default_factory=lambda: ["z3_smt"]
    )  # Languages to try: z3_smt, lean4
    formal_verification_timeout: float = 30.0  # Timeout for proof search (seconds)

    # Byzantine consensus configuration (for consensus="byzantine")
    # PBFT-style fault-tolerant consensus tolerating f faulty nodes where n >= 3f+1
    # Adapted from claude-flow (MIT License)
    byzantine_fault_tolerance: float = 0.33  # Max fraction of faulty agents (default 1/3)
    byzantine_phase_timeout: float = 30.0  # Timeout per PBFT phase (seconds)
    byzantine_max_view_changes: int = 3  # Max leader changes before failure

    def get_round_phase(self, round_number: int) -> Optional[RoundPhase]:
        """Get the phase configuration for a specific round.

        Args:
            round_number: 0-indexed round number (0 = Context Gathering, 1-7 = debate, 8 = Adjudication)

        Returns:
            RoundPhase for the round, or None if not using structured phases
        """
        if not self.use_structured_phases:
            return None

        phases = self.round_phases or STRUCTURED_ROUND_PHASES
        if 0 <= round_number < len(phases):
            return phases[round_number]
        return None


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
        return (
            protocol.user_vote_intensity_min_multiplier
            + (1.0 - protocol.user_vote_intensity_min_multiplier) * ratio
        )
    else:
        # Above neutral: interpolate between 1.0 and max_multiplier
        # intensity=neutral -> 1.0, intensity=scale -> max_multiplier
        ratio = (intensity - neutral) / (scale - neutral) if scale > neutral else 0
        return 1.0 + (protocol.user_vote_intensity_max_multiplier - 1.0) * ratio


# =============================================================================
# Aragora.ai Web UI Default Protocol
# =============================================================================
# This preset is used for debates launched from the aragora.ai main input field.
# CLI, SDK, and API users retain full flexibility with custom configurations.

ARAGORA_AI_PROTOCOL = DebateProtocol(
    # 9-round structured format
    rounds=9,
    use_structured_phases=True,
    round_phases=None,  # Uses STRUCTURED_ROUND_PHASES (9 rounds)
    # Judge-based consensus for final decision
    consensus="judge",
    consensus_threshold=0.6,
    # All agents participate
    topology="all-to-all",
    proposer_count=-1,  # All agents propose
    critic_count=-1,  # All agents critique
    # Near-impossible early stopping (require 95% uniform consensus)
    early_stopping=True,
    early_stop_threshold=0.95,
    min_rounds_before_early_stop=7,
    # High convergence bar to prevent premature consensus
    convergence_detection=True,
    convergence_threshold=0.95,
    divergence_threshold=0.40,
    # Enable Trickster for hollow consensus detection
    enable_trickster=True,
    trickster_sensitivity=0.7,
    # Enable all quality features
    enable_calibration=True,
    enable_rhetorical_observer=True,
    enable_evolution=True,
    enable_evidence_weighting=True,
    verify_claims_during_consensus=True,
    enable_research=True,
    # Role rotation for cognitive diversity
    role_rotation=True,
    role_matching=True,
    # Extended timeouts for 9-round debates
    timeout_seconds=1800,  # 30 minutes total
    round_timeout_seconds=150,  # 2.5 minutes per round
    debate_rounds_timeout_seconds=900,  # 15 minutes for debate rounds phase
    # Enable breakpoints for human intervention
    enable_breakpoints=True,
)

# =============================================================================
# Aragora.ai Light Protocol - Quick Debates
# =============================================================================
# Fast 4-round format for simple questions. ~5 minutes vs ~30 minutes for full.
# Selected via debate_format="light" in API requests.

STRUCTURED_LIGHT_ROUND_PHASES: list[RoundPhase] = [
    RoundPhase(
        number=0,
        name="Quick Context",
        description="Minimal context gathering, core facts only",
        focus="Essential background, key facts",
        cognitive_mode="Researcher",
    ),
    RoundPhase(
        number=1,
        name="Initial Positions",
        description="Establish key viewpoints and main arguments",
        focus="Core positions, primary reasoning",
        cognitive_mode="Analyst",
    ),
    RoundPhase(
        number=2,
        name="Critique & Synthesis",
        description="Combined challenge and integration phase",
        focus="Key disagreements, emerging consensus",
        cognitive_mode="Skeptic",
    ),
    RoundPhase(
        number=3,
        name="Quick Resolution",
        description="Fast judge decision and brief synthesis",
        focus="Final answer, key takeaways",
        cognitive_mode="Adjudicator",
    ),
]

ARAGORA_AI_LIGHT_PROTOCOL = DebateProtocol(
    # 4-round quick format
    rounds=4,
    use_structured_phases=True,
    round_phases=STRUCTURED_LIGHT_ROUND_PHASES,
    # Judge-based for speed
    consensus="judge",
    consensus_threshold=0.6,
    # All agents participate but fewer rounds
    topology="all-to-all",
    proposer_count=-1,
    critic_count=-1,
    # Aggressive early stopping (70% agreement)
    early_stopping=True,
    early_stop_threshold=0.7,
    min_rounds_before_early_stop=2,
    # Lower convergence bar for faster consensus
    convergence_detection=True,
    convergence_threshold=0.8,
    divergence_threshold=0.40,
    # Disable compute-intensive features for speed
    enable_trickster=False,
    enable_calibration=False,
    enable_rhetorical_observer=False,
    enable_evolution=False,
    enable_evidence_weighting=False,
    verify_claims_during_consensus=False,
    enable_research=False,  # Skip web research
    # Simpler roles (no rotation)
    role_rotation=False,
    role_matching=False,
    # Tight timeouts for quick resolution
    timeout_seconds=300,  # 5 minutes total
    round_timeout_seconds=60,  # 1 minute per round
    debate_rounds_timeout_seconds=180,  # 3 minutes for debate rounds
    # No breakpoints in light mode
    enable_breakpoints=False,
)
