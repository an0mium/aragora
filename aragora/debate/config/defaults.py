"""
Centralized defaults for the debate module.

This module extracts magic numbers and hardcoded values from across the debate
package into a single, documented, frozen dataclass. All values can be overridden
via environment variables where appropriate.

Usage:
    from aragora.debate.config.defaults import DEBATE_DEFAULTS

    # Access defaults
    threshold = DEBATE_DEFAULTS.convergence_threshold

    # Or get a fresh instance (useful for testing)
    from aragora.debate.config.defaults import get_debate_defaults
    defaults = get_debate_defaults()

Environment Variable Overrides:
    Many values can be overridden via environment variables with the
    ARAGORA_DEBATE_ prefix. For example:
    - ARAGORA_DEBATE_CONVERGENCE_THRESHOLD=0.90
    - ARAGORA_DEBATE_DIVERGENCE_THRESHOLD=0.35
    - ARAGORA_DEBATE_AGENT_TIMEOUT=45.0
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Final


def _get_env_float(name: str, default: float) -> float:
    """Get a float from environment variable with fallback."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_int(name: str, default: int) -> int:
    """Get an int from environment variable with fallback."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class DebateDefaults:
    """
    Centralized defaults for the debate module.

    This frozen dataclass contains all magic numbers and default values
    used across the debate package. Values are organized by category
    with clear documentation.

    All thresholds are in the range [0.0, 1.0] unless otherwise noted.
    All timeouts are in seconds unless otherwise noted.
    """

    # =========================================================================
    # Convergence Detection
    # =========================================================================
    # Thresholds for semantic convergence detection between debate rounds.
    # See: aragora/debate/convergence.py

    convergence_threshold: float = 0.85
    """
    Similarity threshold for declaring convergence.
    When minimum pairwise similarity >= this value, debate has converged.
    Range: [0.0, 1.0]. Higher = stricter convergence requirements.
    """

    divergence_threshold: float = 0.40
    """
    Similarity threshold for detecting divergence.
    When minimum pairwise similarity < this value, positions are splitting.
    Range: [0.0, 1.0]. Lower = more tolerance for position differences.
    """

    min_rounds_before_convergence_check: int = 1
    """
    Minimum debate rounds before checking for convergence.
    Prevents premature termination of debates.
    """

    consecutive_stable_rounds_for_convergence: int = 1
    """
    Number of consecutive stable rounds needed to declare convergence.
    Higher values require more sustained agreement.
    """

    # Advanced convergence metrics thresholds
    argument_diversity_convergence_threshold: float = 0.3
    """
    Diversity score below which arguments are considered converging.
    Lower diversity = agents focusing on same points.
    """

    evidence_overlap_convergence_threshold: float = 0.6
    """
    Citation overlap score above which evidence is considered converging.
    Higher overlap = agents citing same sources.
    """

    stance_volatility_stability_threshold: float = 0.2
    """
    Volatility score below which positions are considered stable.
    Lower volatility = agents maintaining consistent positions.
    """

    # Convergence metric weights for overall score
    convergence_weight_semantic: float = 0.4
    convergence_weight_diversity: float = 0.2
    convergence_weight_evidence: float = 0.2
    convergence_weight_stability: float = 0.2

    # Argument similarity threshold for uniqueness detection
    argument_uniqueness_similarity_threshold: float = 0.7
    """
    Similarity threshold for argument deduplication.
    Arguments with similarity > this are considered duplicates.
    """

    # =========================================================================
    # Consensus Detection
    # =========================================================================
    # Thresholds for consensus voting and detection.
    # See: aragora/debate/consensus.py, aragora/debate/protocol.py

    consensus_threshold: float = 0.6
    """
    Fraction of votes needed for majority consensus.
    Used when consensus mode is 'majority'.
    """

    strong_consensus_agreement_ratio: float = 0.8
    """
    Agreement ratio threshold for 'strong' consensus.
    Used in ConsensusProof.has_strong_consensus.
    """

    strong_consensus_confidence: float = 0.7
    """
    Minimum confidence for 'strong' consensus.
    Used alongside agreement ratio.
    """

    high_severity_dissent_threshold: float = 0.7
    """
    Dissent severity threshold for identifying blind spots.
    Dissents with severity >= this are flagged as potential gaps.
    """

    low_agreement_blind_spot_threshold: float = 0.6
    """
    Agreement ratio below which consensus indicates multiple valid perspectives.
    Used for blind spot detection.
    """

    # Claim support ratio thresholds
    claim_unanimous_support_ratio: float = 0.9
    """Support ratio >= this indicates unanimous agreement on claim."""

    claim_majority_support_ratio: float = 0.6
    """Support ratio >= this indicates majority agreement on claim."""

    # =========================================================================
    # Timeouts (seconds)
    # =========================================================================
    # Timeout values for various debate operations.
    # See: aragora/debate/orchestrator.py, aragora/debate/protocol.py

    agent_timeout_seconds: float = 30.0
    """
    Default timeout for a single agent response.
    Individual agents should respond within this time.
    """

    round_timeout_seconds: int = 90
    """
    Per-round timeout for all agents to complete.
    Should exceed agent_timeout to allow parallel completion.
    """

    debate_rounds_phase_timeout_seconds: int = 420
    """
    Total timeout for all debate rounds (7 minutes).
    Covers the entire DebateRoundsPhase execution.
    """

    debate_total_timeout_seconds: int = 1200
    """
    Total timeout for entire debate (20 minutes).
    Maximum time for a complete debate run.
    """

    verification_timeout_seconds: float = 5.0
    """
    Timeout for claim verification operations.
    Per-verification timeout for Z3/formal checks.
    """

    judge_evaluation_timeout_seconds: float = 10.0
    """
    Timeout for LLM-based evaluation (e.g., rubric scoring).
    Used in bias mitigation and quality assessment.
    """

    # =========================================================================
    # Retry and Limits
    # =========================================================================
    # Maximum values and retry counts for various operations.

    max_challenges_per_round: int = 3
    """Maximum Trickster challenges per debate round."""

    max_interventions_total: int = 5
    """Maximum total Trickster interventions per debate."""

    intervention_cooldown_rounds: int = 1
    """Minimum rounds between Trickster interventions."""

    max_parallel_scenarios: int = 3
    """Maximum parallel scenario executions."""

    max_checkpoints_default: int = 100
    """Default limit for checkpoint listing."""

    max_history_entries: int = 1000
    """Maximum entries in agent channel history."""

    max_recent_messages_rlm: int = 5
    """Number of recent messages to keep at full detail during RLM compression."""

    # =========================================================================
    # Quality and Scoring Thresholds
    # =========================================================================
    # Thresholds for quality gates, evidence scoring, and filtering.

    min_quality_threshold: float = 0.65
    """
    Minimum acceptable evidence quality score.
    Evidence below this may trigger Trickster challenges.
    """

    hollow_detection_threshold: float = 0.5
    """
    Alert severity threshold for hollow consensus detection.
    Consensus with quality below this triggers intervention.
    """

    quality_gate_threshold: float = 0.6
    """
    Minimum quality score for response filtering.
    Responses below this are filtered by ML quality gate.
    """

    consensus_early_termination_threshold: float = 0.85
    """
    Probability threshold for ML-based early termination.
    When consensus estimator exceeds this, debate may end early.
    """

    early_stop_threshold: float = 0.85
    """
    Fraction of agents saying 'stop' to trigger early termination.
    Used in early_stopping protocol feature.
    """

    # =========================================================================
    # Evidence and Citation Scoring
    # =========================================================================
    # Weights and bonuses for evidence-based scoring.
    # See: aragora/debate/bias_mitigation.py

    evidence_citation_bonus: float = 0.15
    """Bonus per evidence citation in vote weighting."""

    verification_weight_bonus: float = 0.2
    """Weight boost for verified claims."""

    # Verbosity penalty configuration
    verbosity_target_length: int = 1000
    """Ideal proposal length in characters."""

    verbosity_penalty_threshold: float = 3.0
    """Penalize proposals > this multiple of target length."""

    verbosity_max_penalty: float = 0.3
    """Maximum weight reduction for verbose proposals (30%)."""

    # Rubric scoring defaults
    rubric_no_citations_score: float = 0.2
    """Score when proposal has no citations."""

    rubric_base_citation_score: float = 0.3
    """Base score for having citations."""

    rubric_per_citation_bonus: float = 0.2
    """Bonus per valid citation (capped at 1.0)."""

    rubric_uncertainty_base: float = 0.2
    """Base score for epistemic humility."""

    rubric_uncertainty_per_marker: float = 0.12
    """Bonus per uncertainty marker found."""

    rubric_reasoning_base: float = 0.3
    """Base score for clear reasoning."""

    rubric_reasoning_per_marker: float = 0.1
    """Bonus per reasoning marker found."""

    rubric_structure_bonus: float = 0.15
    """Bonus for structured format (bullets, headers)."""

    rubric_counterargument_base: float = 0.2
    """Base score for counterargument handling."""

    rubric_counter_per_marker: float = 0.12
    """Bonus per counterargument marker found."""

    rubric_synthesis_base: float = 0.2
    """Base score for integrative synthesis."""

    rubric_synthesis_per_marker: float = 0.15
    """Bonus per synthesis marker found."""

    rubric_neutral_default: float = 0.5
    """Default neutral score when evaluation fails."""

    # =========================================================================
    # Bias Mitigation
    # =========================================================================
    # Configuration for position shuffling and self-vote handling.
    # See: aragora/debate/bias_mitigation.py

    position_shuffling_permutations: int = 3
    """Number of random orderings for position shuffling."""

    self_vote_downweight_factor: float = 0.5
    """Weight multiplier for self-votes (reduces self-preference)."""

    # =========================================================================
    # Trickster Configuration
    # =========================================================================
    # Hollow consensus detection and challenge settings.
    # See: aragora/debate/trickster.py

    trickster_sensitivity: float = 0.5
    """
    Default Trickster sensitivity (0.0-1.0).
    Higher = more aggressive hollow consensus detection.
    """

    trickster_redundancy_threshold: float = 0.7
    """
    Cross-proposal redundancy score threshold.
    Below this, proposals are too similar (potential echo chamber).
    """

    # =========================================================================
    # RLM Compression
    # =========================================================================
    # Recursive Language Model compression settings.
    # See: aragora/debate/arena_config.py

    rlm_compression_threshold: int = 3000
    """Character count above which to trigger RLM compression."""

    rlm_compression_round_threshold: int = 3
    """Start auto-compression after this many rounds."""

    # =========================================================================
    # Memory and Knowledge Management
    # =========================================================================
    # Thresholds for memory operations and Knowledge Mound integration.

    extraction_min_confidence: float = 0.3
    """Minimum debate confidence to trigger knowledge extraction."""

    coordinator_min_confidence_for_mound: float = 0.7
    """Minimum confidence to write to Knowledge Mound."""

    revalidation_staleness_threshold: float = 0.7
    """Staleness score threshold for triggering revalidation."""

    revalidation_check_interval_seconds: int = 3600
    """Interval between staleness checks (1 hour)."""

    # Memory tier thresholds
    outcome_memory_success_threshold: float = 0.7
    """Minimum confidence for memory tier promotion."""

    outcome_memory_usage_threshold: int = 3
    """Successful uses before promotion to higher tier."""

    feedback_loop_min_debates: int = 3
    """Minimum debates before applying feedback loop adjustments."""

    feedback_loop_weight: float = 0.15
    """Weight for feedback loop adjustments."""

    feedback_loop_decay: float = 0.9
    """Decay factor for old feedback."""

    # =========================================================================
    # Broadcast and Export Thresholds
    # =========================================================================
    # Confidence thresholds for triggering post-debate actions.

    broadcast_min_confidence: float = 0.8
    """Minimum confidence to trigger audio/video broadcast."""

    training_export_min_confidence: float = 0.75
    """Minimum confidence to export training data."""

    breeding_threshold: float = 0.8
    """Minimum confidence to trigger population evolution."""

    post_debate_workflow_threshold: float = 0.7
    """Minimum confidence to trigger post-debate workflow."""

    receipt_min_confidence: float = 0.6
    """Minimum confidence to generate cryptographic receipt."""

    bead_min_confidence: float = 0.5
    """Minimum confidence to create a decision bead."""

    # =========================================================================
    # Scenario Analysis
    # =========================================================================
    # Thresholds for scenario comparison and analysis.
    # See: aragora/debate/scenarios.py

    scenario_conclusions_similarity_threshold: float = 0.6
    """Jaccard similarity threshold for considering conclusions similar."""

    scenario_confidence_difference_threshold: float = 0.2
    """Confidence difference above which to note divergence."""

    scenario_high_consistency_threshold: float = 0.8
    """Consistency score above which results are 'highly consistent'."""

    scenario_moderate_consistency_threshold: float = 0.5
    """Consistency score above which results are 'mostly consistent'."""

    # =========================================================================
    # Calibration and ML Integration
    # =========================================================================
    # Configuration for calibration tracking and ML delegation.

    trickster_calibration_min_samples: int = 20
    """Minimum outcomes before Trickster calibration."""

    trickster_calibration_interval: int = 50
    """Debates between Trickster calibrations."""

    ml_delegation_weight: float = 0.3
    """Weight for ML scoring vs ELO in agent selection."""

    calibration_cost_min_predictions: int = 20
    """Minimum predictions before applying calibration cost."""

    calibration_cost_ece_threshold: float = 0.1
    """Expected calibration error threshold."""

    calibration_cost_overconfident_multiplier: float = 1.3
    """Cost multiplier for overconfident agents."""

    # =========================================================================
    # Cross-Pollination Bridges
    # =========================================================================
    # Weights for performance routing and selection.

    performance_router_latency_weight: float = 0.3
    performance_router_quality_weight: float = 0.4
    performance_router_consistency_weight: float = 0.3

    analytics_selection_diversity_weight: float = 0.2
    analytics_selection_synergy_weight: float = 0.3

    novelty_selection_low_penalty: float = 0.15
    novelty_selection_high_bonus: float = 0.1
    novelty_selection_low_threshold: float = 0.3

    relationship_bias_alliance_threshold: float = 0.7
    relationship_bias_agreement_threshold: float = 0.8
    relationship_bias_vote_penalty: float = 0.3

    # =========================================================================
    # Translation and Caching
    # =========================================================================
    # Cache configuration for translation and other subsystems.

    translation_cache_ttl_seconds: int = 3600
    """Translation cache TTL (1 hour)."""

    translation_cache_max_entries: int = 10000
    """Maximum entries in translation cache."""

    checkpoint_memory_max_entries: int = 100
    """Maximum entries per tier in checkpoint memory snapshot."""

    # =========================================================================
    # Voting and Participation
    # =========================================================================
    # Configuration for voting mechanics and user participation.

    min_participation_ratio: float = 0.5
    """Minimum fraction of agents that must vote."""

    min_participation_count: int = 2
    """Minimum absolute count of agents that must vote."""

    user_vote_weight: float = 0.5
    """Weight of user votes relative to agent votes."""

    vote_grouping_threshold: float = 0.85
    """Similarity threshold for merging vote options."""

    # =========================================================================
    # Agent Limits
    # =========================================================================
    # Agent count constraints for debate configurations.

    min_agents_per_debate: int = 2
    """Minimum agents required for a standard debate."""

    max_agents_per_debate: int = 10
    """Maximum agents allowed in a single debate."""

    byzantine_min_agents: int = 4
    """Minimum agents for Byzantine fault-tolerant consensus (n >= 3f+1)."""

    # =========================================================================
    # Distributed Debate Defaults
    # =========================================================================
    # Defaults for distributed (cross-instance) debates.

    distributed_default_rounds: int = 5
    """Default rounds for distributed debates (fewer than local due to latency)."""

    distributed_consensus_threshold: float = 0.67
    """Consensus threshold for distributed debates (higher than local for reliability)."""

    distributed_proposal_timeout_seconds: float = 60.0
    """Per-proposal timeout in distributed debates."""

    distributed_critique_timeout_seconds: float = 45.0
    """Per-critique timeout in distributed debates."""

    distributed_vote_timeout_seconds: float = 30.0
    """Per-vote timeout in distributed debates."""

    distributed_sync_interval_seconds: float = 5.0
    """State sync interval for distributed coordination."""

    distributed_failover_timeout_seconds: float = 30.0
    """Failover timeout for distributed leader election."""

    # =========================================================================
    # Security Debate Defaults
    # =========================================================================
    # Defaults for security-focused debates.

    security_debate_rounds: int = 3
    """Rounds for security-focused debates (fast, targeted)."""

    security_debate_consensus: str = "majority"
    """Consensus mode for security debates."""

    security_debate_timeout_seconds: int = 300
    """Timeout for security debates (5 minutes)."""

    # =========================================================================
    # Protocol Defaults
    # =========================================================================
    # Default values for DebateProtocol fields.

    default_agreement_intensity: int = 5
    """
    Default agreement intensity (0-10).
    5 = balanced, lower = more disagreement, higher = more agreement.
    """

    default_trickster_sensitivity: float = 0.7
    """Default Trickster sensitivity in protocol."""


# Singleton instance using environment variable overrides
@lru_cache(maxsize=1)
def get_debate_defaults() -> DebateDefaults:
    """
    Get the debate defaults instance with environment variable overrides.

    Returns:
        DebateDefaults: Frozen dataclass with all debate configuration values.

    Environment Variables:
        ARAGORA_DEBATE_CONVERGENCE_THRESHOLD: Override convergence threshold
        ARAGORA_DEBATE_DIVERGENCE_THRESHOLD: Override divergence threshold
        ARAGORA_DEBATE_AGENT_TIMEOUT: Override agent timeout seconds
        ARAGORA_DEBATE_QUALITY_GATE_THRESHOLD: Override quality gate threshold
        ARAGORA_DEBATE_CONSENSUS_THRESHOLD: Override consensus threshold
    """
    return DebateDefaults(
        # Apply environment variable overrides for key configurable values
        convergence_threshold=_get_env_float(
            "ARAGORA_DEBATE_CONVERGENCE_THRESHOLD",
            DebateDefaults.convergence_threshold,
        ),
        divergence_threshold=_get_env_float(
            "ARAGORA_DEBATE_DIVERGENCE_THRESHOLD",
            DebateDefaults.divergence_threshold,
        ),
        agent_timeout_seconds=_get_env_float(
            "ARAGORA_DEBATE_AGENT_TIMEOUT",
            DebateDefaults.agent_timeout_seconds,
        ),
        quality_gate_threshold=_get_env_float(
            "ARAGORA_DEBATE_QUALITY_GATE_THRESHOLD",
            DebateDefaults.quality_gate_threshold,
        ),
        consensus_threshold=_get_env_float(
            "ARAGORA_DEBATE_CONSENSUS_THRESHOLD",
            DebateDefaults.consensus_threshold,
        ),
        debate_total_timeout_seconds=_get_env_int(
            "ARAGORA_DEBATE_TOTAL_TIMEOUT",
            DebateDefaults.debate_total_timeout_seconds,
        ),
        rlm_compression_threshold=_get_env_int(
            "ARAGORA_DEBATE_RLM_COMPRESSION_THRESHOLD",
            DebateDefaults.rlm_compression_threshold,
        ),
    )


# Module-level singleton for convenient access
DEBATE_DEFAULTS: Final[DebateDefaults] = get_debate_defaults()
