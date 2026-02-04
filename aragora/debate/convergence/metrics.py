"""
Convergence result and advanced convergence metrics (G3).

Contains dataclasses for representing convergence detection results
and multi-metric convergence analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# =============================================================================
# Convergence Result
# =============================================================================


@dataclass
class ConvergenceResult:
    """Result of convergence detection check."""

    converged: bool
    status: str  # "converged", "diverging", "refining"
    min_similarity: float
    avg_similarity: float
    per_agent_similarity: dict[str, float] = field(default_factory=dict)
    consecutive_stable_rounds: int = 0


# =============================================================================
# Advanced Convergence Metrics (G3)
# =============================================================================


@dataclass
class ArgumentDiversityMetric:
    """
    Measures diversity of arguments across agents.

    High diversity = agents covering different points (good for exploration)
    Low diversity = agents focusing on same points (may indicate convergence)
    """

    unique_arguments: int
    total_arguments: int
    diversity_score: float  # 0-1, higher = more diverse

    @property
    def is_converging(self) -> bool:
        """Arguments becoming less diverse suggests convergence."""
        return self.diversity_score < 0.3


@dataclass
class EvidenceConvergenceMetric:
    """
    Measures overlap in cited evidence/sources.

    High overlap = agents citing same sources (strong agreement)
    Low overlap = agents using different evidence (disagreement or complementary)
    """

    shared_citations: int
    total_citations: int
    overlap_score: float  # 0-1, higher = more overlap

    @property
    def is_converging(self) -> bool:
        """High citation overlap suggests convergence."""
        return self.overlap_score > 0.6


@dataclass
class StanceVolatilityMetric:
    """
    Measures how often agents change their positions.

    High volatility = agents frequently changing stances (unstable)
    Low volatility = agents maintaining consistent positions (stable)
    """

    stance_changes: int
    total_responses: int
    volatility_score: float  # 0-1, higher = more volatile

    @property
    def is_stable(self) -> bool:
        """Low volatility indicates stable positions."""
        return self.volatility_score < 0.2


@dataclass
class AdvancedConvergenceMetrics:
    """
    Comprehensive convergence metrics for debate analysis.

    Combines multiple signals to provide a nuanced view of
    debate convergence beyond simple text similarity.
    """

    # Core similarity (from ConvergenceDetector)
    semantic_similarity: float

    # Advanced metrics
    argument_diversity: ArgumentDiversityMetric | None = None
    evidence_convergence: EvidenceConvergenceMetric | None = None
    stance_volatility: StanceVolatilityMetric | None = None

    # Aggregate score
    overall_convergence: float = 0.0  # 0-1, higher = more converged

    # Domain context
    domain: str = "general"

    def compute_overall_score(self) -> float:
        """Compute weighted overall convergence score."""
        weights = {
            "semantic": 0.4,
            "diversity": 0.2,
            "evidence": 0.2,
            "stability": 0.2,
        }

        score = self.semantic_similarity * weights["semantic"]

        if self.argument_diversity:
            # Lower diversity = higher convergence
            score += (1 - self.argument_diversity.diversity_score) * weights["diversity"]

        if self.evidence_convergence:
            score += self.evidence_convergence.overlap_score * weights["evidence"]

        if self.stance_volatility:
            # Lower volatility = higher convergence
            score += (1 - self.stance_volatility.volatility_score) * weights["stability"]

        self.overall_convergence = min(1.0, max(0.0, score))
        return self.overall_convergence

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        result = {
            "semantic_similarity": self.semantic_similarity,
            "overall_convergence": self.overall_convergence,
            "domain": self.domain,
        }

        if self.argument_diversity:
            result["argument_diversity"] = {
                "unique_arguments": self.argument_diversity.unique_arguments,
                "total_arguments": self.argument_diversity.total_arguments,
                "diversity_score": self.argument_diversity.diversity_score,
            }

        if self.evidence_convergence:
            result["evidence_convergence"] = {
                "shared_citations": self.evidence_convergence.shared_citations,
                "total_citations": self.evidence_convergence.total_citations,
                "overlap_score": self.evidence_convergence.overlap_score,
            }

        if self.stance_volatility:
            result["stance_volatility"] = {
                "stance_changes": self.stance_volatility.stance_changes,
                "total_responses": self.stance_volatility.total_responses,
                "volatility_score": self.stance_volatility.volatility_score,
            }

        return result
