"""
Semantic convergence detection for multi-agent debates.

Detects when agents' positions have converged, allowing early termination
of debates when further rounds would provide diminishing returns.

Uses a 3-tier fallback for similarity computation:
1. SentenceTransformer (best accuracy, requires sentence-transformers)
2. TF-IDF (good accuracy, requires scikit-learn)
3. Jaccard (always available, zero dependencies)

Inspired by ai-counsel's convergence detection system.

Module Structure
----------------
This module has been split into submodules for maintainability:

- `aragora.debate.cache.embeddings_lru` - EmbeddingCache for text embeddings
- `aragora.debate.similarity.backends` - Similarity computation backends

This file contains:
- ConvergenceResult - Result of convergence check
- Advanced convergence metrics (G3)
- AdvancedConvergenceAnalyzer - Multi-metric analysis
- ConvergenceDetector - Main convergence detection
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Re-export cache utilities
from aragora.debate.cache.embeddings_lru import (
    EmbeddingCache,
    get_embedding_cache,
    reset_embedding_cache,
)

# Re-export similarity backends
from aragora.debate.similarity.backends import (
    SimilarityBackend,
    JaccardBackend,
    TFIDFBackend,
    SentenceTransformerBackend,
    get_similarity_backend,
    _normalize_backend_name,
    _ENV_CONVERGENCE_BACKEND,
    _ENV_SIMILARITY_BACKEND,
)


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
    argument_diversity: Optional[ArgumentDiversityMetric] = None
    evidence_convergence: Optional[EvidenceConvergenceMetric] = None
    stance_volatility: Optional[StanceVolatilityMetric] = None

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


# =============================================================================
# Advanced Convergence Analyzer
# =============================================================================


class AdvancedConvergenceAnalyzer:
    """
    Analyzes debate convergence using multiple metrics.

    Provides a more nuanced view than simple text similarity by
    considering argument diversity, evidence overlap, and stance stability.
    """

    def __init__(self, similarity_backend: Optional[SimilarityBackend] = None):
        """
        Initialize analyzer.

        Args:
            similarity_backend: Backend for text similarity (auto-selects if None)
        """
        if similarity_backend is None:
            # Use best available backend
            try:
                self.backend: SimilarityBackend = SentenceTransformerBackend()
            except ImportError:
                try:
                    self.backend = TFIDFBackend()
                except ImportError:
                    self.backend = JaccardBackend()
        else:
            self.backend = similarity_backend

    def extract_arguments(self, text: str) -> list[str]:
        """
        Extract distinct arguments/claims from text.

        Simple heuristic: split by sentence and filter.
        """
        import re

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        # Filter to substantive sentences (> 5 words)
        arguments = []
        for s in sentences:
            s = s.strip()
            if len(s.split()) > 5:
                arguments.append(s)

        return arguments

    def extract_citations(self, text: str) -> set[str]:
        """
        Extract citations/sources from text.

        Looks for URLs, academic-style citations, and quoted sources.
        """
        import re

        citations = set()

        # URLs
        urls = re.findall(r'https?://[^\s<>"]+', text)
        citations.update(urls)

        # Academic citations like (Author, 2024) or [1]
        academic = re.findall(r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)', text)
        citations.update(academic)

        # Numbered citations [1], [2], etc.
        numbered = re.findall(r'\[\d+\]', text)
        citations.update(numbered)

        # Quoted sources "According to X"
        quoted = re.findall(r'(?:according to|per|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text, re.I)
        citations.update(quoted)

        return citations

    def detect_stance(self, text: str) -> str:
        """
        Detect the stance/position in text.

        Returns: "support", "oppose", "neutral", or "mixed"
        """
        import re

        text_lower = text.lower()

        # Strong support indicators
        support_patterns = r'\b(agree|support|favor|endorse|recommend|should|must|definitely|certainly)\b'
        support_count = len(re.findall(support_patterns, text_lower))

        # Strong oppose indicators
        oppose_patterns = r'\b(disagree|oppose|against|reject|shouldn\'t|must not|definitely not|certainly not)\b'
        oppose_count = len(re.findall(oppose_patterns, text_lower))

        # Neutral indicators
        neutral_patterns = r'\b(depends|unclear|both|however|on the other hand|alternatively)\b'
        neutral_count = len(re.findall(neutral_patterns, text_lower))

        # Determine stance
        if support_count > oppose_count and support_count > neutral_count:
            return "support"
        elif oppose_count > support_count and oppose_count > neutral_count:
            return "oppose"
        elif neutral_count > 0 and support_count > 0 and oppose_count > 0:
            return "mixed"
        else:
            return "neutral"

    def compute_argument_diversity(
        self,
        agent_responses: dict[str, str],
    ) -> ArgumentDiversityMetric:
        """
        Compute argument diversity across agents.

        High diversity = agents making different points.
        """
        all_arguments: list[str] = []
        for text in agent_responses.values():
            all_arguments.extend(self.extract_arguments(text))

        if not all_arguments:
            return ArgumentDiversityMetric(
                unique_arguments=0,
                total_arguments=0,
                diversity_score=0.0,
            )

        # Compute pairwise similarity to find unique arguments
        # Arguments with < 0.7 similarity to all others are "unique"
        unique_count = 0
        for i, arg in enumerate(all_arguments):
            is_unique = True
            for j, other in enumerate(all_arguments):
                if i != j:
                    sim = self.backend.compute_similarity(arg, other)
                    if sim > 0.7:
                        is_unique = False
                        break
            if is_unique:
                unique_count += 1

        diversity_score = unique_count / len(all_arguments) if all_arguments else 0.0

        return ArgumentDiversityMetric(
            unique_arguments=unique_count,
            total_arguments=len(all_arguments),
            diversity_score=diversity_score,
        )

    def compute_evidence_convergence(
        self,
        agent_responses: dict[str, str],
    ) -> EvidenceConvergenceMetric:
        """
        Compute evidence/citation overlap across agents.

        High overlap = agents citing same sources.
        """
        all_citations: list[set[str]] = []
        for text in agent_responses.values():
            all_citations.append(self.extract_citations(text))

        # Flatten for total count
        all_unique = set().union(*all_citations) if all_citations else set()
        total = len(all_unique)

        if total == 0 or len(all_citations) < 2:
            return EvidenceConvergenceMetric(
                shared_citations=0,
                total_citations=0,
                overlap_score=0.0,
            )

        # Find citations shared by at least 2 agents
        shared = set()
        for citation in all_unique:
            count = sum(1 for agent_cites in all_citations if citation in agent_cites)
            if count >= 2:
                shared.add(citation)

        overlap_score = len(shared) / total if total > 0 else 0.0

        return EvidenceConvergenceMetric(
            shared_citations=len(shared),
            total_citations=total,
            overlap_score=overlap_score,
        )

    def compute_stance_volatility(
        self,
        response_history: list[dict[str, str]],
    ) -> StanceVolatilityMetric:
        """
        Compute stance volatility across rounds.

        Args:
            response_history: List of {agent: response} dicts per round

        Returns:
            StanceVolatilityMetric
        """
        if len(response_history) < 2:
            return StanceVolatilityMetric(
                stance_changes=0,
                total_responses=0,
                volatility_score=0.0,
            )

        # Track stance per agent per round
        agent_stances: dict[str, list[str]] = {}
        for round_responses in response_history:
            for agent, text in round_responses.items():
                if agent not in agent_stances:
                    agent_stances[agent] = []
                agent_stances[agent].append(self.detect_stance(text))

        # Count stance changes
        total_changes = 0
        total_responses = 0
        for agent, stances in agent_stances.items():
            total_responses += len(stances)
            for i in range(1, len(stances)):
                if stances[i] != stances[i-1]:
                    total_changes += 1

        volatility_score = total_changes / max(1, total_responses - len(agent_stances))

        return StanceVolatilityMetric(
            stance_changes=total_changes,
            total_responses=total_responses,
            volatility_score=min(1.0, volatility_score),
        )

    def analyze(
        self,
        current_responses: dict[str, str],
        previous_responses: Optional[dict[str, str]] = None,
        response_history: Optional[list[dict[str, str]]] = None,
        domain: str = "general",
    ) -> AdvancedConvergenceMetrics:
        """
        Perform comprehensive convergence analysis.

        Args:
            current_responses: {agent: response} for current round
            previous_responses: {agent: response} for previous round (optional)
            response_history: Full history of responses (optional)
            domain: Debate domain for context

        Returns:
            AdvancedConvergenceMetrics with all computed metrics
        """
        # Compute semantic similarity
        if previous_responses:
            common_agents = set(current_responses.keys()) & set(previous_responses.keys())
            if common_agents:
                similarities = []
                for agent in common_agents:
                    sim = self.backend.compute_similarity(
                        current_responses[agent],
                        previous_responses[agent],
                    )
                    similarities.append(sim)
                semantic_sim = sum(similarities) / len(similarities)
            else:
                semantic_sim = 0.0
        else:
            semantic_sim = 0.0

        # Compute advanced metrics
        arg_diversity = self.compute_argument_diversity(current_responses)
        evidence_conv = self.compute_evidence_convergence(current_responses)

        stance_vol = None
        if response_history:
            stance_vol = self.compute_stance_volatility(response_history)

        # Build result
        metrics = AdvancedConvergenceMetrics(
            semantic_similarity=semantic_sim,
            argument_diversity=arg_diversity,
            evidence_convergence=evidence_conv,
            stance_volatility=stance_vol,
            domain=domain,
        )

        # Compute overall score
        metrics.compute_overall_score()

        return metrics


# =============================================================================
# Convergence Detector
# =============================================================================


class ConvergenceDetector:
    """
    Detects when debate has converged semantically.

    Uses semantic similarity between consecutive rounds to determine
    if agents have reached consensus or are still refining positions.

    Thresholds:
        - converged: ≥85% similarity (agents agree)
        - refining: 40-85% similarity (still improving)
        - diverging: <40% similarity (positions splitting)
    """

    def __init__(
        self,
        convergence_threshold: float = 0.85,
        divergence_threshold: float = 0.40,
        min_rounds_before_check: int = 1,
        consecutive_rounds_needed: int = 1,
    ):
        """
        Initialize convergence detector.

        Args:
            convergence_threshold: Similarity threshold for convergence (default 0.85)
            divergence_threshold: Below this is diverging (default 0.40)
            min_rounds_before_check: Minimum rounds before checking (default 1)
            consecutive_rounds_needed: Stable rounds needed for convergence (default 1)
        """
        self.convergence_threshold = convergence_threshold
        self.divergence_threshold = divergence_threshold
        self.min_rounds_before_check = min_rounds_before_check
        self.consecutive_rounds_needed = consecutive_rounds_needed
        self.consecutive_stable_count = 0
        self.backend = self._select_backend()

        logger.info(
            f"ConvergenceDetector initialized with {self.backend.__class__.__name__}"
        )

    def _select_backend(self) -> SimilarityBackend:
        """
        Select best available similarity backend.

        Tries: SentenceTransformer -> TF-IDF -> Jaccard
        """
        env_override = _normalize_backend_name(
            os.getenv(_ENV_CONVERGENCE_BACKEND, "")
        )
        if env_override:
            try:
                backend = get_similarity_backend(env_override)
                logger.info(f"Using {env_override} backend via {_ENV_CONVERGENCE_BACKEND}")
                return backend
            except Exception as e:
                logger.warning(
                    f"{_ENV_CONVERGENCE_BACKEND}={env_override} failed: {e}. Falling back to auto."
                )

        # Try sentence transformers (best)
        try:
            backend = SentenceTransformerBackend()
            logger.info("Using SentenceTransformerBackend (best accuracy)")
            return backend
        except ImportError as e:
            logger.debug(f"sentence-transformers not available: {e}")
        except (RuntimeError, AttributeError) as e:
            # RuntimeError/AttributeError: transformers/scipy/numpy compatibility issues
            logger.debug(f"sentence-transformers failed to initialize: {e}")
        except OSError as e:
            # OSError can occur when model files are corrupted or missing
            logger.debug(f"sentence-transformers model error: {e}")

        # Try TF-IDF (good)
        try:
            backend = TFIDFBackend()
            logger.info("Using TFIDFBackend (good accuracy)")
            return backend
        except (ImportError, AttributeError, RuntimeError) as e:
            # AttributeError/RuntimeError: scipy/numpy version mismatch
            logger.debug(f"scikit-learn/scipy not available: {e}")

        # Fallback to Jaccard (always available)
        logger.info("Using JaccardBackend (fallback)")
        return JaccardBackend()

    def check_convergence(
        self,
        current_responses: dict[str, str],
        previous_responses: dict[str, str],
        round_number: int,
    ) -> Optional[ConvergenceResult]:
        """
        Check if debate has converged.

        Args:
            current_responses: Agent name -> response text for current round
            previous_responses: Agent name -> response text for previous round
            round_number: Current round number (1-indexed)

        Returns:
            ConvergenceResult or None if too early to check
        """
        # Don't check before minimum rounds
        if round_number <= self.min_rounds_before_check:
            return None

        # Match agents between rounds
        common_agents = set(current_responses.keys()) & set(previous_responses.keys())
        if not common_agents:
            logger.warning("No matching agents between rounds")
            return None

        # Compute similarity for each agent
        per_agent = {}
        agent_list = list(common_agents)

        # Use batch method if available (SentenceTransformerBackend)
        if hasattr(self.backend, 'compute_pairwise_similarities'):
            texts_current = [current_responses[a] for a in agent_list]
            texts_previous = [previous_responses[a] for a in agent_list]
            similarities = self.backend.compute_pairwise_similarities(
                texts_current, texts_previous
            )
            per_agent = dict(zip(agent_list, similarities))
        else:
            # Fallback to individual comparisons
            for agent in agent_list:
                similarity = self.backend.compute_similarity(
                    current_responses[agent], previous_responses[agent]
                )
                per_agent[agent] = similarity

        # Compute aggregate metrics
        similarities = list(per_agent.values())
        min_similarity = min(similarities)
        avg_similarity = sum(similarities) / len(similarities)

        # Determine status
        if min_similarity >= self.convergence_threshold:
            self.consecutive_stable_count += 1
            if self.consecutive_stable_count >= self.consecutive_rounds_needed:
                status = "converged"
                converged = True
            else:
                status = "refining"
                converged = False
        elif min_similarity < self.divergence_threshold:
            status = "diverging"
            converged = False
            self.consecutive_stable_count = 0
        else:
            status = "refining"
            converged = False
            self.consecutive_stable_count = 0

        return ConvergenceResult(
            converged=converged,
            status=status,
            min_similarity=min_similarity,
            avg_similarity=avg_similarity,
            per_agent_similarity=per_agent,
            consecutive_stable_rounds=self.consecutive_stable_count,
        )

    def reset(self) -> None:
        """Reset the consecutive stable count."""
        self.consecutive_stable_count = 0


__all__ = [
    # Cache (re-exported)
    "EmbeddingCache",
    "get_embedding_cache",
    "reset_embedding_cache",
    # Backends (re-exported)
    "SimilarityBackend",
    "JaccardBackend",
    "TFIDFBackend",
    "SentenceTransformerBackend",
    "get_similarity_backend",
    # Convergence
    "ConvergenceResult",
    "ArgumentDiversityMetric",
    "EvidenceConvergenceMetric",
    "StanceVolatilityMetric",
    "AdvancedConvergenceMetrics",
    "AdvancedConvergenceAnalyzer",
    "ConvergenceDetector",
]
