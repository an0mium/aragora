"""
ClaimCheck: Atomic Claim Decomposition and Verification.

Based on: Stepwise claim verification research.

Replaces simple keyword-based evidence grounding with atomic claim
decomposition for finer-grained verification.

Key features:
- Decompose complex claims into atomic sub-claims
- Multi-strategy verification (exact, semantic, inference)
- Confidence propagation from sub-claims to parent
- Integration with existing evidence grounding pipeline
"""

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable
from enum import Enum
import logging
import re
import time

logger = logging.getLogger(__name__)


class VerificationStrategy(Enum):
    """Strategies for verifying a claim against evidence."""

    EXACT_MATCH = "exact_match"  # Direct textual match
    SEMANTIC = "semantic"  # Embedding-based similarity
    INFERENCE = "inference"  # LLM-based logical inference
    CONTRADICTION = "contradiction"  # Check for contradicting evidence
    UNKNOWN = "unknown"  # Could not verify


class VerificationStatus(Enum):
    """Status of a claim verification."""

    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class AtomicClaim:
    """An atomic (indivisible) claim extracted from a complex statement."""

    id: str
    text: str
    parent_claim_id: str | None = None
    claim_type: str = "factual"  # factual, causal, comparative, temporal
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceMatch:
    """A match between a claim and supporting evidence."""

    claim_id: str
    evidence_id: str
    evidence_text: str
    strategy: VerificationStrategy
    match_score: float  # 0.0-1.0
    supporting: bool  # True if supports, False if contradicts
    explanation: str = ""


@dataclass
class ClaimVerificationResult:
    """Result of verifying a single atomic claim."""

    claim: AtomicClaim
    status: VerificationStatus
    confidence: float
    matches: list[EvidenceMatch]
    best_match: EvidenceMatch | None
    verification_time_ms: float = 0.0


@dataclass
class DecompositionResult:
    """Result of decomposing a complex claim."""

    original_claim: str
    atomic_claims: list[AtomicClaim]
    decomposition_strategy: str  # "rule_based" or "llm_based"
    total_claims: int
    decomposition_time_ms: float = 0.0


@dataclass
class FullVerificationResult:
    """Complete verification result for a claim and all sub-claims."""

    original_claim: str
    decomposition: DecompositionResult
    claim_results: list[ClaimVerificationResult]
    overall_status: VerificationStatus
    overall_confidence: float
    supporting_evidence_count: int
    contradicting_evidence_count: int
    total_time_ms: float = 0.0


@dataclass
class ClaimCheckConfig:
    """Configuration for ClaimCheck verifier."""

    # Decomposition settings
    use_llm_decomposition: bool = True
    max_atomic_claims: int = 10
    min_claim_words: int = 3

    # Verification settings
    exact_match_threshold: float = 0.95
    semantic_threshold: float = 0.75
    inference_threshold: float = 0.7
    contradiction_threshold: float = 0.6

    # Confidence settings
    min_matches_for_verified: int = 1
    partial_verification_threshold: float = 0.5

    # Strategy weights
    strategy_weights: dict[str, float] = field(
        default_factory=lambda: {
            "exact_match": 1.0,
            "semantic": 0.85,
            "inference": 0.7,
            "contradiction": 0.9,
        }
    )


class ClaimDecomposer:
    """
    Decomposes complex claims into atomic sub-claims.

    Supports two modes:
    1. Rule-based: Fast, pattern-matching decomposition
    2. LLM-based: More accurate, uses language model

    Example:
        decomposer = ClaimDecomposer()

        result = decomposer.decompose(
            "The Roman Empire fell in 476 AD due to both economic instability and barbarian invasions."
        )

        # Returns atomic claims:
        # - "The Roman Empire fell in 476 AD"
        # - "Economic instability contributed to the fall of the Roman Empire"
        # - "Barbarian invasions contributed to the fall of the Roman Empire"
    """

    # Patterns for rule-based decomposition
    CONJUNCTION_PATTERNS = [
        r"\b(and)\b",
        r"\b(as well as)\b",
        r"\b(along with)\b",
        r"\b(both\s+\w+\s+and)\b",
    ]

    CAUSAL_PATTERNS = [
        r"\b(because)\b",
        r"\b(due to)\b",
        r"\b(as a result of)\b",
        r"\b(caused by)\b",
        r"\b(led to)\b",
        r"\b(resulted in)\b",
    ]

    TEMPORAL_PATTERNS = [
        r"\b(before)\b",
        r"\b(after)\b",
        r"\b(during)\b",
        r"\b(while)\b",
        r"\b(when)\b",
    ]

    def __init__(self, config: ClaimCheckConfig | None = None):
        """Initialize the decomposer.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or ClaimCheckConfig()
        self._claim_counter = 0

    def decompose(
        self,
        claim: str,
        use_llm: bool = False,
        query_fn: Callable | None = None,
    ) -> DecompositionResult:
        """
        Decompose a claim into atomic sub-claims.

        Args:
            claim: The complex claim to decompose
            use_llm: Whether to use LLM for decomposition
            query_fn: Optional async function for LLM queries

        Returns:
            DecompositionResult with atomic claims
        """
        start_time = time.time()

        if use_llm and query_fn is not None:
            # LLM-based decomposition
            # Note: This would need to be made async in real usage
            atomic_claims = self._rule_based_decompose(claim)
            strategy = "rule_based"  # Fallback for sync context
        else:
            # Rule-based decomposition
            atomic_claims = self._rule_based_decompose(claim)
            strategy = "rule_based"

        # Limit claims
        atomic_claims = atomic_claims[: self.config.max_atomic_claims]

        decomposition_time_ms = (time.time() - start_time) * 1000

        return DecompositionResult(
            original_claim=claim,
            atomic_claims=atomic_claims,
            decomposition_strategy=strategy,
            total_claims=len(atomic_claims),
            decomposition_time_ms=decomposition_time_ms,
        )

    def _rule_based_decompose(self, claim: str) -> list[AtomicClaim]:
        """Decompose using rule-based pattern matching."""
        atomic_claims: list[AtomicClaim] = []
        claim = claim.strip()

        # Check for conjunctions
        parts = self._split_conjunctions(claim)

        # Check for causal relationships
        expanded_parts: list[tuple[str, str]] = []
        for part in parts:
            causal = self._extract_causal(part)
            if causal:
                for sub_claim, claim_type in causal:
                    expanded_parts.append((sub_claim, claim_type))
            else:
                expanded_parts.append((part, "factual"))

        # Create atomic claims
        for text, claim_type in expanded_parts:
            text = text.strip()
            if len(text.split()) >= self.config.min_claim_words:
                self._claim_counter += 1
                atomic_claims.append(
                    AtomicClaim(
                        id=f"claim_{self._claim_counter}",
                        text=text,
                        claim_type=claim_type,
                    )
                )

        # If no decomposition happened, return original as atomic
        if not atomic_claims:
            self._claim_counter += 1
            atomic_claims.append(
                AtomicClaim(
                    id=f"claim_{self._claim_counter}",
                    text=claim,
                    claim_type="factual",
                )
            )

        return atomic_claims

    def _split_conjunctions(self, text: str) -> list[str]:
        """Split text on conjunctions."""
        parts = [text]

        for pattern in self.CONJUNCTION_PATTERNS:
            new_parts = []
            for part in parts:
                splits = re.split(pattern, part, flags=re.IGNORECASE)
                new_parts.extend([s.strip() for s in splits if s.strip()])
            parts = new_parts

        return parts

    def _extract_causal(self, text: str) -> list[tuple[str, str]] | None:
        """Extract causal relationships from text."""
        for pattern in self.CAUSAL_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Split into cause and effect
                before = text[: match.start()].strip()
                after = text[match.end() :].strip()

                if before and after:
                    # Determine which is cause and which is effect
                    if any(
                        p in pattern for p in ["because", "due to", "caused by", "as a result of"]
                    ):
                        return [
                            (before, "factual"),
                            (f"{after} contributed to or caused: {before}", "causal"),
                        ]
                    else:  # led to, resulted in
                        return [
                            (before, "factual"),
                            (f"{before} led to: {after}", "causal"),
                        ]

        return None


class ClaimChecker:
    """
    Verifies atomic claims against evidence using multiple strategies.

    Example:
        checker = ClaimChecker()

        result = await checker.verify_claim(
            claim=AtomicClaim(id="1", text="The Roman Empire fell in 476 AD"),
            evidence_texts=["The Western Roman Empire ended in 476 AD when Romulus Augustulus was deposed."],
            query_fn=llm_query,
        )

        print(f"Status: {result.status}, Confidence: {result.confidence:.2f}")
    """

    INFERENCE_PROMPT = """Determine if the following evidence supports, contradicts, or is neutral to the claim.

CLAIM: {claim}

EVIDENCE: {evidence}

Respond in this format:
VERDICT: [SUPPORTS|CONTRADICTS|NEUTRAL]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [Brief explanation]"""

    def __init__(
        self,
        config: ClaimCheckConfig | None = None,
        embedding_fn: Callable | None = None,
    ):
        """Initialize the checker.

        Args:
            config: Configuration options
            embedding_fn: Optional function to get text embeddings
        """
        self.config = config or ClaimCheckConfig()
        self.embedding_fn = embedding_fn
        self._verification_history: list[ClaimVerificationResult] = []

    async def verify_claim(
        self,
        claim: AtomicClaim,
        evidence_texts: list[str],
        evidence_ids: list[str] | None = None,
        query_fn: Callable | None = None,
    ) -> ClaimVerificationResult:
        """
        Verify a single atomic claim against evidence.

        Args:
            claim: The atomic claim to verify
            evidence_texts: List of evidence texts to check against
            evidence_ids: Optional IDs for evidence items
            query_fn: Optional async function for LLM inference

        Returns:
            ClaimVerificationResult with matches and status
        """
        start_time = time.time()

        if evidence_ids is None:
            evidence_ids = [f"evidence_{i}" for i in range(len(evidence_texts))]

        matches: list[EvidenceMatch] = []

        for eid, etext in zip(evidence_ids, evidence_texts):
            # Try each verification strategy
            match = await self._check_evidence(
                claim=claim,
                evidence_id=eid,
                evidence_text=etext,
                query_fn=query_fn,
            )
            if match:
                matches.append(match)

        # Determine status and confidence
        status, confidence, best_match = self._aggregate_results(claim, matches)

        verification_time_ms = (time.time() - start_time) * 1000

        result = ClaimVerificationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            matches=matches,
            best_match=best_match,
            verification_time_ms=verification_time_ms,
        )

        self._verification_history.append(result)

        logger.debug(
            "claim_verified id=%s status=%s confidence=%.2f matches=%d time_ms=%.1f",
            claim.id,
            status.value,
            confidence,
            len(matches),
            verification_time_ms,
        )

        return result

    async def _check_evidence(
        self,
        claim: AtomicClaim,
        evidence_id: str,
        evidence_text: str,
        query_fn: Callable | None,
    ) -> EvidenceMatch | None:
        """Check if evidence supports/contradicts the claim."""
        claim_text = claim.text.lower()
        evidence_lower = evidence_text.lower()

        # Strategy 1: Exact match
        if claim_text in evidence_lower or evidence_lower in claim_text:
            return EvidenceMatch(
                claim_id=claim.id,
                evidence_id=evidence_id,
                evidence_text=evidence_text,
                strategy=VerificationStrategy.EXACT_MATCH,
                match_score=1.0,
                supporting=True,
                explanation="Direct textual match found",
            )

        # Strategy 2: Semantic similarity (if embedding function available)
        if self.embedding_fn:
            try:
                claim_emb = await self.embedding_fn(claim.text)
                evidence_emb = await self.embedding_fn(evidence_text)
                similarity = self._cosine_similarity(claim_emb, evidence_emb)

                if similarity >= self.config.semantic_threshold:
                    return EvidenceMatch(
                        claim_id=claim.id,
                        evidence_id=evidence_id,
                        evidence_text=evidence_text,
                        strategy=VerificationStrategy.SEMANTIC,
                        match_score=similarity,
                        supporting=True,
                        explanation=f"Semantic similarity: {similarity:.3f}",
                    )
            except (ValueError, TypeError, RuntimeError, OSError) as e:
                logger.warning("semantic_check_error: %s", e)

        # Strategy 3: LLM inference (if query function available)
        if query_fn:
            try:
                prompt = self.INFERENCE_PROMPT.format(
                    claim=claim.text,
                    evidence=evidence_text,
                )
                response = await query_fn(prompt)
                match = self._parse_inference_response(
                    response=response,
                    claim_id=claim.id,
                    evidence_id=evidence_id,
                    evidence_text=evidence_text,
                )
                if match and match.match_score >= self.config.inference_threshold:
                    return match
            except (
                ValueError,
                TypeError,
                RuntimeError,
                ConnectionError,
                TimeoutError,
                OSError,
            ) as e:
                logger.warning("inference_check_error: %s", e)

        # Strategy 4: Simple word overlap
        claim_words = set(claim_text.split())
        evidence_words = set(evidence_lower.split())
        overlap = len(claim_words & evidence_words) / max(len(claim_words), 1)

        if overlap >= 0.5:
            return EvidenceMatch(
                claim_id=claim.id,
                evidence_id=evidence_id,
                evidence_text=evidence_text,
                strategy=VerificationStrategy.SEMANTIC,
                match_score=overlap,
                supporting=True,
                explanation=f"Word overlap: {overlap:.2%}",
            )

        return None

    def _parse_inference_response(
        self,
        response: str,
        claim_id: str,
        evidence_id: str,
        evidence_text: str,
    ) -> EvidenceMatch | None:
        """Parse LLM inference response."""
        lines = response.strip().split("\n")

        verdict = "NEUTRAL"
        confidence = 0.5
        explanation = ""

        for line in lines:
            line = line.strip()
            if line.upper().startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip().upper()
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    conf_str = re.sub(r"[^\d.]", "", line.split(":", 1)[1].split()[0])
                    confidence = float(conf_str)
                    if confidence > 1.0:
                        confidence = confidence / 100.0
                except (ValueError, IndexError):
                    confidence = 0.5
            elif line.upper().startswith("EXPLANATION:"):
                explanation = line.split(":", 1)[1].strip()

        if verdict == "NEUTRAL":
            return None

        return EvidenceMatch(
            claim_id=claim_id,
            evidence_id=evidence_id,
            evidence_text=evidence_text,
            strategy=VerificationStrategy.INFERENCE,
            match_score=confidence,
            supporting=(verdict == "SUPPORTS"),
            explanation=explanation or f"LLM verdict: {verdict}",
        )

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np

        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def _aggregate_results(
        self,
        claim: AtomicClaim,
        matches: list[EvidenceMatch],
    ) -> tuple[VerificationStatus, float, EvidenceMatch | None]:
        """Aggregate match results into final status and confidence."""
        if not matches:
            return VerificationStatus.INSUFFICIENT_EVIDENCE, 0.0, None

        supporting = [m for m in matches if m.supporting]
        contradicting = [m for m in matches if not m.supporting]

        # Check for contradictions first
        if contradicting and (
            not supporting
            or max(m.match_score for m in contradicting) > max(m.match_score for m in supporting)
        ):
            best = max(contradicting, key=lambda m: m.match_score)
            return VerificationStatus.CONTRADICTED, best.match_score, best

        if not supporting:
            return VerificationStatus.UNVERIFIED, 0.0, None

        best = max(supporting, key=lambda m: m.match_score)

        # Weight by strategy
        weighted_score = best.match_score * self.config.strategy_weights.get(
            best.strategy.value, 0.5
        )

        if weighted_score >= self.config.semantic_threshold:
            return VerificationStatus.VERIFIED, weighted_score, best
        elif weighted_score >= self.config.partial_verification_threshold:
            return VerificationStatus.PARTIALLY_VERIFIED, weighted_score, best
        else:
            return VerificationStatus.UNVERIFIED, weighted_score, best

    def reset(self) -> None:
        """Reset checker state."""
        self._verification_history.clear()

    def get_metrics(self) -> dict[str, Any]:
        """Get checker metrics for telemetry."""
        if not self._verification_history:
            return {
                "total_verifications": 0,
                "status_distribution": {},
                "avg_confidence": 0.0,
            }

        total = len(self._verification_history)
        statuses = [r.status.value for r in self._verification_history]
        status_counts: dict[str, int] = {}
        for s in statuses:
            status_counts[s] = status_counts.get(s, 0) + 1

        return {
            "total_verifications": total,
            "status_distribution": {k: v / total for k, v in status_counts.items()},
            "status_counts": status_counts,
            "avg_confidence": sum(r.confidence for r in self._verification_history) / total,
            "avg_time_ms": sum(r.verification_time_ms for r in self._verification_history) / total,
        }


class ClaimCheckVerifier:
    """
    Full claim verification pipeline combining decomposition and checking.

    Example:
        verifier = ClaimCheckVerifier()

        result = await verifier.verify(
            claim="The Roman Empire fell in 476 AD due to economic instability.",
            evidence=["The Western Roman Empire ended in 476 AD.", "Economic troubles plagued Rome."],
            query_fn=llm_query,
        )

        print(f"Overall: {result.overall_status}, Confidence: {result.overall_confidence:.2f}")
    """

    def __init__(
        self,
        config: ClaimCheckConfig | None = None,
        embedding_fn: Callable | None = None,
    ):
        """Initialize the verifier.

        Args:
            config: Configuration options
            embedding_fn: Optional function to get embeddings
        """
        self.config = config or ClaimCheckConfig()
        self.decomposer = ClaimDecomposer(config)
        self.checker = ClaimChecker(config, embedding_fn)

    async def verify(
        self,
        claim: str,
        evidence: list[str],
        evidence_ids: list[str] | None = None,
        query_fn: Callable | None = None,
    ) -> FullVerificationResult:
        """
        Fully verify a claim by decomposing and checking each sub-claim.

        Args:
            claim: The complex claim to verify
            evidence: List of evidence texts
            evidence_ids: Optional IDs for evidence items
            query_fn: Optional async function for LLM queries

        Returns:
            FullVerificationResult with complete analysis
        """
        start_time = time.time()

        # Decompose claim
        decomposition = self.decomposer.decompose(
            claim=claim,
            use_llm=self.config.use_llm_decomposition,
            query_fn=query_fn,
        )

        # Verify each atomic claim
        claim_results: list[ClaimVerificationResult] = []
        for atomic_claim in decomposition.atomic_claims:
            result = await self.checker.verify_claim(
                claim=atomic_claim,
                evidence_texts=evidence,
                evidence_ids=evidence_ids,
                query_fn=query_fn,
            )
            claim_results.append(result)

        # Aggregate overall status
        overall_status, overall_confidence = self._aggregate_overall(claim_results)

        # Count evidence types
        supporting = sum(1 for r in claim_results for m in r.matches if m.supporting)
        contradicting = sum(1 for r in claim_results for m in r.matches if not m.supporting)

        total_time_ms = (time.time() - start_time) * 1000

        return FullVerificationResult(
            original_claim=claim,
            decomposition=decomposition,
            claim_results=claim_results,
            overall_status=overall_status,
            overall_confidence=overall_confidence,
            supporting_evidence_count=supporting,
            contradicting_evidence_count=contradicting,
            total_time_ms=total_time_ms,
        )

    def _aggregate_overall(
        self,
        results: list[ClaimVerificationResult],
    ) -> tuple[VerificationStatus, float]:
        """Aggregate results from all sub-claims."""
        if not results:
            return VerificationStatus.INSUFFICIENT_EVIDENCE, 0.0

        statuses = [r.status for r in results]
        confidences = [r.confidence for r in results]

        # If any contradicted, overall is contradicted
        if VerificationStatus.CONTRADICTED in statuses:
            return VerificationStatus.CONTRADICTED, min(confidences)

        # Calculate verification ratio
        verified_count = sum(
            1
            for s in statuses
            if s in (VerificationStatus.VERIFIED, VerificationStatus.PARTIALLY_VERIFIED)
        )
        ratio = verified_count / len(statuses)

        avg_confidence = sum(confidences) / len(confidences)

        if ratio >= 0.8:
            return VerificationStatus.VERIFIED, avg_confidence
        elif ratio >= 0.5:
            return VerificationStatus.PARTIALLY_VERIFIED, avg_confidence
        elif ratio > 0:
            return VerificationStatus.UNVERIFIED, avg_confidence
        else:
            return VerificationStatus.INSUFFICIENT_EVIDENCE, 0.0


# Convenience functions


def create_claim_verifier(
    use_llm: bool = True,
    embedding_fn: Callable | None = None,
    **kwargs: Any,
) -> ClaimCheckVerifier:
    """Create a ClaimCheck verifier with common configuration.

    Args:
        use_llm: Whether to use LLM for decomposition/inference
        embedding_fn: Optional embedding function
        **kwargs: Additional config options

    Returns:
        Configured ClaimCheckVerifier
    """
    config = ClaimCheckConfig(use_llm_decomposition=use_llm, **kwargs)
    return ClaimCheckVerifier(config, embedding_fn)
