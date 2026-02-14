"""Tests for ClaimCheck verifier."""

import pytest
import asyncio
from aragora.evidence.claim_check import (
    ClaimDecomposer,
    ClaimChecker,
    ClaimCheckVerifier,
    ClaimCheckConfig,
    AtomicClaim,
    VerificationStatus,
    VerificationStrategy,
    create_claim_verifier,
)


class TestClaimDecomposer:
    """Test suite for ClaimDecomposer."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        decomposer = ClaimDecomposer()
        assert decomposer.config.max_atomic_claims == 10

    def test_simple_claim_no_decomposition(self) -> None:
        """Test that simple claims stay atomic."""
        decomposer = ClaimDecomposer()

        result = decomposer.decompose("The sky is blue.")

        assert len(result.atomic_claims) == 1
        assert "sky is blue" in result.atomic_claims[0].text.lower()

    def test_conjunction_decomposition(self) -> None:
        """Test splitting on conjunctions."""
        decomposer = ClaimDecomposer()

        result = decomposer.decompose(
            "The Roman Empire was powerful and the Greek civilization was influential."
        )

        assert len(result.atomic_claims) >= 2
        texts = [c.text.lower() for c in result.atomic_claims]
        assert any("roman" in t for t in texts)
        assert any("greek" in t for t in texts)

    def test_causal_decomposition(self) -> None:
        """Test extracting causal relationships."""
        decomposer = ClaimDecomposer()

        result = decomposer.decompose("The economy crashed because of excessive debt.")

        assert len(result.atomic_claims) >= 1
        # Should identify causal relationship
        types = [c.claim_type for c in result.atomic_claims]
        assert "causal" in types or "factual" in types

    def test_complex_claim_decomposition(self) -> None:
        """Test decomposition of complex claim."""
        decomposer = ClaimDecomposer()

        result = decomposer.decompose(
            "The Roman Empire fell in 476 AD due to both economic instability and barbarian invasions."
        )

        assert result.total_claims >= 2
        assert result.decomposition_strategy == "rule_based"

    def test_respects_max_claims(self) -> None:
        """Test that max_atomic_claims is respected."""
        config = ClaimCheckConfig(max_atomic_claims=2)
        decomposer = ClaimDecomposer(config)

        result = decomposer.decompose("A and B and C and D and E are all important.")

        assert len(result.atomic_claims) <= 2


class TestClaimChecker:
    """Test suite for ClaimChecker."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        checker = ClaimChecker()
        assert checker.config.semantic_threshold == 0.75

    @pytest.mark.asyncio
    async def test_exact_match_verification(self) -> None:
        """Test verification with exact text match."""
        checker = ClaimChecker()

        claim = AtomicClaim(id="1", text="The sky is blue")
        evidence = ["The sky is blue during the day.", "Water is wet."]

        result = await checker.verify_claim(
            claim=claim,
            evidence_texts=evidence,
        )

        assert result.status == VerificationStatus.VERIFIED
        assert result.best_match is not None
        assert result.best_match.strategy == VerificationStrategy.EXACT_MATCH

    @pytest.mark.asyncio
    async def test_no_evidence_match(self) -> None:
        """Test verification with no matching evidence."""
        checker = ClaimChecker()

        claim = AtomicClaim(id="1", text="Quantum computing uses qubits.")
        evidence = ["The weather is nice today.", "Cats are mammals."]

        result = await checker.verify_claim(
            claim=claim,
            evidence_texts=evidence,
        )

        # Should be insufficient or unverified (no strong match)
        assert result.status in (
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            VerificationStatus.UNVERIFIED,
        )

    @pytest.mark.asyncio
    async def test_word_overlap_verification(self) -> None:
        """Test verification with significant word overlap."""
        checker = ClaimChecker()

        claim = AtomicClaim(id="1", text="The Roman Empire was powerful.")
        evidence = ["The Roman Empire controlled vast territories and was very powerful."]

        result = await checker.verify_claim(
            claim=claim,
            evidence_texts=evidence,
        )

        # Should find match due to overlap
        assert len(result.matches) > 0

    def test_get_metrics_empty(self) -> None:
        """Test metrics when no verifications done."""
        checker = ClaimChecker()
        metrics = checker.get_metrics()

        assert metrics["total_verifications"] == 0

    @pytest.mark.asyncio
    async def test_get_metrics_with_data(self) -> None:
        """Test metrics after verifications."""
        checker = ClaimChecker()

        claim = AtomicClaim(id="1", text="Test claim")
        await checker.verify_claim(claim, ["Test evidence matching claim"])

        metrics = checker.get_metrics()
        assert metrics["total_verifications"] == 1


class TestClaimCheckVerifier:
    """Test suite for full ClaimCheckVerifier."""

    def test_init(self) -> None:
        """Test initialization."""
        verifier = ClaimCheckVerifier()
        assert verifier.decomposer is not None
        assert verifier.checker is not None

    @pytest.mark.asyncio
    async def test_full_verification_simple(self) -> None:
        """Test full verification of a simple claim."""
        verifier = ClaimCheckVerifier()

        result = await verifier.verify(
            claim="The sky is blue.",
            evidence=["The sky is blue on clear days."],
        )

        assert result.original_claim == "The sky is blue."
        assert result.decomposition.total_claims >= 1
        assert result.overall_status in (
            VerificationStatus.VERIFIED,
            VerificationStatus.PARTIALLY_VERIFIED,
            VerificationStatus.UNVERIFIED,
        )

    @pytest.mark.asyncio
    async def test_full_verification_complex(self) -> None:
        """Test full verification of a complex claim."""
        verifier = ClaimCheckVerifier()

        result = await verifier.verify(
            claim="Rome fell in 476 AD and the economy collapsed.",
            evidence=[
                "The Western Roman Empire ended in 476 AD.",
                "Economic troubles plagued Rome in its final centuries.",
            ],
        )

        assert result.decomposition.total_claims >= 1
        assert result.total_time_ms > 0

    @pytest.mark.asyncio
    async def test_insufficient_evidence(self) -> None:
        """Test with no relevant evidence."""
        verifier = ClaimCheckVerifier()

        result = await verifier.verify(
            claim="Quantum computing will revolutionize cryptography.",
            evidence=["The weather was nice yesterday.", "Cats like to sleep."],
        )

        # Should not be verified
        assert result.overall_status in (
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            VerificationStatus.UNVERIFIED,
        )

    @pytest.mark.asyncio
    async def test_evidence_counts(self) -> None:
        """Test that evidence counts are tracked."""
        verifier = ClaimCheckVerifier()

        result = await verifier.verify(
            claim="Water is essential for life.",
            evidence=[
                "Water is necessary for all known forms of life.",
                "Life requires water to survive.",
            ],
        )

        assert result.supporting_evidence_count >= 0
        assert result.contradicting_evidence_count >= 0


class TestVerificationStatus:
    """Test VerificationStatus enum."""

    def test_all_statuses_have_values(self) -> None:
        """Test that all statuses have string values."""
        assert VerificationStatus.VERIFIED.value == "verified"
        assert VerificationStatus.PARTIALLY_VERIFIED.value == "partially_verified"
        assert VerificationStatus.UNVERIFIED.value == "unverified"
        assert VerificationStatus.CONTRADICTED.value == "contradicted"
        assert VerificationStatus.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"


class TestCreateClaimVerifier:
    """Test the factory function."""

    def test_creates_with_defaults(self) -> None:
        """Test factory creates verifier with defaults."""
        verifier = create_claim_verifier()
        assert isinstance(verifier, ClaimCheckVerifier)

    def test_creates_with_custom_config(self) -> None:
        """Test factory accepts custom configuration."""
        verifier = create_claim_verifier(
            use_llm=False,
            semantic_threshold=0.8,
        )
        assert verifier.config.use_llm_decomposition is False
        assert verifier.config.semantic_threshold == 0.8
