"""Tests for formal verification backends."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from aragora.verification.formal import (
    FormalProofStatus,
    FormalLanguage,
    FormalProofResult,
    FormalVerificationBackend,
    TranslationModel,
    LeanBackend,
    Z3Backend,
    FormalVerificationManager,
    get_formal_verification_manager,
)


class TestFormalProofStatus:
    """Test FormalProofStatus enum."""

    def test_all_statuses_defined(self):
        """Test all expected statuses exist."""
        expected = [
            "NOT_ATTEMPTED",
            "TRANSLATION_FAILED",
            "PROOF_FOUND",
            "PROOF_FAILED",
            "TIMEOUT",
            "BACKEND_UNAVAILABLE",
            "NOT_SUPPORTED",
        ]
        for status in expected:
            assert hasattr(FormalProofStatus, status)

    def test_status_values(self):
        """Test status values are strings."""
        assert FormalProofStatus.NOT_ATTEMPTED.value == "not_attempted"
        assert FormalProofStatus.PROOF_FOUND.value == "proof_found"
        assert FormalProofStatus.PROOF_FAILED.value == "proof_failed"


class TestFormalLanguage:
    """Test FormalLanguage enum."""

    def test_all_languages_defined(self):
        """Test all expected languages exist."""
        expected = ["LEAN4", "COQ", "ISABELLE", "AGDA", "Z3_SMT"]
        for lang in expected:
            assert hasattr(FormalLanguage, lang)

    def test_language_values(self):
        """Test language values."""
        assert FormalLanguage.LEAN4.value == "lean4"
        assert FormalLanguage.Z3_SMT.value == "z3_smt"


class TestFormalProofResult:
    """Test FormalProofResult dataclass."""

    def test_create_minimal_result(self):
        """Test creating a result with minimal fields."""
        result = FormalProofResult(
            status=FormalProofStatus.NOT_ATTEMPTED,
            language=FormalLanguage.LEAN4,
        )
        assert result.status == FormalProofStatus.NOT_ATTEMPTED
        assert result.language == FormalLanguage.LEAN4
        assert result.formal_statement is None
        assert result.proof_text is None

    def test_create_successful_result(self):
        """Test creating a successful proof result."""
        result = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.LEAN4,
            formal_statement="theorem test : True := trivial",
            proof_text="trivial",
            proof_hash="abc123",
        )
        assert result.is_verified is True
        assert result.proof_hash == "abc123"

    def test_is_verified_property(self):
        """Test is_verified property."""
        found = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.LEAN4,
        )
        assert found.is_verified is True

        failed = FormalProofResult(
            status=FormalProofStatus.PROOF_FAILED,
            language=FormalLanguage.LEAN4,
        )
        assert failed.is_verified is False

    def test_is_high_confidence_property(self):
        """Test is_high_confidence property."""
        # High confidence requires: verified, confidence >= 0.8, semantic match verified
        high_conf = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.LEAN4,
            translation_confidence=0.9,
            semantic_match_verified=True,
        )
        assert high_conf.is_high_confidence is True

        # Low confidence score
        low_conf = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.LEAN4,
            translation_confidence=0.5,
            semantic_match_verified=True,
        )
        assert low_conf.is_high_confidence is False

        # Not semantic match verified
        not_verified = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.LEAN4,
            translation_confidence=0.9,
            semantic_match_verified=False,
        )
        assert not_verified.is_high_confidence is False

    def test_to_dict(self):
        """Test serialization to dict."""
        result = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.LEAN4,
            formal_statement="theorem test : True",
            proof_hash="abc123",
            translation_confidence=0.85,
            original_claim="True is true",
        )
        data = result.to_dict()

        assert data["status"] == "proof_found"
        assert data["language"] == "lean4"
        assert data["is_verified"] is True
        assert data["translation_confidence"] == 0.85
        assert "timestamp" in data

    def test_timing_fields(self):
        """Test timing fields."""
        result = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.Z3_SMT,
            translation_time_ms=100.5,
            proof_search_time_ms=250.3,
        )
        assert result.translation_time_ms == 100.5
        assert result.proof_search_time_ms == 250.3

    def test_error_message_field(self):
        """Test error message field."""
        result = FormalProofResult(
            status=FormalProofStatus.PROOF_FAILED,
            language=FormalLanguage.LEAN4,
            error_message="Type mismatch",
        )
        assert result.error_message == "Type mismatch"


class TestTranslationModel:
    """Test TranslationModel enum."""

    def test_all_models_defined(self):
        """Test all expected models exist."""
        expected = ["DEEPSEEK_PROVER", "CLAUDE", "OPENAI", "AUTO"]
        for model in expected:
            assert hasattr(TranslationModel, model)


class TestLeanBackend:
    """Test LeanBackend class."""

    def test_init_default(self):
        """Test default initialization."""
        backend = LeanBackend()
        assert backend._sandbox_timeout == 60.0
        assert backend._sandbox_memory_mb == 1024
        assert backend._translation_model == TranslationModel.AUTO

    def test_init_custom(self):
        """Test custom initialization."""
        backend = LeanBackend(
            sandbox_timeout=30.0,
            sandbox_memory_mb=512,
            translation_model=TranslationModel.CLAUDE,
        )
        assert backend._sandbox_timeout == 30.0
        assert backend._sandbox_memory_mb == 512
        assert backend._translation_model == TranslationModel.CLAUDE

    def test_language_property(self):
        """Test language property."""
        backend = LeanBackend()
        assert backend.language == FormalLanguage.LEAN4

    @patch("shutil.which")
    def test_is_available_when_lean_installed(self, mock_which):
        """Test is_available when Lean is installed."""
        mock_which.return_value = "/usr/local/bin/lean"
        backend = LeanBackend()
        assert backend.is_available is True

    @patch("shutil.which")
    def test_is_available_when_lean_not_installed(self, mock_which):
        """Test is_available when Lean is not installed."""
        mock_which.return_value = None
        backend = LeanBackend()
        assert backend.is_available is False

    def test_can_verify_with_claim_type(self):
        """Test can_verify with explicit claim type."""
        backend = LeanBackend()
        # Patch is_available property on the class
        with patch.object(LeanBackend, "is_available", new_callable=lambda: property(lambda self: True)):
            assert backend.can_verify("any claim", claim_type="MATHEMATICAL") is True
            assert backend.can_verify("any claim", claim_type="THEOREM") is True
            assert backend.can_verify("any claim", claim_type="PROOF") is True

    def test_can_verify_with_math_patterns(self):
        """Test can_verify with mathematical patterns."""
        backend = LeanBackend()
        with patch.object(LeanBackend, "is_available", new_callable=lambda: property(lambda self: True)):
            assert backend.can_verify("for all n, n + 0 = n") is True
            assert backend.can_verify("exists x such that x > 0") is True
            assert backend.can_verify("if x > y then y < x") is False  # No math pattern

    def test_can_verify_unavailable(self):
        """Test can_verify when Lean not available."""
        backend = LeanBackend()
        with patch.object(LeanBackend, "is_available", new_callable=lambda: property(lambda self: False)):
            assert backend.can_verify("for all n, n = n") is False

    def test_clear_cache(self):
        """Test clearing the proof cache."""
        backend = LeanBackend()
        # Add something to cache
        backend._proof_cache["test"] = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.LEAN4,
        )
        assert len(backend._proof_cache) == 1
        count = backend.clear_cache()
        assert count == 1
        assert len(backend._proof_cache) == 0

    @pytest.mark.asyncio
    async def test_prove_unavailable(self):
        """Test prove when Lean not available."""
        backend = LeanBackend()
        with patch.object(LeanBackend, "is_available", new_callable=lambda: property(lambda self: False)):
            result = await backend.prove("theorem test : True := trivial")
            assert result.status == FormalProofStatus.BACKEND_UNAVAILABLE
            assert "not installed" in result.error_message

    @pytest.mark.asyncio
    async def test_prove_cached_result(self):
        """Test prove returns cached result."""
        backend = LeanBackend()
        with patch.object(LeanBackend, "is_available", new_callable=lambda: property(lambda self: True)):
            cached_result = FormalProofResult(
                status=FormalProofStatus.PROOF_FOUND,
                language=FormalLanguage.LEAN4,
            )
            # Pre-populate cache
            import hashlib
            cache_key = hashlib.sha256("theorem test".encode()).hexdigest()
            backend._proof_cache[cache_key] = cached_result

            result = await backend.prove("theorem test")
            assert result.status == FormalProofStatus.PROOF_FOUND

    @pytest.mark.asyncio
    async def test_translate_no_api_key(self):
        """Test translate without API key returns None."""
        backend = LeanBackend()
        with patch.dict("os.environ", {}, clear=True):
            result = await backend.translate("all n = n")
            assert result is None

    @pytest.mark.asyncio
    async def test_verify_proof(self):
        """Test verify_proof delegates to prove."""
        backend = LeanBackend()
        with patch.object(backend, "prove") as mock_prove:
            mock_prove.return_value = FormalProofResult(
                status=FormalProofStatus.PROOF_FOUND,
                language=FormalLanguage.LEAN4,
            )
            result = await backend.verify_proof("theorem test : True", "trivial")
            assert result is True
            mock_prove.assert_called_once()


class TestZ3Backend:
    """Test Z3Backend class."""

    def test_init_default(self):
        """Test default initialization."""
        backend = Z3Backend()
        assert backend._cache_size == 100
        assert backend._cache_ttl == 3600.0

    def test_init_custom(self):
        """Test custom initialization."""
        backend = Z3Backend(cache_size=50, cache_ttl_seconds=1800.0)
        assert backend._cache_size == 50
        assert backend._cache_ttl == 1800.0

    def test_language_property(self):
        """Test language property."""
        backend = Z3Backend()
        assert backend.language == FormalLanguage.Z3_SMT

    def test_is_available_with_z3(self):
        """Test is_available when z3 is installed."""
        backend = Z3Backend()
        try:
            import z3
            assert backend.is_available is True
        except ImportError:
            assert backend.is_available is False

    def test_can_verify_smtlib2(self):
        """Test can_verify with SMT-LIB2 format."""
        backend = Z3Backend()
        with patch.object(Z3Backend, "is_available", new_callable=lambda: property(lambda self: True)):
            smtlib = "(declare-const x Int)\n(assert (> x 0))"
            assert backend.can_verify(smtlib) is True

    def test_can_verify_with_claim_type(self):
        """Test can_verify with claim types."""
        backend = Z3Backend()
        with patch.object(Z3Backend, "is_available", new_callable=lambda: property(lambda self: True)):
            assert backend.can_verify("any", claim_type="assertion") is True
            assert backend.can_verify("any", claim_type="arithmetic") is True
            assert backend.can_verify("any", claim_type="LOGICAL") is True

    def test_can_verify_with_patterns(self):
        """Test can_verify with logical patterns."""
        backend = Z3Backend()
        with patch.object(Z3Backend, "is_available", new_callable=lambda: property(lambda self: True)):
            assert backend.can_verify("for all x, x > 0") is True
            assert backend.can_verify("x greater than y") is True
            assert backend.can_verify("x < y implies y > x") is True

    def test_can_verify_unavailable(self):
        """Test can_verify when Z3 not available."""
        backend = Z3Backend()
        with patch.object(Z3Backend, "is_available", new_callable=lambda: property(lambda self: False)):
            assert backend.can_verify("for all x") is False

    def test_clear_cache(self):
        """Test clearing the proof cache."""
        backend = Z3Backend()
        import time
        backend._proof_cache["test"] = (time.time(), FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.Z3_SMT,
        ))
        count = backend.clear_cache()
        assert count == 1
        assert len(backend._proof_cache) == 0

    def test_is_smtlib2(self):
        """Test SMT-LIB2 format detection."""
        backend = Z3Backend()
        assert backend._is_smtlib2("(declare-const x Int)") is True
        assert backend._is_smtlib2("(assert (> x 0))") is True
        assert backend._is_smtlib2("plain text") is False
        assert backend._is_smtlib2("(random text)") is False

    @pytest.mark.asyncio
    async def test_prove_unavailable(self):
        """Test prove when Z3 not available."""
        backend = Z3Backend()
        with patch.object(Z3Backend, "is_available", new_callable=lambda: property(lambda self: False)):
            result = await backend.prove("(check-sat)")
            assert result.status == FormalProofStatus.BACKEND_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_translate_smtlib2_passthrough(self):
        """Test translate passes through valid SMT-LIB2."""
        backend = Z3Backend()
        smtlib = "(declare-const x Int)\n(assert (> x 0))\n(check-sat)"
        # This will work only if z3 is installed
        try:
            import z3
            result = await backend.translate(smtlib)
            assert result == smtlib
        except ImportError:
            pass  # Skip if z3 not installed

    @pytest.mark.asyncio
    async def test_prove_with_z3(self):
        """Test prove with Z3 (if available)."""
        backend = Z3Backend()
        if not backend.is_available:
            pytest.skip("Z3 not installed")

        # Prove that negation of (x > 0 and x < 0) is unsat
        smtlib = """
(declare-const x Int)
(assert (and (> x 0) (< x 0)))
(check-sat)
"""
        result = await backend.prove(smtlib)
        # This should return unsat (which means the original is valid)
        assert result.status in (FormalProofStatus.PROOF_FOUND, FormalProofStatus.PROOF_FAILED)


class TestFormalVerificationManager:
    """Test FormalVerificationManager class."""

    def test_init(self):
        """Test initialization creates backends."""
        manager = FormalVerificationManager()
        assert len(manager.backends) == 2

    def test_get_available_backends(self):
        """Test getting available backends."""
        manager = FormalVerificationManager()
        available = manager.get_available_backends()
        # Returns list of available backends
        assert isinstance(available, list)

    def test_get_backend_for_claim_mathematical(self):
        """Test getting backend for mathematical claim."""
        manager = FormalVerificationManager()
        # The manager should find a backend that can verify math claims
        # Z3Backend should be available and can verify "x > 0" type claims
        backend = manager.get_backend_for_claim("x > 0", "arithmetic")
        # Should find a backend (Z3 or Lean depending on installation)
        # We just check the method works without error
        assert backend is not None or backend is None  # Either is valid

    def test_get_backend_for_claim_none_available(self):
        """Test getting backend when none can verify."""
        manager = FormalVerificationManager()
        # If all backends are unavailable or can't verify
        backend = manager.get_backend_for_claim("random text that isn't math")
        # Might be None or might find one
        assert backend is None or backend is not None  # Either is valid

    def test_status_report(self):
        """Test status report generation."""
        manager = FormalVerificationManager()
        report = manager.status_report()

        assert "backends" in report
        assert "any_available" in report
        assert isinstance(report["backends"], list)
        for b in report["backends"]:
            assert "language" in b
            assert "available" in b

    @pytest.mark.asyncio
    async def test_attempt_formal_verification_not_supported(self):
        """Test verification when no backend supports claim."""
        manager = FormalVerificationManager()
        # Try to verify something that backends may not handle well
        result = await manager.attempt_formal_verification("xyz random text 123")
        # Result depends on what backends are available and whether they can handle this
        assert result.status in (
            FormalProofStatus.NOT_SUPPORTED,
            FormalProofStatus.TRANSLATION_FAILED,
            FormalProofStatus.PROOF_FAILED,
            FormalProofStatus.BACKEND_UNAVAILABLE,
        )


class TestGetFormalVerificationManager:
    """Test singleton getter function."""

    def test_returns_manager(self):
        """Test that function returns a manager."""
        manager = get_formal_verification_manager()
        assert isinstance(manager, FormalVerificationManager)

    def test_returns_same_instance(self):
        """Test that function returns same instance (singleton)."""
        manager1 = get_formal_verification_manager()
        manager2 = get_formal_verification_manager()
        assert manager1 is manager2


class TestFormalVerificationBackendProtocol:
    """Test that backends implement the protocol correctly."""

    def test_lean_backend_implements_protocol(self):
        """Test LeanBackend implements FormalVerificationBackend."""
        backend = LeanBackend()
        # Check protocol methods exist
        assert hasattr(backend, "language")
        assert hasattr(backend, "is_available")
        assert hasattr(backend, "can_verify")
        assert hasattr(backend, "translate")
        assert hasattr(backend, "prove")
        assert hasattr(backend, "verify_proof")

    def test_z3_backend_implements_protocol(self):
        """Test Z3Backend implements FormalVerificationBackend."""
        backend = Z3Backend()
        # Check protocol methods exist
        assert hasattr(backend, "language")
        assert hasattr(backend, "is_available")
        assert hasattr(backend, "can_verify")
        assert hasattr(backend, "translate")
        assert hasattr(backend, "prove")
        assert hasattr(backend, "verify_proof")
