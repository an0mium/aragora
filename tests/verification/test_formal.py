"""Tests for the formal verification module.

Tests cover:
- FormalProofStatus enum values
- FormalLanguage enum values  
- FormalProofResult dataclass
- FormalVerificationBackend protocol
- LeanBackend implementation
- Z3Backend implementation
- FormalVerificationManager singleton
- Translation model configuration
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

from aragora.verification.formal import (
    FormalLanguage,
    FormalProofResult,
    FormalProofStatus,
    FormalVerificationBackend,
    FormalVerificationManager,
    LeanBackend,
    TranslationModel,
    Z3Backend,
    get_formal_verification_manager,
)


class TestFormalProofStatus:
    """Tests for FormalProofStatus enum."""

    def test_all_statuses_exist(self):
        """All expected status values should be defined."""
        assert FormalProofStatus.NOT_ATTEMPTED.value == "not_attempted"
        assert FormalProofStatus.TRANSLATION_FAILED.value == "translation_failed"
        assert FormalProofStatus.PROOF_FOUND.value == "proof_found"
        assert FormalProofStatus.PROOF_FAILED.value == "proof_failed"
        assert FormalProofStatus.TIMEOUT.value == "timeout"
        assert FormalProofStatus.BACKEND_UNAVAILABLE.value == "backend_unavailable"
        assert FormalProofStatus.NOT_SUPPORTED.value == "not_supported"

    def test_status_count(self):
        """Should have exactly 7 status values."""
        assert len(FormalProofStatus) == 7


class TestFormalLanguage:
    """Tests for FormalLanguage enum."""

    def test_all_languages_exist(self):
        """All expected languages should be defined."""
        assert FormalLanguage.LEAN4.value == "lean4"
        assert FormalLanguage.COQ.value == "coq"
        assert FormalLanguage.ISABELLE.value == "isabelle"
        assert FormalLanguage.AGDA.value == "agda"
        assert FormalLanguage.Z3_SMT.value == "z3_smt"

    def test_language_count(self):
        """Should have exactly 5 languages."""
        assert len(FormalLanguage) == 5


class TestFormalProofResult:
    """Tests for FormalProofResult dataclass."""

    def test_minimal_creation(self):
        """Should create with required fields only."""
        result = FormalProofResult(
            status=FormalProofStatus.NOT_ATTEMPTED,
            language=FormalLanguage.LEAN4,
        )
        assert result.status == FormalProofStatus.NOT_ATTEMPTED
        assert result.language == FormalLanguage.LEAN4
        assert result.formal_statement is None
        assert result.proof_text is None

    def test_successful_proof_result(self):
        """Should store successful proof details."""
        result = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.LEAN4,
            formal_statement="theorem test : 1 + 1 = 2",
            proof_text="by rfl",
            proof_hash="abc123",
            translation_time_ms=100.0,
            proof_search_time_ms=500.0,
        )
        assert result.status == FormalProofStatus.PROOF_FOUND
        assert result.formal_statement is not None
        assert result.proof_text == "by rfl"

    def test_is_verified_property(self):
        """is_verified should be True only for PROOF_FOUND."""
        verified = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.LEAN4,
        )
        not_verified = FormalProofResult(
            status=FormalProofStatus.PROOF_FAILED,
            language=FormalLanguage.LEAN4,
        )
        assert verified.is_verified is True
        assert not_verified.is_verified is False


class TestTranslationModel:
    """Tests for TranslationModel enum."""

    def test_model_values(self):
        """All expected models should be defined."""
        assert TranslationModel.DEEPSEEK_PROVER.value == "deepseek_prover"
        assert TranslationModel.CLAUDE.value == "claude"
        assert TranslationModel.OPENAI.value == "openai"
        assert TranslationModel.AUTO.value == "auto"

    def test_default_model(self):
        """Should have a default model."""
        assert TranslationModel.DEEPSEEK_PROVER is not None
        assert TranslationModel.AUTO is not None


class TestLeanBackend:
    """Tests for LeanBackend class."""

    def test_initialization(self):
        """Should initialize with default values."""
        backend = LeanBackend()
        assert backend.language == FormalLanguage.LEAN4
        assert backend._sandbox_timeout is not None

    def test_initialization_custom_timeout(self):
        """Should accept custom sandbox timeout."""
        backend = LeanBackend(sandbox_timeout=60.0)
        assert backend._sandbox_timeout == 60.0

    def test_initialization_custom_model(self):
        """Should accept custom translation model."""
        backend = LeanBackend(translation_model=TranslationModel.CLAUDE)
        assert backend._translation_model == TranslationModel.CLAUDE

    def test_is_available_without_lean(self):
        """Should return False when Lean is not installed."""
        backend = LeanBackend()
        with patch("shutil.which", return_value=None):
            # is_available is a property, not an async method
            is_available = backend.is_available
            assert is_available is False

    def test_can_verify_without_lean(self):
        """Should return False when Lean is not installed."""
        backend = LeanBackend()
        with patch("shutil.which", return_value=None):
            # can_verify checks is_available internally
            assert backend.can_verify("for all x, x = x") is False


class TestZ3Backend:
    """Tests for Z3Backend class."""

    def test_initialization(self):
        """Should initialize with default values."""
        backend = Z3Backend()
        assert backend.language == FormalLanguage.Z3_SMT
        assert backend._cache_size is not None

    def test_initialization_custom_cache_size(self):
        """Should accept custom cache size."""
        backend = Z3Backend(cache_size=200)
        assert backend._cache_size == 200

    def test_initialization_custom_cache_ttl(self):
        """Should accept custom cache TTL."""
        backend = Z3Backend(cache_ttl_seconds=7200.0)
        assert backend._cache_ttl == 7200.0

    def test_is_available(self):
        """Should check if Z3 is available."""
        backend = Z3Backend()
        # is_available is a property that checks for z3 package
        # Just test the property doesn't crash
        result = backend.is_available
        assert isinstance(result, bool)

    def test_can_verify_arithmetic(self):
        """Should be able to verify arithmetic claims."""
        backend = Z3Backend()
        if not backend.is_available:
            pytest.skip("Z3 not available")

        # can_verify checks if the claim is suitable
        result = backend.can_verify("x > 0 implies x >= 0")
        assert isinstance(result, bool)


class TestFormalVerificationManager:
    """Tests for FormalVerificationManager singleton."""

    def test_initialization(self):
        """Should initialize with default backends."""
        manager = FormalVerificationManager()
        # Manager initializes with Z3Backend and LeanBackend by default
        assert len(manager.backends) == 2

    def test_backends_types(self):
        """Should have correct backend types."""
        manager = FormalVerificationManager()
        # First backend is Z3, second is Lean
        assert manager.backends[0].language == FormalLanguage.Z3_SMT
        assert manager.backends[1].language == FormalLanguage.LEAN4

    def test_get_available_backends(self):
        """Should filter to available backends."""
        manager = FormalVerificationManager()
        # get_available_backends is synchronous and checks is_available property
        available = manager.get_available_backends()
        # Result should be a list
        assert isinstance(available, list)

    def test_get_backend_for_claim(self):
        """Should find appropriate backend for claims."""
        manager = FormalVerificationManager()
        # get_backend_for_claim returns the first suitable backend
        backend = manager.get_backend_for_claim("x > 0", "MATHEMATICAL")
        # Backend could be None if none available
        assert backend is None or hasattr(backend, "language")


class TestGetFormalVerificationManager:
    """Tests for get_formal_verification_manager function."""

    def test_returns_singleton(self):
        """Should return the same instance on repeated calls."""
        manager1 = get_formal_verification_manager()
        manager2 = get_formal_verification_manager()
        assert manager1 is manager2

    def test_returns_manager_instance(self):
        """Should return FormalVerificationManager instance."""
        manager = get_formal_verification_manager()
        assert isinstance(manager, FormalVerificationManager)


class TestBackendProtocol:
    """Tests for FormalVerificationBackend protocol compliance."""

    def test_lean_backend_implements_protocol(self):
        """LeanBackend should implement FormalVerificationBackend."""
        backend = LeanBackend()
        # language is a property
        assert hasattr(backend, "language")
        assert backend.language == FormalLanguage.LEAN4
        # is_available is a property (not a method)
        assert hasattr(backend, "is_available")
        # can_verify, translate, prove, verify_proof are methods
        assert hasattr(backend, "can_verify")
        assert callable(backend.can_verify)
        assert hasattr(backend, "translate")
        assert callable(backend.translate)
        assert hasattr(backend, "prove")
        assert callable(backend.prove)

    def test_z3_backend_implements_protocol(self):
        """Z3Backend should implement FormalVerificationBackend."""
        backend = Z3Backend()
        # language is a property
        assert hasattr(backend, "language")
        assert backend.language == FormalLanguage.Z3_SMT
        # is_available is a property (not a method)
        assert hasattr(backend, "is_available")
        # can_verify, translate, prove are methods
        assert hasattr(backend, "can_verify")
        assert callable(backend.can_verify)
        assert hasattr(backend, "translate")
        assert callable(backend.translate)
        assert hasattr(backend, "prove")
        assert callable(backend.prove)
