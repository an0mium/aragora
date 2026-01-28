"""
Tests for aragora.verification.formal module.

Tests the formal verification backends including:
- FormalProofStatus and FormalLanguage enums
- FormalProofResult dataclass
- LeanBackend (stub implementation)
- Z3Backend (SMT solver)
- FormalVerificationManager
"""

import hashlib
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import centralized skip markers for Z3
from tests.conftest import requires_z3, REQUIRES_Z3

from aragora.verification.formal import (
    FormalLanguage,
    FormalProofResult,
    FormalProofStatus,
    FormalVerificationBackend,
    FormalVerificationManager,
    LeanBackend,
    Z3Backend,
    get_formal_verification_manager,
)


# ==============================================================================
# FormalProofStatus Enum Tests
# ==============================================================================


class TestFormalProofStatus:
    """Tests for FormalProofStatus enum."""

    def test_not_attempted_value(self):
        """NOT_ATTEMPTED has correct value."""
        assert FormalProofStatus.NOT_ATTEMPTED.value == "not_attempted"

    def test_translation_failed_value(self):
        """TRANSLATION_FAILED has correct value."""
        assert FormalProofStatus.TRANSLATION_FAILED.value == "translation_failed"

    def test_proof_found_value(self):
        """PROOF_FOUND has correct value."""
        assert FormalProofStatus.PROOF_FOUND.value == "proof_found"

    def test_proof_failed_value(self):
        """PROOF_FAILED has correct value."""
        assert FormalProofStatus.PROOF_FAILED.value == "proof_failed"

    def test_timeout_value(self):
        """TIMEOUT has correct value."""
        assert FormalProofStatus.TIMEOUT.value == "timeout"

    def test_backend_unavailable_value(self):
        """BACKEND_UNAVAILABLE has correct value."""
        assert FormalProofStatus.BACKEND_UNAVAILABLE.value == "backend_unavailable"

    def test_not_supported_value(self):
        """NOT_SUPPORTED has correct value."""
        assert FormalProofStatus.NOT_SUPPORTED.value == "not_supported"


# ==============================================================================
# FormalLanguage Enum Tests
# ==============================================================================


class TestFormalLanguage:
    """Tests for FormalLanguage enum."""

    def test_lean4_value(self):
        """LEAN4 has correct value."""
        assert FormalLanguage.LEAN4.value == "lean4"

    def test_coq_value(self):
        """COQ has correct value."""
        assert FormalLanguage.COQ.value == "coq"

    def test_isabelle_value(self):
        """ISABELLE has correct value."""
        assert FormalLanguage.ISABELLE.value == "isabelle"

    def test_agda_value(self):
        """AGDA has correct value."""
        assert FormalLanguage.AGDA.value == "agda"

    def test_z3_smt_value(self):
        """Z3_SMT has correct value."""
        assert FormalLanguage.Z3_SMT.value == "z3_smt"


# ==============================================================================
# FormalProofResult Dataclass Tests
# ==============================================================================


class TestFormalProofResult:
    """Tests for FormalProofResult dataclass."""

    def test_minimal_creation(self):
        """Can create with just required fields."""
        result = FormalProofResult(
            status=FormalProofStatus.NOT_ATTEMPTED,
            language=FormalLanguage.Z3_SMT,
        )
        assert result.status == FormalProofStatus.NOT_ATTEMPTED
        assert result.language == FormalLanguage.Z3_SMT

    def test_default_values(self):
        """Default values are set correctly."""
        result = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.LEAN4,
        )
        assert result.formal_statement is None
        assert result.proof_text is None
        assert result.proof_hash is None
        assert result.translation_time_ms == 0.0
        assert result.proof_search_time_ms == 0.0
        assert result.error_message == ""
        assert result.prover_version == ""
        assert isinstance(result.timestamp, datetime)

    def test_full_creation(self):
        """Can create with all fields."""
        ts = datetime(2026, 1, 6, 12, 0, 0)
        result = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.Z3_SMT,
            formal_statement="(assert (= x 1))",
            proof_text="QED",
            proof_hash="abc123",
            translation_time_ms=100.5,
            proof_search_time_ms=250.3,
            error_message="",
            prover_version="z3-4.12.1",
            timestamp=ts,
        )
        assert result.formal_statement == "(assert (= x 1))"
        assert result.proof_text == "QED"
        assert result.proof_hash == "abc123"
        assert result.translation_time_ms == 100.5
        assert result.proof_search_time_ms == 250.3
        assert result.prover_version == "z3-4.12.1"
        assert result.timestamp == ts

    def test_is_verified_true_when_proof_found(self):
        """is_verified returns True when status is PROOF_FOUND."""
        result = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.Z3_SMT,
        )
        assert result.is_verified is True

    def test_is_verified_false_for_other_statuses(self):
        """is_verified returns False for non-PROOF_FOUND statuses."""
        statuses = [
            FormalProofStatus.NOT_ATTEMPTED,
            FormalProofStatus.TRANSLATION_FAILED,
            FormalProofStatus.PROOF_FAILED,
            FormalProofStatus.TIMEOUT,
            FormalProofStatus.BACKEND_UNAVAILABLE,
            FormalProofStatus.NOT_SUPPORTED,
        ]
        for status in statuses:
            result = FormalProofResult(status=status, language=FormalLanguage.Z3_SMT)
            assert result.is_verified is False, f"Expected False for {status}"

    def test_to_dict_contains_required_keys(self):
        """to_dict returns dict with all expected keys."""
        result = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.Z3_SMT,
        )
        d = result.to_dict()
        # Base keys always present
        expected_keys = {
            "status",
            "language",
            "formal_statement",
            "proof_hash",
            "is_verified",
            "translation_time_ms",
            "proof_search_time_ms",
            "error_message",
            "prover_version",
            "timestamp",
            # LLM-translated proof fields (always present)
            "translation_confidence",
            "is_high_confidence",
            "semantic_match_verified",
            # Note: original_claim and confidence_warning are conditional
            # (only present when non-empty)
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_serializes_enum_values(self):
        """to_dict converts enums to their string values."""
        result = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.LEAN4,
        )
        d = result.to_dict()
        assert d["status"] == "proof_found"
        assert d["language"] == "lean4"

    def test_to_dict_serializes_datetime(self):
        """to_dict converts timestamp to ISO format."""
        ts = datetime(2026, 1, 6, 12, 30, 45)
        result = FormalProofResult(
            status=FormalProofStatus.NOT_ATTEMPTED,
            language=FormalLanguage.Z3_SMT,
            timestamp=ts,
        )
        d = result.to_dict()
        assert d["timestamp"] == "2026-01-06T12:30:45"

    def test_to_dict_includes_is_verified(self):
        """to_dict includes computed is_verified property."""
        result = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.Z3_SMT,
        )
        d = result.to_dict()
        assert d["is_verified"] is True


# ==============================================================================
# LeanBackend Tests
# ==============================================================================


class TestLeanBackend:
    """Tests for LeanBackend class."""

    @pytest.fixture
    def backend(self):
        """Create a LeanBackend instance."""
        return LeanBackend()

    def test_language_is_lean4(self, backend):
        """language property returns LEAN4."""
        assert backend.language == FormalLanguage.LEAN4

    def test_is_available_checks_lean_command(self, backend):
        """is_available checks for lean and lake commands."""
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: (
                f"/usr/bin/{cmd}" if cmd in ["lean", "lake"] else None
            )
            assert backend.is_available is True

    def test_is_available_false_when_lean_missing(self, backend):
        """is_available returns False when lean command missing."""
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda cmd: None if cmd == "lean" else f"/usr/bin/{cmd}"
            assert backend.is_available is False

    def test_is_available_false_when_lean_missing_fresh_backend(self, backend):
        """is_available returns False when lean command missing (fresh backend)."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            # Create new backend to avoid cached version
            from aragora.verification.formal import LeanBackend

            new_backend = LeanBackend()
            new_backend._lean_version = None  # Reset cached version
            assert new_backend.is_available is False

    def test_can_verify_with_math_patterns(self, backend):
        """can_verify returns True for mathematical claims when backend available."""
        with patch("shutil.which", return_value="/usr/bin/lean"):
            # Create fresh backend so is_available uses mocked shutil.which
            from aragora.verification.formal import LeanBackend

            fresh_backend = LeanBackend()
            # Mathematical patterns should be verifiable
            assert fresh_backend.can_verify("for all n, n + 0 = n") is True
            assert fresh_backend.can_verify("prove that prime numbers are infinite") is True
            assert fresh_backend.can_verify("theorem about even numbers") is True
            # Non-math claims should not be verifiable
            assert fresh_backend.can_verify("the sky is blue") is False

    @pytest.mark.asyncio
    async def test_translate_returns_none_without_api_key(self, backend):
        """translate returns None when no API key is set."""
        with patch.dict("os.environ", {}, clear=True):
            result = await backend.translate("1 + 1 = 2")
            assert result is None

    @pytest.mark.asyncio
    async def test_prove_returns_unavailable_when_lean_not_installed(self, backend):
        """prove returns BACKEND_UNAVAILABLE status when Lean not installed."""
        with patch("shutil.which", return_value=None):
            # Create fresh backend so is_available uses mocked shutil.which
            from aragora.verification.formal import LeanBackend

            fresh_backend = LeanBackend()
            result = await fresh_backend.prove("theorem test : True := trivial")
            assert result.status == FormalProofStatus.BACKEND_UNAVAILABLE
            assert result.language == FormalLanguage.LEAN4
            # Should mention Lean not installed
            assert "Lean" in result.error_message or "not installed" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_verify_proof_returns_false(self, backend):
        """verify_proof always returns False (stub)."""
        result = await backend.verify_proof("theorem test : True := trivial", "trivial")
        assert result is False


# ==============================================================================
# Z3Backend Tests
# ==============================================================================


class TestZ3Backend:
    """Tests for Z3Backend class."""

    @pytest.fixture
    def backend(self):
        """Create a Z3Backend instance."""
        return Z3Backend()

    @pytest.fixture
    def backend_with_translator(self):
        """Create Z3Backend with mock LLM translator."""
        translator = AsyncMock(return_value="(declare-const x Int)\n(assert (= x 1))\n(check-sat)")
        return Z3Backend(llm_translator=translator)

    def test_language_is_z3_smt(self, backend):
        """language property returns Z3_SMT."""
        assert backend.language == FormalLanguage.Z3_SMT

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    def test_is_available_when_z3_installed(self, backend):
        """is_available returns True when z3 is installed."""
        assert backend.is_available is True

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    def test_z3_version_returns_string(self, backend):
        """z3_version returns version string."""
        version = backend.z3_version
        assert isinstance(version, str)
        assert "z3" in version.lower() or version == "unknown"

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    def test_can_verify_smtlib2_format(self, backend):
        """can_verify returns True for SMT-LIB2 format claims."""
        smtlib = "(declare-const x Int)\n(assert (= x 1))"
        assert backend.can_verify(smtlib) is True

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    def test_can_verify_arithmetic_claim_type(self, backend):
        """can_verify returns True for arithmetic claim type."""
        assert backend.can_verify("x > y", claim_type="arithmetic") is True

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    def test_can_verify_logical_claim_type(self, backend):
        """can_verify returns True for logical claim type."""
        assert backend.can_verify("p implies q", claim_type="logical") is True

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    def test_can_verify_quantifiable_patterns(self, backend):
        """can_verify returns True for claims with quantifiable patterns."""
        claims = [
            "for all x, x > 0",
            "there exists y such that y = 0",
            "if x > y then x >= y",
            "x is greater than y",
        ]
        for claim in claims:
            assert backend.can_verify(claim) is True, f"Should verify: {claim}"

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    def test_can_verify_math_patterns(self, backend):
        """can_verify returns True for claims with math notation."""
        claims = ["x > y", "a + b = c", "x ≤ y", "p → q", "a ∧ b"]
        for claim in claims:
            assert backend.can_verify(claim) is True, f"Should verify: {claim}"

    def test_can_verify_returns_false_when_unavailable(self):
        """can_verify returns False when Z3 not available."""
        backend = Z3Backend()
        with patch.object(Z3Backend, "is_available", False):
            assert backend.can_verify("x > y") is False

    def test_is_smtlib2_detects_valid_format(self, backend):
        """_is_smtlib2 detects valid SMT-LIB2 syntax."""
        valid = [
            "(declare-const x Int)",
            "(assert (= x 1))",
            "(check-sat)",
            "(define-fun f () Int 1)",
        ]
        for stmt in valid:
            assert backend._is_smtlib2(stmt) is True, f"Should detect: {stmt}"

    def test_is_smtlib2_rejects_invalid_format(self, backend):
        """_is_smtlib2 rejects non-SMT-LIB2 text."""
        invalid = [
            "x > y",
            "for all x, x >= 0",
            "theorem test : True := trivial",
        ]
        for stmt in invalid:
            assert backend._is_smtlib2(stmt) is False, f"Should reject: {stmt}"

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    def test_validate_smtlib2_accepts_valid(self, backend):
        """_validate_smtlib2 accepts valid SMT-LIB2."""
        valid = "(declare-const x Int)\n(assert (= x 1))\n(check-sat)"
        assert backend._validate_smtlib2(valid) is True

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    def test_validate_smtlib2_rejects_invalid(self, backend):
        """_validate_smtlib2 rejects malformed SMT-LIB2."""
        invalid = "(declare-const x Int\n(missing paren)"
        assert backend._validate_smtlib2(invalid) is False

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    @pytest.mark.asyncio
    async def test_translate_passthrough_smtlib2(self, backend):
        """translate returns valid SMT-LIB2 unchanged."""
        smtlib = "(declare-const x Int)\n(assert (= x 1))\n(check-sat)"
        result = await backend.translate(smtlib)
        assert result == smtlib

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    @pytest.mark.asyncio
    async def test_translate_returns_none_for_complex_claim(self, backend):
        """translate returns None for claims it cannot translate."""
        # Complex natural language without LLM
        result = await backend.translate("The eigenvalues of a symmetric matrix are real")
        assert result is None

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    @pytest.mark.asyncio
    async def test_prove_valid_claim(self, backend):
        """prove returns PROOF_FOUND for valid claims."""
        # x > 0 and y > 0 implies x + y > 0
        smtlib = """
(declare-const x Int)
(declare-const y Int)
(assert (not (=> (and (> x 0) (> y 0)) (> (+ x y) 0))))
(check-sat)
"""
        result = await backend.prove(smtlib)
        assert result.status == FormalProofStatus.PROOF_FOUND
        assert result.is_verified is True
        assert result.proof_text is not None
        assert result.proof_hash is not None

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    @pytest.mark.asyncio
    async def test_prove_invalid_claim(self, backend):
        """prove returns PROOF_FAILED for invalid claims with counterexample."""
        # x > 0 implies x > 10 (false for x=5)
        smtlib = """
(declare-const x Int)
(assert (not (=> (> x 0) (> x 10))))
(check-sat)
"""
        result = await backend.prove(smtlib)
        assert result.status == FormalProofStatus.PROOF_FAILED
        assert "COUNTEREXAMPLE" in result.proof_text

    @pytest.mark.asyncio
    async def test_prove_returns_unavailable_when_z3_missing(self):
        """prove returns BACKEND_UNAVAILABLE when Z3 not installed."""
        backend = Z3Backend()
        with patch.object(Z3Backend, "is_available", False):
            result = await backend.prove("(check-sat)")
            assert result.status == FormalProofStatus.BACKEND_UNAVAILABLE

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    @pytest.mark.asyncio
    async def test_prove_handles_parse_error(self, backend):
        """prove returns TRANSLATION_FAILED for malformed input."""
        result = await backend.prove("(invalid smtlib (missing parens")
        assert result.status == FormalProofStatus.TRANSLATION_FAILED
        assert "parse" in result.error_message.lower()

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    @pytest.mark.asyncio
    async def test_prove_records_timing(self, backend):
        """prove records proof search time."""
        smtlib = "(declare-const x Int)\n(assert (= x 1))\n(check-sat)"
        result = await backend.prove(smtlib)
        assert result.proof_search_time_ms >= 0

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    @pytest.mark.asyncio
    async def test_prove_records_version(self, backend):
        """prove records prover version."""
        smtlib = "(declare-const x Int)\n(assert (= x 1))\n(check-sat)"
        result = await backend.prove(smtlib)
        assert "z3" in result.prover_version.lower()

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    @pytest.mark.asyncio
    async def test_verify_proof_returns_true_for_valid(self, backend):
        """verify_proof returns True for valid proofs."""
        smtlib = """
(declare-const x Int)
(declare-const y Int)
(assert (not (=> (and (> x 0) (> y 0)) (> (+ x y) 0))))
(check-sat)
"""
        result = await backend.verify_proof(smtlib, "QED")
        assert result is True

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    @pytest.mark.asyncio
    async def test_verify_proof_returns_false_for_invalid(self, backend):
        """verify_proof returns False for invalid claims."""
        # False claim
        smtlib = """
(declare-const x Int)
(assert (not (=> (> x 0) (> x 10))))
(check-sat)
"""
        result = await backend.verify_proof(smtlib, "QED")
        assert result is False


# ==============================================================================
# FormalVerificationManager Tests
# ==============================================================================


class TestFormalVerificationManager:
    """Tests for FormalVerificationManager class."""

    @pytest.fixture
    def manager(self):
        """Create a FormalVerificationManager instance."""
        return FormalVerificationManager()

    def test_has_z3_backend(self, manager):
        """Manager includes Z3 backend."""
        languages = [b.language for b in manager.backends]
        assert FormalLanguage.Z3_SMT in languages

    def test_has_lean_backend(self, manager):
        """Manager includes Lean backend."""
        languages = [b.language for b in manager.backends]
        assert FormalLanguage.LEAN4 in languages

    def test_z3_is_first_backend(self, manager):
        """Z3 is tried before Lean (simpler, faster)."""
        assert manager.backends[0].language == FormalLanguage.Z3_SMT

    def test_get_available_backends_filters_unavailable(self, manager):
        """get_available_backends only returns available backends."""
        available = manager.get_available_backends()
        for backend in available:
            assert backend.is_available is True

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    def test_get_backend_for_claim_arithmetic(self, manager):
        """get_backend_for_claim returns Z3 for arithmetic claims."""
        backend = manager.get_backend_for_claim("x > y", claim_type="arithmetic")
        assert backend is not None
        assert backend.language == FormalLanguage.Z3_SMT

    def test_get_backend_for_claim_returns_none_for_unsupported(self, manager):
        """get_backend_for_claim returns None when no backend can verify."""
        # Patch both backends to return False for can_verify
        with patch.object(Z3Backend, "can_verify", return_value=False):
            with patch.object(LeanBackend, "can_verify", return_value=False):
                backend = manager.get_backend_for_claim("unsupported claim")
                assert backend is None

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    @pytest.mark.asyncio
    async def test_attempt_formal_verification_success(self, manager):
        """attempt_formal_verification succeeds for valid Z3 claims."""
        # Already in SMT-LIB2 format
        smtlib = """
(declare-const x Int)
(declare-const y Int)
(assert (not (=> (and (> x 0) (> y 0)) (> (+ x y) 0))))
(check-sat)
"""
        result = await manager.attempt_formal_verification(smtlib)
        assert result.status == FormalProofStatus.PROOF_FOUND

    @pytest.mark.asyncio
    async def test_attempt_formal_verification_not_supported(self, manager):
        """attempt_formal_verification returns NOT_SUPPORTED when no backend available."""
        with patch.object(manager, "get_backend_for_claim", return_value=None):
            result = await manager.attempt_formal_verification("unsupported claim")
            assert result.status == FormalProofStatus.NOT_SUPPORTED

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    @pytest.mark.asyncio
    async def test_attempt_formal_verification_translation_failed(self, manager):
        """attempt_formal_verification returns TRANSLATION_FAILED when translation fails."""
        # Complex claim that can't be translated without LLM
        result = await manager.attempt_formal_verification(
            "The Riemann hypothesis is true",
            claim_type="logical",
        )
        # Should fail translation since no LLM available
        assert result.status in [
            FormalProofStatus.TRANSLATION_FAILED,
            FormalProofStatus.NOT_SUPPORTED,
        ]

    def test_status_report_structure(self, manager):
        """status_report returns correctly structured dict."""
        report = manager.status_report()
        assert "backends" in report
        assert "any_available" in report
        assert isinstance(report["backends"], list)
        assert isinstance(report["any_available"], bool)

    def test_status_report_backend_entries(self, manager):
        """status_report includes language and available for each backend."""
        report = manager.status_report()
        for entry in report["backends"]:
            assert "language" in entry
            assert "available" in entry
            assert isinstance(entry["language"], str)
            assert isinstance(entry["available"], bool)


# ==============================================================================
# Singleton Tests
# ==============================================================================


class TestGetFormalVerificationManager:
    """Tests for get_formal_verification_manager singleton."""

    def test_returns_manager_instance(self):
        """Returns a FormalVerificationManager instance."""
        manager = get_formal_verification_manager()
        assert isinstance(manager, FormalVerificationManager)

    def test_returns_same_instance(self):
        """Returns the same instance on repeated calls."""
        manager1 = get_formal_verification_manager()
        manager2 = get_formal_verification_manager()
        assert manager1 is manager2

    def test_manager_has_backends(self):
        """Returned manager has backends configured."""
        manager = get_formal_verification_manager()
        assert len(manager.backends) >= 2


# ==============================================================================
# Protocol Compliance Tests
# ==============================================================================


class TestFormalVerificationBackendProtocol:
    """Tests that backends comply with the FormalVerificationBackend protocol."""

    def test_lean_backend_is_protocol_compliant(self):
        """LeanBackend satisfies FormalVerificationBackend protocol."""
        backend = LeanBackend()
        assert isinstance(backend, FormalVerificationBackend)

    def test_z3_backend_is_protocol_compliant(self):
        """Z3Backend satisfies FormalVerificationBackend protocol."""
        backend = Z3Backend()
        assert isinstance(backend, FormalVerificationBackend)


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestFormalVerificationIntegration:
    """Integration tests for formal verification workflow."""

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    @pytest.mark.asyncio
    async def test_full_z3_verification_workflow(self):
        """Test complete Z3 verification from claim to proof."""
        manager = FormalVerificationManager()

        # A simple transitivity claim in SMT-LIB2
        smtlib = """
(declare-const a Int)
(declare-const b Int)
(declare-const c Int)
(assert (not (=> (and (> a b) (> b c)) (> a c))))
(check-sat)
"""
        result = await manager.attempt_formal_verification(smtlib)

        assert result.status == FormalProofStatus.PROOF_FOUND
        assert result.is_verified is True
        assert result.language == FormalLanguage.Z3_SMT
        assert result.proof_hash is not None

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    @pytest.mark.asyncio
    async def test_proof_result_serialization(self):
        """Test that proof results can be serialized to dict."""
        backend = Z3Backend()
        smtlib = """
(declare-const x Int)
(assert (not (=> (> x 0) (>= x 0))))
(check-sat)
"""
        result = await backend.prove(smtlib)
        d = result.to_dict()

        # Should be JSON-serializable
        import json

        serialized = json.dumps(d)
        assert isinstance(serialized, str)

        # Should round-trip
        deserialized = json.loads(serialized)
        assert deserialized["status"] == result.status.value
        assert deserialized["language"] == result.language.value

    @pytest.mark.skipif(requires_z3, reason=REQUIRES_Z3)
    @pytest.mark.asyncio
    async def test_counterexample_detection(self):
        """Test that counterexamples are properly detected and reported."""
        backend = Z3Backend()

        # False claim: all positive numbers are greater than 100
        smtlib = """
(declare-const x Int)
(assert (not (=> (> x 0) (> x 100))))
(check-sat)
"""
        result = await backend.prove(smtlib)

        assert result.status == FormalProofStatus.PROOF_FAILED
        assert "COUNTEREXAMPLE" in result.proof_text
        # The counterexample should contain a concrete value

    @pytest.mark.asyncio
    async def test_manager_prefers_available_backend(self):
        """Manager uses available backends over unavailable ones."""
        manager = FormalVerificationManager()
        claim = "(declare-const x Int)\n(assert (= x 1))\n(check-sat)"

        backend = manager.get_backend_for_claim(claim)

        if backend is not None:
            assert backend.is_available is True
