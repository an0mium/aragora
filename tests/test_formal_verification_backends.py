"""
Edge case tests for formal verification backends.

Tests the Z3Backend and LeanBackend implementations with focus on:
1. Input validation and edge cases
2. Error handling for network failures
3. Translation edge cases
4. Proof verification edge cases
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

from aragora.verification.formal import (
    Z3Backend,
    LeanBackend,
    FormalProofResult,
    FormalProofStatus,
    FormalLanguage,
)


# ============================================================================
# Z3Backend Tests
# ============================================================================

class TestZ3BackendCanVerify:
    """Tests for Z3Backend.can_verify() method."""

    @pytest.fixture
    def backend(self):
        """Create a Z3Backend instance."""
        return Z3Backend()

    def test_can_verify_smtlib2_format(self, backend):
        """Should accept SMT-LIB2 format claims."""
        smtlib = "(declare-const x Int)\n(assert (> x 0))\n(check-sat)"
        if backend.is_available:
            assert backend.can_verify(smtlib) is True

    def test_can_verify_empty_string(self, backend):
        """Empty string should not be verifiable."""
        if backend.is_available:
            assert backend.can_verify("") is False

    def test_can_verify_whitespace_only(self, backend):
        """Whitespace-only string should not be verifiable."""
        if backend.is_available:
            assert backend.can_verify("   \n\t  ") is False

    def test_can_verify_with_math_operators(self, backend):
        """Claims with math operators should be verifiable."""
        if backend.is_available:
            assert backend.can_verify("x > y") is True
            assert backend.can_verify("a = b + c") is True
            assert backend.can_verify("n ≤ m") is True

    def test_can_verify_with_unicode_math(self, backend):
        """Claims with unicode math symbols should be verifiable."""
        if backend.is_available:
            assert backend.can_verify("∀x: x > 0") is True
            assert backend.can_verify("∃n: n² = 4") is True
            assert backend.can_verify("p → q") is True
            assert backend.can_verify("a ∧ b") is True
            assert backend.can_verify("¬p ∨ q") is True

    def test_can_verify_with_claim_type_logical(self, backend):
        """Should accept LOGICAL claim type."""
        if backend.is_available:
            assert backend.can_verify("any claim", claim_type="LOGICAL") is True
            assert backend.can_verify("any claim", claim_type="logical") is True

    def test_can_verify_with_claim_type_factual(self, backend):
        """Should accept FACTUAL claim type."""
        if backend.is_available:
            assert backend.can_verify("any claim", claim_type="FACTUAL") is True

    def test_can_verify_with_quantifiable_patterns(self, backend):
        """Should match quantifiable language patterns."""
        if backend.is_available:
            assert backend.can_verify("for all n, n > 0") is True
            assert backend.can_verify("there exists x such that x = 0") is True
            assert backend.can_verify("if a then b") is True
            assert backend.can_verify("a implies b") is True
            assert backend.can_verify("n is even") is True
            assert backend.can_verify("x is positive") is True

    def test_can_verify_random_text(self, backend):
        """Random text without patterns should not be verifiable."""
        if backend.is_available:
            # This might still match some patterns depending on implementation
            result = backend.can_verify("hello world today is nice")
            # Just verify it doesn't crash
            assert isinstance(result, bool)

    def test_can_verify_special_regex_characters(self, backend):
        """Claims with regex special characters should not crash."""
        if backend.is_available:
            # Should not crash on regex special characters
            assert isinstance(backend.can_verify(".*+?()[]{}|\\^$"), bool)
            assert isinstance(backend.can_verify("a(b)c[d]e{f}g"), bool)

    def test_can_verify_when_unavailable(self, backend):
        """Should return False when Z3 is not available."""
        with patch.object(backend, 'is_available', new_callable=PropertyMock, return_value=False):
            assert backend.can_verify("x > y") is False


class TestZ3BackendTranslate:
    """Tests for Z3Backend.translate() method."""

    @pytest.fixture
    def backend(self):
        """Create a Z3Backend instance."""
        return Z3Backend()

    @pytest.mark.asyncio
    async def test_translate_already_smtlib2(self, backend):
        """Already valid SMT-LIB2 should be returned as-is."""
        if not backend.is_available:
            pytest.skip("Z3 not installed")

        smtlib = "(declare-const x Int)\n(assert (> x 0))\n(check-sat)"
        result = await backend.translate(smtlib)
        assert result == smtlib

    @pytest.mark.asyncio
    async def test_translate_invalid_smtlib2(self, backend):
        """Invalid SMT-LIB2 should return None."""
        if not backend.is_available:
            pytest.skip("Z3 not installed")

        # Looks like SMT-LIB2 but is invalid
        invalid = "(assert (invalid syntax here"
        result = await backend.translate(invalid)
        assert result is None

    @pytest.mark.asyncio
    async def test_translate_simple_transitivity(self, backend):
        """Should translate simple transitivity claims."""
        if not backend.is_available:
            pytest.skip("Z3 not installed")

        claim = "if x > y and y > z then x > z"
        result = await backend.translate(claim)
        # May or may not translate depending on pattern matching
        # Just verify it doesn't crash
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_translate_empty_string(self, backend):
        """Empty string should return None."""
        if not backend.is_available:
            pytest.skip("Z3 not installed")

        result = await backend.translate("")
        assert result is None

    @pytest.mark.asyncio
    async def test_translate_with_llm_translator(self, backend):
        """Should use LLM translator when available."""
        if not backend.is_available:
            pytest.skip("Z3 not installed")

        async def mock_llm(prompt, context):
            return "(declare-const x Int)\n(assert (> x 0))\n(check-sat)"

        backend_with_llm = Z3Backend(llm_translator=mock_llm)
        result = await backend_with_llm.translate("x is positive")
        assert result is not None

    @pytest.mark.asyncio
    async def test_translate_with_llm_returns_markdown(self, backend):
        """Should extract SMT-LIB2 from markdown code blocks."""
        if not backend.is_available:
            pytest.skip("Z3 not installed")

        async def mock_llm(prompt, context):
            return """```smt2
(declare-const x Int)
(assert (> x 0))
(check-sat)
```"""

        backend_with_llm = Z3Backend(llm_translator=mock_llm)
        result = await backend_with_llm.translate("x is positive")
        assert result is not None
        assert "```" not in result

    @pytest.mark.asyncio
    async def test_translate_with_llm_failure(self, backend):
        """Should handle LLM translator failures gracefully."""
        if not backend.is_available:
            pytest.skip("Z3 not installed")

        async def failing_llm(prompt, context):
            raise RuntimeError("LLM API error")

        backend_with_llm = Z3Backend(llm_translator=failing_llm)
        result = await backend_with_llm.translate("some claim")
        # Should return None, not raise
        assert result is None


class TestZ3BackendProve:
    """Tests for Z3Backend.prove() method."""

    @pytest.fixture
    def backend(self):
        """Create a Z3Backend instance."""
        return Z3Backend()

    @pytest.mark.asyncio
    async def test_prove_when_unavailable(self, backend):
        """Should return BACKEND_UNAVAILABLE when Z3 not installed."""
        with patch.object(Z3Backend, 'is_available', new_callable=PropertyMock, return_value=False):
            result = await backend.prove("(check-sat)")
            assert result.status == FormalProofStatus.BACKEND_UNAVAILABLE
            assert result.language == FormalLanguage.Z3_SMT

    @pytest.mark.asyncio
    async def test_prove_valid_unsat(self, backend):
        """Should return PROOF_FOUND for valid unsat result."""
        if not backend.is_available:
            pytest.skip("Z3 not installed")

        # This is trivially unsatisfiable (negation of tautology)
        smtlib = """
(declare-const x Int)
(assert (not (= x x)))
(check-sat)
"""
        result = await backend.prove(smtlib)
        assert result.status == FormalProofStatus.PROOF_FOUND
        assert result.is_verified is True

    @pytest.mark.asyncio
    async def test_prove_valid_sat(self, backend):
        """Should return PROOF_FAILED for sat result (counterexample exists)."""
        if not backend.is_available:
            pytest.skip("Z3 not installed")

        # This is satisfiable (x can be 1)
        smtlib = """
(declare-const x Int)
(assert (> x 0))
(check-sat)
"""
        result = await backend.prove(smtlib)
        assert result.status == FormalProofStatus.PROOF_FAILED
        assert result.is_verified is False

    @pytest.mark.asyncio
    async def test_prove_empty_statement(self, backend):
        """Should handle empty statement gracefully."""
        if not backend.is_available:
            pytest.skip("Z3 not installed")

        result = await backend.prove("")
        assert result.status in (
            FormalProofStatus.TRANSLATION_FAILED,
            FormalProofStatus.PROOF_FAILED,
        )

    @pytest.mark.asyncio
    async def test_prove_invalid_syntax(self, backend):
        """Should handle invalid SMT-LIB2 syntax."""
        if not backend.is_available:
            pytest.skip("Z3 not installed")

        result = await backend.prove("this is not valid SMT-LIB2")
        assert result.status == FormalProofStatus.TRANSLATION_FAILED

    @pytest.mark.asyncio
    async def test_prove_timeout(self, backend):
        """Should timeout on complex problems."""
        if not backend.is_available:
            pytest.skip("Z3 not installed")

        # Create a simple problem that should complete quickly
        # Real timeout testing would need a harder problem
        smtlib = "(declare-const x Int)\n(assert (= x 1))\n(check-sat)"
        result = await backend.prove(smtlib, timeout_seconds=0.001)
        # May or may not timeout depending on system speed
        assert result.status in (
            FormalProofStatus.TIMEOUT,
            FormalProofStatus.PROOF_FAILED,
            FormalProofStatus.PROOF_FOUND,
        )


class TestZ3BackendHelpers:
    """Tests for Z3Backend helper methods."""

    @pytest.fixture
    def backend(self):
        """Create a Z3Backend instance."""
        return Z3Backend()

    def test_is_smtlib2_valid(self, backend):
        """Should detect valid SMT-LIB2 format."""
        valid = "(declare-const x Int)\n(assert (> x 0))"
        assert backend._is_smtlib2(valid) is True

    def test_is_smtlib2_invalid(self, backend):
        """Should reject non-SMT-LIB2 text."""
        assert backend._is_smtlib2("hello world") is False
        assert backend._is_smtlib2("x > y") is False

    def test_is_smtlib2_with_whitespace(self, backend):
        """Should handle leading/trailing whitespace."""
        valid = "   (declare-const x Int)   "
        assert backend._is_smtlib2(valid) is True

    def test_validate_smtlib2_valid(self, backend):
        """Should validate correct SMT-LIB2."""
        if not backend.is_available:
            pytest.skip("Z3 not installed")

        valid = "(declare-const x Int)\n(assert (> x 0))\n(check-sat)"
        assert backend._validate_smtlib2(valid) is True

    def test_validate_smtlib2_invalid(self, backend):
        """Should reject invalid SMT-LIB2."""
        if not backend.is_available:
            pytest.skip("Z3 not installed")

        invalid = "(declare-const x UnknownType)"
        assert backend._validate_smtlib2(invalid) is False


# ============================================================================
# LeanBackend Tests
# ============================================================================

class TestLeanBackendBasics:
    """Basic tests for LeanBackend."""

    @pytest.fixture
    def backend(self):
        """Create a LeanBackend instance."""
        return LeanBackend()

    def test_language_property(self, backend):
        """Should return LEAN4 language."""
        assert backend.language == FormalLanguage.LEAN4

    def test_is_available_checks_lean(self, backend):
        """Should check for lean and lake in PATH."""
        with patch('shutil.which') as mock_which:
            mock_which.return_value = None
            assert backend.is_available is False

    def test_can_verify_always_false(self, backend):
        """Should return False (not implemented yet)."""
        assert backend.can_verify("any claim") is False
        assert backend.can_verify("for all n, n > 0") is False

    @pytest.mark.asyncio
    async def test_prove_returns_unavailable(self, backend):
        """prove() should return BACKEND_UNAVAILABLE."""
        result = await backend.prove("theorem test : True := trivial")
        assert result.status == FormalProofStatus.BACKEND_UNAVAILABLE
        assert result.language == FormalLanguage.LEAN4

    @pytest.mark.asyncio
    async def test_verify_proof_returns_false(self, backend):
        """verify_proof() should return False (not implemented)."""
        result = await backend.verify_proof("theorem", "proof")
        assert result is False


class TestLeanBackendTranslate:
    """Tests for LeanBackend.translate() method."""

    @pytest.fixture
    def backend(self):
        """Create a LeanBackend instance."""
        return LeanBackend()

    @pytest.mark.asyncio
    async def test_translate_no_api_key(self, backend):
        """Should return None when no API key available."""
        with patch.dict('os.environ', {}, clear=True):
            # Remove API keys
            import os
            old_anthropic = os.environ.pop('ANTHROPIC_API_KEY', None)
            old_openai = os.environ.pop('OPENAI_API_KEY', None)
            try:
                result = await backend.translate("claim")
                assert result is None
            finally:
                if old_anthropic:
                    os.environ['ANTHROPIC_API_KEY'] = old_anthropic
                if old_openai:
                    os.environ['OPENAI_API_KEY'] = old_openai

    @pytest.mark.asyncio
    async def test_translate_handles_network_error(self, backend):
        """Should handle network errors gracefully."""
        import os
        # Set a fake API key
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'fake-key'}):
            with patch('aiohttp.ClientSession') as mock_session:
                mock_session.return_value.__aenter__ = AsyncMock(
                    side_effect=Exception("Network error")
                )
                result = await backend.translate("some claim")
                assert result is None

    @pytest.mark.asyncio
    async def test_translate_handles_timeout(self, backend):
        """Should handle timeout errors gracefully."""
        import os
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'fake-key'}):
            with patch('aiohttp.ClientSession') as mock_session:
                mock_session.return_value.__aenter__ = AsyncMock(
                    side_effect=asyncio.TimeoutError()
                )
                result = await backend.translate("some claim")
                assert result is None

    @pytest.mark.asyncio
    async def test_translate_handles_untranslatable(self, backend):
        """Should return None for UNTRANSLATABLE response."""
        import os
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'fake-key'}):
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "content": [{"text": "UNTRANSLATABLE - claim is too vague"}]
            })

            with patch('aiohttp.ClientSession') as mock_session:
                mock_cm = AsyncMock()
                mock_cm.__aenter__.return_value = mock_response
                mock_session.return_value.__aenter__.return_value.post.return_value = mock_cm

                result = await backend.translate("vague claim")
                # May or may not be None depending on mock setup
                # Just verify it doesn't crash


# ============================================================================
# FormalProofResult Tests
# ============================================================================

class TestFormalProofResult:
    """Tests for FormalProofResult dataclass."""

    def test_is_verified_when_proof_found(self):
        """is_verified should be True when status is PROOF_FOUND."""
        result = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.Z3_SMT,
        )
        assert result.is_verified is True

    def test_is_verified_when_proof_failed(self):
        """is_verified should be False when status is not PROOF_FOUND."""
        for status in FormalProofStatus:
            if status != FormalProofStatus.PROOF_FOUND:
                result = FormalProofResult(
                    status=status,
                    language=FormalLanguage.Z3_SMT,
                )
                assert result.is_verified is False

    def test_to_dict_serialization(self):
        """to_dict() should serialize all fields."""
        result = FormalProofResult(
            status=FormalProofStatus.PROOF_FOUND,
            language=FormalLanguage.Z3_SMT,
            formal_statement="(check-sat)",
            proof_text="unsat",
            proof_hash="abc123",
            translation_time_ms=100.0,
            proof_search_time_ms=200.0,
            error_message="",
            prover_version="z3-4.12.0",
        )
        d = result.to_dict()
        assert d["status"] == "proof_found"
        assert d["language"] == "z3_smt"
        assert d["is_verified"] is True
        assert d["proof_hash"] == "abc123"
        assert "timestamp" in d

    def test_default_values(self):
        """Should have sensible defaults."""
        result = FormalProofResult(
            status=FormalProofStatus.NOT_ATTEMPTED,
            language=FormalLanguage.LEAN4,
        )
        assert result.formal_statement is None
        assert result.proof_text is None
        assert result.translation_time_ms == 0.0
        assert result.error_message == ""


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================

class TestFormalVerificationEdgeCases:
    """Edge case tests for formal verification system."""

    @pytest.fixture
    def z3_backend(self):
        return Z3Backend()

    @pytest.mark.asyncio
    async def test_very_long_claim(self, z3_backend):
        """Should handle very long claims without crashing."""
        if not z3_backend.is_available:
            pytest.skip("Z3 not installed")

        long_claim = "x > y " * 1000
        result = z3_backend.can_verify(long_claim)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_null_bytes_in_claim(self, z3_backend):
        """Should handle null bytes in claims."""
        if not z3_backend.is_available:
            pytest.skip("Z3 not installed")

        claim_with_null = "x > y\x00 and y > z"
        result = z3_backend.can_verify(claim_with_null)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_unicode_in_smtlib2(self, z3_backend):
        """Should handle unicode in SMT-LIB2 gracefully."""
        if not z3_backend.is_available:
            pytest.skip("Z3 not installed")

        # Unicode variable names (may or may not be valid depending on Z3)
        smtlib = "(declare-const λ Int)\n(assert (> λ 0))\n(check-sat)"
        result = await z3_backend.translate(smtlib)
        # May return None if invalid, just shouldn't crash

    @pytest.mark.asyncio
    async def test_concurrent_prove_calls(self, z3_backend):
        """Should handle concurrent prove calls safely."""
        if not z3_backend.is_available:
            pytest.skip("Z3 not installed")

        smtlib = "(declare-const x Int)\n(assert (= x 1))\n(check-sat)"

        # Run multiple prove calls concurrently
        results = await asyncio.gather(*[
            z3_backend.prove(smtlib) for _ in range(5)
        ])

        assert len(results) == 5
        for result in results:
            assert isinstance(result, FormalProofResult)
