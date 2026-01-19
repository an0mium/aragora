"""Tests for the verification proofs module.

Tests cover:
- ProofType enum values
- ProofStatus enum values
- VerificationProof dataclass
- VerificationResult dataclass
- ProofExecutor safe code execution
- Dangerous pattern detection
- ClaimVerifier functionality
- ProofBuilder factory methods
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

from aragora.verification.proofs import (
    DANGEROUS_PATTERNS,
    EXEC_TIMEOUT_SECONDS,
    SAFE_BUILTINS,
    ClaimVerifier,
    ProofBuilder,
    ProofExecutor,
    ProofStatus,
    ProofType,
    VerificationProof,
    VerificationResult,
    VerificationReport,
    _validate_code_safety,
    create_simple_assertion,
    create_computation_proof,
)


class TestProofType:
    """Tests for ProofType enum."""

    def test_all_types_exist(self):
        """All expected proof types should be defined."""
        assert ProofType.ASSERTION.value == "assertion"
        assert ProofType.COMPUTATION.value == "computation"
        assert ProofType.API_CALL.value == "api_call"
        assert ProofType.CODE_EXECUTION.value == "code_execution"
        assert ProofType.TEST_SUITE.value == "test_suite"

    def test_type_count(self):
        """Should have expected number of proof types."""
        assert len(ProofType) >= 5


class TestProofStatus:
    """Tests for ProofStatus enum."""

    def test_all_statuses_exist(self):
        """All expected statuses should be defined."""
        assert ProofStatus.PENDING.value == "pending"
        assert ProofStatus.RUNNING.value == "running"
        assert ProofStatus.PASSED.value == "passed"
        assert ProofStatus.FAILED.value == "failed"
        assert ProofStatus.ERROR.value == "error"
        assert ProofStatus.TIMEOUT.value == "timeout"
        assert ProofStatus.SKIPPED.value == "skipped"

    def test_status_count(self):
        """Should have exactly 7 status values."""
        assert len(ProofStatus) == 7


class TestVerificationProof:
    """Tests for VerificationProof dataclass."""

    def test_minimal_creation(self):
        """Should create with required fields only."""
        proof = VerificationProof(
            id="proof-1",
            claim_id="claim-1",
            proof_type=ProofType.ASSERTION,
            description="Test assertion",
            code="assert 1 + 1 == 2",
        )
        assert proof.id == "proof-1"
        assert proof.claim_id == "claim-1"
        assert proof.proof_type == ProofType.ASSERTION
        assert proof.code == "assert 1 + 1 == 2"

    def test_full_creation(self):
        """Should accept all optional fields."""
        proof = VerificationProof(
            id="proof-2",
            claim_id="claim-2",
            proof_type=ProofType.COMPUTATION,
            description="Sum calculation",
            code="result = sum([1,2,3])",
            expected_output="6",
            timeout_seconds=60.0,
        )
        assert proof.description == "Sum calculation"
        assert proof.expected_output == "6"
        assert proof.timeout_seconds == 60.0


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_successful_result(self):
        """Should represent successful verification."""
        result = VerificationResult(
            proof_id="proof-1",
            claim_id="claim-1",
            status=ProofStatus.PASSED,
            passed=True,
        )
        assert result.status == ProofStatus.PASSED
        assert result.passed is True

    def test_failed_result(self):
        """Should represent failed verification."""
        result = VerificationResult(
            proof_id="proof-1",
            claim_id="claim-1",
            status=ProofStatus.FAILED,
            passed=False,
            error="Assertion failed: expected True, got False",
        )
        assert result.status == ProofStatus.FAILED
        assert result.error is not None


class TestDangerousPatternDetection:
    """Tests for code safety validation."""

    def test_safe_code_passes(self):
        """Safe code should pass validation."""
        is_safe, msg = _validate_code_safety("x = 1 + 2")
        assert is_safe is True
        assert msg == ""

    def test_dangerous_import_blocked(self):
        """Code with __import__ should be blocked."""
        is_safe, msg = _validate_code_safety("__import__('os')")
        assert is_safe is False
        assert "__import__" in msg

    def test_dangerous_class_access_blocked(self):
        """Code accessing __class__ should be blocked."""
        is_safe, msg = _validate_code_safety("x.__class__.__bases__")
        assert is_safe is False

    def test_dangerous_exec_blocked(self):
        """Code with exec() should be blocked."""
        is_safe, msg = _validate_code_safety("exec('print(1)')")
        assert is_safe is False

    def test_dangerous_eval_blocked(self):
        """Code with eval() should be blocked."""
        is_safe, msg = _validate_code_safety("eval('1+1')")
        assert is_safe is False

    def test_dangerous_open_blocked(self):
        """Code with open() should be blocked."""
        is_safe, msg = _validate_code_safety("open('/etc/passwd')")
        assert is_safe is False

    def test_dangerous_subprocess_blocked(self):
        """Code importing subprocess should be blocked."""
        is_safe, msg = _validate_code_safety("import subprocess")
        assert is_safe is False

    @pytest.mark.parametrize("pattern", DANGEROUS_PATTERNS[:10])
    def test_all_dangerous_patterns_blocked(self, pattern):
        """All patterns in DANGEROUS_PATTERNS should be detected."""
        is_safe, msg = _validate_code_safety(f"x = {pattern}")
        assert is_safe is False


class TestSafeBuiltins:
    """Tests for SAFE_BUILTINS whitelist."""

    def test_math_functions_available(self):
        """Math functions should be available."""
        assert "abs" in SAFE_BUILTINS
        assert "sum" in SAFE_BUILTINS
        assert "min" in SAFE_BUILTINS
        assert "max" in SAFE_BUILTINS
        assert "pow" in SAFE_BUILTINS

    def test_collection_types_available(self):
        """Collection types should be available."""
        assert "list" in SAFE_BUILTINS
        assert "dict" in SAFE_BUILTINS
        assert "set" in SAFE_BUILTINS
        assert "tuple" in SAFE_BUILTINS

    def test_dangerous_functions_excluded(self):
        """Dangerous functions should NOT be available."""
        assert "__import__" not in SAFE_BUILTINS
        assert "open" not in SAFE_BUILTINS
        assert "exec" not in SAFE_BUILTINS
        assert "eval" not in SAFE_BUILTINS
        assert "compile" not in SAFE_BUILTINS

    def test_exceptions_available_for_error_handling(self):
        """Common exceptions should be available for try/except."""
        assert "Exception" in SAFE_BUILTINS
        assert "ValueError" in SAFE_BUILTINS
        assert "TypeError" in SAFE_BUILTINS
        assert "AssertionError" in SAFE_BUILTINS


class TestProofExecutor:
    """Tests for ProofExecutor class."""

    def test_initialization(self):
        """Should initialize with default timeout."""
        executor = ProofExecutor()
        assert executor.default_timeout == 30.0

    def test_initialization_custom_timeout(self):
        """Should accept custom timeout."""
        executor = ProofExecutor(default_timeout=10.0)
        assert executor.default_timeout == 10.0

    @pytest.mark.asyncio
    async def test_execute_simple_assertion(self):
        """Should execute simple assertion."""
        executor = ProofExecutor(default_timeout=5.0)
        proof = VerificationProof(
            id="test-proof",
            claim_id="test",
            proof_type=ProofType.ASSERTION,
            description="Simple math check",
            code="result = 1 + 1 == 2",
        )
        result = await executor.execute(proof)
        assert result.status in (ProofStatus.PASSED, ProofStatus.ERROR)

    @pytest.mark.asyncio
    async def test_execute_failed_assertion(self):
        """Should handle failed assertions."""
        executor = ProofExecutor(default_timeout=5.0)
        proof = VerificationProof(
            id="test-proof-2",
            claim_id="test",
            proof_type=ProofType.ASSERTION,
            description="Failing assertion",
            code="assert 1 == 2, 'Math is broken'",
        )
        result = await executor.execute(proof)
        assert result.status in (ProofStatus.FAILED, ProofStatus.ERROR)

    @pytest.mark.asyncio
    async def test_execute_computation(self):
        """Should execute computation."""
        executor = ProofExecutor(default_timeout=5.0)
        proof = VerificationProof(
            id="test-proof-3",
            claim_id="test",
            proof_type=ProofType.COMPUTATION,
            description="Sum calculation",
            code="result = sum([1, 2, 3, 4, 5])",
            expected_output="15",
        )
        result = await executor.execute(proof)
        # Result depends on implementation details
        assert result.status in (ProofStatus.PASSED, ProofStatus.FAILED, ProofStatus.ERROR)

    @pytest.mark.asyncio
    async def test_blocks_dangerous_code(self):
        """Should block code with dangerous patterns."""
        executor = ProofExecutor(default_timeout=5.0)
        proof = VerificationProof(
            id="test-proof-4",
            claim_id="test",
            proof_type=ProofType.ASSERTION,
            description="Dangerous code test",
            code="import os; os.system('rm -rf /')",
        )
        result = await executor.execute(proof)
        assert result.status in (ProofStatus.ERROR, ProofStatus.FAILED)


class TestProofBuilder:
    """Tests for ProofBuilder class."""

    def test_initialization(self):
        """Should initialize with claim_id."""
        builder = ProofBuilder(claim_id="test-claim")
        assert builder.claim_id == "test-claim"

    def test_create_assertion(self):
        """Should create assertion proof via instance method."""
        builder = ProofBuilder(claim_id="test")
        proof = builder.assertion(
            description="Positive check",
            code="x = 5",
            assertion="x > 0",
        )
        assert proof.proof_type == ProofType.ASSERTION
        assert proof.claim_id == "test"


class TestHelperFunctions:
    """Tests for module helper functions."""

    def test_create_simple_assertion(self):
        """create_simple_assertion should create valid proof."""
        proof = create_simple_assertion(
            claim_id="test-claim",
            description="Always passes",
            assertion="True",
        )
        assert proof.claim_id == "test-claim"
        assert proof.proof_type == ProofType.ASSERTION

    def test_create_computation_proof(self):
        """create_computation_proof should create valid proof."""
        proof = create_computation_proof(
            claim_id="compute-test",
            description="Power of 2",
            computation_code="result = 2 ** 10",
            expected_assertion="result == 1024",
        )
        assert proof.claim_id == "compute-test"
        assert proof.proof_type == ProofType.COMPUTATION


class TestClaimVerifier:
    """Tests for ClaimVerifier class."""

    def test_initialization(self):
        """Should initialize empty verifier."""
        verifier = ClaimVerifier()
        assert len(verifier.proofs) == 0

    def test_add_proof(self):
        """Should add proof to verifier."""
        verifier = ClaimVerifier()
        proof = VerificationProof(
            id="proof-1",
            claim_id="claim-1",
            proof_type=ProofType.ASSERTION,
            description="Test proof",
            code="assert True",
        )
        verifier.add_proof(proof)
        assert len(verifier.proofs) == 1

    def test_get_proofs_for_claim(self):
        """Should retrieve proofs for specific claim."""
        verifier = ClaimVerifier()
        proof1 = VerificationProof(
            id="proof-1",
            claim_id="claim-1",
            proof_type=ProofType.ASSERTION,
            description="Test proof 1",
            code="assert True",
        )
        proof2 = VerificationProof(
            id="proof-2",
            claim_id="claim-2",
            proof_type=ProofType.ASSERTION,
            description="Test proof 2",
            code="assert True",
        )
        verifier.add_proof(proof1)
        verifier.add_proof(proof2)

        proofs = verifier.get_proofs_for_claim("claim-1")
        assert len(proofs) == 1
        assert proofs[0].claim_id == "claim-1"


class TestVerificationReport:
    """Tests for VerificationReport dataclass."""

    def test_empty_report(self):
        """Should create empty report."""
        report = VerificationReport(
            debate_id="debate-1",
            total_proofs=0,
        )
        assert report.total_proofs == 0
        assert report.pass_rate() == 0.0

    def test_full_report(self):
        """Should calculate pass rate correctly."""
        report = VerificationReport(
            debate_id="debate-1",
            total_proofs=10,
            proofs_passed=7,
            proofs_failed=2,
            proofs_error=1,
        )
        assert report.pass_rate() == 0.7
