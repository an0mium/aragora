"""Tests for executable verification proofs."""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from aragora.verification.proofs import (
    EXEC_TIMEOUT_SECONDS,
    DANGEROUS_PATTERNS,
    SAFE_BUILTINS,
    ProofType,
    ProofStatus,
    VerificationProof,
    VerificationResult,
    ProofExecutor,
    ClaimVerifier,
    VerificationReport,
    ProofBuilder,
    create_simple_assertion,
    create_computation_proof,
    verify_claim_set,
    _validate_code_safety,
)


class TestConstants:
    """Test module constants."""

    def test_exec_timeout_is_reasonable(self):
        """Test timeout is reasonable for proofs."""
        assert EXEC_TIMEOUT_SECONDS >= 1.0
        assert EXEC_TIMEOUT_SECONDS <= 60.0

    def test_dangerous_patterns_include_exploits(self):
        """Test dangerous patterns include known exploits."""
        # Check for sandbox escape patterns
        assert "__class__" in DANGEROUS_PATTERNS
        assert "__builtins__" in DANGEROUS_PATTERNS
        assert "__import__" in DANGEROUS_PATTERNS
        assert "exec(" in DANGEROUS_PATTERNS
        assert "eval(" in DANGEROUS_PATTERNS
        assert "os." in DANGEROUS_PATTERNS
        assert "subprocess" in DANGEROUS_PATTERNS

    def test_safe_builtins_excludes_dangerous(self):
        """Test safe builtins don't include dangerous functions."""
        dangerous = ["__import__", "open", "exec", "eval", "compile"]
        for d in dangerous:
            assert d not in SAFE_BUILTINS

    def test_safe_builtins_includes_math(self):
        """Test safe builtins include math functions."""
        math_funcs = ["abs", "sum", "max", "min", "pow", "round"]
        for f in math_funcs:
            assert f in SAFE_BUILTINS

    def test_safe_builtins_includes_collections(self):
        """Test safe builtins include collection types."""
        collections = ["list", "dict", "set", "tuple", "frozenset"]
        for c in collections:
            assert c in SAFE_BUILTINS


class TestValidateCodeSafety:
    """Test code safety validation."""

    def test_safe_code_passes(self):
        """Test that safe code passes validation."""
        safe_codes = [
            "x = 1 + 2",
            "result = sum([1, 2, 3])",
            "items = list(range(10))",
            "assert 1 == 1",
        ]
        for code in safe_codes:
            is_safe, error = _validate_code_safety(code)
            assert is_safe is True, f"Code should be safe: {code}"

    def test_dangerous_code_blocked(self):
        """Test that dangerous code is blocked."""
        dangerous_codes = [
            "exec('print(1)')",
            "eval('1+1')",
            "import os; os.system('ls')",
            "open('/etc/passwd').read()",
            "x.__class__.__bases__",
            "__import__('os')",
        ]
        for code in dangerous_codes:
            is_safe, error = _validate_code_safety(code)
            assert is_safe is False, f"Code should be dangerous: {code}"
            assert error != ""


class TestProofType:
    """Test ProofType enum."""

    def test_all_types_defined(self):
        """Test all expected proof types exist."""
        expected = [
            "ASSERTION",
            "CODE_EXECUTION",
            "API_CALL",
            "COMPUTATION",
            "TEST_SUITE",
            "PROPERTY_CHECK",
            "STATIC_ANALYSIS",
            "MANUAL",
        ]
        for t in expected:
            assert hasattr(ProofType, t)

    def test_type_values(self):
        """Test type values."""
        assert ProofType.ASSERTION.value == "assertion"
        assert ProofType.COMPUTATION.value == "computation"


class TestProofStatus:
    """Test ProofStatus enum."""

    def test_all_statuses_defined(self):
        """Test all expected statuses exist."""
        expected = [
            "PENDING",
            "RUNNING",
            "PASSED",
            "FAILED",
            "ERROR",
            "SKIPPED",
            "TIMEOUT",
        ]
        for s in expected:
            assert hasattr(ProofStatus, s)


class TestVerificationProof:
    """Test VerificationProof dataclass."""

    def test_create_minimal_proof(self):
        """Test creating a proof with minimal fields."""
        proof = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.ASSERTION,
            description="Test proof",
            code="assert True",
        )
        assert proof.id == "test1"
        assert proof.claim_id == "claim1"
        assert proof.status == ProofStatus.PENDING

    def test_auto_generate_id(self):
        """Test ID is auto-generated if empty."""
        proof = VerificationProof(
            id="",
            claim_id="claim1",
            proof_type=ProofType.ASSERTION,
            description="Test",
            code="x = 1",
        )
        assert proof.id != ""
        assert len(proof.id) == 12

    def test_auto_compute_hash(self):
        """Test hash is auto-computed."""
        proof = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.ASSERTION,
            description="Test",
            code="x = 1",
        )
        assert proof.proof_hash != ""
        assert len(proof.proof_hash) == 16

    def test_same_code_same_hash(self):
        """Test same code produces same hash."""
        proof1 = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.ASSERTION,
            description="Test",
            code="x = 1",
            assertion="x == 1",
        )
        proof2 = VerificationProof(
            id="test2",
            claim_id="claim2",
            proof_type=ProofType.ASSERTION,
            description="Different desc",
            code="x = 1",
            assertion="x == 1",
        )
        assert proof1.proof_hash == proof2.proof_hash

    def test_to_dict(self):
        """Test serialization to dict."""
        proof = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.ASSERTION,
            description="Test proof",
            code="x = 1",
            assertion="x == 1",
            timeout_seconds=10.0,
        )
        data = proof.to_dict()

        assert data["id"] == "test1"
        assert data["claim_id"] == "claim1"
        assert data["proof_type"] == "assertion"
        assert data["code"] == "x = 1"
        assert data["timeout_seconds"] == 10.0

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "id": "test1",
            "claim_id": "claim1",
            "proof_type": "assertion",
            "description": "Test",
            "code": "x = 1",
            "assertion": "x == 1",
            "status": "passed",
            "run_count": 5,
        }
        proof = VerificationProof.from_dict(data)

        assert proof.id == "test1"
        assert proof.proof_type == ProofType.ASSERTION
        assert proof.status == ProofStatus.PASSED
        assert proof.run_count == 5

    def test_dependencies_default_empty(self):
        """Test dependencies default to empty list."""
        proof = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.ASSERTION,
            description="Test",
            code="x = 1",
        )
        assert proof.dependencies == []


class TestVerificationResult:
    """Test VerificationResult dataclass."""

    def test_create_result(self):
        """Test creating a verification result."""
        result = VerificationResult(
            proof_id="proof1",
            claim_id="claim1",
            status=ProofStatus.PASSED,
            passed=True,
            output="Success",
        )
        assert result.proof_id == "proof1"
        assert result.passed is True

    def test_to_dict(self):
        """Test serialization to dict."""
        result = VerificationResult(
            proof_id="proof1",
            claim_id="claim1",
            status=ProofStatus.FAILED,
            passed=False,
            error="Assertion failed",
            assertion_value=False,
        )
        data = result.to_dict()

        assert data["proof_id"] == "proof1"
        assert data["status"] == "failed"
        assert data["passed"] is False
        assert data["error"] == "Assertion failed"

    def test_assertion_details(self):
        """Test assertion-specific fields."""
        result = VerificationResult(
            proof_id="proof1",
            claim_id="claim1",
            status=ProofStatus.PASSED,
            passed=True,
            assertion_value=True,
            assertion_details="x == 1 evaluated to True",
        )
        assert result.assertion_value is True
        assert "evaluated" in result.assertion_details

    def test_output_comparison_fields(self):
        """Test output comparison fields."""
        result = VerificationResult(
            proof_id="proof1",
            claim_id="claim1",
            status=ProofStatus.FAILED,
            passed=False,
            output_matched=False,
            output_diff="Expected: foo\nActual: bar",
        )
        assert result.output_matched is False
        assert "Expected" in result.output_diff


class TestProofExecutor:
    """Test ProofExecutor class."""

    def test_init_default(self):
        """Test default initialization."""
        executor = ProofExecutor()
        assert executor.allow_network is False
        assert executor.allow_filesystem is False
        assert executor.default_timeout == 30.0

    def test_init_custom(self):
        """Test custom initialization."""
        executor = ProofExecutor(
            allow_network=True,
            allow_filesystem=True,
            default_timeout=60.0,
            max_output_size=5000,
        )
        assert executor.allow_network is True
        assert executor.allow_filesystem is True
        assert executor.default_timeout == 60.0
        assert executor.max_output_size == 5000

    @pytest.mark.asyncio
    async def test_execute_skips_network_required(self):
        """Test execution skips proof requiring network."""
        executor = ProofExecutor(allow_network=False)
        proof = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.API_CALL,
            description="Network test",
            code="import requests",
            requires_network=True,
        )
        result = await executor.execute(proof)
        assert result.status == ProofStatus.SKIPPED
        assert "network" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_skips_filesystem_required(self):
        """Test execution skips proof requiring filesystem."""
        executor = ProofExecutor(allow_filesystem=False)
        proof = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.CODE_EXECUTION,
            description="File test",
            code="with open('test.txt') as f: pass",
            requires_filesystem=True,
        )
        result = await executor.execute(proof)
        assert result.status == ProofStatus.SKIPPED
        assert "filesystem" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_simple_assertion_pass(self):
        """Test executing a simple passing assertion."""
        executor = ProofExecutor()
        proof = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.ASSERTION,
            description="Simple math",
            code="x = 1 + 1",
            assertion="x == 2",
            timeout_seconds=5.0,
        )
        result = await executor.execute(proof)
        assert result.status == ProofStatus.PASSED
        assert result.passed is True
        assert result.assertion_value is True

    @pytest.mark.asyncio
    async def test_execute_simple_assertion_fail(self):
        """Test executing a failing assertion."""
        executor = ProofExecutor()
        proof = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.ASSERTION,
            description="Wrong math",
            code="x = 1 + 1",
            assertion="x == 3",
            timeout_seconds=5.0,
        )
        result = await executor.execute(proof)
        assert result.status == ProofStatus.FAILED
        assert result.passed is False
        assert result.assertion_value is False

    @pytest.mark.asyncio
    async def test_execute_code_without_assertion(self):
        """Test executing code without explicit assertion."""
        executor = ProofExecutor()
        proof = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.ASSERTION,
            description="Code only",
            code="x = 1 + 1",
            timeout_seconds=5.0,
        )
        result = await executor.execute(proof)
        assert result.status == ProofStatus.PASSED

    @pytest.mark.asyncio
    async def test_execute_updates_proof_status(self):
        """Test execution updates proof status."""
        executor = ProofExecutor()
        proof = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.ASSERTION,
            description="Status test",
            code="x = 1",
            assertion="x == 1",
            timeout_seconds=5.0,
        )
        assert proof.status == ProofStatus.PENDING
        assert proof.run_count == 0

        await executor.execute(proof)

        assert proof.status == ProofStatus.PASSED
        assert proof.run_count == 1
        assert proof.last_run is not None

    @pytest.mark.asyncio
    async def test_execute_computation(self):
        """Test executing a computation proof."""
        executor = ProofExecutor()
        # Use iterative approach since recursive functions don't work in sandbox
        proof = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.COMPUTATION,
            description="Factorial iterative",
            code="""
result = 1
for i in range(1, 6):
    result = result * i
""",
            assertion="result == 120",
            timeout_seconds=5.0,
        )
        result = await executor.execute(proof)
        assert result.status == ProofStatus.PASSED
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_dangerous_code_blocked(self):
        """Test that dangerous code is blocked."""
        executor = ProofExecutor()
        proof = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.CODE_EXECUTION,
            description="Dangerous",
            code="import os; os.system('ls')",
            timeout_seconds=5.0,
        )
        result = await executor.execute(proof)
        # Should fail due to dangerous pattern
        assert result.status in (ProofStatus.ERROR, ProofStatus.FAILED)


class TestClaimVerifier:
    """Test ClaimVerifier class."""

    def test_init_default(self):
        """Test default initialization."""
        verifier = ClaimVerifier()
        assert verifier.executor is not None
        assert len(verifier.proofs) == 0

    def test_init_with_executor(self):
        """Test initialization with custom executor."""
        executor = ProofExecutor(allow_network=True)
        verifier = ClaimVerifier(executor=executor)
        assert verifier.executor.allow_network is True

    def test_add_proof(self):
        """Test adding a proof."""
        verifier = ClaimVerifier()
        proof = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.ASSERTION,
            description="Test",
            code="x = 1",
        )
        verifier.add_proof(proof)

        assert "test1" in verifier.proofs
        assert "claim1" in verifier.claim_proofs
        assert "test1" in verifier.claim_proofs["claim1"]

    def test_get_proofs_for_claim(self):
        """Test getting proofs for a claim."""
        verifier = ClaimVerifier()
        proof1 = VerificationProof(
            id="p1", claim_id="c1", proof_type=ProofType.ASSERTION,
            description="Test1", code="x = 1",
        )
        proof2 = VerificationProof(
            id="p2", claim_id="c1", proof_type=ProofType.ASSERTION,
            description="Test2", code="y = 2",
        )
        proof3 = VerificationProof(
            id="p3", claim_id="c2", proof_type=ProofType.ASSERTION,
            description="Test3", code="z = 3",
        )

        verifier.add_proof(proof1)
        verifier.add_proof(proof2)
        verifier.add_proof(proof3)

        c1_proofs = verifier.get_proofs_for_claim("c1")
        assert len(c1_proofs) == 2
        assert proof1 in c1_proofs
        assert proof2 in c1_proofs

        c2_proofs = verifier.get_proofs_for_claim("c2")
        assert len(c2_proofs) == 1
        assert proof3 in c2_proofs

    def test_get_proofs_for_nonexistent_claim(self):
        """Test getting proofs for nonexistent claim."""
        verifier = ClaimVerifier()
        proofs = verifier.get_proofs_for_claim("nonexistent")
        assert proofs == []

    @pytest.mark.asyncio
    async def test_verify_claim(self):
        """Test verifying all proofs for a claim."""
        verifier = ClaimVerifier()
        proof = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.ASSERTION,
            description="Test",
            code="x = 1",
            assertion="x == 1",
            timeout_seconds=5.0,
        )
        verifier.add_proof(proof)

        results = await verifier.verify_claim("claim1")
        assert len(results) == 1
        assert results[0].passed is True

    @pytest.mark.asyncio
    async def test_verify_all(self):
        """Test verifying all proofs."""
        verifier = ClaimVerifier()
        proof1 = VerificationProof(
            id="p1", claim_id="c1", proof_type=ProofType.ASSERTION,
            description="Test1", code="x = 1", assertion="x == 1",
            timeout_seconds=5.0,
        )
        proof2 = VerificationProof(
            id="p2", claim_id="c2", proof_type=ProofType.ASSERTION,
            description="Test2", code="y = 2", assertion="y == 2",
            timeout_seconds=5.0,
        )
        verifier.add_proof(proof1)
        verifier.add_proof(proof2)

        results = await verifier.verify_all()
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_get_claim_verification_status_no_proofs(self):
        """Test status for claim with no proofs."""
        verifier = ClaimVerifier()
        status = verifier.get_claim_verification_status("claim1")

        assert status["claim_id"] == "claim1"
        assert status["has_proofs"] is False
        assert status["verified"] is False

    @pytest.mark.asyncio
    async def test_get_claim_verification_status_verified(self):
        """Test status for verified claim."""
        verifier = ClaimVerifier()
        proof = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.ASSERTION,
            description="Test",
            code="x = 1",
            assertion="x == 1",
            timeout_seconds=5.0,
        )
        verifier.add_proof(proof)
        await verifier.verify_claim("claim1")

        status = verifier.get_claim_verification_status("claim1")
        assert status["has_proofs"] is True
        assert status["verified"] is True
        assert status["passed_count"] == 1


class TestVerificationReport:
    """Test VerificationReport dataclass."""

    def test_create_report(self):
        """Test creating a report."""
        report = VerificationReport(debate_id="debate1")
        assert report.debate_id == "debate1"
        assert report.total_claims == 0
        assert report.total_proofs == 0

    def test_verification_rate_zero_claims(self):
        """Test verification rate with zero claims."""
        report = VerificationReport(debate_id="debate1")
        assert report.verification_rate() == 0.0

    def test_verification_rate_calculation(self):
        """Test verification rate calculation."""
        report = VerificationReport(
            debate_id="debate1",
            claims_with_proofs=10,
            claims_verified=7,
        )
        assert report.verification_rate() == 0.7

    def test_pass_rate_zero_executed(self):
        """Test pass rate with zero executed proofs."""
        report = VerificationReport(debate_id="debate1")
        assert report.pass_rate() == 0.0

    def test_pass_rate_calculation(self):
        """Test pass rate calculation."""
        report = VerificationReport(
            debate_id="debate1",
            proofs_passed=8,
            proofs_failed=2,
            proofs_error=0,
        )
        assert report.pass_rate() == 0.8

    def test_to_dict(self):
        """Test serialization to dict."""
        report = VerificationReport(
            debate_id="debate1",
            total_claims=10,
            claims_with_proofs=8,
            claims_verified=6,
            total_proofs=15,
            proofs_passed=12,
        )
        data = report.to_dict()

        assert data["debate_id"] == "debate1"
        assert data["statistics"]["total_claims"] == 10
        assert data["proofs"]["total"] == 15
        assert data["proofs"]["passed"] == 12

    def test_generate_summary(self):
        """Test generating human-readable summary."""
        report = VerificationReport(
            debate_id="debate1",
            total_claims=10,
            claims_with_proofs=8,
            claims_verified=6,
            total_proofs=15,
            proofs_passed=12,
            proofs_failed=2,
        )
        summary = report.generate_summary()

        assert "debate1" in summary
        assert "8/10" in summary
        assert "12" in summary


class TestProofBuilder:
    """Test ProofBuilder helper class."""

    def test_create_assertion(self):
        """Test creating an assertion proof."""
        builder = ProofBuilder(claim_id="claim1", created_by="test")
        proof = builder.assertion(
            description="Test assertion",
            code="x = 1",
            assertion="x == 1",
        )
        assert proof.claim_id == "claim1"
        assert proof.proof_type == ProofType.ASSERTION
        assert proof.created_by == "test"

    def test_create_output_check(self):
        """Test creating an output check proof."""
        builder = ProofBuilder(claim_id="claim1")
        proof = builder.output_check(
            description="Test output",
            code="print('hello')",
            expected_output="hello",
        )
        assert proof.proof_type == ProofType.CODE_EXECUTION
        assert proof.expected_output == "hello"

    def test_create_computation(self):
        """Test creating a computation proof."""
        builder = ProofBuilder(claim_id="claim1")
        proof = builder.computation(
            description="Test computation",
            code="result = 2 ** 10",
            assertion="result == 1024",
        )
        assert proof.proof_type == ProofType.COMPUTATION

    def test_create_property_check(self):
        """Test creating a property check proof."""
        builder = ProofBuilder(claim_id="claim1")
        proof = builder.property_check(
            description="Test property",
            code="values = [1, 2, 3, 4, 5]",
            property_assertion="all(v > 0 for v in values)",
        )
        assert proof.proof_type == ProofType.PROPERTY_CHECK


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_simple_assertion(self):
        """Test create_simple_assertion function."""
        proof = create_simple_assertion(
            claim_id="claim1",
            description="Simple test",
            assertion="1 + 1 == 2",
        )
        assert proof.claim_id == "claim1"
        assert proof.proof_type == ProofType.ASSERTION
        assert proof.assertion == "1 + 1 == 2"
        assert proof.code == ""

    def test_create_computation_proof(self):
        """Test create_computation_proof function."""
        proof = create_computation_proof(
            claim_id="claim1",
            description="Computation test",
            computation_code="x = 5 * 5",
            expected_assertion="x == 25",
        )
        assert proof.proof_type == ProofType.COMPUTATION
        assert "x = 5 * 5" in proof.code

    @pytest.mark.asyncio
    async def test_verify_claim_set(self):
        """Test verify_claim_set function."""
        claims = [
            ("c1", "One plus one equals two"),
            ("c2", "Two times two equals four"),
        ]
        proofs = [
            VerificationProof(
                id="p1", claim_id="c1",
                proof_type=ProofType.ASSERTION,
                description="Addition",
                code="result = 1 + 1",
                assertion="result == 2",
                timeout_seconds=5.0,
            ),
            VerificationProof(
                id="p2", claim_id="c2",
                proof_type=ProofType.ASSERTION,
                description="Multiplication",
                code="result = 2 * 2",
                assertion="result == 4",
                timeout_seconds=5.0,
            ),
        ]

        report = await verify_claim_set(claims, proofs)

        assert report.total_claims == 2
        assert report.total_proofs == 2
        assert report.proofs_passed == 2
        assert report.claims_verified == 2


class TestSecuritySandbox:
    """Test security of sandbox execution."""

    @pytest.mark.asyncio
    async def test_import_blocked(self):
        """Test that imports are blocked."""
        executor = ProofExecutor()
        proof = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.CODE_EXECUTION,
            description="Import test",
            code="__import__('os')",
            timeout_seconds=5.0,
        )
        result = await executor.execute(proof)
        assert result.status in (ProofStatus.ERROR, ProofStatus.FAILED)

    @pytest.mark.asyncio
    async def test_file_access_blocked(self):
        """Test that file access is blocked."""
        executor = ProofExecutor()
        proof = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.CODE_EXECUTION,
            description="File test",
            code="open('/etc/passwd').read()",
            timeout_seconds=5.0,
        )
        result = await executor.execute(proof)
        assert result.status in (ProofStatus.ERROR, ProofStatus.FAILED)

    @pytest.mark.asyncio
    async def test_class_introspection_blocked(self):
        """Test that class introspection is blocked."""
        executor = ProofExecutor()
        proof = VerificationProof(
            id="test1",
            claim_id="claim1",
            proof_type=ProofType.CODE_EXECUTION,
            description="Class introspection",
            code="().__class__.__bases__[0].__subclasses__()",
            timeout_seconds=5.0,
        )
        result = await executor.execute(proof)
        assert result.status in (ProofStatus.ERROR, ProofStatus.FAILED)
