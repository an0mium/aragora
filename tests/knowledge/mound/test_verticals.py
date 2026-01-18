"""
Tests for vertical-specific knowledge modules.

Tests the BaseVerticalKnowledge interface and SoftwareKnowledge implementation.
Other verticals (legal, healthcare, accounting, research) will be tested
as they are implemented.
"""

import pytest
from datetime import datetime, timedelta

from aragora.knowledge.mound.verticals.base import (
    BaseVerticalKnowledge,
    ComplianceCheckResult,
    PatternMatch,
    VerticalCapabilities,
    VerticalFact,
)
from aragora.knowledge.mound.verticals.software import (
    SecretPattern,
    SoftwareKnowledge,
    VulnerabilityPattern,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def software_vertical():
    """Create a SoftwareKnowledge instance."""
    return SoftwareKnowledge()


@pytest.fixture
def sample_vulnerable_code():
    """Sample code with various vulnerabilities."""
    return '''
import os
import subprocess

# SQL Injection vulnerability
def get_user(user_id):
    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
    return cursor.fetchone()

# Command Injection
def run_command(cmd):
    os.system(cmd)
    subprocess.call(cmd, shell=True)

# XSS vulnerability
def render_html(user_input):
    return f"<div>{user_input}</div>" + document.write(user_input)

# Hardcoded credentials
API_KEY = "sk-1234567890abcdef"
password = "mysecretpassword123"

# Weak cryptography
import hashlib
hash = hashlib.md5(data)
'''


@pytest.fixture
def sample_code_with_secrets():
    """Sample code with exposed secrets."""
    return '''
# AWS credentials
aws_access_key = "AKIAIOSFODNN7EXAMPLE"
aws_secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

# GitHub token
github_token = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Private key
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA...
-----END RSA PRIVATE KEY-----

# JWT token
jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
'''


@pytest.fixture
def clean_code():
    """Sample code without vulnerabilities."""
    return '''
import os
from secrets import token_urlsafe

def get_user(user_id: int):
    """Safely get user by ID using parameterized query."""
    return db.query(User).filter(User.id == user_id).first()

def generate_token():
    """Generate a secure random token."""
    return token_urlsafe(32)
'''


# ============================================================================
# VerticalFact Tests
# ============================================================================


class TestVerticalFact:
    """Tests for VerticalFact dataclass."""

    def test_basic_creation(self):
        """Test creating a basic fact."""
        fact = VerticalFact(
            id="test_fact_1",
            vertical="software",
            content="SQL injection vulnerability detected",
            category="vulnerability",
            confidence=0.8,
        )

        assert fact.id == "test_fact_1"
        assert fact.vertical == "software"
        assert fact.confidence == 0.8

    def test_adjusted_confidence_fresh(self):
        """Test adjusted confidence for fresh fact."""
        fact = VerticalFact(
            id="test_fact",
            vertical="software",
            content="test",
            category="test",
            confidence=0.8,
            staleness_days=0,
            decay_rate=0.1,
        )

        # Fresh fact should have no decay
        assert fact.adjusted_confidence == 0.8

    def test_adjusted_confidence_with_decay(self):
        """Test adjusted confidence decays over time."""
        fact = VerticalFact(
            id="test_fact",
            vertical="software",
            content="test",
            category="test",
            confidence=0.8,
            staleness_days=5,
            decay_rate=0.1,  # 10% per day
        )

        # After 5 days at 0.1/day = 50% decay
        # adjusted = 0.8 * (1 - 0.5) = 0.4
        assert fact.adjusted_confidence == pytest.approx(0.4)

    def test_adjusted_confidence_caps_decay(self):
        """Test that decay is capped at 90%."""
        fact = VerticalFact(
            id="test_fact",
            vertical="software",
            content="test",
            category="test",
            confidence=0.8,
            staleness_days=100,  # Very stale
            decay_rate=0.1,
        )

        # Decay capped at 0.9, so minimum is 0.8 * 0.1 = 0.08
        assert fact.adjusted_confidence == pytest.approx(0.08)

    def test_needs_reverification(self):
        """Test reverification detection."""
        fresh_fact = VerticalFact(
            id="fresh",
            vertical="software",
            content="test",
            category="test",
            confidence=0.8,
            staleness_days=0,
        )
        assert not fresh_fact.needs_reverification

        stale_fact = VerticalFact(
            id="stale",
            vertical="software",
            content="test",
            category="test",
            confidence=0.8,
            staleness_days=10,
            decay_rate=0.1,
        )
        # Adjusted confidence is well below 50% of original
        assert stale_fact.needs_reverification

    def test_refresh(self):
        """Test refreshing a fact."""
        fact = VerticalFact(
            id="test",
            vertical="software",
            content="test",
            category="test",
            confidence=0.5,
            staleness_days=10,
        )

        original_verified = fact.verified_at
        fact.refresh(new_confidence=0.9)

        assert fact.staleness_days == 0
        assert fact.confidence == 0.9
        assert fact.verified_at > original_verified

    def test_to_dict(self):
        """Test conversion to dictionary."""
        fact = VerticalFact(
            id="test",
            vertical="software",
            content="test content",
            category="vulnerability",
            confidence=0.8,
            metadata={"severity": "high"},
        )

        d = fact.to_dict()

        assert d["id"] == "test"
        assert d["vertical"] == "software"
        assert d["content"] == "test content"
        assert d["confidence"] == 0.8
        assert d["metadata"]["severity"] == "high"
        assert "adjusted_confidence" in d


# ============================================================================
# VerticalCapabilities Tests
# ============================================================================


class TestVerticalCapabilities:
    """Tests for VerticalCapabilities."""

    def test_default_capabilities(self):
        """Test default capability values."""
        caps = VerticalCapabilities()

        assert caps.supports_pattern_detection is True
        assert caps.supports_cross_reference is False
        assert caps.supports_compliance_check is False
        assert caps.requires_llm is False
        assert caps.requires_vector_search is True

    def test_custom_capabilities(self):
        """Test custom capability configuration."""
        caps = VerticalCapabilities(
            supports_compliance_check=True,
            compliance_frameworks=["OWASP", "CWE"],
            document_types=["code", "config"],
        )

        assert caps.supports_compliance_check is True
        assert "OWASP" in caps.compliance_frameworks
        assert "code" in caps.document_types


# ============================================================================
# SoftwareKnowledge Tests
# ============================================================================


class TestSoftwareKnowledge:
    """Tests for SoftwareKnowledge vertical."""

    def test_vertical_properties(self, software_vertical):
        """Test basic vertical properties."""
        assert software_vertical.vertical_id == "software"
        assert software_vertical.display_name == "Software Development"
        assert "vulnerability" in software_vertical.description.lower() or "security" in software_vertical.description.lower()

    def test_capabilities(self, software_vertical):
        """Test software vertical capabilities."""
        caps = software_vertical.capabilities

        assert caps.supports_pattern_detection is True
        assert caps.supports_compliance_check is True
        assert caps.requires_llm is False
        assert "OWASP" in caps.compliance_frameworks
        assert "CWE" in caps.compliance_frameworks
        assert "code" in caps.document_types

    def test_decay_rates(self, software_vertical):
        """Test software-specific decay rates."""
        rates = software_vertical.decay_rates

        # Vulnerabilities should decay faster than best practices
        assert rates["vulnerability"] > rates["best_practice"]
        # Secrets should decay fastest
        assert rates["secret"] > rates["vulnerability"]
        # Licenses are stable
        assert rates["license"] < rates["default"]

    def test_get_decay_rate(self, software_vertical):
        """Test decay rate lookup."""
        assert software_vertical.get_decay_rate("vulnerability") == 0.05
        assert software_vertical.get_decay_rate("unknown") == 0.02  # default

    # --------------------------------------------------------------------------
    # Fact Extraction Tests
    # --------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_extract_sql_injection(self, software_vertical, sample_vulnerable_code):
        """Test SQL injection detection."""
        facts = await software_vertical.extract_facts(sample_vulnerable_code)

        sql_facts = [f for f in facts if "SQL" in f.content]
        assert len(sql_facts) >= 1

        sql_fact = sql_facts[0]
        assert sql_fact.category == "vulnerability"
        assert sql_fact.provenance.get("cwe") == "CWE-89"
        assert sql_fact.metadata.get("severity") == "critical"

    @pytest.mark.asyncio
    async def test_extract_command_injection(self, software_vertical, sample_vulnerable_code):
        """Test command injection detection."""
        facts = await software_vertical.extract_facts(sample_vulnerable_code)

        cmd_facts = [f for f in facts if "Command" in f.content]
        assert len(cmd_facts) >= 1

        cmd_fact = cmd_facts[0]
        assert cmd_fact.provenance.get("cwe") == "CWE-78"
        assert cmd_fact.metadata.get("severity") == "critical"

    @pytest.mark.asyncio
    async def test_extract_hardcoded_credentials(self, software_vertical, sample_vulnerable_code):
        """Test hardcoded credential detection."""
        facts = await software_vertical.extract_facts(sample_vulnerable_code)

        cred_facts = [f for f in facts if "Hardcoded" in f.content or "credential" in f.content.lower()]
        assert len(cred_facts) >= 1

    @pytest.mark.asyncio
    async def test_extract_secrets(self, software_vertical, sample_code_with_secrets):
        """Test secret detection."""
        facts = await software_vertical.extract_facts(sample_code_with_secrets)

        secret_facts = [f for f in facts if f.category == "secret"]
        assert len(secret_facts) >= 1

        # Check we detect AWS keys
        aws_facts = [f for f in secret_facts if "AWS" in f.content]
        assert len(aws_facts) >= 1

        # Secrets should have high confidence
        for fact in secret_facts:
            assert fact.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_extract_weak_crypto(self, software_vertical, sample_vulnerable_code):
        """Test weak cryptography detection."""
        facts = await software_vertical.extract_facts(sample_vulnerable_code)

        crypto_facts = [f for f in facts if "cryptograph" in f.content.lower() or "MD5" in f.content]
        assert len(crypto_facts) >= 1

    @pytest.mark.asyncio
    async def test_extract_from_clean_code(self, software_vertical, clean_code):
        """Test that clean code produces no vulnerability facts."""
        facts = await software_vertical.extract_facts(clean_code)

        # Clean code should have no vulnerability or secret facts
        vuln_facts = [f for f in facts if f.category in ("vulnerability", "secret")]
        assert len(vuln_facts) == 0

    @pytest.mark.asyncio
    async def test_extract_with_metadata(self, software_vertical, sample_vulnerable_code):
        """Test that metadata is passed through to facts."""
        facts = await software_vertical.extract_facts(
            sample_vulnerable_code,
            metadata={"file": "app.py", "line": 10},
        )

        for fact in facts:
            assert fact.metadata.get("file") == "app.py"

    # --------------------------------------------------------------------------
    # Fact Validation Tests
    # --------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_validate_vulnerability_fact(self, software_vertical):
        """Test vulnerability fact validation."""
        fact = VerticalFact(
            id="vuln_1",
            vertical="software",
            content="SQL injection detected",
            category="vulnerability",
            confidence=0.8,
        )

        is_valid, new_confidence = await software_vertical.validate_fact(fact)

        assert is_valid is True
        # Vulnerability confidence should decrease slightly without re-verification
        assert new_confidence < fact.confidence

    @pytest.mark.asyncio
    async def test_validate_secret_fact(self, software_vertical):
        """Test secret fact validation."""
        fact = VerticalFact(
            id="secret_1",
            vertical="software",
            content="AWS key detected",
            category="secret",
            confidence=0.9,
        )

        is_valid, new_confidence = await software_vertical.validate_fact(fact)

        assert is_valid is True
        # Secret confidence drops faster (should be rotated)
        assert new_confidence < fact.confidence * 0.85

    @pytest.mark.asyncio
    async def test_validate_other_fact(self, software_vertical):
        """Test validation of other fact types."""
        fact = VerticalFact(
            id="bp_1",
            vertical="software",
            content="Use type hints",
            category="best_practice",
            confidence=0.7,
        )

        is_valid, new_confidence = await software_vertical.validate_fact(fact)

        assert is_valid is True
        # Other facts get slight confidence boost on re-validation
        assert new_confidence >= fact.confidence

    # --------------------------------------------------------------------------
    # Pattern Detection Tests
    # --------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_detect_recurring_vulnerability_pattern(self, software_vertical):
        """Test detection of recurring vulnerability patterns."""
        # Create multiple injection vulnerabilities
        facts = [
            software_vertical.create_fact(
                content=f"SQL injection in module {i}",
                category="vulnerability",
                metadata={"category": "injection"},
            )
            for i in range(3)
        ]

        patterns = await software_vertical.detect_patterns(facts)

        recurring = [p for p in patterns if p.pattern_type == "recurring_vulnerability"]
        assert len(recurring) >= 1
        assert recurring[0].confidence >= 0.7

    @pytest.mark.asyncio
    async def test_detect_scattered_secrets_pattern(self, software_vertical):
        """Test detection of scattered secrets pattern."""
        facts = [
            software_vertical.create_fact(
                content="AWS key in config.py",
                category="secret",
            ),
            software_vertical.create_fact(
                content="API key in utils.py",
                category="secret",
            ),
        ]

        patterns = await software_vertical.detect_patterns(facts)

        scattered = [p for p in patterns if p.pattern_type == "secret_management"]
        assert len(scattered) >= 1

    @pytest.mark.asyncio
    async def test_no_patterns_for_single_fact(self, software_vertical):
        """Test that single facts don't generate patterns."""
        facts = [
            software_vertical.create_fact(
                content="One vulnerability",
                category="vulnerability",
            )
        ]

        patterns = await software_vertical.detect_patterns(facts)

        # Should not detect recurring patterns with only one fact
        recurring = [p for p in patterns if p.pattern_type == "recurring_vulnerability"]
        assert len(recurring) == 0

    # --------------------------------------------------------------------------
    # Compliance Checking Tests
    # --------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_owasp_compliance_check(self, software_vertical, sample_vulnerable_code):
        """Test OWASP compliance checking."""
        facts = await software_vertical.extract_facts(sample_vulnerable_code)
        results = await software_vertical.check_compliance(facts, "OWASP")

        # Should have at least one OWASP violation
        assert len(results) >= 1

        # Check result structure
        result = results[0]
        assert result.framework == "OWASP"
        assert result.passed is False
        assert len(result.findings) >= 1

    @pytest.mark.asyncio
    async def test_cwe_compliance_check(self, software_vertical, sample_vulnerable_code):
        """Test CWE compliance checking."""
        facts = await software_vertical.extract_facts(sample_vulnerable_code)
        results = await software_vertical.check_compliance(facts, "CWE")

        # Should find CWE violations
        assert len(results) >= 1

        # Results should reference specific CWEs
        for result in results:
            assert result.framework == "CWE"
            assert "CWE" in result.rule_id

    @pytest.mark.asyncio
    async def test_compliance_with_clean_code(self, software_vertical, clean_code):
        """Test compliance checking on clean code."""
        facts = await software_vertical.extract_facts(clean_code)
        results = await software_vertical.check_compliance(facts, "OWASP")

        # Clean code should have no violations
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_compliance_frameworks(self, software_vertical):
        """Test listing available compliance frameworks."""
        frameworks = await software_vertical.get_compliance_frameworks()

        assert "OWASP" in frameworks
        assert "CWE" in frameworks

    # --------------------------------------------------------------------------
    # Helper Method Tests
    # --------------------------------------------------------------------------

    def test_create_fact_helper(self, software_vertical):
        """Test the create_fact helper method."""
        fact = software_vertical.create_fact(
            content="Test finding",
            category="vulnerability",
            confidence=0.8,
            metadata={"file": "test.py"},
        )

        assert fact.vertical == "software"
        assert fact.content == "Test finding"
        assert fact.category == "vulnerability"
        assert fact.confidence == 0.8
        assert fact.metadata["file"] == "test.py"
        assert fact.id.startswith("software_")

        # Decay rate should match category
        assert fact.decay_rate == software_vertical.get_decay_rate("vulnerability")


# ============================================================================
# PatternMatch Tests
# ============================================================================


class TestPatternMatch:
    """Tests for PatternMatch dataclass."""

    def test_pattern_creation(self):
        """Test creating a pattern match."""
        pattern = PatternMatch(
            pattern_id="p1",
            pattern_name="Recurring SQL Injection",
            pattern_type="recurring_vulnerability",
            description="Multiple SQL injection vulnerabilities found",
            confidence=0.85,
            supporting_facts=["f1", "f2", "f3"],
            metadata={"count": 3},
        )

        assert pattern.pattern_id == "p1"
        assert pattern.confidence == 0.85
        assert len(pattern.supporting_facts) == 3


# ============================================================================
# ComplianceCheckResult Tests
# ============================================================================


class TestComplianceCheckResult:
    """Tests for ComplianceCheckResult dataclass."""

    def test_result_creation(self):
        """Test creating a compliance check result."""
        result = ComplianceCheckResult(
            rule_id="owasp_injection",
            rule_name="A03:2021 Injection",
            framework="OWASP",
            passed=False,
            severity="high",
            findings=["SQL injection in get_user()"],
            evidence=["fact_1"],
            recommendations=["Use parameterized queries"],
            confidence=0.9,
        )

        assert result.rule_id == "owasp_injection"
        assert result.passed is False
        assert result.severity == "high"
        assert len(result.recommendations) == 1
