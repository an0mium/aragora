"""
Automated Security Regression Test Suite.

SOC 2 Controls: CC6.1, CC6.6, CC6.7, CC7.1, CC7.2

Covers OWASP Top 10 vulnerabilities and common security issues.
These tests are designed to catch regressions, not audit the entire codebase.

Run with: pytest tests/security/test_security_regression.py -v
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


# =============================================================================
# Test Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "aragora"

# Directories to skip during scanning
SKIP_DIRS = {
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    ".pytest_cache",
    "dist",
    "build",
    ".mypy_cache",
}


def should_skip_path(path: Path) -> bool:
    """Check if path should be skipped."""
    for part in path.parts:
        if part in SKIP_DIRS or part.startswith("."):
            return True
    return False


def is_test_file(path: Path) -> bool:
    """Check if file is a test file."""
    return "test" in str(path).lower() or "fixture" in str(path).lower()


# =============================================================================
# A01: Broken Access Control Tests
# =============================================================================


class TestBrokenAccessControl:
    """OWASP A01: Broken Access Control vulnerability tests."""

    def test_rbac_module_exists(self):
        """Verify RBAC infrastructure exists."""
        rbac_dir = SRC_ROOT / "rbac"
        assert rbac_dir.exists(), "RBAC module not found"

        # Check for essential RBAC files
        essential_files = ["models.py", "middleware.py", "defaults.py"]
        for f in essential_files:
            assert (rbac_dir / f).exists(), f"RBAC file {f} not found"

    def test_rbac_enforcer_exists(self):
        """Verify RBAC enforcer is implemented."""
        rbac_dir = SRC_ROOT / "rbac"
        if not rbac_dir.exists():
            pytest.skip("RBAC module not found")

        # Check for enforcer implementation
        found_enforcer = False
        for py_file in rbac_dir.rglob("*.py"):
            content = py_file.read_text()
            if "class" in content and "enforcer" in content.lower():
                found_enforcer = True
                break

        assert found_enforcer, "RBAC enforcer class not found"

    def test_route_permissions_defined(self):
        """Verify route permissions are defined."""
        middleware_file = SRC_ROOT / "rbac" / "middleware.py"
        if not middleware_file.exists():
            pytest.skip("RBAC middleware not found")

        content = middleware_file.read_text()
        assert "RoutePermission" in content or "route" in content.lower(), (
            "Route permissions not defined"
        )


# =============================================================================
# A02: Cryptographic Failures Tests
# =============================================================================


class TestCryptographicFailures:
    """OWASP A02: Cryptographic Failures vulnerability tests."""

    def test_encryption_service_exists(self):
        """Verify encryption service exists."""
        encryption_file = SRC_ROOT / "security" / "encryption.py"
        assert encryption_file.exists(), "Encryption service not found"

    def test_encryption_uses_strong_algorithms(self):
        """Verify encryption uses AES or Fernet (strong algorithms)."""
        encryption_file = SRC_ROOT / "security" / "encryption.py"
        if not encryption_file.exists():
            pytest.skip("Encryption service not found")

        content = encryption_file.read_text()

        # Should use strong encryption
        strong_patterns = ["AES", "Fernet", "ChaCha20", "AESGCM", "cryptography"]
        has_strong = any(p in content for p in strong_patterns)

        assert has_strong, "Encryption service should use strong algorithms"

    def test_no_weak_algorithms_in_encryption(self):
        """Verify no weak algorithms in encryption service."""
        encryption_file = SRC_ROOT / "security" / "encryption.py"
        if not encryption_file.exists():
            pytest.skip("Encryption service not found")

        content = encryption_file.read_text().upper()

        # Should NOT use weak algorithms for encryption
        weak_patterns = ["DES.", "RC4", "BLOWFISH"]
        for weak in weak_patterns:
            assert weak not in content, f"Weak algorithm {weak} found in encryption"


# =============================================================================
# A03: Injection Tests
# =============================================================================


class TestInjection:
    """OWASP A03: Injection vulnerability tests."""

    def test_tenant_isolation_uses_parameterized_queries(self):
        """Verify tenant isolation uses parameterized SQL."""
        isolation_file = SRC_ROOT / "tenancy" / "isolation.py"
        if not isolation_file.exists():
            pytest.skip("Tenant isolation not found")

        content = isolation_file.read_text()

        # Should use parameterized queries
        assert "params" in content.lower() or "parameter" in content.lower(), (
            "Tenant isolation should use parameterized queries"
        )

    def test_no_dangerous_eval_with_user_input(self):
        """Verify no eval() with user input patterns."""
        dangerous_files = []

        for py_file in SRC_ROOT.rglob("*.py"):
            if should_skip_path(py_file) or is_test_file(py_file):
                continue

            content = py_file.read_text()

            # Look for dangerous eval patterns with request/user input
            if re.search(r"eval\s*\([^)]*(?:request|user_input|body)", content, re.I):
                dangerous_files.append(py_file.name)

        assert len(dangerous_files) == 0, f"Dangerous eval in: {dangerous_files}"


# =============================================================================
# A05: Security Misconfiguration Tests
# =============================================================================


class TestSecurityMisconfiguration:
    """OWASP A05: Security Misconfiguration vulnerability tests."""

    def test_production_guards_exist(self):
        """Verify production guards module exists."""
        guards_file = SRC_ROOT / "storage" / "production_guards.py"
        assert guards_file.exists(), "Production guards module not found"

    def test_environment_detection_exists(self):
        """Verify environment detection for production vs dev."""
        guards_file = SRC_ROOT / "storage" / "production_guards.py"
        if not guards_file.exists():
            pytest.skip("Production guards not found")

        content = guards_file.read_text()

        # Should detect production environment
        assert "production" in content.lower() or "ARAGORA_ENV" in content, (
            "Production environment detection missing"
        )


# =============================================================================
# A07: Authentication Failures Tests
# =============================================================================


class TestAuthenticationFailures:
    """OWASP A07: Identification and Authentication Failures tests."""

    def test_rate_limiting_infrastructure_exists(self):
        """Verify rate limiting exists."""
        resilience_file = SRC_ROOT / "resilience.py"
        assert resilience_file.exists(), "Resilience module not found"

        content = resilience_file.read_text()
        assert "rate" in content.lower() or "limit" in content.lower(), (
            "Rate limiting not found in resilience module"
        )

    def test_jwt_or_token_auth_exists(self):
        """Verify token-based authentication exists."""
        auth_files = list(SRC_ROOT.rglob("*auth*.py")) + list(SRC_ROOT.rglob("*jwt*.py"))

        found_token_auth = False
        for auth_file in auth_files:
            if should_skip_path(auth_file) or is_test_file(auth_file):
                continue

            content = auth_file.read_text()
            if "jwt" in content.lower() or "token" in content.lower():
                found_token_auth = True
                break

        assert found_token_auth, "Token-based authentication not found"


# =============================================================================
# A09: Security Logging Failures Tests
# =============================================================================


class TestSecurityLoggingFailures:
    """OWASP A09: Security Logging and Monitoring Failures tests."""

    def test_audit_module_exists(self):
        """Verify audit logging infrastructure exists."""
        audit_dir = SRC_ROOT / "audit"
        assert audit_dir.exists(), "Audit module not found"

    def test_audit_has_security_logging(self):
        """Verify audit module has security event logging."""
        audit_dir = SRC_ROOT / "audit"
        if not audit_dir.exists():
            pytest.skip("Audit module not found")

        # Look for security audit functions
        found_security_audit = False
        for py_file in audit_dir.rglob("*.py"):
            content = py_file.read_text()
            if "security" in content.lower() and "audit" in content.lower():
                found_security_audit = True
                break

        assert found_security_audit, "Security audit logging not found"


# =============================================================================
# Secrets Detection Tests
# =============================================================================


class TestSecretsDetection:
    """Tests to detect hardcoded secrets in the codebase."""

    def test_no_openai_keys_in_source(self):
        """Verify no OpenAI API keys in source files."""
        found_keys = []

        for py_file in SRC_ROOT.rglob("*.py"):
            if should_skip_path(py_file) or is_test_file(py_file):
                continue

            content = py_file.read_text()

            # OpenAI keys start with sk-
            if re.search(r"sk-[a-zA-Z0-9]{20,}", content):
                found_keys.append(py_file.name)

        assert len(found_keys) == 0, f"OpenAI keys found in: {found_keys}"

    def test_no_github_tokens_in_source(self):
        """Verify no GitHub tokens in source files."""
        found_tokens = []

        for py_file in SRC_ROOT.rglob("*.py"):
            if should_skip_path(py_file) or is_test_file(py_file):
                continue

            content = py_file.read_text()

            # GitHub tokens
            if re.search(r"gh[po]_[a-zA-Z0-9]{36,}", content):
                found_tokens.append(py_file.name)

        assert len(found_tokens) == 0, f"GitHub tokens found in: {found_tokens}"

    def test_no_private_keys_in_source(self):
        """Verify no actual private keys in source files."""
        found_keys = []

        # Files that legitimately reference key patterns (classifiers, audit types)
        allowed_files = {"classifier.py", "security.py", "software.py", "audit_types"}

        for py_file in SRC_ROOT.rglob("*.py"):
            if should_skip_path(py_file) or is_test_file(py_file):
                continue

            # Skip files that legitimately reference key patterns for detection
            if any(allowed in str(py_file) for allowed in allowed_files):
                continue

            content = py_file.read_text()

            # Look for actual private key content (not just pattern references)
            if "-----BEGIN" in content and "PRIVATE KEY-----" in content:
                # Check if it's a real key (has key content) vs pattern reference
                if re.search(r"-----BEGIN[^-]+PRIVATE KEY-----\n[A-Za-z0-9+/=\n]{50,}", content):
                    found_keys.append(py_file.name)

        assert len(found_keys) == 0, f"Private keys found in: {found_keys}"


# =============================================================================
# Tenant Isolation Tests
# =============================================================================


class TestTenantIsolation:
    """Tests for multi-tenant isolation."""

    def test_tenant_isolation_module_exists(self):
        """Verify tenant isolation module exists."""
        isolation_file = SRC_ROOT / "tenancy" / "isolation.py"
        assert isolation_file.exists(), "Tenant isolation module not found"

    def test_tenant_context_module_exists(self):
        """Verify tenant context module exists."""
        context_file = SRC_ROOT / "tenancy" / "context.py"
        assert context_file.exists(), "Tenant context module not found"

    def test_isolation_violation_exception_exists(self):
        """Verify isolation violation exception is defined."""
        isolation_file = SRC_ROOT / "tenancy" / "isolation.py"
        if not isolation_file.exists():
            pytest.skip("Tenant isolation not found")

        content = isolation_file.read_text()
        assert "IsolationViolation" in content, "IsolationViolation exception not found"


# =============================================================================
# Webhook Security Tests
# =============================================================================


class TestWebhookSecurity:
    """Tests for webhook security."""

    def test_webhook_security_module_exists(self):
        """Verify webhook security module exists."""
        webhook_file = SRC_ROOT / "connectors" / "chat" / "webhook_security.py"
        assert webhook_file.exists(), "Webhook security module not found"

    def test_webhook_verification_required_in_production(self):
        """Verify webhook verification is enforced in production."""
        webhook_file = SRC_ROOT / "connectors" / "chat" / "webhook_security.py"
        if not webhook_file.exists():
            pytest.skip("Webhook security module not found")

        content = webhook_file.read_text()

        # Should enforce verification in production
        assert "production" in content.lower(), (
            "Webhook security should enforce verification in production"
        )


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for secure error handling."""

    def test_error_handler_middleware_exists(self):
        """Verify error handler middleware exists."""
        error_file = SRC_ROOT / "server" / "middleware" / "error_handler.py"
        assert error_file.exists(), "Error handler middleware not found"

    def test_error_codes_module_exists(self):
        """Verify error codes are standardized."""
        error_codes_file = SRC_ROOT / "server" / "error_codes.py"
        assert error_codes_file.exists(), "Error codes module not found"


# =============================================================================
# Summary Report
# =============================================================================


class TestSecuritySummary:
    """Summary tests to verify overall security posture."""

    def test_security_module_structure(self):
        """Verify security module has proper structure."""
        security_dir = SRC_ROOT / "security"
        assert security_dir.exists(), "Security module not found"

        # Should have key security files
        expected_files = ["encryption.py"]
        for f in expected_files:
            assert (security_dir / f).exists(), f"Security file {f} not found"

    def test_minimum_security_coverage(self):
        """Verify minimum security infrastructure exists."""
        required_modules = [
            SRC_ROOT / "rbac",
            SRC_ROOT / "security",
            SRC_ROOT / "audit",
            SRC_ROOT / "tenancy",
        ]

        missing = [str(m) for m in required_modules if not m.exists()]
        assert len(missing) == 0, f"Missing security modules: {missing}"
