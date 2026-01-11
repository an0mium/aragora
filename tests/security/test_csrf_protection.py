"""
CSRF protection security tests.

Verifies that CSRF protections are properly implemented for
state-changing operations.
"""

import hmac
import hashlib
import secrets
import time
import pytest
from unittest.mock import Mock, patch


# =============================================================================
# CSRF Token Generation Tests
# =============================================================================


class TestCSRFTokenGeneration:
    """Test CSRF token generation security."""

    def test_token_has_sufficient_entropy(self):
        """CSRF tokens should have at least 128 bits of entropy."""
        def generate_csrf_token() -> str:
            return secrets.token_urlsafe(32)  # 256 bits

        token = generate_csrf_token()

        # Should be at least 43 chars (32 bytes base64url)
        assert len(token) >= 43

        # Should be unique
        tokens = {generate_csrf_token() for _ in range(100)}
        assert len(tokens) == 100

    def test_tokens_are_unpredictable(self):
        """CSRF tokens should not be predictable."""
        def generate_csrf_token() -> str:
            return secrets.token_urlsafe(32)

        tokens = [generate_csrf_token() for _ in range(10)]

        # No sequential patterns
        for i in range(len(tokens) - 1):
            # Tokens should have minimal character overlap
            common = sum(1 for a, b in zip(tokens[i], tokens[i + 1]) if a == b)
            assert common < len(tokens[i]) / 2  # Less than 50% overlap

    def test_token_bound_to_session(self):
        """CSRF tokens should be bound to session."""
        def generate_session_csrf(session_id: str, secret: bytes) -> str:
            data = f"{session_id}:{secrets.token_urlsafe(16)}"
            sig = hmac.new(secret, data.encode(), hashlib.sha256).hexdigest()
            return f"{data}.{sig}"

        def verify_session_csrf(token: str, session_id: str, secret: bytes) -> bool:
            parts = token.rsplit(".", 1)
            if len(parts) != 2:
                return False
            data, provided_sig = parts
            if not data.startswith(f"{session_id}:"):
                return False
            expected_sig = hmac.new(secret, data.encode(), hashlib.sha256).hexdigest()
            return hmac.compare_digest(provided_sig, expected_sig)

        secret = b"server-secret-key"
        session1 = "session-abc123"
        session2 = "session-xyz789"

        token1 = generate_session_csrf(session1, secret)

        # Token should verify for correct session
        assert verify_session_csrf(token1, session1, secret) is True

        # Token should NOT verify for different session
        assert verify_session_csrf(token1, session2, secret) is False

    def test_token_has_expiry(self):
        """CSRF tokens should expire after reasonable time."""
        def generate_csrf_with_expiry(secret: bytes, ttl: int = 3600) -> str:
            expiry = int(time.time()) + ttl
            data = f"{expiry}:{secrets.token_urlsafe(16)}"
            sig = hmac.new(secret, data.encode(), hashlib.sha256).hexdigest()
            return f"{data}.{sig}"

        def verify_csrf_with_expiry(token: str, secret: bytes) -> bool:
            parts = token.rsplit(".", 1)
            if len(parts) != 2:
                return False
            data, provided_sig = parts

            # Check signature
            expected_sig = hmac.new(secret, data.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(provided_sig, expected_sig):
                return False

            # Check expiry
            try:
                expiry_str = data.split(":")[0]
                expiry = int(expiry_str)
                if time.time() > expiry:
                    return False
            except (ValueError, IndexError):
                return False

            return True

        secret = b"secret-key"

        # Valid token (1 hour TTL)
        valid_token = generate_csrf_with_expiry(secret, ttl=3600)
        assert verify_csrf_with_expiry(valid_token, secret) is True

        # Expired token (negative TTL for testing)
        expired_token = generate_csrf_with_expiry(secret, ttl=-1)
        assert verify_csrf_with_expiry(expired_token, secret) is False


# =============================================================================
# CSRF Token Validation Tests
# =============================================================================


class TestCSRFTokenValidation:
    """Test CSRF token validation."""

    def test_rejects_missing_token(self):
        """Requests without CSRF token should be rejected."""
        def validate_request(headers: dict) -> bool:
            csrf_token = headers.get("X-CSRF-Token")
            if not csrf_token:
                return False
            return True

        assert validate_request({"X-CSRF-Token": "valid-token"}) is True
        assert validate_request({}) is False
        assert validate_request({"X-CSRF-Token": ""}) is False

    def test_rejects_invalid_token(self):
        """Invalid CSRF tokens should be rejected."""
        valid_tokens = {"token-abc123", "token-xyz789"}

        def validate_csrf(token: str) -> bool:
            return token in valid_tokens

        assert validate_csrf("token-abc123") is True
        assert validate_csrf("token-invalid") is False
        assert validate_csrf("") is False

    def test_rejects_reused_token(self):
        """CSRF tokens should not be reusable (one-time use)."""
        used_tokens = set()

        def validate_and_consume(token: str, valid_tokens: set) -> bool:
            if token not in valid_tokens:
                return False
            if token in used_tokens:
                return False  # Already used
            used_tokens.add(token)
            return True

        valid_tokens = {"token-abc123"}

        # First use should succeed
        assert validate_and_consume("token-abc123", valid_tokens) is True

        # Second use should fail (replay attack)
        assert validate_and_consume("token-abc123", valid_tokens) is False

    def test_token_comparison_is_constant_time(self):
        """Token comparison should be constant-time to prevent timing attacks."""
        import hmac

        valid_token = "a" * 64

        def secure_compare(a: str, b: str) -> bool:
            return hmac.compare_digest(a, b)

        # Use constant-time comparison
        assert secure_compare(valid_token, valid_token) is True
        assert secure_compare(valid_token, "b" * 64) is False
        assert secure_compare(valid_token, valid_token[:-1]) is False


# =============================================================================
# State-Changing Operation Protection Tests
# =============================================================================


class TestStateChangingOperations:
    """Test that state-changing operations require CSRF protection."""

    def test_post_requests_require_csrf(self):
        """POST requests should require CSRF token."""
        def check_csrf_required(method: str, path: str) -> bool:
            # Safe methods don't need CSRF
            if method.upper() in ["GET", "HEAD", "OPTIONS"]:
                return False
            # State-changing methods need CSRF
            return True

        assert check_csrf_required("POST", "/api/debates") is True
        assert check_csrf_required("PUT", "/api/debates/123") is True
        assert check_csrf_required("DELETE", "/api/debates/123") is True
        assert check_csrf_required("PATCH", "/api/debates/123") is True
        assert check_csrf_required("GET", "/api/debates") is False
        assert check_csrf_required("HEAD", "/api/debates") is False

    def test_api_endpoints_protected(self):
        """Critical API endpoints should have CSRF protection."""
        CSRF_PROTECTED_ENDPOINTS = {
            "/api/debates",  # Create debate
            "/api/gauntlet",  # Run gauntlet
            "/api/settings",  # Update settings
            "/api/user/profile",  # Update profile
            "/api/org/members",  # Manage members
            "/api/billing",  # Billing changes
        }

        def is_csrf_protected(path: str) -> bool:
            for protected in CSRF_PROTECTED_ENDPOINTS:
                if path.startswith(protected):
                    return True
            return False

        assert is_csrf_protected("/api/debates") is True
        assert is_csrf_protected("/api/debates/123") is True
        assert is_csrf_protected("/api/gauntlet/run") is True
        assert is_csrf_protected("/api/billing/subscribe") is True
        assert is_csrf_protected("/api/health") is False

    def test_csrf_exempt_endpoints(self):
        """Some endpoints can be CSRF exempt (e.g., webhooks with signatures)."""
        CSRF_EXEMPT = {
            "/api/webhooks/stripe",  # Has own signature verification
            "/api/webhooks/github",
            "/api/auth/login",  # Pre-session
            "/api/auth/register",
        }

        def needs_csrf(path: str) -> bool:
            return path not in CSRF_EXEMPT

        assert needs_csrf("/api/webhooks/stripe") is False
        assert needs_csrf("/api/auth/login") is False
        assert needs_csrf("/api/debates") is True


# =============================================================================
# Double Submit Cookie Tests
# =============================================================================


class TestDoubleSubmitCookie:
    """Test double submit cookie pattern."""

    def test_cookie_matches_header(self):
        """CSRF cookie value should match header value."""
        def validate_double_submit(cookie_value: str, header_value: str) -> bool:
            if not cookie_value or not header_value:
                return False
            return hmac.compare_digest(cookie_value, header_value)

        # Matching values
        assert validate_double_submit("token123", "token123") is True

        # Mismatched values
        assert validate_double_submit("token123", "different") is False

        # Missing values
        assert validate_double_submit("", "token123") is False
        assert validate_double_submit("token123", "") is False

    def test_cookie_attributes_secure(self):
        """CSRF cookie should have secure attributes."""
        class MockCookie:
            def __init__(
                self,
                httponly: bool = True,
                secure: bool = True,
                samesite: str = "Strict",
                path: str = "/",
            ):
                self.httponly = httponly
                self.secure = secure
                self.samesite = samesite
                self.path = path

        def validate_cookie_security(cookie: MockCookie) -> list[str]:
            issues = []
            # HttpOnly should be False for CSRF (JavaScript needs to read it)
            # But if using double-submit, the token needs to be readable
            if cookie.secure is False:
                issues.append("Cookie not Secure")
            if cookie.samesite not in ["Strict", "Lax"]:
                issues.append("SameSite not set properly")
            return issues

        # Secure cookie
        secure_cookie = MockCookie(httponly=False, secure=True, samesite="Strict")
        assert len(validate_cookie_security(secure_cookie)) == 0

        # Insecure cookie
        insecure_cookie = MockCookie(secure=False, samesite="None")
        issues = validate_cookie_security(insecure_cookie)
        assert len(issues) == 2


# =============================================================================
# Origin/Referer Validation Tests
# =============================================================================


class TestOriginRefererValidation:
    """Test Origin/Referer header validation as additional CSRF defense."""

    def test_validates_origin_header(self):
        """Origin header should match allowed origins."""
        ALLOWED_ORIGINS = {
            "https://aragora.ai",
            "https://app.aragora.ai",
        }

        def validate_origin(origin: str) -> bool:
            return origin in ALLOWED_ORIGINS

        assert validate_origin("https://aragora.ai") is True
        assert validate_origin("https://app.aragora.ai") is True
        assert validate_origin("https://evil.com") is False
        assert validate_origin("https://aragora.ai.evil.com") is False

    def test_validates_referer_header(self):
        """Referer header should be from allowed origins."""
        ALLOWED_ORIGIN = "https://aragora.ai"

        def validate_referer(referer: str) -> bool:
            if not referer:
                return False
            from urllib.parse import urlparse

            parsed = urlparse(referer)
            origin = f"{parsed.scheme}://{parsed.netloc}"
            return origin == ALLOWED_ORIGIN

        assert validate_referer("https://aragora.ai/debates/123") is True
        assert validate_referer("https://aragora.ai/") is True
        assert validate_referer("https://evil.com/page") is False
        assert validate_referer("") is False

    def test_null_origin_rejected(self):
        """Null origin (privacy proxies) should be handled carefully."""
        def validate_origin_strict(origin: str) -> bool:
            if origin == "null" or not origin:
                return False  # Reject null origin
            # Continue with normal validation
            return origin.startswith("https://aragora.ai")

        assert validate_origin_strict("https://aragora.ai") is True
        assert validate_origin_strict("null") is False
        assert validate_origin_strict("") is False


# =============================================================================
# CSRF in SPA Context Tests
# =============================================================================


class TestCSRFInSPA:
    """Test CSRF protection in Single Page Application context."""

    def test_ajax_requests_include_csrf(self):
        """AJAX requests should include CSRF token in header."""
        def make_request_headers(csrf_token: str) -> dict:
            return {
                "Content-Type": "application/json",
                "X-CSRF-Token": csrf_token,
                "X-Requested-With": "XMLHttpRequest",
            }

        headers = make_request_headers("token123")
        assert headers["X-CSRF-Token"] == "token123"
        assert headers["X-Requested-With"] == "XMLHttpRequest"

    def test_fetch_credentials_same_origin(self):
        """Fetch requests should use credentials: 'same-origin'."""
        # This is a client-side concern, but we can test the expectation
        SAFE_CREDENTIALS_MODES = {"same-origin", "include"}

        def is_safe_credentials(mode: str) -> bool:
            return mode in SAFE_CREDENTIALS_MODES

        assert is_safe_credentials("same-origin") is True
        assert is_safe_credentials("include") is True  # For cross-origin with CSRF
        assert is_safe_credentials("omit") is False  # No cookies sent

    def test_api_rejects_cross_origin_without_cors(self):
        """API should reject cross-origin requests without proper CORS."""
        def check_cors_preflight(
            origin: str, method: str, allowed_origins: set
        ) -> dict:
            if origin not in allowed_origins:
                return {"status": 403, "headers": {}}

            return {
                "status": 200,
                "headers": {
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE",
                    "Access-Control-Allow-Headers": "X-CSRF-Token, Content-Type",
                    "Access-Control-Allow-Credentials": "true",
                },
            }

        allowed = {"https://aragora.ai"}

        # Allowed origin
        response = check_cors_preflight("https://aragora.ai", "POST", allowed)
        assert response["status"] == 200
        assert "X-CSRF-Token" in response["headers"]["Access-Control-Allow-Headers"]

        # Disallowed origin
        response = check_cors_preflight("https://evil.com", "POST", allowed)
        assert response["status"] == 403


# =============================================================================
# CSRF Token Per-Form Tests
# =============================================================================


class TestPerFormCSRFToken:
    """Test per-form CSRF tokens (more secure than session-wide)."""

    def test_token_tied_to_action(self):
        """CSRF token can be tied to specific action."""
        def generate_action_token(action: str, secret: bytes) -> str:
            data = f"{action}:{secrets.token_urlsafe(16)}"
            sig = hmac.new(secret, data.encode(), hashlib.sha256).hexdigest()
            return f"{data}.{sig}"

        def verify_action_token(
            token: str, expected_action: str, secret: bytes
        ) -> bool:
            parts = token.rsplit(".", 1)
            if len(parts) != 2:
                return False
            data, provided_sig = parts

            # Check action matches
            if not data.startswith(f"{expected_action}:"):
                return False

            # Check signature
            expected_sig = hmac.new(secret, data.encode(), hashlib.sha256).hexdigest()
            return hmac.compare_digest(provided_sig, expected_sig)

        secret = b"secret-key"

        # Generate token for "create_debate"
        token = generate_action_token("create_debate", secret)

        # Should verify for correct action
        assert verify_action_token(token, "create_debate", secret) is True

        # Should reject for different action
        assert verify_action_token(token, "delete_debate", secret) is False

    def test_token_tied_to_resource(self):
        """CSRF token can be tied to specific resource."""
        def generate_resource_token(
            resource_type: str, resource_id: str, secret: bytes
        ) -> str:
            data = f"{resource_type}:{resource_id}:{secrets.token_urlsafe(16)}"
            sig = hmac.new(secret, data.encode(), hashlib.sha256).hexdigest()
            return f"{data}.{sig}"

        def verify_resource_token(
            token: str,
            expected_type: str,
            expected_id: str,
            secret: bytes,
        ) -> bool:
            parts = token.rsplit(".", 1)
            if len(parts) != 2:
                return False
            data, provided_sig = parts

            expected_prefix = f"{expected_type}:{expected_id}:"
            if not data.startswith(expected_prefix):
                return False

            expected_sig = hmac.new(secret, data.encode(), hashlib.sha256).hexdigest()
            return hmac.compare_digest(provided_sig, expected_sig)

        secret = b"secret-key"

        token = generate_resource_token("debate", "123", secret)

        # Should verify for correct resource
        assert verify_resource_token(token, "debate", "123", secret) is True

        # Should reject for different resource
        assert verify_resource_token(token, "debate", "456", secret) is False
        assert verify_resource_token(token, "gauntlet", "123", secret) is False
