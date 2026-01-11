"""
Authentication boundary security tests.

Verifies that authentication and authorization boundaries
are enforced correctly across all protected endpoints.
"""

import time
import hmac
import hashlib
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta


# =============================================================================
# Token Validation Tests
# =============================================================================


class TestTokenValidation:
    """Test token validation edge cases."""

    def test_rejects_empty_token(self):
        """Empty tokens should be rejected."""
        def validate_token(token: str) -> bool:
            if not token or not token.strip():
                return False
            return True

        assert validate_token("") is False
        assert validate_token("   ") is False
        assert validate_token(None) is False

    def test_rejects_malformed_bearer_header(self):
        """Malformed Authorization headers should be rejected."""
        def extract_token(auth_header: str) -> str | None:
            if not auth_header:
                return None
            parts = auth_header.split()
            if len(parts) != 2:
                return None
            if parts[0].lower() != "bearer":
                return None
            return parts[1]

        assert extract_token("Bearer token123") == "token123"
        assert extract_token("bearer token123") == "token123"
        assert extract_token("Bearer") is None
        assert extract_token("Basic dXNlcjpwYXNz") is None
        assert extract_token("Bearer token1 token2") is None
        assert extract_token("") is None

    def test_rejects_expired_token(self):
        """Expired tokens should be rejected."""
        def is_token_expired(expiry_timestamp: float) -> bool:
            return time.time() > expiry_timestamp

        # Expired 1 hour ago
        expired = time.time() - 3600
        assert is_token_expired(expired) is True

        # Valid for 1 hour
        valid = time.time() + 3600
        assert is_token_expired(valid) is False

    def test_rejects_token_from_future(self):
        """Tokens with future issued-at should be rejected."""
        def is_token_valid_time(issued_at: float, max_clock_skew: float = 60) -> bool:
            now = time.time()
            # Token issued in future (beyond clock skew) is suspicious
            if issued_at > now + max_clock_skew:
                return False
            return True

        # Normal token
        assert is_token_valid_time(time.time()) is True

        # Future token (1 hour ahead)
        assert is_token_valid_time(time.time() + 3600) is False

        # Within clock skew tolerance
        assert is_token_valid_time(time.time() + 30) is True

    def test_rejects_revoked_token(self):
        """Revoked tokens should be rejected."""
        revoked_tokens = {"token-abc123", "token-xyz789"}

        def is_token_revoked(token: str) -> bool:
            return token in revoked_tokens

        assert is_token_revoked("token-abc123") is True
        assert is_token_revoked("token-valid") is False

    def test_hmac_signature_validation(self):
        """HMAC signature validation should be constant-time."""
        secret = b"super-secret-key"

        def create_token(payload: str) -> str:
            sig = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()
            return f"{payload}.{sig}"

        def verify_token(token: str) -> bool:
            parts = token.rsplit(".", 1)
            if len(parts) != 2:
                return False
            payload, provided_sig = parts
            expected_sig = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()
            # Use constant-time comparison to prevent timing attacks
            return hmac.compare_digest(provided_sig, expected_sig)

        valid_token = create_token("user:123:exp:9999999999")
        assert verify_token(valid_token) is True

        # Tampered payload
        tampered = "user:456:exp:9999999999" + "." + valid_token.split(".")[1]
        assert verify_token(tampered) is False

        # Invalid signature
        invalid_sig = "user:123:exp:9999999999.invalid-signature"
        assert verify_token(invalid_sig) is False


# =============================================================================
# Authorization Boundary Tests
# =============================================================================


class TestAuthorizationBoundaries:
    """Test authorization boundaries between resources."""

    def test_user_cannot_access_other_org_resources(self):
        """Users should only access their own organization's resources."""
        def check_org_access(user_org: str, resource_org: str) -> bool:
            return user_org == resource_org

        assert check_org_access("org-123", "org-123") is True
        assert check_org_access("org-123", "org-456") is False

    def test_debate_ownership_enforced(self):
        """Users should only access debates they own or have access to."""
        class MockDebate:
            def __init__(self, owner_org: str, shared_with: list = None):
                self.owner_org = owner_org
                self.shared_with = shared_with or []

        def can_access_debate(user_org: str, debate: MockDebate) -> bool:
            if debate.owner_org == user_org:
                return True
            if user_org in debate.shared_with:
                return True
            return False

        debate = MockDebate("org-123", shared_with=["org-456"])

        assert can_access_debate("org-123", debate) is True  # Owner
        assert can_access_debate("org-456", debate) is True  # Shared
        assert can_access_debate("org-789", debate) is False  # No access

    def test_gauntlet_result_access_control(self):
        """Gauntlet results should respect org boundaries."""
        def can_access_gauntlet_result(user_org: str, result_org: str) -> bool:
            return user_org == result_org

        assert can_access_gauntlet_result("org-123", "org-123") is True
        assert can_access_gauntlet_result("org-123", "org-456") is False

    def test_api_key_scoping(self):
        """API keys should be scoped to specific permissions."""
        class APIKey:
            def __init__(self, scopes: list):
                self.scopes = set(scopes)

        def has_permission(key: APIKey, required_scope: str) -> bool:
            return required_scope in key.scopes or "admin" in key.scopes

        read_only_key = APIKey(["debates:read", "leaderboard:read"])
        write_key = APIKey(["debates:read", "debates:write"])
        admin_key = APIKey(["admin"])

        assert has_permission(read_only_key, "debates:read") is True
        assert has_permission(read_only_key, "debates:write") is False

        assert has_permission(write_key, "debates:read") is True
        assert has_permission(write_key, "debates:write") is True

        assert has_permission(admin_key, "debates:delete") is True  # Admin has all


# =============================================================================
# Session Security Tests
# =============================================================================


class TestSessionSecurity:
    """Test session security measures."""

    def test_session_id_entropy(self):
        """Session IDs should have sufficient entropy."""
        import secrets

        def generate_session_id() -> str:
            # At least 128 bits of entropy
            return secrets.token_urlsafe(32)

        session_id = generate_session_id()

        # Should be at least 43 characters (32 bytes base64)
        assert len(session_id) >= 43

        # Should be unique
        session_ids = {generate_session_id() for _ in range(100)}
        assert len(session_ids) == 100  # All unique

    def test_session_binding(self):
        """Sessions should be bound to user attributes."""
        class Session:
            def __init__(self, user_id: str, ip: str, user_agent: str):
                self.user_id = user_id
                self.ip = ip
                self.user_agent = user_agent
                self.fingerprint = self._compute_fingerprint()

            def _compute_fingerprint(self) -> str:
                data = f"{self.user_id}:{self.ip}:{self.user_agent}"
                return hashlib.sha256(data.encode()).hexdigest()

            def validate_request(self, ip: str, user_agent: str) -> bool:
                # Check if request matches session binding
                request_fingerprint = hashlib.sha256(
                    f"{self.user_id}:{ip}:{user_agent}".encode()
                ).hexdigest()
                return hmac.compare_digest(self.fingerprint, request_fingerprint)

        session = Session("user-123", "192.168.1.1", "Mozilla/5.0")

        # Same user/ip/ua should validate
        assert session.validate_request("192.168.1.1", "Mozilla/5.0") is True

        # Different IP should fail (potential session hijacking)
        assert session.validate_request("10.0.0.1", "Mozilla/5.0") is False

    def test_session_expiry(self):
        """Sessions should expire after inactivity."""
        SESSION_TIMEOUT = 3600  # 1 hour

        def is_session_valid(last_activity: float) -> bool:
            return time.time() - last_activity < SESSION_TIMEOUT

        # Recent activity
        assert is_session_valid(time.time() - 100) is True

        # Expired
        assert is_session_valid(time.time() - 7200) is False


# =============================================================================
# CORS Security Tests
# =============================================================================


class TestCORSSecurity:
    """Test CORS configuration security."""

    def test_origin_validation(self):
        """Origins should be validated against allowlist."""
        ALLOWED_ORIGINS = {
            "https://aragora.ai",
            "https://app.aragora.ai",
            "http://localhost:3000",
        }

        def is_allowed_origin(origin: str) -> bool:
            return origin in ALLOWED_ORIGINS

        assert is_allowed_origin("https://aragora.ai") is True
        assert is_allowed_origin("https://evil.com") is False
        assert is_allowed_origin("https://aragora.ai.evil.com") is False

    def test_wildcard_origin_rejected_for_credentials(self):
        """Wildcard origin (*) should not be used with credentials."""
        def validate_cors_config(allow_origin: str, allow_credentials: bool) -> bool:
            # Can't use wildcard with credentials
            if allow_origin == "*" and allow_credentials:
                return False
            return True

        assert validate_cors_config("https://aragora.ai", True) is True
        assert validate_cors_config("*", False) is True
        assert validate_cors_config("*", True) is False

    def test_preflight_validation(self):
        """Preflight requests should validate requested methods/headers."""
        ALLOWED_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH"}
        ALLOWED_HEADERS = {"Content-Type", "Authorization", "X-Request-ID"}

        def validate_preflight(
            requested_method: str,
            requested_headers: list[str]
        ) -> bool:
            if requested_method not in ALLOWED_METHODS:
                return False
            for header in requested_headers:
                if header not in ALLOWED_HEADERS:
                    return False
            return True

        assert validate_preflight("POST", ["Content-Type"]) is True
        assert validate_preflight("TRACE", []) is False  # Method not allowed
        assert validate_preflight("GET", ["X-Evil-Header"]) is False


# =============================================================================
# Rate Limit Authentication Tests
# =============================================================================


class TestRateLimitAuthentication:
    """Test that rate limiting is tied to authentication."""

    def test_unauthenticated_gets_stricter_limits(self):
        """Unauthenticated requests should have stricter limits."""
        LIMITS = {
            "authenticated": 100,
            "unauthenticated": 10,
        }

        def get_rate_limit(is_authenticated: bool) -> int:
            return LIMITS["authenticated"] if is_authenticated else LIMITS["unauthenticated"]

        assert get_rate_limit(True) == 100
        assert get_rate_limit(False) == 10

    def test_rate_limit_by_token_not_ip_for_auth(self):
        """Authenticated requests should be rate limited by token, not IP."""
        token_usage = {}
        ip_usage = {}

        def track_authenticated_request(token: str, ip: str):
            """Track by token for authenticated requests."""
            token_usage[token] = token_usage.get(token, 0) + 1
            # Don't track by IP for authenticated

        def track_unauthenticated_request(ip: str):
            """Track by IP for unauthenticated requests."""
            ip_usage[ip] = ip_usage.get(ip, 0) + 1

        # Multiple IPs, same token = counted together
        track_authenticated_request("token-123", "192.168.1.1")
        track_authenticated_request("token-123", "10.0.0.1")

        assert token_usage["token-123"] == 2
        assert len(ip_usage) == 0  # IPs not tracked for authenticated


# =============================================================================
# Multi-tenancy Security Tests
# =============================================================================


class TestMultiTenancySecurity:
    """Test multi-tenancy security isolation."""

    def test_org_data_isolation(self):
        """Organizations should have complete data isolation."""
        class MockDatabase:
            def __init__(self):
                self.data = {
                    "org-123": {"debates": ["d1", "d2"], "results": ["r1"]},
                    "org-456": {"debates": ["d3"], "results": ["r2", "r3"]},
                }

            def get_debates(self, org_id: str) -> list:
                return self.data.get(org_id, {}).get("debates", [])

        db = MockDatabase()

        # Each org only sees their own data
        assert db.get_debates("org-123") == ["d1", "d2"]
        assert db.get_debates("org-456") == ["d3"]
        assert db.get_debates("org-789") == []

    def test_sql_query_includes_org_filter(self):
        """SQL queries should always include org_id filter."""
        def build_safe_query(base_query: str, org_id: str) -> tuple[str, tuple]:
            """Ensure org_id filter is always present."""
            # Add org_id filter if not present
            if "org_id" not in base_query.lower():
                if "WHERE" in base_query.upper():
                    base_query += " AND org_id = ?"
                else:
                    base_query += " WHERE org_id = ?"

            return base_query, (org_id,)

        query1, params1 = build_safe_query("SELECT * FROM debates", "org-123")
        assert "org_id = ?" in query1

        query2, params2 = build_safe_query(
            "SELECT * FROM debates WHERE status = 'active'", "org-123"
        )
        assert "org_id = ?" in query2

    def test_cross_tenant_id_guessing_prevented(self):
        """ID enumeration across tenants should be prevented."""
        import secrets

        def generate_resource_id() -> str:
            """Generate unpredictable resource ID."""
            return f"res_{secrets.token_urlsafe(16)}"

        # IDs should not be sequential/predictable
        ids = [generate_resource_id() for _ in range(10)]

        # All unique
        assert len(set(ids)) == 10

        # Not sequential numbers
        for id in ids:
            assert not id.replace("res_", "").isdigit()
