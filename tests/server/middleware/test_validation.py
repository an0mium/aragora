"""
Tests for aragora.server.middleware.validation - Request validation middleware.

Tests cover:
- RouteValidation dataclass and matching
- ValidationConfig settings
- ValidationResult properties
- ValidationMiddleware.validate_request() for various scenarios
- Query parameter validation
- Body size validation
- Path segment validation
- Schema validation
- Metrics tracking
- Utility functions
"""

from __future__ import annotations

import pytest


# ===========================================================================
# Test RouteValidation
# ===========================================================================


class TestRouteValidation:
    """Tests for RouteValidation dataclass."""

    def test_pattern_compiled_from_string(self):
        """Pattern string should be compiled to regex."""
        from aragora.server.middleware.validation import RouteValidation

        rule = RouteValidation(r"^/api/debates$", "GET")
        assert hasattr(rule.pattern, "match")  # Compiled regex has match method

    def test_matches_correct_path_and_method(self):
        """Rule should match correct path and method."""
        from aragora.server.middleware.validation import RouteValidation

        rule = RouteValidation(r"^/api/debates$", "GET")
        assert rule.matches("/api/debates", "GET") is True
        assert rule.matches("/api/debates", "get") is True  # Case insensitive method

    def test_does_not_match_wrong_path(self):
        """Rule should not match wrong path."""
        from aragora.server.middleware.validation import RouteValidation

        rule = RouteValidation(r"^/api/debates$", "GET")
        assert rule.matches("/api/agents", "GET") is False
        assert rule.matches("/api/debates/123", "GET") is False

    def test_does_not_match_wrong_method(self):
        """Rule should not match wrong HTTP method."""
        from aragora.server.middleware.validation import RouteValidation

        rule = RouteValidation(r"^/api/debates$", "GET")
        assert rule.matches("/api/debates", "POST") is False
        assert rule.matches("/api/debates", "DELETE") is False

    def test_wildcard_method_matches_all(self):
        """Wildcard method should match all HTTP methods."""
        from aragora.server.middleware.validation import RouteValidation

        rule = RouteValidation(r"^/api/health$", "*")
        assert rule.matches("/api/health", "GET") is True
        assert rule.matches("/api/health", "POST") is True
        assert rule.matches("/api/health", "DELETE") is True

    def test_default_max_body_size(self):
        """Default max body size should be 1MB."""
        from aragora.server.middleware.validation import RouteValidation

        rule = RouteValidation(r"^/api/debates$", "POST")
        assert rule.max_body_size == 1_048_576


# ===========================================================================
# Test ValidationConfig
# ===========================================================================


class TestValidationConfig:
    """Tests for ValidationConfig dataclass."""

    def test_default_values(self):
        """Default config should be enabled, non-blocking."""
        from aragora.server.middleware.validation import ValidationConfig

        config = ValidationConfig()
        assert config.enabled is True
        assert config.blocking is False
        assert config.log_all is False
        assert config.max_body_size == 10_485_760  # 10MB

    def test_custom_values(self):
        """Custom config values should be respected."""
        from aragora.server.middleware.validation import ValidationConfig

        config = ValidationConfig(
            enabled=False,
            blocking=True,
            log_all=True,
            max_body_size=5_000_000,
        )
        assert config.enabled is False
        assert config.blocking is True
        assert config.log_all is True
        assert config.max_body_size == 5_000_000


# ===========================================================================
# Test ValidationResult
# ===========================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        """Valid result should have no errors."""
        from aragora.server.middleware.validation import ValidationResult

        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.errors == []
        assert result.error_message == ""

    def test_invalid_result_with_errors(self):
        """Invalid result should contain error messages."""
        from aragora.server.middleware.validation import ValidationResult

        result = ValidationResult(
            valid=False,
            errors=["Missing required parameter", "Body too large"],
        )
        assert result.valid is False
        assert len(result.errors) == 2
        assert "Missing required parameter" in result.error_message
        assert "Body too large" in result.error_message

    def test_error_message_joins_errors(self):
        """Error message should join multiple errors with semicolons."""
        from aragora.server.middleware.validation import ValidationResult

        result = ValidationResult(
            valid=False,
            errors=["Error 1", "Error 2", "Error 3"],
        )
        assert result.error_message == "Error 1; Error 2; Error 3"


# ===========================================================================
# Test ValidationMiddleware
# ===========================================================================


class TestValidationMiddleware:
    """Tests for ValidationMiddleware class."""

    def test_disabled_validation_returns_valid(self):
        """Disabled validation should always return valid."""
        from aragora.server.middleware.validation import (
            ValidationConfig,
            ValidationMiddleware,
        )

        config = ValidationConfig(enabled=False)
        middleware = ValidationMiddleware(config=config)

        result = middleware.validate_request("/api/debates", "GET")
        assert result.valid is True

    def test_no_matching_rule_returns_valid(self):
        """No matching rule should return valid (unvalidated route)."""
        from aragora.server.middleware.validation import (
            ValidationConfig,
            ValidationMiddleware,
        )

        config = ValidationConfig(enabled=True)
        middleware = ValidationMiddleware(config=config, registry=[])

        result = middleware.validate_request("/api/unknown", "GET")
        assert result.valid is True

    def test_tracks_unvalidated_routes(self):
        """Middleware should track routes without validation rules."""
        from aragora.server.middleware.validation import (
            ValidationConfig,
            ValidationMiddleware,
        )

        config = ValidationConfig(enabled=True)
        middleware = ValidationMiddleware(config=config, registry=[])

        middleware.validate_request("/api/unknown", "GET")
        middleware.validate_request("/api/other", "POST")

        unvalidated = middleware.get_unvalidated_routes()
        assert "GET /api/unknown" in unvalidated
        assert "POST /api/other" in unvalidated


class TestQueryParameterValidation:
    """Tests for query parameter validation."""

    def test_required_params_missing(self):
        """Missing required params should fail validation."""
        from aragora.server.middleware.validation import (
            RouteValidation,
            ValidationMiddleware,
        )

        rule = RouteValidation(
            r"^/api/search$",
            "GET",
            required_params=["query"],
        )
        middleware = ValidationMiddleware(registry=[rule])

        result = middleware.validate_request("/api/search", "GET", query_params={})
        assert result.valid is False
        assert "Missing required parameter: query" in result.errors

    def test_required_params_present(self):
        """Present required params should pass validation."""
        from aragora.server.middleware.validation import (
            RouteValidation,
            ValidationMiddleware,
        )

        rule = RouteValidation(
            r"^/api/search$",
            "GET",
            required_params=["query"],
        )
        middleware = ValidationMiddleware(registry=[rule])

        result = middleware.validate_request("/api/search", "GET", query_params={"query": "test"})
        assert result.valid is True

    def test_query_param_range_valid(self):
        """Query param within range should pass."""
        from aragora.server.middleware.validation import (
            RouteValidation,
            ValidationMiddleware,
        )

        rule = RouteValidation(
            r"^/api/debates$",
            "GET",
            query_rules={"limit": (1, 100)},
        )
        middleware = ValidationMiddleware(registry=[rule])

        result = middleware.validate_request("/api/debates", "GET", query_params={"limit": "50"})
        assert result.valid is True

    def test_query_param_range_below_min(self):
        """Query param below min should fail."""
        from aragora.server.middleware.validation import (
            RouteValidation,
            ValidationMiddleware,
        )

        rule = RouteValidation(
            r"^/api/debates$",
            "GET",
            query_rules={"limit": (1, 100)},
        )
        middleware = ValidationMiddleware(registry=[rule])

        result = middleware.validate_request("/api/debates", "GET", query_params={"limit": "0"})
        assert result.valid is False
        assert any("out of range" in err for err in result.errors)

    def test_query_param_range_above_max(self):
        """Query param above max should fail."""
        from aragora.server.middleware.validation import (
            RouteValidation,
            ValidationMiddleware,
        )

        rule = RouteValidation(
            r"^/api/debates$",
            "GET",
            query_rules={"limit": (1, 100)},
        )
        middleware = ValidationMiddleware(registry=[rule])

        result = middleware.validate_request("/api/debates", "GET", query_params={"limit": "500"})
        assert result.valid is False
        assert any("out of range" in err for err in result.errors)


class TestBodyValidation:
    """Tests for request body validation."""

    def test_body_size_within_limit(self):
        """Body within size limit should pass."""
        from aragora.server.middleware.validation import (
            RouteValidation,
            ValidationMiddleware,
        )

        rule = RouteValidation(
            r"^/api/debates$",
            "POST",
            max_body_size=1000,
        )
        middleware = ValidationMiddleware(registry=[rule])

        result = middleware.validate_request("/api/debates", "POST", body=b"x" * 500)
        assert result.valid is True

    def test_body_size_exceeds_limit(self):
        """Body exceeding size limit should fail."""
        from aragora.server.middleware.validation import (
            RouteValidation,
            ValidationMiddleware,
        )

        rule = RouteValidation(
            r"^/api/debates$",
            "POST",
            max_body_size=100,
        )
        middleware = ValidationMiddleware(registry=[rule])

        result = middleware.validate_request("/api/debates", "POST", body=b"x" * 500)
        assert result.valid is False
        assert any("body too large" in err.lower() for err in result.errors)


class TestPathValidation:
    """Tests for path segment validation."""

    def test_valid_path_segment(self):
        """Valid path segment should pass validation."""
        from aragora.server.middleware.validation import (
            RouteValidation,
            ValidationMiddleware,
        )

        def validate_id(s: str) -> tuple[bool, str]:
            if s.startswith("debate_"):
                return True, ""
            return False, "must start with debate_"

        rule = RouteValidation(
            r"^/api/debates/([^/]+)$",
            "GET",
            path_validators={"debate_id": validate_id},
        )
        middleware = ValidationMiddleware(registry=[rule])

        result = middleware.validate_request("/api/debates/debate_123", "GET")
        assert result.valid is True

    def test_invalid_path_segment(self):
        """Invalid path segment should fail validation."""
        from aragora.server.middleware.validation import (
            RouteValidation,
            ValidationMiddleware,
        )

        def validate_id(s: str) -> tuple[bool, str]:
            if s.startswith("debate_"):
                return True, ""
            return False, "must start with debate_"

        rule = RouteValidation(
            r"^/api/debates/([^/]+)$",
            "GET",
            path_validators={"debate_id": validate_id},
        )
        middleware = ValidationMiddleware(registry=[rule])

        result = middleware.validate_request("/api/debates/invalid_123", "GET")
        assert result.valid is False
        assert any("Invalid debate_id" in err for err in result.errors)


class TestMetrics:
    """Tests for validation metrics tracking."""

    def test_validation_count_increments(self):
        """Validation count should increment on each request."""
        from aragora.server.middleware.validation import (
            RouteValidation,
            ValidationMiddleware,
        )

        rule = RouteValidation(r"^/api/test$", "GET")
        middleware = ValidationMiddleware(registry=[rule])

        middleware.validate_request("/api/test", "GET")
        middleware.validate_request("/api/test", "GET")
        middleware.validate_request("/api/test", "GET")

        metrics = middleware.get_metrics()
        assert metrics["total_validations"] == 3

    def test_failure_count_tracks_failures(self):
        """Failure count should track failed validations."""
        from aragora.server.middleware.validation import (
            RouteValidation,
            ValidationMiddleware,
        )

        rule = RouteValidation(
            r"^/api/test$",
            "GET",
            required_params=["required"],
        )
        middleware = ValidationMiddleware(registry=[rule])

        middleware.validate_request("/api/test", "GET", query_params={})  # Fail
        middleware.validate_request("/api/test", "GET", query_params={"required": "v"})  # Pass
        middleware.validate_request("/api/test", "GET", query_params={})  # Fail

        metrics = middleware.get_metrics()
        assert metrics["failures"] == 2
        assert metrics["failure_rate"] == pytest.approx(2 / 3)


# ===========================================================================
# Test Utility Functions
# ===========================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_validation_middleware_defaults(self):
        """create_validation_middleware should create with defaults."""
        from aragora.server.middleware.validation import create_validation_middleware

        middleware = create_validation_middleware()
        assert middleware.config.enabled is True
        assert middleware.config.blocking is False

    def test_create_validation_middleware_blocking(self):
        """create_validation_middleware should respect blocking param."""
        from aragora.server.middleware.validation import create_validation_middleware

        middleware = create_validation_middleware(blocking=True)
        assert middleware.config.blocking is True

    def test_add_route_validation(self):
        """add_route_validation should add to global registry."""
        from aragora.server.middleware.validation import (
            VALIDATION_REGISTRY,
            add_route_validation,
        )

        initial_count = len(VALIDATION_REGISTRY)

        add_route_validation(
            pattern=r"^/api/test/unique/endpoint$",
            method="POST",
            required_params=["data"],
        )

        assert len(VALIDATION_REGISTRY) == initial_count + 1

        # Clean up
        VALIDATION_REGISTRY.pop()


# ===========================================================================
# Test Registry Patterns
# ===========================================================================


class TestTier1Schemas:
    """Tests for Tier 1 validation schemas (auth, org, billing)."""

    def test_auth_register_schema_registered(self):
        """POST /api/auth/register should have body schema."""
        from aragora.server.middleware.validation import VALIDATION_REGISTRY

        matches = [r for r in VALIDATION_REGISTRY if r.matches("/api/auth/register", "POST")]
        assert len(matches) >= 1
        rule = matches[0]
        assert rule.body_schema is not None
        assert rule.max_body_size == 10_000

    def test_auth_login_schema_registered(self):
        """POST /api/auth/login should have body schema."""
        from aragora.server.middleware.validation import VALIDATION_REGISTRY

        matches = [r for r in VALIDATION_REGISTRY if r.matches("/api/auth/login", "POST")]
        assert len(matches) >= 1
        assert matches[0].body_schema is not None

    def test_auth_mfa_verify_schema_registered(self):
        """POST /api/auth/mfa/verify should have body schema."""
        from aragora.server.middleware.validation import VALIDATION_REGISTRY

        matches = [r for r in VALIDATION_REGISTRY if r.matches("/api/auth/mfa/verify", "POST")]
        assert len(matches) >= 1
        assert matches[0].body_schema is not None

    def test_org_invite_schema_registered(self):
        """POST /api/org/{id}/invite should have body schema."""
        from aragora.server.middleware.validation import VALIDATION_REGISTRY

        matches = [r for r in VALIDATION_REGISTRY if r.matches("/api/org/test-org/invite", "POST")]
        assert len(matches) >= 1
        assert matches[0].body_schema is not None

    def test_billing_checkout_schema_registered(self):
        """POST /api/billing/checkout should have body schema."""
        from aragora.server.middleware.validation import VALIDATION_REGISTRY

        matches = [r for r in VALIDATION_REGISTRY if r.matches("/api/billing/checkout", "POST")]
        assert len(matches) >= 1
        assert matches[0].body_schema is not None

    def test_billing_portal_schema_registered(self):
        """POST /api/billing/portal should have body schema."""
        from aragora.server.middleware.validation import VALIDATION_REGISTRY

        matches = [r for r in VALIDATION_REGISTRY if r.matches("/api/billing/portal", "POST")]
        assert len(matches) >= 1
        assert matches[0].body_schema is not None

    def test_v1_routes_also_match(self):
        """v1 routes should also match the patterns."""
        from aragora.server.middleware.validation import VALIDATION_REGISTRY

        # Check v1 auth routes
        matches = [r for r in VALIDATION_REGISTRY if r.matches("/api/v1/auth/login", "POST")]
        assert len(matches) >= 1

        # Check v1 billing routes
        matches = [r for r in VALIDATION_REGISTRY if r.matches("/api/v1/billing/checkout", "POST")]
        assert len(matches) >= 1


class TestValidationRegistry:
    """Tests for built-in validation registry."""

    def test_debates_post_registered(self):
        """POST /api/debates should be registered."""
        from aragora.server.middleware.validation import VALIDATION_REGISTRY

        matches = [r for r in VALIDATION_REGISTRY if r.matches("/api/debates", "POST")]
        assert len(matches) >= 1

    def test_debates_get_has_limit_offset_rules(self):
        """GET /api/debates should have limit/offset rules."""
        from aragora.server.middleware.validation import VALIDATION_REGISTRY

        matches = [r for r in VALIDATION_REGISTRY if r.matches("/api/debates", "GET")]
        assert len(matches) >= 1
        rule = matches[0]
        assert "limit" in rule.query_rules
        assert "offset" in rule.query_rules

    def test_debate_by_id_has_path_validator(self):
        """GET /api/debates/{id} should have path validator."""
        from aragora.server.middleware.validation import VALIDATION_REGISTRY

        matches = [r for r in VALIDATION_REGISTRY if r.matches("/api/debates/test-123", "GET")]
        assert len(matches) >= 1
        rule = matches[0]
        assert "debate_id" in rule.path_validators
