"""
Request validation middleware.

Provides centralized validation for API requests based on route patterns.
Initially operates in warning mode (logs but doesn't block) to identify
handlers that need validation, then can be promoted to blocking mode.

Usage:
    1. Register schemas in VALIDATION_REGISTRY
    2. Add ValidationMiddleware to server middleware stack
    3. Validation errors are logged and optionally blocked
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple

from aragora.server.validation.schema import (
    BILLING_PORTAL_SCHEMA,
    CHECKOUT_SESSION_SCHEMA,
    MEMBER_ROLE_SCHEMA,
    MFA_CODE_SCHEMA,
    MFA_DISABLE_SCHEMA,
    MFA_VERIFY_SCHEMA,
    ORG_INVITE_SCHEMA,
    ORG_SWITCH_SCHEMA,
    ORG_UPDATE_SCHEMA,
    PASSWORD_CHANGE_SCHEMA,
    TOKEN_REFRESH_SCHEMA,
    TOKEN_REVOKE_SCHEMA,
    USER_LOGIN_SCHEMA,
    USER_REGISTER_SCHEMA,
    USER_UPDATE_SCHEMA,
    validate_against_schema,
)
from aragora.server.validation.entities import (
    validate_agent_name,
    validate_debate_id,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Configuration
# =============================================================================


@dataclass
class RouteValidation:
    """Validation rules for a route pattern.

    Attributes:
        pattern: Regex pattern matching the route
        method: HTTP method (GET, POST, etc.) or * for all
        body_schema: Schema for POST/PUT body validation
        query_rules: Dict of param_name -> (min, max) for numeric params
        required_params: List of required query parameter names
        path_validators: Dict of path segment name -> validator function
        max_body_size: Maximum body size in bytes (default 1MB)
    """

    pattern: str | Pattern[str]
    method: str
    body_schema: Optional[dict] = None
    query_rules: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)
    path_validators: Dict[str, Callable[[str], Tuple[bool, str]]] = field(default_factory=dict)
    max_body_size: int = 1_048_576  # 1MB default

    def __post_init__(self) -> None:
        if isinstance(self.pattern, str):
            self.pattern = re.compile(self.pattern)

    def matches(self, path: str, method: str) -> bool:
        """Check if this rule matches the request."""
        if self.method != "*" and self.method.upper() != method.upper():
            return False
        # Pattern is always compiled in __post_init__
        pattern: Pattern[str] = self.pattern  # type: ignore[assignment]
        return bool(pattern.match(path))


# =============================================================================
# Validation Registry
# =============================================================================

# Common query parameter rules
LIMIT_OFFSET_RULES = {"limit": (1, 100), "offset": (0, 100000)}
PAGINATION_RULES = {"page": (1, 10000), "per_page": (1, 100)}

# Route validation registry - add schemas here as they're created
VALIDATION_REGISTRY: List[RouteValidation] = [
    # =========================================================================
    # Debates - core functionality
    # =========================================================================
    RouteValidation(
        r"^/api/(v1/)?debates?$",
        "POST",
        required_params=[],
        query_rules={},
    ),
    RouteValidation(
        r"^/api/(v1/)?debates?$",
        "GET",
        query_rules=LIMIT_OFFSET_RULES,
    ),
    RouteValidation(
        r"^/api/(v1/)?debates?/([^/]+)$",
        "GET",
        path_validators={"debate_id": validate_debate_id},
    ),
    # =========================================================================
    # Agents
    # =========================================================================
    RouteValidation(
        r"^/api/(v1/)?agents?$",
        "GET",
        query_rules=LIMIT_OFFSET_RULES,
    ),
    RouteValidation(
        r"^/api/(v1/)?agents?/([^/]+)$",
        "GET",
        path_validators={"agent_name": validate_agent_name},
    ),
    # =========================================================================
    # Auth - Tier 1 (Security Critical)
    # =========================================================================
    RouteValidation(
        r"^/api/(v1/)?auth/register$",
        "POST",
        body_schema=USER_REGISTER_SCHEMA,
        max_body_size=10_000,  # 10KB for registration
    ),
    RouteValidation(
        r"^/api/(v1/)?auth/login$",
        "POST",
        body_schema=USER_LOGIN_SCHEMA,
        max_body_size=5_000,  # 5KB for login
    ),
    RouteValidation(
        r"^/api/(v1/)?auth/password$",
        "POST",
        body_schema=PASSWORD_CHANGE_SCHEMA,
        max_body_size=5_000,
    ),
    RouteValidation(
        r"^/api/(v1/)?auth/refresh$",
        "POST",
        body_schema=TOKEN_REFRESH_SCHEMA,
        max_body_size=10_000,
    ),
    RouteValidation(
        r"^/api/(v1/)?auth/revoke$",
        "POST",
        body_schema=TOKEN_REVOKE_SCHEMA,
        max_body_size=10_000,
    ),
    RouteValidation(
        r"^/api/(v1/)?auth/me$",
        "PUT",
        body_schema=USER_UPDATE_SCHEMA,
        max_body_size=5_000,
    ),
    # MFA endpoints
    RouteValidation(
        r"^/api/(v1/)?auth/mfa/enable$",
        "POST",
        body_schema=MFA_CODE_SCHEMA,
        max_body_size=1_000,
    ),
    RouteValidation(
        r"^/api/(v1/)?auth/mfa/disable$",
        "POST",
        body_schema=MFA_DISABLE_SCHEMA,
        max_body_size=1_000,
    ),
    RouteValidation(
        r"^/api/(v1/)?auth/mfa/verify$",
        "POST",
        body_schema=MFA_VERIFY_SCHEMA,
        max_body_size=5_000,
    ),
    RouteValidation(
        r"^/api/(v1/)?auth/mfa/backup-codes$",
        "POST",
        body_schema=MFA_CODE_SCHEMA,
        max_body_size=1_000,
    ),
    # =========================================================================
    # Organizations - Tier 1 (Security Critical)
    # =========================================================================
    RouteValidation(
        r"^/api/(v1/)?org/([^/]+)$",
        "PUT",
        body_schema=ORG_UPDATE_SCHEMA,
        max_body_size=100_000,  # Allow for settings object
    ),
    RouteValidation(
        r"^/api/(v1/)?org/([^/]+)/invite$",
        "POST",
        body_schema=ORG_INVITE_SCHEMA,
        max_body_size=5_000,
    ),
    RouteValidation(
        r"^/api/(v1/)?org/([^/]+)/members/([^/]+)/role$",
        "PUT",
        body_schema=MEMBER_ROLE_SCHEMA,
        max_body_size=1_000,
    ),
    RouteValidation(
        r"^/api/(v1/)?user/organizations/switch$",
        "POST",
        body_schema=ORG_SWITCH_SCHEMA,
        max_body_size=1_000,
    ),
    RouteValidation(
        r"^/api/(v1/)?user/organizations/default$",
        "POST",
        body_schema=ORG_SWITCH_SCHEMA,
        max_body_size=1_000,
    ),
    # =========================================================================
    # Billing - Tier 1 (Financial Operations)
    # =========================================================================
    RouteValidation(
        r"^/api/(v1/)?billing/checkout$",
        "POST",
        body_schema=CHECKOUT_SESSION_SCHEMA,
        max_body_size=10_000,
    ),
    RouteValidation(
        r"^/api/(v1/)?billing/portal$",
        "POST",
        body_schema=BILLING_PORTAL_SCHEMA,
        max_body_size=5_000,
    ),
    # Cancel/resume don't require body validation (auth context only)
    RouteValidation(
        r"^/api/(v1/)?billing/cancel$",
        "POST",
        max_body_size=1_000,
    ),
    RouteValidation(
        r"^/api/(v1/)?billing/resume$",
        "POST",
        max_body_size=1_000,
    ),
]


# =============================================================================
# Validation Middleware
# =============================================================================


@dataclass
class ValidationConfig:
    """Configuration for validation middleware.

    Attributes:
        enabled: Whether validation is enabled
        blocking: If True, block invalid requests; if False, just log warnings
        log_all: Log all validation attempts, not just failures
        max_body_size: Default max body size (can be overridden per route)
    """

    enabled: bool = True
    blocking: bool = False  # Start in warning mode
    log_all: bool = False
    max_body_size: int = 10_485_760  # 10MB default


@dataclass
class ValidationResult:
    """Result of request validation."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def error_message(self) -> str:
        return "; ".join(self.errors) if self.errors else ""


class ValidationMiddleware:
    """Middleware for validating API requests.

    Validates requests based on registered route patterns and schemas.
    Can operate in warning mode (log only) or blocking mode.
    """

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        registry: Optional[List[RouteValidation]] = None,
    ) -> None:
        self.config = config or ValidationConfig()
        self.registry = registry or VALIDATION_REGISTRY

        # Metrics
        self._validation_count = 0
        self._validation_failures = 0
        self._unvalidated_routes: set[str] = set()

    def validate_request(
        self,
        path: str,
        method: str,
        query_params: Optional[Dict[str, Any]] = None,
        body: Optional[bytes] = None,
        body_parsed: Optional[dict] = None,
    ) -> ValidationResult:
        """Validate an incoming request.

        Args:
            path: Request path
            method: HTTP method
            query_params: Query parameters
            body: Raw request body (for size check)
            body_parsed: Parsed JSON body (for schema validation)

        Returns:
            ValidationResult with validation status and any errors
        """
        if not self.config.enabled:
            return ValidationResult(valid=True)

        self._validation_count += 1
        result = ValidationResult(valid=True)
        query_params = query_params or {}

        # Find matching validation rule
        rule = self._find_rule(path, method)
        if rule is None:
            # No validation rule - track for coverage metrics
            route_key = f"{method} {path.split('?')[0]}"
            if route_key not in self._unvalidated_routes:
                self._unvalidated_routes.add(route_key)
                if self.config.log_all:
                    logger.debug(f"No validation rule for: {route_key}")
            return result

        # Validate body size
        if body is not None and len(body) > rule.max_body_size:
            result.valid = False
            result.errors.append(
                f"Request body too large: {len(body)} bytes (max {rule.max_body_size})"
            )

        # Validate required params
        for param in rule.required_params:
            if param not in query_params or not query_params[param]:
                result.valid = False
                result.errors.append(f"Missing required parameter: {param}")

        # Validate query param ranges
        for param, (min_val, max_val) in rule.query_rules.items():
            if param in query_params:
                try:
                    val = int(query_params[param])
                    if val < min_val or val > max_val:
                        result.valid = False
                        result.errors.append(
                            f"Parameter '{param}' out of range: {val} (allowed {min_val}-{max_val})"
                        )
                except (ValueError, TypeError):
                    result.warnings.append(f"Parameter '{param}' is not a valid integer")

        # Validate path segments
        for name, validator in rule.path_validators.items():
            segment = self._extract_path_segment(path, name)
            if segment:
                is_valid, err = validator(segment)
                if not is_valid:
                    result.valid = False
                    result.errors.append(f"Invalid {name}: {err}")

        # Validate body schema
        if rule.body_schema and body_parsed is not None:
            schema_result = validate_against_schema(body_parsed, rule.body_schema)
            if not schema_result.is_valid:
                result.valid = False
                result.errors.append(f"Body validation failed: {schema_result.error}")

        # Log result
        if not result.valid:
            self._validation_failures += 1
            log_level = logging.WARNING if not self.config.blocking else logging.ERROR
            logger.log(
                log_level,
                f"Validation failed for {method} {path}: {result.error_message}",
            )
        elif self.config.log_all:
            logger.debug(f"Validation passed for {method} {path}")

        return result

    def _find_rule(self, path: str, method: str) -> Optional[RouteValidation]:
        """Find the first matching validation rule."""
        for rule in self.registry:
            if rule.matches(path, method):
                return rule
        return None

    def _extract_path_segment(self, path: str, name: str) -> Optional[str]:
        """Extract a named segment from the path."""
        parts = path.strip("/").split("/")

        # Common segment positions
        segment_positions = {
            "debate_id": 2,  # /api/debates/{id}
            "agent_name": 2,  # /api/agents/{name}
            "receipt_id": 2,  # /api/receipts/{id}
            "workflow_id": 2,  # /api/workflows/{id}
        }

        pos = segment_positions.get(name)
        if pos is not None and len(parts) > pos:
            return parts[pos]

        return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get validation metrics."""
        return {
            "total_validations": self._validation_count,
            "failures": self._validation_failures,
            "failure_rate": (
                self._validation_failures / self._validation_count
                if self._validation_count > 0
                else 0
            ),
            "unvalidated_route_count": len(self._unvalidated_routes),
            "blocking_mode": self.config.blocking,
        }

    def get_unvalidated_routes(self) -> List[str]:
        """Get list of routes that have no validation rules."""
        return sorted(self._unvalidated_routes)


# =============================================================================
# Utility Functions
# =============================================================================


def create_validation_middleware(
    blocking: bool = False,
    enabled: bool = True,
) -> ValidationMiddleware:
    """Create a validation middleware with standard configuration.

    Args:
        blocking: If True, return errors for invalid requests
        enabled: If False, skip all validation

    Returns:
        Configured ValidationMiddleware instance
    """
    config = ValidationConfig(
        enabled=enabled,
        blocking=blocking,
        log_all=False,
    )
    return ValidationMiddleware(config=config)


def add_route_validation(
    pattern: str,
    method: str,
    body_schema: Optional[dict] = None,
    query_rules: Optional[Dict[str, Tuple[int, int]]] = None,
    required_params: Optional[List[str]] = None,
) -> None:
    """Add a validation rule to the global registry.

    Args:
        pattern: Regex pattern for the route
        method: HTTP method
        body_schema: Schema for body validation
        query_rules: Rules for query parameter ranges
        required_params: Required query parameters
    """
    rule = RouteValidation(
        pattern=pattern,
        method=method,
        body_schema=body_schema,
        query_rules=query_rules or {},
        required_params=required_params or [],
    )
    VALIDATION_REGISTRY.append(rule)
    logger.info(f"Added validation rule: {method} {pattern}")
