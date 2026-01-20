"""
Gauntlet Error Codes and Exceptions.

Provides standardized error codes for the Gauntlet API to enable:
- Machine-parseable error responses
- Consistent error handling across clients
- Clear documentation of failure modes
- Integration with monitoring/alerting systems

Error Code Format: GAUNTLET_XXX
- 1XX: Input validation errors
- 2XX: Authentication/authorization errors
- 3XX: Resource errors (not found, quota exceeded)
- 4XX: Execution errors (runtime failures)
- 5XX: System errors (infrastructure failures)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class GauntletErrorCode(str, Enum):
    """Standardized error codes for Gauntlet API responses."""

    # 1XX: Input Validation Errors
    INVALID_INPUT = "GAUNTLET_100"
    INPUT_TOO_LARGE = "GAUNTLET_101"
    INVALID_INPUT_TYPE = "GAUNTLET_102"
    INVALID_PERSONA = "GAUNTLET_103"
    INVALID_AGENTS = "GAUNTLET_104"
    INVALID_PROFILE = "GAUNTLET_105"
    MISSING_REQUIRED_FIELD = "GAUNTLET_106"
    INVALID_FORMAT = "GAUNTLET_107"

    # 2XX: Authentication/Authorization Errors
    NOT_AUTHENTICATED = "GAUNTLET_200"
    INSUFFICIENT_PERMISSIONS = "GAUNTLET_201"
    TOKEN_EXPIRED = "GAUNTLET_202"
    INVALID_API_KEY = "GAUNTLET_203"
    RBAC_DENIED = "GAUNTLET_204"

    # 3XX: Resource Errors
    GAUNTLET_NOT_FOUND = "GAUNTLET_300"
    RECEIPT_NOT_FOUND = "GAUNTLET_301"
    PERSONA_NOT_FOUND = "GAUNTLET_302"
    QUOTA_EXCEEDED = "GAUNTLET_303"
    RATE_LIMITED = "GAUNTLET_304"
    RESOURCE_LOCKED = "GAUNTLET_305"

    # 4XX: Execution Errors
    GAUNTLET_FAILED = "GAUNTLET_400"
    AGENT_UNAVAILABLE = "GAUNTLET_401"
    EXECUTION_TIMEOUT = "GAUNTLET_402"
    CONSENSUS_FAILED = "GAUNTLET_403"
    VERIFICATION_FAILED = "GAUNTLET_404"
    INCOMPLETE_RESULT = "GAUNTLET_405"
    NOT_COMPLETED = "GAUNTLET_406"

    # 5XX: System Errors
    INTERNAL_ERROR = "GAUNTLET_500"
    STORAGE_ERROR = "GAUNTLET_501"
    SERVICE_UNAVAILABLE = "GAUNTLET_502"
    CONFIGURATION_ERROR = "GAUNTLET_503"
    SIGNING_ERROR = "GAUNTLET_504"


@dataclass
class GauntletError:
    """Structured error response for Gauntlet API."""

    code: GauntletErrorCode
    message: str
    details: Optional[dict[str, Any]] = None
    http_status: int = 400

    def to_dict(self) -> dict[str, Any]:
        """Convert to API response dict."""
        result: dict[str, Any] = {
            "error": True,
            "code": self.code.value,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result


# Pre-defined error instances for common cases
ERRORS = {
    "invalid_input": GauntletError(
        code=GauntletErrorCode.INVALID_INPUT,
        message="Invalid input content",
        http_status=400,
    ),
    "input_too_large": GauntletError(
        code=GauntletErrorCode.INPUT_TOO_LARGE,
        message="Input exceeds maximum size limit",
        details={"max_size_kb": 1024},
        http_status=413,
    ),
    "invalid_input_type": GauntletError(
        code=GauntletErrorCode.INVALID_INPUT_TYPE,
        message="Invalid input type",
        details={"valid_types": ["spec", "architecture", "policy", "code", "strategy", "contract"]},
        http_status=400,
    ),
    "invalid_persona": GauntletError(
        code=GauntletErrorCode.INVALID_PERSONA,
        message="Invalid or unknown persona",
        http_status=400,
    ),
    "not_authenticated": GauntletError(
        code=GauntletErrorCode.NOT_AUTHENTICATED,
        message="Authentication required",
        http_status=401,
    ),
    "insufficient_permissions": GauntletError(
        code=GauntletErrorCode.INSUFFICIENT_PERMISSIONS,
        message="Insufficient permissions for this operation",
        http_status=403,
    ),
    "gauntlet_not_found": GauntletError(
        code=GauntletErrorCode.GAUNTLET_NOT_FOUND,
        message="Gauntlet run not found",
        http_status=404,
    ),
    "receipt_not_found": GauntletError(
        code=GauntletErrorCode.RECEIPT_NOT_FOUND,
        message="Receipt not found for this gauntlet run",
        http_status=404,
    ),
    "quota_exceeded": GauntletError(
        code=GauntletErrorCode.QUOTA_EXCEEDED,
        message="Monthly gauntlet quota exceeded",
        http_status=429,
    ),
    "rate_limited": GauntletError(
        code=GauntletErrorCode.RATE_LIMITED,
        message="Rate limit exceeded, please retry later",
        http_status=429,
    ),
    "not_completed": GauntletError(
        code=GauntletErrorCode.NOT_COMPLETED,
        message="Gauntlet run not yet completed",
        http_status=400,
    ),
    "execution_timeout": GauntletError(
        code=GauntletErrorCode.EXECUTION_TIMEOUT,
        message="Gauntlet execution timed out",
        http_status=408,
    ),
    "internal_error": GauntletError(
        code=GauntletErrorCode.INTERNAL_ERROR,
        message="Internal server error",
        http_status=500,
    ),
    "storage_error": GauntletError(
        code=GauntletErrorCode.STORAGE_ERROR,
        message="Failed to access storage",
        http_status=500,
    ),
}


def gauntlet_error_response(
    error_key: str,
    details: Optional[dict[str, Any]] = None,
    message_override: Optional[str] = None,
) -> tuple[dict[str, Any], int]:
    """
    Create a standardized error response.

    Args:
        error_key: Key in ERRORS dict
        details: Additional details to include
        message_override: Custom message to use instead of default

    Returns:
        Tuple of (response_dict, http_status_code)

    Example:
        >>> body, status = gauntlet_error_response("gauntlet_not_found", {"id": "gauntlet-123"})
        >>> # Returns: ({"error": True, "code": "GAUNTLET_300", ...}, 404)
    """
    base_error = ERRORS.get(error_key, ERRORS["internal_error"])

    response: dict[str, Any] = {
        "error": True,
        "code": base_error.code.value,
        "message": message_override or base_error.message,
    }

    # Merge details
    merged_details = {}
    if base_error.details:
        merged_details.update(base_error.details)
    if details:
        merged_details.update(details)
    if merged_details:
        response["details"] = merged_details

    return response, base_error.http_status
