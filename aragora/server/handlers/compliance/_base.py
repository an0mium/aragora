"""
Base utilities for compliance handlers.

Shared imports, helper functions, and common utilities used across
compliance handler modules.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import rate_limit
from aragora.rbac.decorators import PermissionDeniedError, require_permission
from aragora.observability.metrics import track_handler
from aragora.storage.audit_store import get_audit_store
from aragora.storage.receipt_store import get_receipt_store
from aragora.privacy.deletion import get_deletion_scheduler, get_legal_hold_manager
from aragora.deletion_coordinator import get_deletion_coordinator

logger = logging.getLogger(__name__)


def extract_user_id_from_headers(headers: dict[str, str] | None) -> str:
    """
    Extract user ID from Authorization header.

    Falls back to 'compliance_api' if no valid auth is present.
    This ensures audit trails identify the actual user making compliance requests.
    """
    if not headers:
        return "compliance_api"

    auth_header = headers.get("Authorization", "") or headers.get("authorization", "")
    if not auth_header or not auth_header.startswith("Bearer "):
        return "compliance_api"

    token = auth_header[7:]

    # Check if it's an API key (ara_xxx format)
    if token.startswith("ara_"):
        # API keys don't contain user info directly, use key prefix as identifier
        return f"api_key:{token[:12]}..."

    # Try to decode JWT to get user_id
    try:
        from aragora.billing.auth.tokens import validate_access_token

        payload = validate_access_token(token)
        if payload and payload.user_id:
            return payload.user_id
    except (ImportError, ValueError, AttributeError):
        pass

    return "compliance_api"


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse timestamp from string (ISO date or unix timestamp)."""
    if not value:
        return None

    try:
        ts = float(value)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except ValueError:
        pass

    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt
    except (ValueError, AttributeError):
        pass

    return None


__all__ = [
    # Re-exported from base
    "BaseHandler",
    "HandlerResult",
    "error_response",
    "json_response",
    "rate_limit",
    "PermissionDeniedError",
    "require_permission",
    "track_handler",
    # Storage
    "get_audit_store",
    "get_receipt_store",
    # Privacy/deletion
    "get_deletion_scheduler",
    "get_legal_hold_manager",
    "get_deletion_coordinator",
    # Standard library
    "hashlib",
    "html",
    "json",
    "logging",
    "datetime",
    "timezone",
    "timedelta",
    "Any",
    "Optional",
    # Local utilities
    "logger",
    "extract_user_id_from_headers",
    "parse_timestamp",
]
