"""
Billing core helpers - Shared validation, utility, and idempotency functions.

Extracted from core.py for use by core.py and its mixin submodules
(core_webhooks.py, core_reporting.py).
"""

from __future__ import annotations

import re
import sys
from datetime import datetime



# --- Input validation helpers for financial operations ---

# ISO date pattern: YYYY-MM-DD
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Maximum number of usage events returned in an export to prevent DoS
_MAX_EXPORT_ROWS = 10_000


def _validate_iso_date(value: str | None) -> str | None:
    """Validate and return an ISO date string (YYYY-MM-DD), or None if invalid."""
    if value is None:
        return None
    if not isinstance(value, str) or not _ISO_DATE_RE.match(value):
        return None
    # Verify it actually parses as a real date
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
    return value


def _safe_positive_int(value: str, default: int, maximum: int) -> int:
    """Parse a string to a bounded non-negative integer."""
    try:
        parsed = int(value)
    except (ValueError, TypeError):
        return default
    if parsed < 0:
        return default
    return min(parsed, maximum)


# --- Admin billing compatibility helpers ---


def _get_admin_billing_callable(name: str, fallback):
    """Resolve a callable from admin.billing for test patching."""
    admin_billing = sys.modules.get("aragora.server.handlers.admin.billing")
    if admin_billing is not None:
        candidate = getattr(admin_billing, name, None)
        if callable(candidate) and candidate is not fallback:
            return candidate
    return fallback


# --- Webhook idempotency tracking ---
# Uses aragora.storage.webhook_store for persistence across restarts


def _is_duplicate_webhook(event_id: str) -> bool:
    """Check if webhook event was already processed."""
    from aragora.storage.webhook_store import get_webhook_store

    store = get_webhook_store()
    return store.is_processed(event_id)


def _mark_webhook_processed(event_id: str, result: str = "success") -> None:
    """Mark webhook event as processed."""
    from aragora.storage.webhook_store import get_webhook_store

    store = get_webhook_store()
    store.mark_processed(event_id, result)
