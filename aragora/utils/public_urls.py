"""Helpers for building public Aragora web URLs."""

from __future__ import annotations

import os
from urllib.parse import urlencode


def resolve_public_base_url(*, base_url: str | None = None) -> str:
    """Resolve the canonical public Aragora base URL."""
    return (
        base_url
        or os.environ.get("ARAGORA_PUBLIC_URL")
        or os.environ.get("ARAGORA_BASE_URL")
        or "https://aragora.ai"
    ).rstrip("/")


def public_receipt_url(receipt_id: str, *, base_url: str | None = None) -> str:
    """Build the canonical public receipt URL for external clients."""
    if not receipt_id:
        return ""

    resolved_base_url = resolve_public_base_url(base_url=base_url)
    return f"{resolved_base_url}/receipts?{urlencode({'id': receipt_id})}"


def public_receipt_share_url(token: str, *, base_url: str | None = None) -> str:
    """Build the canonical public share-token URL for a receipt."""
    if not token:
        return ""

    resolved_base_url = resolve_public_base_url(base_url=base_url)
    return f"{resolved_base_url}/api/v2/receipts/share/{token}"
