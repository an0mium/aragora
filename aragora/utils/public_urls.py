"""Helpers for building public Aragora web URLs."""

from __future__ import annotations

import os
from urllib.parse import urlencode


def public_receipt_url(receipt_id: str, *, base_url: str | None = None) -> str:
    """Build the canonical public receipt URL for external clients."""
    if not receipt_id:
        return ""

    resolved_base_url = (
        base_url or os.environ.get("ARAGORA_PUBLIC_URL", "https://aragora.ai")
    ).rstrip("/")
    return f"{resolved_base_url}/receipts?{urlencode({'id': receipt_id})}"
