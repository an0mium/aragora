"""Helpers for canonical receipt links exposed to humans."""

from __future__ import annotations

import os
from urllib.parse import urlencode


def build_public_receipt_url(receipt_id: str, base_url: str | None = None) -> str:
    """Return the browser-facing receipt route used by the live frontend."""
    resolved_base = (
        base_url or os.environ.get("ARAGORA_PUBLIC_URL") or "https://aragora.ai"
    ).rstrip("/")
    return f"{resolved_base}/receipts?{urlencode({'id': receipt_id})}"


__all__ = ["build_public_receipt_url"]
