"""Compatibility shim for compliance handler imports."""

from __future__ import annotations

from aragora.server.handlers.compliance.handler import *  # noqa: F401,F403  # type: ignore[assignment]
from aragora.storage.receipt_store import get_receipt_store
from aragora.storage.audit_store import get_audit_store

__all__ = [*globals().get("__all__", []), "get_receipt_store", "get_audit_store"]
