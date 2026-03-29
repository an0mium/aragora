"""Unified facade over gauntlet and storage receipt stores.

Phase 3 of Decision Integrity Kernel: converges the two receipt stores
so callers get state-machine semantics (gauntlet) and durable persistence
(storage) through a single interface.

Neither store is replaced -- this is a thin delegation layer.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class ReceiptStoreFacade:
    """Thin facade that delegates to gauntlet store (state machine) and
    storage store (durable persistence).

    The gauntlet store is authoritative for lifecycle state (CREATED ->
    APPROVED -> EXECUTED | EXPIRED).  The storage store provides durable
    SQLite/PostgreSQL persistence with signature verification, date-range
    queries, and retention policies.
    """

    def persist_and_save(
        self,
        receipt_id: str,
        receipt_data: dict[str, Any],
        *,
        signature: str | None = None,
        signature_key_id: str | None = None,
        signed_at: str | None = None,
        signature_algorithm: str | None = None,
        state: str = "CREATED",
    ) -> None:
        """Write to both stores.  Gauntlet for state machine, storage for durability."""

        from aragora.gauntlet.receipt_store import ReceiptState, get_receipt_store

        gauntlet = get_receipt_store()
        gauntlet.persist(
            receipt_id,
            receipt_data,
            signature=signature,
            signature_key_id=signature_key_id,
            signed_at=signed_at,
            signature_algorithm=signature_algorithm,
            state=ReceiptState.normalize(state),
        )

        # Best-effort write to durable storage store
        try:
            from aragora.storage.receipt_store import get_receipt_store as get_storage_store

            storage = get_storage_store()
            signed_receipt_data = None
            if signature:
                signed_receipt_data = {
                    "signature": signature,
                    "signature_metadata": {
                        "algorithm": signature_algorithm or "",
                        "key_id": signature_key_id or "",
                        "timestamp": signed_at or "",
                    },
                }
            storage.save(receipt_data, signed_receipt=signed_receipt_data)
        except Exception as exc:  # noqa: BLE001 - durable store is best-effort only
            logger.debug("Storage store write skipped for %s: %s", receipt_id, exc)

    def get_canonical(self, receipt_id: str) -> dict[str, Any] | None:
        """Read from gauntlet store first (state), fall back to storage store."""

        from aragora.gauntlet.receipt_store import get_receipt_store

        gauntlet = get_receipt_store()
        stored = gauntlet.get(receipt_id)
        if stored is not None:
            return stored.to_dict()

        # Fall back to storage store
        try:
            from aragora.storage.receipt_store import get_receipt_store as get_storage_store

            storage = get_storage_store()
            durable = storage.get(receipt_id)
            if durable is not None:
                return durable.to_dict()
        except (ImportError, RuntimeError, OSError, ValueError) as exc:
            logger.debug("Storage store read failed for %s: %s", receipt_id, exc)

        return None

    def transition(self, receipt_id: str, new_state: str) -> None:
        """Transition in gauntlet store, log in storage store."""

        from aragora.gauntlet.receipt_store import ReceiptState, get_receipt_store

        gauntlet = get_receipt_store()
        gauntlet.transition(receipt_id, ReceiptState.normalize(new_state))

    def verify(self, receipt_id: str) -> bool:
        """Verify signature via gauntlet store's verify_receipt."""

        from aragora.gauntlet.receipt_store import get_receipt_store

        gauntlet = get_receipt_store()
        return gauntlet.verify_receipt(receipt_id)


# -- module singleton -------------------------------------------------------

_facade_singleton: ReceiptStoreFacade | None = None
_facade_lock = threading.Lock()


def get_receipt_store_facade() -> ReceiptStoreFacade:
    """Get or create the module-level ReceiptStoreFacade singleton."""
    global _facade_singleton
    if _facade_singleton is None:
        with _facade_lock:
            if _facade_singleton is None:
                _facade_singleton = ReceiptStoreFacade()
    return _facade_singleton


def reset_receipt_store_facade() -> None:
    """Reset the singleton (for testing)."""
    global _facade_singleton
    _facade_singleton = None


__all__ = [
    "ReceiptStoreFacade",
    "get_receipt_store_facade",
    "reset_receipt_store_facade",
]
