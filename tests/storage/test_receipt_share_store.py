from __future__ import annotations

import tempfile
from pathlib import Path

from aragora.storage.receipt_share_store import ReceiptShareStore


def _store() -> ReceiptShareStore:
    tmpdir = tempfile.TemporaryDirectory()
    store = ReceiptShareStore(Path(tmpdir.name) / "receipt_shares.db")
    store._tmpdir = tmpdir  # keep tempdir alive for the test duration
    return store


def test_consume_access_increments_and_returns_updated_row() -> None:
    store = _store()
    store.save(
        token="token-1",
        receipt_id="receipt-1",
        expires_at=4102444800.0,
        max_accesses=3,
    )

    consumed = store.consume_access("token-1", now=1700000000.0)

    assert consumed is not None
    assert consumed["receipt_id"] == "receipt-1"
    assert consumed["access_count"] == 1
    assert store.get_by_token("token-1")["access_count"] == 1


def test_consume_access_respects_max_accesses() -> None:
    store = _store()
    store.save(
        token="limited-token",
        receipt_id="receipt-2",
        expires_at=4102444800.0,
        max_accesses=1,
    )

    first = store.consume_access("limited-token", now=1700000000.0)
    second = store.consume_access("limited-token", now=1700000001.0)

    assert first is not None
    assert first["access_count"] == 1
    assert second is None
    assert store.get_by_token("limited-token")["access_count"] == 1


def test_consume_access_rejects_expired_tokens() -> None:
    store = _store()
    store.save(
        token="expired-token",
        receipt_id="receipt-3",
        expires_at=100.0,
        max_accesses=None,
    )

    consumed = store.consume_access("expired-token", now=101.0)

    assert consumed is None
    assert store.get_by_token("expired-token")["access_count"] == 0
