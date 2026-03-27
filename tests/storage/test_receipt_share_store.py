"""Tests for receipt share token persistence and access consumption."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

from aragora.storage.receipt_share_store import ReceiptShareStore


def test_consume_access_returns_updated_count(tmp_path) -> None:
    """Consuming a share token should atomically return the incremented count."""
    store = ReceiptShareStore(tmp_path / "receipt_shares.db")
    store.save(
        token="share-token",
        receipt_id="receipt-123",
        expires_at=time.time() + 3600,
        max_accesses=2,
    )

    result = store.consume_access("share-token")

    assert result["status"] == "ok"
    assert result["share_info"]["access_count"] == 1
    assert store.get_by_token("share-token")["access_count"] == 1


def test_consume_access_is_atomic_under_race(tmp_path) -> None:
    """Only one concurrent access may consume a single-use share token."""
    store = ReceiptShareStore(tmp_path / "receipt_shares.db")
    store.save(
        token="limited-token",
        receipt_id="receipt-123",
        expires_at=time.time() + 3600,
        max_accesses=1,
    )

    def consume() -> dict[str, object]:
        return store.consume_access("limited-token")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(consume)
        second = executor.submit(consume)
        results = [first.result(), second.result()]

    assert sorted(result["status"] for result in results) == ["limit_reached", "ok"]
    assert store.get_by_token("limited-token")["access_count"] == 1


def test_consume_access_does_not_increment_expired_token(tmp_path) -> None:
    """Expired tokens should be rejected without incrementing access_count."""
    store = ReceiptShareStore(tmp_path / "receipt_shares.db")
    store.save(
        token="expired-token",
        receipt_id="receipt-123",
        expires_at=time.time() - 60,
        max_accesses=2,
    )

    result = store.consume_access("expired-token")

    assert result["status"] == "expired"
    assert result["share_info"]["access_count"] == 0
    assert store.get_by_token("expired-token")["access_count"] == 0
