"""Tests for ReceiptStoreFacade — Phase 3 receipt store convergence."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from aragora.gauntlet.receipt_store import (
    ReceiptState,
    StoredReceipt,
    get_receipt_store,
    reset_receipt_store,
)
from aragora.pipeline.receipt_store_facade import (
    ReceiptStoreFacade,
    get_receipt_store_facade,
    reset_receipt_store_facade,
)


@pytest.fixture(autouse=True)
def _clean_singletons():
    """Reset singletons before and after each test."""
    reset_receipt_store()
    reset_receipt_store_facade()
    yield
    reset_receipt_store()
    reset_receipt_store_facade()


def _sample_receipt_data(receipt_id: str = "r-001") -> dict:
    return {
        "receipt_id": receipt_id,
        "gauntlet_id": "g-001",
        "debate_id": "d-001",
        "verdict": "APPROVED",
        "confidence": 0.95,
        "risk_level": "LOW",
        "risk_score": 0.1,
        "checksum": "abc123",
        "timestamp": 1700000000.0,
    }


class TestPersistAndSave:
    """persist_and_save writes to gauntlet store and best-effort to storage."""

    def test_writes_to_gauntlet_store(self):
        facade = ReceiptStoreFacade()
        data = _sample_receipt_data()
        facade.persist_and_save(
            "r-001",
            data,
            signature="sig123",
            state="CREATED",
        )

        gauntlet = get_receipt_store()
        stored = gauntlet.get("r-001")
        assert stored is not None
        assert stored.receipt_id == "r-001"
        assert stored.state == ReceiptState.CREATED
        assert stored.signature == "sig123"
        assert stored.receipt_data["receipt_id"] == "r-001"
        assert stored.receipt_data["checksum"] == "abc123"

    def test_accepts_lowercase_state(self):
        facade = ReceiptStoreFacade()
        facade.persist_and_save("r-002", _sample_receipt_data("r-002"), state="created")

        gauntlet = get_receipt_store()
        stored = gauntlet.get("r-002")
        assert stored is not None
        assert stored.state == ReceiptState.CREATED

    def test_storage_failure_does_not_raise(self):
        """Best-effort storage write — backend write failures are swallowed."""
        facade = ReceiptStoreFacade()
        mock_storage = MagicMock()
        mock_storage.save.side_effect = sqlite3.OperationalError("readonly database")

        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            return_value=mock_storage,
        ):
            facade.persist_and_save("r-003", _sample_receipt_data("r-003"))

        gauntlet = get_receipt_store()
        assert gauntlet.get("r-003") is not None

    def test_writes_same_canonical_payload_to_both_stores(self):
        facade = ReceiptStoreFacade()
        mock_storage = MagicMock()
        payload = {
            "verdict": "APPROVED",
            "confidence": 0.95,
            "artifact_hash": "hash-004",
            "receipt": {},
        }

        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            return_value=mock_storage,
        ):
            facade.persist_and_save("r-004", payload, state="CREATED")

        gauntlet = get_receipt_store()
        stored = gauntlet.get("r-004")
        assert stored is not None
        assert mock_storage.save.call_args.args[0] == stored.receipt_data
        assert stored.receipt_data["receipt_id"] == "r-004"
        assert stored.receipt_data["debate_id"] == "r-004"
        assert stored.receipt_data["gauntlet_id"] == "r-004"
        assert stored.receipt_data["checksum"] == "hash-004"
        assert stored.receipt_data["receipt"]["id"] == "r-004"
        assert stored.receipt_data["receipt"]["artifact_hash"] == "hash-004"


class TestGetCanonical:
    """get_canonical prefers gauntlet store, falls back to storage."""

    def test_returns_from_gauntlet(self):
        facade = ReceiptStoreFacade()
        facade.persist_and_save("r-010", _sample_receipt_data("r-010"))

        result = facade.get_canonical("r-010")
        assert result is not None
        assert result["receipt_id"] == "r-010"
        assert result["state"] == "CREATED"
        assert result["checksum"] == "abc123"
        assert "receipt_data" not in result

    def test_returns_none_when_missing(self):
        facade = ReceiptStoreFacade()
        result = facade.get_canonical("nonexistent")
        assert result is None

    def test_falls_back_to_storage_store(self):
        """When gauntlet has no entry, try storage store."""
        facade = ReceiptStoreFacade()
        # gauntlet is empty; mock storage store
        mock_stored = MagicMock()
        mock_stored.to_full_dict.return_value = {
            "receipt_id": "r-fallback",
            "gauntlet_id": "g-fallback",
            "artifact_hash": "hash-fallback",
            "receipt": {"id": "r-fallback"},
        }
        mock_storage = MagicMock()
        mock_storage.get.return_value = mock_stored

        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            return_value=mock_storage,
        ):
            result = facade.get_canonical("r-fallback")

        assert result is not None
        assert result["receipt_id"] == "r-fallback"
        assert result["artifact_hash"] == "hash-fallback"
        assert result["receipt"]["id"] == "r-fallback"


class TestTransition:
    """transition delegates to gauntlet store."""

    def test_transitions_state(self):
        facade = ReceiptStoreFacade()
        facade.persist_and_save("r-020", _sample_receipt_data("r-020"), state="CREATED")

        facade.transition("r-020", "APPROVED")

        gauntlet = get_receipt_store()
        stored = gauntlet.get("r-020")
        assert stored is not None
        assert stored.state == ReceiptState.APPROVED

    def test_case_insensitive_transition(self):
        facade = ReceiptStoreFacade()
        facade.persist_and_save("r-021", _sample_receipt_data("r-021"), state="CREATED")

        facade.transition("r-021", "approved")

        gauntlet = get_receipt_store()
        stored = gauntlet.get("r-021")
        assert stored is not None
        assert stored.state == ReceiptState.APPROVED


class TestVerify:
    """verify delegates to gauntlet store's verify_receipt."""

    def test_verify_missing_returns_false(self):
        facade = ReceiptStoreFacade()
        assert facade.verify("nonexistent") is False

    def test_verify_unsigned_returns_false(self):
        facade = ReceiptStoreFacade()
        facade.persist_and_save("r-030", _sample_receipt_data("r-030"))
        assert facade.verify("r-030") is False


class TestSingleton:
    """get_receipt_store_facade returns a stable singleton."""

    def test_returns_same_instance(self):
        a = get_receipt_store_facade()
        b = get_receipt_store_facade()
        assert a is b

    def test_reset_clears_singleton(self):
        a = get_receipt_store_facade()
        reset_receipt_store_facade()
        b = get_receipt_store_facade()
        assert a is not b
