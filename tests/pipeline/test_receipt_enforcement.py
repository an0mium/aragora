"""Tests for the canonical receipt enforcement helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aragora.config.feature_flags import reset_flag_registry
from aragora.gauntlet.receipt_models import DecisionReceipt
from aragora.gauntlet.receipt_store import ReceiptState, get_receipt_store, reset_receipt_store
from aragora.pipeline.receipt_enforcement import (
    ReceiptEnforcementError,
    ReceiptExemption,
    is_receipt_enforcement_enabled,
    require_receipt_gate,
    transition_receipt_executed,
)


@pytest.fixture(autouse=True)
def _reset_globals() -> None:
    reset_receipt_store()
    reset_flag_registry()
    yield
    reset_receipt_store()
    reset_flag_registry()


def _debate_result() -> SimpleNamespace:
    return SimpleNamespace(
        debate_id="debate-receipt-enforcement",
        task="Ship the approved execution plan",
        final_answer="Proceed with the approved execution plan.",
        confidence=0.92,
        consensus_reached=True,
        rounds_used=3,
        duration_seconds=1.25,
        consensus_strength="majority",
        participants=["claude", "codex", "gemini"],
        dissenting_views=["critic: verify rollback path"],
        messages=[],
        votes=[],
        winner="codex",
    )


def _persist_signed_receipt(
    *,
    receipt_id: str = "receipt-1",
    state: ReceiptState = ReceiptState.APPROVED,
    tamper_signature: bool = False,
    tamper_integrity: bool = False,
) -> None:
    receipt = DecisionReceipt.from_debate_result(_debate_result())
    receipt.receipt_id = receipt_id
    if tamper_integrity:
        receipt.artifact_hash = "tampered-artifact-hash"
    else:
        receipt.artifact_hash = receipt._calculate_hash()
    receipt.sign()

    store = get_receipt_store()
    store.persist(
        receipt_id=receipt_id,
        receipt_data=receipt.to_dict(),
        signature=receipt.signature,
        signature_key_id=receipt.signature_key_id,
        signed_at=receipt.signed_at,
        signature_algorithm=receipt.signature_algorithm,
        state=state,
    )
    if tamper_signature:
        stored = store.get(receipt_id)
        assert stored is not None
        stored.signature = "tampered-signature"


def test_known_domains_default_to_disabled() -> None:
    assert is_receipt_enforcement_enabled("openclaw") is False
    assert is_receipt_enforcement_enabled("canvas") is False
    assert is_receipt_enforcement_enabled("computer_use") is False
    assert is_receipt_enforcement_enabled("inbox") is False
    assert is_receipt_enforcement_enabled("shared_inbox") is False


def test_feature_flag_env_override_enables_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAGORA_RECEIPT_ENFORCEMENT_OPENCLAW", "true")
    reset_flag_registry()

    assert is_receipt_enforcement_enabled("openclaw") is True


def test_require_receipt_gate_skips_when_flag_disabled() -> None:
    result = require_receipt_gate(
        action_domain="openclaw",
        action_type="execute_action",
        actor_id="user-1",
        resource_id="action-1",
        receipt_id=None,
    )

    assert result is None


def test_require_receipt_gate_allows_documented_exemption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARAGORA_RECEIPT_ENFORCEMENT_OPENCLAW", "true")
    reset_flag_registry()

    result = require_receipt_gate(
        action_domain="openclaw",
        action_type="execute_action",
        actor_id="admin-user",
        resource_id="action-2",
        exempt=ReceiptExemption(
            reason="Legacy admin remediation path",
            approved_by="security-admin",
            category="legacy_admin",
        ),
    )

    assert result is None


def test_require_receipt_gate_returns_stored_receipt_for_valid_signed_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARAGORA_RECEIPT_ENFORCEMENT_OPENCLAW", "true")
    reset_flag_registry()
    _persist_signed_receipt(receipt_id="receipt-ok")

    stored = require_receipt_gate(
        action_domain="openclaw",
        action_type="execute_action",
        actor_id="user-42",
        resource_id="action-3",
        receipt_id="receipt-ok",
    )

    assert stored is not None
    assert stored.receipt_id == "receipt-ok"
    assert stored.state == ReceiptState.APPROVED


def test_require_receipt_gate_raises_without_receipt_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARAGORA_RECEIPT_ENFORCEMENT_OPENCLAW", "true")
    reset_flag_registry()

    with pytest.raises(ReceiptEnforcementError, match="requires an approved execution receipt"):
        require_receipt_gate(
            action_domain="openclaw",
            action_type="execute_action",
            actor_id="user-42",
            resource_id="action-4",
        )


def test_require_receipt_gate_raises_for_wrong_receipt_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARAGORA_RECEIPT_ENFORCEMENT_OPENCLAW", "true")
    reset_flag_registry()
    _persist_signed_receipt(receipt_id="receipt-expired", state=ReceiptState.EXPIRED)

    with pytest.raises(ReceiptEnforcementError, match="expected APPROVED"):
        require_receipt_gate(
            action_domain="openclaw",
            action_type="execute_action",
            actor_id="user-42",
            resource_id="action-5",
            receipt_id="receipt-expired",
        )


def test_require_receipt_gate_raises_for_tampered_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARAGORA_RECEIPT_ENFORCEMENT_OPENCLAW", "true")
    reset_flag_registry()
    _persist_signed_receipt(receipt_id="receipt-badsig", tamper_signature=True)

    with pytest.raises(ReceiptEnforcementError, match="signature verification"):
        require_receipt_gate(
            action_domain="openclaw",
            action_type="execute_action",
            actor_id="user-42",
            resource_id="action-6",
            receipt_id="receipt-badsig",
        )


def test_require_receipt_gate_raises_for_integrity_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARAGORA_RECEIPT_ENFORCEMENT_OPENCLAW", "true")
    reset_flag_registry()
    _persist_signed_receipt(receipt_id="receipt-badintegrity", tamper_integrity=True)

    with pytest.raises(ReceiptEnforcementError, match="integrity verification"):
        require_receipt_gate(
            action_domain="openclaw",
            action_type="execute_action",
            actor_id="user-42",
            resource_id="action-7",
            receipt_id="receipt-badintegrity",
        )


def test_transition_receipt_executed_advances_state() -> None:
    _persist_signed_receipt(receipt_id="receipt-transition")

    updated = transition_receipt_executed("receipt-transition")

    assert updated.state == ReceiptState.EXECUTED


def test_transition_receipt_executed_is_idempotent_for_executed_receipt() -> None:
    _persist_signed_receipt(receipt_id="receipt-executed")
    first = transition_receipt_executed("receipt-executed")
    second = transition_receipt_executed("receipt-executed")

    assert first.state == ReceiptState.EXECUTED
    assert second.state == ReceiptState.EXECUTED
