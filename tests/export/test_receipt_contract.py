"""Tests for the canonical DecisionReceipt import contract."""

from __future__ import annotations


def test_receipt_contract_exports_canonical_and_legacy_classes() -> None:
    from aragora.export.decision_receipt import DecisionReceipt as ExportDecisionReceipt
    from aragora.gauntlet.receipt import DecisionReceipt as GauntletDecisionReceipt
    from aragora.receipts import DecisionReceipt, LegacyDecisionReceipt

    assert DecisionReceipt is GauntletDecisionReceipt
    assert LegacyDecisionReceipt is ExportDecisionReceipt


def test_legacy_receipt_import_path_round_trips_schema_fields() -> None:
    from aragora.export.decision_receipt import DecisionReceipt

    receipt = DecisionReceipt(receipt_id="receipt-test", gauntlet_id="gauntlet-test")
    payload = receipt.to_dict()
    restored = DecisionReceipt.from_dict(payload)

    assert restored.receipt_id == payload["receipt_id"]
    assert restored.gauntlet_id == payload["gauntlet_id"]
    assert restored.schema_version == payload["schema_version"]
