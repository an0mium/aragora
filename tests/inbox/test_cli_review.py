"""Tests for receipt-backed CLI inbox review."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from aragora.inbox.cli_review import CLIReviewLoop
from aragora.inbox.trust_wedge import InboxWedgeAction, ReceiptState, TriageDecision


def _make_decision() -> TriageDecision:
    return TriageDecision.create(
        final_action="ignore",
        confidence=0.4,
        dissent_summary="",
        receipt_id="receipt-1",
        provider_route="gmail",
        cost_usd=0.0125,
        latency_seconds=1.25,
        execution_tier="baseline",
    )


def _make_envelope(
    decision: TriageDecision,
    *,
    action: str,
    state: ReceiptState,
):
    updated = TriageDecision.create(
        final_action=action,
        confidence=decision.confidence,
        dissent_summary=decision.dissent_summary,
        receipt_id=decision.receipt_id,
        auto_approval_eligible=state is ReceiptState.APPROVED,
        receipt_state=state.value,
        intent=decision.intent,
        provider_route=decision.provider_route,
        label_id=decision.label_id,
        blocked_by_policy=decision.blocked_by_policy,
        cost_usd=decision.cost_usd,
        latency_seconds=decision.latency_seconds,
        execution_tier=decision.execution_tier,
        escalation_reasons=decision.escalation_reasons,
        suppressed_diagnostics_count=decision.suppressed_diagnostics_count,
    )
    return SimpleNamespace(
        intent=decision.intent,
        decision=updated,
        receipt=SimpleNamespace(receipt_id=decision.receipt_id, state=state),
    )


def test_review_batch_uses_receipt_review_for_approve():
    decision = _make_decision()
    review_fn = MagicMock(
        return_value=_make_envelope(decision, action="ignore", state=ReceiptState.APPROVED)
    )

    loop = CLIReviewLoop(
        input_fn=lambda _prompt: "a",
        print_fn=lambda *_args, **_kwargs: None,
        review_fn=review_fn,
    )

    results = loop.review_batch([decision])

    review_fn.assert_called_once_with(
        "receipt-1",
        choice="approve",
        edited_action=None,
        edited_rationale=None,
        label_id=None,
    )
    assert results[0]["action_taken"] == "approve"
    assert decision.receipt_state == ReceiptState.APPROVED.value


def test_review_batch_uses_receipt_review_for_edit():
    decision = _make_decision()
    answers = iter(["e", "archive"])
    review_fn = MagicMock(
        return_value=_make_envelope(decision, action="archive", state=ReceiptState.CREATED)
    )

    loop = CLIReviewLoop(
        input_fn=lambda _prompt: next(answers),
        print_fn=lambda *_args, **_kwargs: None,
        review_fn=review_fn,
    )

    results = loop.review_batch([decision])

    review_fn.assert_called_once_with(
        "receipt-1",
        choice="edit",
        edited_action="archive",
        edited_rationale=None,
        label_id=None,
    )
    assert results[0]["action_taken"] == "edit"
    assert decision.final_action == InboxWedgeAction.ARCHIVE
    assert decision.receipt_state == ReceiptState.CREATED.value


def test_review_batch_displays_manual_review_reason():
    decision = TriageDecision.create(
        final_action="ignore",
        confidence=0.0,
        dissent_summary="No consensus reached; manual review required.",
        receipt_id="receipt-2",
        provider_route="gmail",
        blocked_by_policy=True,
        execution_tier="escalated",
        escalation_reasons=["policy:block", "confidence:low"],
        suppressed_diagnostics_count=2,
    )
    printed: list[str] = []

    loop = CLIReviewLoop(
        input_fn=lambda _prompt: "s",
        print_fn=lambda *args, **_kwargs: printed.append(" ".join(str(arg) for arg in args)),
    )

    loop.review_batch([decision])

    output = "\n".join(printed)
    assert "No consensus reached; manual review required." in output
    assert "Route   : gmail" in output
    assert "Tier    : escalated" in output
    assert "Escal.  : policy:block, confidence:low" in output
    assert "Suppres.: 2" in output


def test_review_batch_copies_review_metadata_from_envelope():
    decision = _make_decision()
    updated = TriageDecision.create(
        final_action="archive",
        confidence=0.9,
        dissent_summary="policy override",
        receipt_id="receipt-1",
        auto_approval_eligible=False,
        receipt_state=ReceiptState.CREATED.value,
        provider_route="gmail+policy",
        blocked_by_policy=True,
        cost_usd=0.0321,
        latency_seconds=2.75,
        execution_tier="escalated",
        escalation_reasons=["policy:block"],
        suppressed_diagnostics_count=3,
    )
    review_fn = MagicMock(
        return_value=SimpleNamespace(
            intent=decision.intent,
            decision=updated,
            receipt=SimpleNamespace(receipt_id="receipt-1", state=ReceiptState.CREATED),
        )
    )

    loop = CLIReviewLoop(
        input_fn=lambda _prompt: "a",
        print_fn=lambda *_args, **_kwargs: None,
        review_fn=review_fn,
    )

    loop.review_batch([decision])

    assert decision.final_action == InboxWedgeAction.ARCHIVE
    assert decision.confidence == 0.9
    assert decision.dissent_summary == "policy override"
    assert decision.provider_route == "gmail+policy"
    assert decision.blocked_by_policy is True
    assert decision.cost_usd == 0.0321
    assert decision.latency_seconds == 2.75
    assert decision.execution_tier == "escalated"
    assert decision.escalation_reasons == ["policy:block"]
    assert decision.suppressed_diagnostics_count == 3
