from __future__ import annotations

from types import SimpleNamespace

from aragora.inbox.auto_approval import AutoApprovalPolicy
from aragora.inbox.trust_wedge import InboxWedgeAction, ReceiptState, TriageDecision


def _decision(*, subject: str, action: InboxWedgeAction = InboxWedgeAction.ARCHIVE) -> TriageDecision:
    decision = TriageDecision.create(
        final_action=action,
        confidence=0.95,
        dissent_summary="",
    )
    decision.receipt_state = ReceiptState.CREATED.value
    decision.intent = SimpleNamespace(_subject=subject)
    return decision


def test_auto_approval_allows_routine_promotional_archive():
    policy = AutoApprovalPolicy()

    assert policy.can_auto_approve(_decision(subject="Game on. Your next trip = $25"))


def test_auto_approval_blocks_financial_receipt_subjects():
    policy = AutoApprovalPolicy()

    assert not policy.can_auto_approve(_decision(subject="Your receipt from Loom, Inc."))
    assert not policy.can_auto_approve(_decision(subject="Your Statement is Ready"))
    assert not policy.can_auto_approve(_decision(subject="Thank you for your payment"))
