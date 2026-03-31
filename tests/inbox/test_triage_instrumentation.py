"""Tests for triage instrumentation: feedback, review queue, digest, calibration."""

from __future__ import annotations

import json

import pytest

from aragora.gauntlet.signing import HMACSigner, ReceiptSigner
from aragora.inbox.trust_wedge import (
    ActionIntent,
    InboxTrustWedgeStore,
    TriageDecision,
)


def _build_intent(
    *,
    action: str = "archive",
    message_id: str = "msg-1",
    subject: str = "Test Subject",
    sender: str = "test@example.com",
    confidence: float = 0.91,
) -> ActionIntent:
    intent = ActionIntent.create(
        provider="gmail",
        user_id="user-1",
        message_id=message_id,
        action=action,
        content_hash=ActionIntent.compute_content_hash(subject, "body"),
        synthesized_rationale="Debated rationale",
        confidence=confidence,
        provider_route="direct",
        debate_id="debate-123",
    )
    # Attach display metadata
    intent._subject = subject
    intent._sender = sender
    intent._snippet = "snippet"
    return intent


def _build_decision(
    *,
    action: str = "archive",
    confidence: float = 0.91,
    blocked_by_policy: bool = False,
    execution_tier: str = "fast",
) -> TriageDecision:
    decision = TriageDecision.create(
        final_action=action,
        confidence=confidence,
        dissent_summary="",
        blocked_by_policy=blocked_by_policy,
    )
    decision.execution_tier = execution_tier
    return decision


@pytest.fixture
def store(tmp_path):
    signer = ReceiptSigner(HMACSigner(secret_key=b"\x01" * 32, key_id="test-key"))
    s = InboxTrustWedgeStore(db_path=str(tmp_path / "wedge.db"))
    yield s
    s.close()


def _create_receipt(
    store,
    *,
    confidence=0.91,
    action="archive",
    blocked=False,
    execution_tier="fast",
    subject="Test",
    sender="test@example.com",
    message_id="msg-1",
):
    """Helper to create a receipt and return its ID."""
    signer = ReceiptSigner(HMACSigner(secret_key=b"\x01" * 32, key_id="test-key"))
    intent = _build_intent(
        action=action,
        message_id=message_id,
        subject=subject,
        sender=sender,
        confidence=confidence,
    )
    decision = _build_decision(
        action=action,
        confidence=confidence,
        blocked_by_policy=blocked,
        execution_tier=execution_tier,
    )
    from aragora.inbox.trust_wedge import InboxTrustWedgeService
    from aragora.services.email_actions import EmailActionsService

    service = InboxTrustWedgeService(
        email_actions_service=EmailActionsService(),
        store=store,
        signer=signer,
    )
    envelope = service.create_receipt(intent, decision)
    return envelope.receipt.receipt_id


# ---------------------------------------------------------------------------
# Feedback tests
# ---------------------------------------------------------------------------


class TestRecordFeedback:
    def test_records_and_retrieves(self, store):
        rid = _create_receipt(store)
        fid = store.record_feedback(rid, label="good")
        assert fid
        stats = store.get_feedback_stats()
        assert stats["good"] == 1
        assert stats["total"] == 1
        assert stats["accuracy"] == 1.0

    def test_rejects_invalid_label(self, store):
        rid = _create_receipt(store)
        with pytest.raises(ValueError, match="Invalid label"):
            store.record_feedback(rid, label="excellent")

    def test_updates_review_choice(self, store):
        rid = _create_receipt(store)
        store.record_feedback(rid, label="bad")
        envelope = store.get_receipt(rid)
        assert envelope is not None
        assert envelope.review_choice == "label_bad"

    def test_multiple_labels_counted(self, store):
        for i, label in enumerate(["good", "good", "bad", "skip"]):
            rid = _create_receipt(store, message_id=f"msg-{i}")
            store.record_feedback(rid, label=label)
        stats = store.get_feedback_stats()
        assert stats["good"] == 2
        assert stats["bad"] == 1
        assert stats["skip"] == 1
        assert stats["accuracy"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Review queue tests
# ---------------------------------------------------------------------------


class TestReviewQueue:
    def test_blocked_first(self, store):
        rid_normal = _create_receipt(store, confidence=0.95, message_id="msg-normal")
        rid_blocked = _create_receipt(
            store, confidence=0.95, blocked=True, message_id="msg-blocked"
        )
        queue = store.list_review_queue(limit=10)
        assert len(queue) >= 2
        # Blocked should come first
        ids = [item["receipt_id"] for item in queue]
        assert ids.index(rid_blocked) < ids.index(rid_normal)

    def test_low_confidence_before_high(self, store):
        rid_low = _create_receipt(store, confidence=0.3, message_id="msg-low")
        rid_high = _create_receipt(store, confidence=0.99, message_id="msg-high")
        queue = store.list_review_queue(limit=10)
        ids = [item["receipt_id"] for item in queue]
        assert ids.index(rid_low) < ids.index(rid_high)

    def test_escalated_before_fast(self, store):
        rid_fast = _create_receipt(store, execution_tier="fast", message_id="msg-fast")
        rid_esc = _create_receipt(store, execution_tier="escalated", message_id="msg-esc")
        queue = store.list_review_queue(limit=10)
        ids = [item["receipt_id"] for item in queue]
        assert ids.index(rid_esc) < ids.index(rid_fast)

    def test_excludes_reviewed_by_default(self, store):
        rid = _create_receipt(store)
        store.record_feedback(rid, label="good")
        queue = store.list_review_queue(limit=10)
        ids = [item["receipt_id"] for item in queue]
        assert rid not in ids

    def test_includes_reviewed_when_flag_set(self, store):
        rid = _create_receipt(store)
        store.record_feedback(rid, label="good")
        queue = store.list_review_queue(limit=10, include_reviewed=True)
        ids = [item["receipt_id"] for item in queue]
        assert rid in ids


# ---------------------------------------------------------------------------
# Digest tests
# ---------------------------------------------------------------------------


class TestDigestData:
    def test_groups_by_action(self, store):
        _create_receipt(store, action="archive", message_id="msg-a1")
        _create_receipt(store, action="archive", message_id="msg-a2")
        _create_receipt(store, action="ignore", message_id="msg-i1")
        data = store.get_digest_data(since_hours=1.0)
        assert data["total"] == 3
        assert data["by_action"]["archive"] == 2
        assert data["by_action"]["ignore"] == 1

    def test_groups_by_domain_from_intent_json(self, store):
        """Domain grouping works when _sender is in the stored intent_json."""
        # Create receipts then patch intent_json to include _sender (simulating production)
        import json as json_mod

        for i, sender in enumerate(["a@foo.com", "b@foo.com", "c@bar.com"]):
            rid = _create_receipt(store, message_id=f"msg-domain-{i}")
            with store._cursor() as cursor:
                row = cursor.execute(
                    "SELECT intent_json FROM inbox_trust_receipts WHERE receipt_id = ?",
                    (rid,),
                ).fetchone()
                intent = json_mod.loads(row["intent_json"])
                intent["_sender"] = sender
                cursor.execute(
                    "UPDATE inbox_trust_receipts SET intent_json = ? WHERE receipt_id = ?",
                    (json_mod.dumps(intent), rid),
                )
        data = store.get_digest_data(since_hours=1.0)
        assert data["by_domain"].get("foo.com") == 2
        assert data["by_domain"].get("bar.com") == 1

    def test_computes_avg_confidence(self, store):
        _create_receipt(store, confidence=0.8, message_id="msg-1")
        _create_receipt(store, confidence=1.0, message_id="msg-2")
        data = store.get_digest_data(since_hours=1.0)
        assert data["avg_confidence"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_perfect_calibration(self, store):
        """All good at high confidence = Brier ~0."""
        for i in range(5):
            rid = _create_receipt(store, confidence=0.95, message_id=f"msg-{i}")
            store.record_feedback(rid, label="good")

        from aragora.inbox.triage_instrumentation import compute_triage_calibration

        cal = compute_triage_calibration(store)
        assert cal["overall_accuracy"] == 1.0
        assert cal["overall_brier"] < 0.01

    def test_bad_calibration(self, store):
        """All bad at high confidence = high Brier."""
        for i in range(5):
            rid = _create_receipt(store, confidence=0.95, message_id=f"msg-{i}")
            store.record_feedback(rid, label="bad")

        from aragora.inbox.triage_instrumentation import compute_triage_calibration

        cal = compute_triage_calibration(store)
        assert cal["overall_accuracy"] == 0.0
        assert cal["overall_brier"] > 0.8

    def test_threshold_suggestion(self, store):
        """Miscalibrated low bucket triggers suggestion."""
        # Create high-confidence good decisions
        for i in range(5):
            rid = _create_receipt(store, confidence=0.95, message_id=f"msg-good-{i}")
            store.record_feedback(rid, label="good")
        # Create low-confidence bad decisions
        for i in range(5):
            rid = _create_receipt(store, confidence=0.5, message_id=f"msg-bad-{i}")
            store.record_feedback(rid, label="bad")

        from aragora.inbox.triage_instrumentation import (
            compute_triage_calibration,
            suggest_threshold_adjustment,
        )

        cal = compute_triage_calibration(store)
        suggestion = suggest_threshold_adjustment(cal)
        # Should suggest raising threshold since low-confidence bucket has 0% accuracy
        assert suggestion is not None or cal["total_labeled"] < 10

    def test_empty_feedback_returns_defaults(self, store):
        from aragora.inbox.triage_instrumentation import compute_triage_calibration

        cal = compute_triage_calibration(store)
        assert cal["total_labeled"] == 0
        assert cal["overall_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Audit prompt tests
# ---------------------------------------------------------------------------


class TestAuditPrompt:
    def test_build_audit_prompt_formats_items(self):
        from aragora.inbox.triage_instrumentation import build_audit_prompt

        items = [
            {
                "receipt_id": "abc",
                "action": "archive",
                "confidence": 0.95,
                "subject": "Sale!",
                "sender": "spam@vendor.com",
                "blocked": False,
            },
        ]
        prompt = build_audit_prompt(items)
        assert "abc" in prompt
        assert "archive" in prompt
        assert "spam@vendor.com" in prompt
        assert "adversarial auditor" in prompt.lower()
