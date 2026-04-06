from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aragora.storage.landing_review_store import (
    get_landing_review_store,
    reset_landing_review_store,
)


def test_landing_review_store_persists_events_and_feedback(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "landing_review.sqlite3"
    monkeypatch.setenv("ARAGORA_LANDING_REVIEW_DB_PATH", str(db_path))
    reset_landing_review_store()

    store = get_landing_review_store()
    now = datetime.now(timezone.utc)
    store.record_event(
        event_type="preflight_shown",
        client_tag="ip:test-client",
        data={"question_length": 88},
        timestamp=now.isoformat(),
    )
    store.record_feedback(
        {
            "id": "lfb_test",
            "timestamp": now.isoformat(),
            "client_tag": "ip:test-client",
            "question": "Should I microwave chicken nuggets?",
            "interpreted_question": "Is it safe to reheat pre-cooked chicken nuggets?",
            "final_answer_preview": "Yes, reheat until hot throughout.",
            "result_warning": None,
            "result_mode": "preview",
            "debate_id": "debate-123",
            "verdict": "needs_review",
            "participant_count": 3,
            "rewritten": True,
        }
    )

    reset_landing_review_store()
    reopened = get_landing_review_store()

    events = reopened.list_recent_events(window_seconds=3600)
    feedback = reopened.list_recent_feedback(window_seconds=3600, limit=10)
    assert reopened.count_events() == 1
    assert reopened.count_feedback() == 1
    assert events == [
        {
            "event_type": "preflight_shown",
            "client_tag": "ip:test-client",
            "data": {"question_length": 88},
            "timestamp": now.isoformat(),
        }
    ]
    assert feedback == [
        {
            "id": "lfb_test",
            "timestamp": now.isoformat(),
            "client_tag": "ip:test-client",
            "question": "Should I microwave chicken nuggets?",
            "interpreted_question": "Is it safe to reheat pre-cooked chicken nuggets?",
            "final_answer_preview": "Yes, reheat until hot throughout.",
            "result_warning": None,
            "result_mode": "preview",
            "debate_id": "debate-123",
            "verdict": "needs_review",
            "participant_count": 3,
            "rewritten": True,
            "review_status": "pending",
            "reviewed_at": None,
            "reviewed_by": None,
        }
    ]

    reset_landing_review_store()


def test_landing_review_store_filters_by_window(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "landing_review.sqlite3"
    monkeypatch.setenv("ARAGORA_LANDING_REVIEW_DB_PATH", str(db_path))
    reset_landing_review_store()

    store = get_landing_review_store()
    now = datetime.now(timezone.utc)
    store.record_event(
        event_type="preflight_shown",
        client_tag="ip:recent",
        data={"question_length": 42},
        timestamp=now.isoformat(),
    )
    store.record_event(
        event_type="preflight_shown",
        client_tag="ip:stale",
        data={"question_length": 12},
        timestamp=(now - timedelta(days=7)).isoformat(),
    )

    recent_events = store.list_recent_events(window_seconds=3600)
    assert len(recent_events) == 1
    assert recent_events[0]["client_tag"] == "ip:recent"

    reset_landing_review_store()


def test_landing_review_store_updates_review_status(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "landing_review.sqlite3"
    monkeypatch.setenv("ARAGORA_LANDING_REVIEW_DB_PATH", str(db_path))
    reset_landing_review_store()

    store = get_landing_review_store()
    now = datetime.now(timezone.utc)
    store.record_feedback(
        {
            "id": "lfb_reviewable",
            "timestamp": now.isoformat(),
            "client_tag": "ip:test-client",
            "question": "Should I microwave chicken nuggets?",
            "interpreted_question": "Is it safe to reheat pre-cooked chicken nuggets?",
            "final_answer_preview": "Yes, reheat until hot throughout.",
            "result_warning": None,
            "result_mode": "preview",
            "debate_id": "debate-123",
            "verdict": "needs_review",
            "participant_count": 3,
            "rewritten": False,
        }
    )

    reviewed_at = (now + timedelta(minutes=5)).isoformat()
    assert store.update_feedback_review(
        report_id="lfb_reviewable",
        review_status="resolved",
        reviewed_at=reviewed_at,
        reviewed_by="owner@aragora.ai",
    )

    feedback = store.list_recent_feedback(window_seconds=3600, limit=10)
    assert feedback[0]["review_status"] == "resolved"
    assert feedback[0]["reviewed_at"] == reviewed_at
    assert feedback[0]["reviewed_by"] == "owner@aragora.ai"

    assert not store.update_feedback_review(
        report_id="missing",
        review_status="reviewed",
        reviewed_at=reviewed_at,
        reviewed_by="owner@aragora.ai",
    )

    reset_landing_review_store()
