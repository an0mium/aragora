"""HTTP-layer tests for the rolling-window triage metrics endpoint (#6373).

Endpoint under test:
  ``GET /api/v1/review-queue/triage-metrics``

The endpoint is served by the existing ``ReviewQueueHandler`` (gap
#6373 keeps the triage-metrics route cohesive with the rest of the
review-queue surface).

Covered behaviours:
  - 401 when unauthenticated
  - 403 when authenticated without the ``review_queue:read`` permission
  - 200 with both windows populated when receipts exist
  - 200 with sparse-data suppression in the empty tree
  - ETag round-trip returns 304 on ``If-None-Match``
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers import review_queue as rq


UTC = timezone.utc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    rq._review_queue_limiter._buckets.clear()


@pytest.fixture
def rq_root(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ARAGORA_REVIEW_QUEUE_ROOT", str(tmp_path))
    (tmp_path / "receipts").mkdir()
    return tmp_path


@pytest.fixture
def handler():
    return rq.ReviewQueueHandler(ctx={})


def _mock_handler(method: str = "GET", headers_extra: dict[str, str] | None = None) -> MagicMock:
    h = MagicMock()
    h.command = method
    h.client_address = ("127.0.0.1", 11111)
    hdrs = {
        "Authorization": "Bearer test",
        "Host": "localhost:8080",
    }
    if headers_extra:
        hdrs.update(headers_extra)
    h.headers = hdrs
    return h


def _auth_user(permissions=None) -> MagicMock:
    user = MagicMock()
    user.user_id = "armand"
    user.email = "armand@example.com"
    user.permissions = permissions if permissions is not None else ["review_queue:read"]
    user.roles = []
    user.role = "user"
    user.is_admin = False
    return user


def _write_receipt(
    root: Path,
    *,
    pr_number: int,
    session_id: str,
    action: str,
    reviewed_at: str | None = None,
    machine_recommendation: str = "approve_candidate",
    queue_bucket: str = "needs_attention",
    elapsed_seconds: float | None = 15.0,
) -> Path:
    payload: dict[str, Any] = {
        "session_id": session_id,
        "reviewed_at": reviewed_at or datetime.now(UTC).isoformat(),
        "actor": "armand",
        "action": action,
        "reason": "",
        "pr_number": pr_number,
        "pr_url": f"https://example.com/pr/{pr_number}",
        "head_sha": "a" * 40,
        "base_sha": "b" * 40,
        "packet_sha": "sha256:deadbeef",
        "queue_bucket": queue_bucket,
        "machine_recommendation": machine_recommendation,
        "github_event": action.upper(),
    }
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = elapsed_seconds
    path = root / "receipts" / f"pr-{pr_number}-{session_id}-{action}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _parse(result) -> dict[str, Any]:
    return json.loads(result.body)


# ---------------------------------------------------------------------------
# Auth / permission gate
# ---------------------------------------------------------------------------


class TestAuthGate:
    @pytest.mark.no_auto_auth
    def test_requires_authentication(self, handler, rq_root):
        mock_h = _mock_handler()
        with patch.object(handler, "get_current_user", return_value=None):
            result = handler.handle("/api/v1/review-queue/triage-metrics", {}, mock_h)
        assert result.status_code == 401

    @pytest.mark.no_auto_auth
    def test_requires_review_queue_read_permission(self, handler, rq_root):
        """An authenticated user without the permission gets 403.

        Opts out of the shared handler-test auto-auth fixture (which
        patches ``has_permission`` to always return True) so we exercise
        the real permission check.
        """
        mock_h = _mock_handler()
        low_user = _auth_user(permissions=[])
        with patch.object(handler, "get_current_user", return_value=low_user):
            result = handler.handle("/api/v1/review-queue/triage-metrics", {}, mock_h)
        assert result.status_code == 403

    @pytest.mark.no_auto_auth
    def test_admin_user_is_allowed(self, handler, rq_root):
        mock_h = _mock_handler()
        admin = _auth_user(permissions=["admin"])
        with patch.object(handler, "get_current_user", return_value=admin):
            result = handler.handle("/api/v1/review-queue/triage-metrics", {}, mock_h)
        assert result.status_code == 200


# ---------------------------------------------------------------------------
# Shape of the response
# ---------------------------------------------------------------------------


class TestResponseShape:
    def test_returns_both_windows_with_suppressed_metrics_when_empty(self, handler, rq_root):
        mock_h = _mock_handler()
        with patch.object(handler, "get_current_user", return_value=_auth_user()):
            result = handler.handle("/api/v1/review-queue/triage-metrics", {}, mock_h)
        assert result.status_code == 200
        data = _parse(result)
        assert set(data["windows"].keys()) == {"7d", "30d"}
        for window in data["windows"].values():
            assert window["total_decisions"] == 0
            assert window["escalation_rate"] is None
            assert "notes" in window
        assert "drift" in data
        assert data["commitment"] == "docs/THESIS.md Commitment 5"

    def test_populated_windows_produce_rates(self, handler, rq_root):
        """With 10+ escalated receipts, the 7-day window returns
        escalation_rate=1.0 and a non-None median duration."""
        now = datetime.now(UTC)
        for i in range(12):
            _write_receipt(
                rq_root,
                pr_number=100 + i,
                session_id=f"s{i}",
                action="request_changes",
                reviewed_at=(now - timedelta(hours=i + 1)).isoformat(),
                queue_bucket="needs_attention",
                elapsed_seconds=20.0 + i,
            )
        mock_h = _mock_handler()
        with patch.object(handler, "get_current_user", return_value=_auth_user()):
            result = handler.handle("/api/v1/review-queue/triage-metrics", {}, mock_h)
        assert result.status_code == 200
        data = _parse(result)
        seven = data["windows"]["7d"]
        assert seven["total_decisions"] == 12
        assert seven["escalation_rate"] == 1.0
        assert seven["settlement_duration_median_s"] is not None
        assert seven["counts"]["escalations"] == 12
        # correlation remains null because final_outcome is not in
        # current receipts — honest-partial-coverage.
        assert seven["human_override_outcome_correlation"] is None
        assert "outcome" in seven["notes"]["human_override_outcome_correlation"].lower()


# ---------------------------------------------------------------------------
# ETag round-trip
# ---------------------------------------------------------------------------


class TestETagRoundtrip:
    def test_returns_etag_and_honors_if_none_match(self, handler, rq_root):
        mock_h = _mock_handler()
        with patch.object(handler, "get_current_user", return_value=_auth_user()):
            first = handler.handle("/api/v1/review-queue/triage-metrics", {}, mock_h)
        assert first.status_code == 200
        etag = first.headers.get("ETag")
        assert etag and etag.startswith('"')

        # Second request with matching If-None-Match should get 304.
        mock_h2 = _mock_handler(headers_extra={"If-None-Match": etag})
        with patch.object(handler, "get_current_user", return_value=_auth_user()):
            second = handler.handle("/api/v1/review-queue/triage-metrics", {}, mock_h2)
        assert second.status_code == 304
        assert second.body == b""
        assert second.headers.get("ETag") == etag

    def test_etag_changes_when_receipts_change(self, handler, rq_root):
        mock_h = _mock_handler()
        with patch.object(handler, "get_current_user", return_value=_auth_user()):
            first = handler.handle("/api/v1/review-queue/triage-metrics", {}, mock_h)
        etag_before = first.headers.get("ETag")

        # Add a receipt — this changes the rolling-window counts, so
        # the generated_at timestamp or body should force a new ETag.
        _write_receipt(
            rq_root,
            pr_number=777,
            session_id="snew",
            action="request_changes",
            queue_bucket="needs_attention",
        )
        mock_h2 = _mock_handler(headers_extra={"If-None-Match": etag_before})
        with patch.object(handler, "get_current_user", return_value=_auth_user()):
            second = handler.handle("/api/v1/review-queue/triage-metrics", {}, mock_h2)
        # Either a new ETag with 200, or at minimum not a 304 — confirm
        # the response is NOT served from the cache.
        assert second.status_code == 200
        assert second.headers.get("ETag") != etag_before
