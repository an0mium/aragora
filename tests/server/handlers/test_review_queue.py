"""Tests for aragora.server.handlers.review_queue — PDB UI v0.

These tests cover the six endpoints exposed to the browser review surface:

- GET  /api/v1/review-queue/prs
- GET  /api/v1/review-queue/prs/{n}/brief
- POST /api/v1/review-queue/prs/{n}/approve
- POST /api/v1/review-queue/prs/{n}/request-changes
- POST /api/v1/review-queue/prs/{n}/defer
- GET  /api/v1/review-queue/stats
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
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Clear the rate-limit buckets between tests."""
    rq._review_queue_limiter._buckets.clear()


@pytest.fixture
def rq_root(tmp_path, monkeypatch):
    """Redirect the review-queue artifact root at a tmp dir."""
    monkeypatch.setenv("ARAGORA_REVIEW_QUEUE_ROOT", str(tmp_path))
    # Force rediscovery — the module caches nothing, but be explicit for future-proofing.
    return tmp_path


@pytest.fixture
def handler():
    return rq.ReviewQueueHandler(ctx={})


def _mock_handler(method: str = "GET", body: bytes = b"") -> MagicMock:
    h = MagicMock()
    h.command = method
    h.client_address = ("127.0.0.1", 11111)
    h.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
        "Authorization": "Bearer test",
        "Host": "localhost:8080",
    }
    h.rfile = MagicMock()
    h.rfile.read.return_value = body
    return h


def _parse(result) -> dict[str, Any]:
    return json.loads(result.body)


def _write_brief(root: Path, pr_number: int, head_sha: str, **extra: Any) -> Path:
    briefs = root / "briefs"
    briefs.mkdir(parents=True, exist_ok=True)
    path = briefs / f"pr-{pr_number}-{head_sha[:12]}.json"
    payload = {
        "pr_number": pr_number,
        "head_sha": head_sha,
        "verdict": "approve_candidate",
        "confidence": 4,
        "logic": "",
        "security": "",
        "maintainability": "",
        "skeptic": "",
    }
    payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _fake_pr(
    *,
    number: int,
    title: str = "Example PR",
    head_sha: str = "abc123def4567890",
    author: str = "armand",
    additions: int = 10,
    deletions: int = 2,
    files: list[str] | None = None,
    checks: list[dict[str, Any]] | None = None,
    created_at: str | None = None,
    is_draft: bool = False,
) -> dict[str, Any]:
    return {
        "number": number,
        "title": title,
        "url": f"https://github.com/synaptent/aragora/pull/{number}",
        "headRefOid": head_sha,
        "isDraft": is_draft,
        "author": {"login": author},
        "labels": [],
        "additions": additions,
        "deletions": deletions,
        "changedFiles": len(files or []),
        "createdAt": created_at or (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        "updatedAt": (datetime.now(UTC) - timedelta(minutes=10)).isoformat(),
        "statusCheckRollup": checks or [{"conclusion": "SUCCESS", "status": "COMPLETED"}],
        "files": [{"path": p} for p in (files or ["aragora/server/foo.py"])],
    }


# ---------------------------------------------------------------------------
# Instantiation & routing basics
# ---------------------------------------------------------------------------


class TestBasics:
    def test_can_handle_unversioned(self, handler):
        assert handler.can_handle("/api/review-queue/prs") is True

    def test_can_handle_versioned(self, handler):
        assert handler.can_handle("/api/v1/review-queue/prs") is True

    def test_cannot_handle_unrelated(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_handle_unknown_subpath_returns_404(self, handler):
        mock_h = _mock_handler("GET")
        result = handler.handle("/api/review-queue/unknown", {}, mock_h)
        assert result.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/review-queue/prs
# ---------------------------------------------------------------------------


class TestListPRs:
    def test_returns_prs_from_gh(self, handler, rq_root):
        prs = [_fake_pr(number=42, title="Fix bug"), _fake_pr(number=43, title="New feat")]
        with patch.object(rq, "_run_gh", return_value=(0, json.dumps(prs), "")):
            mock_h = _mock_handler("GET")
            result = handler.handle("/api/v1/review-queue/prs", {}, mock_h)
        assert result.status_code == 200
        data = _parse(result)
        assert data["total"] == 2
        assert data["degraded"] is False
        numbers = {p["number"] for p in data["prs"]}
        assert numbers == {42, 43}
        # Verify shape
        pr = data["prs"][0]
        assert "ci" in pr
        assert pr["ci"]["success"] == 1
        assert pr["ci"]["total"] == 1
        assert pr["brief_present"] is False
        assert pr["verdict"] is None
        assert pr["deferred"] is False
        assert "aragora/server" in pr["touched_subsystems"]

    def test_returns_degraded_when_gh_missing(self, handler, rq_root):
        with patch.object(rq, "_run_gh", return_value=(127, "", "gh CLI not found")):
            mock_h = _mock_handler("GET")
            result = handler.handle("/api/v1/review-queue/prs", {}, mock_h)
        assert result.status_code == 200
        data = _parse(result)
        assert data["degraded"] is True
        assert data["prs"] == []
        assert "gh" in data["reason"].lower()

    def test_attaches_brief_verdict_when_present(self, handler, rq_root):
        _write_brief(rq_root, 99, "a" * 40, verdict="needs_human_attention", confidence=3)
        prs = [_fake_pr(number=99, head_sha="a" * 40)]
        with patch.object(rq, "_run_gh", return_value=(0, json.dumps(prs), "")):
            mock_h = _mock_handler("GET")
            result = handler.handle("/api/v1/review-queue/prs", {}, mock_h)
        data = _parse(result)
        pr = data["prs"][0]
        assert pr["brief_present"] is True
        assert pr["verdict"] == "needs_human_attention"
        assert pr["confidence"] == 3

    def test_deferred_prs_flagged(self, handler, rq_root):
        deadline = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
        (rq_root / "deferred.json").write_text(
            json.dumps({"42": {"deferred_until": deadline}}), encoding="utf-8"
        )
        prs = [_fake_pr(number=42), _fake_pr(number=43)]
        with patch.object(rq, "_run_gh", return_value=(0, json.dumps(prs), "")):
            mock_h = _mock_handler("GET")
            result = handler.handle("/api/v1/review-queue/prs", {}, mock_h)
        data = _parse(result)
        by_n = {p["number"]: p for p in data["prs"]}
        assert by_n[42]["deferred"] is True
        assert by_n[43]["deferred"] is False
        assert data["deferred_count"] == 1
        assert data["visible"] == 1

    def test_expired_deferral_ignored(self, handler, rq_root):
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        (rq_root / "deferred.json").write_text(
            json.dumps({"42": {"deferred_until": past}}), encoding="utf-8"
        )
        prs = [_fake_pr(number=42)]
        with patch.object(rq, "_run_gh", return_value=(0, json.dumps(prs), "")):
            mock_h = _mock_handler("GET")
            result = handler.handle("/api/v1/review-queue/prs", {}, mock_h)
        data = _parse(result)
        assert data["prs"][0]["deferred"] is False


# ---------------------------------------------------------------------------
# GET /api/v1/review-queue/prs/{n}/brief
# ---------------------------------------------------------------------------


class TestGetBrief:
    def test_returns_brief_when_present(self, handler, rq_root):
        _write_brief(rq_root, 123, "b" * 40, verdict="approve_candidate", confidence=5)
        mock_h = _mock_handler("GET")
        result = handler.handle("/api/v1/review-queue/prs/123/brief", {}, mock_h)
        assert result.status_code == 200
        data = _parse(result)
        assert data["brief"]["pr_number"] == 123
        assert data["brief"]["verdict"] == "approve_candidate"

    def test_returns_404_when_absent(self, handler, rq_root):
        mock_h = _mock_handler("GET")
        result = handler.handle("/api/v1/review-queue/prs/999/brief", {}, mock_h)
        assert result.status_code == 404

    def test_invalid_pr_number_returns_400(self, handler, rq_root):
        mock_h = _mock_handler("GET")
        result = handler.handle("/api/v1/review-queue/prs/abc/brief", {}, mock_h)
        assert result.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/v1/review-queue/prs/{n}/approve
# ---------------------------------------------------------------------------


def _auth_user() -> MagicMock:
    user = MagicMock()
    user.user_id = "armand"
    user.email = "armand@example.com"
    return user


class TestApprove:
    def test_approve_shells_out_to_gh(self, handler, rq_root):
        mock_h = _mock_handler("POST", body=b'{"note":"LGTM","decision_seconds":12}')
        with patch.object(handler, "require_auth_or_error", return_value=(_auth_user(), None)):
            with patch.object(rq, "_run_gh", return_value=(0, "Reviewed PR #42", "")) as mrun:
                result = handler.handle_post("/api/v1/review-queue/prs/42/approve", {}, mock_h)
        assert result.status_code == 200
        # Verify gh invoked with correct args
        args, _ = mrun.call_args
        assert args[0][0] == "pr"
        assert "--approve" in args[0]
        assert "42" in args[0]
        assert "--body" in args[0]

        data = _parse(result)
        assert data["status"] == "ok"
        assert data["action"] == "approve"
        # Stats persisted
        stats_files = list(rq_root.glob("session-*.json"))
        assert len(stats_files) == 1
        stats_data = json.loads(stats_files[0].read_text())
        assert stats_data["approved"] == 1
        assert stats_data["streak"] == 1
        assert stats_data["decision_count"] == 1

    def test_approve_requires_auth(self, handler, rq_root):
        from aragora.server.handlers.utils.responses import HandlerResult

        err = HandlerResult(
            status_code=401, content_type="application/json", body=b'{"error":"no"}'
        )
        mock_h = _mock_handler("POST", body=b"{}")
        with patch.object(handler, "require_auth_or_error", return_value=(None, err)):
            result = handler.handle_post("/api/v1/review-queue/prs/42/approve", {}, mock_h)
        assert result.status_code == 401

    def test_approve_surfaces_gh_auth_error_as_403(self, handler, rq_root):
        mock_h = _mock_handler("POST", body=b"{}")
        with patch.object(handler, "require_auth_or_error", return_value=(_auth_user(), None)):
            with patch.object(rq, "_run_gh", return_value=(4, "", "authentication required")):
                result = handler.handle_post("/api/v1/review-queue/prs/42/approve", {}, mock_h)
        assert result.status_code == 403


class TestRequestChanges:
    def test_request_changes_requires_reason(self, handler, rq_root):
        mock_h = _mock_handler("POST", body=b'{"reason":""}')
        with patch.object(handler, "require_auth_or_error", return_value=(_auth_user(), None)):
            result = handler.handle_post("/api/v1/review-queue/prs/42/request-changes", {}, mock_h)
        assert result.status_code == 400

    def test_request_changes_shells_out_to_gh(self, handler, rq_root):
        mock_h = _mock_handler("POST", body=b'{"reason":"Tests missing","decision_seconds":30}')
        with patch.object(handler, "require_auth_or_error", return_value=(_auth_user(), None)):
            with patch.object(rq, "_run_gh", return_value=(0, "Changes requested", "")) as mrun:
                result = handler.handle_post(
                    "/api/v1/review-queue/prs/42/request-changes", {}, mock_h
                )
        assert result.status_code == 200
        data = _parse(result)
        assert data["action"] == "request_changes"
        args, _ = mrun.call_args
        assert "--request-changes" in args[0]
        assert "--body" in args[0]


class TestDefer:
    def test_defer_writes_local_state(self, handler, rq_root):
        mock_h = _mock_handler("POST", body=b'{"reason":"after coffee","hours":4}')
        with patch.object(handler, "require_auth_or_error", return_value=(_auth_user(), None)):
            # Defer must NOT call gh
            with patch.object(rq, "_run_gh") as mrun:
                result = handler.handle_post("/api/v1/review-queue/prs/42/defer", {}, mock_h)
        assert result.status_code == 200
        mrun.assert_not_called()
        data = _parse(result)
        assert data["status"] == "ok"
        assert data["action"] == "defer"
        assert "deferred_until" in data

        deferred = json.loads((rq_root / "deferred.json").read_text())
        assert "42" in deferred
        assert deferred["42"]["reason"] == "after coffee"
        assert "deferred_until" in deferred["42"]

    def test_defer_hides_from_next_list(self, handler, rq_root):
        # 1. Defer PR 42
        mock_h = _mock_handler("POST", body=b'{"reason":"later","hours":4}')
        with patch.object(handler, "require_auth_or_error", return_value=(_auth_user(), None)):
            with patch.object(rq, "_run_gh"):
                handler.handle_post("/api/v1/review-queue/prs/42/defer", {}, mock_h)

        # 2. Subsequent GET marks it deferred
        prs = [_fake_pr(number=42), _fake_pr(number=43)]
        with patch.object(rq, "_run_gh", return_value=(0, json.dumps(prs), "")):
            mock_h2 = _mock_handler("GET")
            result = handler.handle("/api/v1/review-queue/prs", {}, mock_h2)
        data = _parse(result)
        by_n = {p["number"]: p for p in data["prs"]}
        assert by_n[42]["deferred"] is True
        assert by_n[43]["deferred"] is False


# ---------------------------------------------------------------------------
# GET /api/v1/review-queue/stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_empty_stats(self, handler, rq_root):
        mock_h = _mock_handler("GET")
        result = handler.handle("/api/v1/review-queue/stats", {}, mock_h)
        assert result.status_code == 200
        data = _parse(result)
        stats = data["stats"]
        assert stats["approved"] == 0
        assert stats["request_changes"] == 0
        assert stats["deferred"] == 0
        assert stats["streak"] == 0
        assert stats["decision_count"] == 0
        assert stats["median_decision_seconds"] is None

    def test_median_after_several_actions(self, handler, rq_root):
        with patch.object(handler, "require_auth_or_error", return_value=(_auth_user(), None)):
            with patch.object(rq, "_run_gh", return_value=(0, "", "")):
                mock_h_a = _mock_handler("POST", body=b'{"decision_seconds":10}')
                handler.handle_post("/api/v1/review-queue/prs/42/approve", {}, mock_h_a)
                mock_h_b = _mock_handler("POST", body=b'{"decision_seconds":20}')
                handler.handle_post("/api/v1/review-queue/prs/43/approve", {}, mock_h_b)
        mock_h_stats = _mock_handler("GET")
        result = handler.handle("/api/v1/review-queue/stats", {}, mock_h_stats)
        data = _parse(result)
        stats = data["stats"]
        assert stats["approved"] == 2
        assert stats["decision_count"] == 2
        # Mean used as "median" proxy for v0 (single-user session).
        assert stats["median_decision_seconds"] == pytest.approx(15.0)
        assert stats["streak"] == 2

    def test_streak_resets_on_request_changes(self, handler, rq_root):
        with patch.object(handler, "require_auth_or_error", return_value=(_auth_user(), None)):
            with patch.object(rq, "_run_gh", return_value=(0, "", "")):
                handler.handle_post(
                    "/api/v1/review-queue/prs/42/approve", {}, _mock_handler("POST", b"{}")
                )
                handler.handle_post(
                    "/api/v1/review-queue/prs/43/approve", {}, _mock_handler("POST", b"{}")
                )
                handler.handle_post(
                    "/api/v1/review-queue/prs/44/request-changes",
                    {},
                    _mock_handler("POST", b'{"reason":"boom"}'),
                )
        result = handler.handle("/api/v1/review-queue/stats", {}, _mock_handler("GET"))
        data = _parse(result)
        assert data["stats"]["streak"] == 0
        assert data["stats"]["approved"] == 2
        assert data["stats"]["request_changes"] == 1


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_parse_pr_number_valid(self):
        assert rq._parse_pr_number("42") == 42
        assert rq._parse_pr_number("#123") == 123

    def test_parse_pr_number_invalid(self):
        assert rq._parse_pr_number("abc") is None
        assert rq._parse_pr_number("") is None
        assert rq._parse_pr_number("-1") is None
        assert rq._parse_pr_number(str(rq.MAX_PR_NUMBER + 1)) is None

    def test_subsystem_for(self):
        assert rq._subsystem_for("aragora/server/foo.py") == "aragora/server"
        assert rq._subsystem_for("docs/plans/x.md") == "docs"
        assert rq._subsystem_for("") == "(root)"

    def test_summarize_checks_mixed(self):
        checks = [
            {"conclusion": "SUCCESS", "status": "COMPLETED"},
            {"conclusion": "FAILURE", "status": "COMPLETED"},
            {"conclusion": "", "status": "IN_PROGRESS"},
            {"conclusion": "SKIPPED", "status": "COMPLETED"},  # ignored
        ]
        summary = rq._summarize_checks(checks)
        assert summary == {"success": 1, "failure": 1, "pending": 1, "total": 3}
