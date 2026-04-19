from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.review_queue import ReviewQueueHandler, ReviewQueueRow
from aragora.server.handlers.utils.responses import HandlerResult


def _parse_body(result: HandlerResult) -> dict:
    return json.loads(result.body)


def _make_handler(method: str = "GET", body: bytes = b"") -> MagicMock:
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {
        "Authorization": "Bearer test-token",
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
    }
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    return handler


@pytest.fixture
def handler() -> ReviewQueueHandler:
    return ReviewQueueHandler(ctx={})


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / ".aragora" / "review-queue").mkdir(parents=True)
    return root


def _row(number: int = 6308) -> ReviewQueueRow:
    return ReviewQueueRow(
        number=number,
        title=f"PR {number}",
        url=f"https://github.com/synaptent/aragora/pull/{number}",
        head_sha="abc123def456",
        author="an0mium",
        is_draft=False,
        mergeable="MERGEABLE",
        review_decision="REVIEW_REQUIRED",
        labels=["codex"],
        additions=12,
        deletions=3,
        changed_files=2,
        checks_summary="5/5 green",
        lane="ready_now",
        lane_reason="all green",
        created_at="2026-04-19T12:00:00Z",
        updated_at="2026-04-19T12:30:00Z",
        status_check_rollup=[
            {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
            {"name": "typecheck", "status": "COMPLETED", "conclusion": "SUCCESS"},
        ],
    )


class TestReviewQueueHandlerBasics:
    def test_can_handle_versioned_route(self, handler: ReviewQueueHandler) -> None:
        assert handler.can_handle("/api/v1/review-queue/prs") is True
        assert handler.can_handle("/api/v1/review-queue/stats") is True
        assert handler.can_handle("/api/v1/debates") is False


class TestManualBriefFallback:
    def test_brief_endpoint_reads_manual_markdown(
        self, handler: ReviewQueueHandler, repo_root: Path
    ) -> None:
        brief_file = repo_root / ".aragora" / "review-queue" / "manual-briefs-2026-04-19.md"
        brief_file.write_text(
            """
# Manual PDB Briefs

## #6308 · `fix(ci): cancel test-fast only on synchronize events`

**Verdict:** ✓ APPROVE · **Confidence:** 5/5 (single-model) · **Scope:** 1 line in `.github/workflows/test.yml`

**Logic:** Correct YAML.

**Security:** None.

**Maintainability:** Small follow-up likely.

**Skeptic:** If cancellations persist, root cause is elsewhere.

**Recommended action:** Approve first.
""".strip()
            + "\n",
            encoding="utf-8",
        )

        mock_http = _make_handler()
        with patch(
            "aragora.server.handlers.review_queue._current_repo_root", return_value=repo_root
        ):
            with patch(
                "aragora.server.handlers.review_queue._build_packet",
                return_value=SimpleNamespace(head_sha="abc123def456"),
            ):
                result = handler.handle("/api/v1/review-queue/prs/6308/brief", {}, mock_http)

        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["brief"]["source"] == "manual_markdown"
        assert body["brief"]["verdict"] == "approve_candidate"
        assert body["brief"]["confidence"] == 5
        assert body["brief"]["logic"] == "Correct YAML."


class TestListAndDefer:
    def test_list_excludes_deferred_prs_by_default(
        self, handler: ReviewQueueHandler, repo_root: Path
    ) -> None:
        deferred_path = repo_root / ".aragora" / "review-queue" / "deferred.json"
        deferred_path.write_text(
            json.dumps(
                {
                    "6308": {
                        "pr_number": 6308,
                        "reason": "later",
                        "deferred_at": "2026-04-19T12:00:00+00:00",
                        "deferred_until": "2099-04-19T16:00:00+00:00",
                    }
                }
            ),
            encoding="utf-8",
        )

        mock_http = _make_handler()
        with patch(
            "aragora.server.handlers.review_queue._current_repo_root", return_value=repo_root
        ):
            with patch(
                "aragora.server.handlers.review_queue._build_queue_rows",
                return_value=[_row(6308)],
            ):
                with patch(
                    "aragora.server.handlers.review_queue._build_packet",
                    return_value=SimpleNamespace(
                        touched_subsystems=("aragora/swarm",),
                        high_risk_paths_touched=(),
                        machine_recommendation="approve_candidate",
                        machine_recommendation_reason="all green",
                    ),
                ):
                    result = handler.handle("/api/v1/review-queue/prs", {}, mock_http)

        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["prs"] == []

    def test_defer_endpoint_persists_local_state(
        self, handler: ReviewQueueHandler, repo_root: Path
    ) -> None:
        mock_http = _make_handler("POST", body=b'{"reason":"needs fresh eyes"}')

        with patch(
            "aragora.server.handlers.review_queue._current_repo_root", return_value=repo_root
        ):
            result = handler.handle("/api/v1/review-queue/prs/6308/defer", {}, mock_http)

        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["deferred"]["pr_number"] == 6308
        saved = json.loads((repo_root / ".aragora" / "review-queue" / "deferred.json").read_text())
        assert saved["6308"]["reason"] == "needs fresh eyes"


class TestSettlement:
    def test_approve_endpoint_wraps_existing_settlement_flow(
        self, handler: ReviewQueueHandler, repo_root: Path
    ) -> None:
        packet = SimpleNamespace(
            pr_number=6308,
            url="https://github.com/synaptent/aragora/pull/6308",
            head_sha="abc123def456",
            base_sha="base123",
            queue_bucket="ready_now",
            machine_recommendation="approve_candidate",
            packet_sha="sha256:test",
        )
        receipt = {
            "pr_number": 6308,
            "action": "approve",
            "receipt_path": str(repo_root / ".aragora" / "review-queue" / "receipts" / "test.json"),
        }
        mock_http = _make_handler("POST", body=b"{}")

        with patch(
            "aragora.server.handlers.review_queue._current_repo_root", return_value=repo_root
        ):
            with patch(
                "aragora.server.handlers.review_queue._require_clean_worktree"
            ) as require_clean:
                with patch(
                    "aragora.server.handlers.review_queue._build_packet", return_value=packet
                ):
                    with patch(
                        "aragora.server.handlers.review_queue._settle_packet",
                        return_value=SimpleNamespace(to_dict=lambda: receipt),
                    ) as settle_packet:
                        result = handler.handle(
                            "/api/v1/review-queue/prs/6308/approve", {}, mock_http
                        )

        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["receipt"]["action"] == "approve"
        require_clean.assert_called_once_with(repo_root)
        settle_packet.assert_called_once()

    def test_request_changes_requires_reason(
        self, handler: ReviewQueueHandler, repo_root: Path
    ) -> None:
        mock_http = _make_handler("POST", body=b"{}")

        with patch(
            "aragora.server.handlers.review_queue._current_repo_root", return_value=repo_root
        ):
            result = handler.handle(
                "/api/v1/review-queue/prs/6308/request-changes",
                {},
                mock_http,
            )

        assert result is not None
        assert result.status_code == 400
        body = _parse_body(result)
        assert "requires a non-empty reason" in body["error"]
