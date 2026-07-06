from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import scripts.retrigger_cancelled_pr_runs as retrigger
from scripts.retrigger_cancelled_pr_runs import (
    compute_retriggerable_runs,
    main,
    prune_marker,
)

NOW = datetime(2026, 6, 6, 21, 0, 0, tzinfo=timezone.utc)
RECENT = "2026-06-06T20:59:00Z"  # 1 min before NOW
OLD = "2026-06-06T19:00:00Z"  # 2 h before NOW
PR_EVENTS = {"pull_request", "pull_request_target"}


def make_run(**over: Any) -> dict[str, Any]:
    run = {
        "id": 1,
        "event": "pull_request",
        "conclusion": "cancelled",
        "status": "completed",
        "head_branch": "feat/x",
        "head_sha": "sha-x",
        "run_attempt": 1,
        "run_number": 1,
        "workflow_id": 100,
        "name": "Portability Lint",
        "created_at": RECENT,
        "pull_requests": [{"number": 123}],
    }
    run.update(over)
    return run


def _compute(runs: list[dict[str, Any]], active_heads: dict[str, str], **kw: Any):
    params: dict[str, Any] = {
        "active_heads": active_heads,
        "cancel_events": PR_EVENTS,
        "now": NOW,
        "ttl_minutes": 60,
    }
    params.update(kw)
    return compute_retriggerable_runs(runs, **params)


def test_genuine_cancelled_non_superseded_is_selected() -> None:
    runs = [make_run(id=1, head_branch="feat/a", head_sha="sha-a")]
    eligible, reasons, candidates = _compute(runs, {"feat/a": "sha-a"})
    assert candidates == 1
    assert reasons == {}
    assert [e["run_id"] for e in eligible] == [1]
    assert eligible[0]["rerun_command"] == "gh run rerun 1"


def test_superseded_sha_is_skipped() -> None:
    runs = [make_run(id=2, head_branch="feat/b", head_sha="old-sha")]
    eligible, reasons, candidates = _compute(runs, {"feat/b": "new-sha"})
    assert eligible == []
    assert reasons == {"superseded-sha": 1}
    assert candidates == 1


def test_draft_or_closed_branch_is_skipped() -> None:
    # Draft PRs are excluded from active_heads upstream, so the branch is absent.
    runs = [make_run(id=3, head_branch="feat/c", head_sha="sha-c")]
    eligible, reasons, _ = _compute(runs, {})
    assert eligible == []
    assert reasons == {"draft-or-closed": 1}


def test_ttl_expired_is_skipped() -> None:
    runs = [make_run(id=4, head_branch="feat/d", head_sha="sha-d", created_at=OLD)]
    eligible, reasons, _ = _compute(runs, {"feat/d": "sha-d"})
    assert eligible == []
    assert reasons == {"ttl-expired": 1}


def test_loop_guard_marker_is_honored() -> None:
    runs = [make_run(id=5, head_branch="feat/e", head_sha="sha-e")]
    eligible, reasons, _ = _compute(runs, {"feat/e": "sha-e"}, already_retriggered={5})
    assert eligible == []
    assert reasons == {"already-retriggered": 1}


def test_superseded_by_newer_run_is_skipped() -> None:
    runs = [
        make_run(
            id=6,
            head_branch="feat/f",
            head_sha="sha-f",
            run_number=1,
            created_at="2026-06-06T20:50:00Z",
        ),
        make_run(
            id=7,
            head_branch="feat/f",
            head_sha="sha-f",
            run_number=2,
            conclusion="success",
            created_at="2026-06-06T20:55:00Z",
        ),
    ]
    eligible, reasons, candidates = _compute(runs, {"feat/f": "sha-f"})
    assert eligible == []
    assert reasons == {"superseded-by-newer-run": 1}
    # only the cancelled run is a candidate; the success sibling is not counted
    assert candidates == 1


def test_newer_non_pr_sibling_does_not_supersede() -> None:
    # A newer push/workflow_dispatch run on the same branch+workflow+SHA must not
    # suppress re-running a still-current cancelled PR run.
    runs = [
        make_run(
            id=20,
            head_branch="feat/p",
            head_sha="sha-p",
            run_number=1,
            created_at="2026-06-06T20:50:00Z",
        ),
        make_run(
            id=21,
            event="push",
            conclusion="success",
            head_branch="feat/p",
            head_sha="sha-p",
            run_number=2,
            created_at="2026-06-06T20:55:00Z",
        ),
    ]
    eligible, reasons, candidates = _compute(runs, {"feat/p": "sha-p"})
    assert [e["run_id"] for e in eligible] == [20]
    assert reasons == {}
    assert candidates == 1


def test_newer_different_sha_sibling_does_not_supersede() -> None:
    # A newer run for a *different* head SHA on the same branch+workflow must not
    # suppress the cancelled run that still matches the current PR head.
    runs = [
        make_run(
            id=22,
            head_branch="feat/q",
            head_sha="sha-q",
            run_number=1,
            created_at="2026-06-06T20:50:00Z",
        ),
        make_run(
            id=23,
            head_branch="feat/q",
            head_sha="other-sha",
            conclusion="success",
            run_number=2,
            created_at="2026-06-06T20:55:00Z",
        ),
    ]
    eligible, reasons, candidates = _compute(runs, {"feat/q": "sha-q"})
    assert [e["run_id"] for e in eligible] == [22]
    assert reasons == {}
    assert candidates == 1


def test_max_attempts_guard_is_honored() -> None:
    runs = [make_run(id=8, head_branch="feat/g", head_sha="sha-g", run_attempt=2)]
    eligible, reasons, _ = _compute(runs, {"feat/g": "sha-g"}, max_attempts=2)
    assert eligible == []
    assert reasons == {"max-attempts": 1}


def test_non_pr_and_non_cancelled_runs_are_not_candidates() -> None:
    runs = [
        make_run(id=9, event="push", head_branch="feat/h", head_sha="sha-h"),
        make_run(id=10, conclusion="success", head_branch="feat/h", head_sha="sha-h"),
    ]
    eligible, reasons, candidates = _compute(runs, {"feat/h": "sha-h"})
    assert eligible == []
    assert reasons == {}
    assert candidates == 0


def test_prune_marker_drops_old_entries() -> None:
    data = {"1": RECENT, "2": "2026-06-05T10:00:00Z"}
    pruned = prune_marker(data, now=NOW, retention_hours=24)
    assert pruned == {"1": RECENT}


def test_main_scopes_to_pr_and_writes_receipt(tmp_path, monkeypatch, capsys) -> None:
    class FakeClient:
        def __init__(self, repo: str, token: str) -> None:
            assert repo == "synaptent/aragora"
            assert token == "token"

        def get_pull(self, pr_number: int) -> dict[str, Any]:
            assert pr_number == 123
            return {
                "state": "open",
                "draft": False,
                "head": {"ref": "feat/a", "sha": "sha-a"},
            }

        def list_recent_workflow_runs(
            self,
            max_runs: int,
            *,
            branch: str | None = None,
            event: str | None = None,
        ) -> list[dict[str, Any]]:
            assert max_runs == 300
            assert branch == "feat/a"
            if event != "pull_request":
                return []
            return [
                make_run(id=31, head_branch="feat/a", head_sha="sha-a"),
                make_run(
                    id=32,
                    head_branch="feat/a",
                    head_sha="sha-a",
                    pull_requests=[{"number": 456}],
                ),
            ]

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(retrigger, "GitHubClient", FakeClient)

    rc = main(
        [
            "--repo",
            "synaptent/aragora",
            "--pr",
            "123",
            "--ttl-minutes",
            "100000",
            "--receipt-dir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["pr"] == 123
    assert summary["scope"] == "pr-123"
    assert summary["scoped_branch"] == "feat/a"
    assert summary["scoped_sha"] == "sha-a"
    assert summary["scanned"] == 1
    assert summary["eligible"] == 1
    assert [run["run_id"] for run in summary["eligible_runs"]] == [31]
    assert summary["dry_run"] is True

    receipt_path = tmp_path / summary["receipt"].split("/")[-1]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == "retrigger-cancelled-pr-runs-receipt/v1"
    assert receipt["repo"] == "synaptent/aragora"
    assert receipt["scope"] == "pr-123"
    assert receipt["dry_run"] is True
    assert receipt["eligible_run_ids"] == [31]
    assert receipt["head_shas"] == ["sha-a"]


def test_main_apply_records_rerun_results_in_receipt(tmp_path, monkeypatch, capsys) -> None:
    class FakeClient:
        def __init__(self, repo: str, token: str) -> None:
            self.repo = repo

        def list_open_pulls(self) -> list[dict[str, Any]]:
            return [
                {
                    "state": "open",
                    "draft": False,
                    "head": {"ref": "feat/a", "sha": "sha-a"},
                }
            ]

        def list_recent_workflow_runs(self, max_runs: int) -> list[dict[str, Any]]:
            return [make_run(id=41, head_branch="feat/a", head_sha="sha-a")]

        def rerun_workflow_run(self, run_id: int) -> tuple[bool, str]:
            assert run_id == 41
            return True, "rerun_requested"

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(retrigger, "GitHubClient", FakeClient)

    rc = main(
        [
            "--repo",
            "synaptent/aragora",
            "--ttl-minutes",
            "100000",
            "--apply",
            "--receipt-dir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["dry_run"] is False
    assert summary["applied"] == 1
    assert summary["rerun_results"] == [{"run_id": 41, "ok": True, "message": "rerun_requested"}]

    receipt_path = tmp_path / summary["receipt"].split("/")[-1]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["dry_run"] is False
    assert receipt["rerun_run_ids"] == [41]


def test_client_classifies_app_token_rerun_permission_failure(monkeypatch) -> None:
    client = retrigger.GitHubClient(repo="synaptent/aragora", token="token")

    def fail_post(path: str, payload: dict[str, Any] | None = None) -> None:
        assert path == "/repos/synaptent/aragora/actions/runs/61/rerun"
        assert payload is None
        raise retrigger.GitHubApiError(
            "GitHub API POST /rerun failed: 403 Forbidden\n"
            '{"message":"Resource not accessible by integration"}'
        )

    monkeypatch.setattr(client, "post", fail_post)

    ok, message = client.rerun_workflow_run(61)

    assert ok is False
    assert message == retrigger.APP_TOKEN_RERUN_PERMISSION_RESULT


def test_main_apply_permission_denied_writes_human_packet_and_receipt(
    tmp_path, monkeypatch, capsys
) -> None:
    class FakeClient:
        def __init__(self, repo: str, token: str) -> None:
            self.repo = repo

        def list_open_pulls(self) -> list[dict[str, Any]]:
            return [
                {
                    "state": "open",
                    "draft": False,
                    "head": {"ref": "feat/a", "sha": "sha-a"},
                }
            ]

        def list_recent_workflow_runs(self, max_runs: int) -> list[dict[str, Any]]:
            return [
                make_run(
                    id=61,
                    name="Portability Lint",
                    head_branch="feat/a",
                    head_sha="sha-a",
                )
            ]

        def rerun_workflow_run(self, run_id: int) -> tuple[bool, str]:
            assert run_id == 61
            return False, retrigger.APP_TOKEN_RERUN_PERMISSION_RESULT

    receipt_dir = tmp_path / "receipts"
    packet_dir = tmp_path / "operator-packets"

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(retrigger, "GitHubClient", FakeClient)

    rc = main(
        [
            "--repo",
            "synaptent/aragora",
            "--ttl-minutes",
            "100000",
            "--apply",
            "--receipt-dir",
            str(receipt_dir),
            "--operator-packet-dir",
            str(packet_dir),
        ]
    )

    assert rc == retrigger.OPERATOR_ACTION_EXIT
    summary = json.loads(capsys.readouterr().out)
    assert summary["applied"] == 0
    assert summary["apply_failed"] == 1
    assert summary["operator_action_required"] is True
    assert summary["permission_denied_reruns"][0]["run_id"] == 61
    assert summary["human_rerun_commands"] == ["gh run rerun 61 --failed"]

    packet_path = packet_dir / summary["operator_packet"].split("/")[-1]
    packet = packet_path.read_text(encoding="utf-8")
    assert "Resource not accessible by integration" in packet
    assert "https://github.com/synaptent/aragora/actions/runs/61" in packet
    assert "`gh run rerun 61 --failed`" in packet

    receipt_path = receipt_dir / summary["receipt"].split("/")[-1]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["operator_action_required"] is True
    assert receipt["permission_denied_run_ids"] == [61]
    assert receipt["human_rerun_commands"] == ["gh run rerun 61 --failed"]
    assert receipt["operator_packet"] == summary["operator_packet"]


def test_receipt_write_failure_is_reported_without_failing(monkeypatch, capsys) -> None:
    class FakeClient:
        def __init__(self, repo: str, token: str) -> None:
            self.repo = repo

        def list_open_pulls(self) -> list[dict[str, Any]]:
            return [
                {
                    "state": "open",
                    "draft": False,
                    "head": {"ref": "feat/a", "sha": "sha-a"},
                }
            ]

        def list_recent_workflow_runs(self, max_runs: int) -> list[dict[str, Any]]:
            return [make_run(id=51, head_branch="feat/a", head_sha="sha-a")]

    def fail_receipt(**_: Any) -> str:
        raise OSError("receipt path unavailable")

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(retrigger, "GitHubClient", FakeClient)
    monkeypatch.setattr(retrigger, "_write_receipt", fail_receipt)

    rc = main(["--repo", "synaptent/aragora", "--ttl-minutes", "100000"])

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["eligible"] == 1
    assert summary["receipt"] == ""
    assert summary["receipt_error"] == "receipt path unavailable"
