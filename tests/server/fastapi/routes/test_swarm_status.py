from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import FastAPI

os.environ.setdefault("ARAGORA_USE_SECRETS_MANAGER", "0")

from aragora.server.fastapi.routes import swarm_status
from aragora.swarm.preflight import PreflightReceipt


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_swarm_status_summary_counts_deliverables_as_success(tmp_path: Path) -> None:
    metrics_path = tmp_path / "boss_metrics.jsonl"
    _write_jsonl(
        metrics_path,
        [
            {
                "timestamp": "2026-04-14T02:20:00Z",
                "issue_number": 101,
                "terminal_class": "deliverable_pr_created",
                "outcome": "completed",
                "elapsed_seconds": 12,
            },
            {
                "timestamp": "2026-04-14T02:21:00Z",
                "issue_number": 102,
                "terminal_class": "success_pr_created",
                "outcome": "completed",
                "elapsed_seconds": 18,
            },
        ],
    )

    summary = swarm_status.swarm_status_summary(metrics_path=metrics_path)

    assert summary["status"] == "active"
    assert summary["total_ticks"] == 2
    assert summary["unique_issues_attempted"] == 2
    assert summary["unique_issues_succeeded"] == 2
    assert summary["success_rate"] == 1.0
    assert summary["tick_success_rate"] == 1.0
    assert summary["terminal_class_distribution"] == {
        "deliverable_pr_created": 1,
        "success_pr_created": 1,
    }
    assert summary["latest_tick"]["issue_number"] == 102


def test_swarm_status_summary_uses_issue_truth_for_success_rate(tmp_path: Path) -> None:
    metrics_path = tmp_path / "boss_metrics.jsonl"
    _write_jsonl(
        metrics_path,
        [
            {
                "timestamp": "2026-04-14T02:20:00Z",
                "issue_number": 101,
                "terminal_class": "blocked_auth_failure",
                "outcome": "needs_human",
                "elapsed_seconds": 12,
            },
            {
                "timestamp": "2026-04-14T02:21:00Z",
                "issue_number": 101,
                "terminal_class": "deliverable_pr_created",
                "outcome": "completed",
                "elapsed_seconds": 18,
            },
        ],
    )

    summary = swarm_status.swarm_status_summary(metrics_path=metrics_path)

    assert summary["unique_issues_attempted"] == 1
    assert summary["unique_issues_succeeded"] == 1
    assert summary["success_rate"] == 1.0
    assert summary["tick_success_rate"] == 0.5
    assert summary["recent_blockers"] == [
        {
            "issue_number": 101,
            "terminal_class": "blocked_auth_failure",
            "failure_reason": None,
            "blocker_kind": None,
            "blocker_evidence": None,
            "issue_title": None,
        }
    ]


def test_swarm_status_summary_surfaces_compact_blocker_evidence(tmp_path: Path) -> None:
    metrics_path = tmp_path / "boss_metrics.jsonl"
    _write_jsonl(
        metrics_path,
        [
            {
                "timestamp": "2026-04-14T02:20:00Z",
                "issue_number": 201,
                "terminal_class": "blocked_auth_failure",
                "outcome": "needs_human",
                "failure_reason": "auth",
                "blocker_kind": "credentials",
                "blocker_evidence": "  API key missing   for Anthropic provider  " + ("x" * 260),
                "issue_title": "Repair auth preflight",
                "elapsed_seconds": 12,
            }
        ],
    )

    summary = swarm_status.swarm_status_summary(metrics_path=metrics_path)

    assert summary["recent_blockers"] == [
        {
            "issue_number": 201,
            "terminal_class": "blocked_auth_failure",
            "failure_reason": "auth",
            "blocker_kind": "credentials",
            "blocker_evidence": "API key missing for Anthropic provider " + ("x" * 198) + "...",
            "issue_title": "Repair auth preflight",
        }
    ]


def test_preflight_check_returns_receipt_dict() -> None:
    fake_receipt = MagicMock()
    fake_receipt.to_dict.return_value = {"receipt_id": "receipt-1", "passed": True}

    with (
        patch(
            "aragora.swarm.credential_envelope.CredentialEnvelope.from_environment",
            return_value=object(),
        ) as from_environment,
        patch("aragora.swarm.preflight.run_preflight", return_value=fake_receipt) as run_preflight,
    ):
        result = swarm_status.preflight_check(
            agent="codex",
            base_ref="origin/main",
            skip_publication=False,
        )

    assert result == {"receipt_id": "receipt-1", "passed": True}
    from_environment.assert_called_once()
    assert run_preflight.call_args.kwargs["repo_root"] == Path.cwd()
    assert run_preflight.call_args.kwargs["agent"] == "codex"
    assert run_preflight.call_args.kwargs["base_ref"] == "origin/main"
    assert run_preflight.call_args.kwargs["skip_publication"] is False


def test_list_preflight_receipts_returns_empty_without_receipt_dir(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    assert swarm_status.list_preflight_receipts() == []


def test_list_preflight_receipts_reads_valid_receipts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    receipt_dir = tmp_path / ".aragora" / "receipts" / "preflight"
    receipt_dir.mkdir(parents=True)

    receipt = PreflightReceipt(
        receipt_id="preflight-scratch-1",
        envelope_seal="seal-1",
        repo_root=str(tmp_path),
        check_type="scratch_validation",
        started_at="2026-04-14T02:20:00Z",
        finished_at="2026-04-14T02:21:00Z",
        passed=True,
        cache_key="scratch-key",
        ttl_seconds=600,
        expires_at="2026-04-14T02:31:00Z",
    )
    (receipt_dir / "receipt-ok.json").write_text(
        json.dumps(receipt.to_dict()),
        encoding="utf-8",
    )
    (receipt_dir / "receipt-bad.json").write_text("{not-json", encoding="utf-8")

    receipts = swarm_status.list_preflight_receipts()

    assert receipts == [
        {
            "receipt_id": "preflight-scratch-1",
            "check_type": "scratch_validation",
            "passed": True,
            "started_at": "2026-04-14T02:20:00Z",
            "finished_at": "2026-04-14T02:21:00Z",
            "expires_at": "2026-04-14T02:31:00Z",
            "cache_key": "scratch-key",
        }
    ]


def test_register_routes_adds_swarm_status_endpoints() -> None:
    app = FastAPI()

    swarm_status.register_routes(app)

    route_paths = {route.path for route in app.routes}
    assert "/api/v1/swarm/status" in route_paths
    assert "/api/v1/swarm/preflight" in route_paths
    assert "/api/v1/swarm/preflight/receipts" in route_paths
