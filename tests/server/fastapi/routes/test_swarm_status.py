from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("ARAGORA_USE_SECRETS_MANAGER", "0")

from aragora.swarm.preflight import PreflightReceipt

_SWARM_STATUS_PATH = (
    Path(__file__).resolve().parents[4]
    / "aragora"
    / "server"
    / "fastapi"
    / "routes"
    / "swarm_status.py"
)
_SWARM_STATUS_SPEC = importlib.util.spec_from_file_location(
    "swarm_status_test_target",
    _SWARM_STATUS_PATH,
)
assert _SWARM_STATUS_SPEC is not None and _SWARM_STATUS_SPEC.loader is not None
swarm_status = importlib.util.module_from_spec(_SWARM_STATUS_SPEC)
_SWARM_STATUS_SPEC.loader.exec_module(swarm_status)


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
            "issue_title": None,
        }
    ]


def test_preflight_check_returns_receipt_and_admission_gate() -> None:
    fake_receipt = MagicMock()
    fake_receipt.to_dict.return_value = {"receipt_id": "receipt-1", "passed": True}
    fake_receipt.artifacts = {"expected_contract_checksum": "contract-sha"}
    fake_receipt.failure_terminal_class = SimpleNamespace(value="blocked_auth_failure")
    fake_gate = MagicMock()
    fake_gate.to_dict.return_value = {
        "gate_type": "dispatch_ready",
        "verdict": "blocked",
        "failure_classes": ["blocked_auth_failure"],
    }

    with (
        patch(
            "aragora.swarm.credential_envelope.CredentialEnvelope.from_environment",
            return_value=object(),
        ) as from_environment,
        patch(
            "aragora.swarm.preflight.run_contract_preflight_receipt",
            return_value=fake_receipt,
        ) as run_contract_preflight_receipt,
        patch(
            "aragora.swarm.preflight.evaluate_preflight_receipt_gate",
            return_value=fake_gate,
        ) as evaluate_preflight_receipt_gate,
    ):
        result = swarm_status.preflight_check(
            agent="codex",
            base_ref="origin/main",
            skip_publication=False,
            contract_path="~/tmp/contract.json",
        )

    assert result == {
        "mode": "swarm-preflight",
        "receipt": {"receipt_id": "receipt-1", "passed": True},
        "admission_gate": {
            "gate_type": "dispatch_ready",
            "verdict": "blocked",
            "failure_classes": ["blocked_auth_failure"],
        },
        "failure_terminal_class": "blocked_auth_failure",
    }
    from_environment.assert_called_once()
    assert run_contract_preflight_receipt.call_args.kwargs["repo_root"] == Path.cwd()
    assert run_contract_preflight_receipt.call_args.kwargs["agent"] == "codex"
    assert run_contract_preflight_receipt.call_args.kwargs["base_ref"] == "origin/main"
    assert run_contract_preflight_receipt.call_args.kwargs["skip_publication"] is False
    assert (
        run_contract_preflight_receipt.call_args.kwargs["contract_path"]
        == Path("~/tmp/contract.json").expanduser()
    )
    assert evaluate_preflight_receipt_gate.call_args.kwargs["check_type"] == "remote_publish"
    assert (
        evaluate_preflight_receipt_gate.call_args.kwargs["expected_contract_checksum"]
        == "contract-sha"
    )


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


def test_swarm_preflight_route_requires_contract_path() -> None:
    app = FastAPI()
    swarm_status.register_routes(app)
    client = TestClient(app)

    response = client.post("/api/v1/swarm/preflight")

    assert response.status_code == 422
    assert any(item["loc"][-1] == "contract_path" for item in response.json()["detail"])
