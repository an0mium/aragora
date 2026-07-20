from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_scripts_dir = str(_REPO_ROOT / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import render_decision_integrity_dogfood_dashboard as mod  # noqa: E402

from aragora.gauntlet.receipt_models import DecisionReceipt  # noqa: E402


def _completed(
    args: Sequence[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(list(args), returncode, stdout=stdout, stderr=stderr)


def _valid_receipt_payload(receipt_id: str) -> dict[str, object]:
    receipt = DecisionReceipt(
        receipt_id=receipt_id,
        gauntlet_id=f"gauntlet-{receipt_id}",
        timestamp="2026-07-02T00:00:00Z",
        input_summary="settlement",
        input_hash="input-hash",
        risk_summary={"critical": 0, "high": 0, "medium": 0, "low": 0},
        attacks_attempted=0,
        attacks_successful=0,
        probes_run=0,
        vulnerabilities_found=0,
        verdict="PASS",
        confidence=0.9,
        robustness_score=1.0,
    )
    return receipt.to_dict()


class _FakeRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.receipts = {
            path: json.dumps(_valid_receipt_payload(path.split("/")[-1]))
            for path in mod.SETTLEMENT_RECEIPT_PATHS
        }

    def __call__(self, args: Sequence[str], timeout: int) -> subprocess.CompletedProcess[str]:
        del timeout
        command = list(args)
        self.commands.append(command)
        if command[:5] == ["gh", "api", "-X", "GET", "search/issues"]:
            query = next(item.removeprefix("q=") for item in command if item.startswith("q="))
            if query.endswith("is:merged merged:2026-06-05..2026-07-05"):
                return _completed(command, stdout="10\n")
            if '"independent model review"' in query:
                return _completed(command, stdout="8\n")
            if '"Verdict: PASS"' in query:
                return _completed(command, stdout="7\n")
            if '"merge-quorum"' in query:
                return _completed(command, stdout="6\n")
            if '"CHANGES-REQUESTED"' in query:
                return _completed(command, stdout="3\n")
            if '"[P0]"' in query:
                return _completed(command, stdout="0\n")
            if '"[P1]"' in query:
                return _completed(command, stdout="2\n")
            if '"[P2]"' in query:
                return _completed(command, stdout="4\n")
            if '"[P3]"' in query:
                return _completed(command, stdout="1\n")
            if '"exact-head"' in query:
                return _completed(command, stdout="5\n")
            return _completed(command, stdout="0\n")
        if command == ["git", "fetch", "origin", mod.SETTLEMENT_RECEIPT_BRANCH]:
            return _completed(command, stdout="")
        if command[:2] == ["git", "show"]:
            path = command[2].removeprefix("FETCH_HEAD:")
            return _completed(command, stdout=self.receipts[path])
        return _completed(command, returncode=1, stderr="unexpected command")


def _metric(payload: dict[str, object], metric_id: str) -> dict[str, object]:
    metrics = payload["metrics"]
    assert isinstance(metrics, list)
    for item in metrics:
        assert isinstance(item, dict)
        if item["metric_id"] == metric_id:
            return item
    raise AssertionError(f"missing metric {metric_id}")


def test_build_payload_collects_report_only_metrics_and_receipts(tmp_path: Path) -> None:
    receipt_root = tmp_path / "receipts"
    receipt_root.mkdir()
    (receipt_root / "MERGE_EXECUTOR_RECEIPT_20260705T000000Z_PR9001.json").write_text(
        json.dumps(
            {
                "schema": "merge-executor-receipt/v1",
                "pr": 9001,
                "head_sha": "abc123",
                "packet_entry": {"head_sha": "abc123"},
            }
        ),
        encoding="utf-8",
    )
    (receipt_root / "ignore.json").write_text('{"schema":"other"}', encoding="utf-8")
    runner = _FakeRunner()

    payload = mod.build_payload(
        repo="synaptent/aragora",
        window_start="2026-06-05",
        window_end="2026-07-05",
        now=mod._parse_utc_timestamp("2026-07-05T12:00:00Z"),
        runner=runner,
        receipt_roots=[receipt_root],
    )

    assert payload["report_only"] is True
    assert _metric(payload, "merged_prs")["value"] == "10"
    assert _metric(payload, "independent_model_review_coverage")["value"] == "8/10 (80.0%)"
    assert _metric(payload, "exact_head_marker_coverage")["value"] == "5/10 (50.0%)"
    assert _metric(payload, "settlement_receipts_verified")["value"] == "3/3"
    local = _metric(payload, "operator_local_merge_executor_receipts")
    assert local["value"] == "1"
    details = local["details"]
    assert isinstance(details, dict)
    assert details["exact_head_receipt_count"] == 1
    assert all(command[:3] != ["gh", "issue", "comment"] for command in runner.commands)
    assert all(command[:3] != ["gh", "pr", "comment"] for command in runner.commands)


def test_search_command_failure_marks_metric_failed() -> None:
    def failing_runner(args: Sequence[str], timeout: int) -> subprocess.CompletedProcess[str]:
        del timeout
        return _completed(args, returncode=1, stderr="rate limited")

    metrics = mod.collect_github_search_metrics(
        repo="synaptent/aragora",
        window_start="2026-06-05",
        window_end="2026-07-05",
        now=mod._parse_utc_timestamp("2026-07-05T12:00:00Z"),
        runner=failing_runner,
    )

    assert metrics
    assert {metric.status for metric in metrics} == {"failed"}
    assert all("do not carry forward" in metric.failure_behavior for metric in metrics)
    assert all(metric.error == "rate limited" for metric in metrics)


def test_stale_metric_is_labeled_without_changing_value() -> None:
    metric = mod.DashboardMetric(
        metric_id="x",
        label="X",
        value="42",
        status="ok",
        last_updated_at="2026-07-01T00:00:00Z",
        source_query="query",
        command="command",
        stale_after_hours=24,
        failure_behavior="fail closed",
    )

    out = mod._apply_staleness(
        metric,
        now=mod._parse_utc_timestamp("2026-07-05T00:00:00Z"),
    )

    assert out.value == "42"
    assert out.status == "stale"
    assert "freshness SLA" in out.caveat


def test_render_markdown_surfaces_gap_and_regenerate_command() -> None:
    payload = {
        "generated_at": "2026-07-05T12:00:00Z",
        "repo": "synaptent/aragora",
        "issue": 8861,
        "source_artifact": mod.SOURCE_ARTIFACT,
        "window": {"start": "2026-06-05", "end": "2026-07-05"},
        "metrics": [
            {
                "metric_id": "operator_local_merge_executor_receipts",
                "label": "Operator-local merge-executor receipts observed",
                "value": "0",
                "status": "missing",
                "last_updated_at": "2026-07-05T12:00:00Z",
                "source_query": "/tmp/receipts",
                "command": "find /tmp/receipts",
                "stale_after_hours": 168,
                "failure_behavior": "mark missing",
                "caveat": "local-only",
                "details": {"receipt_count": 0, "exact_head_receipt_count": 0},
            }
        ],
        "known_gaps": [
            {
                "metric_id": "exact_head_marker_coverage",
                "status": "limited",
                "gap": "phrase-marker proxy only",
            }
        ],
    }

    markdown = mod.render_markdown(payload)

    assert "# Decision-Integrity Dogfood Dashboard" in markdown
    assert "Report-only generated companion" in markdown
    assert "`exact_head_marker_coverage` status `limited`: phrase-marker proxy only" in markdown
    assert "python3 scripts/render_decision_integrity_dogfood_dashboard.py" in markdown
