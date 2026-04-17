from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import run_benchmark_corpus_recurrence as mod  # noqa: E402


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_metrics(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_filter_open_issue_numbers_only_keeps_open_issues() -> None:
    def runner(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
        cwd: str | None = None,
    ) -> SimpleNamespace:
        del capture_output, text, check, cwd
        issue_number = int(cmd[3])
        state = "OPEN" if issue_number in {1001, 1003} else "CLOSED"
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"number": issue_number, "state": state}),
            stderr="",
        )

    assert mod.filter_open_issue_numbers(
        "synaptent/aragora",
        [1001, 1002, 1003],
        runner=runner,
    ) == [1001, 1003]


def test_build_boss_loop_command_uses_explicit_issue_list_contract() -> None:
    command = mod.build_boss_loop_command(
        repo="synaptent/aragora",
        issue_numbers=[1064, 2712],
    )

    assert command[:4] == [sys.executable, "-m", "aragora.cli.main", "swarm"]
    assert "--boss-issue-list" in command
    assert command[command.index("--boss-issue-list") + 1] == "1064,2712"
    assert command[command.index("--max-ticks") + 1] == "6"
    assert command[command.index("--max-consecutive-failures") + 1] == "5"
    assert command[command.index("--autonomy") + 1] == "fire_and_forget"


def test_run_recurrence_rotates_metrics_appends_closed_rows_and_dispatches_open_issue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus_path = _write_json(
        tmp_path / "corpus.json",
        {
            "corpus_id": "tw-01-bounded-execution-v1",
            "revision": 8,
            "recorded_on": "2026-04-14",
            "success_contract": "mergeable_pr_or_merged_pr",
            "issues": [
                {"issue_id": 1064, "title": "Closed issue"},
                {"issue_id": 2712, "title": "Open issue"},
            ],
        },
    )
    metrics_path = tmp_path / "boss_metrics.jsonl"
    metrics_path.write_text('{"old": true}\n', encoding="utf-8")

    def fake_append_iteration_metrics(
        *,
        metrics_jsonl_path: str | None,
        outcome_learner_window: int,
        deferred_queue_depth: int,
        iteration: int,
        issue_number: int | None,
        worker_result: dict[str, object],
        elapsed_seconds: float,
        files_changed: int,
        tests_run: int,
        tests_passed: int,
    ) -> None:
        del outcome_learner_window, deferred_queue_depth, iteration, elapsed_seconds
        del files_changed, tests_run, tests_passed
        assert metrics_jsonl_path is not None
        payload = {
            "issue_number": issue_number,
            "issue_title": worker_result.get("issue_title"),
            "worker_status": worker_result.get("status"),
            "worker_outcome": worker_result.get("outcome"),
            "terminal_class": "issue_already_resolved",
        }
        with Path(metrics_jsonl_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    monkeypatch.setattr(mod, "append_iteration_metrics", fake_append_iteration_metrics)

    def runner(
        cmd: list[str],
        *,
        capture_output: bool = False,
        text: bool = False,
        check: bool = False,
        cwd: str | None = None,
    ) -> SimpleNamespace:
        del capture_output, text, check, cwd
        if cmd[:3] == ["gh", "issue", "view"]:
            issue_number = int(cmd[3])
            state = "OPEN" if issue_number == 2712 else "CLOSED"
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"number": issue_number, "state": state}),
                stderr="",
            )

        assert cmd[:4] == [sys.executable, "-m", "aragora.cli.main", "swarm"]
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "issue_number": 2712,
                        "issue_title": "Open issue",
                        "worker_status": "completed",
                        "worker_outcome": "pr_adopted",
                        "publish_action": "pr_created",
                        "terminal_class": "deliverable_pr_created",
                    }
                )
                + "\n"
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    summary = mod.run_recurrence(
        corpus_path=corpus_path,
        repo="synaptent/aragora",
        metrics_file=metrics_path,
        runner=runner,
    )

    assert summary["open_issue_numbers"] == [2712]
    assert summary["closed_issue_numbers"] == [1064]
    assert summary["synthetic_resolved_issue_numbers"] == [1064]
    assert summary["recorded_issue_numbers"] == [1064, 2712]
    assert summary["missing_issue_numbers"] == []
    assert summary["rotated_metrics"]["archived_existing_file"] is True
    archived_path = Path(str(summary["rotated_metrics"]["archive_path"]))
    assert archived_path.exists()

    payloads = _load_metrics(metrics_path)
    assert [payload["issue_number"] for payload in payloads] == [1064, 2712]
    assert payloads[0]["terminal_class"] == "issue_already_resolved"
    assert payloads[1]["terminal_class"] == "deliverable_pr_created"


def test_run_recurrence_raises_when_open_issue_still_missing_from_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus_path = _write_json(
        tmp_path / "corpus.json",
        {
            "corpus_id": "tw-01-bounded-execution-v1",
            "revision": 8,
            "recorded_on": "2026-04-14",
            "success_contract": "mergeable_pr_or_merged_pr",
            "issues": [{"issue_id": 2712, "title": "Open issue"}],
        },
    )
    metrics_path = tmp_path / "boss_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    def fake_append_iteration_metrics(
        *,
        metrics_jsonl_path: str | None,
        outcome_learner_window: int,
        deferred_queue_depth: int,
        iteration: int,
        issue_number: int | None,
        worker_result: dict[str, object],
        elapsed_seconds: float,
        files_changed: int,
        tests_run: int,
        tests_passed: int,
    ) -> None:
        del metrics_jsonl_path, outcome_learner_window, deferred_queue_depth, iteration
        del issue_number, worker_result, elapsed_seconds, files_changed, tests_run, tests_passed

    monkeypatch.setattr(mod, "append_iteration_metrics", fake_append_iteration_metrics)

    def runner(
        cmd: list[str],
        *,
        capture_output: bool = False,
        text: bool = False,
        check: bool = False,
        cwd: str | None = None,
    ) -> SimpleNamespace:
        del capture_output, text, check, cwd
        if cmd[:3] == ["gh", "issue", "view"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"number": 2712, "state": "OPEN"}),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="missing issue numbers 2712"):
        mod.run_recurrence(
            corpus_path=corpus_path,
            repo="synaptent/aragora",
            metrics_file=metrics_path,
            runner=runner,
        )
