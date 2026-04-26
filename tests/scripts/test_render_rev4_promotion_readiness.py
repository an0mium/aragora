from __future__ import annotations

import json
import sys
from pathlib import Path

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import render_rev4_promotion_readiness as mod  # noqa: E402


def _corpus(issue_ids: list[int]) -> dict[str, object]:
    return {
        "corpus_id": "tw-01-bounded-execution-v1",
        "revision": 4,
        "status": "staging",
        "recorded_on": "2026-04-18",
        "promotion_target": "docs/benchmarks/corpus.json",
        "issues": [
            {
                "issue_id": issue_id,
                "title": f"Issue {issue_id}",
                "execution_class": "missing_test_coverage"
                if issue_id % 2
                else "exception_narrowing",
                "scope_hint": ["example.py"],
                "source_reference": f"github.com/synaptent/aragora/issues/{issue_id}",
            }
            for issue_id in issue_ids
        ],
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_build_readiness_reports_gap_to_promotion_floor() -> None:
    readiness = mod.build_readiness(
        corpus=_corpus(list(range(1001, 1031))),
        metrics_rows=[
            {"issue_number": 1001, "terminal_class": "blocked_not_dispatch_bounded"},
            {"issue_number": 1003, "terminal_class": "deliverable_pr_created"},
        ],
        min_dispatched=3,
    )

    assert readiness["status"] == "needs_more_dispatch_evidence"
    assert readiness["needed_for_minimum"] == 1
    assert readiness["dispatch"]["dispatched_issue_ids"] == [1001, 1003]
    assert readiness["dispatch"]["missing_issue_ids"][:2] == [1002, 1004]
    assert readiness["dispatch"]["recommended_next_issue_ids"] == [1002]
    assert readiness["execution_classes"]["missing_test_coverage"] == {
        "total": 15,
        "dispatched": 2,
        "missing": 13,
    }
    assert readiness["execution_classes"]["exception_narrowing"] == {
        "total": 15,
        "dispatched": 0,
        "missing": 15,
    }


def test_build_readiness_reports_promotion_ready_at_floor() -> None:
    readiness = mod.build_readiness(
        corpus=_corpus(list(range(1001, 1031))),
        metrics_rows=[
            {"issue_number": 1001, "terminal_class": "blocked_not_dispatch_bounded"},
            {"issue_number": 1002, "terminal_class": "deliverable_pr_created"},
        ],
        min_dispatched=2,
    )

    assert readiness["status"] == "promotion_ready"
    assert readiness["needed_for_minimum"] == 0


def test_main_writes_markdown_readiness(tmp_path: Path) -> None:
    corpus_path = _write_json(tmp_path / "corpus.json", _corpus([1001, 1002, 1003, 1004]))
    metrics_path = _write_metrics(
        tmp_path / "boss_metrics.jsonl",
        [{"issue_number": 1001, "terminal_class": "deliverable_pr_created"}],
    )
    output_path = tmp_path / "readiness.md"

    exit_code = mod.main(
        [
            "--corpus",
            str(corpus_path),
            "--metrics",
            str(metrics_path),
            "--output",
            str(output_path),
            "--min-dispatched",
            "2",
            "--generated-at",
            "2026-04-25T00:00:00Z",
        ]
    )

    markdown = output_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "Last updated: 2026-04-25T00:00:00Z" in markdown
    assert "Status: `manifest_below_h1_floor`" in markdown
    assert "| Staged issues with metrics evidence | 1 |" in markdown
    assert "`#1002`" in markdown


def test_main_json_mode_emits_readiness(tmp_path: Path, capsys) -> None:
    corpus_path = _write_json(tmp_path / "corpus.json", _corpus(list(range(1001, 1031))))
    metrics_path = _write_metrics(
        tmp_path / "boss_metrics.jsonl",
        [{"issue_number": 1001, "terminal_class": "deliverable_pr_created"}],
    )

    exit_code = mod.main(
        [
            "--corpus",
            str(corpus_path),
            "--metrics",
            str(metrics_path),
            "--min-dispatched",
            "1",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "promotion_ready"
    assert payload["dispatch"]["dispatched_issue_count"] == 1
