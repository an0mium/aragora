"""Tests for scripts/boss_metrics_health.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import boss_metrics_health as mod  # noqa: E402


def _write(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    p = tmp_path / "boss_metrics.jsonl"
    p.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return p


def test_load_metrics_skips_malformed_lines(tmp_path: Path) -> None:
    p = tmp_path / "boss.jsonl"
    p.write_text(
        '{"issue_number": 1, "terminal_class": "blocked_auth_failure"}\n'
        "this is not json\n"
        '{"issue_number": 2, "terminal_class": "deliverable_pr_created"}\n',
        encoding="utf-8",
    )
    rows = mod.load_metrics(p)
    assert len(rows) == 2
    assert rows[0]["issue_number"] == 1
    assert rows[1]["issue_number"] == 2


def test_load_metrics_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mod.load_metrics(tmp_path / "missing.jsonl")


def test_row_is_skip_dispatch_skip_reason() -> None:
    assert mod._row_is_skip({"dispatch_skip_reason": "no_work_orders"})


def test_row_is_skip_terminal_class_in_skip_set() -> None:
    assert mod._row_is_skip({"terminal_class": "blocked_auth_failure"})
    assert mod._row_is_skip({"terminal_class": "rescue_no_deliverable"})


def test_row_is_skip_terminal_class_deliverable_not_skip() -> None:
    assert not mod._row_is_skip({"terminal_class": "deliverable_pr_created"})
    assert not mod._row_is_skip({"terminal_class": "deliverable_branch_pushed"})


def test_row_is_skip_unknown_terminal_class_not_skip() -> None:
    assert not mod._row_is_skip({"terminal_class": "novel_class"})
    assert not mod._row_is_skip({})


def test_top_issues_by_skip_count_orders_by_count() -> None:
    rows = [
        {"issue_number": 100, "terminal_class": "blocked_auth_failure"},
        {"issue_number": 100, "terminal_class": "blocked_auth_failure"},
        {"issue_number": 100, "terminal_class": "blocked_sanitation_failed"},
        {"issue_number": 200, "terminal_class": "blocked_auth_failure"},
        {"issue_number": 200, "terminal_class": "deliverable_pr_created"},  # not skip
        {"issue_number": 300, "terminal_class": "rescue_no_deliverable"},
    ]
    out = mod.top_issues_by_skip_count(rows, top_n=3)
    assert len(out) == 3
    assert out[0]["issue_number"] == 100
    assert out[0]["skip_count"] == 3
    assert out[0]["last_terminal_class"] == "blocked_sanitation_failed"
    assert out[1]["issue_number"] in {200, 300}
    assert out[1]["skip_count"] == 1


def test_top_issues_drops_invalid_issue_numbers() -> None:
    rows = [
        {"issue_number": None, "terminal_class": "blocked_auth_failure"},
        {"issue_number": -5, "terminal_class": "blocked_auth_failure"},
        {"issue_number": "abc", "terminal_class": "blocked_auth_failure"},
        {"issue_number": 0, "terminal_class": "blocked_auth_failure"},
        {"issue_number": 7, "terminal_class": "blocked_auth_failure"},
    ]
    out = mod.top_issues_by_skip_count(rows, top_n=10)
    assert len(out) == 1
    assert out[0]["issue_number"] == 7


def test_top_issues_respects_top_n_limit() -> None:
    rows = [{"issue_number": n, "terminal_class": "blocked_auth_failure"} for n in range(1, 11)]
    out = mod.top_issues_by_skip_count(rows, top_n=3)
    assert len(out) == 3


def test_aggregate_skip_reasons_prefers_dispatch_skip_reason() -> None:
    rows = [
        {"dispatch_skip_reason": "no_work_orders"},
        {"dispatch_skip_reason": "no_work_orders"},
        {"dispatch_skip_reason": "needs_human_no_prompt"},
        {"terminal_class": "blocked_auth_failure"},
    ]
    out = mod.aggregate_skip_reasons(rows)
    assert out["dispatch_skip_reason:no_work_orders"] == 2
    assert out["dispatch_skip_reason:needs_human_no_prompt"] == 1
    assert out["terminal_class:blocked_auth_failure"] == 1


def test_aggregate_skip_reasons_empty() -> None:
    assert mod.aggregate_skip_reasons([]) == {}


def test_aggregate_skip_reasons_ignores_deliverable_terminal() -> None:
    rows = [
        {"terminal_class": "deliverable_pr_created"},
        {"terminal_class": "deliverable_branch_pushed"},
    ]
    assert mod.aggregate_skip_reasons(rows) == {}


def test_detect_stale_loops_threshold() -> None:
    rows = []
    for n, count in [(100, 12), (200, 5), (300, 11)]:
        for _ in range(count):
            rows.append({"issue_number": n, "terminal_class": "blocked_auth_failure"})
    out = mod.detect_stale_loops(rows, min_skip_rows=10)
    issue_numbers = [r["issue_number"] for r in out]
    assert 100 in issue_numbers
    assert 300 in issue_numbers
    assert 200 not in issue_numbers
    assert out[0]["issue_number"] == 100
    assert out[0]["skip_count"] == 12


def test_detect_stale_loops_empty_when_no_skip_rows() -> None:
    rows = [{"issue_number": 1, "terminal_class": "deliverable_pr_created"}] * 20
    assert mod.detect_stale_loops(rows, min_skip_rows=10) == []


def test_render_scorecard_shape() -> None:
    rows = [
        {"issue_number": 1, "terminal_class": "blocked_auth_failure"},
        {"issue_number": 1, "terminal_class": "blocked_auth_failure"},
        {"issue_number": 2, "terminal_class": "deliverable_pr_created"},
        {"issue_number": 3, "terminal_class": "deliverable_branch_pushed"},
        {"dispatch_skip_reason": "no_work_orders", "issue_number": 4},
    ]
    out = mod.render_scorecard(rows, top_n=10, stale_threshold=2)
    assert out["total_rows"] == 5
    assert out["skip_rows"] == 3
    assert out["deliverable_rows"] == 2
    assert "skip_reason_counts" in out
    assert "top_issues_by_skip_count" in out
    assert out["stale_threshold"] == 2
    assert "stale_loops" in out


def test_render_markdown_includes_all_sections() -> None:
    rows = [
        {"issue_number": 1, "terminal_class": "blocked_auth_failure"},
        {"issue_number": 1, "terminal_class": "blocked_auth_failure"},
    ]
    scorecard = mod.render_scorecard(rows, top_n=5, stale_threshold=10)
    md = mod.render_markdown(scorecard)
    assert "boss-loop metrics health scorecard" in md
    assert "Top skip reasons" in md
    assert "Top issues by skip count" in md
    assert "Stale loops" in md
    assert "#1" in md


def test_render_markdown_with_stale_loops() -> None:
    rows = [{"issue_number": 100, "terminal_class": "blocked_auth_failure"}] * 12
    scorecard = mod.render_scorecard(rows, top_n=5, stale_threshold=10)
    md = mod.render_markdown(scorecard)
    assert "Stale loops (>= 10 skip rows)" in md
    assert "#100" in md


def test_render_markdown_uses_non_default_stale_threshold() -> None:
    rows = [{"issue_number": 100, "terminal_class": "blocked_auth_failure"}] * 12
    scorecard = mod.render_scorecard(rows, top_n=5, stale_threshold=7)
    md = mod.render_markdown(scorecard)
    assert "Stale loops (>= 7 skip rows)" in md
    assert "Stale loops (>= 1 issues)" not in md


def test_main_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = _write(
        tmp_path,
        [
            {"issue_number": 1, "terminal_class": "blocked_auth_failure"},
            {"issue_number": 1, "terminal_class": "blocked_auth_failure"},
        ],
    )
    rc = mod.main(["--metrics", str(p), "--top-n", "5", "--format", "json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["total_rows"] == 2
    assert payload["skip_rows"] == 2


def test_main_markdown_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = _write(tmp_path, [{"issue_number": 1, "terminal_class": "blocked_auth_failure"}])
    rc = mod.main(["--metrics", str(p), "--format", "markdown"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "boss-loop metrics health scorecard" in out
