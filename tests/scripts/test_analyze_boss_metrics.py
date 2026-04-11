"""Tests for analyze_boss_metrics script."""

from pathlib import Path

from scripts.analyze_boss_metrics import analyze_boss_metrics, render_text


def test_analyze_boss_metrics_fixture():
    root = Path(__file__).resolve().parents[2]
    metrics_path = root / "benchmarks/fixtures/swarm/sample_boss_metrics.jsonl"
    signals_path = root / "benchmarks/fixtures/swarm/sample_outcome_signals.jsonl"

    report = analyze_boss_metrics(metrics_path=metrics_path, signals_path=signals_path)
    metrics_summary = report["metrics_summary"]

    assert metrics_summary["totals"]["records"] == 3
    assert metrics_summary["deliverables"]["count"] == 1
    assert metrics_summary["publish_actions"]["opened_pr"] == 1

    signals_summary = report["signals_summary"]
    assert signals_summary is not None
    assert signals_summary["total_signals"] == 3
    assert signals_summary["by_loop"]["boss"]["total"] == 2

    text = render_text(report)
    assert "Boss Metrics Summary" in text
    assert "deliverable rate" in text
