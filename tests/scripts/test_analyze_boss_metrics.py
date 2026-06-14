"""Tests for analyze_boss_metrics script."""

from pathlib import Path

from scripts.analyze_boss_metrics import analyze_boss_metrics, analyze_metrics, render_text


def test_analyze_boss_metrics_fixture():
    root = Path(__file__).resolve().parents[2]
    metrics_path = root / "benchmarks/fixtures/swarm/sample_boss_metrics.jsonl"
    signals_path = root / "benchmarks/fixtures/swarm/sample_outcome_signals.jsonl"

    report = analyze_boss_metrics(metrics_path=metrics_path, signals_path=signals_path)
    metrics_summary = report["metrics_summary"]

    assert metrics_summary["totals"]["records"] == 3
    assert metrics_summary["deliverables"]["count"] == 1
    assert metrics_summary["publish_actions"]["opened_pr"] == 1

    terminal_truth = report["terminal_truth_benchmark"]
    assert terminal_truth["no_rescue_rate"] == 0.333
    assert terminal_truth["families"] == {"success": 1, "blocked": 1, "rescue": 1}
    assert terminal_truth["classes"] == {
        "deliverable_branch_pushed": 1,
        "blocked_not_dispatch_bounded": 1,
        "rescue_worker_crash": 1,
    }

    signals_summary = report["signals_summary"]
    assert signals_summary is not None
    assert signals_summary["total_signals"] == 3
    assert signals_summary["by_loop"]["boss"]["total"] == 2

    text = render_text(report)
    assert "Boss Metrics Summary" in text
    assert "deliverable rate" in text
    assert "terminal-truth families" in text
    assert "terminal-truth classes" in text


def test_analyze_metrics_surfaces_invalid_numeric_fields():
    summary = analyze_metrics(
        [
            {"prompt_chars": 30, "enriched_context_chars": 60},
            {"prompt_chars": True, "enriched_context_chars": -1},
            {"prompt_chars": "100", "enriched_context_chars": False},
        ]
    )

    assert summary["prompt_chars"] == {"total": 30, "avg": 10.0}
    assert summary["enriched_context_chars"] == {"total": 60, "avg": 20.0}
    assert summary["invalid_numeric_metrics"] == {
        "enriched_context_chars": 2,
        "prompt_chars": 2,
    }


def test_render_text_includes_invalid_numeric_metrics():
    text = render_text(
        {
            "metrics_summary": {
                "totals": {"records": 1},
                "prompt_chars": {"avg": 0},
                "enriched_context_chars": {"avg": 0},
                "deliverables": {"rate": 0},
                "publish_actions": {},
                "failure_taxonomy": {},
                "invalid_numeric_metrics": {"prompt_chars": 1},
            },
            "terminal_truth_benchmark": {},
        }
    )

    assert "invalid numeric metrics" in text
    assert "prompt_chars: 1" in text
