from __future__ import annotations

import json
import sys
from pathlib import Path

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import reporting as mod  # noqa: E402


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_corpus(path: Path) -> Path:
    return _write_json(
        path,
        {
            "corpus_id": "tw-01-bounded-execution-v1",
            "revision": 2,
        },
    )


def test_build_scorecard_reads_reconcile_summary_payload(tmp_path: Path) -> None:
    corpus_path = _write_corpus(tmp_path / "corpus.json")
    report_path = _write_json(
        tmp_path / "reconcile.json",
        {
            "generated_at": "2026-04-15T23:05:32Z",
            "summary": {
                "attempted_issue_count": 2,
                "truth_success_rate": 0.5,
                "no_rescue_truth_success_rate": 0.25,
                "merged_issue_rate": 0.75,
            },
            "issues": [{"had_rescue": True}, {"had_rescue": False}],
        },
    )

    scorecard = mod.build_scorecard([report_path], corpus_path=corpus_path)

    assert scorecard["corpus"] == {
        "corpus_id": "tw-01-bounded-execution-v1",
        "revision": 2,
    }
    assert scorecard["runs"] == [
        {
            "attempted_issue_count": 2,
            "date": "2026-04-15",
            "delta_no_rescue_pp": None,
            "merged_issue_rate": 0.75,
            "no_rescue_truth_success_rate": 0.25,
            "rescue_count": 1,
            "trend": "baseline",
            "truth_success_rate": 0.5,
        }
    ]


def test_build_scorecard_reads_published_truth_artifact_payload(
    tmp_path: Path,
) -> None:
    corpus_path = _write_corpus(tmp_path / "corpus.json")
    report_path = _write_json(
        tmp_path / "truth-artifact.json",
        {
            "generated_at": "2026-04-15T23:05:32Z",
            "coverage": {"attempted_issue_count": 5},
            "primary_metrics": {
                "truth_success_rate": 1.0,
                "no_rescue_truth_success_rate": 0.8,
                "merged_only_rate": 0.6,
            },
            "issues": [{"had_rescue": False}, {"had_rescue": True}],
        },
    )

    run = mod.build_scorecard([report_path], corpus_path=corpus_path)["runs"][0]

    assert run["attempted_issue_count"] == 5
    assert run["truth_success_rate"] == 1.0
    assert run["no_rescue_truth_success_rate"] == 0.8
    assert run["merged_issue_rate"] == 0.6
    assert run["rescue_count"] == 1
