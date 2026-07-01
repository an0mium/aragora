"""The review-counts extractor (used by the marketplace action) must emit
correct GitHub-Actions key=value outputs and degrade safely on missing/garbage
input."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.extract_review_counts import render_outputs


def _parse(block: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in block.splitlines())


def test_counts_from_a_real_review_json(tmp_path: Path):
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "critical_issues": [{"x": 1}],
                "high_issues": [{"x": 1}, {"x": 2}],
                "medium_issues": [],
                "low_issues": [{"x": 1}],
                "unanimous_critiques": [{"x": 1}],
                "risk_areas": [],
                "split_opinions": [{"x": 1}, {"x": 2}],
                "agreement_score": 0.75,
            }
        ),
        encoding="utf-8",
    )
    out = _parse(render_outputs(review))
    assert out["critical_count"] == "1"
    assert out["high_count"] == "2"
    assert out["total_count"] == "4"  # 1 + 2 + 0 + 1
    assert out["split_opinions_count"] == "2"
    assert out["agreement_score"] == "0.75"
    assert out["review_json_path"].endswith("review.json")


def test_missing_file_degrades_to_zeros(tmp_path: Path):
    out = _parse(render_outputs(tmp_path / "nope.json"))
    assert out["review_json_path"] == ""
    assert out["total_count"] == "0"
    assert out["agreement_score"] == "0.0"


def test_garbage_json_degrades_to_zeros(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    out = _parse(render_outputs(bad))
    assert out["total_count"] == "0"
