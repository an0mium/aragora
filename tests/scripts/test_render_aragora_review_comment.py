from __future__ import annotations

from scripts.render_aragora_review_comment import (
    TRUNCATION_MARKER,
    render_review_comment,
)


def test_filters_empty_structured_findings_from_count_and_footer() -> None:
    comment = render_review_comment(
        {
            "summary": "Reviewer prose identifies several advisory findings.",
            "critical_issues": [{"title": "", "description": ""}],
            "high_issues": [
                {
                    "title": "Validate branch ownership",
                    "description": "The merge helper should verify the owner before mutation.",
                    "file": "scripts/merge.py",
                    "line": 42,
                }
            ],
            "medium_issues": [],
            "low_issues": [],
        }
    )

    assert "**1 finding(s)** across the diff" in comment
    assert "### [HIGH] Finding" in comment
    assert "Validate branch ownership" in comment
    assert "### [CRITICAL] Finding\n**Finding**" not in comment


def test_near_size_limit_truncates_summary_with_marker_and_preserves_findings() -> None:
    findings = [
        {
            "title": f"Finding {idx}",
            "description": f"Description for finding {idx}.",
            "file": f"file_{idx}.py",
            "line": idx,
        }
        for idx in range(1, 9)
    ]
    comment = render_review_comment(
        {
            "summary": "This advisory summary is intentionally long. " * 500,
            "critical_issues": findings[:2],
            "high_issues": findings[2:5],
            "medium_issues": findings[5:7],
            "low_issues": findings[7:],
        },
        max_chars=4_000,
    )

    assert len(comment) <= 4_000
    assert TRUNCATION_MARKER in comment
    assert "**8 finding(s)** across the diff" in comment
    assert "Finding 8" in comment
    assert "### [CRITICAL] Finding\n**Finding**" not in comment
