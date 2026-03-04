from __future__ import annotations

from scripts.pr_admission_controller import (
    classify_stream_from_files,
    evaluate_admission,
    extract_stream_from_labels,
    select_admitted_pr_numbers,
)


def test_extract_stream_from_labels_supports_lane_prefixes() -> None:
    labels = [{"name": "bug"}, {"name": "lane:CI-Hardening"}]
    assert extract_stream_from_labels(labels) == "ci-hardening"


def test_extract_stream_from_labels_supports_stream_slash_prefix() -> None:
    labels = [{"name": "stream/frontend"}]
    assert extract_stream_from_labels(labels) == "frontend"


def test_extract_stream_from_labels_returns_none_when_missing() -> None:
    assert extract_stream_from_labels([{"name": "enhancement"}]) is None


def test_classify_stream_from_files_docs_only() -> None:
    files = ["docs/README.md", "docs/guide/setup.mdx"]
    assert classify_stream_from_files(files) == "docs"


def test_classify_stream_from_files_ci_only() -> None:
    files = [".github/workflows/lint.yml", "scripts/pr_stale_run_gc.py"]
    assert classify_stream_from_files(files) == "ci"


def test_classify_stream_from_files_frontend() -> None:
    files = ["aragora/live/src/components/landing/HeroSection.tsx"]
    assert classify_stream_from_files(files) == "frontend"


def test_select_admitted_pr_numbers_oldest_first() -> None:
    ready_prs = [
        {"number": 200, "created_at": "2026-03-04T01:00:00Z"},
        {"number": 201, "created_at": "2026-03-04T01:01:00Z"},
        {"number": 202, "created_at": "2026-03-04T01:02:00Z"},
    ]
    stream_by_pr = {200: "ci", 201: "ci", 202: "ci"}
    admitted = select_admitted_pr_numbers(
        ready_prs=ready_prs,
        stream_by_pr=stream_by_pr,
        target_stream="ci",
        max_ready_per_stream=1,
    )
    assert admitted == {200}


def test_evaluate_admission_advisory_mode_is_non_blocking() -> None:
    class FakeClient:
        def get_pull(self, number: int) -> dict[str, object]:
            return {
                "number": number,
                "state": "open",
                "draft": False,
                "base": {"ref": "main"},
                "labels": [{"name": "lane:ci"}],
            }

        def list_open_pulls(self, _base_branch: str) -> list[dict[str, object]]:
            return [
                {
                    "number": 100,
                    "draft": False,
                    "created_at": "2026-03-04T01:00:00Z",
                    "labels": [{"name": "lane:ci"}],
                },
                {
                    "number": 101,
                    "draft": False,
                    "created_at": "2026-03-04T01:01:00Z",
                    "labels": [{"name": "lane:ci"}],
                },
            ]

        def list_pull_files(self, _number: int) -> list[str]:
            return []

    rc = evaluate_admission(
        client=FakeClient(),  # type: ignore[arg-type]
        current_pr_number=101,
        max_ready_per_stream=1,
        enforce=False,
    )
    assert rc == 0
