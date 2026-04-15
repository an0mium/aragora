#!/usr/bin/env python3
"""Render a repo-tracked B0 benchmark truth status summary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_PATH = REPO_ROOT / "docs" / "benchmarks" / "corpus.json"
DEFAULT_TRUTH_ROOT = REPO_ROOT / "docs" / "status" / "generated" / "benchmark_truth_artifacts"
DEFAULT_SCORECARD_ROOT = REPO_ROOT / "docs" / "status" / "generated" / "benchmark_scorecards"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "status" / "B0_BENCHMARK_TRUTH_STATUS.md"

SUCCESS_CLASSES = frozenset(
    {
        "success_merged",
        "success_pr_created",
        "deliverable_pr_created",
    }
)
FAILURE_CLASSES = frozenset(
    {
        "rescue_timeout",
        "rescue_worker_crash",
        "rescue_no_deliverable",
        "blocked_not_dispatch_bounded",
        "blocked_validation_target_missing",
        "blocked_sanitation_failed",
        "blocked_auth_failure",
        "blocked_no_runner",
    }
)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "benchmark-corpus"


def _repo_stable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload at {path} must be an object")
    return payload


def load_corpus(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    issues = payload.get("issues")
    if not isinstance(issues, list) or not issues:
        raise ValueError(f"Corpus at {path} must contain a non-empty 'issues' list")
    return payload


def resolve_latest_paths(
    *,
    corpus_path: Path,
    truth_root: Path,
    scorecard_root: Path,
) -> dict[str, Path]:
    corpus = load_corpus(corpus_path)
    corpus_id = str(corpus.get("corpus_id") or "").strip()
    revision = int(corpus.get("revision", 0) or 0)
    slug = _slugify(corpus_id)
    return {
        "truth_corpus_latest": truth_root / slug / "latest.json",
        "truth_revision_latest": truth_root / slug / f"rev-{revision}" / "latest.json",
        "scorecard_corpus_latest": scorecard_root / slug / "latest.json",
        "scorecard_revision_latest": scorecard_root / slug / f"rev-{revision}" / "latest.json",
    }


def _payload_corpus_identity(payload: dict[str, Any]) -> tuple[str, int]:
    corpus = dict(payload.get("corpus") or {})
    return (
        str(corpus.get("corpus_id") or "").strip(),
        int(corpus.get("revision", 0) or 0),
    )


def _load_expected_latest_payload(
    *,
    path: Path,
    label: str,
    expected_corpus_id: str,
    expected_revision: int,
) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    payload = _load_json(path)
    payload_corpus_id, payload_revision = _payload_corpus_identity(payload)
    if payload_corpus_id != expected_corpus_id:
        raise SystemExit(
            f"{label} corpus_id mismatch: expected {expected_corpus_id!r}, "
            f"got {payload_corpus_id!r} at {path}"
        )
    if payload_revision != expected_revision:
        raise SystemExit(
            f"{label} revision mismatch: expected {expected_revision}, "
            f"got {payload_revision} at {path}"
        )
    return payload


def _format_percent(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.1%}"
    return "n/a"


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None or value == "":
        return "n/a"
    return str(value)


def _render_mapping(mapping: dict[str, Any]) -> list[str]:
    if not mapping:
        return ["- none"]
    return [f"- `{key}`: {_format_value(value)}" for key, value in mapping.items()]


def _render_stale_closed_issues(issues: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for issue in issues:
        issue_number = _format_value(issue.get("issue_number"))
        title = _format_value(issue.get("issue_title"))
        closed_at = _format_value(issue.get("issue_closed_at"))
        state_reason = _format_value(issue.get("issue_state_reason"))
        truth_state = _format_value(issue.get("truth_state"))
        lines.append(
            f"- `#{issue_number}` `{title}`: closed `{closed_at}`, "
            f"reason `{state_reason}`, truth `{truth_state}`"
        )
    return lines or ["- none"]


def _normalize_proxy_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    proxy_metrics = dict(payload)
    terminal_class_distribution = dict(proxy_metrics.get("terminal_class_distribution") or {})

    if "unique_issues_neutral" not in proxy_metrics:
        attempted = proxy_metrics.get("unique_issues_attempted")
        succeeded = proxy_metrics.get("unique_issues_succeeded")
        failed = proxy_metrics.get("unique_issues_failed")
        if all(isinstance(value, (int, float)) for value in (attempted, succeeded, failed)):
            proxy_metrics["unique_issues_neutral"] = max(
                int(attempted) - int(succeeded) - int(failed),
                0,
            )

    if "neutral_classes" not in proxy_metrics and terminal_class_distribution:
        proxy_metrics["neutral_classes"] = {
            key: value
            for key, value in terminal_class_distribution.items()
            if key not in SUCCESS_CLASSES and key not in FAILURE_CLASSES
        }

    return proxy_metrics


def render_status_markdown(
    *,
    corpus_path: Path,
    truth_path: Path,
    scorecard_path: Path,
    truth_payload: dict[str, Any],
    scorecard_payload: dict[str, Any],
    latest_paths: dict[str, Path],
) -> str:
    corpus = dict(scorecard_payload.get("corpus") or truth_payload.get("corpus") or {})
    coverage = dict(scorecard_payload.get("coverage") or truth_payload.get("coverage") or {})
    truth_metrics = dict(
        scorecard_payload.get("truth_metrics") or truth_payload.get("primary_metrics") or {}
    )
    proxy_metrics = _normalize_proxy_metrics(dict(scorecard_payload.get("proxy_metrics") or {}))
    previous_artifact = dict(scorecard_payload.get("previous_artifact") or {})
    deltas = dict(scorecard_payload.get("deltas") or {})
    failure_distribution = dict(scorecard_payload.get("failure_class_distribution") or {})
    rescue_counts = dict(scorecard_payload.get("rescue_counts_by_type") or {})
    neutral_classes = dict(proxy_metrics.get("neutral_classes") or {})
    corpus_freshness = dict(truth_payload.get("corpus_freshness") or {})
    stale_closed_issues = [
        item
        for item in list(corpus_freshness.get("stale_closed_issues") or [])
        if isinstance(item, dict)
    ]
    generated_at = (
        str(scorecard_payload.get("generated_at") or "").strip()
        or str(truth_payload.get("generated_at") or "").strip()
        or "unknown"
    )

    lines = [
        "# B0 Benchmark Truth Status",
        "",
        f"Last updated: {generated_at}",
        "",
        "This is the repo-tracked recurring `TW-02` publication surface for the fixed benchmark corpus.",
        "",
        "## Corpus",
        "",
        f"- Corpus manifest: `{_repo_stable_path(corpus_path)}`",
        f"- Corpus id: `{_format_value(corpus.get('corpus_id'))}`",
        f"- Revision: `{_format_value(corpus.get('revision'))}`",
        f"- Recorded on: `{_format_value(corpus.get('recorded_on'))}`",
        f"- Success contract: `{_format_value(corpus.get('success_contract'))}`",
        f"- Coverage status: `{_format_value(coverage.get('status'))}`",
        (
            f"- Coverage: `{_format_value(coverage.get('attempted_issue_count'))}`/"
            f"`{_format_value(corpus.get('issue_count'))}` issues attempted"
        ),
    ]
    missing_issue_numbers = list(coverage.get("missing_issue_numbers") or [])
    if missing_issue_numbers:
        lines.append(
            "- Missing corpus issues: " + ", ".join(f"`{item}`" for item in missing_issue_numbers)
        )
    lines.extend(
        [
            "",
            "## Published Paths",
            "",
            f"- Latest truth artifact: `{_repo_stable_path(truth_path)}`",
            f"- Latest scorecard: `{_repo_stable_path(scorecard_path)}`",
            f"- Revision-scoped truth pointer: `{_repo_stable_path(latest_paths['truth_revision_latest'])}`",
            f"- Revision-scoped scorecard pointer: `{_repo_stable_path(latest_paths['scorecard_revision_latest'])}`",
            "",
            "## Truth Metrics",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Truth success rate | {_format_percent(truth_metrics.get('truth_success_rate'))} |",
            f"| No-rescue truth success rate | {_format_percent(truth_metrics.get('no_rescue_truth_success_rate'))} |",
            f"| Merged-only rate | {_format_percent(truth_metrics.get('merged_only_rate'))} |",
            "",
            "## Proxy Metrics",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Proxy no-rescue success rate | {_format_percent(proxy_metrics.get('no_rescue_success_rate'))} |",
            f"| Unique issues attempted | {_format_value(proxy_metrics.get('unique_issues_attempted'))} |",
            f"| Unique issues succeeded | {_format_value(proxy_metrics.get('unique_issues_succeeded'))} |",
            f"| Unique issues failed | {_format_value(proxy_metrics.get('unique_issues_failed'))} |",
            f"| Unique issues neutral | {_format_value(proxy_metrics.get('unique_issues_neutral'))} |",
            f"| Total ticks | {_format_value(proxy_metrics.get('total_ticks'))} |",
        ]
    )
    if neutral_classes:
        lines.extend(
            [
                "",
                "Proxy note: neutral issue outcomes are current-corpus rows that were neither fresh success nor failure, such as `issue_already_resolved`.",
                "",
                "## Proxy Neutral Class Distribution",
                "",
                *_render_mapping(neutral_classes),
            ]
        )
    if stale_closed_issues:
        lines.extend(
            [
                "",
                "## Corpus Freshness Alerts",
                "",
                "Truth metrics still reflect the frozen corpus revision. Closed issues without linked PR truth should be retired or replaced in the next corpus revision.",
                "",
                *_render_stale_closed_issues(stale_closed_issues),
            ]
        )
    lines.extend(
        [
            "",
            "## Failure Class Distribution",
            "",
            *_render_mapping(failure_distribution),
            "",
            "## Rescue Counts By Type",
            "",
            *_render_mapping(rescue_counts),
        ]
    )
    if previous_artifact:
        lines.extend(
            [
                "",
                "## Previous Published Artifact",
                "",
                f"- Previous artifact path: `{_format_value(previous_artifact.get('path'))}`",
                f"- Previous generated_at: `{_format_value(previous_artifact.get('generated_at'))}`",
            ]
        )
    if deltas:
        lines.extend(
            [
                "",
                "## Deltas",
                "",
                *_render_mapping(deltas),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def write_output(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"Benchmark corpus manifest (default: {DEFAULT_CORPUS_PATH})",
    )
    parser.add_argument(
        "--truth-root",
        type=Path,
        default=DEFAULT_TRUTH_ROOT,
        help=f"Tracked truth-artifact root (default: {DEFAULT_TRUTH_ROOT})",
    )
    parser.add_argument(
        "--scorecard-root",
        type=Path,
        default=DEFAULT_SCORECARD_ROOT,
        help=f"Tracked scorecard root (default: {DEFAULT_SCORECARD_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Markdown status output path (default: {DEFAULT_OUTPUT})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus_path = args.corpus.resolve()
    truth_root = args.truth_root.resolve()
    scorecard_root = args.scorecard_root.resolve()
    output_path = args.output.resolve()
    if not corpus_path.exists():
        raise SystemExit(f"corpus file not found: {corpus_path}")

    corpus = load_corpus(corpus_path)
    latest_paths = resolve_latest_paths(
        corpus_path=corpus_path,
        truth_root=truth_root,
        scorecard_root=scorecard_root,
    )
    expected_corpus_id = str(corpus.get("corpus_id") or "").strip()
    expected_revision = int(corpus.get("revision", 0) or 0)
    truth_path = latest_paths["truth_corpus_latest"]
    scorecard_path = latest_paths["scorecard_corpus_latest"]
    truth_payload = _load_expected_latest_payload(
        path=truth_path,
        label="truth artifact latest.json",
        expected_corpus_id=expected_corpus_id,
        expected_revision=expected_revision,
    )
    scorecard_payload = _load_expected_latest_payload(
        path=scorecard_path,
        label="scorecard latest.json",
        expected_corpus_id=expected_corpus_id,
        expected_revision=expected_revision,
    )
    _load_expected_latest_payload(
        path=latest_paths["truth_revision_latest"],
        label="truth artifact revision latest.json",
        expected_corpus_id=expected_corpus_id,
        expected_revision=expected_revision,
    )
    _load_expected_latest_payload(
        path=latest_paths["scorecard_revision_latest"],
        label="scorecard revision latest.json",
        expected_corpus_id=expected_corpus_id,
        expected_revision=expected_revision,
    )

    content = render_status_markdown(
        corpus_path=corpus_path,
        truth_path=truth_path,
        scorecard_path=scorecard_path,
        truth_payload=truth_payload,
        scorecard_payload=scorecard_payload,
        latest_paths=latest_paths,
    )
    written = write_output(output_path, content)
    print(str(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
