#!/usr/bin/env python3
"""Build a corpus-linked benchmark truth artifact for TW-01/TW-02."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.terminal_truth import TerminalClass  # noqa: E402
from scripts.reconcile_b0_pr_truth import (  # noqa: E402
    DEFAULT_METRICS_PATH,
    GitHubTruthClient,
    IssueMetricsAggregate,
    IssueTruthRecord,
    reconcile_issue_truth,
    report_to_json as _unused_report_to_json,
    resolve_metrics_path,
    resolve_terminal_class,
    load_metrics_rows,
)

DEFAULT_CORPUS_PATH = REPO_ROOT / "docs" / "benchmarks" / "corpus.json"
DEFAULT_PUBLISH_DIR = REPO_ROOT / ".aragora" / "benchmark_truth_artifacts"


def load_corpus(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Corpus at {path} must be a JSON object")
    issues = payload.get("issues")
    if not isinstance(issues, list) or not issues:
        raise ValueError(f"Corpus at {path} must contain a non-empty 'issues' list")
    return payload


def _corpus_issue_numbers(corpus: dict[str, Any]) -> list[int]:
    issue_numbers: list[int] = []
    for item in list(corpus.get("issues") or []):
        if not isinstance(item, dict):
            continue
        issue_number = int(item.get("issue_id", 0) or 0)
        if issue_number > 0:
            issue_numbers.append(issue_number)
    return sorted(issue_numbers)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _corpus_membership_sha256(issue_numbers: list[int]) -> str:
    normalized = json.dumps(issue_numbers, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(normalized)


def aggregate_corpus_issues(
    rows: list[dict[str, Any]],
    corpus: dict[str, Any],
) -> list[IssueMetricsAggregate]:
    by_issue: dict[int, IssueMetricsAggregate] = {}
    corpus_issues = [item for item in list(corpus.get("issues") or []) if isinstance(item, dict)]
    for item in corpus_issues:
        issue_number = int(item.get("issue_id", 0) or 0)
        if issue_number <= 0:
            continue
        by_issue[issue_number] = IssueMetricsAggregate(
            issue_number=issue_number,
            title=str(item.get("title") or "").strip(),
            row_count=0,
            proxy_pr_signal=False,
            had_rescue=False,
        )

    corpus_issue_numbers = set(by_issue)
    for row in rows:
        row_issue_number: object = row.get("issue_number")
        if not isinstance(row_issue_number, int) or row_issue_number not in corpus_issue_numbers:
            continue
        aggregate = by_issue[row_issue_number]
        aggregate.row_count += 1
        terminal_class = resolve_terminal_class(row)
        publish_action = str(row.get("publish_action", "") or "").strip().lower()
        worker_outcome = str(row.get("worker_outcome", "") or "").strip().lower()
        if (
            terminal_class is TerminalClass.DELIVERABLE_PR_CREATED
            or publish_action in {"pr_created", "existing_pr", "discovered_after_push"}
            or worker_outcome in {"pr_adopted"}
        ):
            aggregate.proxy_pr_signal = True
        if terminal_class.value.startswith("rescue_"):
            aggregate.had_rescue = True
    return [by_issue[number] for number in sorted(by_issue)]


def _failure_distributions(
    rows: list[dict[str, Any]],
    *,
    corpus_issue_numbers: set[int],
) -> tuple[dict[str, int], dict[str, int]]:
    failure_counts: Counter[str] = Counter()
    rescue_counts: Counter[str] = Counter()
    for row in rows:
        issue_number = row.get("issue_number")
        if not isinstance(issue_number, int) or issue_number not in corpus_issue_numbers:
            continue
        terminal_class = resolve_terminal_class(row).value
        if terminal_class.startswith("deliverable_") or terminal_class == "issue_already_resolved":
            continue
        failure_counts[terminal_class] += 1
        if terminal_class.startswith("rescue_"):
            rescue_counts[terminal_class] += 1
    return dict(sorted(failure_counts.items())), dict(sorted(rescue_counts.items()))


def _missing_corpus_issue_numbers(aggregates: list[IssueMetricsAggregate]) -> list[int]:
    return [
        aggregate.issue_number
        for aggregate in aggregates
        if aggregate.issue_number > 0 and aggregate.row_count <= 0
    ]


def _coerce_utc_datetime(value: str | None = None) -> dt.datetime:
    if value:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = dt.datetime.now(dt.UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC).replace(microsecond=0)


def normalize_generated_at(value: str | None = None) -> str:
    return _coerce_utc_datetime(value).isoformat().replace("+00:00", "Z")


def _repo_stable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "benchmark-corpus"


def _corpus_publish_dir(*, publish_dir: Path, corpus: dict[str, Any]) -> Path:
    corpus_id = _slugify(str(corpus.get("corpus_id") or "benchmark-corpus"))
    return publish_dir / corpus_id


def _revision_publish_dir(*, publish_dir: Path, corpus: dict[str, Any]) -> Path:
    revision = int(corpus.get("revision", 0) or 0)
    return _corpus_publish_dir(publish_dir=publish_dir, corpus=corpus) / f"rev-{revision}"


def resolve_published_artifact_path(
    *,
    publish_dir: Path,
    artifact: dict[str, Any],
) -> Path:
    corpus = artifact.get("corpus")
    if not isinstance(corpus, dict):
        corpus = {}
    generated_at = artifact.get("generated_at")
    timestamp = _coerce_utc_datetime(
        generated_at if isinstance(generated_at, str) else None
    ).strftime("%Y%m%dT%H%M%SZ")
    filename = f"truth-{timestamp}.json"
    return _revision_publish_dir(publish_dir=publish_dir, corpus=corpus) / filename


def resolve_latest_artifact_paths(
    *,
    publish_dir: Path,
    artifact: dict[str, Any],
) -> dict[str, Path]:
    corpus = artifact.get("corpus")
    if not isinstance(corpus, dict):
        corpus = {}
    return {
        "corpus_latest": _corpus_publish_dir(publish_dir=publish_dir, corpus=corpus)
        / "latest.json",
        "revision_latest": _revision_publish_dir(publish_dir=publish_dir, corpus=corpus)
        / "latest.json",
    }


def build_benchmark_truth_artifact(
    *,
    repo: str,
    metrics_file: Path,
    corpus_path: Path,
    client: GitHubTruthClient | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    normalized_generated_at = normalize_generated_at(generated_at)
    rows = load_metrics_rows(metrics_file)
    corpus = load_corpus(corpus_path)
    membership_issue_numbers = _corpus_issue_numbers(corpus)
    aggregates = aggregate_corpus_issues(rows, corpus)
    truth_client = client or GitHubTruthClient()
    records: list[IssueTruthRecord] = []
    for aggregate in aggregates:
        if aggregate.row_count <= 0:
            records.append(
                IssueTruthRecord(
                    issue_number=aggregate.issue_number,
                    issue_title=aggregate.title,
                    proxy_pr_signal=False,
                    had_rescue=False,
                    truth_state="not_attempted",
                    truth_success=False,
                    no_rescue_truth_success=False,
                )
            )
            continue
        records.append(reconcile_issue_truth(repo, aggregate, truth_client))
    corpus_issue_count = len(aggregates)
    attempted_issue_count = sum(1 for aggregate in aggregates if aggregate.row_count > 0)
    truth_success_issue_count = sum(1 for record in records if record.truth_success)
    no_rescue_truth_success_issue_count = sum(
        1 for record in records if record.no_rescue_truth_success
    )
    merged_issue_count = sum(1 for record in records if record.truth_state == "merged_pr")
    proxy_pr_signal_issue_count = sum(1 for aggregate in aggregates if aggregate.proxy_pr_signal)
    missing_issue_numbers = _missing_corpus_issue_numbers(aggregates)
    run_complete = not missing_issue_numbers
    failure_class_distribution, rescue_counts_by_type = _failure_distributions(
        rows,
        corpus_issue_numbers={aggregate.issue_number for aggregate in aggregates},
    )
    return {
        "generated_at": normalized_generated_at,
        "repo": repo,
        "metrics_file": _repo_stable_path(metrics_file),
        "corpus": {
            "path": _repo_stable_path(corpus_path),
            "corpus_id": str(corpus.get("corpus_id") or "").strip(),
            "revision": int(corpus.get("revision", 0) or 0),
            "recorded_on": str(corpus.get("recorded_on") or "").strip(),
            "success_contract": str(corpus.get("success_contract") or "").strip(),
            "manifest_sha256": _sha256_bytes(corpus_path.read_bytes()),
            "membership_sha256": _corpus_membership_sha256(membership_issue_numbers),
            "membership_issue_numbers": membership_issue_numbers,
            "issue_count": corpus_issue_count,
        },
        "run_status": "complete" if run_complete else "incomplete",
        "coverage": {
            "attempted_issue_count": attempted_issue_count,
            "missing_issue_count": len(missing_issue_numbers),
            "missing_issue_numbers": missing_issue_numbers,
            "is_complete": run_complete,
            "status": "complete" if run_complete else "incomplete",
        },
        "primary_metrics": {
            "truth_success_rate": round(truth_success_issue_count / corpus_issue_count, 4)
            if corpus_issue_count
            else 0.0,
            "no_rescue_truth_success_rate": round(
                no_rescue_truth_success_issue_count / corpus_issue_count,
                4,
            )
            if corpus_issue_count
            else 0.0,
            "merged_only_rate": round(merged_issue_count / corpus_issue_count, 4)
            if corpus_issue_count
            else 0.0,
        },
        "proxy_metrics": {
            "attempted_issue_count": attempted_issue_count,
            "proxy_pr_signal_issue_count": proxy_pr_signal_issue_count,
            "proxy_pr_signal_issue_rate": round(
                proxy_pr_signal_issue_count / corpus_issue_count,
                4,
            )
            if corpus_issue_count
            else 0.0,
            "note": "Proxy metrics are secondary. Truth metrics remain mergeable_pr OR merged_pr.",
        },
        "failure_class_distribution": failure_class_distribution,
        "rescue_counts_by_type": rescue_counts_by_type,
        "issues": [record.to_dict() for record in records],
    }


def write_artifact(path: Path, artifact: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def publish_artifact_bundle(
    *,
    publish_dir: Path,
    artifact: dict[str, Any],
) -> dict[str, Path]:
    timestamped_path = write_artifact(
        resolve_published_artifact_path(publish_dir=publish_dir, artifact=artifact),
        artifact,
    )
    latest_paths = resolve_latest_artifact_paths(publish_dir=publish_dir, artifact=artifact)
    return {
        "timestamped": timestamped_path,
        "corpus_latest": write_artifact(latest_paths["corpus_latest"], artifact),
        "revision_latest": write_artifact(latest_paths["revision_latest"], artifact),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="synaptent/aragora", help="GitHub repo owner/name")
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Metrics JSONL file (default: {DEFAULT_METRICS_PATH})",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help=f"Benchmark corpus manifest (default: {DEFAULT_CORPUS_PATH})",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional artifact output path")
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    parser.add_argument(
        "--fail-incomplete",
        action="store_true",
        help="Exit non-zero when the artifact does not cover every corpus issue",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Write a timestamped artifact plus stable latest.json pointers under the repo-stable publish path",
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        default=None,
        help=f"Optional publish root override (default: {DEFAULT_PUBLISH_DIR})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics_file = resolve_metrics_path(args.metrics_file)
    corpus_path = args.corpus.resolve()
    if not metrics_file.exists():
        raise SystemExit(f"metrics file not found: {metrics_file}")
    if not corpus_path.exists():
        raise SystemExit(f"corpus file not found: {corpus_path}")
    artifact = build_benchmark_truth_artifact(
        repo=str(args.repo),
        metrics_file=metrics_file,
        corpus_path=corpus_path,
    )
    publish_dir: Path | None = None
    if args.publish_dir is not None:
        publish_dir = args.publish_dir.resolve()
    elif args.publish:
        publish_dir = DEFAULT_PUBLISH_DIR
    if args.json or (args.output is None and publish_dir is None):
        print(json.dumps(artifact, indent=2, sort_keys=True))
    is_complete = artifact.get("run_status") == "complete"
    if args.fail_incomplete and not is_complete:
        missing_issue_numbers = list(
            (artifact.get("coverage") or {}).get("missing_issue_numbers") or []
        )
        missing_suffix = ", ".join(str(item) for item in missing_issue_numbers) or "unknown"
        print(
            f"incomplete corpus coverage: missing issue numbers {missing_suffix}",
            file=sys.stderr,
        )
        return 2
    if args.output is not None:
        output_path = write_artifact(args.output.resolve(), artifact)
        print(str(output_path))
    if publish_dir is not None:
        published_paths = publish_artifact_bundle(
            publish_dir=publish_dir,
            artifact=artifact,
        )
        published_path = published_paths["timestamped"]
        if args.json:
            print(str(published_path), file=sys.stderr)
        else:
            print(str(published_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
