#!/usr/bin/env python3
"""Build a corpus-linked benchmark truth artifact for TW-01/TW-02."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
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
DEFAULT_FRESHNESS_MAP_PATH = REPO_ROOT / "docs" / "benchmarks" / "benchmark_corpus_freshness.json"
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


def _corpus_freshness(records: list[IssueTruthRecord]) -> dict[str, Any]:
    stale_closed_issues = [
        {
            "issue_number": record.issue_number,
            "issue_title": record.issue_title,
            "issue_url": record.issue_url,
            "issue_state": record.issue_state,
            "issue_state_reason": record.issue_state_reason,
            "issue_closed_at": record.issue_closed_at,
            "truth_state": record.truth_state,
            "stale_corpus_reason": record.stale_corpus_reason,
        }
        for record in records
        if record.stale_corpus_issue
    ]
    linkage_errors = [
        {
            "issue_number": record.issue_number,
            "issue_title": record.issue_title,
            "issue_url": record.issue_url,
            "issue_state": record.issue_state,
            "issue_state_reason": record.issue_state_reason,
            "issue_closed_at": record.issue_closed_at,
            "truth_state": record.truth_state,
            "linkage_status": record.linkage_status,
            "linkage_error": record.linkage_error,
        }
        for record in records
        if record.linkage_verification_incomplete
    ]
    status = "fresh"
    if stale_closed_issues:
        status = "stale_closed_issues_detected"
    elif linkage_errors:
        status = "linkage_verification_incomplete"
    return {
        "status": status,
        "stale_closed_issue_count": len(stale_closed_issues),
        "stale_closed_issue_numbers": [
            item["issue_number"] for item in stale_closed_issues if item["issue_number"] > 0
        ],
        "stale_closed_issues": stale_closed_issues,
        "linkage_error_count": len(linkage_errors),
        "linkage_errors": linkage_errors,
    }


def load_corpus_freshness_map_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "entries": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Corpus freshness map at {path} must be a JSON object")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"Corpus freshness map at {path} must contain an 'entries' list")
    return {
        "schema_version": int(payload.get("schema_version", 1) or 1),
        "entries": entries,
    }


def write_corpus_freshness_map_payload(path: Path, payload: dict[str, Any]) -> Path:
    entries = [
        entry
        for entry in list(payload.get("entries") or [])
        if isinstance(entry, dict)
        and str(entry.get("corpus_id") or "").strip()
        and str(entry.get("title") or "").strip()
    ]
    entries.sort(
        key=lambda entry: (
            str(entry.get("corpus_id") or "").strip(),
            int(entry.get("revision", 0) or 0),
            str(entry.get("title") or "").strip(),
        )
    )
    normalized = {
        "schema_version": int(payload.get("schema_version", 1) or 1),
        "entries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _issue_target_url(*, repo: str, target: str, url: str = "") -> str:
    if url.strip():
        return url.strip()
    match = re.fullmatch(r"#(\d+)", target.strip())
    if not match:
        return ""
    return f"https://github.com/{repo}/issues/{match.group(1)}"


def _freshness_entry_key(*, corpus_id: str, revision: int) -> tuple[str, int]:
    return corpus_id.strip(), int(revision)


def _normalize_stale_issue_numbers(values: Any) -> list[int]:
    normalized = {
        int(item) for item in list(values or []) if isinstance(item, int) and int(item) > 0
    }
    return sorted(normalized)


def _freshness_issue_target_number(*, target_kind: str, target: str) -> int | None:
    if target_kind.strip().lower() != "issue":
        return None
    match = re.fullmatch(r"#(\d+)", target.strip())
    if not match:
        return None
    issue_number = int(match.group(1))
    return issue_number if issue_number > 0 else None


def _freshness_entry_target_is_open(
    *,
    entry: dict[str, Any],
    repo: str,
    client: GitHubTruthClient | None,
) -> bool:
    target_kind = str(entry.get("target_kind") or "").strip() or "issue"
    target = str(entry.get("target") or "").strip()
    issue_number = _freshness_issue_target_number(target_kind=target_kind, target=target)
    if target_kind.strip().lower() != "issue":
        return True
    if issue_number is None or client is None:
        return False
    try:
        issue = client.get_issue(repo, issue_number)
    except (RuntimeError, subprocess.TimeoutExpired, OSError, json.JSONDecodeError, KeyError):
        return False
    return str(issue.get("state") or "").strip().upper() == "OPEN"


def _linked_corpus_freshness_entries(
    *,
    artifact: dict[str, Any],
    freshness_map_path: Path,
    repo: str,
    client: GitHubTruthClient | None = None,
) -> list[dict[str, Any]]:
    corpus = dict(artifact.get("corpus") or {})
    corpus_id = str(corpus.get("corpus_id") or "").strip()
    revision = int(corpus.get("revision", 0) or 0)
    stale_issue_numbers = _normalize_stale_issue_numbers(
        (artifact.get("corpus_freshness") or {}).get("stale_closed_issue_numbers") or []
    )
    if not corpus_id or not stale_issue_numbers:
        return []

    payload = load_corpus_freshness_map_payload(freshness_map_path)
    linked_entries: list[dict[str, Any]] = []
    for entry in list(payload.get("entries") or []):
        if not isinstance(entry, dict):
            continue
        if _freshness_entry_key(
            corpus_id=str(entry.get("corpus_id") or "").strip(),
            revision=int(entry.get("revision", 0) or 0),
        ) != _freshness_entry_key(corpus_id=corpus_id, revision=revision):
            continue
        target = str(entry.get("target") or "").strip()
        if not target:
            continue
        entry_stale_issue_numbers = _normalize_stale_issue_numbers(
            entry.get("stale_issue_numbers") or []
        )
        if entry_stale_issue_numbers != stale_issue_numbers:
            continue
        if not _freshness_entry_target_is_open(entry=entry, repo=repo, client=client):
            continue
        linked_entries.append(
            {
                "corpus_id": corpus_id,
                "revision": revision,
                "target_kind": str(entry.get("target_kind") or "").strip() or "issue",
                "target": target,
                "title": str(entry.get("title") or "").strip(),
                "notes": str(entry.get("notes") or "").strip(),
                "stale_issue_numbers": entry_stale_issue_numbers,
                "url": _issue_target_url(
                    repo=repo,
                    target=target,
                    url=str(entry.get("url") or ""),
                ),
            }
        )
    return linked_entries


def build_corpus_freshness_issue_drafts(
    *,
    artifact: dict[str, Any],
    freshness_map_path: Path,
    repo: str,
    client: GitHubTruthClient | None = None,
) -> list[dict[str, Any]]:
    corpus = dict(artifact.get("corpus") or {})
    corpus_id = str(corpus.get("corpus_id") or "").strip()
    revision = int(corpus.get("revision", 0) or 0)
    stale_closed_issues = [
        dict(item)
        for item in list((artifact.get("corpus_freshness") or {}).get("stale_closed_issues") or [])
        if isinstance(item, dict)
    ]
    if not corpus_id or not stale_closed_issues:
        return []
    if _linked_corpus_freshness_entries(
        artifact=artifact,
        freshness_map_path=freshness_map_path,
        repo=repo,
        client=client,
    ):
        return []

    title = f"[TW-02] Restock stale issues in {corpus_id} rev-{revision}"
    stale_issue_numbers = [
        int(item.get("issue_number", 0) or 0)
        for item in stale_closed_issues
        if int(item.get("issue_number", 0) or 0) > 0
    ]
    stale_issue_lines: list[str] = []
    for item in stale_closed_issues:
        issue_number = int(item.get("issue_number", 0) or 0)
        if issue_number <= 0:
            continue
        issue_url = str(item.get("issue_url") or "").strip() or _issue_target_url(
            repo=repo,
            target=f"#{issue_number}",
        )
        truth_state = str(item.get("truth_state") or "").strip() or "n/a"
        stale_issue_lines.append(
            f"- #{issue_number} "
            f"`{str(item.get('issue_title') or '').strip()}` "
            f"({issue_url}, truth `{truth_state}`)"
        )
    stale_issue_lines_text = "\n".join(stale_issue_lines) or "- none"
    body = (
        "## Goal\n"
        "Refresh the fixed benchmark corpus so TW-02 only measures live bounded issues.\n\n"
        "## Why now\n"
        f"The recurring truth artifact for `{corpus_id}` revision `{revision}` reports "
        f"{len(stale_closed_issues)} stale closed corpus issue(s), so the repo-tracked "
        "benchmark surface is publishing a known stale membership alert.\n\n"
        "## Evidence\n"
        f"- Corpus id: `{corpus_id}`\n"
        f"- Revision: `{revision}`\n"
        f"- Freshness map: `{_repo_stable_path(freshness_map_path)}`\n"
        f"- Stale closed issue count: {len(stale_closed_issues)}\n\n"
        "### Stale closed issues\n"
        f"{stale_issue_lines_text}\n\n"
        "## Acceptance\n"
        "- Update `docs/benchmarks/corpus.json` with an explicit revision that retires or replaces each stale closed issue.\n"
        "- Re-run the recurring benchmark truth publication until `corpus_freshness.status` is `fresh`.\n"
        "- Keep the recurring TW-02 status surface linked to this follow-up issue until the stale set is cleared.\n"
    )
    return [
        {
            "corpus_id": corpus_id,
            "revision": revision,
            "stale_issue_numbers": stale_issue_numbers,
            "title": title,
            "body": body,
        }
    ]


def find_existing_issue_by_title(*, repo: str, title: str) -> dict[str, Any] | None:
    result = subprocess.run(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "all",
            "--search",
            title,
            "--json",
            "number,title,url,state",
            "--limit",
            "100",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "gh issue list failed")
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        return None
    for item in payload:
        if not isinstance(item, dict):
            continue
        if str(item.get("title") or "").strip() != title:
            continue
        if str(item.get("state") or "").strip().lower() != "open":
            continue
        number = int(item.get("number", 0) or 0)
        if number <= 0:
            continue
        return {
            "number": number,
            "title": str(item.get("title") or "").strip(),
            "url": str(item.get("url") or "").strip(),
            "state": str(item.get("state") or "").strip().lower(),
        }
    return None


def create_issue_for_draft(*, repo: str, draft: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run(
        [
            "gh",
            "issue",
            "create",
            "--repo",
            repo,
            "--title",
            str(draft.get("title") or "").strip(),
            "--body",
            str(draft.get("body") or "").strip(),
            "--label",
            "boss-ready",
            "--label",
            "autonomous",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "gh issue create failed")
    url = str(result.stdout or "").strip().splitlines()[-1].strip()
    match = re.search(r"/issues/(\d+)$", url)
    if not match:
        raise RuntimeError(f"could not parse issue URL from gh output: {url}")
    return {
        "number": int(match.group(1)),
        "title": str(draft.get("title") or "").strip(),
        "url": url,
        "state": "open",
    }


def _upsert_corpus_freshness_entry(
    *,
    entries_by_key: dict[tuple[str, int], dict[str, Any]],
    draft: dict[str, Any],
    issue: dict[str, Any],
) -> None:
    corpus_id = str(draft.get("corpus_id") or "").strip()
    revision = int(draft.get("revision", 0) or 0)
    stale_issue_numbers = [
        int(item)
        for item in list(draft.get("stale_issue_numbers") or [])
        if isinstance(item, int) and item > 0
    ]
    existing = dict(
        entries_by_key.get(_freshness_entry_key(corpus_id=corpus_id, revision=revision), {}) or {}
    )
    notes = str(existing.get("notes") or "").strip()
    entries_by_key[_freshness_entry_key(corpus_id=corpus_id, revision=revision)] = {
        "corpus_id": corpus_id,
        "revision": revision,
        "stale_issue_numbers": stale_issue_numbers,
        "target_kind": "issue",
        "target": f"#{int(issue['number'])}",
        "title": str(issue.get("title") or "").strip(),
        "url": str(issue.get("url") or "").strip(),
        "notes": notes or "Auto-linked by recurring TW-02 publication.",
    }


def ensure_corpus_freshness_issue_linkage(
    *,
    issue_drafts: list[dict[str, Any]],
    freshness_map_path: Path,
    repo: str,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    payload = load_corpus_freshness_map_payload(freshness_map_path)
    entries_by_key = {
        _freshness_entry_key(
            corpus_id=str(entry.get("corpus_id") or "").strip(),
            revision=int(entry.get("revision", 0) or 0),
        ): dict(entry)
        for entry in list(payload.get("entries") or [])
        if isinstance(entry, dict) and str(entry.get("corpus_id") or "").strip()
    }
    results: list[dict[str, Any]] = []
    changed = False

    for draft in issue_drafts:
        corpus_id = str(draft.get("corpus_id") or "").strip()
        revision = int(draft.get("revision", 0) or 0)
        title = str(draft.get("title") or "").strip()
        if not corpus_id or not title:
            continue
        try:
            existing = find_existing_issue_by_title(repo=repo, title=title)
            if existing:
                _upsert_corpus_freshness_entry(
                    entries_by_key=entries_by_key,
                    draft=draft,
                    issue=existing,
                )
                results.append(
                    {
                        "corpus_id": corpus_id,
                        "revision": revision,
                        "action": "linked_existing_issue",
                        "target_kind": "issue",
                        "target": f"#{existing['number']}",
                        "url": existing["url"],
                    }
                )
                changed = True
                continue
            if dry_run:
                results.append(
                    {
                        "corpus_id": corpus_id,
                        "revision": revision,
                        "action": "dry_run_issue_create",
                        "target_kind": "issue",
                        "target": title,
                    }
                )
                continue
            created = create_issue_for_draft(repo=repo, draft=draft)
            _upsert_corpus_freshness_entry(
                entries_by_key=entries_by_key,
                draft=draft,
                issue=created,
            )
            results.append(
                {
                    "corpus_id": corpus_id,
                    "revision": revision,
                    "action": "created_issue",
                    "target_kind": "issue",
                    "target": f"#{created['number']}",
                    "url": created["url"],
                }
            )
            changed = True
        except (RuntimeError, subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as exc:
            results.append(
                {
                    "corpus_id": corpus_id,
                    "revision": revision,
                    "action": "error",
                    "error": str(exc),
                }
            )

    if changed and not dry_run:
        write_corpus_freshness_map_payload(
            freshness_map_path,
            {
                "schema_version": int(payload.get("schema_version", 1) or 1),
                "entries": list(entries_by_key.values()),
            },
        )
    return results


def attach_corpus_freshness_follow_up(
    *,
    artifact: dict[str, Any],
    freshness_map_path: Path,
    repo: str,
    issue_linkage_results: list[dict[str, Any]] | None = None,
    client: GitHubTruthClient | None = None,
) -> dict[str, Any]:
    linked_issues = _linked_corpus_freshness_entries(
        artifact=artifact,
        freshness_map_path=freshness_map_path,
        repo=repo,
        client=client,
    )
    issue_drafts = build_corpus_freshness_issue_drafts(
        artifact=artifact,
        freshness_map_path=freshness_map_path,
        repo=repo,
        client=client,
    )
    corpus_freshness = dict(artifact.get("corpus_freshness") or {})
    corpus_freshness.update(
        {
            "issue_map_path": _repo_stable_path(freshness_map_path),
            "linked_issues": linked_issues,
            "linked_issue_count": len(linked_issues),
            "issue_drafts": issue_drafts,
            "unlinked_issue_count": len(issue_drafts),
            "issue_linkage_results": issue_linkage_results or [],
        }
    )
    artifact["corpus_freshness"] = corpus_freshness
    return artifact


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
    freshness_map_path: Path | None = None,
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
    artifact = {
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
        "corpus_freshness": _corpus_freshness(records),
        "issues": [record.to_dict() for record in records],
    }
    if freshness_map_path is not None:
        return attach_corpus_freshness_follow_up(
            artifact=artifact,
            freshness_map_path=freshness_map_path,
            repo=repo,
            client=truth_client,
        )
    return artifact


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
    parser.add_argument(
        "--freshness-map",
        type=Path,
        default=DEFAULT_FRESHNESS_MAP_PATH,
        help=f"Tracked benchmark corpus freshness map (default: {DEFAULT_FRESHNESS_MAP_PATH})",
    )
    parser.add_argument(
        "--ensure-issues",
        action="store_true",
        help="Create or relink a bounded follow-up issue when stale closed corpus issues are detected.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics_file = resolve_metrics_path(args.metrics_file)
    corpus_path = args.corpus.resolve()
    if not metrics_file.exists():
        raise SystemExit(f"metrics file not found: {metrics_file}")
    if not corpus_path.exists():
        raise SystemExit(f"corpus file not found: {corpus_path}")
    truth_client = GitHubTruthClient()
    artifact = build_benchmark_truth_artifact(
        repo=str(args.repo),
        metrics_file=metrics_file,
        corpus_path=corpus_path,
        client=truth_client,
        freshness_map_path=args.freshness_map.resolve(),
    )
    if args.ensure_issues:
        issue_drafts = [
            dict(item)
            for item in list((artifact.get("corpus_freshness") or {}).get("issue_drafts") or [])
            if isinstance(item, dict)
        ]
        issue_linkage_results = ensure_corpus_freshness_issue_linkage(
            issue_drafts=issue_drafts,
            freshness_map_path=args.freshness_map.resolve(),
            repo=str(args.repo),
            dry_run=bool(args.dry_run),
        )
        artifact = attach_corpus_freshness_follow_up(
            artifact=artifact,
            freshness_map_path=args.freshness_map.resolve(),
            repo=str(args.repo),
            issue_linkage_results=issue_linkage_results,
            client=truth_client,
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
