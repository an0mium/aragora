#!/usr/bin/env python3
"""Reconcile the live boss-ready queue against proof-first canonical lanes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.github_app_auth import github_cli_env  # noqa: E402
from aragora.swarm.proof_first_queue import classify_proof_first_queue_issue  # noqa: E402

DEFAULT_REPO = "synaptent/aragora"
DEFAULT_QUEUE_LABEL = "boss-ready"
DEFAULT_DOCS_ISSUE_TITLE = "[CS-01..03] Reconcile docs/status surfaces to current proof"
GITHUB_CONNECTIVITY_ERROR_TOKENS = (
    "error connecting to api.github.com",
    "could not resolve host: github.com",
    "api rate limit already exceeded",
    "graphql: api rate limit",
    "secondary rate limit",
)


def is_github_connectivity_error(message: str) -> bool:
    lowered = str(message or "").strip().lower()
    return any(token in lowered for token in GITHUB_CONNECTIVITY_ERROR_TOKENS)


def github_read_env() -> dict[str, str]:
    return github_cli_env()


def github_write_env() -> dict[str, str]:
    # The installation token is intentionally read-optimized for quota isolation.
    # Label mutation and issue creation must fall back to the user's gh
    # credentials, even when this process inherited an app-token gh env.
    env = github_cli_env(prefer_app=False)
    if str(env.get("ARAGORA_GITHUB_AUTH_SOURCE") or "").strip() == "github_app_installation":
        env.pop("GH_TOKEN", None)
        env.pop("GITHUB_TOKEN", None)
        env.pop("ARAGORA_GITHUB_AUTH_SOURCE", None)
    return env


def list_open_queue_issues(*, repo: str, label: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--label",
            label,
            "--json",
            "number,title,body,labels,url",
            "--limit",
            "200",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=github_read_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "gh issue list failed")
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        raise RuntimeError("gh issue list returned a non-list payload")
    return [item for item in payload if isinstance(item, dict)]


def remove_queue_label(*, repo: str, issue_number: int, label: str) -> None:
    result = subprocess.run(
        [
            "gh",
            "issue",
            "edit",
            str(issue_number),
            "--repo",
            repo,
            "--remove-label",
            label,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=github_write_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "gh issue edit failed")


def load_status_reconciliation_report(*, repo_root: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["python3", "scripts/reconcile_status_docs.py", "--json"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or result.stdout.strip() or "status reconciliation failed"
        )
    payload = json.loads(result.stdout or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("status reconciliation returned a non-object payload")
    return payload


def docs_proof_drift_detected(report: dict[str, Any]) -> bool:
    summary = dict(report.get("summary") or {})
    return int(summary.get("critical", 0) or 0) > 0 or int(summary.get("warning", 0) or 0) > 0


def build_docs_proof_drift_issue(report: dict[str, Any]) -> dict[str, Any] | None:
    if not docs_proof_drift_detected(report):
        return None
    findings = [
        dict(item)
        for item in list(report.get("findings") or [])
        if isinstance(item, dict)
        and str(item.get("severity") or "").strip().lower() in {"critical", "warning"}
    ]
    lines = [
        "## Goal",
        "Reconcile roadmap/status/positioning surfaces so external claims stay narrower than current measured proof.",
        "",
        "## Why now",
        "The recurring docs reconciliation check found proof drift that should become exactly one bounded `CS-01..03` follow-up issue.",
        "",
        "## Findings",
    ]
    for finding in findings:
        lines.append(
            f"- `{str(finding.get('severity') or '').strip()}` "
            f"`{str(finding.get('source') or '').strip()}`: "
            f"{str(finding.get('message') or '').strip()}"
        )
    lines.extend(
        [
            "",
            "## Acceptance",
            "- Update the affected docs/status surfaces so claims stay narrower than the current proof surfaces.",
            "- Keep `docs/status/NEXT_STEPS_CANONICAL.md`, `docs/status/B0_BENCHMARK_TRUTH_STATUS.md`, and `docs/status/TW03_RESCUE_PRODUCTIZATION_STATUS.md` mutually consistent.",
            "- Re-run `python3 scripts/reconcile_status_docs.py --json` until warning and critical findings are cleared.",
        ]
    )
    return {
        "title": DEFAULT_DOCS_ISSUE_TITLE,
        "body": "\n".join(lines),
        "labels": ["boss-ready", "autonomous"],
    }


def find_existing_open_issue_by_title(*, repo: str, title: str) -> dict[str, Any] | None:
    result = subprocess.run(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--search",
            title,
            "--json",
            "number,title,url",
            "--limit",
            "100",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=github_read_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "gh issue list failed")
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        return None
    for item in payload:
        if not isinstance(item, dict):
            continue
        if str(item.get("title") or "").strip() == title:
            return item
    return None


def create_issue(*, repo: str, title: str, body: str, labels: list[str]) -> dict[str, Any]:
    cmd = ["gh", "issue", "create", "--repo", repo, "--title", title, "--body", body]
    for label in labels:
        cmd.extend(["--label", label])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=github_write_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "gh issue create failed")
    url = str(result.stdout or "").strip().splitlines()[-1].strip()
    return {"title": title, "url": url}


def reconcile_proof_first_queue(
    *,
    repo: str,
    repo_root: Path,
    apply: bool = False,
    queue_label: str = DEFAULT_QUEUE_LABEL,
) -> dict[str, Any]:
    docs_report = load_status_reconciliation_report(repo_root=repo_root)
    github_status: dict[str, Any] = {"available": True}
    try:
        issues = list_open_queue_issues(repo=repo, label=queue_label)
    except RuntimeError as exc:
        if not is_github_connectivity_error(str(exc)):
            raise
        docs_issue_payload = build_docs_proof_drift_issue(docs_report)
        docs_issue_action = None
        if docs_issue_payload is not None:
            docs_issue_action = {
                "action": "deferred_github_unavailable",
                "title": docs_issue_payload["title"],
                "error": str(exc),
            }
        return {
            "repo": repo,
            "queue_label": queue_label,
            "kept": [],
            "removed": [],
            "docs_issue": docs_issue_action,
            "docs_report_summary": dict(docs_report.get("summary") or {}),
            "github_status": {
                "available": False,
                "operation": "list_open_queue_issues",
                "error": str(exc),
            },
        }
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []

    for issue in issues:
        labels = [
            str(item.get("name") or "").strip()
            for item in list(issue.get("labels") or [])
            if isinstance(item, dict)
        ]
        decision = classify_proof_first_queue_issue(
            str(issue.get("title") or "").strip(),
            str(issue.get("body") or "").strip(),
            labels=labels,
            repo_root=repo_root,
        )
        record = {
            "number": int(issue.get("number", 0) or 0),
            "title": str(issue.get("title") or "").strip(),
            "url": str(issue.get("url") or "").strip(),
            "lane": decision.lane,
            "reason": decision.reason,
        }
        if decision.allowed:
            kept.append(record)
            continue
        if apply and record["number"] > 0:
            remove_queue_label(repo=repo, issue_number=record["number"], label=queue_label)
            record["action"] = "removed_label"
        else:
            record["action"] = "would_remove_label"
        removed.append(record)

    docs_issue_payload = build_docs_proof_drift_issue(docs_report)
    docs_issue_action: dict[str, Any] | None = None
    if docs_issue_payload is not None:
        try:
            existing = find_existing_open_issue_by_title(
                repo=repo, title=docs_issue_payload["title"]
            )
        except RuntimeError as exc:
            if not is_github_connectivity_error(str(exc)):
                raise
            github_status = {
                "available": False,
                "operation": "find_existing_open_issue_by_title",
                "error": str(exc),
            }
            docs_issue_action = {
                "action": "deferred_github_unavailable",
                "title": docs_issue_payload["title"],
                "error": str(exc),
            }
        else:
            if existing is not None:
                docs_issue_action = {
                    "action": "existing_issue",
                    "title": docs_issue_payload["title"],
                    "number": int(existing.get("number", 0) or 0),
                    "url": str(existing.get("url") or "").strip(),
                }
            elif apply:
                try:
                    created = create_issue(
                        repo=repo,
                        title=str(docs_issue_payload["title"]),
                        body=str(docs_issue_payload["body"]),
                        labels=list(docs_issue_payload["labels"]),
                    )
                except RuntimeError as exc:
                    if not is_github_connectivity_error(str(exc)):
                        raise
                    github_status = {
                        "available": False,
                        "operation": "create_issue",
                        "error": str(exc),
                    }
                    docs_issue_action = {
                        "action": "deferred_github_unavailable",
                        "title": docs_issue_payload["title"],
                        "error": str(exc),
                    }
                else:
                    docs_issue_action = {
                        "action": "created_issue",
                        "title": docs_issue_payload["title"],
                        "url": str(created.get("url") or "").strip(),
                    }
            else:
                docs_issue_action = {
                    "action": "would_create_issue",
                    "title": docs_issue_payload["title"],
                }

    return {
        "repo": repo,
        "queue_label": queue_label,
        "kept": kept,
        "removed": removed,
        "docs_issue": docs_issue_action,
        "docs_report_summary": dict(docs_report.get("summary") or {}),
        "github_status": github_status,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reconcile proof-first boss-ready queue.")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--queue-label", default=DEFAULT_QUEUE_LABEL)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = reconcile_proof_first_queue(
        repo=args.repo,
        repo_root=Path(args.repo_root).resolve(),
        apply=args.apply,
        queue_label=args.queue_label,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"kept={len(report['kept'])} removed={len(report['removed'])} "
            f"docs_issue={report.get('docs_issue', {}).get('action') if report.get('docs_issue') else 'none'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
