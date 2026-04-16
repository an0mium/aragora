#!/usr/bin/env python3
"""Publish recurring TW-03 rescue productization artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.harvest_rescue_classes import DEFAULT_PRODUCTIZATION_MAP_PATH  # noqa: E402
from scripts.rescue_to_fixtures import (  # noqa: E402
    build_issue_drafts,
    build_issue_title,
    load_rescue_productization_report,
)

DEFAULT_RESCUE_LEDGER_PATH = Path.home() / ".aragora" / "rescue_events.jsonl"
DEFAULT_PUBLISH_DIR = REPO_ROOT / ".aragora" / "rescue_productization"


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


def resolve_published_report_path(
    *,
    publish_dir: Path,
    generated_at: str,
) -> Path:
    timestamp = _coerce_utc_datetime(generated_at).strftime("%Y%m%dT%H%M%SZ")
    return publish_dir / f"rescue-productization-{timestamp}.json"


def resolve_latest_report_path(*, publish_dir: Path) -> Path:
    return publish_dir / "latest.json"


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_productization_map_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "entries": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Productization map at {path} must be a JSON object")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"Productization map at {path} must contain an 'entries' list")
    return {
        "schema_version": int(payload.get("schema_version", 1) or 1),
        "entries": entries,
    }


def write_productization_map_payload(path: Path, payload: dict[str, Any]) -> Path:
    entries = [
        entry
        for entry in list(payload.get("entries") or [])
        if isinstance(entry, dict) and str(entry.get("class") or "").strip()
    ]
    entries.sort(key=lambda entry: str(entry.get("class") or "").strip())
    normalized = {
        "schema_version": int(payload.get("schema_version", 1) or 1),
        "entries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _issue_ref(number: int) -> str:
    return f"#{number}"


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


def _upsert_issue_entry(
    *,
    entries_by_class: dict[str, dict[str, Any]],
    class_name: str,
    issue: dict[str, Any],
) -> None:
    existing = dict(entries_by_class.get(class_name, {}) or {})
    notes = str(existing.get("notes") or "").strip()
    entries_by_class[class_name] = {
        "class": class_name,
        "target_kind": "issue",
        "target": _issue_ref(int(issue["number"])),
        "title": str(issue.get("title") or "").strip(),
        "notes": notes or "Auto-linked by recurring TW-03 harvest.",
    }


def ensure_issue_linkage(
    *,
    issue_drafts: list[dict[str, Any]],
    productization_map_path: Path,
    repo: str,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    payload = load_productization_map_payload(productization_map_path)
    entries_by_class = {
        str(entry.get("class") or "").strip(): dict(entry)
        for entry in list(payload.get("entries") or [])
        if isinstance(entry, dict) and str(entry.get("class") or "").strip()
    }
    results: list[dict[str, Any]] = []
    changed = False

    for draft in issue_drafts:
        class_name = str(draft.get("class") or "").strip()
        title = str(draft.get("title") or "").strip() or build_issue_title(class_name)
        if not class_name or not title:
            continue
        try:
            existing = find_existing_issue_by_title(repo=repo, title=title)
            if existing:
                _upsert_issue_entry(
                    entries_by_class=entries_by_class, class_name=class_name, issue=existing
                )
                results.append(
                    {
                        "class": class_name,
                        "action": "linked_existing_issue",
                        "target_kind": "issue",
                        "target": _issue_ref(existing["number"]),
                        "url": existing["url"],
                    }
                )
                changed = True
                continue
            if dry_run:
                results.append(
                    {
                        "class": class_name,
                        "action": "dry_run_issue_create",
                        "target_kind": "issue",
                        "target": title,
                    }
                )
                continue
            created = create_issue_for_draft(repo=repo, draft=draft)
            _upsert_issue_entry(
                entries_by_class=entries_by_class, class_name=class_name, issue=created
            )
            results.append(
                {
                    "class": class_name,
                    "action": "created_issue",
                    "target_kind": "issue",
                    "target": _issue_ref(created["number"]),
                    "url": created["url"],
                }
            )
            changed = True
        except (RuntimeError, subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as exc:
            results.append(
                {
                    "class": class_name,
                    "action": "error",
                    "error": str(exc),
                }
            )

    if changed and not dry_run:
        write_productization_map_payload(
            productization_map_path,
            {
                "schema_version": int(payload.get("schema_version", 1) or 1),
                "entries": list(entries_by_class.values()),
            },
        )
    return results


def build_published_report(
    *,
    ledger_path: Path,
    productization_map_path: Path,
    repo: str,
    generated_at: str | None = None,
    threshold: int = 2,
    recent_limit: int = 500,
    example_limit: int = 5,
    one_off_limit: int = 20,
    ensure_issues: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    normalized_generated_at = normalize_generated_at(generated_at)
    initial_report = load_rescue_productization_report(
        ledger_path=ledger_path,
        threshold=threshold,
        recent_limit=recent_limit,
        example_limit=example_limit,
        one_off_limit=one_off_limit,
        productization_map_path=productization_map_path,
    )
    initial_issue_drafts = build_issue_drafts(initial_report)

    issue_linkage_results: list[dict[str, Any]] = []
    if ensure_issues and initial_issue_drafts:
        issue_linkage_results = ensure_issue_linkage(
            issue_drafts=initial_issue_drafts,
            productization_map_path=productization_map_path,
            repo=repo,
            dry_run=dry_run,
        )

    final_report = load_rescue_productization_report(
        ledger_path=ledger_path,
        threshold=threshold,
        recent_limit=recent_limit,
        example_limit=example_limit,
        one_off_limit=one_off_limit,
        productization_map_path=productization_map_path,
    )
    final_issue_drafts = build_issue_drafts(final_report)
    return {
        "generated_at": normalized_generated_at,
        "repo": repo,
        "ledger_path": _repo_stable_path(ledger_path),
        "productization_map_path": _repo_stable_path(productization_map_path),
        "summary": final_report.get("summary") or {},
        "repeated_classes": final_report.get("repeated_classes") or [],
        "one_off_classes": final_report.get("one_off_classes") or [],
        "below_threshold_classes": final_report.get("below_threshold_classes") or [],
        "initial_issue_drafts": initial_issue_drafts,
        "issue_linkage_results": issue_linkage_results,
        "issue_drafts": final_issue_drafts,
    }


def publish_report_bundle(
    *,
    publish_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Path]:
    timestamped_path = write_json(
        resolve_published_report_path(
            publish_dir=publish_dir,
            generated_at=str(payload.get("generated_at") or ""),
        ),
        payload,
    )
    return {
        "timestamped": timestamped_path,
        "latest": write_json(resolve_latest_report_path(publish_dir=publish_dir), payload),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_RESCUE_LEDGER_PATH,
        help=f"Path to the RescueEvent ledger (default: {DEFAULT_RESCUE_LEDGER_PATH})",
    )
    parser.add_argument(
        "--productization-map",
        type=Path,
        default=DEFAULT_PRODUCTIZATION_MAP_PATH,
        help=f"Tracked rescue-productization map (default: {DEFAULT_PRODUCTIZATION_MAP_PATH})",
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        default=DEFAULT_PUBLISH_DIR,
        help=f"Directory for latest/timestamped report JSON (default: {DEFAULT_PUBLISH_DIR})",
    )
    parser.add_argument("--repo", default="synaptent/aragora")
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--recent-limit", type=int, default=500)
    parser.add_argument("--example-limit", type=int, default=5)
    parser.add_argument("--one-off-limit", type=int, default=20)
    parser.add_argument(
        "--ensure-issues",
        action="store_true",
        help="Create or relink bounded follow-on issues for unlinked repeated rescue classes.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_published_report(
        ledger_path=args.path,
        productization_map_path=args.productization_map,
        repo=str(args.repo),
        threshold=max(1, args.threshold),
        recent_limit=max(1, args.recent_limit),
        example_limit=max(1, args.example_limit),
        one_off_limit=max(0, args.one_off_limit),
        ensure_issues=bool(args.ensure_issues),
        dry_run=bool(args.dry_run),
    )
    published = publish_report_bundle(
        publish_dir=args.publish_dir.resolve(),
        payload=payload,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(str(published["timestamped"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
