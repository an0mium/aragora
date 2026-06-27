#!/usr/bin/env python3
"""Extract a redacted PR-decision corpus for the Advocate Feasibility Test."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "aft.pr_decision.v1"
SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{12,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{12,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^'\"\s,}]{8,}"),
)


def redact_text(value: str) -> str:
    redacted = value
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted


def redact_payload(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    if isinstance(value, dict):
        return {str(key): redact_payload(item) for key, item in value.items()}
    return value


def deterministic_split_key(pr_number: int, *, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{pr_number}".encode("utf-8")).hexdigest()


def deterministic_split(pr_number: int, *, seed: str, holdout_ratio: float) -> str:
    digest = deterministic_split_key(pr_number, seed=seed)
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "holdout" if bucket < holdout_ratio else "train"


def deterministic_holdout_count(
    total_examples: int,
    *,
    holdout_ratio: float,
    min_holdout_count: int,
) -> int:
    ratio_count = round(total_examples * holdout_ratio)
    if total_examples >= 200:
        ratio_count = max(ratio_count, min_holdout_count)
    return max(1, min(total_examples - 1, ratio_count)) if total_examples > 1 else total_examples


def assign_deterministic_splits(
    examples: list[dict[str, Any]],
    *,
    seed: str,
    holdout_ratio: float,
    min_holdout_count: int,
) -> list[dict[str, Any]]:
    holdout_count = deterministic_holdout_count(
        len(examples),
        holdout_ratio=holdout_ratio,
        min_holdout_count=min_holdout_count,
    )
    holdout_ids = {
        int(example["pr_number"])
        for example in sorted(
            examples,
            key=lambda item: deterministic_split_key(int(item["pr_number"]), seed=seed),
        )[:holdout_count]
    }
    for example in examples:
        example["split"] = "holdout" if int(example["pr_number"]) in holdout_ids else "train"
    return examples


def _label_for_pr(pr: dict[str, Any]) -> str:
    state = str(pr.get("state") or "").upper()
    merge_state = str(pr.get("mergeStateStatus") or "").upper()
    if pr.get("mergedAt") or state == "MERGED":
        return "accept"
    if state == "CLOSED":
        return "block"
    if pr.get("isDraft"):
        return "ask_user"
    if merge_state in {"DIRTY", "BLOCKED", "UNKNOWN"}:
        return "challenge"
    return "challenge"


def _compact_files(pr: dict[str, Any]) -> list[str]:
    files = pr.get("files") or []
    if isinstance(files, list):
        result = []
        for item in files:
            if isinstance(item, dict):
                path = item.get("path") or item.get("filename")
            else:
                path = item
            if path:
                result.append(str(path))
        return sorted(set(result))[:40]
    return []


def _compact_file_paths(paths: list[Any]) -> list[str]:
    result = []
    for path in paths:
        if path:
            result.append(redact_text(str(path)))
    return sorted(set(result))[:40]


def fetch_pr_files(repo: str, pr_number: int) -> list[str]:
    cmd = [
        "gh",
        "pr",
        "view",
        str(pr_number),
        "--repo",
        repo,
        "--json",
        "files",
    ]
    result = subprocess.run(
        cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    payload = json.loads(result.stdout)
    return _compact_files({"files": payload.get("files") or []})


def enrich_examples_with_file_paths(
    examples: list[dict[str, Any]], *, repo: str
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for example in examples:
        features = dict(example.get("context_features") or {})
        changed_files = _compact_file_paths(list(features.get("changed_files") or []))
        changed_files_count = int(features.get("changed_files_count") or 0)
        if not changed_files and changed_files_count > 0 and example.get("pr_number") is not None:
            try:
                changed_files = fetch_pr_files(repo, int(example["pr_number"]))
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError):
                changed_files = []
        if changed_files:
            features["changed_files"] = changed_files
            summary = str(example.get("artifact_summary") or "")
            if " | files=" not in summary:
                example["artifact_summary"] = summary + " | files=" + ", ".join(changed_files[:12])
        example["context_features"] = features
        enriched.append(redact_payload(example))
    return enriched


def _packet_features(packet: dict[str, Any] | None) -> dict[str, Any]:
    if not packet:
        return {}
    entry = packet
    entries = packet.get("entries")
    if isinstance(entries, list) and entries:
        entry = entries[0]
    return {
        "tier": entry.get("tier"),
        "requires_human_risk_settlement": entry.get("requires_human_risk_settlement"),
        "verdict": entry.get("verdict"),
        "status": entry.get("status"),
        "admin_squash_allowed": entry.get("admin_squash_allowed"),
        "unresolved_dissent": entry.get("unresolved_dissent"),
    }


def pr_to_example(
    pr: dict[str, Any],
    *,
    seed: str,
    holdout_ratio: float,
    packet: dict[str, Any] | None = None,
    settlement: dict[str, Any] | None = None,
    log_hint: str | None = None,
) -> dict[str, Any]:
    number = int(pr["number"])
    files = _compact_files(pr)
    additions = int(pr.get("additions") or 0)
    deletions = int(pr.get("deletions") or 0)
    changed_files = int(pr.get("changedFiles") or len(files) or 0)
    title = redact_text(str(pr.get("title") or ""))
    state = str(pr.get("state") or "UNKNOWN")
    merge_state = str(pr.get("mergeStateStatus") or "UNKNOWN")
    label = _label_for_pr(pr)
    packet_context = _packet_features(packet)
    context_features: dict[str, Any] = {
        "pr_number": number,
        "state": state,
        "is_draft": bool(pr.get("isDraft")),
        "merge_state": merge_state,
        "additions": additions,
        "deletions": deletions,
        "changed_files_count": changed_files,
        "changed_files": files,
        "author_login": (pr.get("author") or {}).get("login")
        if isinstance(pr.get("author"), dict)
        else None,
        "labels": sorted(
            str(label_item.get("name") if isinstance(label_item, dict) else label_item)
            for label_item in (pr.get("labels") or [])
            if label_item
        ),
        **packet_context,
    }
    if settlement:
        context_features["settlement_action"] = settlement.get("action")
    if log_hint:
        context_features["queue_log_hint"] = redact_text(log_hint)[:240]

    summary_bits = [
        f"PR #{number}: {title}",
        f"state={state}",
        f"draft={bool(pr.get('isDraft'))}",
        f"merge_state={merge_state}",
        f"additions={additions}",
        f"deletions={deletions}",
        f"changed_files={changed_files}",
    ]
    if files:
        summary_bits.append("files=" + ", ".join(files[:12]))

    return redact_payload(
        {
            "schema_version": SCHEMA_VERSION,
            "task_type": "pr_triage",
            "artifact_id": f"pr-{number}",
            "pr_number": number,
            "url": pr.get("url"),
            "artifact_summary": " | ".join(summary_bits),
            "proposed_action": "merge",
            "context_features": context_features,
            "label": label,
            "split": deterministic_split(number, seed=seed, holdout_ratio=holdout_ratio),
            "source": {
                "github_pr": True,
                "review_packet": packet is not None,
                "settlement_receipt": settlement is not None,
                "queue_log": log_hint is not None,
            },
        }
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_prs_from_source(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("prs", "items", "pull_requests"):
            if isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, dict)]
    raise ValueError(f"Could not find PR list in {path}")


def load_examples_from_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    examples.append(payload)
    return examples


def fetch_prs(repo: str, limit: int) -> list[dict[str, Any]]:
    fields = ",".join(
        [
            "number",
            "title",
            "state",
            "isDraft",
            "mergeStateStatus",
            "headRefOid",
            "headRefName",
            "baseRefName",
            "author",
            "labels",
            "url",
            "mergedAt",
            "closedAt",
            "createdAt",
            "updatedAt",
            "additions",
            "deletions",
            "changedFiles",
        ]
    )
    cmd = [
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "all",
        "--limit",
        str(limit),
        "--json",
        fields,
    ]
    result = subprocess.run(
        cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return json.loads(result.stdout)


def _load_review_packets(paths: list[Path]) -> dict[int, dict[str, Any]]:
    packets: dict[int, dict[str, Any]] = {}
    for path in paths:
        payload = _load_json(path)
        candidates = payload.get("entries") if isinstance(payload, dict) else None
        if isinstance(candidates, list):
            for entry in candidates:
                if isinstance(entry, dict) and entry.get("pr_number") is not None:
                    packets[int(entry["pr_number"])] = entry
        elif isinstance(payload, dict) and payload.get("pr_number") is not None:
            packets[int(payload["pr_number"])] = payload
    return packets


def _load_settlements(paths: list[Path]) -> dict[int, dict[str, Any]]:
    receipts: dict[int, dict[str, Any]] = {}
    for root in paths:
        candidates = [root] if root.is_file() else sorted(root.glob("*.json"))
        for path in candidates:
            try:
                payload = _load_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            number = payload.get("pr_number") or payload.get("pr")
            if number is not None:
                receipts[int(number)] = payload
    return receipts


def _load_queue_log_hints(paths: list[Path]) -> dict[int, str]:
    hints: dict[int, str] = {}
    pattern = re.compile(r"#(?P<number>\d+).{0,160}", re.IGNORECASE)
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in pattern.finditer(text):
            number = int(match.group("number"))
            snippet = match.group(0)
            if any(word in snippet.lower() for word in ("merged", "skipped", "blocked", "closed")):
                hints.setdefault(number, snippet)
    return hints


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="synaptent/aragora")
    parser.add_argument("--source-json", type=Path, help="Offline gh-pr JSON fixture/source")
    parser.add_argument("--source-jsonl", type=Path, help="Existing corpus JSONL to enrich")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", default="aft-v0")
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument(
        "--min-holdout-count",
        type=int,
        default=50,
        help="Minimum deterministic holdout size once a corpus has at least 200 examples.",
    )
    parser.add_argument(
        "--recompute-splits",
        action="store_true",
        help="Reassign deterministic train/holdout splits when enriching an existing corpus.",
    )
    parser.add_argument(
        "--fetch-missing-files",
        action="store_true",
        help="Fetch sanitized changed-file paths for examples that only have a file count.",
    )
    parser.add_argument("--review-packet", type=Path, action="append", default=[])
    parser.add_argument("--settlement-path", type=Path, action="append", default=[])
    parser.add_argument("--queue-log", type=Path, action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not 0.0 < args.holdout_ratio < 1.0:
        raise SystemExit("--holdout-ratio must be between 0 and 1")
    if args.min_holdout_count < 0:
        raise SystemExit("--min-holdout-count must be non-negative")

    if args.source_json and args.source_jsonl:
        raise SystemExit("--source-json and --source-jsonl are mutually exclusive")

    if args.source_jsonl:
        examples = load_examples_from_jsonl(args.source_jsonl)
        if args.fetch_missing_files:
            examples = enrich_examples_with_file_paths(examples, repo=args.repo)
        if args.recompute_splits:
            examples = assign_deterministic_splits(
                examples,
                seed=args.seed,
                holdout_ratio=args.holdout_ratio,
                min_holdout_count=args.min_holdout_count,
            )
        examples.sort(key=lambda item: int(item["pr_number"]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "output": str(args.output),
                    "examples": len(examples),
                    "holdout": sum(1 for item in examples if item["split"] == "holdout"),
                    "train": sum(1 for item in examples if item["split"] == "train"),
                },
                sort_keys=True,
            )
        )
        return 0

    prs = (
        load_prs_from_source(args.source_json)
        if args.source_json
        else fetch_prs(args.repo, args.limit)
    )
    packets = _load_review_packets(args.review_packet)
    settlements = _load_settlements(args.settlement_path)
    log_hints = _load_queue_log_hints(args.queue_log)

    examples = [
        pr_to_example(
            pr,
            seed=args.seed,
            holdout_ratio=args.holdout_ratio,
            packet=packets.get(int(pr["number"])),
            settlement=settlements.get(int(pr["number"])),
            log_hint=log_hints.get(int(pr["number"])),
        )
        for pr in prs
        if pr.get("number") is not None
    ]
    examples = assign_deterministic_splits(
        examples,
        seed=args.seed,
        holdout_ratio=args.holdout_ratio,
        min_holdout_count=args.min_holdout_count,
    )
    examples.sort(key=lambda item: int(item["pr_number"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "output": str(args.output),
                "examples": len(examples),
                "holdout": sum(1 for item in examples if item["split"] == "holdout"),
                "train": sum(1 for item in examples if item["split"] == "train"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
