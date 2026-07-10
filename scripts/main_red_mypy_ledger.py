#!/usr/bin/env python3
"""Render an advisory ownership ledger from truthful main-red mypy output.

The script deliberately keeps GitHub access outside the reporting boundary.
Callers provide a JSON snapshot containing issue comments and pull requests;
the default live path runs the same full typecheck tier as the required CI
check in an explicitly selected repository worktree.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TYPECHECK_COMMAND: tuple[str, ...] = ("bash", "scripts/test_tiers.sh", "typecheck")
DIAGNOSTIC_COMMAND: tuple[str, ...] = (
    "python",
    "-m",
    "mypy",
    "aragora/",
    "--ignore-missing-imports",
    "--show-error-codes",
    "--no-color-output",
    "--no-pretty",
    "--no-error-summary",
)
_MYPY_ERROR_RE = re.compile(
    r"^(?P<path>[^:\n]+):(?P<line>\d+)(?::(?P<column>\d+))?: error: "
    r"(?P<message>.*?)(?:\s{2}\[(?P<code>[^]]+)\])?$"
)
_FIELD_RE = re.compile(r"^(?P<name>[a-z_]+):\s*(?P<value>.*)$", re.IGNORECASE)


@dataclass(frozen=True)
class MypyFinding:
    path: str
    line: int
    column: int | None
    message: str
    code: str | None

    @property
    def identity(self) -> str:
        location = f"{self.path}:{self.line}"
        if self.column is not None:
            location += f":{self.column}"
        suffix = f" [{self.code}]" if self.code else ""
        return f"{location}: {self.message}{suffix}"


@dataclass(frozen=True)
class Claim:
    owner: str
    branch: str | None
    files: frozenset[str]
    expires_at: datetime


@dataclass(frozen=True)
class PullRequest:
    number: int
    branch: str
    state: str
    files: frozenset[str]


@dataclass(frozen=True)
class WorkStatus:
    kind: str
    label: str


@dataclass(frozen=True)
class BucketRow:
    bucket: str
    error_count: int
    file_count: int
    open_pr_error_count: int
    covered_error_count: int
    unclaimed_error_count: int
    status: str
    example: str


@dataclass(frozen=True)
class TypecheckResult:
    gate_exit: int
    gate_output: str
    diagnostic_exit: int | None = None
    diagnostic_output: str = ""


def _normalize_path(value: object) -> str:
    path = str(value or "").strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def parse_mypy_output(output: str) -> list[MypyFinding]:
    """Parse both ``file:line:error`` and ``file:line:column:error`` forms."""
    findings: list[MypyFinding] = []
    for raw_line in output.splitlines():
        match = _MYPY_ERROR_RE.match(raw_line.strip())
        if not match:
            continue
        findings.append(
            MypyFinding(
                path=_normalize_path(match.group("path")),
                line=int(match.group("line")),
                column=int(match.group("column")) if match.group("column") else None,
                message=match.group("message").strip(),
                code=match.group("code"),
            )
        )
    return findings


def _parse_timestamp(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _claim_blocks(body: str) -> list[str]:
    matches = list(re.finditer(r"(?m)^CLAIM\s*$", body))
    blocks: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        blocks.append(body[match.end() : end])
    return blocks


def parse_claim_comments(comments: Sequence[dict[str, Any]], *, now: datetime) -> list[Claim]:
    """Extract non-expired exact-file claims from issue comments."""
    claims: list[Claim] = []
    for comment in comments:
        body = str(comment.get("body") or "")
        for block in _claim_blocks(body):
            fields: dict[str, str] = {}
            files: list[str] = []
            reading_files = False
            for raw_line in block.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                field_match = _FIELD_RE.match(line)
                if field_match:
                    name = field_match.group("name").lower()
                    fields[name] = field_match.group("value").strip()
                    reading_files = name == "files"
                    continue
                if reading_files and line.startswith("-"):
                    normalized = _normalize_path(line[1:])
                    if normalized and "<" not in normalized:
                        files.append(normalized)

            expires_at = _parse_timestamp(fields.get("expires_at"))
            owner = fields.get("owner", "").strip()
            if not owner or not expires_at or expires_at <= now or not files:
                continue
            branch = fields.get("branch") or None
            claims.append(
                Claim(
                    owner=owner,
                    branch=branch,
                    files=frozenset(files),
                    expires_at=expires_at,
                )
            )
    return claims


def parse_pull_requests(raw_prs: Sequence[dict[str, Any]]) -> list[PullRequest]:
    pull_requests: list[PullRequest] = []
    for raw_pr in raw_prs:
        try:
            number = int(raw_pr.get("number"))
        except (TypeError, ValueError):
            continue
        branch = str(raw_pr.get("headRefName") or raw_pr.get("branch") or "").strip()
        state = str(raw_pr.get("state") or "OPEN").strip().upper()
        raw_files = raw_pr.get("files") or []
        files = {
            _normalize_path(item.get("path") if isinstance(item, dict) else item)
            for item in raw_files
        }
        files.discard("")
        pull_requests.append(
            PullRequest(number=number, branch=branch, state=state, files=frozenset(files))
        )
    return pull_requests


def load_work_snapshot(
    path: Path | None, *, now: datetime
) -> tuple[list[Claim], list[PullRequest]]:
    if path is None:
        return [], []
    raw = json.loads(path.read_text(encoding="utf-8"))
    comments = list(raw.get("comments") or [])
    body = raw.get("body")
    if body:
        comments.append({"body": body})
    return (
        parse_claim_comments(comments, now=now),
        parse_pull_requests(list(raw.get("pull_requests") or [])),
    )


def reconcile_file_statuses(
    claims: Sequence[Claim], pull_requests: Sequence[PullRequest]
) -> dict[str, WorkStatus]:
    """Map exact paths to the strongest live work state."""
    statuses: dict[str, WorkStatus] = {}
    prs_by_branch = {pr.branch: pr for pr in pull_requests if pr.branch}

    for pr in pull_requests:
        if pr.state == "MERGED":
            status = WorkStatus("merged", f"MERGED PR #{pr.number}")
        elif pr.state == "OPEN":
            status = WorkStatus("open_pr", f"OPEN PR #{pr.number}")
        else:
            continue
        for path in pr.files:
            statuses[path] = status

    for claim in claims:
        linked_pr = prs_by_branch.get(claim.branch or "")
        if linked_pr and linked_pr.state in {"OPEN", "MERGED"}:
            kind = "merged" if linked_pr.state == "MERGED" else "open_pr"
            label = f"{linked_pr.state} PR #{linked_pr.number}"
            status = WorkStatus(kind, label)
        else:
            status = WorkStatus("claimed", f"CLAIMED {claim.owner}")
        for path in claim.files:
            statuses.setdefault(path, status)
    return statuses


def bucket_for_path(path: str) -> str:
    parts = _normalize_path(path).split("/")
    if parts[0] == "aragora" and len(parts) > 1:
        return "/".join(parts[:2])
    if parts[0] in {"scripts", "tests"}:
        return parts[0]
    return parts[0] or "unknown"


def _escape_markdown(value: str) -> str:
    return " ".join(value.split()).replace("|", r"\|")


def build_bucket_rows(
    findings: Sequence[MypyFinding], statuses: dict[str, WorkStatus]
) -> list[BucketRow]:
    grouped: dict[str, list[MypyFinding]] = defaultdict(list)
    for finding in findings:
        grouped[bucket_for_path(finding.path)].append(finding)

    rows: list[BucketRow] = []
    for bucket, bucket_findings in grouped.items():
        labels: set[str] = set()
        open_pr_errors = 0
        covered_errors = 0
        unclaimed_errors = 0
        for finding in bucket_findings:
            status = statuses.get(finding.path)
            if status is None:
                unclaimed_errors += 1
                continue
            labels.add(status.label)
            covered_errors += 1
            if status.kind == "open_pr":
                open_pr_errors += 1

        if unclaimed_errors:
            status_text = "UNCLAIMED"
            if labels:
                status_text += f" ({covered_errors}/{len(bucket_findings)} covered: "
                status_text += ", ".join(sorted(labels)) + ")"
        else:
            status_text = ", ".join(sorted(labels)) or "UNCLAIMED"

        example = sorted(bucket_findings, key=lambda item: (item.path, item.line))[0].identity
        rows.append(
            BucketRow(
                bucket=bucket,
                error_count=len(bucket_findings),
                file_count=len({finding.path for finding in bucket_findings}),
                open_pr_error_count=open_pr_errors,
                covered_error_count=covered_errors,
                unclaimed_error_count=unclaimed_errors,
                status=status_text,
                example=_escape_markdown(example),
            )
        )
    return sorted(rows, key=lambda row: (-row.error_count, row.bucket))


def render_markdown(
    findings: Sequence[MypyFinding],
    rows: Sequence[BucketRow],
    *,
    head_sha: str,
    command: Sequence[str],
    command_exit: int | None,
    diagnostic_command: Sequence[str] | None,
    diagnostic_exit: int | None,
    gate_false_green: bool,
    enforce_requested: bool,
) -> str:
    error_count = len(findings)
    file_count = len({finding.path for finding in findings})
    open_pr_errors = sum(row.open_pr_error_count for row in rows)
    covered_errors = sum(row.covered_error_count for row in rows)
    unclaimed_buckets = sum(1 for row in rows if row.unclaimed_error_count)
    open_pr_pct = (open_pr_errors / error_count * 100) if error_count else 0.0

    lines = [
        "# Main-red mypy surface ledger",
        "",
        f"- Head: `{head_sha}`",
        f"- Command: `{shlex.join(command)}`",
        f"- Command exit: `{command_exit if command_exit is not None else 'captured-output'}`",
        "- Mode: advisory (never changes CI, claims, baselines, or halt state)",
    ]
    if diagnostic_command is not None:
        lines.append(f"- Diagnostic command: `{shlex.join(diagnostic_command)}`")
        lines.append(f"- Diagnostic exit: `{diagnostic_exit}`")
    if gate_false_green:
        lines.append(
            "- Gate truth mismatch: **FALSE GREEN** - canonical gate passed while the "
            "format-stable diagnostic found mypy errors"
        )
    if enforce_requested:
        lines.extend(
            [
                "- Enforcement: requested but intentionally not implemented in this advisory release",
            ]
        )
    lines.extend(
        [
            "",
            "| Bucket | Errors | Files | Status | One example error |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row.bucket}` | {row.error_count} | {row.file_count} | "
            f"{_escape_markdown(row.status)} | `{row.example}` |"
        )
    lines.extend(
        [
            "",
            f"**Totals:** {error_count} errors | {file_count} files | "
            f"{open_pr_pct:.1f}% covered by open PRs ({open_pr_errors}) | "
            f"{covered_errors} errors covered by any live work | "
            f"{unclaimed_buckets} unclaimed buckets",
        ]
    )
    if command_exit not in (None, 0) and not findings:
        lines.extend(
            [
                "",
                "> Warning: the typecheck command failed but emitted no parseable mypy errors; "
                "this snapshot is not sufficient for drain accounting.",
            ]
        )
    return "\n".join(lines) + "\n"


def run_typecheck(repo_root: Path, *, timeout: int) -> TypecheckResult:
    gate = subprocess.run(
        TYPECHECK_COMMAND,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    gate_output = f"{gate.stdout}\n{gate.stderr}"
    if parse_mypy_output(gate_output):
        return TypecheckResult(gate_exit=gate.returncode, gate_output=gate_output)

    diagnostic = subprocess.run(
        (sys.executable, *DIAGNOSTIC_COMMAND[1:]),
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return TypecheckResult(
        gate_exit=gate.returncode,
        gate_output=gate_output,
        diagnostic_exit=diagnostic.returncode,
        diagnostic_output=f"{diagnostic.stdout}\n{diagnostic.stderr}",
    )


def resolve_typecheck_findings(result: TypecheckResult) -> tuple[list[MypyFinding], bool]:
    """Prefer parseable gate output; otherwise use the format-stable diagnostic."""
    gate_findings = parse_mypy_output(result.gate_output)
    if gate_findings:
        return gate_findings, False
    diagnostic_findings = parse_mypy_output(result.diagnostic_output)
    return diagnostic_findings, bool(result.gate_exit == 0 and diagnostic_findings)


def _git_head(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Pristine repository worktree in which to run the canonical typecheck tier.",
    )
    parser.add_argument(
        "--claims-json",
        type=Path,
        help="Offline snapshot with issue comments and pull_requests file lists.",
    )
    parser.add_argument(
        "--typecheck-output",
        type=Path,
        help="Parse captured typecheck output instead of invoking the command.",
    )
    parser.add_argument("--output", type=Path, help="Also write the Markdown report here.")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--now", help="RFC3339 time used to expire claims deterministically.")
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Reserved stub. Records intent but remains advisory and exits zero.",
    )
    args = parser.parse_args(argv)

    now = _parse_timestamp(args.now) if args.now else datetime.now(timezone.utc)
    if now is None:
        parser.error("--now must be an RFC3339 timestamp with timezone")
    repo_root = args.repo_root.expanduser().resolve()
    claims, pull_requests = load_work_snapshot(args.claims_json, now=now)
    statuses = reconcile_file_statuses(claims, pull_requests)

    if args.typecheck_output:
        command_exit: int | None = None
        raw_output = args.typecheck_output.read_text(encoding="utf-8")
        findings = parse_mypy_output(raw_output)
        diagnostic_command: Sequence[str] | None = None
        diagnostic_exit: int | None = None
        gate_false_green = False
    else:
        typecheck = run_typecheck(repo_root, timeout=args.timeout)
        command_exit = typecheck.gate_exit
        findings, gate_false_green = resolve_typecheck_findings(typecheck)
        diagnostic_command = DIAGNOSTIC_COMMAND if typecheck.diagnostic_exit is not None else None
        diagnostic_exit = typecheck.diagnostic_exit

    rows = build_bucket_rows(findings, statuses)
    report = render_markdown(
        findings,
        rows,
        head_sha=_git_head(repo_root),
        command=TYPECHECK_COMMAND,
        command_exit=command_exit,
        diagnostic_command=diagnostic_command,
        diagnostic_exit=diagnostic_exit,
        gate_false_green=gate_false_green,
        enforce_requested=args.enforce,
    )
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
