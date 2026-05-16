"""Review remote PR heads and optionally run an automated fix/re-review loop."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aragora.connectors.github import GitHubConnector
from aragora.swarm.review_routing import generate_review_response
from aragora.swarm.worker_launcher import LaunchConfig, WorkerLauncher
from aragora.worktree.fleet import resolve_repo_root

logger = logging.getLogger(__name__)

UTC = timezone.utc
MAX_DIFF_CHARS = 60000
_VALID_REVIEW_STATUSES = {"passed", "changes_requested", "blocked_nonreviewable"}
_STATUS_REVIEW_EVENT_BY_STATUS = {
    "passed": "APPROVE",
    "changes_requested": "REQUEST_CHANGES",
    "blocked_nonreviewable": "COMMENT",
}
_OPTIONAL_AGENT_PLACEHOLDERS = {"", "none", "null"}


@dataclass(slots=True)
class PullRequestTarget:
    number: int
    repo: str
    url: str
    title: str
    base_ref: str
    head_ref: str
    head_sha: str
    files: list[str] = field(default_factory=list)
    mergeable: str | None = None
    review_decision: str | None = None
    is_cross_repository: bool = False


@dataclass(slots=True)
class ReviewPass:
    reviewer: str
    reviewed_at: str
    status: str
    summary: str
    findings: list[dict[str, Any]]
    candidate: dict[str, Any] = field(default_factory=dict)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""


@dataclass(slots=True)
class FixPass:
    fixer: str
    started_at: str
    completed_at: str
    status: str
    worktree_path: str
    pushed: bool
    head_sha: str
    commit_shas: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    exit_code: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str | None = None


def add_review_pr_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "review-pr",
        help="Review a live GitHub PR head and optionally run a fixer loop",
        description=(
            "Fetch the current remote PR head, run a reviewer against that truth source, "
            "persist structured findings, and optionally dispatch a fixer in a detached worktree."
        ),
    )
    parser.add_argument("pr", help="PR number or GitHub PR URL")
    parser.add_argument(
        "--repo",
        default=None,
        help="GitHub repo slug override (owner/name). Defaults to the current repo context.",
    )
    parser.add_argument(
        "--reviewer",
        default="claude",
        help="Preferred review model/provider (default: claude)",
    )
    parser.add_argument(
        "--fixer",
        default=None,
        help="Optional fixer model/provider to run after blocking findings (for example: codex)",
    )
    parser.add_argument(
        "--auto-rerun",
        action="store_true",
        help="Re-review the PR head automatically after a successful fixer push",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Directory for run artifacts (default: .aragora/review-pr under repo root)",
    )
    parser.add_argument(
        "--keep-worktree",
        action="store_true",
        help="Keep the detached fixer worktree instead of removing it after the run",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print the final run summary as JSON",
    )
    parser.add_argument(
        "--no-publish-review",
        dest="publish_review",
        action="store_false",
        help="Persist artifacts without posting any GitHub review output.",
    )
    parser.set_defaults(publish_review=True)
    parser.set_defaults(func=cmd_review_pr)


def _normalize_optional_agent(value: object) -> str | None:
    normalized = str(value or "").strip()
    if normalized.lower() in _OPTIONAL_AGENT_PLACEHOLDERS:
        return None
    return normalized


def cmd_review_pr(args: argparse.Namespace) -> int:
    repo_root = resolve_repo_root(Path.cwd())
    result = asyncio.run(
        run_review_pr_loop(
            pr_ref=str(args.pr),
            repo_root=repo_root,
            repo_override=getattr(args, "repo", None),
            reviewer=str(getattr(args, "reviewer", "claude") or "claude"),
            fixer=_normalize_optional_agent(getattr(args, "fixer", None)),
            auto_rerun=bool(getattr(args, "auto_rerun", False)),
            artifact_root=Path(getattr(args, "artifact_dir", "")).resolve()
            if getattr(args, "artifact_dir", None)
            else None,
            keep_worktree=bool(getattr(args, "keep_worktree", False)),
            publish_review=bool(getattr(args, "publish_review", True)),
        )
    )
    if getattr(args, "json_output", False):
        print(json.dumps(result, indent=2))
    else:
        _print_run_summary(result)
    final_status = str(result.get("final_status", "")).strip().lower()
    if final_status == "passed":
        return 0
    if final_status == "changes_requested":
        return 2
    return 1


async def run_review_pr_loop(
    *,
    pr_ref: str,
    repo_root: Path,
    repo_override: str | None = None,
    reviewer: str = "claude",
    fixer: str | None = None,
    auto_rerun: bool = False,
    artifact_root: Path | None = None,
    keep_worktree: bool = False,
    advisory_only: bool = True,
    publish_review: bool = True,
) -> dict[str, Any]:
    fixer = _normalize_optional_agent(fixer)
    target = _fetch_pr_target(pr_ref, repo_override=repo_override, repo_root=repo_root)
    run_dir = _artifact_run_dir(repo_root, target.number, artifact_root=artifact_root)
    review_runs: list[dict[str, Any]] = []
    fix_run: dict[str, Any] | None = None
    observed_head_sha_before = target.head_sha
    observed_head_sha_after = target.head_sha
    head_sha_stale = False
    stale_reason: str | None = None

    diff_text = _fetch_pr_diff(target)
    _write_text(run_dir / "review-1.diff", diff_text)
    review = await _run_review_pass(
        target=target,
        diff_text=diff_text,
        reviewer=reviewer,
        worker_model=fixer or _alternate_agent(reviewer),
        repo_root=repo_root,
    )
    review_runs.append(asdict(review))
    _write_json(run_dir / "review-1.json", asdict(review))

    final_status = review.status
    refreshed_after_review = _fetch_pr_target(
        str(target.number),
        repo_override=target.repo,
        repo_root=repo_root,
    )
    observed_head_sha_after = refreshed_after_review.head_sha
    if (
        observed_head_sha_before
        and observed_head_sha_after
        and observed_head_sha_before != observed_head_sha_after
    ):
        head_sha_stale = True
        final_status = "blocked_nonreviewable"
        stale_reason = (
            f"PR head changed during review: {observed_head_sha_before} -> "
            f"{observed_head_sha_after}"
        )

    if not head_sha_stale and fixer and review.status == "changes_requested":
        fix = await _run_fix_pass(
            target=target,
            review=review,
            fixer=fixer,
            repo_root=repo_root,
            keep_worktree=keep_worktree,
            run_dir=run_dir,
        )
        fix_run = asdict(fix)
        _write_json(run_dir / "fix.json", fix_run)

        if auto_rerun and fix.pushed and fix.status == "applied":
            refreshed_target = _fetch_pr_target(
                str(target.number),
                repo_override=target.repo,
                repo_root=repo_root,
            )
            rerun_diff = _fetch_pr_diff(refreshed_target)
            _write_text(run_dir / "review-2.diff", rerun_diff)
            rerun = await _run_review_pass(
                target=refreshed_target,
                diff_text=rerun_diff,
                reviewer=reviewer,
                worker_model=fixer,
                repo_root=repo_root,
            )
            review_runs.append(asdict(rerun))
            _write_json(run_dir / "review-2.json", asdict(rerun))
            final_status = rerun.status
            target = refreshed_target

    payload = {
        "pr": asdict(target),
        "reviewer": reviewer,
        "fixer": fixer,
        "fixer_requested": fixer is not None,
        "auto_rerun": auto_rerun,
        "github_review_mode": "advisory" if advisory_only else "status",
        "artifact_dir": str(run_dir),
        "review_runs": review_runs,
        "fix_run": fix_run,
        "final_status": final_status,
        "publish_review": publish_review,
        "observed_head_sha_before": observed_head_sha_before,
        "observed_head_sha_after": observed_head_sha_after,
        "head_sha_stale": head_sha_stale,
        "stale_reason": stale_reason,
    }
    if head_sha_stale:
        payload["github_review"] = {
            "posted": False,
            "event": None,
            "mode": "advisory" if advisory_only else "status",
            "url": None,
            "error": stale_reason,
        }
    elif publish_review:
        payload["github_review"] = await _publish_review_outcome(
            target=target,
            review_runs=review_runs,
            fix_run=fix_run,
            final_status=final_status,
            advisory_only=advisory_only,
        )
    else:
        payload["github_review"] = {
            "posted": False,
            "event": None,
            "mode": "advisory" if advisory_only else "status",
            "url": None,
            "error": None,
        }
    _write_json(run_dir / "run.json", payload)
    return payload


def _fetch_pr_target(
    pr_ref: str, *, repo_override: str | None, repo_root: Path
) -> PullRequestTarget:
    number = _parse_pr_number(pr_ref)
    gh_cmd = [
        "gh",
        "pr",
        "view",
        str(number),
        "--json",
        ",".join(
            [
                "number",
                "title",
                "url",
                "headRefName",
                "headRefOid",
                "baseRefName",
                "files",
                "mergeable",
                "reviewDecision",
                "isCrossRepository",
            ]
        ),
    ]
    repo_slug = repo_override or _parse_repo_from_pr_ref(pr_ref)
    if repo_slug:
        gh_cmd.extend(["--repo", repo_slug])
    payload = _run_json_command(gh_cmd, cwd=repo_root)
    repo = repo_slug or _parse_repo_from_pr_ref(str(payload.get("url", ""))) or ""
    files = []
    for item in payload.get("files", []):
        if isinstance(item, dict):
            path = str(item.get("path", "")).strip()
            if path:
                files.append(path)
    return PullRequestTarget(
        number=int(payload.get("number", number)),
        repo=repo,
        url=str(payload.get("url", "")).strip(),
        title=str(payload.get("title", "")).strip() or f"PR #{number}",
        base_ref=str(payload.get("baseRefName", "")).strip() or "main",
        head_ref=str(payload.get("headRefName", "")).strip(),
        head_sha=str(payload.get("headRefOid", "")).strip(),
        files=files,
        mergeable=str(payload.get("mergeable", "")).strip() or None,
        review_decision=str(payload.get("reviewDecision", "")).strip() or None,
        is_cross_repository=bool(payload.get("isCrossRepository", False)),
    )


def _fetch_pr_diff(target: PullRequestTarget) -> str:
    gh_cmd = ["gh", "pr", "diff", str(target.number)]
    if target.repo:
        gh_cmd.extend(["--repo", target.repo])
    proc = _run_command(gh_cmd, cwd=Path.cwd())
    diff_text = proc.stdout
    if len(diff_text) > MAX_DIFF_CHARS:
        diff_text = diff_text[:MAX_DIFF_CHARS] + f"\n\n... [truncated at {MAX_DIFF_CHARS} chars]"
    if not diff_text.strip():
        raise RuntimeError(f"PR #{target.number} has no diff to review")
    return diff_text


async def _run_review_pass(
    *,
    target: PullRequestTarget,
    diff_text: str,
    reviewer: str,
    worker_model: str,
    repo_root: Path,
) -> ReviewPass:
    prompt = _build_review_prompt(target=target, diff_text=diff_text)
    routing = await generate_review_response(
        prompt,
        worker_model=worker_model,
        preferred_review_model=reviewer,
        repo_root=repo_root,
    )
    raw_response = str(routing.get("response", "")).strip()
    parsed = _extract_first_json_object(raw_response)
    status = str(parsed.get("status", "")).strip().lower()
    if status not in _VALID_REVIEW_STATUSES:
        status = "blocked_nonreviewable"
    findings = _normalize_findings(parsed.get("findings", []))
    summary = str(parsed.get("summary", "")).strip()
    if status == "changes_requested" and not findings:
        findings = [{"title": "Blocking changes requested", "body": raw_response, "priority": "P1"}]
    if status == "blocked_nonreviewable" and not findings:
        findings = [
            {
                "title": "Review failed",
                "body": raw_response or "Reviewer returned an invalid payload.",
            }
        ]
    return ReviewPass(
        reviewer=reviewer,
        reviewed_at=_now_iso(),
        status=status,
        summary=summary,
        findings=findings,
        candidate=dict(routing.get("candidate") or {}),
        attempts=[dict(item) for item in routing.get("attempts", []) if isinstance(item, dict)],
        raw_response=raw_response,
    )


async def _run_fix_pass(
    *,
    target: PullRequestTarget,
    review: ReviewPass,
    fixer: str,
    repo_root: Path,
    keep_worktree: bool,
    run_dir: Path,
) -> FixPass:
    started_at = _now_iso()
    if target.is_cross_repository:
        return FixPass(
            fixer=fixer,
            started_at=started_at,
            completed_at=_now_iso(),
            status="unsupported",
            worktree_path="",
            pushed=False,
            head_sha="",
            error="Cross-repository PRs are review-only in the current implementation.",
        )

    if not target.head_ref:
        return FixPass(
            fixer=fixer,
            started_at=started_at,
            completed_at=_now_iso(),
            status="failed",
            worktree_path="",
            pushed=False,
            head_sha="",
            error="PR head branch is missing.",
        )

    managed_root = repo_root / ".worktrees" / "codex-auto"
    managed_root.mkdir(parents=True, exist_ok=True)
    worktree_path = Path(
        tempfile.mkdtemp(
            prefix=f"review-pr-{target.number}-",
            dir=str(managed_root),
        )
    )
    prompt_path = run_dir / "fix_prompt.md"
    prompt_text = _build_fix_prompt(target=target, review=review)
    _write_text(prompt_path, prompt_text)

    try:
        _prepare_fix_worktree(repo_root, target, worktree_path)

        work_order = {
            "work_order_id": f"review-pr-{target.number}-fix",
            "target_agent": fixer,
            "title": f"Address blocking review findings on PR #{target.number}",
            "description": prompt_text,
            "file_scope": list(target.files),
            "expected_tests": [],
            "metadata": {
                "acceptance_criteria": [
                    "Address the blocking findings without widening scope.",
                    "Leave the branch in a pushed, reviewable state.",
                ],
                "constraints": [
                    "Work only on the current PR branch.",
                    "Do not open a new PR.",
                    "Do not revert unrelated existing branch work.",
                ],
            },
        }
        launcher = WorkerLauncher(
            LaunchConfig(
                base_branch=target.base_ref or "main",
                use_managed_session_script=False,
                detach=False,
            )
        )
        worker = await launcher.launch(
            work_order,
            worktree_path=str(worktree_path),
            branch=target.head_ref,
        )
        worker = await launcher.wait(worker.work_order_id)
        pushed = False
        if worker.head_sha and worker.head_sha != worker.initial_head:
            pushed = _push_fixed_branch(worktree_path, target.head_ref)
        status = "applied" if pushed else ("no_changes" if not worker.commit_shas else "failed")
        return FixPass(
            fixer=fixer,
            started_at=started_at,
            completed_at=_now_iso(),
            status=status,
            worktree_path=str(worktree_path),
            pushed=pushed,
            head_sha=worker.head_sha,
            commit_shas=list(worker.commit_shas),
            changed_paths=list(worker.changed_paths),
            exit_code=worker.exit_code,
            stdout_tail=worker.stdout[-4000:],
            stderr_tail=worker.stderr[-4000:],
            error=None if pushed or worker.commit_shas else "Fixer produced no commit.",
        )
    except (RuntimeError, OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.exception("review-pr fixer failed for PR #%s", target.number)
        return FixPass(
            fixer=fixer,
            started_at=started_at,
            completed_at=_now_iso(),
            status="failed",
            worktree_path=str(worktree_path),
            pushed=False,
            head_sha="",
            error=str(exc),
        )
    finally:
        if not keep_worktree:
            _cleanup_worktree(repo_root, worktree_path)


def _prepare_fix_worktree(repo_root: Path, target: PullRequestTarget, worktree_path: Path) -> None:
    _run_command(
        [
            "git",
            "fetch",
            "origin",
            f"refs/heads/{target.head_ref}:refs/remotes/origin/{target.head_ref}",
        ],
        cwd=repo_root,
    )
    _run_command(
        [
            "git",
            "worktree",
            "add",
            "--detach",
            str(worktree_path),
            f"origin/{target.head_ref}",
        ],
        cwd=repo_root,
    )


def _push_fixed_branch(worktree_path: Path, head_ref: str) -> bool:
    try:
        _run_command(
            ["git", "push", "origin", f"HEAD:refs/heads/{head_ref}"],
            cwd=worktree_path,
        )
        return True
    except RuntimeError as exc:
        logger.warning("review-pr push failed for %s: %s", head_ref, exc)
        return False


def _cleanup_worktree(repo_root: Path, worktree_path: Path) -> None:
    cleanup_script = repo_root / "scripts" / "safe_worktree_cleanup.py"
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(cleanup_script),
                "--repo",
                str(repo_root),
                "remove",
                str(worktree_path),
                "--purge-path",
                "--json",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        logger.warning("could not invoke safe review-pr cleanup for %s: %s", worktree_path, exc)
        return
    if proc.returncode != 0:
        logger.warning(
            "review-pr safe cleanup skipped for %s: %s",
            worktree_path,
            proc.stdout.strip() or proc.stderr.strip(),
        )
    parent = worktree_path.parent
    if parent.exists():
        try:
            parent.rmdir()
        except OSError as exc:
            logger.debug("review-pr parent cleanup skipped for %s: %s", parent, exc)


def _build_review_prompt(*, target: PullRequestTarget, diff_text: str) -> str:
    changed_files = "\n".join(f"  - {path}" for path in target.files[:200])
    return (
        "You are reviewing the CURRENT REMOTE HEAD of a GitHub pull request. "
        "Do not rely on local worktree state. Review only what is in the provided diff.\n\n"
        "Respond with strict JSON only using this schema:\n"
        '{"status":"passed|changes_requested|blocked_nonreviewable",'
        '"summary":"short summary",'
        '"findings":[{"title":"...", "body":"...", "file":"optional/path", "priority":"P0|P1|P2|P3"}]}\n\n'
        "Focus on bugs, regressions, security issues, missing tests, and truthfulness gaps. "
        "Avoid style-only nits.\n\n"
        f"PR #{target.number}: {target.title}\n"
        f"URL: {target.url}\n"
        f"Base branch: {target.base_ref}\n"
        f"Head branch: {target.head_ref}\n"
        f"Head SHA: {target.head_sha}\n"
        f"Changed files:\n{changed_files or '  - (not provided)'}\n\n"
        f"--- DIFF START ---\n{diff_text}\n--- DIFF END ---"
    )


def _build_fix_prompt(*, target: PullRequestTarget, review: ReviewPass) -> str:
    findings_text = "\n".join(
        f"- [{item.get('priority', 'P2')}] {item.get('title', 'Finding')}: {item.get('body', '')}"
        for item in review.findings
    )
    scope_text = "\n".join(f"- {path}" for path in target.files[:200])
    return (
        f"Fix the blocking review findings on PR #{target.number}: {target.title}\n\n"
        "You are working on the existing PR branch. Address the review findings directly and do not widen scope.\n\n"
        f"PR URL: {target.url}\n"
        f"Base branch: {target.base_ref}\n"
        f"Head branch: {target.head_ref}\n\n"
        f"Changed-file scope:\n{scope_text or '- (use the relevant existing files)'}\n\n"
        f"Blocking findings:\n{findings_text}\n\n"
        "Success conditions:\n"
        "- Fix the blocking findings in the current PR branch.\n"
        "- Run any targeted validation you can infer from the touched files.\n"
        "- Commit the changes.\n"
        "- Push the current branch back to origin.\n"
        "- If a finding is already fixed on the branch, verify it and leave it alone.\n"
        "- If you hit a real blocker, stop and report the blocker truthfully.\n"
    )


def _artifact_run_dir(repo_root: Path, pr_number: int, *, artifact_root: Path | None) -> Path:
    root = artifact_root or (repo_root / ".aragora" / "review-pr")
    run_dir = root / f"pr-{pr_number}" / _timestamp_slug()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _normalize_findings(raw_findings: Any) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not isinstance(raw_findings, list):
        return findings
    for item in raw_findings:
        if isinstance(item, dict):
            title = str(item.get("title", "")).strip() or "Finding"
            body = str(item.get("body", "")).strip() or title
            finding = {
                "title": title,
                "body": body,
            }
            file_path = str(item.get("file", "")).strip()
            if file_path:
                finding["file"] = file_path
            priority = str(item.get("priority", "")).strip().upper()
            if priority:
                finding["priority"] = priority
            findings.append(finding)
            continue
        text = str(item).strip()
        if text:
            findings.append({"title": text[:120], "body": text})
    return findings


def _extract_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError as exc:
            logger.debug("skipping non-JSON fragment at offset %d: %s", index, exc)
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _parse_pr_number(pr_ref: str) -> int:
    raw = str(pr_ref).strip()
    if raw.isdigit():
        return int(raw)
    match = re.search(r"/pull/(\d+)(?:/|$)", raw)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not parse PR number from {pr_ref!r}")


def _parse_repo_from_pr_ref(pr_ref: str) -> str | None:
    match = re.search(r"github\.com/([^/]+/[^/]+)/pull/\d+(?:/|$)", str(pr_ref).strip())
    return match.group(1) if match else None


def _alternate_agent(agent: str) -> str:
    normalized = str(agent or "").strip().lower()
    return "codex" if normalized == "claude" else "claude"


def _run_json_command(cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    proc = _run_command(cmd, cwd=cwd)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from {' '.join(cmd)}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object from {' '.join(cmd)}")
    return payload


def _run_command(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()[:500]
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)} :: {detail}")
    return proc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def _publish_review_outcome(
    *,
    target: PullRequestTarget,
    review_runs: list[dict[str, Any]],
    fix_run: dict[str, Any] | None,
    final_status: str,
    advisory_only: bool = True,
) -> dict[str, Any]:
    latest_review = review_runs[-1] if review_runs else {}
    event = _github_review_event(final_status, advisory_only=advisory_only)
    pr_url = target.url or (
        f"https://github.com/{target.repo}/pull/{target.number}" if target.repo else ""
    )
    body = _build_github_review_body(
        target=target,
        latest_review=latest_review,
        fix_run=fix_run,
        final_status=final_status,
        review_run_count=len(review_runs),
        advisory_only=advisory_only,
    )
    connector = GitHubConnector(repo=target.repo or None)
    try:
        submission = await connector.submit_pr_review(
            pr_url=pr_url,
            body=body,
            event=event,
        )
    except (RuntimeError, OSError, ValueError) as exc:
        logger.warning(
            "review-pr could not publish GitHub review for PR #%s: %s", target.number, exc
        )
        submission = {"success": False, "error": str(exc)}
    response = submission.get("response")
    review_url = ""
    if isinstance(response, dict):
        review_url = str(response.get("html_url", "")).strip()
    return {
        "posted": bool(submission.get("success")),
        "event": event,
        "mode": "advisory" if advisory_only else "status",
        "url": review_url or None,
        "error": str(submission.get("error", "")).strip() or None,
    }


def _build_github_review_body(
    *,
    target: PullRequestTarget,
    latest_review: dict[str, Any],
    fix_run: dict[str, Any] | None,
    final_status: str,
    review_run_count: int,
    advisory_only: bool = True,
) -> str:
    status_title = _review_title(final_status, advisory_only=advisory_only)
    summary = str(latest_review.get("summary", "")).strip() or _default_review_summary(final_status)
    lines = [
        f"## Aragora review-pr: {status_title}",
        "",
        summary,
        "",
        f"- Review mode: `{'advisory comment only' if advisory_only else 'status review'}`",
        f"- Final status: `{final_status}`",
        f"- Reviewer: `{latest_review.get('reviewer', 'unknown')}`",
        f"- Reviewed at: `{latest_review.get('reviewed_at', 'unknown')}`",
        f"- Head SHA: `{target.head_sha or 'unknown'}`",
    ]
    if advisory_only:
        lines.extend(
            [
                "- Settlement note: machine review is advisory only; it does not approve or block merge on its own.",
            ]
        )
    candidate = latest_review.get("candidate")
    if isinstance(candidate, dict):
        candidate_label = str(candidate.get("label", "")).strip()
        if candidate_label:
            lines.append(f"- Review route: `{candidate_label}`")
    if review_run_count > 1:
        lines.append(f"- Review passes: `{review_run_count}`")
    if fix_run:
        lines.extend(
            [
                "",
                "### Fix Loop",
                f"- Fixer: `{fix_run.get('fixer', 'unknown')}`",
                f"- Status: `{fix_run.get('status', 'unknown')}`",
                f"- Pushed: `{bool(fix_run.get('pushed', False))}`",
            ]
        )
        head_sha = str(fix_run.get("head_sha", "")).strip()
        if head_sha:
            lines.append(f"- Fix head SHA: `{head_sha}`")
        error = str(fix_run.get("error", "")).strip()
        if error:
            lines.append(f"- Fix error: {error}")
    findings = latest_review.get("findings", [])
    if isinstance(findings, list) and findings:
        lines.extend(["", "### Findings"])
        for item in findings:
            if not isinstance(item, dict):
                continue
            priority = str(item.get("priority", "P2")).strip() or "P2"
            title = str(item.get("title", "Finding")).strip() or "Finding"
            body = str(item.get("body", "")).strip() or title
            file_path = str(item.get("file", "")).strip()
            suffix = f" ({file_path})" if file_path else ""
            lines.append(f"- [{priority}] {title}{suffix}: {body}")
    elif final_status == "passed":
        lines.extend(["", "No blocking findings."])
    return "\n".join(lines).strip()


def _default_review_summary(final_status: str) -> str:
    if final_status == "passed":
        return "The latest PR head passed Aragora review."
    if final_status == "changes_requested":
        return "The latest PR head has blocking issues that should be addressed before merge."
    return "Aragora could not produce a reviewable pass/fail result for the latest PR head."


def _github_review_event(final_status: str, *, advisory_only: bool) -> str:
    if advisory_only:
        return "COMMENT"
    return _STATUS_REVIEW_EVENT_BY_STATUS.get(final_status, "COMMENT")


def _review_title(final_status: str, *, advisory_only: bool) -> str:
    if advisory_only:
        return {
            "passed": "advisory pass",
            "changes_requested": "advisory findings",
            "blocked_nonreviewable": "advisory blocked",
        }.get(final_status, final_status or "advisory comment")
    return {
        "passed": "approved",
        "changes_requested": "changes requested",
        "blocked_nonreviewable": "commented",
    }.get(final_status, final_status or "commented")


def _print_run_summary(result: dict[str, Any]) -> None:
    pr = result.get("pr", {})
    final_status = result.get("final_status", "unknown")
    print(f"PR #{pr.get('number')} final status: {final_status}")
    print(f"Artifact dir: {result.get('artifact_dir')}")
    review_runs = result.get("review_runs", [])
    if review_runs:
        latest = review_runs[-1]
        print(
            "Latest review: "
            f"{latest.get('status')} via {latest.get('candidate', {}).get('label', latest.get('reviewer'))}"
        )
        findings = latest.get("findings", [])
        if findings:
            print("Findings:")
            for item in findings:
                print(
                    f"  - [{item.get('priority', 'P2')}] "
                    f"{item.get('title', 'Finding')}: {item.get('body', '')}"
                )
    fix_run = result.get("fix_run")
    if isinstance(fix_run, dict):
        print(
            "Fix run: "
            f"{fix_run.get('status')} via {fix_run.get('fixer')} "
            f"(pushed={fix_run.get('pushed', False)})"
        )
    github_review = result.get("github_review")
    if isinstance(github_review, dict):
        if github_review.get("posted"):
            print(f"GitHub review: posted {github_review.get('event')}")
        else:
            print(f"GitHub review: failed ({github_review.get('error') or 'unknown error'})")
