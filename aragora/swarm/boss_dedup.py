"""Issue deduplication and decomposition helpers for the Boss loop."""

from __future__ import annotations

import logging
import re
from typing import Callable

from aragora.swarm.boss_feed import GitHubIssue

logger = logging.getLogger(__name__)

_DECOMPOSITION_PREFIX_RE = re.compile(r"\[from #\d+\]")
_DEDUP_TITLE_PREFIX_RE = re.compile(r"\[from #\d+\]\s*")
_FILE_SCOPE_HINT_RE = re.compile(
    r"`((?:aragora|tests|scripts|docs|docs-site|sdk|contracts)/[a-zA-Z0-9_/.*-]+(?:\.\w+)?)`"
)


def auto_decompose_stuck_issue(
    issue_number: int | str,
    issues: list[GitHubIssue],
    *,
    repo: str,
    max_retries_per_issue: int,
    label_boss_stuck: Callable[[int | str, str, str], None],
    extract_file_scope_hints: Callable[[str], list[str]],
) -> None:
    """Try to decompose a stuck issue into smaller sub-issues."""
    import subprocess

    issue = next((item for item in issues if item.number == int(issue_number)), None)
    if not issue:
        return

    decomposition_depth = len(_DECOMPOSITION_PREFIX_RE.findall(issue.title))
    max_decomposition_depth = 3
    if decomposition_depth >= max_decomposition_depth:
        label_boss_stuck(
            issue_number,
            repo,
            f"Decomposition depth {decomposition_depth} reached limit of "
            f"{max_decomposition_depth}. Needs manual attention.",
        )
        return

    try:
        pr_check = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--repo",
                repo,
                "--state",
                "merged",
                "--search",
                f"#{issue.number}",
                "--limit",
                "1",
                "--json",
                "number",
                "--jq",
                ".[0].number",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if pr_check.returncode == 0 and pr_check.stdout.strip():
            label_boss_stuck(
                issue_number,
                repo,
                f"PR #{pr_check.stdout.strip()} already merged for this issue.",
            )
            return
    except Exception:
        pass

    existing_titles: set[str] = set()
    try:
        proc = subprocess.run(
            [
                "gh",
                "issue",
                "list",
                "--repo",
                repo,
                "--label",
                "boss-ready",
                "--state",
                "open",
                "--limit",
                "100",
                "--json",
                "title",
                "--jq",
                ".[].title",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode == 0:
            existing_titles = {
                line.strip().lower() for line in proc.stdout.splitlines() if line.strip()
            }
    except Exception:
        pass

    sub_issues_created = 0
    try:
        from aragora.nomic.task_decomposer import TaskDecomposer

        decomposer = TaskDecomposer()
        result = decomposer.analyze(
            issue.body or issue.title,
            file_scope_hints=list(extract_file_scope_hints(issue.body or "")),
        )

        if result.should_decompose and result.subtasks:
            for subtask in result.subtasks[:3]:
                title = f"[from #{issue.number}] {subtask.title}"
                if title.lower() in existing_titles:
                    continue
                scope_lines = (
                    "\n".join(f"- `{path}`" for path in subtask.file_scope)
                    if subtask.file_scope
                    else "- (infer from context)"
                )
                body = (
                    f"Auto-decomposed from #{issue.number} after {max_retries_per_issue} "
                    f"failed autonomous attempts.\n\n"
                    f"## Task\n{subtask.description}\n\n"
                    f"## Files\n{scope_lines}\n\n"
                    f"## Acceptance\n"
                    f"`pytest` on the changed files passes\n\n"
                    f"## Constraints\n"
                    f"- Single-file change preferred\n"
                    f"- Under 100 lines of new/changed code\n"
                    f"- Estimated complexity: {subtask.estimated_complexity}\n"
                )
                try:
                    proc = subprocess.run(
                        [
                            "gh",
                            "issue",
                            "create",
                            "--repo",
                            repo,
                            "--title",
                            title,
                            "--body",
                            body,
                            "--label",
                            "boss-ready",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    if proc.returncode == 0:
                        sub_issues_created += 1
                except Exception:
                    pass
    except Exception as exc:
        logger.debug("Auto-decomposition failed for #%s: %s", issue_number, exc)

    if sub_issues_created > 0:
        comment = (
            f"Boss loop exhausted {max_retries_per_issue} attempts. "
            f"Auto-decomposed into {sub_issues_created} smaller sub-issues with `boss-ready` label."
        )
    else:
        comment = (
            f"Boss loop exhausted {max_retries_per_issue} attempts without "
            f"producing a deliverable. The issue may be too complex for autonomous workers."
        )

    try:
        subprocess.run(
            ["gh", "issue", "comment", str(issue_number), "--repo", repo, "--body", comment],
            capture_output=True,
            timeout=15,
        )
        subprocess.run(
            [
                "gh",
                "issue",
                "edit",
                str(issue_number),
                "--repo",
                repo,
                "--add-label",
                "boss-stuck",
            ],
            capture_output=True,
            timeout=15,
        )
    except Exception:
        pass


def extract_file_scope_hints(body: str) -> list[str]:
    """Extract file paths from an issue body."""
    cleaned = body.replace("\\`", "`")
    return _FILE_SCOPE_HINT_RE.findall(cleaned)


def label_boss_stuck(issue_number: int | str, repo: str, comment: str) -> None:
    """Label an issue as boss-stuck, remove boss-ready, and comment."""
    import subprocess

    try:
        subprocess.run(
            ["gh", "issue", "comment", str(issue_number), "--repo", repo, "--body", comment],
            capture_output=True,
            timeout=15,
        )
        subprocess.run(
            [
                "gh",
                "issue",
                "edit",
                str(issue_number),
                "--repo",
                repo,
                "--add-label",
                "boss-stuck",
                "--remove-label",
                "boss-ready",
            ],
            capture_output=True,
            timeout=15,
        )
    except Exception:
        pass


def semantic_dedup_issues(issues: list[GitHubIssue]) -> list[GitHubIssue]:
    """Use an LLM to cluster semantically duplicate issues."""
    if len(issues) < 6:
        return issues

    import asyncio
    import json as _json
    import os

    try:
        from aragora.agents.base import create_agent

        agent = None
        if os.environ.get("OPENROUTER_API_KEY"):
            agent = create_agent(
                "openrouter",
                name="dedup",
                role="proposer",
                model="deepseek/deepseek-chat",
            )
        elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            agent = create_agent(
                "gemini",
                name="dedup",
                role="proposer",
                model="gemini-2.0-flash",
            )
        if agent is None:
            return issues

        issue_map = {
            str(issue.number): _DEDUP_TITLE_PREFIX_RE.sub("", issue.title).strip()
            for issue in issues
        }
        prompt = (
            "You are deduplicating GitHub issues. Group semantically equivalent tasks. "
            "Return ONLY a JSON array of arrays: [[num1,num2],[num3],...]\n\n"
            + "\n".join(f"#{number}: {title}" for number, title in issue_map.items())
        )

        try:
            asyncio.get_running_loop()
            return issues
        except RuntimeError:
            pass

        raw = asyncio.run(agent.generate(prompt))
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return issues

        clusters = _json.loads(match.group())
        if not isinstance(clusters, list):
            return issues

        kept = {int(cluster[0]) for cluster in clusters if isinstance(cluster, list) and cluster}
        all_clustered = {
            int(number) for cluster in clusters if isinstance(cluster, list) for number in cluster
        }
        deduped = [
            issue for issue in issues if issue.number in kept or issue.number not in all_clustered
        ]
        logger.info("Semantic dedup: %d → %d issues", len(issues), len(deduped))
        return deduped
    except Exception as exc:
        logger.debug("Semantic dedup skipped: %s", exc)
        return issues
