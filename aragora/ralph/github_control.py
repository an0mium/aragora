"""GitHub PR control helpers for Ralph supervisor."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_GITHUB_URL_RE = re.compile(r"https://github\.com/[^\s]+")


class GitHubControlError(RuntimeError):
    """Raised when a GitHub control operation cannot be completed."""


@dataclass(slots=True)
class GitHubCheck:
    name: str
    status: str
    conclusion: str | None = None
    required: bool = False
    details_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "conclusion": self.conclusion,
            "required": self.required,
            "details_url": self.details_url,
        }


@dataclass(slots=True)
class GitHubGateSnapshot:
    pr_url: str
    state: str
    draft: bool
    head_branch: str | None
    base_branch: str | None
    review_decision: str | None
    merge_state_status: str | None
    merge_commit_sha: str | None
    required_checks: list[GitHubCheck]
    advisory_checks: list[GitHubCheck]
    required_checks_green: bool
    required_checks_known: bool
    required_checks_source: str | None
    disposition: str
    blocker_detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pr_url": self.pr_url,
            "state": self.state,
            "draft": self.draft,
            "head_branch": self.head_branch,
            "base_branch": self.base_branch,
            "review_decision": self.review_decision,
            "merge_state_status": self.merge_state_status,
            "merge_commit_sha": self.merge_commit_sha,
            "required_checks": [item.to_dict() for item in self.required_checks],
            "advisory_checks": [item.to_dict() for item in self.advisory_checks],
            "required_checks_green": self.required_checks_green,
            "required_checks_known": self.required_checks_known,
            "required_checks_source": self.required_checks_source,
            "disposition": self.disposition,
            "blocker_detail": self.blocker_detail,
        }


@dataclass(slots=True)
class GitHubMergeResult:
    merged: bool
    action: str
    used_admin: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "merged": self.merged,
            "action": self.action,
            "used_admin": self.used_admin,
            "detail": self.detail,
        }


class GitHubControl:
    """Thin gh CLI wrapper for Ralph PR discovery, gating, and merge."""

    def __init__(self, *, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def find_pr_for_branch(self, branch: str) -> str | None:
        result = self._run_gh(
            [
                "pr",
                "list",
                "--head",
                branch,
                "--state",
                "all",
                "--json",
                "url",
                "--limit",
                "1",
            ],
            raise_on_error=False,
            timeout=15,
        )
        if result.returncode != 0:
            return None
        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                return _optional_text(first.get("url"))
        return None

    def create_pr_for_branch(self, branch: str, target_branch: str) -> str:
        result = self._run_gh(
            ["pr", "create", "--fill", "--head", branch, "--base", target_branch],
            raise_on_error=False,
            timeout=30,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise GitHubControlError(detail or "gh pr create failed")
        url = _extract_first_github_url(result.stdout) or _extract_first_github_url(result.stderr)
        if not url:
            raise GitHubControlError("gh pr create succeeded but returned no PR URL")
        return url

    def fetch_gate_snapshot(self, pr_ref: str) -> GitHubGateSnapshot:
        view = self._pr_view(pr_ref)
        pr_url = _optional_text(view.get("url")) or str(pr_ref).strip()
        state = str(view.get("state", "")).strip().upper()
        draft = bool(view.get("isDraft", False))
        head_branch = _optional_text(view.get("headRefName"))
        base_branch = _optional_text(view.get("baseRefName"))
        review_decision = _optional_text(view.get("reviewDecision"))
        merge_state_status = _optional_text(view.get("mergeStateStatus"))
        merge_commit = view.get("mergeCommit") or {}
        merge_commit_sha = (
            _optional_text(merge_commit.get("oid")) if isinstance(merge_commit, dict) else None
        )

        repo = self._resolve_repo_slug(pr_url)
        required_contexts, known, source, detail = self._required_status_contexts(
            repo=repo,
            base_branch=base_branch,
        )

        parsed_checks = _parse_status_checks(view.get("statusCheckRollup"))
        required_checks, advisory_checks, required_green = _partition_checks(
            parsed_checks,
            required_contexts=required_contexts,
            required_known=known,
        )

        disposition = "merge_now"
        blocker_detail = detail
        if state == "MERGED":
            disposition = "merged"
        elif state and state != "OPEN":
            disposition = "blocked_nonreviewable"
            blocker_detail = blocker_detail or f"PR state is {state}."
        elif draft:
            disposition = "wait_for_review"
        elif review_decision in {"REVIEW_REQUIRED", "CHANGES_REQUESTED"}:
            disposition = "wait_for_review"
        elif not known:
            disposition = "blocked_nonreviewable"
            blocker_detail = blocker_detail or "Required-check truth could not be determined."
        elif not required_green:
            disposition = "wait_for_required_checks"
        elif merge_state_status == "DIRTY":
            disposition = "blocked_nonreviewable"
            blocker_detail = blocker_detail or "PR has merge conflicts."

        return GitHubGateSnapshot(
            pr_url=pr_url,
            state=state,
            draft=draft,
            head_branch=head_branch,
            base_branch=base_branch,
            review_decision=review_decision,
            merge_state_status=merge_state_status,
            merge_commit_sha=merge_commit_sha,
            required_checks=required_checks,
            advisory_checks=advisory_checks,
            required_checks_green=required_green,
            required_checks_known=known,
            required_checks_source=source,
            disposition=disposition,
            blocker_detail=blocker_detail,
        )

    def merge_pr(
        self,
        pr_ref: str,
        *,
        required_checks_green: bool,
        allow_admin: bool,
    ) -> GitHubMergeResult:
        if not required_checks_green:
            return GitHubMergeResult(
                merged=False,
                action="blocked",
                detail="Required checks are not green.",
            )

        normal = self._run_gh(
            ["pr", "merge", pr_ref, "--squash"],
            raise_on_error=False,
            timeout=30,
        )
        if normal.returncode == 0:
            return GitHubMergeResult(
                merged=True,
                action="merge",
                used_admin=False,
                detail=(normal.stdout or "").strip(),
            )

        stderr = (normal.stderr or normal.stdout or "").strip()
        if allow_admin and _looks_like_admin_override_candidate(stderr):
            admin = self._run_gh(
                ["pr", "merge", pr_ref, "--squash", "--admin"],
                raise_on_error=False,
                timeout=30,
            )
            if admin.returncode == 0:
                return GitHubMergeResult(
                    merged=True,
                    action="merge_admin",
                    used_admin=True,
                    detail=(admin.stdout or "").strip(),
                )
            stderr = (admin.stderr or admin.stdout or "").strip()

        return GitHubMergeResult(
            merged=False,
            action="merge_failed",
            used_admin=False,
            detail=stderr or "gh pr merge failed",
        )

    def _pr_view(self, pr_ref: str) -> dict[str, Any]:
        result = self._run_gh(
            [
                "pr",
                "view",
                pr_ref,
                "--json",
                ",".join(
                    [
                        "url",
                        "state",
                        "isDraft",
                        "headRefName",
                        "baseRefName",
                        "reviewDecision",
                        "mergeStateStatus",
                        "mergeCommit",
                        "statusCheckRollup",
                    ]
                ),
            ],
            timeout=20,
        )
        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError) as exc:
            raise GitHubControlError(f"gh pr view returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise GitHubControlError("gh pr view returned a non-object payload")
        return payload

    def _required_status_contexts(
        self,
        *,
        repo: str | None,
        base_branch: str | None,
    ) -> tuple[set[str], bool, str | None, str | None]:
        if not repo or not base_branch:
            return set(), False, None, "Missing repo or base branch for required-check lookup."

        rules_result = self._run_gh(
            ["api", f"repos/{repo}/rules/branches/{base_branch}"],
            raise_on_error=False,
            timeout=20,
        )
        if rules_result.returncode == 0:
            try:
                payload = json.loads(rules_result.stdout)
            except (json.JSONDecodeError, ValueError):
                payload = None
            contexts = _extract_required_contexts_from_rules(payload)
            return contexts, True, "ruleset", None

        protection_result = self._run_gh(
            ["api", f"repos/{repo}/branches/{base_branch}/protection"],
            raise_on_error=False,
            timeout=20,
        )
        stderr = (protection_result.stderr or "").strip().lower()
        if protection_result.returncode == 0:
            try:
                payload = json.loads(protection_result.stdout)
            except (json.JSONDecodeError, ValueError):
                payload = None
            contexts = _extract_required_contexts_from_protection(payload)
            return contexts, True, "branch_protection", None
        if "branch not protected" in stderr or "404" in stderr:
            return set(), True, "branch_protection", None

        detail = (rules_result.stderr or protection_result.stderr or "").strip()
        return set(), False, None, detail or "Unable to determine required GitHub checks."

    def _resolve_repo_slug(self, pr_ref: str) -> str | None:
        parsed = _repo_slug_from_pr_ref(pr_ref)
        if parsed:
            return parsed

        result = self._run_gh(
            ["repo", "view", "--json", "nameWithOwner"],
            raise_on_error=False,
            timeout=15,
        )
        if result.returncode != 0:
            return None
        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(payload, dict):
            return _optional_text(payload.get("nameWithOwner"))
        return None

    def _run_gh(
        self,
        args: list[str],
        *,
        raise_on_error: bool = True,
        timeout: int = 30,
    ) -> subprocess.CompletedProcess[str]:
        cmd = ["gh", *args]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.repo_root),
                timeout=timeout,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
            raise GitHubControlError(f"{' '.join(cmd)} failed: {exc}") from exc

        if raise_on_error and result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise GitHubControlError(detail or f"{' '.join(cmd)} failed")
        return result


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_first_github_url(text: str | None) -> str | None:
    if not text:
        return None
    match = _GITHUB_URL_RE.search(text)
    return match.group(0) if match else None


def _repo_slug_from_pr_ref(pr_ref: str) -> str | None:
    text = str(pr_ref).strip()
    if not text.startswith("http"):
        return None
    parsed = urlparse(text)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 4 and parts[2] == "pull":
        return f"{parts[0]}/{parts[1]}"
    return None


def _looks_like_admin_override_candidate(detail: str) -> bool:
    lowered = str(detail or "").lower()
    return any(
        token in lowered
        for token in (
            "admin",
            "administrator",
            "repository rules",
            "protected branch",
        )
    )


def _extract_required_contexts_from_rules(payload: Any) -> set[str]:
    contexts: set[str] = set()
    rules = (
        payload
        if isinstance(payload, list)
        else payload.get("rules", [])
        if isinstance(payload, dict)
        else []
    )
    if isinstance(rules, dict):
        rules = [rules]
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        parameters = rule.get("parameters")
        if not isinstance(parameters, dict):
            parameters = rule
        for key in ("required_status_checks", "requiredStatusChecks"):
            checks = parameters.get(key, [])
            if not isinstance(checks, list):
                continue
            for check in checks:
                if isinstance(check, dict):
                    name = _optional_text(
                        check.get("context") or check.get("name") or check.get("check_name")
                    )
                else:
                    name = _optional_text(check)
                if name:
                    contexts.add(name)
    return contexts


def _extract_required_contexts_from_protection(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    contexts: set[str] = set()
    required = payload.get("required_status_checks") or {}
    if not isinstance(required, dict):
        return contexts
    for name in required.get("contexts", []) or []:
        text = _optional_text(name)
        if text:
            contexts.add(text)
    for check in required.get("checks", []) or []:
        if isinstance(check, dict):
            text = _optional_text(check.get("context") or check.get("name"))
            if text:
                contexts.add(text)
    return contexts


def _parse_status_checks(payload: Any) -> list[GitHubCheck]:
    items: list[Any]
    if isinstance(payload, dict):
        if isinstance(payload.get("contexts"), dict):
            items = payload["contexts"].get("nodes", []) or []
        else:
            items = payload.get("nodes", []) or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    parsed: list[GitHubCheck] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = _optional_text(item.get("context") or item.get("name"))
        if not name:
            continue
        status = _optional_text(item.get("status") or item.get("state")) or "UNKNOWN"
        conclusion = _optional_text(item.get("conclusion"))
        details_url = _optional_text(item.get("detailsUrl") or item.get("targetUrl"))
        parsed.append(
            GitHubCheck(
                name=name,
                status=status.upper(),
                conclusion=conclusion.upper() if conclusion else None,
                details_url=details_url,
            )
        )
    return parsed


def _partition_checks(
    checks: list[GitHubCheck],
    *,
    required_contexts: set[str],
    required_known: bool,
) -> tuple[list[GitHubCheck], list[GitHubCheck], bool]:
    by_name = {check.name: check for check in checks}
    required_checks: list[GitHubCheck] = []
    advisory_checks: list[GitHubCheck] = []

    if required_known:
        for name in sorted(required_contexts):
            existing = by_name.get(name)
            if existing:
                required_checks.append(
                    GitHubCheck(
                        name=existing.name,
                        status=existing.status,
                        conclusion=existing.conclusion,
                        required=True,
                        details_url=existing.details_url,
                    )
                )
            else:
                required_checks.append(
                    GitHubCheck(name=name, status="MISSING", conclusion=None, required=True)
                )

    advisory_names = set(by_name) - set(required_contexts)
    for name in sorted(advisory_names):
        check = by_name[name]
        advisory_checks.append(
            GitHubCheck(
                name=check.name,
                status=check.status,
                conclusion=check.conclusion,
                required=False,
                details_url=check.details_url,
            )
        )

    required_green = required_known and all(_check_is_green(check) for check in required_checks)
    return required_checks, advisory_checks, required_green


def _check_is_green(check: GitHubCheck) -> bool:
    if check.status in {"QUEUED", "IN_PROGRESS", "PENDING", "EXPECTED", "MISSING", "UNKNOWN"}:
        return False
    if check.conclusion:
        return check.conclusion in {"SUCCESS", "NEUTRAL", "SKIPPED"}
    return check.status in {"COMPLETED", "SUCCESS"}
