"""Task sanitizer for boss-loop admission decisions.

This module is intentionally standalone: it reuses the existing boss validation
and spec semantics, but does not create a parallel orchestration path.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from aragora.swarm.boss_validation import (
    assess_issue_body_sanitation,
    extract_declared_new_file_paths,
    extract_issue_validation_contract,
    extract_pre_dispatch_validation_commands,
    find_missing_pre_dispatch_validation_targets,
)
from aragora.swarm.spec import SwarmSpec

_PATH_RE = re.compile(r"`(?P<path>[^`\n]+/[^`\n]+)`")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s*(?P<heading>.+?)\s*$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s+(?P<text>.+?)\s*$")
_PR_REF_RE = re.compile(
    r"(?:\bPR\s*#(?P<short>\d+)\b|\bpull request\s*#(?P<long>\d+)\b|/pull/(?P<url>\d+)\b)",
    re.IGNORECASE,
)
_COMPLEXITY_RE = re.compile(
    r"\b(?:estimated\s+complexity|complexity)\s*[:=]\s*(?P<value>small|medium|large|high|very high|complex|broad)\b",
    re.IGNORECASE,
)
_MID_SENTENCE_ENDING_RE = re.compile(
    r"(?:,|:|;|/|\(|\[|with|for|to|and|or|but|because|when|while|after|before)$",
    re.IGNORECASE,
)
_CREATE_KEYWORDS = ("create", "new", "add", "generate", "write")
_MODIFY_KEYWORDS = ("modify", "edit", "update", "change", "patch", "touch")
_SCOPE_SECTION_PREFIXES = (
    "file scope",
    "allowed write set",
    "write set",
    "files",
    "scope",
)
_VALIDATION_SECTION_PREFIXES = (
    "validation",
    "validation contract",
    "acceptance criteria",
    "acceptance",
    "test plan",
    "tests",
)
_SEVERITY = {
    "accepted": 0,
    "rewritten": 1,
    "quarantined": 2,
    "dropped": 3,
}


class SanitizationOutcome(str, Enum):
    ACCEPTED = "accepted"
    REWRITTEN = "rewritten"
    DROPPED = "dropped"
    QUARANTINED = "quarantined"


@dataclass(slots=True)
class SanitizationResult:
    outcome: SanitizationOutcome
    original_text: str
    sanitized_text: str
    reason: str
    confidence: float
    checks_failed: list[str]


@dataclass(slots=True)
class _CheckFinding:
    check_name: str
    outcome: SanitizationOutcome
    reason: str
    confidence: float
    rewritten_body: str | None = None


class TaskSanitizer:
    def __init__(self, *, repo_root: Path | None = None) -> None:
        self.repo_root = Path(repo_root or Path.cwd()).resolve()

    def sanitize(self, title: str, body: str) -> SanitizationResult:
        normalized_title = str(title or "").strip()
        normalized_body = str(body or "").strip()
        original_text = self._compose_issue_text(normalized_title, normalized_body)
        working_body = normalized_body
        findings: list[_CheckFinding] = []

        for finding in (
            self._check_description_length(working_body),
            self._check_truncation(working_body),
            self._check_duplicate_of_merged(working_body),
            self._check_contradictory_scope(working_body),
            self._check_impossible_validation(working_body, self.repo_root),
        ):
            if finding is not None:
                findings.append(finding)

        broad_scope = self._check_scope_too_broad(working_body)
        if broad_scope is not None:
            rewritten_body = self._rewrite_broad_scope(working_body)
            findings.append(
                _CheckFinding(
                    check_name=broad_scope.check_name,
                    outcome=broad_scope.outcome,
                    reason=broad_scope.reason,
                    confidence=broad_scope.confidence,
                    rewritten_body=rewritten_body,
                )
            )
            working_body = rewritten_body

        missing_validation = self._check_missing_validation(working_body)
        if missing_validation is not None:
            rewritten_body = self._rewrite_missing_validation(working_body)
            findings.append(
                _CheckFinding(
                    check_name=missing_validation.check_name,
                    outcome=missing_validation.outcome,
                    reason=missing_validation.reason,
                    confidence=missing_validation.confidence,
                    rewritten_body=rewritten_body,
                )
            )
            working_body = rewritten_body

        complexity = self._check_complexity_estimate(normalized_title, working_body)
        if complexity is not None:
            findings.append(complexity)

        if not findings:
            return SanitizationResult(
                outcome=SanitizationOutcome.ACCEPTED,
                original_text=original_text,
                sanitized_text=original_text,
                reason="task accepted unchanged",
                confidence=0.98,
                checks_failed=[],
            )

        severe = max(
            findings,
            key=lambda finding: (_SEVERITY[finding.outcome.value], -findings.index(finding)),
        )
        rewritten_body = (
            working_body if any(f.rewritten_body is not None for f in findings) else normalized_body
        )
        return SanitizationResult(
            outcome=severe.outcome,
            original_text=original_text,
            sanitized_text=self._compose_issue_text(normalized_title, rewritten_body),
            reason=severe.reason,
            confidence=severe.confidence,
            checks_failed=[finding.check_name for finding in findings],
        )

    def _check_description_length(self, body: str) -> _CheckFinding | None:
        body = str(body or "").strip()
        if len(body) >= 40:
            return None
        return _CheckFinding(
            check_name="description_length",
            outcome=SanitizationOutcome.DROPPED,
            reason="task description is too short to dispatch safely",
            confidence=0.99,
        )

    def _check_truncation(self, body: str) -> _CheckFinding | None:
        sanitized_ok, sanitation_reason = assess_issue_body_sanitation(body)
        if not sanitized_ok and sanitation_reason in {
            "task_truncated",
            "auto_decomposed_missing_task",
        }:
            return _CheckFinding(
                check_name="truncation",
                outcome=SanitizationOutcome.DROPPED,
                reason=f"task appears truncated or incomplete ({sanitation_reason})",
                confidence=0.96,
            )

        lines = [line.rstrip() for line in str(body or "").splitlines() if line.strip()]
        if any(line.endswith("\\") for line in lines):
            return _CheckFinding(
                check_name="truncation",
                outcome=SanitizationOutcome.DROPPED,
                reason="task contains a line continuation marker and appears truncated",
                confidence=0.97,
            )
        if not lines:
            return None
        last_line = lines[-1].strip()
        if _MID_SENTENCE_ENDING_RE.search(last_line):
            return _CheckFinding(
                check_name="truncation",
                outcome=SanitizationOutcome.DROPPED,
                reason="task ends mid-sentence and appears truncated",
                confidence=0.9,
            )
        return None

    def _check_contradictory_scope(self, body: str) -> _CheckFinding | None:
        actions_by_path: dict[str, set[str]] = {}
        for path, action in self._extract_scope_actions(body):
            actions_by_path.setdefault(path, set()).add(action)

        conflicts = sorted(
            path
            for path, actions in actions_by_path.items()
            if {"create", "modify"}.issubset(actions)
        )
        if not conflicts:
            return None
        return _CheckFinding(
            check_name="contradictory_scope",
            outcome=SanitizationOutcome.QUARANTINED,
            reason=f"scope contradicts itself for {', '.join(conflicts[:3])}",
            confidence=0.94,
        )

    def _check_impossible_validation(
        self,
        body: str,
        repo_root: Path | None,
    ) -> _CheckFinding | None:
        repo_path = Path(repo_root or self.repo_root).resolve()
        commands = extract_pre_dispatch_validation_commands(body)
        if not commands:
            return None

        missing = find_missing_pre_dispatch_validation_targets(commands, repo_root=repo_path)
        if not missing:
            return None

        declared_new = set(extract_declared_new_file_paths(body))
        unresolved = [path for path in missing if path not in declared_new]
        if not unresolved:
            return None

        return _CheckFinding(
            check_name="impossible_validation",
            outcome=SanitizationOutcome.QUARANTINED,
            reason=f"validation references missing target(s): {', '.join(unresolved)}",
            confidence=0.98,
        )

    def _check_scope_too_broad(self, body: str) -> _CheckFinding | None:
        paths = self._extract_scope_paths(body)
        if len(paths) <= 5:
            return None
        return _CheckFinding(
            check_name="scope_too_broad",
            outcome=SanitizationOutcome.QUARANTINED,
            reason=f"file scope spans {len(paths)} files; quarantine before dispatch",
            confidence=0.93,
        )

    def _check_missing_validation(self, body: str) -> _CheckFinding | None:
        if self._has_validation_contract(body):
            return None
        return _CheckFinding(
            check_name="missing_validation",
            outcome=SanitizationOutcome.REWRITTEN,
            reason="task is missing a validation contract; add a default bounded ruff check",
            confidence=0.91,
        )

    def _check_duplicate_of_merged(self, body: str) -> _CheckFinding | None:
        for pr_number in self._extract_pr_references(body):
            if self._is_pr_merged(pr_number):
                return _CheckFinding(
                    check_name="duplicate_merged_pr",
                    outcome=SanitizationOutcome.DROPPED,
                    reason=f"task references already-merged PR #{pr_number}",
                    confidence=0.97,
                )
        return None

    def _check_complexity_estimate(self, title: str, body: str) -> _CheckFinding | None:
        complexity = self._estimate_complexity(title, body)
        if complexity in {"small", "medium"}:
            return None

        spec = self._build_spec(title, body)
        if spec.work_orders:
            return None

        return _CheckFinding(
            check_name="complexity_estimate",
            outcome=SanitizationOutcome.QUARANTINED,
            reason=f"task complexity is {complexity} without explicit work orders",
            confidence=0.88,
        )

    def _rewrite_missing_validation(self, body: str) -> str:
        if self._has_validation_contract(body):
            return str(body or "").strip()

        target = self._most_specific_scope_path(body)
        validation_command = (
            f"python3 -m ruff check {target}" if target else "python3 -m ruff check aragora/"
        )
        base = str(body or "").rstrip()
        appendix = f"## Validation\n- {validation_command}"
        if not base:
            return appendix
        return f"{base}\n\n{appendix}"

    def _rewrite_broad_scope(self, body: str) -> str:
        chosen = self._most_specific_scope_path(body)
        if not chosen:
            return str(body or "").strip()

        lines = str(body or "").splitlines()
        rewritten: list[str] = []
        in_scope_section = False
        scope_heading_seen = False
        replaced_scope = False

        for raw_line in lines:
            heading_match = _HEADING_RE.match(raw_line)
            if heading_match:
                heading = heading_match.group("heading").strip().rstrip(":").lower()
                if in_scope_section and not self._matches_scope_section(heading):
                    in_scope_section = False
                if self._matches_scope_section(heading):
                    in_scope_section = True
                    scope_heading_seen = True
                    rewritten.append(raw_line.rstrip())
                    if not replaced_scope:
                        rewritten.append(f"- `{chosen}`")
                        replaced_scope = True
                    continue
            if in_scope_section:
                continue
            rewritten.append(raw_line.rstrip())

        result = "\n".join(line for line in rewritten).strip()
        if not scope_heading_seen:
            suffix = f"## File Scope\n- `{chosen}`"
            return f"{result}\n\n{suffix}".strip() if result else suffix
        return result

    def _compose_issue_text(self, title: str, body: str) -> str:
        title = str(title or "").strip()
        body = str(body or "").strip()
        if title and body:
            return f"{title}\n\n{body}"
        return title or body

    def _extract_pr_references(self, body: str) -> list[int]:
        pr_numbers: list[int] = []
        for match in _PR_REF_RE.finditer(str(body or "")):
            value = match.group("short") or match.group("long") or match.group("url") or ""
            try:
                number = int(value)
            except ValueError:
                continue
            if number not in pr_numbers:
                pr_numbers.append(number)
        return pr_numbers

    def _is_pr_merged(self, pr_number: int) -> bool:
        try:
            proc = subprocess.run(
                ["gh", "pr", "view", str(pr_number), "--json", "state,mergedAt"],
                cwd=self.repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
        except (FileNotFoundError, OSError):
            return False
        if proc.returncode != 0:
            return False
        try:
            payload = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            return False
        return bool(payload.get("mergedAt")) or str(payload.get("state", "")).upper() == "MERGED"

    def _build_spec(self, title: str, body: str) -> SwarmSpec:
        return SwarmSpec(
            raw_goal=str(title or "").strip(),
            refined_goal=str(body or "").strip(),
            acceptance_criteria=extract_pre_dispatch_validation_commands(body),
            constraints=SwarmSpec.infer_constraints([title, body]),
            file_scope_hints=self._extract_scope_paths(body),
            work_orders=self._extract_work_orders(body),
        )

    def _extract_work_orders(self, body: str) -> list[dict[str, str]]:
        lines = str(body or "").splitlines()
        work_orders: list[dict[str, str]] = []
        in_breakdown = False
        for raw_line in lines:
            heading_match = _HEADING_RE.match(raw_line)
            if heading_match:
                heading = heading_match.group("heading").strip().rstrip(":").lower()
                in_breakdown = heading in {"task breakdown", "work orders", "work order", "phases"}
                continue
            if re.match(r"^\s*phase\s+\d+\s*:", raw_line, re.IGNORECASE):
                work_orders.append({"title": raw_line.strip()})
                continue
            if not in_breakdown:
                continue
            item_match = _LIST_ITEM_RE.match(raw_line)
            if item_match:
                work_orders.append({"title": item_match.group("text").strip()})
        return work_orders

    def _estimate_complexity(self, title: str, body: str) -> str:
        text = self._compose_issue_text(title, body).lower()
        explicit = _COMPLEXITY_RE.search(text)
        if explicit:
            value = explicit.group("value").lower()
            if value in {"small"}:
                return "small"
            if value in {"medium"}:
                return "medium"
            return "high"

        path_count = len(self._extract_scope_paths(body))
        phase_count = len(
            re.findall(r"^\s*phase\s+\d+\s*:", str(body or ""), re.IGNORECASE | re.MULTILINE)
        )
        if path_count > 5 or phase_count >= 3:
            return "high"
        if (
            any(
                token in text
                for token in ("end-to-end", "e2e", "orchestration", "multi-step", "refactor")
            )
            and path_count >= 3
        ):
            return "high"
        return "medium"

    def _extract_scope_paths(self, body: str) -> list[str]:
        lines = str(body or "").splitlines()
        paths: list[str] = []
        in_scope_section = False
        in_validation_section = False

        for raw_line in lines:
            heading_match = _HEADING_RE.match(raw_line)
            if heading_match:
                heading = heading_match.group("heading").strip().rstrip(":").lower()
                in_scope_section = self._matches_scope_section(heading)
                in_validation_section = self._matches_validation_section(heading)
                continue

            if in_validation_section:
                continue
            if in_scope_section or any(
                keyword in raw_line.lower() for keyword in (*_CREATE_KEYWORDS, *_MODIFY_KEYWORDS)
            ):
                for path in self._extract_paths_from_line(raw_line):
                    if path not in paths:
                        paths.append(path)

        if paths:
            return paths

        for raw_line in lines:
            if any(
                token in raw_line.lower()
                for token in ("pytest ", "ruff check", "python -m pytest", "python3 -m pytest")
            ):
                continue
            for path in self._extract_paths_from_line(raw_line):
                if path not in paths:
                    paths.append(path)
        return paths

    def _extract_scope_actions(self, body: str) -> list[tuple[str, str]]:
        actions: list[tuple[str, str]] = []
        lines = str(body or "").splitlines()
        in_scope_section = False
        in_validation_section = False

        for raw_line in lines:
            heading_match = _HEADING_RE.match(raw_line)
            if heading_match:
                heading = heading_match.group("heading").strip().rstrip(":").lower()
                in_scope_section = self._matches_scope_section(heading)
                in_validation_section = self._matches_validation_section(heading)
                continue
            if in_validation_section:
                continue
            if not in_scope_section and not any(
                keyword in raw_line.lower() for keyword in (*_CREATE_KEYWORDS, *_MODIFY_KEYWORDS)
            ):
                continue

            lowered = raw_line.lower()
            line_actions: set[str] = set()
            # Exclude HTTP method context: "POST/PUT/PATCH/DELETE" should not
            # trigger "patch" as a modify keyword or "delete" as anything.
            cleaned_for_keyword_check = re.sub(
                r"\b(get|post|put|patch|delete|head|options)\s*/\s*"
                r"(get|post|put|patch|delete|head|options)\b",
                "",
                lowered,
                flags=re.IGNORECASE,
            )
            has_modify = any(keyword in cleaned_for_keyword_check for keyword in _MODIFY_KEYWORDS)
            has_create = any(keyword in cleaned_for_keyword_check for keyword in _CREATE_KEYWORDS)
            if has_modify:
                line_actions.add("modify")
                # Only treat as *also* a create when the create keyword is
                # not merely describing what the modification does.  Words
                # like "add" frequently appear in modify annotations
                # (e.g. "modify `foo.py` to add a helper") and should not
                # be misread as a file-creation intent.
                if has_create and not self._create_is_subordinate(lowered):
                    line_actions.add("create")
            elif has_create:
                line_actions.add("create")
            if not line_actions:
                continue
            for path in self._extract_paths_from_line(raw_line):
                for action in sorted(line_actions):
                    actions.append((path, action))
        return actions

    def _extract_paths_from_line(self, line: str) -> list[str]:
        paths: list[str] = []
        for match in _PATH_RE.finditer(str(line or "")):
            path = SwarmSpec.sanitize_file_scope_entry(match.group("path"))
            if path and SwarmSpec.is_concrete_repo_path_hint(path) and path not in paths:
                paths.append(path)
        for path in SwarmSpec.infer_file_scope_hints(str(line or "")):
            if path and SwarmSpec.is_concrete_repo_path_hint(path) and path not in paths:
                paths.append(path)
        return paths

    def _most_specific_scope_path(self, body: str) -> str:
        for path in self._extract_scope_paths(body):
            if SwarmSpec.is_concrete_repo_path_hint(path) and "(" not in path and ")" not in path:
                return path
        return ""

    def _has_validation_contract(self, body: str) -> bool:
        lines = str(body or "").splitlines()
        for raw_line in lines:
            heading_match = _HEADING_RE.match(raw_line)
            if not heading_match:
                continue
            heading = heading_match.group("heading").strip().rstrip(":").lower()
            if self._matches_validation_section(heading):
                return True
        if extract_issue_validation_contract(body):
            return True
        lowered = str(body or "").lower()
        return any(
            token in lowered
            for token in (
                "ruff check",
                "pytest ",
                "python -m pytest",
                "python3 -m pytest",
            )
        )

    @staticmethod
    def _matches_scope_section(heading: str) -> bool:
        return any(
            heading == prefix or heading.startswith(f"{prefix} ")
            for prefix in _SCOPE_SECTION_PREFIXES
        )

    @staticmethod
    def _create_is_subordinate(lowered_line: str) -> bool:
        """Return True when a create keyword (add/create/…) is subordinate to a modify verb.

        Patterns that indicate the create keyword describes *what* the
        modification does rather than a separate file-creation intent:
          - "modify `f.py` to add …"
          - "update `f.py` — add …"
          - "edit `f.py`: add …"
          - "change `f.py` by adding …"
        """
        # If any modify keyword precedes every create keyword, the create
        # keyword is subordinate (it describes the modification content).
        modify_positions = [lowered_line.find(kw) for kw in _MODIFY_KEYWORDS if kw in lowered_line]
        create_positions = [lowered_line.find(kw) for kw in _CREATE_KEYWORDS if kw in lowered_line]
        if not modify_positions or not create_positions:
            return False
        earliest_modify = min(modify_positions)
        earliest_create = min(create_positions)
        return earliest_modify < earliest_create

    @staticmethod
    def _matches_validation_section(heading: str) -> bool:
        return any(
            heading == prefix or heading.startswith(f"{prefix} ")
            for prefix in _VALIDATION_SECTION_PREFIXES
        )


__all__ = [
    "SanitizationOutcome",
    "SanitizationResult",
    "TaskSanitizer",
]
