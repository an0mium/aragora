"""Issue body parsing, validation contract extraction, and pre-dispatch checks.

Provides utilities for sanitizing GitHub issue bodies for worker dispatch,
extracting explicit validation contracts (acceptance criteria, test commands),
running pre-dispatch validation commands, and discovering focused test files.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_VALIDATION_SECTION_PREFIXES = (
    "acceptance criteria",
    "acceptance",
    "test",
    "validation",
    "validation contract",
    "definition of done",
    "done when",
    "test plan",
)
_ISSUE_SECTION_PREFIXES = (
    "summary",
    "context",
    "background",
    "acceptance criteria",
    "acceptance",
    "test",
    "validation",
    "validation contract",
    "definition of done",
    "done when",
    "test plan",
    "scope hints",
    "file scope hints",
    "file scope",
    "implementation rules",
    "constraints",
)
_DISPATCH_CONTEXT_SECTION_PREFIXES = ("summary", "context", "background")
_VALIDATION_INLINE_PREFIXES = (
    "acceptance",
    "acceptance criteria",
    "test",
    "validation",
    "validation contract",
    "definition of done",
    "done when",
    "test plan",
)
_VALIDATION_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s*(?:\[(?: |x|X)\]\s*)?(?P<text>.+?)\s*$")
_MARKDOWN_BOLD_RE = re.compile(r"\*\*(?P<text>.+?)\*\*")
_PRE_DISPATCH_SAFE_COMMAND_PREFIXES = (
    "pytest ",
    "python -m pytest",
    "python3 -m pytest",
    "uv run pytest",
    "uv run python -m pytest",
    "aragora ",
    "python -m aragora",
    "python3 -m aragora",
)
_BACKTICK_COMMAND_RE = re.compile(r"`(?P<command>[^`]+)`")
_EXPLICIT_PYTEST_TARGET_RE = re.compile(r"(?<!\S)(?P<path>tests/\S+?\.py)(?:::\S+)?(?!\S)")
_DECLARED_NEW_FILE_RE = re.compile(r"\(\s*new(?:\s+file)?\s*\)", re.IGNORECASE)
_BACKTICK_PATH_RE = re.compile(r"`(?P<path>[^`\n]+/[^`\n]+)`")


def _ordered_unique_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _normalize_validation_line(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    # GitHub issues commonly use bold inline markers like "**Acceptance:** ..."
    normalized = _MARKDOWN_BOLD_RE.sub(lambda match: match.group("text"), normalized)
    return normalized.strip()


def _match_issue_section_prefix(normalized_lower: str) -> str | None:
    for prefix in _ISSUE_SECTION_PREFIXES:
        if normalized_lower == prefix or normalized_lower.startswith(f"{prefix} "):
            return prefix
    return None


def _normalize_dispatch_text(lines: list[str]) -> str:
    normalized: list[str] = []
    previous_blank = True
    for raw in lines:
        text = str(raw).rstrip()
        if text.strip():
            normalized.append(text)
            previous_blank = False
            continue
        if previous_blank:
            continue
        normalized.append("")
        previous_blank = True
    return "\n".join(normalized).strip()


def sanitize_issue_body_for_dispatch(issue_body: str) -> str:
    """Keep contextual issue text while dropping sections modeled elsewhere."""
    lines = [str(line).rstrip() for line in str(issue_body or "").splitlines()]
    kept: list[str] = []
    active_section: str | None = None

    for raw_line in lines:
        stripped = raw_line.strip()
        normalized = _normalize_validation_line(stripped.lstrip("#").strip())
        normalized_lower = normalized.rstrip(":").strip().lower()
        section_prefix = _match_issue_section_prefix(normalized_lower)

        if section_prefix is not None:
            active_section = section_prefix
            if section_prefix in _DISPATCH_CONTEXT_SECTION_PREFIXES:
                if kept and kept[-1].strip():
                    kept.append("")
                kept.append(normalized.rstrip(":").strip())
            continue

        if active_section and active_section not in _DISPATCH_CONTEXT_SECTION_PREFIXES:
            continue
        kept.append(raw_line.rstrip())

    return _normalize_dispatch_text(kept)


def _compose_issue_dispatch_goal(
    issue_number: int,
    issue_title: str,
    *,
    issue_body: str = "",
    refined_prompt: str = "",
) -> str:
    header = f"[Issue #{issue_number}] {issue_title}".strip()
    body = str(refined_prompt or "").strip() or sanitize_issue_body_for_dispatch(issue_body)
    if body.startswith(header):
        body = body[len(header) :].strip()
    if body:
        return f"{header}\n\n{body}"
    return header


def extract_issue_validation_contract(issue_body: str) -> list[str]:
    """Extract an explicit validation contract from a GitHub issue body.

    Supported forms:
    - bullets/checklists under headings such as "Acceptance Criteria" or "Validation"
    - inline markers such as "Validation: pytest -q ..."
    - standalone pytest commands anywhere in the issue body
    """
    lines = [str(line).rstrip() for line in str(issue_body or "").splitlines()]
    criteria: list[str] = []
    in_validation_section = False

    for raw_line in lines:
        stripped = raw_line.strip()
        normalized = _normalize_validation_line(stripped.lstrip("#").strip())
        normalized_lower = normalized.rstrip(":").strip().lower()

        inline_prefix, _, inline_value = normalized.partition(":")
        if inline_value and inline_prefix.strip().lower() in _VALIDATION_INLINE_PREFIXES:
            criteria.append(inline_value.strip())
            in_validation_section = False
            continue

        section_prefix = _match_issue_section_prefix(normalized_lower)
        if section_prefix is not None:
            in_validation_section = section_prefix in _VALIDATION_SECTION_PREFIXES
            continue

        if stripped.startswith("pytest ") or stripped.startswith("python -m pytest"):
            criteria.append(stripped)
            continue

        if not in_validation_section:
            continue

        if stripped.startswith("#"):
            in_validation_section = False
            continue

        bullet_match = _VALIDATION_BULLET_RE.match(stripped)
        if bullet_match:
            criteria.append(bullet_match.group("text"))
            continue

        if not stripped:
            continue

        if normalized.endswith(":"):
            in_validation_section = False
            continue

        criteria.append(stripped)

    return _ordered_unique_strings(criteria)


def _normalize_pre_dispatch_command(text: str) -> str:
    normalized = str(text).strip()
    if not normalized:
        return ""
    backtick_match = _BACKTICK_COMMAND_RE.search(normalized)
    if backtick_match:
        normalized = backtick_match.group("command").strip()
    if normalized.endswith(" passes."):
        normalized = normalized[: -len(" passes.")].strip()
    if normalized.startswith("aragora "):
        normalized = f"python3 -m aragora.cli.main {normalized[len('aragora ') :].strip()}"
    return normalized


def extract_pre_dispatch_validation_commands(issue_body: str) -> list[str]:
    """Return explicit validation commands that are safe to probe before dispatch."""
    commands: list[str] = []
    for item in extract_issue_validation_contract(issue_body):
        normalized = _normalize_pre_dispatch_command(item)
        if not normalized:
            continue
        if any(normalized.startswith(prefix) for prefix in _PRE_DISPATCH_SAFE_COMMAND_PREFIXES):
            commands.append(normalized)
    return _ordered_unique_strings(commands)


def find_missing_pre_dispatch_validation_targets(
    commands: list[str],
    *,
    repo_root: Path,
) -> list[str]:
    """Return explicit pytest file targets that do not exist on disk.

    This is intentionally narrower than executing the command: a failing test is
    a legitimate bug-fix lane, but a missing pytest target is usually a stale
    issue contract that should stop before dispatch.
    """

    missing: list[str] = []
    for command in commands:
        normalized = _normalize_pre_dispatch_command(command)
        lowered = normalized.lower()
        if not (
            lowered.startswith("pytest ")
            or lowered.startswith("python -m pytest ")
            or lowered.startswith("python3 -m pytest ")
            or lowered.startswith("uv run pytest ")
            or lowered.startswith("uv run python -m pytest ")
        ):
            continue
        for match in _EXPLICIT_PYTEST_TARGET_RE.finditer(normalized):
            path = match.group("path").strip()
            if path and not (repo_root / path).exists():
                missing.append(path)
    return _ordered_unique_strings(missing)


def extract_declared_new_file_paths(issue_body: str) -> list[str]:
    """Return file paths explicitly marked as new in the issue body.

    This keeps the missing-target gate strict for stale contracts while
    allowing lanes whose acceptance criteria intentionally point at a file the
    worker is expected to create.
    """

    declared: list[str] = []
    for raw_line in str(issue_body or "").splitlines():
        line = str(raw_line).strip()
        if not line or not _DECLARED_NEW_FILE_RE.search(line):
            continue
        for match in _BACKTICK_PATH_RE.finditer(line):
            path = str(match.group("path") or "").strip()
            if path:
                declared.append(path)
    return _ordered_unique_strings(declared)


def run_pre_dispatch_validation_commands(
    commands: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Execute bounded validation commands locally before spawning a worker lane."""
    results: list[dict[str, Any]] = []
    timeout = max(1, int(timeout_seconds))
    for command in commands:
        try:
            proc = subprocess.run(
                ["/bin/bash", "-lc", command],
                cwd=str(cwd),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            results.append(
                {
                    "command": command,
                    "status": "timeout",
                }
            )
            return {"satisfied": False, "results": results}
        except (FileNotFoundError, OSError) as exc:
            results.append(
                {
                    "command": command,
                    "status": "error",
                    "detail": str(exc),
                }
            )
            return {"satisfied": False, "results": results}

        results.append(
            {
                "command": command,
                "status": "passed" if proc.returncode == 0 else "failed",
                "returncode": proc.returncode,
            }
        )
        if proc.returncode != 0:
            return {"satisfied": False, "results": results}
    return {"satisfied": True, "results": results}


# ---------------------------------------------------------------------------
# Focused Test Discovery
# ---------------------------------------------------------------------------


def discover_focused_tests(
    repo_path: Path,
    *,
    base_ref: str = "origin/main",
) -> list[str]:
    """Discover test files corresponding to source files changed since *base_ref*.

    Uses the ``tests/`` mirror convention: a source file at
    ``aragora/swarm/boss_loop.py`` maps to ``tests/swarm/test_boss_loop.py``.

    Returns a list of relative paths (strings) for test files that actually
    exist on disk.  Returns an empty list when ``git`` is unavailable or the
    diff is empty.
    """
    try:
        proc = subprocess.run(
            ["git", "diff", "--name-only", base_ref + "..HEAD"],
            capture_output=True,
            text=True,
            cwd=str(repo_path),
            timeout=15,
        )
        if proc.returncode != 0:
            logger.debug("git diff failed (rc=%d): %s", proc.returncode, proc.stderr.strip())
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("discover_focused_tests: git unavailable: %s", exc)
        return []

    changed = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]

    test_paths: list[str] = []
    seen: set[str] = set()

    for filepath in changed:
        parts = Path(filepath).parts
        if not filepath.endswith(".py"):
            continue

        # Source files under aragora/ -> mirror in tests/
        if parts and parts[0] == "aragora" and len(parts) >= 2:
            test_relative = Path("tests") / Path(*parts[1:])
            test_candidate = test_relative.parent / f"test_{test_relative.stem}.py"
            candidate_str = str(test_candidate)
            if candidate_str not in seen and (repo_path / test_candidate).exists():
                seen.add(candidate_str)
                test_paths.append(candidate_str)

        # Changed files already under tests/ -> include directly
        elif parts and parts[0] == "tests" and Path(filepath).name.startswith("test_"):
            if filepath not in seen and (repo_path / filepath).exists():
                seen.add(filepath)
                test_paths.append(filepath)

    return test_paths


def _should_replace_with_focused_tests(command: str) -> bool:
    text = str(command or "").strip()
    lowered = text.lower()
    if "-k" in lowered:
        return False
    if not (lowered.startswith("pytest ") or lowered.startswith("python -m pytest ")):
        return False
    explicit_file_target = re.search(r"(?<!\S)tests/\S+\.py(?:::\S+)?(?!\S)", text)
    if explicit_file_target:
        return False
    return "tests/" in text
