"""Pre-dispatch prompt refinement via codebase context.

Before the boss loop dispatches a worker, this module:
1. Reads the issue + relevant codebase context
2. Finds relevant source files and test patterns
3. Produces a refined prompt with specific files, test patterns,
   and implementation rules

This improves first-attempt success rate by giving workers
the context they need instead of a vague "implement this" prompt.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_refinement_worker_env(refinement: dict[str, Any] | None) -> dict[str, str]:
    """Serialize prompt-refinement hints for worker subprocesses."""
    payload = dict(refinement or {})
    relevant_files = [
        str(item).strip() for item in payload.get("files_to_change", []) if str(item).strip()
    ]
    test_patterns = [
        str(item).strip() for item in payload.get("test_patterns", []) if str(item).strip()
    ]

    env: dict[str, str] = {}
    if relevant_files:
        env["ARAGORA_RELEVANT_FILES"] = os.pathsep.join(relevant_files)
    if test_patterns:
        env["ARAGORA_TEST_PATTERNS"] = os.pathsep.join(test_patterns)
    return env


async def refine_worker_prompt(
    issue_title: str,
    issue_body: str,
    *,
    repo_path: Path | None = None,
    timeout_seconds: float = 45.0,
) -> dict[str, Any]:
    """Refine a vague issue into a concrete worker prompt.

    Returns a dict with 'refined_prompt', 'files_to_change',
    'test_patterns', and 'constraints'.
    """
    repo = repo_path or Path.cwd()
    title = str(issue_title or "").strip()
    body = str(issue_body or "").strip()
    goal = f"{title}\n\n{body}" if body else title

    result: dict[str, Any] = {
        "refined_prompt": goal,
        "files_to_change": [],
        "test_patterns": [],
        "constraints": [],
        "context_gathered": False,
    }

    try:
        keywords = _extract_keywords(title)
        relevant_files = await _find_relevant_files(repo, keywords)
        result["files_to_change"] = relevant_files[:10]

        test_files = await _find_test_files(repo, relevant_files)
        result["test_patterns"] = test_files[:5]

        result["refined_prompt"] = _build_refined_prompt(
            goal=goal,
            relevant_files=relevant_files,
            test_files=test_files,
        )
        result["context_gathered"] = True

    except Exception as exc:
        logger.warning("Prompt refinement failed, using raw goal: %s", exc)

    return result


def _extract_keywords(title: str) -> list[str]:
    """Extract search keywords from an issue title."""
    stop_words = {
        "a",
        "an",
        "the",
        "to",
        "and",
        "or",
        "in",
        "on",
        "for",
        "with",
        "that",
        "this",
        "it",
        "is",
        "are",
        "was",
        "be",
        "do",
        "does",
        "i",
        "want",
        "make",
        "add",
        "should",
        "can",
        "way",
        "how",
        "when",
        "so",
        "let",
        "me",
        "my",
        "get",
        "see",
        "show",
    }
    words = title.lower().split()
    return [
        w.strip(".,!?;:'\"()[]{}") for w in words if w.lower() not in stop_words and len(w) > 2
    ][:8]


async def _find_relevant_files(repo: Path, keywords: list[str]) -> list[str]:
    """Find files matching keywords using grep."""
    if not keywords:
        return []

    pattern = "|".join(keywords[:4])
    try:
        proc = await asyncio.create_subprocess_exec(
            "grep",
            "-rl",
            "--include=*.py",
            "--include=*.tsx",
            "--include=*.ts",
            "-i",
            pattern,
            str(repo / "aragora"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        files = [
            str(Path(f.strip()).relative_to(repo))
            for f in stdout.decode().strip().splitlines()
            if f.strip() and "__pycache__" not in f
        ]
        return sorted(set(files))[:15]
    except Exception:
        return []


async def _find_test_files(repo: Path, source_files: list[str]) -> list[str]:
    """Find test files that correspond to the source files."""
    test_files = []
    for src in source_files[:5]:
        parts = Path(src).parts
        if parts and parts[0] == "aragora":
            test_path = Path("tests") / Path(*parts[1:])
            test_name = f"test_{test_path.stem}.py"
            test_candidate = test_path.parent / test_name
            if (repo / test_candidate).exists():
                test_files.append(str(test_candidate))
    return list(set(test_files))


def _build_refined_prompt(
    *,
    goal: str,
    relevant_files: list[str],
    test_files: list[str],
) -> str:
    """Build a refined worker prompt with codebase context."""
    sections = [goal, ""]

    if relevant_files:
        sections.append("## Relevant Files (grep matches)")
        sections.append("These files are most likely to need changes:")
        for f in relevant_files[:8]:
            sections.append(f"- {f}")
        sections.append("")

    if test_files:
        sections.append("## Existing Test Files")
        sections.append("Follow the patterns in these test files:")
        for f in test_files[:5]:
            sections.append(f"- {f}")
        sections.append("")

    sections.append("## Implementation Rules")
    sections.append("- Read the existing code in the relevant files BEFORE making changes")
    sections.append("- Follow the existing patterns in the test files")
    if test_files:
        for tf in test_files[:3]:
            sections.append(f"- Run `python -m pytest {tf} -x -q` after each change")
    else:
        sections.append("- Run `python -m pytest -x -q` on relevant tests after each change")
    sections.append("- Keep changes minimal and focused")
    sections.append("- Do not modify files outside the relevant scope")
    sections.append(
        "- **IMPORTANT: Commit your changes with `git add -A && git commit -m 'description'`**"
    )
    sections.append("- Each commit message should clearly describe what changed")

    return "\n".join(sections)
