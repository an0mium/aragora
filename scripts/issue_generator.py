#!/usr/bin/env python3
"""
Issue Generator for Nomic Loop

Scans the codebase to generate specific, actionable issues for the nomic loop
to debate. This replaces vague open-ended topics with concrete problems.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class Issue:
    """A specific, actionable issue for the nomic loop to address."""

    id: str
    title: str
    description: str
    category: Literal["bug", "perf", "debt", "feature"]
    priority: int  # 1-5 (1 = highest)
    file_hints: list[str] = field(default_factory=list)
    complexity: Literal["small", "medium", "large"] = "medium"
    source: str = ""  # Where the issue was found (e.g., "TODO", "grep", "analysis")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "priority": self.priority,
            "file_hints": self.file_hints,
            "complexity": self.complexity,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Issue":
        return cls(**data)


class IssueGenerator:
    """Generates specific, actionable issues from codebase analysis."""

    # Patterns to ignore
    IGNORE_DIRS = {
        ".git",
        "__pycache__",
        ".nomic",
        "node_modules",
        ".venv",
        "venv",
        ".comparison",
        "dist",
        "build",
    }
    IGNORE_FILES = {"package-lock.json", "yarn.lock", "poetry.lock"}

    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)
        self._issues: list[Issue] = []

    def scan_for_issues(self) -> list[Issue]:
        """Scan codebase for concrete issues."""
        issues = []

        # 1. Find TODO/FIXME comments
        issues.extend(self._extract_todo_issues())

        # 2. Find broad exception handlers
        issues.extend(self._find_exception_debt())

        # 3. Find large files (>500 LOC)
        issues.extend(self._find_large_files())

        # 4. Find untested modules
        issues.extend(self._find_untested_modules())

        # 5. Find deprecated patterns
        issues.extend(self._find_deprecated_usage())

        # Deduplicate by ID
        seen = set()
        unique_issues = []
        for issue in issues:
            if issue.id not in seen:
                seen.add(issue.id)
                unique_issues.append(issue)

        # Sort by priority
        return sorted(unique_issues, key=lambda i: (i.priority, i.title))

    def _generate_id(self, content: str) -> str:
        """Generate stable ID from content."""
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _run_grep(self, pattern: str, glob: str = "*.py") -> list[tuple[str, int, str]]:
        """Run grep and return (file, line, content) tuples."""
        try:
            result = subprocess.run(
                [
                    "grep",
                    "-rn",
                    "--include",
                    glob,
                    pattern,
                    str(self.repo_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            matches = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    file_path = parts[0]
                    # Skip ignored directories
                    if any(d in file_path for d in self.IGNORE_DIRS):
                        continue
                    try:
                        line_num = int(parts[1])
                        content = parts[2]
                        matches.append((file_path, line_num, content))
                    except ValueError:
                        continue
            return matches
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

    def _extract_todo_issues(self) -> list[Issue]:
        """Convert TODO/FIXME comments to issues."""
        issues = []

        for pattern, priority in [("FIXME", 2), ("TODO", 3), ("HACK", 3), ("XXX", 2)]:
            matches = self._run_grep(pattern)
            for file_path, line_num, content in matches:
                # Extract the actual message
                match = re.search(rf"{pattern}[:\s]+(.+)", content, re.IGNORECASE)
                if match:
                    message = match.group(1).strip()
                    # Skip very short or generic messages
                    if len(message) < 10:
                        continue

                    rel_path = Path(file_path).relative_to(self.repo_path)
                    issue = Issue(
                        id=self._generate_id(f"{file_path}:{line_num}:{pattern}"),
                        title=f"Address {pattern} in {rel_path.name}:{line_num}",
                        description=f"Found {pattern} comment: {message}\n\nLocation: {rel_path}:{line_num}",
                        category="debt",
                        priority=priority,
                        file_hints=[str(rel_path)],
                        complexity="small" if len(message) < 50 else "medium",
                        source=f"{pattern}:{file_path}:{line_num}",
                    )
                    issues.append(issue)

        return issues[:20]  # Limit to prevent overwhelming

    def _find_exception_debt(self) -> list[Issue]:
        """Find broad 'except Exception:' patterns."""
        issues = []
        pattern = r"except\s+Exception\s*:"
        matches = self._run_grep(pattern)

        # Group by file
        by_file: dict[str, list[int]] = {}
        for file_path, line_num, _ in matches:
            rel_path = str(Path(file_path).relative_to(self.repo_path))
            if rel_path not in by_file:
                by_file[rel_path] = []
            by_file[rel_path].append(line_num)

        for rel_path, line_nums in by_file.items():
            issue = Issue(
                id=self._generate_id(f"broad_exception:{rel_path}"),
                title=f"Replace {len(line_nums)} broad exception handlers in {Path(rel_path).name}",
                description=(
                    f"Found {len(line_nums)} instances of 'except Exception:' in {rel_path}.\n\n"
                    f"Lines: {', '.join(str(n) for n in line_nums[:10])}"
                    f"{'...' if len(line_nums) > 10 else ''}\n\n"
                    "These should be replaced with specific exception types for better error handling."
                ),
                category="debt",
                priority=2,
                file_hints=[rel_path],
                complexity="medium" if len(line_nums) <= 5 else "large",
                source=f"grep:except_exception:{rel_path}",
            )
            issues.append(issue)

        return issues

    def _find_large_files(self) -> list[Issue]:
        """Find files larger than 500 lines."""
        issues = []
        threshold = 500

        for py_file in self.repo_path.rglob("*.py"):
            # Skip ignored directories
            if any(d in str(py_file) for d in self.IGNORE_DIRS):
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    lines = sum(1 for _ in f)

                if lines > threshold:
                    rel_path = py_file.relative_to(self.repo_path)
                    issue = Issue(
                        id=self._generate_id(f"large_file:{rel_path}"),
                        title=f"Refactor {rel_path.name} ({lines} LOC)",
                        description=(
                            f"File {rel_path} has {lines} lines of code, exceeding the "
                            f"{threshold} line threshold.\n\n"
                            "Consider breaking it into smaller, focused modules."
                        ),
                        category="debt",
                        priority=3 if lines < 800 else 2,
                        file_hints=[str(rel_path)],
                        complexity="large",
                        source=f"analysis:large_file:{rel_path}",
                    )
                    issues.append(issue)
            except (OSError, UnicodeDecodeError):
                continue

        return sorted(issues, key=lambda i: -int(re.search(r"\((\d+)", i.title).group(1)))[:10]

    def _find_untested_modules(self) -> list[Issue]:
        """Find modules without corresponding test files."""
        issues = []
        test_dir = self.repo_path / "tests"

        if not test_dir.exists():
            return issues

        # Get all test file names
        test_files = set()
        for test_file in test_dir.rglob("test_*.py"):
            # Extract module name from test_foo.py -> foo
            name = test_file.stem
            if name.startswith("test_"):
                test_files.add(name[5:])

        # Check main source files
        src_dirs = [self.repo_path / "aragora", self.repo_path / "scripts"]

        for src_dir in src_dirs:
            if not src_dir.exists():
                continue

            for py_file in src_dir.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                if any(d in str(py_file) for d in self.IGNORE_DIRS):
                    continue

                module_name = py_file.stem
                if module_name not in test_files:
                    rel_path = py_file.relative_to(self.repo_path)

                    # Check file size - only report for substantial files
                    try:
                        with open(py_file, "r", encoding="utf-8") as f:
                            lines = sum(1 for _ in f)
                        if lines < 50:
                            continue
                    except (OSError, UnicodeDecodeError):
                        continue

                    issue = Issue(
                        id=self._generate_id(f"untested:{rel_path}"),
                        title=f"Add tests for {module_name}",
                        description=(
                            f"Module {rel_path} ({lines} LOC) has no corresponding test file.\n\n"
                            f"Expected test file: tests/test_{module_name}.py"
                        ),
                        category="debt",
                        priority=4,
                        file_hints=[str(rel_path)],
                        complexity="medium",
                        source=f"analysis:untested:{rel_path}",
                    )
                    issues.append(issue)

        return issues[:15]  # Limit

    def _find_deprecated_usage(self) -> list[Issue]:
        """Find deprecated patterns and APIs."""
        issues = []

        deprecated_patterns = [
            (
                r"from typing import Optional",
                "Use X | None instead of Optional[X] (Python 3.10+)",
                4,
            ),
            (r"asyncio\.get_event_loop\(\)", "Use asyncio.get_running_loop() instead", 3),
            (r"\.format\(", "Consider using f-strings instead of .format()", 5),
            (r"os\.path\.", "Consider using pathlib.Path instead of os.path", 5),
        ]

        for pattern, description, priority in deprecated_patterns:
            matches = self._run_grep(pattern)
            if len(matches) >= 3:  # Only report if multiple occurrences
                files = list(set(Path(m[0]).relative_to(self.repo_path) for m in matches))
                issue = Issue(
                    id=self._generate_id(f"deprecated:{pattern}"),
                    title=f"Modernize: {description[:50]}",
                    description=(
                        f"{description}\n\n"
                        f"Found {len(matches)} occurrences in {len(files)} files:\n"
                        + "\n".join(f"- {f}" for f in files[:5])
                        + (f"\n... and {len(files) - 5} more" if len(files) > 5 else "")
                    ),
                    category="debt",
                    priority=priority,
                    file_hints=[str(f) for f in files[:5]],
                    complexity="medium" if len(matches) <= 10 else "large",
                    source=f"grep:deprecated:{pattern}",
                )
                issues.append(issue)

        return issues


class IssueSelector:
    """Select next issue for nomic cycle."""

    def __init__(self, backlog: list[Issue], history: Optional[list[dict]] = None):
        self.backlog = backlog
        self.history = {h["id"] for h in (history or [])}

    def select_next(self) -> Optional[Issue]:
        """Select highest priority unworked issue."""
        for issue in self.backlog:
            if issue.id not in self.history:
                return issue
        return None

    def get_top_n(self, n: int = 5) -> list[Issue]:
        """Get top N unworked issues."""
        result = []
        for issue in self.backlog:
            if issue.id not in self.history:
                result.append(issue)
                if len(result) >= n:
                    break
        return result


def load_issue_history(nomic_dir: Path) -> list[dict]:
    """Load issue history from nomic directory."""
    history_file = nomic_dir / "issue_history.json"
    if history_file.exists():
        try:
            return json.loads(history_file.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_issue_attempt(nomic_dir: Path, issue: Issue, outcome: str, cycle: int) -> None:
    """Save issue attempt to history."""
    from datetime import datetime

    history = load_issue_history(nomic_dir)
    history.append(
        {
            "id": issue.id,
            "title": issue.title,
            "cycle": cycle,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat(),
        }
    )

    history_file = nomic_dir / "issue_history.json"
    history_file.write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    import sys

    # When run directly, scan and print issues
    repo_path = Path.cwd()
    if len(sys.argv) > 1:
        repo_path = Path(sys.argv[1])

    print(f"Scanning {repo_path} for issues...")
    generator = IssueGenerator(repo_path)
    issues = generator.scan_for_issues()

    print(f"\nFound {len(issues)} issues:\n")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. [{issue.category}] {issue.title}")
        print(f"   Priority: {issue.priority}, Complexity: {issue.complexity}")
        print(f"   Files: {', '.join(issue.file_hints[:3])}")
        print(f"   {issue.description[:100]}...")
        print()
