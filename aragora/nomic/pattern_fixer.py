"""Batch pattern fixer for finding and fixing all instances of bug classes.

Instead of fixing bugs one-at-a-time as agents discover them, this module
finds ALL instances of a pattern and fixes them atomically.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

ANTIPATTERNS: dict[str, dict[str, str]] = {
    "bare_except": {
        "pattern": r"except\s*(?:Exception)?\s*:\s*(?:pass|\.\.\.)",
        "description": "Bare exception swallowing (except: pass or except Exception: pass)",
    },
    "str_e_leak": {
        "pattern": r"(?:message|error|detail|response).*=.*str\(e\)",
        "description": "str(e) leaked into error responses",
    },
    "missing_logger": {
        "pattern": r"logging\.(debug|info|warning|error|critical)\(",
        "description": "Direct logging.X() call without module-level logger",
    },
    "eval_usage": {
        "pattern": r"\beval\s*\(",
        "description": "eval() call (potential code injection)",
    },
    "shell_true": {
        "pattern": r"subprocess\.\w+\(.*shell\s*=\s*True",
        "description": "shell=True in subprocess (command injection risk)",
    },
    "hardcoded_secret": {
        "pattern": r"(?:api_key|password|secret|token)\s*=\s*[\"'][^\"']{8,}",
        "description": "Potential hardcoded secret",
    },
    "while_true": {
        "pattern": r"while\s+True\s*:",
        "description": "Unbounded while True loop (check for break/return)",
    },
}


@dataclass
class PatternMatch:
    """A single occurrence of a pattern in the codebase."""

    file: str
    line: int
    content: str
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)
    pattern_name: str = ""


@dataclass
class FixResult:
    """Result of applying fixes to pattern matches."""

    matches_found: int
    matches_fixed: int
    files_changed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class PatternFixer:
    """Find and batch-fix antipatterns across the codebase."""

    def __init__(self, codebase_root: str = ".") -> None:
        self.root = Path(codebase_root)

    def find_pattern(
        self,
        pattern: str,
        file_glob: str = "**/*.py",
    ) -> list[PatternMatch]:
        """Search for a regex pattern across files matching the glob."""
        compiled = re.compile(pattern)
        matches: list[PatternMatch] = []

        for path in sorted(self.root.glob(file_glob)):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                continue

            lines = text.splitlines()
            for i, line in enumerate(lines):
                if compiled.search(line):
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    matches.append(
                        PatternMatch(
                            file=str(path),
                            line=i + 1,
                            content=line,
                            context_before=lines[start:i],
                            context_after=lines[i + 1 : end],
                        )
                    )
        return matches

    def find_antipattern(self, name: str) -> list[PatternMatch]:
        """Search for a pre-defined antipattern by name."""
        if name not in ANTIPATTERNS:
            raise ValueError(f"Unknown antipattern: {name!r}. Available: {list(ANTIPATTERNS)}")
        info = ANTIPATTERNS[name]
        results = self.find_pattern(info["pattern"])
        for m in results:
            m.pattern_name = name
        return results

    def list_antipatterns(self) -> dict[str, str]:
        """Return available antipatterns with descriptions."""
        return {name: info["description"] for name, info in ANTIPATTERNS.items()}

    def count_antipatterns(self) -> dict[str, int]:
        """Count occurrences of all pre-defined antipatterns."""
        counts: dict[str, int] = {}
        for name in ANTIPATTERNS:
            counts[name] = len(self.find_antipattern(name))
        return counts

    def fix_pattern(
        self,
        matches: list[PatternMatch],
        replacement: str | Callable[[PatternMatch], str],
    ) -> FixResult:
        """Apply a replacement to all matches.

        If *replacement* is a string it replaces the matched line content.
        If it is a callable, it receives each ``PatternMatch`` and should
        return the replacement line.
        """
        result = FixResult(matches_found=len(matches), matches_fixed=0)

        # Group matches by file, sorted by descending line number so
        # replacements don't shift earlier line numbers.
        by_file: dict[str, list[PatternMatch]] = {}
        for m in matches:
            by_file.setdefault(m.file, []).append(m)

        for filepath, file_matches in by_file.items():
            try:
                path = Path(filepath)
                lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            except (OSError, UnicodeDecodeError) as exc:
                result.errors.append(f"{filepath}: {exc}")
                continue

            file_matches.sort(key=lambda m: m.line, reverse=True)
            changed = False
            for m in file_matches:
                idx = m.line - 1
                if idx < 0 or idx >= len(lines):
                    result.errors.append(f"{filepath}:{m.line}: line out of range")
                    continue
                new_line = replacement(m) if callable(replacement) else replacement
                # Preserve trailing newline
                if lines[idx].endswith("\n") and not new_line.endswith("\n"):
                    new_line += "\n"
                lines[idx] = new_line
                result.matches_fixed += 1
                changed = True

            if changed:
                try:
                    path.write_text("".join(lines), encoding="utf-8")
                    result.files_changed.append(filepath)
                except OSError as exc:
                    result.errors.append(f"{filepath}: write failed: {exc}")

        return result

    def generate_report(self) -> str:
        """Formatted report of all antipattern occurrences."""
        lines: list[str] = ["# Antipattern Report", ""]
        total = 0
        for name, desc in self.list_antipatterns().items():
            matches = self.find_antipattern(name)
            count = len(matches)
            total += count
            lines.append(f"## {name} ({count} occurrences)")
            lines.append(f"  {desc}")
            if matches:
                for m in matches[:5]:
                    lines.append(f"  - {m.file}:{m.line}: {m.content.strip()}")
                if count > 5:
                    lines.append(f"  ... and {count - 5} more")
            lines.append("")

        lines.insert(1, f"Total antipatterns found: {total}")
        lines.insert(2, "")
        return "\n".join(lines)
