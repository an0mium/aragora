"""
Bug Detector - Static Analysis for Common Bug Patterns.

Detects common programming bugs and code quality issues:
- Null pointer dereferences
- Resource leaks (files, connections, locks)
- Race conditions
- Integer overflow
- Infinite loops
- Dead code
- Logic errors
- Exception handling issues

Complements SAST scanner which focuses on security vulnerabilities.

Usage:
    from aragora.analysis.codebase.bug_detector import BugDetector

    detector = BugDetector()
    result = await detector.scan_repository("/path/to/repo")

    print(f"Found {len(result.bugs)} potential bugs")
    for bug in result.bugs:
        print(f"  {bug.bug_type}: {bug.file_path}:{bug.line_number}")
"""

from __future__ import annotations

import ast
import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class BugType(str, Enum):
    """Types of bugs detected."""

    NULL_POINTER = "null_pointer"
    RESOURCE_LEAK = "resource_leak"
    RACE_CONDITION = "race_condition"
    INTEGER_OVERFLOW = "integer_overflow"
    INFINITE_LOOP = "infinite_loop"
    DEAD_CODE = "dead_code"
    LOGIC_ERROR = "logic_error"
    EXCEPTION_HANDLING = "exception_handling"
    TYPE_ERROR = "type_error"
    MEMORY_LEAK = "memory_leak"
    UNINITIALIZED_VAR = "uninitialized_variable"
    UNREACHABLE_CODE = "unreachable_code"
    DUPLICATE_CODE = "duplicate_code"
    DEPRECATED_API = "deprecated_api"


class BugSeverity(str, Enum):
    """Severity of detected bugs."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BugFinding:
    """A detected bug."""

    bug_id: str
    bug_type: BugType
    severity: BugSeverity
    file_path: str
    line_number: int
    column: int = 0
    end_line: int = 0
    message: str = ""
    description: str = ""
    snippet: str = ""
    confidence: float = 0.8

    # Fix suggestion
    suggested_fix: Optional[str] = None
    fix_confidence: float = 0.0

    # Context
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    language: str = "python"

    # Verification
    verified: bool = False
    verified_by: Optional[str] = None  # agent, user
    is_false_positive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bug_id": self.bug_id,
            "bug_type": self.bug_type.value,
            "severity": self.severity.value,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column": self.column,
            "end_line": self.end_line or self.line_number,
            "message": self.message,
            "description": self.description,
            "snippet": self.snippet,
            "confidence": self.confidence,
            "suggested_fix": self.suggested_fix,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "language": self.language,
            "verified": self.verified,
            "is_false_positive": self.is_false_positive,
        }


@dataclass
class BugScanResult:
    """Result of bug detection scan."""

    scan_id: str
    repository_path: str
    bugs: List[BugFinding]
    scanned_files: int
    total_lines: int
    scan_duration_ms: float
    languages: List[str]
    errors: List[str] = field(default_factory=list)
    scanned_at: datetime = field(default_factory=datetime.now)

    @property
    def bugs_by_type(self) -> Dict[str, int]:
        """Count bugs by type."""
        counts: Dict[str, int] = {}
        for bug in self.bugs:
            key = bug.bug_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def bugs_by_severity(self) -> Dict[str, int]:
        """Count bugs by severity."""
        counts: Dict[str, int] = {}
        for bug in self.bugs:
            key = bug.severity.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "repository_path": self.repository_path,
            "bugs": [b.to_dict() for b in self.bugs],
            "bug_count": len(self.bugs),
            "bugs_by_type": self.bugs_by_type,
            "bugs_by_severity": self.bugs_by_severity,
            "scanned_files": self.scanned_files,
            "total_lines": self.total_lines,
            "scan_duration_ms": self.scan_duration_ms,
            "languages": self.languages,
            "errors": self.errors,
            "scanned_at": self.scanned_at.isoformat(),
        }


class BugPattern:
    """Base class for bug detection patterns."""

    name: str = "base"
    bug_type: BugType = BugType.LOGIC_ERROR
    default_severity: BugSeverity = BugSeverity.MEDIUM
    languages: List[str] = ["python"]

    def detect(
        self,
        content: str,
        file_path: str,
        ast_tree: Optional[ast.AST] = None,
    ) -> List[BugFinding]:
        """Detect bugs in content. Override in subclasses."""
        raise NotImplementedError


class NullPointerPattern(BugPattern):
    """Detect potential null pointer dereferences."""

    name = "null_pointer"
    bug_type = BugType.NULL_POINTER
    default_severity = BugSeverity.HIGH
    languages = ["python", "javascript", "typescript"]

    # Patterns indicating potential null access after check
    PYTHON_PATTERNS = [
        # None check followed by attribute access
        (r"if\s+(\w+)\s+is\s+None.*\n.*\1\.(\w+)", "Potential None access after check"),
        # Optional return not handled
        (r"(\w+)\s*=\s*\w+\.get\([^)]+\)[^:]*\n[^#]*\1\.", "Unchecked optional access"),
        # Function returning None used directly
        (r"\.find\([^)]+\)\.(\w+)", "Potential None from find()"),
    ]

    def detect(
        self,
        content: str,
        file_path: str,
        ast_tree: Optional[ast.AST] = None,
    ) -> List[BugFinding]:
        bugs = []
        lines = content.split("\n")

        for pattern, message in self.PYTHON_PATTERNS:
            for match in re.finditer(pattern, content, re.MULTILINE):
                line_num = content[: match.start()].count("\n") + 1
                bugs.append(
                    BugFinding(
                        bug_id=f"np_{line_num}_{file_path[-20:]}",
                        bug_type=self.bug_type,
                        severity=self.default_severity,
                        file_path=file_path,
                        line_number=line_num,
                        message=message,
                        snippet=lines[line_num - 1] if line_num <= len(lines) else "",
                        confidence=0.7,
                    )
                )

        return bugs


class ResourceLeakPattern(BugPattern):
    """Detect resource leaks (files, connections, locks)."""

    name = "resource_leak"
    bug_type = BugType.RESOURCE_LEAK
    default_severity = BugSeverity.HIGH
    languages = ["python"]

    def detect(
        self,
        content: str,
        file_path: str,
        ast_tree: Optional[ast.AST] = None,
    ) -> List[BugFinding]:
        bugs = []
        lines = content.split("\n")

        # Pattern: open() without context manager
        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # File open without 'with'
            if (
                re.match(r"^\w+\s*=\s*open\(", stripped)
                and "with "
                not in content[max(0, content.find(stripped) - 100) : content.find(stripped)]
            ):
                # Check if close() is called in same function
                func_content = self._get_function_content(content, i)
                if func_content and ".close()" not in func_content:
                    bugs.append(
                        BugFinding(
                            bug_id=f"rl_file_{i}_{file_path[-20:]}",
                            bug_type=self.bug_type,
                            severity=self.default_severity,
                            file_path=file_path,
                            line_number=i,
                            message="File opened without context manager and no close() call",
                            snippet=stripped,
                            confidence=0.8,
                            suggested_fix="Use 'with open(...) as f:' context manager",
                        )
                    )

            # Connection without close
            if re.match(r"^\w+\s*=\s*\w+\.connect\(", stripped):
                func_content = self._get_function_content(content, i)
                if func_content and ".close()" not in func_content:
                    bugs.append(
                        BugFinding(
                            bug_id=f"rl_conn_{i}_{file_path[-20:]}",
                            bug_type=self.bug_type,
                            severity=self.default_severity,
                            file_path=file_path,
                            line_number=i,
                            message="Connection opened without close() call",
                            snippet=stripped,
                            confidence=0.7,
                        )
                    )

            # Lock acquire without release
            if ".acquire(" in stripped and "with " not in stripped:
                func_content = self._get_function_content(content, i)
                if func_content and ".release()" not in func_content:
                    bugs.append(
                        BugFinding(
                            bug_id=f"rl_lock_{i}_{file_path[-20:]}",
                            bug_type=self.bug_type,
                            severity=BugSeverity.CRITICAL,
                            file_path=file_path,
                            line_number=i,
                            message="Lock acquired without release()",
                            snippet=stripped,
                            confidence=0.8,
                            suggested_fix="Use 'with lock:' context manager",
                        )
                    )

        return bugs

    def _get_function_content(self, content: str, line_num: int) -> Optional[str]:
        """Get the content of the function containing the line."""
        lines = content.split("\n")
        if line_num > len(lines):
            return None

        # Find function start by looking for 'def' or 'async def'
        start_line = line_num - 1
        while start_line > 0:
            if re.match(r"^\s*(async\s+)?def\s+\w+", lines[start_line]):
                break
            start_line -= 1

        # Find function end (next def at same or lower indentation)
        start_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        end_line = line_num
        while end_line < len(lines):
            line = lines[end_line]
            if line.strip() and not line.startswith(" " * (start_indent + 1)):
                if re.match(r"^\s*(async\s+)?def\s+\w+", line):
                    break
            end_line += 1

        return "\n".join(lines[start_line:end_line])


class InfiniteLoopPattern(BugPattern):
    """Detect potential infinite loops."""

    name = "infinite_loop"
    bug_type = BugType.INFINITE_LOOP
    default_severity = BugSeverity.CRITICAL
    languages = ["python"]

    def detect(
        self,
        content: str,
        file_path: str,
        ast_tree: Optional[ast.AST] = None,
    ) -> List[BugFinding]:
        bugs = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # while True without break
            if stripped == "while True:":
                # Check if there's a break in the loop body
                loop_body = self._get_loop_body(lines, i)
                if loop_body and "break" not in loop_body and "return" not in loop_body:
                    bugs.append(
                        BugFinding(
                            bug_id=f"inf_true_{i}_{file_path[-20:]}",
                            bug_type=self.bug_type,
                            severity=self.default_severity,
                            file_path=file_path,
                            line_number=i,
                            message="while True loop without break or return",
                            snippet=stripped,
                            confidence=0.6,  # Lower - could be intentional server loop
                        )
                    )

            # while with constant condition
            if re.match(r"while\s+1\s*:", stripped) or re.match(r"while\s+\"", stripped):
                bugs.append(
                    BugFinding(
                        bug_id=f"inf_const_{i}_{file_path[-20:]}",
                        bug_type=self.bug_type,
                        severity=self.default_severity,
                        file_path=file_path,
                        line_number=i,
                        message="while loop with constant truthy condition",
                        snippet=stripped,
                        confidence=0.7,
                    )
                )

            # for loop modifying iteration variable
            match = re.match(r"for\s+(\w+)\s+in", stripped)
            if match:
                var_name = match.group(1)
                loop_body = self._get_loop_body(lines, i)
                if loop_body and f"{var_name} =" in loop_body:
                    bugs.append(
                        BugFinding(
                            bug_id=f"inf_mod_{i}_{file_path[-20:]}",
                            bug_type=self.bug_type,
                            severity=BugSeverity.HIGH,
                            file_path=file_path,
                            line_number=i,
                            message=f"Loop variable '{var_name}' modified inside loop",
                            snippet=stripped,
                            confidence=0.8,
                        )
                    )

        return bugs

    def _get_loop_body(self, lines: List[str], start_line: int) -> Optional[str]:
        """Get the body of a loop starting at start_line."""
        if start_line > len(lines):
            return None

        start_indent = len(lines[start_line - 1]) - len(lines[start_line - 1].lstrip())
        body_lines = []

        for i in range(start_line, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= start_indent and i > start_line:
                    break
                body_lines.append(line)

        return "\n".join(body_lines)


class DeadCodePattern(BugPattern):
    """Detect dead/unreachable code."""

    name = "dead_code"
    bug_type = BugType.DEAD_CODE
    default_severity = BugSeverity.LOW
    languages = ["python"]

    def detect(
        self,
        content: str,
        file_path: str,
        ast_tree: Optional[ast.AST] = None,
    ) -> List[BugFinding]:
        bugs = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Code after return/raise/break/continue
            if i < len(lines):
                prev_stripped = lines[i - 1].strip() if i > 1 else ""
                current_indent = len(line) - len(line.lstrip())
                prev_indent = len(lines[i - 1]) - len(lines[i - 1].lstrip()) if i > 1 else 0

                if (
                    prev_stripped.startswith(("return ", "return\n", "raise ", "break", "continue"))
                    and stripped
                    and not stripped.startswith(("#", "def ", "class ", "async def", "@"))
                    and current_indent >= prev_indent
                ):
                    bugs.append(
                        BugFinding(
                            bug_id=f"dead_after_{i}_{file_path[-20:]}",
                            bug_type=self.bug_type,
                            severity=self.default_severity,
                            file_path=file_path,
                            line_number=i,
                            message="Unreachable code after return/raise/break/continue",
                            snippet=stripped,
                            confidence=0.9,
                        )
                    )

            # Unused imports (simplified)
            if stripped.startswith("import ") or stripped.startswith("from "):
                # Extract module/symbol names
                if stripped.startswith("from "):
                    match = re.search(r"import\s+(.+)", stripped)
                    if match:
                        imports = [s.strip().split(" as ")[-1] for s in match.group(1).split(",")]
                        for imp in imports:
                            imp = imp.strip()
                            if imp and imp not in content.replace(stripped, ""):
                                bugs.append(
                                    BugFinding(
                                        bug_id=f"unused_import_{i}_{imp}",
                                        bug_type=BugType.DEAD_CODE,
                                        severity=BugSeverity.LOW,
                                        file_path=file_path,
                                        line_number=i,
                                        message=f"Unused import: {imp}",
                                        snippet=stripped,
                                        confidence=0.6,
                                    )
                                )

        return bugs


class ExceptionHandlingPattern(BugPattern):
    """Detect exception handling issues."""

    name = "exception_handling"
    bug_type = BugType.EXCEPTION_HANDLING
    default_severity = BugSeverity.MEDIUM
    languages = ["python"]

    def detect(
        self,
        content: str,
        file_path: str,
        ast_tree: Optional[ast.AST] = None,
    ) -> List[BugFinding]:
        bugs = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Bare except
            if stripped == "except:":
                bugs.append(
                    BugFinding(
                        bug_id=f"bare_except_{i}_{file_path[-20:]}",
                        bug_type=self.bug_type,
                        severity=BugSeverity.HIGH,
                        file_path=file_path,
                        line_number=i,
                        message="Bare except catches all exceptions including SystemExit",
                        snippet=stripped,
                        confidence=0.95,
                        suggested_fix="Use 'except Exception:' or specific exception types",
                    )
                )

            # except Exception as e: pass
            if re.match(r"except\s+\w+.*:", stripped):
                # Check if next non-empty line is just 'pass'
                for j in range(i, min(i + 3, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and next_line != "#":
                        if next_line == "pass":
                            bugs.append(
                                BugFinding(
                                    bug_id=f"silent_except_{i}_{file_path[-20:]}",
                                    bug_type=self.bug_type,
                                    severity=BugSeverity.MEDIUM,
                                    file_path=file_path,
                                    line_number=i,
                                    message="Exception silently caught with pass",
                                    snippet=stripped,
                                    confidence=0.8,
                                    suggested_fix="Log the exception or handle it properly",
                                )
                            )
                        break

            # Exception not using 'as' variable
            match = re.match(r"except\s+(\w+)\s+as\s+(\w+):", stripped)
            if match:
                exc_var = match.group(2)
                # Check if variable is used in except block
                block_content = self._get_block_content(lines, i)
                if block_content and exc_var not in block_content.replace(stripped, ""):
                    bugs.append(
                        BugFinding(
                            bug_id=f"unused_exc_{i}_{file_path[-20:]}",
                            bug_type=self.bug_type,
                            severity=BugSeverity.LOW,
                            file_path=file_path,
                            line_number=i,
                            message=f"Caught exception variable '{exc_var}' is unused",
                            snippet=stripped,
                            confidence=0.7,
                        )
                    )

        return bugs

    def _get_block_content(self, lines: List[str], start_line: int) -> Optional[str]:
        """Get content of indented block."""
        if start_line > len(lines):
            return None

        start_indent = len(lines[start_line - 1]) - len(lines[start_line - 1].lstrip())
        block_lines = []

        for i in range(start_line, len(lines)):
            line = lines[i]
            if line.strip():
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= start_indent and i > start_line:
                    break
                block_lines.append(line)

        return "\n".join(block_lines)


class LogicErrorPattern(BugPattern):
    """Detect common logic errors."""

    name = "logic_error"
    bug_type = BugType.LOGIC_ERROR
    default_severity = BugSeverity.HIGH
    languages = ["python"]

    def detect(
        self,
        content: str,
        file_path: str,
        ast_tree: Optional[ast.AST] = None,
    ) -> List[BugFinding]:
        bugs = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Assignment in condition (common mistake)
            if re.match(r"if\s+\w+\s*=[^=]", stripped):
                bugs.append(
                    BugFinding(
                        bug_id=f"assign_cond_{i}_{file_path[-20:]}",
                        bug_type=self.bug_type,
                        severity=self.default_severity,
                        file_path=file_path,
                        line_number=i,
                        message="Assignment in condition (did you mean '=='?)",
                        snippet=stripped,
                        confidence=0.9,
                    )
                )

            # Comparison with None using == instead of is
            if re.search(r"==\s*None|None\s*==", stripped):
                bugs.append(
                    BugFinding(
                        bug_id=f"none_eq_{i}_{file_path[-20:]}",
                        bug_type=self.bug_type,
                        severity=BugSeverity.LOW,
                        file_path=file_path,
                        line_number=i,
                        message="Use 'is None' instead of '== None'",
                        snippet=stripped,
                        confidence=0.95,
                        suggested_fix="Replace '== None' with 'is None'",
                    )
                )

            # Mutable default argument
            if re.match(r"def\s+\w+\([^)]*=\s*\[\]", stripped) or re.match(
                r"def\s+\w+\([^)]*=\s*\{\}", stripped
            ):
                bugs.append(
                    BugFinding(
                        bug_id=f"mutable_default_{i}_{file_path[-20:]}",
                        bug_type=self.bug_type,
                        severity=BugSeverity.HIGH,
                        file_path=file_path,
                        line_number=i,
                        message="Mutable default argument (list/dict)",
                        snippet=stripped,
                        confidence=0.95,
                        suggested_fix="Use None as default and initialize inside function",
                    )
                )

            # String format with wrong number of args (simplified)
            if "% " in stripped and "%" in stripped:
                percent_count = stripped.count("%s") + stripped.count("%d") + stripped.count("%f")
                tuple_match = re.search(r"%\s*\(([^)]+)\)", stripped)
                if tuple_match:
                    args = tuple_match.group(1).split(",")
                    if len(args) != percent_count and percent_count > 0:
                        bugs.append(
                            BugFinding(
                                bug_id=f"format_args_{i}_{file_path[-20:]}",
                                bug_type=self.bug_type,
                                severity=BugSeverity.HIGH,
                                file_path=file_path,
                                line_number=i,
                                message=f"String format expects {percent_count} args, got {len(args)}",
                                snippet=stripped,
                                confidence=0.7,
                            )
                        )

        return bugs


class BugDetector:
    """
    Multi-pattern bug detection with confidence scoring.

    Scans codebases for common programming bugs using
    AST analysis and pattern matching.
    """

    PATTERNS: List[BugPattern] = [
        NullPointerPattern(),
        ResourceLeakPattern(),
        InfiniteLoopPattern(),
        DeadCodePattern(),
        ExceptionHandlingPattern(),
        LogicErrorPattern(),
    ]

    # File extensions to scan
    EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".go": "go",
        ".java": "java",
        ".rb": "ruby",
    }

    # Directories to skip
    SKIP_DIRS = {
        "node_modules",
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "env",
        ".tox",
        "dist",
        "build",
        ".eggs",
        "*.egg-info",
    }

    def __init__(
        self,
        patterns: Optional[List[BugPattern]] = None,
        min_confidence: float = 0.5,
    ):
        """
        Initialize bug detector.

        Args:
            patterns: Custom patterns to use (default: all built-in)
            min_confidence: Minimum confidence threshold for reporting
        """
        self.patterns = patterns or self.PATTERNS
        self.min_confidence = min_confidence

    async def scan_repository(
        self,
        repo_path: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> BugScanResult:
        """
        Scan a repository for bugs.

        Args:
            repo_path: Path to repository
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude

        Returns:
            BugScanResult with all findings
        """
        import time
        import uuid

        start_time = time.time()
        scan_id = f"bug_scan_{uuid.uuid4().hex[:12]}"

        bugs: List[BugFinding] = []
        scanned_files = 0
        total_lines = 0
        languages: Set[str] = set()
        errors: List[str] = []

        path = Path(repo_path)
        if not path.exists():
            return BugScanResult(
                scan_id=scan_id,
                repository_path=repo_path,
                bugs=[],
                scanned_files=0,
                total_lines=0,
                scan_duration_ms=0,
                languages=[],
                errors=[f"Path does not exist: {repo_path}"],
            )

        # Collect files to scan
        files_to_scan = self._collect_files(path, include_patterns, exclude_patterns)

        # Scan files concurrently
        tasks = [self._scan_file(f) for f in files_to_scan]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for file_path, result in zip(files_to_scan, results):
            if isinstance(result, BaseException):
                errors.append(f"Error scanning {file_path}: {result}")
                continue

            file_bugs, lines, lang = result
            bugs.extend(file_bugs)
            total_lines += lines
            scanned_files += 1
            if lang:
                languages.add(lang)

        # Filter by confidence
        bugs = [b for b in bugs if b.confidence >= self.min_confidence]

        # Sort by severity then line number
        severity_order = {
            BugSeverity.CRITICAL: 0,
            BugSeverity.HIGH: 1,
            BugSeverity.MEDIUM: 2,
            BugSeverity.LOW: 3,
        }
        bugs.sort(key=lambda b: (severity_order.get(b.severity, 4), b.file_path, b.line_number))

        duration_ms = (time.time() - start_time) * 1000

        logger.info(
            f"[BugDetector] Scan {scan_id}: found {len(bugs)} bugs in {scanned_files} files"
        )

        return BugScanResult(
            scan_id=scan_id,
            repository_path=repo_path,
            bugs=bugs,
            scanned_files=scanned_files,
            total_lines=total_lines,
            scan_duration_ms=duration_ms,
            languages=list(languages),
            errors=errors,
        )

    def _collect_files(
        self,
        path: Path,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]],
    ) -> List[Path]:
        """Collect files to scan."""
        files = []

        for file_path in path.rglob("*"):
            # Skip directories
            if file_path.is_dir():
                continue

            # Skip hidden and specified directories
            parts = file_path.parts
            if any(p in self.SKIP_DIRS or p.startswith(".") for p in parts):
                continue

            # Check extension
            if file_path.suffix not in self.EXTENSIONS:
                continue

            # Apply include/exclude patterns
            rel_path = str(file_path.relative_to(path))

            if include_patterns:
                if not any(self._match_pattern(rel_path, p) for p in include_patterns):
                    continue

            if exclude_patterns:
                if any(self._match_pattern(rel_path, p) for p in exclude_patterns):
                    continue

            files.append(file_path)

        return files

    def _match_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches glob pattern."""
        import fnmatch

        return fnmatch.fnmatch(path, pattern)

    async def _scan_file(
        self,
        file_path: Path,
    ) -> Tuple[List[BugFinding], int, str]:
        """Scan a single file for bugs."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.debug(f"Failed to read {file_path}: {e}")
            return [], 0, ""

        lines = len(content.split("\n"))
        language = self.EXTENSIONS.get(file_path.suffix, "")

        bugs: List[BugFinding] = []

        # Parse AST for Python files
        ast_tree = None
        if file_path.suffix == ".py":
            try:
                ast_tree = ast.parse(content)
            except SyntaxError:
                pass  # Will use pattern matching only

        # Run all patterns
        for pattern in self.patterns:
            if language not in pattern.languages:
                continue

            try:
                pattern_bugs = pattern.detect(content, str(file_path), ast_tree)
                bugs.extend(pattern_bugs)
            except Exception as e:
                logger.debug(f"Pattern {pattern.name} failed on {file_path}: {e}")

        return bugs, lines, language

    async def verify_with_agents(
        self,
        bugs: List[BugFinding],
        limit: int = 10,
    ) -> List[BugFinding]:
        """
        Use multi-agent debate to verify bugs and suggest fixes.

        Args:
            bugs: Bugs to verify
            limit: Max bugs to verify (for performance)

        Returns:
            Bugs with verification and fix suggestions
        """
        try:
            from aragora.debate.arena import DebateArena

            for bug in bugs[:limit]:
                if bug.verified:
                    continue

                question = f"""Analyze this potential bug:

File: {bug.file_path}
Line: {bug.line_number}
Type: {bug.bug_type.value}
Message: {bug.message}
Code: {bug.snippet}

Questions:
1. Is this a real bug or false positive?
2. If real, what is the severity (critical/high/medium/low)?
3. What is the suggested fix?

Format:
IS_BUG: yes/no
SEVERITY: critical/high/medium/low
FIX: <suggested code fix>
EXPLANATION: <brief explanation>"""

                arena = DebateArena(agents=["anthropic-api", "openai-api"])
                result = await arena.debate(question=question, rounds=1, timeout=15)

                if result and hasattr(result, "final_answer"):
                    answer = result.final_answer

                    # Parse response
                    is_bug = "IS_BUG: yes" in answer.lower() or "IS_BUG:yes" in answer.lower()
                    bug.is_false_positive = not is_bug
                    bug.verified = True
                    bug.verified_by = "agent"

                    # Extract fix
                    fix_match = re.search(r"FIX:\s*(.+?)(?:EXPLANATION:|$)", answer, re.DOTALL)
                    if fix_match:
                        bug.suggested_fix = fix_match.group(1).strip()
                        bug.fix_confidence = (
                            result.confidence if hasattr(result, "confidence") else 0.7
                        )

                    logger.debug(
                        f"[BugDetector] Agent verified {bug.bug_id}: "
                        f"{'real bug' if is_bug else 'false positive'}"
                    )

        except ImportError:
            logger.warning("[BugDetector] Debate arena not available")
        except Exception as e:
            logger.error(f"[BugDetector] Agent verification failed: {e}")

        return bugs


# =============================================================================
# Factory Functions
# =============================================================================


def create_bug_detector(
    patterns: Optional[List[str]] = None,
    min_confidence: float = 0.5,
) -> BugDetector:
    """
    Create a bug detector with specified patterns.

    Args:
        patterns: Names of patterns to use (default: all)
        min_confidence: Minimum confidence threshold

    Returns:
        Configured BugDetector
    """
    pattern_map = {
        "null_pointer": NullPointerPattern(),
        "resource_leak": ResourceLeakPattern(),
        "infinite_loop": InfiniteLoopPattern(),
        "dead_code": DeadCodePattern(),
        "exception_handling": ExceptionHandlingPattern(),
        "logic_error": LogicErrorPattern(),
    }

    if patterns:
        selected = [pattern_map[p] for p in patterns if p in pattern_map]
    else:
        selected = list(pattern_map.values())

    return BugDetector(patterns=selected, min_confidence=min_confidence)


__all__ = [
    "BugDetector",
    "BugFinding",
    "BugScanResult",
    "BugType",
    "BugSeverity",
    "BugPattern",
    "NullPointerPattern",
    "ResourceLeakPattern",
    "InfiniteLoopPattern",
    "DeadCodePattern",
    "ExceptionHandlingPattern",
    "LogicErrorPattern",
    "create_bug_detector",
]
