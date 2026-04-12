"""Codebase scanners that produce boss-ready issue candidates.

Each scanner function examines the repo for a specific class of improvement
and returns a list of ``BossIssueCandidate`` objects.  The CLI script
``scripts/generate_boss_issues.py`` consumes these candidates, validates
them through the pre-dispatch gate, deduplicates against open issues,
and creates GitHub issues with the ``boss-ready`` label.
"""

from __future__ import annotations

import json
import hashlib
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aragora.swarm.outcome_learner import load_category_success_rates
from aragora.swarm.terminal_truth import classify_from_metrics


DEFAULT_BOSS_METRICS_PATH = Path(".aragora/overnight/boss_metrics.jsonl")
_CATEGORY_TITLE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^narrow broad except", re.IGNORECASE), "broad_exception"),
    (re.compile(r"^replace silent", re.IGNORECASE), "silent_exception"),
    (re.compile(r"^add unit tests", re.IGNORECASE), "test_coverage"),
    (re.compile(r"^add request body", re.IGNORECASE), "handler_validation"),
    (re.compile(r"^add return type", re.IGNORECASE), "type_annotation"),
    (re.compile(r"^address todo", re.IGNORECASE), "actionable_todo"),
)


@dataclass
class BossIssueCandidate:
    """A candidate issue ready for boss-loop formatting and dispatch."""

    category: str
    title: str
    description: str
    file_scope: list[str]
    new_files: list[str] = field(default_factory=list)
    validation_command: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    estimated_complexity: str = "small"
    expected_success_rate: float = 0.5
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.fingerprint:
            raw = f"{self.category}:{':'.join(sorted(self.file_scope + self.new_files))}"
            self.fingerprint = hashlib.sha256(raw.encode()).hexdigest()[:16]


def infer_issue_category_from_title(title: str | None) -> str | None:
    """Infer scanner category from a boss-issue title."""
    if not title:
        return None

    normalized = title.strip()
    for pattern, category in _CATEGORY_TITLE_PATTERNS:
        if pattern.match(normalized):
            return category
    return None


def _load_prompt_metric_rows(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    try:
        with metrics_path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if float(row.get("prompt_chars", 0) or 0) <= 0:
                    continue
                rows.append(row)
    except OSError:
        return []
    return rows


def _fetch_issue_titles(
    issue_numbers: set[int], *, repo: str = "synaptent/aragora"
) -> dict[int, str]:
    """Fetch GitHub issue titles in batches via gh GraphQL.

    Returns an empty mapping if gh is unavailable or the call fails.
    """
    if not issue_numbers:
        return {}

    owner, sep, name = repo.partition("/")
    if not sep or not owner or not name:
        return {}

    titles: dict[int, str] = {}
    batch_size = 25
    sorted_numbers = sorted(issue_numbers)
    for idx in range(0, len(sorted_numbers), batch_size):
        batch = sorted_numbers[idx : idx + batch_size]
        aliases = "\n".join(
            f"issue_{number}: issue(number: {number}) {{ number title }}" for number in batch
        )
        query = f'query {{ repository(owner: "{owner}", name: "{name}") {{\n{aliases}\n}} }}'
        try:
            result = subprocess.run(
                ["gh", "api", "graphql", "-f", f"query={query}"],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}

        if result.returncode != 0:
            return {}

        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return {}

        repository = payload.get("data", {}).get("repository", {})
        if not isinstance(repository, dict):
            return {}

        for issue in repository.values():
            if not isinstance(issue, dict):
                continue
            number = issue.get("number")
            title = issue.get("title")
            if isinstance(number, int) and isinstance(title, str) and title.strip():
                titles[number] = title.strip()

    return titles


def historical_category_stats(metrics_path: Path) -> dict[str, dict[str, float]]:
    """Compute per-category success and crash rates from prompt-era boss metrics."""
    rows = _load_prompt_metric_rows(metrics_path)
    if not rows:
        return {}

    title_by_issue: dict[int, str] = {}
    missing_titles: set[int] = set()
    for row in rows:
        issue_number = row.get("issue_number")
        if not isinstance(issue_number, int):
            continue
        issue_title = row.get("issue_title")
        if isinstance(issue_title, str) and issue_title.strip():
            title_by_issue[issue_number] = issue_title.strip()
        else:
            missing_titles.add(issue_number)

    title_by_issue.update(
        {
            number: title
            for number, title in _fetch_issue_titles(missing_titles).items()
            if title.strip()
        }
    )

    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "total": 0.0,
            "successes": 0.0,
            "crashes": 0.0,
            "elapsed_seconds": 0.0,
            "success_rate": 0.0,
            "crash_rate": 0.0,
            "avg_elapsed_seconds": 0.0,
        }
    )
    for row in rows:
        issue_number = row.get("issue_number")
        if not isinstance(issue_number, int):
            continue
        category = infer_issue_category_from_title(title_by_issue.get(issue_number))
        if not category:
            continue

        terminal_class = classify_from_metrics(row)
        entry = totals[category]
        entry["total"] += 1.0
        entry["elapsed_seconds"] += float(row.get("elapsed_seconds", 0.0) or 0.0)
        if terminal_class.family == "success":
            entry["successes"] += 1.0
        if terminal_class.value == "rescue_worker_crash":
            entry["crashes"] += 1.0

    finalized: dict[str, dict[str, float]] = {}
    for category, entry in totals.items():
        total = entry["total"]
        if total <= 0:
            continue
        finalized[category] = {
            **entry,
            "success_rate": entry["successes"] / total,
            "crash_rate": entry["crashes"] / total,
            "avg_elapsed_seconds": entry["elapsed_seconds"] / total,
        }
    return finalized


def historical_success_rates(metrics_path: Path) -> dict[str, float]:
    """Return real success rates per scanner category from boss metrics."""
    return {
        category: stats["success_rate"]
        for category, stats in historical_category_stats(metrics_path).items()
    }


def expected_success_rates(
    metrics_path: Path,
    *,
    signal_log_path: Path | None = None,
) -> dict[str, float]:
    """Return category success rates, preferring learner calibration when available."""
    rates = historical_success_rates(metrics_path)
    calibrated_rates = load_category_success_rates(log_path=signal_log_path)
    for category, success_rate in calibrated_rates.items():
        rates[category] = success_rate
    return rates


# ---------------------------------------------------------------------------
# Scanning helpers
# ---------------------------------------------------------------------------

_COMMENT_OR_DOCSTRING_RE = re.compile(r'^\s*#|^\s*"""|\s*"""')


def _is_code_line(line: str) -> bool:
    """Return True if the line is likely executable code (not comment/docstring)."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return False
    return True


def _count_lines(path: Path) -> int:
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _test_mirror_path(module_path: str) -> str:
    """Convert aragora/foo/bar.py -> tests/foo/test_bar.py."""
    parts = Path(module_path).parts
    if parts and parts[0] == "aragora" and len(parts) >= 2:
        relative = Path(*parts[1:])
        return str(relative.parent / f"test_{relative.name}")
    return ""


# ---------------------------------------------------------------------------
# Scanner 1: Untested modules (highest yield)
# ---------------------------------------------------------------------------


def scan_untested_modules(
    repo_root: Path,
    *,
    min_loc: int = 50,
    max_loc: int = 300,
    limit: int = 30,
) -> list[BossIssueCandidate]:
    """Find Python modules without corresponding test files."""
    aragora_dir = repo_root / "aragora"
    tests_dir = repo_root / "tests"
    candidates: list[BossIssueCandidate] = []

    skip_names = {"__init__.py", "__main__.py", "conftest.py"}

    for py_file in sorted(aragora_dir.rglob("*.py")):
        if py_file.name in skip_names:
            continue
        if "migrations" in py_file.parts or "archive" in py_file.parts:
            continue

        rel = str(py_file.relative_to(repo_root))
        test_rel = _test_mirror_path(rel)
        if not test_rel:
            continue

        test_path = (
            repo_root / "tests" / test_rel.split("/", 1)[-1]
            if "/" in test_rel
            else repo_root / "tests" / test_rel
        )
        # Reconstruct properly
        parts = py_file.relative_to(aragora_dir).parts
        test_path = tests_dir / Path(*parts[:-1]) / f"test_{parts[-1]}"

        if test_path.exists():
            continue

        loc = _count_lines(py_file)
        if loc < min_loc or loc > max_loc:
            continue

        test_rel_str = str(test_path.relative_to(repo_root))
        module_rel = str(py_file.relative_to(repo_root))

        candidates.append(
            BossIssueCandidate(
                category="test_coverage",
                title=f"Add unit tests for {'/'.join(parts)}",
                description=(
                    f"Add comprehensive unit tests for `{module_rel}`.\n\n"
                    f"### Requirements\n"
                    f"1. Read `{module_rel}` and identify all public functions and classes\n"
                    f"2. Create `{test_rel_str}` with tests covering:\n"
                    f"   - All public API surface\n"
                    f"   - Edge cases (empty inputs, invalid values)\n"
                    f"   - Error handling paths\n"
                    f"3. Use pytest fixtures for repeated setup\n"
                    f"4. Mock any external services or network calls"
                ),
                file_scope=[module_rel],
                new_files=[test_rel_str],
                validation_command=f"pytest {test_rel_str} -v",
                acceptance_criteria=[
                    "All tests pass",
                    "At least 8 test functions",
                    "No external service calls (mock all dependencies)",
                    "Tests complete in under 10 seconds",
                ],
                estimated_complexity="small" if loc < 150 else "medium",
                expected_success_rate=0.7,
            )
        )

    # Sort by LOC (smaller = easier = higher success)
    candidates.sort(key=lambda c: _count_lines(repo_root / c.file_scope[0]))
    return candidates[:limit]


# ---------------------------------------------------------------------------
# Scanner 2: Silent exception swallowing
# ---------------------------------------------------------------------------

_SILENT_CATCH_RE = re.compile(
    r"except\s+\w[\w.,\s]*:\s*\n\s+pass\s*$",
    re.MULTILINE,
)


def scan_silent_exception_swallowing(
    repo_root: Path,
    *,
    limit: int = 20,
) -> list[BossIssueCandidate]:
    """Find except-pass patterns that silently swallow errors."""
    aragora_dir = repo_root / "aragora"
    candidates: list[BossIssueCandidate] = []
    skip_dirs = {"__pycache__", "archive", "migrations"}

    for py_file in sorted(aragora_dir.rglob("*.py")):
        if any(d in py_file.parts for d in skip_dirs):
            continue
        if py_file.name == "conftest.py":
            continue

        try:
            content = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        matches = list(_SILENT_CATCH_RE.finditer(content))
        if not matches:
            continue

        # Skip if all matches have noqa
        real_matches = []
        for m in matches:
            line_start = content.rfind("\n", 0, m.start()) + 1
            line = content[line_start : content.find("\n", m.start())]
            if "# noqa" not in line and "# intentional" not in line.lower():
                real_matches.append(m)

        if not real_matches:
            continue

        rel = str(py_file.relative_to(repo_root))
        line_numbers = []
        for m in real_matches:
            line_num = content[: m.start()].count("\n") + 1
            line_numbers.append(str(line_num))

        candidates.append(
            BossIssueCandidate(
                category="silent_exception",
                title=f"Replace silent exception swallowing in {py_file.name}",
                description=(
                    f"Replace `except ...: pass` patterns with proper error handling "
                    f"in `{rel}` (lines {', '.join(line_numbers)}).\n\n"
                    f"### Requirements\n"
                    f"1. Read `{rel}` and find all `except ...: pass` patterns\n"
                    f"2. For each, either:\n"
                    f"   - Add `logger.debug(...)` or `logger.warning(...)` to make failures visible\n"
                    f"   - Re-raise with context if the exception should propagate\n"
                    f"   - Add a `# noqa` comment with justification if silence is intentional\n"
                    f"3. Do not change behavior — only add visibility"
                ),
                file_scope=[rel],
                validation_command=f"ruff check {rel}",
                acceptance_criteria=[
                    "No bare `except ...: pass` patterns remain without justification",
                    f"`ruff check {rel}` passes",
                    "Existing tests still pass",
                ],
                estimated_complexity="small",
                expected_success_rate=0.8,
            )
        )

    return candidates[:limit]


# ---------------------------------------------------------------------------
# Scanner 3: Bare except Exception handlers
# ---------------------------------------------------------------------------

_BROAD_EXCEPT_RE = re.compile(r"except\s+Exception\s*(?:as\s+\w+)?\s*:")


def scan_bare_except_handlers(
    repo_root: Path,
    *,
    limit: int = 20,
) -> list[BossIssueCandidate]:
    """Find broad except Exception: handlers that could be narrowed."""
    aragora_dir = repo_root / "aragora"
    candidates: list[BossIssueCandidate] = []
    skip_dirs = {"__pycache__", "archive", "migrations"}

    for py_file in sorted(aragora_dir.rglob("*.py")):
        if any(d in py_file.parts for d in skip_dirs):
            continue

        try:
            content = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        matches = []
        for m in _BROAD_EXCEPT_RE.finditer(content):
            line_start = content.rfind("\n", 0, m.start()) + 1
            line = content[line_start : content.find("\n", m.end())]
            if "# noqa" not in line:
                line_num = content[: m.start()].count("\n") + 1
                matches.append(line_num)

        if not matches:
            continue

        rel = str(py_file.relative_to(repo_root))
        candidates.append(
            BossIssueCandidate(
                category="broad_exception",
                title=f"Narrow broad except Exception in {py_file.name}",
                description=(
                    f"Replace broad `except Exception:` handlers with specific exception "
                    f"types in `{rel}` (lines {', '.join(str(n) for n in matches)}).\n\n"
                    f"### Requirements\n"
                    f"1. Read `{rel}` and identify each `except Exception:` handler\n"
                    f"2. Determine the specific exceptions that can be raised\n"
                    f"3. Replace with specific types (e.g., `ValueError`, `OSError`, `KeyError`)\n"
                    f"4. If the broad catch is intentional (e.g., plugin or callback boundaries), "
                    f"add `# noqa: BLE001` with a comment explaining why"
                ),
                file_scope=[rel],
                validation_command=f"ruff check {rel}",
                acceptance_criteria=[
                    "No `except Exception:` without `# noqa: BLE001` justification",
                    f"`ruff check {rel}` passes",
                    "Existing tests still pass",
                ],
                estimated_complexity="small" if len(matches) <= 3 else "medium",
                expected_success_rate=0.9,
            )
        )

    return candidates[:limit]


# ---------------------------------------------------------------------------
# Scanner 4: Actionable TODOs
# ---------------------------------------------------------------------------

_TODO_RE = re.compile(r"#\s*(TODO|FIXME|HACK|XXX)\s*[:\-]?\s*(.+)", re.IGNORECASE)


def scan_actionable_todos(
    repo_root: Path,
    *,
    min_length: int = 25,
    limit: int = 15,
) -> list[BossIssueCandidate]:
    """Find TODO/FIXME comments with actionable descriptions."""
    aragora_dir = repo_root / "aragora"
    candidates: list[BossIssueCandidate] = []
    skip_dirs = {"__pycache__", "archive", "migrations", "node_modules"}

    for py_file in sorted(aragora_dir.rglob("*.py")):
        if any(d in py_file.parts for d in skip_dirs):
            continue

        try:
            content = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        file_todos: list[tuple[str, str, int]] = []
        for i, line in enumerate(content.splitlines(), 1):
            m = _TODO_RE.search(line)
            if m:
                marker = m.group(1).upper()
                message = m.group(2).strip().rstrip(".")
                if len(message) >= min_length:
                    file_todos.append((marker, message, i))

        if not file_todos:
            continue

        rel = str(py_file.relative_to(repo_root))
        todo_list = "\n".join(
            f"   - Line {line}: `{marker}`: {msg}" for marker, msg, line in file_todos[:5]
        )

        candidates.append(
            BossIssueCandidate(
                category="actionable_todo",
                title=f"Address TODO/FIXME items in {py_file.name}",
                description=(
                    f"Address the following TODO/FIXME items in `{rel}`:\n\n"
                    f"{todo_list}\n\n"
                    f"### Requirements\n"
                    f"1. Read `{rel}` and understand the context around each TODO\n"
                    f"2. Implement the requested change or improvement\n"
                    f"3. Remove the TODO comment after addressing it\n"
                    f"4. If a TODO is no longer relevant, remove it with a brief commit note"
                ),
                file_scope=[rel],
                validation_command=f"ruff check {rel}",
                acceptance_criteria=[
                    "All addressed TODOs are removed",
                    f"`ruff check {rel}` passes",
                    "Existing tests still pass",
                ],
                estimated_complexity="small" if len(file_todos) <= 2 else "medium",
                expected_success_rate=0.5,
            )
        )

    return candidates[:limit]


# ---------------------------------------------------------------------------
# Scanner 5: Handler input validation gaps
# ---------------------------------------------------------------------------

_HANDLER_METHOD_RE = re.compile(
    r"(?:async\s+)?def\s+(handle_(?:post|put|patch|delete)|_handle_(?:post|put|patch|delete))\s*\(",
)


def scan_handler_validation_gaps(
    repo_root: Path,
    *,
    limit: int = 15,
) -> list[BossIssueCandidate]:
    """Find POST/PUT/PATCH/DELETE handlers missing request body validation."""
    handlers_dir = repo_root / "aragora" / "server" / "handlers"
    if not handlers_dir.exists():
        return []

    candidates: list[BossIssueCandidate] = []

    for py_file in sorted(handlers_dir.rglob("*.py")):
        if py_file.name.startswith("__"):
            continue

        try:
            content = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        methods = _HANDLER_METHOD_RE.findall(content)
        if not methods:
            continue

        # Check if file has any validation patterns
        has_validation = any(
            pattern in content
            for pattern in [
                "isinstance(body",
                "if not body",
                "json.loads(",
                "request.json",
                "schema",
                "validate(",
            ]
        )

        if has_validation:
            continue

        rel = str(py_file.relative_to(repo_root))

        candidates.append(
            BossIssueCandidate(
                category="handler_validation",
                title=f"Add request body validation to {py_file.name} handlers",
                description=(
                    f"Add input validation for POST/PUT/PATCH/DELETE handlers in `{rel}`.\n\n"
                    f"### Requirements\n"
                    f"1. Read `{rel}` and identify handler methods: {', '.join(methods)}\n"
                    f"2. Add request body validation:\n"
                    f"   - Check body is not None/empty\n"
                    f"   - Validate expected fields exist and have correct types\n"
                    f"   - Return 400 with clear error message on invalid input\n"
                    f"3. Do not change existing behavior for valid inputs"
                ),
                file_scope=[rel],
                validation_command=f"ruff check {rel}",
                acceptance_criteria=[
                    "All write handlers validate request body",
                    "Invalid input returns 400 with descriptive error",
                    f"`ruff check {rel}` passes",
                    "Existing tests still pass",
                ],
                estimated_complexity="medium",
                expected_success_rate=0.5,
            )
        )

    return candidates[:limit]


# ---------------------------------------------------------------------------
# Scanner 6: Type annotation gaps
# ---------------------------------------------------------------------------

_DEF_WITHOUT_RETURN_RE = re.compile(
    r"^    def\s+(?!_)\w+\s*\([^)]*\)\s*:",
    re.MULTILINE,
)
_HAS_RETURN_TYPE_RE = re.compile(r"\)\s*->\s*")


def scan_type_annotation_gaps(
    repo_root: Path,
    *,
    limit: int = 10,
) -> list[BossIssueCandidate]:
    """Find public methods missing return type annotations."""
    aragora_dir = repo_root / "aragora"
    candidates: list[BossIssueCandidate] = []
    skip_dirs = {"__pycache__", "archive", "migrations"}

    for py_file in sorted(aragora_dir.rglob("*.py")):
        if any(d in py_file.parts for d in skip_dirs):
            continue
        if py_file.name.startswith("__"):
            continue

        try:
            content = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        # Find public method defs without return types
        missing_count = 0
        for m in _DEF_WITHOUT_RETURN_RE.finditer(content):
            line = m.group(0)
            if not _HAS_RETURN_TYPE_RE.search(line):
                missing_count += 1

        if missing_count < 3:  # Only flag files with 3+ missing annotations
            continue

        rel = str(py_file.relative_to(repo_root))
        candidates.append(
            BossIssueCandidate(
                category="type_annotation",
                title=f"Add return type annotations to {py_file.name}",
                description=(
                    f"Add return type annotations to {missing_count} public methods "
                    f"in `{rel}`.\n\n"
                    f"### Requirements\n"
                    f"1. Read `{rel}` and identify public methods missing `-> ...` annotations\n"
                    f"2. Add appropriate return type annotations\n"
                    f"3. Use `None` for methods that don't return a value\n"
                    f"4. Use `Any` only as a last resort"
                ),
                file_scope=[rel],
                validation_command=f"ruff check {rel}",
                acceptance_criteria=[
                    "All public methods have return type annotations",
                    f"`ruff check {rel}` passes",
                ],
                estimated_complexity="small" if missing_count < 10 else "medium",
                expected_success_rate=0.4,
            )
        )

    return candidates[:limit]


# ---------------------------------------------------------------------------
# Aggregate scanner
# ---------------------------------------------------------------------------

CATEGORY_PRIORITY = {
    "broad_exception": 0,
    "silent_exception": 1,
    "test_coverage": 2,
    "handler_validation": 3,
    "actionable_todo": 4,
    "type_annotation": 5,
}


def scan_all(
    repo_root: Path,
    *,
    categories: list[str] | None = None,
    metrics_path: Path | None = None,
    signal_log_path: Path | None = None,
    min_success_rate: float = 0.3,
) -> list[BossIssueCandidate]:
    """Run all scanners and return merged, prioritized candidates."""
    all_scanners = {
        "broad_exception": scan_bare_except_handlers,
        "silent_exception": scan_silent_exception_swallowing,
        "test_coverage": scan_untested_modules,
        "handler_validation": scan_handler_validation_gaps,
        "actionable_todo": scan_actionable_todos,
        "type_annotation": scan_type_annotation_gaps,
    }

    selected = categories or list(all_scanners.keys())
    candidates: list[BossIssueCandidate] = []

    for cat in selected:
        scanner = all_scanners.get(cat)
        if scanner:
            candidates.extend(scanner(repo_root))

    resolved_metrics_path = metrics_path or repo_root / DEFAULT_BOSS_METRICS_PATH
    calibrated_rates = expected_success_rates(
        resolved_metrics_path,
        signal_log_path=signal_log_path,
    )
    for candidate in candidates:
        if candidate.category in calibrated_rates:
            candidate.expected_success_rate = calibrated_rates[candidate.category]

    if min_success_rate > 0:
        candidates = [
            candidate
            for candidate in candidates
            if candidate.expected_success_rate >= min_success_rate
        ]

    # Sort: highest success rate first, then by category priority
    candidates.sort(
        key=lambda c: (
            -c.expected_success_rate,
            CATEGORY_PRIORITY.get(c.category, 99),
        ),
    )
    return candidates
