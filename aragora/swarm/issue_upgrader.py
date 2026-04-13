"""Upgrade low-quality boss-ready issues into high-quality worker-ready specs.

Instead of filtering out hard issues, this module transforms them by:
1. Reading the target module to understand what it actually does
2. Identifying specific functions, dependencies, and test patterns
3. Rewriting the issue body with concrete guidance the worker can follow

This is the B1 (Assist) booster: the system drafts high-quality work orders
from vague inputs, so workers succeed instead of crashing.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
_PATH_RE = re.compile(r"`(?P<path>aragora/[^\s`]+\.py)`")
_TEST_PATH_PREFIX = Path("tests")
_MAX_SIMPLE_LOC = 120
_MAX_SIMPLE_PUBLIC_API = 8
_MAX_SIMPLE_EXTERNAL_WEIGHT = 2
_SKIP_SENTINELS = frozenset({"__init__.py", "__main__.py"})
_CONCRETE_MOCK_GUIDANCE: dict[str, str] = {
    "httpx": "Patch `httpx` clients or request helpers to return deterministic responses.",
    "requests": "Monkeypatch `requests` calls so tests do not hit the network.",
    "aiohttp": "Use async fakes for `aiohttp` sessions and responses.",
    "anthropic": "Stub Anthropic client calls and assert the prompt contract only.",
    "openai": "Stub OpenAI client calls and assert the request payload only.",
    "boto3": "Stub `boto3` clients/resources so tests never reach AWS.",
    "redis": "Replace Redis clients with an in-memory fake or monkeypatched methods.",
    "sqlalchemy": "Mock SQLAlchemy sessions/engines instead of opening a real database.",
    "asyncpg": "Patch asyncpg connection helpers with deterministic fakes.",
    "psycopg": "Stub psycopg connections/cursors so tests stay local.",
    "subprocess": "Patch `subprocess` invocations and assert command construction only.",
}
_KNOWN_LOCAL_IMPORT_PREFIXES = ("aragora", "tests")
_STDLIB_MODULES = set(getattr(sys, "stdlib_module_names", set()))
_FILE_IO_HINTS = (
    "open(",
    "read_text(",
    "write_text(",
    "mkdir(",
    "unlink(",
    "rename(",
    "replace(",
    "iterdir(",
)
_VALUE_ERROR_HINTS = (
    "int(",
    "float(",
    "decimal(",
    "fromisoformat(",
    "strptime(",
    "uuid(",
)
_KEY_ERROR_HINTS = (
    "payload[",
    "data[",
    "mapping[",
    "headers[",
    "meta[",
    "config[",
)
_CLEANUP_CONTEXT_HINTS = (
    "close(",
    "cleanup",
    "shutdown",
    "teardown",
    "best effort",
    "best-effort",
    "release(",
    "cancel(",
    "unlink(",
    "remove(",
)


@dataclass
class UpgradedIssue:
    """An issue that's been enriched with concrete module analysis."""

    original_title: str
    original_body: str
    upgraded_title: str
    upgraded_body: str
    module_summary: str
    functions_found: list[str]
    loc: int
    imports: list[str]
    complexity: str  # "trivial", "simple", "medium", "complex"
    upgrade_method: str  # "llm" or "heuristic"


@dataclass(slots=True)
class _ModuleAnalysis:
    docstring: str
    public_functions: list[str]
    public_classes: list[str]
    public_methods: dict[str, list[str]]
    imports: list[str]
    external_imports: list[str]
    mock_hints: list[str]
    loc: int
    complexity: str
    has_async: bool
    has_useful_public_api: bool
    is_obvious_reexport_or_empty: bool
    external_dependency_weight: int


@dataclass(slots=True)
class _BroadExceptSite:
    line_number: int
    suggestions: list[str]
    rationale: str


@dataclass(slots=True)
class _SilentExceptionSite:
    line_number: int
    guidance: str


@dataclass(slots=True)
class _MissingReturnAnnotation:
    line_number: int
    name: str
    suggested_type: str


def _read_module(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _primary_module_path(issue_body: str) -> str | None:
    match = _PATH_RE.search(str(issue_body or ""))
    if not match:
        return None
    return str(match.group("path")).strip()


def _iter_code_lines(content: str) -> list[str]:
    lines: list[str] = []
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


def _normalized_import_root(node: ast.AST) -> str:
    if isinstance(node, ast.Import):
        if not node.names:
            return ""
        return str(node.names[0].name.split(".", 1)[0]).strip()
    if isinstance(node, ast.ImportFrom):
        if node.level and not node.module:
            return ""
        return str((node.module or "").split(".", 1)[0]).strip()
    return ""


def _is_external_import(root: str) -> bool:
    if not root:
        return False
    if root in _STDLIB_MODULES:
        return False
    if any(
        root == prefix or root.startswith(f"{prefix}.") for prefix in _KNOWN_LOCAL_IMPORT_PREFIXES
    ):
        return False
    return True


def _mock_hint_for_import(root: str) -> str | None:
    return _CONCRETE_MOCK_GUIDANCE.get(root)


def _estimate_complexity(
    *,
    loc: int,
    public_api_size: int,
    external_dependency_weight: int,
    has_async: bool,
) -> str:
    if loc <= 40 and public_api_size <= 3 and external_dependency_weight == 0 and not has_async:
        return "trivial"
    if (
        loc <= _MAX_SIMPLE_LOC
        and public_api_size <= _MAX_SIMPLE_PUBLIC_API
        and external_dependency_weight <= _MAX_SIMPLE_EXTERNAL_WEIGHT
    ):
        return "simple"
    if loc <= 260 and public_api_size <= 16 and external_dependency_weight <= 4:
        return "medium"
    return "complex"


def _is_obvious_reexport_or_empty(module: ast.Module) -> bool:
    body = list(module.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        return True

    allowed_assign_targets = {"__all__", "__version__"}
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            elif isinstance(node.target, ast.Name):
                targets = [node.target.id]
            if targets and all(target in allowed_assign_targets for target in targets):
                continue
        return False
    return True


def _analyze_module(content: str) -> _ModuleAnalysis | None:
    try:
        module = ast.parse(content)
    except SyntaxError:
        return None

    public_functions: list[str] = []
    public_classes: list[str] = []
    public_methods: dict[str, list[str]] = {}
    imports: list[str] = []
    external_imports: list[str] = []
    mock_hints: list[str] = []
    has_async = False

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            public_functions.append(node.name)
        elif isinstance(node, ast.AsyncFunctionDef) and not node.name.startswith("_"):
            public_functions.append(node.name)
            has_async = True
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            public_classes.append(node.name)
            methods: list[str] = []
            for child in node.body:
                if isinstance(child, ast.AsyncFunctionDef):
                    has_async = True
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and not child.name.startswith("_"):
                    methods.append(child.name)
            public_methods[node.name] = methods
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            rendered = ast.unparse(node).strip()
            imports.append(rendered)
            root = _normalized_import_root(node)
            if _is_external_import(root) and root not in external_imports:
                external_imports.append(root)
                hint = _mock_hint_for_import(root)
                if hint:
                    mock_hints.append(hint)

    public_api_size = len(public_functions) + len(public_classes)
    return _ModuleAnalysis(
        docstring=(ast.get_docstring(module) or "").strip()[:200],
        public_functions=public_functions,
        public_classes=public_classes,
        public_methods=public_methods,
        imports=imports,
        external_imports=external_imports,
        mock_hints=mock_hints,
        loc=len(_iter_code_lines(content)),
        complexity=_estimate_complexity(
            loc=len(_iter_code_lines(content)),
            public_api_size=public_api_size,
            external_dependency_weight=len(external_imports),
            has_async=has_async,
        ),
        has_async=has_async,
        has_useful_public_api=(public_api_size > 0),
        is_obvious_reexport_or_empty=_is_obvious_reexport_or_empty(module),
        external_dependency_weight=len(external_imports),
    )


def _generated_test_path(module_rel: str) -> str:
    parts = Path(module_rel).parts
    if parts and parts[0] == "aragora" and len(parts) >= 2:
        return str(_TEST_PATH_PREFIX / Path(*parts[1:-1]) / f"test_{parts[-1]}")
    return str(_TEST_PATH_PREFIX / f"test_{Path(module_rel).name}")


def _render_public_api_section(analysis: _ModuleAnalysis) -> str:
    lines: list[str] = []
    if analysis.public_functions:
        lines.append("Public functions:")
        lines.extend(f"- `{name}()`" for name in analysis.public_functions[:12])
    if analysis.public_classes:
        if lines:
            lines.append("")
        lines.append("Public classes:")
        for class_name in analysis.public_classes[:8]:
            methods = analysis.public_methods.get(class_name, [])
            if methods:
                method_bits = ", ".join(f"`{method}()`" for method in methods[:5])
                lines.append(f"- `{class_name}` with methods {method_bits}")
            else:
                lines.append(f"- `{class_name}`")
    return "\n".join(lines).strip()


def _render_mock_guidance(analysis: _ModuleAnalysis) -> str:
    if not analysis.mock_hints:
        return "- No external dependency mocking required."
    return "\n".join(f"- {hint}" for hint in analysis.mock_hints[:5])


def _module_upgrade_title(module_rel: str, *, prefix: str) -> str:
    parts = Path(module_rel).parts
    rendered = "/".join(parts[1:]) if len(parts) > 1 else module_rel
    return f"{prefix} {rendered}"


def _content_line(content: str, line_number: int) -> str:
    lines = content.splitlines()
    if 1 <= line_number <= len(lines):
        return lines[line_number - 1]
    return ""


def _segment_for_node(content: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(content, node)
    return str(segment or "").strip()


def _infer_broad_exception_suggestions(try_source: str) -> tuple[list[str], str]:
    normalized = try_source.lower()
    suggestions: list[str] = []
    rationale: list[str] = []

    if any(token in normalized for token in _FILE_IO_HINTS):
        suggestions.append("OSError")
        rationale.append("file or path operations are present")
    if "json.loads(" in normalized:
        suggestions.append("json.JSONDecodeError")
        rationale.append("JSON parsing is present")
    if any(token in normalized for token in _VALUE_ERROR_HINTS):
        suggestions.append("ValueError")
        rationale.append("value parsing/conversion is present")
    if any(token in normalized for token in _KEY_ERROR_HINTS):
        suggestions.append("KeyError")
        rationale.append("mapping lookups are present")

    if suggestions:
        return list(dict.fromkeys(suggestions)), "; ".join(rationale)
    return [], "no deterministic narrow type was inferred from the try block"


def _collect_broad_exception_sites(content: str) -> list[_BroadExceptSite]:
    try:
        module = ast.parse(content)
    except SyntaxError:
        return []

    sites: list[_BroadExceptSite] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Try):
            continue
        try_source = _segment_for_node(content, node)
        for handler in node.handlers:
            if not isinstance(handler.type, ast.Name) or handler.type.id != "Exception":
                continue
            line = _content_line(content, handler.lineno)
            if "# noqa" in line:
                continue
            suggestions, rationale = _infer_broad_exception_suggestions(try_source)
            sites.append(
                _BroadExceptSite(
                    line_number=handler.lineno,
                    suggestions=suggestions,
                    rationale=rationale,
                )
            )
    return sites


def _classify_silent_exception_guidance(handler_line: str, try_source: str) -> str:
    normalized = f"{handler_line}\n{try_source}".lower()
    if any(token in normalized for token in _CLEANUP_CONTEXT_HINTS):
        return (
            "If this is truly best-effort cleanup, keep the silence only with an explicit "
            "`# noqa` justification. Otherwise surface a minimal debug log."
        )
    return (
        "Replace the silent swallow with `logger.warning(...)` or `logger.debug(...)` so "
        "failures become visible without changing control flow."
    )


def _collect_silent_exception_sites(content: str) -> list[_SilentExceptionSite]:
    try:
        module = ast.parse(content)
    except SyntaxError:
        return []

    sites: list[_SilentExceptionSite] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Try):
            continue
        try_source = _segment_for_node(content, node)
        for handler in node.handlers:
            if len(handler.body) != 1 or not isinstance(handler.body[0], ast.Pass):
                continue
            line = _content_line(content, handler.lineno)
            if "# noqa" in line or "# intentional" in line.lower():
                continue
            sites.append(
                _SilentExceptionSite(
                    line_number=handler.lineno,
                    guidance=_classify_silent_exception_guidance(line, try_source),
                )
            )
    return sites


def _infer_return_annotation(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    returns = [child.value for child in ast.walk(node) if isinstance(child, ast.Return)]
    if not returns or all(value is None for value in returns):
        return "None"

    inferred: list[str] = []
    for value in returns:
        if value is None:
            inferred.append("None")
        elif isinstance(value, ast.Constant):
            if value.value is None:
                inferred.append("None")
            elif isinstance(value.value, bool):
                inferred.append("bool")
            elif isinstance(value.value, int):
                inferred.append("int")
            elif isinstance(value.value, float):
                inferred.append("float")
            elif isinstance(value.value, str):
                inferred.append("str")
            else:
                inferred.append("Any")
        elif isinstance(value, ast.Dict):
            inferred.append("dict[str, Any]")
        elif isinstance(value, ast.List | ast.ListComp):
            inferred.append("list[Any]")
        elif isinstance(value, ast.Tuple):
            inferred.append("tuple[Any, ...]")
        elif isinstance(value, ast.Set):
            inferred.append("set[Any]")
        else:
            inferred.append("Any")

    unique = list(dict.fromkeys(inferred))
    if len(unique) == 1:
        return unique[0]
    if set(unique) <= {"None", "Any"}:
        return "Any"
    return "Any"


def _collect_missing_return_annotations(content: str) -> list[_MissingReturnAnnotation]:
    try:
        module = ast.parse(content)
    except SyntaxError:
        return []

    missing: list[_MissingReturnAnnotation] = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith(
            "_"
        ):
            if node.returns is None:
                missing.append(
                    _MissingReturnAnnotation(
                        line_number=node.lineno,
                        name=node.name,
                        suggested_type=_infer_return_annotation(node),
                    )
                )
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            for child in node.body:
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and not child.name.startswith("_"):
                    if child.returns is None:
                        missing.append(
                            _MissingReturnAnnotation(
                                line_number=child.lineno,
                                name=f"{node.name}.{child.name}",
                                suggested_type=_infer_return_annotation(child),
                            )
                        )
                    if len(missing) >= 10:
                        break
        if len(missing) >= 10:
            break
    return missing


def _upgrade_test_coverage_issue(
    *,
    title: str,
    body: str,
    module_rel: str,
    analysis: _ModuleAnalysis,
) -> UpgradedIssue | None:
    if analysis.is_obvious_reexport_or_empty or not analysis.has_useful_public_api:
        return None
    if analysis.complexity not in {"trivial", "simple"}:
        return None
    if analysis.external_dependency_weight > _MAX_SIMPLE_EXTERNAL_WEIGHT:
        return None
    if analysis.external_imports and len(analysis.mock_hints) < len(analysis.external_imports):
        return None

    test_rel = _generated_test_path(module_rel)
    public_api_section = _render_public_api_section(analysis)
    module_purpose = f"**Module purpose:** {analysis.docstring}\n\n" if analysis.docstring else ""
    async_note = (
        "\n### Async handling\n- Use `pytest.mark.asyncio` for async entrypoints or wrap them with `asyncio.run()` in focused unit tests.\n"
        if analysis.has_async
        else ""
    )
    upgraded_body = (
        "## Task\n\n"
        f"Add focused unit tests for `{module_rel}`.\n\n"
        f"{module_purpose}"
        "### Public API to cover\n"
        f"{public_api_section}\n\n"
        "### Mock guidance\n"
        f"{_render_mock_guidance(analysis)}\n"
        f"{async_note}"
        "### File Scope\n"
        f"- `{module_rel}`\n"
        f"- `{test_rel}` (create)\n\n"
        "### Validation\n"
        "```bash\n"
        f"pytest {test_rel} -q\n"
        "```\n\n"
        "### Acceptance Criteria\n"
        "- Tests cover the listed public API directly.\n"
        "- External dependencies stay mocked or faked.\n"
        "- The test file runs with a single focused pytest command.\n"
        "- Keep the lane scoped to this module and its mirrored test file.\n\n"
        "### Constraints\n"
        f"- Estimated complexity: {analysis.complexity}\n"
        "- Do not broaden into decomposition or cross-module planning."
    )
    return UpgradedIssue(
        original_title=title,
        original_body=body,
        upgraded_title=_module_upgrade_title(module_rel, prefix="Add unit tests for"),
        upgraded_body=upgraded_body,
        module_summary=analysis.docstring,
        functions_found=list(analysis.public_functions),
        loc=analysis.loc,
        imports=list(analysis.external_imports[:10]),
        complexity=analysis.complexity,
        upgrade_method="heuristic",
    )


def _upgrade_broad_exception_issue(
    *,
    title: str,
    body: str,
    module_rel: str,
    content: str,
    analysis: _ModuleAnalysis,
) -> UpgradedIssue | None:
    if analysis.complexity not in {"trivial", "simple"}:
        return None
    sites = _collect_broad_exception_sites(content)
    if not sites:
        return None

    site_lines: list[str] = []
    for site in sites[:8]:
        if site.suggestions:
            rendered = ", ".join(f"`{item}`" for item in site.suggestions)
            site_lines.append(
                f"- Line {site.line_number}: likely narrow to {rendered} ({site.rationale})."
            )
        else:
            site_lines.append(
                f"- Line {site.line_number}: no deterministic narrow type; keep `BLE001` only with an explicit boundary justification."
            )

    upgraded_body = (
        "## Task\n\n"
        f"Narrow the broad `except Exception` handlers in `{module_rel}`.\n\n"
        "### Catch sites to inspect\n"
        f"{chr(10).join(site_lines)}\n\n"
        "### File Scope\n"
        f"- `{module_rel}`\n\n"
        "### Validation\n"
        "```bash\n"
        f"ruff check {module_rel}\n"
        "```\n\n"
        "### Acceptance Criteria\n"
        "- Each broad catch is replaced with a specific exception type, or carries `# noqa: BLE001` with a boundary justification.\n"
        "- Keep behavior unchanged beyond narrowing the catch.\n"
        f"- `ruff check {module_rel}` passes.\n\n"
        "### Constraints\n"
        f"- Estimated complexity: {analysis.complexity}\n"
        "- Keep the lane scoped to this module only."
    )
    return UpgradedIssue(
        original_title=title,
        original_body=body,
        upgraded_title=_module_upgrade_title(module_rel, prefix="Narrow broad except handlers in"),
        upgraded_body=upgraded_body,
        module_summary=analysis.docstring,
        functions_found=[f"line {site.line_number}" for site in sites[:10]],
        loc=analysis.loc,
        imports=list(analysis.external_imports[:10]),
        complexity=analysis.complexity,
        upgrade_method="heuristic",
    )


def _upgrade_silent_exception_issue(
    *,
    title: str,
    body: str,
    module_rel: str,
    content: str,
    analysis: _ModuleAnalysis,
) -> UpgradedIssue | None:
    sites = _collect_silent_exception_sites(content)
    if not sites:
        return None

    site_lines = "\n".join(f"- Line {site.line_number}: {site.guidance}" for site in sites[:8])
    upgraded_body = (
        "## Task\n\n"
        f"Replace silent exception swallowing in `{module_rel}` with visible, bounded handling.\n\n"
        "### Catch sites to inspect\n"
        f"{site_lines}\n\n"
        "### File Scope\n"
        f"- `{module_rel}`\n\n"
        "### Validation\n"
        "```bash\n"
        f"ruff check {module_rel}\n"
        "```\n\n"
        "### Acceptance Criteria\n"
        "- Each `except ...: pass` either logs at debug/warning level or carries an explicit intentional-silence justification.\n"
        "- Do not broaden the behavior or change control flow.\n"
        f"- `ruff check {module_rel}` passes.\n\n"
        "### Constraints\n"
        f"- Estimated complexity: {analysis.complexity}\n"
        "- Keep the lane scoped to this module only."
    )
    return UpgradedIssue(
        original_title=title,
        original_body=body,
        upgraded_title=_module_upgrade_title(
            module_rel, prefix="Replace silent exception swallowing in"
        ),
        upgraded_body=upgraded_body,
        module_summary=analysis.docstring,
        functions_found=[f"line {site.line_number}" for site in sites[:10]],
        loc=analysis.loc,
        imports=list(analysis.external_imports[:10]),
        complexity=analysis.complexity,
        upgrade_method="heuristic",
    )


def _upgrade_type_annotation_issue(
    *,
    title: str,
    body: str,
    module_rel: str,
    content: str,
    analysis: _ModuleAnalysis,
) -> UpgradedIssue | None:
    missing = _collect_missing_return_annotations(content)
    if not missing:
        return None

    suggestion_lines = "\n".join(
        f"- Line {item.line_number}: `{item.name}` -> suggest `-> {item.suggested_type}`"
        for item in missing[:10]
    )
    upgraded_body = (
        "## Task\n\n"
        f"Add explicit return type annotations for the missing public callables in `{module_rel}`.\n\n"
        "### Suggested annotations\n"
        f"{suggestion_lines}\n\n"
        "### File Scope\n"
        f"- `{module_rel}`\n\n"
        "### Validation\n"
        "```bash\n"
        f"ruff check {module_rel}\n"
        "```\n\n"
        "### Acceptance Criteria\n"
        "- Annotate the listed public callables first and keep the lane bounded.\n"
        "- Use `None` when a callable does not return a value; use `Any` only when inference is genuinely ambiguous.\n"
        f"- `ruff check {module_rel}` passes.\n\n"
        "### Constraints\n"
        f"- Estimated complexity: {analysis.complexity}\n"
        "- Limit the change to this module."
    )
    return UpgradedIssue(
        original_title=title,
        original_body=body,
        upgraded_title=_module_upgrade_title(module_rel, prefix="Add return type annotations for"),
        upgraded_body=upgraded_body,
        module_summary=analysis.docstring,
        functions_found=[item.name for item in missing[:10]],
        loc=analysis.loc,
        imports=list(analysis.external_imports[:10]),
        complexity=analysis.complexity,
        upgrade_method="heuristic",
    )


def upgrade_issue_heuristic(
    title: str,
    body: str,
    *,
    repo_root: Path,
    category: str = "test_coverage",
) -> UpgradedIssue | None:
    """Upgrade an issue using deterministic heuristic module analysis.

    This prototype stays deterministic and bounded. It upgrades only the issue
    categories that can be made more concrete from local module analysis.
    """
    module_rel = _primary_module_path(body)
    if not module_rel:
        return None

    if Path(module_rel).name in _SKIP_SENTINELS:
        return None
    module_path = repo_root / module_rel
    content = _read_module(module_path)
    if content is None:
        return None

    analysis = _analyze_module(content)
    if analysis is None:
        return None

    if category == "test_coverage":
        return _upgrade_test_coverage_issue(
            title=title,
            body=body,
            module_rel=module_rel,
            analysis=analysis,
        )
    if category == "broad_exception":
        return _upgrade_broad_exception_issue(
            title=title,
            body=body,
            module_rel=module_rel,
            content=content,
            analysis=analysis,
        )
    if category == "silent_exception":
        return _upgrade_silent_exception_issue(
            title=title,
            body=body,
            module_rel=module_rel,
            content=content,
            analysis=analysis,
        )
    if category == "type_annotation":
        return _upgrade_type_annotation_issue(
            title=title,
            body=body,
            module_rel=module_rel,
            content=content,
            analysis=analysis,
        )
    return None


async def upgrade_issue_llm(
    title: str,
    body: str,
    *,
    repo_root: Path,
    model: str = "claude-haiku-4-5-20251001",
    timeout: float = 20.0,
) -> UpgradedIssue | None:
    """Upgrade an issue using LLM analysis for richer understanding."""
    # First do the heuristic analysis for the module data
    heuristic = upgrade_issue_heuristic(title, body, repo_root=repo_root)
    if heuristic is None:
        return None

    # Extract module path and content for LLM context
    path_match = re.search(r"`(aragora/\S+\.py)`", body)
    if not path_match:
        return heuristic  # Fall back to heuristic

    module_rel = path_match.group(1)
    module_path = repo_root / module_rel
    content = _read_module(module_path)
    if content is None:
        return heuristic

    # Truncate for LLM context
    content_truncated = content[:4000]

    prompt = f"""Analyze this Python module and suggest specific test cases.

Module: {module_rel} ({heuristic.loc} lines)
Public functions: {", ".join(heuristic.functions_found[:10])}
Complexity: {heuristic.complexity}

```python
{content_truncated}
```

Return a JSON object with:
{{
  "module_purpose": "one sentence describing what this module does",
  "test_cases": [
    {{"name": "test_function_name", "description": "what to test", "approach": "how to test it"}}
  ],
  "mock_strategy": "what needs mocking and why",
  "edge_cases": ["list of edge cases to cover"]
}}"""

    text: str | None = None
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            import anthropic

            client = anthropic.AsyncAnthropic()
            response = await client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
            text = response.content[0].text.strip()
    except Exception as exc:
        logger.debug("LLM upgrade unavailable: %s", exc)

    if text is None:
        try:
            or_key = os.environ.get("OPENROUTER_API_KEY")
            if or_key:
                import httpx

                async with httpx.AsyncClient(timeout=timeout) as http:
                    resp = await http.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {or_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "anthropic/claude-haiku-4.5",
                            "max_tokens": 512,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    )
                    resp.raise_for_status()
                    text = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.debug("OpenRouter upgrade unavailable: %s", exc)

    if text is None:
        return heuristic  # Fall back

    # Parse LLM response and enrich the heuristic body
    import json as _json

    try:
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = _json.loads(text)

        purpose = data.get("module_purpose", heuristic.module_summary)
        test_cases = data.get("test_cases", [])
        mock_strategy = data.get("mock_strategy", "")
        edge_cases = data.get("edge_cases", [])

        # Enrich the body with LLM insights
        tc_lines = "\n".join(
            f"   - `{tc['name']}`: {tc['description']}"
            for tc in test_cases[:10]
            if isinstance(tc, dict)
        )
        edge_lines = "\n".join(f"   - {e}" for e in edge_cases[:5]) if edge_cases else ""

        enrichment = ""
        if tc_lines:
            enrichment += f"\n\n### Suggested test cases\n{tc_lines}"
        if mock_strategy:
            enrichment += f"\n\n### Mocking strategy\n{mock_strategy}"
        if edge_lines:
            enrichment += f"\n\n### Edge cases to cover\n{edge_lines}"

        # Replace module purpose if LLM gave a better one
        upgraded_body = heuristic.upgraded_body
        if purpose and "Module purpose:" not in upgraded_body:
            upgraded_body = upgraded_body.replace(
                "## Task", f"## Task\n\n**Module purpose:** {purpose}", 1
            )
        upgraded_body += enrichment

        return UpgradedIssue(
            original_title=heuristic.original_title,
            original_body=heuristic.original_body,
            upgraded_title=heuristic.upgraded_title,
            upgraded_body=upgraded_body,
            module_summary=purpose or heuristic.module_summary,
            functions_found=heuristic.functions_found,
            loc=heuristic.loc,
            imports=heuristic.imports,
            complexity=heuristic.complexity,
            upgrade_method="llm",
        )
    except (_json.JSONDecodeError, KeyError, IndexError):
        return heuristic  # Fall back on parse failure
