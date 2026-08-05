#!/usr/bin/env python
"""Validate that OpenAPI spec routes match handler registry.

This script ensures the OpenAPI specification stays in sync with actual
handler implementations by comparing:
1. Routes defined in handler ROUTES attributes
2. Literal routes in server-wired aiohttp registration functions
3. Routes in the OpenAPI spec paths

Usage:
    python scripts/validate_openapi_routes.py
    python scripts/validate_openapi_routes.py --spec docs/api/openapi.json
    python scripts/validate_openapi_routes.py --fail-on-missing  # Exit 1 if routes missing
    python scripts/validate_openapi_routes.py --fail-on-missing --baseline scripts/baselines/validate_openapi_routes.json
    python scripts/validate_openapi_routes.py --json  # Output as JSON
    python scripts/validate_openapi_routes.py --ref "$(git rev-parse HEAD)" --json  # Method-aware plane

The method-aware plane (VAL-CDG-011) additionally emits exact
``(method, normalized_path)`` operation sets with complete route-set algebra,
explicit CONNECT debt, and the ratified one-to-many operation projection bound
to the immutable original-cohort identities.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import os
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

# Ensure local checkout modules take precedence over any globally installed package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    # Direct script execution (python scripts/validate_openapi_routes.py)
    from sdk_path_normalize import normalize_sdk_path
except ModuleNotFoundError:
    # Module import context (pytest importing scripts.validate_openapi_routes)
    from scripts.sdk_path_normalize import normalize_sdk_path

_SERVER_ROUTE_REGISTRATION = (
    _REPO_ROOT / "aragora" / "server" / "stream" / "servers_route_registration.py"
)
_LITERAL_ROUTER_METHODS = frozenset(
    {"add_get", "add_post", "add_put", "add_patch", "add_delete", "add_head", "add_options"}
)

# ---------------------------------------------------------------------------
# Method-aware operation plane (VAL-CDG-011)
# ---------------------------------------------------------------------------

# Exact accepted method sets. No extra verb, lowercase alias, extension key,
# or implicit method is accepted anywhere in the method-aware plane.
RUNTIME_METHOD_SET: tuple[str, ...] = (
    "CONNECT",
    "DELETE",
    "GET",
    "HEAD",
    "OPTIONS",
    "PATCH",
    "POST",
    "PUT",
    "TRACE",
)
OPENAPI_METHOD_SET: tuple[str, ...] = (
    "DELETE",
    "GET",
    "HEAD",
    "OPTIONS",
    "PATCH",
    "POST",
    "PUT",
    "TRACE",
)
_RUNTIME_METHODS = frozenset(RUNTIME_METHOD_SET)
_OPENAPI_OPERATION_KEYS = frozenset(method.lower() for method in OPENAPI_METHOD_SET)
_METHOD_ROUTES_ATTRS = tuple(f"{method}_ROUTES" for method in RUNTIME_METHOD_SET)

# Explicit runtime-method token followed by an /api/ path literal in server
# source. Mirrors the ratified HANDLER_METHOD_PATH_LITERAL_RE_V1 witness rule.
_HANDLER_METHOD_PATH_LITERAL_RE = re.compile(
    r"\b(CONNECT|DELETE|GET|HEAD|OPTIONS|PATCH|POST|PUT|TRACE)"
    r"[ \t]+(/api/[^\s\"'`\\)\]>,;]+)"
)

_DEFAULT_INVENTORY_PATH = _REPO_ROOT / "scripts" / "baselines" / "contract_drift_inventory.json"
_PATH_NORMALIZE_AUTHORITY = "scripts/sdk_path_normalize.py"

# Immutable ratified pins (Contract Drift Governance, VAL-CDG-011).
ORIGINAL_RECORD_ID_SET_SHA256 = "c1235670c183b1887ba3fe4280fa0320f9fd6f4a85b8f346d4332ac2aebbe269"
PROJECTION_RECORD_DIGEST_SET_SHA256 = (
    "2d6790a6f825c53047639d9433f40e3e10b5bfc9e357bcd161f6b341134775e5"
)
OPERATION_PROJECTION_SCHEMA = "cdg-operation-projection-v1"
OPERATION_PROJECTION_SCHEMA_VERSION = 1
ORIGINAL_COHORT_TOTAL = 655
ROUTE_PARITY_RECORD_TOTAL = 57
ROUTE_EDGE_TOTAL = 68
ROUTE_MULTI_EDGE_ORIGINALS = 9
ROUTE_MAX_EDGES = 4
ROUTE_EDGE_DISTRIBUTION = {1: 48, 2: 8, 4: 1}
ROUTE_CATEGORY_COUNTS = {
    "routes_missing_in_spec": 11,
    "routes_orphaned_in_spec": 17,
    "sdk_missing_from_both": 29,
}
# For each route category: the one exclusive source-operation set and its
# deterministic selection rule (exact literal membership of the named baseline
# manifest key at the ratified membership commit).
ROUTE_CATEGORY_SELECTION = {
    "routes_missing_in_spec": {
        "source_manifest": "scripts/baselines/validate_openapi_routes.json",
        "source_json_key": "missing_in_spec",
        "selection_rule": (
            "exact literal membership of missing_in_spec in "
            "scripts/baselines/validate_openapi_routes.json at the ratified membership commit"
        ),
    },
    "routes_orphaned_in_spec": {
        "source_manifest": "scripts/baselines/validate_openapi_routes.json",
        "source_json_key": "orphaned_in_spec",
        "selection_rule": (
            "exact literal membership of orphaned_in_spec in "
            "scripts/baselines/validate_openapi_routes.json at the ratified membership commit"
        ),
    },
    "sdk_missing_from_both": {
        "source_manifest": "scripts/baselines/check_sdk_parity.json",
        "source_json_key": "missing_from_both_sdks",
        "selection_rule": (
            "exact literal membership of missing_from_both_sdks in "
            "scripts/baselines/check_sdk_parity.json at the ratified membership commit"
        ),
    },
}
# The nine ratified multi-edge originals and their exact operation IDs.
MULTI_EDGE_OPERATION_PINS = {
    "cdg1:38bad91cd0b0acf27f9d2e200c4ca52857f09733dbf3d8b42bcd62ffaec12f05": (
        "GET /api/coordination/federation",
        "POST /api/coordination/federation",
    ),
    "cdg1:3c78ca805bba20f77932798c52c2854266b84b08ec0d4d903a21833ca5f94b68": (
        "DELETE /api/debates/active",
        "GET /api/debates/active",
        "PATCH /api/debates/active",
        "POST /api/debates/active",
    ),
    "cdg1:640c7ed5a00c72464a8eedb8c4586b4299e59a1717c7711160d1c86195aa57b1": (
        "GET /api/prompt-engine/runs",
        "POST /api/prompt-engine/runs",
    ),
    "cdg1:67dcccb31c0048218421fdcb01fa3c9ca6706288c54c45448d3d5a5511ef8825": (
        "GET /api/coordination/consent",
        "POST /api/coordination/consent",
    ),
    "cdg1:862f836a6d69ee82839d07627d5fec3f97c03a694d08d00435bd66aba139d27a": (
        "GET /api/prompt-engine/runs/{param}",
        "POST /api/prompt-engine/runs/{param}",
    ),
    "cdg1:90f5e8f23f15046585b6cd415d09b010f4d35320136e87d03d071a7c88ee73e6": (
        "GET /api/api-keys",
        "POST /api/api-keys",
    ),
    "cdg1:a18795756e8d40739f6edf2eca3e83c8010c0d83461eba8d41ef97626badb8cd": (
        "GET /api/costs/debates/{param}/performance",
        "POST /api/costs/debates/{param}/performance",
    ),
    "cdg1:ad1cc27167523e46d27d608c664e8722ca051b0dfd7689e9f4ba8c5b01372fd9": (
        "GET /api/coordination/workspaces",
        "POST /api/coordination/workspaces",
    ),
    "cdg1:d17440376695c88fbaad291032536c6da58ca65274aa02ad23edca27bbcc3b69": (
        "GET /api/costs/debates/{param}/line-items",
        "POST /api/costs/debates/{param}/line-items",
    ),
}
# Expected sdk_language provenance per route category (category-derived).
_ROUTE_CATEGORY_SDK_LANGUAGE = {
    "routes_missing_in_spec": [],
    "routes_orphaned_in_spec": [],
    "sdk_missing_from_both": ["python", "typescript"],
}

# Route validation is an offline contract check. Disable AWS Secrets Manager
# lookups so importing handlers does not stall or fail on network-restricted dev/CI
# environments when no secrets are actually needed.
os.environ.setdefault("ARAGORA_USE_SECRETS_MANAGER", "false")


def load_internal_prefixes(path: str | None = None) -> tuple[str, ...]:
    """Load internal/private route prefixes from a policy file.

    Fails CLOSED: if the policy file is missing or unparseable, exit with an
    error instead of silently falling back — a silent fallback would let
    internal route families drift into (or mask leaks in) the public spec.
    """
    if path is None:
        path = str(_REPO_ROOT / "scripts" / "baselines" / "internal_route_prefixes.json")
    p = Path(path)
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"Error: cannot load internal route policy {p}: {exc}. "
            "Refusing to validate without it (fail closed).",
            file=sys.stderr,
        )
        sys.exit(1)
    prefixes = data.get("prefixes") if isinstance(data, dict) else None
    if not isinstance(prefixes, list):
        print(
            f"Error: internal route policy {p} must contain a 'prefixes' list.",
            file=sys.stderr,
        )
        sys.exit(1)
    invalid = [
        item for item in prefixes if not (isinstance(item, str) and item.startswith("/api/"))
    ]
    if invalid:
        print(
            f"Error: internal route policy {p} has non-'/api/' prefixes: {invalid}. "
            "Comparison keys are '/api/'-rooted, so these entries could never match "
            "and would silently fail to exclude anything.",
            file=sys.stderr,
        )
        sys.exit(1)
    normalized = list(prefixes)
    if not normalized:
        print(
            f"Error: internal route policy {p} contains no valid '/api/' prefixes.",
            file=sys.stderr,
        )
        sys.exit(1)
    return tuple(normalized)


def is_internal_route(route: str, prefixes: tuple[str, ...] | list[str]) -> bool:
    """Whether a route falls inside an internal route family.

    Matches the exact family root or anything under it with the slash intact:
    ``/api/v1/sme`` and ``/api/v1/sme/dashboard`` are internal under
    ``/api/v1/sme/``, but sibling names like ``/api/v1/smear`` are NOT
    (round-1 review P2 on #9360: a bare ``startswith(prefix.rstrip("/"))``
    overmatched). Canonical matcher — every checker that applies the
    internal-route policy must delegate here so exclusions stay identical.
    """
    for prefix in prefixes:
        base = prefix.rstrip("/")
        if route == base or route.startswith(base + "/"):
            return True
    return False


class RouteRegistrationScanError(ValueError):
    """A wired route-registration source could not be inspected safely."""


def _parse_python_source(path: Path) -> ast.Module:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RouteRegistrationScanError(f"cannot read route-registration source {path}: {exc}")
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise RouteRegistrationScanError(f"cannot parse route-registration source {path}: {exc}")


def _resolve_local_module(repo_root: Path, module_name: str) -> Path | None:
    relative = Path(*module_name.split("."))
    candidates = (repo_root / relative.with_suffix(".py"), repo_root / relative / "__init__.py")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _resolve_import_module(
    current_module: str,
    current_path: Path,
    imported: ast.ImportFrom,
) -> str | None:
    """Resolve an absolute module name for a local ``from`` import."""
    if imported.level == 0:
        return imported.module

    package_parts = current_module.split(".")
    if current_path.name != "__init__.py":
        package_parts = package_parts[:-1]
    ascents = imported.level - 1
    if ascents > len(package_parts):
        return None
    if ascents:
        package_parts = package_parts[:-ascents]
    if imported.module:
        package_parts.extend(imported.module.split("."))
    return ".".join(package_parts) or None


def _resolve_registration_function(
    repo_root: Path,
    module_name: str,
    function_name: str,
    visited: set[tuple[str, str]] | None = None,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, Path] | None:
    """Resolve a registrar definition through local package reexports."""
    seen = set() if visited is None else visited
    key = (module_name, function_name)
    if key in seen:
        raise RouteRegistrationScanError(
            f"cyclic route-registration reexport while resolving {module_name}:{function_name}"
        )
    seen.add(key)

    module_path = _resolve_local_module(repo_root, module_name)
    if module_path is None:
        return None
    module_tree = _parse_python_source(module_path)
    definitions = [
        node
        for node in module_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    ]
    if definitions:
        return definitions[-1], module_path

    for node in module_tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        for imported in node.names:
            if (imported.asname or imported.name) != function_name:
                continue
            imported_module = _resolve_import_module(module_name, module_path, node)
            if imported_module is None:
                return None
            return _resolve_registration_function(
                repo_root,
                imported_module,
                imported.name,
                seen,
            )
    return None


def _resolve_class_registration_method(
    repo_root: Path,
    module_name: str,
    class_name: str,
    method_name: str,
    visited: set[tuple[str, str]] | None = None,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef | None, Path, int] | None:
    """Resolve ``Class.method`` for an attribute-called wired registrar.

    Returns ``(method, module_path, implicit_params)`` where
    ``implicit_params`` counts leading parameters bound by the descriptor
    protocol at a ``Class.method(app)`` call site: 0 for ``@staticmethod``
    and undecorated functions (accessing through the class yields the plain
    function), 1 for ``@classmethod`` (``cls`` is bound). A method carrying
    any other decorator resolves to ``(None, module_path, 0)``: its runtime
    call semantics are unproven, so its registrations stay unverified rather
    than fabricated. Class reexports are followed like function reexports;
    ``None`` means the class or method cannot be found at all.
    """
    seen = set() if visited is None else visited
    key = (module_name, f"{class_name}.{method_name}")
    if key in seen:
        raise RouteRegistrationScanError(
            f"cyclic route-registration reexport while resolving "
            f"{module_name}:{class_name}.{method_name}"
        )
    seen.add(key)

    module_path = _resolve_local_module(repo_root, module_name)
    if module_path is None:
        return None
    module_tree = _parse_python_source(module_path)
    class_definitions = [
        node
        for node in module_tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if class_definitions:
        methods = [
            node
            for node in class_definitions[-1].body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        ]
        if not methods:
            return None
        method = methods[-1]
        decorators = {
            decorator.id for decorator in method.decorator_list if isinstance(decorator, ast.Name)
        }
        if len(decorators) != len(method.decorator_list) or decorators - {
            "staticmethod",
            "classmethod",
        }:
            return None, module_path, 0
        return method, module_path, (1 if "classmethod" in decorators else 0)

    for node in module_tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        for imported in node.names:
            if (imported.asname or imported.name) != class_name:
                continue
            imported_module = _resolve_import_module(module_name, module_path, node)
            if imported_module is None:
                return None
            return _resolve_class_registration_method(
                repo_root,
                imported_module,
                imported.name,
                method_name,
                seen,
            )
    return None


def _registrar_parameter_bindings(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    implicit_params: int,
    call: ast.Call,
) -> dict[str, tuple[str, ...]]:
    """Exact literal string bindings for registrar parameters at ONE call site.

    A parameter binds only when its value at this specific call is provably a
    single literal string: an explicit literal argument (positional or
    keyword), or the parameter's literal default when this call cannot
    override it. A call using ``*args``/``**kwargs``, a signature using
    positional-only parameters, and any parameter rebound in the registrar
    body all disable binding — the affected paths stay unverified rather
    than folded from a guessed value.
    """
    if function.args.posonlyargs:
        return {}
    if any(isinstance(argument, ast.Starred) for argument in call.args) or any(
        keyword.arg is None for keyword in call.keywords
    ):
        return {}
    rebound = _names_rebound_in(function.body)
    keyword_values = {
        keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None
    }

    def _literal(value: ast.expr | None, name: str) -> tuple[str, ...] | None:
        if (
            value is not None
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and name not in rebound
        ):
            return (value.value,)
        return None

    bindings: dict[str, tuple[str, ...]] = {}
    positional = function.args.args
    defaults_offset = len(positional) - len(function.args.defaults)
    for index, param in enumerate(positional):
        # skip descriptor-bound leading params and the app/router param itself
        if index <= implicit_params:
            continue
        call_index = index - implicit_params
        value: ast.expr | None
        if call_index < len(call.args):
            value = call.args[call_index]
        elif param.arg in keyword_values:
            value = keyword_values[param.arg]
        elif index >= defaults_offset:
            value = function.args.defaults[index - defaults_offset]
        else:
            value = None
        bound = _literal(value, param.arg)
        if bound is not None:
            bindings[param.arg] = bound
    for param, default in zip(function.args.kwonlyargs, function.args.kw_defaults):
        bound = _literal(keyword_values.get(param.arg, default), param.arg)
        if bound is not None:
            bindings[param.arg] = bound
    return bindings


# ast.TryStar arrived in 3.11; the repo floor is 3.10.
_TRY_NODE_TYPES: tuple[type[ast.AST], ...] = (
    (ast.Try, ast.TryStar) if hasattr(ast, "TryStar") else (ast.Try,)
)


def _contains_loop_flow_escape(nodes: Iterable[ast.AST]) -> bool:
    """Whether ``nodes`` contain a statement that can skip loop iterations.

    ``break``, ``continue``, ``return``, and ``raise`` all mean some bound
    values may never reach a fold site in the loop body, so a folded witness
    could claim registrations that never execute (review round 8:
    ``if not enabled(name): continue`` filtered iterations while the fold
    kept every value). Nested definitions are excluded (they do not execute
    during the loop); everything else counts, including escapes belonging to
    nested loops — conservative, since an outer-loop fold cannot distinguish
    which loop an escape cuts short.
    """
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, (ast.Break, ast.Continue, ast.Return, ast.Raise)):
            return True
        if _contains_loop_flow_escape(ast.iter_child_nodes(node)):
            return True
    return False


def _names_rebound_in(nodes: Iterable[ast.AST]) -> set[str]:
    """Names that may be (re)bound anywhere under ``nodes``.

    Conservative superset: assignment/loop/with/walrus targets, ``del``
    statements, except aliases, import aliases, nested def/class names, and
    ``global``/``nonlocal`` declarations all count — including occurrences
    inside nested definitions that would not actually rebind the enclosing
    scope. A false positive only skips folding (the path stays unverified);
    it can never widen the served claim.
    """
    rebound: set[str] = set()
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                rebound.add(child.id)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                rebound.add(child.name)
            elif isinstance(child, ast.alias):
                rebound.add((child.asname or child.name).partition(".")[0])
            elif isinstance(child, (ast.Global, ast.Nonlocal)):
                rebound.update(child.names)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                rebound.add(child.name)
    return rebound


def _iter_executable_calls(
    nodes: Iterable[ast.AST],
    bindings: dict[str, tuple[str, ...]] | None = None,
) -> Iterator[tuple[ast.Call, dict[str, tuple[str, ...]]]]:
    """Yield executable calls with SOUND literal for-loop string bindings.

    Nested definitions that may never execute are excluded. A ``for`` target
    iterating a tuple/list of string constants binds its exact literal values,
    enabling deterministic folding of f-string paths built from those values
    (no evaluation, no heuristics). A folded witness claims that EVERY bound
    value reaches the call, so the binding is dropped wherever that can no
    longer be proven: when the loop body (or, for the ``else`` binding, the
    else block) may rebind the variable; when any nested loop target shadows
    the name — literal nested loops rebind their own exact values, non-literal
    ones invalidate the fold; when the loop body contains any flow escape
    (``break``/``continue``/``return``/``raise``) that could skip iterations
    (review round 8); and inside every conditionally-executed construct
    (``if``/``while`` bodies, ``try`` blocks, ``match`` cases, ternaries,
    boolean short-circuit operands, comprehensions), where a value-dependent
    guard could filter which loop values actually register. Constant paths in
    those positions still count individually, exactly as in straight-line
    code.
    """
    bound = dict(bindings or {})
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, (ast.For, ast.AsyncFor)):
            loop_bound = dict(bound)
            orelse_bound = dict(bound)
            # Whatever the target binds shadows any inherited binding, even
            # when this loop's own values are not literal.
            for name in _names_rebound_in([node.target]):
                loop_bound.pop(name, None)
                orelse_bound.pop(name, None)
            if isinstance(node.target, ast.Name) and isinstance(node.iter, (ast.Tuple, ast.List)):
                literal_values = [
                    element.value
                    for element in node.iter.elts
                    if isinstance(element, ast.Constant) and isinstance(element.value, str)
                ]
                if (
                    literal_values
                    and len(literal_values) == len(node.iter.elts)
                    and node.target.id not in _names_rebound_in(node.body)
                    and not _contains_loop_flow_escape(node.body)
                ):
                    loop_bound[node.target.id] = tuple(literal_values)
                    if node.target.id not in _names_rebound_in(node.orelse):
                        # After exhaustion the loop variable holds only the
                        # LAST element, so the else-block binding is exactly
                        # that value.
                        orelse_bound[node.target.id] = (literal_values[-1],)
            yield from _iter_executable_calls([node.iter], bound)
            yield from _iter_executable_calls(node.body, loop_bound)
            yield from _iter_executable_calls(node.orelse, orelse_bound)
            continue
        if isinstance(node, (ast.If, ast.While)):
            yield from _iter_executable_calls([node.test], bound)
            yield from _iter_executable_calls(node.body, {})
            yield from _iter_executable_calls(node.orelse, {})
            continue
        if isinstance(node, _TRY_NODE_TYPES):
            # ast.TryStar mirrors ast.Try's block attributes exactly.
            try_node = cast(ast.Try, node)
            conditional_parts: list[ast.AST] = [
                *try_node.body,
                *try_node.handlers,
                *try_node.orelse,
                *try_node.finalbody,
            ]
            yield from _iter_executable_calls(conditional_parts, {})
            continue
        if isinstance(node, ast.Match):
            yield from _iter_executable_calls([node.subject], bound)
            yield from _iter_executable_calls(node.cases, {})
            continue
        if isinstance(node, ast.IfExp):
            yield from _iter_executable_calls([node.test], bound)
            yield from _iter_executable_calls([node.body, node.orelse], {})
            continue
        if isinstance(node, ast.BoolOp):
            yield from _iter_executable_calls(node.values[:1], bound)
            yield from _iter_executable_calls(node.values[1:], {})
            continue
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            yield from _iter_executable_calls(ast.iter_child_nodes(node), {})
            continue
        if isinstance(node, ast.Call):
            yield node, bound
        yield from _iter_executable_calls(ast.iter_child_nodes(node), bound)


def _literal_path_candidates(
    path_argument: ast.expr, bindings: dict[str, tuple[str, ...]]
) -> tuple[str, ...]:
    """Exact literal path values for a route argument, or empty when computed.

    Accepts plain string constants and f-strings whose only interpolations are
    for-loop variables bound to literal string tuples: the fold enumerates the
    exact registered paths deterministically. Anything else stays unverified.
    """
    if isinstance(path_argument, ast.Constant) and isinstance(path_argument.value, str):
        return (path_argument.value,)
    if isinstance(path_argument, ast.JoinedStr):
        candidates = [""]
        for part in path_argument.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                candidates = [prefix + part.value for prefix in candidates]
            elif (
                isinstance(part, ast.FormattedValue)
                and isinstance(part.value, ast.Name)
                and part.conversion == -1
                and part.format_spec is None
                and part.value.id in bindings
            ):
                candidates = [
                    prefix + value for prefix in candidates for value in bindings[part.value.id]
                ]
            else:
                return ()
        return tuple(candidates)
    return ()


def get_wired_function_operations(
    registration_path: Path | None = None,
    repo_root: Path | None = None,
    extended: bool = False,
) -> list[dict[str, Any]]:
    """Statically extract literal wired operations with exact method evidence.

    The server registration module is the authority for which free functions
    participate in startup. Only imported ``register_*`` callables that are
    actually called there are considered. Within those exact function bodies,
    literal ``add_<method>('/api/...')`` calls on the wired app's router
    (``app.router.add_*``) with plain constant paths count, and the router
    method name is the exact HTTP method witness.

    With ``extended=True`` (the method-aware plane), additional exact
    evidence forms count: literal ``add_route("METHOD", '/api/...')`` calls
    with the explicit uppercase method constant as the witness; a receiver
    that is the registrar's first argument itself when that parameter is
    literally named ``router`` (``def register(router): router.add_*``);
    f-string paths whose only interpolations are for-loop variables bound to
    literal string tuples or registrar parameters provably bound to a single
    literal string at the specific call site (explicit literal argument or
    an unoverridden literal default), folded to their exact registered
    values; and attribute-called class registrars
    (``Handler.register_routes(app)`` on a class imported by the wiring
    module, review round 7 — the autonomous handlers register exactly this
    way). The legacy path-level plane keeps the original narrower semantics
    so its pinned baseline is not silently re-scoped (review round 4).

    Local package reexports are followed to their definition. Computed
    paths and methods and unrelated route-registration helpers remain
    unverified.
    """
    root = repo_root or _REPO_ROOT
    wiring_path = registration_path or _SERVER_ROUTE_REGISTRATION
    wiring_tree = _parse_python_source(wiring_path)

    imports: dict[str, tuple[str, str]] = {}
    for node in ast.walk(wiring_tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if not node.module or not node.module.startswith("aragora."):
            continue
        for imported in node.names:
            local_name = imported.asname or imported.name
            imports[local_name] = (node.module, imported.name)

    called_names = {
        node.func.id
        for node in ast.walk(wiring_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    # (registrar function, defining module path, witness symbol, app/router
    # parameter name, call-site parameter bindings)
    extraction_jobs: list[
        tuple[
            ast.FunctionDef | ast.AsyncFunctionDef,
            Path,
            str,
            str,
            dict[str, tuple[str, ...]],
        ]
    ] = []
    for local_name in sorted(called_names & imports.keys()):
        module_name, function_name = imports[local_name]
        if not (local_name.startswith("register_") or function_name.startswith("register_")):
            continue

        resolved = _resolve_registration_function(root, module_name, function_name)
        if resolved is None:
            raise RouteRegistrationScanError(
                f"cannot resolve wired registrar {module_name}:{function_name}"
            )
        function, module_path = resolved
        if not function.args.args:
            continue
        extraction_jobs.append(
            (
                function,
                module_path,
                f"{module_name}:{function_name}",
                function.args.args[0].arg,
                {},
            )
        )

    if extended:
        for node in ast.walk(wiring_tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr.startswith("register_")
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in imports
            ):
                continue
            module_name, imported_name = imports[node.func.value.id]
            resolved_method = _resolve_class_registration_method(
                root, module_name, imported_name, node.func.attr
            )
            if resolved_method is None:
                # Not a local class: the imported name may be a MODULE
                # (``from aragora.server import routes; routes.register_all(app)``),
                # whose attribute is then a plain function registrar.
                resolved_module_function = _resolve_registration_function(
                    root, f"{module_name}.{imported_name}", node.func.attr
                )
                if resolved_module_function is None:
                    raise RouteRegistrationScanError(
                        "cannot resolve wired registrar "
                        f"{module_name}:{imported_name}.{node.func.attr}"
                    )
                module_function, function_module_path = resolved_module_function
                if not module_function.args.args:
                    continue
                extraction_jobs.append(
                    (
                        module_function,
                        function_module_path,
                        f"{module_name}.{imported_name}:{node.func.attr}",
                        module_function.args.args[0].arg,
                        _registrar_parameter_bindings(module_function, 0, node),
                    )
                )
                continue
            method_function, method_module_path, implicit_params = resolved_method
            if method_function is None:
                # Unrecognized decorator: runtime call semantics unproven,
                # registrations stay unverified rather than fabricated.
                continue
            if len(method_function.args.args) <= implicit_params:
                continue
            extraction_jobs.append(
                (
                    method_function,
                    method_module_path,
                    f"{module_name}:{imported_name}.{node.func.attr}",
                    method_function.args.args[implicit_params].arg,
                    _registrar_parameter_bindings(method_function, implicit_params, node),
                )
            )

    operations: list[dict[str, Any]] = []
    seen_operations: set[tuple[str, str, str, int]] = set()
    for function, module_path, symbol, app_argument, seed_bindings in extraction_jobs:
        for call, bindings in _iter_executable_calls(function.body, seed_bindings):
            if not isinstance(call.func, ast.Attribute) or not call.args:
                continue
            receiver = call.func.value
            receiver_is_app_router = (
                isinstance(receiver, ast.Attribute)
                and receiver.attr == "router"
                and isinstance(receiver.value, ast.Name)
                and receiver.value.id == app_argument
            )
            # Extended only: the registrar's first argument IS the router,
            # required to be literally named "router" so an arbitrary
            # first-parameter object with unrelated add_* helpers never
            # counts (review round 4).
            receiver_is_direct_router = (
                extended
                and isinstance(receiver, ast.Name)
                and receiver.id == app_argument
                and app_argument == "router"
            )
            if not (receiver_is_app_router or receiver_is_direct_router):
                continue
            if call.func.attr in _LITERAL_ROUTER_METHODS:
                method = call.func.attr.removeprefix("add_").upper()
                path_argument = call.args[0]
            elif extended and call.func.attr == "add_route" and len(call.args) >= 2:
                method_argument = call.args[0]
                if not (
                    isinstance(method_argument, ast.Constant)
                    and isinstance(method_argument.value, str)
                    and method_argument.value in _RUNTIME_METHODS
                ):
                    # Computed or non-canonical method constants stay
                    # unverified: a method witness must be an exact literal.
                    continue
                method = method_argument.value
                path_argument = call.args[1]
            else:
                continue
            if extended:
                path_candidates = _literal_path_candidates(path_argument, bindings)
            elif isinstance(path_argument, ast.Constant) and isinstance(path_argument.value, str):
                path_candidates = (path_argument.value,)
            else:
                path_candidates = ()
            for path in path_candidates:
                if not path.startswith("/api/"):
                    continue
                # The same class registrar may be attribute-called more than
                # once in the wiring module; identical witnesses dedupe.
                witness_key = (method, path, symbol, call.lineno)
                if witness_key in seen_operations:
                    continue
                seen_operations.add(witness_key)
                operations.append(
                    {
                        "method": method,
                        "path": path,
                        "source_path": str(module_path.relative_to(root))
                        if module_path.is_relative_to(root)
                        else str(module_path),
                        "symbol": symbol,
                        "line": call.lineno,
                    }
                )
    return operations


def get_wired_function_routes(
    registration_path: Path | None = None,
    repo_root: Path | None = None,
) -> set[str]:
    """Path-level view with the ORIGINAL narrow semantics.

    The legacy missing/orphan report and its pinned baseline consume this
    set; it deliberately excludes the extended evidence forms so the
    method-aware plane's wider extraction can never silently re-scope the
    baseline (review round 4).
    """
    return {
        operation["path"]
        for operation in get_wired_function_operations(registration_path, repo_root)
    }


def get_handler_routes() -> set[str]:
    """Extract routes from class metadata and decorator metadata.

    Returns:
        Set of route paths defined across all handlers.
    """
    routes: set[str] = set()

    try:
        from aragora.server.handler_registry import HANDLER_REGISTRY
    except ImportError:
        print("Error: Cannot import handler_registry. Ensure aragora is installed.")
        sys.exit(1)

    for attr_name, handler_ref in HANDLER_REGISTRY:
        handler_class = handler_ref
        # Handler registry entries may be deferred import proxies; resolve
        # them so ROUTES metadata is available for validation.
        resolve = getattr(handler_ref, "resolve", None)
        if callable(resolve):
            try:
                handler_class = resolve()
            except Exception:  # noqa: BLE001 - import failures are non-fatal here
                handler_class = None

        if handler_class is None:
            continue

        # Collect ROUTES attribute
        if hasattr(handler_class, "ROUTES"):
            handler_routes = handler_class.ROUTES
            if isinstance(handler_routes, (list, tuple)):
                routes.update(handler_routes)

        # Collect method-specific routes
        for method_attr in (
            "GET_ROUTES",
            "POST_ROUTES",
            "PUT_ROUTES",
            "PATCH_ROUTES",
            "DELETE_ROUTES",
        ):
            if hasattr(handler_class, method_attr):
                method_routes = getattr(handler_class, method_attr)
                if isinstance(method_routes, (list, tuple)):
                    routes.update(method_routes)

        # Collect decorator-backed OpenAPI metadata for handlers that rely on
        # @api_endpoint instead of legacy ROUTES constants.
        for attr_name in dir(handler_class):
            try:
                attr = getattr(handler_class, attr_name)
            except Exception:  # noqa: BLE001 - defensive: discovery should stay best-effort
                continue
            endpoint = getattr(attr, "_openapi", None)
            path = getattr(endpoint, "path", None)
            if isinstance(path, str) and path:
                routes.add(path)

    return routes


# Deliberately bogus tail segment: no real handler route ends in this, so a
# can_handle that claims it is prefix-matching rather than route-matching.
_CANARY_SEGMENT = "zz-nonexistent-canary-zz"


def _probe_can_handle(can_handle: Any, path: str, method: str) -> bool:
    """Call can_handle(path, method), falling back to the single-arg form."""
    try:
        return bool(can_handle(path, method))
    except TypeError:
        try:
            return bool(can_handle(path))
        except Exception:  # noqa: BLE001 - probing must never fail the gate
            return False
    except Exception:  # noqa: BLE001 - probing must never fail the gate
        return False


def filter_served_orphans(candidates: set[str]) -> tuple[set[str], set[str]]:
    """Split orphan candidates into (truly orphaned, served via can_handle).

    Handlers may serve literal paths through ``can_handle`` logic without
    declaring them in ROUTES. Such spec entries are not orphaned: the server
    routes them. Probe every registered handler's ``can_handle`` against each
    candidate (both /api/v1/ and unversioned /api/ forms, all common methods)
    and drop candidates that a handler accepts.

    Specificity guard: a handler only suppresses a candidate if it rejects a
    canary path derived from the same candidate (a clearly nonexistent tail
    segment). Handlers whose ``can_handle`` also claims the canary are broad
    prefix matchers — accepting their claim would let bogus spec paths hide
    behind them. Every suppression is logged (path + handler class) so it is
    never silent.
    """
    if not candidates:
        return set(), set()

    try:
        from aragora.server.handler_registry import HANDLER_REGISTRY
    except ImportError:
        return set(candidates), set()

    probes: list[tuple[str, Any]] = []
    for _attr_name, handler_ref in HANDLER_REGISTRY:
        handler_class = handler_ref
        resolve = getattr(handler_ref, "resolve", None)
        if callable(resolve):
            try:
                handler_class = resolve()
            except Exception:  # noqa: BLE001 - import failures are non-fatal here
                continue
        if handler_class is None:
            continue
        try:
            instance = handler_class.__new__(handler_class)
        except Exception:  # noqa: BLE001 - exotic metaclasses; skip
            continue
        can_handle = getattr(instance, "can_handle", None)
        if callable(can_handle):
            handler_name = getattr(handler_class, "__name__", repr(handler_class))
            probes.append((handler_name, can_handle))

    methods = ("GET", "POST", "PUT", "PATCH", "DELETE")
    served: set[str] = set()
    suppressions: list[tuple[str, str]] = []
    for candidate in sorted(candidates):
        variants = {candidate}
        if candidate.startswith("/api/v1/"):
            variants.add(candidate.replace("/api/v1/", "/api/", 1))
        # Deliberately no legacy->/api/v1/ variant probe: the live dispatch
        # path (aragora/server/router.py) passes the RAW request path to
        # can_handle with no legacy<->v1 aliasing, so a handler accepting the
        # v1 form is not evidence that the legacy spec path is served.
        # Handlers that genuinely serve both forms (strip_version_prefix in
        # can_handle) already pass the direct legacy probe.
        serving_handler: str | None = None
        for handler_name, can_handle in probes:
            non_specific = False
            for variant in variants:
                for method in methods:
                    if not _probe_can_handle(can_handle, variant, method):
                        continue
                    canary = variant.rstrip("/") + "/" + _CANARY_SEGMENT
                    if _probe_can_handle(can_handle, canary, method):
                        # Claims the canary too: prefix-matching can_handle,
                        # not evidence this specific path is served.
                        non_specific = True
                        break
                    serving_handler = handler_name
                    break
                if serving_handler is not None or non_specific:
                    break
            if serving_handler is not None:
                break
        if serving_handler is not None:
            served.add(candidate)
            suppressions.append((candidate, serving_handler))

    if suppressions:
        print(
            f"Spec-orphan suppressions via can_handle ({len(suppressions)}):",
            file=sys.stderr,
        )
        for path, handler_name in suppressions:
            print(f"  - {path} (served by {handler_name})", file=sys.stderr)

    return set(candidates) - served, served


def _iter_spec_paths(spec_path: str) -> list[Path]:
    """Return the primary spec path plus any supplemental generated snapshots."""
    primary = Path(spec_path)
    candidates = [primary]
    if "_generated" not in primary.stem:
        generated = primary.with_name(f"{primary.stem}_generated{primary.suffix}")
        if generated.exists():
            candidates.append(generated)
    return candidates


def get_openapi_routes(spec_path: str) -> set[str]:
    """Extract all paths from one or more OpenAPI specs.

    Args:
        spec_path: Path to the primary OpenAPI JSON file.

    Returns:
        Set of API paths from the primary spec and any sibling generated snapshot.
    """
    path = Path(spec_path)
    if not path.exists():
        print(f"Error: OpenAPI spec not found at {spec_path}")
        sys.exit(1)

    routes: set[str] = set()
    for candidate in _iter_spec_paths(spec_path):
        with open(candidate) as f:
            spec = json.load(f)
        routes.update(spec.get("paths", {}).keys())
    return routes


def normalize_route(route: str | tuple, *, normalize_version: bool = True) -> str:
    """Normalize a route for comparison.

    Handles:
    - Tuple routes like (method, path) or (method, path, handler)
    - Version prefixes (/api/v1/ vs /api/)
    - Trailing slashes
    - Wildcard patterns (* to {param})

    Args:
        route: Raw route string or tuple containing route path.

    Returns:
        Normalized route for comparison.
    """
    import re

    # Handle tuple routes - extract path (second element for (method, path, ...) format)
    if isinstance(route, tuple):
        route = route[1] if len(route) > 1 else str(route[0])

    # Strip HTTP method prefix from string routes like "POST /api/v1/canvas/pipeline/run"
    _methods = ("GET ", "POST ", "PUT ", "PATCH ", "DELETE ", "HEAD ", "OPTIONS ")
    for m in _methods:
        if route.startswith(m):
            route = route[len(m) :]
            break

    # Strip trailing slash
    route = route.rstrip("/")

    # Normalize version prefix when comparing legacy handler metadata. Literal
    # server wiring must match the exact API version it actually registers.
    if normalize_version and route.startswith("/api/") and not route.startswith("/api/v"):
        route = route.replace("/api/", "/api/v1/", 1)

    # Convert wildcard * to generic {param} for comparison
    # This matches both /debates/* and /debates/{id}
    route = re.sub(r"/\*(/|$)", r"/{param}\1", route)

    # Also normalize common OpenAPI param names to generic {param}
    # so /debates/{id} matches /debates/{param}
    route = re.sub(r"/\{[^}]+\}", "/{param}", route)

    return route


class MethodAwareError(ValueError):
    """The method-aware operation plane could not be proved truthfully."""


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    """Compact sorted-key UTF-8 JSON; no BOM; no trailing LF (CDG canonical)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _operation_id(method: str, path: str) -> str:
    return f"{method} {path}"


def path_normalization_binding() -> dict[str, Any]:
    """Digest binding of the single normalization authority.

    The same ``scripts/sdk_path_normalize.py`` bytes back the route plane, the
    SDK-parity checker, and the inventory; consumers compare this binding
    across outputs for schema/version/digest equality.
    """
    authority = _REPO_ROOT / _PATH_NORMALIZE_AUTHORITY
    try:
        authority_bytes = authority.read_bytes()
    except OSError as exc:
        raise MethodAwareError(
            f"cannot read normalization authority {_PATH_NORMALIZE_AUTHORITY}: {exc}"
        ) from exc
    return {
        "authority_path": _PATH_NORMALIZE_AUTHORITY,
        "authority_sha256": _sha256_hex(authority_bytes),
        "schema": "sdk-path-normalize-v1",
        "version": 1,
    }


def _require_exact_ref(ref: str) -> str:
    # 40 hex is a SHA-1 object name; 64 hex supports SHA-256 object-format repos.
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", ref or ""):
        raise MethodAwareError(
            f"--ref must be an exact 40- or 64-hex commit SHA, got {ref!r}; "
            'run with --ref "$(git rev-parse HEAD)"'
        )
    return ref


def _clean_literal_path(raw: str) -> str | None:
    """Sanitize one explicit method-path literal; None when not operation evidence.

    Wildcard literals are prefix claims, not exact operations: a ``*`` segment
    never witnesses a method-specific operation (VAL-CDG-011 forbids wildcard
    and prefix shortcuts).
    """
    path = raw.rstrip(":.,;")
    if not path.startswith("/api/"):
        return None
    if "*" in path:
        return None
    return path


def _relative_source_path(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return str(candidate.relative_to(_REPO_ROOT))
    except ValueError:
        return str(candidate)


def _evidence_sort_key(witness: dict[str, Any]) -> tuple[str, str, int, str, str]:
    source_line = witness.get("source_line")
    return (
        str(witness.get("evidence_type", "")),
        str(witness.get("source_path", "")),
        source_line if isinstance(source_line, int) else 0,
        str(witness.get("symbol", "")),
        str(witness.get("raw_path_literal", "")),
    )


def _iter_executable_string_constants(tree: ast.Module) -> Iterator[tuple[str, int]]:
    """Yield ``(value, lineno)`` for whole string constants outside docstrings.

    Comments never reach the AST, and module/class/function docstrings are
    excluded positionally. Constants nested inside f-strings are excluded
    structurally: an ``ast.JoinedStr`` fragment such as ``"GET /api/debates/"``
    from ``f"GET /api/debates/{id}"`` is a truncated piece of a computed
    string, never an exact witness.
    """
    excluded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                excluded.add(id(body[0].value))
        elif isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.Constant):
                    excluded.add(id(part))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in excluded
        ):
            yield node.value, node.lineno


def _active_handler_source_files() -> frozenset[str]:
    """Repo-relative source files hosting the ACTIVE-TIER handler classes.

    The set covers each active class's defining file plus every method's
    ``__code__.co_filename`` (re-exported classes such as ``WebhookHandler``
    define methods in a different file than the class's import location).
    Literal witnesses outside this set describe code the server never
    dispatches — inactive tiers or unregistered modules — and therefore prove
    nothing about serving (review round 3 on this packet).
    """
    files: set[str] = set()

    def _add(source_file: str) -> None:
        # Key the census on BOTH the raw and resolved spellings so the
        # membership check in get_handler_source_literal_operations (which
        # resolves the scanned file) agrees under symlinked checkouts.
        files.add(_relative_source_path(source_file))
        files.add(_relative_source_path(str(Path(source_file).resolve())))

    for handler_class in _iter_registry_handler_classes():
        # Walk the full MRO: inherited base/mixin methods dispatch for this
        # handler too, so their defining modules are served surface (review
        # round 4: censusing only the class's own __dict__ excluded base
        # modules and dropped their literals).
        for klass in getattr(handler_class, "__mro__", (handler_class,)):
            if klass is object:
                continue
            try:
                source_file = inspect.getsourcefile(klass)
            except (TypeError, OSError):
                source_file = None
            if source_file:
                _add(source_file)
            for value in vars(klass).values():
                function = getattr(value, "__func__", value)
                code = getattr(function, "__code__", None)
                if code is not None:
                    _add(code.co_filename)
    return frozenset(files)


def get_handler_source_literal_operations(
    server_root: Path | None = None,
    allowed_files: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Exact whole-string ``METHOD /api/...`` literals in executable server source.

    A witness must be an entire executable string constant of exactly
    ``METHOD /api/...`` (the dispatch-key form). Comments, docstrings, and
    f-string fragments are structurally excluded, and prose that merely
    *contains* an operation substring (log lines, error messages, usage text)
    never counts (review rounds 1-2 on this packet: raw text scanning let
    documentation prose and truncated f-string fragments fabricate served
    operations). When ``allowed_files`` is provided, only literals in those
    repo-relative files count: the method-aware plane passes the active-tier
    handler file census so literals in unserved modules are never evidence
    (review round 3).
    """
    root = server_root if server_root is not None else _REPO_ROOT / "aragora" / "server"
    witnesses: list[dict[str, Any]] = []
    for source_file in sorted(root.rglob("*.py")):
        if allowed_files is not None and (
            _relative_source_path(source_file) not in allowed_files
            and _relative_source_path(str(source_file.resolve())) not in allowed_files
        ):
            continue
        try:
            tree = _parse_python_source(source_file)
        except RouteRegistrationScanError as exc:
            # An unparseable server file is unproven served surface; skipping
            # it would fail open in a plane that fails closed everywhere else.
            raise MethodAwareError(f"cannot parse server source {source_file}: {exc}") from exc
        for value, lineno in _iter_executable_string_constants(tree):
            match = _HANDLER_METHOD_PATH_LITERAL_RE.fullmatch(value.strip())
            if match is None:
                continue
            path = _clean_literal_path(match.group(2))
            if path is None:
                continue
            witnesses.append(
                {
                    "evidence_type": "handler_source_method_path_literal",
                    "method": match.group(1),
                    "raw_path_literal": path,
                    "source_path": _relative_source_path(source_file),
                    "source_line": lineno,
                    "match_text": match.group(0),
                }
            )
    return witnesses


def _iter_registry_handler_classes() -> Iterator[Any]:
    """Resolve the ACTIVE-TIER registry census, failing CLOSED on resolution.

    Serving truth mirrors the server exactly: startup filters
    ``HANDLER_REGISTRY`` through ``filter_registry_by_tier(get_active_tiers())``
    before resolving anything, so handlers outside the active tiers are
    genuinely not served and are excluded here for the same reason (review
    round 2: censusing inactive tiers made validation depend on unserved
    handlers). Within the active set, a handler that cannot resolve is
    unproven served surface: silently dropping it would retire
    served-but-undeclared drift and manufacture unserved-spec orphans (review
    round 1), so the method-aware plane refuses to publish a partial census.
    The legacy path-level plane keeps its historical skip behavior.
    """
    try:
        # import_module resolves through sys.modules, honoring test isolation;
        # ``import a.b.c as name`` would bypass a monkeypatched module entry.
        handler_registry = importlib.import_module("aragora.server.handler_registry")
    except ImportError as exc:  # pragma: no cover - environment failure
        raise MethodAwareError(f"cannot import handler_registry: {exc}") from exc

    registry = handler_registry.HANDLER_REGISTRY
    filter_by_tier = getattr(handler_registry, "filter_registry_by_tier", None)
    get_active_tiers = getattr(handler_registry, "get_active_tiers", None)
    if not callable(filter_by_tier) or not callable(get_active_tiers):
        raise MethodAwareError(
            "handler_registry no longer exposes filter_registry_by_tier/"
            "get_active_tiers; the served census cannot mirror server startup"
        )
    active_registry = filter_by_tier(registry, get_active_tiers())

    failures: list[str] = []
    resolved_classes: list[Any] = []
    for attr_name, handler_ref in active_registry:
        handler_class = handler_ref
        resolve = getattr(handler_ref, "resolve", None)
        # Only non-class refs are deferred-import handles. A handler CLASS
        # that happens to define a `resolve` instance method must not be
        # invoked unbound (it would raise TypeError and fail the plane).
        if not inspect.isclass(handler_ref) and callable(resolve):
            try:
                handler_class = resolve()
            except Exception as exc:  # noqa: BLE001 - collected, then failed closed
                failures.append(f"{attr_name}: {type(exc).__name__}: {exc}")
                continue
        if handler_class is None:
            failures.append(f"{attr_name}: resolved to None")
            continue
        resolved_classes.append(handler_class)
    if failures:
        raise MethodAwareError(
            "handler registry resolution failed for "
            f"{len(failures)} active-tier entries; refusing a partial served census: "
            + "; ".join(failures[:5])
        )
    yield from resolved_classes


def _handler_source_path(handler_class: Any) -> str:
    try:
        return _relative_source_path(inspect.getsourcefile(handler_class) or "")
    except (TypeError, OSError):
        return str(getattr(handler_class, "__module__", handler_class))


def get_handler_metadata_operations() -> tuple[list[dict[str, Any]], set[str]]:
    """Method witnesses from handler class metadata, plus path-only routes.

    Method evidence comes only from explicit metadata: method-specific
    ``<METHOD>_ROUTES`` attributes, explicit ``("METHOD", path)`` /
    ``"METHOD /api/..."`` ROUTES entries, and ``@api_endpoint`` decorator
    metadata. A bare path in ``ROUTES`` proves the path is served but never
    proves any method: those paths are returned separately as method-unresolved
    drift (handler path presence is not method evidence).
    """
    witnesses: list[dict[str, Any]] = []
    path_only: set[str] = set()

    def _witness(handler_class: Any, symbol: str, method: str, raw_path: str) -> None:
        path = _clean_literal_path(raw_path)
        if path is None:
            # Wildcard/prefix declarations are never operation evidence; the
            # path stays visible as method-unresolved drift instead.
            if raw_path.startswith("/api/"):
                path_only.add(raw_path)
            return
        witnesses.append(
            {
                "evidence_type": "handler_route_metadata",
                "method": method,
                "raw_path_literal": path,
                "source_path": _handler_source_path(handler_class),
                "symbol": symbol,
            }
        )

    def _witness_route_map(handler_class: Any, symbol: str, route_map: dict[Any, Any]) -> None:
        """Dict metadata: ``{path: [methods]}`` or ``{"METHOD path": handler}``.

        Both live shapes carry explicit methods and are served method
        evidence. Any other value shape is unproven served surface and fails
        closed rather than being silently dropped.
        """
        for key, value in route_map.items():
            if not isinstance(key, str):
                raise MethodAwareError(
                    f"{symbol} maps a non-string key {key!r}; route metadata must be explicit"
                )
            head, _, tail = key.partition(" ")
            if head in _RUNTIME_METHODS and tail.startswith("/api/"):
                # "METHOD /api/..." dispatch key mapping to a handler callable.
                _witness(handler_class, symbol, head, tail)
                continue
            if isinstance(value, (list, tuple)):
                for method in value:
                    if isinstance(method, str) and method in _RUNTIME_METHODS:
                        _witness(handler_class, symbol, method, key)
                    else:
                        # Alien or lowercase verbs are never accepted; the path
                        # is retained as method-unresolved drift, not guessed.
                        path_only.add(key)
                continue
            raise MethodAwareError(
                f"{symbol}[{key!r}] must map to an explicit method list or use "
                f"a 'METHOD /api/...' dispatch key, got {type(value).__name__}"
            )

    for handler_class in _iter_registry_handler_classes():
        class_name = getattr(handler_class, "__name__", repr(handler_class))

        plain_routes = getattr(handler_class, "ROUTES", None)
        if isinstance(plain_routes, dict):
            # Dict-shaped metadata ({path: [methods]}) is explicit method
            # evidence (review round 2: real optional handlers such as
            # expenses/invoices/rlm declare this shape and were silently
            # omitted from the served census).
            _witness_route_map(handler_class, f"{class_name}.ROUTES", plain_routes)
        elif isinstance(plain_routes, (list, tuple)):
            for entry in plain_routes:
                if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                    method, path = entry[0], entry[1]
                    if not isinstance(path, str):
                        raise MethodAwareError(
                            f"{class_name}.ROUTES pair {entry!r} carries a "
                            "non-string path; route metadata must be explicit"
                        )
                    if isinstance(method, str) and method in _RUNTIME_METHODS:
                        _witness(handler_class, f"{class_name}.ROUTES", method, path)
                    else:
                        # Alien or lowercase verbs are never accepted; the path
                        # is retained as method-unresolved drift, not guessed.
                        path_only.add(path)
                    continue
                if not isinstance(entry, str):
                    # An unrecognized metadata shape is unproven served
                    # surface; dropping it silently would erase operations.
                    raise MethodAwareError(
                        f"{class_name}.ROUTES entry {entry!r} has an "
                        "unrecognized shape; route metadata must be explicit"
                    )
                head, _, tail = entry.partition(" ")
                if head in _RUNTIME_METHODS and tail.startswith("/api/"):
                    _witness(handler_class, f"{class_name}.ROUTES", head, tail)
                elif tail.startswith("/api/"):
                    # Unacceptable method token before a real path: incomplete
                    # metadata retained as method-unresolved drift.
                    path_only.add(tail)
                else:
                    path_only.add(entry)
        elif plain_routes is not None:
            raise MethodAwareError(
                f"{class_name}.ROUTES has unrecognized container type "
                f"{type(plain_routes).__name__}; route metadata must be explicit"
            )

        dynamic_routes = getattr(handler_class, "DYNAMIC_ROUTES", None)
        if isinstance(dynamic_routes, dict):
            # Parametrized dispatch metadata is served surface with explicit
            # methods, exactly like dict-shaped ROUTES.
            _witness_route_map(handler_class, f"{class_name}.DYNAMIC_ROUTES", dynamic_routes)
        elif isinstance(dynamic_routes, (list, tuple)):
            # Bare parametrized paths prove served surface but no method:
            # method-unresolved drift, never guessed.
            for entry in dynamic_routes:
                if isinstance(entry, str) and entry.startswith("/api/"):
                    path_only.add(entry)
        elif dynamic_routes is not None:
            raise MethodAwareError(
                f"{class_name}.DYNAMIC_ROUTES has unrecognized container type "
                f"{type(dynamic_routes).__name__}; route metadata must be explicit"
            )

        for method_attr in _METHOD_ROUTES_ATTRS:
            method_routes = getattr(handler_class, method_attr, None)
            if not isinstance(method_routes, (list, tuple)):
                continue
            method = method_attr.removesuffix("_ROUTES")
            for entry in method_routes:
                if isinstance(entry, str):
                    _witness(handler_class, f"{class_name}.{method_attr}", method, entry)

        for attr_name in dir(handler_class):
            try:
                attr = getattr(handler_class, attr_name)
            except Exception:  # noqa: BLE001 - discovery stays best-effort
                continue
            endpoint = getattr(attr, "_openapi", None)
            path = getattr(endpoint, "path", None)
            if not (isinstance(path, str) and path):
                continue
            method = getattr(endpoint, "method", None)
            if isinstance(method, str) and method in _RUNTIME_METHODS:
                _witness(
                    handler_class,
                    f"{class_name}.{attr_name}",
                    method,
                    path,
                )
            else:
                # Missing, lowercase, or alien decorator methods are incomplete
                # metadata: retained as method-unresolved drift, never guessed.
                path_only.add(path)

    return witnesses, path_only


def get_openapi_operation_witnesses(spec_path: str) -> list[dict[str, Any]]:
    """Exact lowercase OpenAPI operation keys on ``/api/``-rooted paths.

    Non-method path-item keys (``parameters``, ``summary``, ``x-*`` extensions,
    uppercase aliases, ``connect``, or any other key outside the exact OpenAPI
    method-key set) are ignored and never become operations. A missing PRIMARY
    spec fails closed: an unreadable ``--spec`` path must never masquerade as
    an empty declared surface (review round 5). Only the optional
    ``*_generated`` sibling snapshot may be legitimately absent.
    """
    primary = Path(spec_path)
    if not primary.exists():
        raise MethodAwareError(f"OpenAPI spec not found: {primary}")
    witnesses: list[dict[str, Any]] = []
    for candidate in _iter_spec_paths(spec_path):
        if not candidate.exists():
            continue
        try:
            spec = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MethodAwareError(f"cannot load OpenAPI spec {candidate}: {exc}") from exc
        autogenerated = "_generated" in candidate.stem
        paths = spec.get("paths", {})
        if not isinstance(paths, dict):
            continue
        for raw_path, path_item in paths.items():
            if not (isinstance(raw_path, str) and raw_path.startswith("/api/")):
                continue
            if not isinstance(path_item, dict):
                continue
            for key in path_item:
                if key not in _OPENAPI_OPERATION_KEYS:
                    continue
                witnesses.append(
                    {
                        "evidence_type": "openapi_operation_key",
                        "method": key.upper(),
                        "operation_key": key,
                        "raw_path_literal": raw_path,
                        "source_path": _relative_source_path(candidate),
                        "autogenerated": autogenerated,
                    }
                )
    return witnesses


def _build_operations(
    witnesses: Iterable[dict[str, Any]],
    ref: str,
    *,
    method_universe: frozenset[str],
) -> dict[str, dict[str, Any]]:
    """Merge witnesses into ``operation_id -> operation`` records."""
    operations: dict[str, dict[str, Any]] = {}
    for witness in witnesses:
        method = witness.get("method")
        if not (isinstance(method, str) and method in method_universe):
            raise MethodAwareError(f"witness carries unacceptable method {method!r}: {witness!r}")
        raw_path = witness["raw_path_literal"]
        path = normalize_sdk_path(raw_path)
        if not path.startswith("/api/"):
            continue
        operation_id = _operation_id(method, path)
        record = operations.setdefault(
            operation_id,
            {
                "operation_id": operation_id,
                "method": method,
                "path": path,
                "source_sha": ref,
                "evidence": [],
            },
        )
        record["evidence"].append(dict(witness))
    for record in operations.values():
        record["evidence"].sort(key=_evidence_sort_key)
    return operations


def _admit_spec_operations(
    spec_all: dict[str, dict[str, Any]],
    served: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Admit spec operations, gating autogenerated-only placeholders.

    Autogenerated snapshot operations count only when exact served-operation
    evidence independently supports the method/path; a placeholder alone can
    neither declare nor retire anything.
    """
    spec: dict[str, dict[str, Any]] = {}
    for operation_id, operation in spec_all.items():
        if any(not witness.get("autogenerated") for witness in operation["evidence"]):
            spec[operation_id] = operation
        elif operation_id in served:
            spec[operation_id] = operation
    return spec


def _v1_alias(path: str) -> str:
    if path.startswith("/api/") and not path.startswith("/api/v"):
        return path.replace("/api/", "/api/v1/", 1)
    return path


def _classify_exposure(operation: dict[str, Any], prefixes: tuple[str, ...]) -> str:
    """Resolve public/internal exposure from exact witness literals.

    The canonical path and every witness literal are classified against the
    internal-route policy in the policy's own ``/api/v1/`` space (unversioned
    literals alias to v1 exactly as the path-level plane normalizes them, so a
    version-form difference alone never manufactures ambiguity). Unanimous
    verdicts resolve exposure; genuinely disagreeing witnesses — e.g. a
    non-v1-versioned serving of a v1-internal family — leave the operation
    unresolved governed debt rather than guessing.
    """
    probes = {_v1_alias(operation["path"])}
    for witness in operation["evidence"]:
        raw = witness.get("raw_path_literal")
        if isinstance(raw, str):
            stripped = raw.split("?", 1)[0].rstrip("/") or raw
            probes.add(_v1_alias(stripped))
    verdicts = {is_internal_route(probe, prefixes) for probe in probes}
    if verdicts == {True}:
        return "internal"
    if verdicts == {False}:
        return "public"
    return "unresolved"


def _sorted_operations(operations: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(operations, key=lambda op: op["operation_id"])
    ids = [op["operation_id"] for op in ordered]
    if len(ids) != len(set(ids)):
        raise MethodAwareError("duplicate operation IDs in a collection")
    return ordered


def _assert_disjoint(collections: dict[str, list[dict[str, Any]]]) -> None:
    names = sorted(collections)
    for index, left in enumerate(names):
        left_ids = {op["operation_id"] for op in collections[left]}
        for right in names[index + 1 :]:
            overlap = left_ids & {op["operation_id"] for op in collections[right]}
            if overlap:
                raise MethodAwareError(
                    f"{left} and {right} must be disjoint; shared: {sorted(overlap)[:5]}"
                )


def build_route_set_algebra(
    served: dict[str, dict[str, Any]],
    spec: dict[str, dict[str, Any]],
    internal_prefixes: tuple[str, ...],
) -> dict[str, Any]:
    """Compute the complete exclusive/exhaustive VAL-CDG-011 operation algebra."""
    # Never mutate caller inputs: annotate copies so repeated calls and shared
    # dicts stay stable.
    served = {operation_id: dict(op) for operation_id, op in served.items()}
    spec = {operation_id: dict(op) for operation_id, op in spec.items()}
    served_ids = set(served)
    spec_ids = set(spec)

    for operation_id, operation in spec.items():
        method = operation["method"]
        if method not in set(OPENAPI_METHOD_SET):
            raise MethodAwareError(
                f"spec operation {operation_id} carries non-OpenAPI method {method}"
            )

    served_but_undeclared_ids = served_ids - spec_ids
    declared_and_served_ids = served_ids & spec_ids
    unserved_spec_ids = spec_ids - served_ids

    for operation_id, operation in served.items():
        if operation["method"] == "CONNECT":
            # Standard OpenAPI cannot declare CONNECT: it must remain explicit
            # served-but-undeclared, non-representable debt. Declared CONNECT
            # would mean the spec-side extractor accepted an alien key.
            if operation_id not in served_but_undeclared_ids:
                raise MethodAwareError(
                    f"served CONNECT operation {operation_id} may never be declared_and_served"
                )
            operation["non_representable_in_standard_openapi"] = True

    exposure: dict[str, list[dict[str, Any]]] = {
        "public": [],
        "internal": [],
        "unresolved": [],
    }
    for operation_id in sorted(served_ids):
        operation = served[operation_id]
        verdict = _classify_exposure(operation, internal_prefixes)
        entry = dict(operation)
        entry["exposure"] = verdict
        exposure[verdict].append(entry)

    def _merge_declared(operation_id: str) -> dict[str, Any]:
        merged = dict(served[operation_id])
        merged["evidence"] = sorted(
            merged["evidence"] + spec[operation_id]["evidence"], key=_evidence_sort_key
        )
        return merged

    collections = {
        "served_operations": _sorted_operations(served.values()),
        "spec_operations": _sorted_operations(spec.values()),
        "public_operations": _sorted_operations(exposure["public"]),
        "internal_operations": _sorted_operations(exposure["internal"]),
        "unresolved_exposure": _sorted_operations(exposure["unresolved"]),
        "served_but_undeclared": _sorted_operations(
            served[operation_id] for operation_id in served_but_undeclared_ids
        ),
        "declared_and_served": _sorted_operations(
            _merge_declared(operation_id) for operation_id in declared_and_served_ids
        ),
        "unserved_spec": _sorted_operations(
            spec[operation_id] for operation_id in unserved_spec_ids
        ),
    }

    # Independently re-prove every union, intersection, difference, and
    # disjointness equation from the reconstructed ID sets (never counts).
    def _ids(name: str) -> set[str]:
        return {op["operation_id"] for op in collections[name]}

    _assert_disjoint(
        {
            name: collections[name]
            for name in ("public_operations", "internal_operations", "unresolved_exposure")
        }
    )
    _assert_disjoint(
        {name: collections[name] for name in ("served_but_undeclared", "declared_and_served")}
    )
    _assert_disjoint({name: collections[name] for name in ("unserved_spec", "declared_and_served")})
    checks = (
        (
            "served = public ⊎ internal ⊎ unresolved",
            _ids("served_operations"),
            _ids("public_operations") | _ids("internal_operations") | _ids("unresolved_exposure"),
        ),
        (
            "served_but_undeclared = served - spec",
            _ids("served_but_undeclared"),
            _ids("served_operations") - _ids("spec_operations"),
        ),
        (
            "declared_and_served = served ∩ spec",
            _ids("declared_and_served"),
            _ids("served_operations") & _ids("spec_operations"),
        ),
        (
            "unserved_spec = spec - served",
            _ids("unserved_spec"),
            _ids("spec_operations") - _ids("served_operations"),
        ),
        (
            "served = served_but_undeclared ⊎ declared_and_served",
            _ids("served_operations"),
            _ids("served_but_undeclared") | _ids("declared_and_served"),
        ),
        (
            "spec = unserved_spec ⊎ declared_and_served",
            _ids("spec_operations"),
            _ids("unserved_spec") | _ids("declared_and_served"),
        ),
    )
    for label, left, right in checks:
        if left != right:
            raise MethodAwareError(
                f"route-set algebra violated: {label}; "
                f"only-left={sorted(left - right)[:5]} only-right={sorted(right - left)[:5]}"
            )
    return collections


def load_operation_projection(inventory_path: str | Path | None = None) -> dict[str, Any]:
    """Load, authenticate, and prune the ratified operation projection.

    Every layer is independently reconstructed from the artifact bytes: the
    immutable 655 original-ID set, per-record canonical digests, the projection
    digest set, the 57 route/parity memberships, and the exact 68 method edges
    with their 48/8/1 distribution and the nine ratified multi-edge originals.
    Count-only reconciliation is rejected: identities, digests, and edge
    operation IDs are compared, never bare counts.
    """
    path = Path(inventory_path) if inventory_path is not None else _DEFAULT_INVENTORY_PATH
    try:
        inventory = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MethodAwareError(f"cannot load contract-drift inventory {path}: {exc}") from exc

    try:
        cohort = inventory["accepted_authority"]["canonical_artifacts"]["original_cohort"]
        originals = cohort["original_records"]
        projection = cohort["operation_projection"]
        records = projection["records"]
    except (KeyError, TypeError) as exc:
        raise MethodAwareError(f"inventory {path} lacks the canonical original cohort: {exc}")

    if len(originals) != ORIGINAL_COHORT_TOTAL:
        raise MethodAwareError(
            f"original cohort must contain exactly {ORIGINAL_COHORT_TOTAL} records, "
            f"found {len(originals)}"
        )

    def _recomputed_id(record: dict[str, Any]) -> str:
        payload = _canonical_json_bytes(
            {
                "category": record["category"],
                "exact_historical_literal_record": record["exact_historical_literal_record"],
                "schema": "cdg-original-record-id-v1",
            }
        )
        return f"cdg1:{_sha256_hex(payload)}"

    original_ids: set[str] = set()
    original_by_id: dict[str, dict[str, Any]] = {}
    for record in originals:
        recomputed = _recomputed_id(record)
        if recomputed != record["original_record_id"]:
            raise MethodAwareError(
                f"original record ID mismatch for {record['original_record_id']}"
            )
        original_ids.add(recomputed)
        original_by_id[recomputed] = record
    if len(original_ids) != ORIGINAL_COHORT_TOTAL:
        raise MethodAwareError("duplicate original_record_id in cohort")

    id_set_digest = _sha256_hex(
        _canonical_json_bytes(
            {
                "original_record_ids": sorted(original_ids),
                "schema": "cdg-original-record-id-set-v1",
            }
        )
    )
    if id_set_digest != ORIGINAL_RECORD_ID_SET_SHA256:
        raise MethodAwareError(
            "immutable original-record-ID set digest mismatch: "
            f"{id_set_digest} != {ORIGINAL_RECORD_ID_SET_SHA256}"
        )

    if len(records) != ORIGINAL_COHORT_TOTAL:
        raise MethodAwareError(
            f"operation projection must contain exactly {ORIGINAL_COHORT_TOTAL} membership "
            f"records, found {len(records)}"
        )
    digests: list[str] = []
    projection_ids: set[str] = set()
    per_category_ids: dict[str, set[str]] = {}
    for record in records:
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        digest = _sha256_hex(_canonical_json_bytes(body))
        if digest != record["record_sha256"]:
            raise MethodAwareError(
                f"projection record digest mismatch for {record['original_record_id']}"
            )
        digests.append(digest)
        if _recomputed_id(record) != record["original_record_id"]:
            raise MethodAwareError(
                f"projection membership identity rewrite for {record['original_record_id']}"
            )
        if record["original_record_id"] in projection_ids:
            raise MethodAwareError(
                f"duplicate projection membership for {record['original_record_id']}"
            )
        projection_ids.add(record["original_record_id"])
        per_category_ids.setdefault(record["category"], set()).add(record["original_record_id"])

    digest_set = _sha256_hex(
        _canonical_json_bytes(
            {
                "record_sha256_values": sorted(digests),
                "schema": "cdg-operation-projection-record-digest-set-v1",
            }
        )
    )
    if digest_set != PROJECTION_RECORD_DIGEST_SET_SHA256:
        raise MethodAwareError(
            "projection record-digest-set mismatch: "
            f"{digest_set} != {PROJECTION_RECORD_DIGEST_SET_SHA256}"
        )

    if projection_ids != original_ids:
        raise MethodAwareError(
            "projection membership IDs must equal the immutable original cohort IDs"
        )
    for category, ids in per_category_ids.items():
        expected = {
            record["original_record_id"] for record in originals if record["category"] == category
        }
        if ids != expected:
            raise MethodAwareError(f"per-category original-ID set changed for {category}")

    route_records = [record for record in records if record["category"] in ROUTE_CATEGORY_SELECTION]
    route_category_counts = dict.fromkeys(ROUTE_CATEGORY_SELECTION, 0)
    for record in route_records:
        route_category_counts[record["category"]] += 1
    if route_category_counts != ROUTE_CATEGORY_COUNTS:
        raise MethodAwareError(
            f"route category counts {route_category_counts} != ratified {ROUTE_CATEGORY_COUNTS}"
        )
    if len(route_records) != ROUTE_PARITY_RECORD_TOTAL:
        raise MethodAwareError(
            f"route/parity membership count {len(route_records)} != {ROUTE_PARITY_RECORD_TOTAL}"
        )

    edge_total = 0
    multi_edge = 0
    distribution: dict[int, int] = {}
    seen_paths: set[str] = set()
    for record in sorted(route_records, key=lambda r: r["original_record_id"]):
        original = original_by_id[record["original_record_id"]]
        if original.get("method") is not None:
            raise MethodAwareError(
                f"path-level original {record['original_record_id']} must keep method=null"
            )
        expected_language = _ROUTE_CATEGORY_SDK_LANGUAGE[record["category"]]
        if record.get("sdk_language") != expected_language:
            raise MethodAwareError(
                f"category-derived sdk_language mismatch for {record['original_record_id']}"
            )
        if record.get("projection_status") != "resolved":
            raise MethodAwareError(
                f"unresolved projection membership {record['original_record_id']}"
            )
        edges = record.get("operation_edges")
        if not isinstance(edges, list) or not edges:
            raise MethodAwareError(f"zero-edge route membership {record['original_record_id']}")
        edge_ids: set[str] = set()
        edge_paths: set[str] = set()
        for edge in edges:
            method = edge.get("method")
            if not (isinstance(method, str) and method in _RUNTIME_METHODS):
                raise MethodAwareError(
                    f"projected edge carries invalid method {method!r} on "
                    f"{record['original_record_id']}"
                )
            normalized_path = edge.get("normalized_path")
            if not (isinstance(normalized_path, str) and normalized_path.startswith("/api/")):
                raise MethodAwareError(
                    f"projected edge lacks normalized path on {record['original_record_id']}"
                )
            operation_id = _operation_id(method, normalized_path)
            if edge.get("normalized_operation") != operation_id:
                raise MethodAwareError(
                    f"edge operation ID mismatch on {record['original_record_id']}"
                )
            if operation_id in edge_ids:
                raise MethodAwareError(
                    f"duplicate edge {operation_id} on {record['original_record_id']}"
                )
            edge_ids.add(operation_id)
            edge_paths.add(normalized_path)
            evidence = edge.get("evidence")
            if not (isinstance(evidence, list) and evidence):
                raise MethodAwareError(
                    f"edge {operation_id} lacks witness provenance on "
                    f"{record['original_record_id']}"
                )
            if not any(witness.get("method") == method for witness in evidence):
                raise MethodAwareError(
                    f"edge {operation_id} lacks an exact-method witness on "
                    f"{record['original_record_id']}"
                )
        if len(edge_paths) != 1:
            raise MethodAwareError(
                f"route membership {record['original_record_id']} fans out across paths "
                f"{sorted(edge_paths)}"
            )
        path_key = next(iter(edge_paths))
        if path_key in seen_paths:
            raise MethodAwareError(
                f"exclusive source-operation sets violated: path {path_key} appears in "
                "two route memberships"
            )
        seen_paths.add(path_key)
        edge_total += len(edges)
        distribution[len(edges)] = distribution.get(len(edges), 0) + 1
        if len(edges) > 1:
            multi_edge += 1
            pinned = MULTI_EDGE_OPERATION_PINS.get(record["original_record_id"])
            if pinned is None or tuple(sorted(edge_ids)) != pinned:
                raise MethodAwareError(
                    f"multi-edge original {record['original_record_id']} deviates from the "
                    f"ratified edges {pinned}"
                )

    if edge_total != ROUTE_EDGE_TOTAL:
        raise MethodAwareError(f"route edge total {edge_total} != {ROUTE_EDGE_TOTAL}")
    if multi_edge != ROUTE_MULTI_EDGE_ORIGINALS:
        raise MethodAwareError(
            f"multi-edge original count {multi_edge} != {ROUTE_MULTI_EDGE_ORIGINALS}"
        )
    if distribution != ROUTE_EDGE_DISTRIBUTION:
        raise MethodAwareError(
            f"edge distribution {distribution} != ratified {ROUTE_EDGE_DISTRIBUTION}"
        )
    if max(distribution) != ROUTE_MAX_EDGES:
        raise MethodAwareError(f"maximum edges {max(distribution)} != {ROUTE_MAX_EDGES}")

    schema_descriptor = {
        "one_to_many_rule": projection["one_to_many_rule"],
        "record_digest_encoding": projection["record_digest_encoding"],
        "schema": projection["schema"],
        "schema_version": OPERATION_PROJECTION_SCHEMA_VERSION,
    }
    if projection["schema"] != OPERATION_PROJECTION_SCHEMA:
        raise MethodAwareError(
            f"projection schema {projection['schema']!r} != {OPERATION_PROJECTION_SCHEMA!r}"
        )

    return {
        "schema": OPERATION_PROJECTION_SCHEMA,
        "schema_version": OPERATION_PROJECTION_SCHEMA_VERSION,
        "schema_sha256": _sha256_hex(_canonical_json_bytes(schema_descriptor)),
        "original_record_id_set_sha256": id_set_digest,
        "record_digest_set_sha256": digest_set,
        "original_cohort_total": len(originals),
        "route_records": sorted(route_records, key=lambda r: r["original_record_id"]),
        "route_category_counts": route_category_counts,
        "route_edge_total": edge_total,
        "route_multi_edge_originals": multi_edge,
        "route_max_edges": max(distribution),
        "route_edge_distribution": {str(k): v for k, v in sorted(distribution.items())},
    }


def validate_method_aware_plane(
    spec_path: str,
    ref: str,
    internal_prefixes_path: str | None = None,
    inventory_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute the complete VAL-CDG-011 method-aware operation plane."""
    ref = _require_exact_ref(ref)
    internal_prefixes = load_internal_prefixes(internal_prefixes_path)

    wired_witnesses: list[dict[str, Any]] = []
    wildcard_registrations: set[str] = set()
    for operation in get_wired_function_operations(extended=True):
        path = _clean_literal_path(operation["path"])
        if path is None:
            # Wildcard registrations are prefix claims; they never witness a
            # method-specific operation and stay method-unresolved drift.
            wildcard_registrations.add(operation["path"])
            continue
        wired_witnesses.append(
            {
                "evidence_type": "wired_router_registration",
                "method": operation["method"],
                "raw_path_literal": path,
                "source_path": operation["source_path"],
                "symbol": operation["symbol"],
                "source_line": operation["line"],
            }
        )
    metadata_witnesses, path_only_routes = get_handler_metadata_operations()
    literal_witnesses = get_handler_source_literal_operations(
        allowed_files=_active_handler_source_files()
    )

    served = _build_operations(
        [*wired_witnesses, *metadata_witnesses, *literal_witnesses],
        ref,
        method_universe=_RUNTIME_METHODS,
    )

    spec_witnesses = get_openapi_operation_witnesses(spec_path)
    spec_all = _build_operations(spec_witnesses, ref, method_universe=frozenset(OPENAPI_METHOD_SET))
    spec = _admit_spec_operations(spec_all, served)

    collections = build_route_set_algebra(served, spec, internal_prefixes)

    served_paths = {operation["path"] for operation in served.values()}
    # Method-unresolved debt comes only from the ACTIVE-TIER census
    # (path_only_routes) plus wildcard wired registrations. The legacy
    # unfiltered get_handler_routes() is deliberately NOT merged: it scans
    # the raw registry without the tier filter, which would report
    # inactive-tier handlers as live governed debt (review round 4).
    method_unresolved = {
        normalize_sdk_path(route)
        for route in (*path_only_routes, *wildcard_registrations)
        if isinstance(route, str) and route.startswith("/api/")
    }
    method_unresolved -= served_paths

    projection = load_operation_projection(inventory_path)

    served_ids = {operation["operation_id"] for operation in collections["served_operations"]}
    spec_ids = {operation["operation_id"] for operation in collections["spec_operations"]}
    reconciliation = {
        "in_served_and_spec": 0,
        "served_only": 0,
        "spec_only": 0,
        "historical_only": 0,
    }
    for record in projection["route_records"]:
        for edge in record["operation_edges"]:
            live_id = _operation_id(edge["method"], normalize_sdk_path(edge["normalized_path"]))
            in_served = live_id in served_ids
            in_spec = live_id in spec_ids
            if in_served and in_spec:
                reconciliation["in_served_and_spec"] += 1
            elif in_served:
                reconciliation["served_only"] += 1
            elif in_spec:
                reconciliation["spec_only"] += 1
            else:
                reconciliation["historical_only"] += 1

    plane: dict[str, Any] = {
        "ref": ref,
        "runtime_method_set": sorted(RUNTIME_METHOD_SET),
        "openapi_method_set": sorted(OPENAPI_METHOD_SET),
        "path_normalization": path_normalization_binding(),
        "method_unresolved_paths": sorted(method_unresolved),
        "method_unresolved_paths_count": len(method_unresolved),
        "operation_projection_schema": projection["schema"],
        "operation_projection_schema_version": projection["schema_version"],
        "operation_projection_schema_sha256": projection["schema_sha256"],
        "original_record_id_set_sha256": projection["original_record_id_set_sha256"],
        "operation_projection_record_digest_set_sha256": projection["record_digest_set_sha256"],
        "original_cohort_total": projection["original_cohort_total"],
        "route_category_keys": sorted(ROUTE_CATEGORY_SELECTION),
        "route_category_counts": projection["route_category_counts"],
        "route_category_selection": ROUTE_CATEGORY_SELECTION,
        "operation_projection": projection["route_records"],
        "operation_projection_route_edge_total": projection["route_edge_total"],
        "operation_projection_multi_edge_originals": projection["route_multi_edge_originals"],
        "operation_projection_max_edges": projection["route_max_edges"],
        "operation_projection_edge_distribution": projection["route_edge_distribution"],
        "operation_projection_live_reconciliation": reconciliation,
    }
    for name, operations in collections.items():
        plane[name] = operations
        plane[f"{name}_count"] = len(operations)
    return plane


def load_wired_routes_for_validation() -> set[str]:
    """Load literal wired routes with exact API-version semantics."""
    try:
        return {
            normalize_route(route, normalize_version=False) for route in get_wired_function_routes()
        }
    except RouteRegistrationScanError as exc:
        print(f"Error: {exc}. Refusing to validate partial route wiring.", file=sys.stderr)
        sys.exit(1)


def filter_wired_orphans(
    candidates: set[str], wired_routes: set[str] | None = None
) -> tuple[set[str], set[str]]:
    """Split orphan candidates by literal evidence from wired route functions."""
    if wired_routes is None:
        wired_routes = load_wired_routes_for_validation()

    served = candidates & wired_routes
    if served:
        print(
            f"Spec-orphan suppressions via wired route registrations ({len(served)}):",
            file=sys.stderr,
        )
        for path in sorted(served):
            print(f"  - {path}", file=sys.stderr)
    return candidates - served, served


def validate_coverage(
    spec_path: str,
    fail_on_missing: bool = False,
    output_json: bool = False,
    baseline_path: str | None = None,
    include_internal: bool = False,
    internal_prefixes_path: str | None = None,
) -> dict[str, Any]:
    """Compare handler routes against OpenAPI spec.

    Args:
        spec_path: Path to OpenAPI spec.
        fail_on_missing: Whether to exit with error if routes missing.
        output_json: Whether to output as JSON.

    Returns:
        Validation results dict.
    """
    handler_routes = get_handler_routes()
    openapi_routes = get_openapi_routes(spec_path)
    wired_routes = load_wired_routes_for_validation()

    # Normalize routes for comparison
    normalized_handler = {normalize_route(r) for r in handler_routes}
    normalized_openapi = {normalize_route(r) for r in openapi_routes}
    normalized_openapi_exact = {normalize_route(r, normalize_version=False) for r in openapi_routes}
    internal_prefixes = load_internal_prefixes(internal_prefixes_path)
    if not include_internal:
        normalized_handler = {
            r for r in normalized_handler if not is_internal_route(r, internal_prefixes)
        }
        normalized_openapi = {
            r for r in normalized_openapi if not is_internal_route(r, internal_prefixes)
        }
        normalized_openapi_exact = {
            r
            for r in normalized_openapi_exact
            if not is_internal_route(r, internal_prefixes)
            and not is_internal_route(normalize_route(r), internal_prefixes)
        }
        wired_routes = {
            r
            for r in wired_routes
            if not is_internal_route(r, internal_prefixes)
            and not is_internal_route(normalize_route(r), internal_prefixes)
        }
    effective_handler_routes = normalized_handler | wired_routes

    # Find discrepancies
    # Routes in handlers but not in OpenAPI (these need to be documented)
    missing_in_spec = (normalized_handler - normalized_openapi) | (
        wired_routes - normalized_openapi_exact
    )

    # Keep orphan candidates in exact spec-path space. Handler metadata uses
    # legacy version normalization, while literal server wiring only serves the
    # exact path it registers.
    missing_handlers = {
        route
        for route in normalized_openapi_exact
        if normalize_route(route) not in normalized_handler
    }

    # Filter out known patterns that may not have explicit ROUTES
    # (e.g., dynamic routes handled by can_handle())
    known_dynamic_patterns = {
        r
        for r in missing_handlers
        if any(
            p in r
            for p in [
                "{",
                "*",
                "debate_id",
                "agent",
                "workspace_id",
                "org_id",
            ]
        )
    }

    # Routes that are truly orphaned in OpenAPI
    orphan_candidates = missing_handlers - known_dynamic_patterns

    # A spec path is only orphaned if no server wiring or registered handler
    # routes it. Free-function registrations and can_handle paths do not always
    # have class-level ROUTES metadata.
    orphan_candidates, served_wired_registration = filter_wired_orphans(
        orphan_candidates, wired_routes
    )
    orphaned_in_spec, served_can_handle = filter_served_orphans(orphan_candidates)
    served_undeclared = served_wired_registration | served_can_handle

    baseline_missing: set[str] = set()
    baseline_orphaned: set[str] = set()
    if baseline_path:
        baseline_file = Path(baseline_path)
        if baseline_file.exists():
            baseline_data = json.loads(baseline_file.read_text())
            baseline_missing = set(baseline_data.get("missing_in_spec", []))
            baseline_orphaned = set(baseline_data.get("orphaned_in_spec", []))

    new_missing_in_spec = sorted(set(missing_in_spec) - baseline_missing)
    new_orphaned_in_spec = sorted(set(orphaned_in_spec) - baseline_orphaned)

    results = {
        "handler_routes_count": len(handler_routes),
        "wired_function_routes_count": len(wired_routes),
        "effective_handler_routes_count": len(effective_handler_routes),
        "openapi_routes_count": len(openapi_routes),
        "missing_in_spec": sorted(missing_in_spec),
        "missing_in_spec_count": len(missing_in_spec),
        "new_missing_in_spec": new_missing_in_spec,
        "new_missing_in_spec_count": len(new_missing_in_spec),
        "orphaned_in_spec": sorted(orphaned_in_spec),
        "orphaned_in_spec_count": len(orphaned_in_spec),
        "served_undeclared": sorted(served_undeclared),
        "served_undeclared_count": len(served_undeclared),
        "served_wired_registration": sorted(served_wired_registration),
        "served_wired_registration_count": len(served_wired_registration),
        "new_orphaned_in_spec": new_orphaned_in_spec,
        "new_orphaned_in_spec_count": len(new_orphaned_in_spec),
        "dynamic_routes_skipped": len(known_dynamic_patterns),
        "coverage_percentage": round(
            (1 - len(missing_in_spec) / max(len(effective_handler_routes), 1)) * 100, 1
        ),
    }

    if output_json:
        print(json.dumps(results, indent=2))
    else:
        print("=" * 60)
        print("OpenAPI Route Validation Report")
        print("=" * 60)
        print(f"Handler metadata routes: {results['handler_routes_count']}")
        print(f"Wired function routes:   {results['wired_function_routes_count']}")
        print(f"Effective handler routes: {results['effective_handler_routes_count']}")
        print(f"OpenAPI routes:          {results['openapi_routes_count']}")
        print(f"Coverage:                {results['coverage_percentage']}%")
        print()
        if baseline_path:
            print(f"New missing in spec vs baseline: {results['new_missing_in_spec_count']}")
            print(f"New orphaned in spec vs baseline: {results['new_orphaned_in_spec_count']}")
            print()

        if missing_in_spec:
            print(f"Routes missing from OpenAPI spec ({len(missing_in_spec)}):")
            for route in sorted(missing_in_spec)[:20]:
                print(f"  - {route}")
            if len(missing_in_spec) > 20:
                print(f"  ... and {len(missing_in_spec) - 20} more")
            print()

        if orphaned_in_spec:
            print(f"OpenAPI routes without handlers ({len(orphaned_in_spec)}):")
            for route in sorted(orphaned_in_spec)[:10]:
                print(f"  - {route}")
            if len(orphaned_in_spec) > 10:
                print(f"  ... and {len(orphaned_in_spec) - 10} more")
            print()

        if not missing_in_spec and not orphaned_in_spec:
            print("All routes properly documented!")

    if fail_on_missing:
        if baseline_path:
            if new_missing_in_spec or new_orphaned_in_spec:
                sys.exit(1)
        elif missing_in_spec:
            sys.exit(1)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate OpenAPI spec routes match handler registry"
    )
    parser.add_argument(
        "--spec",
        default="docs/api/openapi.json",
        help="Path to OpenAPI spec (default: docs/api/openapi.json)",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help=(
            "Exact 40-hex commit SHA to bind operation evidence to; enables the "
            "method-aware VAL-CDG-011 operation plane"
        ),
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit with error code 1 if routes are missing from spec",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--baseline",
        default="scripts/baselines/validate_openapi_routes.json",
        help="Path to baseline drift file (default: scripts/baselines/validate_openapi_routes.json)",
    )
    parser.add_argument(
        "--include-internal",
        action="store_true",
        help="Include known internal/private route families in validation",
    )
    parser.add_argument(
        "--internal-prefixes",
        default="scripts/baselines/internal_route_prefixes.json",
        help=(
            "Path to JSON file with internal route prefixes "
            "(default: scripts/baselines/internal_route_prefixes.json)"
        ),
    )

    args = parser.parse_args()
    if args.ref is not None:
        try:
            plane = validate_method_aware_plane(
                args.spec,
                args.ref,
                internal_prefixes_path=args.internal_prefixes,
            )
        except MethodAwareError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        if args.json:
            print(json.dumps(plane, indent=2))
        else:
            print("=" * 60)
            print("Method-Aware Route Operation Plane (VAL-CDG-011)")
            print("=" * 60)
            for name in (
                "served_operations",
                "spec_operations",
                "public_operations",
                "internal_operations",
                "unresolved_exposure",
                "served_but_undeclared",
                "declared_and_served",
                "unserved_spec",
            ):
                print(f"{name}: {plane[f'{name}_count']}")
            print(f"method_unresolved_paths: {plane['method_unresolved_paths_count']}")
            print(
                "operation_projection: "
                f"{len(plane['operation_projection'])} memberships, "
                f"{plane['operation_projection_route_edge_total']} edges"
            )
        return
    validate_coverage(
        args.spec,
        args.fail_on_missing,
        args.json,
        args.baseline,
        args.include_internal,
        args.internal_prefixes,
    )


if __name__ == "__main__":
    main()
