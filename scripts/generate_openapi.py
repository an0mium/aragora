#!/usr/bin/env python3
"""
Generate OpenAPI 3.1.0 spec for Aragora.

Three-tier strategy:
  1. Canonical registry  -- import aragora.server.openapi.generate_openapi_schema()
  2. Runtime decorator    -- import handler modules, read _endpoint_registry
  3. AST fallback         -- parse handler .py files for @api_endpoint(...) calls

By default the script tries the canonical registry first, merges in any
decorator-registered endpoints that are not already in the canonical spec,
and finally falls back to AST parsing when imports fail (e.g. missing deps).

Usage:
    python scripts/generate_openapi.py
    python scripts/generate_openapi.py --output docs/api/openapi.yaml --format yaml
    python scripts/generate_openapi.py --ast-only          # skip runtime imports
    python scripts/generate_openapi.py --legacy-handlers    # old handler introspection
    python scripts/generate_openapi.py --stdout --format json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HANDLERS_DIR = PROJECT_ROOT / "aragora" / "server" / "handlers"

# Ensure project root is on sys.path for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _get_version() -> str:
    """Read package version from aragora/__version__.py without importing."""
    version_file = PROJECT_ROOT / "aragora" / "__version__.py"
    try:
        tree = ast.parse(version_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__version__":
                        if isinstance(node.value, ast.JoinedStr):
                            # f-string -- fall through to exec
                            break
                        if isinstance(node.value, ast.Constant):
                            return str(node.value.value)
        # Fallback: exec the file in a sandbox
        ns: dict[str, Any] = {}
        exec(compile(version_file.read_text(), version_file, "exec"), ns)  # noqa: S102
        return str(ns.get("__version__", "0.0.0"))
    except Exception:
        return "0.0.0"


API_VERSION = _get_version()


# ============================================================================
# AST-based @api_endpoint extractor
# ============================================================================


def _eval_literal(node: ast.expr) -> Any:
    """Safely evaluate an AST node to a Python literal.

    Handles constants, lists, dicts, and unary operators (e.g., True/False).
    Returns None for anything that cannot be statically resolved.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        items = [_eval_literal(elt) for elt in node.elts]
        if any(v is _UNRESOLVABLE for v in items):
            return _UNRESOLVABLE
        return items
    if isinstance(node, ast.Tuple):
        items = [_eval_literal(elt) for elt in node.elts]
        if any(v is _UNRESOLVABLE for v in items):
            return _UNRESOLVABLE
        return tuple(items)
    if isinstance(node, ast.Dict):
        keys = [_eval_literal(k) if k is not None else None for k in node.keys]
        values = [_eval_literal(v) for v in node.values]
        if any(v is _UNRESOLVABLE for v in keys) or any(v is _UNRESOLVABLE for v in values):
            return _UNRESOLVABLE
        return dict(zip(keys, values))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _eval_literal(node.operand)
        if operand is not _UNRESOLVABLE:
            return -operand
    if isinstance(node, ast.Name):
        # True / False / None are ast.Constant in 3.8+, but be safe
        if node.id == "True":
            return True
        if node.id == "False":
            return False
        if node.id == "None":
            return None
    return _UNRESOLVABLE


# Sentinel for values that cannot be resolved statically
_UNRESOLVABLE = object()


def _extract_decorator_kwargs(call_node: ast.Call) -> dict[str, Any]:
    """Extract keyword arguments from an @api_endpoint(...) call node."""
    kwargs: dict[str, Any] = {}
    for kw in call_node.keywords:
        if kw.arg is None:
            continue  # **kwargs -- skip
        value = _eval_literal(kw.value)
        if value is not _UNRESOLVABLE:
            kwargs[kw.arg] = value
    return kwargs


def _is_api_endpoint_decorator(node: ast.expr) -> ast.Call | None:
    """Return the Call node if this is an @api_endpoint(...) decorator."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name) and func.id == "api_endpoint":
        return node
    if isinstance(func, ast.Attribute) and func.attr == "api_endpoint":
        return node
    return None


def _ast_extract_endpoints_from_file(filepath: Path) -> list[dict[str, Any]]:
    """Parse a single Python file and extract @api_endpoint metadata via AST."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    endpoints: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        # Look for decorated functions and methods
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            call = _is_api_endpoint_decorator(decorator)
            if call is None:
                continue
            kwargs = _extract_decorator_kwargs(call)
            path = kwargs.get("path")
            if not path or not isinstance(path, str):
                continue
            method = kwargs.get("method", "GET")
            if not isinstance(method, str):
                continue
            summary = kwargs.get("summary", "")
            if not isinstance(summary, str):
                summary = ""
            tags = kwargs.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            description = kwargs.get("description", "")
            if not isinstance(description, str):
                description = ""
            parameters = kwargs.get("parameters")
            if not isinstance(parameters, list):
                parameters = []
            request_body = kwargs.get("request_body")
            if not isinstance(request_body, dict):
                request_body = None
            responses = kwargs.get("responses")
            if not isinstance(responses, dict):
                responses = {}
            auth_required = kwargs.get("auth_required", True)
            deprecated = kwargs.get("deprecated", False)
            operation_id = kwargs.get("operation_id")
            if not isinstance(operation_id, str):
                operation_id = node.name

            # Use function docstring as description fallback
            if not description and isinstance(node.body, list) and node.body:
                first_stmt = node.body[0]
                if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
                    doc = first_stmt.value.value
                    if isinstance(doc, str):
                        description = doc.strip()

            if not summary:
                summary = node.name.replace("_", " ").title()

            endpoints.append(
                {
                    "path": path,
                    "method": method.upper(),
                    "summary": summary,
                    "tags": tags,
                    "description": description,
                    "parameters": parameters,
                    "request_body": request_body,
                    "responses": responses,
                    "auth_required": bool(auth_required),
                    "deprecated": bool(deprecated),
                    "operation_id": operation_id,
                    "source_file": str(filepath.relative_to(PROJECT_ROOT)),
                }
            )

    return endpoints


def ast_scan_handlers(handlers_dir: Path | None = None) -> list[dict[str, Any]]:
    """Scan all handler .py files under handlers_dir using AST parsing.

    Returns a flat list of endpoint metadata dicts.
    """
    root = handlers_dir or HANDLERS_DIR
    if not root.is_dir():
        print(f"Warning: handlers directory not found: {root}", file=sys.stderr)
        return []

    all_endpoints: list[dict[str, Any]] = []
    for py_file in sorted(root.rglob("*.py")):
        if py_file.name.startswith("_") and py_file.name != "__init__.py":
            continue
        endpoints = _ast_extract_endpoints_from_file(py_file)
        all_endpoints.extend(endpoints)

    return all_endpoints


# ============================================================================
# Runtime decorator introspection
# ============================================================================


def _runtime_collect_decorator_endpoints() -> list[dict[str, Any]]:
    """Import handler modules and collect endpoints from the global registry."""
    try:
        from aragora.server.handlers.openapi_decorator import get_registered_endpoints

        # Force-import handler modules to trigger decorator registration
        _force_import_handlers()

        endpoints = get_registered_endpoints()
        result: list[dict[str, Any]] = []
        for ep in endpoints:
            result.append(
                {
                    "path": ep.path,
                    "method": ep.method,
                    "summary": ep.summary,
                    "tags": ep.tags,
                    "description": ep.description,
                    "parameters": ep.parameters,
                    "request_body": ep.request_body,
                    "responses": ep.responses,
                    "auth_required": bool(ep.security),
                    "deprecated": ep.deprecated,
                    "operation_id": ep.operation_id,
                    "source": "runtime_decorator",
                }
            )
        return result
    except Exception as exc:
        print(f"Warning: runtime decorator collection failed: {exc}", file=sys.stderr)
        return []


def _force_import_handlers() -> None:
    """Best-effort import of handler modules to trigger decorator registration."""
    import importlib

    if not HANDLERS_DIR.is_dir():
        return
    for py_file in sorted(HANDLERS_DIR.rglob("*.py")):
        if py_file.name.startswith("__"):
            continue
        rel = py_file.relative_to(PROJECT_ROOT)
        module_name = str(rel).replace("/", ".").replace("\\", ".").removesuffix(".py")
        try:
            importlib.import_module(module_name)
        except Exception:
            pass  # Best-effort; AST fallback covers failures


# ============================================================================
# Endpoint list -> OpenAPI paths conversion
# ============================================================================


def _extract_path_params(path: str) -> list[dict[str, Any]]:
    """Extract {param} placeholders from a path and return parameter defs."""
    params: list[dict[str, Any]] = []
    for match in re.finditer(r"\{(\w+)\}", path):
        name = match.group(1)
        params.append(
            {
                "name": name,
                "in": "path",
                "required": True,
                "schema": {"type": "string"},
                "description": f"Path parameter: {name}",
            }
        )
    return params


def endpoints_to_openapi_paths(
    endpoints: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Convert a flat list of endpoint dicts to an OpenAPI paths dict."""
    paths: dict[str, dict[str, Any]] = {}

    for ep in endpoints:
        path = ep["path"]
        method = ep["method"].lower()

        if path not in paths:
            paths[path] = {}
        if method in paths[path]:
            continue  # First-seen wins (canonical > runtime > AST)

        operation: dict[str, Any] = {
            "summary": ep.get("summary", ""),
            "tags": ep.get("tags", []),
        }

        description = ep.get("description", "")
        if description:
            operation["description"] = description

        op_id = ep.get("operation_id")
        if op_id:
            operation["operationId"] = op_id

        # Parameters: merge explicit + auto-detected path params
        explicit_params = ep.get("parameters", [])
        auto_path_params = _extract_path_params(path)
        explicit_names = {p.get("name") for p in explicit_params if isinstance(p, dict)}
        merged_params = list(explicit_params)
        for auto_p in auto_path_params:
            if auto_p["name"] not in explicit_names:
                merged_params.append(auto_p)
        if merged_params:
            operation["parameters"] = merged_params

        # Request body
        req_body = ep.get("request_body")
        if req_body and isinstance(req_body, dict):
            operation["requestBody"] = req_body
        elif method in ("post", "put", "patch") and not req_body:
            operation["requestBody"] = {
                "content": {"application/json": {"schema": {"type": "object"}}},
            }

        # Responses
        responses = ep.get("responses", {})
        if responses and isinstance(responses, dict):
            operation["responses"] = responses
        else:
            operation["responses"] = {
                "200": {
                    "description": "Success",
                    "content": {
                        "application/json": {"schema": {"type": "object"}},
                    },
                },
            }

        # Security
        if ep.get("auth_required", True):
            operation["security"] = [{"bearerAuth": []}]

        if ep.get("deprecated", False):
            operation["deprecated"] = True

        paths[path][method] = operation

    return paths


# ============================================================================
# Full schema assembly
# ============================================================================


def _collect_tags(paths: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """Aggregate unique tags from all operations, return sorted tag objects."""
    tag_set: set[str] = set()
    for path_spec in paths.values():
        for method_spec in path_spec.values():
            if isinstance(method_spec, dict):
                for tag in method_spec.get("tags", []):
                    tag_set.add(tag)
    return [{"name": t, "description": f"{t} operations"} for t in sorted(tag_set)]


def build_openapi_schema(
    paths: dict[str, dict[str, Any]],
    *,
    version: str | None = None,
) -> dict[str, Any]:
    """Assemble the complete OpenAPI 3.1.0 document."""
    sorted_paths = dict(sorted(paths.items()))
    tags = _collect_tags(sorted_paths)

    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Aragora API",
            "description": (
                "Control plane for multi-agent vetted decisionmaking across "
                "organizational knowledge and channels. Orchestrates 15+ AI "
                "models to debate your organization's knowledge and deliver "
                "defensible decisions with full audit trails."
            ),
            "version": version or API_VERSION,
            "contact": {"name": "Aragora Team"},
            "license": {"name": "MIT", "identifier": "MIT"},
        },
        "servers": [
            {"url": "http://localhost:8080", "description": "Development server"},
            {"url": "https://api.aragora.ai", "description": "Production server"},
        ],
        "tags": tags,
        "paths": sorted_paths,
        "components": {
            "schemas": {
                "Error": {
                    "type": "object",
                    "properties": {
                        "error": {"type": "string"},
                        "code": {"type": "string"},
                        "trace_id": {"type": "string"},
                    },
                    "required": ["error"],
                },
            },
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "description": (
                        "API token authentication. Set via ARAGORA_API_TOKEN environment variable."
                    ),
                },
            },
        },
    }


# ============================================================================
# Legacy handler introspection (preserved for --legacy-handlers flag)
# ============================================================================


def _legacy_generate_openapi_schema() -> dict[str, Any]:
    """Generate spec via old handler ROUTES introspection (backward compat)."""
    import inspect

    handlers: list[tuple[str, Any]] = []
    try:
        import aragora.server.handlers as handlers_module

        all_exports = getattr(handlers_module, "__all__", dir(handlers_module))
        for name in all_exports:
            if name.endswith("Handler") and name != "BaseHandler":
                try:
                    cls = getattr(handlers_module, name, None)
                    if cls is not None and hasattr(cls, "ROUTES"):
                        handlers.append((name, cls))
                except Exception:
                    pass
    except ImportError:
        pass

    paths: dict[str, Any] = {}
    tags: dict[str, str] = {}

    tag_map = {
        "System": "System",
        "Debates": "Debates",
        "Agents": "Agents",
        "Pulse": "Pulse",
        "Analytics": "Analytics",
        "Consensus": "Consensus",
        "Memory": "Memory",
        "Gauntlet": "Gauntlet",
    }

    for handler_name, handler_cls in handlers:
        tag_key = handler_name.replace("Handler", "")
        tag = tag_map.get(tag_key, tag_key)
        handler_doc = inspect.getdoc(handler_cls)
        if handler_doc and tag not in tags:
            tags[tag] = handler_doc.split("\n")[0]

        handler_routes = getattr(handler_cls, "ROUTES", [])
        if isinstance(handler_routes, dict):
            handler_routes = list(handler_routes.keys())

        for route in handler_routes:
            if route not in paths:
                paths[route] = {}
            paths[route]["get"] = {
                "summary": f"GET {route}",
                "tags": [tag],
                "responses": {"200": {"description": "Success"}},
            }

    sorted_paths = dict(sorted(paths.items()))
    tag_list = [{"name": t, "description": d or f"{t} operations"} for t, d in sorted(tags.items())]
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Aragora API",
            "description": "Auto-generated from handler ROUTES (legacy mode).",
            "version": API_VERSION,
            "contact": {"name": "Aragora Team"},
            "license": {"name": "MIT"},
        },
        "servers": [{"url": "http://localhost:8080", "description": "Development server"}],
        "tags": tag_list,
        "paths": sorted_paths,
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "description": "API token authentication",
                },
            },
        },
    }


# ============================================================================
# Main generation pipeline
# ============================================================================


def generate_schema(
    *,
    ast_only: bool = False,
    legacy_handlers: bool = False,
    include_runtime: bool = True,
) -> dict[str, Any]:
    """Run the multi-tier generation pipeline and return the OpenAPI schema."""
    if legacy_handlers:
        print("Generating OpenAPI spec from handler ROUTES (legacy)...", file=sys.stderr)
        return _legacy_generate_openapi_schema()

    if ast_only:
        print("Generating OpenAPI spec via AST parsing only...", file=sys.stderr)
        endpoints = ast_scan_handlers()
        paths = endpoints_to_openapi_paths(endpoints)
        return build_openapi_schema(paths)

    # --- Tier 1: canonical registry ---
    canonical_paths: dict[str, dict[str, Any]] = {}
    canonical_ok = False
    try:
        from aragora.server.openapi import generate_openapi_schema as canonical_generate

        canonical_schema = canonical_generate()
        canonical_paths = canonical_schema.get("paths", {})
        canonical_ok = True
        print(
            f"Tier 1 (canonical registry): {_count_operations(canonical_paths)} operations",
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"Tier 1 (canonical registry) failed: {exc}", file=sys.stderr)

    # --- Tier 2: runtime decorator registry ---
    runtime_endpoints: list[dict[str, Any]] = []
    if include_runtime and not ast_only:
        runtime_endpoints = _runtime_collect_decorator_endpoints()
        print(
            f"Tier 2 (runtime decorators): {len(runtime_endpoints)} endpoints",
            file=sys.stderr,
        )

    # --- Tier 3: AST fallback ---
    ast_endpoints = ast_scan_handlers()
    print(
        f"Tier 3 (AST parsing): {len(ast_endpoints)} @api_endpoint calls found",
        file=sys.stderr,
    )

    # --- Merge ---
    # Start with canonical as the base, then layer on decorator and AST endpoints
    # for paths/methods not already present.
    merged_paths = dict(canonical_paths)  # shallow copy of top level

    # Merge runtime decorator endpoints
    if runtime_endpoints:
        decorator_paths = endpoints_to_openapi_paths(runtime_endpoints)
        for path, methods in decorator_paths.items():
            if path not in merged_paths:
                merged_paths[path] = methods
            else:
                for method, spec in methods.items():
                    if method not in merged_paths[path]:
                        merged_paths[path][method] = spec

    # Merge AST endpoints (lowest priority)
    if ast_endpoints:
        ast_paths = endpoints_to_openapi_paths(ast_endpoints)
        for path, methods in ast_paths.items():
            if path not in merged_paths:
                merged_paths[path] = methods
            else:
                for method, spec in methods.items():
                    if method not in merged_paths[path]:
                        merged_paths[path][method] = spec

    # Deduplicate paths that differ only by parameter names (e.g.
    # /api/v1/agent/{name}/introspect vs /api/v1/agent/{param}/introspect).
    # Keep the path with descriptive parameter names over generic {param}.
    merged_paths = _deduplicate_ambiguous_paths(merged_paths)

    if canonical_ok:
        # Rebuild using canonical schema structure but with merged paths
        canonical_schema["paths"] = dict(sorted(merged_paths.items()))
        # Ensure tags reflect any new additions
        existing_tag_names = {t["name"] for t in canonical_schema.get("tags", [])}
        for tag_obj in _collect_tags(merged_paths):
            if tag_obj["name"] not in existing_tag_names:
                canonical_schema.setdefault("tags", []).append(tag_obj)
                existing_tag_names.add(tag_obj["name"])
        return canonical_schema
    else:
        return build_openapi_schema(merged_paths)


def _deduplicate_ambiguous_paths(paths: dict[str, Any]) -> dict[str, Any]:
    """Remove paths that differ only in parameter names.

    When two paths normalize to the same template (e.g.
    /api/v1/agent/{name}/introspect vs /api/v1/agent/{param}/introspect),
    keep the one with descriptive parameter names over generic {param},
    and prefer non-autogenerated specs.
    """
    norm_re = re.compile(r"\{[^}]+\}")

    def _norm(p: str) -> str:
        return norm_re.sub("*", p).rstrip("/")

    groups: dict[str, list[str]] = {}
    for path in paths:
        groups.setdefault(_norm(path), []).append(path)

    result: dict[str, Any] = {}
    for group in groups.values():
        if len(group) == 1:
            result[group[0]] = paths[group[0]]
            continue

        # Pick best: prefer non-autogenerated, then named params, then versioned
        def _score(p: str) -> tuple[int, int, int]:
            spec = paths[p]
            has_auto = any(
                isinstance(op, dict) and op.get("x-autogenerated") is True for op in spec.values()
            )
            return (
                0 if has_auto else 1,
                -p.count("{param}"),
                1 if "/api/v1/" in p or "/api/v2/" in p else 0,
            )

        best = max(group, key=_score)
        result[best] = paths[best]

    removed = len(paths) - len(result)
    if removed:
        print(
            f"Deduplicated {removed} ambiguous path(s) ({len(result)} paths remain)",
            file=sys.stderr,
        )
    return result


def _count_operations(paths: dict[str, dict[str, Any]]) -> int:
    """Count total HTTP method operations across all paths."""
    return sum(len(methods) for methods in paths.values())


# ============================================================================
# Output helpers
# ============================================================================


def save_schema(schema: dict[str, Any], output_path: str, fmt: str = "json") -> int:
    """Save schema to file. Returns endpoint count."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "yaml":
        try:
            import yaml

            content = yaml.dump(schema, default_flow_style=False, sort_keys=False, width=120)
        except ImportError:
            print(
                "Warning: PyYAML not installed, falling back to JSON output",
                file=sys.stderr,
            )
            fmt = "json"
            content = json.dumps(schema, indent=2)
    else:
        content = json.dumps(schema, indent=2)

    path.write_text(content + "\n")

    endpoint_count = _count_operations(schema.get("paths", {}))
    return endpoint_count


def print_summary(schema: dict[str, Any]) -> None:
    """Print a human-readable summary of the generated spec."""
    paths = schema.get("paths", {})
    total_paths = len(paths)
    method_counter: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()

    for path_spec in paths.values():
        for method, op in path_spec.items():
            method_upper = method.upper()
            method_counter[method_upper] += 1
            if isinstance(op, dict):
                for tag in op.get("tags", []):
                    tag_counter[tag] += 1

    total_ops = sum(method_counter.values())

    print("\n--- OpenAPI Generation Summary ---", file=sys.stderr)
    print(f"OpenAPI version : {schema.get('openapi', '?')}", file=sys.stderr)
    print(f"API version     : {schema.get('info', {}).get('version', '?')}", file=sys.stderr)
    print(f"Total paths     : {total_paths}", file=sys.stderr)
    print(f"Total operations: {total_ops}", file=sys.stderr)

    print("\nBy HTTP method:", file=sys.stderr)
    for method in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
        count = method_counter.get(method, 0)
        if count:
            print(f"  {method:8s} {count}", file=sys.stderr)

    print(f"\nBy tag ({len(tag_counter)} tags):", file=sys.stderr)
    for tag, count in tag_counter.most_common(25):
        print(f"  {tag:30s} {count}", file=sys.stderr)
    if len(tag_counter) > 25:
        remaining = sum(c for _, c in tag_counter.most_common()[25:])
        print(f"  {'... (remaining)':30s} {remaining}", file=sys.stderr)

    print("--- End Summary ---\n", file=sys.stderr)


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OpenAPI 3.1.0 spec for Aragora (multi-tier pipeline)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="docs/api/openapi.yaml",
        help="Output file path (default: docs/api/openapi.yaml)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "yaml"],
        default=None,
        help="Output format (auto-detected from --output extension if not set)",
    )
    parser.add_argument(
        "--ast-only",
        action="store_true",
        help="Use only AST parsing (no runtime imports)",
    )
    parser.add_argument(
        "--legacy-handlers",
        action="store_true",
        help="Use legacy handler ROUTES introspection",
    )
    parser.add_argument(
        "--no-runtime",
        action="store_true",
        help="Skip runtime decorator import (tiers 1 + 3 only)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print spec to stdout instead of file",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress summary output",
    )
    args = parser.parse_args()

    # Auto-detect format from output extension
    fmt = args.format
    if fmt is None:
        if args.output.endswith(".yaml") or args.output.endswith(".yml"):
            fmt = "yaml"
        else:
            fmt = "json"

    schema = generate_schema(
        ast_only=args.ast_only,
        legacy_handlers=args.legacy_handlers,
        include_runtime=not args.no_runtime,
    )

    if not args.quiet:
        print_summary(schema)

    if args.stdout:
        if fmt == "yaml":
            try:
                import yaml

                print(yaml.dump(schema, default_flow_style=False, sort_keys=False, width=120))
            except ImportError:
                print(json.dumps(schema, indent=2))
        else:
            print(json.dumps(schema, indent=2))
    else:
        endpoint_count = save_schema(schema, args.output, fmt)
        print(f"Output: {args.output} ({endpoint_count} operations)", file=sys.stderr)


if __name__ == "__main__":
    main()
