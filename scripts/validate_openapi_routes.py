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
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

# Ensure local checkout modules take precedence over any globally installed package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SERVER_ROUTE_REGISTRATION = (
    _REPO_ROOT / "aragora" / "server" / "stream" / "servers_route_registration.py"
)
_LITERAL_ROUTER_METHODS = frozenset(
    {"add_get", "add_post", "add_put", "add_patch", "add_delete", "add_head", "add_options"}
)

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


def _iter_executable_calls(nodes: Iterable[ast.AST]) -> Iterator[ast.Call]:
    """Yield calls while excluding nested definitions that may never execute."""
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call):
            yield node
        yield from _iter_executable_calls(ast.iter_child_nodes(node))


def get_wired_function_routes(
    registration_path: Path | None = None,
    repo_root: Path | None = None,
) -> set[str]:
    """Statically extract literal routes from wired aiohttp registration functions.

    The server registration module is the authority for which free functions
    participate in startup. Only imported ``register_*`` callables that are
    actually called there are considered. Within those exact function bodies,
    only literal ``<first-argument>.router.add_*('/api/...')`` calls count.
    Computed paths, re-exported functions without a local definition, and
    unrelated route-registration helpers remain unverified.
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

    routes: set[str] = set()
    for local_name in sorted(called_names & imports.keys()):
        module_name, function_name = imports[local_name]
        if not (local_name.startswith("register_") or function_name.startswith("register_")):
            continue

        module_path = _resolve_local_module(root, module_name)
        if module_path is None:
            continue
        module_tree = _parse_python_source(module_path)
        definitions = (
            node
            for node in module_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        )
        for function in definitions:
            if not function.args.args:
                continue
            app_argument = function.args.args[0].arg
            for call in _iter_executable_calls(function.body):
                if not isinstance(call.func, ast.Attribute):
                    continue
                if call.func.attr not in _LITERAL_ROUTER_METHODS or not call.args:
                    continue
                receiver = call.func.value
                if not (
                    isinstance(receiver, ast.Attribute)
                    and receiver.attr == "router"
                    and isinstance(receiver.value, ast.Name)
                    and receiver.value.id == app_argument
                ):
                    continue
                path_argument = call.args[0]
                if (
                    isinstance(path_argument, ast.Constant)
                    and isinstance(path_argument.value, str)
                    and path_argument.value.startswith("/api/")
                ):
                    routes.add(path_argument.value)
    return routes


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


def filter_wired_orphans(candidates: set[str]) -> tuple[set[str], set[str]]:
    """Split orphan candidates by literal evidence from wired route functions."""
    try:
        wired_routes = {
            normalize_route(route, normalize_version=False) for route in get_wired_function_routes()
        }
    except RouteRegistrationScanError as exc:
        print(f"Error: {exc}. Refusing to validate partial route wiring.", file=sys.stderr)
        sys.exit(1)

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

    # Normalize routes for comparison
    normalized_handler = {normalize_route(r) for r in handler_routes}
    normalized_openapi = {normalize_route(r) for r in openapi_routes}
    internal_prefixes = load_internal_prefixes(internal_prefixes_path)
    if not include_internal:
        normalized_handler = {
            r for r in normalized_handler if not is_internal_route(r, internal_prefixes)
        }
        normalized_openapi = {
            r for r in normalized_openapi if not is_internal_route(r, internal_prefixes)
        }

    # Find discrepancies
    # Routes in handlers but not in OpenAPI (these need to be documented)
    missing_in_spec = normalized_handler - normalized_openapi

    # Routes in OpenAPI but not in handlers (may be deprecated or generated)
    missing_handlers = normalized_openapi - normalized_handler

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
    orphan_candidates, served_wired_registration = filter_wired_orphans(orphan_candidates)
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
            (1 - len(missing_in_spec) / max(len(handler_routes), 1)) * 100, 1
        ),
    }

    if output_json:
        print(json.dumps(results, indent=2))
    else:
        print("=" * 60)
        print("OpenAPI Route Validation Report")
        print("=" * 60)
        print(f"Handler routes:  {results['handler_routes_count']}")
        print(f"OpenAPI routes:  {results['openapi_routes_count']}")
        print(f"Coverage:        {results['coverage_percentage']}%")
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
