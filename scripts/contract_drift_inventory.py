import hashlib
import json
import re
import subprocess
from os.path import relpath
from pathlib import Path
from typing import Any

try:
    from sdk_path_normalize import normalize_sdk_path
except ModuleNotFoundError:
    from scripts.sdk_path_normalize import normalize_sdk_path  # type: ignore[no-redef]

METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}
CATEGORIES = tuple(
    "python_sdk_drift typescript_sdk_drift routes_missing_in_spec "
    "routes_orphaned_in_spec sdk_missing_from_both".split()
)
CLASSES = {"retained-intentional-growth", "resolved", "stale-sdk", "generated-artifact"}
_METHOD = r"(?P<method>GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)"
PY_RE = re.compile(
    r"(?:await\s+)?self\._client\._?request\(\s*[\"']"
    + _METHOD
    + r"[\"']\s*,\s*f?[\"'](?P<path>/[^\"']+)[\"']"
)
TS_PATTERNS = (
    re.compile(
        r"this(?:\.client)?\.request(?:<[^(]*>)?\(\s*[\"']"
        + _METHOD
        + r"[\"']\s*,\s*[`\"'](?P<path>/[^`\"']+)[`\"']"
    ),
    re.compile(
        r"this\.client\.(?P<method>get|post|put|patch|delete|options|head)"
        r"(?:<[^(]*>)?\(\s*[`\"'](?P<path>/[^`\"']+)[`\"']"
    ),
    re.compile(
        r"this\.invoke(?:<[^(]*>)?\(\s*[\"'][^\"']+[\"']\s*,\s*\[[^\]]*\]\s*,\s*"
        r"[\"']" + _METHOD + r"[\"']\s*,\s*[`\"'](?P<path>/[^`\"']+)[`\"']"
    ),
)


class InventoryError(RuntimeError):
    pass


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        _json({key: item for key, item in value.items() if key != "inventory_sha256"}).encode()
    ).hexdigest()


def _object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise InventoryError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InventoryError(f"{label} must be a JSON object: {path}")
    return value


def _path(value: str) -> str:
    value = re.sub(r"\$?\{[^}]+\}", "{param}", value)
    return normalize_sdk_path(value)


def extract_python_endpoints(content: str) -> set[tuple[str, str]]:
    return {(match["method"].lower(), _path(match["path"])) for match in PY_RE.finditer(content)}


def extract_typescript_endpoints(content: str) -> set[tuple[str, str]]:
    return {
        (match["method"].lower(), _path(match["path"]))
        for pattern in TS_PATTERNS
        for match in pattern.finditer(content)
    }


def _scan(root: Path, extractor: Any, suffix: str) -> dict[str, set[tuple[str, str]]]:
    if not root.is_dir():
        raise InventoryError(f"{suffix} SDK namespace directory not found: {root}")
    result = {}
    for path in sorted(root.glob(f"*.{suffix}")):
        if path.stem.startswith("_") or path.name == "index.ts":
            continue
        endpoints = extractor(path.read_text(encoding="utf-8"))
        if endpoints:
            result[path.stem] = endpoints
    return result


def scan_typescript_sdk_by_namespace(
    root: Path,
) -> tuple[dict[str, set[tuple[str, str]]], set[tuple[str, str]]]:
    result = _scan(root, extract_typescript_endpoints, "ts")
    return (
        {key: value for key, value in result.items() if key != "openapi"},
        result.get("openapi", set()),
    )


def load_openapi_endpoints(paths: list[Path]) -> set[tuple[str, str]]:
    if not paths:
        raise InventoryError("at least one OpenAPI input is required")
    result: set[tuple[str, str]] = set()
    for path in paths:
        spec_paths = _object(path, "OpenAPI input").get("paths")
        if not isinstance(spec_paths, dict):
            raise InventoryError(f"OpenAPI input 'paths' must be an object: {path}")
        for raw_path, operations in spec_paths.items():
            if not isinstance(raw_path, str) or not isinstance(operations, dict):
                raise InventoryError(f"malformed OpenAPI path entry in {path}: {raw_path!r}")
            for method, operation in operations.items():
                if isinstance(method, str) and method.lower() in METHODS:
                    if not isinstance(operation, dict):
                        raise InventoryError(f"malformed OpenAPI operation: {raw_path!r}")
                    result.add((method.lower(), normalize_sdk_path(raw_path)))
    return result


def _baselines(verify: dict[str, Any], routes: dict[str, Any], parity: dict[str, Any]):
    values = {
        "python_sdk_drift": verify.get("python_sdk_drift"),
        "typescript_sdk_drift": verify.get("typescript_sdk_drift"),
        "routes_missing_in_spec": routes.get("missing_in_spec"),
        "routes_orphaned_in_spec": routes.get("orphaned_in_spec"),
        "sdk_missing_from_both": parity.get("missing_from_both_sdks"),
    }
    parsed = {}
    for category, entries in values.items():
        if not isinstance(entries, list) or not all(isinstance(entry, str) for entry in entries):
            raise InventoryError(f"baseline category must be a string list: {category}")
        parsed[category] = entries
    return parsed


def _split(category: str, entry: str) -> tuple[str | None, str]:
    if category.startswith(("python_", "typescript_")):
        parts = entry.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() not in METHODS:
            raise InventoryError(f"malformed SDK baseline entry in {category}: {entry!r}")
        return parts[0].upper(), _path(parts[1])
    return None, _path(entry)


def _build_inventory(
    live: dict[str, set[Any]],
    *,
    handler_paths: set[str],
    generated_typescript_drift: set[tuple[str, str]],
    resolved_baseline_entries: dict[str, set[str]],
    source_sha: str,
    inputs: list[str] | None = None,
) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}

    def record(path: str) -> dict[str, Any]:
        return records.setdefault(
            _path(path),
            {"categories": {"live": set(), "historical": set(), "generated": set()}, "methods": {}},
        )

    def add(
        path: str,
        category: str,
        method: str | None = None,
        state: str = "live",
    ):
        item = record(path)
        item["categories"][state].add(category)
        if method:
            item["methods"].setdefault(f"{state}:{category}", set()).add(method.upper())

    for category, values in live.items():
        for value in values:
            if category.startswith(("python_", "typescript_")):
                method, path = value
                add(path, category, method)
            else:
                add(value, category)
    for method, path in generated_typescript_drift:
        add(path, "typescript_openapi_namespace", method, "generated")
    for category, entries in resolved_baseline_entries.items():
        for entry in entries:
            method, path = _split(category, entry)
            add(path, category, method, "historical")

    handlers = {_path(path) for path in handler_paths if not path.endswith("/*")}
    prefixes = {_path(path[:-2]) for path in handler_paths if path.endswith("/*")}

    def covered(path: str) -> bool:
        return path in handlers or any(path.startswith(prefix + "/") for prefix in prefixes)

    items: list[dict[str, Any]] = []
    for path, value in sorted(records.items()):
        categories = {state: sorted(found) for state, found in value["categories"].items()}
        methods = {key: sorted(found) for key, found in sorted(value["methods"].items())}
        current = set(categories["live"])
        handler_backed = covered(path) or "routes_missing_in_spec" in current
        if current and (handler_backed or {"python_sdk_drift", "typescript_sdk_drift"} <= current):
            classification = "retained-intentional-growth"
        elif current:
            classification = "stale-sdk"
        elif categories["generated"]:
            classification = "generated-artifact"
        else:
            classification = "resolved"

        items.append(
            {
                "path": path,
                "categories": categories,
                "methods": methods,
                "classification": classification,
            }
        )

    raw = {category: len(live[category]) for category in CATEGORIES}
    counts = {
        name: sum(item["classification"] == name for item in items) for name in sorted(CLASSES)
    }
    payload = {
        "schema_version": 1,
        "provenance": {
            "source_sha": source_sha,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs": inputs or [],
            "exclusions": ["sdk/typescript/src/namespaces/openapi.ts"],
            "method_limitations": ["Handler classifications are path-level; methods unavailable."],
        },
        "summary": {
            "raw_category_counts": raw,
            "raw_live_total": sum(raw.values()),
            "deduplicated_live_total": sum(bool(item["categories"]["live"]) for item in items),
            "classification_counts": counts,
            "accepted_item_count": len(items),
            "open_item_count": counts["stale-sdk"],
            "judgment_sensitive_count": counts["retained-intentional-growth"],
            "generated_typescript_false_positive_count": len(generated_typescript_drift),
        },
        "items": items,
    }
    payload["inventory_sha256"] = _digest(payload)
    return payload


def _sha(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def _handlers(parity: Any) -> set[str]:
    from aragora.server.handler_registry import HANDLER_REGISTRY

    failures = []
    for name, reference in HANDLER_REGISTRY:
        try:
            handler = getattr(reference, "resolve", lambda: reference)()
            if handler is None:
                raise ImportError("resolved to None")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc}")
    if failures:
        raise InventoryError(f"handler extraction incomplete: {'; '.join(failures[:5])}")
    result = parity.extract_handler_routes_with_status()
    if not result.available:
        raise InventoryError(result.error or "handler route extraction unavailable")
    paths = {_path(path) for routes in result.routes.values() for path in routes}
    if not paths:
        raise InventoryError("handler extraction returned no routes")
    return paths


def build_live_inventory(
    repo_root: Path,
    *,
    verify_baseline: Path | None = None,
    routes_baseline: Path | None = None,
    parity_baseline: Path | None = None,
    extra_specs: list[Path] | None = None,
) -> dict[str, Any]:
    if __package__:
        from scripts import check_sdk_parity as parity, validate_openapi_routes as routes
    else:
        import check_sdk_parity as parity, validate_openapi_routes as routes  # type: ignore[no-redef]  # noqa: E401

    baseline_paths = (
        verify_baseline or repo_root / "scripts/baselines/verify_sdk_contracts.json",
        routes_baseline or repo_root / "scripts/baselines/validate_openapi_routes.json",
        parity_baseline or repo_root / "scripts/baselines/check_sdk_parity.json",
    )
    baseline = _baselines(
        _object(baseline_paths[0], "verify SDK baseline"),
        _object(baseline_paths[1], "route baseline"),
        _object(baseline_paths[2], "SDK parity baseline"),
    )
    specs = [
        repo_root / "docs/api/openapi.json",
        repo_root / "docs/api/openapi_generated.json",
        *(extra_specs or []),
    ]
    openapi = load_openapi_endpoints(specs)
    py_root = repo_root / "sdk/python/aragora_sdk/namespaces"
    ts_root = repo_root / "sdk/typescript/src/namespaces"
    py_ns = _scan(py_root, extract_python_endpoints, "py")
    ts_ns, generated = scan_typescript_sdk_by_namespace(ts_root)
    py = (set().union(*py_ns.values()) if py_ns else set()) - openapi
    ts = (set().union(*ts_ns.values()) if ts_ns else set()) - openapi
    generated -= openapi
    handlers = _handlers(parity)

    route_handlers = routes.get_handler_routes()
    if not route_handlers:
        raise InventoryError("route validation handler extraction returned no routes")
    policy = _object(
        repo_root / "scripts/baselines/internal_route_prefixes.json", "internal route prefix policy"
    ).get("prefixes")
    if not isinstance(policy, list) or not all(isinstance(item, str) for item in policy):
        raise InventoryError("internal route prefix policy must contain a string-list 'prefixes'")
    internal = tuple(item.rstrip("/") for item in policy)
    handler_routes = {routes.normalize_route(path) for path in route_handlers}
    spec_routes = {
        routes.normalize_route(path)
        for spec in specs
        for path in _object(spec, "OpenAPI input")["paths"]
    }
    handler_routes = {path for path in handler_routes if not path.startswith(internal)}
    spec_routes = {path for path in spec_routes if not path.startswith(internal)}
    missing = {_path(path) for path in handler_routes - spec_routes}
    absent = spec_routes - handler_routes
    dynamic = {path for path in absent if "{" in path or "*" in path}
    orphaned = {_path(path) for path in absent - dynamic}

    report = parity.build_parity_report(
        {"canonical": sorted(handlers)},
        {name: {path for _, path in endpoints} for name, endpoints in py_ns.items()},
        {name: {path for _, path in endpoints} for name, endpoints in ts_ns.items()},
        {path for _, path in openapi},
        handler_routes_available=True,
    )
    both = {path for path in report["gaps"]["missing_from_both_sdks"] if path.startswith("/api/")}
    live = {
        "python_sdk_drift": {f"{method.upper()} {path}" for method, path in py},
        "typescript_sdk_drift": {f"{method.upper()} {path}" for method, path in ts},
        "routes_missing_in_spec": missing,
        "routes_orphaned_in_spec": orphaned,
        "sdk_missing_from_both": both,
    }
    resolved = {category: set(entries) - live[category] for category, entries in baseline.items()}

    display = lambda path: relpath(path, repo_root)  # noqa: E731

    return _build_inventory(
        {
            "python_sdk_drift": py,
            "typescript_sdk_drift": ts,
            "routes_missing_in_spec": missing,
            "routes_orphaned_in_spec": orphaned,
            "sdk_missing_from_both": both,
        },
        handler_paths=handlers | route_handlers,
        generated_typescript_drift=generated,
        resolved_baseline_entries=resolved,
        source_sha=_sha(repo_root),
        inputs=[display(path) for path in specs]
        + [
            f"{py_root.relative_to(repo_root)}/*.py",
            f"{ts_root.relative_to(repo_root)}/*.ts",
            *(display(path) for path in baseline_paths),
        ],
    )


def validate_inventory(payload: dict[str, Any]) -> dict[str, Any]:
    provenance, summary, items = (
        payload.get("provenance"),
        payload.get("summary"),
        payload.get("items"),
    )
    if payload.get("schema_version") != 1:
        raise InventoryError("unsupported inventory schema_version")
    if (
        not isinstance(provenance, dict)
        or not re.fullmatch(r"[0-9a-f]{40}", str(provenance.get("source_sha", "")))
        or provenance.get("generator_sha256")
        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise InventoryError("inventory lacks valid source/generator provenance")
    if not isinstance(summary, dict) or not isinstance(items, list):
        raise InventoryError("inventory must contain summary and items")
    try:
        paths = [item["path"] for item in items]
        valid = all(
            isinstance(item, dict)
            and isinstance(item["path"], str)
            and item["path"].startswith("/")
            and item["classification"] in CLASSES
            and set(item["categories"]) == {"live", "historical", "generated"}
            and isinstance(item["methods"], dict)
            for item in items
        )
    except (KeyError, TypeError):
        valid, paths = False, []
    if not valid or len(paths) != len(set(paths)):
        raise InventoryError("inventory contains malformed or duplicate items")
    counts = {name: sum(item["classification"] == name for item in items) for name in CLASSES}
    if (
        summary.get("classification_counts") != {name: counts[name] for name in sorted(CLASSES)}
        or summary.get("open_item_count") != counts["stale-sdk"]
    ):
        raise InventoryError("inventory totals are inconsistent")
    if payload.get("inventory_sha256") != _digest(payload):
        raise InventoryError("inventory digest mismatch")
    return payload


def load_inventory(path: Path) -> dict[str, Any]:
    return validate_inventory(_object(path, "contract drift inventory"))


def inventory_coverage_errors(current: dict[str, Any], accepted: dict[str, Any]) -> list[str]:
    validate_inventory(current)
    validate_inventory(accepted)
    prior = {item["path"]: item for item in accepted["items"]}
    errors = []
    for item in current["items"]:
        old = prior.get(item["path"])
        if old is None:
            errors.append(f"unclassified live path: {item['path']}")
            continue
        for state, categories in item["categories"].items():
            old_categories = set(old["categories"][state])
            if state == "historical":
                old_categories |= set(old["categories"]["live"])
            if not set(categories).issubset(old_categories):
                errors.append(f"unclassified {state} category growth for {item['path']}")
        for key, methods in item["methods"].items():
            state, category = key.split(":", 1)
            old_methods = set(old["methods"].get(key, []))
            if state == "historical":
                old_methods |= set(old["methods"].get(f"live:{category}", []))
            if not set(methods).issubset(old_methods):
                errors.append(f"unclassified {state} category method growth for {item['path']}")
        if (
            item["classification"] == "stale-sdk" != old["classification"]
            or old["classification"] == "stale-sdk"
            and item["classification"] == "retained-intentional-growth"
        ):
            errors.append(
                f"classification changed for {item['path']}: "
                f"{old['classification']} -> {item['classification']}"
            )
    return errors


def write_inventory(path: Path, payload: dict[str, Any]) -> None:
    validate_inventory(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(payload) + "\n", encoding="utf-8")
