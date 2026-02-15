#!/usr/bin/env python3
"""Batch-add OpenAPI stub entries for TS SDK endpoints that are missing from openapi.json.

Reads all TypeScript SDK namespace files, extracts their endpoint references,
compares against the OpenAPI spec (including openapi_generated.json), and adds
minimal stub entries for any missing paths/methods.

Usage:
    python scripts/batch_add_openapi_stubs.py            # Dry run (report only)
    python scripts/batch_add_openapi_stubs.py --apply     # Write changes to openapi.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}

# Regex patterns matching the verify_sdk_contracts.py extraction logic exactly
TS_REQUEST_RE = re.compile(
    r"this\.client\.request\(\s*['\"](?P<method>[A-Z]+)['\"]\s*,"
    r"\s*(?P<path>`[^`]+`|'[^']+'|\"[^\"]+\")"
)
TS_DIRECT_RE = re.compile(
    r"this\.client\.(?P<method>get|post|put|delete|patch)\("
    r"\s*(?P<path>`[^`]+`|'[^']+'|\"[^\"]+\")"
)

# Map path prefix segments to OpenAPI tags
PREFIX_TO_TAG = {
    "accounting": "Accounting",
    "admin": "Admin",
    "agents": "Agents",
    "analytics": "Analytics",
    "audit": "Audit",
    "backups": "Backups",
    "batch": "Batch",
    "bindings": "Bindings",
    "bots": "Bots",
    "checkpoints": "Checkpoints",
    "code-review": "Codebase",
    "compliance": "Compliance",
    "computer-use": "Computer Use",
    "connectors": "Connectors",
    "cross-pollination": "Cross-Pollination",
    "debates": "Debates",
    "decisions": "Decisions",
    "devices": "Devices",
    "email": "Email",
    "evolution": "Evolution",
    "external-agents": "Agents",
    "facts": "Knowledge",
    "feedback": "Feedback",
    "gateway": "Gateway",
    "gauntlet": "Gauntlet",
    "genesis": "Genesis",
    "gmail": "Gmail",
    "gusto": "Accounting",
    "inbox": "Inbox",
    "incidents": "Monitoring",
    "integrations": "Integrations",
    "keys": "Keys",
    "km": "Knowledge Mound",
    "knowledge": "Knowledge",
    "leaderboard": "Leaderboard",
    "learning": "Learning",
    "media": "Media",
    "memory": "Memory",
    "notifications": "Notifications",
    "onboarding": "Onboarding",
    "oncall": "Monitoring",
    "orchestration": "Debates",
    "org": "Admin",
    "outlook": "Email",
    "partners": "Integrations",
    "payments": "Payments",
    "personas": "Personas",
    "plugins": "Plugins",
    "podcast": "Media",
    "policies": "Policies",
    "probes": "Probes",
    "pulse": "Pulse",
    "rbac": "Security",
    "replays": "Replays",
    "repository": "Codebase",
    "rlm": "RLM",
    "services": "Monitoring",
    "skills": "Skills",
    "teams": "Teams",
    "transcription": "Transcription",
    "users": "Users",
    "voice": "Audio",
    "webhooks": "Webhooks",
    "workflow": "Workflow Templates",
}


def _normalize(path: str) -> str:
    """Normalize a path for comparison (mirrors verify_sdk_contracts.py)."""
    path = path.split("?", 1)[0]
    path = re.sub(r"\$\{[^}]+\}", "{param}", path)
    path = re.sub(r"\{[^}]+\}", "{param}", path)
    path = re.sub(r"^/api/v\d+/", "/api/", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return path


def load_openapi_endpoints(spec_paths: list[Path]) -> tuple[set[tuple[str, str]], dict[str, str]]:
    """Load all endpoints from OpenAPI specs.

    Returns:
        Tuple of (set of (method, normalized_path), dict mapping normalized_path to original_path)
    """
    endpoints: set[tuple[str, str]] = set()
    norm_to_orig: dict[str, str] = {}

    for spec_path in spec_paths:
        if not spec_path.exists():
            continue
        spec = json.loads(spec_path.read_text())
        for path, ops in spec.get("paths", {}).items():
            norm = _normalize(path)
            if norm not in norm_to_orig:
                norm_to_orig[norm] = path
            for method in ops:
                if method.lower() in HTTP_METHODS:
                    endpoints.add((method.lower(), norm))

    return endpoints, norm_to_orig


def extract_ts_endpoints() -> list[tuple[str, str, str]]:
    """Extract all endpoint references from TypeScript SDK namespaces.

    Returns:
        List of (namespace, method, raw_path) tuples
    """
    ts_dir = PROJECT_ROOT / "sdk" / "typescript" / "src" / "namespaces"
    if not ts_dir.exists():
        return []

    results = []
    for ts_file in sorted(ts_dir.glob("*.ts")):
        if ts_file.name.startswith("_") or ts_file.name == "index.ts" or ts_file.name == "CLAUDE.md":
            continue

        ns = ts_file.stem
        content = ts_file.read_text()

        for m in TS_REQUEST_RE.finditer(content):
            raw = m.group("path")[1:-1]  # strip quotes/backticks
            results.append((ns, m.group("method").lower(), raw))

        for m in TS_DIRECT_RE.finditer(content):
            raw = m.group("path")[1:-1]
            results.append((ns, m.group("method").lower(), raw))

    return results


def _infer_tag(path: str) -> str:
    """Infer an OpenAPI tag from a path."""
    # Strip /api/ prefix and get first segment
    cleaned = re.sub(r"^/api/(v\d+/)?", "", path)
    first_seg = cleaned.split("/")[0] if cleaned else ""
    return PREFIX_TO_TAG.get(first_seg, "Undocumented")


def _path_to_openapi(normalized_path: str) -> str:
    """Convert a normalized path (with {param}) to a valid OpenAPI path.

    Uses descriptive parameter names based on path context.
    For example: /api/debates/{param}/notes/{param}
    becomes:     /api/v1/debates/{debate_id}/notes/{note_id}
    """
    # First, add version prefix if missing
    if not re.match(r"^/api/v\d+/", normalized_path):
        path = normalized_path.replace("/api/", "/api/v1/", 1)
    else:
        path = normalized_path

    # Non-api paths (like /inbox/*) keep as-is
    if not path.startswith("/api/"):
        pass  # keep as-is

    # Replace {param} placeholders with descriptive names
    segments = path.split("/")
    param_count = 0
    result_segments = []

    used_names: set[str] = set()
    for i, seg in enumerate(segments):
        if seg == "{param}":
            param_count += 1
            # Use the previous non-param segment to infer a name
            prev = "item"
            for j in range(i - 1, -1, -1):
                if segments[j] != "{param}" and not segments[j].startswith("{"):
                    prev = segments[j]
                    break
            # Singularize common suffixes
            if prev.endswith("ies"):
                name = prev[:-3] + "y_id"
            elif prev.endswith("ses"):
                name = prev[:-2] + "_id"
            elif prev.endswith("s"):
                name = prev[:-1] + "_id"
            else:
                name = prev + "_id"
            # Ensure uniqueness if multiple params
            if name in used_names:
                name = f"sub_{name}"
            used_names.add(name)
            result_segments.append("{" + name + "}")
        else:
            result_segments.append(seg)

    return "/".join(result_segments)


def _build_stub_operation(method: str, path: str, tag: str) -> dict:
    """Build a minimal stub operation entry."""
    op: dict = {
        "summary": "Autogenerated placeholder (spec pending)",
        "tags": [tag],
        "responses": {
            "200": {
                "description": "OK"
            }
        },
        "x-autogenerated": True,
        "x-method-inferred": False,
        "deprecated": True,
        "x-aragora-stability": "deprecated",
    }

    # Add path parameters
    params = re.findall(r"\{([^}]+)\}", path)
    if params:
        op["parameters"] = [
            {
                "name": p,
                "in": "path",
                "required": True,
                "schema": {"type": "string"},
                "description": f"Path parameter: {p}",
            }
            for p in params
        ]

    # Add requestBody for POST/PUT/PATCH
    if method in ("post", "put", "patch"):
        op["requestBody"] = {
            "required": False,
            "content": {
                "application/json": {
                    "schema": {"type": "object"}
                }
            }
        }

    return op


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-add OpenAPI stubs for missing TS SDK endpoints"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to openapi.json (default: dry run)",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=PROJECT_ROOT / "docs" / "api" / "openapi.json",
        help="Path to OpenAPI spec to modify",
    )
    args = parser.parse_args()

    # Load existing specs
    spec_paths = [args.spec]
    gen_path = PROJECT_ROOT / "docs" / "api" / "openapi_generated.json"
    if gen_path.exists():
        spec_paths.append(gen_path)

    openapi_eps, norm_to_orig = load_openapi_endpoints(spec_paths)
    print(f"Existing OpenAPI endpoints: {len(openapi_eps)}")

    # Extract TS SDK endpoints
    ts_endpoints = extract_ts_endpoints()
    print(f"TS SDK endpoint references: {len(ts_endpoints)}")

    # Find drift (TS endpoints not in any spec)
    drift_items: list[tuple[str, str, str]] = []  # (ns, method, normalized_path)
    seen = set()
    for ns, method, raw_path in ts_endpoints:
        norm = _normalize(raw_path)
        key = (method, norm)
        if key not in openapi_eps and key not in seen:
            drift_items.append((ns, method, norm))
            seen.add(key)

    print(f"TS drift items to add: {len(drift_items)}")

    if not drift_items:
        print("No drift items to add. OpenAPI spec is up to date.")
        return 0

    # Load the main spec for modification
    spec = json.loads(args.spec.read_text())
    paths = spec.setdefault("paths", {})

    # Track what we add
    new_paths = 0
    new_methods = 0

    for ns, method, norm_path in sorted(drift_items, key=lambda x: (x[2], x[1])):
        tag = _infer_tag(norm_path)

        # Check if the normalized path already exists (just needs a new method)
        if norm_path in norm_to_orig:
            orig_path = norm_to_orig[norm_path]
            if orig_path in paths:
                if method not in paths[orig_path]:
                    paths[orig_path][method] = _build_stub_operation(method, orig_path, tag)
                    new_methods += 1
                continue

        # Need to create a new path entry
        openapi_path = _path_to_openapi(norm_path)

        # Check if this openapi_path already exists (from a previous iteration)
        if openapi_path not in paths:
            paths[openapi_path] = {}
            new_paths += 1

        if method not in paths[openapi_path]:
            paths[openapi_path][method] = _build_stub_operation(method, openapi_path, tag)
            new_methods += 1

    print(f"\nNew path entries: {new_paths}")
    print(f"New method stubs: {new_methods}")
    print(f"Total new operations: {new_paths + new_methods}")

    if args.apply:
        # Sort paths for consistency
        spec["paths"] = dict(sorted(spec["paths"].items()))

        # Also update openapi_generated.json to keep them in sync
        args.spec.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
        print(f"\nWrote updated spec to {args.spec}")

        # Also write to openapi_generated.json
        if gen_path.exists():
            gen_spec = json.loads(gen_path.read_text())
            gen_paths = gen_spec.setdefault("paths", {})
            for path_key, methods in spec["paths"].items():
                if path_key not in gen_paths:
                    gen_paths[path_key] = {}
                for m, op in methods.items():
                    if m not in gen_paths[path_key] and isinstance(op, dict):
                        gen_paths[path_key][m] = op
            gen_spec["paths"] = dict(sorted(gen_spec["paths"].items()))
            gen_path.write_text(json.dumps(gen_spec, indent=2, ensure_ascii=False) + "\n")
            print(f"Wrote updated generated spec to {gen_path}")
    else:
        print("\nDry run complete. Use --apply to write changes.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
