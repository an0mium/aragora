#!/usr/bin/env python3
"""
Verify SDK contracts against OpenAPI specification.

Checks that all SDK namespace endpoints exist in the OpenAPI spec and
reports coverage metrics. Used as a CI gate for SDK generation.

Usage:
    python scripts/verify_sdk_contracts.py
    python scripts/verify_sdk_contracts.py --strict  # Fail on any drift
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}

PY_REQUEST_RE = re.compile(
    r'self\._client\._request\(\s*["\'](?P<method>GET|POST|PUT|PATCH|DELETE)["\']'
    r'\s*,\s*(?:f?["\'])(?P<path>/api/[^"\']+)["\']'
)
TS_REQUEST_RE = re.compile(
    r"this\.client\.request\(\s*['\"](?P<method>[A-Z]+)['\"]\s*,"
    r"\s*(?P<path>`[^`]+`|'[^']+'|\"[^\"]+\")"
)
TS_DIRECT_RE = re.compile(
    r"this\.client\.(?P<method>get|post|put|delete|patch)\("
    r"\s*(?P<path>`[^`]+`|'[^']+'|\"[^\"]+\")"
)


def _normalize(path: str) -> str:
    path = path.split("?", 1)[0]
    path = re.sub(r"\$\{[^}]+\}", "{param}", path)
    path = re.sub(r"\{[^}]+\}", "{param}", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return path


def _load_openapi_endpoints(spec_path: Path) -> set[tuple[str, str]]:
    spec = json.loads(spec_path.read_text())
    endpoints: set[tuple[str, str]] = set()
    for path, ops in spec.get("paths", {}).items():
        for method in ops:
            if method.lower() in HTTP_METHODS:
                endpoints.add((method.lower(), _normalize(path)))
    return endpoints


def _extract_py(content: str) -> set[tuple[str, str]]:
    eps: set[tuple[str, str]] = set()
    for m in PY_REQUEST_RE.finditer(content):
        path = _normalize(m.group("path"))
        if path.startswith("/api/"):
            eps.add((m.group("method").lower(), path))
    return eps


def _extract_ts(content: str) -> set[tuple[str, str]]:
    eps: set[tuple[str, str]] = set()
    for m in TS_REQUEST_RE.finditer(content):
        eps.add((m.group("method").lower(), _normalize(m.group("path")[1:-1])))
    for m in TS_DIRECT_RE.finditer(content):
        eps.add((m.group("method").lower(), _normalize(m.group("path")[1:-1])))
    return eps


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SDK contracts against OpenAPI.")
    parser.add_argument("--strict", action="store_true", help="Fail on any drift")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    spec_path = repo / "docs/api/openapi.json"
    if not spec_path.exists():
        print("ERROR: docs/api/openapi.json not found", file=sys.stderr)
        return 1

    openapi_eps = _load_openapi_endpoints(spec_path)
    print(f"OpenAPI spec: {len(openapi_eps)} endpoints")

    # Check Python SDK
    py_dir = repo / "sdk/python/aragora_sdk/namespaces"
    py_ns = sorted(p.stem for p in py_dir.glob("*.py") if not p.stem.startswith("_"))
    py_total = 0
    py_drift: list[tuple[str, str, str]] = []

    for ns in py_ns:
        content = (py_dir / f"{ns}.py").read_text()
        eps = _extract_py(content)
        py_total += len(eps)
        for ep in sorted(eps - openapi_eps):
            py_drift.append((ns, ep[0].upper(), ep[1]))

    # Check TypeScript SDK
    ts_dir = repo / "sdk/typescript/src/namespaces"
    ts_ns = sorted(p.stem for p in ts_dir.glob("*.ts") if not p.stem.startswith("_"))
    ts_total = 0
    ts_drift: list[tuple[str, str, str]] = []

    for ns in ts_ns:
        content = (ts_dir / f"{ns}.ts").read_text()
        eps = _extract_ts(content)
        ts_total += len(eps)
        for ep in sorted(eps - openapi_eps):
            ts_drift.append((ns, ep[0].upper(), ep[1]))

    # Parity check
    py_ns_set = set(py_ns)
    ts_ns_set = {name.replace("-", "_") for name in ts_ns}
    py_only = sorted(py_ns_set - ts_ns_set)
    ts_only = sorted(ts_ns_set - py_ns_set)

    # Report
    print(f"\nPython SDK:     {len(py_ns)} namespaces, {py_total} endpoint references")
    print(f"TypeScript SDK: {len(ts_ns)} namespaces, {ts_total} endpoint references")

    print(f"\nParity: Python-only={len(py_only)}, TypeScript-only={len(ts_only)}")
    if py_only:
        print(f"  Python-only namespaces: {py_only[:10]}")
    if ts_only:
        print(f"  TypeScript-only namespaces: {ts_only[:10]}")

    has_drift = False
    if py_drift:
        print(f"\nPython SDK drift ({len(py_drift)} endpoints not in spec):")
        for ns, method, path in py_drift[:20]:
            print(f"  {ns}: {method} {path}")
        has_drift = True

    if ts_drift:
        print(f"\nTypeScript SDK drift ({len(ts_drift)} endpoints not in spec):")
        for ns, method, path in ts_drift[:20]:
            print(f"  {ns}: {method} {path}")
        has_drift = True

    # Stability manifest check
    manifest_path = repo / "aragora/server/openapi/stability_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        stable = manifest.get("stable", [])
        missing_stable = []
        for entry in stable:
            parts = entry.split(" ", 1)
            if len(parts) == 2:
                method, path = parts[0].lower(), _normalize(parts[1])
                if (method, path) not in openapi_eps:
                    missing_stable.append(entry)
        print(
            f"\nStability manifest: {len(stable)} stable, {len(missing_stable)} missing from spec"
        )
        if missing_stable:
            for entry in missing_stable[:10]:
                print(f"  MISSING: {entry}")
            has_drift = True

    if not has_drift:
        print("\nAll SDK contracts verified!")
        return 0

    if args.strict:
        print("\nFAILED: SDK/API drift detected (--strict mode)")
        return 1

    print("\nWARNING: SDK/API drift detected (use --strict to fail)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
