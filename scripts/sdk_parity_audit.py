#!/usr/bin/env python3
"""
Generate an SDK parity report by comparing SDK endpoints to OpenAPI.

Outputs a Markdown report suitable for docs/SDK_PARITY.md.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator

HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}


@dataclass(frozen=True)
class Endpoint:
    method: str
    path: str

    def normalized(self) -> Endpoint:
        return Endpoint(self.method, normalize_path(self.path))

    def display(self) -> str:
        return f"{self.method} {self.path}"


def normalize_path(path: str) -> str:
    path = path.split("?", 1)[0]
    if (
        path.startswith("/api/")
        and not path.startswith("/api/v1/")
        and not path.startswith("/api/v2/")
    ):
        path = path.replace("/api/", "/api/v1/", 1)
    return re.sub(r"{[^}]+}", "{param}", path)


def normalize_template(path: str) -> str:
    # Strip query string fragments embedded via template literals
    if "?" in path:
        path = path.split("?", 1)[0]
    # Drop common query-string template suffixes (e.g., `${params}`, `${query}`)
    path = re.sub(r"\$\{(params|query|queryString|qs|searchParams|options|filters)\}$", "", path)
    # Replace remaining interpolations with path params
    path = re.sub(r"\$\{[^}]+\}", "{param}", path)
    return normalize_path(path)


def load_openapi(path: Path) -> tuple[set[Endpoint], int]:
    data = json.loads(path.read_text())
    endpoints: set[Endpoint] = set()
    deprecated_count = 0
    for raw_path, methods in data.get("paths", {}).items():
        path_deprecated = isinstance(methods, dict) and methods.get("deprecated") is True
        for method, _ in methods.items():
            method_upper = method.upper()
            if method_upper in HTTP_METHODS:
                operation = methods[method]
                operation_deprecated = (
                    isinstance(operation, dict) and operation.get("deprecated") is True
                )
                if path_deprecated or operation_deprecated:
                    deprecated_count += 1
                    continue
                endpoints.add(Endpoint(method_upper, raw_path).normalized())
    return endpoints, deprecated_count


def openapi_response_schema_endpoints(spec: dict[str, object]) -> set[Endpoint]:
    """Return endpoints that define a 2xx response schema (or 204)."""

    def has_schema(response: dict[str, object]) -> bool:
        content = response.get("content")
        if not isinstance(content, dict):
            return False
        for payload in content.values():
            if isinstance(payload, dict) and payload.get("schema"):
                return True
        return False

    coverage: dict[Endpoint, bool] = {}
    paths = spec.get("paths", {})
    if not isinstance(paths, dict):
        return set()
    for raw_path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        path_deprecated = methods.get("deprecated") is True
        for method, operation in methods.items():
            method_upper = str(method).upper()
            if method_upper not in HTTP_METHODS:
                continue
            if not isinstance(operation, dict):
                continue
            if path_deprecated or operation.get("deprecated") is True:
                continue
            responses = operation.get("responses", {})
            if not isinstance(responses, dict):
                continue
            success_codes = [code for code in responses if str(code).startswith("2")]
            ok = False
            for code in success_codes:
                response = responses.get(code, {})
                if str(code) == "204":
                    ok = True
                    break
                if isinstance(response, dict) and has_schema(response):
                    ok = True
                    break
            key = Endpoint(method_upper, raw_path).normalized()
            if key in coverage:
                coverage[key] = coverage[key] and ok
            else:
                coverage[key] = ok
    return {endpoint for endpoint, ok in coverage.items() if ok}


REQUEST_RE = re.compile(
    r"\brequest[^\n]*?\(\s*['\"](?P<method>GET|POST|PUT|PATCH|DELETE)['\"]\s*,\s*(?P<path>`[^`]+`|'[^']+'|\"[^\"]+\")",
    re.MULTILINE,
)


def parse_ts_sdk(paths: Iterable[Path]) -> set[Endpoint]:
    endpoints: set[Endpoint] = set()
    for path in paths:
        text = path.read_text()
        for pattern in (REQUEST_RE, CLIENT_CALL_RE):
            for match in pattern.finditer(text):
                method = match.group("method").upper()
                raw = match.group("path")
                literal = raw[1:-1]
                if raw.startswith("`"):
                    literal = normalize_template(literal)
                else:
                    literal = normalize_path(literal)
                endpoints.add(Endpoint(method, literal).normalized())
    return endpoints


CLIENT_CALL_RE = re.compile(
    r"\b(?:this\.)?client\.(?P<method>get|post|put|patch|delete)\b[^\n,]*\(\s*(?P<path>`[^`]+`|'[^']+'|\"[^\"]+\")",
    re.MULTILINE,
)


def parse_ts_client(paths: Iterable[Path]) -> set[Endpoint]:
    endpoints: set[Endpoint] = set()
    for path in paths:
        text = path.read_text()
        for pattern in (REQUEST_RE, CLIENT_CALL_RE):
            for match in pattern.finditer(text):
                method = match.group("method").upper()
                raw = match.group("path")
                literal = raw[1:-1]
                if raw.startswith("`"):
                    literal = normalize_template(literal)
                else:
                    literal = normalize_path(literal)
                endpoints.add(Endpoint(method, literal).normalized())
    return endpoints


# Pattern for _client._get("path") style (legacy)
PY_CALL_LEGACY_RE = re.compile(
    r"\b_client\._(?P<method>get|post|put|patch|delete)\(\s*(?P<path>f?\"[^\"]+\"|f?'[^']+')",
    re.MULTILINE,
)

# Pattern for _client.request("GET", "path") and _client._request("GET", "path") styles
PY_CALL_RE = re.compile(
    r"\b_client\._?request\(\s*['\"](?P<method>GET|POST|PUT|PATCH|DELETE)['\"][,\s]*(?P<path>f?\"[^\"]+\"|f?'[^']+')",
    re.MULTILINE,
)


def parse_python_sdk(paths: Iterable[Path]) -> set[Endpoint]:
    endpoints: set[Endpoint] = set()
    for path in paths:
        text = path.read_text()
        # Match both legacy (_client._get) and current (_client.request) patterns
        for pattern in [PY_CALL_RE, PY_CALL_LEGACY_RE]:
            for match in pattern.finditer(text):
                method = match.group("method").upper()
                raw = match.group("path")
                if raw.startswith("f"):
                    literal = raw[2:-1]  # Strip f" and trailing "
                    literal = re.sub(r"{[^}]+}", "{param}", literal)
                else:
                    literal = raw[1:-1]  # Strip " and trailing "
                literal = normalize_path(literal)
                endpoints.add(Endpoint(method, literal).normalized())
    return endpoints


def iter_files(root: Path, suffix: str) -> Iterator[Path]:
    if not root.exists():
        return iter(())
    return (p for p in root.rglob(f"*{suffix}") if p.is_file())


def categorize(path: str) -> str:
    parts = [p for p in path.strip("/").split("/") if p]
    if not parts:
        return "root"
    if parts[0] == "api":
        parts = parts[1:]
    if parts and parts[0] in ("v1", "v2"):
        parts = parts[1:]
    if not parts:
        return "api"
    return parts[0].replace("-", "_")


def coverage_by_category(
    openapi: set[Endpoint], sdk: set[Endpoint]
) -> list[tuple[str, int, int, float]]:
    openapi_by_cat: dict[str, set[Endpoint]] = defaultdict(set)
    for ep in openapi:
        openapi_by_cat[categorize(ep.path)].add(ep)

    rows: list[tuple[str, int, int, float]] = []
    for category, endpoints in openapi_by_cat.items():
        covered = endpoints & sdk
        coverage = (len(covered) / len(endpoints)) * 100 if endpoints else 0.0
        rows.append((category, len(endpoints), len(covered), coverage))
    rows.sort(key=lambda row: row[1], reverse=True)
    return rows


def format_table(rows: list[tuple[str, int, int, float]], label: str) -> str:
    lines = [
        f"| Category | OpenAPI | {label} Covered | {label} Coverage |",
        "|----------|---------|----------------|------------------|",
    ]
    for category, total, covered, coverage in rows:
        lines.append(f"| {category} | {total} | {covered} | {coverage:.1f}% |")
    return "\n".join(lines)


def format_endpoint_list(title: str, endpoints: list[Endpoint], limit: int = 40) -> str:
    lines = [f"### {title}", ""]
    if not endpoints:
        lines.append("_None_")
        return "\n".join(lines)
    lines.append("<details>")
    lines.append("<summary>Show endpoints</summary>")
    lines.append("")
    lines.append("```text")
    for ep in endpoints[:limit]:
        lines.append(ep.display())
    if len(endpoints) > limit:
        lines.append(f"... +{len(endpoints) - limit} more")
    lines.append("```")
    lines.append("</details>")
    return "\n".join(lines)


def render_report(
    openapi: set[Endpoint],
    ts_sdk: set[Endpoint],
    py_sdk: set[Endpoint],
    ts_client: set[Endpoint],
    deprecated_count: int,
) -> str:
    openapi_total = len(openapi)
    ts_total = len(ts_sdk)
    py_total = len(py_sdk)
    ts_covered = len(openapi & ts_sdk)
    py_covered = len(openapi & py_sdk)
    ts_coverage = (ts_covered / openapi_total) * 100 if openapi_total else 0.0
    py_coverage = (py_covered / openapi_total) * 100 if openapi_total else 0.0

    ts_extras = sorted(ts_sdk - openapi, key=lambda ep: ep.display())
    py_extras = sorted(py_sdk - openapi, key=lambda ep: ep.display())
    ts_missing = sorted(openapi - ts_sdk, key=lambda ep: ep.display())
    py_missing = sorted(openapi - py_sdk, key=lambda ep: ep.display())

    ts_rows = coverage_by_category(openapi, ts_sdk)
    py_rows = coverage_by_category(openapi, py_sdk)

    lines = [
        "# SDK-API Parity Report",
        "",
        f"Generated: {date.today().isoformat()}",
        "OpenAPI Source: `docs/api/openapi.json`",
        "Generated by: `scripts/sdk_parity_audit.py`",
        "",
        "## Executive Summary",
        "",
        "| Metric | OpenAPI | sdk/typescript | aragora-py |",
        "|--------|---------|----------------|------------|",
        f"| Total endpoints | {openapi_total} | {ts_total} | {py_total} |",
        f"| Covered endpoints | - | {ts_covered} | {py_covered} |",
        f"| Coverage | - | {ts_coverage:.1f}% | {py_coverage:.1f}% |",
        f"| Deprecated endpoints (excluded) | {deprecated_count} | - | - |",
        "",
        "## Scope Notes",
        "",
        "- Coverage excludes deprecated endpoints from the OpenAPI total.",
        "- `/api` is normalized to `/api/v1` for SDK comparisons.",
        "- `sdk/typescript` primarily targets `/api` with `/api/v1` aliases.",
        "- `aragora-py` targets `/api/v1` endpoints.",
        "- `aragora-js` is a lightweight `/api/v1` client (reported below for reference).",
        "",
        "## Coverage by Category (sdk/typescript)",
        "",
        format_table(ts_rows, "TS"),
        "",
        "## Coverage by Category (aragora-py)",
        "",
        format_table(py_rows, "PY"),
        "",
        format_endpoint_list("Missing in sdk/typescript (OpenAPI - TS)", ts_missing),
        "",
        format_endpoint_list("Missing in aragora-py (OpenAPI - PY)", py_missing),
        "",
        format_endpoint_list("sdk/typescript endpoints not in OpenAPI", ts_extras),
        "",
        format_endpoint_list("aragora-py endpoints not in OpenAPI", py_extras),
        "",
        "## Reference: aragora-js (Lightweight /api/v1 client)",
        "",
    ]

    if py_total and py_coverage == 0.0:
        insert_at = lines.index("## Coverage by Category (sdk/typescript)")
        lines[insert_at:insert_at] = [
            "- Note: `/api/v1` debate endpoints are not fully represented in the OpenAPI spec, "
            "so coverage for `aragora-py` will read low until those paths are added.",
            "",
        ]

    if ts_client:
        lines.append(f"- Endpoint count: {len(ts_client)}")
    else:
        lines.append("- Endpoint count: 0 (not scanned)")

    return "\n".join(lines).rstrip() + "\n"


def render_stability_manifest(
    stable: set[Endpoint],
    *,
    criteria: str,
) -> dict[str, object]:
    entries = sorted(
        (ep.normalized().display() for ep in stable),
        key=lambda item: (item.split(" ", 1)[0], item.split(" ", 1)[1]),
    )
    return {
        "generated_at": date.today().isoformat(),
        "criteria": criteria,
        "stable": entries,
    }


def _load_stability_overrides(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    overrides: dict[str, list[str]] = {}
    if isinstance(data, dict):
        for key in ("beta", "internal"):
            value = data.get(key)
            if isinstance(value, list):
                overrides[key] = [v for v in value if isinstance(v, str)]
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SDK parity report")
    parser.add_argument("--openapi", default="docs/api/openapi.json")
    parser.add_argument("--ts-sdk", default="sdk/typescript/src")
    parser.add_argument("--py-sdk", default="sdk/python/aragora/namespaces")
    parser.add_argument("--ts-client", default="aragora-js/src")
    parser.add_argument("--output", help="Write report to file")
    parser.add_argument(
        "--stable-out",
        help="Write a stability manifest (OpenAPI endpoints covered by SDKs) to file",
    )
    parser.add_argument(
        "--stable-require",
        choices=("both", "either", "ts", "py"),
        default="both",
        help="SDK coverage requirement for stable endpoints (default: both)",
    )
    args = parser.parse_args()

    openapi_path = Path(args.openapi)
    ts_sdk_dir = Path(args.ts_sdk)
    py_sdk_dir = Path(args.py_sdk)
    ts_client_dir = Path(args.ts_client)

    openapi, deprecated_count = load_openapi(openapi_path)
    ts_sdk = parse_ts_sdk(iter_files(ts_sdk_dir, ".ts"))
    py_sdk = parse_python_sdk(iter_files(py_sdk_dir, ".py"))
    ts_client = parse_ts_client(iter_files(ts_client_dir, ".ts"))

    report = render_report(openapi, ts_sdk, py_sdk, ts_client, deprecated_count)

    if args.output:
        Path(args.output).write_text(report)
        print(f"Wrote {args.output}")
    else:
        print(report)

    if args.stable_out:
        if args.stable_require == "both":
            stable = openapi & ts_sdk & py_sdk
            criteria = "OpenAPI endpoints covered by both TypeScript and Python SDKs"
        elif args.stable_require == "either":
            stable = openapi & (ts_sdk | py_sdk)
            criteria = "OpenAPI endpoints covered by either TypeScript or Python SDK"
        elif args.stable_require == "ts":
            stable = openapi & ts_sdk
            criteria = "OpenAPI endpoints covered by TypeScript SDK"
        else:
            stable = openapi & py_sdk
            criteria = "OpenAPI endpoints covered by Python SDK"
        spec = json.loads(openapi_path.read_text())
        schema_covered = openapi_response_schema_endpoints(spec)
        stable = stable & schema_covered
        criteria = f"{criteria}; includes 2xx response schema coverage"
        manifest = render_stability_manifest(stable, criteria=criteria)
        overrides = _load_stability_overrides(Path(args.stable_out))
        for key, values in overrides.items():
            manifest[key] = values
        Path(args.stable_out).write_text(json.dumps(manifest, indent=2))
        print(f"Wrote {args.stable_out}")


if __name__ == "__main__":
    main()
