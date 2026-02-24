"""
Auto-discovered SDK contract matrix tests.

Validates that every SDK namespace file (Python and TypeScript) references only
endpoints that exist in the OpenAPI spec, and that both SDKs provide equivalent
namespace coverage.

Stage 4 (#175): API/SDK contract hardening.

Budget mechanism: TS SDK namespaces with known OpenAPI spec gaps are tracked
in a budget file.  The test xfails for namespaces listed in the budget,
and hard-fails if a previously-passing namespace regresses.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}

# -- Endpoint extraction patterns --

# Python: self._client.request("METHOD", "/api/...") or self._client._request(...)
PY_REQUEST_RE = re.compile(
    r'self\._client\._?request\(\s*["\'](?P<method>GET|POST|PUT|PATCH|DELETE)["\']'
    r'\s*,\s*(?:f?["\'])(?P<path>/api/[^"\']+)["\']'
)

# TypeScript: this.client.request('METHOD', `path`) or this.client.request<Type>('METHOD', `path`)
TS_REQUEST_RE = re.compile(
    r"this\.client\.request(?:<[^(]+>)?\(\s*['\"](?P<method>[A-Z]+)['\"]\s*,"
    r"\s*(?P<path>`[^`]+`|'[^']+'|\"[^\"]+\")"
)
# TypeScript: this.client.get(`path`)
TS_DIRECT_RE = re.compile(
    r"this\.client\.(?P<method>get|post|put|delete|patch)\("
    r"\s*(?P<path>`[^`]+`|'[^']+'|\"[^\"]+\")"
)


def _repo_root() -> Path:
    # Try __file__ first, fall back to CWD if the expected marker doesn't exist
    root = Path(__file__).resolve().parents[3]
    if (root / "pyproject.toml").exists():
        return root
    # Fallback: CWD should be the repo root in CI
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists():
        return cwd
    return root


def _normalize_path(path: str) -> str:
    """Normalize an API path for comparison."""
    path = path.split("?", 1)[0]
    # Normalize template parameters
    path = re.sub(r"\$\{[^}]+\}", "{param}", path)
    path = re.sub(r"\{[^}]+\}", "{param}", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return path


def _extract_py_endpoints(content: str) -> set[tuple[str, str]]:
    """Extract endpoint references from a Python SDK namespace file."""
    endpoints: set[tuple[str, str]] = set()
    for match in PY_REQUEST_RE.finditer(content):
        method = match.group("method").lower()
        raw_path = match.group("path")
        # Remove f-string expressions: {var} → {param}
        normalized = _normalize_path(raw_path)
        if normalized.startswith("/api/"):
            endpoints.add((method, normalized))
    return endpoints


def _extract_ts_endpoints(content: str) -> set[tuple[str, str]]:
    """Extract endpoint references from a TypeScript SDK namespace file."""
    endpoints: set[tuple[str, str]] = set()
    for match in TS_REQUEST_RE.finditer(content):
        method = match.group("method").lower()
        raw_path = match.group("path")[1:-1]  # Strip quotes/backticks
        endpoints.add((method, _normalize_path(raw_path)))
    for match in TS_DIRECT_RE.finditer(content):
        method = match.group("method").lower()
        raw_path = match.group("path")[1:-1]
        endpoints.add((method, _normalize_path(raw_path)))
    return endpoints


# -- Namespace discovery --

# Namespaces to skip (utility files, not actual API namespaces)
PY_SKIP = {"__init__", "__pycache__"}
TS_SKIP = {"__tests__", "index", "types", "utils", "base"}


def _discover_py_namespaces() -> list[str]:
    ns_dir = _repo_root() / "sdk/python/aragora_sdk/namespaces"
    if not ns_dir.is_dir():
        return []
    return sorted(
        p.stem for p in ns_dir.glob("*.py") if p.stem not in PY_SKIP and not p.stem.startswith("_")
    )


def _discover_ts_namespaces() -> list[str]:
    ns_dir = _repo_root() / "sdk/typescript/src/namespaces"
    if not ns_dir.is_dir():
        return []
    return sorted(
        p.stem for p in ns_dir.glob("*.ts") if p.stem not in TS_SKIP and not p.stem.startswith("_")
    )


# -- Fixtures --


@pytest.fixture(scope="module")
def openapi_spec() -> dict:
    spec_path = _repo_root() / "docs/api/openapi.json"
    assert spec_path.exists(), f"docs/api/openapi.json not found (tried {spec_path})"
    return json.loads(spec_path.read_text())


@pytest.fixture(scope="module")
def openapi_endpoints(openapi_spec: dict) -> set[tuple[str, str]]:
    endpoints: set[tuple[str, str]] = set()
    for path, operations in openapi_spec.get("paths", {}).items():
        for method in operations:
            if method.lower() in HTTP_METHODS:
                endpoints.add((method.lower(), _normalize_path(path)))
    return endpoints


# -- Budget loading --


def _load_py_budget() -> set[str]:
    """Load Python SDK namespaces with known OpenAPI gaps from the budget file."""
    budget_path = _repo_root() / "scripts/baselines/contract_matrix_py_budget.json"
    if not budget_path.exists():
        return set()
    data = json.loads(budget_path.read_text())
    return set(data.get("namespaces_with_gaps", []))


def _load_ts_budget() -> set[str]:
    """Load TS SDK namespaces with known OpenAPI gaps from the budget file."""
    budget_path = _repo_root() / "scripts/baselines/contract_matrix_ts_budget.json"
    if not budget_path.exists():
        return set()
    data = json.loads(budget_path.read_text())
    return set(data.get("namespaces_with_gaps", []))


_PY_BUDGET_NAMESPACES = _load_py_budget()
_TS_BUDGET_NAMESPACES = _load_ts_budget()


# -- Contract tests --


@pytest.mark.parametrize("namespace", _discover_py_namespaces())
def test_python_sdk_endpoints_in_openapi(
    namespace: str, openapi_endpoints: set[tuple[str, str]]
) -> None:
    """Every endpoint referenced by a Python SDK namespace must exist in OpenAPI."""
    ns_file = _repo_root() / "sdk/python/aragora_sdk/namespaces" / f"{namespace}.py"
    if not ns_file.exists():
        pytest.skip(f"Namespace file not found: {namespace}.py")
    content = ns_file.read_text()
    sdk_eps = _extract_py_endpoints(content)
    if not sdk_eps:
        pytest.skip(f"No endpoints extracted from {namespace}.py")
    missing = sorted(sdk_eps - openapi_endpoints)
    if missing and namespace in _PY_BUDGET_NAMESPACES:
        pytest.xfail(f"Known gap: '{namespace}' has {len(missing)} endpoints not in OpenAPI spec")
    assert not missing, (
        f"Python SDK namespace '{namespace}' references endpoints not in OpenAPI spec: {missing}"
    )


@pytest.mark.parametrize("namespace", _discover_ts_namespaces())
def test_typescript_sdk_endpoints_in_openapi(
    namespace: str, openapi_endpoints: set[tuple[str, str]]
) -> None:
    """Every endpoint referenced by a TypeScript SDK namespace must exist in OpenAPI."""
    ns_file = _repo_root() / "sdk/typescript/src/namespaces" / f"{namespace}.ts"
    if not ns_file.exists():
        pytest.skip(f"Namespace file not found: {namespace}.ts")
    content = ns_file.read_text()
    sdk_eps = _extract_ts_endpoints(content)
    if not sdk_eps:
        pytest.skip(f"No endpoints extracted from {namespace}.ts")
    missing = sorted(sdk_eps - openapi_endpoints)
    if missing and namespace in _TS_BUDGET_NAMESPACES:
        pytest.xfail(f"Known gap: '{namespace}' has {len(missing)} endpoints not in OpenAPI spec")
    assert not missing, (
        f"TypeScript SDK namespace '{namespace}' references endpoints not in OpenAPI spec: {missing}"
    )


def test_sdk_parity_namespace_coverage() -> None:
    """Both SDKs should cover the same set of namespaces (by name, kebab→snake)."""
    py_ns = set(_discover_py_namespaces())
    ts_ns = {name.replace("-", "_") for name in _discover_ts_namespaces()}

    py_only = sorted(py_ns - ts_ns)
    ts_only = sorted(ts_ns - py_ns)

    # Allow up to 5 mismatches (some platform-specific namespaces)
    max_drift = 5
    assert len(py_only) <= max_drift, (
        f"Python SDK has {len(py_only)} namespaces not in TypeScript SDK: {py_only}"
    )
    assert len(ts_only) <= max_drift, (
        f"TypeScript SDK has {len(ts_only)} namespaces not in Python SDK: {ts_only}"
    )


def test_stability_manifest_endpoints_exist(
    openapi_endpoints: set[tuple[str, str]],
) -> None:
    """Every endpoint in the stability manifest must still exist in the spec."""
    manifest_path = _repo_root() / "aragora/server/openapi/stability_manifest.json"
    if not manifest_path.exists():
        pytest.skip(f"stability_manifest.json not found (tried {manifest_path})")
    manifest = json.loads(manifest_path.read_text())
    stable = manifest.get("stable", [])

    missing = []
    for entry in stable:
        parts = entry.split(" ", 1)
        if len(parts) != 2:
            continue
        method, path = parts[0].lower(), _normalize_path(parts[1])
        if (method, path) not in openapi_endpoints:
            missing.append(entry)

    assert not missing, (
        f"{len(missing)} stable endpoints missing from OpenAPI spec (first 10): {missing[:10]}"
    )


def test_stability_manifest_minimum_count(
    openapi_endpoints: set[tuple[str, str]],
) -> None:
    """Stability manifest should have a minimum baseline of stable endpoints."""
    manifest_path = _repo_root() / "aragora/server/openapi/stability_manifest.json"
    if not manifest_path.exists():
        pytest.skip(f"stability_manifest.json not found (tried {manifest_path})")
    manifest = json.loads(manifest_path.read_text())
    stable_count = len(manifest.get("stable", []))

    # Should never drop below 400 stable endpoints
    assert stable_count >= 400, (
        f"Stability manifest has only {stable_count} endpoints (minimum: 400). "
        "Stable endpoints should not be removed without API deprecation process."
    )
