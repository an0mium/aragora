"""Contract parity tests for the Dashboard SDK namespace.

Verifies that Python SDK and TypeScript SDK define the same dashboard
endpoints, ensuring contract consistency across SDK surfaces.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# =========================================================================
# Canonical endpoint definitions (source of truth: server handler routes)
# =========================================================================

DASHBOARD_ENDPOINTS: list[tuple[str, str]] = [
    # Core dashboard
    ("GET", "/api/v1/dashboard"),
    ("GET", "/api/v1/dashboard/overview"),
    ("GET", "/api/v1/dashboard/stats"),
    ("GET", "/api/v1/dashboard/activity"),
    ("GET", "/api/v1/dashboard/inbox-summary"),
    ("GET", "/api/v1/dashboard/quick-actions"),
    ("POST", "/api/v1/dashboard/quick-actions/{id}"),
    ("GET", "/api/v1/dashboard/debates"),
    ("GET", "/api/v1/dashboard/debates/{id}"),
    ("GET", "/api/v1/dashboard/stat-cards"),
    ("GET", "/api/v1/dashboard/team-performance"),
    ("GET", "/api/v1/dashboard/team-performance/{id}"),
    ("GET", "/api/v1/dashboard/top-senders"),
    ("GET", "/api/v1/dashboard/labels"),
    ("GET", "/api/v1/dashboard/urgent"),
    ("POST", "/api/v1/dashboard/urgent/{id}/dismiss"),
    ("GET", "/api/v1/dashboard/pending-actions"),
    ("POST", "/api/v1/dashboard/pending-actions/{id}/complete"),
    ("GET", "/api/v1/dashboard/search"),
    ("POST", "/api/v1/dashboard/export"),
    # Gastown dashboard
    ("GET", "/api/v1/dashboard/gastown/overview"),
    ("GET", "/api/v1/dashboard/gastown/agents"),
    ("GET", "/api/v1/dashboard/gastown/beads"),
    ("GET", "/api/v1/dashboard/gastown/convoys"),
    ("GET", "/api/v1/dashboard/gastown/metrics"),
    # Outcome dashboard
    ("GET", "/api/v1/outcome-dashboard"),
    ("GET", "/api/v1/outcome-dashboard/quality"),
    ("GET", "/api/v1/outcome-dashboard/agents"),
    ("GET", "/api/v1/outcome-dashboard/history"),
    ("GET", "/api/v1/outcome-dashboard/calibration"),
    # Usage dashboard
    ("GET", "/api/v1/usage/summary"),
    ("GET", "/api/v1/usage/breakdown"),
    ("GET", "/api/v1/usage/roi"),
    ("GET", "/api/v1/usage/export"),
    ("GET", "/api/v1/usage/budget-status"),
    # Spend analytics dashboard
    ("GET", "/api/v1/analytics/spend/summary"),
    ("GET", "/api/v1/analytics/spend/trends"),
    ("GET", "/api/v1/analytics/spend/by-agent"),
    ("GET", "/api/v1/analytics/spend/by-decision"),
    ("GET", "/api/v1/analytics/spend/budget"),
]


def _normalize_path(path: str) -> str:
    """Normalize a path by replacing specific IDs with {id}."""
    path = re.sub(r"\{[a-z_]+\}", "{id}", path)
    path = re.sub(r"\{self\.[a-z_]+\}", "{id}", path)
    return path


def _extract_python_sdk_paths(filepath: Path) -> list[tuple[str, str]]:
    """Extract (method, path) tuples from a Python SDK namespace file."""
    source = filepath.read_text()
    tree = ast.parse(source)
    endpoints = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "request":
            continue
        if len(node.args) < 2:
            continue
        method_node = node.args[0]
        path_node = node.args[1]

        if isinstance(method_node, ast.Constant) and isinstance(method_node.value, str):
            method = method_node.value
        else:
            continue

        if isinstance(path_node, ast.Constant) and isinstance(path_node.value, str):
            path = path_node.value
        elif isinstance(path_node, ast.JoinedStr):
            parts = []
            for v in path_node.values:
                if isinstance(v, ast.Constant):
                    parts.append(v.value)
                else:
                    parts.append("{id}")
            path = "".join(parts)
        else:
            continue

        endpoints.append((method, _normalize_path(path)))

    return endpoints


def _extract_ts_paths(filepath: Path) -> list[tuple[str, str]]:
    """Extract (method, path) tuples from a TypeScript namespace file."""
    source = filepath.read_text()
    endpoints = []

    pattern = re.compile(
        r"this\.client\.request[^(]*\(\s*['\"](\w+)['\"],\s*"
        r"(?:['\"]([^'\"]+)['\"]|`([^`]+)`)"
    )

    for match in pattern.finditer(source):
        method = match.group(1)
        path = match.group(2) or match.group(3)
        path = re.sub(r"\$\{[^}]+\}", "{id}", path)
        endpoints.append((method, _normalize_path(path)))

    return endpoints


# =========================================================================
# Dashboard parity tests
# =========================================================================


class TestDashboardContractParity:
    """Verify all SDK surfaces match the server's Dashboard endpoints."""

    def _unique_endpoints(self, endpoints: list[tuple[str, str]]) -> set[tuple[str, str]]:
        return set(endpoints)

    def test_python_sdk_covers_all_endpoints(self):
        """Python SDK dashboard covers every server Dashboard endpoint."""
        sdk_file = ROOT / "sdk/python/aragora_sdk/namespaces/dashboard.py"
        sdk_paths = self._unique_endpoints(_extract_python_sdk_paths(sdk_file))
        canonical = set(DASHBOARD_ENDPOINTS)

        missing = canonical - sdk_paths
        assert not missing, (
            f"Python SDK dashboard missing {len(missing)} endpoints:\n"
            + "\n".join(f"  {m} {p}" for m, p in sorted(missing))
        )

    def test_typescript_sdk_covers_all_endpoints(self):
        """TypeScript SDK dashboard covers every server Dashboard endpoint."""
        ts_file = ROOT / "sdk/typescript/src/namespaces/dashboard.ts"
        ts_paths = self._unique_endpoints(_extract_ts_paths(ts_file))
        canonical = set(DASHBOARD_ENDPOINTS)

        missing = canonical - ts_paths
        assert not missing, (
            f"TypeScript SDK dashboard missing {len(missing)} endpoints:\n"
            + "\n".join(f"  {m} {p}" for m, p in sorted(missing))
        )

    def test_python_sdk_method_count(self):
        """Python SDK dashboard has both sync and async classes with matching methods."""
        from aragora_sdk.namespaces.dashboard import AsyncDashboardAPI, DashboardAPI

        sync_methods = [m for m in dir(DashboardAPI) if not m.startswith("_")]
        async_methods = [m for m in dir(AsyncDashboardAPI) if not m.startswith("_")]

        assert set(sync_methods) == set(async_methods), (
            f"Sync/async method mismatch. "
            f"Only in sync: {set(sync_methods) - set(async_methods)}. "
            f"Only in async: {set(async_methods) - set(sync_methods)}"
        )

    def test_python_sdk_has_minimum_methods(self):
        """Python SDK dashboard has at least 35 methods (15+ new + 21 existing)."""
        from aragora_sdk.namespaces.dashboard import DashboardAPI

        methods = [m for m in dir(DashboardAPI) if not m.startswith("_")]
        assert len(methods) >= 35, (
            f"Expected at least 35 dashboard methods, got {len(methods)}: {methods}"
        )

    def test_typescript_sdk_has_minimum_methods(self):
        """TypeScript SDK dashboard has at least 35 methods."""
        ts_file = ROOT / "sdk/typescript/src/namespaces/dashboard.ts"
        ts_paths = _extract_ts_paths(ts_file)
        assert len(ts_paths) >= 35, (
            f"Expected at least 35 TypeScript SDK endpoints, got {len(ts_paths)}"
        )


# =========================================================================
# Cross-surface consistency
# =========================================================================


class TestDashboardCrossSurfaceConsistency:
    """Verify Python and TypeScript SDKs agree on dashboard paths."""

    def test_python_ts_path_agreement(self):
        """Python and TypeScript SDKs hit the same dashboard paths."""
        py_file = ROOT / "sdk/python/aragora_sdk/namespaces/dashboard.py"
        ts_file = ROOT / "sdk/typescript/src/namespaces/dashboard.ts"
        py_paths = set(_extract_python_sdk_paths(py_file))
        ts_paths = set(_extract_ts_paths(ts_file))

        only_in_py = py_paths - ts_paths
        only_in_ts = ts_paths - py_paths

        # Filter out known aliases/extras that don't indicate a gap
        # Python has get_overview (GET /api/v1/dashboard) + get_overview_page (GET /overview)
        # TypeScript has getDashboard (POST /api/v1/dashboard) as an extra
        allowed_py_only = {
            ("GET", "/api/v1/dashboard"),  # Python get_overview
        }
        allowed_ts_only = {
            ("POST", "/api/v1/dashboard"),  # TypeScript getDashboard
        }

        real_py_only = only_in_py - allowed_py_only
        real_ts_only = only_in_ts - allowed_ts_only

        assert not real_py_only, f"Paths only in Python SDK: {real_py_only}"
        assert not real_ts_only, f"Paths only in TypeScript SDK: {real_ts_only}"

    def test_endpoint_count_parity(self):
        """Python and TypeScript SDKs have similar endpoint counts (within 3)."""
        py_file = ROOT / "sdk/python/aragora_sdk/namespaces/dashboard.py"
        ts_file = ROOT / "sdk/typescript/src/namespaces/dashboard.ts"
        py_count = len(set(_extract_python_sdk_paths(py_file)))
        ts_count = len(set(_extract_ts_paths(ts_file)))

        diff = abs(py_count - ts_count)
        assert diff <= 3, (
            f"Endpoint count difference too large: Python={py_count}, TypeScript={ts_count} "
            f"(diff={diff})"
        )
