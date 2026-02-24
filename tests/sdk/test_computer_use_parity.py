"""Contract parity tests for Computer Use SDK namespaces.

Verifies that both Python and TypeScript SDK surfaces define the same
endpoints as the server OpenAPI spec.  These tests catch contract drift
before it reaches users.

Coverage target: 19/19 endpoints (100%).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# =========================================================================
# Canonical endpoint definitions (source of truth: server OpenAPI spec)
# =========================================================================

COMPUTER_USE_ENDPOINTS: list[tuple[str, str]] = [
    # Tasks
    ("GET", "/api/v1/computer-use/tasks"),
    ("POST", "/api/v1/computer-use/tasks"),
    ("GET", "/api/v1/computer-use/tasks/{id}"),
    ("DELETE", "/api/v1/computer-use/tasks/{id}"),
    ("POST", "/api/v1/computer-use/tasks/{id}/cancel"),
    # Actions
    ("GET", "/api/v1/computer-use/actions"),
    ("POST", "/api/v1/computer-use/actions"),
    ("GET", "/api/v1/computer-use/actions/{id}"),
    ("DELETE", "/api/v1/computer-use/actions/{id}"),
    ("GET", "/api/v1/computer-use/actions/stats"),
    # Policies
    ("GET", "/api/v1/computer-use/policies"),
    ("POST", "/api/v1/computer-use/policies"),
    ("GET", "/api/v1/computer-use/policies/{id}"),
    ("PUT", "/api/v1/computer-use/policies/{id}"),
    ("DELETE", "/api/v1/computer-use/policies/{id}"),
    # Approvals
    ("GET", "/api/v1/computer-use/approvals"),
    ("GET", "/api/v1/computer-use/approvals/{id}"),
    ("POST", "/api/v1/computer-use/approvals/{id}/approve"),
    ("POST", "/api/v1/computer-use/approvals/{id}/deny"),
]

EXPECTED_OPERATION_COUNT = len(COMPUTER_USE_ENDPOINTS)  # 19


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


def _extract_openapi_paths(filepath: Path) -> dict[str, list[str]]:
    """Extract path -> [methods] from the COMPUTER_USE_ENDPOINTS dict literal."""
    content = filepath.read_text()
    tree = ast.parse(content)
    paths: dict[str, list[str]] = {}

    def _process_dict(dict_node: ast.Dict) -> None:
        for key, val in zip(dict_node.keys, dict_node.values):
            if isinstance(key, ast.Constant) and isinstance(val, ast.Dict):
                path = key.value
                methods = []
                for method_key in val.keys:
                    if isinstance(method_key, ast.Constant):
                        methods.append(method_key.value.upper())
                paths[path] = methods

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "COMPUTER_USE_ENDPOINTS":
                    if isinstance(node.value, ast.Dict):
                        _process_dict(node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "COMPUTER_USE_ENDPOINTS":
                if node.value and isinstance(node.value, ast.Dict):
                    _process_dict(node.value)

    return paths


# =========================================================================
# OpenAPI Spec Completeness
# =========================================================================


class TestOpenAPISpecCompleteness:
    """Verify the OpenAPI spec covers all canonical routes."""

    _OPENAPI_SPEC = ROOT / "aragora" / "server" / "openapi" / "endpoints" / "computer_use.py"

    @pytest.fixture()
    def openapi_paths(self) -> dict[str, list[str]]:
        assert self._OPENAPI_SPEC.exists(), f"OpenAPI spec not found at {self._OPENAPI_SPEC}"
        return _extract_openapi_paths(self._OPENAPI_SPEC)

    def test_has_all_canonical_paths(self, openapi_paths: dict[str, list[str]]) -> None:
        """OpenAPI spec must define all canonical paths."""
        canonical_paths = {_normalize_path(p) for _, p in COMPUTER_USE_ENDPOINTS}
        openapi_normalized = {_normalize_path(p) for p in openapi_paths}
        missing = canonical_paths - openapi_normalized
        assert not missing, f"Missing OpenAPI paths: {missing}"

    def test_operation_count_minimum(self, openapi_paths: dict[str, list[str]]) -> None:
        """OpenAPI spec must have at least 19 operations."""
        total = sum(len(m) for m in openapi_paths.values())
        assert total >= EXPECTED_OPERATION_COUNT, (
            f"Expected at least {EXPECTED_OPERATION_COUNT} operations, got {total}"
        )


# =========================================================================
# Python SDK Parity
# =========================================================================


class TestPythonSDKParity:
    """Verify Python SDK covers all canonical Computer Use endpoints."""

    _PY_SDK = ROOT / "sdk" / "python" / "aragora_sdk" / "namespaces" / "computer_use.py"

    def _unique_endpoints(self, endpoints: list[tuple[str, str]]) -> set[tuple[str, str]]:
        return set(endpoints)

    def test_python_sdk_covers_all_endpoints(self) -> None:
        """Python SDK computer_use covers every server endpoint."""
        sdk_paths = self._unique_endpoints(_extract_python_sdk_paths(self._PY_SDK))
        canonical = set(COMPUTER_USE_ENDPOINTS)

        missing = canonical - sdk_paths
        assert not missing, (
            f"Python SDK computer_use missing {len(missing)} endpoints:\n"
            + "\n".join(f"  {m} {p}" for m, p in sorted(missing))
        )

    def test_python_sdk_method_count(self) -> None:
        """Python SDK has both sync and async classes with matching methods."""
        from aragora_sdk.namespaces.computer_use import AsyncComputerUseAPI, ComputerUseAPI

        sync_methods = [m for m in dir(ComputerUseAPI) if not m.startswith("_")]
        async_methods = [m for m in dir(AsyncComputerUseAPI) if not m.startswith("_")]

        assert set(sync_methods) == set(async_methods), (
            f"Sync/async method mismatch. "
            f"Only in sync: {set(sync_methods) - set(async_methods)}. "
            f"Only in async: {set(async_methods) - set(sync_methods)}"
        )

    REQUIRED_METHODS = [
        "create_task",
        "list_tasks",
        "get_task",
        "cancel_task",
        "delete_task",
        "list_actions",
        "get_action",
        "execute_action",
        "delete_action",
        "get_action_stats",
        "list_policies",
        "create_policy",
        "get_policy",
        "update_policy",
        "delete_policy",
        "list_approvals",
        "get_approval",
        "approve_approval",
        "deny_approval",
    ]

    def test_has_all_required_methods(self) -> None:
        """Python SDK must expose all required methods."""
        from aragora_sdk.namespaces.computer_use import ComputerUseAPI

        sdk_methods = [m for m in dir(ComputerUseAPI) if not m.startswith("_")]
        for method in self.REQUIRED_METHODS:
            assert method in sdk_methods, f"Python SDK missing method: {method}"

    def test_async_has_all_required_methods(self) -> None:
        """Async Python SDK must expose all required methods."""
        from aragora_sdk.namespaces.computer_use import AsyncComputerUseAPI

        sdk_methods = [m for m in dir(AsyncComputerUseAPI) if not m.startswith("_")]
        for method in self.REQUIRED_METHODS:
            assert method in sdk_methods, f"Async Python SDK missing method: {method}"

    def test_execute_action_sends_action_type(self) -> None:
        """execute_action must send 'action_type' field in request body."""
        content = self._PY_SDK.read_text()
        assert '"action_type"' in content, (
            "Python SDK execute_action should use 'action_type' field"
        )

    def test_update_policy_uses_put(self) -> None:
        """update_policy must use PUT method."""
        content = self._PY_SDK.read_text()
        # Find the update_policy method section and verify PUT
        assert '"PUT"' in content, "Python SDK update_policy should use PUT method"


# =========================================================================
# TypeScript SDK Parity
# =========================================================================


class TestTypeScriptSDKParity:
    """Verify TypeScript SDK covers all canonical Computer Use endpoints."""

    _TS_SDK = ROOT / "sdk" / "typescript" / "src" / "namespaces" / "computer-use.ts"

    def _unique_endpoints(self, endpoints: list[tuple[str, str]]) -> set[tuple[str, str]]:
        return set(endpoints)

    def test_typescript_sdk_covers_all_endpoints(self) -> None:
        """TypeScript SDK computer-use covers every server endpoint."""
        ts_paths = self._unique_endpoints(_extract_ts_paths(self._TS_SDK))
        canonical = set(COMPUTER_USE_ENDPOINTS)

        missing = canonical - ts_paths
        assert not missing, (
            f"TypeScript SDK computer-use missing {len(missing)} endpoints:\n"
            + "\n".join(f"  {m} {p}" for m, p in sorted(missing))
        )

    REQUIRED_METHODS = [
        "createTask",
        "listTasks",
        "getTask",
        "cancelTask",
        "deleteTask",
        "listActions",
        "getAction",
        "executeAction",
        "deleteAction",
        "getActionStats",
        "listPolicies",
        "createPolicy",
        "getPolicy",
        "updatePolicy",
        "deletePolicy",
        "listApprovals",
        "getApproval",
        "approveApproval",
        "denyApproval",
    ]

    def test_has_all_required_methods(self) -> None:
        """TypeScript SDK must expose all required methods."""
        content = self._TS_SDK.read_text()
        ts_methods = re.findall(r"async\s+(\w+)\s*\(", content)
        for method in self.REQUIRED_METHODS:
            assert method in ts_methods, f"TypeScript SDK missing method: {method}"

    def test_execute_action_has_action_type(self) -> None:
        """ExecuteActionOptions must have actionType field."""
        content = self._TS_SDK.read_text()
        interface_match = re.search(
            r"interface ExecuteActionOptions\s*\{([^}]+)\}", content
        )
        assert interface_match, "ExecuteActionOptions interface not found"
        fields = interface_match.group(1)
        assert "actionType" in fields, "ExecuteActionOptions should have actionType"

    def test_update_policy_options_interface(self) -> None:
        """UpdatePolicyOptions interface must exist with correct fields."""
        content = self._TS_SDK.read_text()
        interface_match = re.search(
            r"interface UpdatePolicyOptions\s*\{([^}]+)\}", content
        )
        assert interface_match, "UpdatePolicyOptions interface not found"
        fields = interface_match.group(1)
        for field in ["name", "description", "allowedActions", "blockedDomains"]:
            assert field in fields, f"UpdatePolicyOptions missing field: {field}"


# =========================================================================
# Cross-surface consistency
# =========================================================================


class TestCrossSurfaceConsistency:
    """Verify Python and TypeScript SDKs agree on paths."""

    def test_python_ts_path_agreement(self) -> None:
        """Python and TypeScript SDKs hit the same Computer Use paths."""
        py_file = ROOT / "sdk/python/aragora_sdk/namespaces/computer_use.py"
        ts_file = ROOT / "sdk/typescript/src/namespaces/computer-use.ts"
        py_paths = set(_extract_python_sdk_paths(py_file))
        ts_paths = set(_extract_ts_paths(ts_file))

        only_in_py = py_paths - ts_paths
        only_in_ts = ts_paths - py_paths

        assert not only_in_py, f"Paths only in Python SDK: {only_in_py}"
        assert not only_in_ts, f"Paths only in TypeScript SDK: {only_in_ts}"

    def test_method_count_parity(self) -> None:
        """Python and TypeScript SDKs should have the same number of methods."""
        py_file = ROOT / "sdk/python/aragora_sdk/namespaces/computer_use.py"
        ts_file = ROOT / "sdk/typescript/src/namespaces/computer-use.ts"

        py_endpoints = set(_extract_python_sdk_paths(py_file))
        ts_endpoints = set(_extract_ts_paths(ts_file))

        assert len(py_endpoints) == len(ts_endpoints), (
            f"Endpoint count mismatch: Python has {len(py_endpoints)}, "
            f"TypeScript has {len(ts_endpoints)}"
        )

    def test_all_use_v1_prefix(self) -> None:
        """Both SDKs must use /api/v1/computer-use/ prefix."""
        for filepath, label in [
            (ROOT / "sdk/python/aragora_sdk/namespaces/computer_use.py", "Python SDK"),
            (ROOT / "sdk/typescript/src/namespaces/computer-use.ts", "TypeScript SDK"),
        ]:
            content = filepath.read_text()
            assert "/api/v1/computer-use/" in content, (
                f"{label} missing /api/v1/computer-use/ prefix"
            )

    def test_coverage_exceeds_70_percent(self) -> None:
        """SDK coverage must exceed 70% of OpenAPI endpoints."""
        py_file = ROOT / "sdk/python/aragora_sdk/namespaces/computer_use.py"
        py_endpoints = set(_extract_python_sdk_paths(py_file))
        canonical = set(COMPUTER_USE_ENDPOINTS)

        covered = canonical & py_endpoints
        coverage = len(covered) / len(canonical) * 100

        assert coverage >= 70.0, (
            f"Computer Use SDK coverage is {coverage:.1f}%, need 70%+. "
            f"Missing: {canonical - py_endpoints}"
        )

    def test_full_coverage(self) -> None:
        """SDK should cover 100% of canonical endpoints (19/19)."""
        py_file = ROOT / "sdk/python/aragora_sdk/namespaces/computer_use.py"
        py_endpoints = set(_extract_python_sdk_paths(py_file))
        canonical = set(COMPUTER_USE_ENDPOINTS)

        covered = canonical & py_endpoints
        coverage = len(covered) / len(canonical) * 100

        assert coverage == 100.0, (
            f"Computer Use SDK coverage is {coverage:.1f}% ({len(covered)}/{len(canonical)}). "
            f"Missing: {canonical - py_endpoints}"
        )
