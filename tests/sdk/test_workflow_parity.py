"""Contract parity tests for Workflow SDK namespaces.

Verifies that Python SDK and TypeScript SDK both cover all workflow
backend endpoints.  These tests catch contract drift before it reaches users.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# =========================================================================
# Canonical workflow endpoints (source of truth: WorkflowHandler in handler.py)
# =========================================================================

WORKFLOW_ENDPOINTS: list[tuple[str, str]] = [
    # CRUD
    ("GET", "/api/v1/workflows"),
    ("POST", "/api/v1/workflows"),
    ("GET", "/api/v1/workflows/{id}"),
    ("PUT", "/api/v1/workflows/{id}"),
    ("DELETE", "/api/v1/workflows/{id}"),
    # Execution
    ("POST", "/api/v1/workflows/{id}/execute"),
    ("POST", "/api/v1/workflows/{id}/simulate"),
    ("GET", "/api/v1/workflows/{id}/status"),
    # Versions
    ("GET", "/api/v1/workflows/{id}/versions"),
    ("POST", "/api/v1/workflows/{id}/versions/{id}/restore"),
    # Top-level templates
    ("GET", "/api/v1/workflow-templates"),
    # Approvals
    ("GET", "/api/v1/workflow-approvals"),
    ("POST", "/api/v1/workflow-approvals/{id}/resolve"),
    # Top-level executions
    ("GET", "/api/v1/workflow-executions"),
    ("GET", "/api/v1/workflow-executions/{id}"),
    ("DELETE", "/api/v1/workflow-executions/{id}"),
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

    # Match this.request('METHOD', '/path') pattern
    request_pattern = re.compile(
        r"this\.(?:client\.)?request[^(]*\(\s*['\"](\w+)['\"],\s*"
        r"(?:['\"]([^'\"]+)['\"]|`([^`]+)`)"
    )
    # Match this.invoke<T>('name', [args], 'METHOD', '/path') pattern
    invoke_pattern = re.compile(
        r"this\.invoke[^(]*\([^,]+,\s*\[[^\]]*\],\s*['\"](\w+)['\"],\s*"
        r"(?:['\"]([^'\"]+)['\"]|`([^`]+)`)",
        re.DOTALL,
    )
    # Also extract from @route JSDoc comments as fallback
    jsdoc_pattern = re.compile(r"@route\s+(\w+)\s+(/api/v1/\S+)")

    for match in request_pattern.finditer(source):
        method = match.group(1)
        path = match.group(2) or match.group(3)
        path = re.sub(r"\$\{[^}]+\}", "{id}", path)
        endpoints.append((method, _normalize_path(path)))

    for match in invoke_pattern.finditer(source):
        method = match.group(1)
        path = match.group(2) or match.group(3)
        path = re.sub(r"\$\{[^}]+\}", "{id}", path)
        endpoints.append((method, _normalize_path(path)))

    for match in jsdoc_pattern.finditer(source):
        method = match.group(1)
        path = re.sub(r"\{[a-z_]+\}", "{id}", match.group(2))
        endpoints.append((method, _normalize_path(path)))

    return endpoints


# =========================================================================
# Python SDK parity
# =========================================================================


class TestWorkflowPythonSDKParity:
    """Verify Python SDK covers all workflow endpoints."""

    def _unique_endpoints(self, endpoints: list[tuple[str, str]]) -> set[tuple[str, str]]:
        return set(endpoints)

    def test_python_sdk_covers_all_endpoints(self):
        """Python SDK workflows covers every server Workflow endpoint."""
        sdk_file = ROOT / "sdk/python/aragora_sdk/namespaces/workflows.py"
        sdk_paths = self._unique_endpoints(_extract_python_sdk_paths(sdk_file))
        canonical = set(WORKFLOW_ENDPOINTS)

        missing = canonical - sdk_paths
        assert not missing, (
            f"Python SDK workflows missing {len(missing)} endpoints:\n"
            + "\n".join(f"  {m} {p}" for m, p in sorted(missing))
        )

    def test_python_sdk_method_count(self):
        """Python SDK workflows has both sync and async classes with matching methods."""
        from aragora_sdk.namespaces.workflows import AsyncWorkflowsAPI, WorkflowsAPI

        sync_methods = {m for m in dir(WorkflowsAPI) if not m.startswith("_")}
        async_methods = {m for m in dir(AsyncWorkflowsAPI) if not m.startswith("_")}

        assert sync_methods == async_methods, (
            f"Sync/async method mismatch. "
            f"Only in sync: {sync_methods - async_methods}. "
            f"Only in async: {async_methods - sync_methods}"
        )

    def test_new_methods_exist(self):
        """Verify the newly added methods exist on both sync and async classes."""
        from aragora_sdk.namespaces.workflows import AsyncWorkflowsAPI, WorkflowsAPI

        required_methods = [
            "simulate",
            "get_status",
            "restore_version",
            "list_approvals",
            "resolve_approval",
        ]

        for method_name in required_methods:
            assert hasattr(WorkflowsAPI, method_name), (
                f"WorkflowsAPI missing method: {method_name}"
            )
            assert hasattr(AsyncWorkflowsAPI, method_name), (
                f"AsyncWorkflowsAPI missing method: {method_name}"
            )


# =========================================================================
# TypeScript SDK parity
# =========================================================================


class TestWorkflowTypeScriptSDKParity:
    """Verify TypeScript SDK covers all workflow endpoints."""

    def _unique_endpoints(self, endpoints: list[tuple[str, str]]) -> set[tuple[str, str]]:
        return set(endpoints)

    def test_typescript_sdk_covers_all_endpoints(self):
        """TypeScript SDK workflows covers every server Workflow endpoint."""
        ts_file = ROOT / "sdk/typescript/src/namespaces/workflows.ts"
        ts_paths = self._unique_endpoints(_extract_ts_paths(ts_file))
        canonical = set(WORKFLOW_ENDPOINTS)

        missing = canonical - ts_paths
        assert not missing, (
            f"TypeScript SDK workflows missing {len(missing)} endpoints:\n"
            + "\n".join(f"  {m} {p}" for m, p in sorted(missing))
        )

    def test_typescript_has_approval_methods(self):
        """TypeScript SDK has listApprovals and resolveApproval methods."""
        ts_file = ROOT / "sdk/typescript/src/namespaces/workflows.ts"
        source = ts_file.read_text()

        assert "listApprovals" in source, "TypeScript SDK missing listApprovals method"
        assert "resolveApproval" in source, "TypeScript SDK missing resolveApproval method"

    def test_typescript_has_execution_methods(self):
        """TypeScript SDK has workflow execution management methods."""
        ts_file = ROOT / "sdk/typescript/src/namespaces/workflows.ts"
        source = ts_file.read_text()

        assert "listWorkflowExecutions" in source, "TypeScript SDK missing listWorkflowExecutions"
        assert "getWorkflowExecution" in source, "TypeScript SDK missing getWorkflowExecution"
        assert "terminateExecution" in source, "TypeScript SDK missing terminateExecution"


# =========================================================================
# Cross-surface consistency
# =========================================================================


class TestWorkflowCrossSurfaceConsistency:
    """Verify Python and TypeScript SDKs agree on paths."""

    def test_workflow_python_ts_canonical_agreement(self):
        """Both SDKs cover the same canonical endpoints."""
        py_file = ROOT / "sdk/python/aragora_sdk/namespaces/workflows.py"
        ts_file = ROOT / "sdk/typescript/src/namespaces/workflows.ts"
        py_paths = set(_extract_python_sdk_paths(py_file))
        ts_paths = set(_extract_ts_paths(ts_file))

        canonical = set(WORKFLOW_ENDPOINTS)

        py_coverage = canonical & py_paths
        ts_coverage = canonical & ts_paths

        # Both must fully cover canonical
        py_missing = canonical - py_coverage
        ts_missing = canonical - ts_coverage

        assert not py_missing, f"Python missing canonical endpoints: {py_missing}"
        assert not ts_missing, f"TypeScript missing canonical endpoints: {ts_missing}"

    def test_workflow_approvals_path_agreement(self):
        """Both SDKs use the same approval paths."""
        py_file = ROOT / "sdk/python/aragora_sdk/namespaces/workflows.py"
        ts_file = ROOT / "sdk/typescript/src/namespaces/workflows.ts"
        py_paths = set(_extract_python_sdk_paths(py_file))
        ts_paths = set(_extract_ts_paths(ts_file))

        approval_endpoints = {
            ("GET", "/api/v1/workflow-approvals"),
            ("POST", "/api/v1/workflow-approvals/{id}/resolve"),
        }

        assert approval_endpoints.issubset(py_paths), (
            f"Python SDK missing approval endpoints: {approval_endpoints - py_paths}"
        )
        assert approval_endpoints.issubset(ts_paths), (
            f"TypeScript SDK missing approval endpoints: {approval_endpoints - ts_paths}"
        )

    def test_execute_accepts_inputs(self):
        """Both SDKs' execute method accepts input parameters."""
        from aragora_sdk.namespaces.workflows import AsyncWorkflowsAPI, WorkflowsAPI
        import inspect

        sync_sig = inspect.signature(WorkflowsAPI.execute)
        async_sig = inspect.signature(AsyncWorkflowsAPI.execute)

        assert "inputs" in sync_sig.parameters, "Sync execute missing 'inputs' parameter"
        assert "inputs" in async_sig.parameters, "Async execute missing 'inputs' parameter"

        # TypeScript check
        ts_file = ROOT / "sdk/typescript/src/namespaces/workflows.ts"
        source = ts_file.read_text()
        # Find execute method and verify it takes inputs
        assert "inputs?: Record<string, unknown>" in source, (
            "TypeScript execute method should accept inputs parameter"
        )
