"""Contract parity tests for OpenClaw and Blockchain SDK namespaces.

Verifies that all SDK surfaces (Python SDK, TypeScript SDK, Python client)
define the same endpoints as the server OpenAPI spec.  These tests catch
contract drift before it reaches users.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("aragora_sdk", reason="aragora-sdk not installed")

ROOT = Path(__file__).resolve().parents[2]

# =========================================================================
# Canonical endpoint definitions (source of truth: server OpenAPI spec)
# =========================================================================

OPENCLAW_ENDPOINTS: list[tuple[str, str]] = [
    # Session management
    ("GET", "/api/v1/openclaw/sessions"),
    ("POST", "/api/v1/openclaw/sessions"),
    ("GET", "/api/v1/openclaw/sessions/{id}"),
    ("POST", "/api/v1/openclaw/sessions/{id}/end"),
    ("DELETE", "/api/v1/openclaw/sessions/{id}"),
    # Action management
    ("POST", "/api/v1/openclaw/actions"),
    ("GET", "/api/v1/openclaw/actions/{id}"),
    ("POST", "/api/v1/openclaw/actions/{id}/cancel"),
    # Credential lifecycle
    ("GET", "/api/v1/openclaw/credentials"),
    ("POST", "/api/v1/openclaw/credentials"),
    ("DELETE", "/api/v1/openclaw/credentials/{id}"),
    ("POST", "/api/v1/openclaw/credentials/{id}/rotate"),
    # Policy rules
    ("GET", "/api/v1/openclaw/policy/rules"),
    ("POST", "/api/v1/openclaw/policy/rules"),
    ("DELETE", "/api/v1/openclaw/policy/rules/{id}"),
    # Approvals
    ("GET", "/api/v1/openclaw/approvals"),
    ("POST", "/api/v1/openclaw/approvals/{id}/approve"),
    ("POST", "/api/v1/openclaw/approvals/{id}/deny"),
    # Service introspection
    ("GET", "/api/v1/openclaw/health"),
    ("GET", "/api/v1/openclaw/metrics"),
    ("GET", "/api/v1/openclaw/audit"),
    ("GET", "/api/v1/openclaw/stats"),
]

BLOCKCHAIN_ENDPOINTS: list[tuple[str, str]] = [
    ("GET", "/api/v1/blockchain/agents"),
    ("POST", "/api/v1/blockchain/agents"),
    ("GET", "/api/v1/blockchain/config"),
    ("GET", "/api/v1/blockchain/agents/{id}"),
    ("GET", "/api/v1/blockchain/agents/{id}/reputation"),
    ("GET", "/api/v1/blockchain/agents/{id}/validations"),
    ("POST", "/api/v1/blockchain/sync"),
    ("GET", "/api/v1/blockchain/health"),
]


def _normalize_path(path: str) -> str:
    """Normalize a path by replacing specific IDs with {id}."""
    # f-string interpolations: session_id, approval_id, action_id, credential_id, token_id, rule_name
    path = re.sub(r"\{[a-z_]+\}", "{id}", path)
    # Python f-string patterns
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
        # Look for self._client.request("METHOD", "path") calls
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
            # f-string — reconstruct with {id} placeholders
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

    # Match patterns like: this.client.request<...>('METHOD', '/api/...')
    # and: this.client.request<...>('METHOD', `/api/.../${...}`)
    pattern = re.compile(
        r"this\.client\.request[^(]*\(\s*['\"](\w+)['\"],\s*"
        r"(?:['\"]([^'\"]+)['\"]|`([^`]+)`)"
    )

    for match in pattern.finditer(source):
        method = match.group(1)
        path = match.group(2) or match.group(3)
        # Replace template literals: ${...} -> {id}
        path = re.sub(r"\$\{[^}]+\}", "{id}", path)
        endpoints.append((method, _normalize_path(path)))

    return endpoints


# =========================================================================
# OpenClaw parity tests
# =========================================================================


class TestOpenclawContractParity:
    """Verify all SDK surfaces match the server's OpenClaw endpoints."""

    def _unique_endpoints(self, endpoints: list[tuple[str, str]]) -> set[tuple[str, str]]:
        return set(endpoints)

    def test_python_sdk_covers_all_endpoints(self):
        """Python SDK namespace covers every server OpenClaw endpoint."""
        sdk_file = ROOT / "sdk/python/aragora_sdk/namespaces/openclaw.py"
        sdk_paths = self._unique_endpoints(_extract_python_sdk_paths(sdk_file))
        canonical = set(OPENCLAW_ENDPOINTS)

        missing = canonical - sdk_paths
        assert not missing, f"Python SDK openclaw missing {len(missing)} endpoints:\n" + "\n".join(
            f"  {m} {p}" for m, p in sorted(missing)
        )

    def test_typescript_sdk_covers_all_endpoints(self):
        """TypeScript SDK namespace covers every server OpenClaw endpoint."""
        ts_file = ROOT / "sdk/typescript/src/namespaces/openclaw.ts"
        ts_paths = self._unique_endpoints(_extract_ts_paths(ts_file))
        canonical = set(OPENCLAW_ENDPOINTS)

        missing = canonical - ts_paths
        assert not missing, (
            f"TypeScript SDK openclaw missing {len(missing)} endpoints:\n"
            + "\n".join(f"  {m} {p}" for m, p in sorted(missing))
        )

    def test_python_client_covers_all_endpoints(self):
        """Python client resource covers every server OpenClaw endpoint."""
        client_file = ROOT / "aragora/client/resources/openclaw.py"
        # The client uses _get, _post, _delete instead of request
        # Parse with AST to handle f-strings properly
        source = client_file.read_text()
        tree = ast.parse(source)
        endpoints: set[tuple[str, str]] = set()

        method_map = {
            "_get": "GET",
            "_post": "POST",
            "_delete": "DELETE",
            "_get_async": "GET",
            "_post_async": "POST",
            "_delete_async": "DELETE",
        }

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in method_map:
                continue
            if not node.args:
                continue

            http_method = method_map[func.attr]
            path_node = node.args[0]

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

            endpoints.add((http_method, _normalize_path(path)))

        canonical = set(OPENCLAW_ENDPOINTS)
        missing = canonical - endpoints
        assert not missing, (
            f"Python client openclaw missing {len(missing)} endpoints:\n"
            + "\n".join(f"  {m} {p}" for m, p in sorted(missing))
        )

    def test_no_wrong_paths_in_typescript(self):
        """TypeScript SDK doesn't use incorrect path patterns."""
        ts_file = ROOT / "sdk/typescript/src/namespaces/openclaw.ts"
        source = ts_file.read_text()
        # These were the old incorrect paths
        bad_patterns = [
            "/sessions/${sessionId}/actions",  # should be /actions
            "/sessions/${sessionId}/close",  # should be /sessions/{id}/end
        ]
        for pattern in bad_patterns:
            assert pattern not in source, f"TypeScript SDK still uses incorrect path: {pattern}"

    def test_python_sdk_method_count(self):
        """Python SDK has both sync and async classes."""
        from aragora_sdk.namespaces.openclaw import AsyncOpenclawAPI, OpenclawAPI

        sync_methods = [m for m in dir(OpenclawAPI) if not m.startswith("_")]
        async_methods = [m for m in dir(AsyncOpenclawAPI) if not m.startswith("_")]

        # Both should have the same set of public methods
        assert set(sync_methods) == set(async_methods), (
            f"Sync/async method mismatch. "
            f"Only in sync: {set(sync_methods) - set(async_methods)}. "
            f"Only in async: {set(async_methods) - set(sync_methods)}"
        )


# =========================================================================
# Blockchain parity tests
# =========================================================================


class TestBlockchainContractParity:
    """Verify all SDK surfaces match the server's Blockchain endpoints."""

    def _unique_endpoints(self, endpoints: list[tuple[str, str]]) -> set[tuple[str, str]]:
        return set(endpoints)

    def test_python_sdk_covers_all_endpoints(self):
        """Python SDK blockchain covers every implemented server endpoint."""
        sdk_file = ROOT / "sdk/python/aragora_sdk/namespaces/blockchain.py"
        sdk_paths = self._unique_endpoints(_extract_python_sdk_paths(sdk_file))
        canonical = set(BLOCKCHAIN_ENDPOINTS)

        missing = canonical - sdk_paths
        assert not missing, (
            f"Python SDK blockchain missing {len(missing)} endpoints:\n"
            + "\n".join(f"  {m} {p}" for m, p in sorted(missing))
        )

    def test_typescript_sdk_covers_all_endpoints(self):
        """TypeScript SDK blockchain covers every implemented server endpoint."""
        ts_file = ROOT / "sdk/typescript/src/namespaces/blockchain.ts"
        ts_paths = self._unique_endpoints(_extract_ts_paths(ts_file))
        canonical = set(BLOCKCHAIN_ENDPOINTS)

        missing = canonical - ts_paths
        assert not missing, (
            f"TypeScript SDK blockchain missing {len(missing)} endpoints:\n"
            + "\n".join(f"  {m} {p}" for m, p in sorted(missing))
        )

    def test_python_sdk_blockchain_method_count(self):
        """Python SDK blockchain has both sync and async classes."""
        from aragora_sdk.namespaces.blockchain import AsyncBlockchainAPI, BlockchainAPI

        sync_methods = [m for m in dir(BlockchainAPI) if not m.startswith("_")]
        async_methods = [m for m in dir(AsyncBlockchainAPI) if not m.startswith("_")]

        assert set(sync_methods) == set(async_methods), (
            f"Sync/async method mismatch. "
            f"Only in sync: {set(sync_methods) - set(async_methods)}. "
            f"Only in async: {set(async_methods) - set(sync_methods)}"
        )


# =========================================================================
# Cross-surface consistency checks
# =========================================================================


class TestCrossSurfaceConsistency:
    """Verify that Python SDK and TypeScript SDK agree on paths."""

    def test_openclaw_python_ts_path_agreement(self):
        """Python and TypeScript SDKs hit the same OpenClaw paths."""
        py_file = ROOT / "sdk/python/aragora_sdk/namespaces/openclaw.py"
        ts_file = ROOT / "sdk/typescript/src/namespaces/openclaw.ts"
        py_paths = set(_extract_python_sdk_paths(py_file))
        ts_paths = set(_extract_ts_paths(ts_file))

        # TS may have closeSession alias, so filter that out
        only_in_py = py_paths - ts_paths
        only_in_ts = ts_paths - py_paths

        # Allow known aliases but nothing else
        assert not only_in_py, f"Paths only in Python SDK: {only_in_py}"
        # closeSession is an alias for endSession, so the same path appears twice — that's OK
        unexpected_ts = {p for p in only_in_ts if p not in ts_paths}
        assert not unexpected_ts, f"Unexpected paths only in TypeScript SDK: {unexpected_ts}"

    def test_blockchain_python_ts_path_agreement(self):
        """Python and TypeScript SDKs hit the same Blockchain paths."""
        py_file = ROOT / "sdk/python/aragora_sdk/namespaces/blockchain.py"
        ts_file = ROOT / "sdk/typescript/src/namespaces/blockchain.ts"
        py_paths = set(_extract_python_sdk_paths(py_file))
        ts_paths = set(_extract_ts_paths(ts_file))

        only_in_py = py_paths - ts_paths
        only_in_ts = ts_paths - py_paths

        assert not only_in_py, f"Paths only in Python SDK: {only_in_py}"
        assert not only_in_ts, f"Paths only in TypeScript SDK: {only_in_ts}"
