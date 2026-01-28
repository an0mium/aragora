"""
SDK Parity Validation Tests.

Validates that the Python SDK has methods covering server handler endpoints.
This test helps track SDK coverage and identify gaps.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import Any

import pytest


def get_sdk_methods() -> dict[str, list[str]]:
    """Extract all public methods from SDK namespace classes."""
    sdk_path = Path(__file__).parent.parent.parent / "sdk" / "python" / "aragora" / "namespaces"
    methods: dict[str, list[str]] = {}

    if not sdk_path.exists():
        pytest.skip("SDK path not found")

    for py_file in sdk_path.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        namespace = py_file.stem
        methods[namespace] = []

        try:
            content = py_file.read_text()
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Skip async classes for now - they mirror sync classes
                    if node.name.startswith("Async"):
                        continue

                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # Skip private methods
                            if not item.name.startswith("_"):
                                methods[namespace].append(item.name)
        except SyntaxError:
            continue

    return methods


def get_handler_routes() -> dict[str, list[str]]:
    """Extract route patterns from handler files."""
    handlers_path = Path(__file__).parent.parent.parent / "aragora" / "server" / "handlers"
    routes: dict[str, list[str]] = {}

    if not handlers_path.exists():
        pytest.skip("Handlers path not found")

    for py_file in handlers_path.glob("*.py"):
        if py_file.name.startswith("_") or py_file.name == "base.py":
            continue

        handler_name = py_file.stem
        routes[handler_name] = []

        try:
            content = py_file.read_text()

            # Look for ROUTES class attribute
            route_pattern = re.compile(r"ROUTES\s*=\s*\[(.*?)\]", re.DOTALL)
            match = route_pattern.search(content)
            if match:
                route_str = match.group(1)
                # Extract quoted strings
                found_routes = re.findall(r'"(/[^"]+)"', route_str)
                routes[handler_name].extend(found_routes)

            # Also look for @app.route or @router.route decorators
            decorator_pattern = re.compile(
                r'@(?:app|router)\.(get|post|put|delete|patch)\("([^"]+)"'
            )
            for _method, route in decorator_pattern.findall(content):
                routes[handler_name].append(route)

        except Exception:
            continue

    return routes


class TestSDKStructure:
    """Tests for SDK structure and imports."""

    def test_sdk_namespaces_exist(self):
        """Verify core SDK namespaces exist."""
        sdk_methods = get_sdk_methods()

        core_namespaces = [
            "debates",
            "agents",
            "knowledge",
            "consensus",
            "analytics",
            "auth",
            "workflows",
            "organizations",
            "workspaces",
        ]

        for ns in core_namespaces:
            assert ns in sdk_methods, f"Missing core namespace: {ns}"

    def test_sdk_namespaces_have_methods(self):
        """Verify SDK namespaces have at least some methods."""
        sdk_methods = get_sdk_methods()

        for namespace, methods in sdk_methods.items():
            # Each namespace should have at least one method
            assert len(methods) > 0, f"Namespace {namespace} has no methods"

    def test_knowledge_mound_methods_exist(self):
        """Verify Knowledge Mound SDK methods were added."""
        sdk_methods = get_sdk_methods()

        knowledge_methods = sdk_methods.get("knowledge", [])

        mound_methods = [
            "mound_query",
            "add_node",
            "get_node",
            "list_nodes",
            "get_graph",
            "get_lineage",
            "sync_continuum",
            "sync_consensus",
            "register_region",
            "list_regions",
            "federation_status",
            "export_d3",
            "detect_contradictions",
        ]

        for method in mound_methods:
            assert method in knowledge_methods, f"Missing Knowledge Mound method: {method}"

    def test_consensus_methods_exist(self):
        """Verify Consensus SDK methods exist."""
        sdk_methods = get_sdk_methods()

        consensus_methods = sdk_methods.get("consensus", [])

        expected = [
            "get_similar_debates",
            "get_settled_topics",
            "get_stats",
            "get_dissents",
            "get_contrarian_views",
            "get_risk_warnings",
            "get_domain_history",
        ]

        for method in expected:
            assert method in consensus_methods, f"Missing Consensus method: {method}"

    def test_analytics_methods_exist(self):
        """Verify Analytics SDK methods exist."""
        sdk_methods = get_sdk_methods()

        analytics_methods = sdk_methods.get("analytics", [])

        expected = [
            "disagreements",
            "role_rotation",
            "consensus_quality",
            "ranking_stats",
            "memory_stats",
            "debates_overview",
            "agent_leaderboard",
            "token_usage",
            "cost_breakdown",
            "flip_summary",
            "deliberation_summary",
        ]

        for method in expected:
            assert method in analytics_methods, f"Missing Analytics method: {method}"


class TestSDKParityMetrics:
    """Tests that report SDK parity metrics."""

    def test_sdk_method_count(self):
        """Report total SDK method count."""
        sdk_methods = get_sdk_methods()

        total_methods = sum(len(methods) for methods in sdk_methods.values())

        # Should have at least 100 methods across all namespaces
        assert total_methods >= 100, f"SDK has only {total_methods} methods, expected >= 100"

        # Log the count for visibility
        print(f"\nTotal SDK methods: {total_methods}")
        print(f"Namespaces: {len(sdk_methods)}")

        for ns, methods in sorted(sdk_methods.items()):
            print(f"  {ns}: {len(methods)} methods")

    def test_handler_coverage_report(self):
        """Report handler route coverage."""
        routes = get_handler_routes()

        total_routes = sum(len(r) for r in routes.values())

        print(f"\nTotal handler routes: {total_routes}")
        print(f"Handlers: {len(routes)}")

        for handler, handler_routes in sorted(routes.items()):
            if handler_routes:
                print(f"  {handler}: {len(handler_routes)} routes")


class TestSDKImports:
    """Tests that SDK imports work correctly."""

    def test_client_import(self):
        """Verify client can be imported."""
        from sdk.python.aragora.client import AragoraClient, AragoraAsyncClient

        assert AragoraClient is not None
        assert AragoraAsyncClient is not None

    def test_exceptions_import(self):
        """Verify exceptions can be imported."""
        from sdk.python.aragora.exceptions import (
            AragoraError,
            AuthenticationError,
            AuthorizationError,
            NotFoundError,
            RateLimitError,
            ValidationError,
            ServerError,
        )

        assert AragoraError is not None
        assert AuthenticationError is not None

    def test_namespace_imports(self):
        """Verify namespace APIs can be imported."""
        from sdk.python.aragora.namespaces.debates import DebatesAPI
        from sdk.python.aragora.namespaces.knowledge import KnowledgeAPI
        from sdk.python.aragora.namespaces.consensus import ConsensusAPI
        from sdk.python.aragora.namespaces.analytics import AnalyticsAPI
        from sdk.python.aragora.namespaces.agents import AgentsAPI

        assert DebatesAPI is not None
        assert KnowledgeAPI is not None
        assert ConsensusAPI is not None
        assert AnalyticsAPI is not None
        assert AgentsAPI is not None
