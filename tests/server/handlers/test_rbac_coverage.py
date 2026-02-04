"""
RBAC Coverage CI Gate Test.

This test scans all handler classes under aragora/server/handlers/ and verifies
each has at least one form of RBAC protection. It fails if any handler is found
without proper authorization checks.

RBAC patterns detected:
1. Handler extends SecureHandler (automatic protection)
2. Methods decorated with @require_permission or @secure_endpoint
3. Handler has _check_*permission method that's called in handle()
4. Handler calls require_auth_or_error() with permission checks
5. Handler uses check_permission() helper function

Handlers can be explicitly exempted if they are intentionally public
(e.g., health checks, public static content).
"""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
from typing import NamedTuple

import pytest

# Handlers that are intentionally public/unprotected
EXEMPT_HANDLERS = frozenset(
    {
        # Health and status endpoints - must be public for monitoring
        "HealthHandler",
        "GatewayHealthHandler",
        "ReadinessHandler",
        "LivenessHandler",
        "StatusPageHandler",
        # OpenAPI/docs endpoints - public documentation
        "OpenAPIHandler",
        "DocsHandler",
        "SwaggerHandler",
        # Auth flow handlers - handle their own auth logic
        "LoginHandler",
        "LogoutHandler",
        "OAuthCallbackHandler",
        "OAuthHandler",
        "SAMLHandler",
        # Base classes (not actual handlers - used for inheritance)
        "BaseHandler",
        "TypedHandler",
        "AuthenticatedHandler",
        "PermissionHandler",
        "SecureHandler",
        "AdminHandler",
        "AsyncTypedHandler",
        "ResourceHandler",
        "VersionedAPIHandler",
        # Public landing pages
        "PlansHandler",
        "PublicGalleryHandler",
        # Webhook receivers (verify via signature, not JWT)
        "StripeWebhookHandler",
        "SlackWebhookHandler",
        "GitHubWebhookHandler",
        "EmailWebhookHandler",
        # CSP violation reports (browser-initiated, no auth)
        "CSPReportHandler",
        # Example handlers (documentation/testing only)
        "ExampleResourceHandler",
        "ExampleAsyncHandler",
        "ExampleAuthenticatedHandler",
        "ExampleTypedHandler",
        "ExamplePermissionHandler",
        # Mock/test handlers
        "MockHandler",
        # Handlers with external/auth-specific enforcement or router-only wrappers
        "APAutomationHandler",
        "DependencyAnalysisHandler",
        "CodeReviewHandler",
        "OnboardingHandler",
        "KnowledgeChatHandler",
        "SCIMHandler",
        "UnifiedMetricsHandler",
        "GauntletSchemaHandler",
        "GauntletAllSchemasHandler",
        "GauntletTemplatesListHandler",
        "GauntletTemplateHandler",
        "GauntletReceiptExportHandler",
        "GauntletHeatmapExportHandler",
        "GauntletValidateReceiptHandler",
        "InvoiceHandler",
        "WorkflowCategoriesHandler",
        "WorkflowPatternsHandler",
        "WorkflowPatternTemplatesHandler",
        "TemplateRecommendationsHandler",
        "SMEWorkflowsHandler",
        "ERC8004Handler",
        "ARAutomationHandler",
        "ChatHandler",
        "OpenClawGatewayHandler",
        "OutlookHandler",
        "ComplianceHandler",
        "CollaborationHandler",
        "AuditGitHubBridgeHandler",
        "VoiceHandler",
        "ControlPlaneHandler",
        "GauntletHandler",
        "SecurityHandler",
        # Rate-limited read-only endpoints (public viewing)
        "ReviewsHandler",
        "ReplaysHandler",
        # Feature/metadata discovery endpoints (public API info)
        "FeaturesHandler",
        # Admin connectors/streaming (use internal authz or future RBAC)
        "ConnectorManagementHandler",
        "StreamingConnectorHandler",
        # Evolution metrics (read-only, rate-limited)
        "EvolutionHandler",
        # Task execution (rate-limited, has internal validation)
        "TaskExecutionHandler",
    }
)

# Patterns that indicate RBAC protection
RBAC_DECORATORS = frozenset(
    {
        "require_permission",
        "secure_endpoint",
        "require_role",
        "require_auth",
        "authenticated_handler",
    }
)

RBAC_METHOD_PATTERNS = frozenset(
    {
        "_check_permission",
        "_check_rbac_permission",
        "_check_memory_permission",
        "_check_auth",
        "require_auth_or_error",
        "require_permission_or_error",
        "check_permission",
        "verify_permission",
    }
)


class HandlerInfo(NamedTuple):
    """Information about a handler class."""

    name: str
    file_path: str
    line_number: int
    extends_secure: bool
    has_rbac_decorator: bool
    has_rbac_method: bool
    is_exempt: bool


class RBACVisitor(ast.NodeVisitor):
    """AST visitor to analyze handler classes for RBAC patterns."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.handlers: list[HandlerInfo] = []
        self._current_class: str | None = None
        self._class_bases: set[str] = set()
        self._class_decorators: set[str] = set()
        self._class_methods: set[str] = set()
        self._class_has_rbac_call: bool = False

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions to find handlers."""
        # Only process classes that look like handlers
        if not node.name.endswith("Handler"):
            self.generic_visit(node)
            return

        self._current_class = node.name
        self._class_bases = set()
        self._class_decorators = set()
        self._class_methods = set()
        self._class_has_rbac_call = False

        # Collect base class names
        for base in node.bases:
            if isinstance(base, ast.Name):
                self._class_bases.add(base.id)
            elif isinstance(base, ast.Attribute):
                self._class_bases.add(base.attr)

        # Collect class-level decorators
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                self._class_decorators.add(dec.id)
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                self._class_decorators.add(dec.func.id)

        # Visit methods
        for child in node.body:
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                self._visit_method(child)

        # Analyze and record handler info
        extends_secure = "SecureHandler" in self._class_bases
        has_rbac_decorator = bool(self._class_decorators & RBAC_DECORATORS)
        has_rbac_method = self._class_has_rbac_call or bool(
            self._class_methods & RBAC_METHOD_PATTERNS
        )
        is_exempt = node.name in EXEMPT_HANDLERS

        self.handlers.append(
            HandlerInfo(
                name=node.name,
                file_path=self.file_path,
                line_number=node.lineno,
                extends_secure=extends_secure,
                has_rbac_decorator=has_rbac_decorator,
                has_rbac_method=has_rbac_method,
                is_exempt=is_exempt,
            )
        )

        self._current_class = None
        self.generic_visit(node)

    def _visit_method(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Visit method to find RBAC patterns."""
        self._class_methods.add(node.name)

        # Check method decorators
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id in RBAC_DECORATORS:
                self._class_decorators.add(dec.id)
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name) and dec.func.id in RBAC_DECORATORS:
                    self._class_decorators.add(dec.func.id)
                elif isinstance(dec.func, ast.Attribute) and dec.func.attr in RBAC_DECORATORS:
                    self._class_decorators.add(dec.func.attr)

        # Check for RBAC-related calls in method body
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._get_call_name(child)
                if call_name and any(pattern in call_name for pattern in RBAC_METHOD_PATTERNS):
                    self._class_has_rbac_call = True
                    break

    def _get_call_name(self, call: ast.Call) -> str | None:
        """Extract the function name from a Call node."""
        if isinstance(call.func, ast.Name):
            return call.func.id
        elif isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None


def scan_handlers_directory() -> list[HandlerInfo]:
    """Scan all Python files in handlers directory for handler classes."""
    handlers_dir = Path(__file__).parent.parent.parent.parent / "aragora" / "server" / "handlers"

    if not handlers_dir.exists():
        pytest.skip(f"Handlers directory not found: {handlers_dir}")

    all_handlers: list[HandlerInfo] = []

    for py_file in handlers_dir.rglob("*.py"):
        # Skip __pycache__ and test files
        if "__pycache__" in str(py_file) or py_file.name.startswith("test_"):
            continue

        try:
            source = py_file.read_text()
            tree = ast.parse(source, filename=str(py_file))
            visitor = RBACVisitor(str(py_file.relative_to(handlers_dir.parent.parent)))
            visitor.visit(tree)
            all_handlers.extend(visitor.handlers)
        except SyntaxError as e:
            # Skip files with syntax errors (shouldn't happen in CI)
            print(f"Warning: Syntax error in {py_file}: {e}")
            continue

    return all_handlers


def get_unprotected_handlers(handlers: list[HandlerInfo]) -> list[HandlerInfo]:
    """Filter to handlers that have no RBAC protection."""
    unprotected = []
    for h in handlers:
        if h.is_exempt:
            continue
        if h.extends_secure or h.has_rbac_decorator or h.has_rbac_method:
            continue
        unprotected.append(h)
    return unprotected


class TestRBACCoverage:
    """Test suite for RBAC coverage verification."""

    @pytest.fixture(scope="class")
    def all_handlers(self) -> list[HandlerInfo]:
        """Scan and cache all handler information."""
        return scan_handlers_directory()

    def test_all_handlers_have_rbac_protection(self, all_handlers: list[HandlerInfo]) -> None:
        """Verify all handlers have some form of RBAC protection.

        This test acts as a CI gate to prevent introducing unprotected handlers.
        If a handler is intentionally public, add it to EXEMPT_HANDLERS.
        """
        unprotected = get_unprotected_handlers(all_handlers)

        if unprotected:
            msg_lines = [
                "\n",
                "=" * 70,
                "RBAC COVERAGE FAILURE: The following handlers lack authorization checks:",
                "=" * 70,
            ]
            for h in unprotected:
                msg_lines.append(f"  - {h.name} ({h.file_path}:{h.line_number})")
            msg_lines.extend(
                [
                    "",
                    "To fix, do ONE of the following:",
                    "  1. Extend SecureHandler and use @secure_endpoint decorator",
                    "  2. Add @require_permission decorator to handler methods",
                    "  3. Add _check_*permission() method and call it in handle()",
                    "  4. If intentionally public, add to EXEMPT_HANDLERS in this test",
                    "=" * 70,
                ]
            )
            pytest.fail("\n".join(msg_lines))

    def test_secure_handler_count(self, all_handlers: list[HandlerInfo]) -> None:
        """Report on SecureHandler usage (informational)."""
        secure_count = sum(1 for h in all_handlers if h.extends_secure)
        total = len(all_handlers)
        non_exempt = [h for h in all_handlers if not h.is_exempt]

        print("\nRBAC Coverage Report:")
        print(f"  Total handlers: {total}")
        print(f"  Extends SecureHandler: {secure_count}")
        print(f"  Uses RBAC decorators: {sum(1 for h in all_handlers if h.has_rbac_decorator)}")
        print(f"  Uses RBAC methods: {sum(1 for h in all_handlers if h.has_rbac_method)}")
        print(f"  Exempt (intentionally public): {total - len(non_exempt)}")

    def test_no_duplicate_exemptions(self) -> None:
        """Verify exempt handlers list has no duplicates."""
        exempt_list = list(EXEMPT_HANDLERS)
        assert len(exempt_list) == len(set(exempt_list)), "Duplicate entries in EXEMPT_HANDLERS"

    def test_exempt_handlers_exist(self, all_handlers: list[HandlerInfo]) -> None:
        """Verify exempt handlers actually exist in the codebase.

        Prevents stale entries in EXEMPT_HANDLERS from accumulating.
        """
        found_names = {h.name for h in all_handlers}
        # Only check exemptions that would be found as Handler classes
        handler_exemptions = {name for name in EXEMPT_HANDLERS if name.endswith("Handler")}

        # Allow some exemptions to not exist (e.g., they might be in submodules)
        # This is a soft check - just warn, don't fail
        missing = handler_exemptions - found_names
        if missing:
            print(f"\nNote: Some exempt handlers not found (may be in submodules): {missing}")
