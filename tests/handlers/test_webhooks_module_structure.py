"""Structural regression tests for the webhook handler package."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import warnings


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_NAME = "aragora.server.handlers.webhooks"

EXPECTED_EXPORTS = [
    "GITHUB_APP_ROUTES",
    "handle_github_webhook",
    "WebhookHandler",
    "WebhookStore",
    "WebhookConfig",
    "get_webhook_store",
    "generate_signature",
    "verify_signature",
    "WEBHOOK_EVENTS",
    "RBAC_AVAILABLE",
    "check_permission",
    "validate_webhook_url",
]

EXPECTED_PUBLIC_NAMES = [
    "GITHUB_APP_ROUTES",
    "RBAC_AVAILABLE",
    "WEBHOOK_EVENTS",
    "WebhookConfig",
    "WebhookHandler",
    "WebhookStore",
    "check_permission",
    "generate_signature",
    "get_webhook_store",
    "github_app",
    "handle_github_webhook",
    "importlib",
    "validate_webhook_url",
    "verify_signature",
]

EXPECTED_ROUTES = [
    "POST /api/webhooks",
    "GET /api/webhooks",
    "GET /api/webhooks/events",
    "GET /api/webhooks/slo/status",
    "POST /api/webhooks/slo/test",
    "GET /api/webhooks/:id",
    "DELETE /api/webhooks/:id",
    "PATCH /api/webhooks/:id",
    "POST /api/webhooks/:id/test",
    "GET /api/webhooks/dead-letter",
    "GET /api/webhooks/dead-letter/:id",
    "POST /api/webhooks/dead-letter/:id/retry",
    "DELETE /api/webhooks/dead-letter/:id",
    "GET /api/webhooks/queue/stats",
]

EXPECTED_V1_ROUTES = [
    "/api/v1/webhooks",
    "/api/v1/webhooks/events",
    "/api/v1/webhooks/events/categories",
    "/api/v1/webhooks/slo/status",
    "/api/v1/webhooks/slo/test",
    "/api/v1/webhooks/dead-letter",
    "/api/v1/webhooks/queue/stats",
    "/api/v1/webhooks/bulk",
    "/api/v1/webhooks/pause-all",
    "/api/v1/webhooks/resume-all",
]


def _run_import_probe() -> dict[str, object]:
    script = textwrap.dedent(
        """
        import importlib
        import importlib.machinery
        import importlib.util
        import json
        from pathlib import Path
        import sys
        import warnings

        root = Path(sys.argv[1])
        legacy_file = root / "aragora/server/handlers/webhooks.py"
        canonical_file = root / "aragora/server/handlers/webhook_management.py"
        implementation_paths = {
            path.resolve() for path in (legacy_file, canonical_file) if path.exists()
        }
        execution_count = 0
        original_exec_module = importlib.machinery.SourceFileLoader.exec_module

        def counted_exec_module(loader, module):
            global execution_count
            if Path(loader.path).resolve() in implementation_paths:
                execution_count += 1
            return original_exec_module(loader, module)

        importlib.machinery.SourceFileLoader.exec_module = counted_exec_module
        legacy_module = None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", DeprecationWarning)
                if legacy_file.exists():
                    spec = importlib.util.spec_from_file_location(
                        "aragora.server.handlers.webhooks",
                        legacy_file,
                    )
                    assert spec is not None and spec.loader is not None
                    legacy_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(legacy_module)

                package = importlib.import_module("aragora.server.handlers.webhooks")
                try:
                    implementation = importlib.import_module(
                        "aragora.server.handlers.webhook_management"
                    )
                except ModuleNotFoundError:
                    implementation = package._webhooks_module
        finally:
            importlib.machinery.SourceFileLoader.exec_module = original_exec_module

        relevant_warnings = [
            str(item.message)
            for item in caught
            if issubclass(item.category, DeprecationWarning)
            and str(item.message).startswith(
                "aragora.server.handlers.webhooks"
            )
        ]
        print(json.dumps({
            "execution_count": execution_count,
            "handler_module": package.WebhookHandler.__module__,
            "implementation_name": implementation.__name__,
            "legacy_file_exists": legacy_file.exists(),
            "legacy_handler_is_package_handler": (
                None
                if legacy_module is None
                else legacy_module.WebhookHandler is package.WebhookHandler
            ),
            "package_handler_is_implementation": (
                package.WebhookHandler is implementation.WebhookHandler
            ),
            "compat_module_alias_is_implementation": (
                package._webhooks_module is implementation
            ),
            "relevant_warning_count": len(relevant_warnings),
            "webhooks_module_registered": "webhooks_module" in sys.modules,
        }, sort_keys=True))
        """
    )
    env = os.environ.copy()
    env.update(
        {
            "ARAGORA_SECRETS_STRICT": "false",
            "AWS_CONFIG_FILE": "/dev/null",
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(REPO_ROOT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_webhook_implementation_executes_once_under_canonical_identity() -> None:
    """The implementation has one source path, module identity, and class object."""
    probe = _run_import_probe()

    assert probe == {
        "compat_module_alias_is_implementation": True,
        "execution_count": 1,
        "handler_module": "aragora.server.handlers.webhook_management",
        "implementation_name": "aragora.server.handlers.webhook_management",
        "legacy_file_exists": False,
        "legacy_handler_is_package_handler": None,
        "package_handler_is_implementation": True,
        "relevant_warning_count": 1,
        "webhooks_module_registered": False,
    }


def test_canonical_module_is_silent_and_package_shim_warns() -> None:
    """Only the retired package-level implementation path emits the move warning."""
    script = textwrap.dedent(
        """
        import importlib
        import json
        import warnings

        with warnings.catch_warnings(record=True) as canonical_warnings:
            warnings.simplefilter("always", DeprecationWarning)
            canonical = importlib.import_module(
                "aragora.server.handlers.webhook_management"
            )
        with warnings.catch_warnings(record=True) as shim_warnings:
            warnings.simplefilter("always", DeprecationWarning)
            shim = importlib.import_module("aragora.server.handlers.webhooks")

        def relevant(items):
            return [
                str(item.message)
                for item in items
                if issubclass(item.category, DeprecationWarning)
                and str(item.message).startswith("aragora.server.handlers.webhooks")
            ]

        print(json.dumps({
            "canonical_warnings": relevant(canonical_warnings),
            "shim_handler_is_canonical": shim.WebhookHandler is canonical.WebhookHandler,
            "shim_warnings": relevant(shim_warnings),
        }, sort_keys=True))
        """
    )
    env = os.environ.copy()
    env.update(
        {
            "ARAGORA_SECRETS_STRICT": "false",
            "AWS_CONFIG_FILE": "/dev/null",
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "canonical_warnings": [],
        "shim_handler_is_canonical": True,
        "shim_warnings": [
            "aragora.server.handlers.webhooks is deprecated as the webhook management "
            "implementation home; use aragora.server.handlers.webhook_management instead."
        ],
    }


def test_webhook_routes_registration_and_public_surface_match_baseline() -> None:
    """The structural move preserves exports, route tables, and registration order."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        module = __import__(PACKAGE_NAME, fromlist=["WebhookHandler"])
        from aragora.server.handler_registry.admin import ADMIN_HANDLER_REGISTRY
        from aragora.server.handlers._lazy_imports import ALL_HANDLER_NAMES, HANDLER_MODULES

    assert module.__all__ == EXPECTED_EXPORTS
    assert (
        sorted(name for name in vars(module) if not name.startswith("_")) == EXPECTED_PUBLIC_NAMES
    )
    assert module.WebhookHandler.routes == EXPECTED_ROUTES
    assert module.WebhookHandler.ROUTES == EXPECTED_V1_ROUTES

    assert HANDLER_MODULES["WebhookHandler"] == PACKAGE_NAME
    lazy_names = list(ALL_HANDLER_NAMES)
    lazy_index = lazy_names.index("WebhookHandler")
    assert lazy_index == 136
    assert lazy_names[lazy_index - 3 : lazy_index + 4] == [
        "FormalVerificationHandler",
        "SlackHandler",
        "EvidenceHandler",
        "WebhookHandler",
        "CodebaseAuditHandler",
        "AdminHandler",
        "SecurityHandler",
    ]

    registry_names = [name for name, _ in ADMIN_HANDLER_REGISTRY]
    registry_index = registry_names.index("_webhook_handler")
    assert registry_index == 50
    assert registry_names[registry_index - 3 : registry_index + 4] == [
        "_integration_health_handler",
        "_automation_handler",
        "_workflow_handler",
        "_webhook_handler",
        "_queue_handler",
        "_workflow_templates_handler",
        "_workflow_patterns_handler",
    ]
    registry_entry = dict(ADMIN_HANDLER_REGISTRY)["_webhook_handler"]
    assert registry_entry.resolve() is module.WebhookHandler


def test_package_shim_has_no_dynamic_file_loader() -> None:
    """The compatibility package uses a normal canonical import, not exec loading."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        package = __import__(PACKAGE_NAME, fromlist=["_webhooks_module"])

    source = Path(package.__file__).read_text(encoding="utf-8")
    assert "spec_from_file_location" not in source
    assert "module_from_spec" not in source
    assert "exec_module" not in source
    assert package._webhooks_module.__name__ == "aragora.server.handlers.webhook_management"
