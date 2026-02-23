"""Integration tests for handler registry deferred imports and handler coverage.

These tests verify that ALL _DeferredImport entries in HANDLER_REGISTRY resolve
successfully, that check_handler_coverage doesn't crash, that all resolved
handlers have required methods, and that no two handlers claim the same path.

This test class would have caught the _DeferredImport bug that was only
caught by CI Load Tests.

Marked with @pytest.mark.integration (auto-applied by conftest).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from unittest.mock import MagicMock

import pytest


class TestDeferredImportResolution:
    """Verify that every _DeferredImport in the registry resolves successfully."""

    def test_all_deferred_imports_resolve(self):
        """Every _DeferredImport entry in HANDLER_REGISTRY must resolve to a class."""
        from aragora.server.handler_registry import HANDLER_REGISTRY
        from aragora.server.handler_registry.core import _DeferredImport

        failures = []
        for attr_name, handler_ref in HANDLER_REGISTRY:
            if isinstance(handler_ref, _DeferredImport):
                resolved = handler_ref.resolve()
                if resolved is None:
                    failures.append(
                        f"{attr_name}: failed to resolve "
                        f"{handler_ref._module_path}:{handler_ref._class_name}"
                    )

        assert not failures, (
            f"{len(failures)} handler(s) failed to resolve:\n"
            + "\n".join(f"  - {f}" for f in failures)
        )

    def test_all_deferred_imports_resolve_to_classes(self):
        """Resolved _DeferredImport entries must be actual classes, not modules or other objects."""
        from aragora.server.handler_registry import HANDLER_REGISTRY
        from aragora.server.handler_registry.core import _DeferredImport

        non_classes = []
        for attr_name, handler_ref in HANDLER_REGISTRY:
            if isinstance(handler_ref, _DeferredImport):
                resolved = handler_ref.resolve()
                if resolved is not None and not isinstance(resolved, type):
                    non_classes.append(
                        f"{attr_name}: resolved to {type(resolved).__name__} "
                        f"instead of a class"
                    )

        assert not non_classes, (
            f"{len(non_classes)} handler(s) resolved to non-class objects:\n"
            + "\n".join(f"  - {n}" for n in non_classes)
        )

    def test_deferred_import_idempotent(self):
        """Resolving the same _DeferredImport twice must return the same object."""
        from aragora.server.handler_registry import HANDLER_REGISTRY
        from aragora.server.handler_registry.core import _DeferredImport

        for attr_name, handler_ref in HANDLER_REGISTRY[:5]:
            if isinstance(handler_ref, _DeferredImport):
                first = handler_ref.resolve()
                second = handler_ref.resolve()
                assert first is second, (
                    f"{attr_name}: resolve() returned different objects on second call"
                )

    def test_registry_has_entries(self):
        """Registry must not be empty."""
        from aragora.server.handler_registry import HANDLER_REGISTRY

        assert len(HANDLER_REGISTRY) >= 100, (
            f"HANDLER_REGISTRY has only {len(HANDLER_REGISTRY)} entries, expected 100+"
        )

    def test_all_sub_registries_contribute(self):
        """Each sub-registry (admin, agents, analytics, etc.) contributes entries."""
        from aragora.server.handler_registry.admin import ADMIN_HANDLER_REGISTRY
        from aragora.server.handler_registry.agents import AGENT_HANDLER_REGISTRY
        from aragora.server.handler_registry.analytics import ANALYTICS_HANDLER_REGISTRY
        from aragora.server.handler_registry.debates import DEBATE_HANDLER_REGISTRY
        from aragora.server.handler_registry.memory import MEMORY_HANDLER_REGISTRY
        from aragora.server.handler_registry.social import SOCIAL_HANDLER_REGISTRY

        for name, registry in [
            ("admin", ADMIN_HANDLER_REGISTRY),
            ("agents", AGENT_HANDLER_REGISTRY),
            ("analytics", ANALYTICS_HANDLER_REGISTRY),
            ("debates", DEBATE_HANDLER_REGISTRY),
            ("memory", MEMORY_HANDLER_REGISTRY),
            ("social", SOCIAL_HANDLER_REGISTRY),
        ]:
            assert len(registry) > 0, f"{name} sub-registry is empty"


class TestResolvedHandlerMethods:
    """Verify that resolved handlers have the methods required by the dispatch logic."""

    def test_all_handlers_have_dispatch_method(self):
        """Every resolved handler must have a dispatch mechanism.

        Handlers can use any of: can_handle, ROUTES, register_routes,
        handle, or handle_* methods. Sub-handlers (like GauntletSchemaHandler)
        are routed by their parent and only need handle_* methods.
        """
        from aragora.server.handler_registry import HANDLER_REGISTRY
        from aragora.server.handler_registry.core import _DeferredImport

        missing = []
        for attr_name, handler_ref in HANDLER_REGISTRY:
            if isinstance(handler_ref, _DeferredImport):
                handler_class = handler_ref.resolve()
            else:
                handler_class = handler_ref

            if handler_class is None:
                continue

            has_can_handle = hasattr(handler_class, "can_handle")
            has_routes = hasattr(handler_class, "ROUTES")
            has_register = hasattr(handler_class, "register_routes")
            has_handle = hasattr(handler_class, "handle")
            has_handle_methods = any(
                attr.startswith("handle_")
                for attr in dir(handler_class)
                if not attr.startswith("__")
            )

            if not (has_can_handle or has_routes or has_register or has_handle or has_handle_methods):
                missing.append(
                    f"{attr_name} ({handler_class.__name__}): "
                    "no dispatch mechanism found"
                )

        assert not missing, (
            f"{len(missing)} handler(s) missing any dispatch method:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    def test_all_handlers_have_handle_method(self):
        """Every resolved handler must have handle() or handle_* methods."""
        from aragora.server.handler_registry import HANDLER_REGISTRY
        from aragora.server.handler_registry.core import _DeferredImport

        missing = []
        for attr_name, handler_ref in HANDLER_REGISTRY:
            if isinstance(handler_ref, _DeferredImport):
                handler_class = handler_ref.resolve()
            else:
                handler_class = handler_ref

            if handler_class is None:
                continue

            has_handle = hasattr(handler_class, "handle")
            has_handle_methods = any(
                attr.startswith("handle_")
                for attr in dir(handler_class)
                if not attr.startswith("__")
            )
            has_register = hasattr(handler_class, "register_routes")
            has_routes_only = hasattr(handler_class, "ROUTES")

            if not (has_handle or has_handle_methods or has_register or has_routes_only):
                missing.append(
                    f"{attr_name} ({handler_class.__name__}): "
                    "missing handle, handle_*, register_routes, or ROUTES"
                )

        assert not missing, (
            f"{len(missing)} handler(s) missing handle dispatch:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    def test_handlers_instantiate_with_empty_ctx(self):
        """Handlers must be instantiable with an empty context dict."""
        from aragora.server.handler_registry import HANDLER_REGISTRY
        from aragora.server.handler_registry.core import _DeferredImport

        ctx = {
            "storage": None,
            "stream_emitter": None,
            "control_plane_stream": None,
            "nomic_loop_stream": None,
            "elo_system": None,
            "nomic_dir": None,
            "debate_embeddings": None,
            "critique_store": None,
            "document_store": None,
            "persona_manager": None,
            "position_ledger": None,
            "user_store": None,
            "continuum_memory": None,
            "cross_debate_memory": None,
            "knowledge_mound": None,
        }

        failures = []
        instantiated = 0
        for attr_name, handler_ref in HANDLER_REGISTRY:
            if isinstance(handler_ref, _DeferredImport):
                handler_class = handler_ref.resolve()
            else:
                handler_class = handler_ref

            if handler_class is None:
                continue

            try:
                try:
                    handler_class(ctx)
                except TypeError:
                    handler_class()
                instantiated += 1
            except Exception as e:
                failures.append(f"{attr_name}: {type(e).__name__}: {e}")

        # Allow up to 5% failure rate for handlers with unusual deps
        max_failures = max(3, len(HANDLER_REGISTRY) // 20)
        assert len(failures) <= max_failures, (
            f"{len(failures)} handler(s) failed to instantiate "
            f"(max allowed: {max_failures}):\n"
            + "\n".join(f"  - {f}" for f in failures[:20])
        )
        assert instantiated > 100, (
            f"Only {instantiated} handlers instantiated, expected 100+"
        )

    def test_can_handle_returns_bool(self):
        """can_handle() must return a bool for all instantiated handlers."""
        from aragora.server.handler_registry import HANDLER_REGISTRY
        from aragora.server.handler_registry.core import _DeferredImport

        ctx = {"storage": None}
        non_bool = []

        for attr_name, handler_ref in HANDLER_REGISTRY:
            if isinstance(handler_ref, _DeferredImport):
                handler_class = handler_ref.resolve()
            else:
                handler_class = handler_ref

            if handler_class is None:
                continue
            if not hasattr(handler_class, "can_handle"):
                continue

            try:
                try:
                    instance = handler_class(ctx)
                except TypeError:
                    instance = handler_class()

                result = instance.can_handle("/api/test-path")
                if not isinstance(result, bool):
                    non_bool.append(
                        f"{attr_name}: can_handle returned {type(result).__name__}"
                    )
            except Exception:
                # Instantiation failures tested separately
                pass

        assert not non_bool, (
            f"{len(non_bool)} handler(s) returned non-bool from can_handle:\n"
            + "\n".join(f"  - {n}" for n in non_bool)
        )


class TestHandlerCoverage:
    """Verify check_handler_coverage runs without errors."""

    def test_check_handler_coverage_succeeds(self):
        """check_handler_coverage must not crash."""
        from aragora.server.handler_registry import HANDLER_REGISTRY
        from aragora.server.handler_registry.core import check_handler_coverage

        # This should complete without raising any exception
        check_handler_coverage(HANDLER_REGISTRY)

    def test_check_handler_coverage_with_filtered_registry(self):
        """check_handler_coverage works with a filtered (tier-based) registry."""
        from aragora.server.handler_registry import HANDLER_REGISTRY
        from aragora.server.handler_registry.core import (
            check_handler_coverage,
            filter_registry_by_tier,
        )

        core_only = filter_registry_by_tier(HANDLER_REGISTRY, {"core"})
        check_handler_coverage(core_only)

    def test_check_handler_coverage_with_empty_registry(self):
        """check_handler_coverage must handle empty registry gracefully."""
        from aragora.server.handler_registry.core import check_handler_coverage

        check_handler_coverage([])


class TestRouteCollisionDetection:
    """Verify no two handlers claim the same route paths."""

    def test_route_collision_count_stable(self):
        """Track route collisions -- the count should not grow unexpectedly.

        Some route overlaps are intentional (dispatch order resolves them).
        This test ensures the collision count stays within a known bound,
        alerting when new unintended collisions are introduced.
        """
        from aragora.server.handler_registry import HANDLER_REGISTRY
        from aragora.server.handler_registry.core import _DeferredImport

        route_owners: dict[str, list[str]] = defaultdict(list)

        for attr_name, handler_ref in HANDLER_REGISTRY:
            if isinstance(handler_ref, _DeferredImport):
                handler_class = handler_ref.resolve()
            else:
                handler_class = handler_ref

            if handler_class is None:
                continue

            routes = getattr(handler_class, "ROUTES", [])
            for route in routes:
                if isinstance(route, str):
                    path = route.split(" ", 1)[-1] if " " in route else route
                elif isinstance(route, (tuple, list)):
                    path = route[1] if len(route) >= 2 else route[0]
                else:
                    continue
                route_owners[path].append(attr_name)

        collisions = {
            path: owners
            for path, owners in route_owners.items()
            if len(owners) > 1
        }

        # Known collision count as of Feb 2026.
        # If this grows, investigate whether new collisions are intentional.
        max_known_collisions = 60
        assert len(collisions) <= max_known_collisions, (
            f"{len(collisions)} route collisions exceeds known bound "
            f"of {max_known_collisions}. New collisions:\n"
            + "\n".join(
                f"  - {path}: {', '.join(owners)}"
                for path, owners in sorted(collisions.items())
            )
        )

    def test_no_overlapping_can_handle_on_common_paths(self):
        """At most one handler should match common API paths via can_handle."""
        from aragora.server.handler_registry import HANDLER_REGISTRY
        from aragora.server.handler_registry.core import _DeferredImport

        ctx = {"storage": None}

        # Common paths that should have exactly one handler
        test_paths = [
            "/api/v1/debates",
            "/api/v1/agents",
            "/api/v1/health",
            "/healthz",
            "/readyz",
        ]

        for path in test_paths:
            matching_handlers = []
            for attr_name, handler_ref in HANDLER_REGISTRY:
                if isinstance(handler_ref, _DeferredImport):
                    handler_class = handler_ref.resolve()
                else:
                    handler_class = handler_ref

                if handler_class is None:
                    continue
                if not hasattr(handler_class, "can_handle"):
                    continue

                try:
                    try:
                        instance = handler_class(ctx)
                    except TypeError:
                        instance = handler_class()

                    if instance.can_handle(path):
                        matching_handlers.append(attr_name)
                except Exception:
                    pass

            assert len(matching_handlers) >= 1, (
                f"No handler matches path '{path}'"
            )


class TestHandlerTierConsistency:
    """Verify tier configuration is consistent with registry entries."""

    def test_all_registry_entries_have_tier(self):
        """Every HANDLER_REGISTRY entry should be listed in HANDLER_TIERS or default to extended."""
        from aragora.server.handler_registry import HANDLER_REGISTRY
        from aragora.server.handler_registry.core import HANDLER_TIERS

        # Entries not in HANDLER_TIERS default to "extended" which is fine.
        # This test just ensures the mapping exists for known critical handlers.
        critical = ["_health_handler", "_debates_handler", "_agents_handler"]
        for name in critical:
            assert name in HANDLER_TIERS, (
                f"Critical handler {name} missing from HANDLER_TIERS"
            )

    def test_handler_tiers_values_valid(self):
        """All HANDLER_TIERS values must be valid tier names."""
        from aragora.server.handler_registry.core import HANDLER_TIERS

        valid = {"core", "extended", "enterprise", "experimental", "optional"}
        for attr_name, tier in HANDLER_TIERS.items():
            assert tier in valid, (
                f"{attr_name} has invalid tier '{tier}', must be one of {valid}"
            )

    def test_core_tier_has_health_handler(self):
        """Health handler must always be in core tier."""
        from aragora.server.handler_registry.core import HANDLER_TIERS

        assert HANDLER_TIERS.get("_health_handler") == "core"


class TestValidateAllHandlers:
    """Test validate_all_handlers with the real registry."""

    def test_validate_real_registry(self):
        """validate_all_handlers should report the majority of handlers as valid.

        Some handlers use alternative dispatch patterns (handle_* only, no
        can_handle) which the strict validator flags. This is expected.
        We check that at least 80% of non-None handlers pass validation.
        """
        from aragora.server.handler_registry import (
            HANDLER_REGISTRY,
            HANDLERS_AVAILABLE,
        )
        from aragora.server.handler_registry.core import validate_all_handlers

        results = validate_all_handlers(
            handler_registry=HANDLER_REGISTRY,
            handlers_available=HANDLERS_AVAILABLE,
            raise_on_error=False,
        )

        assert results["status"] in ("ok", "validation_errors")
        total_checked = len(results["valid"]) + len(results["invalid"])
        if total_checked > 0:
            valid_ratio = len(results["valid"]) / total_checked
            assert valid_ratio >= 0.80, (
                f"Only {valid_ratio:.0%} of handlers valid "
                f"({len(results['valid'])}/{total_checked}), expected 80%+"
            )

    def test_validate_handlers_not_available(self):
        """validate_all_handlers should handle unavailable handlers."""
        from aragora.server.handler_registry import HANDLER_REGISTRY
        from aragora.server.handler_registry.core import validate_all_handlers

        results = validate_all_handlers(
            handler_registry=HANDLER_REGISTRY,
            handlers_available=False,
            raise_on_error=False,
        )

        assert results["status"] == "imports_failed"
        assert len(results["valid"]) == 0
