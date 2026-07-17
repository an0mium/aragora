"""Tests for scripts/validate_openapi_routes.py baseline behavior."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

import scripts.validate_openapi_routes as validate_openapi_routes


def test_fail_on_missing_passes_when_only_baseline_drift(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        validate_openapi_routes, "get_handler_routes", lambda: {"/api/v1/a", "/api/v1/b"}
    )
    monkeypatch.setattr(validate_openapi_routes, "get_openapi_routes", lambda _spec: {"/api/v1/b"})

    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "missing_in_spec": ["/api/v1/a"],
                "orphaned_in_spec": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    results = validate_openapi_routes.validate_coverage(
        "ignored.json",
        fail_on_missing=True,
        output_json=False,
        baseline_path=str(baseline),
    )
    assert results["missing_in_spec_count"] == 1
    assert results["new_missing_in_spec_count"] == 0


def test_fail_on_missing_fails_on_new_drift(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        validate_openapi_routes, "get_handler_routes", lambda: {"/api/v1/a", "/api/v1/b"}
    )
    monkeypatch.setattr(validate_openapi_routes, "get_openapi_routes", lambda _spec: {"/api/v1/b"})

    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "missing_in_spec": [],
                "orphaned_in_spec": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        validate_openapi_routes.validate_coverage(
            "ignored.json",
            fail_on_missing=True,
            output_json=False,
            baseline_path=str(baseline),
        )
    assert excinfo.value.code == 1


def test_internal_prefixes_are_excluded_by_default(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        validate_openapi_routes,
        "get_handler_routes",
        lambda: {"/api/v1/control-plane/agents"},
    )
    monkeypatch.setattr(validate_openapi_routes, "get_openapi_routes", lambda _spec: set())

    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"missing_in_spec": [], "orphaned_in_spec": []}\n', encoding="utf-8")

    results = validate_openapi_routes.validate_coverage(
        "ignored.json",
        fail_on_missing=True,
        output_json=False,
        baseline_path=str(baseline),
    )
    assert results["missing_in_spec_count"] == 0


def test_get_handler_routes_resolves_deferred_imports(monkeypatch):
    class DummyHandler:
        ROUTES = ["/api/v1/test/routes"]
        GET_ROUTES = ["/api/v1/test/get"]

    class DummyDeferred:
        def resolve(self):
            return DummyHandler

    fake_registry = types.SimpleNamespace(HANDLER_REGISTRY=[("_dummy", DummyDeferred())])
    monkeypatch.setitem(sys.modules, "aragora.server.handler_registry", fake_registry)

    routes = validate_openapi_routes.get_handler_routes()
    assert "/api/v1/test/routes" in routes
    assert "/api/v1/test/get" in routes


def test_get_handler_routes_includes_api_endpoint_metadata(monkeypatch):
    endpoint = types.SimpleNamespace(path="/api/v1/coordination/fleet/status")

    class DummyHandler:
        def handle(self):
            return None

    setattr(DummyHandler.handle, "_openapi", endpoint)

    fake_registry = types.SimpleNamespace(HANDLER_REGISTRY=[("_dummy", DummyHandler)])
    monkeypatch.setitem(sys.modules, "aragora.server.handler_registry", fake_registry)

    routes = validate_openapi_routes.get_handler_routes()

    assert "/api/v1/coordination/fleet/status" in routes


def test_get_wired_function_routes_accepts_only_called_literal_app_routes(tmp_path: Path):
    registration = tmp_path / "aragora" / "server" / "stream" / "registration.py"
    handler = tmp_path / "aragora" / "server" / "handlers" / "wired.py"
    registration.parent.mkdir(parents=True)
    handler.parent.mkdir(parents=True)
    registration.write_text(
        """
from aragora.server.handlers.wired import register_routes as register_wired
from aragora.server.handlers.wired import register_unused

def install(app):
    register_wired(app)
""".lstrip(),
        encoding="utf-8",
    )
    handler.write_text(
        """
def register_routes(app):
    computed = "/api/v1/computed"
    app.router.add_get("/api/v1/wired", object())
    app.router.add_post("/api/wired-legacy", object())
    app.router.add_put(computed, object())
    other.router.add_delete("/api/v1/other-router", object())
    def dormant():
        app.router.add_get("/api/v1/nested-unwired", object())

def register_unused(app):
    app.router.add_get("/api/v1/unwired", object())
""".lstrip(),
        encoding="utf-8",
    )

    routes = validate_openapi_routes.get_wired_function_routes(registration, tmp_path)

    assert routes == {"/api/v1/wired", "/api/wired-legacy"}
    assert {validate_openapi_routes.normalize_route(route) for route in routes} == {
        "/api/v1/wired",
        "/api/v1/wired-legacy",
    }


def test_get_wired_function_routes_fails_closed_on_unparseable_source(tmp_path: Path):
    registration = tmp_path / "registration.py"
    registration.write_text("def broken(:\n", encoding="utf-8")

    with pytest.raises(validate_openapi_routes.RouteRegistrationScanError):
        validate_openapi_routes.get_wired_function_routes(registration, tmp_path)


def test_wired_function_routes_include_known_server_registrations():
    routes = validate_openapi_routes.get_wired_function_routes()

    assert {
        "/api/v1/accounting/callback",
        "/api/v1/accounting/gusto/callback",
        "/api/v1/accounting/gusto/connect",
        "/api/v1/accounting/gusto/disconnect",
        "/api/v1/accounting/report",
        "/api/v1/codebase/quick-scan",
        "/api/v1/codebase/quick-scans",
    } <= routes


def test_validate_coverage_uses_wired_routes_only_to_disprove_orphans(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(validate_openapi_routes, "get_handler_routes", lambda: set())
    monkeypatch.setattr(
        validate_openapi_routes,
        "get_openapi_routes",
        lambda _spec: {"/api/v1/wired-legacy", "/api/v1/dark"},
    )
    monkeypatch.setattr(
        validate_openapi_routes,
        "get_wired_function_routes",
        lambda: {"/api/wired-legacy"},
    )

    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"missing_in_spec": [], "orphaned_in_spec": []}\n', encoding="utf-8")

    results = validate_openapi_routes.validate_coverage(
        "ignored.json",
        baseline_path=str(baseline),
        include_internal=True,
    )

    assert results["missing_in_spec"] == []
    assert results["orphaned_in_spec"] == ["/api/v1/dark"]
    assert results["served_wired_registration"] == ["/api/v1/wired-legacy"]


def test_validate_coverage_treats_decorator_routes_as_implemented(monkeypatch, tmp_path: Path):
    endpoint = types.SimpleNamespace(path="/api/v1/coordination/swarm/integrator")

    class DummyHandler:
        def handle(self):
            return None

    setattr(DummyHandler.handle, "_openapi", endpoint)

    fake_registry = types.SimpleNamespace(HANDLER_REGISTRY=[("_dummy", DummyHandler)])
    monkeypatch.setitem(sys.modules, "aragora.server.handler_registry", fake_registry)
    monkeypatch.setattr(
        validate_openapi_routes,
        "get_openapi_routes",
        lambda _spec: {"/api/v1/coordination/swarm/integrator"},
    )

    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"missing_in_spec": [], "orphaned_in_spec": []}\n', encoding="utf-8")

    results = validate_openapi_routes.validate_coverage(
        "ignored.json",
        fail_on_missing=False,
        output_json=True,
        baseline_path=str(baseline),
        include_internal=True,
    )

    assert "/api/v1/coordination/swarm/integrator" not in results["orphaned_in_spec"]


def test_get_openapi_routes_includes_sibling_generated_snapshot(tmp_path: Path):
    spec = tmp_path / "openapi.json"
    generated = tmp_path / "openapi_generated.json"
    spec.write_text(json.dumps({"paths": {"/api/v1/canonical": {"get": {}}}}), encoding="utf-8")
    generated.write_text(
        json.dumps({"paths": {"/api/v1/generated": {"get": {}}}}), encoding="utf-8"
    )

    routes = validate_openapi_routes.get_openapi_routes(str(spec))

    assert "/api/v1/canonical" in routes
    assert "/api/v1/generated" in routes


def test_validate_coverage_counts_prompt_engine_registry_routes() -> None:
    results = validate_openapi_routes.validate_coverage(
        "docs/api/openapi.json",
        output_json=True,
    )

    assert "/api/v1/prompt-engine/run" not in results["orphaned_in_spec"]
    assert "/api/v1/prompt-engine/decompose" not in results["orphaned_in_spec"]


def test_filter_served_orphans_drops_can_handle_routes(monkeypatch):
    class ServingHandler:
        def can_handle(self, path: str, method: str = "GET") -> bool:
            return path in ("/api/v1/served/route", "/api/served/route")

    fake_registry = types.SimpleNamespace(HANDLER_REGISTRY=[("_serving", ServingHandler)])
    monkeypatch.setitem(sys.modules, "aragora.server.handler_registry", fake_registry)

    orphaned, served = validate_openapi_routes.filter_served_orphans(
        {"/api/v1/served/route", "/api/v1/dark/route"}
    )

    assert served == {"/api/v1/served/route"}
    assert orphaned == {"/api/v1/dark/route"}


def test_filter_served_orphans_survives_broken_can_handle(monkeypatch):
    class BrokenHandler:
        def can_handle(self, path: str, method: str = "GET") -> bool:
            raise AttributeError("uninitialized state")

    fake_registry = types.SimpleNamespace(HANDLER_REGISTRY=[("_broken", BrokenHandler)])
    monkeypatch.setitem(sys.modules, "aragora.server.handler_registry", fake_registry)

    orphaned, served = validate_openapi_routes.filter_served_orphans({"/api/v1/x"})

    assert orphaned == {"/api/v1/x"}
    assert served == set()


def test_load_internal_prefixes_fails_closed_on_missing_file(tmp_path: Path):
    missing = tmp_path / "nope" / "internal_route_prefixes.json"

    with pytest.raises(SystemExit) as excinfo:
        validate_openapi_routes.load_internal_prefixes(str(missing))
    assert excinfo.value.code == 1


def test_load_internal_prefixes_fails_closed_on_unparseable_file(tmp_path: Path):
    policy = tmp_path / "internal_route_prefixes.json"
    policy.write_text("{not json", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        validate_openapi_routes.load_internal_prefixes(str(policy))
    assert excinfo.value.code == 1


def test_load_internal_prefixes_fails_closed_on_invalid_shape(tmp_path: Path):
    policy = tmp_path / "internal_route_prefixes.json"
    policy.write_text('{"prefixes": "not-a-list"}', encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        validate_openapi_routes.load_internal_prefixes(str(policy))
    assert excinfo.value.code == 1


def test_load_internal_prefixes_reads_repo_policy_by_default():
    prefixes = validate_openapi_routes.load_internal_prefixes()

    assert prefixes
    assert all(p.startswith("/api/") for p in prefixes)


def test_filter_served_orphans_rejects_prefix_matching_can_handle(monkeypatch):
    class BroadHandler:
        def can_handle(self, path: str, method: str = "GET") -> bool:
            return path.startswith("/api/v1/broad") or path.startswith("/api/broad")

    fake_registry = types.SimpleNamespace(HANDLER_REGISTRY=[("_broad", BroadHandler)])
    monkeypatch.setitem(sys.modules, "aragora.server.handler_registry", fake_registry)

    orphaned, served = validate_openapi_routes.filter_served_orphans({"/api/v1/broad/route"})

    # BroadHandler also claims the nonexistent canary path, so its can_handle
    # is non-specific and must not suppress the orphan.
    assert orphaned == {"/api/v1/broad/route"}
    assert served == set()


def test_filter_served_orphans_logs_suppressions(monkeypatch, capsys):
    class ServingHandler:
        def can_handle(self, path: str, method: str = "GET") -> bool:
            return path in ("/api/v1/served/route", "/api/served/route")

    fake_registry = types.SimpleNamespace(HANDLER_REGISTRY=[("_serving", ServingHandler)])
    monkeypatch.setitem(sys.modules, "aragora.server.handler_registry", fake_registry)

    orphaned, served = validate_openapi_routes.filter_served_orphans({"/api/v1/served/route"})

    assert served == {"/api/v1/served/route"}
    assert orphaned == set()
    err = capsys.readouterr().err
    assert "/api/v1/served/route" in err
    assert "ServingHandler" in err


def test_validate_coverage_excludes_served_orphans(monkeypatch, tmp_path: Path):
    class ServingHandler:
        ROUTES = ["/api/v1/declared"]

        def can_handle(self, path: str, method: str = "GET") -> bool:
            return path == "/api/v1/served-only"

    fake_registry = types.SimpleNamespace(HANDLER_REGISTRY=[("_serving", ServingHandler)])
    monkeypatch.setitem(sys.modules, "aragora.server.handler_registry", fake_registry)
    monkeypatch.setattr(
        validate_openapi_routes,
        "get_openapi_routes",
        lambda _spec: {"/api/v1/declared", "/api/v1/served-only", "/api/v1/dark"},
    )

    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"missing_in_spec": [], "orphaned_in_spec": []}\n', encoding="utf-8")

    results = validate_openapi_routes.validate_coverage(
        "ignored.json",
        fail_on_missing=False,
        output_json=True,
        baseline_path=str(baseline),
        include_internal=True,
    )

    assert results["orphaned_in_spec"] == ["/api/v1/dark"]
    assert results["served_undeclared"] == ["/api/v1/served-only"]


def test_is_internal_route_matches_exact_family_not_sibling_names():
    """Round-1 review P2 on #9360: startswith on the prefix without its
    trailing slash overmatches — /api/v1/smear must NOT fall under the
    /api/v1/sme/ internal family."""
    prefixes = ("/api/v1/sme/",)
    assert validate_openapi_routes.is_internal_route("/api/v1/sme/dashboard", prefixes)
    assert validate_openapi_routes.is_internal_route("/api/v1/sme", prefixes)
    assert not validate_openapi_routes.is_internal_route("/api/v1/smear", prefixes)
    assert not validate_openapi_routes.is_internal_route("/api/v1/smear/x", prefixes)
