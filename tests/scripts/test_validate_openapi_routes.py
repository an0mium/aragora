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
