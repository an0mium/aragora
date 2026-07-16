from pathlib import Path

import pytest

import scripts.contract_drift_inventory as inventory
from scripts import check_contract_drift_ratchet as ratchet


def _build(py=(), ts=(), resolved=None, handlers=()):
    live = dict.fromkeys(inventory.CATEGORIES, frozenset())
    live["python_sdk_drift"], live["typescript_sdk_drift"] = set(py), set(ts)
    return inventory._build_inventory(
        live,
        handler_paths=set(handlers),
        generated_typescript_drift=set(),
        resolved_baseline_entries=resolved or {},
        source_sha="0" * 40,
    )


def test_python_and_typescript_request_variants() -> None:
    assert inventory.extract_python_endpoints(
        'self._client.request("GET", "/api/v1/a")\n'
        'await self._client._request("POST", f"/api/v1/b/{item_id}")\n'
        'self._client.request("GET", "/scim/v2/users")'
    ) == {("get", "/api/a"), ("post", "/api/b/{param}"), ("get", "/scim/v2/users")}
    assert inventory.extract_typescript_endpoints(
        "this.client.request('GET', '/auth/x');"
        "this.client.post(`/api/v1/y/${id}`);"
        "this.invoke('legacy', [id], 'DELETE', `/api/v1/z/${id}`);"
    ) == {("get", "/auth/x"), ("post", "/api/y/{param}"), ("delete", "/api/z/{param}")}


def test_generated_namespace_exclusion_and_openapi_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "curated.ts").write_text("this.client.request('GET','/api/v1/a')")
    (tmp_path / "openapi.ts").write_text("this.client.request('GET','/api/v1/generated')")
    namespaces, generated = inventory.scan_typescript_sdk_by_namespace(tmp_path)
    assert (set().union(*namespaces.values()), generated) == (
        {("get", "/api/a")},
        {("get", "/api/generated")},
    )
    malformed = tmp_path / "openapi.json"
    malformed.write_text('{"paths":[]}')
    with pytest.raises(inventory.InventoryError, match="paths"):
        inventory.load_openapi_endpoints([malformed])
    malformed.write_text('{"paths":{"/api/x":{"get":null}}}')
    with pytest.raises(inventory.InventoryError, match="operation"):
        inventory.load_openapi_endpoints([malformed])
    with pytest.raises(inventory.InventoryError, match="not found"):
        inventory.load_openapi_endpoints([tmp_path / "missing.json"])


def test_deduplication_and_method_state_coverage() -> None:
    accepted = _build(
        py={("get", "/api/shared")},
        ts={("post", "/api/shared")},
        resolved={"python_sdk_drift": {"POST /api/shared"}},
    )
    current = _build(
        py={("get", "/api/shared"), ("post", "/api/shared")},
        ts={("post", "/api/shared")},
        resolved={"python_sdk_drift": set()},
    )
    assert accepted["summary"]["raw_live_total"] == 2
    assert accepted["summary"]["deduplicated_live_total"] == 1
    counts = ratchet._count_current_total(accepted)
    assert counts["raw_total_items"] == 2
    assert counts["deduplicated_live_items"] == 1 and counts["total_items"] == 0
    assert (
        "live category method growth" in inventory.inventory_coverage_errors(current, accepted)[0]
    )
    resolved = _build(py=set(), resolved={"python_sdk_drift": {"GET /api/shared"}})
    assert inventory.inventory_coverage_errors(resolved, _build(py={("get", "/api/shared")})) == []
    endpoint = {("get", "/api/budgets/{param}/overrides")}
    stale = _build(py=endpoint, handlers={"/api/budgets/{param}"})
    candidate = _build(py=endpoint, handlers={"/api/budgets/*"})
    assert stale["items"][0]["classification"] == "stale-sdk"
    assert "classification changed" in inventory.inventory_coverage_errors(candidate, stale)[0]


def test_handler_registry_resolution_fails_closed(monkeypatch) -> None:
    import aragora.server.handler_registry as registry
    from scripts import check_sdk_parity

    monkeypatch.setattr(registry, "HANDLER_REGISTRY", [("broken", None)])
    with pytest.raises(inventory.InventoryError, match="incomplete"):
        inventory._handlers(check_sdk_parity)
