from __future__ import annotations

import pytest

from aragora.server.openapi import generate_openapi_schema
from aragora.server.openapi.endpoints.agent_bridge import AGENT_BRIDGE_ENDPOINTS
from aragora.server.openapi.endpoints.response_schemas import (
    AGENT_BRIDGE_EVENTS_RESPONSE_SCHEMA,
    AGENT_BRIDGE_RUN_DETAIL_SCHEMA,
    AGENT_BRIDGE_RUN_LIST_RESPONSE_SCHEMA,
    AGENT_BRIDGE_TRANSCRIPT_RESPONSE_SCHEMA,
)


@pytest.fixture(autouse=True)
def _openapi_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAGORA_SECRETS_STRICT", "false")


def test_agent_bridge_operations_are_registered() -> None:
    schema = generate_openapi_schema()
    paths = schema["paths"]

    assert "/api/v1/agent-bridge/runs" in paths
    assert "/api/v1/agent-bridge/runs/{run_id}" in paths
    assert "/api/v1/agent-bridge/runs/{run_id}/events" in paths
    assert "/api/v1/agent-bridge/runs/{run_id}/transcript" in paths


def test_agent_bridge_response_schemas_match_contract_constants() -> None:
    schema = generate_openapi_schema()
    paths = schema["paths"]

    assert (
        paths["/api/v1/agent-bridge/runs"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        == AGENT_BRIDGE_RUN_LIST_RESPONSE_SCHEMA
    )
    assert (
        paths["/api/v1/agent-bridge/runs/{run_id}"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        == AGENT_BRIDGE_RUN_DETAIL_SCHEMA
    )
    assert (
        paths["/api/v1/agent-bridge/runs/{run_id}/events"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        == AGENT_BRIDGE_EVENTS_RESPONSE_SCHEMA
    )
    assert (
        paths["/api/v1/agent-bridge/runs/{run_id}/transcript"]["get"]["responses"]["200"][
            "content"
        ]["application/json"]["schema"]
        == AGENT_BRIDGE_TRANSCRIPT_RESPONSE_SCHEMA
    )


def test_agent_bridge_operations_have_operation_ids() -> None:
    schema = generate_openapi_schema()
    paths = schema["paths"]

    for path, methods in AGENT_BRIDGE_ENDPOINTS.items():
        for method_name in methods:
            operation_id = paths[path][method_name]["operationId"]
            assert isinstance(operation_id, str)
            assert operation_id
