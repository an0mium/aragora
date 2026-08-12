from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from aragora.server.handlers.admin.credits import CreditsAdminHandler
from aragora.server.handlers.features.integrations import IntegrationsHandler
from aragora.server.openapi import generate_openapi_schema
from aragora.storage.integration_memory import InMemoryIntegrationStore


ROOT = Path(__file__).resolve().parents[3]
OPENAPI_PATH = ROOT / "docs/api/openapi.json"
GENERATED_OPENAPI_PATH = ROOT / "docs/api/openapi_generated.json"


def _documented_success_statuses(spec_path: Path, path: str, method: str) -> set[int]:
    spec = json.loads(spec_path.read_text())
    responses = spec["paths"][path][method]["responses"]
    return {int(status) for status in responses if status.isdigit() and 200 <= int(status) < 300}


@pytest.mark.asyncio
async def test_credit_issue_success_statuses_match_runtime_handler() -> None:
    transaction = SimpleNamespace(to_dict=lambda: {"id": "credit-1"})
    manager = SimpleNamespace(issue_credit=AsyncMock(return_value=transaction))
    runtime_method = inspect.unwrap(CreditsAdminHandler.issue_credit)

    with patch(
        "aragora.server.handlers.admin.credits.get_credit_manager",
        return_value=manager,
    ):
        result = await runtime_method(
            CreditsAdminHandler(),
            "org-1",
            {
                "amount_cents": 100,
                "description": "Contract status probe",
            },
            "admin-1",
        )

    runtime_statuses = {result.status_code}
    assert runtime_statuses == {201}
    for path in (
        "/api/admin/credits/{org_id}/issue",
        "/api/v1/admin/credits/{org_id}/issue",
    ):
        assert _documented_success_statuses(OPENAPI_PATH, path, "post") == runtime_statuses


@pytest.mark.asyncio
async def test_integration_put_success_statuses_match_create_and_update_runtime() -> None:
    store = InMemoryIntegrationStore()
    handler = IntegrationsHandler(server_context={})

    with patch(
        "aragora.server.handlers.features.integrations.get_integration_store",
        return_value=store,
    ):
        created = await handler.configure_integration("email", {}, user_id="user-1")
        updated = await handler.configure_integration(
            "email",
            {"enabled": False},
            user_id="user-1",
        )

    runtime_statuses = {created.status_code, updated.status_code}
    assert runtime_statuses == {200, 201}
    for path in (
        "/api/integrations/{type}",
        "/api/v1/integrations/{type}",
    ):
        assert _documented_success_statuses(OPENAPI_PATH, path, "put") == runtime_statuses

    generated_runtime_schema = generate_openapi_schema()
    generated_responses = generated_runtime_schema["paths"]["/api/v1/integrations/{type}"]["put"][
        "responses"
    ]
    assert {
        int(status)
        for status in generated_responses
        if status.isdigit() and 200 <= int(status) < 300
    } == runtime_statuses

    assert (
        _documented_success_statuses(
            GENERATED_OPENAPI_PATH,
            "/api/v1/integrations/{type}",
            "put",
        )
        == runtime_statuses
    )
