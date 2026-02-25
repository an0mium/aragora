"""Tests for recently added Python SDK parity routes."""

from __future__ import annotations

from unittest.mock import call, patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient
from aragora_sdk.namespaces.audit import AsyncAuditAPI
from aragora_sdk.namespaces.debates import AsyncDebatesAPI
from aragora_sdk.namespaces.selection import AsyncSelectionAPI
from aragora_sdk.namespaces.tasks import AsyncTasksAPI
from aragora_sdk.namespaces.templates import AsyncTemplatesAPI


class TestSyncParityRoutes:
    """Sync route mapping coverage for newly added parity paths."""

    def test_new_sync_route_mappings(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"ok": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            client.audit.get_resource_history("debate", "deb_123")
            client.selection.get_scorer("elo-scorer")
            client.selection.get_team_selector("diversity-selector")
            client.selection.get_role_assigner("capability-assigner")
            client.debates.get_shared("share_123")
            client.tasks.get("task_123")
            client.tasks.update("task_123", status="done")
            client.tasks.delete("task_123")
            client.templates.get_registered("tpl_123")
            client.templates.update_registered("tpl_123", name="Updated")
            client.templates.delete_registered("tpl_123")

            expected_calls = [
                call(
                    "GET",
                    "/api/v1/audit/resource/deb_123/history",
                    params={"resource_type": "debate"},
                ),
                call("GET", "/api/v1/selection/scorers/elo-scorer"),
                call("GET", "/api/v1/selection/team-selectors/diversity-selector"),
                call("GET", "/api/v1/selection/role-assigners/capability-assigner"),
                call("GET", "/api/v1/shared/share_123"),
                call("GET", "/api/v2/tasks/task_123"),
                call("PUT", "/api/v2/tasks/task_123", json={"status": "done"}),
                call("DELETE", "/api/v2/tasks/task_123"),
                call("GET", "/api/v1/templates/registry/tpl_123"),
                call("PUT", "/api/v1/templates/registry/tpl_123", json={"name": "Updated"}),
                call("DELETE", "/api/v1/templates/registry/tpl_123"),
            ]
            mock_request.assert_has_calls(expected_calls)
            client.close()


class TestAsyncParityRoutes:
    """Async route mapping coverage for newly added parity paths."""

    @pytest.mark.asyncio
    async def test_new_async_route_mappings(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"ok": True}
            async with AragoraAsyncClient(
                base_url="https://api.aragora.ai", api_key="test-key"
            ) as client:
                audit = AsyncAuditAPI(client)
                selection = AsyncSelectionAPI(client)
                debates = AsyncDebatesAPI(client)
                tasks = AsyncTasksAPI(client)
                templates = AsyncTemplatesAPI(client)

                await audit.get_resource_history("debate", "deb_123")
                await selection.get_scorer("elo-scorer")
                await selection.get_team_selector("diversity-selector")
                await selection.get_role_assigner("capability-assigner")
                await debates.get_shared("share_123")
                await tasks.get("task_123")
                await tasks.update("task_123", status="done")
                await tasks.delete("task_123")
                await templates.get_registered("tpl_123")
                await templates.update_registered("tpl_123", name="Updated")
                await templates.delete_registered("tpl_123")

                expected_calls = [
                    call(
                        "GET",
                        "/api/v1/audit/resource/deb_123/history",
                        params={"resource_type": "debate"},
                    ),
                    call("GET", "/api/v1/selection/scorers/elo-scorer"),
                    call("GET", "/api/v1/selection/team-selectors/diversity-selector"),
                    call("GET", "/api/v1/selection/role-assigners/capability-assigner"),
                    call("GET", "/api/v1/shared/share_123"),
                    call("GET", "/api/v2/tasks/task_123"),
                    call("PUT", "/api/v2/tasks/task_123", json={"status": "done"}),
                    call("DELETE", "/api/v2/tasks/task_123"),
                    call("GET", "/api/v1/templates/registry/tpl_123"),
                    call("PUT", "/api/v1/templates/registry/tpl_123", json={"name": "Updated"}),
                    call("DELETE", "/api/v1/templates/registry/tpl_123"),
                ]
                mock_request.assert_has_calls(expected_calls)
