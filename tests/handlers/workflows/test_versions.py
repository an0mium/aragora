"""Tests for workflow version management (aragora/server/handlers/workflows/versions.py).

Covers both async functions exported by the versions module:
- get_workflow_versions: retrieve version history with tenant and limit filtering
- restore_workflow_version: restore a workflow to a specific historical version

Each function is tested for:
- Happy-path behavior
- Error/edge cases (not found, empty results)
- Interaction with store and update_workflow
- Default parameter values
- Argument forwarding
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.workflows.versions import (
    get_workflow_versions,
    restore_workflow_version,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATCH_MODULE = "aragora.server.handlers.workflows.versions"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_workflow(
    workflow_id: str = "wf_test",
    name: str = "Test Workflow",
    version: str = "1.0.0",
) -> MagicMock:
    """Create a mock WorkflowDefinition instance."""
    wf = MagicMock()
    wf.id = workflow_id
    wf.name = name
    wf.version = version
    wf.to_dict.return_value = {
        "id": workflow_id,
        "name": name,
        "version": version,
    }
    wf.clone.return_value = wf  # clone returns itself by default
    return wf


def _make_version_entry(
    workflow_id: str = "wf_test",
    version: str = "1.0.0",
    name: str = "Test Workflow",
) -> dict[str, Any]:
    """Create a mock version history entry dict."""
    return {
        "workflow_id": workflow_id,
        "version": version,
        "name": name,
        "created_at": "2025-01-01T00:00:00+00:00",
    }


def _make_mock_store(
    versions: list[dict[str, Any]] | None = None,
    get_version_result: MagicMock | None = None,
) -> MagicMock:
    """Create a mock PersistentWorkflowStore."""
    store = MagicMock()
    store.get_versions.return_value = versions or []
    store.get_version.return_value = get_version_result
    return store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_store():
    """Provide a mock store and patch _get_store to return it."""
    store = _make_mock_store()
    with patch(f"{PATCH_MODULE}._get_store", return_value=store):
        yield store


@pytest.fixture
def mock_update():
    """Patch update_workflow to a controllable AsyncMock."""
    with patch(
        f"{PATCH_MODULE}.update_workflow",
        new_callable=AsyncMock,
    ) as update_mock:
        yield update_mock


# ===========================================================================
# get_workflow_versions
# ===========================================================================


class TestGetWorkflowVersions:
    """Tests for get_workflow_versions function."""

    @pytest.mark.asyncio
    async def test_returns_empty_list(self, mock_store):
        """Returns empty list when no versions exist."""
        mock_store.get_versions.return_value = []
        result = await get_workflow_versions("wf_missing")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_version_history(self, mock_store):
        """Returns list of version dicts from the store."""
        versions = [
            _make_version_entry("wf_1", "1.0.0"),
            _make_version_entry("wf_1", "1.0.1"),
            _make_version_entry("wf_1", "1.0.2"),
        ]
        mock_store.get_versions.return_value = versions
        result = await get_workflow_versions("wf_1")
        assert len(result) == 3
        assert result[0]["version"] == "1.0.0"
        assert result[2]["version"] == "1.0.2"

    @pytest.mark.asyncio
    async def test_passes_workflow_id(self, mock_store):
        """Forwards workflow_id to the store."""
        await get_workflow_versions("wf_42")
        args = mock_store.get_versions.call_args[0]
        assert args[0] == "wf_42"

    @pytest.mark.asyncio
    async def test_passes_tenant_id(self, mock_store):
        """Forwards tenant_id to the store."""
        await get_workflow_versions("wf_1", tenant_id="org-acme")
        args = mock_store.get_versions.call_args[0]
        assert args[1] == "org-acme"

    @pytest.mark.asyncio
    async def test_passes_limit(self, mock_store):
        """Forwards limit to the store."""
        await get_workflow_versions("wf_1", limit=5)
        args = mock_store.get_versions.call_args[0]
        assert args[2] == 5

    @pytest.mark.asyncio
    async def test_default_tenant_id(self, mock_store):
        """Default tenant_id is 'default'."""
        await get_workflow_versions("wf_1")
        args = mock_store.get_versions.call_args[0]
        assert args[1] == "default"

    @pytest.mark.asyncio
    async def test_default_limit(self, mock_store):
        """Default limit is 20."""
        await get_workflow_versions("wf_1")
        args = mock_store.get_versions.call_args[0]
        assert args[2] == 20

    @pytest.mark.asyncio
    async def test_custom_tenant_and_limit(self, mock_store):
        """Both tenant_id and limit can be overridden together."""
        await get_workflow_versions("wf_1", tenant_id="corp", limit=100)
        args = mock_store.get_versions.call_args[0]
        assert args[0] == "wf_1"
        assert args[1] == "corp"
        assert args[2] == 100

    @pytest.mark.asyncio
    async def test_calls_get_store(self):
        """Verifies _get_store is called to obtain the store."""
        store = _make_mock_store()
        with patch(f"{PATCH_MODULE}._get_store", return_value=store) as get_store_mock:
            await get_workflow_versions("wf_1")
            get_store_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_single_version(self, mock_store):
        """Returns list with one entry when only one version exists."""
        mock_store.get_versions.return_value = [_make_version_entry("wf_1", "1.0.0")]
        result = await get_workflow_versions("wf_1")
        assert len(result) == 1
        assert result[0]["workflow_id"] == "wf_1"

    @pytest.mark.asyncio
    async def test_limit_zero(self, mock_store):
        """Limit of 0 is passed through to the store."""
        mock_store.get_versions.return_value = []
        await get_workflow_versions("wf_1", limit=0)
        args = mock_store.get_versions.call_args[0]
        assert args[2] == 0

    @pytest.mark.asyncio
    async def test_preserves_version_entry_structure(self, mock_store):
        """Returned dicts match the structure from the store."""
        entry = {
            "workflow_id": "wf_1",
            "version": "2.0.0",
            "name": "My WF",
            "created_at": "2025-06-01T12:00:00+00:00",
            "extra_field": "extra_value",
        }
        mock_store.get_versions.return_value = [entry]
        result = await get_workflow_versions("wf_1")
        assert result[0] is entry


# ===========================================================================
# restore_workflow_version
# ===========================================================================


class TestRestoreWorkflowVersion:
    """Tests for restore_workflow_version function."""

    @pytest.mark.asyncio
    async def test_restores_existing_version(self, mock_store, mock_update):
        """Restores a workflow when the version exists."""
        old_wf = _make_mock_workflow("wf_1", "Old Workflow", "1.0.0")
        mock_store.get_version.return_value = old_wf
        mock_update.return_value = {"id": "wf_1", "name": "Old Workflow", "version": "1.0.1"}

        result = await restore_workflow_version("wf_1", "1.0.0")

        assert result is not None
        assert result["id"] == "wf_1"
        mock_store.get_version.assert_called_once_with("wf_1", "1.0.0")

    @pytest.mark.asyncio
    async def test_returns_none_when_version_not_found(self, mock_store, mock_update):
        """Returns None when the specified version does not exist."""
        mock_store.get_version.return_value = None
        result = await restore_workflow_version("wf_1", "99.0.0")
        assert result is None
        mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_clones_old_workflow(self, mock_store, mock_update):
        """Clones the old workflow with the original ID and name."""
        old_wf = _make_mock_workflow("wf_1", "Restored WF", "1.0.0")
        cloned_wf = _make_mock_workflow("wf_1", "Restored WF", "1.0.0")
        old_wf.clone.return_value = cloned_wf
        mock_store.get_version.return_value = old_wf
        mock_update.return_value = cloned_wf.to_dict()

        await restore_workflow_version("wf_1", "1.0.0")

        old_wf.clone.assert_called_once_with(new_id="wf_1", new_name="Restored WF")

    @pytest.mark.asyncio
    async def test_calls_update_workflow(self, mock_store, mock_update):
        """Calls update_workflow with the cloned workflow's dict."""
        old_wf = _make_mock_workflow("wf_1", "WF", "1.0.0")
        cloned_wf = _make_mock_workflow("wf_1", "WF", "1.0.0")
        cloned_dict = {"id": "wf_1", "name": "WF", "version": "1.0.0"}
        cloned_wf.to_dict.return_value = cloned_dict
        old_wf.clone.return_value = cloned_wf
        mock_store.get_version.return_value = old_wf
        mock_update.return_value = cloned_dict

        await restore_workflow_version("wf_1", "1.0.0")

        mock_update.assert_called_once_with("wf_1", cloned_dict, "default")

    @pytest.mark.asyncio
    async def test_passes_tenant_id_to_update(self, mock_store, mock_update):
        """Forwards tenant_id to update_workflow."""
        old_wf = _make_mock_workflow("wf_1", "WF", "1.0.0")
        mock_store.get_version.return_value = old_wf
        mock_update.return_value = old_wf.to_dict()

        await restore_workflow_version("wf_1", "1.0.0", tenant_id="org-acme")

        call_args = mock_update.call_args[0]
        assert call_args[2] == "org-acme"

    @pytest.mark.asyncio
    async def test_default_tenant_id(self, mock_store, mock_update):
        """Default tenant_id is 'default'."""
        old_wf = _make_mock_workflow("wf_1", "WF", "1.0.0")
        mock_store.get_version.return_value = old_wf
        mock_update.return_value = old_wf.to_dict()

        await restore_workflow_version("wf_1", "1.0.0")

        call_args = mock_update.call_args[0]
        assert call_args[2] == "default"

    @pytest.mark.asyncio
    async def test_uses_old_workflow_name_for_clone(self, mock_store, mock_update):
        """Clone uses the old workflow's name attribute."""
        old_wf = _make_mock_workflow("wf_1", "Historical Name", "0.5.0")
        mock_store.get_version.return_value = old_wf
        mock_update.return_value = old_wf.to_dict()

        await restore_workflow_version("wf_1", "0.5.0")

        old_wf.clone.assert_called_once_with(new_id="wf_1", new_name="Historical Name")

    @pytest.mark.asyncio
    async def test_returns_update_result(self, mock_store, mock_update):
        """Returns whatever update_workflow returns."""
        old_wf = _make_mock_workflow("wf_1", "WF", "1.0.0")
        mock_store.get_version.return_value = old_wf
        expected = {"id": "wf_1", "name": "WF", "version": "2.0.0", "restored": True}
        mock_update.return_value = expected

        result = await restore_workflow_version("wf_1", "1.0.0")

        assert result == expected

    @pytest.mark.asyncio
    async def test_update_returns_none(self, mock_store, mock_update):
        """When update_workflow returns None, restore also returns None."""
        old_wf = _make_mock_workflow("wf_1", "WF", "1.0.0")
        mock_store.get_version.return_value = old_wf
        mock_update.return_value = None

        result = await restore_workflow_version("wf_1", "1.0.0")

        assert result is None

    @pytest.mark.asyncio
    async def test_calls_get_store(self):
        """Verifies _get_store is called to obtain the store."""
        store = _make_mock_store()
        with (
            patch(f"{PATCH_MODULE}._get_store", return_value=store) as get_store_mock,
            patch(f"{PATCH_MODULE}.update_workflow", new_callable=AsyncMock),
        ):
            await restore_workflow_version("wf_1", "1.0.0")
            get_store_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_workflow_id_and_version_to_store(self, mock_store, mock_update):
        """Passes correct workflow_id and version to store.get_version."""
        mock_store.get_version.return_value = None
        await restore_workflow_version("wf_abc123", "3.2.1")
        mock_store.get_version.assert_called_once_with("wf_abc123", "3.2.1")


# ===========================================================================
# __all__ exports
# ===========================================================================


class TestModuleExports:
    """Test that the module exports the expected symbols."""

    def test_exports_get_workflow_versions(self):
        """get_workflow_versions is in __all__."""
        from aragora.server.handlers.workflows.versions import __all__ as exports

        assert "get_workflow_versions" in exports

    def test_exports_restore_workflow_version(self):
        """restore_workflow_version is in __all__."""
        from aragora.server.handlers.workflows.versions import __all__ as exports

        assert "restore_workflow_version" in exports

    def test_exports_count(self):
        """Exactly 2 items are exported."""
        from aragora.server.handlers.workflows.versions import __all__ as exports

        assert len(exports) == 2


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_restore_with_empty_version_string(self, mock_store, mock_update):
        """Empty version string is passed through to the store."""
        mock_store.get_version.return_value = None
        result = await restore_workflow_version("wf_1", "")
        assert result is None
        mock_store.get_version.assert_called_once_with("wf_1", "")

    @pytest.mark.asyncio
    async def test_restore_with_different_workflow_ids(self, mock_store, mock_update):
        """Workflow ID is correctly passed to both get_version and update_workflow."""
        old_wf = _make_mock_workflow("wf_xyz", "WF XYZ", "1.0.0")
        mock_store.get_version.return_value = old_wf
        mock_update.return_value = old_wf.to_dict()

        await restore_workflow_version("wf_xyz", "1.0.0")

        mock_store.get_version.assert_called_once_with("wf_xyz", "1.0.0")
        call_args = mock_update.call_args[0]
        assert call_args[0] == "wf_xyz"

    @pytest.mark.asyncio
    async def test_clone_gets_workflow_id_as_new_id(self, mock_store, mock_update):
        """Clone receives the current workflow_id, not the old version's id."""
        old_wf = _make_mock_workflow("wf_original", "Original", "0.1.0")
        mock_store.get_version.return_value = old_wf
        mock_update.return_value = old_wf.to_dict()

        await restore_workflow_version("wf_original", "0.1.0")

        old_wf.clone.assert_called_once_with(new_id="wf_original", new_name="Original")

    @pytest.mark.asyncio
    async def test_get_versions_with_empty_workflow_id(self, mock_store):
        """Empty workflow_id is passed through to the store."""
        mock_store.get_versions.return_value = []
        result = await get_workflow_versions("")
        assert result == []
        args = mock_store.get_versions.call_args[0]
        assert args[0] == ""

    @pytest.mark.asyncio
    async def test_get_versions_returns_store_value_directly(self, mock_store):
        """Return value is exactly what the store returns, no transformation."""
        sentinel = [{"unique": True}]
        mock_store.get_versions.return_value = sentinel
        result = await get_workflow_versions("wf_1")
        assert result is sentinel
