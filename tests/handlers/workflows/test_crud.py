"""Tests for workflow CRUD operations (aragora/server/handlers/workflows/crud.py).

Covers all five async functions exported by the crud module:
- list_workflows: list with filtering (tenant, category, tags, search, pagination)
- get_workflow: retrieve by ID and tenant
- create_workflow: create with validation, ID generation, audit
- update_workflow: update with version increment, validation, audit
- delete_workflow: delete with audit logging

Each function is tested for:
- Happy-path behavior
- Error/edge cases
- Interaction with store and audit
- Validation failures
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from aragora.server.handlers.workflows.crud import (
    list_workflows,
    get_workflow,
    create_workflow,
    update_workflow,
    delete_workflow,
    _get_workflow_definition_cls,
    _get_audit_fn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_workflow(
    workflow_id: str = "wf_test",
    name: str = "Test Workflow",
    version: str = "1.0.0",
    created_by: str = "user-1",
    tenant_id: str = "default",
    created_at: datetime | None = None,
) -> MagicMock:
    """Create a mock WorkflowDefinition instance."""
    wf = MagicMock()
    wf.id = workflow_id
    wf.name = name
    wf.version = version
    wf.created_by = created_by
    wf.tenant_id = tenant_id
    wf.created_at = created_at or datetime(2025, 1, 1, tzinfo=timezone.utc)
    wf.to_dict.return_value = {
        "id": workflow_id,
        "name": name,
        "version": version,
        "created_by": created_by,
        "tenant_id": tenant_id,
    }
    wf.validate.return_value = (True, [])
    return wf


def _make_mock_store(
    workflows: list[MagicMock] | None = None,
    total: int = 0,
    get_result: MagicMock | None = None,
    delete_result: bool = True,
) -> MagicMock:
    """Create a mock PersistentWorkflowStore."""
    store = MagicMock()
    store.list_workflows.return_value = (workflows or [], total)
    store.get_workflow.return_value = get_result
    store.save_workflow.return_value = None
    store.save_version.return_value = None
    store.delete_workflow.return_value = delete_result
    return store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PATCH_MODULE = "aragora.server.handlers.workflows.crud"


@pytest.fixture
def mock_store():
    """Provide a mock store and patch _get_store to return it."""
    store = _make_mock_store()
    with patch(f"{PATCH_MODULE}._get_store", return_value=store):
        yield store


@pytest.fixture
def mock_audit():
    """Patch audit_data to a mock for verifying audit calls."""
    audit_mock = MagicMock()
    with (
        patch(f"{PATCH_MODULE}.audit_data", audit_mock),
        patch(f"{PATCH_MODULE}._get_audit_fn", return_value=audit_mock),
    ):
        yield audit_mock


@pytest.fixture
def mock_wf_cls():
    """Patch WorkflowDefinition to a controllable mock class."""
    cls_mock = MagicMock()
    with (
        patch(f"{PATCH_MODULE}.WorkflowDefinition", cls_mock),
        patch(f"{PATCH_MODULE}._get_workflow_definition_cls", return_value=cls_mock),
    ):
        yield cls_mock


# ===========================================================================
# list_workflows
# ===========================================================================


class TestListWorkflows:
    """Test list_workflows function."""

    @pytest.mark.asyncio
    async def test_returns_empty_list(self, mock_store):
        """Listing with no workflows returns empty list and zero count."""
        mock_store.list_workflows.return_value = ([], 0)
        result = await list_workflows()
        assert result["workflows"] == []
        assert result["total_count"] == 0
        assert result["limit"] == 50
        assert result["offset"] == 0

    @pytest.mark.asyncio
    async def test_returns_workflows_as_dicts(self, mock_store):
        """Workflows are serialized via to_dict."""
        wf1 = _make_mock_workflow("wf_1", "First")
        wf2 = _make_mock_workflow("wf_2", "Second")
        mock_store.list_workflows.return_value = ([wf1, wf2], 2)

        result = await list_workflows()

        assert len(result["workflows"]) == 2
        assert result["total_count"] == 2
        wf1.to_dict.assert_called_once()
        wf2.to_dict.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_tenant_id(self, mock_store):
        """Tenant ID is forwarded to the store."""
        mock_store.list_workflows.return_value = ([], 0)
        await list_workflows(tenant_id="acme")
        call_kwargs = mock_store.list_workflows.call_args[1]
        assert call_kwargs["tenant_id"] == "acme"

    @pytest.mark.asyncio
    async def test_passes_category_filter(self, mock_store):
        """Category filter is forwarded to the store."""
        mock_store.list_workflows.return_value = ([], 0)
        await list_workflows(category="legal")
        call_kwargs = mock_store.list_workflows.call_args[1]
        assert call_kwargs["category"] == "legal"

    @pytest.mark.asyncio
    async def test_passes_tags_filter(self, mock_store):
        """Tags filter is forwarded to the store."""
        mock_store.list_workflows.return_value = ([], 0)
        await list_workflows(tags=["review", "compliance"])
        call_kwargs = mock_store.list_workflows.call_args[1]
        assert call_kwargs["tags"] == ["review", "compliance"]

    @pytest.mark.asyncio
    async def test_passes_search_filter(self, mock_store):
        """Search query is forwarded to the store."""
        mock_store.list_workflows.return_value = ([], 0)
        await list_workflows(search="contract")
        call_kwargs = mock_store.list_workflows.call_args[1]
        assert call_kwargs["search"] == "contract"

    @pytest.mark.asyncio
    async def test_passes_pagination(self, mock_store):
        """Limit and offset are forwarded to the store."""
        mock_store.list_workflows.return_value = ([], 0)
        await list_workflows(limit=10, offset=20)
        call_kwargs = mock_store.list_workflows.call_args[1]
        assert call_kwargs["limit"] == 10
        assert call_kwargs["offset"] == 20

    @pytest.mark.asyncio
    async def test_limit_and_offset_in_response(self, mock_store):
        """Response includes requested limit and offset."""
        mock_store.list_workflows.return_value = ([], 0)
        result = await list_workflows(limit=25, offset=5)
        assert result["limit"] == 25
        assert result["offset"] == 5

    @pytest.mark.asyncio
    async def test_default_parameters(self, mock_store):
        """Default parameters are correct."""
        mock_store.list_workflows.return_value = ([], 0)
        await list_workflows()
        call_kwargs = mock_store.list_workflows.call_args[1]
        assert call_kwargs["tenant_id"] == "default"
        assert call_kwargs["category"] is None
        assert call_kwargs["tags"] is None
        assert call_kwargs["search"] is None
        assert call_kwargs["limit"] == 50
        assert call_kwargs["offset"] == 0

    @pytest.mark.asyncio
    async def test_total_count_reflects_store(self, mock_store):
        """Total count matches what the store reports, not just returned items."""
        wf = _make_mock_workflow("wf_1", "Only One")
        # Store says 100 total but returns 1 item (pagination)
        mock_store.list_workflows.return_value = ([wf], 100)
        result = await list_workflows(limit=1, offset=0)
        assert result["total_count"] == 100
        assert len(result["workflows"]) == 1


# ===========================================================================
# get_workflow
# ===========================================================================


class TestGetWorkflow:
    """Test get_workflow function."""

    @pytest.mark.asyncio
    async def test_returns_workflow_dict(self, mock_store):
        """Returns workflow serialized as dict when found."""
        wf = _make_mock_workflow("wf_123", "My WF")
        mock_store.get_workflow.return_value = wf

        result = await get_workflow("wf_123")
        assert result is not None
        assert result["id"] == "wf_123"
        wf.to_dict.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, mock_store):
        """Returns None when workflow does not exist."""
        mock_store.get_workflow.return_value = None

        result = await get_workflow("wf_missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_passes_workflow_id_and_tenant(self, mock_store):
        """Passes workflow_id and tenant_id to the store."""
        mock_store.get_workflow.return_value = None
        await get_workflow("wf_42", tenant_id="org-acme")
        mock_store.get_workflow.assert_called_once_with("wf_42", "org-acme")

    @pytest.mark.asyncio
    async def test_default_tenant_id(self, mock_store):
        """Default tenant_id is 'default'."""
        mock_store.get_workflow.return_value = None
        await get_workflow("wf_1")
        mock_store.get_workflow.assert_called_once_with("wf_1", "default")


# ===========================================================================
# create_workflow
# ===========================================================================


class TestCreateWorkflow:
    """Test create_workflow function."""

    @pytest.mark.asyncio
    async def test_creates_workflow_with_provided_id(self, mock_store, mock_audit, mock_wf_cls):
        """When data includes an ID, it is preserved."""
        wf_mock = _make_mock_workflow("wf_custom")
        mock_wf_cls.from_dict.return_value = wf_mock

        data = {"id": "wf_custom", "name": "Custom WF"}
        result = await create_workflow(data)

        assert result["id"] == "wf_custom"
        mock_store.save_workflow.assert_called_once_with(wf_mock)
        mock_store.save_version.assert_called_once_with(wf_mock)

    @pytest.mark.asyncio
    async def test_generates_id_when_missing(self, mock_store, mock_audit, mock_wf_cls):
        """When data has no ID, a wf_* ID is generated."""
        wf_mock = _make_mock_workflow("wf_generated")
        mock_wf_cls.from_dict.return_value = wf_mock

        data = {"name": "Auto ID"}
        result = await create_workflow(data)

        # The data dict should have gotten an ID assigned
        assert data["id"].startswith("wf_")
        assert len(data["id"]) > 3  # wf_ + at least some hex chars

    @pytest.mark.asyncio
    async def test_generates_id_when_empty_string(self, mock_store, mock_audit, mock_wf_cls):
        """Empty string ID triggers auto-generation."""
        wf_mock = _make_mock_workflow("wf_gen")
        mock_wf_cls.from_dict.return_value = wf_mock

        data = {"id": "", "name": "Empty ID"}
        await create_workflow(data)
        assert data["id"].startswith("wf_")

    @pytest.mark.asyncio
    async def test_sets_tenant_id(self, mock_store, mock_audit, mock_wf_cls):
        """Tenant ID is set on the data before creating."""
        wf_mock = _make_mock_workflow()
        mock_wf_cls.from_dict.return_value = wf_mock

        data = {"name": "WF"}
        await create_workflow(data, tenant_id="org-acme")
        assert data["tenant_id"] == "org-acme"

    @pytest.mark.asyncio
    async def test_sets_created_by(self, mock_store, mock_audit, mock_wf_cls):
        """Created by user is set on the data."""
        wf_mock = _make_mock_workflow()
        mock_wf_cls.from_dict.return_value = wf_mock

        data = {"name": "WF"}
        await create_workflow(data, created_by="alice")
        assert data["created_by"] == "alice"

    @pytest.mark.asyncio
    async def test_sets_timestamps(self, mock_store, mock_audit, mock_wf_cls):
        """Created_at and updated_at are set to current time."""
        wf_mock = _make_mock_workflow()
        mock_wf_cls.from_dict.return_value = wf_mock

        data = {"name": "WF"}
        await create_workflow(data)
        assert "created_at" in data
        assert "updated_at" in data
        assert data["created_at"] == data["updated_at"]
        # Verify it's a valid ISO format datetime
        dt = datetime.fromisoformat(data["created_at"])
        assert dt.tzinfo is not None  # UTC-aware

    @pytest.mark.asyncio
    async def test_validation_failure_raises_value_error(self, mock_store, mock_audit, mock_wf_cls):
        """Invalid workflow raises ValueError with error details."""
        wf_mock = MagicMock()
        wf_mock.validate.return_value = (False, ["name missing", "no steps"])
        mock_wf_cls.from_dict.return_value = wf_mock

        data = {"name": "Bad WF"}
        with pytest.raises(ValueError, match="Invalid workflow"):
            await create_workflow(data)

        # Store should NOT have been called
        mock_store.save_workflow.assert_not_called()
        mock_store.save_version.assert_not_called()

    @pytest.mark.asyncio
    async def test_validation_error_includes_all_errors(self, mock_store, mock_audit, mock_wf_cls):
        """All validation errors are joined in the exception message."""
        wf_mock = MagicMock()
        wf_mock.validate.return_value = (False, ["error A", "error B"])
        mock_wf_cls.from_dict.return_value = wf_mock

        data = {"name": "Bad"}
        with pytest.raises(ValueError, match="error A, error B"):
            await create_workflow(data)

    @pytest.mark.asyncio
    async def test_saves_workflow_and_version(self, mock_store, mock_audit, mock_wf_cls):
        """Both save_workflow and save_version are called."""
        wf_mock = _make_mock_workflow()
        mock_wf_cls.from_dict.return_value = wf_mock

        data = {"name": "WF"}
        await create_workflow(data)

        mock_store.save_workflow.assert_called_once_with(wf_mock)
        mock_store.save_version.assert_called_once_with(wf_mock)

    @pytest.mark.asyncio
    async def test_audit_logged_on_create(self, mock_store, mock_audit, mock_wf_cls):
        """Audit data is logged after creation."""
        wf_mock = _make_mock_workflow("wf_audited", "Audited WF")
        mock_wf_cls.from_dict.return_value = wf_mock

        data = {"name": "Audited WF"}
        await create_workflow(data, tenant_id="t1", created_by="bob")

        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["user_id"] == "bob"
        assert call_kwargs["resource_type"] == "workflow"
        assert call_kwargs["action"] == "create"
        assert call_kwargs["workflow_name"] == "Audited WF"
        assert call_kwargs["tenant_id"] == "t1"

    @pytest.mark.asyncio
    async def test_audit_uses_system_when_no_created_by(self, mock_store, mock_audit, mock_wf_cls):
        """When created_by is empty, audit uses 'system'."""
        wf_mock = _make_mock_workflow("wf_sys", "Sys WF")
        mock_wf_cls.from_dict.return_value = wf_mock

        data = {"name": "Sys WF"}
        await create_workflow(data, created_by="")

        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["user_id"] == "system"

    @pytest.mark.asyncio
    async def test_returns_workflow_dict(self, mock_store, mock_audit, mock_wf_cls):
        """Returns the created workflow as a dict."""
        wf_mock = _make_mock_workflow("wf_ret", "Return WF")
        wf_mock.to_dict.return_value = {"id": "wf_ret", "name": "Return WF"}
        mock_wf_cls.from_dict.return_value = wf_mock

        data = {"name": "Return WF"}
        result = await create_workflow(data)
        assert result == {"id": "wf_ret", "name": "Return WF"}

    @pytest.mark.asyncio
    async def test_default_tenant_and_created_by(self, mock_store, mock_audit, mock_wf_cls):
        """Defaults: tenant_id='default', created_by=''."""
        wf_mock = _make_mock_workflow()
        mock_wf_cls.from_dict.return_value = wf_mock

        data = {"name": "Defaults"}
        await create_workflow(data)
        assert data["tenant_id"] == "default"
        assert data["created_by"] == ""


# ===========================================================================
# update_workflow
# ===========================================================================


class TestUpdateWorkflow:
    """Test update_workflow function."""

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, mock_store):
        """Returns None if the workflow doesn't exist."""
        mock_store.get_workflow.return_value = None
        result = await update_workflow("wf_missing", {"name": "Updated"})
        assert result is None
        mock_store.save_workflow.assert_not_called()

    @pytest.mark.asyncio
    async def test_preserves_metadata(self, mock_store, mock_audit, mock_wf_cls):
        """Preserves id, tenant_id, created_by, created_at from existing workflow."""
        existing = _make_mock_workflow("wf_1", "Original", version="1.0.0", created_by="alice")
        existing.created_at = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_store.get_workflow.return_value = existing

        updated_wf = _make_mock_workflow("wf_1", "Updated", version="1.0.1")
        mock_wf_cls.from_dict.return_value = updated_wf

        data = {"name": "Updated"}
        await update_workflow("wf_1", data)

        assert data["id"] == "wf_1"
        assert data["created_by"] == "alice"
        assert data["created_at"] == "2025-06-15T12:00:00+00:00"

    @pytest.mark.asyncio
    async def test_increments_patch_version(self, mock_store, mock_audit, mock_wf_cls):
        """Version is incremented from the existing workflow's version."""
        existing = _make_mock_workflow("wf_1", version="1.2.3")
        mock_store.get_workflow.return_value = existing

        updated_wf = _make_mock_workflow("wf_1", version="1.2.4")
        mock_wf_cls.from_dict.return_value = updated_wf

        data = {"name": "Updated"}
        await update_workflow("wf_1", data)

        assert data["version"] == "1.2.4"

    @pytest.mark.asyncio
    async def test_increments_single_version(self, mock_store, mock_audit, mock_wf_cls):
        """Single-segment version like '5' increments to '6'."""
        existing = _make_mock_workflow("wf_1", version="5")
        mock_store.get_workflow.return_value = existing

        updated_wf = _make_mock_workflow("wf_1")
        mock_wf_cls.from_dict.return_value = updated_wf

        data = {"name": "Updated"}
        await update_workflow("wf_1", data)
        assert data["version"] == "6"

    @pytest.mark.asyncio
    async def test_increments_two_part_version(self, mock_store, mock_audit, mock_wf_cls):
        """Two-part version like '2.9' increments to '2.10'."""
        existing = _make_mock_workflow("wf_1", version="2.9")
        mock_store.get_workflow.return_value = existing

        updated_wf = _make_mock_workflow("wf_1")
        mock_wf_cls.from_dict.return_value = updated_wf

        data = {"name": "Updated"}
        await update_workflow("wf_1", data)
        assert data["version"] == "2.10"

    @pytest.mark.asyncio
    async def test_sets_updated_at(self, mock_store, mock_audit, mock_wf_cls):
        """updated_at is set to current time."""
        existing = _make_mock_workflow("wf_1")
        mock_store.get_workflow.return_value = existing

        updated_wf = _make_mock_workflow("wf_1")
        mock_wf_cls.from_dict.return_value = updated_wf

        data = {"name": "Updated"}
        await update_workflow("wf_1", data)

        assert "updated_at" in data
        dt = datetime.fromisoformat(data["updated_at"])
        assert dt.tzinfo is not None

    @pytest.mark.asyncio
    async def test_validation_failure_raises_value_error(self, mock_store, mock_audit, mock_wf_cls):
        """Invalid update raises ValueError."""
        existing = _make_mock_workflow("wf_1")
        mock_store.get_workflow.return_value = existing

        bad_wf = MagicMock()
        bad_wf.validate.return_value = (False, ["steps required"])
        mock_wf_cls.from_dict.return_value = bad_wf

        data = {"name": "Bad Update"}
        with pytest.raises(ValueError, match="Invalid workflow.*steps required"):
            await update_workflow("wf_1", data)

        mock_store.save_workflow.assert_not_called()

    @pytest.mark.asyncio
    async def test_saves_workflow_and_version(self, mock_store, mock_audit, mock_wf_cls):
        """Both save_workflow and save_version are called after successful update."""
        existing = _make_mock_workflow("wf_1")
        mock_store.get_workflow.return_value = existing

        updated_wf = _make_mock_workflow("wf_1")
        mock_wf_cls.from_dict.return_value = updated_wf

        data = {"name": "Updated"}
        await update_workflow("wf_1", data)

        mock_store.save_workflow.assert_called_once_with(updated_wf)
        mock_store.save_version.assert_called_once_with(updated_wf)

    @pytest.mark.asyncio
    async def test_audit_logged_on_update(self, mock_store, mock_audit, mock_wf_cls):
        """Audit data is logged after update."""
        existing = _make_mock_workflow("wf_1", version="1.0.0")
        mock_store.get_workflow.return_value = existing

        updated_wf = _make_mock_workflow("wf_1", version="1.0.1")
        updated_wf.version = "1.0.1"
        mock_wf_cls.from_dict.return_value = updated_wf

        data = {"name": "Updated"}
        await update_workflow("wf_1", data, tenant_id="org-x")

        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["user_id"] == "system"
        assert call_kwargs["resource_type"] == "workflow"
        assert call_kwargs["resource_id"] == "wf_1"
        assert call_kwargs["action"] == "update"
        assert call_kwargs["new_version"] == "1.0.1"
        assert call_kwargs["tenant_id"] == "org-x"

    @pytest.mark.asyncio
    async def test_returns_updated_dict(self, mock_store, mock_audit, mock_wf_cls):
        """Returns the updated workflow as a dict."""
        existing = _make_mock_workflow("wf_1")
        mock_store.get_workflow.return_value = existing

        updated_wf = _make_mock_workflow("wf_1", "Updated")
        updated_wf.to_dict.return_value = {"id": "wf_1", "name": "Updated"}
        mock_wf_cls.from_dict.return_value = updated_wf

        data = {"name": "Updated"}
        result = await update_workflow("wf_1", data)
        assert result == {"id": "wf_1", "name": "Updated"}

    @pytest.mark.asyncio
    async def test_default_tenant_id(self, mock_store, mock_audit, mock_wf_cls):
        """Default tenant_id is 'default'."""
        existing = _make_mock_workflow("wf_1")
        mock_store.get_workflow.return_value = existing

        updated_wf = _make_mock_workflow("wf_1")
        mock_wf_cls.from_dict.return_value = updated_wf

        data = {"name": "Updated"}
        await update_workflow("wf_1", data)
        assert data["tenant_id"] == "default"

    @pytest.mark.asyncio
    async def test_preserves_none_created_at(self, mock_store, mock_audit, mock_wf_cls):
        """When existing workflow has no created_at, None is preserved."""
        existing = _make_mock_workflow("wf_1")
        existing.created_at = None
        mock_store.get_workflow.return_value = existing

        updated_wf = _make_mock_workflow("wf_1")
        mock_wf_cls.from_dict.return_value = updated_wf

        data = {"name": "Updated"}
        await update_workflow("wf_1", data)
        assert data["created_at"] is None


# ===========================================================================
# delete_workflow
# ===========================================================================


class TestDeleteWorkflow:
    """Test delete_workflow function."""

    @pytest.mark.asyncio
    async def test_deletes_successfully(self, mock_store, mock_audit):
        """Returns True when workflow is deleted."""
        mock_store.delete_workflow.return_value = True
        result = await delete_workflow("wf_1")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self, mock_store, mock_audit):
        """Returns False when workflow doesn't exist."""
        mock_store.delete_workflow.return_value = False
        result = await delete_workflow("wf_missing")
        assert result is False

    @pytest.mark.asyncio
    async def test_passes_workflow_id_and_tenant(self, mock_store, mock_audit):
        """Passes correct arguments to store."""
        mock_store.delete_workflow.return_value = True
        await delete_workflow("wf_42", tenant_id="org-acme")
        mock_store.delete_workflow.assert_called_once_with("wf_42", "org-acme")

    @pytest.mark.asyncio
    async def test_default_tenant_id(self, mock_store, mock_audit):
        """Default tenant_id is 'default'."""
        mock_store.delete_workflow.return_value = True
        await delete_workflow("wf_1")
        mock_store.delete_workflow.assert_called_once_with("wf_1", "default")

    @pytest.mark.asyncio
    async def test_audit_logged_on_delete(self, mock_store, mock_audit):
        """Audit data is logged when deletion succeeds."""
        mock_store.delete_workflow.return_value = True
        await delete_workflow("wf_1", tenant_id="t1")

        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["user_id"] == "system"
        assert call_kwargs["resource_type"] == "workflow"
        assert call_kwargs["resource_id"] == "wf_1"
        assert call_kwargs["action"] == "delete"
        assert call_kwargs["tenant_id"] == "t1"

    @pytest.mark.asyncio
    async def test_no_audit_when_not_found(self, mock_store, mock_audit):
        """Audit data is NOT logged when workflow doesn't exist."""
        mock_store.delete_workflow.return_value = False
        await delete_workflow("wf_missing")
        mock_audit.assert_not_called()


# ===========================================================================
# _get_workflow_definition_cls (override resolution)
# ===========================================================================


class TestGetWorkflowDefinitionCls:
    """Test the _get_workflow_definition_cls helper."""

    def test_returns_default_when_no_override(self):
        """Returns the default WorkflowDefinition when no override exists."""
        from aragora.workflow.types import WorkflowDefinition as RealCls

        result = _get_workflow_definition_cls()
        assert result is RealCls

    def test_returns_override_when_set(self, monkeypatch):
        """Returns override when the package module has a different class."""
        fake_cls = type("FakeWorkflowDefinition", (), {})
        pkg = sys.modules.get("aragora.server.handlers.workflows")
        if pkg is not None:
            monkeypatch.setattr(pkg, "WorkflowDefinition", fake_cls)
            result = _get_workflow_definition_cls()
            assert result is fake_cls


# ===========================================================================
# _get_audit_fn (override resolution)
# ===========================================================================


class TestGetAuditFn:
    """Test the _get_audit_fn helper."""

    def test_returns_default_when_no_override(self):
        """Returns the default audit_data function."""
        from aragora.audit.unified import audit_data as real_audit

        result = _get_audit_fn()
        assert result is real_audit

    def test_returns_override_when_set(self, monkeypatch):
        """Returns override when the package module has a different function."""
        fake_fn = MagicMock()
        pkg = sys.modules.get("aragora.server.handlers.workflows")
        if pkg is not None:
            monkeypatch.setattr(pkg, "audit_data", fake_fn)
            result = _get_audit_fn()
            assert result is fake_fn


# ===========================================================================
# Integration-style tests (multiple operations)
# ===========================================================================


class TestCrudIntegration:
    """Integration-like tests that combine multiple CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_then_get(self, mock_store, mock_audit, mock_wf_cls):
        """Create workflow then retrieve it."""
        wf_mock = _make_mock_workflow("wf_int", "Integration WF")
        mock_wf_cls.from_dict.return_value = wf_mock

        # Create
        data = {"name": "Integration WF"}
        created = await create_workflow(data)
        assert created["id"] == "wf_int"

        # Now configure store to return it for get
        mock_store.get_workflow.return_value = wf_mock
        fetched = await get_workflow("wf_int")
        assert fetched is not None
        assert fetched["id"] == "wf_int"

    @pytest.mark.asyncio
    async def test_create_update_delete_sequence(self, mock_store, mock_audit, mock_wf_cls):
        """Full lifecycle: create, update, delete."""
        # Create
        create_wf = _make_mock_workflow("wf_life", "Lifecycle", version="1.0.0")
        mock_wf_cls.from_dict.return_value = create_wf
        data = {"name": "Lifecycle"}
        await create_workflow(data)

        # Update
        mock_store.get_workflow.return_value = create_wf
        update_wf = _make_mock_workflow("wf_life", "Updated Lifecycle", version="1.0.1")
        mock_wf_cls.from_dict.return_value = update_wf
        result = await update_workflow("wf_life", {"name": "Updated Lifecycle"})
        assert result is not None

        # Delete
        mock_store.delete_workflow.return_value = True
        deleted = await delete_workflow("wf_life")
        assert deleted is True

        # Verify audit was called 3 times (create, update, delete)
        assert mock_audit.call_count == 3

    @pytest.mark.asyncio
    async def test_update_nonexistent_then_create(self, mock_store, mock_audit, mock_wf_cls):
        """Attempting to update a nonexistent workflow returns None, then create succeeds."""
        mock_store.get_workflow.return_value = None
        result = await update_workflow("wf_new", {"name": "New"})
        assert result is None

        wf_mock = _make_mock_workflow("wf_new", "New WF")
        mock_wf_cls.from_dict.return_value = wf_mock
        created = await create_workflow({"name": "New WF"})
        assert created["id"] == "wf_new"

    @pytest.mark.asyncio
    async def test_list_empty_create_list_again(self, mock_store, mock_audit, mock_wf_cls):
        """List empty store, create workflow, list should now contain it."""
        # Initial empty list
        mock_store.list_workflows.return_value = ([], 0)
        result1 = await list_workflows()
        assert result1["total_count"] == 0

        # Create
        wf = _make_mock_workflow("wf_new")
        mock_wf_cls.from_dict.return_value = wf
        await create_workflow({"name": "New"})

        # After creation, configure store to return the new workflow
        mock_store.list_workflows.return_value = ([wf], 1)
        result2 = await list_workflows()
        assert result2["total_count"] == 1


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_create_with_none_id_field(self, mock_store, mock_audit, mock_wf_cls):
        """Data with id=None triggers auto-generation."""
        wf_mock = _make_mock_workflow()
        mock_wf_cls.from_dict.return_value = wf_mock

        data = {"id": None, "name": "NoneID"}
        await create_workflow(data)
        assert data["id"] is not None
        assert data["id"].startswith("wf_")

    @pytest.mark.asyncio
    async def test_list_with_zero_limit(self, mock_store):
        """Limit of 0 is passed through to store."""
        mock_store.list_workflows.return_value = ([], 0)
        result = await list_workflows(limit=0)
        call_kwargs = mock_store.list_workflows.call_args[1]
        assert call_kwargs["limit"] == 0

    @pytest.mark.asyncio
    async def test_list_with_large_offset(self, mock_store):
        """Large offset results in empty list."""
        mock_store.list_workflows.return_value = ([], 10)
        result = await list_workflows(offset=10000)
        assert result["workflows"] == []
        assert result["offset"] == 10000

    @pytest.mark.asyncio
    async def test_create_workflow_id_format(self, mock_store, mock_audit, mock_wf_cls):
        """Auto-generated ID follows wf_<12 hex chars> format."""
        wf_mock = _make_mock_workflow()
        mock_wf_cls.from_dict.return_value = wf_mock

        data = {"name": "Format Test"}
        await create_workflow(data)
        wf_id = data["id"]
        assert wf_id.startswith("wf_")
        hex_part = wf_id[3:]
        assert len(hex_part) == 12
        # Should be valid hex
        int(hex_part, 16)

    @pytest.mark.asyncio
    async def test_update_preserves_tenant_isolation(self, mock_store, mock_audit, mock_wf_cls):
        """Update respects tenant_id for isolation."""
        existing = _make_mock_workflow("wf_1", tenant_id="org-a")
        mock_store.get_workflow.return_value = existing

        updated_wf = _make_mock_workflow("wf_1")
        mock_wf_cls.from_dict.return_value = updated_wf

        data = {"name": "Updated"}
        await update_workflow("wf_1", data, tenant_id="org-a")

        mock_store.get_workflow.assert_called_once_with("wf_1", "org-a")
        assert data["tenant_id"] == "org-a"

    @pytest.mark.asyncio
    async def test_delete_respects_tenant_isolation(self, mock_store, mock_audit):
        """Delete only removes from the specified tenant."""
        mock_store.delete_workflow.return_value = True
        await delete_workflow("wf_1", tenant_id="org-b")
        mock_store.delete_workflow.assert_called_once_with("wf_1", "org-b")

    @pytest.mark.asyncio
    async def test_list_with_empty_tags(self, mock_store):
        """Empty tags list is passed through."""
        mock_store.list_workflows.return_value = ([], 0)
        await list_workflows(tags=[])
        call_kwargs = mock_store.list_workflows.call_args[1]
        assert call_kwargs["tags"] == []

    @pytest.mark.asyncio
    async def test_list_with_empty_search(self, mock_store):
        """Empty search string is passed through."""
        mock_store.list_workflows.return_value = ([], 0)
        await list_workflows(search="")
        call_kwargs = mock_store.list_workflows.call_args[1]
        assert call_kwargs["search"] == ""

    @pytest.mark.asyncio
    async def test_create_multiple_workflows_unique_ids(self, mock_store, mock_audit, mock_wf_cls):
        """Multiple creates without explicit IDs generate unique IDs."""
        wf_mock = _make_mock_workflow()
        mock_wf_cls.from_dict.return_value = wf_mock

        ids = set()
        for i in range(5):
            data = {"name": f"WF {i}"}
            await create_workflow(data)
            ids.add(data["id"])

        assert len(ids) == 5  # All unique

    @pytest.mark.asyncio
    async def test_update_with_complex_version(self, mock_store, mock_audit, mock_wf_cls):
        """Version with 4+ parts increments correctly."""
        existing = _make_mock_workflow("wf_1", version="1.2.3.4")
        mock_store.get_workflow.return_value = existing

        updated_wf = _make_mock_workflow("wf_1")
        mock_wf_cls.from_dict.return_value = updated_wf

        data = {"name": "Updated"}
        await update_workflow("wf_1", data)
        assert data["version"] == "1.2.3.5"
