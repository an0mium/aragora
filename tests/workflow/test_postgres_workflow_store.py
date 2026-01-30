"""
Comprehensive tests for aragora.workflow.postgres_workflow_store.

Uses mocked PostgreSQL operations (self.execute, self.fetch_one, self.fetch_all)
to test the PostgresWorkflowStore without requiring a real database.
Follows the same mocking pattern as tests/memory/test_postgres_continuum.py.
"""

import json
import sys
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workflow_def(
    wf_id="wf_test",
    name="Test Workflow",
    description="A test workflow",
    version="1.0.0",
    tenant_id="default",
    category="general",
    tags=None,
    is_template=False,
    created_by="tester",
    num_steps=1,
):
    """Create a WorkflowDefinition for testing."""
    from aragora.workflow.types import (
        WorkflowDefinition,
        StepDefinition,
        WorkflowCategory,
    )

    steps = []
    for i in range(num_steps):
        steps.append(
            StepDefinition(
                id=f"step_{i}",
                name=f"Step {i}",
                step_type="task",
                config={"task_type": "function"},
            )
        )

    wf = WorkflowDefinition(
        id=wf_id,
        name=name,
        description=description,
        version=version,
        steps=steps,
        tags=tags or [],
        is_template=is_template,
        created_by=created_by,
        tenant_id=tenant_id,
        created_at=datetime.now(timezone.utc),
    )
    try:
        wf.category = WorkflowCategory(category)
    except ValueError:
        wf.category = WorkflowCategory.GENERAL
    return wf


def _make_mock_row(data: dict[str, Any]) -> MagicMock:
    """Create a mock asyncpg.Record-like row that supports dict-style access."""
    row = MagicMock()
    row.__getitem__ = lambda self, key: data[key]
    row.__contains__ = lambda self, key: key in data
    row.keys = lambda: data.keys()
    return row


def _make_store():
    """Create a PostgresWorkflowStore with mocked pool and connection."""
    from aragora.workflow.postgres_workflow_store import PostgresWorkflowStore

    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.fetchrow = AsyncMock(return_value=None)

    store = PostgresWorkflowStore(mock_pool)

    # Create async context manager for connection
    @asynccontextmanager
    async def mock_connection_ctx():
        yield mock_conn

    # Patch the connection method to bypass real pool
    store.connection = mock_connection_ctx
    store._initialized = True

    return store, mock_conn


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------


class TestSchemaDefinition:
    """Tests for PostgresWorkflowStore schema constants."""

    def test_schema_name_defined(self):
        from aragora.workflow.postgres_workflow_store import PostgresWorkflowStore

        assert PostgresWorkflowStore.SCHEMA_NAME == "workflow_store"

    def test_schema_version_is_positive(self):
        from aragora.workflow.postgres_workflow_store import PostgresWorkflowStore

        assert PostgresWorkflowStore.SCHEMA_VERSION >= 1

    def test_initial_schema_contains_workflows_table(self):
        from aragora.workflow.postgres_workflow_store import PostgresWorkflowStore

        assert "CREATE TABLE IF NOT EXISTS workflows" in PostgresWorkflowStore.INITIAL_SCHEMA

    def test_initial_schema_contains_versions_table(self):
        from aragora.workflow.postgres_workflow_store import PostgresWorkflowStore

        assert "CREATE TABLE IF NOT EXISTS workflow_versions" in PostgresWorkflowStore.INITIAL_SCHEMA

    def test_initial_schema_contains_templates_table(self):
        from aragora.workflow.postgres_workflow_store import PostgresWorkflowStore

        assert "CREATE TABLE IF NOT EXISTS workflow_templates" in PostgresWorkflowStore.INITIAL_SCHEMA

    def test_initial_schema_contains_executions_table(self):
        from aragora.workflow.postgres_workflow_store import PostgresWorkflowStore

        assert "CREATE TABLE IF NOT EXISTS workflow_executions" in PostgresWorkflowStore.INITIAL_SCHEMA

    def test_initial_schema_uses_jsonb(self):
        from aragora.workflow.postgres_workflow_store import PostgresWorkflowStore

        assert "JSONB" in PostgresWorkflowStore.INITIAL_SCHEMA

    def test_initial_schema_uses_timestamptz(self):
        from aragora.workflow.postgres_workflow_store import PostgresWorkflowStore

        assert "TIMESTAMPTZ" in PostgresWorkflowStore.INITIAL_SCHEMA

    def test_initial_schema_has_indexes(self):
        from aragora.workflow.postgres_workflow_store import PostgresWorkflowStore

        schema = PostgresWorkflowStore.INITIAL_SCHEMA
        assert "idx_workflows_tenant" in schema
        assert "idx_workflows_category" in schema
        assert "idx_executions_workflow" in schema
        assert "idx_executions_status" in schema
        assert "idx_executions_tenant" in schema

    def test_initial_schema_has_foreign_keys(self):
        from aragora.workflow.postgres_workflow_store import PostgresWorkflowStore

        schema = PostgresWorkflowStore.INITIAL_SCHEMA
        assert "REFERENCES workflows(id)" in schema


# ---------------------------------------------------------------------------
# Workflow CRUD
# ---------------------------------------------------------------------------


class TestSaveWorkflow:
    """Tests for save_workflow."""

    @pytest.mark.asyncio
    async def test_save_workflow_calls_execute(self):
        store, mock_conn = _make_store()
        wf = _make_workflow_def()
        await store.save_workflow(wf)
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_workflow_uses_upsert(self):
        store, mock_conn = _make_store()
        wf = _make_workflow_def()
        await store.save_workflow(wf)
        sql = mock_conn.execute.call_args[0][0]
        assert "INSERT INTO workflows" in sql
        assert "ON CONFLICT (id) DO UPDATE" in sql

    @pytest.mark.asyncio
    async def test_save_workflow_passes_correct_id(self):
        store, mock_conn = _make_store()
        wf = _make_workflow_def(wf_id="wf_unique_42")
        await store.save_workflow(wf)
        args = mock_conn.execute.call_args[0]
        assert args[1] == "wf_unique_42"

    @pytest.mark.asyncio
    async def test_save_workflow_passes_tenant_id(self):
        store, mock_conn = _make_store()
        wf = _make_workflow_def(tenant_id="tenant_xyz")
        await store.save_workflow(wf)
        args = mock_conn.execute.call_args[0]
        assert args[2] == "tenant_xyz"

    @pytest.mark.asyncio
    async def test_save_workflow_serializes_definition_as_json(self):
        store, mock_conn = _make_store()
        wf = _make_workflow_def(name="JSON Test")
        await store.save_workflow(wf)
        args = mock_conn.execute.call_args[0]
        # definition_json is arg index 7 (after id, tenant_id, name, desc, category, version)
        definition_json = args[7]
        parsed = json.loads(definition_json)
        assert parsed["name"] == "JSON Test"

    @pytest.mark.asyncio
    async def test_save_workflow_serializes_tags_as_json(self):
        store, mock_conn = _make_store()
        wf = _make_workflow_def(tags=["alpha", "beta"])
        await store.save_workflow(wf)
        args = mock_conn.execute.call_args[0]
        tags_json = args[8]
        parsed = json.loads(tags_json)
        assert parsed == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_save_workflow_handles_category_enum(self):
        store, mock_conn = _make_store()
        wf = _make_workflow_def(category="legal")
        await store.save_workflow(wf)
        args = mock_conn.execute.call_args[0]
        # category is at index 5 (after id, tenant_id, name, desc)
        assert args[5] == "legal"

    @pytest.mark.asyncio
    async def test_save_workflow_handles_different_categories(self):
        """Verify various WorkflowCategory enum values are extracted correctly."""
        store, mock_conn = _make_store()
        for cat in ("general", "legal", "healthcare", "finance"):
            mock_conn.execute.reset_mock()
            wf = _make_workflow_def(category=cat)
            await store.save_workflow(wf)
            args = mock_conn.execute.call_args[0]
            assert args[5] == cat


class TestGetWorkflow:
    """Tests for get_workflow."""

    @pytest.mark.asyncio
    async def test_get_workflow_returns_none_when_not_found(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = None
        result = await store.get_workflow("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_workflow_returns_definition_when_found(self):
        store, mock_conn = _make_store()
        wf = _make_workflow_def(wf_id="wf_found", name="Found Workflow")
        definition_data = wf.to_dict()

        mock_conn.fetchrow.return_value = _make_mock_row(
            {"definition": json.dumps(definition_data)}
        )
        result = await store.get_workflow("wf_found")
        assert result is not None
        assert result.id == "wf_found"
        assert result.name == "Found Workflow"

    @pytest.mark.asyncio
    async def test_get_workflow_handles_dict_definition(self):
        """When definition is already a dict (not string), it should still work."""
        store, mock_conn = _make_store()
        wf = _make_workflow_def(wf_id="wf_dict")
        definition_data = wf.to_dict()

        # Return the dict directly (not JSON string)
        mock_conn.fetchrow.return_value = _make_mock_row(
            {"definition": definition_data}
        )
        result = await store.get_workflow("wf_dict")
        assert result is not None
        assert result.id == "wf_dict"

    @pytest.mark.asyncio
    async def test_get_workflow_filters_by_tenant(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = None
        await store.get_workflow("wf_1", tenant_id="tenant_a")
        call_args = mock_conn.fetchrow.call_args[0]
        assert call_args[1] == "wf_1"
        assert call_args[2] == "tenant_a"

    @pytest.mark.asyncio
    async def test_get_workflow_preserves_steps(self):
        store, mock_conn = _make_store()
        wf = _make_workflow_def(num_steps=3)
        mock_conn.fetchrow.return_value = _make_mock_row(
            {"definition": json.dumps(wf.to_dict())}
        )
        result = await store.get_workflow("wf_test")
        assert result is not None
        assert len(result.steps) == 3

    @pytest.mark.asyncio
    async def test_get_workflow_preserves_tags(self):
        store, mock_conn = _make_store()
        wf = _make_workflow_def(tags=["urgent", "review"])
        mock_conn.fetchrow.return_value = _make_mock_row(
            {"definition": json.dumps(wf.to_dict())}
        )
        result = await store.get_workflow("wf_test")
        assert result is not None
        assert "urgent" in result.tags
        assert "review" in result.tags


class TestListWorkflows:
    """Tests for list_workflows."""

    @pytest.mark.asyncio
    async def test_list_workflows_empty(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = _make_mock_row({"cnt": 0})
        mock_conn.fetch.return_value = []
        workflows, total = await store.list_workflows()
        assert total == 0
        assert workflows == []

    @pytest.mark.asyncio
    async def test_list_workflows_returns_count_and_items(self):
        store, mock_conn = _make_store()
        wf1 = _make_workflow_def(wf_id="wf_1", name="WF 1")
        wf2 = _make_workflow_def(wf_id="wf_2", name="WF 2")

        mock_conn.fetchrow.return_value = _make_mock_row({"cnt": 2})
        mock_conn.fetch.return_value = [
            _make_mock_row({"definition": json.dumps(wf1.to_dict())}),
            _make_mock_row({"definition": json.dumps(wf2.to_dict())}),
        ]

        workflows, total = await store.list_workflows()
        assert total == 2
        assert len(workflows) == 2
        assert workflows[0].id == "wf_1"
        assert workflows[1].id == "wf_2"

    @pytest.mark.asyncio
    async def test_list_workflows_with_category_filter(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = _make_mock_row({"cnt": 1})
        wf = _make_workflow_def(wf_id="wf_legal", category="legal")
        mock_conn.fetch.return_value = [
            _make_mock_row({"definition": json.dumps(wf.to_dict())}),
        ]

        workflows, total = await store.list_workflows(category="legal")
        assert total == 1
        # Verify category parameter was passed in query
        fetchrow_sql = mock_conn.fetchrow.call_args[0][0]
        assert "category" in fetchrow_sql

    @pytest.mark.asyncio
    async def test_list_workflows_with_search(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = _make_mock_row({"cnt": 1})
        wf = _make_workflow_def(wf_id="wf_match", name="Alpha Process")
        mock_conn.fetch.return_value = [
            _make_mock_row({"definition": json.dumps(wf.to_dict())}),
        ]

        workflows, total = await store.list_workflows(search="Alpha")
        assert total == 1
        fetchrow_sql = mock_conn.fetchrow.call_args[0][0]
        assert "ILIKE" in fetchrow_sql

    @pytest.mark.asyncio
    async def test_list_workflows_pagination(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = _make_mock_row({"cnt": 10})
        wf = _make_workflow_def(wf_id="wf_page")
        mock_conn.fetch.return_value = [
            _make_mock_row({"definition": json.dumps(wf.to_dict())}),
        ]

        workflows, total = await store.list_workflows(limit=3, offset=6)
        assert total == 10
        # Verify limit and offset are passed
        fetch_args = mock_conn.fetch.call_args[0]
        assert 3 in fetch_args
        assert 6 in fetch_args

    @pytest.mark.asyncio
    async def test_list_workflows_tag_filter_applied_post_fetch(self):
        """Tags are filtered in Python after SQL fetch."""
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = _make_mock_row({"cnt": 2})
        wf1 = _make_workflow_def(wf_id="wf_1", tags=["urgent"])
        wf2 = _make_workflow_def(wf_id="wf_2", tags=["archive"])
        mock_conn.fetch.return_value = [
            _make_mock_row({"definition": json.dumps(wf1.to_dict())}),
            _make_mock_row({"definition": json.dumps(wf2.to_dict())}),
        ]

        workflows, total = await store.list_workflows(tags=["urgent"])
        # total is from SQL (no tag filter), but results are filtered in Python
        assert total == 2
        assert len(workflows) == 1
        assert workflows[0].id == "wf_1"

    @pytest.mark.asyncio
    async def test_list_workflows_tenant_isolation(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = _make_mock_row({"cnt": 1})
        mock_conn.fetch.return_value = []

        await store.list_workflows(tenant_id="my_tenant")
        # Verify tenant_id is the first param
        fetchrow_args = mock_conn.fetchrow.call_args[0]
        assert fetchrow_args[1] == "my_tenant"


class TestDeleteWorkflow:
    """Tests for delete_workflow."""

    @pytest.mark.asyncio
    async def test_delete_workflow_returns_true_when_deleted(self):
        store, mock_conn = _make_store()
        mock_conn.execute.return_value = "DELETE 1"
        result = await store.delete_workflow("wf_del")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_workflow_returns_false_when_not_found(self):
        store, mock_conn = _make_store()
        mock_conn.execute.return_value = "DELETE 0"
        result = await store.delete_workflow("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_workflow_respects_tenant(self):
        store, mock_conn = _make_store()
        mock_conn.execute.return_value = "DELETE 0"
        await store.delete_workflow("wf_1", tenant_id="wrong_tenant")
        call_args = mock_conn.execute.call_args[0]
        assert call_args[1] == "wf_1"
        assert call_args[2] == "wrong_tenant"


# ---------------------------------------------------------------------------
# Version Management
# ---------------------------------------------------------------------------


class TestVersionManagement:
    """Tests for save_version, get_versions, get_version."""

    @pytest.mark.asyncio
    async def test_save_version_calls_execute(self):
        store, mock_conn = _make_store()
        wf = _make_workflow_def(version="1.0.0")
        await store.save_version(wf)
        mock_conn.execute.assert_called_once()
        sql = mock_conn.execute.call_args[0][0]
        assert "INSERT INTO workflow_versions" in sql

    @pytest.mark.asyncio
    async def test_save_version_passes_workflow_id_and_version(self):
        store, mock_conn = _make_store()
        wf = _make_workflow_def(wf_id="wf_v", version="2.5.0")
        await store.save_version(wf)
        args = mock_conn.execute.call_args[0]
        assert args[1] == "wf_v"
        assert args[2] == "2.5.0"

    @pytest.mark.asyncio
    async def test_save_version_serializes_definition(self):
        store, mock_conn = _make_store()
        wf = _make_workflow_def(name="Version Test")
        await store.save_version(wf)
        args = mock_conn.execute.call_args[0]
        definition_json = args[3]
        parsed = json.loads(definition_json)
        assert parsed["name"] == "Version Test"

    @pytest.mark.asyncio
    async def test_get_versions_returns_list(self):
        store, mock_conn = _make_store()
        now = datetime.now(timezone.utc)

        mock_conn.fetch.return_value = [
            _make_mock_row({
                "version": "1.0.0",
                "created_by": "alice",
                "created_at": now,
                "definition": json.dumps({"steps": [{"id": "s1"}]}),
            }),
            _make_mock_row({
                "version": "1.1.0",
                "created_by": "bob",
                "created_at": now,
                "definition": json.dumps({"steps": [{"id": "s1"}, {"id": "s2"}]}),
            }),
        ]

        versions = await store.get_versions("wf_test")
        assert len(versions) == 2
        assert versions[0]["version"] == "1.0.0"
        assert versions[0]["created_by"] == "alice"
        assert versions[0]["step_count"] == 1
        assert versions[1]["version"] == "1.1.0"
        assert versions[1]["step_count"] == 2

    @pytest.mark.asyncio
    async def test_get_versions_empty(self):
        store, mock_conn = _make_store()
        mock_conn.fetch.return_value = []
        versions = await store.get_versions("wf_empty")
        assert versions == []

    @pytest.mark.asyncio
    async def test_get_versions_respects_limit(self):
        store, mock_conn = _make_store()
        mock_conn.fetch.return_value = []
        await store.get_versions("wf_test", limit=5)
        call_args = mock_conn.fetch.call_args[0]
        assert call_args[2] == 5

    @pytest.mark.asyncio
    async def test_get_versions_handles_string_created_at(self):
        """When created_at is a string (not datetime), it should be cast via str()."""
        store, mock_conn = _make_store()
        mock_conn.fetch.return_value = [
            _make_mock_row({
                "version": "1.0.0",
                "created_by": "tester",
                "created_at": "2024-01-15T10:00:00+00:00",
                "definition": json.dumps({"steps": []}),
            }),
        ]
        versions = await store.get_versions("wf_test")
        assert len(versions) == 1
        assert versions[0]["created_at"] == "2024-01-15T10:00:00+00:00"

    @pytest.mark.asyncio
    async def test_get_versions_handles_dict_definition(self):
        """When definition is already a dict, not a JSON string."""
        store, mock_conn = _make_store()
        now = datetime.now(timezone.utc)
        mock_conn.fetch.return_value = [
            _make_mock_row({
                "version": "1.0.0",
                "created_by": "tester",
                "created_at": now,
                "definition": {"steps": [{"id": "s1"}, {"id": "s2"}, {"id": "s3"}]},
            }),
        ]
        versions = await store.get_versions("wf_test")
        assert versions[0]["step_count"] == 3

    @pytest.mark.asyncio
    async def test_get_version_returns_definition(self):
        store, mock_conn = _make_store()
        wf = _make_workflow_def(wf_id="wf_v", version="1.0.0", num_steps=2)
        mock_conn.fetchrow.return_value = _make_mock_row(
            {"definition": json.dumps(wf.to_dict())}
        )
        result = await store.get_version("wf_v", "1.0.0")
        assert result is not None
        assert result.version == "1.0.0"
        assert len(result.steps) == 2

    @pytest.mark.asyncio
    async def test_get_version_returns_none_when_not_found(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = None
        result = await store.get_version("wf_v", "99.0.0")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_version_handles_dict_definition(self):
        store, mock_conn = _make_store()
        wf = _make_workflow_def(wf_id="wf_dict_v", version="2.0.0")
        mock_conn.fetchrow.return_value = _make_mock_row(
            {"definition": wf.to_dict()}
        )
        result = await store.get_version("wf_dict_v", "2.0.0")
        assert result is not None
        assert result.id == "wf_dict_v"


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


class TestTemplateOperations:
    """Tests for template CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_template_calls_execute(self):
        store, mock_conn = _make_store()
        tmpl = _make_workflow_def(wf_id="tmpl_1", is_template=True)
        await store.save_template(tmpl)
        mock_conn.execute.assert_called_once()
        sql = mock_conn.execute.call_args[0][0]
        assert "INSERT INTO workflow_templates" in sql
        assert "ON CONFLICT (id) DO UPDATE" in sql

    @pytest.mark.asyncio
    async def test_save_template_serializes_tags(self):
        store, mock_conn = _make_store()
        tmpl = _make_workflow_def(wf_id="tmpl_t", tags=["fast", "automated"])
        await store.save_template(tmpl)
        args = mock_conn.execute.call_args[0]
        tags_json = args[6]
        parsed = json.loads(tags_json)
        assert parsed == ["fast", "automated"]

    @pytest.mark.asyncio
    async def test_get_template_returns_definition(self):
        store, mock_conn = _make_store()
        tmpl = _make_workflow_def(wf_id="tmpl_found", name="My Template")
        mock_conn.fetchrow.return_value = _make_mock_row(
            {"definition": json.dumps(tmpl.to_dict())}
        )
        result = await store.get_template("tmpl_found")
        assert result is not None
        assert result.name == "My Template"

    @pytest.mark.asyncio
    async def test_get_template_returns_none_when_not_found(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = None
        result = await store.get_template("ghost")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_template_handles_dict_definition(self):
        store, mock_conn = _make_store()
        tmpl = _make_workflow_def(wf_id="tmpl_dict")
        mock_conn.fetchrow.return_value = _make_mock_row(
            {"definition": tmpl.to_dict()}
        )
        result = await store.get_template("tmpl_dict")
        assert result is not None
        assert result.id == "tmpl_dict"

    @pytest.mark.asyncio
    async def test_list_templates_all(self):
        store, mock_conn = _make_store()
        t1 = _make_workflow_def(wf_id="t1", name="Template 1")
        t2 = _make_workflow_def(wf_id="t2", name="Template 2")
        mock_conn.fetch.return_value = [
            _make_mock_row({"definition": json.dumps(t1.to_dict())}),
            _make_mock_row({"definition": json.dumps(t2.to_dict())}),
        ]
        templates = await store.list_templates()
        assert len(templates) == 2

    @pytest.mark.asyncio
    async def test_list_templates_by_category(self):
        store, mock_conn = _make_store()
        t1 = _make_workflow_def(wf_id="t1", category="legal")
        mock_conn.fetch.return_value = [
            _make_mock_row({"definition": json.dumps(t1.to_dict())}),
        ]
        templates = await store.list_templates(category="legal")
        assert len(templates) == 1
        # Verify category was passed in query
        sql = mock_conn.fetch.call_args[0][0]
        assert "category" in sql

    @pytest.mark.asyncio
    async def test_list_templates_without_category_uses_different_query(self):
        store, mock_conn = _make_store()
        mock_conn.fetch.return_value = []
        await store.list_templates()
        sql = mock_conn.fetch.call_args[0][0]
        assert "WHERE category" not in sql

    @pytest.mark.asyncio
    async def test_list_templates_tag_filter_post_fetch(self):
        store, mock_conn = _make_store()
        t1 = _make_workflow_def(wf_id="t1", tags=["fast"])
        t2 = _make_workflow_def(wf_id="t2", tags=["slow"])
        mock_conn.fetch.return_value = [
            _make_mock_row({"definition": json.dumps(t1.to_dict())}),
            _make_mock_row({"definition": json.dumps(t2.to_dict())}),
        ]
        templates = await store.list_templates(tags=["fast"])
        assert len(templates) == 1
        assert templates[0].id == "t1"

    @pytest.mark.asyncio
    async def test_list_templates_limit(self):
        store, mock_conn = _make_store()
        mock_conn.fetch.return_value = []
        await store.list_templates(limit=5)
        call_args = mock_conn.fetch.call_args[0]
        assert 5 in call_args

    @pytest.mark.asyncio
    async def test_increment_template_usage(self):
        store, mock_conn = _make_store()
        await store.increment_template_usage("tmpl_1")
        mock_conn.execute.assert_called_once()
        sql = mock_conn.execute.call_args[0][0]
        assert "usage_count = usage_count + 1" in sql
        assert mock_conn.execute.call_args[0][1] == "tmpl_1"


# ---------------------------------------------------------------------------
# Executions
# ---------------------------------------------------------------------------


class TestExecutionOperations:
    """Tests for execution CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_execution_calls_execute(self):
        store, mock_conn = _make_store()
        exec_data = {
            "id": "exec_1",
            "workflow_id": "wf_test",
            "tenant_id": "default",
            "status": "running",
            "inputs": {"question": "What?"},
            "outputs": {},
            "steps": [{"id": "s1", "status": "completed"}],
            "error": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "duration_ms": None,
        }
        await store.save_execution(exec_data)
        mock_conn.execute.assert_called_once()
        sql = mock_conn.execute.call_args[0][0]
        assert "INSERT INTO workflow_executions" in sql
        assert "ON CONFLICT (id) DO UPDATE" in sql

    @pytest.mark.asyncio
    async def test_save_execution_serializes_json_fields(self):
        store, mock_conn = _make_store()
        exec_data = {
            "id": "exec_json",
            "inputs": {"key": "value"},
            "outputs": {"result": 42},
            "steps": [{"id": "s1"}],
        }
        await store.save_execution(exec_data)
        args = mock_conn.execute.call_args[0]
        # inputs is at index 5, outputs at 6, steps at 7
        assert json.loads(args[5]) == {"key": "value"}
        assert json.loads(args[6]) == {"result": 42}
        assert json.loads(args[7]) == [{"id": "s1"}]

    @pytest.mark.asyncio
    async def test_save_execution_defaults(self):
        store, mock_conn = _make_store()
        exec_data = {"id": "exec_min"}
        await store.save_execution(exec_data)
        args = mock_conn.execute.call_args[0]
        # workflow_id defaults to None
        assert args[2] is None
        # tenant_id defaults to "default"
        assert args[3] == "default"
        # status defaults to "pending"
        assert args[4] == "pending"

    @pytest.mark.asyncio
    async def test_get_execution_returns_none_when_not_found(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = None
        result = await store.get_execution("ghost")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_execution_returns_dict(self):
        store, mock_conn = _make_store()
        now = datetime.now(timezone.utc)
        mock_conn.fetchrow.return_value = _make_mock_row({
            "id": "exec_1",
            "workflow_id": "wf_test",
            "tenant_id": "default",
            "status": "completed",
            "inputs": json.dumps({"question": "What?"}),
            "outputs": json.dumps({"result": 42}),
            "steps": json.dumps([{"id": "s1"}]),
            "error": None,
            "started_at": now,
            "completed_at": now,
            "duration_ms": 1234.5,
        })
        result = await store.get_execution("exec_1")
        assert result is not None
        assert result["id"] == "exec_1"
        assert result["status"] == "completed"
        assert result["inputs"] == {"question": "What?"}
        assert result["outputs"] == {"result": 42}
        assert result["steps"] == [{"id": "s1"}]
        assert result["duration_ms"] == 1234.5

    @pytest.mark.asyncio
    async def test_get_execution_handles_dict_json_fields(self):
        """When inputs/outputs/steps are already dicts/lists (not strings)."""
        store, mock_conn = _make_store()
        now = datetime.now(timezone.utc)
        mock_conn.fetchrow.return_value = _make_mock_row({
            "id": "exec_dict",
            "workflow_id": "wf_test",
            "tenant_id": "default",
            "status": "completed",
            "inputs": {"key": "val"},
            "outputs": {"out": 1},
            "steps": [{"id": "s1"}],
            "error": None,
            "started_at": now,
            "completed_at": now,
            "duration_ms": 100.0,
        })
        result = await store.get_execution("exec_dict")
        assert result["inputs"] == {"key": "val"}
        assert result["outputs"] == {"out": 1}
        assert result["steps"] == [{"id": "s1"}]

    @pytest.mark.asyncio
    async def test_get_execution_handles_none_json_fields(self):
        """When inputs/outputs/steps are None."""
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = _make_mock_row({
            "id": "exec_none",
            "workflow_id": "wf_test",
            "tenant_id": "default",
            "status": "pending",
            "inputs": None,
            "outputs": None,
            "steps": None,
            "error": None,
            "started_at": None,
            "completed_at": None,
            "duration_ms": None,
        })
        result = await store.get_execution("exec_none")
        assert result["inputs"] == {}
        assert result["outputs"] == {}
        assert result["steps"] == []

    @pytest.mark.asyncio
    async def test_get_execution_handles_string_timestamps(self):
        """When started_at/completed_at are strings (not datetime)."""
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = _make_mock_row({
            "id": "exec_str_ts",
            "workflow_id": "wf",
            "tenant_id": "default",
            "status": "completed",
            "inputs": "{}",
            "outputs": "{}",
            "steps": "[]",
            "error": None,
            "started_at": "2024-01-15T10:00:00",
            "completed_at": "2024-01-15T10:05:00",
            "duration_ms": 300000.0,
        })
        result = await store.get_execution("exec_str_ts")
        assert result["started_at"] == "2024-01-15T10:00:00"
        assert result["completed_at"] == "2024-01-15T10:05:00"

    @pytest.mark.asyncio
    async def test_get_execution_error_field(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = _make_mock_row({
            "id": "exec_err",
            "workflow_id": "wf",
            "tenant_id": "default",
            "status": "failed",
            "inputs": "{}",
            "outputs": "{}",
            "steps": "[]",
            "error": "Something went wrong",
            "started_at": None,
            "completed_at": None,
            "duration_ms": None,
        })
        result = await store.get_execution("exec_err")
        assert result["error"] == "Something went wrong"


class TestListExecutions:
    """Tests for list_executions."""

    @pytest.mark.asyncio
    async def test_list_executions_empty(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = _make_mock_row({"cnt": 0})
        mock_conn.fetch.return_value = []
        executions, total = await store.list_executions()
        assert total == 0
        assert executions == []

    @pytest.mark.asyncio
    async def test_list_executions_returns_items_and_count(self):
        store, mock_conn = _make_store()
        now = datetime.now(timezone.utc)
        mock_conn.fetchrow.return_value = _make_mock_row({"cnt": 2})
        mock_conn.fetch.return_value = [
            _make_mock_row({
                "id": "e1", "workflow_id": "wf", "tenant_id": "default",
                "status": "completed", "inputs": "{}", "outputs": "{}",
                "steps": "[]", "error": None, "started_at": now,
                "completed_at": now, "duration_ms": 100.0,
            }),
            _make_mock_row({
                "id": "e2", "workflow_id": "wf", "tenant_id": "default",
                "status": "failed", "inputs": "{}", "outputs": "{}",
                "steps": "[]", "error": "err", "started_at": now,
                "completed_at": now, "duration_ms": 200.0,
            }),
        ]
        executions, total = await store.list_executions()
        assert total == 2
        assert len(executions) == 2
        assert executions[0]["id"] == "e1"
        assert executions[1]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_list_executions_by_workflow_id(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = _make_mock_row({"cnt": 1})
        mock_conn.fetch.return_value = []
        await store.list_executions(workflow_id="wf_specific")
        fetchrow_sql = mock_conn.fetchrow.call_args[0][0]
        assert "workflow_id" in fetchrow_sql

    @pytest.mark.asyncio
    async def test_list_executions_by_status(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = _make_mock_row({"cnt": 1})
        mock_conn.fetch.return_value = []
        await store.list_executions(status="failed")
        fetchrow_sql = mock_conn.fetchrow.call_args[0][0]
        assert "status" in fetchrow_sql

    @pytest.mark.asyncio
    async def test_list_executions_tenant_isolation(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = _make_mock_row({"cnt": 0})
        mock_conn.fetch.return_value = []
        await store.list_executions(tenant_id="special_tenant")
        fetchrow_args = mock_conn.fetchrow.call_args[0]
        assert fetchrow_args[1] == "special_tenant"

    @pytest.mark.asyncio
    async def test_list_executions_pagination(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = _make_mock_row({"cnt": 20})
        mock_conn.fetch.return_value = []
        await store.list_executions(limit=5, offset=10)
        fetch_args = mock_conn.fetch.call_args[0]
        assert 5 in fetch_args
        assert 10 in fetch_args

    @pytest.mark.asyncio
    async def test_list_executions_combined_filters(self):
        """Test workflow_id + status + tenant_id together."""
        store, mock_conn = _make_store()
        mock_conn.fetchrow.return_value = _make_mock_row({"cnt": 0})
        mock_conn.fetch.return_value = []
        await store.list_executions(
            workflow_id="wf_combo",
            tenant_id="t_combo",
            status="running",
        )
        fetchrow_sql = mock_conn.fetchrow.call_args[0][0]
        assert "tenant_id" in fetchrow_sql
        assert "workflow_id" in fetchrow_sql
        assert "status" in fetchrow_sql


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for database error scenarios."""

    @pytest.mark.asyncio
    async def test_save_workflow_propagates_db_error(self):
        store, mock_conn = _make_store()
        mock_conn.execute.side_effect = Exception("connection lost")
        wf = _make_workflow_def()
        with pytest.raises(Exception, match="connection lost"):
            await store.save_workflow(wf)

    @pytest.mark.asyncio
    async def test_get_workflow_propagates_db_error(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.side_effect = Exception("timeout")
        with pytest.raises(Exception, match="timeout"):
            await store.get_workflow("wf_1")

    @pytest.mark.asyncio
    async def test_list_workflows_propagates_db_error(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.side_effect = Exception("pool exhausted")
        with pytest.raises(Exception, match="pool exhausted"):
            await store.list_workflows()

    @pytest.mark.asyncio
    async def test_save_execution_propagates_db_error(self):
        store, mock_conn = _make_store()
        mock_conn.execute.side_effect = Exception("disk full")
        with pytest.raises(Exception, match="disk full"):
            await store.save_execution({"id": "exec_err"})

    @pytest.mark.asyncio
    async def test_get_execution_propagates_db_error(self):
        store, mock_conn = _make_store()
        mock_conn.fetchrow.side_effect = Exception("network error")
        with pytest.raises(Exception, match="network error"):
            await store.get_execution("exec_1")


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_module_exports_store_class(self):
        from aragora.workflow.postgres_workflow_store import __all__

        assert "PostgresWorkflowStore" in __all__

    def test_store_class_importable(self):
        from aragora.workflow.postgres_workflow_store import PostgresWorkflowStore

        assert PostgresWorkflowStore is not None
