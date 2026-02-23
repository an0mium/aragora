"""Tests for workflow template management (aragora/server/handlers/workflows/templates.py).

Covers all public and internal functions in the templates module:
- list_templates() - async, lists templates with optional category/tags filters
- get_template() - async, retrieves a template by ID
- create_workflow_from_template() - async, creates a workflow from a template
- register_template() - sync, registers a workflow as a template
- _create_contract_review_template() - sync, builds the contract review template
- _create_code_review_template() - sync, builds the code review template
- _register_builtin_templates() - sync, registers both built-in templates
- _load_yaml_templates() - sync, loads YAML templates from disk
- load_yaml_templates_async() - async, loads YAML templates for PostgreSQL
- register_builtin_templates_async() - async, registers built-in templates for PG
- initialize_templates() - sync, orchestrates full template initialization
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from aragora.server.handlers.workflows.templates import (
    list_templates,
    get_template,
    create_workflow_from_template,
    register_template,
    _create_contract_review_template,
    _create_code_review_template,
    _register_builtin_templates,
    _load_yaml_templates,
    load_yaml_templates_async,
    register_builtin_templates_async,
    initialize_templates,
)


# ---------------------------------------------------------------------------
# Module-level patch targets
# ---------------------------------------------------------------------------

PATCH_MOD = "aragora.server.handlers.workflows.templates"
PATCH_STORAGE_FACTORY = "aragora.storage.factory"
PATCH_TEMPLATE_LOADER = "aragora.workflow.template_loader"
PATCH_PERSISTENT_STORE = "aragora.workflow.persistent_store"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_template(template_id="t1", name="Template 1"):
    """Create a mock template object with a to_dict() method."""
    t = MagicMock()
    t.id = template_id
    t.name = name
    t.is_template = True
    t.to_dict.return_value = {"id": template_id, "name": name, "is_template": True}
    return t


def _make_mock_store(templates=None, template_map=None):
    """Create a mock persistent store.

    Args:
        templates: List of mock templates to return from list_templates.
        template_map: Dict mapping template_id -> template for get_template.
    """
    store = MagicMock()
    store.list_templates.return_value = templates or []
    if template_map is not None:
        store.get_template.side_effect = lambda tid: template_map.get(tid)
    else:
        store.get_template.return_value = None
    return store


# ---------------------------------------------------------------------------
# list_templates
# ---------------------------------------------------------------------------


class TestListTemplates:
    """Test list_templates() async function."""

    @pytest.mark.asyncio
    async def test_returns_empty_list(self):
        store = _make_mock_store(templates=[])
        with patch(f"{PATCH_MOD}._get_store", return_value=store):
            result = await list_templates()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_template_dicts(self):
        t1 = _make_mock_template("t1", "Template 1")
        t2 = _make_mock_template("t2", "Template 2")
        store = _make_mock_store(templates=[t1, t2])
        with patch(f"{PATCH_MOD}._get_store", return_value=store):
            result = await list_templates()
        assert len(result) == 2
        assert result[0]["id"] == "t1"
        assert result[1]["id"] == "t2"

    @pytest.mark.asyncio
    async def test_passes_category_filter(self):
        store = _make_mock_store(templates=[])
        with patch(f"{PATCH_MOD}._get_store", return_value=store):
            await list_templates(category="legal")
        store.list_templates.assert_called_once_with(category="legal", tags=None)

    @pytest.mark.asyncio
    async def test_passes_tags_filter(self):
        store = _make_mock_store(templates=[])
        with patch(f"{PATCH_MOD}._get_store", return_value=store):
            await list_templates(tags=["review", "compliance"])
        store.list_templates.assert_called_once_with(category=None, tags=["review", "compliance"])

    @pytest.mark.asyncio
    async def test_passes_both_filters(self):
        store = _make_mock_store(templates=[])
        with patch(f"{PATCH_MOD}._get_store", return_value=store):
            await list_templates(category="code", tags=["security"])
        store.list_templates.assert_called_once_with(category="code", tags=["security"])

    @pytest.mark.asyncio
    async def test_calls_to_dict_on_each_template(self):
        t1 = _make_mock_template("t1", "T1")
        store = _make_mock_store(templates=[t1])
        with patch(f"{PATCH_MOD}._get_store", return_value=store):
            result = await list_templates()
        t1.to_dict.assert_called_once()
        assert result[0]["name"] == "T1"


# ---------------------------------------------------------------------------
# get_template
# ---------------------------------------------------------------------------


class TestGetTemplate:
    """Test get_template() async function."""

    @pytest.mark.asyncio
    async def test_returns_template_dict_when_found(self):
        t = _make_mock_template("t1", "Found Template")
        store = _make_mock_store(template_map={"t1": t})
        with patch(f"{PATCH_MOD}._get_store", return_value=store):
            result = await get_template("t1")
        assert result is not None
        assert result["id"] == "t1"
        assert result["name"] == "Found Template"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        store = _make_mock_store(template_map={})
        with patch(f"{PATCH_MOD}._get_store", return_value=store):
            result = await get_template("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_calls_store_get_template(self):
        store = _make_mock_store()
        with patch(f"{PATCH_MOD}._get_store", return_value=store):
            await get_template("t_abc")
        store.get_template.assert_called_once_with("t_abc")


# ---------------------------------------------------------------------------
# create_workflow_from_template
# ---------------------------------------------------------------------------


class TestCreateWorkflowFromTemplate:
    """Test create_workflow_from_template() async function."""

    @pytest.mark.asyncio
    async def test_raises_when_template_not_found(self):
        store = _make_mock_store(template_map={})
        with patch(f"{PATCH_MOD}._get_store", return_value=store):
            with pytest.raises(ValueError, match="Template not found"):
                await create_workflow_from_template("missing_id", "New WF")

    @pytest.mark.asyncio
    async def test_creates_workflow_from_template(self):
        template = _make_mock_template("t1", "Template")
        cloned = MagicMock()
        cloned.to_dict.return_value = {"id": "wf_new", "name": "My WF"}
        template.clone.return_value = cloned

        store = _make_mock_store(template_map={"t1": template})
        mock_create = AsyncMock(return_value={"id": "wf_new", "name": "My WF", "status": "created"})

        with patch(f"{PATCH_MOD}._get_store", return_value=store), \
             patch(f"{PATCH_MOD}.create_workflow", mock_create):
            result = await create_workflow_from_template("t1", "My WF")

        assert result["id"] == "wf_new"
        template.clone.assert_called_once_with(new_name="My WF")
        store.increment_template_usage.assert_called_once_with("t1")

    @pytest.mark.asyncio
    async def test_increments_usage_count(self):
        template = _make_mock_template("t1", "Template")
        cloned = MagicMock()
        cloned.to_dict.return_value = {"id": "wf_new", "name": "My WF"}
        template.clone.return_value = cloned

        store = _make_mock_store(template_map={"t1": template})
        mock_create = AsyncMock(return_value={"id": "wf_new"})

        with patch(f"{PATCH_MOD}._get_store", return_value=store), \
             patch(f"{PATCH_MOD}.create_workflow", mock_create):
            await create_workflow_from_template("t1", "My WF")

        store.increment_template_usage.assert_called_once_with("t1")

    @pytest.mark.asyncio
    async def test_passes_tenant_and_created_by(self):
        template = _make_mock_template("t1", "Template")
        cloned = MagicMock()
        cloned.to_dict.return_value = {"id": "wf_new", "name": "My WF"}
        template.clone.return_value = cloned

        store = _make_mock_store(template_map={"t1": template})
        mock_create = AsyncMock(return_value={"id": "wf_new"})

        with patch(f"{PATCH_MOD}._get_store", return_value=store), \
             patch(f"{PATCH_MOD}.create_workflow", mock_create):
            await create_workflow_from_template(
                "t1", "My WF", tenant_id="acme", created_by="user1"
            )

        # create_workflow is called with (workflow_dict, tenant_id, created_by)
        args = mock_create.call_args
        # Positional: to_dict result, then tenant_id and created_by
        assert args[0][1] == "acme"
        assert args[0][2] == "user1"

    @pytest.mark.asyncio
    async def test_applies_customizations(self):
        template = _make_mock_template("t1", "Template")
        cloned = MagicMock()
        cloned.to_dict.return_value = {"id": "wf_new", "name": "My WF", "description": "old"}
        template.clone.return_value = cloned

        store = _make_mock_store(template_map={"t1": template})
        mock_create = AsyncMock(return_value={"id": "wf_custom"})

        mock_wf_class = MagicMock()
        customized = MagicMock()
        customized.to_dict.return_value = {"id": "wf_new", "name": "My WF", "description": "new"}
        mock_wf_class.from_dict.return_value = customized

        with patch(f"{PATCH_MOD}._get_store", return_value=store), \
             patch(f"{PATCH_MOD}.create_workflow", mock_create), \
             patch(f"{PATCH_MOD}.WorkflowDefinition", mock_wf_class):
            await create_workflow_from_template(
                "t1", "My WF", customizations={"description": "new"}
            )

        # Verify WorkflowDefinition.from_dict was called with merged customizations
        from_dict_arg = mock_wf_class.from_dict.call_args[0][0]
        assert from_dict_arg["description"] == "new"

    @pytest.mark.asyncio
    async def test_no_customizations_skips_merge(self):
        template = _make_mock_template("t1", "Template")
        cloned = MagicMock()
        cloned.to_dict.return_value = {"id": "wf_new", "name": "My WF"}
        template.clone.return_value = cloned

        store = _make_mock_store(template_map={"t1": template})
        mock_create = AsyncMock(return_value={"id": "wf_new"})

        with patch(f"{PATCH_MOD}._get_store", return_value=store), \
             patch(f"{PATCH_MOD}.create_workflow", mock_create), \
             patch(f"{PATCH_MOD}.WorkflowDefinition") as mock_wf:
            await create_workflow_from_template("t1", "My WF")

        # When no customizations, WorkflowDefinition.from_dict should NOT be called
        mock_wf.from_dict.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_tenant_and_created_by(self):
        template = _make_mock_template("t1", "Template")
        cloned = MagicMock()
        cloned.to_dict.return_value = {"id": "wf_new", "name": "My WF"}
        template.clone.return_value = cloned

        store = _make_mock_store(template_map={"t1": template})
        mock_create = AsyncMock(return_value={"id": "wf_new"})

        with patch(f"{PATCH_MOD}._get_store", return_value=store), \
             patch(f"{PATCH_MOD}.create_workflow", mock_create):
            await create_workflow_from_template("t1", "My WF")

        args = mock_create.call_args
        assert args[0][1] == "default"
        assert args[0][2] == ""


# ---------------------------------------------------------------------------
# register_template
# ---------------------------------------------------------------------------


class TestRegisterTemplate:
    """Test register_template() sync function."""

    def test_sets_is_template_flag(self):
        workflow = MagicMock()
        workflow.is_template = False
        store = _make_mock_store()

        mock_sb = MagicMock()
        mock_sb.POSTGRES = "postgres"
        mock_sb.SUPABASE = "supabase"

        with patch(f"{PATCH_MOD}._get_store", return_value=store), \
             patch(f"{PATCH_STORAGE_FACTORY}.get_storage_backend", return_value="sqlite"), \
             patch(f"{PATCH_STORAGE_FACTORY}.StorageBackend", mock_sb):
            register_template(workflow)

        assert workflow.is_template is True

    def test_saves_template_for_sqlite_backend(self):
        workflow = MagicMock()
        store = _make_mock_store()

        mock_sb = MagicMock()
        mock_sb.POSTGRES = "postgres"
        mock_sb.SUPABASE = "supabase"

        with patch(f"{PATCH_MOD}._get_store", return_value=store), \
             patch(f"{PATCH_STORAGE_FACTORY}.get_storage_backend", return_value="sqlite"), \
             patch(f"{PATCH_STORAGE_FACTORY}.StorageBackend", mock_sb):
            register_template(workflow)

        store.save_template.assert_called_once_with(workflow)

    def test_defers_for_postgres_backend(self):
        workflow = MagicMock()
        workflow.id = "t_pg"
        store = _make_mock_store()

        # Need backend == StorageBackend.POSTGRES to be True
        mock_sb = MagicMock()
        mock_sb.POSTGRES = "postgres"
        mock_sb.SUPABASE = "supabase"

        with patch(f"{PATCH_MOD}._get_store", return_value=store), \
             patch(f"{PATCH_STORAGE_FACTORY}.get_storage_backend", return_value="postgres"), \
             patch(f"{PATCH_STORAGE_FACTORY}.StorageBackend", mock_sb):
            register_template(workflow)

        store.save_template.assert_not_called()

    def test_defers_for_supabase_backend(self):
        workflow = MagicMock()
        workflow.id = "t_supa"
        store = _make_mock_store()

        mock_sb = MagicMock()
        mock_sb.POSTGRES = "postgres"
        mock_sb.SUPABASE = "supabase"

        with patch(f"{PATCH_MOD}._get_store", return_value=store), \
             patch(f"{PATCH_STORAGE_FACTORY}.get_storage_backend", return_value="supabase"), \
             patch(f"{PATCH_STORAGE_FACTORY}.StorageBackend", mock_sb):
            register_template(workflow)

        store.save_template.assert_not_called()


# ---------------------------------------------------------------------------
# _create_contract_review_template
# ---------------------------------------------------------------------------


class TestCreateContractReviewTemplate:
    """Test _create_contract_review_template() factory function."""

    def test_returns_workflow_definition(self):
        template = _create_contract_review_template()
        assert template is not None

    def test_template_id(self):
        template = _create_contract_review_template()
        assert template.id == "template_contract_review"

    def test_template_name(self):
        template = _create_contract_review_template()
        assert template.name == "Contract Review"

    def test_template_is_marked_as_template(self):
        template = _create_contract_review_template()
        assert template.is_template is True

    def test_has_legal_category(self):
        template = _create_contract_review_template()
        from aragora.workflow.types import WorkflowCategory
        assert template.category == WorkflowCategory.LEGAL

    def test_has_expected_tags(self):
        template = _create_contract_review_template()
        assert "legal" in template.tags
        assert "contracts" in template.tags
        assert "review" in template.tags
        assert "compliance" in template.tags

    def test_has_icon(self):
        template = _create_contract_review_template()
        assert template.icon == "document-text"

    def test_has_six_steps(self):
        template = _create_contract_review_template()
        assert len(template.steps) == 6

    def test_step_ids(self):
        template = _create_contract_review_template()
        step_ids = [s.id for s in template.steps]
        assert "extract" in step_ids
        assert "analyze" in step_ids
        assert "risk_check" in step_ids
        assert "human_review" in step_ids
        assert "auto_approve" in step_ids
        assert "store_result" in step_ids

    def test_step_types(self):
        template = _create_contract_review_template()
        step_map = {s.id: s for s in template.steps}
        assert step_map["extract"].step_type == "agent"
        assert step_map["analyze"].step_type == "debate"
        assert step_map["risk_check"].step_type == "decision"
        assert step_map["human_review"].step_type == "human_checkpoint"
        assert step_map["auto_approve"].step_type == "task"
        assert step_map["store_result"].step_type == "memory_write"

    def test_has_transitions(self):
        template = _create_contract_review_template()
        assert len(template.transitions) == 2
        trans_ids = [t.id for t in template.transitions]
        assert "high_risk_route" in trans_ids
        assert "low_risk_route" in trans_ids

    def test_transition_from_risk_check(self):
        template = _create_contract_review_template()
        for t in template.transitions:
            assert t.from_step == "risk_check"

    def test_transition_targets(self):
        template = _create_contract_review_template()
        trans_map = {t.id: t for t in template.transitions}
        assert trans_map["high_risk_route"].to_step == "human_review"
        assert trans_map["low_risk_route"].to_step == "auto_approve"

    def test_extract_step_config(self):
        template = _create_contract_review_template()
        step_map = {s.id: s for s in template.steps}
        config = step_map["extract"].config
        assert config["agent_type"] == "claude"
        assert "prompt_template" in config

    def test_human_review_has_checklist(self):
        template = _create_contract_review_template()
        step_map = {s.id: s for s in template.steps}
        config = step_map["human_review"].config
        assert len(config["checklist"]) == 3
        assert config["timeout_seconds"] == 86400

    def test_steps_have_visual_data(self):
        template = _create_contract_review_template()
        for step in template.steps:
            assert step.visual is not None

    def test_description_set(self):
        template = _create_contract_review_template()
        assert "contract" in template.description.lower()

    def test_analyze_step_has_debate_config(self):
        template = _create_contract_review_template()
        step_map = {s.id: s for s in template.steps}
        config = step_map["analyze"].config
        assert "agents" in config
        assert config["rounds"] == 2

    def test_step_next_steps_chain(self):
        template = _create_contract_review_template()
        step_map = {s.id: s for s in template.steps}
        assert "analyze" in step_map["extract"].next_steps
        assert "risk_check" in step_map["analyze"].next_steps
        assert "store_result" in step_map["human_review"].next_steps
        assert "store_result" in step_map["auto_approve"].next_steps


# ---------------------------------------------------------------------------
# _create_code_review_template
# ---------------------------------------------------------------------------


class TestCreateCodeReviewTemplate:
    """Test _create_code_review_template() factory function."""

    def test_returns_workflow_definition(self):
        template = _create_code_review_template()
        assert template is not None

    def test_template_id(self):
        template = _create_code_review_template()
        assert template.id == "template_code_review"

    def test_template_name(self):
        template = _create_code_review_template()
        assert template.name == "Code Security Review"

    def test_template_is_marked_as_template(self):
        template = _create_code_review_template()
        assert template.is_template is True

    def test_has_code_category(self):
        template = _create_code_review_template()
        from aragora.workflow.types import WorkflowCategory
        assert template.category == WorkflowCategory.CODE

    def test_has_expected_tags(self):
        template = _create_code_review_template()
        assert "code" in template.tags
        assert "security" in template.tags
        assert "review" in template.tags
        assert "OWASP" in template.tags

    def test_has_icon(self):
        template = _create_code_review_template()
        assert template.icon == "code"

    def test_has_three_steps(self):
        template = _create_code_review_template()
        assert len(template.steps) == 3

    def test_step_ids(self):
        template = _create_code_review_template()
        step_ids = [s.id for s in template.steps]
        assert "scan" in step_ids
        assert "debate" in step_ids
        assert "summarize" in step_ids

    def test_step_types(self):
        template = _create_code_review_template()
        step_map = {s.id: s for s in template.steps}
        assert step_map["scan"].step_type == "agent"
        assert step_map["debate"].step_type == "debate"
        assert step_map["summarize"].step_type == "agent"

    def test_scan_uses_codex_agent(self):
        template = _create_code_review_template()
        step_map = {s.id: s for s in template.steps}
        assert step_map["scan"].config["agent_type"] == "codex"

    def test_debate_uses_adversarial_topology(self):
        template = _create_code_review_template()
        step_map = {s.id: s for s in template.steps}
        assert step_map["debate"].config["topology"] == "adversarial"

    def test_summarize_uses_claude_agent(self):
        template = _create_code_review_template()
        step_map = {s.id: s for s in template.steps}
        assert step_map["summarize"].config["agent_type"] == "claude"

    def test_steps_have_visual_data(self):
        template = _create_code_review_template()
        for step in template.steps:
            assert step.visual is not None

    def test_no_conditional_transitions(self):
        """Code review template has linear flow, no conditional transitions."""
        template = _create_code_review_template()
        assert not template.transitions

    def test_description_set(self):
        template = _create_code_review_template()
        assert "security" in template.description.lower()

    def test_step_next_steps_chain(self):
        template = _create_code_review_template()
        step_map = {s.id: s for s in template.steps}
        assert "debate" in step_map["scan"].next_steps
        assert "summarize" in step_map["debate"].next_steps

    def test_debate_has_security_agents(self):
        template = _create_code_review_template()
        step_map = {s.id: s for s in template.steps}
        agents = step_map["debate"].config["agents"]
        assert "security_analyst" in agents
        assert "penetration_tester" in agents


# ---------------------------------------------------------------------------
# _register_builtin_templates
# ---------------------------------------------------------------------------


class TestRegisterBuiltinTemplates:
    """Test _register_builtin_templates() sync function."""

    def test_registers_two_templates(self):
        with patch(f"{PATCH_MOD}.register_template") as mock_register:
            _register_builtin_templates()
        assert mock_register.call_count == 2

    def test_registers_contract_review(self):
        with patch(f"{PATCH_MOD}.register_template") as mock_register:
            _register_builtin_templates()
        template_ids = [c[0][0].id for c in mock_register.call_args_list]
        assert "template_contract_review" in template_ids

    def test_registers_code_review(self):
        with patch(f"{PATCH_MOD}.register_template") as mock_register:
            _register_builtin_templates()
        template_ids = [c[0][0].id for c in mock_register.call_args_list]
        assert "template_code_review" in template_ids


# ---------------------------------------------------------------------------
# _load_yaml_templates
# ---------------------------------------------------------------------------


class TestLoadYamlTemplates:
    """Test _load_yaml_templates() sync function."""

    def test_loads_new_templates(self):
        mock_templates = {
            "yaml_t1": _make_mock_template("yaml_t1", "YAML Template 1"),
            "yaml_t2": _make_mock_template("yaml_t2", "YAML Template 2"),
        }
        store = _make_mock_store(template_map={})

        mock_sb = MagicMock()
        mock_sb.POSTGRES = "postgres"
        mock_sb.SUPABASE = "supabase"

        with patch(f"{PATCH_TEMPLATE_LOADER}.load_templates", return_value=mock_templates), \
             patch(f"{PATCH_MOD}._get_store", return_value=store), \
             patch(f"{PATCH_STORAGE_FACTORY}.get_storage_backend", return_value="sqlite"), \
             patch(f"{PATCH_STORAGE_FACTORY}.StorageBackend", mock_sb):
            _load_yaml_templates()

        assert store.save_template.call_count == 2

    def test_skips_existing_templates(self):
        existing = _make_mock_template("yaml_t1", "Existing")
        mock_templates = {
            "yaml_t1": _make_mock_template("yaml_t1", "YAML Template 1"),
        }
        store = _make_mock_store(template_map={"yaml_t1": existing})

        mock_sb = MagicMock()
        mock_sb.POSTGRES = "postgres"
        mock_sb.SUPABASE = "supabase"

        with patch(f"{PATCH_TEMPLATE_LOADER}.load_templates", return_value=mock_templates), \
             patch(f"{PATCH_MOD}._get_store", return_value=store), \
             patch(f"{PATCH_STORAGE_FACTORY}.get_storage_backend", return_value="sqlite"), \
             patch(f"{PATCH_STORAGE_FACTORY}.StorageBackend", mock_sb):
            _load_yaml_templates()

        store.save_template.assert_not_called()

    def test_skips_for_postgres_backend(self):
        mock_sb = MagicMock()
        mock_sb.POSTGRES = "postgres"
        mock_sb.SUPABASE = "supabase"

        with patch(f"{PATCH_TEMPLATE_LOADER}.load_templates") as mock_load, \
             patch(f"{PATCH_STORAGE_FACTORY}.get_storage_backend", return_value="postgres"), \
             patch(f"{PATCH_STORAGE_FACTORY}.StorageBackend", mock_sb):
            _load_yaml_templates()

        mock_load.assert_not_called()

    def test_skips_for_supabase_backend(self):
        mock_sb = MagicMock()
        mock_sb.POSTGRES = "postgres"
        mock_sb.SUPABASE = "supabase"

        with patch(f"{PATCH_TEMPLATE_LOADER}.load_templates") as mock_load, \
             patch(f"{PATCH_STORAGE_FACTORY}.get_storage_backend", return_value="supabase"), \
             patch(f"{PATCH_STORAGE_FACTORY}.StorageBackend", mock_sb):
            _load_yaml_templates()

        mock_load.assert_not_called()

    def test_handles_import_error(self):
        """ImportError from template_loader is caught gracefully."""
        with patch.dict("sys.modules", {"aragora.workflow.template_loader": None}):
            # Should not raise -- the local import will fail with ImportError
            _load_yaml_templates()

    def test_handles_os_error(self):
        """OSError from reading YAML files is caught gracefully."""
        store = _make_mock_store()
        mock_sb = MagicMock()
        mock_sb.POSTGRES = "postgres"
        mock_sb.SUPABASE = "supabase"

        with patch(f"{PATCH_TEMPLATE_LOADER}.load_templates", side_effect=OSError("disk error")), \
             patch(f"{PATCH_MOD}._get_store", return_value=store), \
             patch(f"{PATCH_STORAGE_FACTORY}.get_storage_backend", return_value="sqlite"), \
             patch(f"{PATCH_STORAGE_FACTORY}.StorageBackend", mock_sb):
            # Should not raise
            _load_yaml_templates()

    def test_handles_value_error(self):
        """ValueError from parsing YAML is caught gracefully."""
        store = _make_mock_store()
        mock_sb = MagicMock()
        mock_sb.POSTGRES = "postgres"
        mock_sb.SUPABASE = "supabase"

        with patch(f"{PATCH_TEMPLATE_LOADER}.load_templates", side_effect=ValueError("bad yaml")), \
             patch(f"{PATCH_MOD}._get_store", return_value=store), \
             patch(f"{PATCH_STORAGE_FACTORY}.get_storage_backend", return_value="sqlite"), \
             patch(f"{PATCH_STORAGE_FACTORY}.StorageBackend", mock_sb):
            # Should not raise
            _load_yaml_templates()

    def test_handles_key_error(self):
        """KeyError from parsing YAML is caught gracefully."""
        store = _make_mock_store()
        mock_sb = MagicMock()
        mock_sb.POSTGRES = "postgres"
        mock_sb.SUPABASE = "supabase"

        with patch(f"{PATCH_TEMPLATE_LOADER}.load_templates", side_effect=KeyError("missing key")), \
             patch(f"{PATCH_MOD}._get_store", return_value=store), \
             patch(f"{PATCH_STORAGE_FACTORY}.get_storage_backend", return_value="sqlite"), \
             patch(f"{PATCH_STORAGE_FACTORY}.StorageBackend", mock_sb):
            # Should not raise
            _load_yaml_templates()

    def test_handles_type_error(self):
        """TypeError from parsing YAML is caught gracefully."""
        store = _make_mock_store()
        mock_sb = MagicMock()
        mock_sb.POSTGRES = "postgres"
        mock_sb.SUPABASE = "supabase"

        with patch(f"{PATCH_TEMPLATE_LOADER}.load_templates", side_effect=TypeError("bad type")), \
             patch(f"{PATCH_MOD}._get_store", return_value=store), \
             patch(f"{PATCH_STORAGE_FACTORY}.get_storage_backend", return_value="sqlite"), \
             patch(f"{PATCH_STORAGE_FACTORY}.StorageBackend", mock_sb):
            # Should not raise
            _load_yaml_templates()

    def test_loads_mix_of_new_and_existing(self):
        """Some templates already exist, only new ones are saved."""
        existing = _make_mock_template("yaml_t1", "Existing")
        new_template = _make_mock_template("yaml_t2", "New")
        mock_templates = {
            "yaml_t1": _make_mock_template("yaml_t1", "YAML T1"),
            "yaml_t2": new_template,
        }
        store = _make_mock_store(template_map={"yaml_t1": existing})

        mock_sb = MagicMock()
        mock_sb.POSTGRES = "postgres"
        mock_sb.SUPABASE = "supabase"

        with patch(f"{PATCH_TEMPLATE_LOADER}.load_templates", return_value=mock_templates), \
             patch(f"{PATCH_MOD}._get_store", return_value=store), \
             patch(f"{PATCH_STORAGE_FACTORY}.get_storage_backend", return_value="sqlite"), \
             patch(f"{PATCH_STORAGE_FACTORY}.StorageBackend", mock_sb):
            _load_yaml_templates()

        store.save_template.assert_called_once_with(new_template)


# ---------------------------------------------------------------------------
# load_yaml_templates_async
# ---------------------------------------------------------------------------


class TestLoadYamlTemplatesAsync:
    """Test load_yaml_templates_async() for PostgreSQL backends."""

    @pytest.mark.asyncio
    async def test_loads_new_templates(self):
        t1 = _make_mock_template("yaml_t1", "YAML T1")
        mock_templates = {"yaml_t1": t1}
        mock_store = AsyncMock()
        mock_store.get_template.return_value = None

        async def _get_store():
            return mock_store

        with patch(f"{PATCH_TEMPLATE_LOADER}.load_templates", return_value=mock_templates), \
             patch(f"{PATCH_PERSISTENT_STORE}.get_async_workflow_store", side_effect=_get_store):
            await load_yaml_templates_async()

        mock_store.save_template.assert_called_once_with(t1)

    @pytest.mark.asyncio
    async def test_skips_existing_templates(self):
        existing = _make_mock_template("yaml_t1", "Existing")
        mock_templates = {"yaml_t1": _make_mock_template("yaml_t1", "New")}
        mock_store = AsyncMock()
        mock_store.get_template.return_value = existing

        async def _get_store():
            return mock_store

        with patch(f"{PATCH_TEMPLATE_LOADER}.load_templates", return_value=mock_templates), \
             patch(f"{PATCH_PERSISTENT_STORE}.get_async_workflow_store", side_effect=_get_store):
            await load_yaml_templates_async()

        mock_store.save_template.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_import_error(self):
        with patch.dict("sys.modules", {"aragora.workflow.template_loader": None}):
            # Should not raise
            await load_yaml_templates_async()

    @pytest.mark.asyncio
    async def test_handles_os_error(self):
        mock_store = AsyncMock()

        async def _get_store():
            return mock_store

        with patch(f"{PATCH_TEMPLATE_LOADER}.load_templates", side_effect=OSError("disk")), \
             patch(f"{PATCH_PERSISTENT_STORE}.get_async_workflow_store", side_effect=_get_store):
            # Should not raise
            await load_yaml_templates_async()

    @pytest.mark.asyncio
    async def test_handles_value_error(self):
        mock_store = AsyncMock()

        async def _get_store():
            return mock_store

        with patch(f"{PATCH_TEMPLATE_LOADER}.load_templates", side_effect=ValueError("bad")), \
             patch(f"{PATCH_PERSISTENT_STORE}.get_async_workflow_store", side_effect=_get_store):
            # Should not raise
            await load_yaml_templates_async()

    @pytest.mark.asyncio
    async def test_handles_key_error(self):
        mock_store = AsyncMock()

        async def _get_store():
            return mock_store

        with patch(f"{PATCH_TEMPLATE_LOADER}.load_templates", side_effect=KeyError("missing")), \
             patch(f"{PATCH_PERSISTENT_STORE}.get_async_workflow_store", side_effect=_get_store):
            # Should not raise
            await load_yaml_templates_async()

    @pytest.mark.asyncio
    async def test_handles_type_error(self):
        mock_store = AsyncMock()

        async def _get_store():
            return mock_store

        with patch(f"{PATCH_TEMPLATE_LOADER}.load_templates", side_effect=TypeError("bad")), \
             patch(f"{PATCH_PERSISTENT_STORE}.get_async_workflow_store", side_effect=_get_store):
            # Should not raise
            await load_yaml_templates_async()

    @pytest.mark.asyncio
    async def test_loads_multiple_new(self):
        t1 = _make_mock_template("yaml_t1", "T1")
        t2 = _make_mock_template("yaml_t2", "T2")
        mock_templates = {"yaml_t1": t1, "yaml_t2": t2}
        mock_store = AsyncMock()
        mock_store.get_template.return_value = None

        async def _get_store():
            return mock_store

        with patch(f"{PATCH_TEMPLATE_LOADER}.load_templates", return_value=mock_templates), \
             patch(f"{PATCH_PERSISTENT_STORE}.get_async_workflow_store", side_effect=_get_store):
            await load_yaml_templates_async()

        assert mock_store.save_template.call_count == 2


# ---------------------------------------------------------------------------
# register_builtin_templates_async
# ---------------------------------------------------------------------------


class TestRegisterBuiltinTemplatesAsync:
    """Test register_builtin_templates_async() for PostgreSQL backends."""

    @pytest.mark.asyncio
    async def test_registers_two_templates(self):
        mock_store = AsyncMock()
        mock_store.get_template.return_value = None

        async def _get_store():
            return mock_store

        with patch(f"{PATCH_PERSISTENT_STORE}.get_async_workflow_store", side_effect=_get_store):
            await register_builtin_templates_async()

        assert mock_store.save_template.call_count == 2

    @pytest.mark.asyncio
    async def test_sets_is_template_true(self):
        mock_store = AsyncMock()
        mock_store.get_template.return_value = None

        async def _get_store():
            return mock_store

        with patch(f"{PATCH_PERSISTENT_STORE}.get_async_workflow_store", side_effect=_get_store):
            await register_builtin_templates_async()

        for c in mock_store.save_template.call_args_list:
            template = c[0][0]
            assert template.is_template is True

    @pytest.mark.asyncio
    async def test_skips_existing_templates(self):
        existing = _make_mock_template("template_contract_review", "Existing")
        mock_store = AsyncMock()
        # First call returns existing (contract review), second returns None (code review)
        mock_store.get_template.side_effect = [existing, None]

        async def _get_store():
            return mock_store

        with patch(f"{PATCH_PERSISTENT_STORE}.get_async_workflow_store", side_effect=_get_store):
            await register_builtin_templates_async()

        # Only code review should be saved
        assert mock_store.save_template.call_count == 1

    @pytest.mark.asyncio
    async def test_handles_os_error(self):
        mock_store = AsyncMock()
        mock_store.get_template.side_effect = OSError("db down")

        async def _get_store():
            return mock_store

        with patch(f"{PATCH_PERSISTENT_STORE}.get_async_workflow_store", side_effect=_get_store):
            # Should not raise
            await register_builtin_templates_async()

    @pytest.mark.asyncio
    async def test_handles_value_error(self):
        mock_store = AsyncMock()
        mock_store.get_template.side_effect = ValueError("bad")

        async def _get_store():
            return mock_store

        with patch(f"{PATCH_PERSISTENT_STORE}.get_async_workflow_store", side_effect=_get_store):
            # Should not raise
            await register_builtin_templates_async()

    @pytest.mark.asyncio
    async def test_handles_runtime_error(self):
        mock_store = AsyncMock()
        mock_store.get_template.side_effect = RuntimeError("pool closed")

        async def _get_store():
            return mock_store

        with patch(f"{PATCH_PERSISTENT_STORE}.get_async_workflow_store", side_effect=_get_store):
            # Should not raise
            await register_builtin_templates_async()

    @pytest.mark.asyncio
    async def test_registers_contract_and_code_review(self):
        mock_store = AsyncMock()
        mock_store.get_template.return_value = None

        async def _get_store():
            return mock_store

        with patch(f"{PATCH_PERSISTENT_STORE}.get_async_workflow_store", side_effect=_get_store):
            await register_builtin_templates_async()

        saved_ids = [c[0][0].id for c in mock_store.save_template.call_args_list]
        assert "template_contract_review" in saved_ids
        assert "template_code_review" in saved_ids


# ---------------------------------------------------------------------------
# initialize_templates
# ---------------------------------------------------------------------------


class TestInitializeTemplates:
    """Test initialize_templates() orchestration function."""

    def test_calls_register_and_load(self):
        with patch(f"{PATCH_MOD}._register_builtin_templates") as mock_register, \
             patch(f"{PATCH_MOD}._load_yaml_templates") as mock_load:
            initialize_templates()

        mock_register.assert_called_once()
        mock_load.assert_called_once()

    def test_register_called_before_load(self):
        call_order = []

        def mock_register():
            call_order.append("register")

        def mock_load():
            call_order.append("load")

        with patch(f"{PATCH_MOD}._register_builtin_templates", side_effect=mock_register), \
             patch(f"{PATCH_MOD}._load_yaml_templates", side_effect=mock_load):
            initialize_templates()

        assert call_order == ["register", "load"]


# ---------------------------------------------------------------------------
# __all__ exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Verify __all__ contains expected exports."""

    def test_list_templates_exported(self):
        from aragora.server.handlers.workflows import templates
        assert "list_templates" in templates.__all__

    def test_get_template_exported(self):
        from aragora.server.handlers.workflows import templates
        assert "get_template" in templates.__all__

    def test_create_workflow_from_template_exported(self):
        from aragora.server.handlers.workflows import templates
        assert "create_workflow_from_template" in templates.__all__

    def test_register_template_exported(self):
        from aragora.server.handlers.workflows import templates
        assert "register_template" in templates.__all__

    def test_load_yaml_templates_async_exported(self):
        from aragora.server.handlers.workflows import templates
        assert "load_yaml_templates_async" in templates.__all__

    def test_register_builtin_templates_async_exported(self):
        from aragora.server.handlers.workflows import templates
        assert "register_builtin_templates_async" in templates.__all__

    def test_initialize_templates_exported(self):
        from aragora.server.handlers.workflows import templates
        assert "initialize_templates" in templates.__all__

    def test_internal_creators_exported(self):
        from aragora.server.handlers.workflows import templates
        assert "_create_contract_review_template" in templates.__all__
        assert "_create_code_review_template" in templates.__all__
