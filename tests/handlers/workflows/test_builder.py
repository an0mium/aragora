"""Tests for WorkflowBuilderHandler (aragora/server/handlers/workflows/builder.py).

Covers all routes and behavior of the WorkflowBuilderHandler class:
- can_handle() routing for all ROUTES
- GET  /api/v1/workflows/step-types       - Step type catalog
- POST /api/v1/workflows/generate          - NL-to-workflow generation
- POST /api/v1/workflows/auto-layout       - Auto-layout positions
- POST /api/v1/workflows/from-pattern      - Pattern-based creation
- POST /api/v1/workflows/validate          - Workflow validation
- POST /api/v1/workflows/{id}/replay       - Execution replay
- Error handling for ImportError, ValueError, TypeError, KeyError, etc.
- read_json_body_validated failure paths
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.workflows.builder import WorkflowBuilderHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PATCH_MOD = "aragora.server.handlers.workflows.builder"


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to WorkflowBuilderHandler."""

    def __init__(
        self,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        content_type: str = "application/json",
    ):
        self.command = method
        self.rfile = MagicMock()
        self.client_address = ("127.0.0.1", 12345)

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers = {
                "Content-Length": str(len(raw)),
                "Content-Type": content_type,
            }
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {
                "Content-Length": "2",
                "Content-Type": content_type,
            }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a WorkflowBuilderHandler instance."""
    return WorkflowBuilderHandler(ctx={})


@pytest.fixture
def mock_http():
    """Factory for creating mock HTTP handlers."""

    def _create(method="GET", body=None, content_type="application/json"):
        return _MockHTTPHandler(method=method, body=body, content_type=content_type)

    return _create


# ---------------------------------------------------------------------------
# can_handle routing
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Test WorkflowBuilderHandler.can_handle() routing."""

    def test_generate(self, handler):
        assert handler.can_handle("/api/v1/workflows/generate") is True

    def test_auto_layout(self, handler):
        assert handler.can_handle("/api/v1/workflows/auto-layout") is True

    def test_step_types(self, handler):
        assert handler.can_handle("/api/v1/workflows/step-types") is True

    def test_from_pattern(self, handler):
        assert handler.can_handle("/api/v1/workflows/from-pattern") is True

    def test_validate(self, handler):
        assert handler.can_handle("/api/v1/workflows/validate") is True

    def test_replay_with_id(self, handler):
        assert handler.can_handle("/api/v1/workflows/wf_123/replay") is True

    def test_replay_with_uuid(self, handler):
        assert handler.can_handle("/api/v1/workflows/abc-def-ghi/replay") is True

    def test_unrelated_path_rejected(self, handler):
        assert handler.can_handle("/api/v1/workflows") is False

    def test_unrelated_workflow_subpath(self, handler):
        assert handler.can_handle("/api/v1/workflows/wf_123/execute") is False

    def test_different_api(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_replay_only_suffix(self, handler):
        """Path that just ends with /replay but isn't under /workflows/."""
        assert handler.can_handle("/api/v1/other/replay") is False

    def test_root_path(self, handler):
        assert handler.can_handle("/") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False


# ---------------------------------------------------------------------------
# GET /api/v1/workflows/step-types
# ---------------------------------------------------------------------------


class TestGetStepTypes:
    """Test GET step-types endpoint."""

    def test_returns_full_catalog(self, handler, mock_http):
        """Should return all step types when no category filter."""
        mock_info_a = MagicMock()
        mock_info_a.to_dict.return_value = {"name": "llm_call", "category": "ai"}
        mock_info_b = MagicMock()
        mock_info_b.to_dict.return_value = {"name": "http_request", "category": "io"}

        catalog = {"llm_call": mock_info_a, "http_request": mock_info_b}
        categories = ["ai", "io", "logic"]

        with patch(f"{PATCH_MOD}.WorkflowBuilderHandler._get_step_catalog") as original:
            # Don't mock - call the real method, but mock the imports
            original.side_effect = None  # undo
        # Actually test by patching the inner imports
        with (
            patch(
                f"{PATCH_MOD}.WorkflowBuilderHandler._get_step_catalog",
                wraps=handler._get_step_catalog,
            ),
        ):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.workflow.step_catalog": MagicMock(
                        get_step_catalog=MagicMock(return_value=catalog),
                        list_step_categories=MagicMock(return_value=categories),
                    ),
                },
            ):
                result = handler._get_step_catalog({})

        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 2
        assert len(body["step_types"]) == 2
        assert body["categories"] == ["ai", "io", "logic"]

    def test_filters_by_category(self, handler, mock_http):
        """Should filter step types by category query param."""
        mock_info_a = MagicMock()
        mock_info_a.to_dict.return_value = {"name": "llm_call", "category": "ai"}
        mock_info_b = MagicMock()
        mock_info_b.to_dict.return_value = {"name": "http_request", "category": "io"}

        catalog = {"llm_call": mock_info_a, "http_request": mock_info_b}
        categories = ["ai", "io"]

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.step_catalog": MagicMock(
                    get_step_catalog=MagicMock(return_value=catalog),
                    list_step_categories=MagicMock(return_value=categories),
                ),
            },
        ):
            result = handler._get_step_catalog({"category": "ai"})

        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 1
        assert body["step_types"][0]["name"] == "llm_call"

    def test_category_filter_no_match(self, handler):
        """Category filter that matches nothing returns empty list."""
        mock_info = MagicMock()
        mock_info.to_dict.return_value = {"name": "llm_call", "category": "ai"}

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.step_catalog": MagicMock(
                    get_step_catalog=MagicMock(return_value={"llm_call": mock_info}),
                    list_step_categories=MagicMock(return_value=["ai"]),
                ),
            },
        ):
            result = handler._get_step_catalog({"category": "nonexistent"})

        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 0
        assert body["step_types"] == []

    def test_import_error_returns_503(self, handler):
        """Should return 503 when step_catalog module is not available."""
        with patch.dict("sys.modules", {"aragora.workflow.step_catalog": None}):
            result = handler._get_step_catalog({})

        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "").lower()

    def test_handle_get_routes_to_step_types(self, handler, mock_http):
        """The handle() GET dispatch should route step-types correctly."""
        mock_info = MagicMock()
        mock_info.to_dict.return_value = {"name": "test", "category": "test"}
        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.step_catalog": MagicMock(
                    get_step_catalog=MagicMock(return_value={"test": mock_info}),
                    list_step_categories=MagicMock(return_value=["test"]),
                ),
            },
        ):
            result = handler.handle("/api/v1/workflows/step-types", {}, mock_http())

        assert _status(result) == 200

    def test_handle_get_unknown_path_returns_none(self, handler, mock_http):
        """handle() GET with non step-types path returns None."""
        result = handler.handle("/api/v1/workflows/generate", {}, mock_http())
        assert result is None

    def test_empty_catalog(self, handler):
        """Empty catalog returns zero items."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.step_catalog": MagicMock(
                    get_step_catalog=MagicMock(return_value={}),
                    list_step_categories=MagicMock(return_value=[]),
                ),
            },
        ):
            result = handler._get_step_catalog({})

        body = _body(result)
        assert body["count"] == 0
        assert body["step_types"] == []
        assert body["categories"] == []


# ---------------------------------------------------------------------------
# POST /api/v1/workflows/generate
# ---------------------------------------------------------------------------


class TestGenerateWorkflow:
    """Test POST generate endpoint."""

    def test_quick_mode_success(self, handler):
        """Quick mode should call build_quick and return result."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"workflow_id": "wf_1", "steps": []}

        mock_builder_class = MagicMock()
        mock_builder_instance = MagicMock()
        mock_builder_instance.build_quick.return_value = mock_result
        mock_builder_class.return_value = mock_builder_instance

        mock_config_class = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.nl_builder": MagicMock(
                    NLWorkflowBuilder=mock_builder_class,
                    NLBuildConfig=mock_config_class,
                ),
            },
        ):
            result = handler._generate_workflow({"description": "Send email on new lead"})

        assert _status(result) == 200
        body = _body(result)
        assert body["workflow_id"] == "wf_1"
        mock_builder_instance.build_quick.assert_called_once_with(
            "Send email on new lead",
            category=None,
        )

    def test_quick_mode_with_category(self, handler):
        """Quick mode should pass category to build_quick."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"workflow_id": "wf_1"}

        mock_builder = MagicMock()
        mock_builder.return_value.build_quick.return_value = mock_result

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.nl_builder": MagicMock(
                    NLWorkflowBuilder=mock_builder,
                    NLBuildConfig=MagicMock(),
                ),
            },
        ):
            result = handler._generate_workflow(
                {
                    "description": "classify documents",
                    "category": "data_processing",
                }
            )

        assert _status(result) == 200
        mock_builder.return_value.build_quick.assert_called_once_with(
            "classify documents",
            category="data_processing",
        )

    def test_async_mode_calls_build(self, handler):
        """Non-quick mode should call build() via _run_async."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"workflow_id": "wf_2", "mode": "full"}

        mock_builder = MagicMock()
        mock_coro = MagicMock()
        mock_builder.return_value.build.return_value = mock_coro

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.nl_builder": MagicMock(
                    NLWorkflowBuilder=mock_builder,
                    NLBuildConfig=MagicMock(),
                ),
            },
        ):
            with patch(f"{PATCH_MOD}._run_async", return_value=mock_result) as mock_run:
                result = handler._generate_workflow(
                    {
                        "description": "complex workflow",
                        "mode": "full",
                        "category": "ops",
                        "agents": ["agent1", "agent2"],
                    }
                )

        assert _status(result) == 200
        mock_run.assert_called_once_with(mock_coro)
        mock_builder.return_value.build.assert_called_once_with(
            "complex workflow",
            category="ops",
            agents=["agent1", "agent2"],
        )

    def test_empty_description_returns_400(self, handler):
        """Missing description should return 400."""
        result = handler._generate_workflow({"description": ""})
        assert _status(result) == 400
        assert "description" in _body(result).get("error", "").lower()

    def test_missing_description_returns_400(self, handler):
        """Body without description key should return 400."""
        result = handler._generate_workflow({})
        assert _status(result) == 400

    def test_import_error_returns_503(self, handler):
        """Should return 503 when nl_builder module is not available."""
        with patch.dict("sys.modules", {"aragora.workflow.nl_builder": None}):
            result = handler._generate_workflow({"description": "test workflow"})

        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "").lower()

    def test_type_error_returns_400(self, handler):
        """TypeError during generation should return 400."""
        mock_builder = MagicMock()
        mock_builder.return_value.build_quick.side_effect = TypeError("bad arg")

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.nl_builder": MagicMock(
                    NLWorkflowBuilder=mock_builder,
                    NLBuildConfig=MagicMock(),
                ),
            },
        ):
            result = handler._generate_workflow({"description": "a workflow"})

        assert _status(result) == 400
        assert "failed" in _body(result).get("error", "").lower()

    def test_value_error_returns_400(self, handler):
        """ValueError during generation should return 400."""
        mock_builder = MagicMock()
        mock_builder.return_value.build_quick.side_effect = ValueError("invalid mode")

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.nl_builder": MagicMock(
                    NLWorkflowBuilder=mock_builder,
                    NLBuildConfig=MagicMock(),
                ),
            },
        ):
            result = handler._generate_workflow({"description": "a workflow"})

        assert _status(result) == 400

    def test_default_mode_is_quick(self, handler):
        """When no mode specified, should default to 'quick'."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {}

        mock_config = MagicMock()
        mock_builder = MagicMock()
        mock_builder.return_value.build_quick.return_value = mock_result

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.nl_builder": MagicMock(
                    NLWorkflowBuilder=mock_builder,
                    NLBuildConfig=mock_config,
                ),
            },
        ):
            handler._generate_workflow({"description": "test"})

        mock_config.assert_called_once_with(mode="quick")


# ---------------------------------------------------------------------------
# POST /api/v1/workflows/auto-layout
# ---------------------------------------------------------------------------


class TestAutoLayout:
    """Test POST auto-layout endpoint."""

    def test_flow_layout_success(self, handler):
        """Default flow layout should return positions."""
        mock_pos = MagicMock()
        mock_pos.to_dict.return_value = {"step_id": "s1", "x": 0, "y": 0}

        steps = [{"id": "s1", "name": "Start"}]
        transitions = [{"from": "s1", "to": "s2"}]

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.layout": MagicMock(
                    flow_layout=MagicMock(return_value=[mock_pos]),
                    grid_layout=MagicMock(),
                ),
            },
        ):
            result = handler._auto_layout(
                {
                    "steps": steps,
                    "transitions": transitions,
                }
            )

        body = _body(result)
        assert _status(result) == 200
        assert body["layout"] == "flow"
        assert body["count"] == 1
        assert len(body["positions"]) == 1

    def test_grid_layout(self, handler):
        """Grid layout with custom columns."""
        mock_pos_a = MagicMock()
        mock_pos_a.to_dict.return_value = {"step_id": "s1", "x": 0, "y": 0}
        mock_pos_b = MagicMock()
        mock_pos_b.to_dict.return_value = {"step_id": "s2", "x": 1, "y": 0}

        steps = [{"id": "s1"}, {"id": "s2"}]

        mock_layout = MagicMock()
        mock_grid_fn = MagicMock(return_value=[mock_pos_a, mock_pos_b])

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.layout": MagicMock(
                    flow_layout=MagicMock(),
                    grid_layout=mock_grid_fn,
                ),
            },
        ):
            result = handler._auto_layout(
                {
                    "steps": steps,
                    "layout": "grid",
                    "columns": 4,
                }
            )

        body = _body(result)
        assert _status(result) == 200
        assert body["layout"] == "grid"
        assert body["count"] == 2
        mock_grid_fn.assert_called_once_with(steps, columns=4)

    def test_grid_layout_default_columns(self, handler):
        """Grid layout defaults to 3 columns."""
        mock_grid_fn = MagicMock(return_value=[])

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.layout": MagicMock(
                    flow_layout=MagicMock(),
                    grid_layout=mock_grid_fn,
                ),
            },
        ):
            handler._auto_layout(
                {
                    "steps": [{"id": "s1"}],
                    "layout": "grid",
                }
            )

        mock_grid_fn.assert_called_once_with([{"id": "s1"}], columns=3)

    def test_empty_steps_returns_400(self, handler):
        """Empty steps list should return 400."""
        result = handler._auto_layout({"steps": []})
        assert _status(result) == 400
        assert "steps" in _body(result).get("error", "").lower()

    def test_missing_steps_returns_400(self, handler):
        """Missing steps key should return 400."""
        result = handler._auto_layout({})
        assert _status(result) == 400

    def test_import_error_returns_503(self, handler):
        """Should return 503 when layout module is not available."""
        with patch.dict("sys.modules", {"aragora.workflow.layout": None}):
            result = handler._auto_layout({"steps": [{"id": "s1"}]})

        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "").lower()

    def test_key_error_returns_400(self, handler):
        """KeyError during layout should return 400."""
        mock_flow_fn = MagicMock(side_effect=KeyError("missing_key"))

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.layout": MagicMock(
                    flow_layout=mock_flow_fn,
                    grid_layout=MagicMock(),
                ),
            },
        ):
            result = handler._auto_layout(
                {
                    "steps": [{"id": "s1"}],
                    "transitions": [],
                }
            )

        assert _status(result) == 400
        assert "failed" in _body(result).get("error", "").lower()

    def test_type_error_returns_400(self, handler):
        """TypeError during layout should return 400."""
        mock_flow_fn = MagicMock(side_effect=TypeError("bad type"))

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.layout": MagicMock(
                    flow_layout=mock_flow_fn,
                    grid_layout=MagicMock(),
                ),
            },
        ):
            result = handler._auto_layout(
                {
                    "steps": [{"id": "s1"}],
                    "transitions": [],
                }
            )

        assert _status(result) == 400

    def test_default_transitions_empty(self, handler):
        """Missing transitions defaults to empty list."""
        mock_flow_fn = MagicMock(return_value=[])

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.layout": MagicMock(
                    flow_layout=mock_flow_fn,
                    grid_layout=MagicMock(),
                ),
            },
        ):
            handler._auto_layout({"steps": [{"id": "s1"}]})

        mock_flow_fn.assert_called_once_with([{"id": "s1"}], [])

    def test_default_layout_is_flow(self, handler):
        """Default layout type should be 'flow'."""
        mock_flow_fn = MagicMock(return_value=[])

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.layout": MagicMock(
                    flow_layout=mock_flow_fn,
                    grid_layout=MagicMock(),
                ),
            },
        ):
            result = handler._auto_layout({"steps": [{"id": "s1"}]})

        body = _body(result)
        assert body["layout"] == "flow"
        mock_flow_fn.assert_called_once()


# ---------------------------------------------------------------------------
# POST /api/v1/workflows/from-pattern
# ---------------------------------------------------------------------------


class TestCreateFromPattern:
    """Test POST from-pattern endpoint."""

    def test_success(self, handler):
        """Should create workflow from pattern name."""
        mock_workflow = MagicMock()
        mock_workflow.to_dict.return_value = {"id": "wf_pattern_1", "pattern": "sequential"}

        mock_pattern = MagicMock()
        mock_pattern.create.return_value = mock_workflow

        mock_create_fn = MagicMock(return_value=mock_pattern)
        mock_pattern_type = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.patterns.base": MagicMock(PatternType=mock_pattern_type),
                "aragora.workflow.patterns": MagicMock(create_pattern=mock_create_fn),
            },
        ):
            result = handler._create_from_pattern({"pattern": "sequential"})

        assert _status(result) == 201
        body = _body(result)
        assert body["id"] == "wf_pattern_1"

    def test_with_optional_kwargs(self, handler):
        """Should pass name, agents, and task to create_pattern."""
        mock_workflow = MagicMock()
        mock_workflow.to_dict.return_value = {}

        mock_pattern = MagicMock()
        mock_pattern.create.return_value = mock_workflow

        mock_create_fn = MagicMock(return_value=mock_pattern)
        mock_pattern_type = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.patterns.base": MagicMock(PatternType=mock_pattern_type),
                "aragora.workflow.patterns": MagicMock(create_pattern=mock_create_fn),
            },
        ):
            result = handler._create_from_pattern(
                {
                    "pattern": "fan_out",
                    "name": "My Workflow",
                    "agents": ["agent1"],
                    "task": "Analyze data",
                }
            )

        assert _status(result) == 201
        # Verify kwargs passed to create_pattern
        call_kwargs = mock_create_fn.call_args[1]
        assert call_kwargs["name"] == "My Workflow"
        assert call_kwargs["agents"] == ["agent1"]
        assert call_kwargs["task"] == "Analyze data"

    def test_empty_pattern_returns_400(self, handler):
        """Empty pattern name should return 400."""
        result = handler._create_from_pattern({"pattern": ""})
        assert _status(result) == 400
        assert "pattern" in _body(result).get("error", "").lower()

    def test_missing_pattern_returns_400(self, handler):
        """Missing pattern key should return 400."""
        result = handler._create_from_pattern({})
        assert _status(result) == 400

    def test_unknown_pattern_returns_400(self, handler):
        """Unknown pattern type should return 400 with pattern name."""
        mock_pattern_type = MagicMock(side_effect=ValueError("not a valid pattern"))

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.patterns.base": MagicMock(PatternType=mock_pattern_type),
                "aragora.workflow.patterns": MagicMock(),
            },
        ):
            result = handler._create_from_pattern({"pattern": "nonexistent"})

        assert _status(result) == 400
        assert "nonexistent" in _body(result).get("error", "").lower()

    def test_import_error_returns_503(self, handler):
        """Should return 503 when patterns module is not available."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.patterns.base": None,
            },
        ):
            result = handler._create_from_pattern({"pattern": "sequential"})

        assert _status(result) == 503

    def test_type_error_returns_400(self, handler):
        """TypeError during pattern creation should return 400."""
        mock_pattern_type = MagicMock()
        mock_create_fn = MagicMock(side_effect=TypeError("bad type"))

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.patterns.base": MagicMock(PatternType=mock_pattern_type),
                "aragora.workflow.patterns": MagicMock(create_pattern=mock_create_fn),
            },
        ):
            result = handler._create_from_pattern({"pattern": "sequential"})

        assert _status(result) == 400
        assert "failed" in _body(result).get("error", "").lower()

    def test_attribute_error_returns_400(self, handler):
        """AttributeError during pattern creation should return 400."""
        mock_pattern_type = MagicMock()
        mock_create_fn = MagicMock(side_effect=AttributeError("no attribute"))

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.patterns.base": MagicMock(PatternType=mock_pattern_type),
                "aragora.workflow.patterns": MagicMock(create_pattern=mock_create_fn),
            },
        ):
            result = handler._create_from_pattern({"pattern": "sequential"})

        assert _status(result) == 400

    def test_no_optional_kwargs(self, handler):
        """Should work fine without optional name/agents/task."""
        mock_workflow = MagicMock()
        mock_workflow.to_dict.return_value = {"id": "wf_1"}

        mock_pattern = MagicMock()
        mock_pattern.create.return_value = mock_workflow

        mock_create_fn = MagicMock(return_value=mock_pattern)
        mock_pattern_type = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.patterns.base": MagicMock(PatternType=mock_pattern_type),
                "aragora.workflow.patterns": MagicMock(create_pattern=mock_create_fn),
            },
        ):
            result = handler._create_from_pattern({"pattern": "sequential"})

        assert _status(result) == 201
        # No extra kwargs should be passed
        call_kwargs = mock_create_fn.call_args[1]
        assert "name" not in call_kwargs
        assert "agents" not in call_kwargs
        assert "task" not in call_kwargs


# ---------------------------------------------------------------------------
# POST /api/v1/workflows/validate
# ---------------------------------------------------------------------------


class TestValidateWorkflow:
    """Test POST validate endpoint."""

    def test_success(self, handler):
        """Valid workflow definition should return validation result."""
        mock_definition = MagicMock()
        mock_from_dict = MagicMock(return_value=mock_definition)

        mock_validation_result = MagicMock()
        mock_validation_result.to_dict.return_value = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }
        mock_validate_fn = MagicMock(return_value=mock_validation_result)

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.types": MagicMock(
                    WorkflowDefinition=MagicMock(from_dict=mock_from_dict),
                ),
                "aragora.workflow.validation": MagicMock(
                    validate_workflow=mock_validate_fn,
                ),
            },
        ):
            result = handler._validate_workflow({"name": "test", "steps": []})

        assert _status(result) == 200
        body = _body(result)
        assert body["valid"] is True
        assert body["errors"] == []

    def test_import_error_returns_503(self, handler):
        """Should return 503 when validation module is not available."""
        with patch.dict("sys.modules", {"aragora.workflow.types": None}):
            result = handler._validate_workflow({"name": "test"})

        assert _status(result) == 503

    def test_key_error_returns_400(self, handler):
        """KeyError during validation should return 400."""
        mock_from_dict = MagicMock(side_effect=KeyError("missing field"))

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.types": MagicMock(
                    WorkflowDefinition=MagicMock(from_dict=mock_from_dict),
                ),
                "aragora.workflow.validation": MagicMock(),
            },
        ):
            result = handler._validate_workflow({})

        assert _status(result) == 400
        assert "invalid" in _body(result).get("error", "").lower()

    def test_type_error_returns_400(self, handler):
        """TypeError during validation should return 400."""
        mock_from_dict = MagicMock(side_effect=TypeError("wrong type"))

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.types": MagicMock(
                    WorkflowDefinition=MagicMock(from_dict=mock_from_dict),
                ),
                "aragora.workflow.validation": MagicMock(),
            },
        ):
            result = handler._validate_workflow({"name": "test"})

        assert _status(result) == 400

    def test_value_error_returns_400(self, handler):
        """ValueError during validation should return 400."""
        mock_from_dict = MagicMock(side_effect=ValueError("bad value"))

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.types": MagicMock(
                    WorkflowDefinition=MagicMock(from_dict=mock_from_dict),
                ),
                "aragora.workflow.validation": MagicMock(),
            },
        ):
            result = handler._validate_workflow({"name": "test"})

        assert _status(result) == 400

    def test_validation_with_errors(self, handler):
        """Workflow with validation errors should still return 200 with error details."""
        mock_definition = MagicMock()
        mock_from_dict = MagicMock(return_value=mock_definition)

        mock_validation_result = MagicMock()
        mock_validation_result.to_dict.return_value = {
            "valid": False,
            "errors": ["Step 'end' has no incoming transitions"],
            "warnings": ["Unused variable 'x'"],
        }
        mock_validate_fn = MagicMock(return_value=mock_validation_result)

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.types": MagicMock(
                    WorkflowDefinition=MagicMock(from_dict=mock_from_dict),
                ),
                "aragora.workflow.validation": MagicMock(
                    validate_workflow=mock_validate_fn,
                ),
            },
        ):
            result = handler._validate_workflow({"name": "test", "steps": []})

        assert _status(result) == 200
        body = _body(result)
        assert body["valid"] is False
        assert len(body["errors"]) == 1


# ---------------------------------------------------------------------------
# POST /api/v1/workflows/{id}/replay
# ---------------------------------------------------------------------------


class TestReplayWorkflow:
    """Test POST replay endpoint."""

    def test_success(self, handler):
        """Should replay a workflow with given inputs."""
        mock_workflow = {"id": "wf_123", "name": "Test"}
        mock_exec_result = {"execution_id": "exec_1", "status": "completed"}

        with (
            patch(f"{PATCH_MOD}._run_async") as mock_run_async,
        ):
            # First call: get_workflow, second call: execute_workflow
            mock_run_async.side_effect = [mock_workflow, mock_exec_result]

            # Need to patch the imports inside the method
            mock_get_wf = MagicMock()
            mock_execute_wf = MagicMock()

            with patch.dict(
                "sys.modules",
                {
                    "aragora.server.handlers.workflows.execution": MagicMock(
                        execute_workflow=mock_execute_wf,
                    ),
                    "aragora.server.handlers.workflows.crud": MagicMock(
                        get_workflow=mock_get_wf,
                    ),
                },
            ):
                result = handler._replay_workflow(
                    "/api/v1/workflows/wf_123/replay",
                    {"inputs": {"key": "value"}},
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["execution_id"] == "exec_1"

    def test_workflow_not_found(self, handler):
        """Should return 404 when workflow does not exist."""
        with patch(f"{PATCH_MOD}._run_async") as mock_run_async:
            mock_run_async.return_value = None

            mock_get_wf = MagicMock()
            with patch.dict(
                "sys.modules",
                {
                    "aragora.server.handlers.workflows.execution": MagicMock(),
                    "aragora.server.handlers.workflows.crud": MagicMock(
                        get_workflow=mock_get_wf,
                    ),
                },
            ):
                result = handler._replay_workflow(
                    "/api/v1/workflows/wf_999/replay",
                    {},
                )

        assert _status(result) == 404
        assert "wf_999" in _body(result).get("error", "")

    def test_invalid_path_format(self, handler):
        """Invalid replay path should return 400."""
        result = handler._replay_workflow("/api/v1/workflows/replay", {})
        assert _status(result) == 400
        assert "invalid" in _body(result).get("error", "").lower()

    def test_too_short_path(self, handler):
        """Path with fewer than 5 parts should return 400."""
        result = handler._replay_workflow("/api/v1/replay", {})
        assert _status(result) == 400

    def test_path_without_replay_at_correct_index(self, handler):
        """Path where parts[4] != 'replay' should return 400."""
        result = handler._replay_workflow("/api/v1/workflows/wf_123/execute", {})
        assert _status(result) == 400

    def test_extracts_workflow_id_from_path(self, handler):
        """Should correctly extract workflow_id from path parts[3]."""
        with patch(f"{PATCH_MOD}._run_async") as mock_run_async:
            mock_run_async.side_effect = [{"id": "my-uuid-id"}, {"ok": True}]

            mock_get_wf = MagicMock()
            mock_execute_wf = MagicMock()

            with patch.dict(
                "sys.modules",
                {
                    "aragora.server.handlers.workflows.execution": MagicMock(
                        execute_workflow=mock_execute_wf,
                    ),
                    "aragora.server.handlers.workflows.crud": MagicMock(
                        get_workflow=mock_get_wf,
                    ),
                },
            ):
                handler._replay_workflow(
                    "/api/v1/workflows/my-uuid-id/replay",
                    {"inputs": {}},
                )

        # The first call to _run_async should be get_workflow("my-uuid-id")
        first_call = mock_run_async.call_args_list[0]
        # The coro passed is get_workflow("my-uuid-id")
        mock_get_wf.assert_called_once_with("my-uuid-id")

    def test_default_inputs_empty(self, handler):
        """Default inputs should be empty dict when not provided."""
        with patch(f"{PATCH_MOD}._run_async") as mock_run_async:
            mock_run_async.side_effect = [{"id": "wf_1"}, {"status": "ok"}]

            mock_execute_wf = MagicMock()
            mock_get_wf = MagicMock()

            with patch.dict(
                "sys.modules",
                {
                    "aragora.server.handlers.workflows.execution": MagicMock(
                        execute_workflow=mock_execute_wf,
                    ),
                    "aragora.server.handlers.workflows.crud": MagicMock(
                        get_workflow=mock_get_wf,
                    ),
                },
            ):
                handler._replay_workflow(
                    "/api/v1/workflows/wf_1/replay",
                    {},
                )

        # execute_workflow should be called with inputs={}
        mock_execute_wf.assert_called_once_with("wf_1", inputs={})

    def test_value_error_returns_404(self, handler):
        """ValueError during replay should return 404."""
        with patch(f"{PATCH_MOD}._run_async") as mock_run_async:
            mock_run_async.side_effect = ValueError("workflow not found")

            mock_get_wf = MagicMock()
            with patch.dict(
                "sys.modules",
                {
                    "aragora.server.handlers.workflows.execution": MagicMock(),
                    "aragora.server.handlers.workflows.crud": MagicMock(
                        get_workflow=mock_get_wf,
                    ),
                },
            ):
                result = handler._replay_workflow(
                    "/api/v1/workflows/wf_1/replay",
                    {},
                )

        assert _status(result) == 404
        assert "failed" in _body(result).get("error", "").lower()

    def test_connection_error_returns_503(self, handler):
        """ConnectionError during replay should return 503."""
        with patch(f"{PATCH_MOD}._run_async") as mock_run_async:
            mock_run_async.side_effect = ConnectionError("service down")

            mock_get_wf = MagicMock()
            with patch.dict(
                "sys.modules",
                {
                    "aragora.server.handlers.workflows.execution": MagicMock(),
                    "aragora.server.handlers.workflows.crud": MagicMock(
                        get_workflow=mock_get_wf,
                    ),
                },
            ):
                result = handler._replay_workflow(
                    "/api/v1/workflows/wf_1/replay",
                    {},
                )

        assert _status(result) == 503
        assert "unavailable" in _body(result).get("error", "").lower()

    def test_timeout_error_returns_503(self, handler):
        """TimeoutError during replay should return 503."""
        with patch(f"{PATCH_MOD}._run_async") as mock_run_async:
            mock_run_async.side_effect = TimeoutError("timed out")

            mock_get_wf = MagicMock()
            with patch.dict(
                "sys.modules",
                {
                    "aragora.server.handlers.workflows.execution": MagicMock(),
                    "aragora.server.handlers.workflows.crud": MagicMock(
                        get_workflow=mock_get_wf,
                    ),
                },
            ):
                result = handler._replay_workflow(
                    "/api/v1/workflows/wf_1/replay",
                    {},
                )

        assert _status(result) == 503

    def test_key_error_returns_500(self, handler):
        """KeyError during replay should return 500."""
        with patch(f"{PATCH_MOD}._run_async") as mock_run_async:
            mock_run_async.side_effect = KeyError("bad_key")

            mock_get_wf = MagicMock()
            with patch.dict(
                "sys.modules",
                {
                    "aragora.server.handlers.workflows.execution": MagicMock(),
                    "aragora.server.handlers.workflows.crud": MagicMock(
                        get_workflow=mock_get_wf,
                    ),
                },
            ):
                result = handler._replay_workflow(
                    "/api/v1/workflows/wf_1/replay",
                    {},
                )

        assert _status(result) == 500
        assert "internal" in _body(result).get("error", "").lower()

    def test_type_error_returns_500(self, handler):
        """TypeError during replay should return 500."""
        with patch(f"{PATCH_MOD}._run_async") as mock_run_async:
            mock_run_async.side_effect = TypeError("bad type")

            mock_get_wf = MagicMock()
            with patch.dict(
                "sys.modules",
                {
                    "aragora.server.handlers.workflows.execution": MagicMock(),
                    "aragora.server.handlers.workflows.crud": MagicMock(
                        get_workflow=mock_get_wf,
                    ),
                },
            ):
                result = handler._replay_workflow(
                    "/api/v1/workflows/wf_1/replay",
                    {},
                )

        assert _status(result) == 500

    def test_attribute_error_returns_500(self, handler):
        """AttributeError during replay should return 500."""
        with patch(f"{PATCH_MOD}._run_async") as mock_run_async:
            mock_run_async.side_effect = AttributeError("no attr")

            mock_get_wf = MagicMock()
            with patch.dict(
                "sys.modules",
                {
                    "aragora.server.handlers.workflows.execution": MagicMock(),
                    "aragora.server.handlers.workflows.crud": MagicMock(
                        get_workflow=mock_get_wf,
                    ),
                },
            ):
                result = handler._replay_workflow(
                    "/api/v1/workflows/wf_1/replay",
                    {},
                )

        assert _status(result) == 500

    def test_import_error_for_execution_module(self, handler):
        """Should handle ImportError when execution module is missing."""
        # When the local import fails, it raises ImportError which is not caught
        # by any handler, but get dispatched through handle_post's @handle_errors
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.handlers.workflows.execution": None,
            },
        ):
            # ImportError from the import inside _replay_workflow won't be caught
            # because the except clauses only catch ValueError, ConnectionError, etc.
            # It will propagate up. Let's verify that.
            with pytest.raises((ImportError, ModuleNotFoundError)):
                handler._replay_workflow(
                    "/api/v1/workflows/wf_1/replay",
                    {},
                )


# ---------------------------------------------------------------------------
# POST dispatch (handle_post)
# ---------------------------------------------------------------------------


class TestHandlePost:
    """Test POST request dispatch through handle_post."""

    def test_routes_to_generate(self, handler, mock_http):
        """POST to /generate should invoke _generate_workflow."""
        http = mock_http(method="POST", body={"description": "test"})

        with patch.object(
            handler,
            "_generate_workflow",
            return_value=MagicMock(
                status_code=200,
                body=b"{}",
                content_type="application/json",
            ),
        ) as mock_gen:
            result = handler.handle_post(
                "/api/v1/workflows/generate",
                {},
                http,
            )

        mock_gen.assert_called_once()
        assert _status(result) == 200

    def test_routes_to_auto_layout(self, handler, mock_http):
        """POST to /auto-layout should invoke _auto_layout."""
        http = mock_http(method="POST", body={"steps": [{"id": "s1"}]})

        with patch.object(
            handler,
            "_auto_layout",
            return_value=MagicMock(
                status_code=200,
                body=b"{}",
                content_type="application/json",
            ),
        ) as mock_layout:
            result = handler.handle_post(
                "/api/v1/workflows/auto-layout",
                {},
                http,
            )

        mock_layout.assert_called_once()

    def test_routes_to_from_pattern(self, handler, mock_http):
        """POST to /from-pattern should invoke _create_from_pattern."""
        http = mock_http(method="POST", body={"pattern": "sequential"})

        with patch.object(
            handler,
            "_create_from_pattern",
            return_value=MagicMock(
                status_code=201,
                body=b"{}",
                content_type="application/json",
            ),
        ) as mock_pattern:
            result = handler.handle_post(
                "/api/v1/workflows/from-pattern",
                {},
                http,
            )

        mock_pattern.assert_called_once()

    def test_routes_to_validate(self, handler, mock_http):
        """POST to /validate should invoke _validate_workflow."""
        http = mock_http(method="POST", body={"name": "test"})

        with patch.object(
            handler,
            "_validate_workflow",
            return_value=MagicMock(
                status_code=200,
                body=b"{}",
                content_type="application/json",
            ),
        ) as mock_validate:
            result = handler.handle_post(
                "/api/v1/workflows/validate",
                {},
                http,
            )

        mock_validate.assert_called_once()

    def test_routes_to_replay(self, handler, mock_http):
        """POST to /workflows/{id}/replay should invoke _replay_workflow."""
        http = mock_http(method="POST", body={"inputs": {}})

        with patch.object(
            handler,
            "_replay_workflow",
            return_value=MagicMock(
                status_code=200,
                body=b"{}",
                content_type="application/json",
            ),
        ) as mock_replay:
            result = handler.handle_post(
                "/api/v1/workflows/wf_123/replay",
                {},
                http,
            )

        mock_replay.assert_called_once()

    def test_unknown_post_path_returns_none(self, handler, mock_http):
        """POST to unrecognized path should return None."""
        http = mock_http(method="POST", body={})
        result = handler.handle_post(
            "/api/v1/workflows/unknown",
            {},
            http,
        )
        assert result is None

    def test_invalid_json_body_returns_error(self, handler, mock_http):
        """Invalid JSON body should return error from read_json_body_validated."""
        http = _MockHTTPHandler(method="POST")
        # Simulate invalid JSON
        http.rfile.read.return_value = b"not json"
        http.headers = {
            "Content-Length": "8",
            "Content-Type": "application/json",
        }
        result = handler.handle_post(
            "/api/v1/workflows/generate",
            {},
            http,
        )
        # Should get an error response (either 400 from body parse or from handler)
        assert result is not None
        assert _status(result) == 400

    def test_missing_content_type_with_body(self, handler):
        """POST with body but no Content-Type should return 415."""
        http = _MockHTTPHandler(method="POST", body={"description": "test"})
        # Remove Content-Type
        http.headers = {"Content-Length": "30"}

        result = handler.handle_post(
            "/api/v1/workflows/generate",
            {},
            http,
        )
        assert result is not None
        assert _status(result) == 415


# ---------------------------------------------------------------------------
# ROUTES constant
# ---------------------------------------------------------------------------


class TestRoutes:
    """Test the ROUTES class attribute."""

    def test_routes_contains_expected_paths(self, handler):
        """ROUTES should contain all documented endpoints."""
        routes = WorkflowBuilderHandler.ROUTES
        assert "/api/v1/workflows/generate" in routes
        assert "/api/v1/workflows/auto-layout" in routes
        assert "/api/v1/workflows/step-types" in routes
        assert "/api/v1/workflows/from-pattern" in routes
        assert "/api/v1/workflows/validate" in routes
        assert "/api/v1/workflows/*/replay" in routes

    def test_routes_count(self, handler):
        """ROUTES should have exactly 6 entries."""
        assert len(WorkflowBuilderHandler.ROUTES) == 6


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    """Test WorkflowBuilderHandler initialization."""

    def test_default_ctx(self):
        """Default ctx should be empty dict."""
        h = WorkflowBuilderHandler()
        assert h.ctx == {}

    def test_custom_ctx(self):
        """Should accept custom context."""
        ctx = {"store": MagicMock()}
        h = WorkflowBuilderHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_none_ctx_defaults_to_empty(self):
        """None ctx should default to empty dict."""
        h = WorkflowBuilderHandler(ctx=None)
        assert h.ctx == {}


# ---------------------------------------------------------------------------
# Integration: full POST dispatch with internal method execution
# ---------------------------------------------------------------------------


class TestIntegrationPostGenerate:
    """Integration tests for POST /generate via handle_post."""

    def test_full_generate_quick_flow(self, handler, mock_http):
        """Full path: handle_post -> _generate_workflow -> build_quick."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"name": "Generated Workflow"}

        mock_builder = MagicMock()
        mock_builder.return_value.build_quick.return_value = mock_result

        http = mock_http(method="POST", body={"description": "send weekly email"})

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.nl_builder": MagicMock(
                    NLWorkflowBuilder=mock_builder,
                    NLBuildConfig=MagicMock(),
                ),
            },
        ):
            result = handler.handle_post(
                "/api/v1/workflows/generate",
                {},
                http,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "Generated Workflow"

    def test_full_validate_flow(self, handler, mock_http):
        """Full path: handle_post -> _validate_workflow -> validate_workflow."""
        mock_definition = MagicMock()
        mock_from_dict = MagicMock(return_value=mock_definition)

        mock_validation = MagicMock()
        mock_validation.to_dict.return_value = {"valid": True, "errors": []}

        http = mock_http(method="POST", body={"name": "test", "steps": []})

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.types": MagicMock(
                    WorkflowDefinition=MagicMock(from_dict=mock_from_dict),
                ),
                "aragora.workflow.validation": MagicMock(
                    validate_workflow=MagicMock(return_value=mock_validation),
                ),
            },
        ):
            result = handler.handle_post(
                "/api/v1/workflows/validate",
                {},
                http,
            )

        assert _status(result) == 200
        assert _body(result)["valid"] is True

    def test_full_auto_layout_flow(self, handler, mock_http):
        """Full path: handle_post -> _auto_layout -> flow_layout."""
        mock_pos = MagicMock()
        mock_pos.to_dict.return_value = {"step_id": "s1", "x": 100, "y": 50}

        http = mock_http(
            method="POST",
            body={"steps": [{"id": "s1"}], "transitions": []},
        )

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.layout": MagicMock(
                    flow_layout=MagicMock(return_value=[mock_pos]),
                    grid_layout=MagicMock(),
                ),
            },
        ):
            result = handler.handle_post(
                "/api/v1/workflows/auto-layout",
                {},
                http,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 1
        assert body["positions"][0]["x"] == 100

    def test_full_from_pattern_flow(self, handler, mock_http):
        """Full path: handle_post -> _create_from_pattern -> create_pattern."""
        mock_workflow = MagicMock()
        mock_workflow.to_dict.return_value = {"id": "wf_p1"}

        mock_pattern = MagicMock()
        mock_pattern.create.return_value = mock_workflow

        http = mock_http(method="POST", body={"pattern": "sequential"})

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.patterns.base": MagicMock(PatternType=MagicMock()),
                "aragora.workflow.patterns": MagicMock(
                    create_pattern=MagicMock(return_value=mock_pattern),
                ),
            },
        ):
            result = handler.handle_post(
                "/api/v1/workflows/from-pattern",
                {},
                http,
            )

        assert _status(result) == 201


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_replay_path_with_trailing_slash(self, handler):
        """Replay path with trailing slash should still be parsed correctly."""
        # strip("/") will remove trailing slash, so parts should be correct
        with patch(f"{PATCH_MOD}._run_async") as mock_run_async:
            mock_run_async.side_effect = [{"id": "wf_1"}, {"ok": True}]

            with patch.dict(
                "sys.modules",
                {
                    "aragora.server.handlers.workflows.execution": MagicMock(),
                    "aragora.server.handlers.workflows.crud": MagicMock(),
                },
            ):
                result = handler._replay_workflow(
                    "/api/v1/workflows/wf_1/replay/",
                    {},
                )

        # The trailing slash gets stripped, parts[4] should still be "replay"
        # Actually with trailing slash: strip("/") gives "api/v1/workflows/wf_1/replay"
        # So parts = ["api", "v1", "workflows", "wf_1", "replay"] - correct
        assert _status(result) == 200

    def test_replay_path_with_special_chars_in_id(self, handler):
        """Workflow ID with special characters."""
        with patch(f"{PATCH_MOD}._run_async") as mock_run_async:
            mock_run_async.side_effect = [{"id": "wf-abc_123"}, {"status": "done"}]

            with patch.dict(
                "sys.modules",
                {
                    "aragora.server.handlers.workflows.execution": MagicMock(),
                    "aragora.server.handlers.workflows.crud": MagicMock(),
                },
            ):
                result = handler._replay_workflow(
                    "/api/v1/workflows/wf-abc_123/replay",
                    {},
                )

        assert _status(result) == 200

    def test_generate_description_whitespace_only(self, handler):
        """Description with only whitespace should still be passed (not empty string)."""
        # "   " is truthy in Python, so it passes the `if not description` check
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {}

        mock_builder = MagicMock()
        mock_builder.return_value.build_quick.return_value = mock_result

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.nl_builder": MagicMock(
                    NLWorkflowBuilder=mock_builder,
                    NLBuildConfig=MagicMock(),
                ),
            },
        ):
            result = handler._generate_workflow({"description": "   "})

        # Whitespace is truthy, so it proceeds
        assert _status(result) == 200

    def test_auto_layout_many_steps(self, handler):
        """Auto-layout with many steps should work."""
        positions = []
        for i in range(100):
            p = MagicMock()
            p.to_dict.return_value = {"step_id": f"s{i}", "x": i * 10, "y": 0}
            positions.append(p)

        steps = [{"id": f"s{i}"} for i in range(100)]

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.layout": MagicMock(
                    flow_layout=MagicMock(return_value=positions),
                    grid_layout=MagicMock(),
                ),
            },
        ):
            result = handler._auto_layout({"steps": steps})

        body = _body(result)
        assert body["count"] == 100

    def test_from_pattern_with_only_name(self, handler):
        """Pattern creation with only name kwarg."""
        mock_workflow = MagicMock()
        mock_workflow.to_dict.return_value = {}

        mock_pattern = MagicMock()
        mock_pattern.create.return_value = mock_workflow

        mock_create_fn = MagicMock(return_value=mock_pattern)

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.patterns.base": MagicMock(PatternType=MagicMock()),
                "aragora.workflow.patterns": MagicMock(create_pattern=mock_create_fn),
            },
        ):
            handler._create_from_pattern(
                {
                    "pattern": "sequential",
                    "name": "My Flow",
                }
            )

        call_kwargs = mock_create_fn.call_args[1]
        assert call_kwargs["name"] == "My Flow"
        assert "agents" not in call_kwargs
        assert "task" not in call_kwargs

    def test_handle_post_empty_body(self, handler):
        """POST with empty body {} should still route correctly."""
        http = _MockHTTPHandler(method="POST", body={})
        # generate expects description, so should get 400
        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.nl_builder": MagicMock(
                    NLWorkflowBuilder=MagicMock(),
                    NLBuildConfig=MagicMock(),
                ),
            },
        ):
            result = handler.handle_post(
                "/api/v1/workflows/generate",
                {},
                http,
            )

        assert _status(result) == 400
