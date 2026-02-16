"""
Tests for OpenClaw Workflow Steps.

Tests cover:
- OpenClawActionConfig creation and defaults
- VALID_ACTION_TYPES contents
- OpenClawActionStep construction, config validation, execute, and template resolution
- OpenClawSessionStep construction, config validation, create/end session
- register_openclaw_steps() registration function
- Error handling: ImportError, generic exceptions, missing fields, skip mode

Since the import chain from aragora.workflow.step has Python 3.11 compat issues
in the security module, we mock BaseStep and WorkflowContext before importing
the openclaw module.
"""

import sys
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# Mock BaseStep and WorkflowContext (cannot import from aragora.workflow.step)
# ============================================================================


@dataclass
class MockWorkflowContext:
    """Stand-in for WorkflowContext since the real one can't be imported."""

    workflow_id: str = "wf-test"
    definition_id: str = "def-test"
    inputs: dict = field(default_factory=dict)
    step_outputs: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    current_step_id: str | None = None
    current_step_config: dict = field(default_factory=dict)

    def get_input(self, key, default=None):
        return self.inputs.get(key, default)

    def get_step_output(self, step_id, default=None):
        return self.step_outputs.get(step_id, default)

    def get_state(self, key, default=None):
        return self.state.get(key, default)

    def set_state(self, key, value):
        self.state[key] = value

    def get_config(self, key, default=None):
        return self.current_step_config.get(key, default)


class MockBaseStep:
    """Stand-in for BaseStep since the real one can't be imported."""

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        self._name = name
        self._config = config or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    def validate_config(self) -> bool:
        return True

    async def execute(self, context) -> Any:
        raise NotImplementedError

    async def checkpoint(self) -> dict[str, Any]:
        return {}

    async def restore(self, state: dict[str, Any]) -> None:
        pass


# Patch the module before importing openclaw, then immediately restore
_mock_step_module = MagicMock()
_mock_step_module.BaseStep = MockBaseStep
_mock_step_module.WorkflowContext = MockWorkflowContext

_original_step = sys.modules.get("aragora.workflow.step")
_original_workflow = sys.modules.get("aragora.workflow")
_workflow_was_absent = "aragora.workflow" not in sys.modules

sys.modules["aragora.workflow.step"] = _mock_step_module
if _workflow_was_absent:
    sys.modules["aragora.workflow"] = MagicMock()

from aragora.workflow.nodes.openclaw import (  # noqa: E402
    VALID_ACTION_TYPES,
    OpenClawActionConfig,
    OpenClawActionStep,
    OpenClawSessionStep,
    register_openclaw_steps,
)

# Immediately restore real modules so other test files are not affected
if _original_step is not None:
    sys.modules["aragora.workflow.step"] = _original_step
elif "aragora.workflow.step" in sys.modules:
    del sys.modules["aragora.workflow.step"]
if _original_workflow is not None:
    sys.modules["aragora.workflow"] = _original_workflow
elif _workflow_was_absent and "aragora.workflow" in sys.modules:
    del sys.modules["aragora.workflow"]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def action_config_defaults():
    """Return a default OpenClawActionConfig."""
    return OpenClawActionConfig()


@pytest.fixture
def mock_context():
    """Return a basic MockWorkflowContext."""
    return MockWorkflowContext()


@pytest.fixture
def mock_proxy():
    """Return an AsyncMock proxy with common methods."""
    proxy = AsyncMock()
    proxy.execute_action = AsyncMock(
        return_value={
            "success": True,
            "action_id": "act-123",
            "result": {"output": "ok"},
            "execution_time_ms": 42,
            "audit_id": "aud-456",
            "requires_approval": False,
        }
    )
    proxy.create_session = AsyncMock()
    proxy.end_session = AsyncMock()
    return proxy


@pytest.fixture
def shell_step():
    """Return an OpenClawActionStep configured for a shell command."""
    return OpenClawActionStep(
        name="Run ls",
        config={
            "action_type": "shell",
            "session_id": "sess-1",
            "command": "ls -la",
        },
    )


@pytest.fixture
def session_create_step():
    """Return an OpenClawSessionStep configured for session creation."""
    return OpenClawSessionStep(
        name="Create session",
        config={
            "operation": "create",
            "workspace_id": "/workspace/test",
            "roles": ["developer"],
        },
    )


@pytest.fixture
def session_end_step():
    """Return an OpenClawSessionStep configured for ending a session."""
    return OpenClawSessionStep(
        name="End session",
        config={
            "operation": "end",
            "session_id": "sess-abc",
        },
    )


# ============================================================================
# VALID_ACTION_TYPES Tests
# ============================================================================


class TestValidActionTypes:
    """Tests for the VALID_ACTION_TYPES frozenset."""

    def test_contains_shell(self):
        """Shell action type should be valid."""
        assert "shell" in VALID_ACTION_TYPES

    def test_contains_file_actions(self):
        """File action types should all be valid."""
        assert "file_read" in VALID_ACTION_TYPES
        assert "file_write" in VALID_ACTION_TYPES
        assert "file_delete" in VALID_ACTION_TYPES

    def test_contains_browser_actions(self):
        """Browser-related action types should be valid."""
        assert "browser" in VALID_ACTION_TYPES
        assert "screenshot" in VALID_ACTION_TYPES

    def test_contains_api(self):
        """API action type should be valid."""
        assert "api" in VALID_ACTION_TYPES

    def test_contains_keyboard_mouse(self):
        """Keyboard and mouse action types should be valid."""
        assert "keyboard" in VALID_ACTION_TYPES
        assert "mouse" in VALID_ACTION_TYPES

    def test_total_count(self):
        """Exactly 7 action types should be defined."""
        assert len(VALID_ACTION_TYPES) == 9

    def test_is_frozenset(self):
        """VALID_ACTION_TYPES should be immutable."""
        assert isinstance(VALID_ACTION_TYPES, frozenset)


# ============================================================================
# OpenClawActionConfig Tests
# ============================================================================


class TestOpenClawActionConfig:
    """Tests for the OpenClawActionConfig dataclass."""

    def test_defaults(self, action_config_defaults):
        """Default config should have sensible values."""
        cfg = action_config_defaults
        assert cfg.action_type == "shell"
        assert cfg.session_id == ""
        assert cfg.command == ""
        assert cfg.path == ""
        assert cfg.content == ""
        assert cfg.url == ""
        assert cfg.params == {}
        assert cfg.timeout_seconds == 60.0
        assert cfg.require_approval is False
        assert cfg.on_failure == "error"

    def test_custom_values(self):
        """Custom values should override defaults."""
        cfg = OpenClawActionConfig(
            action_type="browser",
            session_id="sess-42",
            url="https://example.com",
            timeout_seconds=120.0,
            require_approval=True,
            on_failure="skip",
        )
        assert cfg.action_type == "browser"
        assert cfg.session_id == "sess-42"
        assert cfg.url == "https://example.com"
        assert cfg.timeout_seconds == 120.0
        assert cfg.require_approval is True
        assert cfg.on_failure == "skip"

    def test_params_dict_independent(self):
        """Each config instance should have its own params dict."""
        cfg1 = OpenClawActionConfig()
        cfg2 = OpenClawActionConfig()
        cfg1.params["key"] = "value"
        assert "key" not in cfg2.params


# ============================================================================
# OpenClawActionStep Construction Tests
# ============================================================================


class TestOpenClawActionStepConstruction:
    """Tests for OpenClawActionStep construction and config parsing."""

    def test_basic_construction(self, shell_step):
        """Step should store name and parsed config."""
        assert shell_step.name == "Run ls"
        assert shell_step._step_config.action_type == "shell"
        assert shell_step._step_config.session_id == "sess-1"
        assert shell_step._step_config.command == "ls -la"

    def test_step_type_class_attr(self):
        """step_type class attribute should be 'openclaw_action'."""
        assert OpenClawActionStep.step_type == "openclaw_action"

    def test_no_config(self):
        """Constructing with no config should use defaults."""
        step = OpenClawActionStep(name="Empty")
        assert step._step_config.action_type == "shell"
        assert step._step_config.command == ""
        assert step._step_config.session_id == ""

    def test_file_write_config(self):
        """File write config should parse path and content."""
        step = OpenClawActionStep(
            name="Write file",
            config={
                "action_type": "file_write",
                "session_id": "sess-1",
                "path": "/tmp/test.txt",
                "content": "hello world",
            },
        )
        assert step._step_config.action_type == "file_write"
        assert step._step_config.path == "/tmp/test.txt"
        assert step._step_config.content == "hello world"

    def test_api_config_with_params(self):
        """API config should accept extra params."""
        step = OpenClawActionStep(
            name="API call",
            config={
                "action_type": "api",
                "session_id": "sess-1",
                "params": {"endpoint": "/v1/data", "method": "GET"},
            },
        )
        assert step._step_config.action_type == "api"
        assert step._step_config.params["endpoint"] == "/v1/data"


# ============================================================================
# OpenClawActionStep validate_config Tests
# ============================================================================


class TestOpenClawActionStepValidation:
    """Tests for OpenClawActionStep.validate_config()."""

    def test_valid_shell_config(self):
        """Shell config with command should pass validation."""
        step = OpenClawActionStep(
            name="Shell",
            config={"action_type": "shell", "command": "echo hi"},
        )
        assert step.validate_config() is True

    def test_invalid_action_type(self):
        """Unknown action type should fail validation."""
        step = OpenClawActionStep(
            name="Bad",
            config={"action_type": "invalid_type"},
        )
        assert step.validate_config() is False

    def test_shell_missing_command(self):
        """Shell action without command should fail validation."""
        step = OpenClawActionStep(
            name="No cmd",
            config={"action_type": "shell"},
        )
        assert step.validate_config() is False

    def test_file_read_missing_path(self):
        """File read without path should fail validation."""
        step = OpenClawActionStep(
            name="No path",
            config={"action_type": "file_read"},
        )
        assert step.validate_config() is False

    def test_file_write_missing_path(self):
        """File write without path should fail validation."""
        step = OpenClawActionStep(
            name="No path",
            config={"action_type": "file_write", "content": "data"},
        )
        assert step.validate_config() is False

    def test_file_delete_missing_path(self):
        """File delete without path should fail validation."""
        step = OpenClawActionStep(
            name="No path",
            config={"action_type": "file_delete"},
        )
        assert step.validate_config() is False

    def test_file_read_with_path_valid(self):
        """File read with path should pass validation."""
        step = OpenClawActionStep(
            name="Read",
            config={"action_type": "file_read", "path": "/tmp/data.txt"},
        )
        assert step.validate_config() is True

    def test_browser_missing_url(self):
        """Browser action without URL should fail validation."""
        step = OpenClawActionStep(
            name="Browse",
            config={"action_type": "browser"},
        )
        assert step.validate_config() is False

    def test_screenshot_missing_url(self):
        """Screenshot action without URL should fail validation."""
        step = OpenClawActionStep(
            name="Screenshot",
            config={"action_type": "screenshot"},
        )
        assert step.validate_config() is False

    def test_browser_with_url_valid(self):
        """Browser action with URL should pass validation."""
        step = OpenClawActionStep(
            name="Browse",
            config={"action_type": "browser", "url": "https://example.com"},
        )
        assert step.validate_config() is True

    def test_api_valid(self):
        """API action should pass validation (no required fields beyond type)."""
        step = OpenClawActionStep(
            name="API",
            config={"action_type": "api"},
        )
        assert step.validate_config() is True


# ============================================================================
# OpenClawActionStep execute Tests
# ============================================================================


class TestOpenClawActionStepExecute:
    """Tests for OpenClawActionStep.execute()."""

    @pytest.mark.asyncio
    async def test_execute_shell_success(self, shell_step, mock_proxy, mock_context):
        """Successful shell execution should return structured result."""
        with patch.object(shell_step, "_get_proxy", return_value=mock_proxy):
            result = await shell_step.execute(mock_context)

        assert result["success"] is True
        assert result["action_type"] == "shell"
        assert result["action_id"] == "act-123"
        assert result["execution_time_ms"] == 42
        assert result["audit_id"] == "aud-456"
        assert result["requires_approval"] is False

    @pytest.mark.asyncio
    async def test_execute_missing_session_id(self, mock_context):
        """Execute without session_id should return error dict."""
        step = OpenClawActionStep(
            name="No session",
            config={"action_type": "shell", "command": "echo hi"},
        )
        result = await step.execute(mock_context)

        assert result["success"] is False
        assert "session_id is required" in result["error"]
        assert result["action_type"] == "shell"

    @pytest.mark.asyncio
    async def test_execute_builds_shell_payload(self, mock_proxy, mock_context):
        """Shell execution should send command in payload."""
        step = OpenClawActionStep(
            name="Shell cmd",
            config={
                "action_type": "shell",
                "session_id": "sess-1",
                "command": "echo hello",
            },
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            await step.execute(mock_context)

        mock_proxy.execute_action.assert_called_once()
        call_kwargs = mock_proxy.execute_action.call_args[1]
        assert call_kwargs["session_id"] == "sess-1"
        assert call_kwargs["action_type"] == "shell"
        assert call_kwargs["input"]["command"] == "echo hello"
        assert call_kwargs["metadata"]["workflow_id"] == "wf-test"

    @pytest.mark.asyncio
    async def test_execute_builds_file_read_payload(self, mock_proxy, mock_context):
        """File read execution should send path in payload."""
        step = OpenClawActionStep(
            name="Read file",
            config={
                "action_type": "file_read",
                "session_id": "sess-1",
                "path": "/workspace/file.txt",
            },
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            await step.execute(mock_context)

        call_kwargs = mock_proxy.execute_action.call_args[1]
        assert call_kwargs["input"]["path"] == "/workspace/file.txt"

    @pytest.mark.asyncio
    async def test_execute_builds_file_write_payload(self, mock_proxy, mock_context):
        """File write execution should send path and content in payload."""
        step = OpenClawActionStep(
            name="Write file",
            config={
                "action_type": "file_write",
                "session_id": "sess-1",
                "path": "/workspace/out.txt",
                "content": "output data",
            },
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            await step.execute(mock_context)

        call_kwargs = mock_proxy.execute_action.call_args[1]
        assert call_kwargs["input"]["path"] == "/workspace/out.txt"
        assert call_kwargs["input"]["content"] == "output data"

    @pytest.mark.asyncio
    async def test_execute_builds_browser_payload(self, mock_proxy, mock_context):
        """Browser execution should send URL in payload."""
        step = OpenClawActionStep(
            name="Browse",
            config={
                "action_type": "browser",
                "session_id": "sess-1",
                "url": "https://example.com",
            },
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            await step.execute(mock_context)

        call_kwargs = mock_proxy.execute_action.call_args[1]
        assert call_kwargs["input"]["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_execute_builds_api_payload(self, mock_proxy, mock_context):
        """API execution should merge params into input."""
        step = OpenClawActionStep(
            name="API",
            config={
                "action_type": "api",
                "session_id": "sess-1",
                "params": {"endpoint": "/v1/items", "method": "POST"},
            },
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            await step.execute(mock_context)

        call_kwargs = mock_proxy.execute_action.call_args[1]
        assert call_kwargs["input"]["endpoint"] == "/v1/items"
        assert call_kwargs["input"]["method"] == "POST"

    @pytest.mark.asyncio
    async def test_execute_failure_on_failure_error(self, mock_proxy, mock_context):
        """When proxy returns failure and on_failure=error, result shows failure."""
        mock_proxy.execute_action.return_value = {
            "success": False,
            "error": "permission denied",
        }
        step = OpenClawActionStep(
            name="Fail",
            config={
                "action_type": "shell",
                "session_id": "sess-1",
                "command": "rm -rf /",
                "on_failure": "error",
            },
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            result = await step.execute(mock_context)

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_execute_failure_on_failure_skip(self, mock_proxy, mock_context):
        """When proxy returns failure and on_failure=skip, result is skipped."""
        mock_proxy.execute_action.return_value = {
            "success": False,
            "error": "permission denied",
        }
        step = OpenClawActionStep(
            name="Skip fail",
            config={
                "action_type": "shell",
                "session_id": "sess-1",
                "command": "restricted",
                "on_failure": "skip",
            },
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            result = await step.execute(mock_context)

        assert result["success"] is False
        assert result["skipped"] is True
        assert result["error"] == "permission denied"

    @pytest.mark.asyncio
    async def test_execute_exception_on_failure_error(self, mock_proxy, mock_context):
        """When proxy raises and on_failure=error, error is returned."""
        mock_proxy.execute_action.side_effect = RuntimeError("connection lost")
        step = OpenClawActionStep(
            name="Crash",
            config={
                "action_type": "shell",
                "session_id": "sess-1",
                "command": "echo",
                "on_failure": "error",
            },
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            result = await step.execute(mock_context)

        assert result["success"] is False
        assert result["error"] == "OpenClaw action failed"
        assert "skipped" not in result

    @pytest.mark.asyncio
    async def test_execute_exception_on_failure_skip(self, mock_proxy, mock_context):
        """When proxy raises and on_failure=skip, result is skipped."""
        mock_proxy.execute_action.side_effect = RuntimeError("timeout")
        step = OpenClawActionStep(
            name="Crash skip",
            config={
                "action_type": "shell",
                "session_id": "sess-1",
                "command": "echo",
                "on_failure": "skip",
            },
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            result = await step.execute(mock_context)

        assert result["success"] is False
        assert result["skipped"] is True
        assert "OpenClaw action failed" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_import_error(self, mock_context):
        """When gateway module is not available, ImportError is caught."""
        step = OpenClawActionStep(
            name="No gateway",
            config={
                "action_type": "shell",
                "session_id": "sess-1",
                "command": "echo hi",
            },
        )

        async def raise_import_error():
            raise ImportError("No module named 'aragora.gateway'")

        with patch.object(step, "_get_proxy", side_effect=ImportError("not available")):
            result = await step.execute(mock_context)

        assert result["success"] is False
        assert "not available" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_result_with_to_dict(self, mock_proxy, mock_context):
        """When proxy returns object with to_dict(), it should be used."""
        result_obj = MagicMock()
        result_obj.to_dict.return_value = {
            "success": True,
            "action_id": "act-obj-1",
            "result": {"data": "from_obj"},
            "execution_time_ms": 10,
            "audit_id": "aud-obj-1",
            "requires_approval": False,
        }
        mock_proxy.execute_action.return_value = result_obj

        step = OpenClawActionStep(
            name="Object result",
            config={
                "action_type": "shell",
                "session_id": "sess-1",
                "command": "echo",
            },
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            result = await step.execute(mock_context)

        assert result["success"] is True
        assert result["action_id"] == "act-obj-1"
        assert result["result"] == {"data": "from_obj"}

    @pytest.mark.asyncio
    async def test_execute_result_raw_string(self, mock_proxy, mock_context):
        """When proxy returns a non-dict, non-to_dict object, raw string is used."""
        mock_proxy.execute_action.return_value = "plain text response"

        step = OpenClawActionStep(
            name="Raw result",
            config={
                "action_type": "shell",
                "session_id": "sess-1",
                "command": "echo",
            },
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            result = await step.execute(mock_context)

        # The raw string gets wrapped: {"raw": "plain text response"}
        # Then success defaults to True since "success" key is absent
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_uses_context_step_config_override(self, mock_proxy, mock_context):
        """current_step_config on context should override step config."""
        step = OpenClawActionStep(
            name="Override",
            config={
                "action_type": "shell",
                "session_id": "sess-1",
                "command": "original",
            },
        )
        # Override command via context
        mock_context.current_step_config = {"command": "overridden"}

        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            await step.execute(mock_context)

        call_kwargs = mock_proxy.execute_action.call_args[1]
        assert call_kwargs["input"]["command"] == "overridden"


# ============================================================================
# OpenClawActionStep _resolve Tests
# ============================================================================


class TestOpenClawActionStepResolve:
    """Tests for template variable resolution via _resolve()."""

    def test_no_template(self, shell_step, mock_context):
        """Plain strings should pass through unchanged."""
        assert shell_step._resolve("hello", mock_context) == "hello"

    def test_empty_string(self, shell_step, mock_context):
        """Empty string should return empty string."""
        assert shell_step._resolve("", mock_context) == ""

    def test_step_output_field_resolution(self, shell_step, mock_context):
        """Template {step.<id>.<field>} should resolve from step outputs."""
        mock_context.step_outputs = {"create_session": {"session_id": "sess-resolved"}}
        result = shell_step._resolve("{step.create_session.session_id}", mock_context)
        assert result == "sess-resolved"

    def test_step_output_without_field(self, shell_step, mock_context):
        """Template {step.<id>} without field should return str(output)."""
        mock_context.step_outputs = {"prev": "previous_value"}
        result = shell_step._resolve("{step.prev}", mock_context)
        assert result == "previous_value"

    def test_input_resolution(self, shell_step, mock_context):
        """Template {input.<key>} should resolve from workflow inputs."""
        mock_context.inputs = {"project_dir": "/workspace/proj"}
        result = shell_step._resolve("{input.project_dir}", mock_context)
        assert result == "/workspace/proj"

    def test_unresolvable_template(self, shell_step, mock_context):
        """Unresolvable template should remain unchanged."""
        result = shell_step._resolve("{step.missing.field}", mock_context)
        assert result == "{step.missing.field}"

    def test_mixed_text_and_template(self, shell_step, mock_context):
        """Templates embedded in other text should resolve inline."""
        mock_context.step_outputs = {"setup": {"dir": "/home/user"}}
        result = shell_step._resolve("ls {step.setup.dir}/docs", mock_context)
        assert result == "ls /home/user/docs"

    def test_multiple_templates(self, shell_step, mock_context):
        """Multiple templates in the same string should all resolve."""
        mock_context.inputs = {"user": "alice"}
        mock_context.step_outputs = {"session": {"id": "s1"}}
        result = shell_step._resolve("user={input.user} session={step.session.id}", mock_context)
        assert result == "user=alice session=s1"

    @pytest.mark.asyncio
    async def test_resolve_used_in_execute(self, mock_proxy, mock_context):
        """Template resolution should be applied during execute."""
        mock_context.step_outputs = {"create_session": {"session_id": "sess-dynamic"}}
        step = OpenClawActionStep(
            name="Template step",
            config={
                "action_type": "shell",
                "session_id": "{step.create_session.session_id}",
                "command": "whoami",
            },
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            await step.execute(mock_context)

        call_kwargs = mock_proxy.execute_action.call_args[1]
        assert call_kwargs["session_id"] == "sess-dynamic"


# ============================================================================
# OpenClawSessionStep Construction Tests
# ============================================================================


class TestOpenClawSessionStepConstruction:
    """Tests for OpenClawSessionStep construction."""

    def test_basic_construction(self, session_create_step):
        """Session step should store name and config."""
        assert session_create_step.name == "Create session"
        assert session_create_step._config["operation"] == "create"
        assert session_create_step._config["workspace_id"] == "/workspace/test"
        assert session_create_step._config["roles"] == ["developer"]

    def test_step_type_class_attr(self):
        """step_type class attribute should be 'openclaw_session'."""
        assert OpenClawSessionStep.step_type == "openclaw_session"

    def test_no_config(self):
        """Constructing with no config should use empty dict."""
        step = OpenClawSessionStep(name="Default")
        assert step._config == {}


# ============================================================================
# OpenClawSessionStep validate_config Tests
# ============================================================================


class TestOpenClawSessionStepValidation:
    """Tests for OpenClawSessionStep.validate_config()."""

    def test_create_valid(self, session_create_step):
        """Create operation should pass validation."""
        assert session_create_step.validate_config() is True

    def test_end_with_session_id_valid(self, session_end_step):
        """End operation with session_id should pass validation."""
        assert session_end_step.validate_config() is True

    def test_end_missing_session_id(self):
        """End operation without session_id should fail validation."""
        step = OpenClawSessionStep(
            name="Bad end",
            config={"operation": "end"},
        )
        assert step.validate_config() is False

    def test_invalid_operation(self):
        """Invalid operation should fail validation."""
        step = OpenClawSessionStep(
            name="Bad op",
            config={"operation": "restart"},
        )
        assert step.validate_config() is False

    def test_default_operation_create(self):
        """No explicit operation should default to create and pass."""
        step = OpenClawSessionStep(name="Default op", config={})
        assert step.validate_config() is True


# ============================================================================
# OpenClawSessionStep execute Tests
# ============================================================================


class TestOpenClawSessionStepExecute:
    """Tests for OpenClawSessionStep.execute()."""

    @pytest.mark.asyncio
    async def test_create_session_success(self, mock_proxy, mock_context):
        """Successful session creation should return session details."""
        session_obj = MagicMock()
        session_obj.session_id = "sess-new-123"
        mock_proxy.create_session.return_value = session_obj

        step = OpenClawSessionStep(
            name="Create",
            config={
                "operation": "create",
                "workspace_id": "/workspace/proj",
                "roles": ["admin"],
            },
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            result = await step.execute(mock_context)

        assert result["success"] is True
        assert result["operation"] == "create"
        assert result["session_id"] == "sess-new-123"
        assert result["workspace_id"] == "/workspace/proj"

        mock_proxy.create_session.assert_called_once_with(
            user_id="workflow",
            tenant_id="default",
            workspace_id="/workspace/proj",
            roles=["admin"],
        )

    @pytest.mark.asyncio
    async def test_create_session_with_metadata(self, mock_proxy, mock_context):
        """Session creation should pull user_id and tenant_id from metadata."""
        mock_context.metadata = {
            "user_id": "user-42",
            "tenant_id": "tenant-7",
        }
        session_obj = MagicMock()
        session_obj.session_id = "sess-meta"
        mock_proxy.create_session.return_value = session_obj

        step = OpenClawSessionStep(
            name="Meta create",
            config={"operation": "create"},
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            result = await step.execute(mock_context)

        assert result["success"] is True
        assert result["user_id"] == "user-42"
        assert result["tenant_id"] == "tenant-7"

        mock_proxy.create_session.assert_called_once_with(
            user_id="user-42",
            tenant_id="tenant-7",
            workspace_id="/workspace",
            roles=["user"],
        )

    @pytest.mark.asyncio
    async def test_create_session_string_fallback(self, mock_proxy, mock_context):
        """When session has no session_id attr, str() is used."""
        mock_proxy.create_session.return_value = "plain-session-id"

        step = OpenClawSessionStep(
            name="String session",
            config={"operation": "create"},
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            result = await step.execute(mock_context)

        assert result["success"] is True
        assert result["session_id"] == "plain-session-id"

    @pytest.mark.asyncio
    async def test_end_session_success(self, mock_proxy, mock_context):
        """Successful session end should return confirmation."""
        step = OpenClawSessionStep(
            name="End",
            config={
                "operation": "end",
                "session_id": "sess-to-end",
            },
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            result = await step.execute(mock_context)

        assert result["success"] is True
        assert result["operation"] == "end"
        assert result["session_id"] == "sess-to-end"
        mock_proxy.end_session.assert_called_once_with("sess-to-end")

    @pytest.mark.asyncio
    async def test_end_session_with_template(self, mock_proxy, mock_context):
        """End session should resolve template variables in session_id."""
        mock_context.step_outputs = {"create_step": {"session_id": "sess-from-template"}}
        step = OpenClawSessionStep(
            name="End template",
            config={
                "operation": "end",
                "session_id": "{step.create_step.session_id}",
            },
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            result = await step.execute(mock_context)

        assert result["success"] is True
        assert result["session_id"] == "sess-from-template"
        mock_proxy.end_session.assert_called_once_with("sess-from-template")

    @pytest.mark.asyncio
    async def test_end_session_missing_session_id(self, mock_proxy, mock_context):
        """End session without session_id should return error."""
        step = OpenClawSessionStep(
            name="End no id",
            config={"operation": "end", "session_id": ""},
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            result = await step.execute(mock_context)

        assert result["success"] is False
        assert "session_id is required" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_import_error(self, mock_context):
        """ImportError should be caught and reported."""
        step = OpenClawSessionStep(
            name="No gateway",
            config={"operation": "create"},
        )
        with patch.object(step, "_get_proxy", side_effect=ImportError("gateway missing")):
            result = await step.execute(mock_context)

        assert result["success"] is False
        assert "not available" in result["error"]
        assert result["operation"] == "create"

    @pytest.mark.asyncio
    async def test_execute_generic_exception(self, mock_proxy, mock_context):
        """Generic exception during session creation should be caught."""
        mock_proxy.create_session.side_effect = RuntimeError("db down")
        step = OpenClawSessionStep(
            name="Crash",
            config={"operation": "create"},
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            result = await step.execute(mock_context)

        assert result["success"] is False
        assert "failed" in result["error"].lower()
        assert result["operation"] == "create"

    @pytest.mark.asyncio
    async def test_execute_unknown_operation(self, mock_proxy, mock_context):
        """Unknown operation should return error dict."""
        step = OpenClawSessionStep(
            name="Unknown op",
            config={"operation": "restart"},
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            result = await step.execute(mock_context)

        assert result["success"] is False
        assert "Unknown operation" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_uses_context_step_config_override(self, mock_proxy, mock_context):
        """current_step_config on context should merge with step config."""
        session_obj = MagicMock()
        session_obj.session_id = "sess-override"
        mock_proxy.create_session.return_value = session_obj

        step = OpenClawSessionStep(
            name="Override",
            config={
                "operation": "create",
                "workspace_id": "/workspace/orig",
            },
        )
        mock_context.current_step_config = {"workspace_id": "/workspace/override"}

        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            result = await step.execute(mock_context)

        assert result["workspace_id"] == "/workspace/override"


# ============================================================================
# register_openclaw_steps Tests
# ============================================================================


class TestRegisterOpenclawSteps:
    """Tests for the register_openclaw_steps() helper function."""

    def test_register_calls_register_step_type(self):
        """register_openclaw_steps should register both step types."""
        mock_register = MagicMock()
        with patch(
            "aragora.workflow.nodes.openclaw.register_step_type",
            mock_register,
            create=True,
        ):
            # We need to patch the import inside the function
            mock_nodes_module = MagicMock()
            mock_nodes_module.register_step_type = mock_register
            with patch.dict(
                sys.modules,
                {"aragora.workflow.nodes": mock_nodes_module},
            ):
                register_openclaw_steps()

        assert mock_register.call_count == 2
        mock_register.assert_any_call("openclaw_action", OpenClawActionStep)
        mock_register.assert_any_call("openclaw_session", OpenClawSessionStep)

    def test_register_handles_import_error(self):
        """register_openclaw_steps should not raise if import fails."""
        with patch.dict(
            sys.modules,
            {"aragora.workflow.nodes": None},
        ):
            # Should not raise
            register_openclaw_steps()


# ============================================================================
# Integration-style Tests
# ============================================================================


class TestOpenClawStepIntegration:
    """Integration-style tests combining session and action steps."""

    @pytest.mark.asyncio
    async def test_session_then_action_workflow(self, mock_proxy, mock_context):
        """Simulate creating a session then using it in an action step."""
        # Step 1: Create session
        session_obj = MagicMock()
        session_obj.session_id = "sess-integration"
        mock_proxy.create_session.return_value = session_obj

        session_step = OpenClawSessionStep(
            name="setup",
            config={"operation": "create", "workspace_id": "/workspace"},
        )
        with patch.object(session_step, "_get_proxy", return_value=mock_proxy):
            session_result = await session_step.execute(mock_context)

        assert session_result["success"] is True

        # Step 2: Store output like the engine would
        mock_context.step_outputs["setup"] = session_result

        # Step 3: Execute action using template reference
        action_step = OpenClawActionStep(
            name="run_command",
            config={
                "action_type": "shell",
                "session_id": "{step.setup.session_id}",
                "command": "ls -la",
            },
        )
        with patch.object(action_step, "_get_proxy", return_value=mock_proxy):
            action_result = await action_step.execute(mock_context)

        assert action_result["success"] is True
        call_kwargs = mock_proxy.execute_action.call_args[1]
        assert call_kwargs["session_id"] == "sess-integration"

    @pytest.mark.asyncio
    async def test_file_delete_payload(self, mock_proxy, mock_context):
        """File delete action should send path in payload."""
        step = OpenClawActionStep(
            name="Delete",
            config={
                "action_type": "file_delete",
                "session_id": "sess-1",
                "path": "/tmp/old.log",
            },
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            await step.execute(mock_context)

        call_kwargs = mock_proxy.execute_action.call_args[1]
        assert call_kwargs["input"]["path"] == "/tmp/old.log"
        assert call_kwargs["action_type"] == "file_delete"

    @pytest.mark.asyncio
    async def test_screenshot_payload(self, mock_proxy, mock_context):
        """Screenshot action should send URL in payload."""
        step = OpenClawActionStep(
            name="Screenshot",
            config={
                "action_type": "screenshot",
                "session_id": "sess-1",
                "url": "https://example.com/page",
            },
        )
        with patch.object(step, "_get_proxy", return_value=mock_proxy):
            await step.execute(mock_context)

        call_kwargs = mock_proxy.execute_action.call_args[1]
        assert call_kwargs["input"]["url"] == "https://example.com/page"
        assert call_kwargs["action_type"] == "screenshot"
