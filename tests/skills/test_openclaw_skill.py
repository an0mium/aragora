"""
Tests for OpenClaw Skill wrapper.

Tests cover:
- Manifest declaration
- Action routing (shell, file_read, file_write, file_delete, browser)
- Session management per context
- Error handling (missing action, invalid action, missing params)
- Proxy result conversion
- Approval workflow handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from aragora.skills.builtin.openclaw_skill import OpenClawSkill, _ACTION_TYPE_MAP
from aragora.skills.base import SkillCapability, SkillContext, SkillManifest, SkillResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_proxy_result(
    success=True,
    action_id="act-1",
    requires_approval=False,
    result=None,
    error=None,
    policy_decision_value="allow",
    execution_time_ms=10.0,
    audit_id="audit-1",
    approval_id=None,
):
    """Build a mock ProxyActionResult."""
    pr = MagicMock()
    pr.success = success
    pr.action_id = action_id
    pr.requires_approval = requires_approval
    pr.result = result if result is not None else {"output": "ok"}
    pr.error = error
    pr.policy_decision = MagicMock()
    pr.policy_decision.value = policy_decision_value
    pr.execution_time_ms = execution_time_ms
    pr.audit_id = audit_id
    pr.approval_id = (
        approval_id if approval_id is not None else ("appr-1" if requires_approval else None)
    )
    return pr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_proxy():
    proxy = AsyncMock()
    session = MagicMock()
    session.session_id = "sess-123"
    proxy.create_session = AsyncMock(return_value=session)
    proxy.execute_action = AsyncMock(return_value=make_proxy_result())
    return proxy


@pytest.fixture
def skill(mock_proxy):
    return OpenClawSkill(proxy=mock_proxy)


@pytest.fixture
def context():
    return SkillContext(user_id="user-1", tenant_id="tenant-1")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_manifest_name_and_capabilities(self, skill):
        m = skill.manifest
        assert isinstance(m, SkillManifest)
        assert m.name == "openclaw"
        expected_caps = {
            SkillCapability.SHELL_EXECUTION,
            SkillCapability.READ_LOCAL,
            SkillCapability.WRITE_LOCAL,
            SkillCapability.WEB_FETCH,
            SkillCapability.CODE_EXECUTION,
        }
        assert set(m.capabilities) == expected_caps


# ---------------------------------------------------------------------------
# Action routing -- success paths
# ---------------------------------------------------------------------------


class TestActionSuccess:
    @pytest.mark.asyncio
    async def test_execute_shell_success(self, skill, mock_proxy, context):
        mock_proxy.execute_action.return_value = make_proxy_result(
            result={"stdout": "hello"},
        )

        result = await skill.execute({"action": "shell", "command": "echo hello"}, context)

        assert result.success
        assert result.data["action"] == "shell"
        assert result.data["result"] == {"stdout": "hello"}
        mock_proxy.execute_action.assert_awaited_once_with(
            session_id="sess-123",
            action_type="shell",
            command="echo hello",
        )

    @pytest.mark.asyncio
    async def test_execute_file_read_success(self, skill, mock_proxy, context):
        mock_proxy.execute_action.return_value = make_proxy_result(
            result={"content": "file body"},
        )

        result = await skill.execute(
            {"action": "file_read", "path": "/workspace/README.md"}, context
        )

        assert result.success
        assert result.data["action"] == "file_read"
        mock_proxy.execute_action.assert_awaited_once_with(
            session_id="sess-123",
            action_type="file_read",
            path="/workspace/README.md",
        )

    @pytest.mark.asyncio
    async def test_execute_file_write_success(self, skill, mock_proxy, context):
        mock_proxy.execute_action.return_value = make_proxy_result(
            result={"bytes_written": 5},
        )

        result = await skill.execute(
            {"action": "file_write", "path": "/workspace/out.txt", "content": "hello"},
            context,
        )

        assert result.success
        assert result.data["action"] == "file_write"
        mock_proxy.execute_action.assert_awaited_once_with(
            session_id="sess-123",
            action_type="file_write",
            path="/workspace/out.txt",
            content="hello",
        )

    @pytest.mark.asyncio
    async def test_execute_file_delete_success(self, skill, mock_proxy, context):
        mock_proxy.execute_action.return_value = make_proxy_result(
            result={"deleted": True},
        )

        result = await skill.execute(
            {"action": "file_delete", "path": "/workspace/tmp.txt"},
            context,
        )

        assert result.success
        assert result.data["action"] == "file_delete"
        mock_proxy.execute_action.assert_awaited_once_with(
            session_id="sess-123",
            action_type="file_delete",
            path="/workspace/tmp.txt",
        )

    @pytest.mark.asyncio
    async def test_execute_browser_success(self, skill, mock_proxy, context):
        mock_proxy.execute_action.return_value = make_proxy_result(
            result={"title": "Example Domain"},
        )

        result = await skill.execute(
            {"action": "browser", "url": "https://example.com"},
            context,
        )

        assert result.success
        assert result.data["action"] == "browser"
        mock_proxy.execute_action.assert_awaited_once_with(
            session_id="sess-123",
            action_type="browser",
            url="https://example.com",
        )

    @pytest.mark.asyncio
    async def test_execute_screenshot_routed_as_browser(self, skill, mock_proxy, context):
        """The 'screenshot' action is handled by _execute_browser as well."""
        mock_proxy.execute_action.return_value = make_proxy_result(
            result={"screenshot": "base64..."},
        )

        result = await skill.execute(
            {"action": "screenshot", "url": "https://example.com"},
            context,
        )

        assert result.success
        assert result.data["action"] == "screenshot"
        mock_proxy.execute_action.assert_awaited_once_with(
            session_id="sess-123",
            action_type="screenshot",
            url="https://example.com",
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_execute_missing_action(self, skill, context):
        result = await skill.execute({}, context)

        assert not result.success
        assert result.error_code == "missing_action"
        assert "Action is required" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_invalid_action(self, skill, context):
        result = await skill.execute({"action": "hack_the_planet"}, context)

        assert not result.success
        assert result.error_code == "invalid_action"
        assert "hack_the_planet" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_shell_missing_command(self, skill, mock_proxy, context):
        result = await skill.execute({"action": "shell"}, context)

        assert not result.success
        assert result.error_code == "missing_command"

    @pytest.mark.asyncio
    async def test_execute_file_read_missing_path(self, skill, mock_proxy, context):
        result = await skill.execute({"action": "file_read"}, context)

        assert not result.success
        assert result.error_code == "missing_path"

    @pytest.mark.asyncio
    async def test_execute_file_write_missing_path(self, skill, mock_proxy, context):
        result = await skill.execute({"action": "file_write", "content": "abc"}, context)

        assert not result.success
        assert result.error_code == "missing_path"

    @pytest.mark.asyncio
    async def test_execute_file_delete_missing_path(self, skill, mock_proxy, context):
        result = await skill.execute({"action": "file_delete"}, context)

        assert not result.success
        assert result.error_code == "missing_path"

    @pytest.mark.asyncio
    async def test_execute_browser_missing_url(self, skill, mock_proxy, context):
        result = await skill.execute({"action": "browser"}, context)

        assert not result.success
        assert result.error_code == "missing_url"

    @pytest.mark.asyncio
    async def test_execute_handles_proxy_exception(self, skill, mock_proxy, context):
        mock_proxy.execute_action.side_effect = RuntimeError("connection lost")

        result = await skill.execute({"action": "shell", "command": "ls"}, context)

        assert not result.success
        assert "connection lost" in result.error_message


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


class TestSessionManagement:
    @pytest.mark.asyncio
    async def test_session_created_per_context(self, skill, mock_proxy):
        ctx_a = SkillContext(user_id="alice", tenant_id="t1")
        ctx_b = SkillContext(user_id="bob", tenant_id="t1")

        await skill.execute({"action": "shell", "command": "ls"}, ctx_a)
        await skill.execute({"action": "shell", "command": "ls"}, ctx_b)

        assert mock_proxy.create_session.await_count == 2

    @pytest.mark.asyncio
    async def test_session_reused_same_context(self, skill, mock_proxy, context):
        await skill.execute({"action": "shell", "command": "ls"}, context)
        await skill.execute({"action": "shell", "command": "pwd"}, context)

        # Session should only be created once for the same user/tenant
        mock_proxy.create_session.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_session_key_includes_tenant(self, skill, mock_proxy):
        ctx_a = SkillContext(user_id="alice", tenant_id="t1")
        ctx_b = SkillContext(user_id="alice", tenant_id="t2")

        await skill.execute({"action": "shell", "command": "ls"}, ctx_a)
        await skill.execute({"action": "shell", "command": "ls"}, ctx_b)

        # Different tenants => different sessions
        assert mock_proxy.create_session.await_count == 2


# ---------------------------------------------------------------------------
# Proxy result conversion
# ---------------------------------------------------------------------------


class TestProxyResultConversion:
    @pytest.mark.asyncio
    async def test_to_skill_result_success_metadata(self, skill, mock_proxy, context):
        mock_proxy.execute_action.return_value = make_proxy_result(
            action_id="act-42",
            execution_time_ms=123.4,
            audit_id="audit-99",
            result={"stdout": "data"},
        )

        result = await skill.execute({"action": "shell", "command": "ls"}, context)

        assert result.success
        assert result.data["action_id"] == "act-42"
        assert result.data["execution_time_ms"] == 123.4
        assert result.data["audit_id"] == "audit-99"
        assert result.metadata.get("provider") == "openclaw"

    @pytest.mark.asyncio
    async def test_to_skill_result_denied(self, skill, mock_proxy, context):
        mock_proxy.execute_action.return_value = make_proxy_result(
            success=False,
            error=None,
            policy_decision_value="deny",
        )

        result = await skill.execute({"action": "shell", "command": "rm -rf /"}, context)

        assert not result.success
        assert result.error_code == "action_denied"
        assert "deny" in result.error_message

    @pytest.mark.asyncio
    async def test_to_skill_result_denied_with_error(self, skill, mock_proxy, context):
        mock_proxy.execute_action.return_value = make_proxy_result(
            success=False,
            error="Dangerous command blocked",
            policy_decision_value="deny",
        )

        result = await skill.execute({"action": "shell", "command": "rm -rf /"}, context)

        assert not result.success
        assert result.error_message == "Dangerous command blocked"

    @pytest.mark.asyncio
    async def test_to_skill_result_requires_approval(self, skill, mock_proxy, context):
        mock_proxy.execute_action.return_value = make_proxy_result(
            success=True,
            requires_approval=True,
            approval_id="appr-77",
        )

        result = await skill.execute({"action": "shell", "command": "sudo reboot"}, context)

        assert not result.success
        assert result.error_code == "requires_approval"
        assert "appr-77" in result.error_message
