"""
Tests for OpenClaw Secure Proxy.

Tests cover:
- Policy engine (rules, evaluation, YAML loading)
- Secure proxy (sessions, actions, approvals)
- Action sandbox (workspace isolation, command filtering)
- Protocol intercept hooks
"""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.gateway.openclaw_policy import (
    ActionRequest,
    ActionType,
    OpenClawPolicy,
    PolicyDecision,
    PolicyRule,
    create_enterprise_policy,
)
from aragora.gateway.openclaw_proxy import (
    OpenClawSecureProxy,
    PendingApproval,
    ProxyActionResult,
    ProxySession,
)
from aragora.gateway.openclaw_sandbox import (
    OpenClawActionSandbox,
    SandboxConfig,
    SandboxSession,
)


# =============================================================================
# Policy Engine Tests
# =============================================================================


class TestPolicyRule:
    """Tests for PolicyRule matching."""

    def test_matches_action_type(self):
        """Should match specified action types."""
        rule = PolicyRule(
            name="test",
            action_types=[ActionType.SHELL, ActionType.FILE_READ],
            decision=PolicyDecision.ALLOW,
        )

        assert rule.matches_action_type(ActionType.SHELL)
        assert rule.matches_action_type(ActionType.FILE_READ)
        assert not rule.matches_action_type(ActionType.BROWSER)

    def test_matches_path_allow_patterns(self):
        """Should match path allow patterns."""
        rule = PolicyRule(
            name="test",
            action_types=[ActionType.FILE_READ],
            decision=PolicyDecision.ALLOW,
            path_patterns=["/workspace/**", "/tmp/*"],
        )

        assert rule.matches_path("/workspace/file.txt")
        assert rule.matches_path("/workspace/subdir/file.py")
        assert rule.matches_path("/tmp/data")
        assert not rule.matches_path("/etc/passwd")

    def test_matches_path_deny_patterns(self):
        """Should reject paths matching deny patterns."""
        rule = PolicyRule(
            name="test",
            action_types=[ActionType.FILE_READ],
            decision=PolicyDecision.ALLOW,
            path_patterns=["**/*"],
            path_deny_patterns=["/etc/**", "/root/**"],
        )

        assert rule.matches_path("/workspace/file.txt")
        assert not rule.matches_path("/etc/passwd")
        assert not rule.matches_path("/root/.bashrc")

    def test_matches_command_patterns(self):
        """Should match command patterns."""
        rule = PolicyRule(
            name="test",
            action_types=[ActionType.SHELL],
            decision=PolicyDecision.ALLOW,
            command_patterns=[r"^ls\s+", r"^cat\s+"],
        )

        assert rule.matches_command("ls -la")
        assert rule.matches_command("cat file.txt")
        assert not rule.matches_command("rm -rf /")

    def test_matches_command_deny_patterns(self):
        """Should reject commands matching deny patterns."""
        rule = PolicyRule(
            name="test",
            action_types=[ActionType.SHELL],
            decision=PolicyDecision.DENY,
            command_deny_patterns=[r"rm\s+-rf\s+/", r"sudo"],
        )

        assert not rule.matches_command("rm -rf /")
        assert not rule.matches_command("sudo apt install")
        assert rule.matches_command("ls -la")

    def test_matches_url_patterns(self):
        """Should match URL patterns."""
        rule = PolicyRule(
            name="test",
            action_types=[ActionType.BROWSER],
            decision=PolicyDecision.ALLOW,
            url_patterns=["https://*", "http://example.com/*"],
        )

        assert rule.matches_url("https://google.com")
        assert rule.matches_url("http://example.com/page")
        assert not rule.matches_url("file:///etc/passwd")


class TestOpenClawPolicy:
    """Tests for OpenClawPolicy engine."""

    def test_default_init(self):
        """Should initialize with no rules."""
        policy = OpenClawPolicy()

        assert len(policy.get_rules()) == 0
        assert policy._default_decision == PolicyDecision.DENY

    def test_load_from_dict(self):
        """Should load policy from dictionary."""
        policy_dict = {
            "version": 1,
            "default_decision": "allow",
            "rules": [
                {
                    "name": "block_etc",
                    "action_types": ["file_read"],
                    "decision": "deny",
                    "priority": 100,
                    "path_patterns": ["/etc/**"],
                }
            ],
        }

        policy = OpenClawPolicy(policy_dict=policy_dict)

        assert len(policy.get_rules()) == 1
        assert policy._default_decision == PolicyDecision.ALLOW

    def test_load_from_yaml_file(self):
        """Should load policy from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
version: 1
default_decision: deny
rules:
  - name: allow_workspace
    action_types:
      - file_read
      - file_write
    decision: allow
    path_patterns:
      - "/workspace/**"
""")
            f.flush()

            policy = OpenClawPolicy(policy_file=f.name)

            assert len(policy.get_rules()) == 1
            rule = policy.get_rule("allow_workspace")
            assert rule is not None
            assert ActionType.FILE_READ in rule.action_types

    def test_evaluate_matching_rule(self):
        """Should return matched rule decision."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "allow_ls",
                        "action_types": ["shell"],
                        "decision": "allow",
                        "command_patterns": [r"^ls\s+"],
                    }
                ],
            }
        )

        request = ActionRequest(
            action_type=ActionType.SHELL,
            user_id="user-1",
            session_id="sess-1",
            command="ls -la",
        )

        result = policy.evaluate(request)

        assert result.decision == PolicyDecision.ALLOW
        assert result.matched_rule is not None
        assert result.matched_rule.name == "allow_ls"

    def test_evaluate_no_match_uses_default(self):
        """Should use default decision when no rule matches."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [],
            }
        )

        request = ActionRequest(
            action_type=ActionType.SHELL,
            user_id="user-1",
            session_id="sess-1",
            command="echo hello",
        )

        result = policy.evaluate(request)

        assert result.decision == PolicyDecision.DENY
        assert result.matched_rule is None

    def test_evaluate_priority_order(self):
        """Should evaluate rules in priority order."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "allow_all_shell",
                        "action_types": ["shell"],
                        "decision": "allow",
                        "priority": 1,
                    },
                    {
                        "name": "block_sudo",
                        "action_types": ["shell"],
                        "decision": "deny",
                        "priority": 100,
                        "command_patterns": [r"^sudo\s+"],
                    },
                ],
            }
        )

        # sudo should be blocked (higher priority)
        request = ActionRequest(
            action_type=ActionType.SHELL,
            user_id="user-1",
            session_id="sess-1",
            command="sudo apt install",
        )
        result = policy.evaluate(request)
        assert result.decision == PolicyDecision.DENY

        # ls should be allowed (lower priority rule)
        request = ActionRequest(
            action_type=ActionType.SHELL,
            user_id="user-1",
            session_id="sess-1",
            command="ls -la",
        )
        result = policy.evaluate(request)
        assert result.decision == PolicyDecision.ALLOW

    def test_evaluate_role_based_rules(self):
        """Should respect role-based rule conditions."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "admin_all_access",
                        "action_types": ["shell"],
                        "decision": "allow",
                        "allowed_roles": ["admin"],
                    },
                ],
            }
        )

        # Admin should have access
        request = ActionRequest(
            action_type=ActionType.SHELL,
            user_id="admin-user",
            session_id="sess-1",
            command="sudo reboot",
            roles=["admin"],
        )
        result = policy.evaluate(request)
        assert result.decision == PolicyDecision.ALLOW

        # Non-admin should be denied
        request = ActionRequest(
            action_type=ActionType.SHELL,
            user_id="regular-user",
            session_id="sess-2",
            command="sudo reboot",
            roles=["user"],
        )
        result = policy.evaluate(request)
        assert result.decision == PolicyDecision.DENY

    def test_evaluate_rate_limiting(self):
        """Should enforce rate limits."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "rate_limited_shell",
                        "action_types": ["shell"],
                        "decision": "allow",
                        "rate_limit": 3,
                        "rate_limit_window": 60,
                    },
                ],
            }
        )

        request = ActionRequest(
            action_type=ActionType.SHELL,
            user_id="user-1",
            session_id="sess-1",
            command="echo hello",
        )

        # First 3 should be allowed
        for _ in range(3):
            result = policy.evaluate(request)
            assert result.decision == PolicyDecision.ALLOW

        # 4th should be denied (rate limited)
        result = policy.evaluate(request)
        assert result.decision == PolicyDecision.DENY
        assert "rate limit" in result.reason.lower()

    def test_create_enterprise_policy(self):
        """Should create valid enterprise policy."""
        policy = create_enterprise_policy()

        assert len(policy.get_rules()) >= 5
        assert policy._default_decision == PolicyDecision.DENY

        # Should block /etc
        request = ActionRequest(
            action_type=ActionType.FILE_READ,
            user_id="user-1",
            session_id="sess-1",
            path="/etc/passwd",
        )
        result = policy.evaluate(request)
        assert result.decision == PolicyDecision.DENY

    def test_add_and_remove_rule(self):
        """Should add and remove rules."""
        policy = OpenClawPolicy()

        rule = PolicyRule(
            name="test_rule",
            action_types=[ActionType.SHELL],
            decision=PolicyDecision.ALLOW,
        )

        policy.add_rule(rule)
        assert policy.get_rule("test_rule") is not None

        policy.remove_rule("test_rule")
        assert policy.get_rule("test_rule") is None

    def test_to_dict_and_save(self):
        """Should export and save policy."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "test_rule",
                        "action_types": ["shell"],
                        "decision": "allow",
                    }
                ],
            }
        )

        exported = policy.to_dict()
        assert exported["version"] == 1
        assert len(exported["rules"]) == 1

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            policy.save_to_file(f.name)
            loaded = OpenClawPolicy(policy_file=f.name)
            assert len(loaded.get_rules()) == 1


# =============================================================================
# Secure Proxy Tests
# =============================================================================


class TestOpenClawSecureProxy:
    """Tests for OpenClawSecureProxy."""

    @pytest.mark.asyncio
    async def test_create_session(self):
        """Should create proxy session."""
        proxy = OpenClawSecureProxy()

        session = await proxy.create_session(
            user_id="user-1",
            tenant_id="tenant-1",
            roles=["developer"],
        )

        assert session.session_id
        assert session.user_id == "user-1"
        assert session.tenant_id == "tenant-1"
        assert "developer" in session.roles

    @pytest.mark.asyncio
    async def test_end_session(self):
        """Should end proxy session."""
        proxy = OpenClawSecureProxy()

        session = await proxy.create_session(user_id="user-1", tenant_id="tenant-1")
        assert proxy.get_session(session.session_id) is not None

        result = await proxy.end_session(session.session_id)
        assert result is True
        assert proxy.get_session(session.session_id) is None

    @pytest.mark.asyncio
    async def test_session_limit_per_user(self):
        """Should enforce max sessions per user."""
        proxy = OpenClawSecureProxy(max_sessions_per_user=2)

        session1 = await proxy.create_session(user_id="user-1", tenant_id="t1")
        session2 = await proxy.create_session(user_id="user-1", tenant_id="t1")
        session3 = await proxy.create_session(user_id="user-1", tenant_id="t1")

        # session1 should have been cleaned up
        assert proxy.get_session(session1.session_id) is None
        assert proxy.get_session(session2.session_id) is not None
        assert proxy.get_session(session3.session_id) is not None

    @pytest.mark.asyncio
    async def test_execute_action_allowed(self):
        """Should allow actions that pass policy."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "allow",
                "rules": [],
            }
        )
        proxy = OpenClawSecureProxy(policy=policy)

        session = await proxy.create_session(user_id="user-1", tenant_id="t1")
        result = await proxy.execute_action(
            session_id=session.session_id,
            action_type="shell",
            command="ls -la",
        )

        assert result.success is True
        assert result.policy_decision == PolicyDecision.ALLOW

    @pytest.mark.asyncio
    async def test_execute_action_denied(self):
        """Should deny actions that fail policy."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [],
            }
        )
        proxy = OpenClawSecureProxy(policy=policy)

        session = await proxy.create_session(user_id="user-1", tenant_id="t1")
        result = await proxy.execute_action(
            session_id=session.session_id,
            action_type="shell",
            command="rm -rf /",
        )

        assert result.success is False
        assert result.policy_decision == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_execute_action_requires_approval(self):
        """Should handle approval workflow."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "approve_sudo",
                        "action_types": ["shell"],
                        "decision": "require_approval",
                        "command_patterns": [r"^sudo\s+"],
                    }
                ],
            }
        )

        approval_requests = []
        proxy = OpenClawSecureProxy(
            policy=policy,
            approval_callback=lambda a: approval_requests.append(a),
        )

        session = await proxy.create_session(user_id="user-1", tenant_id="t1")
        result = await proxy.execute_action(
            session_id=session.session_id,
            action_type="shell",
            command="sudo apt install vim",
        )

        assert result.success is False
        assert result.policy_decision == PolicyDecision.REQUIRE_APPROVAL
        assert result.requires_approval is True
        assert result.approval_id is not None
        assert len(approval_requests) == 1

    @pytest.mark.asyncio
    async def test_approve_action(self):
        """Should execute approved action."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "approve_all",
                        "action_types": ["shell"],
                        "decision": "require_approval",
                    }
                ],
            }
        )
        proxy = OpenClawSecureProxy(policy=policy)

        session = await proxy.create_session(user_id="user-1", tenant_id="t1")
        pending = await proxy.execute_action(
            session_id=session.session_id,
            action_type="shell",
            command="echo hello",
        )

        result = await proxy.approve_action(
            approval_id=pending.approval_id,
            approver_id="admin-1",
        )

        assert result.success is True
        assert result.policy_decision == PolicyDecision.ALLOW

    @pytest.mark.asyncio
    async def test_deny_approval(self):
        """Should handle denied approval."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "approve_all",
                        "action_types": ["shell"],
                        "decision": "require_approval",
                    }
                ],
            }
        )
        proxy = OpenClawSecureProxy(policy=policy)

        session = await proxy.create_session(user_id="user-1", tenant_id="t1")
        pending = await proxy.execute_action(
            session_id=session.session_id,
            action_type="shell",
            command="sudo rm -rf /",
        )

        result = await proxy.deny_approval(
            approval_id=pending.approval_id,
            denier_id="admin-1",
            reason="Too dangerous",
        )

        assert result is True
        assert len(proxy.get_pending_approvals()) == 0

    @pytest.mark.asyncio
    async def test_audit_callback(self):
        """Should emit audit events."""
        audit_events = []
        proxy = OpenClawSecureProxy(
            audit_callback=lambda e: audit_events.append(e),
        )

        session = await proxy.create_session(user_id="user-1", tenant_id="t1")
        await proxy.execute_action(
            session_id=session.session_id,
            action_type="shell",
            command="ls",
        )
        await proxy.end_session(session.session_id)

        event_types = [e["event_type"] for e in audit_events]
        assert "session_created" in event_types
        assert "session_ended" in event_types

    @pytest.mark.asyncio
    async def test_invalid_session(self):
        """Should reject actions for invalid session."""
        proxy = OpenClawSecureProxy()

        result = await proxy.execute_action(
            session_id="invalid-session",
            action_type="shell",
            command="ls",
        )

        assert result.success is False
        assert "session" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Should return proxy statistics."""
        proxy = OpenClawSecureProxy()

        session = await proxy.create_session(user_id="user-1", tenant_id="t1")
        await proxy.execute_action(
            session_id=session.session_id,
            action_type="shell",
            command="ls",
        )

        stats = proxy.get_stats()

        assert stats["sessions_created"] == 1
        assert stats["active_sessions"] == 1
        assert stats["actions_allowed"] >= 0


# =============================================================================
# Action Sandbox Tests
# =============================================================================


class TestOpenClawActionSandbox:
    """Tests for OpenClawActionSandbox."""

    @pytest.mark.asyncio
    async def test_create_sandbox(self):
        """Should create sandbox for session."""
        sandbox = OpenClawActionSandbox()

        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
            tenant_id="tenant-1",
        )

        assert session.sandbox_id
        assert session.session_id == "sess-1"
        assert Path(session.workspace_path).exists()

        # Cleanup
        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_destroy_sandbox(self):
        """Should destroy sandbox and clean up."""
        sandbox = OpenClawActionSandbox()

        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )
        workspace = Path(session.workspace_path)
        assert workspace.exists()

        result = await sandbox.destroy_sandbox(session.sandbox_id)
        assert result is True
        assert not workspace.exists()

    @pytest.mark.asyncio
    async def test_execute_shell_allowed(self):
        """Should execute allowed shell commands."""
        sandbox = OpenClawActionSandbox()

        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.execute_shell(
            sandbox_id=session.sandbox_id,
            command="echo hello",
        )

        assert result.success is True
        assert "hello" in result.output.get("stdout", "")

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_execute_shell_blocked(self):
        """Should block dangerous commands."""
        sandbox = OpenClawActionSandbox()

        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.execute_shell(
            sandbox_id=session.sandbox_id,
            command="sudo apt install malware",
        )

        assert result.success is False
        assert "blocked" in result.error.lower() or "sudo" in result.error.lower()

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_read_file_in_workspace(self):
        """Should read files in workspace."""
        sandbox = OpenClawActionSandbox()

        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Create a test file
        test_file = Path(session.workspace_path) / "test.txt"
        test_file.write_text("hello world")

        result = await sandbox.read_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/test.txt",
        )

        assert result.success is True
        assert result.output == "hello world"

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_write_file_in_workspace(self):
        """Should write files in workspace."""
        sandbox = OpenClawActionSandbox()

        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        result = await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/output.txt",
            content="test content",
        )

        assert result.success is True

        # Verify file was created
        output_file = Path(session.workspace_path) / "output.txt"
        assert output_file.exists()
        assert output_file.read_text() == "test content"

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_write_file_size_limit(self):
        """Should enforce file size limits."""
        config = SandboxConfig(max_file_size_mb=1)
        sandbox = OpenClawActionSandbox(default_config=config)

        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Try to write 2MB file
        large_content = "x" * (2 * 1024 * 1024)
        result = await sandbox.write_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/large.txt",
            content=large_content,
        )

        assert result.success is False
        assert "size" in result.error.lower()

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_delete_file_in_workspace(self):
        """Should delete files in workspace."""
        sandbox = OpenClawActionSandbox()

        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Create a test file
        test_file = Path(session.workspace_path) / "to_delete.txt"
        test_file.write_text("delete me")

        result = await sandbox.delete_file(
            sandbox_id=session.sandbox_id,
            path="/workspace/to_delete.txt",
        )

        assert result.success is True
        assert not test_file.exists()

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self):
        """Should block path traversal attacks."""
        sandbox = OpenClawActionSandbox()

        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        # Try to read /etc/passwd via traversal
        result = await sandbox.read_file(
            sandbox_id=session.sandbox_id,
            path="../../../etc/passwd",
        )

        # Should either fail or not return /etc/passwd content
        if result.success:
            assert "root:" not in str(result.output)

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_get_sandbox_for_session(self):
        """Should retrieve sandbox by session ID."""
        sandbox = OpenClawActionSandbox()

        session = await sandbox.create_sandbox(
            session_id="sess-1",
            user_id="user-1",
        )

        found = sandbox.get_sandbox_for_session("sess-1")
        assert found is not None
        assert found.sandbox_id == session.sandbox_id

        await sandbox.destroy_sandbox(session.sandbox_id)

    @pytest.mark.asyncio
    async def test_cleanup_all(self):
        """Should clean up all sandboxes."""
        sandbox = OpenClawActionSandbox()

        await sandbox.create_sandbox(session_id="sess-1", user_id="user-1")
        await sandbox.create_sandbox(session_id="sess-2", user_id="user-2")

        count = await sandbox.cleanup_all()
        assert count == 2
        assert sandbox.get_stats()["active_sandboxes"] == 0

    def test_get_stats(self):
        """Should return sandbox statistics."""
        sandbox = OpenClawActionSandbox()
        stats = sandbox.get_stats()

        assert "sandboxes_created" in stats
        assert "commands_executed" in stats
        assert "files_read" in stats


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for proxy + sandbox."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Should complete full proxy + sandbox workflow."""
        # Create components
        policy = create_enterprise_policy()
        sandbox = OpenClawActionSandbox()
        proxy = OpenClawSecureProxy(policy=policy)

        # Create session
        session = await proxy.create_session(
            user_id="user-1",
            tenant_id="tenant-1",
            workspace_id="ws-1",
            roles=["developer"],
        )

        # Create sandbox
        sandbox_session = await sandbox.create_sandbox(
            session_id=session.session_id,
            user_id=session.user_id,
            tenant_id=session.tenant_id,
        )

        # Execute safe command
        result = await sandbox.execute_shell(
            sandbox_id=sandbox_session.sandbox_id,
            command="pwd",
        )
        assert result.success is True

        # Try dangerous command (should be blocked by policy)
        policy_result = proxy._policy.evaluate(
            ActionRequest(
                action_type=ActionType.SHELL,
                user_id=session.user_id,
                session_id=session.session_id,
                command="rm -rf /",
            )
        )
        assert policy_result.decision == PolicyDecision.DENY

        # Cleanup
        await sandbox.destroy_sandbox(sandbox_session.sandbox_id)
        await proxy.end_session(session.session_id)
