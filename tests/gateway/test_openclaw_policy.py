"""
Tests for OpenClaw Policy Engine.

Tests policy-based access control for OpenClaw actions including:
- Shell commands, file access, browser operations, and API calls
- Path glob matching and command pattern filtering
- Role-based policy overrides via RBAC integration
- Approval workflows for sensitive operations
- Policy precedence (deny > require_approval > allow)
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from aragora.gateway.openclaw_policy import (
    ActionRequest,
    ActionType,
    OpenClawPolicy,
    PolicyDecision,
    PolicyEvaluationResult,
    PolicyRule,
    create_enterprise_policy,
)


class TestActionType:
    """Tests for ActionType enum."""

    def test_action_type_values(self) -> None:
        """Test action type enum values."""
        assert ActionType.SHELL.value == "shell"
        assert ActionType.FILE_READ.value == "file_read"
        assert ActionType.FILE_WRITE.value == "file_write"
        assert ActionType.FILE_DELETE.value == "file_delete"
        assert ActionType.BROWSER.value == "browser"
        assert ActionType.API.value == "api"
        assert ActionType.SCREENSHOT.value == "screenshot"
        assert ActionType.KEYBOARD.value == "keyboard"
        assert ActionType.MOUSE.value == "mouse"


class TestPolicyDecision:
    """Tests for PolicyDecision enum."""

    def test_policy_decision_values(self) -> None:
        """Test policy decision enum values."""
        assert PolicyDecision.ALLOW.value == "allow"
        assert PolicyDecision.DENY.value == "deny"
        assert PolicyDecision.REQUIRE_APPROVAL.value == "require_approval"


class TestPolicyRule:
    """Tests for PolicyRule dataclass."""

    def test_default_rule(self) -> None:
        """Test rule creation with defaults."""
        rule = PolicyRule(
            name="test_rule",
            action_types=[ActionType.SHELL],
            decision=PolicyDecision.ALLOW,
        )

        assert rule.name == "test_rule"
        assert ActionType.SHELL in rule.action_types
        assert rule.decision == PolicyDecision.ALLOW
        assert rule.priority == 0
        assert rule.path_patterns == []
        assert rule.command_patterns == []
        assert rule.workspace_only is False

    def test_rule_with_patterns(self) -> None:
        """Test rule creation with patterns."""
        rule = PolicyRule(
            name="file_rule",
            action_types=[ActionType.FILE_READ, ActionType.FILE_WRITE],
            decision=PolicyDecision.ALLOW,
            priority=50,
            path_patterns=["/tmp/*", "/workspace/**"],
            path_deny_patterns=["/etc/*", "/sys/*"],
        )

        assert len(rule.action_types) == 2
        assert rule.priority == 50
        assert "/tmp/*" in rule.path_patterns
        assert "/etc/*" in rule.path_deny_patterns

    def test_matches_action_type(self) -> None:
        """Test action type matching."""
        rule = PolicyRule(
            name="shell_rule",
            action_types=[ActionType.SHELL, ActionType.FILE_READ],
            decision=PolicyDecision.ALLOW,
        )

        assert rule.matches_action_type(ActionType.SHELL) is True
        assert rule.matches_action_type(ActionType.FILE_READ) is True
        assert rule.matches_action_type(ActionType.FILE_WRITE) is False
        assert rule.matches_action_type(ActionType.BROWSER) is False


class TestPolicyRulePathMatching:
    """Tests for PolicyRule path pattern matching."""

    def test_path_matches_allow_pattern(self) -> None:
        """Test path matching against allow patterns."""
        rule = PolicyRule(
            name="tmp_rule",
            action_types=[ActionType.FILE_READ],
            decision=PolicyDecision.ALLOW,
            path_patterns=["/tmp/*"],
        )

        assert rule.matches_path("/tmp/test.txt") is True
        assert rule.matches_path("/tmp/subdir") is True
        assert rule.matches_path("/home/user/file.txt") is False

    def test_path_matches_glob_pattern(self) -> None:
        """Test path glob matching with ** patterns."""
        rule = PolicyRule(
            name="workspace_rule",
            action_types=[ActionType.FILE_READ],
            decision=PolicyDecision.ALLOW,
            path_patterns=["/workspace/**"],
        )

        assert rule.matches_path("/workspace/file.txt") is True
        assert rule.matches_path("/workspace/deep/nested/file.py") is True
        assert rule.matches_path("/other/file.txt") is False

    def test_path_deny_patterns_take_precedence(self) -> None:
        """Test that deny patterns are checked first."""
        rule = PolicyRule(
            name="mixed_rule",
            action_types=[ActionType.FILE_READ],
            decision=PolicyDecision.ALLOW,
            path_patterns=["/home/*"],
            path_deny_patterns=["/home/secret/*"],
        )

        assert rule.matches_path("/home/user/file.txt") is True
        assert rule.matches_path("/home/secret/password.txt") is False

    def test_path_no_patterns_allows_all(self) -> None:
        """Test that no patterns matches any path."""
        rule = PolicyRule(
            name="any_rule",
            action_types=[ActionType.FILE_READ],
            decision=PolicyDecision.ALLOW,
        )

        assert rule.matches_path("/any/path/here.txt") is True
        assert rule.matches_path("/etc/passwd") is True
        assert rule.matches_path(None) is True

    def test_path_null_with_patterns(self) -> None:
        """Test null path behavior with patterns."""
        rule = PolicyRule(
            name="pattern_rule",
            action_types=[ActionType.FILE_READ],
            decision=PolicyDecision.ALLOW,
            path_patterns=["/tmp/*"],
        )

        # None path doesn't match when patterns are required
        assert rule.matches_path(None) is False


class TestPolicyRuleCommandMatching:
    """Tests for PolicyRule command pattern matching."""

    def test_command_matches_allow_pattern(self) -> None:
        """Test command matching against allow patterns."""
        rule = PolicyRule(
            name="ls_rule",
            action_types=[ActionType.SHELL],
            decision=PolicyDecision.ALLOW,
            command_patterns=[r"^ls\s+"],
        )

        assert rule.matches_command("ls -la") is True
        assert rule.matches_command("ls /tmp") is True
        assert rule.matches_command("rm -rf /") is False

    def test_command_deny_patterns_take_precedence(self) -> None:
        """Test that command deny patterns are checked first."""
        rule = PolicyRule(
            name="shell_rule",
            action_types=[ActionType.SHELL],
            decision=PolicyDecision.ALLOW,
            command_patterns=[r".*"],
            command_deny_patterns=[r"rm\s+-rf"],
        )

        assert rule.matches_command("ls -la") is True
        assert rule.matches_command("rm -rf /") is False

    def test_command_regex_patterns(self) -> None:
        """Test regex patterns for commands."""
        rule = PolicyRule(
            name="dev_commands",
            action_types=[ActionType.SHELL],
            decision=PolicyDecision.ALLOW,
            command_patterns=[r"^(python|node|npm|git)\s+"],
        )

        assert rule.matches_command("python script.py") is True
        assert rule.matches_command("node app.js") is True
        assert rule.matches_command("npm install") is True
        assert rule.matches_command("git status") is True
        assert rule.matches_command("sudo rm -rf /") is False

    def test_command_null_with_patterns(self) -> None:
        """Test null command behavior with patterns."""
        rule = PolicyRule(
            name="pattern_rule",
            action_types=[ActionType.SHELL],
            decision=PolicyDecision.ALLOW,
            command_patterns=[r"^ls"],
        )

        # None command doesn't match when patterns are required
        assert rule.matches_command(None) is False


class TestPolicyRuleUrlMatching:
    """Tests for PolicyRule URL pattern matching."""

    def test_url_matches_allow_pattern(self) -> None:
        """Test URL matching against allow patterns."""
        rule = PolicyRule(
            name="api_rule",
            action_types=[ActionType.API],
            decision=PolicyDecision.ALLOW,
            url_patterns=["https://api.example.com/*"],
        )

        assert rule.matches_url("https://api.example.com/v1/users") is True
        assert rule.matches_url("https://malicious.com/data") is False

    def test_url_deny_patterns_take_precedence(self) -> None:
        """Test that URL deny patterns are checked first."""
        rule = PolicyRule(
            name="browser_rule",
            action_types=[ActionType.BROWSER],
            decision=PolicyDecision.ALLOW,
            url_patterns=["https://*"],
            url_deny_patterns=[r"localhost", r"127\.0\.0\.1"],
        )

        assert rule.matches_url("https://example.com") is True
        assert rule.matches_url("https://localhost:8080") is False
        assert rule.matches_url("http://127.0.0.1:3000") is False


class TestActionRequest:
    """Tests for ActionRequest dataclass."""

    def test_basic_request(self) -> None:
        """Test basic action request creation."""
        request = ActionRequest(
            action_type=ActionType.SHELL,
            user_id="user-123",
            session_id="session-456",
            command="ls -la",
        )

        assert request.action_type == ActionType.SHELL
        assert request.user_id == "user-123"
        assert request.command == "ls -la"
        assert request.workspace_id == "default"
        assert request.roles == []

    def test_request_with_roles(self) -> None:
        """Test action request with roles."""
        request = ActionRequest(
            action_type=ActionType.FILE_WRITE,
            user_id="user-123",
            session_id="session-456",
            path="/workspace/file.txt",
            roles=["admin", "developer"],
        )

        assert "admin" in request.roles
        assert "developer" in request.roles

    def test_request_with_tenant(self) -> None:
        """Test action request with tenant context."""
        request = ActionRequest(
            action_type=ActionType.API,
            user_id="user-123",
            session_id="session-456",
            url="https://api.example.com/data",
            tenant_id="tenant-789",
        )

        assert request.tenant_id == "tenant-789"


class TestOpenClawPolicy:
    """Tests for OpenClawPolicy engine."""

    def test_empty_policy_denies_by_default(self) -> None:
        """Test that empty policy denies actions by default."""
        policy = OpenClawPolicy()

        request = ActionRequest(
            action_type=ActionType.SHELL,
            user_id="user-123",
            session_id="session-456",
            command="ls -la",
        )

        result = policy.evaluate(request)

        assert result.decision == PolicyDecision.DENY
        assert result.matched_rule is None
        assert "default decision" in result.reason.lower()

    def test_empty_policy_with_allow_default(self) -> None:
        """Test empty policy with allow as default decision."""
        policy = OpenClawPolicy(default_decision=PolicyDecision.ALLOW)

        request = ActionRequest(
            action_type=ActionType.SHELL,
            user_id="user-123",
            session_id="session-456",
            command="ls -la",
        )

        result = policy.evaluate(request)

        assert result.decision == PolicyDecision.ALLOW
        assert result.matched_rule is None

    def test_allow_action_matching_allowlist(self) -> None:
        """Test allowing action that matches allowlist."""
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
            user_id="user-123",
            session_id="session-456",
            command="ls -la /tmp",
        )

        result = policy.evaluate(request)

        assert result.decision == PolicyDecision.ALLOW
        assert result.matched_rule is not None
        assert result.matched_rule.name == "allow_ls"

    def test_deny_action_matching_denylist(self) -> None:
        """Test denying action that matches denylist."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "allow",
                "rules": [
                    {
                        "name": "block_rm_rf",
                        "action_types": ["shell"],
                        "decision": "deny",
                        "priority": 100,
                        "command_patterns": [r"rm\s+-rf"],
                    }
                ],
            }
        )

        request = ActionRequest(
            action_type=ActionType.SHELL,
            user_id="user-123",
            session_id="session-456",
            command="rm -rf /important",
        )

        result = policy.evaluate(request)

        assert result.decision == PolicyDecision.DENY
        assert result.matched_rule.name == "block_rm_rf"

    def test_require_approval_for_sensitive_operations(self) -> None:
        """Test require_approval decision for sensitive operations."""
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

        request = ActionRequest(
            action_type=ActionType.SHELL,
            user_id="user-123",
            session_id="session-456",
            command="sudo apt-get update",
        )

        result = policy.evaluate(request)

        assert result.decision == PolicyDecision.REQUIRE_APPROVAL
        assert result.matched_rule.name == "approve_sudo"
        assert result.approval_workflow == "default"


class TestPolicyPrecedence:
    """Tests for policy precedence rules."""

    def test_higher_priority_rule_wins(self) -> None:
        """Test that higher priority rules are evaluated first."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "low_priority_allow",
                        "action_types": ["shell"],
                        "decision": "allow",
                        "priority": 10,
                        "command_patterns": [r".*"],
                    },
                    {
                        "name": "high_priority_deny",
                        "action_types": ["shell"],
                        "decision": "deny",
                        "priority": 100,
                        "command_patterns": [r"rm\s+-rf"],
                    },
                ],
            }
        )

        # rm -rf should be denied by high priority rule
        request = ActionRequest(
            action_type=ActionType.SHELL,
            user_id="user-123",
            session_id="session-456",
            command="rm -rf /tmp/test",
        )

        result = policy.evaluate(request)

        assert result.decision == PolicyDecision.DENY
        assert result.matched_rule.name == "high_priority_deny"

    def test_deny_takes_precedence_via_priority(self) -> None:
        """Test that deny rules can take precedence over allow via priority."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "allow",
                "rules": [
                    {
                        "name": "block_etc",
                        "action_types": ["file_read"],
                        "decision": "deny",
                        "priority": 100,
                        "path_patterns": ["/etc/*"],
                    },
                    {
                        "name": "allow_all_reads",
                        "action_types": ["file_read"],
                        "decision": "allow",
                        "priority": 10,
                    },
                ],
            }
        )

        request = ActionRequest(
            action_type=ActionType.FILE_READ,
            user_id="user-123",
            session_id="session-456",
            path="/etc/passwd",
        )

        result = policy.evaluate(request)

        assert result.decision == PolicyDecision.DENY
        assert result.matched_rule.name == "block_etc"

    def test_require_approval_between_allow_and_deny(self) -> None:
        """Test require_approval with appropriate priority."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "allow_safe_commands",
                        "action_types": ["shell"],
                        "decision": "allow",
                        "priority": 10,
                        "command_patterns": [r"^(ls|cat|echo)\s+"],
                    },
                    {
                        "name": "approve_elevated",
                        "action_types": ["shell"],
                        "decision": "require_approval",
                        "priority": 50,
                        "command_patterns": [r"^sudo\s+"],
                    },
                    {
                        "name": "block_dangerous",
                        "action_types": ["shell"],
                        "decision": "deny",
                        "priority": 100,
                        "command_patterns": [r"rm\s+-rf\s+/"],
                    },
                ],
            }
        )

        # Safe command - allowed
        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.SHELL,
                user_id="user-123",
                session_id="session-456",
                command="ls -la",
            )
        )
        assert result.decision == PolicyDecision.ALLOW

        # Elevated command - require approval
        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.SHELL,
                user_id="user-123",
                session_id="session-456",
                command="sudo apt-get install vim",
            )
        )
        assert result.decision == PolicyDecision.REQUIRE_APPROVAL

        # Dangerous command - denied
        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.SHELL,
                user_id="user-123",
                session_id="session-456",
                command="rm -rf /",
            )
        )
        assert result.decision == PolicyDecision.DENY


class TestRoleBasedPolicyOverride:
    """Tests for role-based policy overrides."""

    def test_allowed_roles_grants_access(self) -> None:
        """Test that allowed_roles grants access to specific roles."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "admin_shell_access",
                        "action_types": ["shell"],
                        "decision": "allow",
                        "allowed_roles": ["admin", "superuser"],
                    }
                ],
            }
        )

        # Admin can execute
        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.SHELL,
                user_id="user-123",
                session_id="session-456",
                command="any_command",
                roles=["admin"],
            )
        )
        assert result.decision == PolicyDecision.ALLOW

        # Non-admin is denied (rule doesn't match)
        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.SHELL,
                user_id="user-456",
                session_id="session-789",
                command="any_command",
                roles=["user"],
            )
        )
        assert result.decision == PolicyDecision.DENY

    def test_denied_roles_blocks_access(self) -> None:
        """Test that denied_roles blocks specific roles."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "allow_most_users",
                        "action_types": ["file_read"],
                        "decision": "allow",
                        "denied_roles": ["restricted", "guest"],
                    }
                ],
            }
        )

        # Regular user can access
        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.FILE_READ,
                user_id="user-123",
                session_id="session-456",
                path="/workspace/file.txt",
                roles=["user"],
            )
        )
        assert result.decision == PolicyDecision.ALLOW

        # Restricted user is denied (rule skipped)
        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.FILE_READ,
                user_id="user-456",
                session_id="session-789",
                path="/workspace/file.txt",
                roles=["restricted"],
            )
        )
        assert result.decision == PolicyDecision.DENY

    def test_role_override_with_rbac_checker(self) -> None:
        """Test RBAC checker integration."""
        mock_rbac = MagicMock()

        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "admin_rule",
                        "action_types": ["shell"],
                        "decision": "allow",
                        "allowed_roles": ["admin"],
                    }
                ],
            },
            rbac_checker=mock_rbac,
        )

        # RBAC checker is stored
        assert policy._rbac_checker is mock_rbac


class TestPathGlobMatching:
    """Tests for path glob pattern matching in file policies."""

    def test_tmp_wildcard_allows(self) -> None:
        """Test /tmp/* allows temp files."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "allow_tmp",
                        "action_types": ["file_read", "file_write"],
                        "decision": "allow",
                        "path_patterns": ["/tmp/*"],
                    }
                ],
            }
        )

        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.FILE_WRITE,
                user_id="user-123",
                session_id="session-456",
                path="/tmp/test.txt",
            )
        )
        assert result.decision == PolicyDecision.ALLOW

    def test_etc_wildcard_denies(self) -> None:
        """Test /etc/* denies system files."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "allow",
                "rules": [
                    {
                        "name": "block_etc",
                        "action_types": ["file_read", "file_write", "file_delete"],
                        "decision": "deny",
                        "priority": 100,
                        "path_patterns": ["/etc/*"],
                    }
                ],
            }
        )

        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.FILE_READ,
                user_id="user-123",
                session_id="session-456",
                path="/etc/passwd",
            )
        )
        assert result.decision == PolicyDecision.DENY

    def test_double_star_glob_for_recursive(self) -> None:
        """Test ** pattern for recursive matching."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "allow_workspace",
                        "action_types": ["file_read", "file_write"],
                        "decision": "allow",
                        "path_patterns": ["/workspace/**"],
                    }
                ],
            }
        )

        # Direct child
        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.FILE_READ,
                user_id="user-123",
                session_id="session-456",
                path="/workspace/file.txt",
            )
        )
        assert result.decision == PolicyDecision.ALLOW

        # Deeply nested
        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.FILE_READ,
                user_id="user-123",
                session_id="session-456",
                path="/workspace/deep/nested/dir/file.py",
            )
        )
        assert result.decision == PolicyDecision.ALLOW

    def test_multiple_path_patterns(self) -> None:
        """Test multiple path patterns in one rule."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "allow_safe_dirs",
                        "action_types": ["file_read"],
                        "decision": "allow",
                        "path_patterns": ["/tmp/*", "/var/tmp/*", "/home/*/projects/*"],
                    }
                ],
            }
        )

        assert (
            policy.evaluate(
                ActionRequest(
                    action_type=ActionType.FILE_READ,
                    user_id="user-123",
                    session_id="session-456",
                    path="/tmp/file.txt",
                )
            ).decision
            == PolicyDecision.ALLOW
        )

        assert (
            policy.evaluate(
                ActionRequest(
                    action_type=ActionType.FILE_READ,
                    user_id="user-123",
                    session_id="session-456",
                    path="/var/tmp/cache.dat",
                )
            ).decision
            == PolicyDecision.ALLOW
        )

        assert (
            policy.evaluate(
                ActionRequest(
                    action_type=ActionType.FILE_READ,
                    user_id="user-123",
                    session_id="session-456",
                    path="/home/john/projects/app.py",
                )
            ).decision
            == PolicyDecision.ALLOW
        )


class TestShellCommandFiltering:
    """Tests for shell command filtering."""

    def test_allowed_commands(self) -> None:
        """Test allowed shell commands pass."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "allow_dev_commands",
                        "action_types": ["shell"],
                        "decision": "allow",
                        "command_patterns": [
                            r"^ls\s+",
                            r"^cat\s+",
                            r"^python\s+",
                            r"^git\s+",
                        ],
                    }
                ],
            }
        )

        allowed_commands = ["ls -la", "cat file.txt", "python script.py", "git status"]

        for cmd in allowed_commands:
            result = policy.evaluate(
                ActionRequest(
                    action_type=ActionType.SHELL,
                    user_id="user-123",
                    session_id="session-456",
                    command=cmd,
                )
            )
            assert result.decision == PolicyDecision.ALLOW, f"Expected ALLOW for: {cmd}"

    def test_denied_commands(self) -> None:
        """Test denied shell commands are blocked."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "allow",
                "rules": [
                    {
                        "name": "block_dangerous",
                        "action_types": ["shell"],
                        "decision": "deny",
                        "priority": 100,
                        "command_patterns": [
                            r"rm\s+-rf\s+/",
                            r"^sudo\s+rm",
                            r"mkfs\.",
                            r"dd\s+if=.*of=/dev/",
                        ],
                    }
                ],
            }
        )

        blocked_commands = [
            "rm -rf /",
            "rm -rf /home",
            "sudo rm -rf /var",
            "mkfs.ext4 /dev/sda1",
            "dd if=/dev/zero of=/dev/sda",
        ]

        for cmd in blocked_commands:
            result = policy.evaluate(
                ActionRequest(
                    action_type=ActionType.SHELL,
                    user_id="user-123",
                    session_id="session-456",
                    command=cmd,
                )
            )
            assert result.decision == PolicyDecision.DENY, f"Expected DENY for: {cmd}"

    def test_command_with_deny_patterns_in_rule(self) -> None:
        """Test command_deny_patterns filter dangerous variants."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "allow_rm_safe",
                        "action_types": ["shell"],
                        "decision": "allow",
                        "command_patterns": [r"^rm\s+"],
                        "command_deny_patterns": [r"-rf\s+/", r"--no-preserve-root"],
                    }
                ],
            }
        )

        # Safe rm is allowed
        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.SHELL,
                user_id="user-123",
                session_id="session-456",
                command="rm temp.txt",
            )
        )
        assert result.decision == PolicyDecision.ALLOW

        # Dangerous rm -rf / is blocked by deny pattern
        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.SHELL,
                user_id="user-123",
                session_id="session-456",
                command="rm -rf /",
            )
        )
        # Rule doesn't match due to deny pattern, falls to default
        assert result.decision == PolicyDecision.DENY


class TestApiEndpointFiltering:
    """Tests for API endpoint/URL filtering."""

    def test_allowed_api_urls(self) -> None:
        """Test allowed API URLs pass."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "allow_api",
                        "action_types": ["api"],
                        "decision": "allow",
                        "url_patterns": [
                            "https://api.example.com/*",
                            "https://*.github.com/*",
                        ],
                    }
                ],
            }
        )

        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.API,
                user_id="user-123",
                session_id="session-456",
                url="https://api.example.com/v1/users",
            )
        )
        assert result.decision == PolicyDecision.ALLOW

    def test_blocked_localhost_urls(self) -> None:
        """Test localhost URLs are blocked.

        Note: url_deny_patterns cause the rule's matches_url() to return False,
        meaning the rule doesn't match. To block URLs, use a deny rule with
        url_patterns (allowlist) or set default_decision to deny.
        """
        # Method: Use default deny and only allow external URLs
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "allow_external_only",
                        "action_types": ["api", "browser"],
                        "decision": "allow",
                        "priority": 100,
                        "url_patterns": [
                            r"https://.*",  # Only allow HTTPS
                        ],
                        "url_deny_patterns": [
                            r"localhost",
                            r"127\.0\.0\.1",
                            r"0\.0\.0\.0",
                        ],
                    }
                ],
            }
        )

        local_urls = [
            "http://localhost:8080/api",
            "http://127.0.0.1:3000/data",
            "http://0.0.0.0:5000/endpoint",
        ]

        for url in local_urls:
            result = policy.evaluate(
                ActionRequest(
                    action_type=ActionType.API,
                    user_id="user-123",
                    session_id="session-456",
                    url=url,
                )
            )
            # These URLs fail to match the allow rule (due to deny patterns or http://),
            # so they fall through to default deny
            assert result.decision == PolicyDecision.DENY, f"Expected DENY for: {url}"


class TestPolicyYamlLoading:
    """Tests for policy YAML loading and validation."""

    def test_load_from_yaml_file(self) -> None:
        """Test loading policy from YAML file."""
        policy_yaml = """
version: 1
default_decision: deny
rules:
  - name: allow_reads
    action_types:
      - file_read
    decision: allow
    path_patterns:
      - "/tmp/*"
  - name: block_writes
    action_types:
      - file_write
    decision: deny
    priority: 100
    path_patterns:
      - "/etc/*"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(policy_yaml)
            f.flush()

            policy = OpenClawPolicy(policy_file=f.name)

            assert len(policy.get_rules()) == 2
            assert policy._default_decision == PolicyDecision.DENY

            # Verify rules loaded correctly
            allow_rule = policy.get_rule("allow_reads")
            assert allow_rule is not None
            assert allow_rule.decision == PolicyDecision.ALLOW

            block_rule = policy.get_rule("block_writes")
            assert block_rule is not None
            assert block_rule.priority == 100

            # Cleanup
            Path(f.name).unlink()

    def test_load_from_dict(self) -> None:
        """Test loading policy from dictionary."""
        policy_dict = {
            "version": 2,
            "default_decision": "allow",
            "rules": [
                {
                    "name": "test_rule",
                    "action_types": ["shell"],
                    "decision": "require_approval",
                    "command_patterns": [r"^sudo"],
                }
            ],
        }

        policy = OpenClawPolicy(policy_dict=policy_dict)

        assert policy._version == 2
        assert policy._default_decision == PolicyDecision.ALLOW
        assert len(policy.get_rules()) == 1

    def test_file_not_found_raises_error(self) -> None:
        """Test that missing policy file raises error."""
        with pytest.raises(FileNotFoundError):
            OpenClawPolicy(policy_file="/nonexistent/policy.yaml")

    def test_save_to_file(self) -> None:
        """Test saving policy to YAML file."""
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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        policy.save_to_file(temp_path)

        # Verify saved correctly
        with open(temp_path) as f:
            saved_data = yaml.safe_load(f)

        assert saved_data["version"] == 1
        assert saved_data["default_decision"] == "deny"
        assert len(saved_data["rules"]) == 1

        Path(temp_path).unlink()

    def test_to_dict_export(self) -> None:
        """Test exporting policy to dictionary."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 3,
                "default_decision": "require_approval",
                "rules": [
                    {
                        "name": "export_test",
                        "action_types": ["file_read"],
                        "decision": "allow",
                        "path_patterns": ["/safe/*"],
                        "tags": ["safety", "test"],
                    }
                ],
            }
        )

        exported = policy.to_dict()

        assert exported["version"] == 3
        assert exported["default_decision"] == "require_approval"
        assert len(exported["rules"]) == 1
        assert exported["rules"][0]["name"] == "export_test"
        assert "safety" in exported["rules"][0]["tags"]


class TestInvalidPolicyConfiguration:
    """Tests for invalid policy configuration handling."""

    def test_invalid_action_type_raises_error(self) -> None:
        """Test invalid action type raises error."""
        with pytest.raises(ValueError):
            OpenClawPolicy(
                policy_dict={
                    "version": 1,
                    "rules": [
                        {
                            "name": "bad_rule",
                            "action_types": ["invalid_action"],
                            "decision": "allow",
                        }
                    ],
                }
            )

    def test_invalid_decision_raises_error(self) -> None:
        """Test invalid decision raises error."""
        with pytest.raises(ValueError):
            OpenClawPolicy(
                policy_dict={
                    "version": 1,
                    "rules": [
                        {
                            "name": "bad_rule",
                            "action_types": ["shell"],
                            "decision": "invalid_decision",
                        }
                    ],
                }
            )

    def test_empty_rules_list_is_valid(self) -> None:
        """Test that empty rules list is valid."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [],
            }
        )

        assert len(policy.get_rules()) == 0

    def test_missing_name_uses_default(self) -> None:
        """Test that missing rule name uses default."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "rules": [
                    {
                        "action_types": ["shell"],
                        "decision": "allow",
                    }
                ],
            }
        )

        rule = policy.get_rules()[0]
        assert rule.name == "unnamed"


class TestWildcardPatterns:
    """Tests for wildcard patterns in policies."""

    def test_single_wildcard_in_path(self) -> None:
        """Test single * wildcard in path patterns."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "user_homes",
                        "action_types": ["file_read"],
                        "decision": "allow",
                        "path_patterns": ["/home/*/documents/*"],
                    }
                ],
            }
        )

        # Matches
        assert (
            policy.evaluate(
                ActionRequest(
                    action_type=ActionType.FILE_READ,
                    user_id="user-123",
                    session_id="session-456",
                    path="/home/john/documents/report.pdf",
                )
            ).decision
            == PolicyDecision.ALLOW
        )

        # Doesn't match - too deep
        assert (
            policy.evaluate(
                ActionRequest(
                    action_type=ActionType.FILE_READ,
                    user_id="user-123",
                    session_id="session-456",
                    path="/home/john/documents/subdir/report.pdf",
                )
            ).decision
            == PolicyDecision.DENY
        )

    def test_question_mark_wildcard(self) -> None:
        """Test ? wildcard for single character matching."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "log_files",
                        "action_types": ["file_read"],
                        "decision": "allow",
                        "path_patterns": ["/var/log/app?.log"],
                    }
                ],
            }
        )

        # Matches single character
        assert (
            policy.evaluate(
                ActionRequest(
                    action_type=ActionType.FILE_READ,
                    user_id="user-123",
                    session_id="session-456",
                    path="/var/log/app1.log",
                )
            ).decision
            == PolicyDecision.ALLOW
        )

        # Doesn't match multiple characters
        assert (
            policy.evaluate(
                ActionRequest(
                    action_type=ActionType.FILE_READ,
                    user_id="user-123",
                    session_id="session-456",
                    path="/var/log/app123.log",
                )
            ).decision
            == PolicyDecision.DENY
        )


class TestPolicyEvaluationPerformance:
    """Tests for policy evaluation performance."""

    def test_evaluation_returns_timing(self) -> None:
        """Test that evaluation returns timing information."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "allow_all",
                        "action_types": ["shell"],
                        "decision": "allow",
                    }
                ],
            }
        )

        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.SHELL,
                user_id="user-123",
                session_id="session-456",
                command="ls -la",
            )
        )

        assert result.evaluation_time_ms >= 0
        assert result.evaluation_time_ms < 1000  # Should be fast

    def test_many_rules_performance(self) -> None:
        """Test evaluation performance with many rules."""
        # Create policy with 100 rules
        rules = []
        for i in range(100):
            rules.append(
                {
                    "name": f"rule_{i}",
                    "action_types": ["shell"],
                    "decision": "allow",
                    "priority": i,
                    "command_patterns": [f"^command_{i}"],
                }
            )

        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": rules,
            }
        )

        # Evaluate multiple times and check performance
        start = time.time()
        for _ in range(100):
            policy.evaluate(
                ActionRequest(
                    action_type=ActionType.SHELL,
                    user_id="user-123",
                    session_id="session-456",
                    command="command_50",
                )
            )
        elapsed = time.time() - start

        # 100 evaluations should complete in under 1 second
        assert elapsed < 1.0, f"Performance too slow: {elapsed:.2f}s for 100 evaluations"


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    def test_rate_limit_enforcement(self) -> None:
        """Test that rate limits are enforced."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "rate_limited_api",
                        "action_types": ["api"],
                        "decision": "allow",
                        "rate_limit": 3,
                        "rate_limit_window": 60,
                    }
                ],
            }
        )

        request = ActionRequest(
            action_type=ActionType.API,
            user_id="user-123",
            session_id="session-456",
            url="https://api.example.com/endpoint",
        )

        # First 3 requests should succeed
        for i in range(3):
            result = policy.evaluate(request)
            assert result.decision == PolicyDecision.ALLOW, f"Request {i + 1} should be allowed"

        # 4th request should be rate limited
        result = policy.evaluate(request)
        assert result.decision == PolicyDecision.DENY
        assert result.metadata.get("rate_limited") is True
        assert "rate limit" in result.reason.lower()


class TestWorkspaceRestriction:
    """Tests for workspace-only restrictions."""

    def test_workspace_only_allows_workspace_paths(self) -> None:
        """Test workspace_only restriction allows workspace paths."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "workspace_files",
                        "action_types": ["file_read", "file_write"],
                        "decision": "allow",
                        "workspace_only": True,
                    }
                ],
            }
        )

        # Workspace path allowed
        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.FILE_READ,
                user_id="user-123",
                session_id="session-456",
                workspace_id="project-1",
                path="/workspace/project-1/file.txt",
            )
        )
        assert result.decision == PolicyDecision.ALLOW

    def test_workspace_only_blocks_external_paths(self) -> None:
        """Test workspace_only restriction blocks external paths."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "workspace_files",
                        "action_types": ["file_read", "file_write"],
                        "decision": "allow",
                        "workspace_only": True,
                    }
                ],
            }
        )

        # External path blocked (rule doesn't match)
        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.FILE_READ,
                user_id="user-123",
                session_id="session-456",
                workspace_id="project-1",
                path="/etc/passwd",
            )
        )
        assert result.decision == PolicyDecision.DENY


class TestEventCallback:
    """Tests for event callback functionality."""

    def test_event_callback_called_on_evaluation(self) -> None:
        """Test that event callback is called on evaluation."""
        events = []

        def capture_event(event_type: str, data: dict) -> None:
            events.append({"type": event_type, "data": data})

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
            },
            event_callback=capture_event,
        )

        policy.evaluate(
            ActionRequest(
                action_type=ActionType.SHELL,
                user_id="user-123",
                session_id="session-456",
                command="ls -la",
            )
        )

        assert len(events) == 1
        assert events[0]["type"] == "policy_evaluated"
        assert events[0]["data"]["result"]["decision"] == "allow"

    def test_event_callback_exception_handled(self) -> None:
        """Test that event callback exceptions are handled gracefully."""

        def failing_callback(event_type: str, data: dict) -> None:
            raise RuntimeError("Callback failed")

        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "rules": [
                    {
                        "name": "test_rule",
                        "action_types": ["shell"],
                        "decision": "allow",
                    }
                ],
            },
            event_callback=failing_callback,
        )

        # Should not raise, even with failing callback
        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.SHELL,
                user_id="user-123",
                session_id="session-456",
                command="ls -la",
            )
        )

        assert result.decision == PolicyDecision.ALLOW


class TestRuleManagement:
    """Tests for rule management methods."""

    def test_add_rule(self) -> None:
        """Test adding a rule to policy."""
        policy = OpenClawPolicy()

        rule = PolicyRule(
            name="dynamic_rule",
            action_types=[ActionType.SHELL],
            decision=PolicyDecision.ALLOW,
            priority=50,
        )

        policy.add_rule(rule)

        assert len(policy.get_rules()) == 1
        assert policy.get_rule("dynamic_rule") is not None

    def test_remove_rule(self) -> None:
        """Test removing a rule from policy."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "rules": [
                    {
                        "name": "to_remove",
                        "action_types": ["shell"],
                        "decision": "allow",
                    },
                    {
                        "name": "to_keep",
                        "action_types": ["file_read"],
                        "decision": "allow",
                    },
                ],
            }
        )

        assert len(policy.get_rules()) == 2

        removed = policy.remove_rule("to_remove")

        assert removed is True
        assert len(policy.get_rules()) == 1
        assert policy.get_rule("to_remove") is None
        assert policy.get_rule("to_keep") is not None

    def test_remove_nonexistent_rule(self) -> None:
        """Test removing a nonexistent rule."""
        policy = OpenClawPolicy()

        removed = policy.remove_rule("nonexistent")

        assert removed is False

    def test_get_rules_returns_copy(self) -> None:
        """Test that get_rules returns a copy."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "rules": [
                    {
                        "name": "test_rule",
                        "action_types": ["shell"],
                        "decision": "allow",
                    }
                ],
            }
        )

        rules = policy.get_rules()
        rules.clear()

        # Original should be unchanged
        assert len(policy.get_rules()) == 1


class TestEnterprisePolicy:
    """Tests for enterprise policy creation."""

    def test_create_enterprise_policy(self) -> None:
        """Test enterprise policy factory function."""
        policy = create_enterprise_policy()

        # Should have multiple security rules
        rules = policy.get_rules()
        assert len(rules) > 5

        # Should have default deny
        assert policy._default_decision == PolicyDecision.DENY

    def test_enterprise_policy_blocks_system_directories(self) -> None:
        """Test enterprise policy blocks system directories."""
        policy = create_enterprise_policy()

        dangerous_paths = ["/etc/passwd", "/sys/kernel", "/proc/cpuinfo", "/root/.ssh/id_rsa"]

        for path in dangerous_paths:
            result = policy.evaluate(
                ActionRequest(
                    action_type=ActionType.FILE_READ,
                    user_id="user-123",
                    session_id="session-456",
                    path=path,
                )
            )
            assert result.decision == PolicyDecision.DENY, f"Expected DENY for: {path}"

    def test_enterprise_policy_blocks_dangerous_commands(self) -> None:
        """Test enterprise policy blocks dangerous shell commands."""
        policy = create_enterprise_policy()

        dangerous_commands = [
            "rm -rf /",
            "mkfs.ext4 /dev/sda1",
            "dd if=/dev/zero of=/dev/sda",
            "chmod 777 /",
        ]

        for cmd in dangerous_commands:
            result = policy.evaluate(
                ActionRequest(
                    action_type=ActionType.SHELL,
                    user_id="user-123",
                    session_id="session-456",
                    command=cmd,
                )
            )
            assert result.decision == PolicyDecision.DENY, f"Expected DENY for: {cmd}"

    def test_enterprise_policy_requires_approval_for_sudo(self) -> None:
        """Test enterprise policy requires approval for sudo commands."""
        policy = create_enterprise_policy()

        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.SHELL,
                user_id="user-123",
                session_id="session-456",
                command="sudo apt-get update",
            )
        )

        assert result.decision == PolicyDecision.REQUIRE_APPROVAL

    def test_enterprise_policy_allows_workspace_files(self) -> None:
        """Test enterprise policy allows workspace file operations."""
        policy = create_enterprise_policy()

        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.FILE_WRITE,
                user_id="user-123",
                session_id="session-456",
                workspace_id="project-1",
                path="/workspace/project-1/app.py",
            )
        )

        assert result.decision == PolicyDecision.ALLOW

    def test_enterprise_policy_allows_dev_commands(self) -> None:
        """Test enterprise policy allows common development commands."""
        policy = create_enterprise_policy()

        dev_commands = ["ls -la", "python script.py", "git status", "npm install"]

        for cmd in dev_commands:
            result = policy.evaluate(
                ActionRequest(
                    action_type=ActionType.SHELL,
                    user_id="user-123",
                    session_id="session-456",
                    command=cmd,
                )
            )
            assert result.decision == PolicyDecision.ALLOW, f"Expected ALLOW for: {cmd}"


class TestPolicyEvaluationResult:
    """Tests for PolicyEvaluationResult dataclass."""

    def test_result_fields(self) -> None:
        """Test result dataclass fields."""
        rule = PolicyRule(
            name="test_rule",
            action_types=[ActionType.SHELL],
            decision=PolicyDecision.ALLOW,
        )

        result = PolicyEvaluationResult(
            decision=PolicyDecision.ALLOW,
            matched_rule=rule,
            reason="Test reason",
            evaluation_time_ms=1.5,
            requires_audit=True,
            approval_workflow=None,
            metadata={"key": "value"},
        )

        assert result.decision == PolicyDecision.ALLOW
        assert result.matched_rule == rule
        assert result.reason == "Test reason"
        assert result.evaluation_time_ms == 1.5
        assert result.requires_audit is True
        assert result.approval_workflow is None
        assert result.metadata["key"] == "value"

    def test_result_with_approval_workflow(self) -> None:
        """Test result with approval workflow."""
        result = PolicyEvaluationResult(
            decision=PolicyDecision.REQUIRE_APPROVAL,
            matched_rule=None,
            reason="Needs approval",
            evaluation_time_ms=0.5,
            approval_workflow="security_review",
        )

        assert result.approval_workflow == "security_review"


class TestSpecialActionTypes:
    """Tests for special action types (screenshot, keyboard, mouse)."""

    def test_screenshot_action(self) -> None:
        """Test screenshot action type evaluation."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "allow_screenshots",
                        "action_types": ["screenshot"],
                        "decision": "allow",
                        "rate_limit": 5,
                        "rate_limit_window": 60,
                    }
                ],
            }
        )

        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.SCREENSHOT,
                user_id="user-123",
                session_id="session-456",
            )
        )

        assert result.decision == PolicyDecision.ALLOW

    def test_keyboard_action(self) -> None:
        """Test keyboard action type evaluation."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "allow_keyboard",
                        "action_types": ["keyboard"],
                        "decision": "allow",
                    }
                ],
            }
        )

        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.KEYBOARD,
                user_id="user-123",
                session_id="session-456",
            )
        )

        assert result.decision == PolicyDecision.ALLOW

    def test_mouse_action(self) -> None:
        """Test mouse action type evaluation."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "allow_mouse",
                        "action_types": ["mouse"],
                        "decision": "allow",
                    }
                ],
            }
        )

        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.MOUSE,
                user_id="user-123",
                session_id="session-456",
            )
        )

        assert result.decision == PolicyDecision.ALLOW


class TestBrowserActions:
    """Tests for browser action type."""

    def test_browser_url_patterns(self) -> None:
        """Test browser action with URL patterns."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "deny",
                "rules": [
                    {
                        "name": "allow_https",
                        "action_types": ["browser"],
                        "decision": "allow",
                        "url_patterns": ["https://*"],
                    }
                ],
            }
        )

        # HTTPS allowed
        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.BROWSER,
                user_id="user-123",
                session_id="session-456",
                url="https://example.com",
            )
        )
        assert result.decision == PolicyDecision.ALLOW

        # HTTP not matched (falls to default deny)
        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.BROWSER,
                user_id="user-123",
                session_id="session-456",
                url="http://insecure.com",
            )
        )
        assert result.decision == PolicyDecision.DENY

    def test_browser_blocks_file_urls(self) -> None:
        """Test browser blocks file:// URLs."""
        policy = OpenClawPolicy(
            policy_dict={
                "version": 1,
                "default_decision": "allow",
                "rules": [
                    {
                        "name": "block_file_urls",
                        "action_types": ["browser"],
                        "decision": "deny",
                        "priority": 100,
                        "url_deny_patterns": [r"^file://"],
                    }
                ],
            }
        )

        result = policy.evaluate(
            ActionRequest(
                action_type=ActionType.BROWSER,
                user_id="user-123",
                session_id="session-456",
                url="file:///etc/passwd",
            )
        )

        assert result.decision == PolicyDecision.DENY
