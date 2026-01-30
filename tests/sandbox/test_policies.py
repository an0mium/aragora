"""
Tests for Sandbox Policy Definitions.

Tests cover:
- ToolPolicy creation and configuration
- ToolPolicyChecker for tool, path, and network access
- Policy rule matching (ToolRule, PathRule, NetworkRule)
- ResourceLimit enforcement
- Default, strict, and permissive policy factories
- Audit logging
"""

import pytest

from aragora.sandbox.policies import (
    NetworkRule,
    PathRule,
    PolicyAction,
    ResourceLimit,
    ResourceType,
    ToolPolicy,
    ToolPolicyChecker,
    ToolRule,
    create_default_policy,
    create_permissive_policy,
    create_strict_policy,
)


class TestToolRule:
    """Tests for ToolRule dataclass."""

    def test_exact_match(self):
        """Test exact pattern matching."""
        rule = ToolRule(pattern=r"^python$", action=PolicyAction.ALLOW)

        assert rule.matches("python")
        assert not rule.matches("python3")
        assert not rule.matches("pythonw")

    def test_regex_match(self):
        """Test regex pattern matching."""
        rule = ToolRule(pattern=r"^python\d*$", action=PolicyAction.ALLOW)

        assert rule.matches("python")
        assert rule.matches("python3")
        assert rule.matches("python2")
        assert not rule.matches("pythonw")

    def test_wildcard_match(self):
        """Test wildcard-like regex matching."""
        rule = ToolRule(pattern=r"^pip.*$", action=PolicyAction.ALLOW)

        assert rule.matches("pip")
        assert rule.matches("pip3")
        assert rule.matches("pip-compile")

    def test_rule_with_reason(self):
        """Test rule with reason."""
        rule = ToolRule(
            pattern=r"^rm$",
            action=PolicyAction.DENY,
            reason="Dangerous system tool",
        )

        assert rule.matches("rm")
        assert rule.action == PolicyAction.DENY
        assert rule.reason == "Dangerous system tool"

    def test_rule_with_resource_limits(self):
        """Test rule with resource limits."""
        rule = ToolRule(
            pattern=r"^node$",
            action=PolicyAction.ALLOW,
            resource_limits={"max_memory_mb": 256},
        )

        assert rule.resource_limits["max_memory_mb"] == 256


class TestPathRule:
    """Tests for PathRule dataclass."""

    def test_path_match(self):
        """Test path pattern matching."""
        rule = PathRule(pattern=r"^/tmp/sandbox/.*$", action=PolicyAction.ALLOW)

        assert rule.matches("/tmp/sandbox/test.py")
        assert rule.matches("/tmp/sandbox/subdir/file.txt")
        assert not rule.matches("/tmp/other/file.txt")

    def test_read_only_path(self):
        """Test read-only path rule."""
        rule = PathRule(
            pattern=r"^/usr/lib/.*$",
            action=PolicyAction.ALLOW,
            read_allowed=True,
            write_allowed=False,
        )

        assert rule.read_allowed is True
        assert rule.write_allowed is False

    def test_read_write_path(self):
        """Test read-write path rule."""
        rule = PathRule(
            pattern=r"^/workspace/.*$",
            action=PolicyAction.ALLOW,
            read_allowed=True,
            write_allowed=True,
        )

        assert rule.read_allowed is True
        assert rule.write_allowed is True

    def test_denied_path(self):
        """Test denied path rule."""
        rule = PathRule(
            pattern=r"^/etc/passwd$",
            action=PolicyAction.DENY,
            reason="System file",
        )

        assert rule.matches("/etc/passwd")
        assert rule.action == PolicyAction.DENY


class TestNetworkRule:
    """Tests for NetworkRule dataclass."""

    def test_localhost_match(self):
        """Test localhost matching."""
        rule = NetworkRule(
            host_pattern=r"^localhost$",
            port_range=(80, 443),
            protocols=["http", "https"],
        )

        assert rule.matches("localhost", 80, "http")
        assert rule.matches("localhost", 443, "https")
        assert not rule.matches("localhost", 8080, "http")  # Port out of range
        assert not rule.matches("example.com", 80, "http")  # Host doesn't match

    def test_ip_address_match(self):
        """Test IP address pattern matching."""
        rule = NetworkRule(
            host_pattern=r"^127\.0\.0\.1$",
            port_range=(1024, 65535),
        )

        assert rule.matches("127.0.0.1", 8080, "http")
        assert not rule.matches("192.168.1.1", 8080, "http")

    def test_domain_pattern_match(self):
        """Test domain pattern matching."""
        rule = NetworkRule(
            host_pattern=r".*\.example\.com$",
            port_range=(80, 443),
        )

        assert rule.matches("api.example.com", 443, "https")
        assert rule.matches("subdomain.example.com", 80, "http")
        assert not rule.matches("example.org", 80, "http")

    def test_protocol_filtering(self):
        """Test protocol filtering."""
        rule = NetworkRule(
            host_pattern=r"^.*$",
            protocols=["https"],
        )

        assert rule.matches("example.com", 443, "https")
        assert not rule.matches("example.com", 80, "http")


class TestResourceLimit:
    """Tests for ResourceLimit dataclass."""

    def test_default_limits(self):
        """Test default resource limits."""
        limits = ResourceLimit()

        assert limits.max_memory_mb == 512
        assert limits.max_cpu_percent == 100
        assert limits.max_execution_seconds == 60
        assert limits.max_processes == 10
        assert limits.max_file_size_mb == 10
        assert limits.max_files_created == 100
        assert limits.max_network_requests == 50

    def test_custom_limits(self):
        """Test custom resource limits."""
        limits = ResourceLimit(
            max_memory_mb=256,
            max_cpu_percent=50,
            max_execution_seconds=30,
            max_processes=5,
        )

        assert limits.max_memory_mb == 256
        assert limits.max_cpu_percent == 50
        assert limits.max_execution_seconds == 30
        assert limits.max_processes == 5


class TestToolPolicy:
    """Tests for ToolPolicy dataclass."""

    def test_default_policy_creation(self):
        """Test creating a policy with defaults."""
        policy = ToolPolicy(name="test")

        assert policy.name == "test"
        assert policy.default_tool_action == PolicyAction.DENY
        assert policy.default_path_action == PolicyAction.DENY
        assert policy.default_network_action == PolicyAction.DENY
        assert policy.audit_denials is True

    def test_add_tool_allowlist(self):
        """Test adding tools to allowlist."""
        policy = ToolPolicy(name="test")
        policy.add_tool_allowlist(["python3", "node"], reason="Safe interpreters")

        assert len(policy.tool_rules) == 2
        assert policy.tool_rules[0].pattern == "python3"
        assert policy.tool_rules[0].action == PolicyAction.ALLOW

    def test_add_tool_denylist(self):
        """Test adding tools to denylist."""
        policy = ToolPolicy(name="test")
        policy.add_tool_denylist([r"^rm$", r"^sudo$"], reason="Dangerous tools")

        assert len(policy.tool_rules) == 2
        assert all(r.action == PolicyAction.DENY for r in policy.tool_rules)

    def test_add_path_allowlist(self):
        """Test adding paths to allowlist."""
        policy = ToolPolicy(name="test")
        policy.add_path_allowlist(
            [r"^/tmp/.*$", r"^/workspace/.*$"],
            read=True,
            write=True,
            reason="Workspace directories",
        )

        assert len(policy.path_rules) == 2
        assert policy.path_rules[0].read_allowed is True
        assert policy.path_rules[0].write_allowed is True

    def test_add_network_allowlist(self):
        """Test adding network hosts to allowlist."""
        policy = ToolPolicy(name="test")
        policy.add_network_allowlist(
            [r"^localhost$", r"^127\.0\.0\.1$"],
            ports=(1024, 65535),
            reason="Local development",
        )

        assert len(policy.network_rules) == 2
        assert policy.network_rules[0].port_range == (1024, 65535)


class TestToolPolicyChecker:
    """Tests for ToolPolicyChecker."""

    @pytest.fixture
    def default_checker(self):
        """Create a checker with default policy."""
        policy = create_default_policy()
        return ToolPolicyChecker(policy)

    @pytest.fixture
    def strict_checker(self):
        """Create a checker with strict policy."""
        policy = create_strict_policy()
        return ToolPolicyChecker(policy)

    @pytest.fixture
    def permissive_checker(self):
        """Create a checker with permissive policy."""
        policy = create_permissive_policy()
        return ToolPolicyChecker(policy)

    def test_check_allowed_tool(self, default_checker):
        """Test checking an allowed tool."""
        allowed, reason = default_checker.check_tool("python3")

        assert allowed is True
        assert reason == "Safe standard tools"

    def test_check_denied_tool(self, default_checker):
        """Test checking a denied tool."""
        allowed, reason = default_checker.check_tool("rm")

        assert allowed is False
        assert reason == "Dangerous system tools"

    def test_check_unknown_tool_default_deny(self, default_checker):
        """Test checking unknown tool with default deny."""
        allowed, reason = default_checker.check_tool("unknown_tool")

        assert allowed is False
        assert reason == "default policy"

    def test_check_path_read_allowed(self, default_checker):
        """Test checking allowed read path."""
        allowed, reason = default_checker.check_path("/tmp/sandbox/test.py", "read")

        assert allowed is True
        assert reason == "Sandbox workspace"

    def test_check_path_write_allowed(self, default_checker):
        """Test checking allowed write path."""
        allowed, reason = default_checker.check_path("/workspace/test.py", "write")

        assert allowed is True
        assert reason == "Sandbox workspace"

    def test_check_path_read_only(self, default_checker):
        """Test checking read-only path for write."""
        allowed, reason = default_checker.check_path("/usr/lib/python3/test.py", "write")

        assert allowed is False
        assert reason == "System libraries (read-only)"

    def test_check_path_default_deny(self, default_checker):
        """Test checking path with default deny."""
        allowed, reason = default_checker.check_path("/etc/passwd", "read")

        assert allowed is False
        assert reason == "default policy"

    def test_check_network_localhost_allowed(self, default_checker):
        """Test checking allowed localhost network."""
        allowed, reason = default_checker.check_network("localhost", 8080, "http")

        assert allowed is True
        assert reason == "Localhost for testing"

    def test_check_network_external_denied(self, default_checker):
        """Test checking external network denied."""
        allowed, reason = default_checker.check_network("example.com", 443, "https")

        assert allowed is False
        assert reason == "default policy"

    def test_get_resource_limits(self, default_checker):
        """Test getting resource limits."""
        limits = default_checker.get_resource_limits()

        assert isinstance(limits, ResourceLimit)
        assert limits.max_memory_mb > 0

    def test_audit_log_recorded(self, default_checker):
        """Test that audit log is recorded."""
        default_checker.clear_audit_log()

        default_checker.check_tool("python3")
        default_checker.check_tool("rm")

        log = default_checker.get_audit_log()

        assert len(log) == 2
        assert log[0]["resource"] == "python3"
        assert log[0]["allowed"] is True
        assert log[1]["resource"] == "rm"
        assert log[1]["allowed"] is False

    def test_clear_audit_log(self, default_checker):
        """Test clearing audit log."""
        default_checker.check_tool("python3")
        default_checker.clear_audit_log()

        log = default_checker.get_audit_log()
        assert len(log) == 0

    def test_strict_policy_limited_tools(self, strict_checker):
        """Test strict policy has limited tools."""
        allowed_cat, _ = strict_checker.check_tool("cat")
        allowed_python, _ = strict_checker.check_tool("python3")

        assert allowed_cat is True
        assert allowed_python is False  # Not in strict allowlist

    def test_strict_policy_limited_resources(self, strict_checker):
        """Test strict policy has limited resources."""
        limits = strict_checker.get_resource_limits()

        assert limits.max_memory_mb == 256
        assert limits.max_cpu_percent == 50
        assert limits.max_execution_seconds == 30
        assert limits.max_network_requests == 0

    def test_permissive_policy_allows_most_tools(self, permissive_checker):
        """Test permissive policy allows most tools."""
        allowed_python, _ = permissive_checker.check_tool("python3")
        allowed_custom, _ = permissive_checker.check_tool("custom_tool")

        assert allowed_python is True
        assert allowed_custom is True

    def test_permissive_policy_generous_resources(self, permissive_checker):
        """Test permissive policy has generous resources."""
        limits = permissive_checker.get_resource_limits()

        assert limits.max_memory_mb == 2048
        assert limits.max_cpu_percent == 200
        assert limits.max_execution_seconds == 300


class TestPolicyFactories:
    """Tests for policy factory functions."""

    def test_create_default_policy(self):
        """Test default policy factory."""
        policy = create_default_policy()

        assert policy.name == "default"
        assert policy.default_tool_action == PolicyAction.DENY
        assert len(policy.tool_rules) > 0
        assert len(policy.path_rules) > 0
        assert len(policy.network_rules) > 0

    def test_create_strict_policy(self):
        """Test strict policy factory."""
        policy = create_strict_policy()

        assert policy.name == "strict"
        assert policy.audit_all is True
        assert policy.resource_limits.max_memory_mb < 512

    def test_create_permissive_policy(self):
        """Test permissive policy factory."""
        policy = create_permissive_policy()

        assert policy.name == "permissive"
        assert policy.default_tool_action == PolicyAction.ALLOW
        assert policy.default_path_action == PolicyAction.ALLOW
        assert policy.resource_limits.max_memory_mb > 1024


class TestPolicyAction:
    """Tests for PolicyAction enum."""

    def test_policy_actions(self):
        """Test policy action values."""
        assert PolicyAction.ALLOW.value == "allow"
        assert PolicyAction.DENY.value == "deny"
        assert PolicyAction.AUDIT.value == "audit"

    def test_audit_action_allows_but_logs(self):
        """Test audit action allows but logs."""
        policy = ToolPolicy(name="test")
        policy.tool_rules.append(ToolRule(pattern=r"^special_tool$", action=PolicyAction.AUDIT))

        checker = ToolPolicyChecker(policy)
        allowed, reason = checker.check_tool("special_tool")

        # AUDIT action should allow
        assert allowed is True


class TestResourceType:
    """Tests for ResourceType enum."""

    def test_resource_types(self):
        """Test resource type values."""
        assert ResourceType.FILE_READ.value == "file_read"
        assert ResourceType.FILE_WRITE.value == "file_write"
        assert ResourceType.NETWORK.value == "network"
        assert ResourceType.PROCESS.value == "process"
        assert ResourceType.ENVIRONMENT.value == "environment"
        assert ResourceType.MEMORY.value == "memory"
        assert ResourceType.CPU.value == "cpu"


class TestPolicyChainedRules:
    """Tests for policy rule chaining and precedence."""

    def test_first_match_wins(self):
        """Test that first matching rule wins."""
        policy = ToolPolicy(name="test")
        # Add deny rule first
        policy.add_tool_denylist([r"^python$"], reason="Denied first")
        # Then add allow rule
        policy.add_tool_allowlist([r"^python$"], reason="Allowed second")

        checker = ToolPolicyChecker(policy)
        allowed, reason = checker.check_tool("python")

        # First rule (deny) should win
        assert allowed is False
        assert reason == "Denied first"

    def test_specific_rule_before_general(self):
        """Test specific rules should come before general."""
        policy = ToolPolicy(name="test")
        # Specific deny for rm
        policy.add_tool_denylist([r"^rm$"], reason="rm is dangerous")
        # General allow for shell tools
        policy.add_tool_allowlist([r"^.*$"], reason="Allow all")

        checker = ToolPolicyChecker(policy)

        rm_allowed, rm_reason = checker.check_tool("rm")
        ls_allowed, ls_reason = checker.check_tool("ls")

        assert rm_allowed is False
        assert rm_reason == "rm is dangerous"
        assert ls_allowed is True
        assert ls_reason == "Allow all"

    def test_multiple_path_rules(self):
        """Test multiple path rules."""
        policy = ToolPolicy(name="test")
        # Deny sensitive files
        policy.path_rules.append(
            PathRule(pattern=r"^/workspace/\.env$", action=PolicyAction.DENY, reason="Sensitive")
        )
        # Allow workspace
        policy.add_path_allowlist([r"^/workspace/.*$"], read=True, write=True, reason="Workspace")

        checker = ToolPolicyChecker(policy)

        env_allowed, _ = checker.check_path("/workspace/.env", "read")
        code_allowed, _ = checker.check_path("/workspace/code.py", "read")

        assert env_allowed is False
        assert code_allowed is True
