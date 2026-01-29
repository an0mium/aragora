"""Tests for computer-use policy definitions."""

import pytest

from aragora.computer_use.actions import ActionType, ClickAction, TypeAction
from aragora.computer_use.policies import (
    ActionRule,
    BoundaryRule,
    ComputerPolicy,
    ComputerPolicyChecker,
    DomainRule,
    PolicyDecision,
    RateLimits,
    TextPatternRule,
    create_default_computer_policy,
    create_readonly_computer_policy,
    create_strict_computer_policy,
)


class TestPolicyDecision:
    """Test PolicyDecision enum."""

    def test_all_decisions_defined(self):
        """Verify all decision types exist."""
        expected = ["allow", "deny", "audit", "require_approval"]
        actual = [d.value for d in PolicyDecision]
        assert set(expected) == set(actual)


class TestActionRule:
    """Test ActionRule dataclass."""

    def test_create_allow_rule(self):
        """Test creating an allow rule."""
        rule = ActionRule(
            action_type=ActionType.CLICK,
            decision=PolicyDecision.ALLOW,
            reason="Standard click action",
        )
        assert rule.action_type == ActionType.CLICK
        assert rule.decision == PolicyDecision.ALLOW
        assert rule.reason == "Standard click action"

    def test_create_deny_rule(self):
        """Test creating a deny rule."""
        rule = ActionRule(
            action_type=ActionType.DRAG,
            decision=PolicyDecision.DENY,
            reason="Drag disabled",
        )
        assert rule.decision == PolicyDecision.DENY

    def test_rate_limits(self):
        """Test rule with rate limits."""
        rule = ActionRule(
            action_type=ActionType.TYPE,
            max_per_session=100,
            cooldown_seconds=0.5,
        )
        assert rule.max_per_session == 100
        assert rule.cooldown_seconds == 0.5


class TestDomainRule:
    """Test DomainRule dataclass."""

    def test_match_exact_domain(self):
        """Test matching exact domain."""
        rule = DomainRule(pattern=r"^https://example\.com.*$")
        assert rule.matches("https://example.com/page")
        assert not rule.matches("https://other.com/page")

    def test_match_localhost(self):
        """Test matching localhost."""
        rule = DomainRule(pattern=r"^https?://localhost(:\d+)?(/.*)?$")
        assert rule.matches("http://localhost:8080/api")
        assert rule.matches("https://localhost/")
        assert not rule.matches("https://example.com")

    def test_case_insensitive_match(self):
        """Test case-insensitive matching."""
        rule = DomainRule(pattern=r"^https://Example\.com.*$")
        # Note: matches() uses re.IGNORECASE
        assert rule.matches("https://example.com/")


class TestTextPatternRule:
    """Test TextPatternRule dataclass."""

    def test_create_rule(self):
        """Test creating a text pattern rule."""
        rule = TextPatternRule(
            pattern=r"password\s*[:=]\s*\S+",
            decision=PolicyDecision.DENY,
            reason="Credential protection",
        )
        assert rule.pattern == r"password\s*[:=]\s*\S+"
        assert rule.decision == PolicyDecision.DENY

    def test_applies_to_specific_actions(self):
        """Test rule applies to specific action types."""
        rule = TextPatternRule(
            pattern=r"secret",
            applies_to=[ActionType.TYPE, ActionType.KEY],
        )
        assert ActionType.TYPE in rule.applies_to
        assert ActionType.KEY in rule.applies_to
        assert ActionType.CLICK not in rule.applies_to


class TestBoundaryRule:
    """Test BoundaryRule dataclass."""

    def test_default_boundaries(self):
        """Test default screen boundaries."""
        rule = BoundaryRule()
        assert rule.min_x == 0
        assert rule.min_y == 0
        assert rule.max_x == 3840  # 4K width
        assert rule.max_y == 2160  # 4K height

    def test_custom_boundaries(self):
        """Test custom boundaries."""
        rule = BoundaryRule(max_x=1920, max_y=1080)
        assert rule.max_x == 1920
        assert rule.max_y == 1080


class TestRateLimits:
    """Test RateLimits dataclass."""

    def test_default_limits(self):
        """Test default rate limits."""
        limits = RateLimits()
        assert limits.max_actions_per_minute == 60
        assert limits.max_clicks_per_minute == 30
        assert limits.max_keystrokes_per_minute == 300

    def test_custom_limits(self):
        """Test custom rate limits."""
        limits = RateLimits(
            max_actions_per_minute=30,
            max_clicks_per_minute=15,
        )
        assert limits.max_actions_per_minute == 30
        assert limits.max_clicks_per_minute == 15


class TestComputerPolicy:
    """Test ComputerPolicy dataclass."""

    def test_create_empty_policy(self):
        """Test creating an empty policy."""
        policy = ComputerPolicy(name="test")
        assert policy.name == "test"
        assert policy.default_decision == PolicyDecision.DENY
        assert len(policy.action_rules) == 0

    def test_add_action_allowlist(self):
        """Test adding actions to allowlist."""
        policy = ComputerPolicy(name="test")
        policy.add_action_allowlist(
            [ActionType.CLICK, ActionType.TYPE],
            reason="Test allowlist",
        )
        assert len(policy.action_rules) == 2
        assert all(r.decision == PolicyDecision.ALLOW for r in policy.action_rules)

    def test_add_action_denylist(self):
        """Test adding actions to denylist."""
        policy = ComputerPolicy(name="test")
        policy.add_action_denylist(
            [ActionType.DRAG],
            reason="Drag disabled",
        )
        assert len(policy.action_rules) == 1
        assert policy.action_rules[0].decision == PolicyDecision.DENY

    def test_add_domain_allowlist(self):
        """Test adding domains to allowlist."""
        policy = ComputerPolicy(name="test")
        policy.add_domain_allowlist(
            [r"^https://example\.com.*$"],
            reason="Trusted domain",
        )
        assert len(policy.domain_rules) == 1

    def test_add_sensitive_text_patterns(self):
        """Test adding sensitive text patterns."""
        policy = ComputerPolicy(name="test")
        policy.add_sensitive_text_patterns(
            [r"password", r"secret"],
            reason="Credential protection",
        )
        assert len(policy.text_pattern_rules) == 2

    def test_safety_settings(self):
        """Test safety setting defaults."""
        policy = ComputerPolicy(name="test")
        assert policy.require_screenshot_before_action is True
        assert policy.require_human_approval_for_sensitive is True
        assert policy.block_credential_fields is True
        assert policy.audit_all_actions is True

    def test_session_limits(self):
        """Test session limit defaults."""
        policy = ComputerPolicy(name="test")
        assert policy.max_actions_per_task == 100
        assert policy.max_consecutive_errors == 3
        assert policy.timeout_per_action_seconds == 10.0


class TestComputerPolicyChecker:
    """Test ComputerPolicyChecker."""

    @pytest.fixture
    def default_policy(self):
        """Create default policy for tests."""
        return create_default_computer_policy()

    @pytest.fixture
    def checker(self, default_policy):
        """Create checker with default policy."""
        return ComputerPolicyChecker(default_policy)

    def test_check_allowed_action(self, checker):
        """Test checking an allowed action."""
        action = ClickAction(x=100, y=100)
        allowed, reason = checker.check_action(action)
        assert allowed is True

    def test_check_action_with_domain(self, checker):
        """Test checking action with URL."""
        action = ClickAction(x=100, y=100)
        allowed, reason = checker.check_action(action, current_url="http://localhost:8080")
        assert allowed is True

    def test_deny_unknown_domain(self, checker):
        """Test denying action on unknown domain."""
        action = ClickAction(x=100, y=100)
        allowed, reason = checker.check_action(action, current_url="https://malicious.com")
        assert allowed is False
        assert "not in allowlist" in reason.lower()

    def test_check_coordinates_in_bounds(self, checker):
        """Test checking coordinates within bounds."""
        action = ClickAction(x=500, y=500)
        allowed, reason = checker.check_action(action)
        assert allowed is True

    def test_deny_coordinates_out_of_bounds(self, checker):
        """Test denying coordinates outside bounds."""
        action = ClickAction(x=5000, y=500)
        allowed, reason = checker.check_action(action)
        assert allowed is False
        assert "coordinate" in reason.lower()

    def test_deny_negative_coordinates(self, checker):
        """Test denying negative coordinates."""
        action = ClickAction(x=-100, y=200)
        allowed, reason = checker.check_action(action)
        assert allowed is False

    def test_check_sensitive_text_denied(self, checker):
        """Test denying sensitive text patterns."""
        action = TypeAction(text="password = mysecret123")
        allowed, reason = checker.check_action(action)
        assert allowed is False

    def test_check_normal_text_allowed(self, checker):
        """Test allowing normal text."""
        action = TypeAction(text="Hello, World!")
        allowed, reason = checker.check_action(action)
        assert allowed is True

    def test_max_actions_limit(self, checker):
        """Test max actions per task limit."""
        # Exhaust action limit
        checker._total_actions = 100
        action = ClickAction(x=100, y=100)
        allowed, reason = checker.check_action(action)
        assert allowed is False
        assert "max actions" in reason.lower()

    def test_consecutive_errors_limit(self, checker):
        """Test consecutive errors limit."""
        # Simulate too many errors
        checker._consecutive_errors = 3
        action = ClickAction(x=100, y=100)
        allowed, reason = checker.check_action(action)
        assert allowed is False
        assert "error" in reason.lower()

    def test_record_success_resets_errors(self, checker):
        """Test recording success resets error count."""
        checker._consecutive_errors = 2
        checker.record_success()
        assert checker._consecutive_errors == 0

    def test_record_error_increments(self, checker):
        """Test recording error increments count."""
        checker.record_error()
        assert checker._consecutive_errors == 1
        checker.record_error()
        assert checker._consecutive_errors == 2

    def test_reset_clears_state(self, checker):
        """Test reset clears all state."""
        checker._total_actions = 50
        checker._consecutive_errors = 2
        checker.reset()
        assert checker._total_actions == 0
        assert checker._consecutive_errors == 0

    def test_get_audit_log(self, checker):
        """Test getting audit log."""
        action = ClickAction(x=100, y=100)
        checker.check_action(action, current_url="http://localhost")
        log = checker.get_audit_log()
        assert len(log) >= 1
        assert log[0]["action_type"] == "click"

    def test_get_stats(self, checker):
        """Test getting session stats."""
        checker._total_actions = 10
        checker._consecutive_errors = 1
        stats = checker.get_stats()
        assert stats["total_actions"] == 10
        assert stats["consecutive_errors"] == 1


class TestDefaultComputerPolicy:
    """Test default computer policy factory."""

    def test_creates_policy(self):
        """Test factory creates valid policy."""
        policy = create_default_computer_policy()
        assert policy.name == "default"
        assert policy.default_decision == PolicyDecision.DENY

    def test_allows_standard_actions(self):
        """Test default policy allows standard actions."""
        policy = create_default_computer_policy()
        checker = ComputerPolicyChecker(policy)

        for action_type in [
            ActionType.SCREENSHOT,
            ActionType.CLICK,
            ActionType.TYPE,
            ActionType.SCROLL,
        ]:
            allowed, _ = checker._check_action_type(action_type)
            assert allowed is True, f"{action_type} should be allowed"

    def test_allows_localhost(self):
        """Test default policy allows localhost."""
        policy = create_default_computer_policy()
        checker = ComputerPolicyChecker(policy)
        allowed, _ = checker._check_domain("http://localhost:3000/api")
        assert allowed is True

    def test_blocks_credentials(self):
        """Test default policy blocks credential patterns."""
        policy = create_default_computer_policy()
        checker = ComputerPolicyChecker(policy)

        # API key pattern
        action = TypeAction(text="sk-1234567890abcdef1234567890abcdef")
        allowed, _ = checker.check_action(action)
        assert allowed is False


class TestStrictComputerPolicy:
    """Test strict computer policy factory."""

    def test_creates_policy(self):
        """Test factory creates valid policy."""
        policy = create_strict_computer_policy()
        assert policy.name == "strict"

    def test_limits_actions(self):
        """Test strict policy has lower limits."""
        policy = create_strict_computer_policy()
        assert policy.max_actions_per_task == 50
        assert policy.max_consecutive_errors == 2

    def test_stricter_rate_limits(self):
        """Test strict policy has stricter rate limits."""
        policy = create_strict_computer_policy()
        assert policy.rate_limits.max_actions_per_minute == 30


class TestReadonlyComputerPolicy:
    """Test readonly computer policy factory."""

    def test_creates_policy(self):
        """Test factory creates valid policy."""
        policy = create_readonly_computer_policy()
        assert policy.name == "readonly"

    def test_allows_observation_actions(self):
        """Test readonly allows observation actions."""
        policy = create_readonly_computer_policy()
        checker = ComputerPolicyChecker(policy)

        allowed, _ = checker._check_action_type(ActionType.SCREENSHOT)
        assert allowed is True

        allowed, _ = checker._check_action_type(ActionType.SCROLL)
        assert allowed is True

    def test_blocks_interactive_actions(self):
        """Test readonly blocks interactive actions."""
        policy = create_readonly_computer_policy()
        checker = ComputerPolicyChecker(policy)

        allowed, _ = checker._check_action_type(ActionType.CLICK)
        assert allowed is False

        allowed, _ = checker._check_action_type(ActionType.TYPE)
        assert allowed is False

        allowed, _ = checker._check_action_type(ActionType.DRAG)
        assert allowed is False
