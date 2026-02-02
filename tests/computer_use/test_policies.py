from aragora.computer_use.actions import ScreenshotAction, TypeAction
from aragora.computer_use.policies import (
    ComputerPolicyChecker,
    PolicyDecision,
    create_default_computer_policy,
)


def test_evaluate_action_requires_approval_for_sensitive_type():
    policy = create_default_computer_policy()
    checker = ComputerPolicyChecker(policy)

    action = TypeAction(text="hello")
    decision, reason = checker.evaluate_action(
        action,
        current_url="http://localhost:8080",
        enforce_sensitive_approvals=True,
    )

    assert decision == PolicyDecision.REQUIRE_APPROVAL
    assert "Sensitive action" in reason


def test_evaluate_action_allows_screenshot():
    policy = create_default_computer_policy()
    checker = ComputerPolicyChecker(policy)

    action = ScreenshotAction()
    decision, _ = checker.evaluate_action(
        action,
        current_url="http://localhost:8080",
        enforce_sensitive_approvals=True,
    )

    assert decision == PolicyDecision.ALLOW


def test_evaluate_action_denies_sensitive_text_patterns():
    policy = create_default_computer_policy()
    checker = ComputerPolicyChecker(policy)

    action = TypeAction(text="password=supersecret")
    decision, reason = checker.evaluate_action(
        action,
        current_url="http://localhost:8080",
        enforce_sensitive_approvals=False,
    )

    assert decision == PolicyDecision.DENY
    assert "credential" in reason.lower()
