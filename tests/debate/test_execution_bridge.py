"""Tests for ExecutionBridge — auto-triggering downstream actions."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.execution_bridge import (
    ActionResult,
    ActionRule,
    ActionType,
    BridgeConfig,
    ExecutionBridge,
    create_default_bridge,
)


@dataclass
class MockDebateResult:
    consensus: str = "Deploy safely"
    final_answer: str = "Deploy safely"
    confidence: float = 0.85
    domain: str = "deployment"
    rounds_used: int = 3
    participants: list[str] | None = None
    task: str = "How to deploy"


@pytest.fixture
def bridge():
    return ExecutionBridge()


@pytest.fixture
def default_bridge():
    return create_default_bridge()


class TestActionRule:
    """Tests for individual action rule matching."""

    def test_rule_matches_when_condition_true(self):
        rule = ActionRule(
            name="test",
            condition=lambda r: r.get("confidence", 0) > 0.8,
            action_type=ActionType.NOTIFICATION,
            min_confidence=0.7,
        )
        assert rule.matches({"confidence": 0.9})

    def test_rule_no_match_below_min_confidence(self):
        rule = ActionRule(
            name="test",
            condition=lambda r: True,
            action_type=ActionType.NOTIFICATION,
            min_confidence=0.8,
        )
        assert not rule.matches({"confidence": 0.5})

    def test_rule_handles_condition_errors(self):
        rule = ActionRule(
            name="test",
            condition=lambda r: r["missing_key"],
            action_type=ActionType.NOTIFICATION,
        )
        assert not rule.matches({})

    def test_rule_default_min_confidence_zero(self):
        rule = ActionRule(
            name="test",
            condition=lambda r: True,
            action_type=ActionType.NOTIFICATION,
        )
        assert rule.matches({"confidence": 0.0})


class TestExecutionBridge:
    """Tests for the ExecutionBridge."""

    def test_register_action(self, bridge):
        rule = bridge.register_action(
            name="test_action",
            condition=lambda r: True,
            action_type=ActionType.NOTIFICATION,
        )
        assert len(bridge.rules) == 1
        assert rule.name == "test_action"

    def test_rules_sorted_by_priority(self, bridge):
        bridge.register_action(
            name="low",
            condition=lambda r: True,
            action_type=ActionType.NOTIFICATION,
            priority=10,
        )
        bridge.register_action(
            name="high",
            condition=lambda r: True,
            action_type=ActionType.NOTIFICATION,
            priority=100,
        )
        assert bridge.rules[0].name == "high"
        assert bridge.rules[1].name == "low"

    def test_evaluate_returns_results(self, bridge):
        bridge.register_action(
            name="always_match",
            condition=lambda r: True,
            action_type=ActionType.OUTCOME_VERIFICATION,
        )
        with patch(
            "aragora.debate.execution_bridge.ExecutionBridge._execute_outcome_verification"
        ) as mock:
            mock.return_value = ActionResult(
                rule_name="always_match",
                action_type=ActionType.OUTCOME_VERIFICATION,
                success=True,
            )
            results = bridge.evaluate_and_execute(
                debate_id="d-1",
                debate_result=MockDebateResult(),
                confidence=0.85,
            )
        assert len(results) == 1
        assert results[0].success

    def test_evaluate_skips_disabled_rules(self, bridge):
        rule = bridge.register_action(
            name="disabled",
            condition=lambda r: True,
            action_type=ActionType.NOTIFICATION,
        )
        rule.enabled = False
        results = bridge.evaluate_and_execute(
            debate_id="d-1",
            debate_result=MockDebateResult(),
            confidence=0.85,
        )
        assert len(results) == 0

    def test_evaluate_skips_non_matching_rules(self, bridge):
        bridge.register_action(
            name="never_match",
            condition=lambda r: False,
            action_type=ActionType.NOTIFICATION,
        )
        results = bridge.evaluate_and_execute(
            debate_id="d-1",
            debate_result=MockDebateResult(),
            confidence=0.85,
        )
        assert len(results) == 0


class TestDefaultBridge:
    """Tests for create_default_bridge."""

    def test_default_bridge_has_rules(self, default_bridge):
        assert len(default_bridge.rules) >= 3

    def test_default_bridge_rule_names(self, default_bridge):
        names = {r.name for r in default_bridge.rules}
        assert "auto_verify_outcome" in names
        assert "notify_high_confidence" in names
        assert "improve_low_confidence" in names
        assert "escalate_gauntlet_failure" in names

    def test_auto_verify_matches_everything(self, default_bridge):
        verify_rule = next(r for r in default_bridge.rules if r.name == "auto_verify_outcome")
        assert verify_rule.matches({"confidence": 0.0})
        assert verify_rule.matches({"confidence": 1.0})

    def test_high_confidence_notification_threshold(self, default_bridge):
        notify_rule = next(r for r in default_bridge.rules if r.name == "notify_high_confidence")
        assert not notify_rule.matches({"confidence": 0.7, "consensus_reached": True})
        assert notify_rule.matches({"confidence": 0.9, "consensus_reached": True})
        assert not notify_rule.matches({"confidence": 0.9, "consensus_reached": False})

    def test_low_confidence_improvement(self, default_bridge):
        improve_rule = next(r for r in default_bridge.rules if r.name == "improve_low_confidence")
        assert improve_rule.matches({"confidence": 0.3, "consensus_reached": True})
        assert improve_rule.matches({"confidence": 0.8, "consensus_reached": False})
        assert not improve_rule.matches({"confidence": 0.8, "consensus_reached": True})


class TestActionExecution:
    """Tests for individual action type execution."""

    def test_workflow_disabled(self):
        config = BridgeConfig(enable_auto_workflows=False)
        bridge = ExecutionBridge(config=config)
        rule = ActionRule(
            name="test",
            condition=lambda r: True,
            action_type=ActionType.WORKFLOW,
            config={"workflow_template": "test"},
        )
        result = bridge._execute_workflow(rule, {"debate_id": "d-1"})
        assert not result.success
        assert "disabled" in result.detail

    def test_notification_disabled(self):
        config = BridgeConfig(enable_auto_notifications=False)
        bridge = ExecutionBridge(config=config)
        rule = ActionRule(
            name="test",
            condition=lambda r: True,
            action_type=ActionType.NOTIFICATION,
        )
        result = bridge._execute_notification(rule, {"debate_id": "d-1"})
        assert not result.success

    def test_outcome_verification_disabled(self):
        config = BridgeConfig(enable_outcome_verification=False)
        bridge = ExecutionBridge(config=config)
        rule = ActionRule(
            name="test",
            condition=lambda r: True,
            action_type=ActionType.OUTCOME_VERIFICATION,
        )
        result = bridge._execute_outcome_verification(rule, {"debate_id": "d-1"})
        assert not result.success

    def test_pr_creation_disabled_by_default(self):
        bridge = ExecutionBridge()
        rule = ActionRule(
            name="test",
            condition=lambda r: True,
            action_type=ActionType.PR_CREATION,
        )
        result = bridge._execute_pr_creation(rule, {"debate_id": "d-1"}, MockDebateResult())
        assert not result.success
        assert "disabled" in result.detail

    def test_webhook_no_url(self):
        bridge = ExecutionBridge()
        rule = ActionRule(
            name="test",
            condition=lambda r: True,
            action_type=ActionType.WEBHOOK,
        )
        result = bridge._execute_webhook(rule, {"debate_id": "d-1"})
        assert not result.success
        assert "webhook_url" in result.detail


class TestActionResult:
    """Tests for ActionResult serialization."""

    def test_to_dict(self):
        result = ActionResult(
            rule_name="test",
            action_type=ActionType.NOTIFICATION,
            success=True,
            detail="Sent",
        )
        d = result.to_dict()
        assert d["rule_name"] == "test"
        assert d["action_type"] == "notification"
        assert d["success"] is True


class TestBridgeConfig:
    """Tests for BridgeConfig defaults."""

    def test_defaults(self):
        config = BridgeConfig()
        assert config.enable_auto_workflows is True
        assert config.enable_auto_notifications is True
        assert config.enable_outcome_verification is True
        assert config.enable_pr_creation is False
        assert config.default_min_confidence == 0.7

    def test_custom_config(self):
        config = BridgeConfig(
            enable_pr_creation=True,
            pr_min_confidence=0.9,
        )
        assert config.enable_pr_creation is True
        assert config.pr_min_confidence == 0.9
