"""
Tests for Receipt Workflow Triggers.

Covers:
- ReceiptWorkflowTrigger rule registration and management
- Rule evaluation against receipt data
- Priority ordering
- Default trigger rules
- Edge cases
"""

from __future__ import annotations

import pytest

from aragora.workflow.triggers import (
    ReceiptWorkflowTrigger,
    TriggerResult,
    TriggerRule,
    create_default_triggers,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def trigger():
    return ReceiptWorkflowTrigger()


@pytest.fixture
def sample_fail_receipt():
    return {
        "receipt_id": "REC-FAIL-001",
        "verdict": "FAIL",
        "confidence": 0.95,
        "robustness_score": 0.3,
        "vulnerabilities_found": 5,
    }


@pytest.fixture
def sample_pass_receipt():
    return {
        "receipt_id": "REC-PASS-001",
        "verdict": "PASS",
        "confidence": 0.95,
        "robustness_score": 0.9,
        "vulnerabilities_found": 0,
    }


@pytest.fixture
def sample_conditional_receipt():
    return {
        "receipt_id": "REC-COND-001",
        "verdict": "CONDITIONAL",
        "confidence": 0.7,
        "robustness_score": 0.65,
        "vulnerabilities_found": 2,
    }


# ============================================================================
# Registration Tests
# ============================================================================


class TestRuleRegistration:
    def test_register_rule(self, trigger):
        rule = trigger.register_rule(
            name="test_rule",
            condition=lambda r: True,
            workflow_template="test_workflow",
        )
        assert isinstance(rule, TriggerRule)
        assert rule.name == "test_rule"
        assert len(trigger.rules) == 1

    def test_register_multiple_rules(self, trigger):
        trigger.register_rule("r1", lambda r: True, "wf1")
        trigger.register_rule("r2", lambda r: True, "wf2")
        assert len(trigger.rules) == 2

    def test_rules_sorted_by_priority(self, trigger):
        trigger.register_rule("low", lambda r: True, "wf_low", priority=10)
        trigger.register_rule("high", lambda r: True, "wf_high", priority=100)
        trigger.register_rule("mid", lambda r: True, "wf_mid", priority=50)
        assert trigger.rules[0].name == "high"
        assert trigger.rules[1].name == "mid"
        assert trigger.rules[2].name == "low"

    def test_register_with_metadata(self, trigger):
        rule = trigger.register_rule(
            "meta",
            lambda r: True,
            "wf",
            metadata={"team": "security"},
        )
        assert rule.metadata == {"team": "security"}

    def test_register_disabled_rule(self, trigger):
        rule = trigger.register_rule(
            "disabled",
            lambda r: True,
            "wf",
            enabled=False,
        )
        assert rule.enabled is False


# ============================================================================
# Rule Management Tests
# ============================================================================


class TestRuleManagement:
    def test_remove_rule(self, trigger):
        trigger.register_rule("to_remove", lambda r: True, "wf")
        assert trigger.remove_rule("to_remove") is True
        assert len(trigger.rules) == 0

    def test_remove_nonexistent_rule(self, trigger):
        assert trigger.remove_rule("missing") is False

    def test_enable_rule(self, trigger):
        trigger.register_rule("r1", lambda r: True, "wf", enabled=False)
        assert trigger.enable_rule("r1") is True
        assert trigger.rules[0].enabled is True

    def test_disable_rule(self, trigger):
        trigger.register_rule("r1", lambda r: True, "wf", enabled=True)
        assert trigger.disable_rule("r1") is True
        assert trigger.rules[0].enabled is False

    def test_enable_nonexistent(self, trigger):
        assert trigger.enable_rule("missing") is False

    def test_disable_nonexistent(self, trigger):
        assert trigger.disable_rule("missing") is False


# ============================================================================
# Evaluation Tests
# ============================================================================


class TestEvaluation:
    def test_matching_rule_returns_result(self, trigger, sample_fail_receipt):
        trigger.register_rule(
            "fail_check",
            lambda r: r.get("verdict") == "FAIL",
            "escalation",
        )
        results = trigger.evaluate(sample_fail_receipt)
        assert len(results) == 1
        assert results[0].rule_name == "fail_check"
        assert results[0].workflow_template == "escalation"
        assert results[0].matched is True
        assert results[0].receipt_id == "REC-FAIL-001"

    def test_no_matching_rules(self, trigger, sample_pass_receipt):
        trigger.register_rule(
            "fail_only",
            lambda r: r.get("verdict") == "FAIL",
            "escalation",
        )
        results = trigger.evaluate(sample_pass_receipt)
        assert len(results) == 0

    def test_multiple_matches(self, trigger, sample_fail_receipt):
        trigger.register_rule("r1", lambda r: r.get("verdict") == "FAIL", "wf1")
        trigger.register_rule("r2", lambda r: r.get("vulnerabilities_found", 0) > 3, "wf2")
        results = trigger.evaluate(sample_fail_receipt)
        assert len(results) == 2

    def test_disabled_rules_skipped(self, trigger, sample_fail_receipt):
        trigger.register_rule(
            "disabled_rule",
            lambda r: True,
            "wf",
            enabled=False,
        )
        results = trigger.evaluate(sample_fail_receipt)
        assert len(results) == 0

    def test_priority_order_in_results(self, trigger, sample_fail_receipt):
        trigger.register_rule("low", lambda r: True, "wf_low", priority=10)
        trigger.register_rule("high", lambda r: True, "wf_high", priority=100)
        results = trigger.evaluate(sample_fail_receipt)
        assert results[0].rule_name == "high"
        assert results[1].rule_name == "low"

    def test_condition_error_handled_gracefully(self, trigger, sample_fail_receipt):
        def bad_condition(r):
            raise ValueError("boom")

        trigger.register_rule("bad", bad_condition, "wf")
        results = trigger.evaluate(sample_fail_receipt)
        assert len(results) == 0

    def test_evaluate_first_returns_highest_priority(self, trigger, sample_fail_receipt):
        trigger.register_rule("low", lambda r: True, "wf_low", priority=10)
        trigger.register_rule("high", lambda r: True, "wf_high", priority=100)
        result = trigger.evaluate_first(sample_fail_receipt)
        assert result is not None
        assert result.rule_name == "high"

    def test_evaluate_first_returns_none_when_no_match(self, trigger, sample_pass_receipt):
        trigger.register_rule("fail_only", lambda r: r.get("verdict") == "FAIL", "wf")
        result = trigger.evaluate_first(sample_pass_receipt)
        assert result is None

    def test_result_to_dict(self, trigger, sample_fail_receipt):
        trigger.register_rule("r1", lambda r: True, "wf", metadata={"key": "val"})
        results = trigger.evaluate(sample_fail_receipt)
        d = results[0].to_dict()
        assert d["rule_name"] == "r1"
        assert d["workflow_template"] == "wf"
        assert d["matched"] is True
        assert d["metadata"] == {"key": "val"}


# ============================================================================
# Default Triggers Tests
# ============================================================================


class TestDefaultTriggers:
    def test_creates_trigger_with_rules(self):
        trigger = create_default_triggers()
        assert len(trigger.rules) == 3

    def test_fail_triggers_escalation(self, sample_fail_receipt):
        trigger = create_default_triggers()
        results = trigger.evaluate(sample_fail_receipt)
        templates = [r.workflow_template for r in results]
        assert "escalation_review" in templates

    def test_conditional_triggers_review(self, sample_conditional_receipt):
        trigger = create_default_triggers()
        results = trigger.evaluate(sample_conditional_receipt)
        templates = [r.workflow_template for r in results]
        assert "conditional_review" in templates

    def test_high_confidence_pass_triggers_archive(self, sample_pass_receipt):
        trigger = create_default_triggers()
        results = trigger.evaluate(sample_pass_receipt)
        templates = [r.workflow_template for r in results]
        assert "auto_archive" in templates

    def test_low_confidence_pass_no_archive(self):
        trigger = create_default_triggers()
        receipt = {
            "receipt_id": "REC-LP",
            "verdict": "PASS",
            "confidence": 0.6,
        }
        results = trigger.evaluate(receipt)
        templates = [r.workflow_template for r in results]
        assert "auto_archive" not in templates
