"""Tests for the OpenRouter-backed RescuePlanner."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from aragora.swarm.rescue_planner import (
    ActionPlan,
    RescueAction,
    _parse_action_plan,
    plan_rescue,
)


class TestParseActionPlan:
    def test_valid_json(self) -> None:
        raw = json.dumps(
            {
                "action": "send_followup",
                "reason": "session stalled after test failure",
                "confidence": 0.85,
                "proposed_prompt": "Check the test output and fix the assertion.",
            }
        )
        plan = _parse_action_plan(raw)
        assert plan.action == "send_followup"
        assert plan.confidence == 0.85
        assert "stalled" in plan.reason

    def test_json_embedded_in_prose(self) -> None:
        raw = 'Here is my recommendation:\n{"action": "restart_from_state", "reason": "worker crashed", "confidence": 0.7}\nDone.'
        plan = _parse_action_plan(raw)
        assert plan.action == "restart_from_state"
        assert plan.confidence == 0.7

    def test_invalid_action_falls_back_to_escalate(self) -> None:
        raw = json.dumps({"action": "delete_everything", "reason": "bad", "confidence": 0.9})
        plan = _parse_action_plan(raw)
        assert plan.action == "escalate"

    def test_low_confidence_escalates(self) -> None:
        raw = json.dumps({"action": "send_followup", "reason": "maybe", "confidence": 0.1})
        plan = _parse_action_plan(raw)
        assert plan.action == "escalate"
        assert "Low confidence" in plan.reason

    def test_empty_response_escalates(self) -> None:
        plan = _parse_action_plan("")
        assert plan.action == "escalate"

    def test_non_json_escalates(self) -> None:
        plan = _parse_action_plan("I don't know what to do.")
        assert plan.action == "escalate"

    def test_escalate_at_low_confidence_is_kept(self) -> None:
        raw = json.dumps({"action": "escalate", "reason": "genuinely stuck", "confidence": 0.1})
        plan = _parse_action_plan(raw)
        assert plan.action == "escalate"
        assert plan.reason == "genuinely stuck"

    def test_proposed_prompt_truncated(self) -> None:
        raw = json.dumps(
            {
                "action": "send_followup",
                "reason": "needs fix",
                "confidence": 0.8,
                "proposed_prompt": "x" * 1000,
            }
        )
        plan = _parse_action_plan(raw)
        assert len(plan.proposed_prompt) <= 500


class TestPlanRescue:
    def test_no_context_escalates(self) -> None:
        plan = plan_rescue()
        assert plan.action == "escalate"
        assert "No context" in plan.reason

    def test_no_api_key_escalates(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            plan = plan_rescue(session_summary="worker stalled")
        assert plan.action == "escalate"
        assert "API key" in plan.reason

    def test_successful_call(self) -> None:
        mock_response = json.dumps(
            {
                "action": "send_followup",
                "reason": "test failure needs fix",
                "confidence": 0.9,
                "proposed_prompt": "Fix the failing assertion in test_foo.py",
            }
        )
        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
            patch(
                "aragora.swarm.rescue_planner._call_openrouter",
                return_value=mock_response,
            ),
        ):
            plan = plan_rescue(
                session_summary="Worker ran tests, 1 failed",
                blocker_evidence="AssertionError in test_foo.py:42",
            )
        assert plan.action == "send_followup"
        assert plan.confidence == 0.9

    def test_api_error_escalates(self) -> None:
        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
            patch(
                "aragora.swarm.rescue_planner._call_openrouter",
                side_effect=ConnectionError("network down"),
            ),
        ):
            plan = plan_rescue(session_summary="stuck")
        assert plan.action == "escalate"
        assert "ConnectionError" in plan.reason


class TestActionPlan:
    def test_to_dict(self) -> None:
        plan = ActionPlan(
            action="send_followup",
            reason="test",
            confidence=0.8,
        )
        d = plan.to_dict()
        assert d["action"] == "send_followup"
        assert d["confidence"] == 0.8


class TestRescueAction:
    def test_all_actions_have_values(self) -> None:
        assert len(RescueAction) == 7
        assert RescueAction.WAIT_FOR_CI.value == "wait_for_ci"
        assert RescueAction.ESCALATE.value == "escalate"
