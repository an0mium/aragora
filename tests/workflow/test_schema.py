"""
Comprehensive tests for aragora.workflow.schema validation.

Tests cover: validation gates, cycle detection, unreachable steps,
resource limits, step config validation, transition validation,
condition safety, and Pydantic schema models.
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _minimal_workflow(**overrides):
    """Helper to build a minimal valid workflow dict."""
    base = {
        "id": "wf_test",
        "name": "Test Workflow",
        "steps": [
            {
                "id": "step1",
                "name": "Step 1",
                "step_type": "task",
                "config": {"task_type": "function"},
            },
        ],
        "transitions": [],
    }
    base.update(overrides)
    return base


def _two_step_workflow(**overrides):
    """Helper: workflow with two linearly connected steps."""
    base = {
        "id": "wf_two",
        "name": "Two Step",
        "steps": [
            {
                "id": "s1",
                "name": "S1",
                "step_type": "task",
                "config": {"task_type": "function"},
                "next_steps": ["s2"],
            },
            {"id": "s2", "name": "S2", "step_type": "task", "config": {"task_type": "function"}},
        ],
        "transitions": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# ValidationResult / ValidationMessage basics
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_initial_state(self):
        from aragora.workflow.schema import ValidationResult

        r = ValidationResult(valid=True)
        assert r.valid is True
        assert r.errors == []
        assert r.warnings == []

    def test_add_error_marks_invalid(self):
        from aragora.workflow.schema import ValidationResult

        r = ValidationResult(valid=True)
        r.add_error("bad", path="x", code="E1")
        assert r.valid is False
        assert len(r.errors) == 1
        assert r.errors[0].message == "bad"
        assert r.errors[0].path == "x"
        assert r.errors[0].code == "E1"

    def test_add_warning_keeps_valid(self):
        from aragora.workflow.schema import ValidationResult

        r = ValidationResult(valid=True)
        r.add_warning("warn")
        assert r.valid is True
        assert len(r.warnings) == 1

    def test_add_info(self):
        from aragora.workflow.schema import ValidationResult, ValidationSeverity

        r = ValidationResult(valid=True)
        r.add_info("info msg", path="p", code="I1")
        assert r.valid is True
        assert len(r.messages) == 1
        assert r.messages[0].severity == ValidationSeverity.INFO

    def test_validation_message_str(self):
        from aragora.workflow.schema import ValidationMessage, ValidationSeverity

        m = ValidationMessage(
            severity=ValidationSeverity.ERROR, message="oops", path="steps[0]", code="E"
        )
        s = str(m)
        assert "[ERROR]" in s
        assert "steps[0]" in s
        assert "oops" in s

    def test_validation_message_str_no_path(self):
        from aragora.workflow.schema import ValidationMessage, ValidationSeverity

        m = ValidationMessage(severity=ValidationSeverity.WARNING, message="hmm")
        s = str(m)
        assert "[WARNING]" in s
        assert "at " not in s


# ---------------------------------------------------------------------------
# Top-level structure validation
# ---------------------------------------------------------------------------


class TestStructureValidation:
    def test_valid_minimal_workflow(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        r = v.validate(_minimal_workflow())
        assert r.valid is True

    def test_missing_id(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow()
        wf["id"] = ""
        r = v.validate(wf)
        assert r.valid is False
        assert any(e.code == "MISSING_ID" for e in r.errors)

    def test_missing_name(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow()
        wf["name"] = ""
        r = v.validate(wf)
        assert r.valid is False
        assert any(e.code == "MISSING_NAME" for e in r.errors)

    def test_no_steps(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow()
        wf["steps"] = []
        r = v.validate(wf)
        assert r.valid is False
        assert any(e.code == "NO_STEPS" for e in r.errors)

    def test_steps_not_list(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow()
        wf["steps"] = "not_a_list"
        r = v.validate(wf)
        assert r.valid is False
        assert any(e.code == "INVALID_STEPS_TYPE" for e in r.errors)

    def test_too_many_steps(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator(max_steps=2)
        steps = [
            {
                "id": f"s{i}",
                "name": f"S{i}",
                "step_type": "task",
                "config": {"task_type": "function"},
            }
            for i in range(5)
        ]
        wf = _minimal_workflow(steps=steps)
        r = v.validate(wf)
        assert any(e.code == "TOO_MANY_STEPS" for e in r.errors)

    def test_too_many_transitions(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator(max_transitions=1)
        transitions = [
            {"id": "t1", "from_step": "step1", "to_step": "step1"},
            {"id": "t2", "from_step": "step1", "to_step": "step1"},
        ]
        wf = _minimal_workflow(transitions=transitions)
        r = v.validate(wf)
        assert any(e.code == "TOO_MANY_TRANSITIONS" for e in r.errors)

    def test_invalid_entry_step(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(entry_step="nonexistent")
        r = v.validate(wf)
        assert any(e.code == "INVALID_ENTRY_STEP" for e in r.errors)

    def test_valid_entry_step(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(entry_step="step1")
        r = v.validate(wf)
        assert r.valid is True


# ---------------------------------------------------------------------------
# Step validation
# ---------------------------------------------------------------------------


class TestStepValidation:
    def test_missing_step_id(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(
            steps=[{"name": "No ID", "step_type": "task", "config": {"task_type": "function"}}]
        )
        r = v.validate(wf)
        assert any(e.code == "MISSING_STEP_ID" for e in r.errors)

    def test_duplicate_step_id(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        steps = [
            {"id": "dup", "name": "A", "step_type": "task", "config": {"task_type": "function"}},
            {"id": "dup", "name": "B", "step_type": "task", "config": {"task_type": "function"}},
        ]
        wf = _minimal_workflow(steps=steps)
        r = v.validate(wf)
        assert any(e.code == "DUPLICATE_STEP_ID" for e in r.errors)

    def test_missing_step_name_is_warning(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(
            steps=[{"id": "s1", "step_type": "task", "config": {"task_type": "function"}}]
        )
        r = v.validate(wf)
        assert r.valid is True  # warning, not error
        assert any(w.code == "MISSING_STEP_NAME" for w in r.warnings)

    def test_invalid_step_type(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(steps=[{"id": "s1", "name": "S1", "step_type": "rocket_launch"}])
        r = v.validate(wf)
        assert any(e.code == "INVALID_STEP_TYPE" for e in r.errors)

    def test_all_valid_step_types(self):
        from aragora.workflow.schema import WorkflowValidator, VALID_STEP_TYPES

        v = WorkflowValidator()
        for stype in VALID_STEP_TYPES:
            config = (
                {"task_type": "function"}
                if stype == "task"
                else {"cases": []}
                if stype == "switch"
                else {}
            )
            wf = _minimal_workflow(
                steps=[{"id": "s1", "name": "S1", "step_type": stype, "config": config}]
            )
            r = v.validate(wf)
            assert not any(e.code == "INVALID_STEP_TYPE" for e in r.errors), (
                f"step type {stype} should be valid"
            )

    def test_task_missing_task_type_config(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(
            steps=[{"id": "s1", "name": "S1", "step_type": "task", "config": {}}]
        )
        r = v.validate(wf)
        assert any(e.code == "MISSING_REQUIRED_CONFIG" for e in r.errors)

    def test_switch_missing_cases_config(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(
            steps=[{"id": "s1", "name": "S1", "step_type": "switch", "config": {}}]
        )
        r = v.validate(wf)
        assert any(e.code == "MISSING_REQUIRED_CONFIG" for e in r.errors)

    def test_unknown_task_type_warning(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(
            steps=[
                {"id": "s1", "name": "S1", "step_type": "task", "config": {"task_type": "alien"}}
            ]
        )
        r = v.validate(wf)
        assert any(w.code == "UNKNOWN_TASK_TYPE" for w in r.warnings)

    def test_http_task_missing_url(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(
            steps=[{"id": "s1", "name": "S1", "step_type": "task", "config": {"task_type": "http"}}]
        )
        r = v.validate(wf)
        assert any(e.code == "MISSING_HTTP_URL" for e in r.errors)

    def test_http_task_with_url_ok(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(
            steps=[
                {
                    "id": "s1",
                    "name": "S1",
                    "step_type": "task",
                    "config": {"task_type": "http", "url": "https://example.com"},
                }
            ]
        )
        r = v.validate(wf)
        assert not any(e.code == "MISSING_HTTP_URL" for e in r.errors)

    def test_agent_step_incomplete_warning(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(
            steps=[{"id": "s1", "name": "S1", "step_type": "agent", "config": {}}]
        )
        r = v.validate(wf)
        assert any(w.code == "INCOMPLETE_AGENT_CONFIG" for w in r.warnings)

    def test_agent_step_with_agent_type_no_warning(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(
            steps=[
                {"id": "s1", "name": "S1", "step_type": "agent", "config": {"agent_type": "claude"}}
            ]
        )
        r = v.validate(wf)
        assert not any(w.code == "INCOMPLETE_AGENT_CONFIG" for w in r.warnings)

    def test_debate_step_no_topic_warning(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(
            steps=[{"id": "s1", "name": "S1", "step_type": "debate", "config": {}}]
        )
        r = v.validate(wf)
        assert any(w.code == "MISSING_DEBATE_TOPIC" for w in r.warnings)

    def test_quick_debate_step_with_question_no_warning(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(
            steps=[
                {
                    "id": "s1",
                    "name": "S1",
                    "step_type": "quick_debate",
                    "config": {"question": "What?"},
                }
            ]
        )
        r = v.validate(wf)
        assert not any(w.code == "MISSING_DEBATE_TOPIC" for w in r.warnings)

    def test_invalid_timeout_zero(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(
            steps=[
                {
                    "id": "s1",
                    "name": "S1",
                    "step_type": "agent",
                    "config": {"agent_type": "x"},
                    "timeout_seconds": 0,
                }
            ]
        )
        r = v.validate(wf)
        assert any(e.code == "INVALID_TIMEOUT" for e in r.errors)

    def test_invalid_timeout_negative(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(
            steps=[
                {
                    "id": "s1",
                    "name": "S1",
                    "step_type": "agent",
                    "config": {"agent_type": "x"},
                    "timeout_seconds": -5,
                }
            ]
        )
        r = v.validate(wf)
        assert any(e.code == "INVALID_TIMEOUT" for e in r.errors)

    def test_long_timeout_warning(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(
            steps=[
                {
                    "id": "s1",
                    "name": "S1",
                    "step_type": "agent",
                    "config": {"agent_type": "x"},
                    "timeout_seconds": 7200,
                }
            ]
        )
        r = v.validate(wf)
        assert any(w.code == "LONG_TIMEOUT" for w in r.warnings)

    def test_default_timeout_ok(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        # No timeout_seconds key uses default 120 which is fine
        wf = _minimal_workflow(
            steps=[{"id": "s1", "name": "S1", "step_type": "agent", "config": {"agent_type": "x"}}]
        )
        r = v.validate(wf)
        assert not any(e.code == "INVALID_TIMEOUT" for e in r.errors)


# ---------------------------------------------------------------------------
# Transition validation
# ---------------------------------------------------------------------------


class TestTransitionValidation:
    def test_valid_transition(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _two_step_workflow(
            transitions=[{"id": "t1", "from_step": "s1", "to_step": "s2", "condition": "True"}]
        )
        r = v.validate(wf)
        assert not any(
            e.code
            in ("MISSING_FROM_STEP", "MISSING_TO_STEP", "UNKNOWN_FROM_STEP", "UNKNOWN_TO_STEP")
            for e in r.errors
        )

    def test_missing_from_step(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _two_step_workflow(transitions=[{"id": "t1", "to_step": "s2"}])
        r = v.validate(wf)
        assert any(e.code == "MISSING_FROM_STEP" for e in r.errors)

    def test_missing_to_step(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _two_step_workflow(transitions=[{"id": "t1", "from_step": "s1"}])
        r = v.validate(wf)
        assert any(e.code == "MISSING_TO_STEP" for e in r.errors)

    def test_unknown_from_step(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _two_step_workflow(transitions=[{"id": "t1", "from_step": "ghost", "to_step": "s2"}])
        r = v.validate(wf)
        assert any(e.code == "UNKNOWN_FROM_STEP" for e in r.errors)

    def test_unknown_to_step(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _two_step_workflow(transitions=[{"id": "t1", "from_step": "s1", "to_step": "ghost"}])
        r = v.validate(wf)
        assert any(e.code == "UNKNOWN_TO_STEP" for e in r.errors)

    def test_duplicate_transition_id_warning(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _two_step_workflow(
            transitions=[
                {"id": "dup", "from_step": "s1", "to_step": "s2"},
                {"id": "dup", "from_step": "s2", "to_step": "s1"},
            ]
        )
        r = v.validate(wf)
        assert any(w.code == "DUPLICATE_TRANSITION_ID" for w in r.warnings)

    def test_invalid_condition_syntax(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _two_step_workflow(
            transitions=[{"id": "t1", "from_step": "s1", "to_step": "s2", "condition": "if ???"}]
        )
        r = v.validate(wf)
        assert any(e.code == "INVALID_CONDITION_SYNTAX" for e in r.errors)

    def test_valid_condition(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _two_step_workflow(
            transitions=[{"id": "t1", "from_step": "s1", "to_step": "s2", "condition": "x > 5"}]
        )
        r = v.validate(wf)
        assert not any(e.code == "INVALID_CONDITION_SYNTAX" for e in r.errors)

    def test_unsafe_condition_warning(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        for dangerous in ["__import__('os')", "exec('code')", "eval('x')", "open('f')"]:
            wf = _two_step_workflow(
                transitions=[
                    {"id": "t1", "from_step": "s1", "to_step": "s2", "condition": dangerous}
                ]
            )
            r = v.validate(wf)
            assert any(w.code == "UNSAFE_CONDITION" for w in r.warnings), (
                f"should warn for: {dangerous}"
            )


# ---------------------------------------------------------------------------
# Graph validation: unreachable steps and cycles
# ---------------------------------------------------------------------------


class TestGraphValidation:
    def test_unreachable_step_warning(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        steps = [
            {"id": "s1", "name": "S1", "step_type": "agent", "config": {"agent_type": "x"}},
            {"id": "s2", "name": "S2", "step_type": "agent", "config": {"agent_type": "x"}},
        ]
        # No transitions or next_steps connecting s1 to s2
        wf = _minimal_workflow(steps=steps)
        r = v.validate(wf)
        assert any(w.code == "UNREACHABLE_STEPS" for w in r.warnings)

    def test_all_reachable_via_next_steps(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        r = v.validate(_two_step_workflow())
        assert not any(w.code == "UNREACHABLE_STEPS" for w in r.warnings)

    def test_all_reachable_via_transitions(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        steps = [
            {"id": "s1", "name": "S1", "step_type": "agent", "config": {"agent_type": "x"}},
            {"id": "s2", "name": "S2", "step_type": "agent", "config": {"agent_type": "x"}},
        ]
        wf = _minimal_workflow(
            steps=steps, transitions=[{"id": "t1", "from_step": "s1", "to_step": "s2"}]
        )
        r = v.validate(wf)
        assert not any(w.code == "UNREACHABLE_STEPS" for w in r.warnings)

    def test_cycle_allowed_by_default(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()  # allow_cycles=True by default
        steps = [
            {
                "id": "s1",
                "name": "S1",
                "step_type": "agent",
                "config": {"agent_type": "x"},
                "next_steps": ["s2"],
            },
            {
                "id": "s2",
                "name": "S2",
                "step_type": "agent",
                "config": {"agent_type": "x"},
                "next_steps": ["s1"],
            },
        ]
        wf = _minimal_workflow(steps=steps)
        r = v.validate(wf)
        assert not any(e.code == "CYCLE_DETECTED" for e in r.errors)

    def test_cycle_detected_when_disallowed(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator(allow_cycles=False)
        steps = [
            {
                "id": "s1",
                "name": "S1",
                "step_type": "agent",
                "config": {"agent_type": "x"},
                "next_steps": ["s2"],
            },
            {
                "id": "s2",
                "name": "S2",
                "step_type": "agent",
                "config": {"agent_type": "x"},
                "next_steps": ["s1"],
            },
        ]
        wf = _minimal_workflow(steps=steps)
        r = v.validate(wf)
        assert any(e.code == "CYCLE_DETECTED" for e in r.errors)

    def test_self_loop_cycle(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator(allow_cycles=False)
        steps = [
            {
                "id": "s1",
                "name": "S1",
                "step_type": "agent",
                "config": {"agent_type": "x"},
                "next_steps": ["s1"],
            },
        ]
        wf = _minimal_workflow(steps=steps)
        r = v.validate(wf)
        assert any(e.code == "CYCLE_DETECTED" for e in r.errors)

    def test_no_cycle_linear(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator(allow_cycles=False)
        r = v.validate(_two_step_workflow())
        assert not any(e.code == "CYCLE_DETECTED" for e in r.errors)

    def test_longer_cycle(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator(allow_cycles=False)
        steps = [
            {
                "id": "a",
                "name": "A",
                "step_type": "agent",
                "config": {"agent_type": "x"},
                "next_steps": ["b"],
            },
            {
                "id": "b",
                "name": "B",
                "step_type": "agent",
                "config": {"agent_type": "x"},
                "next_steps": ["c"],
            },
            {
                "id": "c",
                "name": "C",
                "step_type": "agent",
                "config": {"agent_type": "x"},
                "next_steps": ["a"],
            },
        ]
        wf = _minimal_workflow(steps=steps)
        r = v.validate(wf)
        assert any(e.code == "CYCLE_DETECTED" for e in r.errors)

    def test_unknown_next_step_reference(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        steps = [
            {
                "id": "s1",
                "name": "S1",
                "step_type": "agent",
                "config": {"agent_type": "x"},
                "next_steps": ["phantom"],
            },
        ]
        wf = _minimal_workflow(steps=steps)
        r = v.validate(wf)
        assert any(e.code == "UNKNOWN_NEXT_STEP" for e in r.errors)

    def test_cycle_via_transitions(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator(allow_cycles=False)
        steps = [
            {"id": "s1", "name": "S1", "step_type": "agent", "config": {"agent_type": "x"}},
            {"id": "s2", "name": "S2", "step_type": "agent", "config": {"agent_type": "x"}},
        ]
        transitions = [
            {"id": "t1", "from_step": "s1", "to_step": "s2"},
            {"id": "t2", "from_step": "s2", "to_step": "s1"},
        ]
        wf = _minimal_workflow(steps=steps, transitions=transitions)
        r = v.validate(wf)
        assert any(e.code == "CYCLE_DETECTED" for e in r.errors)


# ---------------------------------------------------------------------------
# Resource limits validation
# ---------------------------------------------------------------------------


class TestLimitsValidation:
    def test_valid_limits(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(
            limits={"max_tokens": 5000, "max_cost_usd": 1.5, "timeout_seconds": 60}
        )
        r = v.validate(wf)
        assert r.valid is True

    def test_invalid_max_tokens_zero(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(limits={"max_tokens": 0})
        r = v.validate(wf)
        assert any(e.code == "INVALID_MAX_TOKENS" for e in r.errors)

    def test_invalid_max_tokens_negative(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(limits={"max_tokens": -10})
        r = v.validate(wf)
        assert any(e.code == "INVALID_MAX_TOKENS" for e in r.errors)

    def test_invalid_max_tokens_string(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(limits={"max_tokens": "many"})
        r = v.validate(wf)
        assert any(e.code == "INVALID_MAX_TOKENS" for e in r.errors)

    def test_high_max_tokens_warning(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(limits={"max_tokens": 2000000})
        r = v.validate(wf)
        assert any(w.code == "HIGH_MAX_TOKENS" for w in r.warnings)

    def test_invalid_max_cost_zero(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(limits={"max_cost_usd": 0})
        r = v.validate(wf)
        assert any(e.code == "INVALID_MAX_COST" for e in r.errors)

    def test_invalid_max_cost_negative(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(limits={"max_cost_usd": -5.0})
        r = v.validate(wf)
        assert any(e.code == "INVALID_MAX_COST" for e in r.errors)

    def test_high_max_cost_warning(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(limits={"max_cost_usd": 200.0})
        r = v.validate(wf)
        assert any(w.code == "HIGH_MAX_COST" for w in r.warnings)

    def test_invalid_timeout_zero(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(limits={"timeout_seconds": 0})
        r = v.validate(wf)
        assert any(e.code == "INVALID_TIMEOUT" for e in r.errors)

    def test_invalid_timeout_negative(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow(limits={"timeout_seconds": -1})
        r = v.validate(wf)
        assert any(e.code == "INVALID_TIMEOUT" for e in r.errors)

    def test_no_limits_ok(self):
        from aragora.workflow.schema import WorkflowValidator

        v = WorkflowValidator()
        wf = _minimal_workflow()
        r = v.validate(wf)
        assert r.valid is True


# ---------------------------------------------------------------------------
# validate_workflow convenience function (YAML/JSON string parsing)
# ---------------------------------------------------------------------------


class TestValidateWorkflowFunction:
    def test_dict_input(self):
        from aragora.workflow.schema import validate_workflow

        r = validate_workflow(_minimal_workflow())
        assert r.valid is True

    def test_yaml_string_input(self):
        import yaml
        from aragora.workflow.schema import validate_workflow

        wf = _minimal_workflow()
        yaml_str = yaml.safe_dump(wf)
        r = validate_workflow(yaml_str)
        assert r.valid is True

    def test_json_string_input(self):
        import json
        from aragora.workflow.schema import validate_workflow

        wf = _minimal_workflow()
        json_str = json.dumps(wf)
        r = validate_workflow(json_str)
        assert r.valid is True

    def test_invalid_yaml_string(self):
        from aragora.workflow.schema import validate_workflow

        r = validate_workflow("{{{invalid yaml")
        assert r.valid is False
        assert any(e.code == "PARSE_ERROR" for e in r.errors)

    def test_non_dict_yaml(self):
        from aragora.workflow.schema import validate_workflow

        r = validate_workflow("- just\n- a\n- list")
        assert r.valid is False
        assert any(e.code == "PARSE_ERROR" for e in r.errors)


# ---------------------------------------------------------------------------
# validate_workflow_file
# ---------------------------------------------------------------------------


class TestValidateWorkflowFile:
    def test_file_not_found(self):
        from aragora.workflow.schema import validate_workflow_file

        r = validate_workflow_file("/tmp/does_not_exist_workflow.yaml")
        assert r.valid is False
        assert any(e.code == "FILE_NOT_FOUND" for e in r.errors)

    def test_valid_file(self, tmp_path):
        import yaml
        from aragora.workflow.schema import validate_workflow_file

        wf = _minimal_workflow()
        p = tmp_path / "wf.yaml"
        p.write_text(yaml.safe_dump(wf))
        r = validate_workflow_file(str(p))
        assert r.valid is True


# ---------------------------------------------------------------------------
# VALID_STEP_TYPES / VALID_TASK_TYPES constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_valid_step_types_is_set(self):
        from aragora.workflow.schema import VALID_STEP_TYPES

        assert isinstance(VALID_STEP_TYPES, set)
        assert "agent" in VALID_STEP_TYPES
        assert "debate" in VALID_STEP_TYPES
        assert "task" in VALID_STEP_TYPES

    def test_valid_task_types_is_set(self):
        from aragora.workflow.schema import VALID_TASK_TYPES

        assert isinstance(VALID_TASK_TYPES, set)
        assert "function" in VALID_TASK_TYPES
        assert "http" in VALID_TASK_TYPES

    def test_step_type_count(self):
        from aragora.workflow.schema import VALID_STEP_TYPES

        # Ensure the set has a reasonable number of types
        assert len(VALID_STEP_TYPES) >= 10


# ---------------------------------------------------------------------------
# Pydantic schema models (if available)
# ---------------------------------------------------------------------------


class TestPydanticSchemas:
    def _skip_if_no_pydantic(self):
        from aragora.workflow.schema import PYDANTIC_AVAILABLE

        if not PYDANTIC_AVAILABLE:
            import pytest

            pytest.skip("Pydantic not available")

    def test_workflow_schema_valid(self):
        self._skip_if_no_pydantic()
        from aragora.workflow.schema import WorkflowSchema

        ws = WorkflowSchema(
            id="wf1",
            name="Test",
            steps=[{"id": "s1", "name": "Step1", "step_type": "agent"}],
        )
        assert ws.id == "wf1"
        assert len(ws.steps) == 1

    def test_workflow_schema_missing_id(self):
        self._skip_if_no_pydantic()
        import pytest
        from aragora.workflow.schema import WorkflowSchema

        with pytest.raises(Exception):
            WorkflowSchema(
                id="", name="Test", steps=[{"id": "s1", "name": "S", "step_type": "agent"}]
            )

    def test_workflow_schema_missing_name(self):
        self._skip_if_no_pydantic()
        import pytest
        from aragora.workflow.schema import WorkflowSchema

        with pytest.raises(Exception):
            WorkflowSchema(
                id="wf1", name="", steps=[{"id": "s1", "name": "S", "step_type": "agent"}]
            )

    def test_workflow_schema_no_steps(self):
        self._skip_if_no_pydantic()
        import pytest
        from aragora.workflow.schema import WorkflowSchema

        with pytest.raises(Exception):
            WorkflowSchema(id="wf1", name="Test", steps=[])

    def test_step_schema_invalid_type(self):
        self._skip_if_no_pydantic()
        import pytest
        from aragora.workflow.schema import StepSchema

        with pytest.raises(Exception):
            StepSchema(id="s1", step_type="invalid_type")

    def test_workflow_schema_invalid_entry_step(self):
        self._skip_if_no_pydantic()
        import pytest
        from aragora.workflow.schema import WorkflowSchema

        with pytest.raises(Exception):
            WorkflowSchema(
                id="wf1",
                name="Test",
                steps=[{"id": "s1", "name": "S", "step_type": "agent"}],
                entry_step="nonexistent",
            )

    def test_workflow_schema_transition_unknown_from(self):
        self._skip_if_no_pydantic()
        import pytest
        from aragora.workflow.schema import WorkflowSchema

        with pytest.raises(Exception):
            WorkflowSchema(
                id="wf1",
                name="Test",
                steps=[{"id": "s1", "name": "S", "step_type": "agent"}],
                transitions=[{"from_step": "ghost", "to_step": "s1"}],
            )

    def test_resource_limits_schema(self):
        self._skip_if_no_pydantic()
        from aragora.workflow.schema import ResourceLimitsSchema

        lim = ResourceLimitsSchema(max_tokens=5000, max_cost_usd=2.0)
        assert lim.max_tokens == 5000
        assert lim.max_cost_usd == 2.0
