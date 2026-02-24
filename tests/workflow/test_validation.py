"""Tests for workflow validation: cycle detection, reachability analysis, and orphan transitions.

These tests target ``aragora.workflow.validation.validate_workflow`` directly,
covering the graph algorithms that prevent runtime failures in workflow execution.
"""

from __future__ import annotations

import pytest

from aragora.workflow.validation import validate_workflow, ValidationResult


# ---------------------------------------------------------------------------
# Lightweight fakes for WorkflowDefinition / StepDefinition / TransitionRule
# ---------------------------------------------------------------------------

class _FakeStep:
    """Minimal stub matching the attributes validation.py reads."""

    def __init__(self, id: str, step_type: str = "action", next_steps: list[str] | None = None, config: dict | None = None):
        self.id = id
        self.step_type = step_type
        self.next_steps = next_steps or []
        self.config = config or {}


class _FakeTransition:
    """Minimal stub matching TransitionRule fields used by validate_workflow."""

    def __init__(self, from_step: str, to_step: str):
        self.from_step = from_step
        self.to_step = to_step


class _FakeDefinition:
    """Minimal stub for WorkflowDefinition."""

    def __init__(
        self,
        steps: list[_FakeStep],
        transitions: list[_FakeTransition] | None = None,
        entry_step: str | None = None,
    ):
        self.steps = steps
        self.transitions = transitions or []
        self.entry_step = entry_step or (steps[0].id if steps else None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _codes(result: ValidationResult) -> list[str]:
    return [m.code for m in result.messages]


def _error_codes(result: ValidationResult) -> list[str]:
    return [m.code for m in result.errors]


def _warning_codes(result: ValidationResult) -> list[str]:
    return [m.code for m in result.warnings]


def _info_codes(result: ValidationResult) -> list[str]:
    return [m.code for m in result.info]


# ===========================================================================
# 1. Empty / trivial definitions
# ===========================================================================

class TestEmptyWorkflow:
    def test_no_steps_is_error(self):
        defn = _FakeDefinition(steps=[])
        result = validate_workflow(defn)
        assert not result.valid
        assert "NO_STEPS" in _error_codes(result)

    def test_single_step_is_valid(self):
        defn = _FakeDefinition(steps=[_FakeStep("a")])
        result = validate_workflow(defn)
        assert result.valid
        assert len(result.errors) == 0


# ===========================================================================
# 2. Entry step validation
# ===========================================================================

class TestEntryStep:
    def test_missing_entry_step_errors(self):
        defn = _FakeDefinition(
            steps=[_FakeStep("a"), _FakeStep("b")],
            entry_step="nonexistent",
        )
        result = validate_workflow(defn)
        assert not result.valid
        assert "MISSING_ENTRY" in _error_codes(result)


# ===========================================================================
# 3. Reachability analysis (BFS from entry)
# ===========================================================================

class TestReachability:
    def test_linear_chain_all_reachable(self):
        """A → B → C: all reachable from A."""
        defn = _FakeDefinition(
            steps=[_FakeStep("a", next_steps=["b"]), _FakeStep("b", next_steps=["c"]), _FakeStep("c")],
            entry_step="a",
        )
        result = validate_workflow(defn)
        assert result.valid
        assert "UNREACHABLE_STEP" not in _warning_codes(result)

    def test_disconnected_step_warns(self):
        """A → B, C (disconnected): C should be warned as unreachable."""
        defn = _FakeDefinition(
            steps=[_FakeStep("a", next_steps=["b"]), _FakeStep("b"), _FakeStep("c")],
            entry_step="a",
        )
        result = validate_workflow(defn)
        assert "UNREACHABLE_STEP" in _warning_codes(result)
        unreachable_msgs = [m for m in result.warnings if m.code == "UNREACHABLE_STEP"]
        assert any("'c'" in m.message for m in unreachable_msgs)

    def test_multiple_disconnected_steps(self):
        """A → B; C, D both disconnected."""
        defn = _FakeDefinition(
            steps=[
                _FakeStep("a", next_steps=["b"]),
                _FakeStep("b"),
                _FakeStep("c"),
                _FakeStep("d"),
            ],
            entry_step="a",
        )
        result = validate_workflow(defn)
        unreachable = [m for m in result.warnings if m.code == "UNREACHABLE_STEP"]
        unreachable_ids = {m.step_id for m in unreachable}
        assert unreachable_ids == {"c", "d"}

    def test_reachable_via_transition_rule(self):
        """B reachable via TransitionRule, not next_steps."""
        defn = _FakeDefinition(
            steps=[_FakeStep("a"), _FakeStep("b")],
            transitions=[_FakeTransition("a", "b")],
            entry_step="a",
        )
        result = validate_workflow(defn)
        assert "UNREACHABLE_STEP" not in _warning_codes(result)

    def test_diamond_all_reachable(self):
        """A → B, A → C, B → D, C → D: all reachable."""
        defn = _FakeDefinition(
            steps=[
                _FakeStep("a", next_steps=["b", "c"]),
                _FakeStep("b", next_steps=["d"]),
                _FakeStep("c", next_steps=["d"]),
                _FakeStep("d"),
            ],
            entry_step="a",
        )
        result = validate_workflow(defn)
        assert "UNREACHABLE_STEP" not in _warning_codes(result)

    def test_reachability_with_cycle_still_warns_orphan(self):
        """A → B → A (cycle), C (disconnected): C should still warn."""
        defn = _FakeDefinition(
            steps=[
                _FakeStep("a", next_steps=["b"]),
                _FakeStep("b", next_steps=["a"]),
                _FakeStep("c"),
            ],
            entry_step="a",
        )
        result = validate_workflow(defn)
        unreachable = [m for m in result.warnings if m.code == "UNREACHABLE_STEP"]
        assert any("'c'" in m.message for m in unreachable)


# ===========================================================================
# 4. Cycle detection (DFS back-edge)
# ===========================================================================

class TestCycleDetection:
    def test_no_cycle_linear(self):
        """A → B → C: no cycle."""
        defn = _FakeDefinition(
            steps=[_FakeStep("a", next_steps=["b"]), _FakeStep("b", next_steps=["c"]), _FakeStep("c")],
        )
        result = validate_workflow(defn)
        assert "CYCLE_DETECTED" not in _error_codes(result)

    def test_simple_cycle_detected(self):
        """A → B → C → A: cycle at C → A."""
        defn = _FakeDefinition(
            steps=[
                _FakeStep("a", next_steps=["b"]),
                _FakeStep("b", next_steps=["c"]),
                _FakeStep("c", next_steps=["a"]),
            ],
        )
        result = validate_workflow(defn)
        assert not result.valid
        assert "CYCLE_DETECTED" in _error_codes(result)

    def test_self_loop_detected(self):
        """A → A: self-loop."""
        defn = _FakeDefinition(
            steps=[_FakeStep("a", next_steps=["a"])],
        )
        result = validate_workflow(defn)
        assert not result.valid
        assert "CYCLE_DETECTED" in _error_codes(result)

    def test_two_node_cycle(self):
        """A → B → A: two-node cycle."""
        defn = _FakeDefinition(
            steps=[
                _FakeStep("a", next_steps=["b"]),
                _FakeStep("b", next_steps=["a"]),
            ],
        )
        result = validate_workflow(defn)
        assert not result.valid
        assert "CYCLE_DETECTED" in _error_codes(result)

    def test_cycle_via_transition_rule(self):
        """Cycle introduced via TransitionRule rather than next_steps."""
        defn = _FakeDefinition(
            steps=[_FakeStep("a"), _FakeStep("b"), _FakeStep("c")],
            transitions=[
                _FakeTransition("a", "b"),
                _FakeTransition("b", "c"),
                _FakeTransition("c", "a"),
            ],
        )
        result = validate_workflow(defn)
        assert not result.valid
        assert "CYCLE_DETECTED" in _error_codes(result)

    def test_loop_step_type_is_allowed(self):
        """Loop-type steps forming a cycle should produce info, not error."""
        defn = _FakeDefinition(
            steps=[
                _FakeStep("a", step_type="action", next_steps=["b"]),
                _FakeStep("b", step_type="loop", next_steps=["a"]),
            ],
        )
        result = validate_workflow(defn)
        # The back edge b→a: step_map[b] has step_type=="loop" → LOOP_CYCLE info
        assert "LOOP_CYCLE" in _info_codes(result)
        # Should NOT produce an error for the loop step's cycle
        cycle_errors = [m for m in result.errors if m.code == "CYCLE_DETECTED" and m.step_id == "b"]
        assert len(cycle_errors) == 0

    def test_multiple_independent_cycles(self):
        """Two disconnected cycles: A→B→A and C→D→C."""
        defn = _FakeDefinition(
            steps=[
                _FakeStep("a", next_steps=["b"]),
                _FakeStep("b", next_steps=["a"]),
                _FakeStep("c", next_steps=["d"]),
                _FakeStep("d", next_steps=["c"]),
            ],
        )
        result = validate_workflow(defn)
        assert not result.valid
        cycle_errors = [m for m in result.errors if m.code == "CYCLE_DETECTED"]
        assert len(cycle_errors) >= 2

    def test_nested_cycle_in_subgraph(self):
        """A → B → C → D → B: inner cycle B→C→D→B."""
        defn = _FakeDefinition(
            steps=[
                _FakeStep("a", next_steps=["b"]),
                _FakeStep("b", next_steps=["c"]),
                _FakeStep("c", next_steps=["d"]),
                _FakeStep("d", next_steps=["b"]),
            ],
        )
        result = validate_workflow(defn)
        assert not result.valid
        assert "CYCLE_DETECTED" in _error_codes(result)

    def test_dag_with_converging_paths_no_cycle(self):
        """A → B, A → C, B → D, C → D: DAG, no cycle."""
        defn = _FakeDefinition(
            steps=[
                _FakeStep("a", next_steps=["b", "c"]),
                _FakeStep("b", next_steps=["d"]),
                _FakeStep("c", next_steps=["d"]),
                _FakeStep("d"),
            ],
        )
        result = validate_workflow(defn)
        assert "CYCLE_DETECTED" not in _error_codes(result)


# ===========================================================================
# 5. Orphan transitions
# ===========================================================================

class TestOrphanTransitions:
    def test_valid_transitions_no_orphans(self):
        defn = _FakeDefinition(
            steps=[_FakeStep("a"), _FakeStep("b")],
            transitions=[_FakeTransition("a", "b")],
        )
        result = validate_workflow(defn)
        assert "ORPHAN_TRANSITION" not in _error_codes(result)

    def test_orphan_source_transition(self):
        """Transition from nonexistent step."""
        defn = _FakeDefinition(
            steps=[_FakeStep("a"), _FakeStep("b")],
            transitions=[_FakeTransition("ghost", "b")],
        )
        result = validate_workflow(defn)
        assert not result.valid
        assert "ORPHAN_TRANSITION" in _error_codes(result)

    def test_orphan_target_transition(self):
        """Transition to nonexistent step."""
        defn = _FakeDefinition(
            steps=[_FakeStep("a"), _FakeStep("b")],
            transitions=[_FakeTransition("a", "ghost")],
        )
        result = validate_workflow(defn)
        assert not result.valid
        assert "ORPHAN_TRANSITION" in _error_codes(result)

    def test_both_orphan_source_and_target(self):
        """Transition where both from and to are nonexistent."""
        defn = _FakeDefinition(
            steps=[_FakeStep("a")],
            transitions=[_FakeTransition("ghost1", "ghost2")],
        )
        result = validate_workflow(defn)
        assert not result.valid
        orphan_errors = [m for m in result.errors if m.code == "ORPHAN_TRANSITION"]
        assert len(orphan_errors) == 2


# ===========================================================================
# 6. ValidationResult API
# ===========================================================================

class TestValidationResultAPI:
    def test_to_dict_structure(self):
        defn = _FakeDefinition(
            steps=[_FakeStep("a", next_steps=["b"]), _FakeStep("b")],
            entry_step="a",
        )
        result = validate_workflow(defn)
        d = result.to_dict()
        assert "valid" in d
        assert "error_count" in d
        assert "warning_count" in d
        assert "messages" in d
        assert isinstance(d["messages"], list)

    def test_errors_property(self):
        defn = _FakeDefinition(steps=[])
        result = validate_workflow(defn)
        assert len(result.errors) > 0
        assert all(m.level == "error" for m in result.errors)

    def test_warnings_property(self):
        defn = _FakeDefinition(
            steps=[_FakeStep("a"), _FakeStep("b")],
            entry_step="a",
        )
        result = validate_workflow(defn)
        assert all(m.level == "warning" for m in result.warnings)


# ===========================================================================
# 7. Combined scenarios
# ===========================================================================

class TestCombinedScenarios:
    def test_cycle_and_orphan_together(self):
        """Workflow with both a cycle and orphan transitions."""
        defn = _FakeDefinition(
            steps=[
                _FakeStep("a", next_steps=["b"]),
                _FakeStep("b", next_steps=["a"]),
            ],
            transitions=[_FakeTransition("ghost", "a")],
        )
        result = validate_workflow(defn)
        assert not result.valid
        assert "CYCLE_DETECTED" in _error_codes(result)
        assert "ORPHAN_TRANSITION" in _error_codes(result)

    def test_complex_valid_workflow(self):
        """Multi-step valid workflow: no cycles, all reachable, no orphans."""
        defn = _FakeDefinition(
            steps=[
                _FakeStep("start", next_steps=["validate"]),
                _FakeStep("validate", next_steps=["process"]),
                _FakeStep("process"),
                _FakeStep("notify"),
                _FakeStep("end"),
            ],
            transitions=[
                _FakeTransition("process", "notify"),
                _FakeTransition("notify", "end"),
            ],
            entry_step="start",
        )
        result = validate_workflow(defn)
        assert result.valid
        assert len(result.errors) == 0
        assert "UNREACHABLE_STEP" not in _warning_codes(result)
