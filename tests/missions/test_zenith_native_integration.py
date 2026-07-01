"""Native Zenith-style mission control tests.

These tests pin the small public surface needed to absorb Zenith's control loop
without importing Zenith as a runtime dependency.
"""

from __future__ import annotations

from aragora.missions import (
    ContractAssertion,
    ContractState,
    Feature,
    Handoff,
    MissionOrchestrator,
    MissionState,
    Status,
    inject_validation_features,
)
from aragora.missions.boundary import (
    MissionBoundaryAction,
    MissionBoundaryController,
    MissionBoundaryEvent,
    apply_boundary_decision,
    layered_validation_kinds,
    validate_contract_coverage,
)


def test_mission_state_persists_contract_and_native_task_shape(tmp_path) -> None:
    state = MissionState(
        mission_id="zenith-mission",
        goal="ship with evidence",
        milestones=["m1"],
        contract=[
            ContractAssertion(
                assertion_id="VAL-API",
                statement="API behavior remains compatible",
                evidence_expectations=["pytest tests/api"],
            )
        ],
        contract_state={"VAL-API": ContractState(status="pending")},
        role_specs={"implementer": {"agent": "codex", "model": "default"}},
        skill_seeds=[{"id": "debug-imports", "body": "Use targeted import checks first."}],
        decision_trace=[{"action": "continue", "reason": "seeded"}],
        features=[
            Feature(
                id="impl-api",
                description="Implement API behavior",
                milestone="m1",
                kind="work",
                fulfills=["VAL-API"],
                metadata={"role_spec_id": "implementer", "skill_seed_ids": ["debug-imports"]},
            ),
            Feature(
                id="validate-api",
                description="Validate API behavior",
                milestone="m1",
                kind="validate",
                skill="validator",
                fulfills=["VAL-API"],
            ),
            Feature(
                id="gate-api",
                description="Seal API milestone",
                milestone="m1",
                kind="gate",
                fulfills=["VAL-API"],
            ),
        ],
    )

    path = tmp_path / "mission.json"
    state.save(path)

    restored = MissionState.load(path)

    assert restored.contract[0].assertion_id == "VAL-API"
    assert restored.contract_state["VAL-API"].status == "pending"
    assert restored.role_specs["implementer"]["agent"] == "codex"
    assert restored.skill_seeds[0]["id"] == "debug-imports"
    assert restored.decision_trace[0]["action"] == "continue"
    assert [feature.kind for feature in restored.features] == ["work", "validate", "gate"]


def test_contract_coverage_requires_exactly_one_active_work_owner() -> None:
    state = MissionState(
        mission_id="m",
        goal="cover assertions",
        contract=[ContractAssertion("VAL-A", "A must hold")],
    )

    assert [err.code for err in validate_contract_coverage(state)] == ["uncovered_assertion"]

    state.features.append(
        Feature(id="impl-a", description="Implement A", milestone="m1", fulfills=["VAL-A"])
    )
    assert validate_contract_coverage(state) == []

    state.features.append(
        Feature(id="impl-a-2", description="Implement A twice", milestone="m1", fulfills=["VAL-A"])
    )
    assert [err.code for err in validate_contract_coverage(state)] == ["over_covered_assertion"]

    state.get("impl-a-2").kind = "validate"
    state.features.append(
        Feature(
            id="gate-a",
            description="Gate A",
            milestone="m1",
            kind="gate",
            fulfills=["VAL-A"],
        )
    )
    assert validate_contract_coverage(state) == []


def test_layered_validation_injection_adds_optional_fidelity_and_gate() -> None:
    state = MissionState(
        mission_id="m",
        goal="validate UI carefully",
        milestones=["ui"],
        features=[
            Feature(
                id="impl-ui",
                description="Implement UI",
                milestone="ui",
                status=Status.COMPLETED,
                fulfills=["VAL-UI"],
                metadata={"paths": ["aragora/live"], "surface": "ui"},
            )
        ],
    )

    injected = inject_validation_features(
        state,
        milestone="ui",
        validation_kinds=layered_validation_kinds(state, "ui"),
        include_gate=True,
    )

    assert [feature.id for feature in injected] == [
        "validate-ui-automated",
        "validate-ui-review",
        "validate-ui-fidelity",
        "gate-ui",
    ]
    assert [feature.kind for feature in injected] == ["validate", "validate", "validate", "gate"]
    assert state.get("gate-ui").preconditions == [
        "feature:validate-ui-automated",
        "feature:validate-ui-review",
        "feature:validate-ui-fidelity",
    ]
    assert state.get("validate-ui-fidelity").metadata["validation_kind"] == "fidelity"


def test_boundary_controller_retries_transient_failure_then_parks() -> None:
    state = MissionState(
        mission_id="m",
        goal="retry transient failures",
        features=[Feature(id="impl", description="Implement", milestone="m1")],
    )
    controller = MissionBoundaryController(max_retries=2)

    first = controller.evaluate(
        state,
        MissionBoundaryEvent(kind="worker_failed", feature_id="impl", reason="test failed"),
    )
    assert first.action == MissionBoundaryAction.RETRY

    apply_boundary_decision(state, first)
    assert state.get("impl").status == Status.PENDING
    assert state.get("impl").retry_count == 1

    second = controller.evaluate(
        state,
        MissionBoundaryEvent(kind="worker_failed", feature_id="impl", reason="test failed again"),
    )
    assert second.action == MissionBoundaryAction.PARK

    apply_boundary_decision(state, second)
    assert state.get("impl").status == Status.BLOCKED
    assert state.decision_trace[-1]["action"] == "park"


def test_boundary_controller_parks_terminal_and_operator_tier_immediately() -> None:
    state = MissionState(
        mission_id="m",
        goal="respect settlement boundaries",
        features=[Feature(id="impl", description="Implement", milestone="m1")],
    )
    controller = MissionBoundaryController(operator_tier=3)

    terminal = controller.evaluate(
        state,
        MissionBoundaryEvent(
            kind="worker_failed",
            feature_id="impl",
            terminal=True,
            reason="operator decision required",
        ),
    )
    assert terminal.action == MissionBoundaryAction.PARK

    tier = controller.evaluate(
        state,
        MissionBoundaryEvent(kind="feature_completed", feature_id="impl", risk_tier=3),
    )
    assert tier.action == MissionBoundaryAction.PARK
    assert "tier-3" in tier.reason


def test_validation_failure_reopens_parent_and_adds_bounded_followup() -> None:
    state = MissionState(
        mission_id="m",
        goal="repair failed validation",
        milestones=["m1"],
        contract=[ContractAssertion("VAL-A", "A must hold")],
        features=[
            Feature(
                id="impl-a",
                description="Implement A",
                milestone="m1",
                status=Status.COMPLETED,
                fulfills=["VAL-A"],
                metadata={"paths": ["aragora/missions/state.py"]},
            ),
            Feature(
                id="validate-m1-automated",
                description="Validate A",
                milestone="m1",
                kind="validate",
                skill="validator",
                metadata={"validation_for": "m1", "validates": ["impl-a"]},
                fulfills=["VAL-A"],
            ),
        ],
    )
    controller = MissionBoundaryController()

    decision = controller.evaluate(
        state,
        MissionBoundaryEvent(
            kind="validation_failed",
            feature_id="validate-m1-automated",
            reason="regression failed",
            failed_assertions=["VAL-A"],
        ),
    )
    assert decision.action == MissionBoundaryAction.PATCH_PLAN

    apply_boundary_decision(state, decision)

    assert state.get("impl-a").status == Status.PENDING
    assert state.get("validate-m1-automated").status == Status.BLOCKED
    followup = state.get("repair-validate-m1-automated-val-a")
    assert followup.kind == "work"
    assert followup.fulfills == ["VAL-A"]
    assert followup.metadata["paths"] == ["aragora/missions/state.py"]
    assert "regression failed" in followup.description


def test_orchestrator_runs_native_contract_through_validators_and_gate(tmp_path) -> None:
    path = tmp_path / "mission.json"
    MissionState(
        mission_id="m",
        goal="ship with boundary control",
        milestones=["ui"],
        contract=[ContractAssertion("VAL-UI", "UI behavior is preserved")],
        contract_state={"VAL-UI": ContractState()},
        features=[
            Feature(
                id="impl-ui",
                description="Implement UI",
                milestone="ui",
                fulfills=["VAL-UI"],
                metadata={"paths": ["aragora/live"], "surface": "ui"},
            )
        ],
    ).save(path)
    seen: list[str] = []

    def dispatch(feature: Feature) -> Handoff:
        seen.append(feature.id)
        return Handoff(success=True)

    done, total = MissionOrchestrator(path).run(dispatch, max_ticks=10)
    final = MissionState.load(path)

    assert (done, total) == (5, 5)
    assert seen == [
        "impl-ui",
        "validate-ui-automated",
        "validate-ui-review",
        "validate-ui-fidelity",
        "gate-ui",
    ]
    assert final.get("gate-ui").status == Status.COMPLETED
    assert final.decision_trace[0]["action"] == "add_validator"
    assert final.contract_state["VAL-UI"].status == "passed"


def test_orchestrator_native_validation_failure_patches_plan(tmp_path) -> None:
    path = tmp_path / "mission.json"
    MissionState(
        mission_id="m",
        goal="repair failed validation",
        milestones=["m1"],
        contract=[ContractAssertion("VAL-A", "A must hold")],
        features=[
            Feature(
                id="impl-a",
                description="Implement A",
                milestone="m1",
                status=Status.COMPLETED,
                fulfills=["VAL-A"],
                metadata={"paths": ["aragora/missions/state.py"]},
            ),
            Feature(
                id="validate-m1-automated",
                description="Validate A",
                milestone="m1",
                kind="validate",
                skill="validator",
                metadata={"validation_for": "m1", "validates": ["impl-a"]},
                fulfills=["VAL-A"],
            ),
        ],
    ).save(path)

    def dispatch(feature: Feature) -> Handoff:
        assert feature.id == "validate-m1-automated"
        return Handoff(success=False, blocked_reason="regression failed")

    assert MissionOrchestrator(path).tick(dispatch) is True
    final = MissionState.load(path)

    assert final.get("impl-a").status == Status.PENDING
    assert final.get("validate-m1-automated").status == Status.BLOCKED
    assert final.get("repair-validate-m1-automated-val-a").fulfills == ["VAL-A"]
    assert final.decision_trace[-1]["action"] == "patch_plan"
