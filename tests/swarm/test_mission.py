"""Tests for aragora.swarm.mission — lineage and gate primitives."""

from __future__ import annotations

import pytest

from aragora.swarm.mission import (
    GateEvaluation,
    GateType,
    GateVerdict,
    MissionContextPolicy,
    MissionEnvelope,
    MissionStage,
    RepairPolicy,
    TranscriptAllowance,
    default_context_policy,
    mission_lineage_payload,
    normalize_context_policies,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestGateType:
    def test_values(self):
        assert GateType.DRAFT_READY == "draft_ready"
        assert GateType.DISPATCH_READY == "dispatch_ready"
        assert GateType.MILESTONE_READY == "milestone_ready"
        assert GateType.PUBLISH_READY == "publish_ready"

    def test_is_str(self):
        for member in GateType:
            assert isinstance(member, str)

    def test_all_members_unique(self):
        values = [m.value for m in GateType]
        assert len(values) == len(set(values))

    def test_membership(self):
        assert "draft_ready" in GateType._value2member_map_
        assert "publish_ready" in GateType._value2member_map_


class TestGateVerdict:
    def test_values(self):
        assert GateVerdict.PASS == "pass"
        assert GateVerdict.BLOCKED == "blocked"
        assert GateVerdict.NEEDS_HUMAN == "needs_human"

    def test_is_str(self):
        for member in GateVerdict:
            assert isinstance(member, str)

    def test_all_members_present(self):
        assert len(list(GateVerdict)) == 3


class TestTranscriptAllowance:
    def test_values(self):
        assert TranscriptAllowance.NONE == "none"
        assert TranscriptAllowance.SUMMARY_ONLY == "summary_only"
        assert TranscriptAllowance.RAW_ALLOWED == "raw_allowed"

    def test_is_str(self):
        for member in TranscriptAllowance:
            assert isinstance(member, str)

    def test_all_members_present(self):
        assert len(list(TranscriptAllowance)) == 3


# ---------------------------------------------------------------------------
# MissionEnvelope tests
# ---------------------------------------------------------------------------


class TestMissionEnvelope:
    def test_defaults(self):
        env = MissionEnvelope()
        assert env.mission_id == ""
        assert env.roadmap_refs == []
        assert env.goal_summary == ""
        assert env.assertion_ids == []
        assert env.evidence_expectations == []

    def test_construction_with_values(self):
        env = MissionEnvelope(
            mission_id="m-001",
            roadmap_refs=["RS-01", "RS-02"],
            goal_summary="Improve throughput",
            assertion_ids=["A1", "A2"],
            evidence_expectations=["receipt", "validation_command"],
        )
        assert env.mission_id == "m-001"
        assert env.roadmap_refs == ["RS-01", "RS-02"]
        assert env.goal_summary == "Improve throughput"

    def test_to_dict_round_trip(self):
        env = MissionEnvelope(
            mission_id=" m-002 ",
            roadmap_refs=["R1", "R2"],
            goal_summary="  goal  ",
            assertion_ids=["A", "B"],
            evidence_expectations=["receipt"],
        )
        d = env.to_dict()
        assert d["mission_id"] == "m-002"
        assert d["goal_summary"] == "goal"
        assert d["roadmap_refs"] == ["R1", "R2"]

    def test_from_dict_round_trip(self):
        original = MissionEnvelope(
            mission_id="m-003",
            roadmap_refs=["R1"],
            goal_summary="Some goal",
            assertion_ids=["A3"],
            evidence_expectations=["evidence_1"],
        )
        restored = MissionEnvelope.from_dict(original.to_dict())
        assert restored.mission_id == original.mission_id
        assert restored.roadmap_refs == original.roadmap_refs
        assert restored.goal_summary == original.goal_summary
        assert restored.assertion_ids == original.assertion_ids
        assert restored.evidence_expectations == original.evidence_expectations

    def test_from_dict_none(self):
        env = MissionEnvelope.from_dict(None)
        assert env.mission_id == ""
        assert env.roadmap_refs == []

    def test_from_dict_empty(self):
        env = MissionEnvelope.from_dict({})
        assert env.mission_id == ""
        assert env.goal_summary == ""

    def test_to_dict_deduplicates_lists(self):
        env = MissionEnvelope(
            mission_id="m-x",
            roadmap_refs=["R1", "R1", "R2"],
            assertion_ids=["A", "A"],
        )
        d = env.to_dict()
        assert d["roadmap_refs"] == ["R1", "R2"]
        assert d["assertion_ids"] == ["A"]

    def test_to_dict_strips_whitespace_in_lists(self):
        env = MissionEnvelope(roadmap_refs=["  R1 ", " R2  "])
        d = env.to_dict()
        assert d["roadmap_refs"] == ["R1", "R2"]

    def test_to_dict_filters_empty_strings_in_lists(self):
        env = MissionEnvelope(assertion_ids=["", "  ", "A"])
        d = env.to_dict()
        assert d["assertion_ids"] == ["A"]

    def test_from_dict_missing_keys(self):
        env = MissionEnvelope.from_dict({"mission_id": "m-x"})
        assert env.mission_id == "m-x"
        assert env.roadmap_refs == []
        assert env.assertion_ids == []

    def test_from_dict_none_values_treated_as_empty(self):
        env = MissionEnvelope.from_dict({"mission_id": None, "roadmap_refs": None})
        assert env.mission_id == ""
        assert env.roadmap_refs == []


# ---------------------------------------------------------------------------
# RepairPolicy tests
# ---------------------------------------------------------------------------


class TestRepairPolicy:
    def test_defaults(self):
        rp = RepairPolicy()
        assert rp.max_repair_rounds == 2
        assert rp.max_validator_rounds == 3
        assert rp.max_stage_wall_time_minutes == 90
        assert rp.escalate_after_terminal_classes == []

    def test_construction_with_values(self):
        rp = RepairPolicy(
            max_repair_rounds=5,
            max_validator_rounds=6,
            max_stage_wall_time_minutes=120,
            escalate_after_terminal_classes=["auth_failure", "oom"],
        )
        assert rp.max_repair_rounds == 5
        assert rp.max_validator_rounds == 6
        assert rp.max_stage_wall_time_minutes == 120
        assert rp.escalate_after_terminal_classes == ["auth_failure", "oom"]

    def test_to_dict_round_trip(self):
        rp = RepairPolicy(
            max_repair_rounds=3,
            max_validator_rounds=4,
            max_stage_wall_time_minutes=60,
            escalate_after_terminal_classes=["timeout"],
        )
        d = rp.to_dict()
        assert d["max_repair_rounds"] == 3
        assert d["max_validator_rounds"] == 4
        assert d["max_stage_wall_time_minutes"] == 60
        assert d["escalate_after_terminal_classes"] == ["timeout"]

    def test_from_dict_round_trip(self):
        original = RepairPolicy(
            max_repair_rounds=1,
            max_validator_rounds=2,
            max_stage_wall_time_minutes=45,
            escalate_after_terminal_classes=["crash"],
        )
        restored = RepairPolicy.from_dict(original.to_dict())
        assert restored.max_repair_rounds == original.max_repair_rounds
        assert restored.max_validator_rounds == original.max_validator_rounds
        assert restored.max_stage_wall_time_minutes == original.max_stage_wall_time_minutes
        assert restored.escalate_after_terminal_classes == original.escalate_after_terminal_classes

    def test_from_dict_none(self):
        rp = RepairPolicy.from_dict(None)
        assert rp.max_repair_rounds == 2
        assert rp.max_validator_rounds == 3

    def test_from_dict_empty(self):
        rp = RepairPolicy.from_dict({})
        assert rp.max_repair_rounds == 2
        assert rp.max_stage_wall_time_minutes == 90

    def test_to_dict_clamps_negative_to_zero(self):
        rp = RepairPolicy(
            max_repair_rounds=-1, max_validator_rounds=-5, max_stage_wall_time_minutes=-10
        )
        d = rp.to_dict()
        assert d["max_repair_rounds"] == 0
        assert d["max_validator_rounds"] == 0
        assert d["max_stage_wall_time_minutes"] == 0

    def test_to_dict_deduplicates_terminal_classes(self):
        rp = RepairPolicy(escalate_after_terminal_classes=["oom", "oom", "crash"])
        d = rp.to_dict()
        assert d["escalate_after_terminal_classes"] == ["oom", "crash"]

    def test_from_dict_missing_keys(self):
        rp = RepairPolicy.from_dict({"max_repair_rounds": 7})
        assert rp.max_repair_rounds == 7
        assert rp.max_validator_rounds == 3  # default

    def test_from_dict_none_values_use_defaults(self):
        rp = RepairPolicy.from_dict({"max_repair_rounds": None, "max_validator_rounds": None})
        assert rp.max_repair_rounds == 2
        assert rp.max_validator_rounds == 3


# ---------------------------------------------------------------------------
# MissionStage tests
# ---------------------------------------------------------------------------


class TestMissionStage:
    def test_defaults(self):
        stage = MissionStage()
        assert stage.stage_id == ""
        assert stage.mission_id == ""
        assert stage.title == ""
        assert stage.assertion_ids == []
        assert stage.file_scope == []
        assert stage.validation_command == ""
        assert stage.acceptance_criteria == []
        assert isinstance(stage.repair_policy, RepairPolicy)

    def test_construction_with_values(self):
        rp = RepairPolicy(max_repair_rounds=4)
        stage = MissionStage(
            stage_id="s-001",
            mission_id="m-001",
            title="Implement feature X",
            assertion_ids=["A1"],
            file_scope=["aragora/feature.py"],
            validation_command="pytest tests/test_feature.py",
            acceptance_criteria=["All tests pass"],
            repair_policy=rp,
        )
        assert stage.stage_id == "s-001"
        assert stage.mission_id == "m-001"
        assert stage.title == "Implement feature X"
        assert stage.repair_policy.max_repair_rounds == 4

    def test_to_dict_round_trip(self):
        stage = MissionStage(
            stage_id="s-002",
            mission_id="m-002",
            title="  Fix bug  ",
            file_scope=["aragora/bug.py"],
            validation_command="  pytest tests/  ",
            acceptance_criteria=["No regressions"],
        )
        d = stage.to_dict()
        assert d["stage_id"] == "s-002"
        assert d["title"] == "Fix bug"
        assert d["validation_command"] == "pytest tests/"
        assert "repair_policy" in d
        assert isinstance(d["repair_policy"], dict)

    def test_from_dict_round_trip(self):
        original = MissionStage(
            stage_id="s-003",
            mission_id="m-003",
            title="Stage Three",
            assertion_ids=["A3"],
            file_scope=["aragora/mod.py"],
            validation_command="make test",
            acceptance_criteria=["Green"],
            repair_policy=RepairPolicy(max_repair_rounds=1),
        )
        restored = MissionStage.from_dict(original.to_dict())
        assert restored.stage_id == original.stage_id
        assert restored.mission_id == original.mission_id
        assert restored.title == original.title
        assert restored.assertion_ids == original.assertion_ids
        assert restored.file_scope == original.file_scope
        assert restored.validation_command == original.validation_command
        assert restored.acceptance_criteria == original.acceptance_criteria
        assert restored.repair_policy.max_repair_rounds == original.repair_policy.max_repair_rounds

    def test_from_dict_none(self):
        stage = MissionStage.from_dict(None)
        assert stage.stage_id == ""
        assert stage.mission_id == ""

    def test_from_dict_empty(self):
        stage = MissionStage.from_dict({})
        assert stage.title == ""
        assert stage.file_scope == []
        assert isinstance(stage.repair_policy, RepairPolicy)

    def test_to_dict_deduplicates_lists(self):
        stage = MissionStage(
            stage_id="s-x",
            assertion_ids=["A", "A", "B"],
            file_scope=["f.py", "f.py"],
            acceptance_criteria=["c", "c"],
        )
        d = stage.to_dict()
        assert d["assertion_ids"] == ["A", "B"]
        assert d["file_scope"] == ["f.py"]
        assert d["acceptance_criteria"] == ["c"]

    def test_from_dict_nested_repair_policy(self):
        stage = MissionStage.from_dict(
            {
                "stage_id": "s-nested",
                "repair_policy": {"max_repair_rounds": 9, "max_validator_rounds": 7},
            }
        )
        assert stage.repair_policy.max_repair_rounds == 9
        assert stage.repair_policy.max_validator_rounds == 7

    def test_from_dict_none_repair_policy_uses_default(self):
        stage = MissionStage.from_dict({"stage_id": "s-no-rp", "repair_policy": None})
        assert stage.repair_policy.max_repair_rounds == 2

    def test_from_dict_missing_keys(self):
        stage = MissionStage.from_dict({"stage_id": "s-partial"})
        assert stage.stage_id == "s-partial"
        assert stage.mission_id == ""
        assert stage.acceptance_criteria == []


# ---------------------------------------------------------------------------
# MissionContextPolicy tests
# ---------------------------------------------------------------------------


class TestMissionContextPolicy:
    def test_construction_required_role(self):
        policy = MissionContextPolicy(role="worker")
        assert policy.role == "worker"
        assert policy.allowed_artifact_classes == []
        assert policy.max_source_count == 0
        assert policy.max_chars == 0
        assert policy.freshness_ttl_seconds == 0
        assert policy.transcript_allowance == TranscriptAllowance.NONE.value
        assert policy.required_sources == []
        assert policy.forbidden_sources == []

    def test_is_resolvable_false_when_defaults(self):
        policy = MissionContextPolicy(role="worker")
        assert policy.is_resolvable() is False

    def test_is_resolvable_false_missing_role(self):
        policy = MissionContextPolicy(
            role="",
            allowed_artifact_classes=["receipt"],
            max_source_count=5,
            max_chars=1000,
            freshness_ttl_seconds=60,
        )
        assert policy.is_resolvable() is False

    def test_is_resolvable_false_no_artifact_classes(self):
        policy = MissionContextPolicy(
            role="worker",
            allowed_artifact_classes=[],
            max_source_count=5,
            max_chars=1000,
            freshness_ttl_seconds=60,
        )
        assert policy.is_resolvable() is False

    def test_is_resolvable_false_zero_source_count(self):
        policy = MissionContextPolicy(
            role="worker",
            allowed_artifact_classes=["receipt"],
            max_source_count=0,
            max_chars=1000,
            freshness_ttl_seconds=60,
        )
        assert policy.is_resolvable() is False

    def test_is_resolvable_false_zero_max_chars(self):
        policy = MissionContextPolicy(
            role="worker",
            allowed_artifact_classes=["receipt"],
            max_source_count=5,
            max_chars=0,
            freshness_ttl_seconds=60,
        )
        assert policy.is_resolvable() is False

    def test_is_resolvable_true_when_fully_populated(self):
        policy = MissionContextPolicy(
            role="worker",
            allowed_artifact_classes=["receipt"],
            max_source_count=5,
            max_chars=1000,
            freshness_ttl_seconds=60,
            transcript_allowance=TranscriptAllowance.NONE.value,
        )
        assert policy.is_resolvable() is True

    def test_is_resolvable_true_with_zero_freshness(self):
        # freshness_ttl_seconds >= 0 is the check, so 0 is valid
        policy = MissionContextPolicy(
            role="auditor",
            allowed_artifact_classes=["evidence"],
            max_source_count=3,
            max_chars=500,
            freshness_ttl_seconds=0,
        )
        assert policy.is_resolvable() is True

    def test_to_dict_round_trip(self):
        policy = MissionContextPolicy(
            role="  validator  ",
            allowed_artifact_classes=["receipt", "summary"],
            max_source_count=6,
            max_chars=12000,
            freshness_ttl_seconds=900,
            transcript_allowance=TranscriptAllowance.SUMMARY_ONLY.value,
            required_sources=["validation_command"],
            forbidden_sources=["raw_worker_transcript"],
        )
        d = policy.to_dict()
        assert d["role"] == "validator"
        assert d["max_source_count"] == 6
        assert d["max_chars"] == 12000
        assert d["transcript_allowance"] == "summary_only"

    def test_from_dict_round_trip(self):
        original = MissionContextPolicy(
            role="worker",
            allowed_artifact_classes=["mission_stage", "swarm_spec"],
            max_source_count=8,
            max_chars=24000,
            freshness_ttl_seconds=3600,
            transcript_allowance=TranscriptAllowance.NONE.value,
            required_sources=["aragora/feature.py"],
            forbidden_sources=["validator_private_notes"],
        )
        restored = MissionContextPolicy.from_dict(original.to_dict())
        assert restored.role == original.role
        assert restored.allowed_artifact_classes == original.allowed_artifact_classes
        assert restored.max_source_count == original.max_source_count
        assert restored.max_chars == original.max_chars
        assert restored.freshness_ttl_seconds == original.freshness_ttl_seconds
        assert restored.transcript_allowance == original.transcript_allowance
        assert restored.required_sources == original.required_sources
        assert restored.forbidden_sources == original.forbidden_sources

    def test_from_dict_none(self):
        policy = MissionContextPolicy.from_dict(None)
        assert policy.role == "worker"

    def test_from_dict_empty(self):
        policy = MissionContextPolicy.from_dict({})
        assert policy.role == "worker"
        assert policy.max_source_count == 0
        assert policy.transcript_allowance == TranscriptAllowance.NONE.value

    def test_from_dict_missing_role_defaults_to_worker(self):
        policy = MissionContextPolicy.from_dict({"max_chars": 5000})
        assert policy.role == "worker"
        assert policy.max_chars == 5000

    def test_from_dict_empty_role_defaults_to_worker(self):
        policy = MissionContextPolicy.from_dict({"role": "  ", "max_chars": 1000})
        assert policy.role == "worker"

    def test_to_dict_clamps_negative_ints_to_zero(self):
        policy = MissionContextPolicy(
            role="worker",
            max_source_count=-5,
            max_chars=-100,
            freshness_ttl_seconds=-300,
        )
        d = policy.to_dict()
        assert d["max_source_count"] == 0
        assert d["max_chars"] == 0
        assert d["freshness_ttl_seconds"] == 0

    def test_to_dict_empty_transcript_allowance_falls_back_to_none(self):
        policy = MissionContextPolicy(role="worker", transcript_allowance="")
        d = policy.to_dict()
        assert d["transcript_allowance"] == TranscriptAllowance.NONE.value

    def test_to_dict_deduplicates_sources(self):
        policy = MissionContextPolicy(
            role="validator",
            allowed_artifact_classes=["a", "a", "b"],
            required_sources=["s1", "s1"],
            forbidden_sources=["f1", "f1"],
        )
        d = policy.to_dict()
        assert d["allowed_artifact_classes"] == ["a", "b"]
        assert d["required_sources"] == ["s1"]
        assert d["forbidden_sources"] == ["f1"]

    def test_from_dict_none_transcript_allowance_falls_back_to_none(self):
        policy = MissionContextPolicy.from_dict({"role": "worker", "transcript_allowance": None})
        assert policy.transcript_allowance == TranscriptAllowance.NONE.value


# ---------------------------------------------------------------------------
# GateEvaluation tests
# ---------------------------------------------------------------------------


class TestGateEvaluation:
    def test_construction_minimal(self):
        ge = GateEvaluation(gate_type=GateType.DRAFT_READY, verdict=GateVerdict.PASS)
        assert ge.gate_type == "draft_ready"
        assert ge.verdict == "pass"
        assert ge.mission_id == ""
        assert ge.stage_id == ""
        assert ge.assertion_ids == []
        assert ge.failure_classes == []
        assert ge.repair_eligible is False
        assert ge.required_evidence == []
        assert ge.notes == ""

    def test_construction_full(self):
        ge = GateEvaluation(
            gate_type=GateType.DISPATCH_READY,
            verdict=GateVerdict.BLOCKED,
            mission_id="m-001",
            stage_id="s-001",
            assertion_ids=["A1"],
            failure_classes=["test_failure"],
            repair_eligible=True,
            required_evidence=["receipt"],
            notes="Missing evidence",
        )
        assert ge.verdict == GateVerdict.BLOCKED
        assert ge.repair_eligible is True
        assert ge.notes == "Missing evidence"

    def test_to_dict_round_trip(self):
        ge = GateEvaluation(
            gate_type="dispatch_ready",
            verdict="blocked",
            mission_id=" m-002 ",
            notes="  some notes  ",
        )
        d = ge.to_dict()
        assert d["gate_type"] == "dispatch_ready"
        assert d["verdict"] == "blocked"
        assert d["mission_id"] == "m-002"
        assert d["notes"] == "some notes"

    def test_from_dict_round_trip(self):
        # Use plain strings so the round-trip stays stable through to_dict/from_dict.
        # Passing enum members through str() produces "GateType.X" (Python 3.11
        # behaviour), so plain string values are the canonical form for round-trips.
        original = GateEvaluation(
            gate_type="milestone_ready",
            verdict="needs_human",
            mission_id="m-003",
            stage_id="s-003",
            assertion_ids=["A3"],
            failure_classes=["ambiguous_result"],
            repair_eligible=False,
            required_evidence=["human_review"],
            notes="Needs review",
        )
        restored = GateEvaluation.from_dict(original.to_dict())
        assert restored.gate_type == original.gate_type
        assert restored.verdict == original.verdict
        assert restored.mission_id == original.mission_id
        assert restored.stage_id == original.stage_id
        assert restored.assertion_ids == original.assertion_ids
        assert restored.failure_classes == original.failure_classes
        assert restored.repair_eligible == original.repair_eligible
        assert restored.required_evidence == original.required_evidence
        assert restored.notes == original.notes

    def test_from_dict_none(self):
        ge = GateEvaluation.from_dict(None)
        assert ge.gate_type == ""
        assert ge.verdict == ""
        assert ge.repair_eligible is False

    def test_from_dict_empty(self):
        ge = GateEvaluation.from_dict({})
        assert ge.gate_type == ""
        assert ge.verdict == ""
        assert ge.failure_classes == []

    def test_to_dict_repair_eligible_bool(self):
        ge_true = GateEvaluation(gate_type="pass", verdict="pass", repair_eligible=True)
        ge_false = GateEvaluation(gate_type="pass", verdict="pass", repair_eligible=False)
        assert ge_true.to_dict()["repair_eligible"] is True
        assert ge_false.to_dict()["repair_eligible"] is False

    def test_from_dict_repair_eligible_truthy_values(self):
        ge = GateEvaluation.from_dict({"gate_type": "g", "verdict": "v", "repair_eligible": 1})
        assert ge.repair_eligible is True

    def test_from_dict_repair_eligible_falsy_values(self):
        ge = GateEvaluation.from_dict({"gate_type": "g", "verdict": "v", "repair_eligible": 0})
        assert ge.repair_eligible is False

    def test_to_dict_deduplicates_lists(self):
        ge = GateEvaluation(
            gate_type="g",
            verdict="v",
            assertion_ids=["A", "A", "B"],
            failure_classes=["f", "f"],
            required_evidence=["e", "e"],
        )
        d = ge.to_dict()
        assert d["assertion_ids"] == ["A", "B"]
        assert d["failure_classes"] == ["f"]
        assert d["required_evidence"] == ["e"]

    def test_from_dict_missing_optional_keys(self):
        ge = GateEvaluation.from_dict({"gate_type": "publish_ready", "verdict": "pass"})
        assert ge.gate_type == "publish_ready"
        assert ge.verdict == "pass"
        assert ge.mission_id == ""
        assert ge.assertion_ids == []


# ---------------------------------------------------------------------------
# default_context_policy tests
# ---------------------------------------------------------------------------


class TestDefaultContextPolicy:
    def test_worker_role_defaults(self):
        policy = default_context_policy("worker")
        assert policy.role == "worker"
        assert "mission_envelope" in policy.allowed_artifact_classes
        assert "swarm_spec" in policy.allowed_artifact_classes
        assert policy.max_chars == 24000
        assert policy.freshness_ttl_seconds == 3600
        assert policy.transcript_allowance == TranscriptAllowance.NONE.value
        assert "raw_worker_transcript" in policy.forbidden_sources
        assert "validator_private_notes" in policy.forbidden_sources

    def test_validator_role_defaults(self):
        policy = default_context_policy("validator")
        assert policy.role == "validator"
        assert "receipt" in policy.allowed_artifact_classes
        assert "validation_command" in policy.allowed_artifact_classes
        assert policy.max_chars == 16000
        assert policy.freshness_ttl_seconds == 900
        assert policy.transcript_allowance == TranscriptAllowance.SUMMARY_ONLY.value
        assert "raw_worker_transcript" in policy.forbidden_sources

    def test_unknown_role_falls_back_to_worker(self):
        policy = default_context_policy("reviewer")
        assert policy.role == "worker"

    def test_auditor_role_falls_back_to_worker(self):
        policy = default_context_policy("auditor")
        assert policy.role == "worker"

    def test_empty_role_falls_back_to_worker(self):
        policy = default_context_policy("")
        assert policy.role == "worker"

    def test_whitespace_role_falls_back_to_worker(self):
        policy = default_context_policy("   ")
        assert policy.role == "worker"

    def test_worker_max_source_count_min_4_no_scope(self):
        policy = default_context_policy("worker", file_scope=[])
        # max(4, min(12, 0 + 4)) = max(4, 4) = 4
        assert policy.max_source_count == 4

    def test_worker_max_source_count_scales_with_scope(self):
        scope = ["a.py", "b.py", "c.py", "d.py", "e.py", "f.py", "g.py", "h.py", "i.py"]
        policy = default_context_policy("worker", file_scope=scope)
        # max(4, min(12, 9+4)) = max(4, min(12, 13)) = max(4, 12) = 12
        assert policy.max_source_count == 12

    def test_worker_required_sources_capped_at_6(self):
        scope = ["a.py", "b.py", "c.py", "d.py", "e.py", "f.py", "g.py", "h.py"]
        policy = default_context_policy("worker", file_scope=scope)
        assert len(policy.required_sources) == 6

    def test_worker_required_sources_empty_when_no_scope(self):
        policy = default_context_policy("worker", file_scope=None)
        assert policy.required_sources == []

    def test_validator_max_source_count_min_4_no_scope_evidence(self):
        policy = default_context_policy("validator", file_scope=[], evidence_expectations=[])
        # max(4, min(8, 0 + max(1, 0))) = max(4, min(8, 1)) = max(4, 1) = 4
        assert policy.max_source_count == 4

    def test_validator_required_sources_use_evidence(self):
        policy = default_context_policy("validator", evidence_expectations=["receipt", "summary"])
        assert "receipt" in policy.required_sources
        assert "summary" in policy.required_sources

    def test_validator_required_sources_default_to_validation_command(self):
        policy = default_context_policy("validator", evidence_expectations=[])
        assert policy.required_sources == ["validation_command"]

    def test_role_is_case_insensitive(self):
        policy = default_context_policy("VALIDATOR")
        assert policy.role == "validator"

    def test_returns_mission_context_policy_instance(self):
        policy = default_context_policy("worker")
        assert isinstance(policy, MissionContextPolicy)

    def test_worker_policy_is_resolvable(self):
        policy = default_context_policy("worker", file_scope=["mod.py"])
        assert policy.is_resolvable() is True

    def test_validator_policy_is_resolvable(self):
        policy = default_context_policy("validator")
        assert policy.is_resolvable() is True


# ---------------------------------------------------------------------------
# normalize_context_policies tests
# ---------------------------------------------------------------------------


class TestNormalizeContextPolicies:
    def test_none_payload_returns_defaults_for_both_roles(self):
        result = normalize_context_policies(None)
        assert set(result.keys()) == {"worker", "validator"}
        assert result["worker"]["role"] == "worker"
        assert result["validator"]["role"] == "validator"

    def test_empty_payload_returns_defaults(self):
        result = normalize_context_policies({})
        assert set(result.keys()) == {"worker", "validator"}

    def test_only_known_roles_in_output(self):
        result = normalize_context_policies({"auditor": {"role": "auditor"}})
        # auditor is not a handled role; result only has worker and validator
        assert set(result.keys()) == {"worker", "validator"}

    def test_valid_worker_policy_used_as_is(self):
        payload = {
            "worker": {
                "role": "worker",
                "allowed_artifact_classes": ["mission_stage"],
                "max_source_count": 5,
                "max_chars": 8000,
                "freshness_ttl_seconds": 1800,
                "transcript_allowance": "none",
            }
        }
        result = normalize_context_policies(payload)
        assert result["worker"]["max_chars"] == 8000
        assert result["worker"]["max_source_count"] == 5

    def test_valid_validator_policy_used_as_is(self):
        payload = {
            "validator": {
                "role": "validator",
                "allowed_artifact_classes": ["receipt"],
                "max_source_count": 3,
                "max_chars": 6000,
                "freshness_ttl_seconds": 600,
                "transcript_allowance": "summary_only",
            }
        }
        result = normalize_context_policies(payload)
        assert result["validator"]["max_chars"] == 6000
        assert result["validator"]["transcript_allowance"] == "summary_only"

    def test_non_mapping_worker_replaced_with_default(self):
        result = normalize_context_policies({"worker": "bad_value"})
        assert result["worker"]["role"] == "worker"
        # default max_chars for worker is 24000
        assert result["worker"]["max_chars"] == 24000

    def test_non_mapping_validator_replaced_with_default(self):
        result = normalize_context_policies({"validator": 42})
        assert result["validator"]["role"] == "validator"

    def test_file_scope_passed_to_defaults(self):
        scope = ["a.py", "b.py", "c.py"]
        result = normalize_context_policies({}, file_scope=scope)
        worker_policy = result["worker"]
        # required_sources should include up to 6 items from scope
        assert worker_policy["required_sources"] == scope

    def test_evidence_expectations_passed_to_validator_default(self):
        evidence = ["receipt", "validation_command"]
        result = normalize_context_policies({}, evidence_expectations=evidence)
        validator_policy = result["validator"]
        assert "receipt" in validator_policy["required_sources"]

    def test_returns_serialized_dicts_not_objects(self):
        result = normalize_context_policies(None)
        for role_policy in result.values():
            assert isinstance(role_policy, dict)

    def test_partial_payload_fills_missing_role(self):
        payload = {
            "worker": {
                "role": "worker",
                "allowed_artifact_classes": ["mission_stage"],
                "max_source_count": 5,
                "max_chars": 8000,
                "freshness_ttl_seconds": 1800,
            }
        }
        result = normalize_context_policies(payload)
        # validator should still be present (filled with default)
        assert "validator" in result
        assert result["validator"]["role"] == "validator"


# ---------------------------------------------------------------------------
# mission_lineage_payload tests
# ---------------------------------------------------------------------------


class TestMissionLineagePayload:
    def test_defaults_all_empty(self):
        payload = mission_lineage_payload()
        assert payload == {
            "mission_id": "",
            "stage_id": "",
            "assertion_ids": [],
            "roadmap_refs": [],
            "evidence_expectations": [],
        }

    def test_with_all_values(self):
        payload = mission_lineage_payload(
            mission_id="m-001",
            stage_id="s-001",
            assertion_ids=["A1", "A2"],
            roadmap_refs=["RS-01"],
            evidence_expectations=["receipt"],
        )
        assert payload["mission_id"] == "m-001"
        assert payload["stage_id"] == "s-001"
        assert payload["assertion_ids"] == ["A1", "A2"]
        assert payload["roadmap_refs"] == ["RS-01"]
        assert payload["evidence_expectations"] == ["receipt"]

    def test_strips_whitespace(self):
        payload = mission_lineage_payload(
            mission_id="  m-002  ",
            stage_id=" s-002 ",
        )
        assert payload["mission_id"] == "m-002"
        assert payload["stage_id"] == "s-002"

    def test_deduplicates_assertion_ids(self):
        payload = mission_lineage_payload(assertion_ids=["A1", "A1", "A2"])
        assert payload["assertion_ids"] == ["A1", "A2"]

    def test_deduplicates_roadmap_refs(self):
        payload = mission_lineage_payload(roadmap_refs=["RS-01", "RS-01", "RS-02"])
        assert payload["roadmap_refs"] == ["RS-01", "RS-02"]

    def test_deduplicates_evidence_expectations(self):
        payload = mission_lineage_payload(evidence_expectations=["receipt", "receipt", "summary"])
        assert payload["evidence_expectations"] == ["receipt", "summary"]

    def test_none_lists_become_empty(self):
        payload = mission_lineage_payload(
            assertion_ids=None,
            roadmap_refs=None,
            evidence_expectations=None,
        )
        assert payload["assertion_ids"] == []
        assert payload["roadmap_refs"] == []
        assert payload["evidence_expectations"] == []

    def test_preserves_insertion_order(self):
        payload = mission_lineage_payload(
            roadmap_refs=["RS-03", "RS-01", "RS-02"],
        )
        assert payload["roadmap_refs"] == ["RS-03", "RS-01", "RS-02"]

    def test_filters_empty_strings_in_lists(self):
        payload = mission_lineage_payload(
            assertion_ids=["", "  ", "A1"],
        )
        assert payload["assertion_ids"] == ["A1"]

    def test_none_mission_id_becomes_empty_string(self):
        payload = mission_lineage_payload(mission_id=None)
        assert payload["mission_id"] == ""

    def test_returns_plain_dict(self):
        payload = mission_lineage_payload(mission_id="m-x", stage_id="s-x")
        assert isinstance(payload, dict)
        assert set(payload.keys()) == {
            "mission_id",
            "stage_id",
            "assertion_ids",
            "roadmap_refs",
            "evidence_expectations",
        }
