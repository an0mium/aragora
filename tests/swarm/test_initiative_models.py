"""Tests for aragora.swarm.initiative_models."""

from __future__ import annotations

from aragora.pipeline.decision_plan.core import PlanStatus
from aragora.swarm.initiative_models import (
    DEFAULT_PLAN_STATUS,
    InitiativeCheckpoint,
    InitiativeMilestone,
    InitiativeRecord,
    InitiativeSlice,
    utcnow_iso,
)


# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

_DEFAULT_STATUS = PlanStatus.CREATED.value  # "created"


# ===========================================================================
# utcnow_iso
# ===========================================================================


class TestUtcnowIso:
    def test_returns_string(self):
        result = utcnow_iso()
        assert isinstance(result, str)

    def test_contains_timezone_offset(self):
        # ISO 8601 with timezone — must contain '+' or 'Z'
        result = utcnow_iso()
        assert "+" in result or "Z" in result

    def test_successive_calls_are_non_decreasing(self):
        first = utcnow_iso()
        second = utcnow_iso()
        assert second >= first


# ===========================================================================
# DEFAULT_PLAN_STATUS
# ===========================================================================


class TestDefaultPlanStatus:
    def test_value_equals_created(self):
        assert DEFAULT_PLAN_STATUS == _DEFAULT_STATUS

    def test_is_string(self):
        assert isinstance(DEFAULT_PLAN_STATUS, str)


# ===========================================================================
# InitiativeSlice
# ===========================================================================


class TestInitiativeSliceDefaults:
    def test_required_fields_stored(self):
        s = InitiativeSlice(slice_id="s1", title="My Slice", description="desc")
        assert s.slice_id == "s1"
        assert s.title == "My Slice"
        assert s.description == "desc"

    def test_default_lists_are_empty(self):
        s = InitiativeSlice(slice_id="s1", title="T", description="D")
        assert s.dependencies == []
        assert s.file_scope == []
        assert s.acceptance_criteria == []
        assert s.validations == []

    def test_default_complexity_is_medium(self):
        s = InitiativeSlice(slice_id="s1", title="T", description="D")
        assert s.estimated_complexity == "medium"

    def test_default_status_is_created(self):
        s = InitiativeSlice(slice_id="s1", title="T", description="D")
        assert s.status == _DEFAULT_STATUS

    def test_default_metadata_is_empty_dict(self):
        s = InitiativeSlice(slice_id="s1", title="T", description="D")
        assert s.metadata == {}

    def test_mutable_defaults_are_independent(self):
        a = InitiativeSlice(slice_id="a", title="A", description="A")
        b = InitiativeSlice(slice_id="b", title="B", description="B")
        a.dependencies.append("x")
        assert b.dependencies == []


class TestInitiativeSliceToDict:
    def _make(self, **kwargs):
        defaults = dict(slice_id="s1", title="T", description="D")
        defaults.update(kwargs)
        return InitiativeSlice(**defaults)

    def test_to_dict_has_all_keys(self):
        d = self._make().to_dict()
        for key in (
            "slice_id",
            "title",
            "description",
            "dependencies",
            "file_scope",
            "acceptance_criteria",
            "validations",
            "estimated_complexity",
            "status",
            "metadata",
        ):
            assert key in d

    def test_to_dict_copies_lists(self):
        s = self._make(dependencies=["a", "b"])
        d = s.to_dict()
        d["dependencies"].append("c")
        assert s.dependencies == ["a", "b"]  # original not mutated

    def test_to_dict_copies_metadata(self):
        s = self._make(metadata={"key": "value"})
        d = s.to_dict()
        d["metadata"]["extra"] = 1
        assert "extra" not in s.metadata


class TestInitiativeSliceFromDict:
    def _full_payload(self):
        return {
            "slice_id": "  s1  ",
            "title": "  My Slice  ",
            "description": "  desc  ",
            "dependencies": ["  dep1  ", "dep2"],
            "file_scope": ["file.py"],
            "acceptance_criteria": ["ac1", "  ac2  "],
            "validations": ["val1"],
            "estimated_complexity": "  high  ",
            "status": "executing",
            "metadata": {"k": "v"},
        }

    def test_round_trip(self):
        s = InitiativeSlice(
            slice_id="s1",
            title="My Slice",
            description="desc",
            dependencies=["dep1", "dep2"],
            file_scope=["file.py"],
            acceptance_criteria=["ac1", "ac2"],
            validations=["val1"],
            estimated_complexity="high",
            status="executing",
            metadata={"k": "v"},
        )
        assert InitiativeSlice.from_dict(s.to_dict()) == s

    def test_strips_whitespace(self):
        payload = self._full_payload()
        s = InitiativeSlice.from_dict(payload)
        assert s.slice_id == "s1"
        assert s.title == "My Slice"
        assert s.description == "desc"
        assert s.dependencies == ["dep1", "dep2"]
        assert s.estimated_complexity == "high"

    def test_empty_payload_uses_defaults(self):
        s = InitiativeSlice.from_dict({})
        assert s.slice_id == ""
        assert s.title == ""
        assert s.description == ""
        assert s.dependencies == []
        assert s.estimated_complexity == "medium"
        assert s.status == _DEFAULT_STATUS
        assert s.metadata == {}

    def test_none_values_use_defaults(self):
        s = InitiativeSlice.from_dict(
            {"estimated_complexity": None, "status": None, "metadata": None}
        )
        assert s.estimated_complexity == "medium"
        assert s.status == _DEFAULT_STATUS
        assert s.metadata == {}

    def test_filters_blank_list_items(self):
        s = InitiativeSlice.from_dict(
            {
                "slice_id": "x",
                "title": "x",
                "description": "x",
                "dependencies": ["  ", "", "real"],
                "file_scope": ["  "],
                "acceptance_criteria": ["", "  ", "criterion"],
                "validations": [""],
            }
        )
        assert s.dependencies == ["real"]
        assert s.file_scope == []
        assert s.acceptance_criteria == ["criterion"]
        assert s.validations == []

    def test_empty_complexity_string_falls_back_to_medium(self):
        s = InitiativeSlice.from_dict({"estimated_complexity": "   "})
        assert s.estimated_complexity == "medium"

    def test_empty_status_string_falls_back_to_default(self):
        s = InitiativeSlice.from_dict({"status": "   "})
        assert s.status == _DEFAULT_STATUS


# ===========================================================================
# InitiativeCheckpoint
# ===========================================================================


class TestInitiativeCheckpointDefaults:
    def test_required_fields(self):
        c = InitiativeCheckpoint(checkpoint_id="cp1", title="Gate")
        assert c.checkpoint_id == "cp1"
        assert c.title == "Gate"
        assert c.description == ""

    def test_default_lists_are_empty(self):
        c = InitiativeCheckpoint(checkpoint_id="cp1", title="G")
        assert c.dependencies == []
        assert c.validations == []

    def test_default_status(self):
        c = InitiativeCheckpoint(checkpoint_id="cp1", title="G")
        assert c.status == _DEFAULT_STATUS

    def test_default_metadata(self):
        c = InitiativeCheckpoint(checkpoint_id="cp1", title="G")
        assert c.metadata == {}

    def test_mutable_defaults_independent(self):
        a = InitiativeCheckpoint(checkpoint_id="a", title="A")
        b = InitiativeCheckpoint(checkpoint_id="b", title="B")
        a.validations.append("v")
        assert b.validations == []


class TestInitiativeCheckpointToDict:
    def test_to_dict_has_all_keys(self):
        d = InitiativeCheckpoint(checkpoint_id="cp1", title="G").to_dict()
        for key in (
            "checkpoint_id",
            "title",
            "description",
            "dependencies",
            "validations",
            "status",
            "metadata",
        ):
            assert key in d

    def test_to_dict_copies_lists(self):
        c = InitiativeCheckpoint(checkpoint_id="cp1", title="G", dependencies=["a"])
        d = c.to_dict()
        d["dependencies"].append("b")
        assert c.dependencies == ["a"]


class TestInitiativeCheckpointFromDict:
    def test_round_trip(self):
        c = InitiativeCheckpoint(
            checkpoint_id="cp1",
            title="Gate",
            description="A gate",
            dependencies=["s1"],
            validations=["v1"],
            status="verifying",
            metadata={"note": "test"},
        )
        assert InitiativeCheckpoint.from_dict(c.to_dict()) == c

    def test_strips_whitespace(self):
        c = InitiativeCheckpoint.from_dict(
            {
                "checkpoint_id": "  cp2  ",
                "title": "  Gate  ",
                "description": "  desc  ",
            }
        )
        assert c.checkpoint_id == "cp2"
        assert c.title == "Gate"
        assert c.description == "desc"

    def test_empty_payload_uses_defaults(self):
        c = InitiativeCheckpoint.from_dict({})
        assert c.checkpoint_id == ""
        assert c.title == ""
        assert c.description == ""
        assert c.dependencies == []
        assert c.validations == []
        assert c.status == _DEFAULT_STATUS
        assert c.metadata == {}

    def test_none_values_use_defaults(self):
        c = InitiativeCheckpoint.from_dict({"status": None, "metadata": None})
        assert c.status == _DEFAULT_STATUS
        assert c.metadata == {}

    def test_filters_blank_list_items(self):
        c = InitiativeCheckpoint.from_dict(
            {
                "checkpoint_id": "cp",
                "title": "T",
                "dependencies": ["", "  ", "real"],
                "validations": ["  "],
            }
        )
        assert c.dependencies == ["real"]
        assert c.validations == []


# ===========================================================================
# InitiativeMilestone
# ===========================================================================


class TestInitiativeMilestoneDefaults:
    def test_required_fields(self):
        m = InitiativeMilestone(milestone_id="m1", title="Phase 1")
        assert m.milestone_id == "m1"
        assert m.title == "Phase 1"
        assert m.description == ""

    def test_default_lists(self):
        m = InitiativeMilestone(milestone_id="m1", title="M")
        assert m.slice_ids == []
        assert m.checkpoint_ids == []

    def test_default_status(self):
        m = InitiativeMilestone(milestone_id="m1", title="M")
        assert m.status == _DEFAULT_STATUS

    def test_default_metadata(self):
        m = InitiativeMilestone(milestone_id="m1", title="M")
        assert m.metadata == {}

    def test_mutable_defaults_independent(self):
        a = InitiativeMilestone(milestone_id="a", title="A")
        b = InitiativeMilestone(milestone_id="b", title="B")
        a.slice_ids.append("s1")
        assert b.slice_ids == []


class TestInitiativeMilestoneToDict:
    def test_to_dict_has_all_keys(self):
        d = InitiativeMilestone(milestone_id="m1", title="M").to_dict()
        for key in (
            "milestone_id",
            "title",
            "description",
            "slice_ids",
            "checkpoint_ids",
            "status",
            "metadata",
        ):
            assert key in d

    def test_to_dict_copies_lists(self):
        m = InitiativeMilestone(milestone_id="m1", title="M", slice_ids=["s1"])
        d = m.to_dict()
        d["slice_ids"].append("s2")
        assert m.slice_ids == ["s1"]


class TestInitiativeMilestoneFromDict:
    def test_round_trip(self):
        m = InitiativeMilestone(
            milestone_id="m1",
            title="Phase 1",
            description="First milestone",
            slice_ids=["s1", "s2"],
            checkpoint_ids=["cp1"],
            status="executing",
            metadata={"priority": 1},
        )
        assert InitiativeMilestone.from_dict(m.to_dict()) == m

    def test_strips_whitespace(self):
        m = InitiativeMilestone.from_dict(
            {"milestone_id": "  m2  ", "title": "  M  ", "description": "  d  "}
        )
        assert m.milestone_id == "m2"
        assert m.title == "M"
        assert m.description == "d"

    def test_empty_payload_uses_defaults(self):
        m = InitiativeMilestone.from_dict({})
        assert m.milestone_id == ""
        assert m.title == ""
        assert m.description == ""
        assert m.slice_ids == []
        assert m.checkpoint_ids == []
        assert m.status == _DEFAULT_STATUS
        assert m.metadata == {}

    def test_none_values_use_defaults(self):
        m = InitiativeMilestone.from_dict({"status": None, "metadata": None})
        assert m.status == _DEFAULT_STATUS
        assert m.metadata == {}

    def test_filters_blank_list_items(self):
        m = InitiativeMilestone.from_dict(
            {
                "milestone_id": "m",
                "title": "T",
                "slice_ids": ["", "  ", "s1"],
                "checkpoint_ids": ["  "],
            }
        )
        assert m.slice_ids == ["s1"]
        assert m.checkpoint_ids == []


# ===========================================================================
# InitiativeRecord
# ===========================================================================


class TestInitiativeRecordDefaults:
    def test_required_fields(self):
        r = InitiativeRecord(initiative_id="i1", title="Init", goal="Do X", rationale="Because Y")
        assert r.initiative_id == "i1"
        assert r.title == "Init"
        assert r.goal == "Do X"
        assert r.rationale == "Because Y"

    def test_default_lists_are_empty(self):
        r = InitiativeRecord(initiative_id="i1", title="I", goal="G", rationale="R")
        assert r.slices == []
        assert r.dependencies == []
        assert r.validations == []
        assert r.milestones == []
        assert r.checkpoints == []

    def test_default_feature_flag_is_none(self):
        r = InitiativeRecord(initiative_id="i1", title="I", goal="G", rationale="R")
        assert r.feature_flag_name is None

    def test_default_status(self):
        r = InitiativeRecord(initiative_id="i1", title="I", goal="G", rationale="R")
        assert r.status == _DEFAULT_STATUS

    def test_default_planner_rationale(self):
        r = InitiativeRecord(initiative_id="i1", title="I", goal="G", rationale="R")
        assert r.planner_rationale == ""

    def test_created_at_and_updated_at_set_automatically(self):
        r = InitiativeRecord(initiative_id="i1", title="I", goal="G", rationale="R")
        assert r.created_at
        assert r.updated_at

    def test_default_metadata(self):
        r = InitiativeRecord(initiative_id="i1", title="I", goal="G", rationale="R")
        assert r.metadata == {}

    def test_mutable_defaults_independent(self):
        a = InitiativeRecord(initiative_id="a", title="A", goal="G", rationale="R")
        b = InitiativeRecord(initiative_id="b", title="B", goal="G", rationale="R")
        a.slices.append(InitiativeSlice(slice_id="s", title="S", description="D"))
        assert b.slices == []


class TestInitiativeRecordTouch:
    def test_touch_updates_updated_at(self):
        r = InitiativeRecord(initiative_id="i1", title="I", goal="G", rationale="R")
        original = r.updated_at
        r.touch()
        # updated_at must be >= original (time can't go backwards)
        assert r.updated_at >= original

    def test_touch_does_not_change_created_at(self):
        r = InitiativeRecord(initiative_id="i1", title="I", goal="G", rationale="R")
        original_created = r.created_at
        r.touch()
        assert r.created_at == original_created


class TestInitiativeRecordToDict:
    def _make(self):
        r = InitiativeRecord(
            initiative_id="i1",
            title="Initiative",
            goal="Build feature",
            rationale="Needed for product",
        )
        r.slices.append(InitiativeSlice(slice_id="s1", title="Slice 1", description="D"))
        r.milestones.append(InitiativeMilestone(milestone_id="m1", title="Milestone 1"))
        r.checkpoints.append(InitiativeCheckpoint(checkpoint_id="cp1", title="Checkpoint 1"))
        return r

    def test_to_dict_has_all_keys(self):
        d = self._make().to_dict()
        for key in (
            "initiative_id",
            "title",
            "goal",
            "rationale",
            "slices",
            "dependencies",
            "validations",
            "feature_flag_name",
            "milestones",
            "checkpoints",
            "status",
            "planner_rationale",
            "created_at",
            "updated_at",
            "metadata",
        ):
            assert key in d, f"Missing key: {key}"

    def test_slices_serialized_as_list_of_dicts(self):
        d = self._make().to_dict()
        assert isinstance(d["slices"], list)
        assert isinstance(d["slices"][0], dict)

    def test_milestones_serialized_as_list_of_dicts(self):
        d = self._make().to_dict()
        assert isinstance(d["milestones"], list)
        assert isinstance(d["milestones"][0], dict)

    def test_checkpoints_serialized_as_list_of_dicts(self):
        d = self._make().to_dict()
        assert isinstance(d["checkpoints"], list)
        assert isinstance(d["checkpoints"][0], dict)

    def test_to_dict_copies_dependencies(self):
        r = InitiativeRecord(
            initiative_id="i1", title="I", goal="G", rationale="R", dependencies=["d1"]
        )
        d = r.to_dict()
        d["dependencies"].append("d2")
        assert r.dependencies == ["d1"]

    def test_feature_flag_none_preserved(self):
        r = InitiativeRecord(initiative_id="i1", title="I", goal="G", rationale="R")
        d = r.to_dict()
        assert d["feature_flag_name"] is None

    def test_feature_flag_value_preserved(self):
        r = InitiativeRecord(
            initiative_id="i1",
            title="I",
            goal="G",
            rationale="R",
            feature_flag_name="my_flag",
        )
        d = r.to_dict()
        assert d["feature_flag_name"] == "my_flag"


class TestInitiativeRecordFromDict:
    def _full_record(self):
        return InitiativeRecord(
            initiative_id="i1",
            title="Initiative",
            goal="Build feature",
            rationale="Needed for product",
            slices=[InitiativeSlice(slice_id="s1", title="S1", description="D")],
            dependencies=["dep1"],
            validations=["val1"],
            feature_flag_name="flag_x",
            milestones=[InitiativeMilestone(milestone_id="m1", title="M1")],
            checkpoints=[InitiativeCheckpoint(checkpoint_id="cp1", title="CP1")],
            status="executing",
            planner_rationale="Because it makes sense",
            metadata={"source": "test"},
        )

    def test_round_trip(self):
        r = self._full_record()
        restored = InitiativeRecord.from_dict(r.to_dict())
        assert restored.initiative_id == r.initiative_id
        assert restored.title == r.title
        assert restored.goal == r.goal
        assert restored.rationale == r.rationale
        assert len(restored.slices) == 1
        assert restored.slices[0].slice_id == "s1"
        assert len(restored.milestones) == 1
        assert restored.milestones[0].milestone_id == "m1"
        assert len(restored.checkpoints) == 1
        assert restored.checkpoints[0].checkpoint_id == "cp1"
        assert restored.dependencies == ["dep1"]
        assert restored.validations == ["val1"]
        assert restored.feature_flag_name == "flag_x"
        assert restored.status == "executing"
        assert restored.planner_rationale == "Because it makes sense"
        assert restored.metadata == {"source": "test"}

    def test_empty_payload_uses_defaults(self):
        r = InitiativeRecord.from_dict({})
        assert r.initiative_id == ""
        assert r.title == ""
        assert r.goal == ""
        assert r.rationale == ""
        assert r.slices == []
        assert r.dependencies == []
        assert r.validations == []
        assert r.feature_flag_name is None
        assert r.milestones == []
        assert r.checkpoints == []
        assert r.status == _DEFAULT_STATUS
        assert r.planner_rationale == ""
        assert r.metadata == {}

    def test_none_values_use_defaults(self):
        r = InitiativeRecord.from_dict(
            {
                "status": None,
                "metadata": None,
                "feature_flag_name": None,
            }
        )
        assert r.status == _DEFAULT_STATUS
        assert r.metadata == {}
        assert r.feature_flag_name is None

    def test_empty_feature_flag_name_becomes_none(self):
        r = InitiativeRecord.from_dict({"feature_flag_name": "   "})
        assert r.feature_flag_name is None

    def test_strips_whitespace_on_string_fields(self):
        r = InitiativeRecord.from_dict(
            {
                "initiative_id": "  i2  ",
                "title": "  T  ",
                "goal": "  G  ",
                "rationale": "  R  ",
                "planner_rationale": "  PR  ",
            }
        )
        assert r.initiative_id == "i2"
        assert r.title == "T"
        assert r.goal == "G"
        assert r.rationale == "R"
        assert r.planner_rationale == "PR"

    def test_filters_blank_dependency_items(self):
        r = InitiativeRecord.from_dict(
            {
                "initiative_id": "i",
                "title": "T",
                "goal": "G",
                "rationale": "R",
                "dependencies": ["", "  ", "real_dep"],
                "validations": ["  "],
            }
        )
        assert r.dependencies == ["real_dep"]
        assert r.validations == []

    def test_non_dict_slice_items_are_skipped(self):
        r = InitiativeRecord.from_dict(
            {
                "initiative_id": "i",
                "title": "T",
                "goal": "G",
                "rationale": "R",
                "slices": [
                    "not_a_dict",
                    None,
                    {"slice_id": "s1", "title": "S", "description": "D"},
                ],
            }
        )
        assert len(r.slices) == 1
        assert r.slices[0].slice_id == "s1"

    def test_non_dict_milestone_items_are_skipped(self):
        r = InitiativeRecord.from_dict(
            {
                "initiative_id": "i",
                "title": "T",
                "goal": "G",
                "rationale": "R",
                "milestones": ["bad", {"milestone_id": "m1", "title": "M"}],
            }
        )
        assert len(r.milestones) == 1

    def test_non_dict_checkpoint_items_are_skipped(self):
        r = InitiativeRecord.from_dict(
            {
                "initiative_id": "i",
                "title": "T",
                "goal": "G",
                "rationale": "R",
                "checkpoints": [42, {"checkpoint_id": "cp1", "title": "C"}],
            }
        )
        assert len(r.checkpoints) == 1

    def test_created_at_preserved_from_payload(self):
        ts = "2025-01-15T10:00:00+00:00"
        r = InitiativeRecord.from_dict(
            {
                "initiative_id": "i",
                "title": "T",
                "goal": "G",
                "rationale": "R",
                "created_at": ts,
                "updated_at": ts,
            }
        )
        assert r.created_at == ts
        assert r.updated_at == ts

    def test_empty_created_at_falls_back_to_fresh_timestamp(self):
        r = InitiativeRecord.from_dict(
            {
                "initiative_id": "i",
                "title": "T",
                "goal": "G",
                "rationale": "R",
                "created_at": "",
                "updated_at": None,
            }
        )
        # Should auto-generate timestamps rather than store empty strings
        assert r.created_at
        assert r.updated_at

    def test_empty_status_falls_back_to_default(self):
        r = InitiativeRecord.from_dict({"status": "   "})
        assert r.status == _DEFAULT_STATUS
