"""Tests for molecule-based work tracking (Gastown-inspired).

Covers MoleculeStatus, MoleculeType, MOLECULE_CAPABILITIES, Molecule dataclass,
MoleculeTracker, and create_round_molecules.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aragora.debate.molecules import (
    MOLECULE_CAPABILITIES,
    AgentProfileLike,
    Molecule,
    MoleculeStatus,
    MoleculeTracker,
    MoleculeType,
    create_round_molecules,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeAgent:
    name: str
    capabilities: set[str]
    elo_rating: float = 1500.0
    availability: float = 1.0


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestMoleculeStatus:
    def test_all_statuses(self):
        assert MoleculeStatus.PENDING.value == "pending"
        assert MoleculeStatus.ASSIGNED.value == "assigned"
        assert MoleculeStatus.IN_PROGRESS.value == "in_progress"
        assert MoleculeStatus.COMPLETED.value == "completed"
        assert MoleculeStatus.FAILED.value == "failed"
        assert MoleculeStatus.BLOCKED.value == "blocked"


class TestMoleculeType:
    def test_all_types(self):
        assert MoleculeType.PROPOSAL.value == "proposal"
        assert MoleculeType.CRITIQUE.value == "critique"
        assert MoleculeType.REVISION.value == "revision"
        assert MoleculeType.SYNTHESIS.value == "synthesis"
        assert MoleculeType.VOTE.value == "vote"
        assert MoleculeType.CONSENSUS_CHECK.value == "consensus_check"
        assert MoleculeType.QUALITY_REVIEW.value == "quality_review"
        assert MoleculeType.FACT_CHECK.value == "fact_check"

    def test_capabilities_mapping(self):
        for mt in MoleculeType:
            assert mt in MOLECULE_CAPABILITIES
            assert isinstance(MOLECULE_CAPABILITIES[mt], set)
            assert len(MOLECULE_CAPABILITIES[mt]) > 0


# ---------------------------------------------------------------------------
# AgentProfileLike protocol
# ---------------------------------------------------------------------------


class TestAgentProfileLike:
    def test_fake_agent_satisfies(self):
        agent = FakeAgent(name="a", capabilities={"reasoning"})
        assert isinstance(agent, AgentProfileLike)


# ---------------------------------------------------------------------------
# Molecule dataclass
# ---------------------------------------------------------------------------


class TestMolecule:
    def test_create(self):
        mol = Molecule.create("d1", MoleculeType.PROPOSAL, 1)
        assert mol.molecule_id.startswith("mol-")
        assert mol.debate_id == "d1"
        assert mol.molecule_type == MoleculeType.PROPOSAL
        assert mol.round_number == 1
        assert mol.status == MoleculeStatus.PENDING
        assert mol.attempts == 0
        assert mol.max_attempts == 3

    def test_auto_capabilities(self):
        mol = Molecule.create("d1", MoleculeType.CRITIQUE, 1)
        assert "analysis" in mol.required_capabilities
        assert "quality_assessment" in mol.required_capabilities

    def test_custom_input(self):
        mol = Molecule.create("d1", MoleculeType.PROPOSAL, 1, input_data={"task": "test"})
        assert mol.input_data == {"task": "test"}

    def test_depends_on(self):
        mol = Molecule.create("d1", MoleculeType.SYNTHESIS, 1, depends_on=["mol-a", "mol-b"])
        assert mol.depends_on == ["mol-a", "mol-b"]

    def test_assign(self):
        mol = Molecule.create("d1", MoleculeType.PROPOSAL, 1)
        mol.assign("claude")
        assert mol.assigned_agent == "claude"
        assert mol.status == MoleculeStatus.ASSIGNED
        assert mol.attempts == 1
        assert "claude" in mol.assignment_history

    def test_start(self):
        mol = Molecule.create("d1", MoleculeType.PROPOSAL, 1)
        mol.assign("claude")
        mol.start()
        assert mol.status == MoleculeStatus.IN_PROGRESS
        assert mol.started_at is not None

    def test_complete(self):
        mol = Molecule.create("d1", MoleculeType.PROPOSAL, 1)
        mol.assign("claude")
        mol.start()
        mol.complete({"result": "done"})
        assert mol.status == MoleculeStatus.COMPLETED
        assert mol.output_data == {"result": "done"}
        assert mol.completed_at is not None

    def test_complete_increases_affinity(self):
        mol = Molecule.create("d1", MoleculeType.PROPOSAL, 1)
        mol.assign("claude")
        mol.start()
        mol.complete({"result": "done"})
        assert mol.agent_affinity["claude"] == 0.6  # 0.5 default + 0.1

    def test_fail(self):
        mol = Molecule.create("d1", MoleculeType.PROPOSAL, 1)
        mol.assign("claude")
        mol.start()
        mol.fail("timeout")
        assert mol.status == MoleculeStatus.FAILED
        assert mol.error_message == "timeout"
        assert mol.assigned_agent is None  # cleared on fail

    def test_fail_decreases_affinity(self):
        mol = Molecule.create("d1", MoleculeType.PROPOSAL, 1)
        mol.assign("claude")
        mol.start()
        mol.fail("timeout")
        assert mol.agent_affinity["claude"] == 0.3  # 0.5 default - 0.2

    def test_can_retry(self):
        mol = Molecule.create("d1", MoleculeType.PROPOSAL, 1)
        assert mol.can_retry()
        mol.attempts = 3
        assert not mol.can_retry()

    def test_to_dict(self):
        mol = Molecule.create("d1", MoleculeType.VOTE, 2)
        d = mol.to_dict()
        assert d["debate_id"] == "d1"
        assert d["molecule_type"] == "vote"
        assert d["round_number"] == 2
        assert d["status"] == "pending"
        assert isinstance(d["required_capabilities"], list)

    def test_from_dict(self):
        mol = Molecule.create("d1", MoleculeType.PROPOSAL, 1)
        mol.assign("claude")
        mol.start()
        mol.complete({"result": "done"})
        d = mol.to_dict()
        restored = Molecule.from_dict(d)
        assert restored.molecule_id == mol.molecule_id
        assert restored.molecule_type == MoleculeType.PROPOSAL
        assert restored.status == MoleculeStatus.COMPLETED
        assert restored.output_data == {"result": "done"}
        assert restored.assigned_agent == "claude"

    def test_roundtrip_serialization(self):
        mol = Molecule.create("d1", MoleculeType.FACT_CHECK, 3, depends_on=["mol-x"])
        mol.agent_affinity = {"a": 0.8, "b": 0.3}
        d = mol.to_dict()
        restored = Molecule.from_dict(d)
        assert restored.depends_on == ["mol-x"]
        assert restored.agent_affinity == {"a": 0.8, "b": 0.3}
        assert restored.molecule_type == MoleculeType.FACT_CHECK


# ---------------------------------------------------------------------------
# MoleculeTracker
# ---------------------------------------------------------------------------


class TestMoleculeTracker:
    @pytest.fixture
    def tracker(self):
        return MoleculeTracker()

    def test_create_molecule(self, tracker):
        mol = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        assert tracker.get_molecule(mol.molecule_id) is mol

    def test_get_debate_molecules(self, tracker):
        m1 = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        m2 = tracker.create_molecule("d1", MoleculeType.CRITIQUE, 1)
        tracker.create_molecule("d2", MoleculeType.PROPOSAL, 1)
        mols = tracker.get_debate_molecules("d1")
        assert len(mols) == 2
        assert m1 in mols
        assert m2 in mols

    def test_get_pending_molecules(self, tracker):
        m1 = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        pending = tracker.get_pending_molecules("d1")
        assert m1 in pending

    def test_pending_respects_dependencies(self, tracker):
        m1 = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        m2 = tracker.create_molecule("d1", MoleculeType.CRITIQUE, 1, depends_on=[m1.molecule_id])
        pending = tracker.get_pending_molecules("d1")
        # m1 is pending (no deps), m2 is blocked
        assert m1 in pending
        assert m2 not in pending
        assert m2.status == MoleculeStatus.BLOCKED

    def test_assign_molecule(self, tracker):
        mol = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        agent = FakeAgent("claude", {"reasoning", "creativity"})
        assert tracker.assign_molecule(mol.molecule_id, agent)
        assert mol.status == MoleculeStatus.ASSIGNED
        assert mol.assigned_agent == "claude"

    def test_assign_fails_missing_capabilities(self, tracker):
        mol = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        agent = FakeAgent("weak", set())  # no capabilities
        assert not tracker.assign_molecule(mol.molecule_id, agent)

    def test_assign_fails_nonexistent(self, tracker):
        agent = FakeAgent("a", {"reasoning"})
        assert not tracker.assign_molecule("nonexistent", agent)

    def test_assign_fails_wrong_status(self, tracker):
        mol = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        agent = FakeAgent("claude", {"reasoning", "creativity"})
        tracker.assign_molecule(mol.molecule_id, agent)
        tracker.start_molecule(mol.molecule_id)
        # Can't assign IN_PROGRESS molecule
        assert not tracker.assign_molecule(mol.molecule_id, agent)

    def test_assign_failed_retry(self, tracker):
        mol = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        agent = FakeAgent("claude", {"reasoning", "creativity"})
        tracker.assign_molecule(mol.molecule_id, agent)
        tracker.start_molecule(mol.molecule_id)
        tracker.fail_molecule(mol.molecule_id, "error")
        # Should be able to reassign after failure
        agent2 = FakeAgent("gpt", {"reasoning", "creativity"})
        assert tracker.assign_molecule(mol.molecule_id, agent2)

    def test_assign_exhausted_retries(self, tracker):
        mol = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        mol.max_attempts = 1
        agent = FakeAgent("claude", {"reasoning", "creativity"})
        tracker.assign_molecule(mol.molecule_id, agent)
        tracker.start_molecule(mol.molecule_id)
        tracker.fail_molecule(mol.molecule_id, "error")
        # Exhausted retries
        assert not tracker.assign_molecule(mol.molecule_id, agent)

    def test_start_molecule(self, tracker):
        mol = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        agent = FakeAgent("claude", {"reasoning", "creativity"})
        tracker.assign_molecule(mol.molecule_id, agent)
        assert tracker.start_molecule(mol.molecule_id)
        assert mol.status == MoleculeStatus.IN_PROGRESS

    def test_start_fails_not_assigned(self, tracker):
        mol = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        assert not tracker.start_molecule(mol.molecule_id)

    def test_complete_molecule(self, tracker):
        mol = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        agent = FakeAgent("claude", {"reasoning", "creativity"})
        tracker.assign_molecule(mol.molecule_id, agent)
        tracker.start_molecule(mol.molecule_id)
        assert tracker.complete_molecule(mol.molecule_id, {"result": "ok"})
        assert mol.status == MoleculeStatus.COMPLETED

    def test_complete_unblocks_dependents(self, tracker):
        m1 = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        m2 = tracker.create_molecule("d1", MoleculeType.CRITIQUE, 1, depends_on=[m1.molecule_id])
        # Mark m2 as blocked first
        tracker.get_pending_molecules("d1")
        assert m2.status == MoleculeStatus.BLOCKED

        # Complete m1 → should unblock m2
        agent = FakeAgent("claude", {"reasoning", "creativity"})
        tracker.assign_molecule(m1.molecule_id, agent)
        tracker.start_molecule(m1.molecule_id)
        tracker.complete_molecule(m1.molecule_id, {"text": "proposal"})
        assert m2.status == MoleculeStatus.PENDING

    def test_complete_nonexistent(self, tracker):
        assert not tracker.complete_molecule("nonexistent", {})

    def test_complete_wrong_status(self, tracker):
        mol = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        assert not tracker.complete_molecule(mol.molecule_id, {})

    def test_fail_molecule(self, tracker):
        mol = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        agent = FakeAgent("claude", {"reasoning", "creativity"})
        tracker.assign_molecule(mol.molecule_id, agent)
        tracker.start_molecule(mol.molecule_id)
        assert tracker.fail_molecule(mol.molecule_id, "timeout")
        assert mol.status == MoleculeStatus.FAILED

    def test_fail_nonexistent(self, tracker):
        assert not tracker.fail_molecule("nonexistent", "error")

    def test_fail_wrong_status(self, tracker):
        mol = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        assert not tracker.fail_molecule(mol.molecule_id, "error")

    def test_workload_tracking(self, tracker):
        m1 = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        m2 = tracker.create_molecule("d1", MoleculeType.CRITIQUE, 1)
        agent = FakeAgent("claude", {"reasoning", "creativity", "analysis", "quality_assessment"})
        tracker.assign_molecule(m1.molecule_id, agent)
        tracker.assign_molecule(m2.molecule_id, agent)
        assert tracker._agent_workload["claude"] == 2

        tracker.start_molecule(m1.molecule_id)
        tracker.complete_molecule(m1.molecule_id, {})
        assert tracker._agent_workload["claude"] == 1

    def test_find_best_agent(self, tracker):
        mol = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        low_elo = FakeAgent("weak", {"reasoning", "creativity"}, elo_rating=1100)
        high_elo = FakeAgent("strong", {"reasoning", "creativity"}, elo_rating=1900)
        best = tracker.find_best_agent(mol, [low_elo, high_elo])
        assert best is high_elo

    def test_find_best_agent_capability_filter(self, tracker):
        mol = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        no_caps = FakeAgent("nocap", set())
        has_caps = FakeAgent("hascap", {"reasoning", "creativity"})
        best = tracker.find_best_agent(mol, [no_caps, has_caps])
        assert best is has_caps

    def test_find_best_agent_none(self, tracker):
        mol = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        assert tracker.find_best_agent(mol, []) is None

    def test_find_best_agent_skips_failed(self, tracker):
        mol = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        agent = FakeAgent("claude", {"reasoning", "creativity"})
        mol.assignment_history.append("claude")
        mol.status = MoleculeStatus.FAILED
        assert tracker.find_best_agent(mol, [agent]) is None

    def test_find_best_agent_workload_penalty(self, tracker):
        mol = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        busy = FakeAgent("busy", {"reasoning", "creativity"}, elo_rating=1500)
        idle = FakeAgent("idle", {"reasoning", "creativity"}, elo_rating=1500)
        tracker._agent_workload["busy"] = 5
        best = tracker.find_best_agent(mol, [busy, idle])
        assert best is idle

    def test_get_progress(self, tracker):
        tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        tracker.create_molecule("d1", MoleculeType.CRITIQUE, 1)
        progress = tracker.get_progress("d1")
        assert progress["total"] == 2
        assert progress["completed"] == 0
        assert progress["progress"] == 0.0
        assert "by_status" in progress
        assert "by_type" in progress

    def test_get_progress_empty(self, tracker):
        progress = tracker.get_progress("empty")
        assert progress["total"] == 0
        assert progress["progress"] == 0.0

    def test_get_progress_partial(self, tracker):
        m1 = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        tracker.create_molecule("d1", MoleculeType.CRITIQUE, 1)
        agent = FakeAgent("claude", {"reasoning", "creativity"})
        tracker.assign_molecule(m1.molecule_id, agent)
        tracker.start_molecule(m1.molecule_id)
        tracker.complete_molecule(m1.molecule_id, {})
        progress = tracker.get_progress("d1")
        assert progress["completed"] == 1
        assert progress["progress"] == 0.5

    def test_clear_debate(self, tracker):
        tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        tracker.create_molecule("d1", MoleculeType.CRITIQUE, 1)
        tracker.clear_debate("d1")
        assert tracker.get_debate_molecules("d1") == []

    def test_clear_debate_updates_workload(self, tracker):
        mol = tracker.create_molecule("d1", MoleculeType.PROPOSAL, 1)
        agent = FakeAgent("claude", {"reasoning", "creativity"})
        tracker.assign_molecule(mol.molecule_id, agent)
        assert tracker._agent_workload["claude"] == 1
        tracker.clear_debate("d1")
        assert tracker._agent_workload["claude"] == 0


# ---------------------------------------------------------------------------
# create_round_molecules
# ---------------------------------------------------------------------------


class TestCreateRoundMolecules:
    def test_two_agents(self):
        tracker = MoleculeTracker()
        mols = create_round_molecules(tracker, "d1", 1, 2, "Rate limiter")
        # 2 proposals + 2 critiques (each agent critiques the other) + 1 synthesis = 5
        assert len(mols) == 5
        types = [m.molecule_type for m in mols]
        assert types.count(MoleculeType.PROPOSAL) == 2
        assert types.count(MoleculeType.CRITIQUE) == 2
        assert types.count(MoleculeType.SYNTHESIS) == 1

    def test_three_agents(self):
        tracker = MoleculeTracker()
        mols = create_round_molecules(tracker, "d1", 1, 3, "Design API")
        # 3 proposals + 6 critiques (3*2) + 1 synthesis = 10
        assert len(mols) == 10

    def test_dependencies_wired(self):
        tracker = MoleculeTracker()
        mols = create_round_molecules(tracker, "d1", 1, 2, "Test")
        critiques = [m for m in mols if m.molecule_type == MoleculeType.CRITIQUE]
        synthesis = [m for m in mols if m.molecule_type == MoleculeType.SYNTHESIS]
        # Each critique depends on one proposal
        for c in critiques:
            assert len(c.depends_on) == 1
        # Synthesis depends on all critiques
        assert len(synthesis[0].depends_on) == len(critiques)

    def test_registered_in_tracker(self):
        tracker = MoleculeTracker()
        create_round_molecules(tracker, "d1", 1, 2, "Test")
        assert len(tracker.get_debate_molecules("d1")) == 5
