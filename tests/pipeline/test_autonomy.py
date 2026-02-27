"""Tests for Autonomy Level Controls."""

from __future__ import annotations

import pytest
from aragora.pipeline.autonomy import AutonomyGate, AutonomyLevel, create_gates


class TestAutonomyLevel:
    def test_from_string(self):
        assert AutonomyLevel.from_string("fully_autonomous") == AutonomyLevel.FULLY_AUTONOMOUS
        assert AutonomyLevel.from_string("propose-and-approve") == AutonomyLevel.PROPOSE_AND_APPROVE
        assert AutonomyLevel.from_string("HUMAN_GUIDED") == AutonomyLevel.HUMAN_GUIDED

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Unknown autonomy level"):
            AutonomyLevel.from_string("invalid")

    def test_orchestrator_config_fully_autonomous(self):
        config = AutonomyLevel.FULLY_AUTONOMOUS.to_orchestrator_config()
        assert config["require_human_approval"] is False
        assert config["auto_commit"] is True
        assert config["auto_merge"] is True

    def test_orchestrator_config_propose_approve(self):
        config = AutonomyLevel.PROPOSE_AND_APPROVE.to_orchestrator_config()
        assert config["require_human_approval"] is True
        assert config["auto_commit"] is False

    def test_orchestrator_config_metrics_driven(self):
        config = AutonomyLevel.METRICS_DRIVEN.to_orchestrator_config()
        assert config["require_human_approval"] is False
        assert config["auto_commit"] is True
        assert config["auto_merge"] is False

    def test_requires_spec_approval(self):
        assert AutonomyLevel.PROPOSE_AND_APPROVE.requires_spec_approval
        assert AutonomyLevel.HUMAN_GUIDED.requires_spec_approval
        assert not AutonomyLevel.FULLY_AUTONOMOUS.requires_spec_approval
        assert not AutonomyLevel.METRICS_DRIVEN.requires_spec_approval

    def test_requires_merge_approval(self):
        assert not AutonomyLevel.FULLY_AUTONOMOUS.requires_merge_approval
        assert AutonomyLevel.PROPOSE_AND_APPROVE.requires_merge_approval
        assert AutonomyLevel.METRICS_DRIVEN.requires_merge_approval

    def test_auto_commits(self):
        assert AutonomyLevel.FULLY_AUTONOMOUS.auto_commits
        assert AutonomyLevel.METRICS_DRIVEN.auto_commits
        assert not AutonomyLevel.PROPOSE_AND_APPROVE.auto_commits
        assert not AutonomyLevel.HUMAN_GUIDED.auto_commits

    def test_skips_interrogation(self):
        assert AutonomyLevel.FULLY_AUTONOMOUS.skips_interrogation
        assert not AutonomyLevel.PROPOSE_AND_APPROVE.skips_interrogation

    def test_approval_level(self):
        assert AutonomyLevel.FULLY_AUTONOMOUS.to_approval_level() == "none"
        assert AutonomyLevel.HUMAN_GUIDED.to_approval_level() == "all_stages"


class TestAutonomyGate:
    def test_fully_autonomous_never_needs(self):
        gate = AutonomyGate(level=AutonomyLevel.FULLY_AUTONOMOUS, stage="spec")
        assert not gate.needs_approval()

    def test_human_guided_always_needs(self):
        gate = AutonomyGate(level=AutonomyLevel.HUMAN_GUIDED, stage="spec")
        assert gate.needs_approval()

    def test_propose_approve_spec(self):
        gate = AutonomyGate(level=AutonomyLevel.PROPOSE_AND_APPROVE, stage="spec")
        assert gate.needs_approval()

    def test_propose_approve_interrogation(self):
        gate = AutonomyGate(level=AutonomyLevel.PROPOSE_AND_APPROVE, stage="interrogation")
        assert not gate.needs_approval()

    def test_metrics_driven_high_quality(self):
        gate = AutonomyGate(level=AutonomyLevel.METRICS_DRIVEN, stage="spec", metrics_threshold=0.8)
        assert not gate.needs_approval(quality_score=0.9)

    def test_metrics_driven_low_quality(self):
        gate = AutonomyGate(level=AutonomyLevel.METRICS_DRIVEN, stage="spec", metrics_threshold=0.8)
        assert gate.needs_approval(quality_score=0.5)

    def test_metrics_driven_always_approves_merge(self):
        gate = AutonomyGate(level=AutonomyLevel.METRICS_DRIVEN, stage="merge")
        assert gate.needs_approval(quality_score=1.0)

    def test_gate_description(self):
        gate = AutonomyGate(level=AutonomyLevel.HUMAN_GUIDED, stage="spec")
        assert "specification" in gate.gate_description.lower()


class TestCreateGates:
    def test_creates_all_stages(self):
        gates = create_gates(AutonomyLevel.PROPOSE_AND_APPROVE)
        assert set(gates.keys()) == {"interrogation", "spec", "execution", "merge"}

    def test_gates_use_level(self):
        gates = create_gates(AutonomyLevel.FULLY_AUTONOMOUS)
        for gate in gates.values():
            assert gate.level == AutonomyLevel.FULLY_AUTONOMOUS
