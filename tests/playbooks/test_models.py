"""
Tests for Playbook data models.
"""

import pytest
from aragora.playbooks.models import Playbook, PlaybookStep, ApprovalGate


class TestApprovalGate:
    """Tests for ApprovalGate dataclass."""

    def test_create_default(self):
        gate = ApprovalGate(name="test", description="Test gate")
        assert gate.required_role == "decision_maker"
        assert gate.auto_approve_if_consensus is False
        assert gate.timeout_hours == 24.0

    def test_to_dict(self):
        gate = ApprovalGate(name="review", description="Review gate", required_role="admin")
        d = gate.to_dict()
        assert d["name"] == "review"
        assert d["required_role"] == "admin"

    def test_from_dict(self):
        data = {
            "name": "approval",
            "description": "Needs approval",
            "required_role": "manager",
            "auto_approve_if_consensus": True,
            "timeout_hours": 48.0,
        }
        gate = ApprovalGate.from_dict(data)
        assert gate.name == "approval"
        assert gate.auto_approve_if_consensus is True
        assert gate.timeout_hours == 48.0

    def test_from_dict_minimal(self):
        gate = ApprovalGate.from_dict({"name": "x"})
        assert gate.name == "x"
        assert gate.description == ""


class TestPlaybookStep:
    """Tests for PlaybookStep dataclass."""

    def test_create(self):
        step = PlaybookStep(name="debate", action="debate", config={"rounds": 3})
        assert step.action == "debate"
        assert step.config["rounds"] == 3

    def test_to_dict(self):
        step = PlaybookStep(name="review", action="review")
        d = step.to_dict()
        assert d["action"] == "review"
        assert d["config"] == {}

    def test_from_dict(self):
        step = PlaybookStep.from_dict(
            {"name": "s1", "action": "notify", "config": {"channels": ["slack"]}}
        )
        assert step.action == "notify"
        assert step.config["channels"] == ["slack"]


class TestPlaybook:
    """Tests for Playbook dataclass."""

    def test_create_minimal(self):
        pb = Playbook(
            id="test",
            name="Test Playbook",
            description="A test",
            category="general",
        )
        assert pb.id == "test"
        assert pb.min_agents == 3
        assert pb.consensus_threshold == 0.7

    def test_to_dict(self):
        pb = Playbook(
            id="test",
            name="Test",
            description="Desc",
            category="finance",
            tags=["a", "b"],
        )
        d = pb.to_dict()
        assert d["id"] == "test"
        assert d["category"] == "finance"
        assert d["tags"] == ["a", "b"]
        assert "approval_gates" in d
        assert "steps" in d

    def test_from_dict(self):
        data = {
            "id": "pb1",
            "name": "Playbook 1",
            "description": "First playbook",
            "category": "healthcare",
            "template_name": "compliance_review",
            "vertical_profile": "healthcare_hipaa",
            "min_agents": 4,
            "approval_gates": [
                {"name": "gate1", "description": "First gate"},
            ],
            "steps": [
                {"name": "step1", "action": "debate"},
            ],
        }
        pb = Playbook.from_dict(data)
        assert pb.id == "pb1"
        assert pb.vertical_profile == "healthcare_hipaa"
        assert len(pb.approval_gates) == 1
        assert len(pb.steps) == 1

    def test_roundtrip(self):
        pb = Playbook(
            id="rt",
            name="Roundtrip",
            description="Test roundtrip",
            category="engineering",
            steps=[PlaybookStep(name="s", action="debate")],
            approval_gates=[ApprovalGate(name="g", description="gate")],
        )
        d = pb.to_dict()
        restored = Playbook.from_dict(d)
        assert restored.id == pb.id
        assert len(restored.steps) == 1
        assert len(restored.approval_gates) == 1
