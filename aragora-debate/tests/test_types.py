"""Tests for aragora_debate.types."""

import sys
import os

# Add src to path for standalone testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aragora_debate.types import (
    Agent,
    Claim,
    Consensus,
    ConsensusMethod,
    Critique,
    DebateConfig,
    DebateResult,
    DecisionReceipt,
    DissentRecord,
    Evidence,
    Message,
    Phase,
    Proposal,
    Verdict,
    Vote,
)


class TestEnums:
    def test_phase_values(self):
        assert Phase.PROPOSE == "propose"
        assert Phase.CRITIQUE == "critique"
        assert Phase.VOTE == "vote"

    def test_consensus_method_values(self):
        assert ConsensusMethod.MAJORITY == "majority"
        assert ConsensusMethod.UNANIMOUS == "unanimous"
        assert ConsensusMethod.SUPERMAJORITY == "supermajority"

    def test_verdict_values(self):
        assert Verdict.APPROVED == "approved"
        assert Verdict.REJECTED == "rejected"
        assert Verdict.NEEDS_REVIEW == "needs_review"


class TestMessage:
    def test_creation(self):
        m = Message(role="proposer", agent="claude", content="hello")
        assert m.role == "proposer"
        assert m.agent == "claude"
        assert m.round == 0

    def test_to_dict(self):
        m = Message(role="voter", agent="gpt4", content="agree", round=2)
        d = m.to_dict()
        assert d["role"] == "voter"
        assert d["agent"] == "gpt4"
        assert d["round"] == 2
        assert "timestamp" in d


class TestCritique:
    def test_content_property(self):
        c = Critique(
            agent="claude",
            target_agent="gpt4",
            target_content="some proposal",
            issues=["weak evidence", "missing data"],
            suggestions=["add citations"],
            reasoning="needs more rigor",
        )
        content = c.content
        assert "weak evidence" in content
        assert "add citations" in content
        assert "needs more rigor" in content

    def test_empty_critique(self):
        c = Critique(agent="a", target_agent="b", target_content="x")
        assert c.content == "(no critique)"


class TestClaim:
    def test_net_evidence_strength(self):
        c = Claim(
            statement="test",
            author="claude",
            supporting_evidence=[Evidence(source="a", content="x", strength=0.8)],
            refuting_evidence=[Evidence(source="b", content="y", strength=0.2)],
        )
        assert abs(c.net_evidence_strength - 0.8) < 0.01

    def test_no_evidence(self):
        c = Claim(statement="test", author="claude")
        assert c.net_evidence_strength == 0.5


class TestConsensus:
    def test_agreement_ratio(self):
        c = Consensus(
            reached=True,
            method=ConsensusMethod.MAJORITY,
            confidence=0.8,
            supporting_agents=["a", "b", "c"],
            dissenting_agents=["d"],
        )
        assert abs(c.agreement_ratio - 0.75) < 0.01

    def test_no_agents(self):
        c = Consensus(
            reached=False,
            method=ConsensusMethod.MAJORITY,
            confidence=0.0,
        )
        assert c.agreement_ratio == 0.0


class TestDissentRecord:
    def test_to_dict(self):
        d = DissentRecord(
            agent="gpt4",
            reasons=["insufficient evidence"],
            alternative_view="should use RAG instead",
        )
        result = d.to_dict()
        assert result["agent"] == "gpt4"
        assert "insufficient evidence" in result["reasons"]
        assert result["alternative_view"] == "should use RAG instead"


class TestDebateConfig:
    def test_defaults(self):
        cfg = DebateConfig()
        assert cfg.rounds == 3
        assert cfg.consensus_method == ConsensusMethod.MAJORITY
        assert cfg.early_stopping is True
        assert cfg.min_rounds == 1

    def test_custom(self):
        cfg = DebateConfig(rounds=5, consensus_method=ConsensusMethod.SUPERMAJORITY)
        assert cfg.rounds == 5
        assert cfg.consensus_method == ConsensusMethod.SUPERMAJORITY


class TestDecisionReceipt:
    def test_to_markdown(self):
        receipt = DecisionReceipt(
            receipt_id="DR-20260211-abc123",
            question="Should we use microservices?",
            verdict=Verdict.APPROVED,
            confidence=0.85,
            consensus=Consensus(
                reached=True,
                method=ConsensusMethod.MAJORITY,
                confidence=0.85,
                supporting_agents=["claude", "gpt4"],
                dissenting_agents=["gemini"],
                dissents=[
                    DissentRecord(
                        agent="gemini",
                        reasons=["monolith is simpler"],
                        alternative_view="keep monolith",
                    )
                ],
            ),
            agents=["claude", "gpt4", "gemini"],
            rounds_used=3,
        )
        md = receipt.to_markdown()
        assert "Decision Receipt" in md
        assert "microservices" in md
        assert "claude" in md
        assert "Dissenting" in md
        assert "gemini" in md

    def test_to_dict(self):
        receipt = DecisionReceipt(
            receipt_id="DR-test",
            question="test",
            verdict=Verdict.REJECTED,
            confidence=0.3,
            consensus=Consensus(
                reached=False,
                method=ConsensusMethod.MAJORITY,
                confidence=0.3,
            ),
            agents=["a", "b"],
            rounds_used=3,
        )
        d = receipt.to_dict()
        assert d["verdict"] == "rejected"
        assert d["rounds_used"] == 3
        assert d["consensus"]["reached"] is False


class TestDebateResult:
    def test_summary(self):
        r = DebateResult(
            task="Should we use React?",
            status="consensus_reached",
            rounds_used=2,
            confidence=0.9,
            consensus_reached=True,
            final_answer="Yes, React is good",
        )
        s = r.summary()
        assert "React" in s
        assert "consensus" in s.lower()

    def test_to_dict(self):
        r = DebateResult(task="test", status="completed", rounds_used=1)
        d = r.to_dict()
        assert d["task"] == "test"
        assert d["status"] == "completed"
