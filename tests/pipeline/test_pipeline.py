"""
Comprehensive tests for the aragora.pipeline module.

Tests cover:
- pr_generator: DecisionMemo, PatchPlan, PRGenerator, generate_pr_artifacts
- risk_register: Risk, RiskRegister, RiskAnalyzer, RiskLevel, RiskCategory, generate_risk_register
- verification_plan: VerificationCase, VerificationPlan, VerificationPlanGenerator,
                     VerificationType, CasePriority, generate_test_plan
- __init__: backward compatibility aliases
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora.export.artifact import ConsensusProof, DebateArtifact, VerificationResult
from aragora.pipeline.pr_generator import (
    DecisionMemo,
    PatchPlan,
    PRGenerator,
    generate_pr_artifacts,
)
from aragora.pipeline.risk_register import (
    Risk,
    RiskAnalyzer,
    RiskCategory,
    RiskLevel,
    RiskRegister,
    generate_risk_register,
)
from aragora.pipeline.verification_plan import (
    CasePriority,
    VerificationCase,
    VerificationPlan,
    VerificationPlanGenerator,
    VerificationType,
    generate_test_plan,
)

# Also test backward compat aliases from __init__
from aragora.pipeline import TestPlan, TestCase, TestPlanGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_artifact(**overrides) -> DebateArtifact:
    """Create a DebateArtifact with sensible defaults."""
    defaults = dict(
        debate_id="debate-001",
        task="Design a rate limiter for the API gateway",
        agents=["claude", "gpt4", "gemini"],
        rounds=3,
        critique_count=4,
        consensus_proof=ConsensusProof(
            reached=True,
            confidence=0.85,
            final_answer="1. Implement token bucket algorithm\n2. Add Redis backend\n- Use sliding window",
            vote_breakdown={"claude": True, "gpt4": True, "gemini": False},
            rounds_used=3,
        ),
        verification_results=[],
        trace_data=None,
        provenance_data=None,
    )
    defaults.update(overrides)
    return DebateArtifact(**defaults)


# ===================================================================
# DecisionMemo tests
# ===================================================================


class TestDecisionMemo:
    def test_creation_defaults(self):
        memo = DecisionMemo(
            debate_id="d1",
            title="Title",
            summary="Sum",
            key_decisions=["A"],
            rationale="R",
            supporting_evidence=[],
            dissenting_views=[],
            open_questions=[],
        )
        assert memo.debate_id == "d1"
        assert memo.consensus_confidence == 0.0
        assert memo.rounds_used == 0
        assert memo.agent_count == 0
        assert memo.created_at  # auto-generated

    def test_to_dict(self):
        memo = DecisionMemo(
            debate_id="d1",
            title="T",
            summary="S",
            key_decisions=["X"],
            rationale="R",
            supporting_evidence=[{"source": "web", "summary": "info"}],
            dissenting_views=["dv"],
            open_questions=["q"],
            consensus_confidence=0.9,
            rounds_used=3,
            agent_count=5,
        )
        d = memo.to_dict()
        assert d["debate_id"] == "d1"
        assert d["consensus_confidence"] == 0.9
        assert d["key_decisions"] == ["X"]
        assert d["supporting_evidence"] == [{"source": "web", "summary": "info"}]

    def test_to_markdown_contains_sections(self):
        memo = DecisionMemo(
            debate_id="d1",
            title="Rate Limiter",
            summary="We decided to use token bucket.",
            key_decisions=["Use token bucket", "Redis backend"],
            rationale="Performance and simplicity",
            supporting_evidence=[{"source": "paper", "summary": "Token bucket is O(1)"}],
            dissenting_views=["Leaky bucket is simpler"],
            open_questions=["What about distributed mode?"],
            consensus_confidence=0.85,
            rounds_used=3,
        )
        md = memo.to_markdown()
        assert "# Decision Memo: Rate Limiter" in md
        assert "85%" in md
        assert "Use token bucket" in md
        assert "Redis backend" in md
        assert "Performance and simplicity" in md
        assert "Leaky bucket is simpler" in md
        assert "What about distributed mode?" in md
        assert "paper" in md

    def test_to_markdown_empty_dissenting_views(self):
        memo = DecisionMemo(
            debate_id="d1",
            title="T",
            summary="S",
            key_decisions=[],
            rationale="R",
            supporting_evidence=[],
            dissenting_views=[],
            open_questions=[],
        )
        md = memo.to_markdown()
        assert "*None recorded*" in md

    def test_to_markdown_empty_open_questions(self):
        memo = DecisionMemo(
            debate_id="d1",
            title="T",
            summary="S",
            key_decisions=[],
            rationale="R",
            supporting_evidence=[],
            dissenting_views=["x"],
            open_questions=[],
        )
        md = memo.to_markdown()
        assert "*None*" in md

    def test_to_markdown_evidence_truncated(self):
        long_summary = "x" * 200
        memo = DecisionMemo(
            debate_id="d1",
            title="T",
            summary="S",
            key_decisions=[],
            rationale="R",
            supporting_evidence=[{"source": "src", "summary": long_summary}],
            dissenting_views=[],
            open_questions=[],
        )
        md = memo.to_markdown()
        # Evidence summary is truncated to 100 chars in markdown
        assert "x" * 100 in md
        assert "x" * 200 not in md


# ===================================================================
# PatchPlan tests
# ===================================================================


class TestPatchPlan:
    def test_creation_defaults(self):
        plan = PatchPlan(
            debate_id="d1",
            title="T",
            description="D",
            steps=[],
            file_changes=[],
            dependencies=[],
            estimated_complexity="low",
        )
        assert plan.estimated_complexity == "low"
        assert plan.created_at  # auto

    def test_to_dict(self):
        plan = PatchPlan(
            debate_id="d1",
            title="T",
            description="D",
            steps=[{"step_num": 1, "action": "Do thing"}],
            file_changes=[{"path": "a.py", "action": "modify", "description": "change"}],
            dependencies=["redis"],
            estimated_complexity="high",
        )
        d = plan.to_dict()
        assert d["debate_id"] == "d1"
        assert d["estimated_complexity"] == "high"
        assert len(d["steps"]) == 1
        assert d["dependencies"] == ["redis"]

    def test_to_markdown_with_steps_and_files(self):
        plan = PatchPlan(
            debate_id="d1",
            title="Rate Limiter Impl",
            description="Implement rate limiting",
            steps=[
                {
                    "step_num": 1,
                    "action": "Add module",
                    "target": "rate_limiter.py",
                    "details": "Create rate limiter module",
                    "verification": "Unit tests",
                }
            ],
            file_changes=[
                {"path": "rate_limiter.py", "action": "create", "description": "New module"}
            ],
            dependencies=["redis", "aiohttp"],
            estimated_complexity="medium",
        )
        md = plan.to_markdown()
        assert "# Patch Plan: Rate Limiter Impl" in md
        assert "MEDIUM" in md
        assert "Step 1" in md
        assert "rate_limiter.py" in md
        assert "redis" in md
        assert "aiohttp" in md

    def test_to_markdown_no_dependencies(self):
        plan = PatchPlan(
            debate_id="d1",
            title="T",
            description="D",
            steps=[],
            file_changes=[],
            dependencies=[],
            estimated_complexity="low",
        )
        md = plan.to_markdown()
        assert "*None*" in md


# ===================================================================
# PRGenerator tests
# ===================================================================


class TestPRGenerator:
    def test_generate_decision_memo_basic(self):
        artifact = _make_artifact()
        gen = PRGenerator(artifact)
        memo = gen.generate_decision_memo()

        assert memo.debate_id == "debate-001"
        assert memo.consensus_confidence == 0.85
        assert memo.rounds_used == 3
        assert memo.agent_count == 3
        assert isinstance(memo.key_decisions, list)

    def test_generate_decision_memo_extracts_decisions(self):
        artifact = _make_artifact(
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                final_answer="1. Use token bucket\n2. Add Redis\n- Configure TTL",
                vote_breakdown={},
                rounds_used=2,
            )
        )
        gen = PRGenerator(artifact)
        memo = gen.generate_decision_memo()
        assert len(memo.key_decisions) > 0

    def test_generate_decision_memo_with_provenance(self):
        artifact = _make_artifact(
            provenance_data={
                "chain": {
                    "records": [
                        {"source_type": "paper", "content": "Token bucket is efficient"},
                        {"source_type": "benchmark", "content": "10k req/s sustained"},
                    ]
                }
            }
        )
        gen = PRGenerator(artifact)
        memo = gen.generate_decision_memo()
        assert len(memo.supporting_evidence) == 2
        assert memo.supporting_evidence[0]["source"] == "paper"

    def test_generate_decision_memo_with_trace_rationale(self):
        artifact = _make_artifact(
            trace_data={
                "events": [
                    {
                        "event_type": "agent_synthesis",
                        "content": {
                            "content": "Synthesis: token bucket wins because of O(1) perf."
                        },
                    }
                ]
            }
        )
        gen = PRGenerator(artifact)
        memo = gen.generate_decision_memo()
        assert "token bucket" in memo.rationale

    def test_generate_decision_memo_no_trace(self):
        artifact = _make_artifact(trace_data=None)
        gen = PRGenerator(artifact)
        memo = gen.generate_decision_memo()
        assert "multi-agent debate" in memo.rationale.lower()

    def test_generate_decision_memo_no_consensus(self):
        artifact = _make_artifact(consensus_proof=None)
        gen = PRGenerator(artifact)
        memo = gen.generate_decision_memo()
        assert memo.consensus_confidence == 0
        assert memo.summary == ""

    def test_generate_decision_memo_long_summary_truncated(self):
        long_answer = "A" * 600
        artifact = _make_artifact(
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                final_answer=long_answer,
                vote_breakdown={},
                rounds_used=2,
            )
        )
        gen = PRGenerator(artifact)
        memo = gen.generate_decision_memo()
        assert memo.summary.endswith("...")
        assert len(memo.summary) == 503  # 500 + "..."

    def test_generate_patch_plan_basic(self):
        artifact = _make_artifact()
        gen = PRGenerator(artifact)
        plan = gen.generate_patch_plan()

        assert plan.debate_id == "debate-001"
        assert plan.description == artifact.task
        assert isinstance(plan.steps, list)
        assert len(plan.steps) > 0

    def test_generate_patch_plan_extracts_steps(self):
        artifact = _make_artifact(
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                final_answer="1. Create the module for handling\n2. Add rate limiting logic here\n3. Write tests for coverage",
                vote_breakdown={},
                rounds_used=2,
            )
        )
        gen = PRGenerator(artifact)
        plan = gen.generate_patch_plan()
        assert len(plan.steps) >= 3

    def test_generate_patch_plan_generic_steps_when_none_found(self):
        artifact = _make_artifact(
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                final_answer="The solution is straightforward.",
                vote_breakdown={},
                rounds_used=2,
            )
        )
        gen = PRGenerator(artifact)
        plan = gen.generate_patch_plan()
        assert len(plan.steps) == 3
        assert plan.steps[0]["action"] == "Review debate conclusions"

    def test_generate_patch_plan_infers_file_changes(self):
        artifact = _make_artifact(
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                final_answer="Modify `rate_limiter.py` and `config.yaml` to add limits.",
                vote_breakdown={},
                rounds_used=2,
            )
        )
        gen = PRGenerator(artifact)
        plan = gen.generate_patch_plan()
        paths = [c["path"] for c in plan.file_changes]
        assert "rate_limiter.py" in paths
        assert "config.yaml" in paths

    def test_estimate_complexity_low(self):
        artifact = _make_artifact(rounds=1, critique_count=1)
        gen = PRGenerator(artifact)
        assert gen._estimate_complexity() == "low"

    def test_estimate_complexity_medium(self):
        artifact = _make_artifact(rounds=2, critique_count=3)
        gen = PRGenerator(artifact)
        assert gen._estimate_complexity() == "medium"

    def test_estimate_complexity_high(self):
        artifact = _make_artifact(rounds=5, critique_count=8)
        gen = PRGenerator(artifact)
        assert gen._estimate_complexity() == "high"

    def test_extract_title_short_task(self):
        artifact = _make_artifact(task="Fix bug")
        gen = PRGenerator(artifact)
        assert gen._extract_title("Fix bug") == "Fix bug"

    def test_extract_title_with_period(self):
        artifact = _make_artifact(task="Fix the bug. Then test.")
        gen = PRGenerator(artifact)
        assert gen._extract_title("Fix the bug. Then test.") == "Fix the bug."

    def test_extract_title_long_task(self):
        long_task = "A" * 100
        artifact = _make_artifact(task=long_task)
        gen = PRGenerator(artifact)
        title = gen._extract_title(long_task)
        assert title.endswith("...")
        assert len(title) == 63  # 60 + "..."

    def test_extract_decisions_numbered(self):
        gen = PRGenerator(_make_artifact())
        decisions = gen._extract_decisions(
            "1. First decision here about something\n2. Second decision here about something"
        )
        assert len(decisions) == 2

    def test_extract_decisions_bullet(self):
        gen = PRGenerator(_make_artifact())
        decisions = gen._extract_decisions(
            "- Bullet decision about something important\n* Another bullet decision text"
        )
        assert len(decisions) == 2

    def test_extract_decisions_fallback_sentences(self):
        gen = PRGenerator(_make_artifact())
        text = "This is a long sentence about the architecture decision we made."
        decisions = gen._extract_decisions(text)
        assert len(decisions) >= 1

    def test_extract_decisions_empty(self):
        gen = PRGenerator(_make_artifact())
        assert gen._extract_decisions("") == []

    def test_extract_decisions_limit_10(self):
        gen = PRGenerator(_make_artifact())
        lines = "\n".join(f"{i}. Decision number {i} about something" for i in range(1, 20))
        decisions = gen._extract_decisions(lines)
        assert len(decisions) <= 10

    def test_extract_dissenting_views_from_consensus(self):
        consensus = MagicMock()
        consensus.dissenting_views = ["View A", "View B"]
        consensus.critiques = []
        gen = PRGenerator(_make_artifact())
        views = gen._extract_dissenting_views(consensus)
        assert "View A" in views
        assert "View B" in views

    def test_extract_dissenting_views_from_critiques(self):
        critique = MagicMock()
        critique.severity = 0.8
        critique.agent = "gpt4"
        critique.issues = ["Security concern about token storage"]
        consensus = MagicMock()
        consensus.dissenting_views = []
        consensus.critiques = [critique]
        gen = PRGenerator(_make_artifact())
        views = gen._extract_dissenting_views(consensus)
        assert any("gpt4" in v for v in views)

    def test_extract_dissenting_views_none(self):
        gen = PRGenerator(_make_artifact())
        assert gen._extract_dissenting_views(None) == []

    def test_extract_dissenting_views_limit_5(self):
        consensus = MagicMock()
        consensus.dissenting_views = [f"View {i}" for i in range(10)]
        consensus.critiques = []
        gen = PRGenerator(_make_artifact())
        views = gen._extract_dissenting_views(consensus)
        assert len(views) <= 5

    def test_extract_open_questions_recurring(self):
        c1 = MagicMock()
        c1.issues = ["how does it scale under load?"]
        c2 = MagicMock()
        c2.issues = ["how does it scale under load?"]
        consensus = MagicMock()
        consensus.critiques = [c1, c2]
        gen = PRGenerator(_make_artifact())
        questions = gen._extract_open_questions(consensus)
        assert len(questions) >= 1

    def test_extract_open_questions_none(self):
        gen = PRGenerator(_make_artifact())
        assert gen._extract_open_questions(None) == []

    def test_extract_open_questions_no_critiques_attr(self):
        consensus = MagicMock(spec=[])  # no critiques attribute
        gen = PRGenerator(_make_artifact())
        assert gen._extract_open_questions(consensus) == []

    def test_extract_open_questions_limit_3(self):
        # Create 5 critiques all raising the same 3+ issues
        critiques = []
        for _ in range(5):
            c = MagicMock()
            c.issues = [
                "question alpha about something",
                "question beta about something",
                "question gamma about something",
                "question delta about something",
            ]
            critiques.append(c)
        consensus = MagicMock()
        consensus.critiques = critiques
        gen = PRGenerator(_make_artifact())
        questions = gen._extract_open_questions(consensus)
        assert len(questions) <= 3


# ===================================================================
# generate_pr_artifacts convenience function
# ===================================================================


class TestGeneratePRArtifacts:
    def test_returns_both_artifacts(self):
        artifact = _make_artifact()
        result = generate_pr_artifacts(artifact)
        assert "decision_memo" in result
        assert "patch_plan" in result
        assert isinstance(result["decision_memo"], DecisionMemo)
        assert isinstance(result["patch_plan"], PatchPlan)


# ===================================================================
# RiskLevel / RiskCategory enums
# ===================================================================


class TestRiskEnums:
    def test_risk_level_values(self):
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_risk_category_values(self):
        assert RiskCategory.TECHNICAL.value == "technical"
        assert RiskCategory.SECURITY.value == "security"
        assert RiskCategory.PERFORMANCE.value == "performance"
        assert RiskCategory.SCALABILITY.value == "scalability"
        assert RiskCategory.MAINTAINABILITY.value == "maintainability"
        assert RiskCategory.COMPATIBILITY.value == "compatibility"
        assert RiskCategory.UNKNOWN.value == "unknown"


# ===================================================================
# Risk dataclass
# ===================================================================


class TestRisk:
    def test_creation_defaults(self):
        r = Risk(
            id="r1",
            title="T",
            description="D",
            level=RiskLevel.LOW,
            category=RiskCategory.TECHNICAL,
            source="test",
        )
        assert r.impact == 0.5
        assert r.likelihood == 0.5
        assert r.mitigation == ""
        assert r.mitigation_status == "proposed"
        assert r.related_critique_ids == []
        assert r.related_claim_ids == []

    def test_risk_score(self):
        r = Risk(
            id="r1",
            title="T",
            description="D",
            level=RiskLevel.HIGH,
            category=RiskCategory.SECURITY,
            source="s",
            impact=0.8,
            likelihood=0.6,
        )
        assert r.risk_score == pytest.approx(0.48)

    def test_to_dict(self):
        r = Risk(
            id="r1",
            title="Title",
            description="Desc",
            level=RiskLevel.CRITICAL,
            category=RiskCategory.PERFORMANCE,
            source="agent",
            impact=0.9,
            likelihood=0.7,
            mitigation="Add cache",
            mitigation_status="in_progress",
            related_critique_ids=["c1"],
            related_claim_ids=["cl1"],
        )
        d = r.to_dict()
        assert d["id"] == "r1"
        assert d["level"] == "critical"
        assert d["category"] == "performance"
        assert d["risk_score"] == pytest.approx(0.63)
        assert d["mitigation"] == "Add cache"
        assert d["mitigation_status"] == "in_progress"


# ===================================================================
# RiskRegister
# ===================================================================


class TestRiskRegister:
    def _make_register(self) -> RiskRegister:
        reg = RiskRegister(debate_id="d1")
        reg.add_risk(
            Risk(
                id="r1",
                title="Low risk",
                description="D",
                level=RiskLevel.LOW,
                category=RiskCategory.TECHNICAL,
                source="s",
            )
        )
        reg.add_risk(
            Risk(
                id="r2",
                title="Med risk",
                description="D",
                level=RiskLevel.MEDIUM,
                category=RiskCategory.SECURITY,
                source="s",
            )
        )
        reg.add_risk(
            Risk(
                id="r3",
                title="High risk",
                description="D",
                level=RiskLevel.HIGH,
                category=RiskCategory.PERFORMANCE,
                source="s",
            )
        )
        reg.add_risk(
            Risk(
                id="r4",
                title="Crit risk",
                description="D",
                level=RiskLevel.CRITICAL,
                category=RiskCategory.SECURITY,
                source="s",
            )
        )
        return reg

    def test_add_risk(self):
        reg = RiskRegister(debate_id="d1")
        assert len(reg.risks) == 0
        reg.add_risk(
            Risk(
                id="r1",
                title="T",
                description="D",
                level=RiskLevel.LOW,
                category=RiskCategory.TECHNICAL,
                source="s",
            )
        )
        assert len(reg.risks) == 1

    def test_get_by_level(self):
        reg = self._make_register()
        assert len(reg.get_by_level(RiskLevel.LOW)) == 1
        assert len(reg.get_by_level(RiskLevel.CRITICAL)) == 1

    def test_get_by_category(self):
        reg = self._make_register()
        assert len(reg.get_by_category(RiskCategory.SECURITY)) == 2

    def test_get_unmitigated(self):
        reg = self._make_register()
        assert len(reg.get_unmitigated()) == 4
        reg.risks[0].mitigation_status = "implemented"
        assert len(reg.get_unmitigated()) == 3

    def test_get_critical_risks(self):
        reg = self._make_register()
        critical = reg.get_critical_risks()
        assert len(critical) == 2  # HIGH + CRITICAL

    def test_summary(self):
        reg = self._make_register()
        s = reg.summary
        assert s["total_risks"] == 4
        assert s["critical"] == 1
        assert s["high"] == 1
        assert s["medium"] == 1
        assert s["low"] == 1
        assert s["unmitigated"] == 4
        assert s["avg_risk_score"] == pytest.approx(0.25)  # 0.5*0.5 for all

    def test_summary_empty(self):
        reg = RiskRegister(debate_id="d1")
        assert reg.summary["avg_risk_score"] == 0

    def test_to_markdown(self):
        reg = self._make_register()
        md = reg.to_markdown()
        assert "# Risk Register" in md
        assert "CRITICAL" in md
        assert "HIGH" in md
        assert "Total Risks" in md

    def test_to_dict(self):
        reg = self._make_register()
        d = reg.to_dict()
        assert d["debate_id"] == "d1"
        assert len(d["risks"]) == 4
        assert "summary" in d
        assert "thresholds" in d

    def test_thresholds_configurable(self):
        reg = RiskRegister(
            debate_id="d1", low_support_threshold=0.3, critical_support_threshold=0.9
        )
        assert reg.low_support_threshold == 0.3
        assert reg.critical_support_threshold == 0.9


# ===================================================================
# RiskAnalyzer
# ===================================================================


class TestRiskAnalyzer:
    def test_analyze_empty_artifact(self):
        artifact = _make_artifact(
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                final_answer="OK",
                vote_breakdown={},
                rounds_used=2,
            ),
            trace_data=None,
            verification_results=[],
        )
        analyzer = RiskAnalyzer(artifact)
        reg = analyzer.analyze()
        assert reg.debate_id == "debate-001"
        assert len(reg.risks) == 0

    def test_analyze_low_confidence_consensus(self):
        artifact = _make_artifact(
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.4,
                final_answer="Maybe",
                vote_breakdown={},
                rounds_used=2,
            ),
        )
        analyzer = RiskAnalyzer(artifact)
        reg = analyzer.analyze()
        low_conf = [r for r in reg.risks if r.id == "consensus-low-confidence"]
        assert len(low_conf) == 1
        assert low_conf[0].level == RiskLevel.HIGH  # <0.5 => HIGH

    def test_analyze_medium_confidence_consensus(self):
        artifact = _make_artifact(
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.6,
                final_answer="Probably",
                vote_breakdown={},
                rounds_used=2,
            ),
        )
        analyzer = RiskAnalyzer(artifact)
        reg = analyzer.analyze()
        low_conf = [r for r in reg.risks if r.id == "consensus-low-confidence"]
        assert len(low_conf) == 1
        assert low_conf[0].level == RiskLevel.MEDIUM

    def test_analyze_no_consensus_reached(self):
        artifact = _make_artifact(
            consensus_proof=ConsensusProof(
                reached=False,
                confidence=0.3,
                final_answer="",
                vote_breakdown={},
                rounds_used=2,
            ),
        )
        analyzer = RiskAnalyzer(artifact)
        reg = analyzer.analyze()
        no_consensus = [r for r in reg.risks if r.id == "consensus-not-reached"]
        assert len(no_consensus) == 1
        assert no_consensus[0].level == RiskLevel.HIGH

    def test_analyze_critiques(self):
        artifact = _make_artifact(
            trace_data={
                "events": [
                    {
                        "event_type": "agent_critique",
                        "agent": "gpt4",
                        "content": {
                            "severity": 0.8,
                            "issues": ["Security vulnerability in auth flow"],
                            "suggestions": ["Use OAuth2 instead"],
                        },
                    }
                ]
            },
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                final_answer="OK",
                vote_breakdown={},
                rounds_used=2,
            ),
        )
        analyzer = RiskAnalyzer(artifact)
        reg = analyzer.analyze()
        assert len(reg.risks) >= 1
        assert reg.risks[0].level == RiskLevel.HIGH  # severity >= 0.8

    def test_analyze_critiques_medium_severity(self):
        artifact = _make_artifact(
            trace_data={
                "events": [
                    {
                        "event_type": "agent_critique",
                        "agent": "gemini",
                        "content": {
                            "severity": 0.65,
                            "issues": ["Performance might degrade under load"],
                            "suggestions": [],
                        },
                    }
                ]
            },
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                final_answer="OK",
                vote_breakdown={},
                rounds_used=2,
            ),
        )
        analyzer = RiskAnalyzer(artifact)
        reg = analyzer.analyze()
        assert len(reg.risks) >= 1
        assert reg.risks[0].level == RiskLevel.MEDIUM

    def test_analyze_critiques_low_severity_ignored(self):
        artifact = _make_artifact(
            trace_data={
                "events": [
                    {
                        "event_type": "agent_critique",
                        "agent": "claude",
                        "content": {
                            "severity": 0.3,
                            "issues": ["Minor style issue"],
                            "suggestions": [],
                        },
                    }
                ]
            },
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                final_answer="OK",
                vote_breakdown={},
                rounds_used=2,
            ),
        )
        analyzer = RiskAnalyzer(artifact)
        reg = analyzer.analyze()
        assert len(reg.risks) == 0  # severity < 0.6 ignored

    def test_analyze_verification_refuted(self):
        artifact = _make_artifact(
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                final_answer="OK",
                vote_breakdown={},
                rounds_used=2,
            ),
            verification_results=[
                VerificationResult(
                    claim_id="c1",
                    claim_text="The algorithm is correct for all inputs",
                    status="refuted",
                    method="z3",
                )
            ],
        )
        analyzer = RiskAnalyzer(artifact)
        reg = analyzer.analyze()
        refuted = [r for r in reg.risks if "verification" in r.id]
        assert len(refuted) == 1
        assert refuted[0].level == RiskLevel.HIGH

    def test_analyze_verification_timeout(self):
        artifact = _make_artifact(
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                final_answer="OK",
                vote_breakdown={},
                rounds_used=2,
            ),
            verification_results=[
                VerificationResult(
                    claim_id="c2",
                    claim_text="Termination is guaranteed",
                    status="timeout",
                    method="lean",
                )
            ],
        )
        analyzer = RiskAnalyzer(artifact)
        reg = analyzer.analyze()
        timeout_risks = [r for r in reg.risks if "verification" in r.id]
        assert len(timeout_risks) == 1
        assert timeout_risks[0].level == RiskLevel.MEDIUM

    def test_analyze_verification_verified_no_risk(self):
        artifact = _make_artifact(
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                final_answer="OK",
                vote_breakdown={},
                rounds_used=2,
            ),
            verification_results=[
                VerificationResult(claim_id="c3", claim_text="Safe", status="verified", method="z3")
            ],
        )
        analyzer = RiskAnalyzer(artifact)
        reg = analyzer.analyze()
        assert len(reg.risks) == 0

    def test_categorize_issue_security(self):
        analyzer = RiskAnalyzer(_make_artifact())
        assert analyzer._categorize_issue("security vulnerability found") == RiskCategory.SECURITY
        assert analyzer._categorize_issue("auth bypass possible") == RiskCategory.SECURITY

    def test_categorize_issue_performance(self):
        analyzer = RiskAnalyzer(_make_artifact())
        assert analyzer._categorize_issue("slow response times") == RiskCategory.PERFORMANCE
        assert analyzer._categorize_issue("high latency detected") == RiskCategory.PERFORMANCE

    def test_categorize_issue_scalability(self):
        analyzer = RiskAnalyzer(_make_artifact())
        assert analyzer._categorize_issue("cannot scale to 1M users") == RiskCategory.SCALABILITY
        assert analyzer._categorize_issue("throughput limited") == RiskCategory.SCALABILITY

    def test_categorize_issue_maintainability(self):
        analyzer = RiskAnalyzer(_make_artifact())
        assert analyzer._categorize_issue("code is too complex") == RiskCategory.MAINTAINABILITY
        assert analyzer._categorize_issue("hard to test") == RiskCategory.MAINTAINABILITY

    def test_categorize_issue_compatibility(self):
        analyzer = RiskAnalyzer(_make_artifact())
        assert analyzer._categorize_issue("version incompatibility") == RiskCategory.COMPATIBILITY
        assert analyzer._categorize_issue("dependency conflict") == RiskCategory.COMPATIBILITY

    def test_categorize_issue_default_technical(self):
        analyzer = RiskAnalyzer(_make_artifact())
        assert analyzer._categorize_issue("something else entirely") == RiskCategory.TECHNICAL

    def test_no_consensus_proof(self):
        artifact = _make_artifact(consensus_proof=None)
        analyzer = RiskAnalyzer(artifact)
        reg = analyzer.analyze()
        assert len([r for r in reg.risks if "consensus" in r.id]) == 0


# ===================================================================
# generate_risk_register convenience function
# ===================================================================


class TestGenerateRiskRegister:
    def test_returns_register(self):
        artifact = _make_artifact()
        reg = generate_risk_register(artifact)
        assert isinstance(reg, RiskRegister)
        assert reg.debate_id == "debate-001"


# ===================================================================
# VerificationType / CasePriority enums
# ===================================================================


class TestVerificationEnums:
    def test_verification_type_values(self):
        assert VerificationType.UNIT.value == "unit"
        assert VerificationType.INTEGRATION.value == "integration"
        assert VerificationType.E2E.value == "e2e"
        assert VerificationType.PERFORMANCE.value == "performance"
        assert VerificationType.SECURITY.value == "security"
        assert VerificationType.REGRESSION.value == "regression"

    def test_case_priority_values(self):
        assert CasePriority.P0.value == "p0"
        assert CasePriority.P1.value == "p1"
        assert CasePriority.P2.value == "p2"
        assert CasePriority.P3.value == "p3"

    def test_not_collected_by_pytest(self):
        assert VerificationType.__test__ is False
        assert CasePriority.__test__ is False


# ===================================================================
# VerificationCase
# ===================================================================


class TestVerificationCase:
    def test_creation_defaults(self):
        tc = VerificationCase(
            id="tc1",
            title="Test",
            description="Desc",
            test_type=VerificationType.UNIT,
            priority=CasePriority.P1,
        )
        assert tc.preconditions == []
        assert tc.steps == []
        assert tc.expected_result == ""
        assert tc.automated is False
        assert tc.implemented is False

    def test_to_dict(self):
        tc = VerificationCase(
            id="tc1",
            title="Test title",
            description="Test desc",
            test_type=VerificationType.INTEGRATION,
            priority=CasePriority.P0,
            preconditions=["DB running"],
            steps=["Call API", "Check response"],
            expected_result="200 OK",
            automated=True,
            implemented=True,
            related_claim_ids=["cl1"],
            related_critique_ids=["cr1"],
        )
        d = tc.to_dict()
        assert d["test_type"] == "integration"
        assert d["priority"] == "p0"
        assert d["automated"] is True
        assert d["implemented"] is True
        assert d["related_claim_ids"] == ["cl1"]

    def test_not_collected_by_pytest(self):
        assert VerificationCase.__test__ is False


# ===================================================================
# VerificationPlan
# ===================================================================


class TestVerificationPlan:
    def _make_plan(self) -> VerificationPlan:
        plan = VerificationPlan(debate_id="d1", title="T", description="D")
        plan.add_test(
            VerificationCase(
                id="t1",
                title="Unit test",
                description="D",
                test_type=VerificationType.UNIT,
                priority=CasePriority.P0,
                implemented=True,
                automated=True,
            )
        )
        plan.add_test(
            VerificationCase(
                id="t2",
                title="Integration test",
                description="D",
                test_type=VerificationType.INTEGRATION,
                priority=CasePriority.P1,
            )
        )
        plan.add_test(
            VerificationCase(
                id="t3",
                title="E2E test",
                description="D",
                test_type=VerificationType.E2E,
                priority=CasePriority.P2,
            )
        )
        return plan

    def test_add_test(self):
        plan = VerificationPlan(debate_id="d1", title="T", description="D")
        assert len(plan.test_cases) == 0
        plan.add_test(
            VerificationCase(
                id="t1",
                title="T",
                description="D",
                test_type=VerificationType.UNIT,
                priority=CasePriority.P0,
            )
        )
        assert len(plan.test_cases) == 1

    def test_get_by_type(self):
        plan = self._make_plan()
        assert len(plan.get_by_type(VerificationType.UNIT)) == 1
        assert len(plan.get_by_type(VerificationType.E2E)) == 1
        assert len(plan.get_by_type(VerificationType.SECURITY)) == 0

    def test_get_by_priority(self):
        plan = self._make_plan()
        assert len(plan.get_by_priority(CasePriority.P0)) == 1
        assert len(plan.get_by_priority(CasePriority.P3)) == 0

    def test_get_unimplemented(self):
        plan = self._make_plan()
        assert len(plan.get_unimplemented()) == 2

    def test_summary(self):
        plan = self._make_plan()
        s = plan.summary
        assert s["total_tests"] == 3
        assert s["by_type"]["unit"] == 1
        assert s["by_type"]["integration"] == 1
        assert s["automated"] == 1
        assert s["implemented"] == 1

    def test_target_coverage_default(self):
        plan = VerificationPlan(debate_id="d1", title="T", description="D")
        assert plan.target_coverage == 0.8

    def test_to_markdown(self):
        plan = self._make_plan()
        plan.critical_paths = ["/api/auth", "/api/data"]
        md = plan.to_markdown()
        assert "# Test Plan:" in md
        assert "Unit test" in md
        assert "/api/auth" in md
        assert "Total Tests" in md

    def test_to_markdown_no_critical_paths(self):
        plan = VerificationPlan(debate_id="d1", title="T", description="D")
        md = plan.to_markdown()
        assert "TBD based on implementation" in md

    def test_to_dict(self):
        plan = self._make_plan()
        d = plan.to_dict()
        assert d["debate_id"] == "d1"
        assert len(d["test_cases"]) == 3
        assert "summary" in d
        assert d["target_coverage"] == 0.8

    def test_not_collected_by_pytest(self):
        assert VerificationPlan.__test__ is False


# ===================================================================
# VerificationPlanGenerator
# ===================================================================


class TestVerificationPlanGenerator:
    def test_generate_basic(self):
        artifact = _make_artifact()
        gen = VerificationPlanGenerator(artifact)
        plan = gen.generate()
        assert plan.debate_id == "debate-001"
        assert len(plan.test_cases) >= 2  # at least smoke + regression

    def test_generate_adds_standard_tests(self):
        artifact = _make_artifact(
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                final_answer="Simple answer",
                vote_breakdown={},
                rounds_used=2,
            ),
            trace_data=None,
            verification_results=[],
        )
        gen = VerificationPlanGenerator(artifact)
        plan = gen.generate()
        ids = [t.id for t in plan.test_cases]
        assert "smoke-1" in ids
        assert "regression-1" in ids

    def test_generate_consensus_tests(self):
        artifact = _make_artifact(
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                final_answer="Implement the rate limiter\nUse Redis for state\nAdd monitoring\nCreate dashboard\nEnsure reliability",
                vote_breakdown={},
                rounds_used=2,
            ),
        )
        gen = VerificationPlanGenerator(artifact)
        plan = gen.generate()
        consensus_tests = [t for t in plan.test_cases if t.id.startswith("consensus-")]
        assert len(consensus_tests) >= 1
        assert len(consensus_tests) <= 5

    def test_generate_critique_tests(self):
        artifact = _make_artifact(
            trace_data={
                "events": [
                    {
                        "event_type": "agent_critique",
                        "event_id": "ev1",
                        "content": {
                            "issues": ["Edge case: empty input", "Edge case: very large input"],
                        },
                    }
                ]
            },
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                final_answer="OK",
                vote_breakdown={},
                rounds_used=2,
            ),
        )
        gen = VerificationPlanGenerator(artifact)
        plan = gen.generate()
        critique_tests = [t for t in plan.test_cases if t.id.startswith("critique-")]
        assert len(critique_tests) >= 1

    def test_generate_verification_tests(self):
        artifact = _make_artifact(
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                final_answer="OK",
                vote_breakdown={},
                rounds_used=2,
            ),
            verification_results=[
                VerificationResult(
                    claim_id="c1", claim_text="Algorithm terminates", status="verified", method="z3"
                ),
            ],
        )
        gen = VerificationPlanGenerator(artifact)
        plan = gen.generate()
        formal_tests = [t for t in plan.test_cases if t.id.startswith("formal-")]
        assert len(formal_tests) == 1
        assert formal_tests[0].priority == CasePriority.P0
        assert formal_tests[0].automated is True

    def test_generate_skips_non_verified(self):
        artifact = _make_artifact(
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                final_answer="OK",
                vote_breakdown={},
                rounds_used=2,
            ),
            verification_results=[
                VerificationResult(
                    claim_id="c1", claim_text="Claim", status="refuted", method="z3"
                ),
            ],
        )
        gen = VerificationPlanGenerator(artifact)
        plan = gen.generate()
        formal_tests = [t for t in plan.test_cases if t.id.startswith("formal-")]
        assert len(formal_tests) == 0

    def test_generate_no_consensus(self):
        artifact = _make_artifact(consensus_proof=None)
        gen = VerificationPlanGenerator(artifact)
        plan = gen.generate()
        assert len(plan.test_cases) >= 2  # smoke + regression

    def test_extract_title_with_period(self):
        artifact = _make_artifact(task="Fix the auth bug. Urgently.")
        gen = VerificationPlanGenerator(artifact)
        assert gen._extract_title() == "Fix the auth bug."

    def test_extract_title_truncation(self):
        artifact = _make_artifact(task="A" * 100)
        gen = VerificationPlanGenerator(artifact)
        title = gen._extract_title()
        assert title.endswith("...")
        assert len(title) == 63

    def test_not_collected_by_pytest(self):
        assert VerificationPlanGenerator.__test__ is False

    def test_skips_non_actionable_consensus_lines(self):
        artifact = _make_artifact(
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                final_answer="The answer is 42.\nNothing here.\nJust a thought.",
                vote_breakdown={},
                rounds_used=2,
            ),
        )
        gen = VerificationPlanGenerator(artifact)
        plan = gen.generate()
        consensus_tests = [t for t in plan.test_cases if t.id.startswith("consensus-")]
        assert len(consensus_tests) == 0


# ===================================================================
# generate_test_plan convenience function
# ===================================================================


class TestGenerateTestPlan:
    def test_returns_plan(self):
        artifact = _make_artifact()
        plan = generate_test_plan(artifact)
        assert isinstance(plan, VerificationPlan)
        assert plan.debate_id == "debate-001"


# ===================================================================
# Backward compatibility aliases
# ===================================================================


class TestBackwardCompatibility:
    def test_test_plan_alias(self):
        assert TestPlan is VerificationPlan

    def test_test_case_alias(self):
        assert TestCase is VerificationCase

    def test_test_plan_generator_alias(self):
        assert TestPlanGenerator is VerificationPlanGenerator
