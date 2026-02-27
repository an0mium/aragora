"""Tests for the Pipeline Outcome Feedback system."""

from __future__ import annotations
from unittest.mock import MagicMock
import pytest
from aragora.pipeline.outcome_feedback import (
    AgentContribution,
    OutcomeFeedbackRecorder,
    PipelineOutcome,
)


@pytest.fixture
def recorder():
    return OutcomeFeedbackRecorder()


def make_outcome(**kwargs) -> PipelineOutcome:
    defaults = {
        "pipeline_id": "test-001",
        "run_type": "self_improvement",
        "domain": "technical",
        "questions_asked": 5,
        "questions_answered": 4,
        "answers_changed_default": 2,
        "spec_completeness": 0.8,
        "execution_succeeded": True,
        "tests_passed": 42,
        "tests_failed": 0,
        "files_changed": 5,
        "human_interventions": 0,
        "total_duration_s": 300.0,
    }
    defaults.update(kwargs)
    return PipelineOutcome(**defaults)


class TestRecording:
    def test_record_basic(self, recorder):
        recorder.record(make_outcome())
        assert len(recorder._outcomes) == 1

    def test_record_multiple(self, recorder):
        for i in range(5):
            recorder.record(make_outcome(pipeline_id=f"run-{i}"))
        assert len(recorder._outcomes) == 5

    def test_get_recent(self, recorder):
        for i in range(10):
            recorder.record(make_outcome(pipeline_id=f"run-{i}"))
        assert len(recorder.get_recent_outcomes(limit=3)) == 3

    def test_get_recent_filtered(self, recorder):
        recorder.record(make_outcome(pipeline_id="self-1", run_type="self_improvement"))
        recorder.record(make_outcome(pipeline_id="user-1", run_type="user_project"))
        recorder.record(make_outcome(pipeline_id="self-2", run_type="self_improvement"))
        assert len(recorder.get_recent_outcomes(run_type="self_improvement")) == 2


class TestQualityScore:
    def test_successful_high_quality(self, recorder):
        o = make_outcome(
            execution_succeeded=True,
            spec_completeness=1.0,
            questions_asked=5,
            questions_answered=5,
            human_interventions=0,
            rollback_triggered=False,
        )
        recorder.record(o)
        assert o.overall_quality_score >= 0.8

    def test_failed_low_quality(self, recorder):
        o = make_outcome(
            execution_succeeded=False,
            tests_passed=0,
            tests_failed=10,
            spec_completeness=0.2,
            questions_asked=5,
            questions_answered=1,
            human_interventions=5,
            rollback_triggered=True,
        )
        recorder.record(o)
        assert o.overall_quality_score < 0.4

    def test_partial_success(self, recorder):
        o = make_outcome(
            execution_succeeded=False, tests_passed=8, tests_failed=2, spec_completeness=0.6
        )
        recorder.record(o)
        assert 0.3 < o.overall_quality_score < 0.8


class TestQualityTrend:
    def test_trend(self, recorder):
        for i in range(5):
            recorder.record(make_outcome(pipeline_id=f"run-{i}", spec_completeness=i * 0.2))
        trend = recorder.get_quality_trend(window=5)
        assert len(trend) == 5 and trend[-1] >= trend[0]


class TestAgentPerformance:
    def test_contributions(self, recorder):
        o = make_outcome(
            agent_contributions=[
                AgentContribution(
                    agent_name="claude", provider="anthropic", phase="debate", influence_score=0.8
                ),
                AgentContribution(
                    agent_name="gpt4", provider="openai", phase="debate", influence_score=0.6
                ),
                AgentContribution(
                    agent_name="claude",
                    provider="anthropic",
                    phase="execution",
                    influence_score=0.9,
                ),
            ]
        )
        recorder.record(o)
        perf = recorder.get_agent_phase_performance()
        assert perf["claude"]["debate"] == 0.8 and perf["gpt4"]["debate"] == 0.6


class TestKMIntegration:
    def test_ingest(self):
        km = MagicMock()
        recorder = OutcomeFeedbackRecorder(knowledge_mound=km)
        recorder.record(make_outcome())
        km.ingest.assert_called_once()
        assert km.ingest.call_args[0][0]["type"] == "pipeline_outcome"

    def test_km_failure_graceful(self):
        km = MagicMock()
        km.ingest.side_effect = RuntimeError("fail")
        recorder = OutcomeFeedbackRecorder(knowledge_mound=km)
        recorder.record(make_outcome())
        assert len(recorder._outcomes) == 1


class TestELOIntegration:
    def test_elo_high_influence(self):
        elo = MagicMock()
        recorder = OutcomeFeedbackRecorder(elo_system=elo)
        recorder.record(
            make_outcome(
                agent_contributions=[
                    AgentContribution(
                        agent_name="claude",
                        provider="anthropic",
                        phase="debate",
                        influence_score=0.8,
                    )
                ]
            )
        )
        elo.update_domain_elo.assert_called_with("claude", "technical:debate", won=True)

    def test_elo_low_influence(self):
        elo = MagicMock()
        recorder = OutcomeFeedbackRecorder(elo_system=elo)
        recorder.record(
            make_outcome(
                agent_contributions=[
                    AgentContribution(
                        agent_name="gpt4", provider="openai", phase="execution", influence_score=0.2
                    )
                ]
            )
        )
        elo.update_domain_elo.assert_called_with("gpt4", "technical:execution", won=False)


class TestSerialization:
    def test_to_dict(self):
        o = make_outcome(
            agent_contributions=[
                AgentContribution(
                    agent_name="claude",
                    provider="anthropic",
                    phase="debate",
                    influence_score=0.8,
                    truth_ratio=0.9,
                )
            ]
        )
        d = o.to_dict()
        assert d["pipeline_id"] == "test-001" and d["execution"]["succeeded"] is True
        assert d["agents"]["contributions"][0]["truth_ratio"] == 0.9
