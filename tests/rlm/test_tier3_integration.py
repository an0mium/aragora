"""
Comprehensive tests for RLM Tier 3 integration.

Covers:
1. PipelineRLMContext dataclass defaults and enriched flag
2. enrich_plan_context with mock RLM, continuum, mound, and debate_result
3. GauntletRLMFinding dataclass fields
4. GauntletRLMAnalysis dataclass fields
5. analyze_debate_for_gauntlet with heuristic pattern detection
6. _parse_gauntlet_findings internal helper
7. TrajectoryLearningData dataclass defaults
8. collect_trajectory_for_learning from mock RLMResult
9. store_trajectory_learning JSONL output
10. load_trajectory_insights round-trip and edge cases
11. Module exports from aragora.rlm
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.rlm.tier3_integration import (
    GauntletRLMAnalysis,
    GauntletRLMFinding,
    PipelineRLMContext,
    TrajectoryLearningData,
    _parse_gauntlet_findings,
    analyze_debate_for_gauntlet,
    collect_trajectory_for_learning,
    enrich_plan_context,
    load_trajectory_insights,
    store_trajectory_learning,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeMessage:
    """Fake debate message with dict-like interface."""

    agent: str
    content: str
    round: int = 0

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _make_debate_result(
    messages: list[dict[str, Any]] | None = None,
    *,
    task: str = "Design a rate limiter",
    debate_id: str = "debate-001",
    consensus_reached: bool = False,
    final_answer: str | None = None,
    confidence: float = 0.8,
) -> MagicMock:
    """Build a mock DebateResult suitable for load_debate_context."""
    result = MagicMock()
    result.messages = messages or []
    result.task = task
    result.debate_id = debate_id
    result.consensus_reached = consensus_reached
    result.final_answer = final_answer
    result.confidence = confidence
    return result


def _make_rlm_result(
    *,
    answer: str = "The answer",
    trajectory_log_path: str | None = "/tmp/log.jsonl",
    rlm_iterations: int = 3,
    code_blocks_executed: int = 5,
    confidence: float = 0.9,
    used_true_rlm: bool = True,
    tokens_processed: int = 1500,
    sub_calls_made: int = 2,
    time_seconds: float = 1.7,
) -> MagicMock:
    """Build a mock RLMResult."""
    result = MagicMock()
    result.answer = answer
    result.trajectory_log_path = trajectory_log_path
    result.rlm_iterations = rlm_iterations
    result.code_blocks_executed = code_blocks_executed
    result.confidence = confidence
    result.used_true_rlm = used_true_rlm
    result.tokens_processed = tokens_processed
    result.sub_calls_made = sub_calls_made
    result.time_seconds = time_seconds
    return result


# =========================================================================
# 1. PipelineRLMContext dataclass
# =========================================================================


class TestPipelineRLMContext:
    """Tests for PipelineRLMContext defaults and fields."""

    def test_default_values(self) -> None:
        ctx = PipelineRLMContext()
        assert ctx.memory_helpers == {}
        assert ctx.knowledge_helpers == {}
        assert ctx.debate_summary == ""
        assert ctx.enriched is False

    def test_enriched_flag_initially_false(self) -> None:
        ctx = PipelineRLMContext()
        assert ctx.enriched is False

    def test_custom_values(self) -> None:
        ctx = PipelineRLMContext(
            memory_helpers={"recall": lambda: None},
            knowledge_helpers={"search": lambda: None},
            debate_summary="Summary here",
            enriched=True,
        )
        assert "recall" in ctx.memory_helpers
        assert "search" in ctx.knowledge_helpers
        assert ctx.debate_summary == "Summary here"
        assert ctx.enriched is True


# =========================================================================
# 2. enrich_plan_context
# =========================================================================


class TestEnrichPlanContext:
    """Tests for enrich_plan_context function."""

    def test_memory_helper_injection(self) -> None:
        rlm = MagicMock()
        rlm.inject_memory_helpers.return_value = {
            "helpers": {"recall": "fn_recall"}
        }
        continuum = MagicMock()

        ctx = enrich_plan_context(rlm, continuum=continuum)

        rlm.inject_memory_helpers.assert_called_once_with(continuum)
        assert ctx.memory_helpers == {"recall": "fn_recall"}
        assert ctx.enriched is True

    def test_knowledge_helper_injection(self) -> None:
        rlm = MagicMock()
        rlm.inject_knowledge_helpers.return_value = {
            "helpers": {"search": "fn_search"}
        }
        mound = MagicMock()

        ctx = enrich_plan_context(
            rlm, mound=mound, workspace_id="ws-001"
        )

        rlm.inject_knowledge_helpers.assert_called_once_with(mound, "ws-001")
        assert ctx.knowledge_helpers == {"search": "fn_search"}
        assert ctx.enriched is True

    def test_knowledge_requires_workspace_id(self) -> None:
        """Knowledge helpers are skipped when workspace_id is empty."""
        rlm = MagicMock()
        mound = MagicMock()

        ctx = enrich_plan_context(rlm, mound=mound, workspace_id="")

        rlm.inject_knowledge_helpers.assert_not_called()
        assert ctx.knowledge_helpers == {}

    def test_debate_summary_generation(self) -> None:
        rlm = MagicMock(spec=[])  # No inject_* methods
        debate = _make_debate_result(
            messages=[
                {"agent": "claude", "content": "Use token bucket", "round": 1},
                {"agent": "gpt4", "content": "Sliding window better", "round": 1},
            ],
            task="Design a rate limiter",
            consensus_reached=True,
            final_answer="Token bucket with sliding window",
        )

        ctx = enrich_plan_context(rlm, debate_result=debate)

        assert "Design a rate limiter" in ctx.debate_summary
        assert "claude" in ctx.debate_summary
        assert "gpt4" in ctx.debate_summary
        assert "Consensus" in ctx.debate_summary
        assert ctx.enriched is True

    def test_debate_summary_no_consensus(self) -> None:
        rlm = MagicMock(spec=[])
        debate = _make_debate_result(
            messages=[
                {"agent": "claude", "content": "Idea A", "round": 1},
            ],
            consensus_reached=False,
        )

        ctx = enrich_plan_context(rlm, debate_result=debate)

        assert "Debate:" in ctx.debate_summary
        # No "Consensus" line when consensus not reached
        assert "Consensus" not in ctx.debate_summary

    def test_none_optional_parameters(self) -> None:
        rlm = MagicMock(spec=[])
        ctx = enrich_plan_context(rlm)
        assert ctx.memory_helpers == {}
        assert ctx.knowledge_helpers == {}
        assert ctx.debate_summary == ""
        assert ctx.enriched is False

    def test_enriched_false_when_nothing_injected(self) -> None:
        rlm = MagicMock(spec=[])
        ctx = enrich_plan_context(rlm)
        assert ctx.enriched is False

    def test_memory_injection_exception_handled(self) -> None:
        rlm = MagicMock()
        rlm.inject_memory_helpers.side_effect = RuntimeError("Memory failure")
        continuum = MagicMock()

        ctx = enrich_plan_context(rlm, continuum=continuum)

        assert ctx.memory_helpers == {}
        assert ctx.enriched is False

    def test_knowledge_injection_exception_handled(self) -> None:
        rlm = MagicMock()
        rlm.inject_knowledge_helpers.side_effect = RuntimeError("Knowledge failure")
        mound = MagicMock()

        ctx = enrich_plan_context(rlm, mound=mound, workspace_id="ws-1")

        assert ctx.knowledge_helpers == {}

    def test_debate_summary_exception_handled(self) -> None:
        rlm = MagicMock(spec=[])
        debate = MagicMock()
        # Force load_debate_context to fail
        debate.messages = None

        with patch(
            "aragora.rlm.debate_helpers.load_debate_context",
            side_effect=Exception("load failed"),
        ):
            ctx = enrich_plan_context(rlm, debate_result=debate)

        assert ctx.debate_summary == ""

    def test_all_enrichments_combined(self) -> None:
        rlm = MagicMock()
        rlm.inject_memory_helpers.return_value = {"helpers": {"mem": True}}
        rlm.inject_knowledge_helpers.return_value = {"helpers": {"know": True}}

        debate = _make_debate_result(
            messages=[{"agent": "a1", "content": "hi", "round": 1}],
            consensus_reached=True,
            final_answer="yes",
        )

        ctx = enrich_plan_context(
            rlm,
            continuum=MagicMock(),
            mound=MagicMock(),
            workspace_id="ws-1",
            debate_result=debate,
        )

        assert ctx.memory_helpers == {"mem": True}
        assert ctx.knowledge_helpers == {"know": True}
        assert ctx.debate_summary != ""
        assert ctx.enriched is True


# =========================================================================
# 3. GauntletRLMFinding dataclass
# =========================================================================


class TestGauntletRLMFinding:
    """Tests for GauntletRLMFinding fields."""

    def test_required_fields(self) -> None:
        finding = GauntletRLMFinding(
            category="logical_fallacy",
            severity="high",
            description="Ad hominem detected",
        )
        assert finding.category == "logical_fallacy"
        assert finding.severity == "high"
        assert finding.description == "Ad hominem detected"

    def test_optional_fields_default(self) -> None:
        finding = GauntletRLMFinding(
            category="weak_argument",
            severity="low",
            description="Assertion without evidence",
        )
        assert finding.source_round is None
        assert finding.source_agent is None
        assert finding.evidence == ""

    def test_all_fields_set(self) -> None:
        finding = GauntletRLMFinding(
            category="missing_evidence",
            severity="medium",
            description="Claim without citation",
            source_round=2,
            source_agent="gpt4",
            evidence="studies show that...",
        )
        assert finding.source_round == 2
        assert finding.source_agent == "gpt4"
        assert finding.evidence == "studies show that..."


# =========================================================================
# 4. GauntletRLMAnalysis dataclass
# =========================================================================


class TestGauntletRLMAnalysis:
    """Tests for GauntletRLMAnalysis fields."""

    def test_basic_construction(self) -> None:
        analysis = GauntletRLMAnalysis(
            findings=[],
            total_findings=0,
        )
        assert analysis.findings == []
        assert analysis.total_findings == 0
        assert analysis.severity_counts == {}
        assert analysis.summary == ""
        assert analysis.raw_rlm_answer == ""

    def test_with_findings(self) -> None:
        f1 = GauntletRLMFinding("cat1", "high", "desc1")
        f2 = GauntletRLMFinding("cat2", "low", "desc2")
        analysis = GauntletRLMAnalysis(
            findings=[f1, f2],
            total_findings=2,
            severity_counts={"high": 1, "low": 1},
            summary="Found 2 issues",
        )
        assert len(analysis.findings) == 2
        assert analysis.severity_counts["high"] == 1


# =========================================================================
# 5. analyze_debate_for_gauntlet
# =========================================================================


class TestAnalyzeDebateForGauntlet:
    """Tests for analyze_debate_for_gauntlet heuristic analysis."""

    def test_detects_ad_hominem(self) -> None:
        debate = _make_debate_result(
            messages=[
                {"agent": "badagent", "content": "You always get this wrong", "round": 1},
            ]
        )
        analysis = analyze_debate_for_gauntlet(MagicMock(spec=[]), debate)

        assert analysis.total_findings >= 1
        categories = [f.category for f in analysis.findings]
        assert "logical_fallacy" in categories

    def test_detects_you_never_pattern(self) -> None:
        debate = _make_debate_result(
            messages=[
                {"agent": "agent1", "content": "You never consider the edge cases", "round": 1},
            ]
        )
        analysis = analyze_debate_for_gauntlet(MagicMock(spec=[]), debate)
        categories = [f.category for f in analysis.findings]
        assert "logical_fallacy" in categories

    def test_detects_unsupported_assertions(self) -> None:
        debate = _make_debate_result(
            messages=[
                {"agent": "lazy", "content": "Obviously this is the right approach", "round": 2},
            ]
        )
        analysis = analyze_debate_for_gauntlet(MagicMock(spec=[]), debate)

        categories = [f.category for f in analysis.findings]
        assert "weak_argument" in categories

    def test_detects_everyone_knows(self) -> None:
        debate = _make_debate_result(
            messages=[
                {"agent": "agent2", "content": "Everyone knows this is true", "round": 1},
            ]
        )
        analysis = analyze_debate_for_gauntlet(MagicMock(spec=[]), debate)
        categories = [f.category for f in analysis.findings]
        assert "weak_argument" in categories

    def test_detects_missing_evidence(self) -> None:
        debate = _make_debate_result(
            messages=[
                {"agent": "citer", "content": "Studies show this works perfectly", "round": 3},
            ]
        )
        analysis = analyze_debate_for_gauntlet(MagicMock(spec=[]), debate)

        categories = [f.category for f in analysis.findings]
        assert "missing_evidence" in categories

    def test_no_missing_evidence_with_citation(self) -> None:
        """When a year/URL is present, 'studies show' should not flag."""
        debate = _make_debate_result(
            messages=[
                {
                    "agent": "citer",
                    "content": "Studies show this works (Smith 2023)",
                    "round": 1,
                },
            ]
        )
        analysis = analyze_debate_for_gauntlet(MagicMock(spec=[]), debate)
        categories = [f.category for f in analysis.findings]
        assert "missing_evidence" not in categories

    def test_severity_counts_dict(self) -> None:
        debate = _make_debate_result(
            messages=[
                {"agent": "a", "content": "You always do this", "round": 1},
                {"agent": "b", "content": "Obviously correct", "round": 1},
            ]
        )
        analysis = analyze_debate_for_gauntlet(MagicMock(spec=[]), debate)

        assert isinstance(analysis.severity_counts, dict)
        total = sum(analysis.severity_counts.values())
        assert total == analysis.total_findings

    def test_focus_areas_filter(self) -> None:
        """Only analyze for specified focus areas."""
        debate = _make_debate_result(
            messages=[
                {"agent": "a", "content": "You always fail", "round": 1},
                {"agent": "b", "content": "Obviously true", "round": 1},
            ]
        )
        # Only look for logical_fallacy, not weak_argument
        analysis = analyze_debate_for_gauntlet(
            MagicMock(spec=[]), debate, focus_areas=["logical_fallacy"]
        )

        categories = [f.category for f in analysis.findings]
        assert "logical_fallacy" in categories
        assert "weak_argument" not in categories

    def test_empty_debate(self) -> None:
        debate = _make_debate_result(messages=[])
        analysis = analyze_debate_for_gauntlet(MagicMock(spec=[]), debate)

        assert analysis.total_findings == 0
        assert analysis.findings == []
        assert "No issues found" in analysis.summary

    def test_failed_debate_context_loading(self) -> None:
        """When load_debate_context raises, return empty analysis."""
        debate = MagicMock()

        with patch(
            "aragora.rlm.debate_helpers.load_debate_context",
            side_effect=Exception("load failed"),
        ):
            analysis = analyze_debate_for_gauntlet(MagicMock(spec=[]), debate)

        assert analysis.total_findings == 0
        assert "failed" in analysis.summary.lower()

    def test_summary_includes_round_and_message_counts(self) -> None:
        debate = _make_debate_result(
            messages=[
                {"agent": "a", "content": "Clean message", "round": 1},
                {"agent": "b", "content": "Another clean one", "round": 2},
            ]
        )
        analysis = analyze_debate_for_gauntlet(MagicMock(spec=[]), debate)
        assert "2 rounds" in analysis.summary or "Analyzed 2 rounds" in analysis.summary

    def test_finding_stores_source_round_and_agent(self) -> None:
        debate = _make_debate_result(
            messages=[
                {"agent": "offender", "content": "You always mess up", "round": 5},
            ]
        )
        analysis = analyze_debate_for_gauntlet(MagicMock(spec=[]), debate)

        assert len(analysis.findings) >= 1
        finding = analysis.findings[0]
        assert finding.source_round == 5
        assert finding.source_agent == "offender"

    def test_evidence_field_populated(self) -> None:
        long_msg = "You always " + "x" * 300
        debate = _make_debate_result(
            messages=[{"agent": "a", "content": long_msg, "round": 1}]
        )
        analysis = analyze_debate_for_gauntlet(MagicMock(spec=[]), debate)

        assert len(analysis.findings) >= 1
        # Evidence should be truncated to first 200 chars
        assert len(analysis.findings[0].evidence) <= 200


# =========================================================================
# 6. _parse_gauntlet_findings (internal helper)
# =========================================================================


class TestParseGauntletFindings:
    """Tests for _parse_gauntlet_findings."""

    def test_parse_json_array(self) -> None:
        raw = json.dumps([
            {
                "category": "logical_fallacy",
                "severity": "high",
                "description": "Ad hominem attack",
                "source_round": 1,
                "source_agent": "agent-x",
                "evidence": "some text",
            },
            {
                "category": "weak_argument",
                "severity": "low",
                "description": "Unsupported claim",
            },
        ])
        findings = _parse_gauntlet_findings(raw)
        assert len(findings) == 2
        assert findings[0].category == "logical_fallacy"
        assert findings[0].severity == "high"
        assert findings[0].source_round == 1
        assert findings[1].category == "weak_argument"
        assert findings[1].evidence == ""

    def test_parse_line_by_line_format(self) -> None:
        raw = (
            "[logical_fallacy] (high) Ad hominem detected\n"
            "[weak_argument] (low) Assertion without evidence\n"
        )
        findings = _parse_gauntlet_findings(raw)
        assert len(findings) == 2
        assert findings[0].category == "logical_fallacy"
        assert findings[0].severity == "high"
        assert "Ad hominem" in findings[0].description
        assert findings[1].severity == "low"

    def test_empty_input(self) -> None:
        assert _parse_gauntlet_findings("") == []

    def test_invalid_json_and_no_line_matches(self) -> None:
        raw = "just some random text\nwith no structure"
        findings = _parse_gauntlet_findings(raw)
        assert findings == []

    def test_comment_lines_skipped(self) -> None:
        raw = (
            "# This is a comment\n"
            "[cat] (sev) Description\n"
            "# Another comment\n"
        )
        findings = _parse_gauntlet_findings(raw)
        assert len(findings) == 1

    def test_json_with_missing_fields(self) -> None:
        raw = json.dumps([{"description": "Incomplete record"}])
        findings = _parse_gauntlet_findings(raw)
        assert len(findings) == 1
        assert findings[0].category == "unknown"
        assert findings[0].severity == "low"


# =========================================================================
# 7. TrajectoryLearningData dataclass
# =========================================================================


class TestTrajectoryLearningData:
    """Tests for TrajectoryLearningData defaults and fields."""

    def test_required_and_default_fields(self) -> None:
        data = TrajectoryLearningData(query="test query", answer="test answer")
        assert data.query == "test query"
        assert data.answer == "test answer"
        assert data.trajectory_log_path is None
        assert data.rlm_iterations == 0
        assert data.code_blocks_executed == 0
        assert data.confidence == 0.0
        assert data.used_true_rlm is False
        assert data.debate_id is None
        assert data.timestamp == ""
        assert data.metadata == {}

    def test_all_fields_populated(self) -> None:
        data = TrajectoryLearningData(
            query="q",
            answer="a",
            trajectory_log_path="/tmp/log.jsonl",
            rlm_iterations=5,
            code_blocks_executed=10,
            confidence=0.95,
            used_true_rlm=True,
            debate_id="d-123",
            timestamp="2026-02-14T00:00:00Z",
            metadata={"key": "value"},
        )
        assert data.trajectory_log_path == "/tmp/log.jsonl"
        assert data.rlm_iterations == 5
        assert data.code_blocks_executed == 10
        assert data.confidence == 0.95
        assert data.used_true_rlm is True
        assert data.debate_id == "d-123"
        assert data.metadata == {"key": "value"}


# =========================================================================
# 8. collect_trajectory_for_learning
# =========================================================================


class TestCollectTrajectoryForLearning:
    """Tests for collect_trajectory_for_learning from RLMResult."""

    def test_extracts_all_fields(self) -> None:
        result = _make_rlm_result()
        data = collect_trajectory_for_learning(
            result, query="What is X?", debate_id="d-42"
        )

        assert data.query == "What is X?"
        assert data.answer == "The answer"
        assert data.trajectory_log_path == "/tmp/log.jsonl"
        assert data.rlm_iterations == 3
        assert data.code_blocks_executed == 5
        assert data.confidence == 0.9
        assert data.used_true_rlm is True
        assert data.debate_id == "d-42"

    def test_timestamp_is_set(self) -> None:
        result = _make_rlm_result()
        data = collect_trajectory_for_learning(result)
        assert data.timestamp != ""
        # Should be ISO format
        assert "T" in data.timestamp

    def test_metadata_includes_tokens_and_timing(self) -> None:
        result = _make_rlm_result(
            tokens_processed=2000,
            sub_calls_made=4,
            time_seconds=3.5,
        )
        data = collect_trajectory_for_learning(result)
        assert data.metadata["tokens_processed"] == 2000
        assert data.metadata["sub_calls_made"] == 4
        assert data.metadata["time_seconds"] == 3.5

    def test_missing_attributes_use_defaults(self) -> None:
        """When RLMResult lacks attributes, getattr defaults kick in."""
        result = MagicMock(spec=[])  # No attributes at all
        data = collect_trajectory_for_learning(result)

        assert data.answer == ""
        assert data.trajectory_log_path is None
        assert data.rlm_iterations == 0
        assert data.confidence == 0.0
        assert data.used_true_rlm is False

    def test_default_query_and_debate_id(self) -> None:
        result = _make_rlm_result()
        data = collect_trajectory_for_learning(result)
        assert data.query == ""
        assert data.debate_id is None


# =========================================================================
# 9. store_trajectory_learning
# =========================================================================


class TestStoreTrajectoryLearning:
    """Tests for store_trajectory_learning JSONL output."""

    def test_writes_jsonl_to_tmp_directory(self, tmp_path: Any) -> None:
        data = TrajectoryLearningData(
            query="test", answer="answer", timestamp="2026-01-01T00:00:00Z"
        )
        output_dir = str(tmp_path / "trajectories")
        path = store_trajectory_learning(data, output_dir=output_dir)

        assert os.path.exists(path)
        assert path.endswith("rlm_trajectories.jsonl")

        with open(path) as f:
            record = json.loads(f.readline())
        assert record["query"] == "test"
        assert record["answer"] == "answer"
        assert record["timestamp"] == "2026-01-01T00:00:00Z"

    def test_appends_to_existing_file(self, tmp_path: Any) -> None:
        output_dir = str(tmp_path / "traj")
        data1 = TrajectoryLearningData(query="q1", answer="a1")
        data2 = TrajectoryLearningData(query="q2", answer="a2")

        store_trajectory_learning(data1, output_dir=output_dir)
        store_trajectory_learning(data2, output_dir=output_dir)

        path = os.path.join(output_dir, "rlm_trajectories.jsonl")
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["query"] == "q1"
        assert json.loads(lines[1])["query"] == "q2"

    def test_creates_directory_if_needed(self, tmp_path: Any) -> None:
        deep_dir = str(tmp_path / "a" / "b" / "c")
        data = TrajectoryLearningData(query="q", answer="a")
        path = store_trajectory_learning(data, output_dir=deep_dir)
        assert os.path.exists(path)

    def test_answer_truncated_to_1000_chars(self, tmp_path: Any) -> None:
        long_answer = "x" * 2000
        data = TrajectoryLearningData(query="q", answer=long_answer)
        output_dir = str(tmp_path / "traj")
        path = store_trajectory_learning(data, output_dir=output_dir)

        with open(path) as f:
            record = json.loads(f.readline())
        assert len(record["answer"]) == 1000

    def test_stores_all_fields(self, tmp_path: Any) -> None:
        data = TrajectoryLearningData(
            query="q",
            answer="a",
            trajectory_log_path="/tmp/log",
            rlm_iterations=7,
            code_blocks_executed=12,
            confidence=0.85,
            used_true_rlm=True,
            debate_id="d-99",
            timestamp="ts",
            metadata={"extra": "data"},
        )
        output_dir = str(tmp_path / "traj")
        path = store_trajectory_learning(data, output_dir=output_dir)

        with open(path) as f:
            record = json.loads(f.readline())
        assert record["rlm_iterations"] == 7
        assert record["code_blocks_executed"] == 12
        assert record["confidence"] == 0.85
        assert record["used_true_rlm"] is True
        assert record["debate_id"] == "d-99"
        assert record["metadata"] == {"extra": "data"}


# =========================================================================
# 10. load_trajectory_insights
# =========================================================================


class TestLoadTrajectoryInsights:
    """Tests for load_trajectory_insights round-trip and edge cases."""

    def test_reads_jsonl_back(self, tmp_path: Any) -> None:
        output_dir = str(tmp_path / "traj")
        for i in range(3):
            data = TrajectoryLearningData(
                query=f"q{i}", answer=f"a{i}", rlm_iterations=i
            )
            store_trajectory_learning(data, output_dir=output_dir)

        entries = load_trajectory_insights(trajectory_dir=output_dir)
        assert len(entries) == 3
        assert entries[0].query == "q0"
        assert entries[2].rlm_iterations == 2

    def test_handles_missing_file(self, tmp_path: Any) -> None:
        entries = load_trajectory_insights(
            trajectory_dir=str(tmp_path / "nonexistent")
        )
        assert entries == []

    def test_respects_limit_parameter(self, tmp_path: Any) -> None:
        output_dir = str(tmp_path / "traj")
        for i in range(10):
            data = TrajectoryLearningData(query=f"q{i}", answer=f"a{i}")
            store_trajectory_learning(data, output_dir=output_dir)

        entries = load_trajectory_insights(trajectory_dir=output_dir, limit=3)
        assert len(entries) == 3
        assert entries[0].query == "q0"
        assert entries[2].query == "q2"

    def test_handles_malformed_lines(self, tmp_path: Any) -> None:
        output_dir = str(tmp_path / "traj")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "rlm_trajectories.jsonl")

        with open(path, "w") as f:
            f.write('{"query": "good", "answer": "a"}\n')
            f.write("not valid json\n")
            f.write('{"query": "also_good", "answer": "b"}\n')

        entries = load_trajectory_insights(trajectory_dir=output_dir)
        assert len(entries) == 2
        assert entries[0].query == "good"
        assert entries[1].query == "also_good"

    def test_skips_empty_lines(self, tmp_path: Any) -> None:
        output_dir = str(tmp_path / "traj")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "rlm_trajectories.jsonl")

        with open(path, "w") as f:
            f.write('{"query": "q1", "answer": "a1"}\n')
            f.write("\n")
            f.write("   \n")
            f.write('{"query": "q2", "answer": "a2"}\n')

        entries = load_trajectory_insights(trajectory_dir=output_dir)
        assert len(entries) == 2

    def test_round_trip_preserves_data(self, tmp_path: Any) -> None:
        output_dir = str(tmp_path / "traj")
        original = TrajectoryLearningData(
            query="complex query",
            answer="detailed answer",
            trajectory_log_path="/logs/t1.jsonl",
            rlm_iterations=5,
            code_blocks_executed=8,
            confidence=0.92,
            used_true_rlm=True,
            debate_id="d-round-trip",
            timestamp="2026-02-14T12:00:00Z",
            metadata={"tokens_processed": 1500},
        )
        store_trajectory_learning(original, output_dir=output_dir)
        loaded = load_trajectory_insights(trajectory_dir=output_dir)

        assert len(loaded) == 1
        entry = loaded[0]
        assert entry.query == original.query
        assert entry.answer == original.answer
        assert entry.trajectory_log_path == original.trajectory_log_path
        assert entry.rlm_iterations == original.rlm_iterations
        assert entry.code_blocks_executed == original.code_blocks_executed
        assert entry.confidence == original.confidence
        assert entry.used_true_rlm == original.used_true_rlm
        assert entry.debate_id == original.debate_id
        assert entry.timestamp == original.timestamp
        assert entry.metadata == original.metadata


# =========================================================================
# 11. Module exports
# =========================================================================


class TestModuleExports:
    """Verify all Tier 3 symbols are accessible from aragora.rlm."""

    def test_pipeline_rlm_context_importable(self) -> None:
        from aragora.rlm import PipelineRLMContext

        assert PipelineRLMContext is not None

    def test_enrich_plan_context_importable(self) -> None:
        from aragora.rlm import enrich_plan_context

        assert callable(enrich_plan_context)

    def test_gauntlet_rlm_finding_importable(self) -> None:
        from aragora.rlm import GauntletRLMFinding

        assert GauntletRLMFinding is not None

    def test_gauntlet_rlm_analysis_importable(self) -> None:
        from aragora.rlm import GauntletRLMAnalysis

        assert GauntletRLMAnalysis is not None

    def test_analyze_debate_for_gauntlet_importable(self) -> None:
        from aragora.rlm import analyze_debate_for_gauntlet

        assert callable(analyze_debate_for_gauntlet)

    def test_trajectory_learning_data_importable(self) -> None:
        from aragora.rlm import TrajectoryLearningData

        assert TrajectoryLearningData is not None

    def test_collect_trajectory_for_learning_importable(self) -> None:
        from aragora.rlm import collect_trajectory_for_learning

        assert callable(collect_trajectory_for_learning)

    def test_store_trajectory_learning_importable(self) -> None:
        from aragora.rlm import store_trajectory_learning

        assert callable(store_trajectory_learning)

    def test_load_trajectory_insights_importable(self) -> None:
        from aragora.rlm import load_trajectory_insights

        assert callable(load_trajectory_insights)
