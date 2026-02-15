"""Tests for ExplainabilityAPI client resource."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.explainability import (
    BatchDebateResult,
    BatchJobStatus,
    Counterfactual,
    DecisionExplanation,
    EvidenceItem,
    ExplainabilityAPI,
    ExplanationFactor,
    VotePivot,
)


@pytest.fixture
def mock_client() -> AragoraClient:
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def api(mock_client: AragoraClient) -> ExplainabilityAPI:
    return ExplainabilityAPI(mock_client)


# ---------------------------------------------------------------------------
# Sample response data
# ---------------------------------------------------------------------------

SAMPLE_FACTOR = {
    "id": "f-1",
    "name": "Cost efficiency",
    "description": "Evaluated cost implications",
    "weight": 0.35,
    "evidence": ["Lower infrastructure costs", "Reduced maintenance"],
    "source_agents": ["agent-a", "agent-b"],
}

SAMPLE_EVIDENCE = {
    "id": "ev-1",
    "content": "Redis handles 100k ops/sec",
    "source": "benchmark_report",
    "confidence": 0.92,
    "round_number": 2,
    "agent_id": "agent-a",
    "supporting_claims": ["claim-1", "claim-2"],
    "contradicting_claims": ["claim-3"],
}

SAMPLE_VOTE_PIVOT = {
    "agent_id": "agent-c",
    "vote_value": "approve",
    "confidence": 0.88,
    "influence_score": 0.72,
    "reasoning": "Strong benchmark evidence",
    "changed_outcome": True,
    "counterfactual_result": "rejected",
}

SAMPLE_COUNTERFACTUAL = {
    "id": "cf-1",
    "scenario": "Without cost data",
    "description": "If cost analysis were unavailable",
    "alternative_outcome": "rejected",
    "probability": 0.65,
    "key_differences": ["No cost comparison", "Relied on opinion only"],
}

SAMPLE_EXPLANATION = {
    "debate_id": "debate-42",
    "decision": "approved",
    "confidence": 0.91,
    "summary": "Approved based on strong performance evidence",
    "factors": [SAMPLE_FACTOR],
    "evidence_chain": [SAMPLE_EVIDENCE],
    "vote_pivots": [SAMPLE_VOTE_PIVOT],
    "counterfactuals": [SAMPLE_COUNTERFACTUAL],
    "generated_at": "2026-02-10T14:30:00Z",
}

SAMPLE_BATCH_STATUS = {
    "batch_id": "batch-99",
    "status": "processing",
    "total_debates": 5,
    "processed_count": 3,
    "success_count": 2,
    "error_count": 1,
    "progress_pct": 60.0,
    "created_at": "2026-02-10T12:00:00Z",
    "completed_at": None,
}

SAMPLE_BATCH_RESULT = {
    "debate_id": "debate-42",
    "status": "success",
    "explanation": SAMPLE_EXPLANATION,
    "error": None,
    "processing_time_ms": 450.5,
}

SAMPLE_BATCH_RESULT_ERROR = {
    "debate_id": "debate-99",
    "status": "error",
    "explanation": None,
    "error": "Debate not found",
    "processing_time_ms": 12.0,
}


# ===========================================================================
# Dataclass construction tests
# ===========================================================================


class TestDataclasses:
    def test_explanation_factor_defaults(self) -> None:
        factor = ExplanationFactor(id="f", name="n", description="d", weight=0.5)
        assert factor.evidence == []
        assert factor.source_agents == []

    def test_explanation_factor_full(self) -> None:
        factor = ExplanationFactor(
            id="f-1",
            name="Cost",
            description="desc",
            weight=0.4,
            evidence=["e1"],
            source_agents=["a1"],
        )
        assert factor.weight == 0.4
        assert factor.evidence == ["e1"]
        assert factor.source_agents == ["a1"]

    def test_evidence_item_defaults(self) -> None:
        ev = EvidenceItem(
            id="e", content="c", source="s", confidence=0.5, round_number=1, agent_id="a"
        )
        assert ev.supporting_claims == []
        assert ev.contradicting_claims == []

    def test_vote_pivot_defaults(self) -> None:
        vp = VotePivot(
            agent_id="a",
            vote_value="approve",
            confidence=0.8,
            influence_score=0.5,
            reasoning="good",
        )
        assert vp.changed_outcome is False
        assert vp.counterfactual_result is None

    def test_counterfactual_defaults(self) -> None:
        cf = Counterfactual(
            id="c", scenario="s", description="d", alternative_outcome="rejected", probability=0.3
        )
        assert cf.key_differences == []

    def test_decision_explanation_defaults(self) -> None:
        exp = DecisionExplanation(debate_id="d", decision="ok", confidence=0.9, summary="sum")
        assert exp.factors == []
        assert exp.evidence_chain == []
        assert exp.vote_pivots == []
        assert exp.counterfactuals == []
        assert exp.generated_at is None

    def test_batch_job_status_defaults(self) -> None:
        bjs = BatchJobStatus(
            batch_id="b",
            status="pending",
            total_debates=0,
            processed_count=0,
            success_count=0,
            error_count=0,
            progress_pct=0.0,
        )
        assert bjs.created_at is None
        assert bjs.completed_at is None

    def test_batch_debate_result_defaults(self) -> None:
        bdr = BatchDebateResult(debate_id="d", status="success")
        assert bdr.explanation is None
        assert bdr.error is None
        assert bdr.processing_time_ms == 0.0


# ===========================================================================
# get_explanation
# ===========================================================================


class TestGetExplanation:
    def test_get_explanation(self, api: ExplainabilityAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_EXPLANATION
        result = api.get_explanation("debate-42")
        assert isinstance(result, DecisionExplanation)
        assert result.debate_id == "debate-42"
        assert result.decision == "approved"
        assert result.confidence == 0.91
        assert result.summary == "Approved based on strong performance evidence"
        assert len(result.factors) == 1
        assert len(result.evidence_chain) == 1
        assert len(result.vote_pivots) == 1
        assert len(result.counterfactuals) == 1
        assert result.generated_at is not None
        assert result.generated_at.year == 2026
        mock_client._get.assert_called_once_with("/api/v1/debates/debate-42/explanation")

    def test_get_explanation_minimal_response(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {}
        result = api.get_explanation("debate-1")
        assert result.debate_id == "debate-1"
        assert result.decision == ""
        assert result.confidence == 0.0
        assert result.summary == ""
        assert result.factors == []
        assert result.evidence_chain == []
        assert result.vote_pivots == []
        assert result.counterfactuals == []
        assert result.generated_at is None

    def test_get_explanation_invalid_datetime(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {**SAMPLE_EXPLANATION, "generated_at": "not-a-date"}
        result = api.get_explanation("debate-42")
        assert result.generated_at is None

    def test_get_explanation_debate_id_fallback(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        """When response has no debate_id, the argument is used."""
        data = {k: v for k, v in SAMPLE_EXPLANATION.items() if k != "debate_id"}
        mock_client._get.return_value = data
        result = api.get_explanation("fallback-id")
        assert result.debate_id == "fallback-id"

    @pytest.mark.asyncio
    async def test_get_explanation_async(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_EXPLANATION)
        result = await api.get_explanation_async("debate-42")
        assert isinstance(result, DecisionExplanation)
        assert result.debate_id == "debate-42"
        assert result.confidence == 0.91
        mock_client._get_async.assert_called_once_with("/api/v1/debates/debate-42/explanation")


# ===========================================================================
# get_evidence
# ===========================================================================


class TestGetEvidence:
    def test_get_evidence(self, api: ExplainabilityAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"evidence": [SAMPLE_EVIDENCE]}
        result = api.get_evidence("debate-42")
        assert len(result) == 1
        ev = result[0]
        assert isinstance(ev, EvidenceItem)
        assert ev.id == "ev-1"
        assert ev.content == "Redis handles 100k ops/sec"
        assert ev.source == "benchmark_report"
        assert ev.confidence == 0.92
        assert ev.round_number == 2
        assert ev.agent_id == "agent-a"
        assert ev.supporting_claims == ["claim-1", "claim-2"]
        assert ev.contradicting_claims == ["claim-3"]
        mock_client._get.assert_called_once_with("/api/v1/debates/debate-42/evidence")

    def test_get_evidence_alternate_key(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        """Accepts 'evidence_chain' key as fallback."""
        mock_client._get.return_value = {"evidence_chain": [SAMPLE_EVIDENCE]}
        result = api.get_evidence("debate-42")
        assert len(result) == 1
        assert result[0].id == "ev-1"

    def test_get_evidence_empty(self, api: ExplainabilityAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        result = api.get_evidence("debate-42")
        assert result == []

    def test_get_evidence_multiple(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        ev2 = {**SAMPLE_EVIDENCE, "id": "ev-2", "round_number": 3}
        mock_client._get.return_value = {"evidence": [SAMPLE_EVIDENCE, ev2]}
        result = api.get_evidence("debate-42")
        assert len(result) == 2
        assert result[1].id == "ev-2"
        assert result[1].round_number == 3

    @pytest.mark.asyncio
    async def test_get_evidence_async(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"evidence": [SAMPLE_EVIDENCE]})
        result = await api.get_evidence_async("debate-42")
        assert len(result) == 1
        assert result[0].id == "ev-1"

    @pytest.mark.asyncio
    async def test_get_evidence_async_alternate_key(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"evidence_chain": [SAMPLE_EVIDENCE]})
        result = await api.get_evidence_async("debate-42")
        assert len(result) == 1


# ===========================================================================
# get_vote_pivots
# ===========================================================================


class TestGetVotePivots:
    def test_get_vote_pivots(self, api: ExplainabilityAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"pivots": [SAMPLE_VOTE_PIVOT]}
        result = api.get_vote_pivots("debate-42")
        assert len(result) == 1
        vp = result[0]
        assert isinstance(vp, VotePivot)
        assert vp.agent_id == "agent-c"
        assert vp.vote_value == "approve"
        assert vp.confidence == 0.88
        assert vp.influence_score == 0.72
        assert vp.reasoning == "Strong benchmark evidence"
        assert vp.changed_outcome is True
        assert vp.counterfactual_result == "rejected"
        mock_client._get.assert_called_once_with("/api/v1/debates/debate-42/votes/pivots")

    def test_get_vote_pivots_alternate_key(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        """Accepts 'vote_pivots' key as fallback."""
        mock_client._get.return_value = {"vote_pivots": [SAMPLE_VOTE_PIVOT]}
        result = api.get_vote_pivots("debate-42")
        assert len(result) == 1
        assert result[0].agent_id == "agent-c"

    def test_get_vote_pivots_empty(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {}
        result = api.get_vote_pivots("debate-42")
        assert result == []

    def test_vote_pivot_no_counterfactual_result(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        pivot_data = {k: v for k, v in SAMPLE_VOTE_PIVOT.items() if k != "counterfactual_result"}
        mock_client._get.return_value = {"pivots": [pivot_data]}
        result = api.get_vote_pivots("debate-42")
        assert result[0].counterfactual_result is None

    @pytest.mark.asyncio
    async def test_get_vote_pivots_async(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"pivots": [SAMPLE_VOTE_PIVOT]})
        result = await api.get_vote_pivots_async("debate-42")
        assert len(result) == 1
        assert result[0].changed_outcome is True

    @pytest.mark.asyncio
    async def test_get_vote_pivots_async_alternate_key(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"vote_pivots": [SAMPLE_VOTE_PIVOT]})
        result = await api.get_vote_pivots_async("debate-42")
        assert len(result) == 1


# ===========================================================================
# get_counterfactuals
# ===========================================================================


class TestGetCounterfactuals:
    def test_get_counterfactuals(self, api: ExplainabilityAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"counterfactuals": [SAMPLE_COUNTERFACTUAL]}
        result = api.get_counterfactuals("debate-42")
        assert len(result) == 1
        cf = result[0]
        assert isinstance(cf, Counterfactual)
        assert cf.id == "cf-1"
        assert cf.scenario == "Without cost data"
        assert cf.description == "If cost analysis were unavailable"
        assert cf.alternative_outcome == "rejected"
        assert cf.probability == 0.65
        assert cf.key_differences == ["No cost comparison", "Relied on opinion only"]
        mock_client._get.assert_called_once_with("/api/v1/debates/debate-42/counterfactuals")

    def test_get_counterfactuals_empty(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {}
        result = api.get_counterfactuals("debate-42")
        assert result == []

    def test_get_counterfactuals_missing_fields(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"counterfactuals": [{}]}
        result = api.get_counterfactuals("debate-42")
        assert len(result) == 1
        cf = result[0]
        assert cf.id == ""
        assert cf.scenario == ""
        assert cf.probability == 0.0
        assert cf.key_differences == []

    @pytest.mark.asyncio
    async def test_get_counterfactuals_async(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"counterfactuals": [SAMPLE_COUNTERFACTUAL]}
        )
        result = await api.get_counterfactuals_async("debate-42")
        assert len(result) == 1
        assert result[0].id == "cf-1"


# ===========================================================================
# get_summary
# ===========================================================================


class TestGetSummary:
    def test_get_summary_default_format(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"summary": "The debate concluded with approval."}
        result = api.get_summary("debate-42")
        assert result == "The debate concluded with approval."
        mock_client._get.assert_called_once_with(
            "/api/v1/debates/debate-42/summary", params={"format": "text"}
        )

    def test_get_summary_markdown_format(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"summary": "## Approved\nStrong evidence."}
        result = api.get_summary("debate-42", format="markdown")
        assert "## Approved" in result
        mock_client._get.assert_called_once_with(
            "/api/v1/debates/debate-42/summary", params={"format": "markdown"}
        )

    def test_get_summary_html_format(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"summary": "<h2>Approved</h2>"}
        result = api.get_summary("debate-42", format="html")
        assert "<h2>" in result

    def test_get_summary_empty(self, api: ExplainabilityAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        result = api.get_summary("debate-42")
        assert result == ""

    @pytest.mark.asyncio
    async def test_get_summary_async(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"summary": "Approved."})
        result = await api.get_summary_async("debate-42")
        assert result == "Approved."
        mock_client._get_async.assert_called_once_with(
            "/api/v1/debates/debate-42/summary", params={"format": "text"}
        )

    @pytest.mark.asyncio
    async def test_get_summary_async_with_format(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"summary": "# Result"})
        result = await api.get_summary_async("debate-42", format="markdown")
        assert result == "# Result"
        mock_client._get_async.assert_called_once_with(
            "/api/v1/debates/debate-42/summary", params={"format": "markdown"}
        )


# ===========================================================================
# create_batch
# ===========================================================================


class TestCreateBatch:
    def test_create_batch_defaults(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = SAMPLE_BATCH_STATUS
        result = api.create_batch(["d1", "d2", "d3"])
        assert isinstance(result, BatchJobStatus)
        assert result.batch_id == "batch-99"
        assert result.status == "processing"
        assert result.total_debates == 5
        assert result.progress_pct == 60.0
        mock_client._post.assert_called_once()
        body = mock_client._post.call_args[0][1]
        assert body["debate_ids"] == ["d1", "d2", "d3"]
        assert body["options"]["include_evidence"] is True
        assert body["options"]["include_counterfactuals"] is False

    def test_create_batch_custom_options(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = SAMPLE_BATCH_STATUS
        api.create_batch(
            ["d1"],
            include_evidence=False,
            include_counterfactuals=True,
        )
        body = mock_client._post.call_args[0][1]
        assert body["options"]["include_evidence"] is False
        assert body["options"]["include_counterfactuals"] is True

    def test_create_batch_url(self, api: ExplainabilityAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_BATCH_STATUS
        api.create_batch(["d1"])
        url = mock_client._post.call_args[0][0]
        assert url == "/api/v1/explainability/batch"

    @pytest.mark.asyncio
    async def test_create_batch_async(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value=SAMPLE_BATCH_STATUS)
        result = await api.create_batch_async(["d1", "d2"])
        assert result.batch_id == "batch-99"
        body = mock_client._post_async.call_args[0][1]
        assert body["debate_ids"] == ["d1", "d2"]

    @pytest.mark.asyncio
    async def test_create_batch_async_custom_options(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value=SAMPLE_BATCH_STATUS)
        await api.create_batch_async(["d1"], include_evidence=False, include_counterfactuals=True)
        body = mock_client._post_async.call_args[0][1]
        assert body["options"]["include_evidence"] is False
        assert body["options"]["include_counterfactuals"] is True


# ===========================================================================
# get_batch_status
# ===========================================================================


class TestGetBatchStatus:
    def test_get_batch_status(self, api: ExplainabilityAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_BATCH_STATUS
        result = api.get_batch_status("batch-99")
        assert isinstance(result, BatchJobStatus)
        assert result.batch_id == "batch-99"
        assert result.status == "processing"
        assert result.processed_count == 3
        assert result.success_count == 2
        assert result.error_count == 1
        assert result.created_at is not None
        assert result.created_at.year == 2026
        assert result.completed_at is None
        mock_client._get.assert_called_once_with("/api/v1/explainability/batch/batch-99/status")

    def test_get_batch_status_completed(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        completed = {
            **SAMPLE_BATCH_STATUS,
            "status": "completed",
            "processed_count": 5,
            "success_count": 5,
            "error_count": 0,
            "progress_pct": 100.0,
            "completed_at": "2026-02-10T12:05:00Z",
        }
        mock_client._get.return_value = completed
        result = api.get_batch_status("batch-99")
        assert result.status == "completed"
        assert result.completed_at is not None
        assert result.completed_at.year == 2026

    def test_get_batch_status_minimal(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {}
        result = api.get_batch_status("batch-99")
        assert result.batch_id == ""
        assert result.status == "pending"
        assert result.total_debates == 0
        assert result.created_at is None

    def test_get_batch_status_invalid_datetime(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {
            **SAMPLE_BATCH_STATUS,
            "created_at": "garbage",
            "completed_at": "also-garbage",
        }
        result = api.get_batch_status("batch-99")
        assert result.created_at is None
        assert result.completed_at is None

    @pytest.mark.asyncio
    async def test_get_batch_status_async(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_BATCH_STATUS)
        result = await api.get_batch_status_async("batch-99")
        assert result.batch_id == "batch-99"
        mock_client._get_async.assert_called_once_with(
            "/api/v1/explainability/batch/batch-99/status"
        )


# ===========================================================================
# get_batch_results
# ===========================================================================


class TestGetBatchResults:
    def test_get_batch_results(self, api: ExplainabilityAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"results": [SAMPLE_BATCH_RESULT]}
        result = api.get_batch_results("batch-99")
        assert len(result) == 1
        br = result[0]
        assert isinstance(br, BatchDebateResult)
        assert br.debate_id == "debate-42"
        assert br.status == "success"
        assert br.explanation is not None
        assert isinstance(br.explanation, DecisionExplanation)
        assert br.explanation.debate_id == "debate-42"
        assert br.error is None
        assert br.processing_time_ms == 450.5
        mock_client._get.assert_called_once_with("/api/v1/explainability/batch/batch-99/results")

    def test_get_batch_results_with_error(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"results": [SAMPLE_BATCH_RESULT_ERROR]}
        result = api.get_batch_results("batch-99")
        assert len(result) == 1
        br = result[0]
        assert br.status == "error"
        assert br.explanation is None
        assert br.error == "Debate not found"

    def test_get_batch_results_mixed(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {
            "results": [SAMPLE_BATCH_RESULT, SAMPLE_BATCH_RESULT_ERROR]
        }
        result = api.get_batch_results("batch-99")
        assert len(result) == 2
        assert result[0].status == "success"
        assert result[0].explanation is not None
        assert result[1].status == "error"
        assert result[1].explanation is None

    def test_get_batch_results_empty(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {}
        result = api.get_batch_results("batch-99")
        assert result == []

    def test_get_batch_results_missing_fields(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"results": [{}]}
        result = api.get_batch_results("batch-99")
        assert len(result) == 1
        br = result[0]
        assert br.debate_id == ""
        assert br.status == "error"
        assert br.explanation is None
        assert br.processing_time_ms == 0.0

    @pytest.mark.asyncio
    async def test_get_batch_results_async(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"results": [SAMPLE_BATCH_RESULT]})
        result = await api.get_batch_results_async("batch-99")
        assert len(result) == 1
        assert result[0].explanation is not None


# ===========================================================================
# Parser internals
# ===========================================================================


class TestParsers:
    def test_parse_factor_missing_fields(self, api: ExplainabilityAPI) -> None:
        factor = api._parse_factor({})
        assert factor.id == ""
        assert factor.name == ""
        assert factor.description == ""
        assert factor.weight == 0.0
        assert factor.evidence == []
        assert factor.source_agents == []

    def test_parse_evidence_missing_fields(self, api: ExplainabilityAPI) -> None:
        ev = api._parse_evidence({})
        assert ev.id == ""
        assert ev.content == ""
        assert ev.source == ""
        assert ev.confidence == 0.0
        assert ev.round_number == 0
        assert ev.agent_id == ""

    def test_parse_vote_pivot_missing_fields(self, api: ExplainabilityAPI) -> None:
        vp = api._parse_vote_pivot({})
        assert vp.agent_id == ""
        assert vp.vote_value == ""
        assert vp.confidence == 0.0
        assert vp.influence_score == 0.0
        assert vp.reasoning == ""
        assert vp.changed_outcome is False
        assert vp.counterfactual_result is None

    def test_parse_counterfactual_missing_fields(self, api: ExplainabilityAPI) -> None:
        cf = api._parse_counterfactual({})
        assert cf.id == ""
        assert cf.scenario == ""
        assert cf.description == ""
        assert cf.alternative_outcome == ""
        assert cf.probability == 0.0
        assert cf.key_differences == []

    def test_parse_batch_status_missing_fields(self, api: ExplainabilityAPI) -> None:
        bs = api._parse_batch_status({})
        assert bs.batch_id == ""
        assert bs.status == "pending"
        assert bs.total_debates == 0
        assert bs.processed_count == 0
        assert bs.success_count == 0
        assert bs.error_count == 0
        assert bs.progress_pct == 0.0

    def test_parse_batch_result_with_explanation(self, api: ExplainabilityAPI) -> None:
        br = api._parse_batch_result(SAMPLE_BATCH_RESULT)
        assert br.explanation is not None
        assert br.explanation.debate_id == "debate-42"
        assert len(br.explanation.factors) == 1

    def test_parse_batch_result_without_explanation(self, api: ExplainabilityAPI) -> None:
        br = api._parse_batch_result(SAMPLE_BATCH_RESULT_ERROR)
        assert br.explanation is None
        assert br.error == "Debate not found"

    def test_parse_explanation_preserves_nested_data(self, api: ExplainabilityAPI) -> None:
        """Verify all nested dataclasses are correctly parsed."""
        exp = api._parse_explanation(SAMPLE_EXPLANATION, "debate-42")

        # Factor
        assert exp.factors[0].id == "f-1"
        assert exp.factors[0].weight == 0.35
        assert exp.factors[0].evidence == ["Lower infrastructure costs", "Reduced maintenance"]
        assert exp.factors[0].source_agents == ["agent-a", "agent-b"]

        # Evidence
        assert exp.evidence_chain[0].id == "ev-1"
        assert exp.evidence_chain[0].confidence == 0.92
        assert exp.evidence_chain[0].supporting_claims == ["claim-1", "claim-2"]
        assert exp.evidence_chain[0].contradicting_claims == ["claim-3"]

        # Vote pivot
        assert exp.vote_pivots[0].agent_id == "agent-c"
        assert exp.vote_pivots[0].changed_outcome is True
        assert exp.vote_pivots[0].counterfactual_result == "rejected"

        # Counterfactual
        assert exp.counterfactuals[0].id == "cf-1"
        assert exp.counterfactuals[0].probability == 0.65
        assert len(exp.counterfactuals[0].key_differences) == 2


# ===========================================================================
# Integration-like workflow tests
# ===========================================================================


class TestWorkflows:
    def test_full_explanation_workflow(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        """Simulate a full explainability workflow: explanation, evidence, pivots, counterfactuals, summary."""
        mock_client._get.side_effect = [
            SAMPLE_EXPLANATION,
            {"evidence": [SAMPLE_EVIDENCE]},
            {"pivots": [SAMPLE_VOTE_PIVOT]},
            {"counterfactuals": [SAMPLE_COUNTERFACTUAL]},
            {"summary": "Decision: approved."},
        ]

        explanation = api.get_explanation("debate-42")
        assert explanation.decision == "approved"

        evidence = api.get_evidence("debate-42")
        assert len(evidence) == 1

        pivots = api.get_vote_pivots("debate-42")
        assert len(pivots) == 1

        counterfactuals = api.get_counterfactuals("debate-42")
        assert len(counterfactuals) == 1

        summary = api.get_summary("debate-42")
        assert summary == "Decision: approved."

        assert mock_client._get.call_count == 5

    def test_batch_workflow(self, api: ExplainabilityAPI, mock_client: AragoraClient) -> None:
        """Simulate batch create -> poll status -> get results."""
        mock_client._post.return_value = {
            "batch_id": "batch-1",
            "status": "pending",
            "total_debates": 3,
            "processed_count": 0,
            "success_count": 0,
            "error_count": 0,
            "progress_pct": 0.0,
        }

        batch = api.create_batch(["d1", "d2", "d3"])
        assert batch.batch_id == "batch-1"
        assert batch.status == "pending"

        mock_client._get.side_effect = [
            {
                "batch_id": "batch-1",
                "status": "processing",
                "total_debates": 3,
                "processed_count": 2,
                "success_count": 2,
                "error_count": 0,
                "progress_pct": 66.7,
            },
            {
                "batch_id": "batch-1",
                "status": "completed",
                "total_debates": 3,
                "processed_count": 3,
                "success_count": 3,
                "error_count": 0,
                "progress_pct": 100.0,
                "completed_at": "2026-02-10T13:00:00Z",
            },
            {
                "results": [
                    {
                        "debate_id": "d1",
                        "status": "success",
                        "explanation": {
                            "debate_id": "d1",
                            "decision": "approved",
                            "confidence": 0.85,
                            "summary": "Approved d1",
                        },
                        "processing_time_ms": 200.0,
                    },
                    {
                        "debate_id": "d2",
                        "status": "success",
                        "explanation": {
                            "debate_id": "d2",
                            "decision": "rejected",
                            "confidence": 0.70,
                            "summary": "Rejected d2",
                        },
                        "processing_time_ms": 180.0,
                    },
                    {
                        "debate_id": "d3",
                        "status": "success",
                        "explanation": {
                            "debate_id": "d3",
                            "decision": "deferred",
                            "confidence": 0.55,
                            "summary": "Deferred d3",
                        },
                        "processing_time_ms": 300.0,
                    },
                ]
            },
        ]

        # Poll #1
        status = api.get_batch_status("batch-1")
        assert status.status == "processing"
        assert status.progress_pct == 66.7

        # Poll #2
        status = api.get_batch_status("batch-1")
        assert status.status == "completed"
        assert status.completed_at is not None

        # Get results
        results = api.get_batch_results("batch-1")
        assert len(results) == 3
        assert results[0].explanation.decision == "approved"
        assert results[1].explanation.decision == "rejected"
        assert results[2].explanation.decision == "deferred"

    @pytest.mark.asyncio
    async def test_full_explanation_workflow_async(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        """Async version of the full explainability workflow."""
        mock_client._get_async = AsyncMock(
            side_effect=[
                SAMPLE_EXPLANATION,
                {"evidence": [SAMPLE_EVIDENCE]},
                {"pivots": [SAMPLE_VOTE_PIVOT]},
                {"counterfactuals": [SAMPLE_COUNTERFACTUAL]},
                {"summary": "Approved."},
            ]
        )

        explanation = await api.get_explanation_async("debate-42")
        assert explanation.decision == "approved"

        evidence = await api.get_evidence_async("debate-42")
        assert len(evidence) == 1

        pivots = await api.get_vote_pivots_async("debate-42")
        assert len(pivots) == 1

        counterfactuals = await api.get_counterfactuals_async("debate-42")
        assert len(counterfactuals) == 1

        summary = await api.get_summary_async("debate-42")
        assert summary == "Approved."

    @pytest.mark.asyncio
    async def test_batch_workflow_async(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        """Async batch workflow: create -> status -> results."""
        mock_client._post_async = AsyncMock(
            return_value={
                "batch_id": "batch-2",
                "status": "pending",
                "total_debates": 2,
                "processed_count": 0,
                "success_count": 0,
                "error_count": 0,
                "progress_pct": 0.0,
            }
        )

        batch = await api.create_batch_async(["d1", "d2"])
        assert batch.batch_id == "batch-2"

        mock_client._get_async = AsyncMock(
            side_effect=[
                {
                    "batch_id": "batch-2",
                    "status": "completed",
                    "total_debates": 2,
                    "processed_count": 2,
                    "success_count": 2,
                    "error_count": 0,
                    "progress_pct": 100.0,
                },
                {
                    "results": [
                        {
                            "debate_id": "d1",
                            "status": "success",
                            "explanation": {
                                "debate_id": "d1",
                                "decision": "ok",
                                "confidence": 0.9,
                                "summary": "OK",
                            },
                        },
                    ]
                },
            ]
        )

        status = await api.get_batch_status_async("batch-2")
        assert status.status == "completed"

        results = await api.get_batch_results_async("batch-2")
        assert len(results) == 1
        assert results[0].explanation.decision == "ok"


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_explanation_with_empty_nested_lists(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {
            "debate_id": "d-empty",
            "decision": "none",
            "confidence": 0.0,
            "summary": "",
            "factors": [],
            "evidence_chain": [],
            "vote_pivots": [],
            "counterfactuals": [],
        }
        result = api.get_explanation("d-empty")
        assert result.factors == []
        assert result.evidence_chain == []
        assert result.vote_pivots == []
        assert result.counterfactuals == []

    def test_evidence_primary_key_takes_precedence(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        """When both 'evidence' and 'evidence_chain' are present, 'evidence' wins."""
        mock_client._get.return_value = {
            "evidence": [SAMPLE_EVIDENCE],
            "evidence_chain": [SAMPLE_EVIDENCE, SAMPLE_EVIDENCE],
        }
        result = api.get_evidence("debate-42")
        assert len(result) == 1

    def test_vote_pivots_primary_key_takes_precedence(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        """When both 'pivots' and 'vote_pivots' are present, 'pivots' wins."""
        mock_client._get.return_value = {
            "pivots": [SAMPLE_VOTE_PIVOT],
            "vote_pivots": [SAMPLE_VOTE_PIVOT, SAMPLE_VOTE_PIVOT],
        }
        result = api.get_vote_pivots("debate-42")
        assert len(result) == 1

    def test_batch_status_with_non_iso_datetime(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        """Non-ISO datetime strings should not crash, just produce None."""
        mock_client._get.return_value = {
            **SAMPLE_BATCH_STATUS,
            "created_at": 12345,
            "completed_at": True,
        }
        result = api.get_batch_status("batch-99")
        # Numeric / bool values get str()-converted, then fail fromisoformat
        assert result.created_at is None
        assert result.completed_at is None

    def test_multiple_factors_in_explanation(
        self, api: ExplainabilityAPI, mock_client: AragoraClient
    ) -> None:
        factor2 = {**SAMPLE_FACTOR, "id": "f-2", "name": "Performance", "weight": 0.65}
        mock_client._get.return_value = {
            **SAMPLE_EXPLANATION,
            "factors": [SAMPLE_FACTOR, factor2],
        }
        result = api.get_explanation("debate-42")
        assert len(result.factors) == 2
        assert result.factors[0].name == "Cost efficiency"
        assert result.factors[1].name == "Performance"
        total_weight = sum(f.weight for f in result.factors)
        assert total_weight == pytest.approx(1.0)

    def test_constructor_stores_client(self, mock_client: AragoraClient) -> None:
        api = ExplainabilityAPI(mock_client)
        assert api._client is mock_client
