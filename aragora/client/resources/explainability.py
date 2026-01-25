"""
Explainability API resource for the Aragora client.

Provides methods for understanding debate decisions:
- Decision explanations
- Evidence chains
- Vote pivot analysis
- Counterfactual analysis
- Human-readable summaries
- Batch processing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..client import AragoraClient

logger = logging.getLogger(__name__)


@dataclass
class ExplanationFactor:
    """A factor contributing to a decision."""

    id: str
    name: str
    description: str
    weight: float
    evidence: List[str] = field(default_factory=list)
    source_agents: List[str] = field(default_factory=list)


@dataclass
class EvidenceItem:
    """A piece of evidence in the decision chain."""

    id: str
    content: str
    source: str
    confidence: float
    round_number: int
    agent_id: str
    supporting_claims: List[str] = field(default_factory=list)
    contradicting_claims: List[str] = field(default_factory=list)


@dataclass
class VotePivot:
    """Analysis of a vote that influenced the decision."""

    agent_id: str
    vote_value: str
    confidence: float
    influence_score: float
    reasoning: str
    changed_outcome: bool = False
    counterfactual_result: Optional[str] = None


@dataclass
class Counterfactual:
    """A counterfactual analysis scenario."""

    id: str
    scenario: str
    description: str
    alternative_outcome: str
    probability: float
    key_differences: List[str] = field(default_factory=list)


@dataclass
class DecisionExplanation:
    """Full explanation of a debate decision."""

    debate_id: str
    decision: str
    confidence: float
    summary: str
    factors: List[ExplanationFactor] = field(default_factory=list)
    evidence_chain: List[EvidenceItem] = field(default_factory=list)
    vote_pivots: List[VotePivot] = field(default_factory=list)
    counterfactuals: List[Counterfactual] = field(default_factory=list)
    generated_at: Optional[datetime] = None


@dataclass
class BatchJobStatus:
    """Status of a batch explainability job."""

    batch_id: str
    status: str  # pending, processing, completed, partial, failed
    total_debates: int
    processed_count: int
    success_count: int
    error_count: int
    progress_pct: float
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class BatchDebateResult:
    """Result for a single debate in a batch."""

    debate_id: str
    status: str  # success, error, not_found
    explanation: Optional[DecisionExplanation] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0


class ExplainabilityAPI:
    """API interface for decision explainability."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    # =========================================================================
    # Single Debate Explanations
    # =========================================================================

    def get_explanation(self, debate_id: str) -> DecisionExplanation:
        """
        Get full decision explanation for a debate.

        Args:
            debate_id: The debate ID.

        Returns:
            DecisionExplanation object.
        """
        response = self._client._get(f"/api/v1/debates/{debate_id}/explanation")
        return self._parse_explanation(response, debate_id)

    async def get_explanation_async(self, debate_id: str) -> DecisionExplanation:
        """Async version of get_explanation()."""
        response = await self._client._get_async(f"/api/v1/debates/{debate_id}/explanation")
        return self._parse_explanation(response, debate_id)

    def get_evidence(self, debate_id: str) -> List[EvidenceItem]:
        """
        Get evidence chain for a debate decision.

        Args:
            debate_id: The debate ID.

        Returns:
            List of EvidenceItem objects.
        """
        response = self._client._get(f"/api/v1/debates/{debate_id}/evidence")
        evidence = response.get("evidence", response.get("evidence_chain", []))
        return [self._parse_evidence(e) for e in evidence]

    async def get_evidence_async(self, debate_id: str) -> List[EvidenceItem]:
        """Async version of get_evidence()."""
        response = await self._client._get_async(f"/api/v1/debates/{debate_id}/evidence")
        evidence = response.get("evidence", response.get("evidence_chain", []))
        return [self._parse_evidence(e) for e in evidence]

    def get_vote_pivots(self, debate_id: str) -> List[VotePivot]:
        """
        Get vote influence analysis for a debate.

        Args:
            debate_id: The debate ID.

        Returns:
            List of VotePivot objects.
        """
        response = self._client._get(f"/api/v1/debates/{debate_id}/votes/pivots")
        pivots = response.get("pivots", response.get("vote_pivots", []))
        return [self._parse_vote_pivot(p) for p in pivots]

    async def get_vote_pivots_async(self, debate_id: str) -> List[VotePivot]:
        """Async version of get_vote_pivots()."""
        response = await self._client._get_async(f"/api/v1/debates/{debate_id}/votes/pivots")
        pivots = response.get("pivots", response.get("vote_pivots", []))
        return [self._parse_vote_pivot(p) for p in pivots]

    def get_counterfactuals(self, debate_id: str) -> List[Counterfactual]:
        """
        Get counterfactual analysis for a debate.

        Args:
            debate_id: The debate ID.

        Returns:
            List of Counterfactual objects.
        """
        response = self._client._get(f"/api/v1/debates/{debate_id}/counterfactuals")
        counterfactuals = response.get("counterfactuals", [])
        return [self._parse_counterfactual(c) for c in counterfactuals]

    async def get_counterfactuals_async(self, debate_id: str) -> List[Counterfactual]:
        """Async version of get_counterfactuals()."""
        response = await self._client._get_async(f"/api/v1/debates/{debate_id}/counterfactuals")
        counterfactuals = response.get("counterfactuals", [])
        return [self._parse_counterfactual(c) for c in counterfactuals]

    def get_summary(self, debate_id: str, format: str = "text") -> str:
        """
        Get human-readable summary of a debate decision.

        Args:
            debate_id: The debate ID.
            format: Output format (text, markdown, html).

        Returns:
            Summary string.
        """
        response = self._client._get(
            f"/api/v1/debates/{debate_id}/summary", params={"format": format}
        )
        return response.get("summary", "")

    async def get_summary_async(self, debate_id: str, format: str = "text") -> str:
        """Async version of get_summary()."""
        response = await self._client._get_async(
            f"/api/v1/debates/{debate_id}/summary", params={"format": format}
        )
        return response.get("summary", "")

    # =========================================================================
    # Batch Processing
    # =========================================================================

    def create_batch(
        self,
        debate_ids: List[str],
        include_evidence: bool = True,
        include_counterfactuals: bool = False,
    ) -> BatchJobStatus:
        """
        Create a batch explainability job.

        Args:
            debate_ids: List of debate IDs to process.
            include_evidence: Include evidence chains.
            include_counterfactuals: Include counterfactual analysis.

        Returns:
            BatchJobStatus object.
        """
        body = {
            "debate_ids": debate_ids,
            "options": {
                "include_evidence": include_evidence,
                "include_counterfactuals": include_counterfactuals,
            },
        }
        response = self._client._post("/api/v1/explainability/batch", body)
        return self._parse_batch_status(response)

    async def create_batch_async(
        self,
        debate_ids: List[str],
        include_evidence: bool = True,
        include_counterfactuals: bool = False,
    ) -> BatchJobStatus:
        """Async version of create_batch()."""
        body = {
            "debate_ids": debate_ids,
            "options": {
                "include_evidence": include_evidence,
                "include_counterfactuals": include_counterfactuals,
            },
        }
        response = await self._client._post_async("/api/v1/explainability/batch", body)
        return self._parse_batch_status(response)

    def get_batch_status(self, batch_id: str) -> BatchJobStatus:
        """
        Get status of a batch explainability job.

        Args:
            batch_id: The batch job ID.

        Returns:
            BatchJobStatus object.
        """
        response = self._client._get(f"/api/v1/explainability/batch/{batch_id}/status")
        return self._parse_batch_status(response)

    async def get_batch_status_async(self, batch_id: str) -> BatchJobStatus:
        """Async version of get_batch_status()."""
        response = await self._client._get_async(f"/api/v1/explainability/batch/{batch_id}/status")
        return self._parse_batch_status(response)

    def get_batch_results(self, batch_id: str) -> List[BatchDebateResult]:
        """
        Get results of a completed batch job.

        Args:
            batch_id: The batch job ID.

        Returns:
            List of BatchDebateResult objects.
        """
        response = self._client._get(f"/api/v1/explainability/batch/{batch_id}/results")
        results = response.get("results", [])
        return [self._parse_batch_result(r) for r in results]

    async def get_batch_results_async(self, batch_id: str) -> List[BatchDebateResult]:
        """Async version of get_batch_results()."""
        response = await self._client._get_async(f"/api/v1/explainability/batch/{batch_id}/results")
        results = response.get("results", [])
        return [self._parse_batch_result(r) for r in results]

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_explanation(self, data: Dict[str, Any], debate_id: str) -> DecisionExplanation:
        """Parse explanation data into DecisionExplanation object."""
        generated_at = None
        if data.get("generated_at"):
            try:
                generated_at = datetime.fromisoformat(data["generated_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        factors = [self._parse_factor(f) for f in data.get("factors", [])]
        evidence = [self._parse_evidence(e) for e in data.get("evidence_chain", [])]
        pivots = [self._parse_vote_pivot(p) for p in data.get("vote_pivots", [])]
        counterfactuals = [self._parse_counterfactual(c) for c in data.get("counterfactuals", [])]

        return DecisionExplanation(
            debate_id=data.get("debate_id", debate_id),
            decision=data.get("decision", ""),
            confidence=data.get("confidence", 0.0),
            summary=data.get("summary", ""),
            factors=factors,
            evidence_chain=evidence,
            vote_pivots=pivots,
            counterfactuals=counterfactuals,
            generated_at=generated_at,
        )

    def _parse_factor(self, data: Dict[str, Any]) -> ExplanationFactor:
        """Parse factor data into ExplanationFactor object."""
        return ExplanationFactor(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            weight=data.get("weight", 0.0),
            evidence=data.get("evidence", []),
            source_agents=data.get("source_agents", []),
        )

    def _parse_evidence(self, data: Dict[str, Any]) -> EvidenceItem:
        """Parse evidence data into EvidenceItem object."""
        return EvidenceItem(
            id=data.get("id", ""),
            content=data.get("content", ""),
            source=data.get("source", ""),
            confidence=data.get("confidence", 0.0),
            round_number=data.get("round_number", 0),
            agent_id=data.get("agent_id", ""),
            supporting_claims=data.get("supporting_claims", []),
            contradicting_claims=data.get("contradicting_claims", []),
        )

    def _parse_vote_pivot(self, data: Dict[str, Any]) -> VotePivot:
        """Parse vote pivot data into VotePivot object."""
        return VotePivot(
            agent_id=data.get("agent_id", ""),
            vote_value=data.get("vote_value", ""),
            confidence=data.get("confidence", 0.0),
            influence_score=data.get("influence_score", 0.0),
            reasoning=data.get("reasoning", ""),
            changed_outcome=data.get("changed_outcome", False),
            counterfactual_result=data.get("counterfactual_result"),
        )

    def _parse_counterfactual(self, data: Dict[str, Any]) -> Counterfactual:
        """Parse counterfactual data into Counterfactual object."""
        return Counterfactual(
            id=data.get("id", ""),
            scenario=data.get("scenario", ""),
            description=data.get("description", ""),
            alternative_outcome=data.get("alternative_outcome", ""),
            probability=data.get("probability", 0.0),
            key_differences=data.get("key_differences", []),
        )

    def _parse_batch_status(self, data: Dict[str, Any]) -> BatchJobStatus:
        """Parse batch status data into BatchJobStatus object."""
        created_at = None
        completed_at = None

        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(str(data["created_at"]).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        if data.get("completed_at"):
            try:
                completed_at = datetime.fromisoformat(
                    str(data["completed_at"]).replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        return BatchJobStatus(
            batch_id=data.get("batch_id", ""),
            status=data.get("status", "pending"),
            total_debates=data.get("total_debates", 0),
            processed_count=data.get("processed_count", 0),
            success_count=data.get("success_count", 0),
            error_count=data.get("error_count", 0),
            progress_pct=data.get("progress_pct", 0.0),
            created_at=created_at,
            completed_at=completed_at,
        )

    def _parse_batch_result(self, data: Dict[str, Any]) -> BatchDebateResult:
        """Parse batch result data into BatchDebateResult object."""
        explanation = None
        if data.get("explanation"):
            explanation = self._parse_explanation(data["explanation"], data.get("debate_id", ""))

        return BatchDebateResult(
            debate_id=data.get("debate_id", ""),
            status=data.get("status", "error"),
            explanation=explanation,
            error=data.get("error"),
            processing_time_ms=data.get("processing_time_ms", 0.0),
        )


__all__ = [
    "ExplainabilityAPI",
    "DecisionExplanation",
    "ExplanationFactor",
    "EvidenceItem",
    "VotePivot",
    "Counterfactual",
    "BatchJobStatus",
    "BatchDebateResult",
]
