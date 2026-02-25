"""
Composite Epistemic Quality Scorer.

Combines six epistemic quality mechanisms into a single composite metric
per debate/decision:

1. Consensus diversity — Did agents from different providers disagree then converge?
2. Claim decomposition — Were claims properly decomposed and supported?
3. Calibration quality — Brier score / calibration accuracy
4. Uncertainty acknowledgment — Were unknowns explicitly flagged?
5. Provenance completeness — Are sources cited and verifiable?
6. Hollow consensus risk — Trickster detection score (inverted)

Each sub-scorer is independent and optional. Missing data degrades
gracefully to a neutral 0.5 score for the absent component.

Usage:
    from aragora.reasoning.epistemic_scorer import EpistemicScorer

    scorer = EpistemicScorer()
    score = scorer.score_debate(debate_result, votes)
    print(f"Epistemic quality: {score.overall:.2f}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Default neutral score when data is unavailable
_NEUTRAL = 0.5

# Patterns indicating explicit uncertainty acknowledgment in text
_UNCERTAINTY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bI(?:'m| am) (?:not (?:sure|certain)|uncertain)\b", re.IGNORECASE),
    re.compile(r"\b(?:uncertain(?:ty)?|unknown|unclear|ambiguous)\b", re.IGNORECASE),
    re.compile(r"\b(?:might|could|may) (?:be|have)\b", re.IGNORECASE),
    re.compile(r"\b(?:approximately|roughly|estimated)\b", re.IGNORECASE),
    re.compile(r"\blimitation(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bconfidence[:\s]+(?:0\.\d|low|medium)\b", re.IGNORECASE),
    re.compile(r"\b(?:caveat|assumption|risk)\b", re.IGNORECASE),
    re.compile(r"\b(?:don'?t know|cannot determine|insufficient data)\b", re.IGNORECASE),
    re.compile(r"\bexplicit(?:ly)? unknown\b", re.IGNORECASE),
    re.compile(r"\b(?:falsif(?:y|ied|iable)|disprove)\b", re.IGNORECASE),
]

# Known model-provider prefixes used to infer provider from agent name
_PROVIDER_KEYWORDS: dict[str, str] = {
    "claude": "anthropic",
    "anthropic": "anthropic",
    "gpt": "openai",
    "openai": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "gemini": "google",
    "google": "google",
    "grok": "xai",
    "xai": "xai",
    "mistral": "mistral",
    "codestral": "mistral",
    "deepseek": "deepseek",
    "llama": "meta",
    "meta": "meta",
    "qwen": "alibaba",
    "yi": "01ai",
    "kimi": "moonshot",
    "command": "cohere",
    "cohere": "cohere",
}


@dataclass
class EpistemicScorerConfig:
    """Configuration for epistemic scoring weights.

    Each weight controls the contribution of its sub-score to the
    overall composite. Weights are normalized internally, so their
    relative magnitudes are what matter.
    """

    weight_consensus_diversity: float = 1.0
    weight_claim_decomposition: float = 1.0
    weight_calibration_quality: float = 1.0
    weight_uncertainty_acknowledgment: float = 1.0
    weight_provenance_completeness: float = 1.0
    weight_hollow_consensus_risk: float = 1.0


@dataclass
class EpistemicScore:
    """Composite epistemic quality score for a debate or decision.

    Attributes:
        overall: Weighted composite score in [0.0, 1.0].
        consensus_diversity: Provider diversity of agreeing voters.
        claim_decomposition: Ratio of claims with supporting evidence.
        calibration_quality: Agent calibration accuracy (1 - Brier).
        uncertainty_acknowledgment: Explicit uncertainty flagging ratio.
        provenance_completeness: Claims with provenance vs total.
        hollow_consensus_risk: Inverted trickster score (higher = safer).
        components: Raw sub-scores keyed by name for transparency.
    """

    overall: float
    consensus_diversity: float
    claim_decomposition: float
    calibration_quality: float
    uncertainty_acknowledgment: float
    provenance_completeness: float
    hollow_consensus_risk: float
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "overall": round(self.overall, 4),
            "consensus_diversity": round(self.consensus_diversity, 4),
            "claim_decomposition": round(self.claim_decomposition, 4),
            "calibration_quality": round(self.calibration_quality, 4),
            "uncertainty_acknowledgment": round(self.uncertainty_acknowledgment, 4),
            "provenance_completeness": round(self.provenance_completeness, 4),
            "hollow_consensus_risk": round(self.hollow_consensus_risk, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
        }


class EpistemicScorer:
    """Composite scorer combining six epistemic quality mechanisms.

    Each sub-scorer operates independently and produces a value in [0, 1].
    Missing inputs gracefully degrade to a neutral 0.5.
    """

    def __init__(self, config: EpistemicScorerConfig | None = None) -> None:
        self.config = config or EpistemicScorerConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_debate(
        self,
        debate_result: Any,
        votes: list[Any],
        provenance_chain: Any | None = None,
        trickster_report: dict[str, Any] | None = None,
        calibration_data: dict[str, Any] | None = None,
    ) -> EpistemicScore:
        """Score a debate's epistemic quality.

        Args:
            debate_result: A debate result object (or dict) with attributes
                such as ``messages``, ``final_answer``, ``claims``, etc.
            votes: List of ConsensusVote (or dicts with ``agent``, ``vote``,
                ``confidence``, ``reasoning`` keys).
            provenance_chain: Optional ProvenanceChain or ProvenanceManager
                for provenance completeness scoring.
            trickster_report: Optional dict from ``EvidencePoweredTrickster.get_stats()``
                with keys ``hollow_alerts_detected``, ``total_interventions``, etc.
            calibration_data: Optional dict mapping agent names to calibration
                info (e.g. ``{"brier_score": 0.15, "calibration_total": 30}``).

        Returns:
            EpistemicScore with composite and per-dimension scores.
        """
        components: dict[str, float] = {}

        components["consensus_diversity"] = self._score_consensus_diversity(votes)
        components["claim_decomposition"] = self._score_claim_decomposition(debate_result)
        components["calibration_quality"] = self._score_calibration_quality(calibration_data)
        components["uncertainty_acknowledgment"] = self._score_uncertainty(debate_result)
        components["provenance_completeness"] = self._score_provenance(
            debate_result, provenance_chain
        )
        components["hollow_consensus_risk"] = self._score_hollow_consensus(
            trickster_report, votes
        )

        overall = self._compute_weighted_overall(components)

        return EpistemicScore(
            overall=overall,
            consensus_diversity=components["consensus_diversity"],
            claim_decomposition=components["claim_decomposition"],
            calibration_quality=components["calibration_quality"],
            uncertainty_acknowledgment=components["uncertainty_acknowledgment"],
            provenance_completeness=components["provenance_completeness"],
            hollow_consensus_risk=components["hollow_consensus_risk"],
            components=components,
        )

    def score_decision(self, receipt_data: dict[str, Any]) -> EpistemicScore:
        """Score epistemic quality from a decision receipt.

        Decision receipts are dicts typically containing keys like
        ``votes``, ``claims``, ``provenance``, ``trickster``,
        ``calibration``, ``final_answer``, and ``messages``.

        Args:
            receipt_data: Decision receipt dictionary.

        Returns:
            EpistemicScore with composite and per-dimension scores.
        """
        votes = receipt_data.get("votes", [])
        trickster_report = receipt_data.get("trickster")
        calibration_data = receipt_data.get("calibration")
        provenance_chain = receipt_data.get("provenance")

        # Build a lightweight result-like object from receipt
        result_proxy = _ReceiptProxy(receipt_data)

        return self.score_debate(
            debate_result=result_proxy,
            votes=votes,
            provenance_chain=provenance_chain,
            trickster_report=trickster_report,
            calibration_data=calibration_data,
        )

    # ------------------------------------------------------------------
    # Sub-scorers
    # ------------------------------------------------------------------

    def _score_consensus_diversity(self, votes: list[Any]) -> float:
        """Score based on provider diversity among agreeing voters.

        Higher score when agents backed by *different* model providers
        converge on agreement. Unanimous agreement from a single
        provider is less epistemically valuable.
        """
        if not votes:
            return _NEUTRAL

        agreeing_agents: list[str] = []
        for vote in votes:
            vote_type = _get_attr(vote, "vote", "")
            # Accept VoteType enum, string, or dict
            if hasattr(vote_type, "value"):
                vote_type = vote_type.value
            vote_type_str = str(vote_type).lower()

            if vote_type_str in ("agree", "conditional"):
                agent_name = str(_get_attr(vote, "agent", ""))
                if agent_name:
                    agreeing_agents.append(agent_name)

        if not agreeing_agents:
            return 0.0

        providers = set()
        for agent in agreeing_agents:
            provider = _infer_provider(agent)
            providers.add(provider)

        # Score: more unique providers among agreeing agents = higher diversity
        # 1 provider  -> 0.3
        # 2 providers -> 0.6
        # 3+ providers -> 0.8-1.0
        n_providers = len(providers)
        if n_providers <= 1:
            return 0.3
        elif n_providers == 2:
            return 0.6
        else:
            return min(1.0, 0.6 + 0.1 * n_providers)

    def _score_claim_decomposition(self, debate_result: Any) -> float:
        """Score based on claim decomposition and evidence support.

        Looks at the ratio of claims that have supporting evidence
        vs unsupported assertions.
        """
        claims = _get_attr(debate_result, "claims", None)
        if claims is None:
            claims = _get_attr(debate_result, "evidence_chain", None)
        if not claims:
            return _NEUTRAL

        if not isinstance(claims, (list, tuple)):
            return _NEUTRAL

        total = len(claims)
        if total == 0:
            return _NEUTRAL

        supported = 0
        for claim in claims:
            # Check for supporting evidence on the claim
            sup_evidence = _get_attr(claim, "supporting_evidence", None)
            if sup_evidence and len(sup_evidence) > 0:
                supported += 1
                continue

            # Check for evidence_chain or sources
            sources = _get_attr(claim, "sources", None) or _get_attr(claim, "evidence", None)
            if sources and len(sources) > 0:
                supported += 1
                continue

            # Check if claim has a parent (refinement chain)
            parent = _get_attr(claim, "parent_claim_id", None)
            if parent is not None:
                supported += 1

        return supported / total

    def _score_calibration_quality(
        self, calibration_data: dict[str, Any] | None
    ) -> float:
        """Score based on agent calibration accuracy.

        Uses Brier score data when available. Lower Brier = better
        calibration = higher score.
        """
        if not calibration_data:
            return _NEUTRAL

        # calibration_data can be:
        #   - dict of agent_name -> {brier_score, calibration_total}
        #   - dict with a top-level "brier_score" key
        if "brier_score" in calibration_data:
            brier = float(calibration_data["brier_score"])
            return max(0.0, min(1.0, 1.0 - brier))

        brier_scores: list[float] = []
        for agent_data in calibration_data.values():
            if isinstance(agent_data, dict):
                brier = agent_data.get("brier_score")
                total = agent_data.get("calibration_total", 0)
                if brier is not None and total >= 5:
                    brier_scores.append(float(brier))
            elif isinstance(agent_data, (int, float)):
                brier_scores.append(float(agent_data))

        if not brier_scores:
            return _NEUTRAL

        avg_brier = sum(brier_scores) / len(brier_scores)
        return max(0.0, min(1.0, 1.0 - avg_brier))

    def _score_uncertainty(self, debate_result: Any) -> float:
        """Score based on explicit uncertainty acknowledgment.

        Scans agent messages/responses for uncertainty markers like
        confidence bounds, explicit unknowns, and caveats.
        """
        texts = _extract_response_texts(debate_result)
        if not texts:
            return _NEUTRAL

        total_agents = len(texts)
        agents_with_uncertainty = 0

        for text in texts:
            if not text:
                continue
            matches = sum(1 for pat in _UNCERTAINTY_PATTERNS if pat.search(text))
            # An agent acknowledges uncertainty if at least 2 different
            # patterns match (avoids false positives from single words)
            if matches >= 2:
                agents_with_uncertainty += 1

        if total_agents == 0:
            return _NEUTRAL

        # Reward some uncertainty (epistemic humility) but not excessive
        # 0% uncertainty -> 0.2 (overconfident)
        # ~50% uncertainty -> 1.0 (ideal)
        # 100% uncertainty -> 0.7 (too uncertain)
        ratio = agents_with_uncertainty / total_agents
        if ratio < 0.1:
            return 0.2
        elif ratio <= 0.6:
            return 0.4 + ratio
        else:
            return max(0.7, 1.0 - (ratio - 0.6))

    def _score_provenance(
        self,
        debate_result: Any,
        provenance_chain: Any | None,
    ) -> float:
        """Score based on provenance completeness.

        Checks what fraction of claims have associated provenance records
        or citation data.
        """
        claims = _get_attr(debate_result, "claims", None)
        if not claims or not isinstance(claims, (list, tuple)):
            if provenance_chain is not None:
                # Have provenance data but no claims to check against
                records = _get_attr(provenance_chain, "records", None)
                if records and len(records) > 0:
                    return 0.7
            return _NEUTRAL

        total = len(claims)
        if total == 0:
            return _NEUTRAL

        # Check provenance chain for records
        record_ids: set[str] = set()
        if provenance_chain is not None:
            records = _get_attr(provenance_chain, "records", None)
            if records:
                for r in records:
                    rid = _get_attr(r, "id", None)
                    if rid:
                        record_ids.add(str(rid))

            # Also check citation graph if available
            graph = _get_attr(provenance_chain, "graph", None)
            if graph:
                claim_citations = _get_attr(graph, "claim_citations", None)
                if claim_citations:
                    record_ids.update(str(k) for k in claim_citations.keys())

        # Count claims with provenance
        claims_with_provenance = 0
        for claim in claims:
            claim_id = _get_attr(claim, "claim_id", None)

            # Check if claim has provenance in the chain
            if claim_id and str(claim_id) in record_ids:
                claims_with_provenance += 1
                continue

            # Check if claim has inline evidence referencing provenance
            sup_evidence = _get_attr(claim, "supporting_evidence", None)
            if sup_evidence and len(sup_evidence) > 0:
                for ev in sup_evidence:
                    ev_type = _get_attr(ev, "evidence_type", "")
                    if str(ev_type) in ("citation", "data", "tool_output"):
                        claims_with_provenance += 1
                        break

        return claims_with_provenance / total

    def _score_hollow_consensus(
        self,
        trickster_report: dict[str, Any] | None,
        votes: list[Any],
    ) -> float:
        """Score hollow consensus risk (inverted: higher = safer).

        If Trickster ran, uses its alert and intervention counts.
        Otherwise, estimates from vote pattern uniformity.
        """
        if trickster_report is not None:
            alerts = trickster_report.get("hollow_alerts_detected", 0)
            interventions = trickster_report.get("total_interventions", 0)

            # More alerts and interventions indicate higher hollow risk
            # Invert: high alerts => low safety score
            risk = min(1.0, (alerts * 0.2 + interventions * 0.15))
            return max(0.0, 1.0 - risk)

        # Fallback: estimate from vote pattern
        if not votes:
            return _NEUTRAL

        confidences: list[float] = []
        vote_types: list[str] = []
        for vote in votes:
            conf = _get_attr(vote, "confidence", None)
            if conf is not None:
                confidences.append(float(conf))
            vt = _get_attr(vote, "vote", "")
            if hasattr(vt, "value"):
                vt = vt.value
            vote_types.append(str(vt).lower())

        # Suspiciously uniform high confidence is a hollow consensus signal
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            conf_spread = max(confidences) - min(confidences) if len(confidences) > 1 else 0.0

            # Very high avg confidence with near-zero spread is suspicious
            if avg_conf > 0.9 and conf_spread < 0.1:
                return 0.3  # Likely hollow

        # All-agree with no dissent is slightly suspicious
        unique_votes = set(vote_types)
        if len(unique_votes) == 1 and "agree" in unique_votes:
            return 0.5  # Could be hollow, could be genuine

        # Some diversity in votes indicates genuine deliberation
        if len(unique_votes) >= 2:
            return 0.8

        return 0.6

    # ------------------------------------------------------------------
    # Weighted combination
    # ------------------------------------------------------------------

    def _compute_weighted_overall(self, components: dict[str, float]) -> float:
        """Compute weighted average of component scores."""
        weights = {
            "consensus_diversity": self.config.weight_consensus_diversity,
            "claim_decomposition": self.config.weight_claim_decomposition,
            "calibration_quality": self.config.weight_calibration_quality,
            "uncertainty_acknowledgment": self.config.weight_uncertainty_acknowledgment,
            "provenance_completeness": self.config.weight_provenance_completeness,
            "hollow_consensus_risk": self.config.weight_hollow_consensus_risk,
        }

        total_weight = sum(weights.values())
        if total_weight == 0:
            return _NEUTRAL

        weighted_sum = sum(
            components.get(key, _NEUTRAL) * w for key, w in weights.items()
        )
        return max(0.0, min(1.0, weighted_sum / total_weight))


# ======================================================================
# Helpers
# ======================================================================


class _ReceiptProxy:
    """Lightweight proxy to make a receipt dict behave like a result object."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name)


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Get attribute from object or dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _infer_provider(agent_name: str) -> str:
    """Infer model provider from agent name.

    Uses keyword matching against known provider prefixes.
    Returns the agent name itself as fallback (unique provider).
    """
    lower = agent_name.lower()
    for keyword, provider in _PROVIDER_KEYWORDS.items():
        if keyword in lower:
            return provider
    return lower  # Treat unknown agents as their own unique provider


def _extract_response_texts(debate_result: Any) -> list[str]:
    """Extract agent response texts from a debate result."""
    texts: list[str] = []

    # Try messages attribute
    messages = _get_attr(debate_result, "messages", None)
    if messages and isinstance(messages, (list, tuple)):
        for msg in messages:
            content = _get_attr(msg, "content", None)
            if content and isinstance(content, str):
                texts.append(content)
        if texts:
            return texts

    # Try final_answer
    final = _get_attr(debate_result, "final_answer", None)
    if final and isinstance(final, str):
        texts.append(final)

    # Try reasoning_summary
    reasoning = _get_attr(debate_result, "reasoning_summary", None)
    if reasoning and isinstance(reasoning, str):
        texts.append(reasoning)

    # Try responses dict
    responses = _get_attr(debate_result, "responses", None)
    if responses and isinstance(responses, dict):
        for text in responses.values():
            if isinstance(text, str):
                texts.append(text)

    return texts


__all__ = [
    "EpistemicScore",
    "EpistemicScorer",
    "EpistemicScorerConfig",
]
