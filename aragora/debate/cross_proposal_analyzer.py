"""
Cross-Proposal Analyzer for multi-agent evidence validation.

This module analyzes evidence patterns across multiple agent proposals to detect:
- Shared evidence (same sources cited by multiple agents)
- Contradictory evidence (opposite conclusions from similar evidence)
- Evidence gaps (claims in consensus but no evidence from any agent)
- Echo chamber patterns (agents citing each other without external validation)

Key insight: When agents converge, they should converge on EVIDENCE, not just conclusions.
If they agree but cite different (or no) evidence, the consensus is hollow.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from aragora.debate.evidence_quality import EvidenceType

if TYPE_CHECKING:
    from aragora.debate.evidence_linker import EvidenceCoverageResult, EvidenceLink

logger = logging.getLogger(__name__)


# Lazy import to avoid scipy/numpy import failures
_evidence_linker_module = None


def _get_evidence_linker_module():
    """Lazy import of evidence_linker module."""
    global _evidence_linker_module
    if _evidence_linker_module is None:
        try:
            from aragora.debate import evidence_linker as _module

            _evidence_linker_module = _module
        except ImportError as e:
            logger.debug(f"evidence_linker module not available: {e}")
            return None
    return _evidence_linker_module


def _get_evidence_linker_class():
    """Get EvidenceClaimLinker class if available."""
    module = _get_evidence_linker_module()
    if module:
        return module.EvidenceClaimLinker
    return None


@dataclass
class SharedEvidence:
    """Evidence cited by multiple agents."""

    evidence_text: str
    evidence_type: EvidenceType
    agents: list[str]  # Agents who cited this evidence
    claims_supported: list[str]  # Claims this evidence supports

    @property
    def agent_count(self) -> int:
        """Number of agents citing this evidence."""
        return len(self.agents)


@dataclass
class Contradiction:
    """Contradictory evidence between agents."""

    agent1: str
    agent2: str
    topic: str  # The topic/claim being contradicted
    evidence1: str
    evidence2: str
    description: str  # Human-readable description


@dataclass
class EvidenceGap:
    """A claim made without supporting evidence."""

    claim: str
    agents_making_claim: list[str]
    gap_severity: float  # 0-1, higher = more severe


@dataclass
class CrossProposalAnalysis:
    """Complete analysis of evidence patterns across proposals."""

    # Positive signals
    shared_evidence: list[SharedEvidence]
    evidence_corroboration_score: float  # 0-1, how much evidence is shared

    # Warning signals
    contradictory_evidence: list[Contradiction]
    evidence_gaps: list[EvidenceGap]

    # Echo chamber detection
    redundancy_score: float  # 0-1, how much agents echo each other
    unique_evidence_sources: int  # Number of distinct evidence sources
    total_evidence_sources: int  # Total evidence citations

    # Per-agent analysis
    agent_coverage: dict[str, float]  # agent -> coverage score
    weakest_agent: Optional[str]  # Agent with lowest evidence quality

    @property
    def has_concerns(self) -> bool:
        """Whether there are significant evidence concerns."""
        return (
            len(self.evidence_gaps) > 0
            or len(self.contradictory_evidence) > 0
            or self.redundancy_score > 0.7
        )

    @property
    def top_concern(self) -> Optional[str]:
        """Get the most significant concern."""
        if self.evidence_gaps:
            return f"Evidence gap: {self.evidence_gaps[0].claim[:80]}..."
        if self.contradictory_evidence:
            c = self.contradictory_evidence[0]
            return f"Contradiction between {c.agent1} and {c.agent2} on {c.topic}"
        if self.redundancy_score > 0.7:
            return f"Echo chamber: {self.redundancy_score:.0%} redundancy"
        return None


class CrossProposalAnalyzer:
    """
    Analyzes evidence patterns across converging agent proposals.

    This is the key component for detecting hollow consensus. When multiple
    agents agree on a conclusion, this analyzer checks whether they're
    agreeing on evidence too, or just echoing each other.

    Usage:
        analyzer = CrossProposalAnalyzer()

        proposals = {
            "claude": "We should use caching. According to [1], it improves...",
            "gpt": "Caching is the best approach. Studies show...",
            "gemini": "I agree caching helps performance.",
        }

        analysis = analyzer.analyze(proposals)

        if analysis.has_concerns:
            print(f"Warning: {analysis.top_concern}")
    """

    def __init__(
        self,
        linker: Optional[Any] = None,
        min_redundancy_similarity: float = 0.7,
        min_claim_overlap: float = 0.5,
    ):
        """
        Initialize the analyzer.

        Args:
            linker: Evidence-claim linker (created if not provided)
            min_redundancy_similarity: Threshold for redundancy detection
            min_claim_overlap: Threshold for claim overlap detection
        """
        # Lazy load EvidenceClaimLinker to avoid scipy/numpy import failures
        self.linker = linker
        if self.linker is None:
            EvidenceClaimLinker = _get_evidence_linker_class()
            if EvidenceClaimLinker is not None:
                self.linker = EvidenceClaimLinker()
        self.min_redundancy_similarity = min_redundancy_similarity
        self.min_claim_overlap = min_claim_overlap

    def analyze(self, proposals: dict[str, str]) -> CrossProposalAnalysis:
        """
        Analyze evidence patterns across multiple agent proposals.

        Args:
            proposals: Dict of agent name -> proposal text

        Returns:
            CrossProposalAnalysis with complete analysis
        """
        if not proposals or len(proposals) < 2:
            return self._empty_analysis()

        # If linker not available (ML dependencies missing), return empty analysis
        if self.linker is None:
            logger.debug("EvidenceClaimLinker not available, skipping analysis")
            return self._empty_analysis()

        # Get evidence coverage for each agent
        agent_coverage: dict[str, Any] = {}
        agent_links: dict[str, list] = {}

        for agent, text in proposals.items():
            coverage = self.linker.compute_evidence_coverage(text)
            agent_coverage[agent] = coverage
            agent_links[agent] = coverage.links

        # Find shared evidence
        shared = self._find_shared_evidence(agent_links)

        # Find contradictions
        contradictions = self._find_contradictions(agent_coverage, proposals)

        # Find evidence gaps (claims without evidence from anyone)
        gaps = self._find_consensus_gaps(agent_coverage)

        # Calculate redundancy
        redundancy, unique_sources, total_sources = self._calculate_redundancy(
            agent_links, proposals
        )

        # Corroboration score
        corroboration = self._calculate_corroboration(shared, len(proposals))

        # Coverage scores
        coverage_scores = {agent: cov.coverage for agent, cov in agent_coverage.items()}

        # Find weakest agent
        weakest = min(coverage_scores, key=coverage_scores.get) if coverage_scores else None

        return CrossProposalAnalysis(
            shared_evidence=shared,
            evidence_corroboration_score=corroboration,
            contradictory_evidence=contradictions,
            evidence_gaps=gaps,
            redundancy_score=redundancy,
            unique_evidence_sources=unique_sources,
            total_evidence_sources=total_sources,
            agent_coverage=coverage_scores,
            weakest_agent=weakest,
        )

    def _empty_analysis(self) -> CrossProposalAnalysis:
        """Return empty analysis for edge cases."""
        return CrossProposalAnalysis(
            shared_evidence=[],
            evidence_corroboration_score=0.0,
            contradictory_evidence=[],
            evidence_gaps=[],
            redundancy_score=0.0,
            unique_evidence_sources=0,
            total_evidence_sources=0,
            agent_coverage={},
            weakest_agent=None,
        )

    def _find_shared_evidence(
        self,
        agent_links: dict[str, list[EvidenceLink]],
    ) -> list[SharedEvidence]:
        """Find evidence cited by multiple agents."""
        # Group evidence by normalized form
        evidence_map: dict[str, dict] = {}

        for agent, links in agent_links.items():
            for link in links:
                # Normalize evidence text
                normalized = self._normalize_evidence(link.evidence)
                if not normalized:
                    continue

                if normalized not in evidence_map:
                    evidence_map[normalized] = {
                        "text": link.evidence,
                        "type": link.evidence_type,
                        "agents": set(),
                        "claims": [],
                    }

                evidence_map[normalized]["agents"].add(agent)
                evidence_map[normalized]["claims"].append(link.claim)

        # Filter to shared evidence (2+ agents)
        shared = []
        for data in evidence_map.values():
            if len(data["agents"]) >= 2:
                shared.append(
                    SharedEvidence(
                        evidence_text=data["text"],
                        evidence_type=data["type"],
                        agents=list(data["agents"]),
                        claims_supported=list(set(data["claims"])),  # Dedupe claims
                    )
                )

        return shared

    def _normalize_evidence(self, evidence: str) -> str:
        """Normalize evidence for comparison."""
        # Remove whitespace and lowercase
        normalized = re.sub(r"\s+", " ", evidence.lower().strip())
        # Remove punctuation for comparison
        normalized = re.sub(r"[^\w\s]", "", normalized)
        return normalized if len(normalized) > 5 else ""

    def _find_contradictions(
        self,
        agent_coverage: dict[str, EvidenceCoverageResult],
        proposals: dict[str, str],
    ) -> list[Contradiction]:
        """Find contradictory evidence between agents."""
        contradictions = []

        # Look for opposite conclusions on similar topics
        agents = list(proposals.keys())

        for i, agent1 in enumerate(agents):
            for agent2 in agents[i + 1 :]:
                # Find claims that appear to contradict
                for link1 in agent_coverage[agent1].links:
                    for link2 in agent_coverage[agent2].links:
                        if self._are_contradictory(link1, link2):
                            contradictions.append(
                                Contradiction(
                                    agent1=agent1,
                                    agent2=agent2,
                                    topic=self._extract_topic(link1.claim, link2.claim),
                                    evidence1=link1.evidence,
                                    evidence2=link2.evidence,
                                    description=f"{agent1} and {agent2} cite different evidence for similar claims",
                                )
                            )

        return contradictions[:3]  # Limit to top 3

    def _are_contradictory(self, link1: EvidenceLink, link2: EvidenceLink) -> bool:
        """Check if two evidence links are contradictory."""
        # Simple heuristic: claims overlap but evidence differs significantly
        claim_overlap = self._text_similarity(link1.claim, link2.claim)
        evidence_overlap = self._text_similarity(link1.evidence, link2.evidence)

        # Contradiction: similar claims, different evidence
        return claim_overlap > 0.4 and evidence_overlap < 0.2

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity using word overlap."""
        words1 = set(re.findall(r"\b\w{4,}\b", text1.lower()))
        words2 = set(re.findall(r"\b\w{4,}\b", text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _extract_topic(self, claim1: str, claim2: str) -> str:
        """Extract common topic from two claims."""
        words1 = set(re.findall(r"\b\w{4,}\b", claim1.lower()))
        words2 = set(re.findall(r"\b\w{4,}\b", claim2.lower()))
        common = words1 & words2

        if common:
            return " ".join(list(common)[:3])
        return "related topic"

    def _find_consensus_gaps(
        self,
        agent_coverage: dict[str, EvidenceCoverageResult],
    ) -> list[EvidenceGap]:
        """Find claims made by agents but without evidence from anyone."""
        # Collect all claims across agents
        claim_to_agents: dict[str, list[str]] = {}
        claim_to_linked: dict[str, bool] = {}

        for agent, coverage in agent_coverage.items():
            # Get all claims (linked and unlinked)
            all_claims = set()
            for link in coverage.links:
                all_claims.add(link.claim)
                claim_to_linked[link.claim] = link.is_strong_link

            for claim in coverage.unlinked_claims:
                all_claims.add(claim)
                if claim not in claim_to_linked:
                    claim_to_linked[claim] = False

            for claim in all_claims:
                if claim not in claim_to_agents:
                    claim_to_agents[claim] = []
                claim_to_agents[claim].append(agent)

        # Find claims made by multiple agents without strong evidence
        gaps = []
        for claim, agents in claim_to_agents.items():
            if len(agents) >= 2 and not claim_to_linked.get(claim, False):
                # Severity based on how many agents make this claim
                severity = len(agents) / len(agent_coverage)
                gaps.append(
                    EvidenceGap(
                        claim=claim[:200],  # Truncate long claims
                        agents_making_claim=agents,
                        gap_severity=severity,
                    )
                )

        # Sort by severity and return top gaps
        gaps.sort(key=lambda g: g.gap_severity, reverse=True)
        return gaps[:5]

    def _calculate_redundancy(
        self,
        agent_links: dict[str, list[EvidenceLink]],
        proposals: dict[str, str],
    ) -> tuple[float, int, int]:
        """
        Calculate redundancy score (echo chamber detection).

        Returns: (redundancy_score, unique_sources, total_sources)
        """
        # Collect all evidence sources
        all_evidence = []
        unique_evidence = set()

        for links in agent_links.values():
            for link in links:
                normalized = self._normalize_evidence(link.evidence)
                if normalized:
                    all_evidence.append(normalized)
                    unique_evidence.add(normalized)

        total = len(all_evidence)
        unique = len(unique_evidence)

        if total == 0:
            return 0.0, 0, 0

        # Redundancy: 1 - (unique / total)
        # High redundancy = many agents citing same evidence = echo chamber
        redundancy = 1.0 - (unique / total) if total > 0 else 0.0

        return redundancy, unique, total

    def _calculate_corroboration(
        self,
        shared: list[SharedEvidence],
        num_agents: int,
    ) -> float:
        """Calculate evidence corroboration score."""
        if not shared or num_agents < 2:
            return 0.0

        # Score based on how much evidence is shared by multiple agents
        max_sharing = sum(s.agent_count for s in shared)
        potential_sharing = len(shared) * num_agents

        return max_sharing / potential_sharing if potential_sharing > 0 else 0.0


__all__ = [
    "SharedEvidence",
    "Contradiction",
    "EvidenceGap",
    "CrossProposalAnalysis",
    "CrossProposalAnalyzer",
]
