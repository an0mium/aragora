"""
Debate summarization module.

Generates user-friendly summaries from debate results, including:
- One-liner verdict
- Key points and conclusions
- Agreement and disagreement areas
- Confidence assessment
- Actionable next steps
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DebateSummary:
    """Structured summary of a debate result."""

    # Core summary
    one_liner: str = ""
    key_points: list[str] = field(default_factory=list)

    # Consensus analysis
    agreement_areas: list[str] = field(default_factory=list)
    disagreement_areas: list[str] = field(default_factory=list)

    # Confidence and quality
    confidence: float = 0.0
    confidence_label: str = ""  # "high", "medium", "low"
    consensus_strength: str = ""  # "strong", "medium", "weak", "none"

    # Actionable insights
    next_steps: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    # Metadata
    rounds_used: int = 0
    agents_participated: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "one_liner": self.one_liner,
            "key_points": self.key_points,
            "agreement_areas": self.agreement_areas,
            "disagreement_areas": self.disagreement_areas,
            "confidence": self.confidence,
            "confidence_label": self.confidence_label,
            "consensus_strength": self.consensus_strength,
            "next_steps": self.next_steps,
            "caveats": self.caveats,
            "rounds_used": self.rounds_used,
            "agents_participated": self.agents_participated,
            "duration_seconds": self.duration_seconds,
        }


class DebateSummarizer:
    """
    Generates structured summaries from debate results.

    Uses rule-based extraction from debate content to produce
    user-friendly summaries without requiring additional LLM calls.
    """

    # Confidence thresholds
    HIGH_CONFIDENCE = 0.8
    MEDIUM_CONFIDENCE = 0.6

    # Common conclusion patterns
    CONCLUSION_PATTERNS = [
        r"(?:in conclusion|to summarize|therefore|thus|overall)[,:]?\s*(.+?)(?:\.|$)",
        r"(?:the answer is|we conclude that|the recommendation is)[,:]?\s*(.+?)(?:\.|$)",
        r"(?:based on.*?analysis)[,:]?\s*(.+?)(?:\.|$)",
    ]

    # Agreement patterns
    AGREEMENT_PATTERNS = [
        r"(?:all agents agree|unanimous|consensus reached)[,:]?\s*(.+?)(?:\.|$)",
        r"(?:we all support|everyone agrees)[,:]?\s*(.+?)(?:\.|$)",
    ]

    # Disagreement patterns
    DISAGREEMENT_PATTERNS = [
        r"(?:however|but|on the other hand|disagree)[,:]?\s*(.+?)(?:\.|$)",
        r"(?:alternative view|dissenting opinion)[,:]?\s*(.+?)(?:\.|$)",
    ]

    def summarize(self, result: Any) -> DebateSummary:
        """
        Generate a structured summary from a debate result.

        Args:
            result: DebateResult object or dict with debate data

        Returns:
            DebateSummary with extracted insights
        """
        # Handle both DebateResult objects and dicts
        if hasattr(result, "to_dict"):
            data = result
        elif isinstance(result, dict):
            data = _DictWrapper(result)
        else:
            logger.warning(f"Unknown result type: {type(result)}")
            return DebateSummary()

        summary = DebateSummary()

        # Extract metadata
        summary.rounds_used = getattr(data, "rounds_used", 0) or 0
        summary.duration_seconds = getattr(data, "duration_seconds", 0.0) or 0.0
        summary.confidence = getattr(data, "confidence", 0.0) or 0.0

        # Count participating agents
        messages = getattr(data, "messages", []) or []
        agents = set()
        for msg in messages:
            if hasattr(msg, "agent"):
                agents.add(msg.agent)
            elif isinstance(msg, dict):
                agents.add(msg.get("agent", ""))
        summary.agents_participated = len(agents)

        # Determine confidence label
        summary.confidence_label = self._get_confidence_label(summary.confidence)

        # Determine consensus strength
        consensus_reached = getattr(data, "consensus_reached", False)
        consensus_strength = getattr(data, "consensus_strength", "")
        if consensus_reached:
            summary.consensus_strength = consensus_strength or "medium"
        else:
            summary.consensus_strength = "none"

        # Generate one-liner
        summary.one_liner = self._generate_one_liner(data)

        # Extract key points from final answer
        final_answer = getattr(data, "final_answer", "") or ""
        summary.key_points = self._extract_key_points(final_answer)

        # Extract agreement areas
        summary.agreement_areas = self._extract_agreements(data)

        # Extract disagreement areas
        summary.disagreement_areas = self._extract_disagreements(data)

        # Generate next steps
        summary.next_steps = self._generate_next_steps(data)

        # Generate caveats
        summary.caveats = self._generate_caveats(data)

        return summary

    def _get_confidence_label(self, confidence: float) -> str:
        """Map confidence score to label."""
        if confidence >= self.HIGH_CONFIDENCE:
            return "high"
        elif confidence >= self.MEDIUM_CONFIDENCE:
            return "medium"
        else:
            return "low"

    def _generate_one_liner(self, data: Any) -> str:
        """Generate a one-line summary of the debate outcome."""
        consensus_reached = getattr(data, "consensus_reached", False)
        confidence = getattr(data, "confidence", 0.0) or 0.0
        final_answer = getattr(data, "final_answer", "") or ""
        task = getattr(data, "task", "") or ""

        # Extract the core conclusion
        conclusion = self._extract_conclusion(final_answer)

        if consensus_reached:
            if confidence >= self.HIGH_CONFIDENCE:
                prefix = "Agents reached strong consensus"
            else:
                prefix = "Agents reached consensus"
        else:
            dissenting = getattr(data, "dissenting_views", []) or []
            if dissenting:
                prefix = "Agents had mixed opinions"
            else:
                prefix = "No clear consensus reached"

        if conclusion:
            return f"{prefix}: {conclusion}"
        elif task:
            # Fallback to task-based summary
            task_short = task[:80] + "..." if len(task) > 80 else task
            return f"{prefix} on: {task_short}"
        else:
            return prefix

    def _extract_conclusion(self, text: str) -> str:
        """Extract the main conclusion from text."""
        if not text:
            return ""

        # Try conclusion patterns
        for pattern in self.CONCLUSION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                conclusion = match.group(1).strip()
                # Clean up and truncate
                conclusion = re.sub(r"\s+", " ", conclusion)
                if len(conclusion) > 150:
                    conclusion = conclusion[:147] + "..."
                return conclusion

        # Fallback: first sentence if short enough
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if sentences:
            first = sentences[0].strip()
            if len(first) <= 150:
                return first
            return first[:147] + "..."

        return ""

    def _extract_key_points(self, final_answer: str) -> list[str]:
        """Extract key points from the final answer."""
        if not final_answer:
            return []

        key_points = []

        # Look for numbered or bulleted lists
        list_pattern = r"(?:^|\n)\s*(?:\d+[.)]|\*|-)\s*(.+?)(?=\n\s*(?:\d+[.)]|\*|-)|$)"
        matches = re.findall(list_pattern, final_answer, re.MULTILINE)
        if matches:
            for match in matches[:5]:  # Limit to 5 points
                point = match.strip()
                if len(point) > 10:  # Ignore very short items
                    key_points.append(point)

        # If no list found, extract sentences with key phrases
        if not key_points:
            key_phrases = ["important", "key", "critical", "essential", "recommend", "suggest"]
            sentences = re.split(r"(?<=[.!?])\s+", final_answer)
            for sent in sentences:
                if any(phrase in sent.lower() for phrase in key_phrases):
                    clean = sent.strip()
                    if len(clean) > 20 and len(clean) < 200:
                        key_points.append(clean)
                        if len(key_points) >= 3:
                            break

        # If still nothing, take first 3 sentences
        if not key_points:
            sentences = re.split(r"(?<=[.!?])\s+", final_answer)
            for sent in sentences[:3]:
                clean = sent.strip()
                if len(clean) > 20:
                    key_points.append(clean)

        return key_points[:5]

    def _extract_agreements(self, data: Any) -> list[str]:
        """Extract areas where agents agreed."""
        agreements = []

        # From consensus patterns in final answer
        final_answer = getattr(data, "final_answer", "") or ""
        for pattern in self.AGREEMENT_PATTERNS:
            matches = re.findall(pattern, final_answer, re.IGNORECASE)
            for match in matches:
                clean = match.strip()
                if len(clean) > 10:
                    agreements.append(clean)

        # From winning patterns
        winning_patterns = getattr(data, "winning_patterns", []) or []
        for pattern in winning_patterns[:3]:
            if isinstance(pattern, str) and len(pattern) > 10:
                agreements.append(pattern)

        # From high-agreement votes
        votes = getattr(data, "votes", []) or []
        if votes:
            # Group votes by content
            vote_counts: dict[str, int] = {}
            for vote in votes:
                content = ""
                if hasattr(vote, "reasoning"):
                    content = vote.reasoning
                elif isinstance(vote, dict):
                    content = vote.get("reasoning", "")
                if content:
                    # Simplify for grouping
                    key = content[:100].lower()
                    vote_counts[key] = vote_counts.get(key, 0) + 1

            # Find highly agreed items
            for key, count in sorted(vote_counts.items(), key=lambda x: -x[1])[:2]:
                if count >= 2:
                    agreements.append(f"{count} agents agreed: {key[:80]}")

        return agreements[:5]

    def _extract_disagreements(self, data: Any) -> list[str]:
        """Extract areas where agents disagreed."""
        disagreements = []

        # From dissenting views
        dissenting = getattr(data, "dissenting_views", []) or []
        for view in dissenting[:3]:
            if isinstance(view, str) and len(view) > 10:
                # Truncate if too long
                if len(view) > 150:
                    view = view[:147] + "..."
                disagreements.append(view)

        # From disagreement patterns in critiques
        critiques = getattr(data, "critiques", []) or []
        for critique in critiques[:5]:
            content = ""
            if hasattr(critique, "content"):
                content = critique.content
            elif isinstance(critique, dict):
                content = critique.get("content", "")

            for pattern in self.DISAGREEMENT_PATTERNS:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    clean = match.strip()
                    if len(clean) > 10 and clean not in disagreements:
                        disagreements.append(clean)
                        break

        # From debate cruxes
        cruxes = getattr(data, "debate_cruxes", []) or []
        for crux in cruxes[:2]:
            if isinstance(crux, dict):
                claim = crux.get("claim", "")
                if claim and len(claim) > 10:
                    disagreements.append(f"Crux: {claim}")

        return disagreements[:5]

    def _generate_next_steps(self, data: Any) -> list[str]:
        """Generate actionable next steps based on debate outcome."""
        next_steps = []
        consensus_reached = getattr(data, "consensus_reached", False)
        confidence = getattr(data, "confidence", 0.0) or 0.0
        dissenting = getattr(data, "dissenting_views", []) or []
        evidence_suggestions = getattr(data, "evidence_suggestions", []) or []

        if consensus_reached:
            if confidence >= self.HIGH_CONFIDENCE:
                next_steps.append("Proceed with implementation of the agreed approach")
            else:
                next_steps.append("Consider running additional validation before proceeding")

            if dissenting:
                next_steps.append("Review dissenting views for potential edge cases")
        else:
            next_steps.append("Consider running another debate with more specific constraints")
            if len(dissenting) > 1:
                next_steps.append("Identify root causes of disagreement before proceeding")

        # Evidence-based suggestions
        for suggestion in evidence_suggestions[:2]:
            if isinstance(suggestion, dict):
                claim = suggestion.get("claim", "")
                if claim:
                    next_steps.append(f"Gather evidence for: {claim[:60]}")

        # If grounded verdict exists with low score
        grounded = getattr(data, "grounded_verdict", None)
        if grounded:
            score = getattr(grounded, "grounding_score", 1.0)
            if score < 0.5:
                next_steps.append("Improve evidence grounding before finalizing decision")

        return next_steps[:4]

    def _generate_caveats(self, data: Any) -> list[str]:
        """Generate caveats and limitations."""
        caveats = []
        confidence = getattr(data, "confidence", 0.0) or 0.0
        consensus_reached = getattr(data, "consensus_reached", False)
        rounds_used = getattr(data, "rounds_used", 0) or 0

        if confidence < self.MEDIUM_CONFIDENCE:
            caveats.append("Low confidence - results should be validated")

        if not consensus_reached:
            caveats.append("No consensus reached - consider multiple perspectives")

        if rounds_used == 1:
            caveats.append("Single-round debate - limited deliberation")

        # Check for convergence issues
        convergence_status = getattr(data, "convergence_status", "")
        if convergence_status == "diverging":
            caveats.append("Agents were diverging - opinions may be polarized")

        # Check novelty
        avg_novelty = getattr(data, "avg_novelty", 1.0)
        if avg_novelty < 0.3:
            caveats.append("Low novelty - agents may have been repetitive")

        return caveats[:3]


class _DictWrapper:
    """Wrapper to access dict values as attributes."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        return self._data.get(name)


def summarize_debate(result: Any) -> DebateSummary:
    """
    Convenience function to generate a debate summary.

    Args:
        result: DebateResult object or dict

    Returns:
        DebateSummary instance
    """
    summarizer = DebateSummarizer()
    return summarizer.summarize(result)


__all__ = ["DebateSummary", "DebateSummarizer", "summarize_debate"]
