"""
Result formatting for debate conclusions.

Provides formatted output for debate results including:
- Verdict summary
- Final answer
- Vote breakdown
- Dissenting views
- Key cruxes
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.core import DebateResult


class ResultFormatter:
    """
    Formats debate results for human-readable output.

    Usage:
        formatter = ResultFormatter()
        conclusion = formatter.format_conclusion(result)
    """

    def __init__(self, max_answer_length: int = 1000, max_view_length: int = 300):
        """
        Initialize the formatter.

        Args:
            max_answer_length: Maximum characters for final answer display.
            max_view_length: Maximum characters for dissenting view display.
        """
        self.max_answer_length = max_answer_length
        self.max_view_length = max_view_length

    def format_conclusion(self, result: "DebateResult") -> str:
        """
        Format a clear, readable debate conclusion with full context.

        Args:
            result: The debate result to format.

        Returns:
            Formatted conclusion string with sections for verdict,
            final answer, vote breakdown, dissenting views, and cruxes.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("DEBATE CONCLUSION")
        lines.append("=" * 60)

        # Verdict section
        lines.append("\n## VERDICT")
        self._add_verdict(lines, result)

        # Winner (if determined)
        if hasattr(result, "winner") and result.winner:
            lines.append(f"Winner: {result.winner}")

        # Final answer section
        lines.append("\n## FINAL ANSWER")
        self._add_final_answer(lines, result)

        # Vote breakdown (if available)
        self._add_vote_breakdown(lines, result)

        # Dissenting views (if any)
        self._add_dissenting_views(lines, result)

        # Debate cruxes (key disagreement points)
        self._add_cruxes(lines, result)

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def _add_verdict(self, lines: list[str], result: "DebateResult") -> None:
        """Add verdict section to output."""
        if result.consensus_reached:
            lines.append(f"Consensus: YES ({result.confidence:.0%} agreement)")
            if hasattr(result, "consensus_strength") and result.consensus_strength:
                lines.append(f"Strength: {result.consensus_strength.upper()}")
        else:
            lines.append(f"Consensus: NO (only {result.confidence:.0%} agreement)")

    def _add_final_answer(self, lines: list[str], result: "DebateResult") -> None:
        """Add final answer section to output."""
        if result.final_answer:
            # Truncate if very long, but show substantial content
            if len(result.final_answer) > self.max_answer_length:
                answer_display = result.final_answer[: self.max_answer_length] + "..."
            else:
                answer_display = result.final_answer
            lines.append(answer_display)
        else:
            lines.append("No final answer determined.")

    def _add_vote_breakdown(self, lines: list[str], result: "DebateResult") -> None:
        """Add vote breakdown section if votes are available."""
        if not (hasattr(result, "votes") and result.votes):
            return

        lines.append("\n## VOTE BREAKDOWN")
        vote_counts = {}
        for vote in result.votes:
            voter = getattr(vote, "voter", "unknown")
            choice = getattr(vote, "choice", "abstain")
            vote_counts[voter] = choice
        for voter, choice in vote_counts.items():
            lines.append(f"  - {voter}: {choice}")

    def _add_dissenting_views(self, lines: list[str], result: "DebateResult") -> None:
        """Add dissenting views section if present."""
        if not (hasattr(result, "dissenting_views") and result.dissenting_views):
            return

        lines.append("\n## DISSENTING VIEWS")
        for i, view in enumerate(result.dissenting_views[:3]):
            if len(view) > self.max_view_length:
                view_display = view[: self.max_view_length] + "..."
            else:
                view_display = view
            lines.append(f"  {i + 1}. {view_display}")

    def _add_cruxes(self, lines: list[str], result: "DebateResult") -> None:
        """Add key cruxes section if present."""
        if not (hasattr(result, "belief_cruxes") and result.belief_cruxes):
            return

        lines.append("\n## KEY CRUXES")
        for crux in result.belief_cruxes[:3]:
            claim = crux.get("claim", "unknown")[:80]
            uncertainty = crux.get("uncertainty", 0)
            lines.append(f"  - {claim}... (uncertainty: {uncertainty:.2f})")


def format_conclusion(result: "DebateResult") -> str:
    """
    Convenience function to format a debate conclusion.

    Args:
        result: The debate result to format.

    Returns:
        Formatted conclusion string.
    """
    formatter = ResultFormatter()
    return formatter.format_conclusion(result)


__all__ = ["ResultFormatter", "format_conclusion"]
