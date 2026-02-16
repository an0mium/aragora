"""
Decision Export Module.

Provides export functionality for Decision entities in multiple formats:
- HTML: Standalone HTML document with embedded styles
- PDF: PDF document (requires weasyprint, graceful fallback)
- Markdown: Plain markdown text

Usage:
    from aragora.explainability.export import export_decision_html, export_decision_pdf

    html = export_decision_html(decision)
    pdf_bytes = export_decision_pdf(decision)  # Returns None if weasyprint unavailable
"""

from __future__ import annotations

import logging

from aragora.explainability.builder import ExplanationBuilder
from aragora.explainability.decision import Decision

logger = logging.getLogger(__name__)

# CSS for exported HTML documents
_EXPORT_CSS = """
body {
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    max-width: 900px;
    margin: 2rem auto;
    padding: 1rem 2rem;
    color: #1a1a1a;
    line-height: 1.6;
}
h1 { color: #111; border-bottom: 2px solid #333; padding-bottom: 0.5rem; }
h2 { color: #333; margin-top: 2rem; }
h3 { color: #555; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; }
th { background-color: #f5f5f5; font-weight: 600; }
tr:nth-child(even) { background-color: #fafafa; }
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.85rem;
    font-weight: 500;
}
.badge-success { background: #d4edda; color: #155724; }
.badge-warning { background: #fff3cd; color: #856404; }
.badge-info { background: #d1ecf1; color: #0c5460; }
.evidence-card {
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    background: #fcfcfc;
}
.meta { color: #666; font-size: 0.9rem; }
.footer { margin-top: 3rem; border-top: 1px solid #ddd; padding-top: 1rem; color: #999; font-size: 0.85rem; }
"""


def _decision_to_html_body(decision: Decision) -> str:
    """Convert a Decision to HTML body content (no wrapper)."""
    parts: list[str] = []

    # Header
    consensus_badge = (
        '<span class="badge badge-success">Consensus Reached</span>'
        if decision.consensus_reached
        else '<span class="badge badge-warning">No Consensus</span>'
    )
    parts.append("<h1>Decision Report</h1>")
    parts.append(f"<p>{consensus_badge}</p>")

    # Summary table
    parts.append("<table>")
    parts.append(f"<tr><th>Decision ID</th><td>{decision.decision_id}</td></tr>")
    parts.append(f"<tr><th>Debate ID</th><td>{decision.debate_id}</td></tr>")
    parts.append(f"<tr><th>Confidence</th><td>{decision.confidence:.0%}</td></tr>")
    parts.append(f"<tr><th>Consensus Type</th><td>{decision.consensus_type}</td></tr>")
    parts.append(f"<tr><th>Rounds Used</th><td>{decision.rounds_used}</td></tr>")
    parts.append(
        f"<tr><th>Agents</th><td>{', '.join(decision.agents_participated) or 'N/A'}</td></tr>"
    )
    parts.append(f"<tr><th>Domain</th><td>{decision.domain}</td></tr>")
    parts.append(f"<tr><th>Timestamp</th><td>{decision.timestamp}</td></tr>")
    parts.append("</table>")

    # Conclusion
    if decision.conclusion:
        parts.append("<h2>Conclusion</h2>")
        # Escape HTML in conclusion text
        escaped = (
            decision.conclusion.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        parts.append(f"<p>{escaped}</p>")

    # Evidence chain
    if decision.evidence_chain:
        parts.append("<h2>Evidence Chain</h2>")
        top_evidence = decision.get_top_evidence(5)
        for ev in top_evidence:
            escaped_content = (
                ev.content[:300].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            parts.append('<div class="evidence-card">')
            parts.append(
                f"<strong>{ev.source}</strong> "
                f'<span class="badge badge-info">{ev.grounding_type}</span> '
                f'<span class="meta">relevance: {ev.relevance_score:.0%}</span>'
            )
            parts.append(f"<p>{escaped_content}</p>")
            parts.append("</div>")
        parts.append(
            f'<p class="meta">Evidence quality score: {decision.evidence_quality_score:.2f}</p>'
        )

    # Vote pivots
    if decision.vote_pivots:
        parts.append("<h2>Vote Analysis</h2>")
        parts.append("<table>")
        parts.append(
            "<tr><th>Agent</th><th>Choice</th><th>Confidence</th>"
            "<th>Influence</th><th>Flip?</th></tr>"
        )
        for vp in decision.vote_pivots[:5]:
            flip_icon = "Yes" if vp.flip_detected else "No"
            parts.append(
                f"<tr><td>{vp.agent}</td><td>{vp.choice}</td>"
                f"<td>{vp.confidence:.0%}</td><td>{vp.influence_score:.0%}</td>"
                f"<td>{flip_icon}</td></tr>"
            )
        parts.append("</table>")
        parts.append(
            f'<p class="meta">Agent agreement score: {decision.agent_agreement_score:.2f}</p>'
        )

    # Confidence attribution
    if decision.confidence_attribution:
        parts.append("<h2>Confidence Factors</h2>")
        parts.append("<table>")
        parts.append("<tr><th>Factor</th><th>Contribution</th><th>Explanation</th></tr>")
        for ca in decision.confidence_attribution:
            parts.append(
                f"<tr><td>{ca.factor}</td><td>{ca.contribution:.0%}</td>"
                f"<td>{ca.explanation}</td></tr>"
            )
        parts.append("</table>")

    # Counterfactuals
    if decision.counterfactuals:
        parts.append("<h2>Counterfactual Analysis</h2>")
        for cf in decision.counterfactuals[:3]:
            parts.append('<div class="evidence-card">')
            parts.append(f"<strong>If:</strong> {cf.condition}<br>")
            parts.append(f"<strong>Then:</strong> {cf.outcome_change}<br>")
            parts.append(
                f'<span class="meta">sensitivity: {cf.sensitivity:.0%}, '
                f"likelihood: {cf.likelihood:.0%}</span>"
            )
            parts.append("</div>")

    # Footer
    parts.append('<div class="footer">')
    parts.append("<p>Generated by Aragora Decision Integrity Platform</p>")
    parts.append(f"<p>Belief stability: {decision.belief_stability_score:.2f}</p>")
    parts.append("</div>")

    return "\n".join(parts)


def export_decision_html(decision: Decision, title: str | None = None) -> str:
    """Export a Decision as a standalone HTML document.

    Args:
        decision: The Decision entity to export.
        title: Optional document title. Defaults to "Decision Report - {debate_id}".

    Returns:
        Complete HTML document as string.
    """
    if title is None:
        title = f"Decision Report - {decision.debate_id}"

    body = _decision_to_html_body(decision)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{_EXPORT_CSS}</style>
</head>
<body>
{body}
</body>
</html>"""


def export_decision_pdf(decision: Decision, title: str | None = None) -> bytes | None:
    """Export a Decision as a PDF document.

    Requires weasyprint to be installed. Returns None if weasyprint is
    not available, allowing callers to implement graceful fallback.

    Args:
        decision: The Decision entity to export.
        title: Optional document title.

    Returns:
        PDF bytes if weasyprint is available, None otherwise.
    """
    html = export_decision_html(decision, title=title)

    try:
        import weasyprint  # type: ignore[import-untyped]

        pdf_bytes: bytes = weasyprint.HTML(string=html).write_pdf()
        return pdf_bytes
    except ImportError:
        logger.info("weasyprint not installed; PDF export unavailable")
        return None
    except OSError as e:
        logger.warning(f"PDF generation failed: {e}")
        return None


def export_decision_markdown(decision: Decision) -> str:
    """Export a Decision as markdown text using ExplanationBuilder.

    Args:
        decision: The Decision entity to export.

    Returns:
        Markdown-formatted summary string.
    """
    builder = ExplanationBuilder()
    return builder.generate_summary(decision)


__all__ = [
    "export_decision_html",
    "export_decision_pdf",
    "export_decision_markdown",
]
