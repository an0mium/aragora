"""
Receipt export utilities for HTML and PDF formats.

Provides receipt_to_html() and receipt_to_pdf() for exporting
DecisionReceipts to user-facing formats.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.gauntlet.receipt_models import DecisionReceipt


def receipt_to_html(receipt: DecisionReceipt) -> str:
    """Render a DecisionReceipt as a standalone HTML document.

    Args:
        receipt: The receipt to render.

    Returns:
        Complete HTML document as a string.
    """
    verdict_class = {
        "PASS": "verdict-pass",
        "CONDITIONAL": "verdict-conditional",
        "FAIL": "verdict-fail",
    }.get(receipt.verdict, "verdict-unknown")

    risk = receipt.risk_summary or {}
    findings_rows = ""
    for vuln in (receipt.vulnerability_details or [])[:20]:
        severity = escape(str(vuln.get("severity", "unknown")))
        title = escape(str(vuln.get("title", "Untitled")))
        desc = escape(str(vuln.get("description", ""))[:300])
        findings_rows += f"<tr><td>{severity}</td><td>{title}</td><td>{desc}</td></tr>\n"

    provenance_rows = ""
    for rec in (receipt.provenance_chain or [])[:50]:
        ts = escape(str(rec.timestamp))
        ev = escape(rec.event_type)
        agent = escape(rec.agent or "")
        desc = escape(rec.description[:200])
        provenance_rows += f"<tr><td>{ts}</td><td>{ev}</td><td>{agent}</td><td>{desc}</td></tr>\n"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Decision Receipt {escape(receipt.receipt_id)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
h1 {{ border-bottom: 2px solid #2563eb; padding-bottom: 8px; }}
.verdict-pass {{ color: #16a34a; }}
.verdict-conditional {{ color: #d97706; }}
.verdict-fail {{ color: #dc2626; }}
.verdict-unknown {{ color: #6b7280; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; }}
th {{ background: #f3f4f6; }}
.meta {{ color: #6b7280; font-size: 0.9em; }}
.hash {{ font-family: monospace; font-size: 0.85em; color: #4b5563; }}
</style>
</head>
<body>
<h1>Decision Receipt</h1>
<p class="meta">
  Receipt ID: <span class="hash">{escape(receipt.receipt_id)}</span><br/>
  Gauntlet ID: <span class="hash">{escape(receipt.gauntlet_id)}</span><br/>
  Generated: {escape(str(receipt.timestamp))}<br/>
  Schema Version: {escape(receipt.schema_version)}
</p>

<h2 class="{verdict_class}">Verdict: {escape(receipt.verdict)}</h2>
<p>Confidence: {receipt.confidence:.1%} | Robustness: {receipt.robustness_score:.1%}</p>
<blockquote>{escape(receipt.verdict_reasoning)}</blockquote>

<h2>Risk Summary</h2>
<table>
<tr><th>Severity</th><th>Count</th></tr>
<tr><td>Critical</td><td>{risk.get("critical", 0)}</td></tr>
<tr><td>High</td><td>{risk.get("high", 0)}</td></tr>
<tr><td>Medium</td><td>{risk.get("medium", 0)}</td></tr>
<tr><td>Low</td><td>{risk.get("low", 0)}</td></tr>
</table>
<p>Attacks: {receipt.attacks_attempted} attempted, {receipt.attacks_successful} successful | Probes: {receipt.probes_run}</p>

{"<h2>Findings</h2><table><tr><th>Severity</th><th>Title</th><th>Description</th></tr>" + findings_rows + "</table>" if findings_rows else ""}

{"<h2>Provenance Chain</h2><table><tr><th>Time</th><th>Event</th><th>Agent</th><th>Description</th></tr>" + provenance_rows + "</table>" if provenance_rows else ""}

<hr/>
<p class="meta hash">Artifact Hash: {escape(receipt.artifact_hash)}</p>
</body>
</html>"""


def receipt_to_pdf(receipt: DecisionReceipt) -> bytes:
    """Render a DecisionReceipt as a minimal PDF.

    This produces a simple text-based PDF without requiring external
    libraries. For production use, consider weasyprint or reportlab.

    Args:
        receipt: The receipt to render.

    Returns:
        PDF content as bytes.
    """
    # Build plain text content for embedding in PDF
    lines = [
        f"Decision Receipt: {receipt.receipt_id}",
        f"Gauntlet ID: {receipt.gauntlet_id}",
        f"Generated: {receipt.timestamp}",
        "",
        f"Verdict: {receipt.verdict}",
        f"Confidence: {receipt.confidence:.1%}",
        f"Robustness Score: {receipt.robustness_score:.1%}",
        f"Reasoning: {receipt.verdict_reasoning}",
        "",
        "Risk Summary:",
    ]

    risk = receipt.risk_summary or {}
    for level in ("critical", "high", "medium", "low"):
        lines.append(f"  {level.capitalize()}: {risk.get(level, 0)}")

    lines.append(
        f"\nAttacks: {receipt.attacks_attempted} attempted, {receipt.attacks_successful} successful"
    )
    lines.append(f"Probes: {receipt.probes_run}")

    if receipt.vulnerability_details:
        lines.append("\nFindings:")
        for vuln in receipt.vulnerability_details[:10]:
            lines.append(f"  [{vuln.get('severity', '?')}] {vuln.get('title', 'Untitled')}")

    lines.append(f"\nArtifact Hash: {receipt.artifact_hash}")

    text = "\n".join(lines)

    # Minimal PDF generation (PDF 1.4 spec)
    return _build_minimal_pdf(text)


def _build_minimal_pdf(text: str) -> bytes:
    """Build a minimal valid PDF with the given text content."""
    # Encode text safely for PDF (replace non-ASCII)
    safe_text = text.encode("ascii", errors="replace").decode("ascii")

    # Split into lines and limit length
    pdf_lines = []
    for line in safe_text.split("\n"):
        while len(line) > 80:
            pdf_lines.append(line[:80])
            line = line[80:]
        pdf_lines.append(line)

    # Build PDF stream content
    stream_lines = ["BT", "/F1 10 Tf"]
    y = 750
    for line in pdf_lines:
        if y < 50:
            break
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_lines.append(f"1 0 0 1 50 {y} Tm")
        stream_lines.append(f"({escaped}) Tj")
        y -= 14
    stream_lines.append("ET")
    stream_content = "\n".join(stream_lines)

    objects: list[str] = []

    # Object 1: Catalog
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj")

    # Object 2: Pages
    objects.append("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj")

    # Object 3: Page
    objects.append(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R "
        "/MediaBox [0 0 612 792] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj"
    )

    # Object 4: Content stream
    objects.append(
        f"4 0 obj\n<< /Length {len(stream_content)} >>\nstream\n{stream_content}\nendstream\nendobj"
    )

    # Object 5: Font
    objects.append("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>\nendobj")

    # Build final PDF
    pdf_parts = ["%PDF-1.4\n"]
    offsets = []
    for obj in objects:
        offsets.append(len("".join(pdf_parts)))
        pdf_parts.append(obj + "\n")

    xref_offset = len("".join(pdf_parts))
    pdf_parts.append("xref\n")
    pdf_parts.append(f"0 {len(objects) + 1}\n")
    pdf_parts.append("0000000000 65535 f \n")
    for offset in offsets:
        pdf_parts.append(f"{offset:010d} 00000 n \n")

    pdf_parts.append("trailer\n")
    pdf_parts.append(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n")
    pdf_parts.append("startxref\n")
    pdf_parts.append(f"{xref_offset}\n")
    pdf_parts.append("%%EOF\n")

    return "".join(pdf_parts).encode("ascii")


__all__ = ["receipt_to_html", "receipt_to_pdf"]
