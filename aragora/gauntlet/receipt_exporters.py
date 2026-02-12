"""
Export functions for Decision Receipts.

Contains the rendering/export logic extracted from the DecisionReceipt class:
- to_markdown -> receipt_to_markdown
- to_html -> receipt_to_html
- to_html_paginated -> receipt_to_html_paginated
- to_sarif -> receipt_to_sarif
- to_csv -> receipt_to_csv

These are called by DecisionReceipt's methods via delegation.
"""

from __future__ import annotations

import hashlib
from html import escape
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .receipt_models import DecisionReceipt, ProvenanceRecord


def receipt_to_markdown(
    receipt: DecisionReceipt,
    include_provenance: bool = True,
    include_evidence: bool = True,
) -> str:
    """Generate markdown report with full provenance and evidence links.

    Args:
        receipt: The DecisionReceipt to render
        include_provenance: Include full provenance chain section
        include_evidence: Include evidence hashes for findings

    Returns:
        Markdown formatted decision receipt
    """
    verdict_emoji = {
        "PASS": "✓",
        "CONDITIONAL": "~",
        "FAIL": "✗",
    }.get(receipt.verdict, "?")

    lines = [
        "# Decision Receipt",
        "",
        f"**Receipt ID:** `{receipt.receipt_id}`",
        f"**Gauntlet ID:** `{receipt.gauntlet_id}`",
        f"**Generated:** {receipt.timestamp}",
        "",
        "---",
        "",
        f"## Verdict: [{verdict_emoji}] {receipt.verdict}",
        "",
        f"**Confidence:** {receipt.confidence:.1%}",
        f"**Robustness Score:** {receipt.robustness_score:.1%}",
        "",
        f"> {receipt.verdict_reasoning}",
        "",
    ]

    # Consensus proof section
    if receipt.consensus_proof:
        lines.extend(
            [
                "---",
                "",
                "## Consensus Proof",
                "",
                f"- **Consensus Reached:** {'Yes' if receipt.consensus_proof.reached else 'No'}",
                f"- **Method:** {receipt.consensus_proof.method}",
                f"- **Confidence:** {receipt.consensus_proof.confidence:.1%}",
            ]
        )
        if receipt.consensus_proof.supporting_agents:
            lines.append(
                f"- **Supporting Agents:** {', '.join(receipt.consensus_proof.supporting_agents)}"
            )
        if receipt.consensus_proof.dissenting_agents:
            lines.append(
                f"- **Dissenting Agents:** {', '.join(receipt.consensus_proof.dissenting_agents)}"
            )
        if receipt.consensus_proof.evidence_hash:
            lines.append(f"- **Evidence Hash:** `{receipt.consensus_proof.evidence_hash}`")
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## Risk Summary",
            "",
            "| Severity | Count |",
            "|----------|-------|",
            f"| Critical | {receipt.risk_summary.get('critical', 0)} |",
            f"| High | {receipt.risk_summary.get('high', 0)} |",
            f"| Medium | {receipt.risk_summary.get('medium', 0)} |",
            f"| Low | {receipt.risk_summary.get('low', 0)} |",
            f"| **Total** | **{receipt.vulnerabilities_found}** |",
            "",
            "---",
            "",
            "## Validation Coverage",
            "",
            f"- **Attacks Attempted:** {receipt.attacks_attempted}",
            f"- **Attacks Successful:** {receipt.attacks_successful}",
            f"- **Probes Run:** {receipt.probes_run}",
            "",
        ]
    )

    if receipt.vulnerability_details:
        lines.append("---")
        lines.append("")
        lines.append("## Critical Findings")
        lines.append("")
        for i, vuln in enumerate(receipt.vulnerability_details[:10], 1):
            finding_id = vuln.get("id", f"F-{i:03d}")
            lines.append(f"### [{finding_id}] {vuln.get('title', 'Unknown')}")
            lines.append("")
            lines.append(
                f"**Severity:** {vuln.get('severity', vuln.get('severity_level', 'unknown')).upper()}"
            )
            lines.append(f"**Category:** {vuln.get('category', 'unknown')}")
            if vuln.get("verified"):
                lines.append("**Verified:** Yes")
            if vuln.get("source"):
                lines.append(f"**Source:** {vuln.get('source')}")
            lines.append("")
            lines.append(vuln.get("description", "")[:500])
            if vuln.get("evidence") and include_evidence:
                lines.append("")
                evidence = vuln.get("evidence", "")
                if isinstance(evidence, str) and len(evidence) > 200:
                    evidence = evidence[:200] + "..."
                lines.append(f"**Evidence:** {evidence}")
                # Generate evidence hash for verification
                evidence_str = str(vuln.get("evidence", "") or vuln.get("description", ""))
                evidence_hash = hashlib.sha256(evidence_str.encode()).hexdigest()[:16]
                lines.append(f"**Evidence Hash:** `{evidence_hash}`")
            if vuln.get("mitigation"):
                lines.append("")
                lines.append(f"**Mitigation:** {vuln.get('mitigation')}")
            lines.append("")

    if receipt.dissenting_views:
        lines.append("---")
        lines.append("")
        lines.append("## Dissenting Views")
        lines.append("")
        for view in receipt.dissenting_views[:5]:
            lines.append(f"- {view}")
        lines.append("")

    # Provenance chain section
    if include_provenance and receipt.provenance_chain:
        lines.append("---")
        lines.append("")
        lines.append("## Provenance Chain")
        lines.append("")
        lines.append("| # | Timestamp | Event | Agent | Description | Evidence Hash |")
        lines.append("|---|-----------|-------|-------|-------------|---------------|")
        for i, record in enumerate(receipt.provenance_chain, 1):
            timestamp = record.timestamp[:19] if record.timestamp else "-"
            event = record.event_type or "-"
            agent = record.agent or "-"
            desc = (
                (record.description[:40] + "...")
                if len(record.description) > 40
                else record.description
            )
            evidence_hash = f"`{record.evidence_hash}`" if record.evidence_hash else "-"
            lines.append(
                f"| {i} | {timestamp} | {event} | {agent} | {desc} | {evidence_hash} |"
            )
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## Integrity Verification",
            "",
            "| Field | Hash |",
            "|-------|------|",
            f"| Input | `{receipt.input_hash}` |",
            f"| Artifact | `{receipt.artifact_hash}` |",
            "",
            "To verify this receipt has not been tampered with, the artifact hash",
            "can be recomputed from the receipt contents and compared.",
            "",
            "---",
            "",
            "*Generated by Aragora Gauntlet*",
        ]
    )

    return "\n".join(lines)


def receipt_to_html(
    receipt: DecisionReceipt,
    max_findings: int = 20,
    max_provenance: int = 50,
) -> str:
    """Export as self-contained HTML document.

    Args:
        receipt: The DecisionReceipt to render
        max_findings: Maximum number of findings to include (default 20)
        max_provenance: Maximum provenance records to include (default 50)
    """
    verdict_color = {
        "PASS": "#28a745",
        "CONDITIONAL": "#ffc107",
        "FAIL": "#dc3545",
    }.get(receipt.verdict, "#6c757d")

    # Use list + join for O(n) complexity instead of O(n^2) string concatenation
    findings_parts: list[str] = []
    for vuln in receipt.vulnerability_details[:max_findings]:
        severity = str(vuln.get("severity", "UNKNOWN")).upper()
        severity_color = {
            "CRITICAL": "#dc3545",
            "HIGH": "#fd7e14",
            "MEDIUM": "#ffc107",
            "LOW": "#28a745",
        }.get(severity, "#6c757d")
        title = escape(str(vuln.get("title", "")))
        description = escape(str(vuln.get("description", "")))
        mitigation = vuln.get("mitigation")
        mitigation_html = ""
        if mitigation:
            mitigation_html = f"<p><em>Mitigation: {escape(str(mitigation))}</em></p>"

        findings_parts.append(
            f'<div class="finding" style="border-left: 4px solid {severity_color};">'
            f'<strong style="color: {severity_color};">[{severity}]</strong> {title}'
            f"<p>{description}</p>"
            f"{mitigation_html}"
            "</div>"
        )
    findings_html = "".join(findings_parts)

    risk_summary = receipt.risk_summary or {}

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Decision Receipt - {escape(receipt.receipt_id)}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .verdict {{ font-size: 22px; font-weight: bold; color: {verdict_color}; margin: 20px 0; padding: 16px; background: #f8f9fa; border-radius: 8px; }}
        .scores {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin: 20px 0; }}
        .score {{ text-align: center; padding: 12px; background: #f8f9fa; border-radius: 8px; }}
        .score-value {{ font-size: 28px; font-weight: bold; color: #333; }}
        .score-label {{ font-size: 12px; color: #666; }}
        .section {{ margin: 24px 0; }}
        .finding {{ margin: 10px 0; padding: 12px; background: #f8f9fa; border-left: 4px solid #ccc; }}
        table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; }}
        .meta {{ font-size: 13px; color: #666; }}
        code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace; }}
    </style>
</head>
<body>
    <h1>Decision Receipt</h1>
    <p class="meta">
        <strong>Receipt ID:</strong> <code>{escape(receipt.receipt_id)}</code><br>
        <strong>Gauntlet ID:</strong> <code>{escape(receipt.gauntlet_id)}</code><br>
        <strong>Generated:</strong> {escape(receipt.timestamp)}
    </p>

    <div class="verdict">
        VERDICT: {escape(receipt.verdict)}
        <div style="font-size: 14px; font-weight: normal; margin-top: 8px;">
            Confidence: {receipt.confidence:.0%} | Robustness: {receipt.robustness_score:.0%}
        </div>
        {f'<div style="font-size: 13px; font-weight: normal; margin-top: 8px;">{escape(receipt.verdict_reasoning)}</div>' if receipt.verdict_reasoning else ""}
    </div>

    <div class="scores">
        <div class="score">
            <div class="score-value">{receipt.confidence:.0%}</div>
            <div class="score-label">Confidence</div>
        </div>
        <div class="score">
            <div class="score-value">{receipt.robustness_score:.0%}</div>
            <div class="score-label">Robustness</div>
        </div>
    </div>

    <div class="section">
        <h2>Risk Summary</h2>
        <table>
            <tr><th>Severity</th><th>Count</th></tr>
            <tr><td>Critical</td><td>{risk_summary.get("critical", 0)}</td></tr>
            <tr><td>High</td><td>{risk_summary.get("high", 0)}</td></tr>
            <tr><td>Medium</td><td>{risk_summary.get("medium", 0)}</td></tr>
            <tr><td>Low</td><td>{risk_summary.get("low", 0)}</td></tr>
            <tr><td><strong>Total</strong></td><td><strong>{receipt.vulnerabilities_found}</strong></td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Coverage</h2>
        <p class="meta">
            Attacks Attempted: {receipt.attacks_attempted}<br>
            Attacks Successful: {receipt.attacks_successful}<br>
            Probes Run: {receipt.probes_run}
        </p>
    </div>

    <div class="section">
        <h2>Findings</h2>
        {findings_html or '<p class="meta">No findings reported.</p>'}
    </div>

    <div class="section">
        <h2>Integrity</h2>
        <p class="meta">
            Input Hash: <code>{escape(receipt.input_hash[:32])}...</code><br>
            Artifact Hash: <code>{escape(receipt.artifact_hash[:32])}...</code>
        </p>
    </div>
{receipt._signature_verification_html()}
</body>
</html>
"""


def receipt_to_html_paginated(
    receipt: DecisionReceipt,
    findings_per_page: int = 10,
    max_provenance: int = 50,
    provenance_sampling: str = "first_last",
) -> str:
    """Export as paginated HTML document optimized for PDF rendering.

    Uses CSS page breaks and provenance sampling to handle large receipts
    efficiently without memory issues during PDF generation.

    Args:
        receipt: The DecisionReceipt to render
        findings_per_page: Number of findings per page (default 10)
        max_provenance: Maximum provenance records to include (default 50)
        provenance_sampling: Sampling strategy for provenance chain:
            - "all": Include all records up to max_provenance
            - "first_last": Include first and last half (default)
            - "sampled": Evenly sample across the chain

    Returns:
        HTML string with CSS page breaks suitable for PDF rendering
    """
    verdict_color = {
        "PASS": "#28a745",
        "CONDITIONAL": "#ffc107",
        "FAIL": "#dc3545",
    }.get(receipt.verdict, "#6c757d")

    # Sample provenance chain based on strategy
    provenance = receipt._sample_provenance(max_provenance, provenance_sampling)

    # Build findings HTML with page breaks
    findings_parts: list[str] = []
    for i, vuln in enumerate(receipt.vulnerability_details):
        # Add page break before each new page (except first)
        if i > 0 and i % findings_per_page == 0:
            findings_parts.append('<div style="page-break-before: always;"></div>')

        severity = str(vuln.get("severity", "UNKNOWN")).upper()
        severity_color = {
            "CRITICAL": "#dc3545",
            "HIGH": "#fd7e14",
            "MEDIUM": "#ffc107",
            "LOW": "#28a745",
        }.get(severity, "#6c757d")
        title = escape(str(vuln.get("title", "")))
        description = escape(str(vuln.get("description", "")))
        mitigation = vuln.get("mitigation")
        mitigation_html = ""
        if mitigation:
            mitigation_html = f"<p><em>Mitigation: {escape(str(mitigation))}</em></p>"

        findings_parts.append(
            f'<div class="finding" style="border-left: 4px solid {severity_color};">'
            f'<strong style="color: {severity_color};">[{severity}]</strong> {title}'
            f"<p>{description}</p>"
            f"{mitigation_html}"
            "</div>"
        )
    findings_html = "".join(findings_parts)

    # Build provenance HTML
    provenance_parts: list[str] = []
    if provenance:
        provenance_parts.append('<div style="page-break-before: always;"></div>')
        provenance_parts.append('<div class="section"><h2>Provenance Chain</h2>')
        provenance_parts.append(
            "<table><tr><th>#</th><th>Timestamp</th><th>Event</th>"
            "<th>Agent</th><th>Description</th></tr>"
        )
        for i, record in enumerate(provenance, 1):
            timestamp = record.timestamp[:19] if record.timestamp else "-"
            event = record.event_type or "-"
            agent = record.agent or "-"
            desc = escape(record.description[:50]) if record.description else "-"
            provenance_parts.append(
                f"<tr><td>{i}</td><td>{escape(timestamp)}</td>"
                f"<td>{escape(event)}</td><td>{escape(agent)}</td>"
                f"<td>{desc}</td></tr>"
            )
        provenance_parts.append("</table></div>")
    provenance_html = "".join(provenance_parts)

    risk_summary = receipt.risk_summary or {}

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Decision Receipt - {escape(receipt.receipt_id)}</title>
    <style>
        @page {{ margin: 2cm; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .verdict {{ font-size: 22px; font-weight: bold; color: {verdict_color}; margin: 20px 0; padding: 16px; background: #f8f9fa; border-radius: 8px; }}
        .section {{ margin: 24px 0; }}
        .finding {{ margin: 10px 0; padding: 12px; background: #f8f9fa; border-left: 4px solid #ccc; }}
        table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #eee; font-size: 11px; }}
        th {{ background: #f8f9fa; }}
        .meta {{ font-size: 13px; color: #666; }}
        code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace; }}
    </style>
</head>
<body>
    <h1>Decision Receipt</h1>
    <p class="meta">
        <strong>Receipt ID:</strong> <code>{escape(receipt.receipt_id)}</code><br>
        <strong>Gauntlet ID:</strong> <code>{escape(receipt.gauntlet_id)}</code><br>
        <strong>Generated:</strong> {escape(receipt.timestamp)}
    </p>

    <div class="verdict">
        VERDICT: {escape(receipt.verdict)}
        <div style="font-size: 14px; font-weight: normal; margin-top: 8px;">
            Confidence: {receipt.confidence:.0%} | Robustness: {receipt.robustness_score:.0%}
        </div>
    </div>

    <div class="section">
        <h2>Risk Summary</h2>
        <table>
            <tr><th>Severity</th><th>Count</th></tr>
            <tr><td>Critical</td><td>{risk_summary.get("critical", 0)}</td></tr>
            <tr><td>High</td><td>{risk_summary.get("high", 0)}</td></tr>
            <tr><td>Medium</td><td>{risk_summary.get("medium", 0)}</td></tr>
            <tr><td>Low</td><td>{risk_summary.get("low", 0)}</td></tr>
            <tr><td><strong>Total</strong></td><td><strong>{receipt.vulnerabilities_found}</strong></td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Findings ({len(receipt.vulnerability_details)} total)</h2>
        {findings_html or '<p class="meta">No findings reported.</p>'}
    </div>

    {provenance_html}

    <div class="section">
        <h2>Integrity</h2>
        <p class="meta">
            Input Hash: <code>{escape(receipt.input_hash[:32])}...</code><br>
            Artifact Hash: <code>{escape(receipt.artifact_hash[:32])}...</code>
        </p>
    </div>
{receipt._signature_verification_html()}
</body>
</html>
"""


def receipt_to_sarif(receipt: DecisionReceipt) -> dict:
    """Export as SARIF 2.1.0 format.

    SARIF (Static Analysis Results Interchange Format) is the OASIS standard
    for exchanging static analysis results. This enables interoperability with:
    - GitHub Security (code scanning)
    - Azure DevOps
    - VS Code SARIF Viewer
    - SonarQube
    - DefectDojo

    Args:
        receipt: The DecisionReceipt to export

    Returns:
        SARIF 2.1.0 dictionary
    """
    # Map severity to SARIF levels
    sarif_level_map = {
        "CRITICAL": "error",
        "HIGH": "error",
        "MEDIUM": "warning",
        "LOW": "note",
    }

    # Map severity to SARIF security-severity scores (CVSS-like)
    sarif_severity_map = {
        "CRITICAL": "9.0",
        "HIGH": "7.0",
        "MEDIUM": "4.0",
        "LOW": "1.0",
    }

    # Build rules from unique vulnerability categories
    rules: list[dict[str, Any]] = []
    rule_ids: dict[str, int] = {}

    for idx, vuln in enumerate(receipt.vulnerability_details):
        category = vuln.get("category", "unknown")
        if category not in rule_ids:
            rule_id = f"ARAGORA-{len(rule_ids) + 1:03d}"
            rule_ids[category] = len(rules)
            rules.append(
                {
                    "id": rule_id,
                    "name": category.replace("_", " ").title(),
                    "shortDescription": {"text": f"Aragora Gauntlet: {category}"},
                    "fullDescription": {"text": f"Security finding in category: {category}"},
                    "helpUri": "https://aragora.ai/docs/gauntlet",
                    "properties": {
                        "security-severity": sarif_severity_map.get(
                            str(vuln.get("severity_level", "MEDIUM")).upper(), "4.0"
                        ),
                        "tags": ["security", "aragora", category],
                    },
                }
            )

    # Build results from vulnerability details
    results = []
    for vuln in receipt.vulnerability_details:
        category = vuln.get("category", "unknown")
        severity = str(vuln.get("severity_level", vuln.get("severity", "MEDIUM"))).upper()
        rule_idx = rule_ids.get(category, 0)
        rule_id = rules[rule_idx]["id"] if rule_idx < len(rules) else "ARAGORA-000"

        result = {
            "ruleId": rule_id,
            "ruleIndex": rule_idx,
            "level": sarif_level_map.get(severity, "warning"),
            "message": {"text": vuln.get("description", vuln.get("title", "Finding"))},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": f"input/{receipt.input_hash[:8]}",
                            "uriBaseId": "GAUNTLET_ROOT",
                        }
                    },
                    "logicalLocations": [
                        {
                            "name": vuln.get("title", "Unknown"),
                            "kind": "finding",
                        }
                    ],
                }
            ],
            "fingerprints": {
                "aragora/v1": hashlib.sha256(
                    f"{vuln.get('id', '')}:{vuln.get('title', '')}".encode()
                ).hexdigest()[:32]
            },
            "properties": {
                "gauntlet_id": receipt.gauntlet_id,
                "receipt_id": receipt.receipt_id,
                "category": category,
                "severity": severity,
                "verified": vuln.get("verified", False),
            },
        }

        # Add fix suggestions if mitigation is present
        if vuln.get("mitigation"):
            result["fixes"] = [{"description": {"text": vuln.get("mitigation", "")}}]

        results.append(result)

    # Build SARIF document
    sarif = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Aragora Gauntlet",
                        "version": "1.0.0",
                        "informationUri": "https://aragora.ai/gauntlet",
                        "rules": rules,
                        "properties": {
                            "verdict": receipt.verdict,
                            "confidence": receipt.confidence,
                            "robustness_score": receipt.robustness_score,
                        },
                    }
                },
                "results": results,
                "invocations": [
                    {
                        "executionSuccessful": True,
                        "endTimeUtc": receipt.timestamp,
                        "properties": {
                            "gauntlet_id": receipt.gauntlet_id,
                            "receipt_id": receipt.receipt_id,
                            "attacks_attempted": receipt.attacks_attempted,
                            "attacks_successful": receipt.attacks_successful,
                            "probes_run": receipt.probes_run,
                        },
                    }
                ],
                "artifacts": [
                    {
                        "location": {
                            "uri": f"input/{receipt.input_hash[:8]}",
                            "uriBaseId": "GAUNTLET_ROOT",
                        },
                        "hashes": {
                            "sha-256": receipt.input_hash,
                        },
                        "length": -1,
                        "properties": {
                            "summary": receipt.input_summary[:200],
                        },
                    }
                ],
                "properties": {
                    "risk_summary": receipt.risk_summary,
                    "artifact_hash": receipt.artifact_hash,
                    "consensus_proof": (
                        receipt.consensus_proof.to_dict() if receipt.consensus_proof else None
                    ),
                },
            }
        ],
    }

    return sarif


def receipt_to_csv(receipt: DecisionReceipt) -> str:
    """Export findings as CSV format.

    Args:
        receipt: The DecisionReceipt to export

    Returns:
        CSV content with vulnerability details
    """
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(
        [
            "Finding ID",
            "Category",
            "Severity",
            "Title",
            "Description",
            "Mitigation",
            "Verified",
            "Source",
        ]
    )

    # Data rows
    for vuln in receipt.vulnerability_details:
        writer.writerow(
            [
                vuln.get("id", ""),
                vuln.get("category", ""),
                vuln.get("severity_level", vuln.get("severity", "")),
                vuln.get("title", ""),
                vuln.get("description", "")[:500],
                vuln.get("mitigation", ""),
                vuln.get("verified", False),
                vuln.get("source", ""),
            ]
        )

    return output.getvalue()
