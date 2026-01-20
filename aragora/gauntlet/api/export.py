"""
Gauntlet Export Utilities - v1.

Provides export functionality for DecisionReceipts and RiskHeatmaps
in multiple formats suitable for integration and archival.

Supported formats:
- JSON (with optional schema validation)
- Markdown (human-readable)
- HTML (self-contained, printable)
- CSV (for spreadsheet import)
- SARIF (Static Analysis Results Interchange Format)
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from aragora.gauntlet.receipt import DecisionReceipt
    from aragora.gauntlet.heatmap import RiskHeatmap


class ReceiptExportFormat(Enum):
    """Export formats for DecisionReceipt."""

    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"
    SARIF = "sarif"  # Static Analysis Results Interchange Format


class HeatmapExportFormat(Enum):
    """Export formats for RiskHeatmap."""

    JSON = "json"
    CSV = "csv"
    SVG = "svg"
    ASCII = "ascii"
    HTML = "html"


@dataclass
class ExportOptions:
    """Options for export operations."""

    # JSON options
    indent: int = 2
    sort_keys: bool = False

    # Content options
    include_provenance: bool = True
    include_config: bool = False
    max_vulnerabilities: int = 100

    # Validation
    validate_schema: bool = False

    # Metadata
    include_export_metadata: bool = True


def export_receipt(
    receipt: "DecisionReceipt",
    format: ReceiptExportFormat = ReceiptExportFormat.JSON,
    options: Optional[ExportOptions] = None,
) -> str:
    """
    Export a DecisionReceipt to the specified format.

    Args:
        receipt: The DecisionReceipt to export
        format: Target export format
        options: Export options

    Returns:
        Exported string representation
    """
    opts = options or ExportOptions()

    if format == ReceiptExportFormat.JSON:
        return _export_receipt_json(receipt, opts)
    elif format == ReceiptExportFormat.MARKDOWN:
        return receipt.to_markdown()
    elif format == ReceiptExportFormat.HTML:
        return receipt.to_html()
    elif format == ReceiptExportFormat.CSV:
        return _export_receipt_csv(receipt, opts)
    elif format == ReceiptExportFormat.SARIF:
        return _export_receipt_sarif(receipt, opts)
    else:
        raise ValueError(f"Unsupported format: {format}")


def export_heatmap(
    heatmap: "RiskHeatmap",
    format: HeatmapExportFormat = HeatmapExportFormat.JSON,
    options: Optional[ExportOptions] = None,
) -> str:
    """
    Export a RiskHeatmap to the specified format.

    Args:
        heatmap: The RiskHeatmap to export
        format: Target export format
        options: Export options

    Returns:
        Exported string representation
    """
    opts = options or ExportOptions()

    if format == HeatmapExportFormat.JSON:
        return _export_heatmap_json(heatmap, opts)
    elif format == HeatmapExportFormat.CSV:
        return _export_heatmap_csv(heatmap, opts)
    elif format == HeatmapExportFormat.SVG:
        return heatmap.to_svg()
    elif format == HeatmapExportFormat.ASCII:
        return heatmap.to_ascii()
    elif format == HeatmapExportFormat.HTML:
        return _export_heatmap_html(heatmap, opts)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _export_receipt_json(receipt: "DecisionReceipt", opts: ExportOptions) -> str:
    """Export receipt as JSON."""
    data = receipt.to_dict()

    # Apply options
    if not opts.include_provenance:
        data.pop("provenance_chain", None)
    if not opts.include_config:
        data.pop("config_used", None)
    if opts.max_vulnerabilities and len(data.get("vulnerability_details", [])) > opts.max_vulnerabilities:
        data["vulnerability_details"] = data["vulnerability_details"][:opts.max_vulnerabilities]
        data["_truncated_vulnerabilities"] = True

    # Add export metadata
    if opts.include_export_metadata:
        data["_export"] = {
            "format": "json",
            "exported_at": datetime.now().isoformat(),
            "schema_version": "1.0.0",
        }

    # Validate if requested
    if opts.validate_schema:
        from .schema import validate_receipt
        is_valid, errors = validate_receipt(data)
        if not is_valid:
            data["_validation_errors"] = errors

    return json.dumps(data, indent=opts.indent, sort_keys=opts.sort_keys)


def _export_receipt_csv(receipt: "DecisionReceipt", opts: ExportOptions) -> str:
    """Export receipt as CSV (findings table)."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Finding ID",
        "Severity",
        "Category",
        "Title",
        "Description",
        "Mitigation",
        "Verified",
    ])

    # Vulnerability rows
    for vuln in receipt.vulnerability_details[:opts.max_vulnerabilities]:
        writer.writerow([
            vuln.get("id", ""),
            vuln.get("severity", vuln.get("severity_level", "")),
            vuln.get("category", ""),
            vuln.get("title", ""),
            vuln.get("description", "")[:500],
            vuln.get("mitigation", ""),
            vuln.get("verified", ""),
        ])

    # Summary row
    writer.writerow([])
    writer.writerow(["Summary"])
    writer.writerow(["Verdict", receipt.verdict])
    writer.writerow(["Confidence", f"{receipt.confidence:.1%}"])
    writer.writerow(["Robustness", f"{receipt.robustness_score:.1%}"])
    writer.writerow(["Total Findings", receipt.vulnerabilities_found])
    writer.writerow(["Receipt ID", receipt.receipt_id])

    return output.getvalue()


def _export_receipt_sarif(receipt: "DecisionReceipt", opts: ExportOptions) -> str:
    """
    Export receipt as SARIF (Static Analysis Results Interchange Format).

    SARIF is a standard for static analysis tools that enables
    integration with IDEs and CI/CD pipelines.

    Specification: https://sarifweb.azurewebsites.net/
    """
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
                        "rules": _generate_sarif_rules(receipt),
                    }
                },
                "results": _generate_sarif_results(receipt, opts),
                "invocations": [
                    {
                        "executionSuccessful": True,
                        "startTimeUtc": receipt.timestamp,
                        "endTimeUtc": receipt.timestamp,
                    }
                ],
            }
        ],
    }

    return json.dumps(sarif, indent=opts.indent)


def _generate_sarif_rules(receipt: "DecisionReceipt") -> List[Dict[str, Any]]:
    """Generate SARIF rules from vulnerability categories."""
    categories = set()
    for vuln in receipt.vulnerability_details:
        cat = vuln.get("category", "general")
        categories.add(cat)

    rules = []
    for i, cat in enumerate(sorted(categories)):
        rules.append({
            "id": f"GAUNTLET-{i+1:03d}",
            "name": cat.replace("_", " ").title(),
            "shortDescription": {"text": f"Gauntlet finding: {cat}"},
            "fullDescription": {"text": f"Vulnerability detected in category: {cat}"},
            "defaultConfiguration": {"level": "warning"},
        })

    return rules


def _generate_sarif_results(receipt: "DecisionReceipt", opts: ExportOptions) -> List[Dict[str, Any]]:
    """Generate SARIF results from vulnerabilities."""
    results = []

    # Map severity to SARIF levels
    severity_map = {
        "critical": "error",
        "high": "error",
        "medium": "warning",
        "low": "note",
    }

    for i, vuln in enumerate(receipt.vulnerability_details[:opts.max_vulnerabilities]):
        severity = str(vuln.get("severity", vuln.get("severity_level", "medium"))).lower()
        level = severity_map.get(severity, "warning")

        result = {
            "ruleId": f"GAUNTLET-{i+1:03d}",
            "level": level,
            "message": {
                "text": vuln.get("title", "Unknown vulnerability"),
            },
            "properties": {
                "category": vuln.get("category", ""),
                "severity": severity,
                "verified": vuln.get("verified", False),
            },
        }

        # Add description if available
        if vuln.get("description"):
            result["message"]["markdown"] = vuln["description"][:1000]

        # Add mitigation as fix
        if vuln.get("mitigation"):
            result["fixes"] = [
                {
                    "description": {"text": vuln["mitigation"][:500]},
                }
            ]

        results.append(result)

    return results


def _export_heatmap_json(heatmap: "RiskHeatmap", opts: ExportOptions) -> str:
    """Export heatmap as JSON."""
    data = heatmap.to_dict()

    if opts.include_export_metadata:
        data["_export"] = {
            "format": "json",
            "exported_at": datetime.now().isoformat(),
        }

    return json.dumps(data, indent=opts.indent, sort_keys=opts.sort_keys)


def _export_heatmap_csv(heatmap: "RiskHeatmap", opts: ExportOptions) -> str:
    """Export heatmap as CSV matrix."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row with severities
    writer.writerow(["Category"] + [s.upper() for s in heatmap.severities])

    # Category rows
    for category in heatmap.categories:
        row = [category]
        for severity in heatmap.severities:
            cell = heatmap.get_cell(category, severity)
            row.append(cell.count if cell else 0)
        writer.writerow(row)

    # Total row
    writer.writerow([])
    writer.writerow(["TOTAL"] + [heatmap.get_severity_total(s) for s in heatmap.severities])

    return output.getvalue()


def _export_heatmap_html(heatmap: "RiskHeatmap", opts: ExportOptions) -> str:
    """Export heatmap as standalone HTML with embedded SVG."""
    from html import escape

    svg = heatmap.to_svg()

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Risk Heatmap</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ margin: 20px 0; padding: 16px; background: #f8f9fa; border-radius: 8px; }}
        .heatmap {{ margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 8px; text-align: center; border: 1px solid #ddd; }}
        th {{ background: #f8f9fa; }}
        .critical {{ background: rgba(220, 38, 38, 0.2); }}
        .high {{ background: rgba(234, 88, 12, 0.2); }}
        .medium {{ background: rgba(234, 179, 8, 0.2); }}
        .low {{ background: rgba(34, 197, 94, 0.2); }}
        .meta {{ font-size: 12px; color: #888; margin-top: 20px; }}
    </style>
</head>
<body>
    <h1>Risk Heatmap</h1>

    <div class="summary">
        <strong>Total Findings:</strong> {heatmap.total_findings}<br>
        <strong>Highest Risk Category:</strong> {escape(heatmap.highest_risk_category or 'N/A')}<br>
        <strong>Highest Severity:</strong> {escape(heatmap.highest_risk_severity or 'N/A')}
    </div>

    <div class="heatmap">
        {svg}
    </div>

    <h2>Data Table</h2>
    <table>
        <tr>
            <th>Category</th>
            {''.join(f'<th class="{s}">{escape(s.upper())}</th>' for s in heatmap.severities)}
            <th>Total</th>
        </tr>
        {''.join(_heatmap_row_html(heatmap, cat) for cat in heatmap.categories)}
        <tr>
            <th>TOTAL</th>
            {''.join(f'<td class="{s}"><strong>{heatmap.get_severity_total(s)}</strong></td>' for s in heatmap.severities)}
            <td><strong>{heatmap.total_findings}</strong></td>
        </tr>
    </table>

    <p class="meta">Generated by Aragora Gauntlet at {datetime.now().isoformat()}</p>
</body>
</html>
"""


def _heatmap_row_html(heatmap: "RiskHeatmap", category: str) -> str:
    """Generate HTML table row for a heatmap category."""
    from html import escape

    cells = []
    total = 0
    for severity in heatmap.severities:
        cell = heatmap.get_cell(category, severity)
        count = cell.count if cell else 0
        total += count
        cells.append(f'<td class="{severity}">{count}</td>')

    return f"<tr><td>{escape(category)}</td>{''.join(cells)}<td>{total}</td></tr>"


# =============================================================================
# Batch Export
# =============================================================================


def export_receipts_bundle(
    receipts: List["DecisionReceipt"],
    format: ReceiptExportFormat = ReceiptExportFormat.JSON,
    options: Optional[ExportOptions] = None,
) -> str:
    """
    Export multiple receipts as a bundle.

    Args:
        receipts: List of DecisionReceipts to export
        format: Target export format
        options: Export options

    Returns:
        Bundle string (JSON array or concatenated documents)
    """
    opts = options or ExportOptions()

    if format == ReceiptExportFormat.JSON:
        bundle = {
            "bundle_type": "decision_receipts",
            "exported_at": datetime.now().isoformat(),
            "count": len(receipts),
            "receipts": [receipt.to_dict() for receipt in receipts],
        }
        return json.dumps(bundle, indent=opts.indent)

    elif format == ReceiptExportFormat.CSV:
        # Combine all vulnerabilities from all receipts
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "Receipt ID",
            "Gauntlet ID",
            "Verdict",
            "Finding ID",
            "Severity",
            "Category",
            "Title",
        ])

        for receipt in receipts:
            for vuln in receipt.vulnerability_details:
                writer.writerow([
                    receipt.receipt_id,
                    receipt.gauntlet_id,
                    receipt.verdict,
                    vuln.get("id", ""),
                    vuln.get("severity", ""),
                    vuln.get("category", ""),
                    vuln.get("title", ""),
                ])

        return output.getvalue()

    else:
        # Concatenate individual exports
        return "\n\n---\n\n".join(
            export_receipt(receipt, format, opts) for receipt in receipts
        )


# =============================================================================
# Streaming Export
# =============================================================================


def stream_receipt_json(
    receipt: "DecisionReceipt",
    chunk_size: int = 4096,
) -> Any:
    """
    Stream a receipt as JSON for large exports.

    Yields chunks of JSON suitable for streaming responses.

    Args:
        receipt: Receipt to export
        chunk_size: Size of each chunk

    Yields:
        String chunks of JSON
    """
    full_json = receipt.to_json()
    for i in range(0, len(full_json), chunk_size):
        yield full_json[i:i + chunk_size]


__all__ = [
    # Enums
    "ReceiptExportFormat",
    "HeatmapExportFormat",
    # Classes
    "ExportOptions",
    # Functions
    "export_receipt",
    "export_heatmap",
    "export_receipts_bundle",
    "stream_receipt_json",
]
