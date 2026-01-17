"""
Audit reporting modules for document analysis.

Provides structured output formats for audit findings:
- DefectReport: Comprehensive defect analysis
- RiskHeatmap: Visual risk assessment
- DiscrepancyMap: Cross-document discrepancy visualization
"""

from aragora.audit.reports.defect_report import (
    DefectReport,
    ReportConfig,
    ReportFormat,
    generate_report,
)

__all__ = [
    "DefectReport",
    "ReportConfig",
    "ReportFormat",
    "generate_report",
]
