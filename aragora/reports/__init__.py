"""
Report Generation Module.

Provides report generation for audit sessions:
- PDF reports for executive summaries
- Markdown reports for documentation
- JSON reports for machine processing
- HTML reports for web viewing

Usage:
    from aragora.reports import AuditReportGenerator, ReportFormat

    generator = AuditReportGenerator()
    report = await generator.generate(session_id, format=ReportFormat.PDF)
"""

from aragora.reports.generator import (
    AuditReportGenerator,
    ReportConfig,
    ReportFormat,
    ReportTemplate,
)

__all__ = [
    "AuditReportGenerator",
    "ReportConfig",
    "ReportFormat",
    "ReportTemplate",
]
