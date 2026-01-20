"""
Gauntlet Stable API - v1.

Provides versioned, documented API for Gauntlet Decision Receipts.

This module defines:
- JSON Schema for DecisionReceipt and RiskHeatmap
- OpenAPI-compatible endpoint handlers
- Vertical audit templates (compliance, security, legal, financial)
- Error response standards (RFC 7807)
- Multi-format export utilities (JSON, Markdown, HTML, CSV, SARIF)

Usage:
    from aragora.gauntlet.api import (
        # Schema validation
        get_receipt_schema,
        validate_receipt,
        DECISION_RECEIPT_SCHEMA,

        # Export
        ReceiptExportFormat,
        export_receipt,
        export_heatmap,

        # Templates
        AuditTemplate,
        COMPLIANCE_TEMPLATE,
        get_template,
        list_templates,
    )

Version: 1.0.0
"""

from aragora.gauntlet.api.schema import (
    # Schemas
    DECISION_RECEIPT_SCHEMA,
    RISK_HEATMAP_SCHEMA,
    PROBLEM_DETAIL_SCHEMA,
    PROVENANCE_RECORD_SCHEMA,
    CONSENSUS_PROOF_SCHEMA,
    RISK_SUMMARY_SCHEMA,
    VULNERABILITY_DETAIL_SCHEMA,
    HEATMAP_CELL_SCHEMA,
    # Utilities
    get_receipt_schema,
    get_heatmap_schema,
    validate_receipt,
    validate_heatmap,
    get_all_schemas,
    to_openapi_schema,
    # Constants
    SCHEMA_VERSION,
)
from aragora.gauntlet.api.templates import (
    # Classes
    AuditTemplate,
    TemplateSection,
    TemplateCategory,
    TemplateFormat,
    # Pre-built templates
    COMPLIANCE_TEMPLATE,
    SECURITY_TEMPLATE,
    LEGAL_TEMPLATE,
    FINANCIAL_TEMPLATE,
    OPERATIONAL_TEMPLATE,
    # Functions
    get_template,
    list_templates,
    register_template,
    create_custom_template,
)
from aragora.gauntlet.api.export import (
    # Enums
    ReceiptExportFormat,
    HeatmapExportFormat,
    # Classes
    ExportOptions,
    # Functions
    export_receipt,
    export_heatmap,
    export_receipts_bundle,
    stream_receipt_json,
)

__all__ = [
    # Schema - Core
    "DECISION_RECEIPT_SCHEMA",
    "RISK_HEATMAP_SCHEMA",
    "PROBLEM_DETAIL_SCHEMA",
    # Schema - Components
    "PROVENANCE_RECORD_SCHEMA",
    "CONSENSUS_PROOF_SCHEMA",
    "RISK_SUMMARY_SCHEMA",
    "VULNERABILITY_DETAIL_SCHEMA",
    "HEATMAP_CELL_SCHEMA",
    # Schema - Utilities
    "get_receipt_schema",
    "get_heatmap_schema",
    "validate_receipt",
    "validate_heatmap",
    "get_all_schemas",
    "to_openapi_schema",
    "SCHEMA_VERSION",
    # Templates - Classes
    "AuditTemplate",
    "TemplateSection",
    "TemplateCategory",
    "TemplateFormat",
    # Templates - Pre-built
    "COMPLIANCE_TEMPLATE",
    "SECURITY_TEMPLATE",
    "LEGAL_TEMPLATE",
    "FINANCIAL_TEMPLATE",
    "OPERATIONAL_TEMPLATE",
    # Templates - Functions
    "get_template",
    "list_templates",
    "register_template",
    "create_custom_template",
    # Export - Enums
    "ReceiptExportFormat",
    "HeatmapExportFormat",
    # Export - Classes
    "ExportOptions",
    # Export - Functions
    "export_receipt",
    "export_heatmap",
    "export_receipts_bundle",
    "stream_receipt_json",
]
