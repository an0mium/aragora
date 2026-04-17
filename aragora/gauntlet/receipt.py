"""
Decision Receipt - Audit-ready output format.

Provides a tamper-evident, comprehensive record of a Gauntlet validation
suitable for compliance, audit trails, and decision documentation.

This module is a thin facade that re-exports from focused submodules:
- receipt_models.py: Core dataclasses (ProvenanceRecord, ConsensusProof,
  DecisionReceipt, CruxReceipt)
- receipt_exporters.py: Export functions (markdown, HTML, SARIF, CSV,
  crux_receipt_to_markdown)
"""

from __future__ import annotations

# Re-export data models
from .receipt_models import (
    ConsensusProof,
    CruxReceipt,
    DecisionReceipt,
    ProvenanceRecord,
    build_crux_receipt,
)

# Re-export export functions
from .receipt_exporters import (
    crux_receipt_to_markdown,
    receipt_to_csv,
    receipt_to_html,
    receipt_to_html_paginated,
    receipt_to_markdown,
    receipt_to_sarif,
)

__all__ = [
    "ConsensusProof",
    "CruxReceipt",
    "DecisionReceipt",
    "ProvenanceRecord",
    "build_crux_receipt",
    "crux_receipt_to_markdown",
    "receipt_to_csv",
    "receipt_to_html",
    "receipt_to_html_paginated",
    "receipt_to_markdown",
    "receipt_to_sarif",
]
