"""
Workflow Template Catalog.

Curated vertical-specific playbook templates that wire together
Aragora's debate engine, compliance checks, and decision receipts
for regulated industries.

Verticals:
- Healthcare: HIPAA/HITECH clinical decision playbooks
- Financial: SOX/Basel III/MiFID II regulatory decision playbooks
- Legal: Precedent-driven analysis with privilege preservation
"""

from __future__ import annotations

from aragora.workflow.templates.catalog.verticals import (
    # Healthcare
    HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
    # Financial
    FINANCIAL_REGULATORY_DECISION_TEMPLATE,
    # Legal
    LEGAL_ANALYSIS_DECISION_TEMPLATE,
    # Registry
    VERTICAL_TEMPLATES,
)

__all__ = [
    "HEALTHCARE_CLINICAL_DECISION_TEMPLATE",
    "FINANCIAL_REGULATORY_DECISION_TEMPLATE",
    "LEGAL_ANALYSIS_DECISION_TEMPLATE",
    "VERTICAL_TEMPLATES",
]
