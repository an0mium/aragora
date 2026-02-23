"""
Vertical-Specific Playbook Templates.

Industry-specific workflow templates that combine compliance checks,
evidence grading, and decision receipts for regulated verticals.

Each template follows the standard WorkflowTemplate dict schema:
- name, description, category, version, tags
- compliance_frameworks: list of regulatory frameworks enforced
- required_agent_types: agent types the template expects
- steps: list of workflow step dicts
- transitions: list of transition dicts
- output_format: description of the template's output structure
- metadata: author, version, estimated duration, etc.
"""

from __future__ import annotations

from typing import Any

from aragora.workflow.templates.catalog.verticals.healthcare import (
    HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
)
from aragora.workflow.templates.catalog.verticals.financial import (
    FINANCIAL_REGULATORY_DECISION_TEMPLATE,
)
from aragora.workflow.templates.catalog.verticals.legal import (
    LEGAL_ANALYSIS_DECISION_TEMPLATE,
)

VERTICAL_TEMPLATES: dict[str, dict[str, Any]] = {
    "vertical/healthcare-clinical-decision": HEALTHCARE_CLINICAL_DECISION_TEMPLATE,
    "vertical/financial-regulatory-decision": FINANCIAL_REGULATORY_DECISION_TEMPLATE,
    "vertical/legal-analysis-decision": LEGAL_ANALYSIS_DECISION_TEMPLATE,
}

__all__ = [
    "HEALTHCARE_CLINICAL_DECISION_TEMPLATE",
    "FINANCIAL_REGULATORY_DECISION_TEMPLATE",
    "LEGAL_ANALYSIS_DECISION_TEMPLATE",
    "VERTICAL_TEMPLATES",
]
