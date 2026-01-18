"""
Industry Workflow Templates.

Pre-built workflow templates for common enterprise use cases:
- Legal: Contract review, due diligence, compliance audit
- Healthcare: HIPAA assessment, medical record review
- Code: Security audit, architecture review, code quality
- Accounting: Financial audit, tax compliance
"""

from aragora.workflow.templates.legal import (
    CONTRACT_REVIEW_TEMPLATE,
    DUE_DILIGENCE_TEMPLATE,
    COMPLIANCE_AUDIT_TEMPLATE,
)
from aragora.workflow.templates.healthcare import (
    HIPAA_ASSESSMENT_TEMPLATE,
    CLINICAL_REVIEW_TEMPLATE,
    PHI_AUDIT_TEMPLATE,
)
from aragora.workflow.templates.code import (
    SECURITY_AUDIT_TEMPLATE,
    ARCHITECTURE_REVIEW_TEMPLATE,
    CODE_QUALITY_TEMPLATE,
)
from aragora.workflow.templates.accounting import (
    FINANCIAL_AUDIT_TEMPLATE,
    SOX_COMPLIANCE_TEMPLATE,
)

# Template registry
WORKFLOW_TEMPLATES = {
    # Legal
    "legal/contract-review": CONTRACT_REVIEW_TEMPLATE,
    "legal/due-diligence": DUE_DILIGENCE_TEMPLATE,
    "legal/compliance-audit": COMPLIANCE_AUDIT_TEMPLATE,
    # Healthcare
    "healthcare/hipaa-assessment": HIPAA_ASSESSMENT_TEMPLATE,
    "healthcare/clinical-review": CLINICAL_REVIEW_TEMPLATE,
    "healthcare/phi-audit": PHI_AUDIT_TEMPLATE,
    # Code
    "code/security-audit": SECURITY_AUDIT_TEMPLATE,
    "code/architecture-review": ARCHITECTURE_REVIEW_TEMPLATE,
    "code/quality-review": CODE_QUALITY_TEMPLATE,
    # Accounting
    "accounting/financial-audit": FINANCIAL_AUDIT_TEMPLATE,
    "accounting/sox-compliance": SOX_COMPLIANCE_TEMPLATE,
}


def get_template(template_id: str) -> dict:
    """Get a workflow template by ID."""
    return WORKFLOW_TEMPLATES.get(template_id)


def list_templates(category: str = None) -> list:
    """List available templates, optionally filtered by category."""
    templates = []
    for template_id, template in WORKFLOW_TEMPLATES.items():
        if category and not template_id.startswith(category):
            continue
        templates.append({
            "id": template_id,
            "name": template.get("name", template_id),
            "description": template.get("description", ""),
            "category": template_id.split("/")[0],
        })
    return templates


__all__ = [
    "WORKFLOW_TEMPLATES",
    "get_template",
    "list_templates",
    # Legal
    "CONTRACT_REVIEW_TEMPLATE",
    "DUE_DILIGENCE_TEMPLATE",
    "COMPLIANCE_AUDIT_TEMPLATE",
    # Healthcare
    "HIPAA_ASSESSMENT_TEMPLATE",
    "CLINICAL_REVIEW_TEMPLATE",
    "PHI_AUDIT_TEMPLATE",
    # Code
    "SECURITY_AUDIT_TEMPLATE",
    "ARCHITECTURE_REVIEW_TEMPLATE",
    "CODE_QUALITY_TEMPLATE",
    # Accounting
    "FINANCIAL_AUDIT_TEMPLATE",
    "SOX_COMPLIANCE_TEMPLATE",
]
