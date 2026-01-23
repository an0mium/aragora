"""
Industry Workflow Templates.

Pre-built workflow templates for common enterprise use cases:
- Legal: Contract review, due diligence, compliance audit
- Healthcare: HIPAA assessment, medical record review
- Code: Security audit, architecture review, code quality
- Accounting: Financial audit, tax compliance
- AI/ML: Model deployment, bias audit, AI governance
- DevOps: CI/CD review, incident response, infrastructure audit
- Product: PRD review, feature specs, launch readiness

Pattern-based workflow templates:
- HiveMind: Parallel agent execution with consensus merge
- MapReduce: Split work, parallel processing, aggregate results
- ReviewCycle: Iterative refinement with convergence check
"""

from typing import Any, Optional

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
from aragora.workflow.templates.ai_ml import (
    MODEL_DEPLOYMENT_TEMPLATE,
    AI_GOVERNANCE_TEMPLATE,
    PROMPT_ENGINEERING_TEMPLATE,
)
from aragora.workflow.templates.devops import (
    CICD_PIPELINE_REVIEW_TEMPLATE,
    INCIDENT_RESPONSE_TEMPLATE,
    INFRASTRUCTURE_AUDIT_TEMPLATE,
)
from aragora.workflow.templates.product import (
    PRD_REVIEW_TEMPLATE,
    FEATURE_SPEC_TEMPLATE,
    USER_RESEARCH_TEMPLATE,
    LAUNCH_READINESS_TEMPLATE,
)
from aragora.workflow.templates.patterns import (
    HIVE_MIND_TEMPLATE,
    MAP_REDUCE_TEMPLATE,
    REVIEW_CYCLE_TEMPLATE,
    PATTERN_TEMPLATES,
    create_hive_mind_workflow,
    create_map_reduce_workflow,
    create_review_cycle_workflow,
    get_pattern_template,
    list_pattern_templates,
)
from aragora.workflow.templates.marketing import (
    AD_PERFORMANCE_REVIEW_TEMPLATE,
    LEAD_TO_CRM_SYNC_TEMPLATE,
    CROSS_PLATFORM_ANALYTICS_TEMPLATE,
    SUPPORT_TICKET_TRIAGE_TEMPLATE,
    ECOMMERCE_ORDER_SYNC_TEMPLATE,
    MARKETING_TEMPLATES,
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
    # AI/ML
    "ai_ml/model-deployment": MODEL_DEPLOYMENT_TEMPLATE,
    "ai_ml/ai-governance": AI_GOVERNANCE_TEMPLATE,
    "ai_ml/prompt-engineering": PROMPT_ENGINEERING_TEMPLATE,
    # DevOps
    "devops/cicd-review": CICD_PIPELINE_REVIEW_TEMPLATE,
    "devops/incident-response": INCIDENT_RESPONSE_TEMPLATE,
    "devops/infrastructure-audit": INFRASTRUCTURE_AUDIT_TEMPLATE,
    # Product
    "product/prd-review": PRD_REVIEW_TEMPLATE,
    "product/feature-spec": FEATURE_SPEC_TEMPLATE,
    "product/user-research": USER_RESEARCH_TEMPLATE,
    "product/launch-readiness": LAUNCH_READINESS_TEMPLATE,
    # Marketing/Advertising
    "marketing/ad-performance-review": AD_PERFORMANCE_REVIEW_TEMPLATE,
    "marketing/lead-to-crm-sync": LEAD_TO_CRM_SYNC_TEMPLATE,
    "marketing/cross-platform-analytics": CROSS_PLATFORM_ANALYTICS_TEMPLATE,
    # Support
    "support/ticket-triage": SUPPORT_TICKET_TRIAGE_TEMPLATE,
    # E-commerce
    "ecommerce/order-sync": ECOMMERCE_ORDER_SYNC_TEMPLATE,
    # Note: Pattern templates (hive-mind, map-reduce, review-cycle) are not included
    # in WORKFLOW_TEMPLATES as they are factory patterns with a different schema.
    # Use PATTERN_TEMPLATES or create_*_workflow() functions instead.
}


def get_template(template_id: str) -> Optional[dict[str, Any]]:
    """Get a workflow template by ID."""
    return WORKFLOW_TEMPLATES.get(template_id)


def list_templates(category: Optional[str] = None) -> list[dict[str, Any]]:
    """List available templates, optionally filtered by category."""
    templates = []
    for template_id, template in WORKFLOW_TEMPLATES.items():
        if category and not template_id.startswith(category):
            continue
        templates.append(
            {
                "id": template_id,
                "name": template.get("name", template_id),
                "description": template.get("description", ""),
                "category": template_id.split("/")[0],
            }
        )
    return templates


# Export packaging utilities
from aragora.workflow.templates.package import (
    TemplatePackage,
    TemplateMetadata,
    TemplateAuthor,
    TemplateDependency,
    TemplateStatus,
    TemplateCategory,
    create_package,
    package_all_templates,
    register_package,
    get_package,
    list_packages,
)

__all__ = [
    "WORKFLOW_TEMPLATES",
    "get_template",
    "list_templates",
    # Packaging
    "TemplatePackage",
    "TemplateMetadata",
    "TemplateAuthor",
    "TemplateDependency",
    "TemplateStatus",
    "TemplateCategory",
    "create_package",
    "package_all_templates",
    "register_package",
    "get_package",
    "list_packages",
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
    # AI/ML
    "MODEL_DEPLOYMENT_TEMPLATE",
    "AI_GOVERNANCE_TEMPLATE",
    "PROMPT_ENGINEERING_TEMPLATE",
    # DevOps
    "CICD_PIPELINE_REVIEW_TEMPLATE",
    "INCIDENT_RESPONSE_TEMPLATE",
    "INFRASTRUCTURE_AUDIT_TEMPLATE",
    # Product
    "PRD_REVIEW_TEMPLATE",
    "FEATURE_SPEC_TEMPLATE",
    "USER_RESEARCH_TEMPLATE",
    "LAUNCH_READINESS_TEMPLATE",
    # Patterns
    "HIVE_MIND_TEMPLATE",
    "MAP_REDUCE_TEMPLATE",
    "REVIEW_CYCLE_TEMPLATE",
    "PATTERN_TEMPLATES",
    "create_hive_mind_workflow",
    "create_map_reduce_workflow",
    "create_review_cycle_workflow",
    "get_pattern_template",
    "list_pattern_templates",
    # Marketing
    "AD_PERFORMANCE_REVIEW_TEMPLATE",
    "LEAD_TO_CRM_SYNC_TEMPLATE",
    "CROSS_PLATFORM_ANALYTICS_TEMPLATE",
    "SUPPORT_TICKET_TRIAGE_TEMPLATE",
    "ECOMMERCE_ORDER_SYNC_TEMPLATE",
    "MARKETING_TEMPLATES",
]
