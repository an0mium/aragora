"""
Built-in deliberation templates.

These templates cover common use cases across different verticals:
- Code: Code review, security audit, architecture decisions
- Legal: Contract review, due diligence, compliance
- Finance: Financial audit, risk assessment
- Healthcare: HIPAA compliance, clinical guidelines
- Compliance: SOC 2, GDPR, regulatory assessments
- Academic: Citation verification, peer review
- General: Research, quick decisions
"""

from aragora.deliberation.templates.base import (
    DeliberationTemplate,
    OutputFormat,
    TeamStrategy,
    TemplateCategory,
)


# =============================================================================
# Code Templates
# =============================================================================

CODE_REVIEW = DeliberationTemplate(
    name="code_review",
    description="Multi-agent code review with security, performance, and maintainability focus",
    category=TemplateCategory.CODE,
    default_agents=["anthropic-api", "openai-api", "codestral"],
    team_strategy=TeamStrategy.BEST_FOR_DOMAIN,
    default_knowledge_sources=["github:pr"],
    output_format=OutputFormat.GITHUB_REVIEW,
    consensus_threshold=0.7,
    max_rounds=3,
    personas=["security", "performance", "maintainability", "readability"],
    tags=["code", "review", "quality", "security", "best-practices"],
)

SECURITY_AUDIT = DeliberationTemplate(
    name="security_audit",
    description="Comprehensive security audit for applications and infrastructure",
    category=TemplateCategory.CODE,
    default_agents=["anthropic-api", "openai-api", "gemini"],
    team_strategy=TeamStrategy.DIVERSE,
    default_knowledge_sources=["github:repo"],
    output_format=OutputFormat.DECISION_RECEIPT,
    consensus_threshold=0.8,
    max_rounds=5,
    personas=["owasp", "penetration", "threat-modeling", "compliance"],
    tags=["security", "audit", "vulnerabilities", "owasp", "penetration"],
)

ARCHITECTURE_DECISION = DeliberationTemplate(
    name="architecture_decision",
    description="Technical architecture decision with trade-off analysis",
    category=TemplateCategory.CODE,
    default_agents=["anthropic-api", "openai-api", "gemini", "mistral", "deepseek"],
    team_strategy=TeamStrategy.DIVERSE,
    default_knowledge_sources=["confluence", "jira", "github"],
    output_format=OutputFormat.DECISION_RECEIPT,
    consensus_threshold=0.6,
    max_rounds=5,
    personas=["architect", "security", "scalability", "cost", "operations"],
    tags=["architecture", "decision", "trade-offs", "design"],
)


# =============================================================================
# Legal Templates
# =============================================================================

CONTRACT_REVIEW = DeliberationTemplate(
    name="contract_review",
    description="Legal contract analysis with risk assessment and compliance verification",
    category=TemplateCategory.LEGAL,
    default_agents=["anthropic-api", "openai-api", "gemini"],
    team_strategy=TeamStrategy.BEST_FOR_DOMAIN,
    default_knowledge_sources=["confluence", "sharepoint"],
    output_format=OutputFormat.DECISION_RECEIPT,
    consensus_threshold=0.8,
    max_rounds=5,
    personas=["legal", "compliance", "risk", "commercial"],
    tags=["legal", "contracts", "review", "compliance", "risk-assessment"],
)

DUE_DILIGENCE = DeliberationTemplate(
    name="due_diligence",
    description="Comprehensive due diligence for M&A and investment decisions",
    category=TemplateCategory.LEGAL,
    default_agents=["anthropic-api", "openai-api", "gemini", "mistral"],
    team_strategy=TeamStrategy.DIVERSE,
    default_knowledge_sources=["confluence", "sharepoint", "financial_docs"],
    output_format=OutputFormat.DECISION_RECEIPT,
    consensus_threshold=0.75,
    max_rounds=7,
    personas=["legal", "financial", "operational", "risk"],
    tags=["legal", "due-diligence", "m&a", "investment", "risk"],
)


# =============================================================================
# Finance Templates
# =============================================================================

FINANCIAL_AUDIT = DeliberationTemplate(
    name="financial_audit",
    description="Financial statement audit with multi-agent verification",
    category=TemplateCategory.FINANCE,
    default_agents=["anthropic-api", "openai-api", "gemini"],
    team_strategy=TeamStrategy.BEST_FOR_DOMAIN,
    default_knowledge_sources=["financial_statements"],
    output_format=OutputFormat.COMPLIANCE_REPORT,
    consensus_threshold=0.85,
    max_rounds=5,
    personas=["auditor", "forensic", "compliance", "tax"],
    tags=["finance", "audit", "accounting", "compliance", "statements"],
)

RISK_ASSESSMENT = DeliberationTemplate(
    name="risk_assessment",
    description="Enterprise risk assessment and mitigation planning",
    category=TemplateCategory.FINANCE,
    default_agents=["anthropic-api", "openai-api"],
    team_strategy=TeamStrategy.BEST_FOR_DOMAIN,
    output_format=OutputFormat.DECISION_RECEIPT,
    consensus_threshold=0.7,
    max_rounds=4,
    personas=["risk", "compliance", "operations", "strategic"],
    tags=["risk", "assessment", "mitigation", "enterprise"],
)


# =============================================================================
# Healthcare Templates
# =============================================================================

HIPAA_COMPLIANCE = DeliberationTemplate(
    name="hipaa_compliance",
    description="HIPAA compliance verification for healthcare data handling",
    category=TemplateCategory.HEALTHCARE,
    default_agents=["anthropic-api", "openai-api"],
    team_strategy=TeamStrategy.BEST_FOR_DOMAIN,
    default_knowledge_sources=["policy_docs"],
    output_format=OutputFormat.COMPLIANCE_REPORT,
    consensus_threshold=0.9,
    max_rounds=4,
    personas=["hipaa", "privacy", "security", "compliance"],
    tags=["healthcare", "hipaa", "compliance", "privacy", "phi"],
)

CLINICAL_REVIEW = DeliberationTemplate(
    name="clinical_review",
    description="Clinical guidelines review against evidence-based medicine",
    category=TemplateCategory.HEALTHCARE,
    default_agents=["anthropic-api", "openai-api", "gemini"],
    team_strategy=TeamStrategy.DIVERSE,
    default_knowledge_sources=["medical_literature"],
    output_format=OutputFormat.DECISION_RECEIPT,
    consensus_threshold=0.8,
    max_rounds=5,
    personas=["clinical", "evidence-based", "specialist", "guidelines"],
    tags=["healthcare", "clinical", "guidelines", "evidence", "treatment"],
)


# =============================================================================
# Compliance Templates
# =============================================================================

COMPLIANCE_CHECK = DeliberationTemplate(
    name="compliance_check",
    description="Regulatory compliance assessment for various frameworks",
    category=TemplateCategory.COMPLIANCE,
    default_agents=["anthropic-api", "openai-api"],
    team_strategy=TeamStrategy.BEST_FOR_DOMAIN,
    default_knowledge_sources=["policy_docs"],
    output_format=OutputFormat.COMPLIANCE_REPORT,
    consensus_threshold=0.9,
    max_rounds=3,
    personas=["sox", "gdpr", "hipaa", "pci-dss"],
    tags=["compliance", "regulatory", "assessment", "soc2", "gdpr"],
)

SOC2_AUDIT = DeliberationTemplate(
    name="soc2_audit",
    description="SOC 2 compliance audit preparation and gap analysis",
    category=TemplateCategory.COMPLIANCE,
    default_agents=["anthropic-api", "openai-api"],
    team_strategy=TeamStrategy.BEST_FOR_DOMAIN,
    default_knowledge_sources=["policy_docs", "procedures"],
    output_format=OutputFormat.COMPLIANCE_REPORT,
    consensus_threshold=0.85,
    max_rounds=5,
    personas=["security", "availability", "processing-integrity", "confidentiality", "privacy"],
    tags=["compliance", "soc2", "audit", "trust-services"],
)

GDPR_ASSESSMENT = DeliberationTemplate(
    name="gdpr_assessment",
    description="GDPR compliance assessment and data protection review",
    category=TemplateCategory.COMPLIANCE,
    default_agents=["anthropic-api", "openai-api"],
    team_strategy=TeamStrategy.BEST_FOR_DOMAIN,
    default_knowledge_sources=["policy_docs", "data_inventory"],
    output_format=OutputFormat.COMPLIANCE_REPORT,
    consensus_threshold=0.85,
    max_rounds=4,
    personas=["privacy", "dpo", "legal", "technical"],
    tags=["compliance", "gdpr", "privacy", "data-protection", "eu"],
)


# =============================================================================
# Academic Templates
# =============================================================================

CITATION_VERIFICATION = DeliberationTemplate(
    name="citation_verification",
    description="Verify academic citations and references for accuracy",
    category=TemplateCategory.ACADEMIC,
    default_agents=["anthropic-api", "openai-api"],
    team_strategy=TeamStrategy.FAST,
    output_format=OutputFormat.STANDARD,
    consensus_threshold=0.7,
    max_rounds=3,
    personas=["verification", "accuracy", "formatting"],
    tags=["academic", "citations", "verification", "research", "references"],
)

PEER_REVIEW = DeliberationTemplate(
    name="peer_review",
    description="Multi-agent academic peer review simulation",
    category=TemplateCategory.ACADEMIC,
    default_agents=["anthropic-api", "openai-api", "gemini"],
    team_strategy=TeamStrategy.DIVERSE,
    output_format=OutputFormat.DECISION_RECEIPT,
    consensus_threshold=0.6,
    max_rounds=5,
    personas=["methodology", "significance", "clarity", "reproducibility"],
    tags=["academic", "peer-review", "research", "methodology"],
)


# =============================================================================
# General Templates
# =============================================================================

QUICK_DECISION = DeliberationTemplate(
    name="quick_decision",
    description="Fast decision with minimal agents for simple questions",
    category=TemplateCategory.GENERAL,
    default_agents=["anthropic-api", "openai-api"],
    team_strategy=TeamStrategy.FAST,
    output_format=OutputFormat.SUMMARY,
    consensus_threshold=0.5,
    max_rounds=2,
    require_consensus=False,
    timeout_seconds=60.0,
    tags=["quick", "fast", "simple", "decision"],
)

RESEARCH_ANALYSIS = DeliberationTemplate(
    name="research_analysis",
    description="General-purpose research and analysis workflow",
    category=TemplateCategory.GENERAL,
    default_agents=["anthropic-api", "openai-api", "gemini"],
    team_strategy=TeamStrategy.DIVERSE,
    output_format=OutputFormat.DECISION_RECEIPT,
    consensus_threshold=0.6,
    max_rounds=5,
    personas=["research", "analysis", "synthesis"],
    tags=["research", "analysis", "general", "investigation"],
)

BRAINSTORM = DeliberationTemplate(
    name="brainstorm",
    description="Creative brainstorming with diverse agent perspectives",
    category=TemplateCategory.GENERAL,
    default_agents=["anthropic-api", "openai-api", "gemini", "mistral", "deepseek"],
    team_strategy=TeamStrategy.DIVERSE,
    output_format=OutputFormat.STANDARD,
    consensus_threshold=0.3,  # Low threshold - want diverse ideas
    max_rounds=4,
    require_consensus=False,
    personas=["creative", "practical", "critical", "visionary"],
    tags=["brainstorm", "creative", "ideas", "innovation"],
)


# =============================================================================
# Built-in Templates Dictionary
# =============================================================================

BUILTIN_TEMPLATES = {
    # Code
    "code_review": CODE_REVIEW,
    "security_audit": SECURITY_AUDIT,
    "architecture_decision": ARCHITECTURE_DECISION,
    # Legal
    "contract_review": CONTRACT_REVIEW,
    "due_diligence": DUE_DILIGENCE,
    # Finance
    "financial_audit": FINANCIAL_AUDIT,
    "risk_assessment": RISK_ASSESSMENT,
    # Healthcare
    "hipaa_compliance": HIPAA_COMPLIANCE,
    "clinical_review": CLINICAL_REVIEW,
    # Compliance
    "compliance_check": COMPLIANCE_CHECK,
    "soc2_audit": SOC2_AUDIT,
    "gdpr_assessment": GDPR_ASSESSMENT,
    # Academic
    "citation_verification": CITATION_VERIFICATION,
    "peer_review": PEER_REVIEW,
    # General
    "quick_decision": QUICK_DECISION,
    "research_analysis": RESEARCH_ANALYSIS,
    "brainstorm": BRAINSTORM,
}
