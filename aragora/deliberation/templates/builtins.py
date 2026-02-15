"""
Built-in deliberation templates.

These templates cover common use cases across different verticals:
- Code: Code review, security audit, architecture decisions
- Legal: Contract review, due diligence, compliance
- Finance: Financial audit, risk assessment
- Healthcare: HIPAA compliance, clinical guidelines
- Compliance: SOC 2, GDPR, regulatory assessments
- Academic: Citation verification, peer review
- Business: Hiring, vendor evaluation, budget allocation, strategy
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
    example_topics=[
        "Review this pull request for security and performance issues",
        "Check this Python module for code quality and best practices",
        "Audit this API endpoint handler for common vulnerabilities",
    ],
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
    example_topics=[
        "Perform a security audit of our web application",
        "Check our infrastructure for OWASP Top 10 vulnerabilities",
        "Assess the security posture of our API gateway",
    ],
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
    example_topics=[
        "Should we migrate from monolith to microservices?",
        "Evaluate database options: PostgreSQL vs DynamoDB for our use case",
        "Design the architecture for a real-time event processing pipeline",
    ],
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
    example_topics=[
        "Review this SaaS vendor contract for unfavorable terms",
        "Analyze the liability clauses in our partnership agreement",
        "Check this NDA for compliance with our data protection policy",
    ],
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
    example_topics=[
        "Evaluate the risks of acquiring this startup",
        "Conduct due diligence on a potential Series B investment",
        "Assess the legal and financial health of a merger target",
    ],
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
    example_topics=[
        "Audit Q3 financial statements for material misstatements",
        "Review revenue recognition practices for compliance",
        "Verify the accuracy of our annual report disclosures",
    ],
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
    example_topics=[
        "Assess cybersecurity risks for our cloud infrastructure",
        "Evaluate supply chain risks for Q4",
        "Identify and prioritize operational risks in our expansion plan",
    ],
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
    example_topics=[
        "Verify our patient data handling meets HIPAA requirements",
        "Audit PHI access controls in our EHR system",
        "Assess HIPAA compliance of our telehealth platform",
    ],
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
    example_topics=[
        "Review treatment guidelines for Type 2 diabetes management",
        "Evaluate clinical evidence for a new cardiac intervention",
        "Assess whether this clinical protocol aligns with current best practices",
    ],
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
    example_topics=[
        "Check if our data processing meets regulatory requirements",
        "Assess compliance gaps before our upcoming audit",
        "Evaluate our SOX controls for financial reporting",
    ],
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
    example_topics=[
        "Prepare for our SOC 2 Type II audit next quarter",
        "Identify gaps in our Trust Services Criteria controls",
        "Review our SOC 2 evidence collection process",
    ],
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
    example_topics=[
        "Assess GDPR compliance for our customer data pipeline",
        "Review our data retention policies against GDPR Article 17",
        "Evaluate our consent management implementation",
    ],
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
    example_topics=[
        "Verify the citations in this research paper draft",
        "Check reference accuracy for our literature review",
        "Validate DOIs and publication details in this bibliography",
    ],
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
    example_topics=[
        "Peer review this machine learning paper submission",
        "Evaluate the methodology of this clinical trial report",
        "Assess the statistical analysis in this research manuscript",
    ],
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
    example_topics=[
        "Should we use Slack or Teams for internal communication?",
        "Is this feature worth building for our next sprint?",
        "Which cloud provider should we pick for this microservice?",
    ],
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
    example_topics=[
        "Research the competitive landscape for AI code review tools",
        "Analyze the impact of remote work on team productivity",
        "Investigate best practices for API versioning strategies",
    ],
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
    example_topics=[
        "Brainstorm new features for our mobile app",
        "Generate creative marketing campaign ideas for Q2",
        "Ideate solutions for improving customer onboarding",
    ],
)


# =============================================================================
# Productivity Templates
# =============================================================================

EMAIL_PRIORITIZATION = DeliberationTemplate(
    name="email_prioritization",
    description="Multi-agent email triage to identify urgent, important, and actionable messages",
    category=TemplateCategory.GENERAL,
    default_agents=["anthropic-api", "openai-api"],
    team_strategy=TeamStrategy.FAST,
    default_knowledge_sources=["gmail", "calendar", "slack"],
    output_format=OutputFormat.SUMMARY,
    consensus_threshold=0.7,
    max_rounds=2,
    require_consensus=True,
    timeout_seconds=30.0,
    personas=["urgency", "relevance", "action-required", "sender-importance"],
    tags=["email", "inbox", "triage", "priority", "productivity"],
    example_topics=[
        "Prioritize my inbox for today's most urgent items",
        "Which of these emails need a response before end of day?",
        "Triage my unread emails by urgency and importance",
    ],
)

INBOX_TRIAGE = DeliberationTemplate(
    name="inbox_triage",
    description="Batch email categorization and folder assignment with spam detection",
    category=TemplateCategory.GENERAL,
    default_agents=["anthropic-api", "openai-api", "gemini"],
    team_strategy=TeamStrategy.FAST,
    default_knowledge_sources=["gmail", "contacts"],
    output_format=OutputFormat.STANDARD,
    consensus_threshold=0.6,
    max_rounds=2,
    require_consensus=False,
    timeout_seconds=60.0,
    personas=["categorization", "spam-detection", "relevance", "urgency"],
    tags=["email", "inbox", "categorization", "folders", "spam"],
    example_topics=[
        "Sort my inbox into categories: urgent, follow-up, FYI, spam",
        "Categorize these 50 unread emails into appropriate folders",
        "Detect and filter spam from my business inbox",
    ],
)

MEETING_PREP = DeliberationTemplate(
    name="meeting_prep",
    description="Prepare for meetings by analyzing relevant emails, docs, and context",
    category=TemplateCategory.GENERAL,
    default_agents=["anthropic-api", "openai-api"],
    team_strategy=TeamStrategy.FAST,
    default_knowledge_sources=["calendar", "gmail", "confluence", "slack"],
    output_format=OutputFormat.SUMMARY,
    consensus_threshold=0.6,
    max_rounds=2,
    personas=["context", "key-points", "action-items", "participants"],
    tags=["meeting", "preparation", "context", "productivity"],
    example_topics=[
        "Prepare me for tomorrow's board meeting",
        "Summarize context and key topics for my 1:1 with the CTO",
        "Gather relevant docs and action items for the sprint retrospective",
    ],
)


# =============================================================================
# Business Templates
# =============================================================================

HIRING_DECISION = DeliberationTemplate(
    name="hiring_decision",
    description="Evaluate a hiring decision with multiple agent perspectives",
    category=TemplateCategory.BUSINESS,
    default_agents=["anthropic-api", "openai-api", "gemini"],
    team_strategy=TeamStrategy.DIVERSE,
    output_format=OutputFormat.DECISION_RECEIPT,
    consensus_threshold=0.7,
    max_rounds=4,
    personas=["talent", "culture-fit", "skills", "compensation"],
    tags=["business", "hiring", "hr", "talent", "recruitment"],
    example_topics=[
        "Should we hire this VP of Engineering candidate?",
        "Evaluate the final two candidates for the product manager role",
        "Assess culture fit and technical skills for a senior hire",
    ],
)

VENDOR_EVALUATION = DeliberationTemplate(
    name="vendor_evaluation",
    description="Compare vendors for a business need",
    category=TemplateCategory.BUSINESS,
    default_agents=["anthropic-api", "openai-api"],
    team_strategy=TeamStrategy.BEST_FOR_DOMAIN,
    output_format=OutputFormat.DECISION_RECEIPT,
    consensus_threshold=0.7,
    max_rounds=4,
    personas=["cost", "quality", "reliability", "support"],
    tags=["business", "vendor", "procurement", "evaluation", "comparison"],
    example_topics=[
        "Compare AWS vs GCP vs Azure for our infrastructure needs",
        "Evaluate CRM vendors: Salesforce vs HubSpot vs Pipedrive",
        "Assess three payment processing vendors for our e-commerce platform",
    ],
)

BUDGET_ALLOCATION = DeliberationTemplate(
    name="budget_allocation",
    description="Debate budget allocation across departments or projects",
    category=TemplateCategory.BUSINESS,
    default_agents=["anthropic-api", "openai-api", "gemini"],
    team_strategy=TeamStrategy.DIVERSE,
    output_format=OutputFormat.DECISION_RECEIPT,
    consensus_threshold=0.7,
    max_rounds=5,
    personas=["finance", "growth", "operations", "risk"],
    tags=["business", "budget", "allocation", "finance", "planning"],
    example_topics=[
        "Allocate next year's engineering budget across teams",
        "Decide how to split marketing budget between channels",
        "Prioritize R&D spending across three product lines",
    ],
)

TOOL_SELECTION = DeliberationTemplate(
    name="tool_selection",
    description="Evaluate and select the best tool for a business need",
    category=TemplateCategory.BUSINESS,
    default_agents=["anthropic-api", "openai-api"],
    team_strategy=TeamStrategy.BEST_FOR_DOMAIN,
    output_format=OutputFormat.SUMMARY,
    consensus_threshold=0.6,
    max_rounds=3,
    personas=["technical", "usability", "cost", "integration"],
    tags=["business", "tools", "selection", "evaluation", "software"],
    example_topics=[
        "Select a project management tool for our team",
        "Choose between Jira, Linear, and Shortcut for issue tracking",
        "Evaluate CI/CD tools: GitHub Actions vs CircleCI vs Jenkins",
    ],
)

PERFORMANCE_REVIEW = DeliberationTemplate(
    name="performance_review",
    description="Structure performance review with multiple perspectives",
    category=TemplateCategory.BUSINESS,
    default_agents=["anthropic-api", "openai-api"],
    team_strategy=TeamStrategy.DIVERSE,
    output_format=OutputFormat.DECISION_RECEIPT,
    consensus_threshold=0.7,
    max_rounds=3,
    personas=["management", "peer", "self-assessment", "goals"],
    tags=["business", "performance", "review", "hr", "assessment"],
    example_topics=[
        "Structure a fair performance review for a senior engineer",
        "Evaluate team member contributions from multiple perspectives",
        "Assess goal completion and set objectives for next quarter",
    ],
)

STRATEGIC_PLANNING = DeliberationTemplate(
    name="strategic_planning",
    description="Evaluate strategic options for business direction",
    category=TemplateCategory.BUSINESS,
    default_agents=["anthropic-api", "openai-api", "gemini", "mistral"],
    team_strategy=TeamStrategy.DIVERSE,
    output_format=OutputFormat.DECISION_RECEIPT,
    consensus_threshold=0.65,
    max_rounds=5,
    personas=["strategy", "market", "operations", "finance", "innovation"],
    tags=["business", "strategy", "planning", "direction", "growth"],
    example_topics=[
        "Should we expand into the European market this year?",
        "Evaluate strategic options for our product roadmap",
        "Plan our go-to-market strategy for the enterprise segment",
    ],
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
    # Productivity
    "email_prioritization": EMAIL_PRIORITIZATION,
    "inbox_triage": INBOX_TRIAGE,
    "meeting_prep": MEETING_PREP,
    # Business
    "hiring_decision": HIRING_DECISION,
    "vendor_evaluation": VENDOR_EVALUATION,
    "budget_allocation": BUDGET_ALLOCATION,
    "tool_selection": TOOL_SELECTION,
    "performance_review": PERFORMANCE_REVIEW,
    "strategic_planning": STRATEGIC_PLANNING,
}
