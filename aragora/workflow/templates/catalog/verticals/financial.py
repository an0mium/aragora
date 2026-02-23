"""
Financial Services Vertical Playbook Template.

SOX/regulatory decision template with:
- Risk assessment stages (market, credit, operational)
- Regulatory compliance checks (SOX, Basel III, MiFID II)
- Audit trail generation with decision receipts
- Segregation of duties enforcement

This is a *playbook* template -- a complete decision workflow that wires
together debate, compliance, risk assessment, and receipt generation for
financial regulatory decisions.
"""

from __future__ import annotations

from typing import Any

FINANCIAL_REGULATORY_DECISION_TEMPLATE: dict[str, Any] = {
    "name": "Financial Regulatory Decision Playbook",
    "description": (
        "End-to-end financial decision workflow with SOX/Basel III/MiFID II "
        "compliance, multi-dimensional risk assessment (market, credit, "
        "operational), segregation of duties enforcement, and audit-ready "
        "decision receipt generation."
    ),
    "category": "finance",
    "version": "1.0",
    "tags": [
        "finance",
        "regulatory",
        "sox",
        "basel-iii",
        "mifid-ii",
        "risk-assessment",
        "segregation-of-duties",
        "decision-receipt",
        "playbook",
    ],
    "compliance_frameworks": ["SOX", "BASEL_III", "MIFID_II", "GAAP", "IFRS"],
    "required_agent_types": [
        "financial_auditor",
        "compliance_officer",
        "sox",
        "internal_auditor",
        "forensic_accountant",
        "risk_analyst",
    ],
    "output_format": {
        "decision_receipt": {
            "type": "gauntlet_receipt",
            "includes": [
                "decision_summary",
                "risk_scores",
                "compliance_attestations",
                "segregation_of_duties_proof",
                "dissenting_opinions",
                "audit_trail",
            ],
        },
        "risk_report": {
            "type": "structured_report",
            "sections": [
                "executive_summary",
                "market_risk_assessment",
                "credit_risk_assessment",
                "operational_risk_assessment",
                "regulatory_impact",
                "recommendation",
            ],
        },
    },
    "steps": [
        # --- Phase 1: Intake and Segregation of Duties ---
        {
            "id": "decision_intake",
            "type": "task",
            "name": "Financial Decision Intake",
            "description": (
                "Capture the financial decision context, classify by type "
                "(investment, lending, trading, regulatory), set materiality "
                "threshold, and identify required approval levels."
            ),
            "config": {
                "task_type": "validate",
                "validation_rules": [
                    "decision_type_classified",
                    "materiality_threshold_set",
                    "approval_levels_identified",
                    "decision_scope_bounded",
                ],
            },
        },
        {
            "id": "segregation_of_duties_check",
            "type": "task",
            "name": "Segregation of Duties Verification",
            "description": (
                "Enforce segregation of duties: verify that proposer, reviewer, "
                "and approver are distinct individuals/roles. SOX Section 404 "
                "requirement."
            ),
            "config": {
                "task_type": "function",
                "function_name": "verify_segregation_of_duties",
                "sod_rules": [
                    "proposer_not_approver",
                    "reviewer_not_proposer",
                    "approver_not_reviewer",
                    "dual_authorization_for_material_decisions",
                ],
            },
        },
        {
            "id": "sod_gate",
            "type": "decision",
            "name": "Segregation of Duties Gate",
            "description": ("Block workflow if segregation of duties requirements are not met."),
            "config": {
                "condition": "sod_check_passed",
                "true_target": "regulatory_compliance_screening",
                "false_target": "sod_remediation",
            },
        },
        {
            "id": "sod_remediation",
            "type": "human_checkpoint",
            "name": "SoD Remediation",
            "description": (
                "Route to compliance for manual SoD remediation when automated checks fail."
            ),
            "config": {
                "approval_type": "remediation",
                "required_role": "compliance_officer",
                "checklist": [
                    "Reassign roles to satisfy SoD requirements",
                    "Document SoD exception if applicable",
                    "Obtain compensating control approval",
                ],
            },
        },
        # --- Phase 2: Regulatory Compliance Screening ---
        {
            "id": "regulatory_compliance_screening",
            "type": "debate",
            "name": "Regulatory Compliance Screening",
            "description": (
                "Multi-agent screening against applicable regulatory "
                "frameworks: SOX, Basel III, MiFID II."
            ),
            "config": {
                "agents": ["sox", "compliance_officer", "internal_auditor"],
                "rounds": 2,
                "topic_template": (
                    "Screen the following financial decision for regulatory "
                    "compliance (SOX, Basel III, MiFID II): {decision_description}. "
                    "Materiality: {materiality_threshold}. "
                    "Decision type: {decision_type}."
                ),
                "compliance_checks": [
                    "sox_section_302_certification",
                    "sox_section_404_internal_controls",
                    "basel_iii_capital_adequacy",
                    "basel_iii_leverage_ratio",
                    "mifid_ii_best_execution",
                    "mifid_ii_suitability",
                    "gaap_revenue_recognition",
                ],
            },
        },
        {
            "id": "regulatory_gate",
            "type": "decision",
            "name": "Regulatory Compliance Gate",
            "description": "Block if regulatory compliance screening fails.",
            "config": {
                "condition": "regulatory_screening_passed",
                "true_target": "historical_context",
                "false_target": "regulatory_remediation",
            },
        },
        {
            "id": "regulatory_remediation",
            "type": "human_checkpoint",
            "name": "Regulatory Remediation",
            "description": (
                "Compliance officer remediates regulatory concerns before the decision can proceed."
            ),
            "config": {
                "approval_type": "remediation",
                "required_role": "compliance_officer",
                "checklist": [
                    "Address identified regulatory gaps",
                    "Document compensating controls",
                    "Obtain regulatory exception approval if needed",
                ],
            },
        },
        # --- Phase 3: Risk Assessment (Market, Credit, Operational) ---
        {
            "id": "historical_context",
            "type": "memory_read",
            "name": "Historical Context Retrieval",
            "description": (
                "Retrieve prior decisions, risk assessments, and market data "
                "relevant to this financial decision."
            ),
            "config": {
                "query_template": (
                    "Financial decision history and risk data for: "
                    "{decision_description}. Include prior risk assessments, "
                    "market conditions, and regulatory findings."
                ),
                "domains": [
                    "finance/decisions",
                    "finance/risk",
                    "compliance/sox",
                    "compliance/regulatory",
                ],
            },
        },
        {
            "id": "market_risk_assessment",
            "type": "debate",
            "name": "Market Risk Assessment",
            "description": (
                "Assess market risk: price volatility, liquidity risk, "
                "interest rate exposure, currency risk."
            ),
            "config": {
                "agents": ["financial_auditor", "risk_analyst", "compliance_officer"],
                "rounds": 2,
                "topic_template": (
                    "Assess market risk for: {decision_description}. "
                    "Historical context: {historical_data}. "
                    "Evaluate: price volatility, liquidity risk, "
                    "interest rate exposure, currency risk."
                ),
                "risk_categories": [
                    "price_volatility",
                    "liquidity_risk",
                    "interest_rate_exposure",
                    "currency_risk",
                ],
            },
        },
        {
            "id": "credit_risk_assessment",
            "type": "debate",
            "name": "Credit Risk Assessment",
            "description": (
                "Assess credit risk: counterparty default probability, "
                "concentration risk, collateral adequacy."
            ),
            "config": {
                "agents": ["financial_auditor", "risk_analyst", "internal_auditor"],
                "rounds": 2,
                "topic_template": (
                    "Assess credit risk for: {decision_description}. "
                    "Evaluate: counterparty default probability, "
                    "concentration risk, collateral adequacy, "
                    "Basel III capital requirements."
                ),
                "risk_categories": [
                    "counterparty_default",
                    "concentration_risk",
                    "collateral_adequacy",
                    "capital_requirements",
                ],
            },
        },
        {
            "id": "operational_risk_assessment",
            "type": "debate",
            "name": "Operational Risk Assessment",
            "description": (
                "Assess operational risk: process failure, system risk, "
                "fraud risk, compliance risk."
            ),
            "config": {
                "agents": [
                    "internal_auditor",
                    "forensic_accountant",
                    "compliance_officer",
                ],
                "rounds": 2,
                "topic_template": (
                    "Assess operational risk for: {decision_description}. "
                    "Evaluate: process failure probability, system risk, "
                    "fraud indicators, internal control adequacy."
                ),
                "risk_categories": [
                    "process_failure",
                    "system_risk",
                    "fraud_risk",
                    "internal_control_adequacy",
                ],
            },
        },
        {
            "id": "risk_consolidation",
            "type": "debate",
            "name": "Consolidated Risk Assessment",
            "description": (
                "Consolidate market, credit, and operational risk scores "
                "into an overall risk profile with recommendation."
            ),
            "config": {
                "agents": [
                    "financial_auditor",
                    "risk_analyst",
                    "compliance_officer",
                    "sox",
                ],
                "rounds": 2,
                "topic_template": (
                    "Consolidate risk assessment for: {decision_description}. "
                    "Market risk: {market_risk_score}. "
                    "Credit risk: {credit_risk_score}. "
                    "Operational risk: {operational_risk_score}. "
                    "Provide overall risk rating and recommendation."
                ),
                "consensus_threshold": 0.75,
            },
        },
        # --- Phase 4: Approval Chain ---
        {
            "id": "materiality_gate",
            "type": "decision",
            "name": "Materiality-Based Routing",
            "description": (
                "Route to appropriate approval level based on materiality "
                "threshold and risk rating."
            ),
            "config": {
                "condition": "requires_executive_approval",
                "true_target": "executive_review",
                "false_target": "manager_review",
            },
        },
        {
            "id": "manager_review",
            "type": "human_checkpoint",
            "name": "Manager Review",
            "description": "Manager-level review for sub-material decisions.",
            "config": {
                "approval_type": "review",
                "required_role": "department_manager",
                "checklist": [
                    "Review risk assessment summary",
                    "Confirm regulatory compliance",
                    "Approve within delegated authority",
                ],
            },
        },
        {
            "id": "executive_review",
            "type": "human_checkpoint",
            "name": "Executive Review (CFO/CRO)",
            "description": (
                "Executive-level review for material decisions requiring "
                "CFO or Chief Risk Officer sign-off."
            ),
            "config": {
                "approval_type": "sign_off",
                "required_roles": ["cfo", "chief_risk_officer"],
                "checklist": [
                    "Review consolidated risk assessment",
                    "Confirm regulatory compliance attestations",
                    "Verify segregation of duties",
                    "Approve materiality impact",
                    "Confirm board notification if required",
                ],
            },
        },
        # --- Phase 5: Audit Trail and Receipt ---
        {
            "id": "generate_audit_trail",
            "type": "task",
            "name": "Generate Audit Trail",
            "description": (
                "Generate comprehensive audit trail documenting the "
                "decision process, participants, evidence, and approvals."
            ),
            "config": {
                "task_type": "function",
                "function_name": "generate_financial_audit_trail",
                "trail_config": {
                    "include_all_debate_rounds": True,
                    "include_risk_scores": True,
                    "include_compliance_checks": True,
                    "include_sod_verification": True,
                    "tamper_evident": True,
                },
            },
        },
        {
            "id": "generate_decision_receipt",
            "type": "task",
            "name": "Generate Decision Receipt",
            "description": (
                "Generate cryptographic decision receipt with full audit trail, "
                "risk scores, and compliance attestations."
            ),
            "config": {
                "task_type": "function",
                "function_name": "generate_gauntlet_receipt",
                "receipt_config": {
                    "include_risk_scores": True,
                    "include_compliance_attestations": True,
                    "include_sod_proof": True,
                    "include_dissenting_opinions": True,
                    "hash_algorithm": "sha256",
                },
            },
        },
        {
            "id": "generate_risk_report",
            "type": "task",
            "name": "Generate Risk Report",
            "description": "Generate structured risk assessment report.",
            "config": {
                "task_type": "transform",
                "template": "financial_risk_report",
            },
        },
        {
            "id": "archive",
            "type": "memory_write",
            "name": "Archive Financial Decision",
            "description": (
                "Archive decision with all supporting evidence for "
                "SOX-compliant retention (7 years)."
            ),
            "config": {
                "domain": "finance/decisions",
                "retention_years": 7,
                "compliance_tags": ["sox", "basel_iii", "mifid_ii"],
            },
        },
    ],
    "transitions": [
        {"from": "decision_intake", "to": "segregation_of_duties_check"},
        {"from": "segregation_of_duties_check", "to": "sod_gate"},
        {"from": "sod_gate", "to": "regulatory_compliance_screening", "condition": "sod_ok"},
        {"from": "sod_gate", "to": "sod_remediation", "condition": "sod_failed"},
        {"from": "sod_remediation", "to": "segregation_of_duties_check"},
        {"from": "regulatory_compliance_screening", "to": "regulatory_gate"},
        {
            "from": "regulatory_gate",
            "to": "historical_context",
            "condition": "compliant",
        },
        {
            "from": "regulatory_gate",
            "to": "regulatory_remediation",
            "condition": "non_compliant",
        },
        {"from": "regulatory_remediation", "to": "regulatory_compliance_screening"},
        {"from": "historical_context", "to": "market_risk_assessment"},
        {"from": "market_risk_assessment", "to": "credit_risk_assessment"},
        {"from": "credit_risk_assessment", "to": "operational_risk_assessment"},
        {"from": "operational_risk_assessment", "to": "risk_consolidation"},
        {"from": "risk_consolidation", "to": "materiality_gate"},
        {
            "from": "materiality_gate",
            "to": "executive_review",
            "condition": "material",
        },
        {
            "from": "materiality_gate",
            "to": "manager_review",
            "condition": "sub_material",
        },
        {"from": "manager_review", "to": "generate_audit_trail", "condition": "approved"},
        {
            "from": "manager_review",
            "to": "risk_consolidation",
            "condition": "rejected",
        },
        {
            "from": "executive_review",
            "to": "generate_audit_trail",
            "condition": "approved",
        },
        {
            "from": "executive_review",
            "to": "risk_consolidation",
            "condition": "rejected",
        },
        {"from": "generate_audit_trail", "to": "generate_decision_receipt"},
        {"from": "generate_decision_receipt", "to": "generate_risk_report"},
        {"from": "generate_risk_report", "to": "archive"},
    ],
    "metadata": {
        "author": "aragora",
        "version": "1.0.0",
        "min_agents": 4,
        "max_agents": 6,
        "estimated_duration_minutes": 20,
        "regulatory_note": (
            "This template enforces SOX Section 302/404 requirements, "
            "Basel III capital and leverage ratios, and MiFID II best "
            "execution and suitability obligations. Segregation of duties "
            "is enforced at workflow level."
        ),
    },
}
