"""
Accounting/Financial Workflow Templates.

Templates for financial audits and SOX compliance.
"""

from typing import Dict, Any

FINANCIAL_AUDIT_TEMPLATE: Dict[str, Any] = {
    "name": "Financial Statement Audit",
    "description": "Multi-agent financial statement audit with substantive testing",
    "category": "accounting",
    "version": "1.0",
    "tags": ["accounting", "audit", "financial", "gaap"],
    "steps": [
        {
            "id": "engagement_setup",
            "type": "task",
            "name": "Engagement Setup",
            "description": "Configure audit engagement parameters",
            "config": {
                "task_type": "validate",
                "validation_rules": [
                    "engagement_letter_signed",
                    "materiality_threshold_set",
                    "audit_period_defined",
                ],
            },
        },
        {
            "id": "risk_assessment",
            "type": "debate",
            "name": "Risk Assessment",
            "description": "Assess inherent and control risks",
            "config": {
                "agents": ["financial_auditor", "internal_auditor", "forensic_accountant"],
                "rounds": 2,
                "topic_template": "Assess audit risks for: {client_profile}",
            },
        },
        {
            "id": "control_testing",
            "type": "debate",
            "name": "Internal Control Testing",
            "description": "Test effectiveness of internal controls",
            "config": {
                "agents": ["internal_auditor", "sox", "financial_auditor"],
                "rounds": 3,
                "topic_template": "Test internal controls: {control_documentation}",
            },
        },
        {
            "id": "parallel_testing",
            "type": "parallel",
            "name": "Substantive Testing",
            "description": "Parallel substantive tests by account area",
            "branches": [
                {
                    "id": "revenue_testing",
                    "steps": [
                        {
                            "id": "revenue_analysis",
                            "type": "debate",
                            "name": "Revenue Recognition Testing",
                            "config": {
                                "agents": ["financial_auditor", "tax_specialist"],
                                "topic_template": "Test revenue recognition (ASC 606): {revenue_data}",
                            },
                        },
                    ],
                },
                {
                    "id": "expense_testing",
                    "steps": [
                        {
                            "id": "expense_analysis",
                            "type": "debate",
                            "name": "Expense Testing",
                            "config": {
                                "agents": ["financial_auditor", "forensic_accountant"],
                                "topic_template": "Test expense classifications: {expense_data}",
                            },
                        },
                    ],
                },
                {
                    "id": "asset_testing",
                    "steps": [
                        {
                            "id": "asset_analysis",
                            "type": "debate",
                            "name": "Asset Valuation Testing",
                            "config": {
                                "agents": ["financial_auditor", "internal_auditor"],
                                "topic_template": "Test asset valuations: {asset_data}",
                            },
                        },
                    ],
                },
                {
                    "id": "liability_testing",
                    "steps": [
                        {
                            "id": "liability_analysis",
                            "type": "debate",
                            "name": "Liability Completeness Testing",
                            "config": {
                                "agents": ["financial_auditor", "compliance_officer"],
                                "topic_template": "Test liability completeness: {liability_data}",
                            },
                        },
                    ],
                },
            ],
        },
        {
            "id": "analytical_review",
            "type": "debate",
            "name": "Analytical Procedures",
            "description": "Perform analytical review procedures",
            "config": {
                "agents": ["financial_auditor", "forensic_accountant", "data_architect"],
                "topic_template": "Analytical review of financials: {financial_statements}",
            },
        },
        {
            "id": "going_concern",
            "type": "debate",
            "name": "Going Concern Assessment",
            "description": "Evaluate going concern assumption",
            "config": {
                "agents": ["financial_auditor", "m_and_a_counsel"],
                "topic_template": "Assess going concern: {financial_indicators}",
            },
        },
        {
            "id": "subsequent_events",
            "type": "debate",
            "name": "Subsequent Events Review",
            "description": "Review subsequent events through report date",
            "config": {
                "agents": ["financial_auditor", "compliance_officer"],
                "topic_template": "Review subsequent events: {event_log}",
            },
        },
        {
            "id": "findings_consolidation",
            "type": "task",
            "name": "Consolidate Findings",
            "description": "Consolidate all audit findings",
            "config": {
                "task_type": "aggregate",
            },
        },
        {
            "id": "management_discussion",
            "type": "human_checkpoint",
            "name": "Management Discussion",
            "description": "Discuss findings with management",
            "config": {
                "approval_type": "review",
                "required_role": "audit_partner",
                "notification_roles": ["client_cfo", "audit_committee"],
            },
        },
        {
            "id": "partner_review",
            "type": "human_checkpoint",
            "name": "Partner Review",
            "description": "Engagement partner final review",
            "config": {
                "approval_type": "sign_off",
                "required_role": "engagement_partner",
                "checklist": [
                    "All testing completed",
                    "All findings resolved or noted",
                    "Independence confirmed",
                    "Quality review completed",
                ],
            },
        },
        {
            "id": "generate_opinion",
            "type": "task",
            "name": "Generate Audit Opinion",
            "description": "Generate audit opinion letter",
            "config": {
                "task_type": "transform",
                "template": "audit_opinion_letter",
            },
        },
        {
            "id": "archive",
            "type": "memory_write",
            "name": "Archive Workpapers",
            "description": "Archive audit workpapers",
            "config": {
                "domain": "financial/audit",
                "retention_years": 7,
            },
        },
    ],
    "transitions": [
        {"from": "engagement_setup", "to": "risk_assessment"},
        {"from": "risk_assessment", "to": "control_testing"},
        {"from": "control_testing", "to": "parallel_testing"},
        {"from": "parallel_testing", "to": "analytical_review"},
        {"from": "analytical_review", "to": "going_concern"},
        {"from": "going_concern", "to": "subsequent_events"},
        {"from": "subsequent_events", "to": "findings_consolidation"},
        {"from": "findings_consolidation", "to": "management_discussion"},
        {"from": "management_discussion", "to": "partner_review"},
        {"from": "partner_review", "to": "generate_opinion", "condition": "approved"},
        {"from": "generate_opinion", "to": "archive"},
    ],
}

SOX_COMPLIANCE_TEMPLATE: Dict[str, Any] = {
    "name": "SOX Compliance Assessment",
    "description": "Sarbanes-Oxley Section 404 internal control assessment",
    "category": "accounting",
    "version": "1.0",
    "tags": ["accounting", "sox", "compliance", "internal-controls"],
    "steps": [
        {
            "id": "scope_definition",
            "type": "task",
            "name": "Define Scope",
            "description": "Define scope of SOX testing",
            "config": {
                "task_type": "validate",
                "validation_rules": [
                    "significant_accounts_identified",
                    "material_processes_mapped",
                    "control_objectives_defined",
                ],
            },
        },
        {
            "id": "process_documentation",
            "type": "memory_read",
            "name": "Gather Process Documentation",
            "description": "Retrieve process and control documentation",
            "config": {
                "query_template": "SOX controls documentation for {fiscal_year}",
                "domains": ["compliance/sox", "operational/processes"],
            },
        },
        {
            "id": "risk_assessment",
            "type": "debate",
            "name": "Control Risk Assessment",
            "description": "Assess risk of material misstatement",
            "config": {
                "agents": ["sox", "financial_auditor", "internal_auditor"],
                "rounds": 2,
                "topic_template": "Assess control risks: {process_documentation}",
            },
        },
        {
            "id": "control_design_testing",
            "type": "debate",
            "name": "Control Design Testing",
            "description": "Test design effectiveness of controls",
            "config": {
                "agents": ["sox", "internal_auditor", "compliance_officer"],
                "rounds": 2,
                "topic_template": "Test control design: {controls}",
            },
        },
        {
            "id": "control_operating_testing",
            "type": "debate",
            "name": "Operating Effectiveness Testing",
            "description": "Test operating effectiveness of controls",
            "config": {
                "agents": ["sox", "internal_auditor", "forensic_accountant"],
                "rounds": 3,
                "topic_template": "Test operating effectiveness: {controls}",
            },
        },
        {
            "id": "itgc_testing",
            "type": "debate",
            "name": "IT General Controls Testing",
            "description": "Test IT general controls",
            "config": {
                "agents": ["sox", "security_engineer", "devops_engineer"],
                "rounds": 2,
                "topic_template": "Test ITGCs: {it_controls}",
            },
        },
        {
            "id": "deficiency_evaluation",
            "type": "debate",
            "name": "Deficiency Evaluation",
            "description": "Evaluate and classify control deficiencies",
            "config": {
                "agents": ["sox", "financial_auditor", "compliance_officer"],
                "topic_template": "Evaluate deficiencies: {test_results}",
            },
        },
        {
            "id": "material_weakness_check",
            "type": "decision",
            "name": "Material Weakness Check",
            "description": "Determine if material weaknesses exist",
            "config": {
                "condition": "material_weakness_identified",
                "true_target": "remediation_assessment",
                "false_target": "management_assertion",
            },
        },
        {
            "id": "remediation_assessment",
            "type": "debate",
            "name": "Remediation Assessment",
            "description": "Assess remediation plans for material weaknesses",
            "config": {
                "agents": ["sox", "internal_auditor", "compliance_officer"],
                "topic_template": "Assess remediation for: {material_weaknesses}",
            },
        },
        {
            "id": "cfo_ceo_review",
            "type": "human_checkpoint",
            "name": "Executive Review",
            "description": "CFO/CEO review of assessment",
            "config": {
                "approval_type": "review",
                "required_roles": ["cfo", "ceo"],
                "checklist": [
                    "Review control deficiencies",
                    "Approve management assertion",
                    "Confirm disclosure completeness",
                ],
            },
        },
        {
            "id": "management_assertion",
            "type": "task",
            "name": "Management Assertion",
            "description": "Prepare management assertion on ICFR",
            "config": {
                "task_type": "transform",
                "template": "sox_management_assertion",
            },
        },
        {
            "id": "audit_committee_presentation",
            "type": "human_checkpoint",
            "name": "Audit Committee Presentation",
            "description": "Present findings to audit committee",
            "config": {
                "approval_type": "presentation",
                "required_role": "audit_committee",
            },
        },
        {
            "id": "generate_report",
            "type": "task",
            "name": "Generate SOX Report",
            "description": "Generate SOX 404 compliance report",
            "config": {
                "task_type": "transform",
                "template": "sox_404_report",
            },
        },
        {
            "id": "archive",
            "type": "memory_write",
            "name": "Archive Documentation",
            "description": "Archive SOX documentation",
            "config": {
                "domain": "compliance/sox",
                "retention_years": 7,
            },
        },
    ],
    "transitions": [
        {"from": "scope_definition", "to": "process_documentation"},
        {"from": "process_documentation", "to": "risk_assessment"},
        {"from": "risk_assessment", "to": "control_design_testing"},
        {"from": "control_design_testing", "to": "control_operating_testing"},
        {"from": "control_operating_testing", "to": "itgc_testing"},
        {"from": "itgc_testing", "to": "deficiency_evaluation"},
        {"from": "deficiency_evaluation", "to": "material_weakness_check"},
        {"from": "material_weakness_check", "to": "remediation_assessment", "condition": "has_mw"},
        {"from": "material_weakness_check", "to": "management_assertion", "condition": "no_mw"},
        {"from": "remediation_assessment", "to": "cfo_ceo_review"},
        {"from": "cfo_ceo_review", "to": "management_assertion"},
        {"from": "management_assertion", "to": "audit_committee_presentation"},
        {"from": "audit_committee_presentation", "to": "generate_report"},
        {"from": "generate_report", "to": "archive"},
    ],
}

# =============================================================================
# Bank Reconciliation Workflow (Plaid + QBO Integration)
# =============================================================================

BANK_RECONCILIATION_TEMPLATE: Dict[str, Any] = {
    "name": "Bank Reconciliation",
    "description": "Automated bank-to-book reconciliation with Plaid and QuickBooks integration",
    "category": "accounting",
    "version": "1.0",
    "tags": ["accounting", "reconciliation", "plaid", "qbo", "banking"],
    "connectors": ["plaid", "qbo"],
    "steps": [
        {
            "id": "define_period",
            "type": "task",
            "name": "Define Reconciliation Period",
            "description": "Set start and end dates for reconciliation",
            "config": {
                "task_type": "validate",
                "validation_rules": [
                    "start_date_valid",
                    "end_date_valid",
                    "period_not_future",
                ],
            },
        },
        {
            "id": "fetch_bank_transactions",
            "type": "connector",
            "name": "Fetch Bank Transactions",
            "description": "Retrieve bank transactions from Plaid",
            "config": {
                "connector": "plaid",
                "operation": "get_transactions",
                "params": {
                    "start_date": "{start_date}",
                    "end_date": "{end_date}",
                    "account_ids": "{selected_accounts}",
                },
                "output_key": "bank_transactions",
            },
        },
        {
            "id": "fetch_book_transactions",
            "type": "connector",
            "name": "Fetch Book Transactions",
            "description": "Retrieve transactions from QuickBooks",
            "config": {
                "connector": "qbo",
                "operation": "list_transactions",
                "params": {
                    "start_date": "{start_date}",
                    "end_date": "{end_date}",
                    "transaction_types": ["Invoice", "Expense", "Payment", "Transfer"],
                },
                "output_key": "book_transactions",
            },
        },
        {
            "id": "auto_match",
            "type": "task",
            "name": "Automated Matching",
            "description": "Run automated transaction matching algorithm",
            "config": {
                "task_type": "function",
                "function_name": "reconciliation_auto_match",
                "params": {
                    "bank_txns": "{bank_transactions}",
                    "book_txns": "{book_transactions}",
                    "tolerance_days": 3,
                    "tolerance_amount": 0.01,
                },
                "output_key": "match_results",
            },
        },
        {
            "id": "check_discrepancies",
            "type": "decision",
            "name": "Check for Discrepancies",
            "description": "Determine if there are unmatched transactions",
            "config": {
                "condition": "match_results.discrepancy_count > 0",
                "true_target": "discrepancy_analysis",
                "false_target": "generate_report",
            },
        },
        {
            "id": "discrepancy_analysis",
            "type": "debate",
            "name": "Discrepancy Analysis",
            "description": "Multi-agent analysis of unmatched transactions",
            "config": {
                "agents": ["financial_auditor", "forensic_accountant", "internal_auditor"],
                "rounds": 2,
                "topic_template": "Analyze reconciliation discrepancies: {match_results.discrepancies}",
            },
        },
        {
            "id": "categorize_transactions",
            "type": "debate",
            "name": "Categorize Unmatched Transactions",
            "description": "Determine proper categorization for unmatched bank transactions",
            "config": {
                "agents": ["financial_auditor", "tax_specialist"],
                "rounds": 1,
                "topic_template": "Suggest QBO account categorization for: {match_results.unmatched_bank}",
            },
        },
        {
            "id": "create_entries_decision",
            "type": "decision",
            "name": "Auto-Create Entries Decision",
            "description": "Decide whether to auto-create QBO entries for unmatched bank transactions",
            "config": {
                "condition": "auto_create_entries_enabled and categorization_confidence >= 0.85",
                "true_target": "create_qbo_entries",
                "false_target": "accountant_review",
            },
        },
        {
            "id": "create_qbo_entries",
            "type": "connector",
            "name": "Create QBO Entries",
            "description": "Create journal entries in QuickBooks for matched bank transactions",
            "config": {
                "connector": "qbo",
                "operation": "create_journal_entries",
                "params": {
                    "entries": "{categorized_transactions}",
                },
                "output_key": "created_entries",
            },
        },
        {
            "id": "accountant_review",
            "type": "human_checkpoint",
            "name": "Accountant Review",
            "description": "Manual review of discrepancies by staff accountant",
            "config": {
                "approval_type": "review",
                "required_role": "staff_accountant",
                "checklist": [
                    "Review unmatched bank transactions",
                    "Review unmatched book transactions",
                    "Approve suggested categorizations",
                    "Flag suspicious activity",
                ],
            },
        },
        {
            "id": "anomaly_check",
            "type": "decision",
            "name": "Anomaly Detection Check",
            "description": "Check for suspicious transactions",
            "config": {
                "condition": "any(t.is_anomaly for t in bank_transactions)",
                "true_target": "fraud_investigation",
                "false_target": "controller_approval",
            },
        },
        {
            "id": "fraud_investigation",
            "type": "debate",
            "name": "Fraud Investigation",
            "description": "Multi-agent investigation of suspicious transactions",
            "config": {
                "agents": ["forensic_accountant", "internal_auditor", "compliance_officer"],
                "rounds": 3,
                "topic_template": "Investigate potential fraud: {anomalous_transactions}",
                "urgent": True,
            },
        },
        {
            "id": "controller_approval",
            "type": "human_checkpoint",
            "name": "Controller Approval",
            "description": "Final approval from accounting controller",
            "config": {
                "approval_type": "sign_off",
                "required_role": "controller",
                "checklist": [
                    "All discrepancies resolved or explained",
                    "Bank balance matches book balance",
                    "No unexplained variances",
                ],
            },
        },
        {
            "id": "generate_report",
            "type": "task",
            "name": "Generate Reconciliation Report",
            "description": "Generate formal bank reconciliation report",
            "config": {
                "task_type": "transform",
                "template": "bank_reconciliation_report",
            },
        },
        {
            "id": "archive",
            "type": "memory_write",
            "name": "Archive Reconciliation",
            "description": "Store reconciliation in knowledge base",
            "config": {
                "domain": "accounting/reconciliation",
                "retention_years": 7,
            },
        },
    ],
    "transitions": [
        {"from": "define_period", "to": "fetch_bank_transactions"},
        {"from": "fetch_bank_transactions", "to": "fetch_book_transactions"},
        {"from": "fetch_book_transactions", "to": "auto_match"},
        {"from": "auto_match", "to": "check_discrepancies"},
        {
            "from": "check_discrepancies",
            "to": "discrepancy_analysis",
            "condition": "has_discrepancies",
        },
        {"from": "check_discrepancies", "to": "generate_report", "condition": "no_discrepancies"},
        {"from": "discrepancy_analysis", "to": "categorize_transactions"},
        {"from": "categorize_transactions", "to": "create_entries_decision"},
        {"from": "create_entries_decision", "to": "create_qbo_entries", "condition": "auto_create"},
        {
            "from": "create_entries_decision",
            "to": "accountant_review",
            "condition": "manual_review",
        },
        {"from": "create_qbo_entries", "to": "accountant_review"},
        {"from": "accountant_review", "to": "anomaly_check"},
        {"from": "anomaly_check", "to": "fraud_investigation", "condition": "anomalies_found"},
        {"from": "anomaly_check", "to": "controller_approval", "condition": "no_anomalies"},
        {"from": "fraud_investigation", "to": "controller_approval"},
        {"from": "controller_approval", "to": "generate_report", "condition": "approved"},
        {"from": "controller_approval", "to": "accountant_review", "condition": "rejected"},
        {"from": "generate_report", "to": "archive"},
    ],
}
