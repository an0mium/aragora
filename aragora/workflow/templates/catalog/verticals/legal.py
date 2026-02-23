"""
Legal Vertical Playbook Template.

Legal analysis decision template with:
- Precedent research stages
- Jurisdictional analysis
- Risk/liability assessment
- Confidentiality classification
- Privilege preservation checks (attorney-client, work product)

This is a *playbook* template -- a complete decision workflow that wires
together debate, precedent analysis, privilege protection, and receipt
generation for legal decisions and opinions.
"""

from __future__ import annotations

from typing import Any

LEGAL_ANALYSIS_DECISION_TEMPLATE: dict[str, Any] = {
    "name": "Legal Analysis Decision Playbook",
    "description": (
        "End-to-end legal analysis workflow with precedent research, "
        "jurisdictional analysis, risk/liability assessment, "
        "confidentiality classification, privilege preservation, "
        "and audit-ready decision receipt generation."
    ),
    "category": "legal",
    "version": "1.0",
    "tags": [
        "legal",
        "precedent",
        "jurisdictional",
        "liability",
        "privilege",
        "confidentiality",
        "decision-receipt",
        "playbook",
    ],
    "compliance_frameworks": [
        "ABA_MODEL_RULES",
        "ATTORNEY_CLIENT_PRIVILEGE",
        "WORK_PRODUCT_DOCTRINE",
        "CONFLICT_OF_INTEREST",
    ],
    "required_agent_types": [
        "contract_analyst",
        "compliance_officer",
        "litigation_support",
        "m_and_a_counsel",
        "legal_researcher",
    ],
    "output_format": {
        "decision_receipt": {
            "type": "gauntlet_receipt",
            "includes": [
                "legal_opinion_summary",
                "precedent_citations",
                "jurisdictional_analysis",
                "risk_liability_assessment",
                "privilege_log",
                "confidentiality_classification",
                "dissenting_opinions",
                "audit_trail",
            ],
        },
        "legal_memorandum": {
            "type": "structured_report",
            "sections": [
                "question_presented",
                "brief_answer",
                "statement_of_facts",
                "applicable_law",
                "analysis",
                "conclusion",
                "recommendation",
            ],
        },
    },
    "steps": [
        # --- Phase 1: Intake and Privilege Protection ---
        {
            "id": "legal_intake",
            "type": "task",
            "name": "Legal Matter Intake",
            "description": (
                "Capture the legal question, classify matter type "
                "(litigation, transactional, regulatory, advisory), "
                "identify jurisdictions, and set confidentiality level."
            ),
            "config": {
                "task_type": "validate",
                "validation_rules": [
                    "legal_question_defined",
                    "matter_type_classified",
                    "jurisdictions_identified",
                    "confidentiality_level_set",
                    "client_identified",
                ],
            },
        },
        {
            "id": "conflict_of_interest_check",
            "type": "task",
            "name": "Conflict of Interest Check",
            "description": (
                "Screen for conflicts of interest across current and former "
                "client representations per ABA Model Rule 1.7/1.9."
            ),
            "config": {
                "task_type": "function",
                "function_name": "check_conflicts_of_interest",
                "conflict_rules": [
                    "current_client_conflict_1_7",
                    "former_client_conflict_1_9",
                    "imputed_disqualification_1_10",
                    "organization_as_client_1_13",
                ],
            },
        },
        {
            "id": "conflict_gate",
            "type": "decision",
            "name": "Conflict Gate",
            "description": "Block if conflicts of interest are detected.",
            "config": {
                "condition": "no_conflicts_detected",
                "true_target": "privilege_classification",
                "false_target": "conflict_resolution",
            },
        },
        {
            "id": "conflict_resolution",
            "type": "human_checkpoint",
            "name": "Conflict Resolution",
            "description": (
                "Senior attorney reviews and resolves conflict of interest "
                "issues -- waiver, ethical wall, or withdrawal."
            ),
            "config": {
                "approval_type": "review",
                "required_role": "senior_attorney",
                "checklist": [
                    "Assess conflict severity",
                    "Determine if informed consent waiver is appropriate",
                    "Implement ethical wall if needed",
                    "Document resolution or recommend withdrawal",
                ],
            },
        },
        {
            "id": "privilege_classification",
            "type": "task",
            "name": "Privilege and Confidentiality Classification",
            "description": (
                "Classify all materials for privilege protection: "
                "attorney-client privilege, work product doctrine, "
                "and confidentiality tier (Public, Internal, Confidential, "
                "Highly Confidential -- Attorney Eyes Only)."
            ),
            "config": {
                "task_type": "function",
                "function_name": "classify_privilege_and_confidentiality",
                "classification_tiers": [
                    "public",
                    "internal",
                    "confidential",
                    "highly_confidential_aeo",
                ],
                "privilege_categories": [
                    "attorney_client_privilege",
                    "work_product_doctrine",
                    "joint_defense_privilege",
                    "common_interest_privilege",
                ],
            },
        },
        {
            "id": "privilege_log_init",
            "type": "task",
            "name": "Initialize Privilege Log",
            "description": (
                "Initialize a privilege log to track all privileged "
                "communications and work product throughout the workflow."
            ),
            "config": {
                "task_type": "function",
                "function_name": "initialize_privilege_log",
                "log_fields": [
                    "document_id",
                    "date",
                    "author",
                    "recipients",
                    "privilege_type",
                    "description",
                ],
            },
        },
        # --- Phase 2: Precedent Research ---
        {
            "id": "precedent_retrieval",
            "type": "memory_read",
            "name": "Precedent Retrieval",
            "description": (
                "Retrieve relevant legal precedents, prior opinions, "
                "and applicable statutes from the knowledge base."
            ),
            "config": {
                "query_template": (
                    "Legal precedents and authorities for: {legal_question}. "
                    "Jurisdictions: {jurisdictions}. "
                    "Matter type: {matter_type}. "
                    "Include: case law, statutes, regulations, prior firm opinions."
                ),
                "domains": [
                    "legal/precedents",
                    "legal/statutes",
                    "legal/opinions",
                    "legal/regulations",
                ],
            },
        },
        {
            "id": "precedent_analysis",
            "type": "debate",
            "name": "Precedent Analysis",
            "description": (
                "Multi-agent analysis of relevant precedents. Assess "
                "applicability, distinguish unfavorable authority, "
                "identify gaps in legal research."
            ),
            "config": {
                "agents": [
                    "legal_researcher",
                    "litigation_support",
                    "contract_analyst",
                ],
                "rounds": 3,
                "topic_template": (
                    "Analyze legal precedents for: {legal_question}. "
                    "Retrieved authorities: {precedents}. "
                    "Assess: binding vs. persuasive authority, "
                    "distinguishable cases, gaps in research."
                ),
                "consensus_threshold": 0.7,
            },
        },
        # --- Phase 3: Jurisdictional Analysis ---
        {
            "id": "jurisdictional_analysis",
            "type": "debate",
            "name": "Jurisdictional Analysis",
            "description": (
                "Analyze jurisdictional considerations: applicable law, "
                "choice of law, forum selection, cross-border implications."
            ),
            "config": {
                "agents": [
                    "litigation_support",
                    "compliance_officer",
                    "legal_researcher",
                ],
                "rounds": 2,
                "topic_template": (
                    "Analyze jurisdictional issues for: {legal_question}. "
                    "Identified jurisdictions: {jurisdictions}. "
                    "Assess: governing law, choice of law rules, "
                    "forum selection, cross-border enforcement."
                ),
                "analysis_dimensions": [
                    "governing_law",
                    "choice_of_law",
                    "forum_selection",
                    "cross_border_enforcement",
                    "statute_of_limitations",
                ],
            },
        },
        # --- Phase 4: Risk and Liability Assessment ---
        {
            "id": "liability_assessment",
            "type": "debate",
            "name": "Risk and Liability Assessment",
            "description": (
                "Multi-agent assessment of legal risks and potential "
                "liability exposure."
            ),
            "config": {
                "agents": [
                    "litigation_support",
                    "contract_analyst",
                    "m_and_a_counsel",
                ],
                "rounds": 3,
                "topic_template": (
                    "Assess risk and liability for: {legal_question}. "
                    "Precedent analysis: {precedent_summary}. "
                    "Jurisdictional analysis: {jurisdictional_summary}. "
                    "Evaluate: probability of adverse outcome, "
                    "magnitude of exposure, mitigation strategies."
                ),
                "consensus_threshold": 0.75,
                "risk_dimensions": [
                    "probability_of_adverse_outcome",
                    "magnitude_of_exposure",
                    "reputational_risk",
                    "regulatory_risk",
                    "mitigation_feasibility",
                ],
            },
        },
        # --- Phase 5: Legal Opinion Synthesis ---
        {
            "id": "opinion_synthesis",
            "type": "debate",
            "name": "Legal Opinion Synthesis",
            "description": (
                "Synthesize precedent analysis, jurisdictional analysis, "
                "and liability assessment into a legal opinion with "
                "recommendation."
            ),
            "config": {
                "agents": [
                    "legal_researcher",
                    "litigation_support",
                    "m_and_a_counsel",
                    "compliance_officer",
                ],
                "rounds": 2,
                "topic_template": (
                    "Synthesize legal opinion for: {legal_question}. "
                    "Precedent analysis: {precedent_summary}. "
                    "Jurisdictional analysis: {jurisdictional_summary}. "
                    "Liability assessment: {liability_summary}. "
                    "Provide: brief answer, analysis, recommendation."
                ),
                "consensus_threshold": 0.8,
            },
        },
        # --- Phase 6: Review and Approval ---
        {
            "id": "privilege_review",
            "type": "task",
            "name": "Privilege Preservation Review",
            "description": (
                "Final review to ensure all privileged communications "
                "are properly logged and protected."
            ),
            "config": {
                "task_type": "function",
                "function_name": "review_privilege_log",
                "checks": [
                    "all_privileged_docs_logged",
                    "no_inadvertent_waiver",
                    "privilege_labels_applied",
                    "distribution_restricted",
                ],
            },
        },
        {
            "id": "senior_attorney_review",
            "type": "human_checkpoint",
            "name": "Senior Attorney Review",
            "description": (
                "Senior attorney reviews the synthesized legal opinion, "
                "precedent analysis, and risk assessment."
            ),
            "config": {
                "approval_type": "review",
                "required_role": "senior_attorney",
                "checklist": [
                    "Verify precedent analysis accuracy",
                    "Confirm jurisdictional analysis completeness",
                    "Review risk/liability assessment",
                    "Validate legal opinion conclusion",
                    "Confirm privilege log completeness",
                ],
                "timeout_hours": 72,
            },
        },
        {
            "id": "partner_sign_off",
            "type": "human_checkpoint",
            "name": "Partner Sign-Off",
            "description": (
                "Managing partner sign-off for final legal opinion."
            ),
            "config": {
                "approval_type": "sign_off",
                "required_role": "partner",
                "checklist": [
                    "Review legal opinion quality",
                    "Confirm risk assessment alignment",
                    "Approve for client delivery",
                    "Verify malpractice risk acceptable",
                ],
            },
        },
        # --- Phase 7: Receipt and Archive ---
        {
            "id": "generate_decision_receipt",
            "type": "task",
            "name": "Generate Decision Receipt",
            "description": (
                "Generate cryptographic decision receipt with full audit "
                "trail, precedent citations, and privilege log."
            ),
            "config": {
                "task_type": "function",
                "function_name": "generate_gauntlet_receipt",
                "receipt_config": {
                    "include_precedent_citations": True,
                    "include_jurisdictional_analysis": True,
                    "include_liability_assessment": True,
                    "include_privilege_log": True,
                    "include_dissenting_opinions": True,
                    "hash_algorithm": "sha256",
                },
            },
        },
        {
            "id": "generate_legal_memorandum",
            "type": "task",
            "name": "Generate Legal Memorandum",
            "description": "Generate structured legal memorandum.",
            "config": {
                "task_type": "transform",
                "template": "legal_memorandum",
            },
        },
        {
            "id": "archive",
            "type": "memory_write",
            "name": "Archive Legal Opinion",
            "description": (
                "Archive legal opinion with supporting materials. "
                "Apply confidentiality and privilege protections."
            ),
            "config": {
                "domain": "legal/opinions",
                "retention_years": 10,
                "compliance_tags": ["privileged", "confidential"],
                "access_control": "attorney_eyes_only",
            },
        },
    ],
    "transitions": [
        {"from": "legal_intake", "to": "conflict_of_interest_check"},
        {"from": "conflict_of_interest_check", "to": "conflict_gate"},
        {
            "from": "conflict_gate",
            "to": "privilege_classification",
            "condition": "no_conflict",
        },
        {
            "from": "conflict_gate",
            "to": "conflict_resolution",
            "condition": "conflict_detected",
        },
        {"from": "conflict_resolution", "to": "privilege_classification"},
        {"from": "privilege_classification", "to": "privilege_log_init"},
        {"from": "privilege_log_init", "to": "precedent_retrieval"},
        {"from": "precedent_retrieval", "to": "precedent_analysis"},
        {"from": "precedent_analysis", "to": "jurisdictional_analysis"},
        {"from": "jurisdictional_analysis", "to": "liability_assessment"},
        {"from": "liability_assessment", "to": "opinion_synthesis"},
        {"from": "opinion_synthesis", "to": "privilege_review"},
        {"from": "privilege_review", "to": "senior_attorney_review"},
        {
            "from": "senior_attorney_review",
            "to": "partner_sign_off",
            "condition": "approved",
        },
        {
            "from": "senior_attorney_review",
            "to": "opinion_synthesis",
            "condition": "rejected",
        },
        {
            "from": "partner_sign_off",
            "to": "generate_decision_receipt",
            "condition": "approved",
        },
        {
            "from": "partner_sign_off",
            "to": "senior_attorney_review",
            "condition": "rejected",
        },
        {"from": "generate_decision_receipt", "to": "generate_legal_memorandum"},
        {"from": "generate_legal_memorandum", "to": "archive"},
    ],
    "metadata": {
        "author": "aragora",
        "version": "1.0.0",
        "min_agents": 3,
        "max_agents": 5,
        "estimated_duration_minutes": 25,
        "regulatory_note": (
            "This template enforces attorney-client privilege, work product "
            "doctrine protections, and conflict of interest screening per "
            "ABA Model Rules. All communications are logged in a privilege "
            "log. Confidentiality classification is applied at intake."
        ),
    },
}
