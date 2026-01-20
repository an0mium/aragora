"""
AI/ML Workflow Templates.

Templates for model deployment, bias audits, ML pipelines, and AI governance.
"""

from typing import Any, Dict

MODEL_DEPLOYMENT_TEMPLATE: Dict[str, Any] = {
    "name": "ML Model Deployment Review",
    "description": "Comprehensive review for ML model production deployment",
    "category": "ai_ml",
    "version": "1.0",
    "tags": ["ai", "ml", "deployment", "production"],
    "steps": [
        {
            "id": "gather_artifacts",
            "type": "task",
            "name": "Gather Model Artifacts",
            "description": "Collect model card, training data docs, and metrics",
            "config": {
                "task_type": "aggregate",
                "sources": ["model_card", "training_data", "evaluation_metrics"],
            },
        },
        {
            "id": "performance_review",
            "type": "debate",
            "name": "Performance Review",
            "description": "Multi-agent review of model performance metrics",
            "config": {
                "agents": ["ml_engineer", "data_scientist", "statistician"],
                "rounds": 2,
                "topic_template": "Review model performance: {evaluation_metrics}",
            },
        },
        {
            "id": "bias_audit",
            "type": "debate",
            "name": "Bias and Fairness Audit",
            "description": "Audit model for demographic bias and fairness",
            "config": {
                "agents": ["ai_ethics_specialist", "data_scientist", "claude"],
                "rounds": 3,
                "topic_template": "Audit for bias: {model_card} {demographic_analysis}",
            },
        },
        {
            "id": "robustness_testing",
            "type": "gauntlet",
            "name": "Robustness Testing",
            "description": "Adversarial testing for model robustness",
            "config": {
                "profile": "ai_act",
                "input_type": "spec",
            },
        },
        {
            "id": "drift_analysis",
            "type": "debate",
            "name": "Drift Analysis Setup",
            "description": "Review data drift and concept drift monitoring plan",
            "config": {
                "agents": ["ml_engineer", "data_engineer"],
                "topic_template": "Review drift monitoring: {monitoring_plan}",
            },
        },
        {
            "id": "inference_review",
            "type": "debate",
            "name": "Inference Pipeline Review",
            "description": "Review inference latency, throughput, and reliability",
            "config": {
                "agents": ["ml_engineer", "devops_engineer", "sre"],
                "rounds": 2,
                "topic_template": "Review inference pipeline: {infrastructure_spec}",
            },
        },
        {
            "id": "rollback_plan",
            "type": "debate",
            "name": "Rollback Plan Review",
            "description": "Review model rollback and incident response plan",
            "config": {
                "agents": ["ml_engineer", "devops_engineer"],
                "topic_template": "Review rollback procedures",
            },
        },
        {
            "id": "bias_gate",
            "type": "decision",
            "name": "Bias Check Gate",
            "description": "Check if model passes fairness thresholds",
            "config": {
                "condition": "max_demographic_disparity <= 0.1",
                "true_target": "ml_lead_review",
                "false_target": "bias_remediation",
            },
        },
        {
            "id": "bias_remediation",
            "type": "debate",
            "name": "Bias Remediation Plan",
            "description": "Generate remediation recommendations",
            "config": {
                "agents": ["ai_ethics_specialist", "data_scientist"],
                "topic_template": "Generate bias remediation plan",
            },
        },
        {
            "id": "ml_lead_review",
            "type": "human_checkpoint",
            "name": "ML Lead Review",
            "description": "ML lead approval for production deployment",
            "config": {
                "approval_type": "sign_off",
                "required_role": "ml_lead",
                "checklist": [
                    "Performance meets SLA",
                    "Bias within acceptable limits",
                    "Monitoring configured",
                    "Rollback tested",
                ],
            },
        },
        {
            "id": "generate_model_card",
            "type": "task",
            "name": "Generate Deployment Model Card",
            "description": "Generate production model card with all findings",
            "config": {
                "task_type": "transform",
                "template": "model_card_v2",
            },
        },
        {
            "id": "store",
            "type": "memory_write",
            "name": "Store Deployment Record",
            "description": "Persist deployment review to knowledge base",
            "config": {
                "domain": "ml/deployments",
            },
        },
    ],
    "transitions": [
        {"from": "gather_artifacts", "to": "performance_review"},
        {"from": "performance_review", "to": "bias_audit"},
        {"from": "bias_audit", "to": "robustness_testing"},
        {"from": "robustness_testing", "to": "drift_analysis"},
        {"from": "drift_analysis", "to": "inference_review"},
        {"from": "inference_review", "to": "rollback_plan"},
        {"from": "rollback_plan", "to": "bias_gate"},
        {"from": "bias_gate", "to": "bias_remediation", "condition": "bias_detected"},
        {"from": "bias_gate", "to": "ml_lead_review", "condition": "no_bias"},
        {"from": "bias_remediation", "to": "gather_artifacts"},
        {"from": "ml_lead_review", "to": "generate_model_card"},
        {"from": "generate_model_card", "to": "store"},
    ],
}

AI_GOVERNANCE_TEMPLATE: Dict[str, Any] = {
    "name": "AI Governance Assessment",
    "description": "EU AI Act and responsible AI governance compliance audit",
    "category": "ai_ml",
    "version": "1.0",
    "tags": ["ai", "governance", "compliance", "ai_act"],
    "steps": [
        {
            "id": "risk_classification",
            "type": "debate",
            "name": "AI System Risk Classification",
            "description": "Classify AI system risk level per EU AI Act",
            "config": {
                "agents": ["ai_ethics_specialist", "legal_analyst", "compliance_officer"],
                "rounds": 2,
                "topic_template": "Classify risk level for: {system_description}",
            },
        },
        {
            "id": "prohibited_check",
            "type": "gauntlet",
            "name": "Prohibited Practices Check",
            "description": "Check for prohibited AI practices",
            "config": {
                "profile": "ai_act",
                "persona": "ai_act",
            },
        },
        {
            "id": "transparency_review",
            "type": "debate",
            "name": "Transparency Requirements Review",
            "description": "Review transparency and explainability measures",
            "config": {
                "agents": ["ai_ethics_specialist", "technical_writer"],
                "topic_template": "Review transparency measures: {documentation}",
            },
        },
        {
            "id": "human_oversight",
            "type": "debate",
            "name": "Human Oversight Review",
            "description": "Review human oversight mechanisms",
            "config": {
                "agents": ["ai_ethics_specialist", "ux_researcher"],
                "topic_template": "Review human oversight: {oversight_plan}",
            },
        },
        {
            "id": "data_governance",
            "type": "debate",
            "name": "Training Data Governance",
            "description": "Review training data governance and documentation",
            "config": {
                "agents": ["data_governance_specialist", "legal_analyst"],
                "topic_template": "Review data governance: {data_documentation}",
            },
        },
        {
            "id": "technical_docs",
            "type": "debate",
            "name": "Technical Documentation Review",
            "description": "Review conformity assessment documentation",
            "config": {
                "agents": ["technical_writer", "compliance_officer"],
                "topic_template": "Review technical docs for conformity",
            },
        },
        {
            "id": "high_risk_gate",
            "type": "decision",
            "name": "High-Risk System Gate",
            "description": "Route based on risk classification",
            "config": {
                "condition": "risk_level in ['high', 'unacceptable']",
                "true_target": "conformity_assessment",
                "false_target": "generate_report",
            },
        },
        {
            "id": "conformity_assessment",
            "type": "debate",
            "name": "Conformity Assessment",
            "description": "Full conformity assessment for high-risk systems",
            "config": {
                "agents": ["compliance_officer", "legal_analyst", "ai_ethics_specialist"],
                "rounds": 3,
                "topic_template": "Conduct conformity assessment",
            },
        },
        {
            "id": "dpo_review",
            "type": "human_checkpoint",
            "name": "DPO/Ethics Board Review",
            "description": "Data Protection Officer or Ethics Board approval",
            "config": {
                "approval_type": "sign_off",
                "required_role": "dpo",
                "checklist": [
                    "Risk classification verified",
                    "Transparency requirements met",
                    "Human oversight adequate",
                    "Technical documentation complete",
                ],
            },
        },
        {
            "id": "generate_report",
            "type": "task",
            "name": "Generate Governance Report",
            "description": "Generate AI governance compliance report",
            "config": {
                "task_type": "transform",
                "template": "ai_governance_report",
            },
        },
        {
            "id": "store",
            "type": "memory_write",
            "name": "Store Governance Record",
            "description": "Persist governance assessment",
            "config": {
                "domain": "compliance/ai_governance",
            },
        },
    ],
    "transitions": [
        {"from": "risk_classification", "to": "prohibited_check"},
        {"from": "prohibited_check", "to": "transparency_review"},
        {"from": "transparency_review", "to": "human_oversight"},
        {"from": "human_oversight", "to": "data_governance"},
        {"from": "data_governance", "to": "technical_docs"},
        {"from": "technical_docs", "to": "high_risk_gate"},
        {"from": "high_risk_gate", "to": "conformity_assessment", "condition": "high_risk"},
        {"from": "high_risk_gate", "to": "generate_report", "condition": "low_risk"},
        {"from": "conformity_assessment", "to": "dpo_review"},
        {"from": "dpo_review", "to": "generate_report"},
        {"from": "generate_report", "to": "store"},
    ],
}

PROMPT_ENGINEERING_TEMPLATE: Dict[str, Any] = {
    "name": "Prompt Engineering Review",
    "description": "Review and optimize prompts for LLM applications",
    "category": "ai_ml",
    "version": "1.0",
    "tags": ["ai", "prompts", "llm", "optimization"],
    "steps": [
        {
            "id": "gather_prompts",
            "type": "task",
            "name": "Gather Prompts",
            "description": "Collect prompts and expected outputs",
            "config": {
                "task_type": "aggregate",
                "sources": ["prompt_library", "test_cases"],
            },
        },
        {
            "id": "clarity_review",
            "type": "debate",
            "name": "Clarity Review",
            "description": "Review prompt clarity and specificity",
            "config": {
                "agents": ["claude", "gpt4", "gemini"],
                "rounds": 2,
                "topic_template": "Review prompt clarity: {prompt_content}",
            },
        },
        {
            "id": "injection_review",
            "type": "debate",
            "name": "Injection Vulnerability Review",
            "description": "Check for prompt injection vulnerabilities",
            "config": {
                "agents": ["security_engineer", "claude"],
                "topic_template": "Check injection risks: {prompt_content}",
            },
        },
        {
            "id": "output_format_review",
            "type": "debate",
            "name": "Output Format Review",
            "description": "Review output formatting and structure",
            "config": {
                "agents": ["claude", "technical_writer"],
                "topic_template": "Review output format specification",
            },
        },
        {
            "id": "edge_case_testing",
            "type": "gauntlet",
            "name": "Edge Case Testing",
            "description": "Test prompts against edge cases",
            "config": {
                "profile": "quick",
                "input_type": "spec",
            },
        },
        {
            "id": "optimization_suggestions",
            "type": "debate",
            "name": "Optimization Suggestions",
            "description": "Generate prompt optimization recommendations",
            "config": {
                "agents": ["claude", "gpt4", "gemini"],
                "rounds": 2,
                "topic_template": "Suggest optimizations: {review_findings}",
            },
        },
        {
            "id": "store",
            "type": "memory_write",
            "name": "Store Optimized Prompts",
            "description": "Persist optimized prompts to library",
            "config": {
                "domain": "ai/prompts",
            },
        },
    ],
    "transitions": [
        {"from": "gather_prompts", "to": "clarity_review"},
        {"from": "clarity_review", "to": "injection_review"},
        {"from": "injection_review", "to": "output_format_review"},
        {"from": "output_format_review", "to": "edge_case_testing"},
        {"from": "edge_case_testing", "to": "optimization_suggestions"},
        {"from": "optimization_suggestions", "to": "store"},
    ],
}
