"""
Product Workflow Templates.

Templates for PRD review, feature specs, and product decision-making.
"""

from typing import Any, Dict

PRD_REVIEW_TEMPLATE: Dict[str, Any] = {
    "name": "PRD Review Workflow",
    "description": "Multi-stakeholder Product Requirements Document review",
    "category": "product",
    "version": "1.0",
    "tags": ["product", "prd", "requirements", "review"],
    "steps": [
        {
            "id": "gather_prd",
            "type": "task",
            "name": "Gather PRD",
            "description": "Collect PRD and supporting documents",
            "config": {
                "task_type": "aggregate",
                "sources": ["prd", "user_research", "competitive_analysis"],
            },
        },
        {
            "id": "problem_review",
            "type": "debate",
            "name": "Problem Statement Review",
            "description": "Review problem definition and user needs",
            "config": {
                "agents": ["product_manager", "ux_researcher", "customer_success"],
                "rounds": 2,
                "topic_template": "Review problem statement: {problem_section}",
            },
        },
        {
            "id": "solution_review",
            "type": "debate",
            "name": "Solution Review",
            "description": "Review proposed solution and alternatives",
            "config": {
                "agents": ["product_manager", "architect", "designer"],
                "rounds": 2,
                "topic_template": "Review proposed solution: {solution_section}",
            },
        },
        {
            "id": "technical_feasibility",
            "type": "debate",
            "name": "Technical Feasibility Review",
            "description": "Engineering assessment of feasibility",
            "config": {
                "agents": ["architect", "tech_lead", "devops_engineer"],
                "rounds": 2,
                "topic_template": "Assess technical feasibility: {solution_section}",
            },
        },
        {
            "id": "success_metrics",
            "type": "debate",
            "name": "Success Metrics Review",
            "description": "Review KPIs and success criteria",
            "config": {
                "agents": ["product_manager", "data_analyst", "business_analyst"],
                "topic_template": "Review success metrics: {metrics_section}",
            },
        },
        {
            "id": "risk_assessment",
            "type": "gauntlet",
            "name": "Risk Assessment",
            "description": "Adversarial review of PRD risks",
            "config": {
                "profile": "quick",
                "input_type": "spec",
            },
        },
        {
            "id": "scope_review",
            "type": "debate",
            "name": "Scope and Timeline Review",
            "description": "Review scope, timeline, and resource requirements",
            "config": {
                "agents": ["product_manager", "tech_lead", "project_manager"],
                "topic_template": "Review scope and timeline: {scope_section}",
            },
        },
        {
            "id": "stakeholder_alignment",
            "type": "human_checkpoint",
            "name": "Stakeholder Alignment",
            "description": "Cross-functional stakeholder sign-off",
            "config": {
                "approval_type": "multi_sign_off",
                "required_roles": ["product_lead", "engineering_lead", "design_lead"],
                "checklist": [
                    "Problem clearly defined",
                    "Solution addresses user needs",
                    "Technical feasibility confirmed",
                    "Success metrics measurable",
                    "Timeline realistic",
                ],
            },
        },
        {
            "id": "generate_summary",
            "type": "task",
            "name": "Generate PRD Summary",
            "description": "Generate executive summary and action items",
            "config": {
                "task_type": "transform",
                "template": "prd_review_summary",
            },
        },
        {
            "id": "store",
            "type": "memory_write",
            "name": "Store PRD Review",
            "description": "Persist PRD review to knowledge base",
            "config": {
                "domain": "product/prds",
            },
        },
    ],
    "transitions": [
        {"from": "gather_prd", "to": "problem_review"},
        {"from": "problem_review", "to": "solution_review"},
        {"from": "solution_review", "to": "technical_feasibility"},
        {"from": "technical_feasibility", "to": "success_metrics"},
        {"from": "success_metrics", "to": "risk_assessment"},
        {"from": "risk_assessment", "to": "scope_review"},
        {"from": "scope_review", "to": "stakeholder_alignment"},
        {"from": "stakeholder_alignment", "to": "generate_summary"},
        {"from": "generate_summary", "to": "store"},
    ],
}

FEATURE_SPEC_TEMPLATE: Dict[str, Any] = {
    "name": "Feature Specification Review",
    "description": "Technical feature specification review workflow",
    "category": "product",
    "version": "1.0",
    "tags": ["product", "feature", "specification", "engineering"],
    "steps": [
        {
            "id": "gather_spec",
            "type": "task",
            "name": "Gather Specification",
            "description": "Collect feature spec and designs",
            "config": {
                "task_type": "aggregate",
                "sources": ["feature_spec", "design_mocks", "api_spec"],
            },
        },
        {
            "id": "requirements_review",
            "type": "debate",
            "name": "Requirements Clarity",
            "description": "Review requirements for clarity and completeness",
            "config": {
                "agents": ["tech_lead", "qa_engineer", "product_manager"],
                "rounds": 2,
                "topic_template": "Review requirements clarity: {requirements}",
            },
        },
        {
            "id": "api_design_review",
            "type": "debate",
            "name": "API Design Review",
            "description": "Review API contracts and interfaces",
            "config": {
                "agents": ["api_design_reviewer", "architect", "security_engineer"],
                "rounds": 2,
                "topic_template": "Review API design: {api_spec}",
            },
        },
        {
            "id": "data_model_review",
            "type": "debate",
            "name": "Data Model Review",
            "description": "Review data models and schema changes",
            "config": {
                "agents": ["data_architect", "tech_lead"],
                "topic_template": "Review data model: {data_model}",
            },
        },
        {
            "id": "security_review",
            "type": "debate",
            "name": "Security Review",
            "description": "Security implications of feature",
            "config": {
                "agents": ["security_engineer", "tech_lead"],
                "topic_template": "Review security implications: {feature_spec}",
            },
        },
        {
            "id": "test_strategy",
            "type": "debate",
            "name": "Test Strategy Review",
            "description": "Review testing approach and coverage",
            "config": {
                "agents": ["qa_engineer", "tech_lead"],
                "topic_template": "Review test strategy: {test_plan}",
            },
        },
        {
            "id": "tech_lead_approval",
            "type": "human_checkpoint",
            "name": "Tech Lead Approval",
            "description": "Technical lead sign-off",
            "config": {
                "approval_type": "sign_off",
                "required_role": "tech_lead",
                "checklist": [
                    "Requirements complete and testable",
                    "API design follows standards",
                    "Data model reviewed",
                    "Security considered",
                    "Test coverage adequate",
                ],
            },
        },
        {
            "id": "create_tasks",
            "type": "task",
            "name": "Create Engineering Tasks",
            "description": "Break down spec into engineering tasks",
            "config": {
                "task_type": "http",
                "endpoint": "/api/tickets/create-batch",
            },
        },
        {
            "id": "store",
            "type": "memory_write",
            "name": "Store Feature Spec",
            "description": "Persist feature spec to knowledge base",
            "config": {
                "domain": "product/features",
            },
        },
    ],
    "transitions": [
        {"from": "gather_spec", "to": "requirements_review"},
        {"from": "requirements_review", "to": "api_design_review"},
        {"from": "api_design_review", "to": "data_model_review"},
        {"from": "data_model_review", "to": "security_review"},
        {"from": "security_review", "to": "test_strategy"},
        {"from": "test_strategy", "to": "tech_lead_approval"},
        {"from": "tech_lead_approval", "to": "create_tasks"},
        {"from": "create_tasks", "to": "store"},
    ],
}

USER_RESEARCH_TEMPLATE: Dict[str, Any] = {
    "name": "User Research Analysis",
    "description": "Analyze user research findings with multi-agent synthesis",
    "category": "product",
    "version": "1.0",
    "tags": ["product", "research", "user", "analysis"],
    "steps": [
        {
            "id": "gather_research",
            "type": "task",
            "name": "Gather Research Data",
            "description": "Collect interview transcripts, survey data, analytics",
            "config": {
                "task_type": "aggregate",
                "sources": ["interviews", "surveys", "analytics", "feedback"],
            },
        },
        {
            "id": "pattern_identification",
            "type": "debate",
            "name": "Pattern Identification",
            "description": "Identify patterns and themes in research data",
            "config": {
                "agents": ["ux_researcher", "data_analyst", "claude"],
                "rounds": 3,
                "topic_template": "Identify patterns: {research_data}",
            },
        },
        {
            "id": "persona_synthesis",
            "type": "debate",
            "name": "Persona Synthesis",
            "description": "Synthesize or update user personas",
            "config": {
                "agents": ["ux_researcher", "product_manager"],
                "rounds": 2,
                "topic_template": "Synthesize personas: {patterns}",
            },
        },
        {
            "id": "journey_mapping",
            "type": "debate",
            "name": "Journey Mapping",
            "description": "Map user journeys and pain points",
            "config": {
                "agents": ["ux_researcher", "designer", "customer_success"],
                "topic_template": "Map user journey: {research_data}",
            },
        },
        {
            "id": "opportunity_identification",
            "type": "debate",
            "name": "Opportunity Identification",
            "description": "Identify product opportunities from research",
            "config": {
                "agents": ["product_manager", "ux_researcher", "business_analyst"],
                "rounds": 2,
                "topic_template": "Identify opportunities: {patterns} {journeys}",
            },
        },
        {
            "id": "prioritization",
            "type": "debate",
            "name": "Opportunity Prioritization",
            "description": "Prioritize opportunities by impact and feasibility",
            "config": {
                "agents": ["product_manager", "architect", "business_analyst"],
                "topic_template": "Prioritize opportunities",
            },
        },
        {
            "id": "pm_review",
            "type": "human_checkpoint",
            "name": "PM Review",
            "description": "Product manager synthesis review",
            "config": {
                "approval_type": "review",
                "required_role": "product_manager",
            },
        },
        {
            "id": "generate_insights",
            "type": "task",
            "name": "Generate Insights Report",
            "description": "Generate research insights document",
            "config": {
                "task_type": "transform",
                "template": "research_insights_report",
            },
        },
        {
            "id": "store",
            "type": "memory_write",
            "name": "Store Research",
            "description": "Persist research to knowledge base",
            "config": {
                "domain": "product/research",
            },
        },
    ],
    "transitions": [
        {"from": "gather_research", "to": "pattern_identification"},
        {"from": "pattern_identification", "to": "persona_synthesis"},
        {"from": "persona_synthesis", "to": "journey_mapping"},
        {"from": "journey_mapping", "to": "opportunity_identification"},
        {"from": "opportunity_identification", "to": "prioritization"},
        {"from": "prioritization", "to": "pm_review"},
        {"from": "pm_review", "to": "generate_insights"},
        {"from": "generate_insights", "to": "store"},
    ],
}

LAUNCH_READINESS_TEMPLATE: Dict[str, Any] = {
    "name": "Launch Readiness Review",
    "description": "Pre-launch readiness assessment across all functions",
    "category": "product",
    "version": "1.0",
    "tags": ["product", "launch", "readiness", "go-to-market"],
    "steps": [
        {
            "id": "gather_status",
            "type": "task",
            "name": "Gather Launch Status",
            "description": "Collect status from all workstreams",
            "config": {
                "task_type": "aggregate",
                "sources": [
                    "engineering_status",
                    "qa_status",
                    "marketing_status",
                    "support_status",
                ],
            },
        },
        {
            "id": "engineering_readiness",
            "type": "debate",
            "name": "Engineering Readiness",
            "description": "Assess engineering completion and quality",
            "config": {
                "agents": ["tech_lead", "qa_engineer", "devops_engineer"],
                "topic_template": "Assess engineering readiness: {engineering_status}",
            },
        },
        {
            "id": "operations_readiness",
            "type": "debate",
            "name": "Operations Readiness",
            "description": "Assess infrastructure and operations readiness",
            "config": {
                "agents": ["devops_engineer", "sre", "security_engineer"],
                "topic_template": "Assess operations readiness: {ops_status}",
            },
        },
        {
            "id": "support_readiness",
            "type": "debate",
            "name": "Support Readiness",
            "description": "Assess customer support readiness",
            "config": {
                "agents": ["customer_success", "product_manager"],
                "topic_template": "Assess support readiness: {support_status}",
            },
        },
        {
            "id": "marketing_readiness",
            "type": "debate",
            "name": "Marketing Readiness",
            "description": "Assess go-to-market readiness",
            "config": {
                "agents": ["product_marketing", "product_manager"],
                "topic_template": "Assess marketing readiness: {marketing_status}",
            },
        },
        {
            "id": "risk_review",
            "type": "gauntlet",
            "name": "Launch Risk Review",
            "description": "Adversarial review of launch risks",
            "config": {
                "profile": "policy",
                "input_type": "strategy",
            },
        },
        {
            "id": "go_no_go_gate",
            "type": "decision",
            "name": "Go/No-Go Gate",
            "description": "Determine launch readiness",
            "config": {
                "condition": "all_green == True",
                "true_target": "exec_approval",
                "false_target": "blocker_resolution",
            },
        },
        {
            "id": "blocker_resolution",
            "type": "debate",
            "name": "Blocker Resolution Plan",
            "description": "Plan to address launch blockers",
            "config": {
                "agents": ["product_manager", "tech_lead", "project_manager"],
                "topic_template": "Plan blocker resolution: {blockers}",
            },
        },
        {
            "id": "exec_approval",
            "type": "human_checkpoint",
            "name": "Executive Approval",
            "description": "Executive sign-off for launch",
            "config": {
                "approval_type": "sign_off",
                "required_role": "executive",
                "checklist": [
                    "Engineering complete and tested",
                    "Operations ready",
                    "Support trained",
                    "Marketing prepared",
                    "No critical risks",
                ],
            },
        },
        {
            "id": "generate_checklist",
            "type": "task",
            "name": "Generate Launch Checklist",
            "description": "Generate final launch checklist",
            "config": {
                "task_type": "transform",
                "template": "launch_checklist",
            },
        },
        {
            "id": "store",
            "type": "memory_write",
            "name": "Store Launch Record",
            "description": "Persist launch record",
            "config": {
                "domain": "product/launches",
            },
        },
    ],
    "transitions": [
        {"from": "gather_status", "to": "engineering_readiness"},
        {"from": "engineering_readiness", "to": "operations_readiness"},
        {"from": "operations_readiness", "to": "support_readiness"},
        {"from": "support_readiness", "to": "marketing_readiness"},
        {"from": "marketing_readiness", "to": "risk_review"},
        {"from": "risk_review", "to": "go_no_go_gate"},
        {"from": "go_no_go_gate", "to": "blocker_resolution", "condition": "blockers_exist"},
        {"from": "go_no_go_gate", "to": "exec_approval", "condition": "ready_to_launch"},
        {"from": "blocker_resolution", "to": "gather_status"},
        {"from": "exec_approval", "to": "generate_checklist"},
        {"from": "generate_checklist", "to": "store"},
    ],
}
