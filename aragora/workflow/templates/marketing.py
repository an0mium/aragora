"""
Marketing and Advertising Workflow Templates.

Templates for:
- Ad performance review across platforms
- Lead-to-CRM sync workflows
- Cross-platform analytics
- Budget optimization
"""

from typing import Any

AD_PERFORMANCE_REVIEW_TEMPLATE: dict[str, Any] = {
    "name": "Ad Performance Review",
    "description": "Multi-agent analysis of advertising performance across platforms",
    "category": "marketing",
    "version": "1.0",
    "tags": ["advertising", "marketing", "analytics", "optimization"],
    "inputs": {
        "platforms": {
            "type": "list",
            "description": "Advertising platforms to analyze",
            "default": ["google_ads", "meta_ads", "linkedin_ads", "microsoft_ads"],
        },
        "date_range_days": {
            "type": "integer",
            "description": "Number of days to analyze",
            "default": 30,
        },
        "budget": {
            "type": "number",
            "description": "Monthly advertising budget",
            "required": False,
        },
    },
    "steps": [
        {
            "id": "data_collection",
            "type": "parallel",
            "name": "Collect Performance Data",
            "description": "Gather performance data from all connected platforms",
            "branches": [
                {
                    "id": "google_ads_data",
                    "steps": [
                        {
                            "id": "fetch_google_metrics",
                            "type": "connector",
                            "name": "Fetch Google Ads Metrics",
                            "config": {
                                "connector": "advertising",
                                "platform": "google_ads",
                                "action": "get_performance",
                                "params": {"days": "{date_range_days}"},
                            },
                        },
                    ],
                },
                {
                    "id": "meta_ads_data",
                    "steps": [
                        {
                            "id": "fetch_meta_metrics",
                            "type": "connector",
                            "name": "Fetch Meta Ads Metrics",
                            "config": {
                                "connector": "advertising",
                                "platform": "meta_ads",
                                "action": "get_insights",
                                "params": {"date_preset": "last_30d"},
                            },
                        },
                    ],
                },
                {
                    "id": "linkedin_ads_data",
                    "steps": [
                        {
                            "id": "fetch_linkedin_metrics",
                            "type": "connector",
                            "name": "Fetch LinkedIn Ads Metrics",
                            "config": {
                                "connector": "advertising",
                                "platform": "linkedin_ads",
                                "action": "get_analytics",
                                "params": {"days": "{date_range_days}"},
                            },
                        },
                    ],
                },
                {
                    "id": "microsoft_ads_data",
                    "steps": [
                        {
                            "id": "fetch_microsoft_metrics",
                            "type": "connector",
                            "name": "Fetch Microsoft Ads Metrics",
                            "config": {
                                "connector": "advertising",
                                "platform": "microsoft_ads",
                                "action": "get_performance",
                                "params": {"days": "{date_range_days}"},
                            },
                        },
                    ],
                },
            ],
        },
        {
            "id": "performance_analysis",
            "type": "debate",
            "name": "Performance Analysis",
            "description": "Multi-agent analysis of advertising performance metrics",
            "config": {
                "agents": ["marketing_analyst", "data_scientist", "cfo"],
                "rounds": 3,
                "topic_template": """
                    Analyze advertising performance across platforms:
                    - Google Ads: {google_ads_data}
                    - Meta Ads: {meta_ads_data}
                    - LinkedIn Ads: {linkedin_ads_data}
                    - Microsoft Ads: {microsoft_ads_data}

                    Evaluate: ROAS, CPA, CTR, conversion rates, and spend efficiency.
                    Budget context: {budget}
                """,
            },
        },
        {
            "id": "audience_analysis",
            "type": "debate",
            "name": "Audience Performance Analysis",
            "description": "Analyze which audiences are performing best",
            "config": {
                "agents": ["marketing_analyst", "customer_insights"],
                "rounds": 2,
                "topic_template": """
                    Based on the performance data, analyze audience segments:
                    - Which demographics are converting best?
                    - What targeting criteria should be expanded/reduced?
                    - Are there audience overlaps causing inefficiency?
                """,
            },
        },
        {
            "id": "budget_recommendations",
            "type": "debate",
            "name": "Budget Allocation Recommendations",
            "description": "Recommend optimal budget distribution",
            "config": {
                "agents": ["marketing_analyst", "cfo", "growth_strategist"],
                "rounds": 2,
                "topic_template": """
                    Based on performance analysis, recommend budget allocation:
                    - Current budget: {budget}
                    - Platform performance: {performance_analysis}
                    - Audience insights: {audience_analysis}

                    Provide specific percentage allocations for each platform.
                """,
            },
        },
        {
            "id": "creative_recommendations",
            "type": "debate",
            "name": "Creative Recommendations",
            "description": "Recommend creative optimizations",
            "config": {
                "agents": ["creative_director", "marketing_analyst"],
                "rounds": 2,
                "topic_template": """
                    Based on performance data, recommend creative improvements:
                    - Which ad formats are performing best?
                    - What messaging resonates with target audiences?
                    - Are there creative fatigue issues?
                """,
            },
        },
        {
            "id": "generate_report",
            "type": "task",
            "name": "Generate Performance Report",
            "description": "Compile final performance report with recommendations",
            "config": {
                "task_type": "synthesize",
                "output_format": "report",
                "sections": [
                    "executive_summary",
                    "platform_performance",
                    "audience_insights",
                    "budget_recommendations",
                    "creative_recommendations",
                    "action_items",
                ],
            },
        },
    ],
    "outputs": {
        "report": "Performance analysis report",
        "recommendations": "List of actionable recommendations",
        "budget_allocation": "Recommended budget distribution by platform",
    },
}


LEAD_TO_CRM_SYNC_TEMPLATE: dict[str, Any] = {
    "name": "Lead-to-CRM Sync",
    "description": "Sync leads from advertising platforms to CRM with enrichment",
    "category": "marketing",
    "version": "1.0",
    "tags": ["crm", "leads", "advertising", "sync"],
    "inputs": {
        "source_platform": {
            "type": "string",
            "description": "Source advertising platform",
            "enum": ["linkedin_ads", "meta_ads", "google_ads"],
        },
        "target_crm": {
            "type": "string",
            "description": "Target CRM platform",
            "default": "hubspot",
        },
        "enrich_data": {
            "type": "boolean",
            "description": "Whether to enrich lead data",
            "default": True,
        },
    },
    "steps": [
        {
            "id": "fetch_leads",
            "type": "connector",
            "name": "Fetch New Leads",
            "description": "Get new leads from advertising platform",
            "config": {
                "connector": "advertising",
                "platform": "{source_platform}",
                "action": "get_leads",
                "params": {"since_hours": 24},
            },
        },
        {
            "id": "deduplicate",
            "type": "task",
            "name": "Deduplicate Leads",
            "description": "Check for existing contacts in CRM",
            "config": {
                "task_type": "transform",
                "operations": [
                    {"type": "lookup", "target": "{target_crm}", "key": "email"},
                    {"type": "filter", "condition": "not_exists"},
                ],
            },
        },
        {
            "id": "enrich_leads",
            "type": "conditional",
            "name": "Enrich Lead Data",
            "description": "Optionally enrich lead data",
            "condition": "{enrich_data}",
            "then_steps": [
                {
                    "id": "enrichment",
                    "type": "connector",
                    "name": "Enrich with External Data",
                    "config": {
                        "connector": "crm",
                        "action": "enrich",
                        "params": {"leads": "{deduplicate}"},
                    },
                },
            ],
            "else_steps": [],
        },
        {
            "id": "qualify_leads",
            "type": "debate",
            "name": "Lead Qualification",
            "description": "AI-powered lead scoring and qualification",
            "config": {
                "agents": ["sales_analyst", "marketing_analyst"],
                "rounds": 2,
                "topic_template": """
                    Score and qualify these leads:
                    {enriched_leads}

                    Consider:
                    - Company size and industry fit
                    - Engagement signals from ad interaction
                    - Lead completeness and data quality
                    - ICP (Ideal Customer Profile) match
                """,
            },
        },
        {
            "id": "create_contacts",
            "type": "connector",
            "name": "Create CRM Contacts",
            "description": "Create contacts in target CRM",
            "config": {
                "connector": "crm",
                "platform": "{target_crm}",
                "action": "sync_lead",
                "params": {
                    "leads": "{qualify_leads}",
                    "source": "{source_platform}",
                },
            },
        },
        {
            "id": "assign_leads",
            "type": "task",
            "name": "Assign to Sales",
            "description": "Route leads to appropriate sales reps",
            "config": {
                "task_type": "route",
                "routing_rules": [
                    {"condition": "score >= 80", "action": "assign_hot_lead"},
                    {"condition": "score >= 50", "action": "assign_warm_lead"},
                    {"condition": "default", "action": "nurture_sequence"},
                ],
            },
        },
        {
            "id": "notify",
            "type": "task",
            "name": "Send Notifications",
            "description": "Notify relevant team members",
            "config": {
                "task_type": "notify",
                "channels": ["slack", "email"],
                "template": "new_leads_synced",
            },
        },
    ],
    "outputs": {
        "leads_synced": "Number of leads synced to CRM",
        "leads_qualified": "Number of qualified leads",
        "assignments": "Lead assignment summary",
    },
}


CROSS_PLATFORM_ANALYTICS_TEMPLATE: dict[str, Any] = {
    "name": "Cross-Platform Analytics",
    "description": "Unified analytics view across marketing and analytics platforms",
    "category": "analytics",
    "version": "1.0",
    "tags": ["analytics", "reporting", "marketing", "bi"],
    "inputs": {
        "analytics_platforms": {
            "type": "list",
            "description": "Analytics platforms to include",
            "default": ["google_analytics", "mixpanel"],
        },
        "advertising_platforms": {
            "type": "list",
            "description": "Ad platforms to include",
            "default": ["google_ads", "meta_ads"],
        },
        "date_range_days": {
            "type": "integer",
            "default": 30,
        },
    },
    "steps": [
        {
            "id": "collect_analytics",
            "type": "parallel",
            "name": "Collect Analytics Data",
            "branches": [
                {
                    "id": "web_analytics",
                    "steps": [
                        {
                            "id": "ga4_data",
                            "type": "connector",
                            "name": "Fetch GA4 Data",
                            "config": {
                                "connector": "analytics",
                                "platform": "google_analytics",
                                "action": "get_report",
                                "params": {
                                    "metrics": ["sessions", "users", "conversions"],
                                    "dimensions": ["source", "medium", "campaign"],
                                },
                            },
                        },
                    ],
                },
                {
                    "id": "product_analytics",
                    "steps": [
                        {
                            "id": "mixpanel_data",
                            "type": "connector",
                            "name": "Fetch Mixpanel Data",
                            "config": {
                                "connector": "analytics",
                                "platform": "mixpanel",
                                "action": "get_insights",
                                "params": {"event": "conversion"},
                            },
                        },
                    ],
                },
                {
                    "id": "ad_performance",
                    "steps": [
                        {
                            "id": "advertising_data",
                            "type": "connector",
                            "name": "Fetch Ad Performance",
                            "config": {
                                "connector": "advertising",
                                "action": "get_cross_platform_performance",
                                "params": {"days": "{date_range_days}"},
                            },
                        },
                    ],
                },
            ],
        },
        {
            "id": "attribution_analysis",
            "type": "debate",
            "name": "Attribution Analysis",
            "description": "Analyze cross-channel attribution",
            "config": {
                "agents": ["marketing_analyst", "data_scientist"],
                "rounds": 2,
                "topic_template": """
                    Analyze attribution across channels:
                    - Web Analytics: {ga4_data}
                    - Product Analytics: {mixpanel_data}
                    - Ad Performance: {advertising_data}

                    Questions to answer:
                    1. What channels drive the most conversions?
                    2. What is the typical customer journey?
                    3. Where are there attribution gaps?
                """,
            },
        },
        {
            "id": "funnel_analysis",
            "type": "debate",
            "name": "Funnel Analysis",
            "description": "Analyze conversion funnel across platforms",
            "config": {
                "agents": ["growth_strategist", "product_analyst"],
                "rounds": 2,
                "topic_template": """
                    Analyze the full conversion funnel:
                    - Traffic sources and quality
                    - Landing page performance
                    - User activation metrics
                    - Conversion bottlenecks
                """,
            },
        },
        {
            "id": "roi_calculation",
            "type": "task",
            "name": "Calculate ROI",
            "description": "Calculate ROI by channel and campaign",
            "config": {
                "task_type": "calculate",
                "calculations": [
                    {"name": "channel_roi", "formula": "(revenue - cost) / cost * 100"},
                    {"name": "cac", "formula": "total_spend / new_customers"},
                    {"name": "ltv_cac_ratio", "formula": "ltv / cac"},
                ],
            },
        },
        {
            "id": "generate_dashboard",
            "type": "task",
            "name": "Generate Dashboard",
            "description": "Create unified analytics dashboard",
            "config": {
                "task_type": "visualize",
                "output_format": "dashboard",
                "widgets": [
                    {"type": "kpi", "metrics": ["sessions", "conversions", "revenue"]},
                    {"type": "chart", "data": "channel_performance"},
                    {"type": "table", "data": "campaign_breakdown"},
                    {"type": "funnel", "data": "conversion_funnel"},
                ],
            },
        },
    ],
    "outputs": {
        "dashboard": "Unified analytics dashboard",
        "attribution_report": "Cross-channel attribution report",
        "roi_analysis": "ROI analysis by channel",
    },
}


SUPPORT_TICKET_TRIAGE_TEMPLATE: dict[str, Any] = {
    "name": "Support Ticket Triage",
    "description": "AI-powered support ticket triage and routing",
    "category": "support",
    "version": "1.0",
    "tags": ["support", "triage", "customer-service", "automation"],
    "inputs": {
        "platforms": {
            "type": "list",
            "description": "Support platforms to monitor",
            "default": ["zendesk", "freshdesk"],
        },
        "auto_respond": {
            "type": "boolean",
            "description": "Enable auto-response for simple queries",
            "default": False,
        },
    },
    "steps": [
        {
            "id": "fetch_tickets",
            "type": "connector",
            "name": "Fetch Open Tickets",
            "description": "Get unassigned/new tickets from all platforms",
            "config": {
                "connector": "support",
                "action": "get_tickets",
                "params": {"status": "open", "limit": 100},
            },
        },
        {
            "id": "categorize_tickets",
            "type": "debate",
            "name": "Categorize Tickets",
            "description": "AI-powered ticket categorization",
            "config": {
                "agents": ["support_analyst", "product_expert"],
                "rounds": 2,
                "topic_template": """
                    Categorize these support tickets:
                    {tickets}

                    Categories:
                    - billing: Payment, invoices, refunds
                    - technical: Bugs, errors, integrations
                    - account: Login, permissions, settings
                    - feature: Requests, suggestions
                    - general: Other inquiries
                """,
            },
        },
        {
            "id": "prioritize_tickets",
            "type": "debate",
            "name": "Prioritize Tickets",
            "description": "Determine ticket priority and urgency",
            "config": {
                "agents": ["support_manager", "customer_success"],
                "rounds": 2,
                "topic_template": """
                    Prioritize these tickets based on:
                    - Customer tier/value
                    - Issue severity
                    - SLA requirements
                    - Sentiment analysis

                    Tickets: {categorize_tickets}
                """,
            },
        },
        {
            "id": "suggest_responses",
            "type": "conditional",
            "name": "Generate Response Suggestions",
            "condition": "{auto_respond}",
            "then_steps": [
                {
                    "id": "generate_responses",
                    "type": "connector",
                    "name": "Generate AI Responses",
                    "config": {
                        "connector": "support",
                        "action": "auto_respond",
                        "params": {"tickets": "{prioritize_tickets}"},
                    },
                },
            ],
        },
        {
            "id": "route_tickets",
            "type": "task",
            "name": "Route to Agents",
            "description": "Assign tickets to appropriate agents/teams",
            "config": {
                "task_type": "route",
                "routing_rules": [
                    {"condition": "priority == 'urgent'", "action": "assign_senior"},
                    {"condition": "category == 'billing'", "action": "assign_billing_team"},
                    {"condition": "category == 'technical'", "action": "assign_tech_team"},
                    {"condition": "default", "action": "round_robin"},
                ],
            },
        },
        {
            "id": "update_tickets",
            "type": "connector",
            "name": "Update Ticket Status",
            "description": "Update tickets with triage results",
            "config": {
                "connector": "support",
                "action": "update_tickets",
                "params": {
                    "tickets": "{route_tickets}",
                    "add_tags": ["ai_triaged"],
                },
            },
        },
        {
            "id": "notify_teams",
            "type": "task",
            "name": "Notify Teams",
            "description": "Alert teams about high-priority tickets",
            "config": {
                "task_type": "notify",
                "conditions": {"priority": ["urgent", "high"]},
                "channels": ["slack"],
            },
        },
    ],
    "outputs": {
        "tickets_triaged": "Number of tickets processed",
        "priority_breakdown": "Tickets by priority",
        "routing_summary": "Assignment summary",
    },
}


ECOMMERCE_ORDER_SYNC_TEMPLATE: dict[str, Any] = {
    "name": "E-commerce Order Sync",
    "description": "Sync orders across e-commerce platforms and accounting",
    "category": "ecommerce",
    "version": "1.0",
    "tags": ["ecommerce", "orders", "accounting", "sync"],
    "inputs": {
        "source_platforms": {
            "type": "list",
            "description": "E-commerce platforms to sync from",
            "default": ["shopify", "walmart"],
        },
        "accounting_platform": {
            "type": "string",
            "description": "Target accounting platform",
            "default": "xero",
        },
        "create_invoices": {
            "type": "boolean",
            "description": "Auto-create invoices",
            "default": True,
        },
    },
    "steps": [
        {
            "id": "fetch_orders",
            "type": "connector",
            "name": "Fetch New Orders",
            "description": "Get orders from e-commerce platforms",
            "config": {
                "connector": "ecommerce",
                "action": "get_orders",
                "params": {
                    "status": "paid",
                    "synced": False,
                    "days": 1,
                },
            },
        },
        {
            "id": "validate_orders",
            "type": "task",
            "name": "Validate Order Data",
            "description": "Ensure order data is complete and valid",
            "config": {
                "task_type": "validate",
                "validation_rules": [
                    "customer_email_present",
                    "shipping_address_complete",
                    "line_items_present",
                    "total_matches_line_items",
                ],
            },
        },
        {
            "id": "map_to_accounting",
            "type": "task",
            "name": "Map to Accounting Format",
            "description": "Transform orders to accounting entries",
            "config": {
                "task_type": "transform",
                "mappings": {
                    "customer": "contact",
                    "line_items": "invoice_lines",
                    "total": "invoice_total",
                    "tax": "tax_amount",
                    "shipping": "shipping_line",
                },
            },
        },
        {
            "id": "create_invoices",
            "type": "conditional",
            "name": "Create Invoices",
            "condition": "{create_invoices}",
            "then_steps": [
                {
                    "id": "invoice_creation",
                    "type": "connector",
                    "name": "Create Accounting Invoices",
                    "config": {
                        "connector": "accounting",
                        "platform": "{accounting_platform}",
                        "action": "create_invoice",
                        "params": {"data": "{map_to_accounting}"},
                    },
                },
            ],
        },
        {
            "id": "reconcile",
            "type": "debate",
            "name": "Reconciliation Check",
            "description": "Verify sync completeness",
            "config": {
                "agents": ["accountant", "operations"],
                "rounds": 1,
                "topic_template": """
                    Verify order sync reconciliation:
                    - Orders fetched: {fetch_orders}
                    - Invoices created: {create_invoices}

                    Flag any discrepancies.
                """,
            },
        },
        {
            "id": "mark_synced",
            "type": "task",
            "name": "Mark Orders Synced",
            "description": "Update sync status on source platforms",
            "config": {
                "task_type": "update",
                "field": "synced_to_accounting",
                "value": True,
            },
        },
    ],
    "outputs": {
        "orders_synced": "Number of orders synced",
        "invoices_created": "Number of invoices created",
        "reconciliation_status": "Reconciliation report",
    },
}


# Export all templates
MARKETING_TEMPLATES = {
    "ad_performance_review": AD_PERFORMANCE_REVIEW_TEMPLATE,
    "lead_to_crm_sync": LEAD_TO_CRM_SYNC_TEMPLATE,
    "cross_platform_analytics": CROSS_PLATFORM_ANALYTICS_TEMPLATE,
    "support_ticket_triage": SUPPORT_TICKET_TRIAGE_TEMPLATE,
    "ecommerce_order_sync": ECOMMERCE_ORDER_SYNC_TEMPLATE,
}

__all__ = [
    "AD_PERFORMANCE_REVIEW_TEMPLATE",
    "LEAD_TO_CRM_SYNC_TEMPLATE",
    "CROSS_PLATFORM_ANALYTICS_TEMPLATE",
    "SUPPORT_TICKET_TRIAGE_TEMPLATE",
    "ECOMMERCE_ORDER_SYNC_TEMPLATE",
    "MARKETING_TEMPLATES",
]
