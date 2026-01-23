"""
Email Management Workflow Templates.

Templates for automated email processing workflows:
- Inbox triage and categorization
- Follow-up tracking and reminders
- Snooze management
- Priority-based routing
"""

from typing import Any

# Email categorization workflow
EMAIL_CATEGORIZATION_TEMPLATE: dict[str, Any] = {
    "name": "Email Categorization Workflow",
    "description": "Automatically categorize and label incoming emails",
    "category": "email",
    "version": "1.0",
    "tags": ["email", "categorization", "inbox", "automation"],
    "steps": [
        {
            "id": "fetch_uncategorized",
            "type": "task",
            "name": "Fetch Uncategorized Emails",
            "description": "Get emails without category labels",
            "config": {
                "task_type": "function",
                "function_name": "fetch_uncategorized_emails",
                "inputs": ["user_id", "limit"],
                "limit": 50,
            },
        },
        {
            "id": "categorize_batch",
            "type": "task",
            "name": "Categorize Emails",
            "description": "Run categorization on batch of emails",
            "config": {
                "task_type": "function",
                "function_name": "categorize_email_batch",
                "inputs": ["emails"],
            },
        },
        {
            "id": "review_low_confidence",
            "type": "debate",
            "name": "Review Low Confidence",
            "description": "Multi-agent review of low confidence categorizations",
            "config": {
                "agents": ["categorization_agent", "context_agent"],
                "rounds": 2,
                "topic_template": "Review categorization for: {email_subject}. Predicted: {category}, confidence: {confidence}",
                "confidence_threshold": 0.6,
            },
        },
        {
            "id": "apply_labels",
            "type": "task",
            "name": "Apply Gmail Labels",
            "description": "Apply category labels to emails in Gmail",
            "config": {
                "task_type": "function",
                "function_name": "apply_category_labels",
                "create_labels_if_missing": True,
            },
        },
        {
            "id": "update_sender_history",
            "type": "task",
            "name": "Update Sender History",
            "description": "Record sender categorization patterns",
            "config": {
                "task_type": "function",
                "function_name": "update_sender_category_patterns",
            },
        },
    ],
    "transitions": [
        {"from": "fetch_uncategorized", "to": "categorize_batch"},
        {"from": "categorize_batch", "to": "review_low_confidence"},
        {"from": "review_low_confidence", "to": "apply_labels"},
        {"from": "apply_labels", "to": "update_sender_history"},
    ],
}


# Follow-up tracking workflow
FOLLOWUP_TRACKING_TEMPLATE: dict[str, Any] = {
    "name": "Follow-Up Tracking Workflow",
    "description": "Track sent emails and manage follow-up reminders",
    "category": "email",
    "version": "1.0",
    "tags": ["email", "followup", "tracking", "reminders"],
    "steps": [
        {
            "id": "scan_sent_folder",
            "type": "task",
            "name": "Scan Sent Folder",
            "description": "Scan sent folder for emails needing follow-up",
            "config": {
                "task_type": "function",
                "function_name": "scan_sent_emails",
                "days_back": 7,
                "exclude_auto_replies": True,
            },
        },
        {
            "id": "check_for_replies",
            "type": "task",
            "name": "Check for Replies",
            "description": "Check if any tracked emails have received replies",
            "config": {
                "task_type": "function",
                "function_name": "check_thread_replies",
            },
        },
        {
            "id": "update_followup_status",
            "type": "task",
            "name": "Update Follow-Up Status",
            "description": "Mark replied threads as resolved",
            "config": {
                "task_type": "function",
                "function_name": "update_followup_statuses",
            },
        },
        {
            "id": "identify_overdue",
            "type": "task",
            "name": "Identify Overdue",
            "description": "Find follow-ups past their expected reply date",
            "config": {
                "task_type": "function",
                "function_name": "identify_overdue_followups",
            },
        },
        {
            "id": "prioritize_followups",
            "type": "debate",
            "name": "Prioritize Follow-Ups",
            "description": "Multi-agent prioritization of overdue follow-ups",
            "config": {
                "agents": ["urgency_agent", "context_agent"],
                "rounds": 2,
                "topic_template": "Which follow-ups are most urgent? Overdue: {overdue_list}",
            },
        },
        {
            "id": "send_reminders",
            "type": "task",
            "name": "Queue Reminders",
            "description": "Queue reminder notifications for user",
            "config": {
                "task_type": "function",
                "function_name": "queue_followup_reminders",
                "max_reminders_per_day": 5,
            },
        },
    ],
    "transitions": [
        {"from": "scan_sent_folder", "to": "check_for_replies"},
        {"from": "check_for_replies", "to": "update_followup_status"},
        {"from": "update_followup_status", "to": "identify_overdue"},
        {"from": "identify_overdue", "to": "prioritize_followups"},
        {"from": "prioritize_followups", "to": "send_reminders"},
    ],
}


# Snooze management workflow
SNOOZE_MANAGEMENT_TEMPLATE: dict[str, Any] = {
    "name": "Snooze Management Workflow",
    "description": "Process snoozed emails and manage wake-ups",
    "category": "email",
    "version": "1.0",
    "tags": ["email", "snooze", "scheduling", "inbox"],
    "steps": [
        {
            "id": "check_due_snoozes",
            "type": "task",
            "name": "Check Due Snoozes",
            "description": "Find snoozed emails ready to wake up",
            "config": {
                "task_type": "function",
                "function_name": "get_due_snoozes",
            },
        },
        {
            "id": "wake_emails",
            "type": "task",
            "name": "Wake Up Emails",
            "description": "Move snoozed emails back to inbox",
            "config": {
                "task_type": "function",
                "function_name": "unsnooze_emails",
                "remove_snooze_label": True,
                "add_to_inbox": True,
            },
        },
        {
            "id": "check_priority_changes",
            "type": "task",
            "name": "Check Priority Changes",
            "description": "Re-evaluate priority of woken emails",
            "config": {
                "task_type": "function",
                "function_name": "reprioritize_emails",
            },
        },
        {
            "id": "notify_user",
            "type": "task",
            "name": "Notify User",
            "description": "Send notification about woken emails",
            "config": {
                "task_type": "function",
                "function_name": "send_snooze_notification",
                "batch_notifications": True,
            },
        },
    ],
    "transitions": [
        {"from": "check_due_snoozes", "to": "wake_emails"},
        {"from": "wake_emails", "to": "check_priority_changes"},
        {"from": "check_priority_changes", "to": "notify_user"},
    ],
}


# Inbox triage workflow
INBOX_TRIAGE_TEMPLATE: dict[str, Any] = {
    "name": "Inbox Triage Workflow",
    "description": "Comprehensive inbox processing with prioritization and routing",
    "category": "email",
    "version": "1.0",
    "tags": ["email", "triage", "prioritization", "routing"],
    "steps": [
        {
            "id": "fetch_new_emails",
            "type": "task",
            "name": "Fetch New Emails",
            "description": "Get unprocessed emails from inbox",
            "config": {
                "task_type": "function",
                "function_name": "fetch_new_emails",
                "max_results": 100,
            },
        },
        {
            "id": "quick_filter",
            "type": "task",
            "name": "Quick Filter",
            "description": "Apply fast pattern-based filtering (spam, newsletters)",
            "config": {
                "task_type": "function",
                "function_name": "quick_filter_emails",
                "auto_archive_newsletters": True,
                "spam_to_junk": True,
            },
        },
        {
            "id": "prioritize_batch",
            "type": "task",
            "name": "Prioritize Emails",
            "description": "Score and prioritize remaining emails",
            "config": {
                "task_type": "function",
                "function_name": "prioritize_email_batch",
                "use_sender_history": True,
            },
        },
        {
            "id": "categorize_emails",
            "type": "task",
            "name": "Categorize Emails",
            "description": "Assign categories to emails",
            "config": {
                "task_type": "function",
                "function_name": "categorize_email_batch",
            },
        },
        {
            "id": "agent_review_critical",
            "type": "debate",
            "name": "Review Critical Emails",
            "description": "Multi-agent review of critical priority emails",
            "config": {
                "agents": ["urgency_agent", "billing_agent", "context_agent"],
                "rounds": 2,
                "filter_priority": "critical",
                "topic_template": "Review critical email: {subject}. From: {sender}",
            },
        },
        {
            "id": "apply_actions",
            "type": "task",
            "name": "Apply Triage Actions",
            "description": "Apply labels, stars, and routing based on triage results",
            "config": {
                "task_type": "function",
                "function_name": "apply_triage_actions",
                "star_critical": True,
                "label_by_category": True,
            },
        },
        {
            "id": "detect_followups",
            "type": "task",
            "name": "Detect Follow-Up Needs",
            "description": "Identify emails that might need follow-up tracking",
            "config": {
                "task_type": "function",
                "function_name": "detect_followup_candidates",
            },
        },
        {
            "id": "suggest_snoozes",
            "type": "task",
            "name": "Suggest Snoozes",
            "description": "Generate snooze suggestions for deferrable emails",
            "config": {
                "task_type": "function",
                "function_name": "generate_snooze_suggestions",
                "target_priority": "low",
            },
        },
        {
            "id": "update_statistics",
            "type": "task",
            "name": "Update Statistics",
            "description": "Update inbox analytics and sender history",
            "config": {
                "task_type": "function",
                "function_name": "update_inbox_statistics",
            },
        },
    ],
    "transitions": [
        {"from": "fetch_new_emails", "to": "quick_filter"},
        {"from": "quick_filter", "to": "prioritize_batch"},
        {"from": "prioritize_batch", "to": "categorize_emails"},
        {"from": "categorize_emails", "to": "agent_review_critical"},
        {"from": "agent_review_critical", "to": "apply_actions"},
        {"from": "apply_actions", "to": "detect_followups"},
        {"from": "detect_followups", "to": "suggest_snoozes"},
        {"from": "suggest_snoozes", "to": "update_statistics"},
    ],
}


# Template registry
EMAIL_MANAGEMENT_TEMPLATES = {
    "email_categorization": EMAIL_CATEGORIZATION_TEMPLATE,
    "followup_tracking": FOLLOWUP_TRACKING_TEMPLATE,
    "snooze_management": SNOOZE_MANAGEMENT_TEMPLATE,
    "inbox_triage": INBOX_TRIAGE_TEMPLATE,
}


def get_email_management_template(name: str) -> dict[str, Any]:
    """
    Get an email management workflow template by name.

    Args:
        name: Template name

    Returns:
        Template dictionary

    Raises:
        KeyError: If template not found
    """
    if name not in EMAIL_MANAGEMENT_TEMPLATES:
        raise KeyError(
            f"Unknown email management template: {name}. "
            f"Available: {list(EMAIL_MANAGEMENT_TEMPLATES.keys())}"
        )
    return EMAIL_MANAGEMENT_TEMPLATES[name]


def list_email_management_templates() -> list[dict[str, str]]:
    """List available email management templates."""
    return [
        {
            "name": name,
            "display_name": template["name"],
            "description": template["description"],
            "tags": template.get("tags", []),
        }
        for name, template in EMAIL_MANAGEMENT_TEMPLATES.items()
    ]
