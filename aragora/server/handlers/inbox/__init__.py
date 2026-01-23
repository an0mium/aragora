"""
Inbox Management Handlers.

API handlers for inbox intelligence features:
- Action item extraction and tracking
- Meeting detection and calendar integration
- Email categorization
- Priority scoring
"""

from .action_items import (
    get_action_items_handlers,
    handle_auto_snooze_meeting,
    handle_batch_extract,
    handle_complete_action,
    handle_detect_meeting,
    handle_extract_action_items,
    handle_get_due_soon,
    handle_list_pending_actions,
    handle_update_action_status,
)

__all__ = [
    # Handler functions
    "handle_extract_action_items",
    "handle_list_pending_actions",
    "handle_complete_action",
    "handle_update_action_status",
    "handle_get_due_soon",
    "handle_batch_extract",
    "handle_detect_meeting",
    "handle_auto_snooze_meeting",
    # Registration helper
    "get_action_items_handlers",
]
