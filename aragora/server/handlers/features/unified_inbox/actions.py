"""Bulk actions for unified inbox messages.

Handles batch operations like archive, mark read/unread, star, and delete.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

VALID_ACTIONS = ["archive", "mark_read", "mark_unread", "star", "delete"]


async def execute_bulk_action(
    tenant_id: str,
    message_ids: list[str],
    action: str,
    store: Any,
) -> dict[str, Any]:
    """Execute a bulk action on the given message IDs.

    Returns a dict with success_count, error_count, and errors list.
    """
    success_count = 0
    errors: list[dict[str, str]] = []

    for msg_id in message_ids:
        try:
            if action == "mark_read":
                updated = await store.update_message_flags(tenant_id, msg_id, is_read=True)
                if not updated:
                    errors.append({"id": msg_id, "error": "Message not found"})
                    continue
            elif action == "mark_unread":
                updated = await store.update_message_flags(tenant_id, msg_id, is_read=False)
                if not updated:
                    errors.append({"id": msg_id, "error": "Message not found"})
                    continue
            elif action == "star":
                updated = await store.update_message_flags(tenant_id, msg_id, is_starred=True)
                if not updated:
                    errors.append({"id": msg_id, "error": "Message not found"})
                    continue
            elif action in ("archive", "delete"):
                deleted = await store.delete_message(tenant_id, msg_id)
                if not deleted:
                    errors.append({"id": msg_id, "error": "Message not found"})
                    continue

            success_count += 1

        except (RuntimeError, ValueError, TypeError, KeyError, OSError) as e:
            logger.warning("Bulk action failed for message %s: %s", msg_id, e)
            errors.append({"id": msg_id, "error": "Action failed"})

    return {
        "action": action,
        "success_count": success_count,
        "error_count": len(errors),
        "errors": errors if errors else None,
    }
