"""Email sender for debate origin result routing."""

from __future__ import annotations

import logging
from typing import Any

from ..models import DebateOrigin
from ..formatting import _format_result_message

logger = logging.getLogger(__name__)


async def _send_email_result(origin: DebateOrigin, result: dict[str, Any]) -> bool:
    """Send result via email."""
    # Use existing email notification system
    try:
        from aragora.server.handlers.social.notifications import send_email_notification  # type: ignore[attr-defined]

        email = origin.metadata.get("email")
        if not email:
            email = origin.channel_id  # channel_id is email for email platform

        subject = "Aragora Debate Complete"
        message = _format_result_message(result, origin, markdown=False, html=True)

        # Fire-and-forget email
        await send_email_notification(
            to_email=email,
            subject=subject,
            body=message,
        )
        logger.info(f"Email result sent to {email}")
        return True

    except ImportError:
        logger.warning("Email notification system not available")
        return False
    except Exception as e:
        logger.error(f"Email result send error: {e}")
        return False
