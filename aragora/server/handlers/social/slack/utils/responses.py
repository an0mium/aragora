"""
Slack response utilities.

Provides helper functions for creating Slack-formatted responses.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from aragora.server.handlers.base import HandlerResult


def slack_response(
    text: str,
    response_type: str = "ephemeral",
    attachments: Optional[List[Dict[str, Any]]] = None,
) -> HandlerResult:
    """Create a basic Slack response.

    Args:
        text: Response text
        response_type: "ephemeral" (only visible to user) or "in_channel" (visible to all)
        attachments: Optional list of attachments

    Returns:
        HandlerResult with Slack-formatted JSON response
    """
    response: Dict[str, Any] = {
        "response_type": response_type,
        "text": text,
    }
    if attachments:
        response["attachments"] = attachments

    return HandlerResult(
        status_code=200,
        content_type="application/json",
        body=json.dumps(response).encode("utf-8"),
    )


def slack_blocks_response(
    blocks: List[Dict[str, Any]],
    response_type: str = "ephemeral",
    text: str = "",
) -> HandlerResult:
    """Create a Slack response with Block Kit blocks.

    Args:
        blocks: Block Kit blocks
        response_type: "ephemeral" (only visible to user) or "in_channel" (visible to all)
        text: Fallback text for notifications

    Returns:
        HandlerResult with Slack-formatted JSON response
    """
    response: Dict[str, Any] = {
        "response_type": response_type,
        "blocks": blocks,
    }
    if text:
        response["text"] = text

    return HandlerResult(
        status_code=200,
        content_type="application/json",
        body=json.dumps(response).encode("utf-8"),
    )
