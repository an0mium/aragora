"""
Built-in Hook Handlers.

Pre-defined handlers for common hook actions that can be referenced
in YAML hook configurations.

Usage in YAML:
    action:
      handler: aragora.hooks.builtin.log_event
      args:
        level: info
        message: "Debate complete: {consensus_reached}"
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

__all__ = [
    # Logging
    "log_event",
    "log_metric",
    # Notifications
    "send_webhook",
    "send_slack_notification",
    # Persistence
    "save_checkpoint",
    "store_fact",
    # Control flow
    "set_context_var",
    "delay_execution",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Logging Handlers
# =============================================================================


async def log_event(
    message: str = "",
    level: str = "info",
    include_context: bool = False,
    **context: Any,
) -> None:
    """
    Log an event with optional context.

    Args:
        message: Log message (supports {field} interpolation)
        level: Log level (debug, info, warning, error)
        include_context: Whether to include full context in log
        **context: Context values for interpolation
    """
    # Interpolate message
    try:
        formatted = message.format(**context)
    except (KeyError, ValueError):
        formatted = message

    # Get log function
    log_fn = getattr(logger, level.lower(), logger.info)

    if include_context:
        # Serialize context for logging
        ctx_str = json.dumps(
            {k: str(v)[:200] for k, v in context.items()},
            default=str,
        )
        log_fn(f"{formatted} | context={ctx_str}")
    else:
        log_fn(formatted)


async def log_metric(
    metric_name: str,
    value: Any,
    tags: Optional[dict[str, str]] = None,
    **context: Any,
) -> None:
    """
    Log a metric value.

    Args:
        metric_name: Name of the metric
        value: Metric value
        tags: Optional tags for the metric
        **context: Additional context
    """
    tags = tags or {}

    # Extract common tags from context
    if "debate_id" in context:
        tags["debate_id"] = str(context["debate_id"])
    if "platform" in context:
        tags["platform"] = str(context["platform"])

    tag_str = ",".join(f"{k}={v}" for k, v in tags.items())
    logger.info(f"METRIC {metric_name}={value} {tag_str}")


# =============================================================================
# Notification Handlers
# =============================================================================


async def send_webhook(
    url: str,
    payload_template: Optional[dict[str, Any]] = None,
    method: str = "POST",
    headers: Optional[dict[str, str]] = None,
    timeout: float = 30.0,
    **context: Any,
) -> bool:
    """
    Send a webhook notification.

    Args:
        url: Webhook URL
        payload_template: Payload template (fields interpolated from context)
        method: HTTP method
        headers: Additional headers
        timeout: Request timeout
        **context: Context for interpolation

    Returns:
        True if webhook was sent successfully
    """
    try:
        import httpx

        # Build payload
        payload: dict[str, Any] = {}
        if payload_template:
            for key, value in payload_template.items():
                if isinstance(value, str) and "{" in value:
                    try:
                        payload[key] = value.format(**context)
                    except (KeyError, ValueError):
                        payload[key] = value
                else:
                    payload[key] = value
        else:
            # Default payload
            payload = {
                "event": context.get("trigger", "hook_event"),
                "timestamp": datetime.utcnow().isoformat(),
                "data": {k: str(v)[:500] for k, v in context.items() if k != "trigger"},
            }

        async with httpx.AsyncClient(timeout=timeout) as client:
            if method.upper() == "POST":
                response = await client.post(url, json=payload, headers=headers)
            elif method.upper() == "PUT":
                response = await client.put(url, json=payload, headers=headers)
            else:
                response = await client.get(url, params=payload, headers=headers)

            if response.is_success:
                logger.debug(f"Webhook sent to {url}")
                return True
            else:
                logger.warning(f"Webhook failed: {response.status_code}")
                return False

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return False


async def send_slack_notification(
    channel: str,
    message_template: str,
    username: str = "Aragora",
    icon_emoji: str = ":robot_face:",
    **context: Any,
) -> bool:
    """
    Send a Slack notification.

    Requires SLACK_WEBHOOK_URL environment variable.

    Args:
        channel: Channel to send to
        message_template: Message template with {field} interpolation
        username: Bot username
        icon_emoji: Bot icon
        **context: Context for interpolation

    Returns:
        True if notification was sent
    """
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not configured")
        return False

    # Interpolate message
    try:
        message = message_template.format(**context)
    except (KeyError, ValueError):
        message = message_template

    return await send_webhook(
        url=webhook_url,
        payload_template={
            "channel": channel,
            "username": username,
            "icon_emoji": icon_emoji,
            "text": message,
        },
        **context,
    )


# =============================================================================
# Persistence Handlers
# =============================================================================


async def save_checkpoint(
    path: str = "checkpoints",
    filename_template: str = "checkpoint_{debate_id}_{timestamp}.json",
    include_fields: Optional[list[str]] = None,
    **context: Any,
) -> Optional[str]:
    """
    Save a checkpoint file.

    Args:
        path: Directory path for checkpoints
        filename_template: Filename template with interpolation
        include_fields: Specific fields to include (None = all)
        **context: Context to save

    Returns:
        Path to saved file, or None if failed
    """
    try:
        # Create directory
        checkpoint_dir = Path(path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        try:
            filename = filename_template.format(
                timestamp=timestamp,
                **context,
            )
        except (KeyError, ValueError):
            filename = f"checkpoint_{timestamp}.json"

        # Filter fields if specified
        if include_fields:
            data = {k: v for k, v in context.items() if k in include_fields}
        else:
            data = context

        # Add metadata
        data["_checkpoint_time"] = datetime.utcnow().isoformat()

        # Save file
        file_path = checkpoint_dir / filename
        with open(file_path, "w") as f:
            json.dump(data, f, default=str, indent=2)

        logger.info(f"Checkpoint saved: {file_path}")
        return str(file_path)

    except Exception as e:
        logger.error(f"Checkpoint save error: {e}")
        return None


async def store_fact(
    fact_type: str,
    content_field: str = "final_answer",
    confidence_field: str = "confidence",
    source: str = "debate",
    **context: Any,
) -> bool:
    """
    Store a fact in the knowledge mound.

    Args:
        fact_type: Type of fact (e.g., "consensus", "finding")
        content_field: Context field containing the fact content
        confidence_field: Context field containing confidence score
        source: Source identifier
        **context: Context containing the fact data

    Returns:
        True if fact was stored successfully
    """
    try:
        from aragora.knowledge.mound import get_knowledge_mound

        mound = get_knowledge_mound()

        content = context.get(content_field, "")
        confidence = context.get(confidence_field, 0.5)
        debate_id = context.get("debate_id", "unknown")

        await mound.store_verified_fact(  # type: ignore[misc]
            content=str(content),
            source=f"{source}:{fact_type}",
            confidence=float(confidence),
            topics=[fact_type] if fact_type else None,
        )

        logger.info(f"Fact stored: {fact_type} from debate {debate_id}")
        return True

    except ImportError:
        logger.debug("Knowledge mound not available")
        return False
    except Exception as e:
        logger.error(f"Fact storage error: {e}")
        return False


# =============================================================================
# Control Flow Handlers
# =============================================================================


async def set_context_var(
    var_name: str,
    value: Any,
    **context: Any,
) -> None:
    """
    Set a context variable for downstream hooks.

    Note: This modifies the context dict in-place.

    Args:
        var_name: Variable name to set
        value: Value to set
        **context: Current context (modified in-place)
    """
    context[var_name] = value


async def delay_execution(
    seconds: float = 1.0,
    **context: Any,
) -> None:
    """
    Delay execution before proceeding.

    Useful for rate limiting or sequencing.

    Args:
        seconds: Delay duration
        **context: Ignored
    """
    import asyncio

    await asyncio.sleep(seconds)
