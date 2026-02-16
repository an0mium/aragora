"""
Slack Debate Management.

This module handles starting and managing debates from Slack,
including integration with DecisionRouter and fallback mechanisms.
"""

import asyncio
import logging
import time
from typing import Any

from .state import _active_debates

logger = logging.getLogger(__name__)


async def start_slack_debate(
    topic: str,
    channel_id: str,
    user_id: str,
    response_url: str = "",
    thread_ts: str | None = None,
    attachments: list[dict[str, Any]] | None = None,
    decision_integrity: dict[str, Any] | bool | None = None,
) -> str:
    """Start a debate from Slack via DecisionRouter.

    Uses the unified DecisionRouter for:
    - Deduplication (prevents duplicate debates for same topic/user)
    - Caching (returns cached results if available)
    - Origin registration for result routing
    """
    import uuid

    debate_id = str(uuid.uuid4())

    try:
        from aragora.core import (
            DecisionConfig,
            DecisionRequest,
            DecisionType,
            InputSource,
            RequestContext,
            ResponseChannel,
            get_decision_router,
        )

        # Create response channel for result routing
        response_channel = ResponseChannel(
            platform="slack",
            channel_id=channel_id,
            user_id=user_id,
            thread_id=thread_ts,
            webhook_url=response_url,
        )

        # Create request context
        context = RequestContext(
            user_id=user_id,
            session_id=f"slack:{channel_id}",
        )

        config = None
        if decision_integrity is not None:
            if isinstance(decision_integrity, bool):
                decision_integrity = {} if decision_integrity else None
            if isinstance(decision_integrity, dict):
                config = DecisionConfig(decision_integrity=decision_integrity)

        request_kwargs = {
            "content": topic,
            "decision_type": DecisionType.DEBATE,
            "source": InputSource.SLACK,
            "response_channels": [response_channel],
            "context": context,
            "attachments": attachments or [],
        }
        if config is not None:
            request_kwargs["config"] = config

        # Create decision request
        request = DecisionRequest(**request_kwargs)  # type: ignore[arg-type]

        # Register origin for result routing (best-effort)
        try:
            from aragora.server.debate_origin import register_debate_origin

            register_debate_origin(
                debate_id=request.request_id,
                platform="slack",
                channel_id=channel_id,
                user_id=user_id,
                thread_id=thread_ts,
                metadata={
                    "topic": topic,
                    "response_url": response_url,
                },
            )
        except (RuntimeError, KeyError, AttributeError, OSError) as exc:
            logger.debug("Failed to register Slack debate origin: %s", exc)

        # Route through DecisionRouter in the background to keep Slack responsive.
        router = get_decision_router()

        def _record_active(debate_key: str) -> None:
            _active_debates[debate_key] = {
                "topic": topic,
                "channel_id": channel_id,
                "user_id": user_id,
                "thread_ts": thread_ts,
                "started_at": time.time(),
            }

        task = asyncio.create_task(router.route(request))

        def _route_done(done_task: asyncio.Task) -> None:
            try:
                result = done_task.result()
            except asyncio.CancelledError:
                return
            except (RuntimeError, ValueError, KeyError, AttributeError, OSError) as exc:  # pragma: no cover - defensive logging
                logger.error(
                    "DecisionRouter task failed for Slack debate %s: %s",
                    request.request_id,
                    exc,
                )
                return

            if result.request_id and result.request_id != request.request_id:
                state = _active_debates.pop(request.request_id, None)
                if state is not None:
                    _active_debates[result.request_id] = state
                try:
                    from aragora.server.debate_origin import register_debate_origin

                    register_debate_origin(
                        debate_id=result.request_id,
                        platform="slack",
                        channel_id=channel_id,
                        user_id=user_id,
                        thread_id=thread_ts,
                        metadata={
                            "topic": topic,
                            "response_url": response_url,
                        },
                    )
                except (RuntimeError, KeyError, AttributeError, OSError, ImportError) as exc:
                    logger.debug("Failed to register dedup Slack origin: %s", exc)
            logger.info("DecisionRouter started debate %s from Slack", result.request_id)

        task.add_done_callback(_route_done)

        # If we can get a quick response (cache/dedup), use its request_id; otherwise fall back.
        debate_key = request.request_id
        try:
            result = await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
        except asyncio.TimeoutError:
            result = None
        except (RuntimeError, ValueError, KeyError, AttributeError, OSError) as exc:
            logger.debug("Failed to get quick debate result: %s", exc)
            result = None

        if result and result.request_id:
            debate_key = result.request_id

        _record_active(debate_key)
        return debate_key

    except ImportError:
        logger.debug("DecisionRouter not available, using fallback")
        return await _fallback_start_debate(topic, channel_id, user_id, debate_id, thread_ts)
    except (RuntimeError, ValueError, KeyError, AttributeError) as e:
        logger.error(f"DecisionRouter failed: {e}, using fallback")
        return await _fallback_start_debate(topic, channel_id, user_id, debate_id, thread_ts)


async def _fallback_start_debate(
    topic: str,
    channel_id: str,
    user_id: str,
    debate_id: str,
    thread_ts: str | None = None,
) -> str:
    """Fallback debate start when DecisionRouter unavailable."""
    # Register origin for result routing
    try:
        from aragora.server.debate_origin import register_debate_origin

        register_debate_origin(
            debate_id=debate_id,
            platform="slack",
            channel_id=channel_id,
            user_id=user_id,
            thread_id=thread_ts,
            metadata={"topic": topic},
        )
    except (RuntimeError, KeyError, AttributeError, OSError) as e:
        logger.warning(f"Failed to register debate origin: {e}")

    # Try to enqueue via Redis queue
    try:
        from aragora.queue import create_debate_job, create_redis_queue

        job = create_debate_job(
            question=topic,
            user_id=user_id,
            metadata={
                "debate_id": debate_id,
                "platform": "slack",
                "channel_id": channel_id,
                "thread_ts": thread_ts,
            },
        )
        queue = await create_redis_queue()
        await queue.enqueue(job)
        logger.info(f"Debate {debate_id} enqueued via Redis queue")
    except ImportError:
        logger.warning("Redis queue not available, debate will run inline")
    except (RuntimeError, OSError, ConnectionError) as e:
        logger.warning(f"Failed to enqueue debate: {e}")

    # Track active debate
    _active_debates[debate_id] = {
        "topic": topic,
        "channel_id": channel_id,
        "user_id": user_id,
        "thread_ts": thread_ts,
        "started_at": time.time(),
    }

    return debate_id


# Backward compatibility alias
_start_slack_debate = start_slack_debate
_fallback_start_debate = _fallback_start_debate


__all__ = [
    "start_slack_debate",
    "_start_slack_debate",
    "_fallback_start_debate",
]
