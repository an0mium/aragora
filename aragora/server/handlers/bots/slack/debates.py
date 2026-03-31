"""
Slack Debate Management.

This module handles starting and managing debates from Slack,
including integration with DecisionRouter and fallback mechanisms.
"""

import asyncio
import logging
import time
from typing import Any

from aragora.config import DEFAULT_ROUNDS

from .state import _active_debates

logger = logging.getLogger(__name__)

_SLACK_FAIL_CLOSED_MIN_CONFIDENCE = 0.7
_SLACK_NON_TRIVIAL_MIN_ROUNDS = 2


def _build_slack_policy(topic: str) -> dict[str, Any]:
    """Classify Slack asks for delivery guardrails."""
    try:
        from aragora.routing.lara_router import LaRARouter, RetrievalMode

        decision = LaRARouter().route(query=topic, doc_tokens=0)
        features = decision.query_features
        is_non_trivial = (
            decision.selected_mode != RetrievalMode.RAG
            or features.is_analytical
            or features.is_multi_hop
            or features.requires_aggregation
            or features.complexity_score >= 0.35
        )
        query_mode = "factual" if not is_non_trivial and features.is_factual else "deliberative"
        min_rounds = 1 if query_mode == "factual" else _SLACK_NON_TRIVIAL_MIN_ROUNDS
        return {
            "query_mode": query_mode,
            "selected_mode": decision.selected_mode.value,
            "routing_confidence": float(decision.confidence),
            "require_consensus": True,
            "fail_closed": True,
            "min_confidence": _SLACK_FAIL_CLOSED_MIN_CONFIDENCE,
            "min_rounds": min_rounds,
        }
    except (ImportError, RuntimeError, AttributeError, ValueError):
        return {
            "query_mode": "deliberative",
            "selected_mode": "debate",
            "routing_confidence": 0.0,
            "require_consensus": True,
            "fail_closed": True,
            "min_confidence": _SLACK_FAIL_CLOSED_MIN_CONFIDENCE,
            "min_rounds": _SLACK_NON_TRIVIAL_MIN_ROUNDS,
        }


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

        slack_policy = _build_slack_policy(topic)

        # Create request context
        context = RequestContext(
            user_id=user_id,
            session_id=f"slack:{channel_id}",
            metadata={"slack_policy": slack_policy},
        )

        config_kwargs: dict[str, Any] = {}
        if decision_integrity is not None:
            if isinstance(decision_integrity, bool):
                decision_integrity = {} if decision_integrity else None
            if isinstance(decision_integrity, dict):
                config_kwargs["decision_integrity"] = decision_integrity
        min_rounds = int(slack_policy.get("min_rounds", 1) or 1)
        if DEFAULT_ROUNDS < min_rounds:
            config_kwargs["rounds"] = min_rounds
        config = DecisionConfig(**config_kwargs) if config_kwargs else None

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
                    "slack_policy": slack_policy,
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
            except (
                RuntimeError,
                ValueError,
                KeyError,
                AttributeError,
                OSError,
            ) as exc:  # pragma: no cover - defensive logging
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
                            "slack_policy": slack_policy,
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
        logger.error("DecisionRouter failed: %s, using fallback", e)
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
        logger.warning("Failed to register debate origin: %s", e)

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
        logger.info("Debate %s enqueued via Redis queue", debate_id)
    except ImportError:
        logger.warning("Redis queue not available, debate will run inline")
    except (RuntimeError, OSError, ConnectionError) as e:
        logger.warning("Failed to enqueue debate: %s", e)

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
