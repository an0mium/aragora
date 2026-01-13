"""
Batch debate operations handler mixin.

Extracted from debates.py for modularity. Provides batch submission,
status checking, and queue management.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from aragora.server.handlers.base import (
    error_response,
    json_response,
    HandlerResult,
    handle_errors,
    safe_error_message,
)
from aragora.server.middleware.tier_enforcement import check_org_quota, increment_org_usage
from aragora.server.handlers.utils.rate_limit import rate_limit
from aragora.server.validation.entities import validate_path_segment, SAFE_ID_PATTERN
from aragora.server.validation.schema import validate_against_schema, BATCH_SUBMIT_SCHEMA
from aragora.exceptions import DebateStartError

if TYPE_CHECKING:
    from aragora.server.handlers.debates import DebatesHandler

logger = logging.getLogger(__name__)


class BatchOperationsMixin:
    """Mixin providing batch debate operations for DebatesHandler."""

    @rate_limit(rpm=10, limiter_name="batch_submit")
    @handle_errors("submit batch")
    def _submit_batch(self: "DebatesHandler", handler) -> HandlerResult:
        """Submit a batch of debates for processing.

        POST body:
            {
                "items": [
                    {
                        "question": str,  # Required
                        "agents": str,    # Optional, default: anthropic-api,openai-api,gemini
                        "rounds": int,    # Optional, default: 3
                        "consensus": str, # Optional, default: majority
                        "priority": int,  # Optional, higher = runs first
                        "metadata": dict  # Optional, custom data
                    },
                    ...
                ],
                "webhook_url": str,      # Optional: callback URL on completion
                "webhook_headers": dict, # Optional: headers for webhook
                "max_parallel": int      # Optional: override default concurrency
            }

        Returns:
            {
                "success": true,
                "batch_id": "batch_abc123",
                "items_queued": 10,
                "status_url": "/api/debates/batch/batch_abc123/status"
            }
        """
        from aragora.config import MAX_CONCURRENT_DEBATES
        from aragora.server.debate_queue import (
            BatchRequest,
            BatchItem,
            sanitize_webhook_headers,
            validate_webhook_url,
        )

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid or missing JSON body", 400)

        # Schema validation for input sanitization
        validation_result = validate_against_schema(body, BATCH_SUBMIT_SCHEMA)
        if not validation_result.is_valid:
            return error_response(validation_result.error, 400)

        items_data = body.get("items", [])
        if not items_data:
            return error_response("items array is required and cannot be empty", 400)

        if len(items_data) > 1000:
            return error_response("Batch cannot exceed 1000 items", 400)

        # Validate and parse items
        items = []
        errors = []
        for i, item_data in enumerate(items_data):
            if not isinstance(item_data, dict):
                errors.append(f"Item {i}: must be an object")
                continue

            question = item_data.get("question", "").strip()
            if not question:
                errors.append(f"Item {i}: question is required")
                continue

            if len(question) > 10000:
                errors.append(f"Item {i}: question exceeds 10,000 characters")
                continue

            try:
                item = BatchItem.from_dict(item_data)
                items.append(item)
            except Exception as e:
                errors.append(f"Item {i}: {str(e)}")

        if errors:
            return error_response(
                f"Validation failed: {'; '.join(errors[:5])}"
                + (f" (and {len(errors) - 5} more)" if len(errors) > 5 else ""),
                400,
            )

        # Check quota before proceeding
        batch_size = len(items)
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = None
        if hasattr(handler, "user_store"):
            user_store = handler.user_store
        elif hasattr(handler.__class__, "user_store"):
            user_store = handler.__class__.user_store

        user_ctx = extract_user_from_request(handler, user_store) if user_store else None

        if user_ctx and user_ctx.is_authenticated and user_ctx.org_id:
            if user_store and hasattr(user_store, "get_organization_by_id"):
                org = user_store.get_organization_by_id(user_ctx.org_id)
                if org:
                    if org.is_at_limit:
                        return json_response(
                            {
                                "error": "quota_exceeded",
                                "code": "QUOTA_EXCEEDED",
                                "limit": org.limits.debates_per_month,
                                "used": org.debates_used_this_month,
                                "remaining": 0,
                                "tier": org.tier.value,
                                "upgrade_url": "/pricing",
                                "message": f"Your {org.tier.value} plan allows {org.limits.debates_per_month} debates per month. Upgrade to increase your limit.",
                            },
                            status=402,
                        )  # Payment Required

                    remaining = org.limits.debates_per_month - org.debates_used_this_month
                    if batch_size > remaining:
                        return json_response(
                            {
                                "error": "quota_insufficient",
                                "code": "QUOTA_INSUFFICIENT",
                                "limit": org.limits.debates_per_month,
                                "used": org.debates_used_this_month,
                                "remaining": remaining,
                                "requested": batch_size,
                                "tier": org.tier.value,
                                "upgrade_url": "/pricing",
                                "message": f"Requested {batch_size} debates but only {remaining} remaining. Upgrade to increase your limit.",
                            },
                            status=402,
                        )  # Payment Required

        webhook_url = body.get("webhook_url")
        if webhook_url:
            is_valid, error_msg = validate_webhook_url(str(webhook_url))
            if not is_valid:
                return error_response(error_msg, 400)

        webhook_headers, header_error = sanitize_webhook_headers(body.get("webhook_headers"))
        if header_error:
            return error_response(header_error, 400)

        max_parallel = body.get("max_parallel")
        if max_parallel is not None:
            try:
                max_parallel = int(max_parallel)
            except (TypeError, ValueError):
                return error_response("max_parallel must be an integer", 400)
            max_parallel = max(1, min(max_parallel, MAX_CONCURRENT_DEBATES))

        # Create batch request
        batch = BatchRequest(
            items=items,
            webhook_url=webhook_url,
            webhook_headers=webhook_headers,
            max_parallel=max_parallel,
        )

        # Submit to queue
        try:
            from aragora.server.debate_queue import get_debate_queue
            from aragora.server.http_utils import run_async

            async def submit():
                queue = await get_debate_queue()

                # Configure executor if not set
                if queue.debate_executor is None:
                    queue.debate_executor = self._create_debate_executor()

                return await queue.submit_batch(batch)

            batch_id = run_async(submit())

            logger.info(f"Batch {batch_id} submitted with {len(items)} items")

            # Increment usage on successful batch submission
            if user_ctx and user_ctx.is_authenticated and user_ctx.org_id:
                if user_store and hasattr(user_store, "increment_usage"):
                    try:
                        user_store.increment_usage(user_ctx.org_id, batch_size)
                        logger.info(
                            f"Incremented batch usage for org {user_ctx.org_id} by {batch_size}"
                        )
                    except Exception as ue:
                        logger.warning(f"Usage increment failed for org {user_ctx.org_id}: {ue}")

            return json_response(
                {
                    "success": True,
                    "batch_id": batch_id,
                    "items_queued": len(items),
                    "status_url": f"/api/debates/batch/{batch_id}/status",
                }
            )

        except Exception as e:
            logger.error(f"Failed to submit batch: {e}", exc_info=True)
            return error_response(safe_error_message(e, "submit batch"), 500)

    def _create_debate_executor(self: "DebatesHandler"):
        """Create a debate executor function for the batch queue."""
        from aragora.server.debate_queue import BatchItem

        async def execute_debate(item: BatchItem):
            """Execute a single debate from batch."""
            from aragora.server.debate_controller import DebateRequest, DebateController
            from aragora.server.debate_factory import DebateFactory
            from aragora.server.stream import SyncEventEmitter

            # Create minimal components for debate execution
            emitter = SyncEventEmitter()
            factory = DebateFactory()

            controller = DebateController(
                factory=factory,
                emitter=emitter,
            )

            request = DebateRequest(
                question=item.question,
                agents_str=item.agents,
                rounds=item.rounds,
                consensus=item.consensus,
            )

            response = controller.start_debate(request)

            if response.success:
                return {
                    "success": True,
                    "debate_id": response.debate_id,
                }
            else:
                raise DebateStartError(
                    debate_id=response.debate_id or "unknown",
                    reason=response.error or "Debate failed to start",
                )

        return execute_debate

    @handle_errors("get batch status")
    def _get_batch_status(self: "DebatesHandler", batch_id: str) -> HandlerResult:
        """Get status of a batch request.

        Returns full batch status including all items.
        """
        is_valid, err = validate_path_segment(batch_id, "batch id", SAFE_ID_PATTERN)
        if not is_valid:
            return error_response(err, 400)

        from aragora.server.debate_queue import get_debate_queue_sync

        queue = get_debate_queue_sync()
        if not queue:
            return error_response("Batch queue not initialized", 503)

        status = queue.get_batch_status(batch_id)
        if not status:
            return error_response(f"Batch not found: {batch_id}", 404)

        return json_response(status)

    @handle_errors("list batches")
    def _list_batches(
        self: "DebatesHandler", limit: int, status_filter: Optional[str] = None
    ) -> HandlerResult:
        """List batch requests.

        Query params:
            limit: Maximum batches to return (default 50, max 100)
            status: Filter by status (pending, processing, completed, etc.)
        """
        from aragora.server.debate_queue import get_debate_queue_sync, BatchStatus

        queue = get_debate_queue_sync()
        if not queue:
            return json_response({"batches": [], "count": 0})

        # Parse status filter
        status = None
        if status_filter:
            try:
                status = BatchStatus(status_filter)
            except ValueError:
                valid = ", ".join(s.value for s in BatchStatus)
                return error_response(f"Invalid status '{status_filter}'. Valid: {valid}", 400)

        batches = queue.list_batches(status=status, limit=limit)

        return json_response(
            {
                "batches": batches,
                "count": len(batches),
            }
        )

    @handle_errors("get queue status")
    def _get_queue_status(self: "DebatesHandler") -> HandlerResult:
        """Get overall queue status.

        Returns queue health and processing statistics.
        """
        from aragora.server.debate_queue import get_debate_queue_sync

        queue = get_debate_queue_sync()
        if not queue:
            return json_response(
                {
                    "active": False,
                    "message": "Queue not initialized",
                }
            )

        # Count batches by status
        all_batches = queue.list_batches(limit=1000)
        status_counts = {}
        for batch in all_batches:
            status = batch.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        return json_response(
            {
                "active": True,
                "max_concurrent": queue.max_concurrent,
                "active_count": queue._active_count,
                "total_batches": len(all_batches),
                "status_counts": status_counts,
            }
        )


__all__ = ["BatchOperationsMixin"]
