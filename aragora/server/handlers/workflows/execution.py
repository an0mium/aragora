"""
Workflow execution operations.

Provides operations for executing workflows and managing execution state.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .core import (
    logger,
    _get_store,
    _get_engine,
    _step_result_to_dict,
)


def _normalize_list(value: Any | None) -> list[str] | None:
    """Normalize strings or sequences into a list of non-empty strings."""
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or None
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or None
    return None


def _extract_notification_context(
    workflow: Any,
    inputs: dict[str, Any],
    tenant_id: str,
    user_id: str | None,
    org_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build metadata and notification settings for workflow execution."""
    base_meta = workflow.metadata if isinstance(getattr(workflow, "metadata", None), dict) else {}

    channel_targets = _normalize_list(
        inputs.get("channel_targets")
        or inputs.get("chat_targets")
        or inputs.get("notify_channels")
        or base_meta.get("channel_targets")
        or base_meta.get("chat_targets")
        or base_meta.get("notify_channels")
    )
    approval_targets = _normalize_list(
        inputs.get("approval_targets") or base_meta.get("approval_targets")
    )

    notify_steps = inputs.get("notify_steps")
    if notify_steps is None:
        notify_steps = base_meta.get("notify_steps", False)
    notify_steps = bool(notify_steps)

    thread_id = (
        inputs.get("thread_id")
        or inputs.get("origin_thread_id")
        or base_meta.get("thread_id")
        or base_meta.get("origin_thread_id")
    )

    thread_id_by_platform = None
    raw_threads = inputs.get("thread_id_by_platform") or base_meta.get("thread_id_by_platform")
    if isinstance(raw_threads, dict):
        thread_id_by_platform = {
            str(key): str(value)
            for key, value in raw_threads.items()
            if key is not None and value is not None
        }

    if not channel_targets and bool(inputs.get("notify_channels") or base_meta.get("notify_channels")):
        try:
            from aragora.approvals.chat import get_default_chat_targets

            channel_targets = get_default_chat_targets() or None
        except Exception:
            channel_targets = None

    metadata = dict(base_meta)
    metadata["tenant_id"] = tenant_id
    if user_id:
        metadata["user_id"] = user_id
    if org_id:
        metadata["org_id"] = org_id
    if channel_targets:
        metadata["channel_targets"] = channel_targets
        metadata.setdefault("chat_targets", channel_targets)
    if approval_targets:
        metadata.setdefault("approval_targets", approval_targets)
    if thread_id:
        metadata["thread_id"] = thread_id
    if thread_id_by_platform:
        metadata["thread_id_by_platform"] = thread_id_by_platform

    notify_config = {
        "channel_targets": channel_targets or [],
        "thread_id": thread_id,
        "thread_id_by_platform": thread_id_by_platform or {},
        "notify_steps": notify_steps,
    }
    return metadata, notify_config


def _should_notify_chat(event_type: str, notify_steps: bool) -> bool:
    """Check whether an event should trigger chat notifications."""
    critical_events = {
        "workflow_start",
        "workflow_complete",
        "workflow_failed",
        "workflow_terminated",
        "workflow_human_approval_required",
        "workflow_human_approval_received",
        "workflow_human_approval_timeout",
    }
    step_events = {
        "workflow_step_complete",
        "workflow_step_failed",
        "workflow_step_skipped",
    }
    if event_type in critical_events:
        return True
    if notify_steps and event_type in step_events:
        return True
    return False


def _format_workflow_message(event_type: str, payload: dict[str, Any]) -> str:
    """Format a human-readable workflow update for chat channels."""
    workflow_name = payload.get("workflow_name") or payload.get("definition_id") or "workflow"
    execution_id = payload.get("workflow_id") or payload.get("execution_id") or ""
    step_name = payload.get("step_name") or payload.get("step_id")

    if event_type == "workflow_start":
        return f"Workflow started: {workflow_name} ({execution_id})"
    if event_type == "workflow_complete":
        return f"Workflow completed: {workflow_name} ({execution_id})"
    if event_type == "workflow_failed":
        error = payload.get("error")
        suffix = f" Error: {error}" if error else ""
        return f"Workflow failed: {workflow_name} ({execution_id}).{suffix}"
    if event_type == "workflow_terminated":
        return f"Workflow terminated: {workflow_name} ({execution_id})"
    if event_type == "workflow_human_approval_required":
        request_id = payload.get("request_id")
        return f"Approval required for {workflow_name} ({execution_id}). Request: {request_id}"
    if event_type == "workflow_human_approval_received":
        status = payload.get("status")
        return f"Approval {status} for {workflow_name} ({execution_id})"
    if event_type == "workflow_human_approval_timeout":
        return f"Approval timed out for {workflow_name} ({execution_id})"
    if event_type.startswith("workflow_step_") and step_name:
        status = str(payload.get("status", event_type.replace("workflow_step_", ""))).upper()
        return f"Step {status}: {step_name} ({workflow_name})"
    return f"Workflow update: {workflow_name} ({execution_id})"


async def _dispatch_chat_message(
    *,
    text: str,
    channel_targets: list[str],
    thread_id: str | None,
    thread_id_by_platform: dict[str, str],
) -> None:
    """Send workflow update to configured chat targets."""
    if not channel_targets:
        return
    try:
        from aragora.approvals.chat import parse_chat_targets
        from aragora.connectors.chat.registry import get_connector
    except Exception:
        return

    parsed = parse_chat_targets(channel_targets)
    for platform, channels in parsed.items():
        connector = get_connector(platform)
        if connector is None or not connector.is_configured:
            continue

        platform_thread = thread_id_by_platform.get(platform) or thread_id
        for channel_id in channels:
            try:
                await connector.send_message(
                    channel_id=channel_id,
                    text=text,
                    thread_id=platform_thread,
                )
            except Exception as exc:
                logger.debug(
                    "Failed to send workflow update to %s:%s: %s",
                    platform,
                    channel_id,
                    exc,
                )


def _schedule_chat_dispatch(coro: Any) -> None:
    """Schedule async chat dispatch without blocking workflow execution."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        loop.create_task(coro)
        return

    try:
        asyncio.run(coro)
    except Exception:
        pass


def _build_event_callback(
    *,
    event_emitter: Any,
    tenant_id: str,
    user_id: str | None,
    org_id: str | None,
    workflow_definition_id: str,
    execution_id: str,
    notify_config: dict[str, Any],
) -> Callable[[str, dict[str, Any]], None]:
    """Create callback that bridges workflow events to stream, webhooks, and chat."""
    channel_targets = notify_config.get("channel_targets") or []
    thread_id = notify_config.get("thread_id")
    thread_id_by_platform = notify_config.get("thread_id_by_platform") or {}
    notify_steps = bool(notify_config.get("notify_steps"))

    def _callback(event_type: str, data: dict[str, Any]) -> None:
        payload = dict(data)
        payload.setdefault("tenant_id", tenant_id)
        if user_id:
            payload.setdefault("user_id", user_id)
        if org_id:
            payload.setdefault("org_id", org_id)
        payload.setdefault("workflow_definition_id", workflow_definition_id)
        payload.setdefault("execution_id", execution_id)

        if event_emitter is not None:
            try:
                from aragora.events.types import StreamEvent, StreamEventType

                stream_type = StreamEventType(event_type)
                event_emitter.emit(StreamEvent(type=stream_type, data=payload))
            except Exception as exc:
                logger.debug("Workflow event emitter failed: %s", exc)

        try:
            from aragora.events.dispatcher import dispatch_event

            dispatch_event(event_type, payload)
        except Exception:
            pass

        if channel_targets and _should_notify_chat(event_type, notify_steps):
            text = _format_workflow_message(event_type, payload)
            _schedule_chat_dispatch(
                _dispatch_chat_message(
                    text=text,
                    channel_targets=channel_targets,
                    thread_id=thread_id,
                    thread_id_by_platform=thread_id_by_platform,
                )
            )

    return _callback


async def execute_workflow(
    workflow_id: str,
    inputs: dict[str, Any] | None = None,
    tenant_id: str = "default",
    user_id: str | None = None,
    org_id: str | None = None,
    event_emitter: Any | None = None,
) -> dict[str, Any]:
    """
    Execute a workflow.

    Args:
        workflow_id: ID of workflow to execute
        inputs: Input parameters for the workflow
        tenant_id: Tenant ID for isolation
        user_id: Optional user ID for audit/event metadata
        org_id: Optional organization ID for audit/event metadata
        event_emitter: Optional event emitter for workflow progress updates

    Returns:
        Execution result
    """
    store = _get_store()
    workflow = store.get_workflow(workflow_id, tenant_id)
    if not workflow:
        raise ValueError(f"Workflow not found: {workflow_id}")

    execution_id = f"exec_{uuid.uuid4().hex[:12]}"
    inputs = inputs or {}

    metadata, notify_config = _extract_notification_context(
        workflow=workflow,
        inputs=inputs,
        tenant_id=tenant_id,
        user_id=user_id,
        org_id=org_id,
    )
    event_callback = _build_event_callback(
        event_emitter=event_emitter,
        tenant_id=tenant_id,
        user_id=user_id,
        org_id=org_id,
        workflow_definition_id=workflow_id,
        execution_id=execution_id,
        notify_config=notify_config,
    )

    # Store execution state - typed as dict[str, Any] to accommodate mixed value types on update
    execution: dict[str, Any] = {
        "id": execution_id,
        "workflow_id": workflow_id,
        "tenant_id": tenant_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
    }
    store.save_execution(execution)

    try:
        result = await _get_engine().execute(
            workflow,
            inputs,
            execution_id,
            metadata=metadata,
            event_callback=event_callback,
        )

        execution.update(
            {
                "status": "completed" if result.success else "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "outputs": result.final_output,
                "steps": [_step_result_to_dict(s) for s in result.steps],
                "error": result.error,
                "duration_ms": result.total_duration_ms,
            }
        )
        store.save_execution(execution)

        from aragora.server.handlers import workflows as workflows_module

        if workflows_module.audit_data is not None:
            workflows_module.audit_data(
                user_id="system",
                resource_type="workflow_execution",
                resource_id=execution_id,
                action="execute",
                workflow_id=workflow_id,
                status=execution["status"],
                tenant_id=tenant_id,
            )

        return execution

    except (ValueError, KeyError, TypeError) as e:
        logger.warning(f"Invalid workflow configuration or inputs: {e}")
        execution.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }
        )
        store.save_execution(execution)
        raise
    except OSError as e:
        logger.error(f"Storage error during workflow execution: {e}")
        execution.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }
        )
        store.save_execution(execution)
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Connection error during workflow execution: {e}")
        execution.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }
        )
        store.save_execution(execution)
        raise


async def get_execution(execution_id: str) -> dict[str, Any] | None:
    """Get execution status and result."""
    store = _get_store()
    return store.get_execution(execution_id)


async def list_executions(
    workflow_id: str | None = None,
    tenant_id: str = "default",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List workflow executions."""
    store = _get_store()
    executions, _ = store.list_executions(
        workflow_id=workflow_id,
        tenant_id=tenant_id,
        limit=limit,
    )
    return executions


async def terminate_execution(execution_id: str) -> bool:
    """Request termination of a running execution."""
    store = _get_store()
    execution = store.get_execution(execution_id)
    if not execution:
        return False

    if execution.get("status") != "running":
        return False

    _get_engine().request_termination("User requested")
    execution["status"] = "terminated"
    execution["completed_at"] = datetime.now(timezone.utc).isoformat()
    store.save_execution(execution)

    return True


__all__ = [
    "execute_workflow",
    "get_execution",
    "list_executions",
    "terminate_execution",
]
