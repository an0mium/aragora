"""
Extension Handlers - HTTP endpoints for extension management.

Provides REST API endpoints for:
- Extension status and statistics
- Gastown workspace/convoy management
- Moltbot inbox/gateway/onboarding management
- Agent Fabric operations
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aragora.rbac.decorators import require_permission
from aragora.server.extensions import get_extension_state

if TYPE_CHECKING:
    from aragora.rbac.models import AuthorizationContext

logger = logging.getLogger(__name__)


# =============================================================================
# Extension Status Endpoints
# =============================================================================


async def handle_extensions_status(ctx: "AuthorizationContext") -> dict[str, Any]:
    """
    GET /api/extensions/status

    Get the status of all extensions.
    """
    state = get_extension_state()
    if not state:
        return {
            "error": "Extensions not initialized",
            "status": "unavailable",
        }

    return {
        "status": "ok",
        "extensions": {
            "agent_fabric": {
                "enabled": state.fabric_enabled,
                "available": state.metadata.get("fabric_available", False),
            },
            "gastown": {
                "enabled": state.gastown_enabled,
                "available": state.metadata.get("gastown_available", False),
            },
            "moltbot": {
                "enabled": state.moltbot_enabled,
                "available": state.metadata.get("moltbot_available", False),
            },
            "computer_use": {
                "enabled": state.computer_use_enabled,
                "available": state.metadata.get("computer_use_available", False),
            },
        },
    }


async def handle_extensions_stats(ctx: "AuthorizationContext") -> dict[str, Any]:
    """
    GET /api/extensions/stats

    Get statistics from all enabled extensions.
    """
    state = get_extension_state()
    if not state:
        return {"error": "Extensions not initialized"}

    stats: dict[str, Any] = {"status": "ok"}

    # Fabric stats
    if state.fabric_enabled and state.fabric:
        try:
            fabric_stats = await state.fabric.get_stats()
            stats["agent_fabric"] = fabric_stats
        except Exception as e:
            stats["agent_fabric"] = {"error": str(e)}

    # Gastown stats
    if state.gastown_enabled and state.coordinator:
        try:
            gastown_stats = await state.coordinator.get_stats()
            stats["gastown"] = gastown_stats
        except Exception as e:
            stats["gastown"] = {"error": str(e)}

    # Moltbot stats
    if state.moltbot_enabled:
        moltbot_stats = {}
        if state.inbox_manager:
            try:
                moltbot_stats["inbox"] = await state.inbox_manager.get_stats()
            except Exception as e:
                moltbot_stats["inbox"] = {"error": str(e)}
        if state.local_gateway:
            try:
                moltbot_stats["gateway"] = await state.local_gateway.get_stats()
            except Exception as e:
                moltbot_stats["gateway"] = {"error": str(e)}
        if state.voice_processor:
            try:
                moltbot_stats["voice"] = await state.voice_processor.get_stats()
            except Exception as e:
                moltbot_stats["voice"] = {"error": str(e)}
        if state.onboarding:
            try:
                moltbot_stats["onboarding"] = await state.onboarding.get_stats()
            except Exception as e:
                moltbot_stats["onboarding"] = {"error": str(e)}
        stats["moltbot"] = moltbot_stats

    return stats


# =============================================================================
# Gastown Endpoints
# =============================================================================


@require_permission("workspaces:read")
async def handle_gastown_workspaces_list(ctx: "AuthorizationContext") -> dict[str, Any]:
    """
    GET /api/extensions/gastown/workspaces

    List all workspaces.
    """
    state = get_extension_state()
    if not state or not state.gastown_enabled:
        return {"error": "Gastown extension not enabled"}

    if not state.workspace_manager:
        return {"error": "Workspace manager not available"}

    try:
        workspaces = await state.workspace_manager.list_workspaces()
        return {
            "status": "ok",
            "workspaces": [
                {
                    "id": w.id,
                    "name": w.config.name,
                    "status": w.status,
                    "rigs": len(w.rigs),
                    "created_at": w.created_at.isoformat(),
                }
                for w in workspaces
            ],
        }
    except Exception as e:
        return {"error": str(e)}


@require_permission("workspaces:write")
async def handle_gastown_workspace_create(
    ctx: "AuthorizationContext",
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    POST /api/extensions/gastown/workspaces

    Create a new workspace.
    """
    state = get_extension_state()
    if not state or not state.gastown_enabled:
        return {"error": "Gastown extension not enabled"}

    if not state.coordinator:
        return {"error": "Coordinator not available"}

    try:
        workspace = await state.coordinator.create_workspace(
            name=data.get("name", "Unnamed Workspace"),
            root_path=data.get("root_path", "/tmp/workspace"),
            description=data.get("description", ""),
            owner_id=ctx.user_id if ctx else "",
            tenant_id=ctx.org_id if ctx else None,
        )
        return {
            "status": "ok",
            "workspace": {
                "id": workspace.id,
                "name": workspace.config.name,
                "status": workspace.status,
            },
        }
    except Exception as e:
        return {"error": str(e)}


@require_permission("convoys:read")
async def handle_gastown_convoys_list(ctx: "AuthorizationContext") -> dict[str, Any]:
    """
    GET /api/extensions/gastown/convoys

    List all convoys.
    """
    state = get_extension_state()
    if not state or not state.gastown_enabled:
        return {"error": "Gastown extension not enabled"}

    if not state.convoy_tracker:
        return {"error": "Convoy tracker not available"}

    try:
        convoys = await state.convoy_tracker.list_convoys()
        return {
            "status": "ok",
            "convoys": [
                {
                    "id": c.id,
                    "title": c.title,
                    "status": c.status.value,
                    "rig_id": c.rig_id,
                    "created_at": c.created_at.isoformat(),
                }
                for c in convoys
            ],
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Moltbot Endpoints
# =============================================================================


@require_permission("inbox:read")
async def handle_moltbot_inbox_messages(ctx: "AuthorizationContext") -> dict[str, Any]:
    """
    GET /api/extensions/moltbot/inbox/messages

    List inbox messages.
    """
    state = get_extension_state()
    if not state or not state.moltbot_enabled:
        return {"error": "Moltbot extension not enabled"}

    if not state.inbox_manager:
        return {"error": "Inbox manager not available"}

    try:
        messages = await state.inbox_manager.list_messages(
            user_id=ctx.user_id if ctx else None,
            limit=100,
        )
        return {
            "status": "ok",
            "messages": [
                {
                    "id": m.id,
                    "channel_id": m.channel_id,
                    "direction": m.direction,
                    "content": m.content[:100] + "..." if len(m.content) > 100 else m.content,
                    "status": m.status.value,
                    "created_at": m.created_at.isoformat(),
                }
                for m in messages
            ],
        }
    except Exception as e:
        return {"error": str(e)}


@require_permission("devices:read")
async def handle_moltbot_gateway_devices(ctx: "AuthorizationContext") -> dict[str, Any]:
    """
    GET /api/extensions/moltbot/gateway/devices

    List registered devices.
    """
    state = get_extension_state()
    if not state or not state.moltbot_enabled:
        return {"error": "Moltbot extension not enabled"}

    if not state.local_gateway:
        return {"error": "Gateway not available"}

    try:
        devices = await state.local_gateway.list_devices()
        return {
            "status": "ok",
            "devices": [
                {
                    "id": d.id,
                    "name": d.config.name,
                    "type": d.config.device_type,
                    "status": d.status,
                    "last_seen": d.last_seen.isoformat() if d.last_seen else None,
                }
                for d in devices
            ],
        }
    except Exception as e:
        return {"error": str(e)}


@require_permission("onboarding:read")
async def handle_moltbot_onboarding_flows(ctx: "AuthorizationContext") -> dict[str, Any]:
    """
    GET /api/extensions/moltbot/onboarding/flows

    List onboarding flows.
    """
    state = get_extension_state()
    if not state or not state.moltbot_enabled:
        return {"error": "Moltbot extension not enabled"}

    if not state.onboarding:
        return {"error": "Onboarding orchestrator not available"}

    try:
        flows = await state.onboarding.list_flows()
        return {
            "status": "ok",
            "flows": [
                {
                    "id": f.id,
                    "name": f.name,
                    "status": f.status,
                    "steps": len(f.steps),
                    "started": f.started_count,
                    "completed": f.completed_count,
                }
                for f in flows
            ],
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Agent Fabric Endpoints
# =============================================================================


@require_permission("agents:read")
async def handle_fabric_agents_list(ctx: "AuthorizationContext") -> dict[str, Any]:
    """
    GET /api/extensions/fabric/agents

    List agents in the fabric.
    """
    state = get_extension_state()
    if not state or not state.fabric_enabled:
        return {"error": "Agent Fabric not enabled"}

    if not state.fabric:
        return {"error": "Fabric not available"}

    try:
        agents = await state.fabric.list_agents()
        return {
            "status": "ok",
            "agents": [
                {
                    "id": a.id,
                    "model": a.config.model,
                    "status": a.status.value,
                    "created_at": a.created_at,
                }
                for a in agents
            ],
        }
    except Exception as e:
        return {"error": str(e)}


@require_permission("tasks:read")
async def handle_fabric_tasks_list(ctx: "AuthorizationContext") -> dict[str, Any]:
    """
    GET /api/extensions/fabric/tasks

    List tasks in the fabric.
    """
    state = get_extension_state()
    if not state or not state.fabric_enabled:
        return {"error": "Agent Fabric not enabled"}

    if not state.fabric:
        return {"error": "Fabric not available"}

    try:
        # Get scheduler stats which includes task info
        stats = await state.fabric.get_stats()
        return {
            "status": "ok",
            "stats": stats,
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Route Registration Helper
# =============================================================================


def get_extension_routes() -> dict[str, tuple]:
    """
    Get all extension routes for registration with the server.

    Returns:
        Dictionary mapping route patterns to (handler, methods) tuples
    """
    return {
        # Status
        "/api/extensions/status": (handle_extensions_status, ["GET"]),
        "/api/extensions/stats": (handle_extensions_stats, ["GET"]),
        # Gastown
        "/api/extensions/gastown/workspaces": (handle_gastown_workspaces_list, ["GET"]),
        "/api/extensions/gastown/workspaces/create": (handle_gastown_workspace_create, ["POST"]),
        "/api/extensions/gastown/convoys": (handle_gastown_convoys_list, ["GET"]),
        # Moltbot
        "/api/extensions/moltbot/inbox/messages": (handle_moltbot_inbox_messages, ["GET"]),
        "/api/extensions/moltbot/gateway/devices": (handle_moltbot_gateway_devices, ["GET"]),
        "/api/extensions/moltbot/onboarding/flows": (handle_moltbot_onboarding_flows, ["GET"]),
        # Agent Fabric
        "/api/extensions/fabric/agents": (handle_fabric_agents_list, ["GET"]),
        "/api/extensions/fabric/tasks": (handle_fabric_tasks_list, ["GET"]),
    }
