"""
MCP Tools for External Integration triggers.

Provides tools for interacting with Zapier, Make, and n8n integrations:
- trigger_external_webhook: Trigger an external automation webhook
- list_integrations: List configured integrations
- test_integration: Test an integration connection
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def trigger_external_webhook_tool(
    platform: str,
    event_type: str,
    data: str = "{}",
) -> Dict[str, Any]:
    """
    Trigger an external automation webhook (Zapier, Make, n8n).

    Args:
        platform: Target platform (zapier, make, n8n)
        event_type: Type of event to trigger
        data: JSON string of event data

    Returns:
        Dict with trigger results
    """
    import json as json_module

    valid_platforms = {"zapier", "make", "n8n"}
    if platform.lower() not in valid_platforms:
        return {"error": f"Invalid platform. Must be one of: {valid_platforms}"}

    # Parse event data
    try:
        event_data = json_module.loads(data) if data else {}
    except json_module.JSONDecodeError:
        return {"error": "Invalid JSON in data parameter"}

    # Add common event metadata
    event_data["timestamp"] = time.time()
    event_data["event_type"] = event_type
    event_data["source"] = "mcp_tool"

    try:
        triggered_count = 0

        if platform == "zapier":
            from aragora.integrations.zapier import get_zapier_integration

            zapier = get_zapier_integration()
            triggered_count = await zapier.fire_trigger(event_type, event_data)

        elif platform == "make":
            from aragora.integrations.make import get_make_integration

            make = get_make_integration()
            triggered_count = await make.trigger_webhooks(event_type, event_data)

        elif platform == "n8n":
            from aragora.integrations.n8n import get_n8n_integration

            n8n = get_n8n_integration()
            triggered_count = await n8n.dispatch_event(event_type, event_data)

        return {
            "platform": platform,
            "event_type": event_type,
            "triggered": triggered_count > 0,
            "webhooks_triggered": triggered_count,
        }

    except ImportError:
        logger.warning(f"{platform} integration not available")
        return {"error": f"{platform} integration module not available"}
    except Exception as e:
        logger.error(f"Failed to trigger {platform} webhook: {e}")
        return {"error": f"Trigger failed: {str(e)}"}


async def list_integrations_tool(
    platform: str = "all",
    workspace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List configured external integrations.

    Args:
        platform: Filter by platform (all, zapier, make, n8n)
        workspace_id: Filter by workspace ID

    Returns:
        Dict with list of configured integrations
    """
    integrations: Dict[str, List[Dict[str, Any]]] = {
        "zapier": [],
        "make": [],
        "n8n": [],
    }

    try:
        if platform in ("all", "zapier"):
            from aragora.integrations.zapier import get_zapier_integration

            zapier = get_zapier_integration()
            apps = zapier.list_apps(workspace_id)
            integrations["zapier"] = [
                {
                    "id": app.id,
                    "workspace_id": app.workspace_id,
                    "active": app.active,
                    "triggers_count": len(app.triggers),
                    "created_at": app.created_at,
                }
                for app in apps
            ]
    except ImportError:
        logger.debug("Zapier integration not available")
    except Exception as e:
        logger.warning(f"Failed to list Zapier integrations: {e}")

    try:
        if platform in ("all", "make"):
            from aragora.integrations.make import get_make_integration

            make = get_make_integration()
            connections = make.list_connections(workspace_id)
            integrations["make"] = [
                {
                    "id": conn.id,
                    "workspace_id": conn.workspace_id,
                    "active": conn.active,
                    "webhooks_count": len(conn.webhooks),
                    "created_at": conn.created_at,
                }
                for conn in connections
            ]
    except ImportError:
        logger.debug("Make integration not available")
    except Exception as e:
        logger.warning(f"Failed to list Make integrations: {e}")

    try:
        if platform in ("all", "n8n"):
            from aragora.integrations.n8n import get_n8n_integration

            n8n = get_n8n_integration()
            credentials = n8n.list_credentials(workspace_id)
            integrations["n8n"] = [
                {
                    "id": cred.id,
                    "workspace_id": cred.workspace_id,
                    "active": cred.active,
                    "webhooks_count": len(cred.webhooks),
                    "created_at": cred.created_at,
                }
                for cred in credentials
            ]
    except ImportError:
        logger.debug("n8n integration not available")
    except Exception as e:
        logger.warning(f"Failed to list n8n integrations: {e}")

    # Calculate totals
    total = sum(len(v) for v in integrations.values())

    return {
        "integrations": integrations if platform == "all" else {platform: integrations.get(platform, [])},
        "total": total,
        "platform_filter": platform,
        "workspace_filter": workspace_id,
    }


async def test_integration_tool(
    platform: str,
    integration_id: str,
) -> Dict[str, Any]:
    """
    Test an integration connection.

    Args:
        platform: Platform (zapier, make, n8n)
        integration_id: ID of the integration to test

    Returns:
        Dict with test results
    """
    valid_platforms = {"zapier", "make", "n8n"}
    if platform.lower() not in valid_platforms:
        return {"error": f"Invalid platform. Must be one of: {valid_platforms}"}

    try:
        if platform == "zapier":
            from aragora.integrations.zapier import get_zapier_integration

            zapier = get_zapier_integration()
            app = zapier.get_app(integration_id)
            if not app:
                return {"error": f"Zapier app {integration_id} not found"}
            return {
                "platform": "zapier",
                "integration_id": integration_id,
                "status": "ok" if app.active else "inactive",
                "triggers_configured": len(app.triggers),
                "total_triggers_fired": app.trigger_count,
            }

        elif platform == "make":
            from aragora.integrations.make import get_make_integration

            make = get_make_integration()
            result = make.test_connection(integration_id)
            return {
                "platform": "make",
                "integration_id": integration_id,
                **result,
            }

        elif platform == "n8n":
            from aragora.integrations.n8n import get_n8n_integration

            n8n = get_n8n_integration()
            cred = n8n.get_credential(integration_id)
            if not cred:
                return {"error": f"n8n credential {integration_id} not found"}
            return {
                "platform": "n8n",
                "integration_id": integration_id,
                "status": "ok" if cred.active else "inactive",
                "webhooks_configured": len(cred.webhooks),
                "total_operations": cred.operation_count,
            }

    except ImportError:
        return {"error": f"{platform} integration module not available"}
    except Exception as e:
        logger.error(f"Failed to test {platform} integration: {e}")
        return {"error": f"Test failed: {str(e)}"}

    return {"error": "Unknown platform"}


async def get_integration_events_tool(
    platform: str,
) -> Dict[str, Any]:
    """
    Get available event types for an integration platform.

    Args:
        platform: Platform (zapier, make, n8n)

    Returns:
        Dict with available event types
    """
    valid_platforms = {"zapier", "make", "n8n"}
    if platform.lower() not in valid_platforms:
        return {"error": f"Invalid platform. Must be one of: {valid_platforms}"}

    try:
        if platform == "zapier":
            from aragora.integrations.zapier import get_zapier_integration

            zapier = get_zapier_integration()
            return {
                "platform": "zapier",
                "trigger_types": zapier.TRIGGER_TYPES,
                "action_types": zapier.ACTION_TYPES,
            }

        elif platform == "make":
            from aragora.integrations.make import get_make_integration

            make = get_make_integration()
            modules = make.MODULE_TYPES
            triggers = {k: v for k, v in modules.items() if v.get("type") == "trigger"}
            actions = {k: v for k, v in modules.items() if v.get("type") == "action"}
            return {
                "platform": "make",
                "trigger_modules": triggers,
                "action_modules": actions,
            }

        elif platform == "n8n":
            from aragora.integrations.n8n import get_n8n_integration

            n8n = get_n8n_integration()
            return {
                "platform": "n8n",
                "event_types": n8n.EVENT_TYPES,
            }

    except ImportError:
        return {"error": f"{platform} integration module not available"}
    except Exception as e:
        logger.error(f"Failed to get {platform} events: {e}")
        return {"error": f"Failed: {str(e)}"}

    return {"error": "Unknown platform"}


# Export all tools
__all__ = [
    "trigger_external_webhook_tool",
    "list_integrations_tool",
    "test_integration_tool",
    "get_integration_events_tool",
]
