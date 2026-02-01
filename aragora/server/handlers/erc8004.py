"""
HTTP Handler for ERC-8004 blockchain operations.

Provides REST API endpoints for interacting with ERC-8004 registries:
- GET /api/v1/blockchain/agents - List on-chain agents
- GET /api/v1/blockchain/agents/{token_id} - Get agent identity
- POST /api/v1/blockchain/agents - Register new agent
- GET /api/v1/blockchain/agents/{token_id}/reputation - Get reputation
- POST /api/v1/blockchain/agents/{token_id}/reputation - Submit feedback
- GET /api/v1/blockchain/agents/{token_id}/validations - Get validations
- GET /api/v1/blockchain/config - Get chain configuration
- POST /api/v1/blockchain/sync - Trigger manual sync
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)
from aragora.server.handlers.openapi_decorator import api_endpoint
from aragora.rbac.decorators import require_permission
from aragora.resilience import with_timeout

logger = logging.getLogger(__name__)

# Lazy-loaded components
_provider = None
_connector = None
_adapter = None


def _get_provider():
    """Get or create the Web3 provider."""
    global _provider
    if _provider is None:
        try:
            from aragora.blockchain.provider import Web3Provider

            _provider = Web3Provider.from_env()
        except ImportError:
            raise ImportError(
                "web3 is required for blockchain endpoints. "
                "Install with: pip install aragora[blockchain]"
            )
        except Exception as e:
            logger.error(f"Failed to create Web3Provider: {e}")
            raise
    return _provider


def _get_connector():
    """Get or create the ERC-8004 connector."""
    global _connector
    if _connector is None:
        from aragora.connectors.blockchain import ERC8004Connector

        _connector = ERC8004Connector.from_env()
    return _connector


def _get_adapter():
    """Get or create the ERC-8004 adapter."""
    global _adapter
    if _adapter is None:
        from aragora.knowledge.mound.adapters.erc8004_adapter import ERC8004Adapter

        _adapter = ERC8004Adapter(provider=_get_provider())
    return _adapter


@api_endpoint(
    method="GET",
    path="/api/v1/blockchain/config",
    summary="Get blockchain configuration",
    description="Returns current chain configuration and connectivity status.",
    tags=["Blockchain"],
    responses={
        "200": {"description": "Configuration returned"},
        "500": {"description": "Configuration error"},
    },
)
@require_permission("blockchain:read")
@with_timeout(10.0)
async def handle_blockchain_config() -> HandlerResult:
    """Get blockchain configuration and connectivity status."""
    try:
        provider = _get_provider()
        config = provider.get_config()

        return success_response(
            {
                "chain_id": config.chain_id,
                "rpc_url": config.rpc_url[:50] + "..."
                if len(config.rpc_url) > 50
                else config.rpc_url,
                "identity_registry": config.identity_registry_address or None,
                "reputation_registry": config.reputation_registry_address or None,
                "validation_registry": config.validation_registry_address or None,
                "block_confirmations": config.block_confirmations,
                "is_connected": provider.is_connected(),
                "health": provider.get_health_status(),
            }
        )
    except ImportError as e:
        return error_response(str(e), status=501)
    except Exception as e:
        logger.error(f"Error getting blockchain config: {e}")
        return error_response(f"Configuration error: {str(e)}")


@api_endpoint(
    method="GET",
    path="/api/v1/blockchain/agents/{token_id}",
    summary="Get agent identity",
    description="Retrieves on-chain agent identity by token ID.",
    tags=["Blockchain", "Agents"],
    responses={
        "200": {"description": "Agent identity returned"},
        "404": {"description": "Agent not found"},
        "500": {"description": "Fetch error"},
    },
)
@require_permission("blockchain:read")
@with_timeout(15.0)
async def handle_get_agent(token_id: int) -> dict[str, Any]:
    """Get agent identity by token ID."""
    try:
        from aragora.blockchain.contracts.identity import IdentityRegistryContract

        provider = _get_provider()
        contract = IdentityRegistryContract(provider)
        identity = contract.get_agent(token_id)

        return success_response(
            {
                "token_id": identity.token_id,
                "owner": identity.owner,
                "agent_uri": identity.agent_uri,
                "wallet_address": identity.wallet_address,
                "chain_id": identity.chain_id,
                "aragora_agent_id": identity.aragora_agent_id,
                "registered_at": identity.registered_at.isoformat()
                if identity.registered_at
                else None,
                "tx_hash": identity.tx_hash,
            }
        )
    except ImportError as e:
        return error_response(str(e), status=501)
    except Exception as e:
        logger.error(f"Error fetching agent {token_id}: {e}")
        return error_response(f"Agent not found: {str(e)}", status=404)


@api_endpoint(
    method="GET",
    path="/api/v1/blockchain/agents/{token_id}/reputation",
    summary="Get agent reputation",
    description="Retrieves aggregated reputation summary for an agent.",
    tags=["Blockchain", "Reputation"],
    responses={
        "200": {"description": "Reputation summary returned"},
        "404": {"description": "Agent or reputation not found"},
        "500": {"description": "Fetch error"},
    },
)
@require_permission("blockchain:read")
@with_timeout(15.0)
async def handle_get_reputation(
    token_id: int,
    tag1: str = "",
    tag2: str = "",
) -> dict[str, Any]:
    """Get reputation summary for an agent."""
    try:
        from aragora.blockchain.contracts.reputation import ReputationRegistryContract

        provider = _get_provider()
        contract = ReputationRegistryContract(provider)
        summary = contract.get_summary(token_id, tag1=tag1, tag2=tag2)

        return success_response(
            {
                "agent_id": summary.agent_id,
                "count": summary.count,
                "summary_value": summary.summary_value,
                "summary_value_decimals": summary.summary_value_decimals,
                "normalized_value": summary.normalized_value,
                "tag1": summary.tag1,
                "tag2": summary.tag2,
            }
        )
    except ImportError as e:
        return error_response(str(e), status=501)
    except Exception as e:
        logger.error(f"Error fetching reputation for agent {token_id}: {e}")
        return error_response(f"Reputation not found: {str(e)}", status=404)


@api_endpoint(
    method="GET",
    path="/api/v1/blockchain/agents/{token_id}/validations",
    summary="Get agent validations",
    description="Retrieves validation summary for an agent.",
    tags=["Blockchain", "Validation"],
    responses={
        "200": {"description": "Validation summary returned"},
        "404": {"description": "Agent or validations not found"},
        "500": {"description": "Fetch error"},
    },
)
@require_permission("blockchain:read")
@with_timeout(15.0)
async def handle_get_validations(
    token_id: int,
    tag: str = "",
) -> dict[str, Any]:
    """Get validation summary for an agent."""
    try:
        from aragora.blockchain.contracts.validation import ValidationRegistryContract

        provider = _get_provider()
        contract = ValidationRegistryContract(provider)
        summary = contract.get_summary(token_id, tag=tag)

        return success_response(
            {
                "agent_id": summary.agent_id,
                "count": summary.count,
                "average_response": summary.average_response,
                "tag": summary.tag,
            }
        )
    except ImportError as e:
        return error_response(str(e), status=501)
    except Exception as e:
        logger.error(f"Error fetching validations for agent {token_id}: {e}")
        return error_response(f"Validations not found: {str(e)}", status=404)


@api_endpoint(
    method="POST",
    path="/api/v1/blockchain/sync",
    summary="Trigger blockchain sync",
    description="Manually triggers synchronization between blockchain and Knowledge Mound.",
    tags=["Blockchain", "Sync"],
    responses={
        "200": {"description": "Sync completed"},
        "500": {"description": "Sync error"},
    },
)
@require_permission("blockchain:write")
@with_timeout(60.0)
async def handle_blockchain_sync(
    sync_identities: bool = True,
    sync_reputation: bool = True,
    sync_validations: bool = True,
    agent_ids: Optional[list[int]] = None,
) -> dict[str, Any]:
    """Trigger manual blockchain sync to Knowledge Mound."""
    try:
        adapter = _get_adapter()
        result = await adapter.sync_to_km(
            agent_ids=agent_ids,
            sync_identities=sync_identities,
            sync_reputation=sync_reputation,
            sync_validations=sync_validations,
        )

        return success_response(
            {
                "records_synced": result.records_synced,
                "records_skipped": result.records_skipped,
                "records_failed": result.records_failed,
                "duration_ms": result.duration_ms,
                "errors": result.errors[:10] if result.errors else [],
            }
        )
    except ImportError as e:
        return error_response(str(e), status=501)
    except Exception as e:
        logger.error(f"Blockchain sync error: {e}")
        return error_response(f"Sync failed: {str(e)}")


@api_endpoint(
    method="GET",
    path="/api/v1/blockchain/health",
    summary="Get blockchain connector health",
    description="Returns health status of the blockchain integration.",
    tags=["Blockchain", "Health"],
    responses={
        "200": {"description": "Health status returned"},
    },
)
@require_permission("blockchain:read")
async def handle_blockchain_health() -> dict[str, Any]:
    """Get blockchain connector health status."""
    try:
        connector = _get_connector()
        health = await connector.health_check()

        adapter_status = {}
        try:
            adapter = _get_adapter()
            adapter_status = adapter.get_health_status()
        except Exception as e:
            adapter_status = {"error": str(e)}

        return success_response(
            {
                "connector": health.to_dict(),
                "adapter": adapter_status,
            }
        )
    except ImportError as e:
        return success_response(
            {
                "available": False,
                "error": str(e),
            }
        )
    except Exception as e:
        return success_response(
            {
                "available": False,
                "error": str(e),
            }
        )


# Handler registry for unified server
BLOCKCHAIN_HANDLERS = {
    "blockchain_config": handle_blockchain_config,
    "blockchain_get_agent": handle_get_agent,
    "blockchain_get_reputation": handle_get_reputation,
    "blockchain_get_validations": handle_get_validations,
    "blockchain_sync": handle_blockchain_sync,
    "blockchain_health": handle_blockchain_health,
}

__all__ = [
    "BLOCKCHAIN_HANDLERS",
    "ERC8004Handler",
    "handle_blockchain_config",
    "handle_get_agent",
    "handle_get_reputation",
    "handle_get_validations",
    "handle_blockchain_sync",
    "handle_blockchain_health",
]


def _get_query_param(query_params: dict[str, Any], name: str, default: str = "") -> str:
    value = query_params.get(name, default)
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


class ERC8004Handler(BaseHandler):
    """Handler for ERC-8004 blockchain API endpoints."""

    ROUTES = [
        "/api/v1/blockchain/config",
        "/api/v1/blockchain/health",
        "/api/v1/blockchain/sync",
        "/api/v1/blockchain/agents",
        "/api/v1/blockchain/agents/*",
    ]

    def can_handle(self, path: str) -> bool:
        return path.startswith("/api/v1/blockchain/")

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        method = handler.command if hasattr(handler, "command") else "GET"

        if path == "/api/v1/blockchain/config" and method == "GET":
            return handle_blockchain_config()

        if path == "/api/v1/blockchain/health" and method == "GET":
            return handle_blockchain_health()

        if path == "/api/v1/blockchain/sync" and method == "POST":
            body = self.read_json_body(handler) or {}
            return handle_blockchain_sync(
                sync_identities=bool(body.get("sync_identities", True)),
                sync_reputation=bool(body.get("sync_reputation", True)),
                sync_validations=bool(body.get("sync_validations", True)),
                agent_ids=body.get("agent_ids"),
            )

        if path == "/api/v1/blockchain/agents":
            if method == "GET":
                return error_response("Agent listing not implemented", status_code=501)
            if method == "POST":
                return error_response("Agent registration not implemented", status_code=501)
            return error_response(f"Method {method} not allowed", status_code=405)

        if path.startswith("/api/v1/blockchain/agents/"):
            suffix = path[len("/api/v1/blockchain/agents/") :]
            parts = [p for p in suffix.split("/") if p]
            if not parts:
                return error_response("Invalid agent path", status=400)
            try:
                token_id = int(parts[0])
            except ValueError:
                return error_response("Invalid token_id", status=400)

            if len(parts) == 1 and method == "GET":
                return handle_get_agent(token_id)

            if len(parts) == 2 and parts[1] == "reputation" and method == "GET":
                tag1 = _get_query_param(query_params, "tag1", "")
                tag2 = _get_query_param(query_params, "tag2", "")
                return handle_get_reputation(token_id, tag1=tag1, tag2=tag2)

            if len(parts) == 2 and parts[1] == "validations" and method == "GET":
                tag = _get_query_param(query_params, "tag", "")
                return handle_get_validations(token_id, tag=tag)

            return error_response("Invalid blockchain agent endpoint", status=400)

        return error_response("Invalid path", status=400)
