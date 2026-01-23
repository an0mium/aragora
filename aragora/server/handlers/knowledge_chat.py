"""
HTTP Handler for Knowledge + Chat Bridge.

Provides REST API endpoints for chat-knowledge integration:
- POST /api/v1/chat/knowledge/search - Search knowledge from chat context
- POST /api/v1/chat/knowledge/inject - Get relevant knowledge for conversation
- POST /api/v1/chat/knowledge/store - Store chat as knowledge
- GET /api/v1/chat/knowledge/channel/:id/summary - Get channel knowledge summary
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)

# Lazy-loaded bridge instance
_bridge = None


def _get_bridge():
    """Get or create the Knowledge + Chat bridge."""
    global _bridge
    if _bridge is None:
        from aragora.services.knowledge_chat_bridge import get_knowledge_chat_bridge

        _bridge = get_knowledge_chat_bridge()
    return _bridge


async def handle_knowledge_search(
    query: str,
    workspace_id: str = "default",
    channel_id: Optional[str] = None,
    user_id: Optional[str] = None,
    scope: str = "workspace",
    strategy: str = "hybrid",
    node_types: Optional[List[str]] = None,
    min_confidence: float = 0.3,
    max_results: int = 10,
) -> Dict[str, Any]:
    """
    Search knowledge from chat context.

    POST /api/v1/chat/knowledge/search
    {
        "query": "What's the policy on remote work?",
        "workspace_id": "ws_123",
        "channel_id": "C123456",
        "scope": "workspace",
        "strategy": "hybrid",
        "node_types": ["policy", "document"],
        "max_results": 10
    }
    """
    from aragora.services.knowledge_chat_bridge import (
        KnowledgeSearchScope,
        RelevanceStrategy,
    )

    try:
        bridge = _get_bridge()

        # Parse enums
        try:
            search_scope = KnowledgeSearchScope(scope)
        except ValueError:
            search_scope = KnowledgeSearchScope.WORKSPACE

        try:
            search_strategy = RelevanceStrategy(strategy)
        except ValueError:
            search_strategy = RelevanceStrategy.HYBRID

        # Execute search
        context = await bridge.search_knowledge(
            query=query,
            workspace_id=workspace_id,
            channel_id=channel_id,
            user_id=user_id,
            scope=search_scope,
            strategy=search_strategy,
            node_types=node_types,
            min_confidence=min_confidence,
            max_results=max_results,
        )

        return {
            "success": True,
            **context.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Knowledge search failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_knowledge_inject(
    messages: List[Dict[str, Any]],
    workspace_id: str = "default",
    channel_id: Optional[str] = None,
    max_context_items: int = 5,
) -> Dict[str, Any]:
    """
    Get relevant knowledge to inject into a conversation.

    POST /api/v1/chat/knowledge/inject
    {
        "messages": [
            {"author": "user1", "content": "What's our vacation policy?"},
            {"author": "user2", "content": "I think it's in the handbook"}
        ],
        "workspace_id": "ws_123",
        "channel_id": "C123456",
        "max_context_items": 5
    }
    """
    try:
        bridge = _get_bridge()

        results = await bridge.inject_knowledge_for_conversation(
            messages=messages,
            workspace_id=workspace_id,
            channel_id=channel_id,
            max_context_items=max_context_items,
        )

        return {
            "success": True,
            "context": [r.to_dict() for r in results],
            "item_count": len(results),
        }

    except Exception as e:
        logger.exception(f"Knowledge injection failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_store_chat_knowledge(
    messages: List[Dict[str, Any]],
    workspace_id: str = "default",
    channel_id: str = "",
    channel_name: str = "",
    platform: str = "unknown",
    node_type: str = "chat_context",
) -> Dict[str, Any]:
    """
    Store chat messages as knowledge.

    POST /api/v1/chat/knowledge/store
    {
        "messages": [
            {"author": "user1", "content": "We decided to use Python 3.11"},
            {"author": "user2", "content": "Agreed, it has better performance"}
        ],
        "workspace_id": "ws_123",
        "channel_id": "C123456",
        "channel_name": "#engineering",
        "platform": "slack"
    }
    """
    try:
        if len(messages) < 2:
            return {
                "success": False,
                "error": "At least 2 messages required",
            }

        bridge = _get_bridge()

        node_id = await bridge.store_chat_as_knowledge(
            messages=messages,
            workspace_id=workspace_id,
            channel_id=channel_id,
            channel_name=channel_name,
            platform=platform,
            node_type=node_type,
        )

        if node_id:
            return {
                "success": True,
                "node_id": node_id,
                "message_count": len(messages),
            }
        else:
            return {
                "success": False,
                "error": "Failed to store knowledge",
            }

    except Exception as e:
        logger.exception(f"Store chat knowledge failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_channel_knowledge_summary(
    channel_id: str,
    workspace_id: str = "default",
    max_items: int = 10,
) -> Dict[str, Any]:
    """
    Get a summary of knowledge related to a channel.

    GET /api/v1/chat/knowledge/channel/:id/summary
    """
    try:
        bridge = _get_bridge()

        summary = await bridge.get_channel_knowledge_summary(
            channel_id=channel_id,
            workspace_id=workspace_id,
            max_items=max_items,
        )

        return {
            "success": True,
            **summary,
        }

    except Exception as e:
        logger.exception(f"Channel summary failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


class KnowledgeChatHandler(BaseHandler):
    """
    HTTP handler for Knowledge + Chat bridge endpoints.
    """

    ROUTES = [
        "/api/v1/chat/knowledge/search",
        "/api/v1/chat/knowledge/inject",
        "/api/v1/chat/knowledge/store",
    ]

    ROUTE_PREFIXES = [
        "/api/v1/chat/knowledge/channel/",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the request."""
        if path in self.ROUTES:
            return True
        for prefix in self.ROUTE_PREFIXES:
            if path.startswith(prefix):
                return True
        return False

    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle GET requests."""
        # GET /api/v1/chat/knowledge/channel/:id/summary
        if path.startswith("/api/v1/chat/knowledge/channel/") and path.endswith("/summary"):
            # Extract channel_id from path
            parts = path.split("/")
            if len(parts) >= 7:
                channel_id = parts[5]
                workspace_id = query_params.get("workspace_id", "default")
                max_items = int(query_params.get("max_items", 10))

                import asyncio

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        handle_channel_knowledge_summary(
                            channel_id=channel_id,
                            workspace_id=workspace_id,
                            max_items=max_items,
                        )
                    )
                finally:
                    loop.close()

                if result.get("success"):
                    return success_response(result)
                else:
                    return error_response(result.get("error", "Unknown error"), 400)

        return None

    def handle_post(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests."""
        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            if path == "/api/v1/chat/knowledge/search":
                query = body.get("query")
                if not query:
                    return error_response("query is required", 400)

                result = loop.run_until_complete(
                    handle_knowledge_search(
                        query=query,
                        workspace_id=body.get("workspace_id", "default"),
                        channel_id=body.get("channel_id"),
                        user_id=body.get("user_id"),
                        scope=body.get("scope", "workspace"),
                        strategy=body.get("strategy", "hybrid"),
                        node_types=body.get("node_types"),
                        min_confidence=body.get("min_confidence", 0.3),
                        max_results=body.get("max_results", 10),
                    )
                )

            elif path == "/api/v1/chat/knowledge/inject":
                messages = body.get("messages")
                if not messages:
                    return error_response("messages is required", 400)

                result = loop.run_until_complete(
                    handle_knowledge_inject(
                        messages=messages,
                        workspace_id=body.get("workspace_id", "default"),
                        channel_id=body.get("channel_id"),
                        max_context_items=body.get("max_context_items", 5),
                    )
                )

            elif path == "/api/v1/chat/knowledge/store":
                messages = body.get("messages")
                if not messages or len(messages) < 2:
                    return error_response("At least 2 messages required", 400)

                result = loop.run_until_complete(
                    handle_store_chat_knowledge(
                        messages=messages,
                        workspace_id=body.get("workspace_id", "default"),
                        channel_id=body.get("channel_id", ""),
                        channel_name=body.get("channel_name", ""),
                        platform=body.get("platform", "unknown"),
                        node_type=body.get("node_type", "chat_context"),
                    )
                )

            else:
                return None

        finally:
            loop.close()

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)


__all__ = [
    "KnowledgeChatHandler",
    "handle_knowledge_search",
    "handle_knowledge_inject",
    "handle_store_chat_knowledge",
    "handle_channel_knowledge_summary",
]
