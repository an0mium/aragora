"""
Aragora Inbox Package.

Provides inbox-to-debate routing, auto-spawning debates for high-priority
incoming messages across all integration channels.

Usage:
    from aragora.inbox import InboxDebateRouter, RouterConfig, get_inbox_debate_router

    # Get or create the global router
    router = get_inbox_debate_router()

    # Or create with custom config
    config = RouterConfig(
        enabled=True,
        priority_threshold="high",
        keyword_patterns=["urgent", "critical", "escalate"],
    )
    router = InboxDebateRouter(config=config)

    # Start listening for events
    await router.start()

    # Manually evaluate a message
    result = router.evaluate_message(message_data)
"""

from aragora.inbox.debate_router import (
    InboxDebateRouter,
    RouterConfig,
    TriggerRule,
    DebateSpawnResult,
    get_inbox_debate_router,
    reset_inbox_debate_router,
)

__all__ = [
    "InboxDebateRouter",
    "RouterConfig",
    "TriggerRule",
    "DebateSpawnResult",
    "get_inbox_debate_router",
    "reset_inbox_debate_router",
]
