"""
Slack slash commands package.

Contains implementations for individual slash commands:
- help: Show available commands
- status: Get system status
- agents: List available agents
- ask: Ask a question
- search: Search debates
- leaderboard: Show ELO rankings
- recent: Show recent debates
- gauntlet: Run a gauntlet challenge
- debate: Start a debate

Mixins:

    from aragora.server.handlers.social.slack.commands import CommandsMixin

    class MyHandler(CommandsMixin):
        pass
"""

from .base import CommandsMixin

__all__ = ["CommandsMixin"]
