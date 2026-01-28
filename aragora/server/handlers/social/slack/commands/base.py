"""
Slack slash command implementations.

Provides mixin classes for handling Slack slash commands.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..utils.responses import slack_response, slack_blocks_response
from ...base import HandlerResult

logger = logging.getLogger(__name__)


class CommandsMixin:
    """Mixin providing basic Slack slash command implementations.

    Should be mixed into a handler class that provides access to Slack utilities.
    """

    def command_help(self) -> HandlerResult:
        """Show help message."""
        help_text = """*Aragora Slash Commands*

*Core Commands:*
`/aragora debate "topic"` - Start a multi-agent debate on a topic
`/aragora ask "question"` - Quick Q&A without full debate
`/aragora gauntlet "statement"` - Run adversarial stress-test validation

*Discovery:*
`/aragora search "query"` - Search debates and evidence
`/aragora recent` - Show recent debates
`/aragora leaderboard` - View agent rankings

*Info:*
`/aragora agents` - List available agents
`/aragora status` - Get system status
`/aragora help` - Show this help message

*Examples:*
- `/aragora debate "Should AI be regulated?"`
- `/aragora ask "What is the capital of France?"`
- `/aragora gauntlet "We should migrate to microservices"`
- `/aragora search "machine learning"`
"""
        return slack_response(help_text, response_type="ephemeral")

    def command_status(self) -> HandlerResult:
        """Get system status."""
        try:
            from aragora.ranking.elo import EloSystem

            store = EloSystem()
            agents = store.get_all_ratings()

            blocks: List[Dict[str, Any]] = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Aragora System Status",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Agents:* {len(agents)}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": "*Status:* Online",
                        },
                    ],
                },
            ]

            return slack_blocks_response(
                blocks,
                text="Aragora is online",
                response_type="ephemeral",
            )

        except ImportError as e:
            logger.warning(f"ELO system not available for status: {e}")
            return slack_response(
                "Status service temporarily unavailable",
                response_type="ephemeral",
            )
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Data error in status command: {e}")
            return slack_response(
                f"Error getting status: {str(e)[:100]}",
                response_type="ephemeral",
            )
        except Exception as e:
            logger.exception(f"Unexpected status command error: {e}")
            return slack_response(
                f"Error getting status: {str(e)[:100]}",
                response_type="ephemeral",
            )

    def command_agents(self) -> HandlerResult:
        """List available agents."""
        try:
            from aragora.ranking.elo import EloSystem

            store = EloSystem()
            agents = store.get_all_ratings()

            if not agents:
                return slack_response(
                    "No agents registered yet.",
                    response_type="ephemeral",
                )

            # Sort by ELO
            sorted_agents = sorted(agents, key=lambda x: x.rating, reverse=True)

            # Build agent list
            agent_lines = []
            for i, agent in enumerate(sorted_agents[:10], 1):
                line = f"{i}. *{agent.agent_id}*: {agent.rating:.0f} ELO"
                agent_lines.append(line)

            blocks: List[Dict[str, Any]] = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Available Agents",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "\n".join(agent_lines) if agent_lines else "No agents available",
                    },
                },
            ]

            if len(sorted_agents) > 10:
                blocks.append(
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"_Showing top 10 of {len(sorted_agents)} agents_",
                            }
                        ],
                    }
                )

            return slack_blocks_response(
                blocks,
                text=f"{len(agents)} agents available",
                response_type="ephemeral",
            )

        except ImportError as e:
            logger.warning(f"ELO system not available for agents: {e}")
            return slack_response(
                "Agent listing temporarily unavailable",
                response_type="ephemeral",
            )
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Data error in agents command: {e}")
            return slack_response(
                f"Error listing agents: {str(e)[:100]}",
                response_type="ephemeral",
            )
        except Exception as e:
            logger.exception(f"Unexpected agents command error: {e}")
            return slack_response(
                f"Error listing agents: {str(e)[:100]}",
                response_type="ephemeral",
            )

    def command_leaderboard(self) -> HandlerResult:
        """Show agent leaderboard."""
        try:
            from aragora.ranking.elo import EloSystem

            store = EloSystem()
            agents = store.get_all_ratings()

            if not agents:
                return slack_response(
                    "No agents ranked yet.",
                    response_type="ephemeral",
                )

            # Sort by ELO
            sorted_agents = sorted(agents, key=lambda x: x.rating, reverse=True)

            # Build leaderboard
            lines = []
            medals = ["Gold", "Silver", "Bronze"]
            for i, agent in enumerate(sorted_agents[:10], 1):
                medal = medals[i - 1] if i <= 3 else str(i)
                line = f"{medal}. *{agent.agent_id}*: {agent.rating:.0f} ELO ({agent.wins}W/{agent.losses}L)"
                lines.append(line)

            blocks: List[Dict[str, Any]] = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Agent Leaderboard",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "\n".join(lines),
                    },
                },
            ]

            return slack_blocks_response(
                blocks,
                text="Agent Leaderboard",
                response_type="ephemeral",
            )

        except ImportError as e:
            logger.warning(f"ELO system not available for leaderboard: {e}")
            return slack_response(
                "Leaderboard temporarily unavailable",
                response_type="ephemeral",
            )
        except Exception as e:
            logger.exception(f"Leaderboard command error: {e}")
            return slack_response(
                f"Error getting leaderboard: {str(e)[:100]}",
                response_type="ephemeral",
            )


__all__ = ["CommandsMixin"]
