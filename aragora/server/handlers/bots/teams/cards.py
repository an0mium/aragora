"""
Teams Bot Adaptive Card action handling.

Handles Adaptive Card actions (invoke activities) including:
- Vote actions
- Summary requests
- View details
- View report
- Share result
- Start debate prompts
- Compose extension actions
- Task module interactions
- Link unfurling
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any

from aragora.audit.unified import audit_data
from aragora.config import DEFAULT_AGENT_LIST, DEFAULT_ROUNDS

from aragora.server.handlers.bots.teams_utils import (
    _active_debates,
    _start_teams_debate,
    _user_votes,
    build_consensus_card,
    build_debate_card,
    get_debate_vote_counts,
)

if TYPE_CHECKING:
    from aragora.server.handlers.bots.teams.handler import TeamsBot

logger = logging.getLogger(__name__)

# Permission constants
PERM_TEAMS_DEBATES_CREATE = "teams:debates:create"
PERM_TEAMS_DEBATES_VOTE = "teams:debates:vote"
PERM_TEAMS_CARDS_RESPOND = "teams:cards:respond"

# Agent display names for UI
AGENT_DISPLAY_NAMES: dict[str, str] = {
    "claude": "Claude",
    "gpt4": "GPT-4",
    "gemini": "Gemini",
    "mistral": "Mistral",
    "deepseek": "DeepSeek",
    "grok": "Grok",
    "qwen": "Qwen",
    "kimi": "Kimi",
    "anthropic-api": "Claude",
    "openai-api": "GPT-4",
}


class TeamsCardActions:
    """Handles Adaptive Card action invokes from Teams.

    Routes card action submits, compose extension actions, and task module
    interactions to the appropriate handlers.
    """

    def __init__(self, bot: TeamsBot):
        """Initialize the card actions handler.

        Args:
            bot: The parent TeamsBot instance.
        """
        self.bot = bot

    async def handle_invoke(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle invoke activity (Adaptive Card actions, compose extensions).

        Routes card action submits and compose extension requests to
        the appropriate handler.
        """
        invoke_name = activity.get("name", "")
        value = activity.get("value", {})
        from_user = activity.get("from", {})
        user_id = from_user.get("id", "")

        logger.info("Teams invoke: %s from %s", invoke_name, user_id)

        # Handle Adaptive Card action submit (most common)
        if invoke_name == "adaptiveCard/action" or not invoke_name:
            return await self._handle_card_action(value, user_id, activity)

        # Compose extension: submit action (messaging extension)
        if invoke_name == "composeExtension/submitAction":
            return await self._handle_compose_extension_submit(value, user_id, activity)

        # Compose extension: query (search in messaging extension)
        if invoke_name == "composeExtension/query":
            return await self._handle_compose_extension_query(value, user_id, activity)

        # Compose extension / task module: fetch task (open dialog)
        if invoke_name in ("composeExtension/fetchTask", "task/fetch"):
            return await self._handle_task_module_fetch(value, user_id, activity)

        # Task module: submit (dialog form submitted)
        if invoke_name == "task/submit":
            return await self._handle_task_module_submit(value, user_id, activity)

        # Messaging extension: link unfurling
        if invoke_name == "composeExtension/queryLink":
            return await self._handle_link_unfurling(value, activity)

        # Default invoke response (required for card actions)
        return {
            "status": 200,
            "body": {"statusCode": 200, "type": "message", "value": "Action processed"},
        }

    async def _handle_card_action(
        self, value: dict[str, Any], user_id: str, activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Route Adaptive Card action submits to specific handlers."""
        action = value.get("action", "")

        # RBAC: Check permission to respond to card actions
        # Skip permission check for help action (always allowed)
        if action != "help":
            perm_error = self.bot._check_permission(activity, PERM_TEAMS_CARDS_RESPOND)
            if perm_error:
                return {
                    "status": 403,
                    "body": {
                        "statusCode": 403,
                        "type": "application/vnd.microsoft.card.adaptive",
                        "value": {
                            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                            "type": "AdaptiveCard",
                            "version": "1.4",
                            "body": [
                                {
                                    "type": "TextBlock",
                                    "text": "Permission Denied",
                                    "weight": "Bolder",
                                    "color": "Attention",
                                },
                                {
                                    "type": "TextBlock",
                                    "text": "You don't have permission to perform this action.",
                                    "wrap": True,
                                },
                            ],
                        },
                    },
                }

        if action == "vote":
            return await self._handle_vote(
                debate_id=value.get("debate_id", ""),
                agent=value.get("agent", value.get("value", "")),
                user_id=user_id,
                activity=activity,
            )
        elif action == "summary":
            return await self._handle_summary(
                debate_id=value.get("debate_id", ""),
                activity=activity,
            )
        elif action == "view_details":
            return await self._handle_view_details(
                debate_id=value.get("debate_id", ""),
                activity=activity,
            )
        elif action == "view_report":
            return await self._handle_view_report(
                debate_id=value.get("debate_id", ""),
                activity=activity,
            )
        elif action == "view_rankings":
            return await self._handle_view_rankings(
                period=value.get("period", "all_time"),
                activity=activity,
            )
        elif action == "watch":
            return await self._handle_watch_debate(
                debate_id=value.get("debate_id", ""),
                user_id=user_id,
                activity=activity,
            )
        elif action == "share":
            return await self._handle_share_result(
                debate_id=value.get("debate_id", ""),
                activity=activity,
            )
        elif action == "start_debate_prompt":
            return await self._handle_start_debate_prompt(activity)
        elif action == "help":
            # Import event processor for help command
            event_processor = self.bot._get_event_processor()
            await event_processor._cmd_help(activity)
            return {
                "status": 200,
                "body": {"statusCode": 200, "type": "message", "value": "Help sent"},
            }

        logger.debug("Unhandled card action: %s", action)
        return {
            "status": 200,
            "body": {"statusCode": 200, "type": "message", "value": "Action acknowledged"},
        }

    async def _handle_vote(
        self, debate_id: str, agent: str, user_id: str, activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle vote action from card."""
        # RBAC: Check permission to vote on debates
        perm_error = self.bot._check_permission(activity, PERM_TEAMS_DEBATES_VOTE, debate_id)
        if perm_error:
            return {
                "status": 403,
                "body": {
                    "statusCode": 403,
                    "type": "application/vnd.microsoft.card.adaptive",
                    "value": {
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": "Permission Denied",
                                "weight": "Bolder",
                                "color": "Attention",
                            },
                            {
                                "type": "TextBlock",
                                "text": "You don't have permission to vote on debates.",
                                "wrap": True,
                            },
                        ],
                    },
                },
            }

        if not debate_id or not agent:
            return {
                "status": 400,
                "body": {"statusCode": 400, "type": "error", "value": "Invalid vote data"},
            }

        if debate_id not in _user_votes:
            _user_votes[debate_id] = {}

        previous_vote = _user_votes[debate_id].get(user_id)
        _user_votes[debate_id][user_id] = agent

        logger.info("Vote recorded: %s voted for %s in %s", user_id, agent, debate_id)

        audit_data(
            user_id=f"teams:{user_id}",
            resource_type="debate_vote",
            resource_id=debate_id,
            action="create",
            vote_option=agent,
            platform="teams",
        )

        if previous_vote and previous_vote != agent:
            message = f"Your vote changed from {previous_vote} to {agent}."
        else:
            message = f"Your vote for {agent} has been recorded!"

        vote_counts = get_debate_vote_counts(debate_id)
        total_votes = sum(vote_counts.values())

        return {
            "status": 200,
            "body": {
                "statusCode": 200,
                "type": "application/vnd.microsoft.card.adaptive",
                "value": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": message,
                            "weight": "Bolder",
                            "color": "Good",
                        },
                        {
                            "type": "TextBlock",
                            "text": f"Total votes cast: {total_votes}",
                            "isSubtle": True,
                            "size": "Small",
                        },
                    ],
                },
            },
        }

    async def _handle_summary(self, debate_id: str, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle summary request - show debate summary as an Adaptive Card."""
        debate_info = _active_debates.get(debate_id)

        if not debate_info:
            return {
                "status": 200,
                "body": {
                    "statusCode": 200,
                    "type": "message",
                    "value": f"Debate {debate_id[:8]}... not found or has completed.",
                },
            }

        topic = debate_info.get("topic", "Unknown")
        started = debate_info.get("started_at", 0)
        elapsed = time.time() - started if started else 0
        vote_counts = get_debate_vote_counts(debate_id)

        facts: list[dict[str, str]] = [
            {"title": "Topic", "value": topic[:100]},
            {"title": "Elapsed", "value": f"{elapsed / 60:.1f} minutes"},
            {"title": "Votes Cast", "value": str(sum(vote_counts.values()))},
        ]

        if vote_counts:
            top_agent = max(vote_counts.keys(), key=lambda k: vote_counts[k])
            facts.append(
                {
                    "title": "Leading Agent",
                    "value": f"{top_agent} ({vote_counts[top_agent]} votes)",
                }
            )

        return {
            "status": 200,
            "body": {
                "statusCode": 200,
                "type": "application/vnd.microsoft.card.adaptive",
                "value": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": "Debate Summary",
                            "weight": "Bolder",
                            "size": "Medium",
                        },
                        {"type": "FactSet", "facts": facts},
                    ],
                },
            },
        }

    async def _handle_view_details(
        self, debate_id: str, activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle view-details action - show detailed debate progress card."""
        debate_info = _active_debates.get(debate_id)

        if not debate_info:
            await self.bot.send_reply(
                activity,
                f"Debate {debate_id[:8]}... not found. It may have completed.",
            )
            return {
                "status": 200,
                "body": {"statusCode": 200, "type": "message", "value": "Debate not found"},
            }

        topic = debate_info.get("topic", "Unknown")
        current_round = debate_info.get("current_round", 1)
        total_rounds = debate_info.get("total_rounds", DEFAULT_ROUNDS)

        try:
            from datetime import datetime

            from aragora.server.handlers.bots.teams_cards import create_debate_progress_card

            card = create_debate_progress_card(
                debate_id=debate_id,
                topic=topic,
                current_round=current_round,
                total_rounds=total_rounds,
                current_phase=debate_info.get("phase", "deliberation"),
                timestamp=datetime.now(),
            )
            await self.bot.send_card(activity, card, f"Debate details: {topic[:80]}")
        except ImportError:
            started = debate_info.get("started_at", 0)
            elapsed = time.time() - started if started else 0
            await self.bot.send_reply(
                activity,
                f"**Debate Details**\n\n"
                f"**Topic:** {topic[:200]}\n"
                f"**Round:** {current_round}/{total_rounds}\n"
                f"**Elapsed:** {elapsed / 60:.1f} minutes\n"
                f"**ID:** {debate_id[:8]}...",
            )

        return {
            "status": 200,
            "body": {"statusCode": 200, "type": "message", "value": "Details sent"},
        }

    async def _handle_view_report(self, debate_id: str, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle view-report action - provide link to full report."""
        await self.bot.send_reply(
            activity,
            f"**Full Report**\n\n"
            f"View the complete debate report and audit trail:\n"
            f"[Open Report](https://aragora.ai/debate/{debate_id})",
        )
        return {
            "status": 200,
            "body": {"statusCode": 200, "type": "message", "value": "Report link sent"},
        }

    async def _handle_view_rankings(self, period: str, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle view-rankings action from leaderboard card."""
        event_processor = self.bot._get_event_processor()
        await event_processor._cmd_leaderboard(activity)
        return {
            "status": 200,
            "body": {"statusCode": 200, "type": "message", "value": "Rankings sent"},
        }

    async def _handle_watch_debate(
        self, debate_id: str, user_id: str, activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle watch action - subscribe user to live debate updates."""
        debate_info = _active_debates.get(debate_id)
        if not debate_info:
            return {
                "status": 200,
                "body": {
                    "statusCode": 200,
                    "type": "message",
                    "value": "Debate not found or already completed.",
                },
            }

        watchers: list[str] = debate_info.setdefault("watchers", [])
        if user_id not in watchers:
            watchers.append(user_id)

        logger.info("User %s watching debate %s", user_id, debate_id)

        return {
            "status": 200,
            "body": {
                "statusCode": 200,
                "type": "message",
                "value": "You will receive updates for this debate.",
            },
        }

    async def _handle_share_result(
        self, debate_id: str, activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle share action - re-post consensus card to channel."""
        debate_info = _active_debates.get(debate_id)
        if not debate_info:
            return {
                "status": 200,
                "body": {
                    "statusCode": 200,
                    "type": "message",
                    "value": "Debate result not available for sharing.",
                },
            }

        topic = debate_info.get("topic", "Unknown")
        vote_counts = get_debate_vote_counts(debate_id)

        card = build_consensus_card(
            debate_id=debate_id,
            topic=topic,
            consensus_reached=debate_info.get("consensus_reached", False),
            confidence=debate_info.get("confidence", 0.0),
            winner=debate_info.get("winner"),
            final_answer=debate_info.get("final_answer"),
            vote_counts=vote_counts,
        )
        await self.bot.send_card(activity, card, f"Debate result: {topic[:100]}")

        return {
            "status": 200,
            "body": {"statusCode": 200, "type": "message", "value": "Result shared"},
        }

    async def _handle_start_debate_prompt(self, activity: dict[str, Any]) -> dict[str, Any]:
        """Handle start-debate-prompt - return a task module for topic input."""
        return {
            "status": 200,
            "body": {
                "task": {
                    "type": "continue",
                    "value": {
                        "title": "Start a Debate",
                        "height": "medium",
                        "width": "medium",
                        "card": {
                            "contentType": "application/vnd.microsoft.card.adaptive",
                            "content": {
                                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                                "type": "AdaptiveCard",
                                "version": "1.4",
                                "body": [
                                    {
                                        "type": "TextBlock",
                                        "text": "What should the agents debate?",
                                        "weight": "Bolder",
                                    },
                                    {
                                        "type": "Input.Text",
                                        "id": "debate_topic",
                                        "placeholder": "Enter your debate topic...",
                                        "isMultiline": True,
                                        "maxLength": 1000,
                                    },
                                ],
                                "actions": [
                                    {
                                        "type": "Action.Submit",
                                        "title": "Start Debate",
                                        "data": {"action": "start_debate_from_task_module"},
                                    },
                                ],
                            },
                        },
                    },
                }
            },
        }

    # =========================================================================
    # Compose extension (messaging extension) handlers
    # =========================================================================

    async def _handle_compose_extension_submit(
        self, value: dict[str, Any], user_id: str, activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle compose extension submit action."""
        command_id = value.get("commandId", "")
        data = value.get("data", {})

        # RBAC: Check permission to create debates via compose extension
        if command_id == "startDebate":
            perm_error = self.bot._check_permission(activity, PERM_TEAMS_DEBATES_CREATE)
            if perm_error:
                return {
                    "status": 200,
                    "body": {
                        "composeExtension": {
                            "type": "message",
                            "text": "You don't have permission to start debates.",
                        }
                    },
                }

        if command_id == "startDebate":
            topic = data.get("topic", data.get("debate_topic", ""))
            if not topic:
                return {
                    "status": 200,
                    "body": {
                        "composeExtension": {
                            "type": "message",
                            "text": "Please provide a debate topic.",
                        }
                    },
                }

            conversation = activity.get("conversation", {})
            service_url = activity.get("serviceUrl", "")
            attachments = activity.get("attachments")
            if not isinstance(attachments, list):
                attachments = []

            debate_id = await _start_teams_debate(
                topic=topic,
                conversation_id=conversation.get("id", ""),
                user_id=user_id,
                service_url=service_url,
                attachments=attachments,
            )

            card = build_debate_card(
                debate_id=debate_id,
                topic=topic,
                agents=[AGENT_DISPLAY_NAMES.get(a, a) for a in DEFAULT_AGENT_LIST[:5]],
                current_round=1,
                total_rounds=DEFAULT_ROUNDS,
            )

            return {
                "status": 200,
                "body": {
                    "composeExtension": {
                        "type": "result",
                        "attachmentLayout": "list",
                        "attachments": [
                            {
                                "contentType": "application/vnd.microsoft.card.adaptive",
                                "content": card,
                                "preview": {
                                    "contentType": "application/vnd.microsoft.card.thumbnail",
                                    "content": {
                                        "title": f"Debate: {topic[:50]}",
                                        "text": f"ID: {debate_id[:8]}...",
                                    },
                                },
                            }
                        ],
                    }
                },
            }

        return {
            "status": 200,
            "body": {
                "composeExtension": {
                    "type": "message",
                    "text": "Command not recognized.",
                }
            },
        }

    async def _handle_compose_extension_query(
        self, value: dict[str, Any], user_id: str, activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle compose extension search query."""
        # RBAC: Check permission to search/read debates via compose extension
        perm_error = self.bot._check_permission(activity, PERM_TEAMS_CARDS_RESPOND)
        if perm_error:
            return {
                "status": 200,
                "body": {
                    "composeExtension": {
                        "type": "message",
                        "text": "You don't have permission to search debates.",
                    }
                },
            }

        query_text = ""
        parameters = value.get("parameters", [])
        for param in parameters:
            if param.get("name") == "query":
                query_text = param.get("value", "")
                break

        results: list[dict[str, Any]] = []
        for debate_id, info in _active_debates.items():
            topic = info.get("topic", "")
            if query_text.lower() in topic.lower() or not query_text:
                card = build_debate_card(
                    debate_id=debate_id,
                    topic=topic,
                    agents=[AGENT_DISPLAY_NAMES.get(a, a) for a in DEFAULT_AGENT_LIST[:5]],
                    current_round=info.get("current_round", 1),
                    total_rounds=info.get("total_rounds", DEFAULT_ROUNDS),
                )

                results.append(
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": card,
                        "preview": {
                            "contentType": "application/vnd.microsoft.card.thumbnail",
                            "content": {
                                "title": f"Debate: {topic[:50]}",
                                "text": f"ID: {debate_id[:8]}...",
                            },
                        },
                    }
                )

                if len(results) >= 10:
                    break

        return {
            "status": 200,
            "body": {
                "composeExtension": {
                    "type": "result",
                    "attachmentLayout": "list",
                    "attachments": results,
                }
            },
        }

    async def _handle_task_module_fetch(
        self, value: dict[str, Any], user_id: str, activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle task module fetch - return a dialog for user input."""
        # RBAC: Check permission to open task module dialogs
        perm_error = self.bot._check_permission(activity, PERM_TEAMS_CARDS_RESPOND)
        if perm_error:
            return {
                "status": 200,
                "body": {
                    "task": {
                        "type": "message",
                        "value": "You don't have permission to open this dialog.",
                    }
                },
            }

        data = value.get("data", {})
        command_id = value.get("commandId", data.get("commandId", ""))

        if command_id == "startDebate" or data.get("action") == "start_debate_prompt":
            return await self._handle_start_debate_prompt(activity)

        return {
            "status": 200,
            "body": {
                "task": {
                    "type": "continue",
                    "value": {
                        "title": "Aragora",
                        "card": {
                            "contentType": "application/vnd.microsoft.card.adaptive",
                            "content": {
                                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                                "type": "AdaptiveCard",
                                "version": "1.4",
                                "body": [
                                    {
                                        "type": "TextBlock",
                                        "text": "Use @Aragora commands to interact.",
                                        "wrap": True,
                                    },
                                ],
                            },
                        },
                    },
                }
            },
        }

    async def _handle_task_module_submit(
        self, value: dict[str, Any], user_id: str, activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle task module form submission."""
        data = value.get("data", {})
        action = data.get("action", "")

        if action == "start_debate_from_task_module":
            # RBAC: Check permission to create debates via task module
            perm_error = self.bot._check_permission(activity, PERM_TEAMS_DEBATES_CREATE)
            if perm_error:
                return {
                    "status": 200,
                    "body": {
                        "task": {
                            "type": "message",
                            "value": "You don't have permission to start debates.",
                        }
                    },
                }

            topic = data.get("debate_topic", "")
            if topic:
                conversation = activity.get("conversation", {})
                service_url = activity.get("serviceUrl", "")
                attachments = activity.get("attachments")
                if not isinstance(attachments, list):
                    attachments = []

                debate_id = await _start_teams_debate(
                    topic=topic,
                    conversation_id=conversation.get("id", ""),
                    user_id=user_id,
                    service_url=service_url,
                    attachments=attachments,
                )

                card = build_debate_card(
                    debate_id=debate_id,
                    topic=topic,
                    agents=[AGENT_DISPLAY_NAMES.get(a, a) for a in DEFAULT_AGENT_LIST[:5]],
                    current_round=1,
                    total_rounds=DEFAULT_ROUNDS,
                    include_vote_buttons=False,
                )
                await self.bot.send_card(activity, card, f"Starting debate: {topic[:100]}")

                audit_data(
                    user_id=f"teams:{user_id}",
                    resource_type="debate",
                    resource_id=debate_id,
                    action="create",
                    platform="teams",
                    task_preview=topic[:100],
                )

        return {"status": 200, "body": {"task": {"type": "message", "value": "Done"}}}

    async def _handle_link_unfurling(
        self, value: dict[str, Any], activity: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle link unfurling for aragora.ai URLs."""
        # RBAC: Check permission to view debate data via link unfurling
        perm_error = self.bot._check_permission(activity, PERM_TEAMS_CARDS_RESPOND)
        if perm_error:
            # Return empty results silently for link unfurling permission denial
            return {
                "status": 200,
                "body": {
                    "composeExtension": {
                        "type": "result",
                        "attachmentLayout": "list",
                        "attachments": [],
                    }
                },
            }

        url = value.get("url", "")

        debate_id_match = re.search(r"/debate/([a-f0-9-]+)", url)
        if debate_id_match:
            debate_id = debate_id_match.group(1)
            debate_info = _active_debates.get(debate_id)

            if debate_info:
                topic = debate_info.get("topic", "Unknown topic")
                card = build_debate_card(
                    debate_id=debate_id,
                    topic=topic,
                    agents=[AGENT_DISPLAY_NAMES.get(a, a) for a in DEFAULT_AGENT_LIST[:5]],
                    current_round=debate_info.get("current_round", 1),
                    total_rounds=debate_info.get("total_rounds", DEFAULT_ROUNDS),
                )

                return {
                    "status": 200,
                    "body": {
                        "composeExtension": {
                            "type": "result",
                            "attachmentLayout": "list",
                            "attachments": [
                                {
                                    "contentType": "application/vnd.microsoft.card.adaptive",
                                    "content": card,
                                    "preview": {
                                        "contentType": "application/vnd.microsoft.card.thumbnail",
                                        "content": {
                                            "title": f"Debate: {topic[:50]}",
                                            "text": f"ID: {debate_id[:8]}...",
                                        },
                                    },
                                }
                            ],
                        }
                    },
                }

        return {
            "status": 200,
            "body": {
                "composeExtension": {
                    "type": "result",
                    "attachmentLayout": "list",
                    "attachments": [],
                }
            },
        }


__all__ = [
    "TeamsCardActions",
    "AGENT_DISPLAY_NAMES",
]
