"""
Telegram callback query and inline query handling.

Handles button presses (votes, view details) and inline bot queries.
"""

from __future__ import annotations

import logging
from typing import Any

from ...base import HandlerResult, json_response
from ..chat_events import emit_message_received, emit_vote_received
from ..telemetry import record_message, record_vote
from . import _common

logger = logging.getLogger(__name__)


def _tg():
    """Lazy import of the telegram package for patchable attribute access."""
    from aragora.server.handlers.social import telegram as telegram_module

    return telegram_module


class TelegramCallbacksMixin:
    """Mixin providing callback query and inline query handling for Telegram."""

    def _handle_message(self, message: dict[str, Any]) -> HandlerResult:
        """Handle incoming text message.

        RBAC Permission Required: telegram:messages:send (for sending responses)
        """
        chat_id = message.get("chat", {}).get("id")
        text = message.get("text", "").strip()
        user = message.get("from", {})
        user_id = user.get("id")
        username = user.get("username", "unknown")

        if not chat_id or not text:
            return json_response({"ok": True})

        logger.info("Telegram message from %s (%s): %s...", username, user_id, text[:50])

        # Parse bot commands
        if text.startswith("/"):
            record_message("telegram", "command")
            return self._handle_command(chat_id, user_id, username, text)

        record_message("telegram", "text")

        # RBAC: Check permission to send messages (for response)
        if not self._check_telegram_user_permission(
            user_id, username, chat_id, _common.PERM_TELEGRAM_MESSAGES_SEND
        ):
            return self._deny_telegram_permission(
                chat_id, _common.PERM_TELEGRAM_MESSAGES_SEND, "send messages"
            )

        # Emit webhook event for message received
        emit_message_received(
            platform="telegram",
            chat_id=str(chat_id),
            user_id=str(user_id),
            username=username,
            message_text=text,
            message_type="text",
        )

        # Handle regular messages as questions/topics
        if len(text) > 10:
            response = (
                f'I received: "{text[:50]}..."\n\n'
                "To start a debate on this topic, use:\n"
                f"/debate {text[:100]}"
            )
        else:
            response = "Send /help to see available commands."

        _tg().create_tracked_task(
            self._send_message_async(chat_id, response),
            name=f"telegram-reply-{chat_id}",
        )

        return json_response({"ok": True})

    def _handle_callback_query(self, callback: dict[str, Any]) -> HandlerResult:
        """Handle inline keyboard button clicks.

        RBAC Permission Required: telegram:callbacks:handle
        """
        callback_id = callback.get("id")
        data = callback.get("data", "")
        user = callback.get("from", {})
        user_id = user.get("id")
        username = user.get("username", "unknown")
        message = callback.get("message", {})
        chat_id = message.get("chat", {}).get("id")

        logger.info("Telegram callback from %s: %s", username, data)

        # RBAC: Check base permission to handle callbacks
        if not self._check_telegram_user_permission(
            user_id, username, chat_id, _common.PERM_TELEGRAM_CALLBACKS_HANDLE
        ):
            _tg().create_tracked_task(
                self._answer_callback_async(
                    callback_id,
                    "Permission denied: You cannot perform this action.",
                    show_alert=True,
                ),
                name=f"telegram-callback-denied-{callback_id}",
            )
            return json_response({"ok": True})

        # Parse callback data
        parts = data.split(":")
        action = parts[0] if parts else ""

        if action == "vote" and len(parts) >= 3:
            debate_id = parts[1]
            vote_option = parts[2]
            # RBAC: Check vote recording permission
            if not self._check_telegram_user_permission(
                user_id, username, chat_id, _common.PERM_TELEGRAM_VOTES_RECORD
            ):
                _tg().create_tracked_task(
                    self._answer_callback_async(
                        callback_id,
                        "Permission denied: You don't have permission to vote.",
                        show_alert=True,
                    ),
                    name=f"telegram-vote-denied-{callback_id}",
                )
                return json_response({"ok": True})
            return self._handle_vote(
                callback_id, chat_id, user_id, username, debate_id, vote_option
            )
        elif action == "details" and len(parts) >= 2:
            debate_id = parts[1]
            # RBAC: Check debate read permission
            if not self._check_telegram_user_permission(
                user_id, username, chat_id, _common.PERM_TELEGRAM_DEBATES_READ
            ):
                _tg().create_tracked_task(
                    self._answer_callback_async(
                        callback_id,
                        "Permission denied: You don't have permission to view debate details.",
                        show_alert=True,
                    ),
                    name=f"telegram-details-denied-{callback_id}",
                )
                return json_response({"ok": True})
            return self._handle_view_details(callback_id, chat_id, user_id, username, debate_id)

        # Answer callback to remove loading state
        _tg().create_tracked_task(
            self._answer_callback_async(callback_id, "Action received"),
            name=f"telegram-callback-ack-{callback_id}",
        )

        return json_response({"ok": True})

    def _handle_vote(
        self,
        callback_id: str,
        chat_id: int,
        user_id: int,
        username: str,
        debate_id: str,
        vote_option: str,
    ) -> HandlerResult:
        """Handle vote callback."""
        logger.info(
            "Vote received: %s -> %s from %s (@%s)", debate_id, vote_option, user_id, username
        )

        # Emit webhook event for vote received
        emit_vote_received(
            platform="telegram",
            chat_id=str(chat_id),
            user_id=str(user_id),
            username=username,
            debate_id=debate_id,
            vote=vote_option,
        )

        # Record vote metrics
        record_vote("telegram", vote_option)

        # Record vote in storage
        try:
            from aragora.server.storage import get_debates_db

            db = get_debates_db()
            if db and hasattr(db, "record_vote"):
                db.record_vote(
                    debate_id=debate_id,
                    voter_id=f"telegram:{user_id}",
                    vote=vote_option,
                    source="telegram",
                )
        except (ImportError, KeyError, OSError, RuntimeError, ValueError) as e:
            logger.warning("Failed to record vote: %s", e)

        emoji = "+" if vote_option == "agree" else "-"
        _tg().create_tracked_task(
            self._answer_callback_async(
                callback_id,
                f"{emoji} Your vote for '{vote_option}' has been recorded!",
                show_alert=True,
            ),
            name=f"telegram-vote-ack-{callback_id}",
        )

        return json_response({"ok": True})

    def _handle_view_details(
        self,
        callback_id: str,
        chat_id: int,
        user_id: int,
        username: str,
        debate_id: str,
    ) -> HandlerResult:
        """Handle view details callback."""
        logger.info(
            "View details requested by %s (@%s) for debate %s", user_id, username, debate_id
        )

        debate_data = None
        try:
            from aragora.server.storage import get_debates_db

            db = get_debates_db()
            if db:
                debate_data = db.get(debate_id)
        except (ImportError, KeyError, OSError, RuntimeError, ValueError) as e:
            logger.warning("Failed to fetch debate: %s", e)

        if not debate_data:
            _tg().create_tracked_task(
                self._answer_callback_async(
                    callback_id,
                    f"Debate {debate_id} not found",
                    show_alert=True,
                ),
                name=f"telegram-details-notfound-{callback_id}",
            )
            return json_response({"ok": True})

        task = debate_data.get("task", "Unknown")
        final_answer = debate_data.get("final_answer", "No conclusion")
        consensus = debate_data.get("consensus_reached", False)
        confidence = debate_data.get("confidence", 0)
        rounds_used = debate_data.get("rounds_used", 0)
        agents = debate_data.get("agents", [])

        agent_list = ", ".join(agents[:5]) if agents else "Unknown"
        if len(agents) > 5:
            agent_list += f" (+{len(agents) - 5} more)"

        response = (
            f"*Debate Details*\n\n"
            f"*Topic:*\n{task[:200]}{'...' if len(task) > 200 else ''}\n\n"
            f"*ID:* `{debate_id}`\n"
            f"*Consensus:* {'Yes' if consensus else 'No'}\n"
            f"*Confidence:* {confidence:.1%}\n"
            f"*Rounds:* {rounds_used}\n"
            f"*Agents:* {agent_list}\n\n"
            f"*Conclusion:*\n{final_answer[:500] if final_answer else 'No conclusion'}{'...' if final_answer and len(final_answer) > 500 else ''}"
        )

        # Answer callback and send details as new message
        _tg().create_tracked_task(
            self._answer_callback_async(callback_id, "Loading details..."),
            name=f"telegram-details-ack-{callback_id}",
        )

        _tg().create_tracked_task(
            self._send_message_async(chat_id, response, parse_mode="Markdown"),
            name=f"telegram-details-{chat_id}",
        )

        return json_response({"ok": True})

    def _handle_inline_query(self, query: dict[str, Any]) -> HandlerResult:
        """Handle inline queries (@bot query)."""
        query_id = query.get("id")
        query_text = query.get("query", "").strip()
        user = query.get("from", {})
        user_id = user.get("id")
        username = user.get("username", "unknown")

        # RBAC: Check basic read permission for inline queries
        if not self._check_telegram_user_permission(
            user_id, username, 0, _common.PERM_TELEGRAM_READ
        ):
            _tg().create_tracked_task(
                self._answer_inline_query_async(query_id, []),
                name=f"telegram-inline-denied-{query_id}",
            )
            return json_response({"ok": True})

        if not query_text or len(query_text) < 5:
            results: list[dict[str, Any]] = []
        else:
            results = [
                {
                    "type": "article",
                    "id": f"debate_{hash(query_text) % 10000}",
                    "title": f"Start debate: {query_text[:50]}...",
                    "description": "Click to start a multi-agent debate on this topic",
                    "input_message_content": {
                        "message_text": f"/debate {query_text}",
                    },
                },
                {
                    "type": "article",
                    "id": f"gauntlet_{hash(query_text) % 10000}",
                    "title": f"Stress-test: {query_text[:50]}...",
                    "description": "Click to run adversarial validation on this statement",
                    "input_message_content": {
                        "message_text": f"/gauntlet {query_text}",
                    },
                },
            ]

        _tg().create_tracked_task(
            self._answer_inline_query_async(query_id, results),
            name=f"telegram-inline-{query_id}",
        )

        return json_response({"ok": True})
