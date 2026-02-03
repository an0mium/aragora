"""
Telegram bot command handling.

Handles /start, /help, /status, /agents, /debate, /gauntlet, /search, /recent, /receipt commands
and their associated async execution flows.
"""

from __future__ import annotations

import logging

from ...base import HandlerResult, json_response
from ..chat_events import (
    emit_command_received,
    emit_debate_completed,
    emit_debate_started,
    emit_gauntlet_completed,
    emit_gauntlet_started,
)
from ..telemetry import (
    record_command,
    record_debate_completed,
    record_debate_failed,
    record_debate_started,
    record_gauntlet_completed,
    record_gauntlet_failed,
    record_gauntlet_started,
)
from . import _common

logger = logging.getLogger(__name__)


def _tg():
    """Lazy import of the telegram package for patchable attribute access."""
    from aragora.server.handlers.social import telegram as telegram_module

    return telegram_module


class TelegramCommandsMixin:
    """Mixin providing command handling for the Telegram bot."""

    def _handle_command(
        self,
        chat_id: int,
        user_id: int,
        username: str,
        text: str,
    ) -> HandlerResult:
        """Handle bot commands.

        RBAC Permission Required: telegram:commands:execute (for command execution)
        """
        # Parse command and arguments
        parts = text.split(maxsplit=1)
        command = parts[0].lower()
        # Remove @botname suffix if present
        if "@" in command:
            command = command.split("@")[0]
        args = parts[1] if len(parts) > 1 else ""

        # Record command metric (strip leading /)
        cmd_name = command[1:] if command.startswith("/") else command

        # RBAC: Check base permission to execute commands
        if not self._check_telegram_user_permission(
            user_id, username, chat_id, _common.PERM_TELEGRAM_COMMANDS_EXECUTE
        ):
            return self._deny_telegram_permission(
                chat_id, _common.PERM_TELEGRAM_COMMANDS_EXECUTE, "execute bot commands"
            )

        # Emit webhook event for command received
        emit_command_received(
            platform="telegram",
            chat_id=str(chat_id),
            user_id=str(user_id),
            username=username,
            command=cmd_name,
            args=args,
        )

        if command == "/start":
            record_command("telegram", "start")
            if not self._check_telegram_user_permission(
                user_id, username, chat_id, _common.PERM_TELEGRAM_READ
            ):
                return self._deny_telegram_permission(
                    chat_id, _common.PERM_TELEGRAM_READ, "view bot information"
                )
            response = self._command_start(username)
        elif command == "/help":
            record_command("telegram", "help")
            if not self._check_telegram_user_permission(
                user_id, username, chat_id, _common.PERM_TELEGRAM_READ
            ):
                return self._deny_telegram_permission(
                    chat_id, _common.PERM_TELEGRAM_READ, "view help information"
                )
            response = self._command_help()
        elif command == "/status":
            record_command("telegram", "status")
            if not self._check_telegram_user_permission(
                user_id, username, chat_id, _common.PERM_TELEGRAM_READ
            ):
                return self._deny_telegram_permission(
                    chat_id, _common.PERM_TELEGRAM_READ, "view system status"
                )
            response = self._command_status()
        elif command == "/agents":
            record_command("telegram", "agents")
            if not self._check_telegram_user_permission(
                user_id, username, chat_id, _common.PERM_TELEGRAM_READ
            ):
                return self._deny_telegram_permission(
                    chat_id, _common.PERM_TELEGRAM_READ, "view agent list"
                )
            response = self._command_agents()
        elif command == "/debate":
            record_command("telegram", "debate")
            if not self._check_telegram_user_permission(
                user_id, username, chat_id, _common.PERM_TELEGRAM_DEBATES_CREATE
            ):
                return self._deny_telegram_permission(
                    chat_id, _common.PERM_TELEGRAM_DEBATES_CREATE, "create debates"
                )
            return self._command_debate(chat_id, user_id, username, args)
        elif command == "/gauntlet":
            record_command("telegram", "gauntlet")
            if not self._check_telegram_user_permission(
                user_id, username, chat_id, _common.PERM_TELEGRAM_GAUNTLET_RUN
            ):
                return self._deny_telegram_permission(
                    chat_id, _common.PERM_TELEGRAM_GAUNTLET_RUN, "run gauntlet stress-tests"
                )
            return self._command_gauntlet(chat_id, user_id, username, args)
        elif command == "/search":
            record_command("telegram", "search")
            if not self._check_telegram_user_permission(
                user_id, username, chat_id, _common.PERM_TELEGRAM_READ
            ):
                return self._deny_telegram_permission(
                    chat_id, _common.PERM_TELEGRAM_READ, "search debates"
                )
            response = self._command_search(args)
        elif command == "/recent":
            record_command("telegram", "recent")
            if not self._check_telegram_user_permission(
                user_id, username, chat_id, _common.PERM_TELEGRAM_READ
            ):
                return self._deny_telegram_permission(
                    chat_id, _common.PERM_TELEGRAM_READ, "view recent debates"
                )
            response = self._command_recent()
        elif command == "/receipt":
            record_command("telegram", "receipt")
            if not self._check_telegram_user_permission(
                user_id, username, chat_id, _common.PERM_TELEGRAM_READ
            ):
                return self._deny_telegram_permission(
                    chat_id, _common.PERM_TELEGRAM_READ, "view receipts"
                )
            response = self._command_receipt(args)
        else:
            record_command("telegram", "unknown")
            response = f"Unknown command: {command}\nSend /help for available commands."

        _tg().create_tracked_task(
            self._send_message_async(chat_id, response),
            name=f"telegram-cmd-{command}-{chat_id}",
        )

        return json_response({"ok": True})

    def _command_start(self, username: str) -> str:
        """Handle /start command."""
        return (
            f"Welcome to Aragora, {username}!\n\n"
            "I can run multi-agent debates and adversarial validations.\n\n"
            "Commands:\n"
            "/debate <topic> - Start a debate\n"
            "/gauntlet <statement> - Stress-test a statement\n"
            "/search <query> - Search past debates\n"
            "/recent - Show recent debates\n"
            "/receipt <id> - View decision receipt\n"
            "/status - System status\n"
            "/agents - List agents\n"
            "/help - Show this help"
        )

    def _command_help(self) -> str:
        """Handle /help command."""
        return (
            "*Aragora Bot Commands*\n\n"
            "/start - Welcome message\n"
            "/debate <topic> - Start a multi-agent debate on a topic\n"
            "/gauntlet <statement> - Run adversarial stress-test\n"
            "/search <query> - Search past debates\n"
            "/recent - Show recent debates\n"
            "/receipt <id> - View decision receipt\n"
            "/status - Get system status\n"
            "/agents - List available agents\n"
            "/help - Show this help\n\n"
            "*Examples:*\n"
            "/debate Should AI be regulated?\n"
            "/gauntlet We should migrate to microservices\n"
            "/search machine learning\n"
            "/receipt abc123"
        )

    def _command_status(self) -> str:
        """Handle /status command."""
        try:
            from aragora.ranking.elo import EloSystem

            store = EloSystem()
            agents = store.get_all_ratings()
            return f"*Aragora Status*\n\nStatus: Online\nAgents: {len(agents)} registered"
        except Exception as e:
            logger.warning("Failed to get status: %s", e)
            return "*Aragora Status*\n\nStatus: Online"

    def _command_agents(self) -> str:
        """Handle /agents command."""
        try:
            from aragora.ranking.elo import EloSystem

            store = EloSystem()
            agents = store.get_all_ratings()

            if not agents:
                return "No agents registered yet."

            agents = sorted(agents, key=lambda a: getattr(a, "elo", 1500), reverse=True)

            lines = ["*Top Agents by ELO:*\n"]
            for i, agent in enumerate(agents[:10]):
                name = getattr(agent, "name", "Unknown")
                elo = getattr(agent, "elo", 1500)
                wins = getattr(agent, "wins", 0)
                medal = ["1.", "2.", "3."][i] if i < 3 else f"{i + 1}."
                lines.append(f"{medal} *{name}* - ELO: {elo:.0f} | Wins: {wins}")

            return "\n".join(lines)
        except Exception as e:
            logger.warning("Failed to list agents: %s", e)
            return "Could not fetch agent list."

    def _command_debate(
        self,
        chat_id: int,
        user_id: int,
        username: str,
        args: str,
    ) -> HandlerResult:
        """Handle /debate command."""
        if not args:
            response = "Please provide a topic.\n\nExample: /debate Should AI be regulated?"
            _tg().create_tracked_task(
                self._send_message_async(chat_id, response),
                name=f"telegram-debate-help-{chat_id}",
            )
            return json_response({"ok": True})

        topic = args.strip().strip("\"'")

        if len(topic) < 10:
            response = "Topic is too short. Please provide more detail."
            _tg().create_tracked_task(
                self._send_message_async(chat_id, response),
                name=f"telegram-debate-short-{chat_id}",
            )
            return json_response({"ok": True})

        if len(topic) > 500:
            response = "Topic is too long. Please limit to 500 characters."
            _tg().create_tracked_task(
                self._send_message_async(chat_id, response),
                name=f"telegram-debate-long-{chat_id}",
            )
            return json_response({"ok": True})

        # Send initial acknowledgment
        _tg().create_tracked_task(
            self._send_message_async(
                chat_id,
                f"*Starting debate on:*\n_{topic}_\n\nRequested by @{username}\nProcessing... (this may take a few minutes)",
                parse_mode="Markdown",
            ),
            name=f"telegram-debate-ack-{chat_id}",
        )

        # Queue the debate asynchronously
        _tg().create_tracked_task(
            self._run_debate_async(chat_id, user_id, username, topic),
            name=f"telegram-debate-{topic[:30]}",
        )

        return json_response({"ok": True})

    async def _run_debate_async(
        self,
        chat_id: int,
        user_id: int,
        username: str,
        topic: str,
        message_id: int | None = None,
    ) -> None:
        """Run debate asynchronously and send result to chat."""
        from aragora.config import DEFAULT_CONSENSUS, DEFAULT_ROUNDS

        record_debate_started("telegram")
        try:
            from aragora import Arena, DebateProtocol, Environment
            from aragora.agents import get_agents_by_names

            # Register debate origin for tracking and potential async routing
            try:
                from aragora.server.debate_origin import register_debate_origin
                import uuid

                debate_id = f"tg-{chat_id}-{uuid.uuid4().hex[:8]}"
                register_debate_origin(
                    debate_id=debate_id,
                    platform="telegram",
                    channel_id=str(chat_id),
                    user_id=str(user_id),
                    message_id=str(message_id) if message_id else None,
                    metadata={"username": username, "topic": topic},
                )
                logger.debug("Registered debate origin: %s", debate_id)
            except ImportError:
                debate_id = None
                logger.debug("Debate origin tracking not available")

            # Emit webhook event for debate started
            emit_debate_started(
                platform="telegram",
                chat_id=str(chat_id),
                user_id=str(user_id),
                username=username,
                topic=topic,
                debate_id=debate_id,
            )

            env = Environment(task=f"Debate: {topic}")

            # Resolve agent binding for this chat
            agent_names = ["anthropic-api", "openai-api"]  # Default agents
            protocol_config = {
                "rounds": 3,
                "consensus": "majority",
                "convergence_detection": False,
                "early_stopping": False,
            }
            try:
                from aragora.server.bindings import get_binding_router, BindingType

                router = get_binding_router()
                resolution = router.resolve(
                    provider="telegram",
                    account_id="default",
                    peer_id=f"chat:{chat_id}",
                    user_id=str(user_id),
                )

                if resolution.matched and resolution.binding:
                    logger.debug(
                        "Binding resolved: %s type=%s reason=%s",
                        resolution.agent_binding,
                        resolution.binding_type,
                        resolution.match_reason,
                    )

                    if resolution.binding_type == BindingType.SPECIFIC_AGENT:
                        agent_names = [resolution.agent_binding]
                    elif resolution.binding_type == BindingType.AGENT_POOL:
                        pool_name = resolution.agent_binding
                        if pool_name == "full-team":
                            agent_names = ["anthropic-api", "openai-api", "mistral-api"]
                        elif pool_name == "fast":
                            agent_names = ["openai-api"]

                    if resolution.config_overrides:
                        protocol_config.update(resolution.config_overrides)

            except ImportError:
                logger.debug("Binding router not available, using default agents")
            except Exception as e:
                logger.debug("Binding resolution failed: %s, using default agents", e)

            agents = get_agents_by_names(agent_names)
            protocol = DebateProtocol(
                rounds=protocol_config.get("rounds", DEFAULT_ROUNDS),
                consensus=protocol_config.get("consensus", DEFAULT_CONSENSUS),
                convergence_detection=protocol_config.get("convergence_detection", False),
                early_stopping=protocol_config.get("early_stopping", False),
            )

            if not agents:
                await self._send_message_async(
                    chat_id,
                    "Failed to start debate: No agents available",
                )
                record_debate_failed("telegram")
                return

            arena = Arena.from_env(env, agents, protocol)
            result = await arena.run()

            consensus_emoji = "+" if result.consensus_reached else "-"
            confidence_pct = f"{result.confidence:.1%}"

            response = (
                f"*Debate Complete!* {consensus_emoji}\n\n"
                f"*Topic:* {topic[:100]}{'...' if len(topic) > 100 else ''}\n\n"
                f"*Consensus:* {'Yes' if result.consensus_reached else 'No'}\n"
                f"*Confidence:* {confidence_pct}\n"
                f"*Rounds:* {result.rounds_used}\n"
                f"*Agents:* {len(agents)}\n\n"
                f"*Conclusion:*\n{result.final_answer[:500] if result.final_answer else 'No conclusion reached'}{'...' if result.final_answer and len(result.final_answer) > 500 else ''}\n\n"
                f"_Debate ID: {result.id}_\n"
                f"_Requested by @{username}_"
            )

            # Send with inline keyboard for voting
            keyboard = {
                "inline_keyboard": [
                    [
                        {"text": "Agree", "callback_data": f"vote:{result.id}:agree"},
                        {"text": "Disagree", "callback_data": f"vote:{result.id}:disagree"},
                    ],
                    [
                        {"text": "View Details", "callback_data": f"details:{result.id}"},
                    ],
                ]
            }

            await self._send_message_async(
                chat_id,
                response,
                parse_mode="Markdown",
                reply_markup=keyboard,
            )

            # Mark debate result as sent for origin tracking
            if debate_id:
                try:
                    from aragora.server.debate_origin import mark_result_sent

                    mark_result_sent(debate_id)
                except ImportError:
                    pass

            # Send voice summary if TTS is enabled
            if _tg().TTS_VOICE_ENABLED:
                await self._send_voice_summary(
                    chat_id,
                    topic,
                    result.final_answer,
                    result.consensus_reached,
                    result.confidence,
                    result.rounds_used,
                )

            # Emit webhook event for debate completed
            emit_debate_completed(
                platform="telegram",
                chat_id=str(chat_id),
                debate_id=result.id,
                topic=topic,
                consensus_reached=result.consensus_reached,
                confidence=result.confidence,
                rounds_used=result.rounds_used,
                final_answer=result.final_answer,
            )

            # Record successful debate completion
            record_debate_completed("telegram", result.consensus_reached)

        except Exception as e:
            logger.error("Telegram debate failed: %s", e, exc_info=True)
            record_debate_failed("telegram")
            await self._send_message_async(
                chat_id,
                f"Debate failed: {str(e)[:100]}",
            )

    def _command_gauntlet(
        self,
        chat_id: int,
        user_id: int,
        username: str,
        args: str,
    ) -> HandlerResult:
        """Handle /gauntlet command."""
        if not args:
            response = "Please provide a statement to stress-test.\n\nExample: /gauntlet We should migrate to microservices"
            _tg().create_tracked_task(
                self._send_message_async(chat_id, response),
                name=f"telegram-gauntlet-help-{chat_id}",
            )
            return json_response({"ok": True})

        statement = args.strip().strip("\"'")

        if len(statement) < 10:
            response = "Statement is too short. Please provide more detail."
            _tg().create_tracked_task(
                self._send_message_async(chat_id, response),
                name=f"telegram-gauntlet-short-{chat_id}",
            )
            return json_response({"ok": True})

        if len(statement) > 1000:
            response = "Statement is too long. Please limit to 1000 characters."
            _tg().create_tracked_task(
                self._send_message_async(chat_id, response),
                name=f"telegram-gauntlet-long-{chat_id}",
            )
            return json_response({"ok": True})

        # Send initial acknowledgment
        _tg().create_tracked_task(
            self._send_message_async(
                chat_id,
                f"*Running Gauntlet stress-test on:*\n_{statement[:200]}{'...' if len(statement) > 200 else ''}_\n\nRequested by @{username}\nRunning adversarial validation...",
                parse_mode="Markdown",
            ),
            name=f"telegram-gauntlet-ack-{chat_id}",
        )

        # Queue the gauntlet asynchronously
        _tg().create_tracked_task(
            self._run_gauntlet_async(chat_id, user_id, username, statement),
            name=f"telegram-gauntlet-{statement[:30]}",
        )

        return json_response({"ok": True})

    async def _run_gauntlet_async(
        self,
        chat_id: int,
        user_id: int,
        username: str,
        statement: str,
    ) -> None:
        """Run gauntlet asynchronously and send result to chat."""
        from aragora.server.http_client_pool import get_http_pool

        record_gauntlet_started("telegram")

        # Emit webhook event for gauntlet started
        emit_gauntlet_started(
            platform="telegram",
            chat_id=str(chat_id),
            user_id=str(user_id),
            username=username,
            statement=statement,
        )

        try:
            pool = get_http_pool()
            async with pool.get_session("telegram_gauntlet") as client:
                resp = await client.post(
                    "http://localhost:8080/api/gauntlet/run",
                    json={
                        "statement": statement,
                        "intensity": "medium",
                        "metadata": {
                            "source": "telegram",
                            "chat_id": chat_id,
                            "user_id": user_id,
                        },
                    },
                    timeout=120,
                )
                data = resp.json()

                if resp.status_code != 200:
                    await self._send_message_async(
                        chat_id,
                        f"Gauntlet failed: {data.get('error', 'Unknown error')}",
                    )
                    record_gauntlet_failed("telegram")
                    return

                run_id = data.get("run_id", "unknown")
                score = data.get("score", 0)
                passed = data.get("passed", False)
                vulnerabilities = data.get("vulnerabilities", [])

                status_emoji = "PASSED" if passed else "FAILED"
                score_pct = f"{score:.1%}"

                response = (
                    f"*Gauntlet Results* {status_emoji}\n\n"
                    f"*Statement:*\n_{statement[:200]}{'...' if len(statement) > 200 else ''}_\n\n"
                    f"*Score:* {score_pct}\n"
                    f"*Status:* {'Passed' if passed else 'Failed'}\n"
                    f"*Vulnerabilities:* {len(vulnerabilities)}\n"
                )

                if vulnerabilities:
                    response += "\n*Issues Found:*\n"
                    for v in vulnerabilities[:5]:
                        desc = v.get("description", "Unknown issue")[:100]
                        response += f"- {desc}\n"
                    if len(vulnerabilities) > 5:
                        response += f"_...and {len(vulnerabilities) - 5} more_\n"

                response += f"\n_Run ID: {run_id}_\n_Requested by @{username}_"

                await self._send_message_async(
                    chat_id,
                    response,
                    parse_mode="Markdown",
                )

                # Emit webhook event for gauntlet completed
                emit_gauntlet_completed(
                    platform="telegram",
                    chat_id=str(chat_id),
                    gauntlet_id=run_id,
                    statement=statement,
                    verdict="passed" if passed else "failed",
                    confidence=score,
                    challenges_passed=len(
                        [v for v in vulnerabilities if not v.get("critical", False)]
                    ),
                    challenges_total=len(vulnerabilities) + 1,
                )

                # Record successful gauntlet completion
                record_gauntlet_completed("telegram", passed)

        except Exception as e:
            logger.error("Telegram gauntlet failed: %s", e, exc_info=True)
            record_gauntlet_failed("telegram")
            await self._send_message_async(
                chat_id,
                f"Gauntlet failed: {str(e)[:100]}",
            )

    def _command_search(self, query: str) -> str:
        """Handle /search command - search past debates."""
        if not query or len(query.strip()) < 3:
            return (
                "Please provide a search query (at least 3 characters).\n\n"
                "Example: /search machine learning"
            )

        query = query.strip()

        try:
            from aragora.storage import get_storage

            db = get_storage()
            if db and hasattr(db, "search"):
                search_results, total = db.search(query, limit=5)
                results = list(search_results)
            else:
                # Fallback: manual search through recent debates
                if db and hasattr(db, "get_recent_debates"):
                    recent = db.get_recent_debates(limit=50)
                    query_lower = query.lower()
                    results = [
                        d
                        for d in recent
                        if query_lower in d.get("topic", "").lower()
                        or query_lower in d.get("conclusion", "").lower()
                    ][:5]
                    total = len(results)
                else:
                    return "Search service is not available."

            if not results:
                return f"No debates found matching: {query}"

            lines = [f"*Search Results for:* _{query}_\n"]
            for i, debate in enumerate(results[:5], 1):
                topic = debate.get("topic", "Unknown topic")[:60]
                debate_id = debate.get("id", "N/A")
                consensus = "Yes" if debate.get("consensus_reached") else "No"
                lines.append(f"{i}. *{topic}*{'...' if len(debate.get('topic', '')) > 60 else ''}")
                lines.append(f"   ID: `{debate_id}` | Consensus: {consensus}")

            if total > 5:
                lines.append(f"\n_Showing 5 of {total} results_")

            return "\n".join(lines)

        except ImportError:
            logger.warning("Storage not available for search")
            return "Search service temporarily unavailable."
        except Exception as e:
            logger.exception(f"Unexpected search error: {e}")
            return f"Search failed: {str(e)[:100]}"

    def _command_recent(self) -> str:
        """Handle /recent command - show recent debates."""
        try:
            from aragora.storage import get_storage

            db = get_storage()
            if not db or not hasattr(db, "get_recent_debates"):
                return "Recent debates service is not available."

            debates = db.get_recent_debates(limit=5)

            if not debates:
                return "No recent debates found.\n\nStart one with /debate <topic>"

            lines = ["*Recent Debates*\n"]
            for i, debate in enumerate(debates[:5], 1):
                topic = debate.get("topic", "Unknown topic")[:50]
                debate_id = debate.get("id", "N/A")
                consensus = "Yes" if debate.get("consensus_reached") else "No"
                confidence = debate.get("confidence", 0)

                lines.append(f"{i}. *{topic}*{'...' if len(debate.get('topic', '')) > 50 else ''}")
                lines.append(f"   ID: `{debate_id}`")
                lines.append(f"   Consensus: {consensus} | Confidence: {confidence:.0%}")

            lines.append("\n_Use /receipt <id> to view decision receipt_")

            return "\n".join(lines)

        except ImportError:
            logger.warning("Storage not available for recent debates")
            return "Recent debates service temporarily unavailable."
        except Exception as e:
            logger.exception(f"Unexpected recent debates error: {e}")
            return f"Failed to get recent debates: {str(e)[:100]}"

    def _command_receipt(self, args: str) -> str:
        """Handle /receipt command - view decision receipt."""
        if not args or not args.strip():
            return (
                "Please provide a debate ID.\n\n"
                "Example: /receipt abc123\n\n"
                "Use /recent to see recent debate IDs."
            )

        debate_id = args.strip()

        try:
            # Try to get receipt from receipt store
            try:
                from aragora.storage.receipt_store import get_receipt_store

                receipt_store = get_receipt_store()
                receipt_data = receipt_store.get(debate_id)

                if receipt_data:
                    return self._format_receipt(receipt_data)
            except ImportError:
                pass

            # Fallback: generate receipt from debate result
            from aragora.storage import get_storage

            db = get_storage()
            if not db:
                return "Receipt service is not available."

            debate = db.get_debate(debate_id)
            if not debate:
                return f"No debate found with ID: {debate_id}"

            # Generate receipt from debate data
            try:
                from aragora.gauntlet.receipt import DecisionReceipt

                receipt = DecisionReceipt.from_dict(debate)
                return self._format_receipt(receipt.to_dict())
            except ImportError:
                # Manual formatting if receipt module unavailable
                return self._format_debate_as_receipt(debate)

        except Exception as e:
            logger.exception(f"Unexpected receipt error: {e}")
            return f"Failed to get receipt: {str(e)[:100]}"

    def _format_receipt(self, receipt_data: dict) -> str:
        """Format a receipt for Telegram display."""
        receipt_id = receipt_data.get("receipt_id", receipt_data.get("id", "N/A"))
        topic = receipt_data.get("topic", receipt_data.get("question", "Unknown"))[:100]
        decision = receipt_data.get("decision", receipt_data.get("conclusion", "N/A"))[:300]
        confidence = receipt_data.get("confidence", 0)
        timestamp = receipt_data.get("timestamp", receipt_data.get("created_at", "N/A"))
        agents = receipt_data.get("agents", receipt_data.get("participants", []))

        lines = [
            "*Decision Receipt*\n",
            f"*Receipt ID:* `{receipt_id}`",
            f"*Topic:* {topic}{'...' if len(receipt_data.get('topic', '')) > 100 else ''}",
            f"*Decision:* {decision}{'...' if len(receipt_data.get('decision', '')) > 300 else ''}",
            f"*Confidence:* {confidence:.0%}"
            if isinstance(confidence, (int, float))
            else f"*Confidence:* {confidence}",
            f"*Timestamp:* {timestamp}",
        ]

        if agents:
            agent_names = [a.get("name", a) if isinstance(a, dict) else str(a) for a in agents[:5]]
            lines.append(f"*Agents:* {', '.join(agent_names)}")

        # Add verification hash if available
        if receipt_data.get("hash"):
            lines.append(f"\n_Verification Hash:_ `{receipt_data['hash'][:16]}...`")

        return "\n".join(lines)

    def _format_debate_as_receipt(self, debate: dict) -> str:
        """Format a debate as a receipt when receipt module unavailable."""
        debate_id = debate.get("id", "N/A")
        topic = debate.get("topic", "Unknown")[:100]
        conclusion = debate.get("conclusion", debate.get("final_answer", "N/A"))[:300]
        consensus = "Yes" if debate.get("consensus_reached") else "No"
        confidence = debate.get("confidence", 0)

        lines = [
            "*Debate Summary*\n",
            f"*Debate ID:* `{debate_id}`",
            f"*Topic:* {topic}",
            f"*Conclusion:* {conclusion}",
            f"*Consensus Reached:* {consensus}",
            f"*Confidence:* {confidence:.0%}"
            if isinstance(confidence, (int, float))
            else f"*Confidence:* {confidence}",
        ]

        if debate.get("rounds_used"):
            lines.append(f"*Rounds:* {debate['rounds_used']}")

        return "\n".join(lines)
