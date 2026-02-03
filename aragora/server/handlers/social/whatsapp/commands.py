"""
WhatsApp bot command implementations.

Handles:
- help - Show available commands
- status - Get system status
- agents - List available agents
- debate <topic> - Start a multi-agent debate
- plan <topic> - Debate with an implementation plan
- implement <topic> - Debate with plan + context snapshot
- gauntlet <statement> - Run adversarial validation
- search <query> - Search past debates
- recent - Show recent debates
- receipt <id> - View decision receipt
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.config import DEFAULT_CONSENSUS, DEFAULT_ROUNDS

from . import config as _config
from .config import (
    PERM_WHATSAPP_DEBATES,
    PERM_WHATSAPP_GAUNTLET,
    TTS_VOICE_ENABLED,
)
from .messaging import (
    send_interactive_buttons,
    send_text_message,
    send_voice_summary,
)
from ..telemetry import (
    record_debate_completed,
    record_debate_failed,
    record_debate_started,
    record_gauntlet_completed,
    record_gauntlet_failed,
    record_gauntlet_started,
)
from ..chat_events import (
    emit_debate_completed,
    emit_debate_started,
    emit_gauntlet_completed,
    emit_gauntlet_started,
)

logger = logging.getLogger(__name__)


def command_help() -> str:
    """Return help message."""
    return (
        "*Aragora Commands*\n\n"
        "*help* - Show this help message\n"
        "*status* - Get system status\n"
        "*agents* - List available agents\n"
        "*debate <topic>* - Start a multi-agent debate\n"
        "*plan <topic>* - Debate with an implementation plan\n"
        "*implement <topic>* - Debate with plan + context snapshot\n"
        "*gauntlet <statement>* - Run adversarial stress-test\n"
        "*search <query>* - Search past debates\n"
        "*recent* - Show recent debates\n"
        "*receipt <id>* - View decision receipt\n\n"
        "*Examples:*\n"
        "debate Should AI be regulated?\n"
        "plan Improve our on-call process\n"
        "implement Automate weekly incident reporting\n"
        "gauntlet We should migrate to microservices\n"
        "search machine learning\n"
        "receipt abc123"
    )


def command_status() -> str:
    """Return status message."""
    try:
        from aragora.ranking.elo import EloSystem

        store = EloSystem()
        agents = store.get_all_ratings()
        return f"*Aragora Status*\n\nStatus: Online\nAgents: {len(agents)} registered"
    except Exception as e:
        logger.warning(f"Failed to get status: {e}")
        return "*Aragora Status*\n\nStatus: Online"


def command_agents() -> str:
    """Return agents list."""
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
            lines.append(f"{i + 1}. *{name}* - ELO: {elo:.0f} | Wins: {wins}")

        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Failed to list agents: {e}")
        return "Could not fetch agent list."


def command_debate(
    handler_instance: Any,
    from_number: str,
    profile_name: str,
    topic: str,
    decision_integrity: dict[str, Any] | bool | None = None,
) -> None:
    """Handle debate command.

    RBAC: Requires whatsapp:debates:create permission.

    Args:
        handler_instance: The WhatsAppHandler instance (for RBAC checks).
        from_number: Sender's WhatsApp phone number.
        profile_name: Sender's profile name.
        topic: Debate topic.
    """
    # RBAC: Check permission to create debates
    perm_error = handler_instance._check_whatsapp_permission(
        from_number, PERM_WHATSAPP_DEBATES, profile_name
    )
    if perm_error:
        _config.create_tracked_task(
            send_text_message(
                from_number,
                "Sorry, you don't have permission to start debates. "
                "Please contact your administrator.",
            ),
            name=f"whatsapp-debate-perm-denied-{from_number}",
        )
        return

    topic = topic.strip().strip("\"'")

    if len(topic) < 10:
        _config.create_tracked_task(
            send_text_message(
                from_number,
                "Topic is too short. Please provide more detail.",
            ),
            name=f"whatsapp-debate-short-{from_number}",
        )
        return

    if len(topic) > 500:
        _config.create_tracked_task(
            send_text_message(
                from_number,
                "Topic is too long. Please limit to 500 characters.",
            ),
            name=f"whatsapp-debate-long-{from_number}",
        )
        return

    # Send acknowledgment
    _config.create_tracked_task(
        send_text_message(
            from_number,
            f"*Starting debate on:*\n_{topic}_\n\nRequested by {profile_name}\nProcessing... (this may take a few minutes)",
        ),
        name=f"whatsapp-debate-ack-{from_number}",
    )

    # Run debate asynchronously
    ctx = getattr(handler_instance, "ctx", {}) or {}
    _config.create_tracked_task(
        run_debate_async(
            from_number,
            profile_name,
            topic,
            decision_integrity=decision_integrity,
            document_store=ctx.get("document_store"),
            evidence_store=ctx.get("evidence_store"),
        ),
        name=f"whatsapp-debate-{topic[:30]}",
    )


async def run_debate_async(
    from_number: str,
    profile_name: str,
    topic: str,
    decision_integrity: dict[str, Any] | bool | None = None,
    document_store: Any | None = None,
    evidence_store: Any | None = None,
) -> None:
    """Run debate and send result."""
    record_debate_started("whatsapp")
    debate_id = None
    try:
        from aragora import Arena, DebateProtocol, Environment
        from aragora.agents import get_agents_by_names

        # Register debate origin for tracking
        try:
            from aragora.server.debate_origin import register_debate_origin
            import uuid

            debate_id = f"wa-{from_number[-8:]}-{uuid.uuid4().hex[:8]}"
            register_debate_origin(
                debate_id=debate_id,
                platform="whatsapp",
                channel_id=from_number,
                user_id=from_number,
                metadata={"profile_name": profile_name, "topic": topic},
            )
            logger.debug(f"Registered WhatsApp debate origin: {debate_id}")
        except ImportError:
            logger.debug("Debate origin tracking not available")

        # Emit webhook event for debate started
        emit_debate_started(
            platform="whatsapp",
            chat_id=from_number,
            user_id=from_number,
            username=profile_name,
            topic=topic,
            debate_id=debate_id,
        )

        env = Environment(task=f"Debate: {topic}")

        # Resolve agent binding for this conversation
        agent_names = ["anthropic-api", "openai-api"]  # Default agents
        protocol_config: dict[str, Any] = {
            "rounds": 3,
            "consensus": "majority",
            "convergence_detection": False,
            "early_stopping": False,
        }
        try:
            from aragora.server.bindings import get_binding_router, BindingType

            router = get_binding_router()
            resolution = router.resolve(
                provider="whatsapp",
                account_id="default",
                peer_id=f"dm:{from_number}",
                user_id=from_number,
            )

            if resolution.matched and resolution.binding:
                logger.debug(
                    f"Binding resolved: {resolution.agent_binding} "
                    f"type={resolution.binding_type} reason={resolution.match_reason}"
                )

                # Apply agent binding
                if resolution.binding_type == BindingType.SPECIFIC_AGENT:
                    agent_names = [resolution.agent_binding]
                elif resolution.binding_type == BindingType.AGENT_POOL:
                    pool_name = resolution.agent_binding
                    if pool_name == "full-team":
                        agent_names = ["anthropic-api", "openai-api", "mistral-api"]
                    elif pool_name == "fast":
                        agent_names = ["openai-api"]

                # Apply config overrides from binding
                if resolution.config_overrides:
                    protocol_config.update(resolution.config_overrides)

        except ImportError:
            logger.debug("Binding router not available, using default agents")
        except Exception as e:
            logger.debug(f"Binding resolution failed: {e}, using default agents")

        agents = get_agents_by_names(agent_names)
        protocol = DebateProtocol(
            rounds=protocol_config.get("rounds", DEFAULT_ROUNDS),
            consensus=protocol_config.get("consensus", DEFAULT_CONSENSUS),
            convergence_detection=protocol_config.get("convergence_detection", False),
            early_stopping=protocol_config.get("early_stopping", False),
        )

        if not agents:
            await send_text_message(
                from_number,
                "Failed to start debate: No agents available",
            )
            record_debate_failed("whatsapp")
            return

        arena = Arena.from_env(env, agents, protocol)
        result = await arena.run()

        response = (
            f"*Debate Complete!*\n\n"
            f"*Topic:* {topic[:100]}{'...' if len(topic) > 100 else ''}\n\n"
            f"*Consensus:* {'Yes' if result.consensus_reached else 'No'}\n"
            f"*Confidence:* {result.confidence:.1%}\n"
            f"*Rounds:* {result.rounds_used}\n"
            f"*Agents:* {len(agents)}\n\n"
            f"*Conclusion:*\n{result.final_answer[:500] if result.final_answer else 'No conclusion'}{'...' if result.final_answer and len(result.final_answer) > 500 else ''}\n\n"
            f"_Debate ID: {result.id}_\n"
            f"_Requested by {profile_name}_"
        )

        # Send result with interactive buttons
        await send_interactive_buttons(
            from_number,
            response,
            [
                {"id": f"vote_agree_{result.id}", "title": "Agree"},
                {"id": f"vote_disagree_{result.id}", "title": "Disagree"},
                {"id": f"details_{result.id}", "title": "View Details"},
            ],
            "Vote on this debate",
        )

        # Mark debate result as sent for origin tracking
        if debate_id:
            try:
                from aragora.server.debate_origin import mark_result_sent

                mark_result_sent(debate_id)
            except ImportError:
                pass

        # Send voice summary if TTS is enabled
        if TTS_VOICE_ENABLED:
            await send_voice_summary(
                from_number,
                topic,
                result.final_answer,
                result.consensus_reached,
                result.confidence,
                result.rounds_used,
            )

        # Emit webhook event for debate completed
        emit_debate_completed(
            platform="whatsapp",
            chat_id=from_number,
            debate_id=result.id,
            topic=topic,
            consensus_reached=result.consensus_reached,
            confidence=result.confidence,
            rounds_used=result.rounds_used,
            final_answer=result.final_answer,
        )

        # Record successful debate completion
        record_debate_completed("whatsapp", result.consensus_reached)

        # Optionally emit decision integrity package
        from aragora.server.decision_integrity_utils import (
            maybe_emit_decision_integrity,
        )

        await maybe_emit_decision_integrity(
            result=result,
            debate_id=debate_id or getattr(result, "debate_id", None),
            arena=arena,
            decision_integrity=decision_integrity,
            document_store=document_store,
            evidence_store=evidence_store,
        )

    except Exception as e:
        logger.error(f"WhatsApp debate failed: {e}", exc_info=True)
        record_debate_failed("whatsapp")
        await send_text_message(
            from_number,
            f"Debate failed: {str(e)[:100]}",
        )


def command_gauntlet(
    handler_instance: Any,
    from_number: str,
    profile_name: str,
    statement: str,
) -> None:
    """Handle gauntlet command.

    RBAC: Requires whatsapp:gauntlet:run permission.

    Args:
        handler_instance: The WhatsAppHandler instance (for RBAC checks).
        from_number: Sender's WhatsApp phone number.
        profile_name: Sender's profile name.
        statement: Statement to stress-test.
    """
    # RBAC: Check permission to run gauntlet stress tests
    perm_error = handler_instance._check_whatsapp_permission(
        from_number, PERM_WHATSAPP_GAUNTLET, profile_name
    )
    if perm_error:
        _config.create_tracked_task(
            send_text_message(
                from_number,
                "Sorry, you don't have permission to run gauntlet stress tests. "
                "Please contact your administrator.",
            ),
            name=f"whatsapp-gauntlet-perm-denied-{from_number}",
        )
        return

    statement = statement.strip().strip("\"'")

    if len(statement) < 10:
        _config.create_tracked_task(
            send_text_message(
                from_number,
                "Statement is too short. Please provide more detail.",
            ),
            name=f"whatsapp-gauntlet-short-{from_number}",
        )
        return

    if len(statement) > 1000:
        _config.create_tracked_task(
            send_text_message(
                from_number,
                "Statement is too long. Please limit to 1000 characters.",
            ),
            name=f"whatsapp-gauntlet-long-{from_number}",
        )
        return

    # Send acknowledgment
    _config.create_tracked_task(
        send_text_message(
            from_number,
            f"*Running Gauntlet stress-test on:*\n_{statement[:200]}{'...' if len(statement) > 200 else ''}_\n\nRequested by {profile_name}\nRunning adversarial validation...",
        ),
        name=f"whatsapp-gauntlet-ack-{from_number}",
    )

    # Run gauntlet asynchronously
    _config.create_tracked_task(
        run_gauntlet_async(from_number, profile_name, statement),
        name=f"whatsapp-gauntlet-{statement[:30]}",
    )


async def run_gauntlet_async(
    from_number: str,
    profile_name: str,
    statement: str,
) -> None:
    """Run gauntlet and send result."""
    from aragora.server.http_client_pool import get_http_pool

    record_gauntlet_started("whatsapp")

    # Emit webhook event for gauntlet started
    emit_gauntlet_started(
        platform="whatsapp",
        chat_id=from_number,
        user_id=from_number,
        username=profile_name,
        statement=statement,
    )

    try:
        pool = get_http_pool()
        async with pool.get_session("whatsapp_gauntlet") as client:
            resp = await client.post(
                "http://localhost:8080/api/gauntlet/run",
                json={
                    "statement": statement,
                    "intensity": "medium",
                    "metadata": {
                        "source": "whatsapp",
                        "from_number": from_number,
                    },
                },
                timeout=120,
            )
            data = resp.json()

            if resp.status_code != 200:
                await send_text_message(
                    from_number,
                    f"Gauntlet failed: {data.get('error', 'Unknown error')}",
                )
                record_gauntlet_failed("whatsapp")
                return

            run_id = data.get("run_id", "unknown")
            score = data.get("score", 0)
            passed = data.get("passed", False)
            vulnerabilities = data.get("vulnerabilities", [])

            response = (
                f"*Gauntlet Results* {'PASSED' if passed else 'FAILED'}\n\n"
                f"*Statement:*\n_{statement[:200]}{'...' if len(statement) > 200 else ''}_\n\n"
                f"*Score:* {score:.1%}\n"
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

            response += f"\n_Run ID: {run_id}_\n_Requested by {profile_name}_"

            await send_text_message(from_number, response)

            # Emit webhook event for gauntlet completed
            emit_gauntlet_completed(
                platform="whatsapp",
                chat_id=from_number,
                gauntlet_id=run_id,
                statement=statement,
                verdict="passed" if passed else "failed",
                confidence=score,
                challenges_passed=len([v for v in vulnerabilities if not v.get("critical", False)]),
                challenges_total=len(vulnerabilities) + 1,
            )

            # Record successful gauntlet completion
            record_gauntlet_completed("whatsapp", passed)

    except Exception as e:
        logger.error(f"WhatsApp gauntlet failed: {e}", exc_info=True)
        record_gauntlet_failed("whatsapp")
        await send_text_message(
            from_number,
            f"Gauntlet failed: {str(e)[:100]}",
        )


def command_search(query: str) -> str:
    """Search past debates.

    Args:
        query: Search query string.

    Returns:
        Formatted search results.
    """
    if not query or len(query.strip()) < 3:
        return (
            "Please provide a search query (at least 3 characters).\n\n"
            "Example: search machine learning"
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
            topic = debate.get("topic", "Unknown topic")[:50]
            debate_id = debate.get("id", "N/A")
            consensus = "Yes" if debate.get("consensus_reached") else "No"
            lines.append(f"{i}. *{topic}*{'...' if len(debate.get('topic', '')) > 50 else ''}")
            lines.append(f"   ID: {debate_id} | Consensus: {consensus}")

        if total > 5:
            lines.append(f"\n_Showing 5 of {total} results_")

        return "\n".join(lines)

    except ImportError:
        logger.warning("Storage not available for search")
        return "Search service temporarily unavailable."
    except Exception as e:
        logger.exception(f"Unexpected search error: {e}")
        return f"Search failed: {str(e)[:100]}"


def command_recent() -> str:
    """Show recent debates.

    Returns:
        Formatted list of recent debates.
    """
    try:
        from aragora.storage import get_storage

        db = get_storage()
        if not db or not hasattr(db, "get_recent_debates"):
            return "Recent debates service is not available."

        debates = db.get_recent_debates(limit=5)

        if not debates:
            return "No recent debates found.\n\nStart one with: debate <topic>"

        lines = ["*Recent Debates*\n"]
        for i, debate in enumerate(debates[:5], 1):
            topic = debate.get("topic", "Unknown topic")[:40]
            debate_id = debate.get("id", "N/A")
            consensus = "Yes" if debate.get("consensus_reached") else "No"
            confidence = debate.get("confidence", 0)

            lines.append(f"{i}. *{topic}*{'...' if len(debate.get('topic', '')) > 40 else ''}")
            lines.append(f"   ID: {debate_id}")
            lines.append(f"   Consensus: {consensus} | Confidence: {confidence:.0%}")

        lines.append("\n_Use receipt <id> to view decision receipt_")

        return "\n".join(lines)

    except ImportError:
        logger.warning("Storage not available for recent debates")
        return "Recent debates service temporarily unavailable."
    except Exception as e:
        logger.exception(f"Unexpected recent debates error: {e}")
        return f"Failed to get recent debates: {str(e)[:100]}"


def command_receipt(debate_id: str) -> str:
    """View decision receipt for a debate.

    Args:
        debate_id: The debate ID to get receipt for.

    Returns:
        Formatted decision receipt.
    """
    if not debate_id or not debate_id.strip():
        return (
            "Please provide a debate ID.\n\n"
            "Example: receipt abc123\n\n"
            "Use recent to see recent debate IDs."
        )

    debate_id = debate_id.strip()

    try:
        # Try to get receipt from receipt store
        try:
            from aragora.storage.receipt_store import get_receipt_store

            receipt_store = get_receipt_store()
            receipt_data = receipt_store.get(debate_id)

            if receipt_data:
                return _format_receipt(receipt_data.to_dict())
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
            return _format_receipt(receipt.to_dict())
        except ImportError:
            # Manual formatting if receipt module unavailable
            return _format_debate_as_receipt(debate)

    except Exception as e:
        logger.exception(f"Unexpected receipt error: {e}")
        return f"Failed to get receipt: {str(e)[:100]}"


def _format_receipt(receipt_data: dict) -> str:
    """Format a receipt for WhatsApp display."""
    receipt_id = receipt_data.get("receipt_id", receipt_data.get("id", "N/A"))
    topic = receipt_data.get("topic", receipt_data.get("question", "Unknown"))[:80]
    decision = receipt_data.get("decision", receipt_data.get("conclusion", "N/A"))[:250]
    confidence = receipt_data.get("confidence", 0)
    timestamp = receipt_data.get("timestamp", receipt_data.get("created_at", "N/A"))
    agents = receipt_data.get("agents", receipt_data.get("participants", []))

    lines = [
        "*Decision Receipt*\n",
        f"*Receipt ID:* {receipt_id}",
        f"*Topic:* {topic}{'...' if len(receipt_data.get('topic', '')) > 80 else ''}",
        f"*Decision:* {decision}{'...' if len(receipt_data.get('decision', '')) > 250 else ''}",
    ]

    if isinstance(confidence, (int, float)):
        lines.append(f"*Confidence:* {confidence:.0%}")
    else:
        lines.append(f"*Confidence:* {confidence}")

    lines.append(f"*Timestamp:* {timestamp}")

    if agents:
        agent_names = [a.get("name", a) if isinstance(a, dict) else str(a) for a in agents[:5]]
        lines.append(f"*Agents:* {', '.join(agent_names)}")

    # Add verification hash if available
    if receipt_data.get("hash"):
        lines.append(f"\n_Verification:_ {receipt_data['hash'][:16]}...")

    return "\n".join(lines)


def _format_debate_as_receipt(debate: dict) -> str:
    """Format a debate as a receipt when receipt module unavailable."""
    debate_id = debate.get("id", "N/A")
    topic = debate.get("topic", "Unknown")[:80]
    conclusion = debate.get("conclusion", debate.get("final_answer", "N/A"))[:250]
    consensus = "Yes" if debate.get("consensus_reached") else "No"
    confidence = debate.get("confidence", 0)

    lines = [
        "*Debate Summary*\n",
        f"*Debate ID:* {debate_id}",
        f"*Topic:* {topic}",
        f"*Conclusion:* {conclusion}",
        f"*Consensus Reached:* {consensus}",
    ]

    if isinstance(confidence, (int, float)):
        lines.append(f"*Confidence:* {confidence:.0%}")
    else:
        lines.append(f"*Confidence:* {confidence}")

    if debate.get("rounds_used"):
        lines.append(f"*Rounds:* {debate['rounds_used']}")

    return "\n".join(lines)
