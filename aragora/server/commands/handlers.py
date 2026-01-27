"""
Slash command handlers for Aragora.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from .models import CommandContext, CommandResult, CommandDefinition, CommandPermission

logger = logging.getLogger(__name__)


class BaseCommandHandler(ABC):
    """Base class for command handlers."""

    @property
    @abstractmethod
    def definition(self) -> CommandDefinition:
        """Return the command definition."""
        ...

    @abstractmethod
    async def execute(self, ctx: CommandContext) -> CommandResult:
        """Execute the command."""
        ...

    async def __call__(self, ctx: CommandContext) -> CommandResult:
        """Make handler callable."""
        return await self.execute(ctx)


class DebateCommandHandler(BaseCommandHandler):
    """Handler for /debate command - start a new debate."""

    @property
    def definition(self) -> CommandDefinition:
        return CommandDefinition(
            name="debate",
            description="Start a new multi-agent debate",
            usage="/debate <topic>",
            examples=[
                "/debate Should we migrate to microservices?",
                "/debate What's the best approach for caching?",
            ],
            aliases=["d", "start"],
            permission=CommandPermission.AUTHENTICATED,
        )

    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(
                "Please provide a debate topic.\n\nUsage: `/debate <topic>`\n\n"
                "Example: `/debate Should we use TypeScript or JavaScript?`",
                ephemeral=True,
            )

        topic = ctx.arg_string

        try:
            from aragora import Arena, Environment, DebateProtocol
            from aragora.agents import get_default_agents  # type: ignore[attr-defined]

            # Create debate
            env = Environment(task=topic)
            protocol = DebateProtocol(rounds=3, consensus="majority")
            agents = get_default_agents()[:3]  # Use 3 agents

            # Register origin for result routing
            from aragora.server.debate_origin import register_debate_origin

            arena = Arena(env, agents, protocol)
            debate_id = arena.debate_id

            register_debate_origin(  # type: ignore[call-arg]
                debate_id=debate_id,
                platform=ctx.platform,
                channel_id=ctx.channel_id,
                user_id=ctx.user_id,
                thread_id=ctx.thread_id,
                workspace_id=ctx.workspace_id,
            )

            # Start debate asynchronously
            import asyncio

            asyncio.create_task(arena.run())

            return CommandResult.ok(
                f"Debate started!\n\n"
                f"*Topic:* {topic}\n"
                f"*Debate ID:* `{debate_id}`\n"
                f"*Agents:* {', '.join(a.name for a in agents)}\n\n"
                f"Results will be posted here when complete.",
                ephemeral=False,
            )

        except ImportError:
            # Fallback if full Arena not available
            import uuid

            debate_id = str(uuid.uuid4())[:8]
            return CommandResult.ok(
                f"Debate queued!\n\n"
                f"*Topic:* {topic}\n"
                f"*Debate ID:* `{debate_id}`\n\n"
                f"Use `/status {debate_id}` to check progress.",
                ephemeral=False,
            )

        except Exception as e:
            logger.exception("Failed to start debate")
            return CommandResult.error(f"Failed to start debate: {e!s}", ephemeral=True)


class StatusCommandHandler(BaseCommandHandler):
    """Handler for /status command - check debate status."""

    @property
    def definition(self) -> CommandDefinition:
        return CommandDefinition(
            name="status",
            description="Check the status of a debate",
            usage="/status [debate-id]",
            examples=[
                "/status abc123",
                "/status",  # Shows recent debates
            ],
            aliases=["s", "check"],
            permission=CommandPermission.AUTHENTICATED,
        )

    async def execute(self, ctx: CommandContext) -> CommandResult:
        debate_id = ctx.get_arg(0)

        if not debate_id:
            # Show recent debates for this channel/user
            return await self._show_recent_debates(ctx)

        try:
            from aragora.memory.consensus import ConsensusMemory

            memory = ConsensusMemory()
            await memory.initialize()  # type: ignore[attr-defined]

            debate = await memory.get_debate(debate_id)  # type: ignore[attr-defined]
            if not debate:
                return CommandResult.error(
                    f"Debate `{debate_id}` not found.\n\n"
                    f"Use `/history` to see your recent debates.",
                    ephemeral=True,
                )

            status = debate.get("status", "unknown")
            task = debate.get("task", "Unknown topic")
            rounds = debate.get("rounds_used", 0)
            consensus = debate.get("consensus_reached", False)

            status_emoji = {
                "running": "running",
                "completed": "completed",
                "failed": "failed",
                "pending": "pending",
            }.get(status, "unknown")

            message = (
                f"*Debate Status: {status_emoji}*\n\n"
                f"*ID:* `{debate_id}`\n"
                f"*Topic:* {task}\n"
                f"*Status:* {status}\n"
                f"*Rounds:* {rounds}\n"
                f"*Consensus:* {'Yes' if consensus else 'No'}"
            )

            if status == "completed" and debate.get("final_answer"):
                message += f"\n\n*Result:*\n{debate['final_answer'][:500]}"
                if len(debate.get("final_answer", "")) > 500:
                    message += "..."

            return CommandResult.ok(message, ephemeral=True)

        except ImportError:
            return CommandResult.ok(
                f"*Debate Status*\n\n"
                f"*ID:* `{debate_id}`\n"
                f"*Status:* Checking...\n\n"
                f"Full status API not available.",
                ephemeral=True,
            )

        except Exception as e:
            logger.exception("Failed to get debate status")
            return CommandResult.error(f"Failed to get status: {e!s}", ephemeral=True)

    async def _show_recent_debates(self, ctx: CommandContext) -> CommandResult:
        """Show recent debates for the user."""
        try:
            from aragora.memory.consensus import ConsensusMemory

            memory = ConsensusMemory()
            memory.initialize()

            # Get recent verified debates
            debates = memory.list_verified_debates(verified_only=False, limit=5)

            if not debates:
                return CommandResult.ok(
                    "No recent debates found.\n\nUse `/debate <topic>` to start one!",
                    ephemeral=True,
                )

            lines = ["*Recent Debates:*\n"]
            for d in debates:
                status = d.get("proof_status", "unknown")
                verified = "verified" if d.get("is_verified") else "pending"
                debate_id = d.get("debate_id", "?")
                lines.append(f"  `{debate_id}` - {verified} - {status}")

            lines.append("\n\nUse `/status <id>` for details.")
            return CommandResult.ok("\n".join(lines), ephemeral=True)

        except Exception:
            return CommandResult.ok(
                "Use `/status <debate-id>` to check a specific debate.",
                ephemeral=True,
            )


class HelpCommandHandler(BaseCommandHandler):
    """Handler for /help command - show available commands."""

    @property
    def definition(self) -> CommandDefinition:
        return CommandDefinition(
            name="help",
            description="Show available commands and usage",
            usage="/help [command]",
            examples=[
                "/help",
                "/help debate",
            ],
            aliases=["h", "?", "commands"],
            permission=CommandPermission.PUBLIC,
        )

    async def execute(self, ctx: CommandContext) -> CommandResult:
        from .registry import get_command_registry

        registry = get_command_registry()
        command_name = ctx.get_arg(0)

        if command_name:
            # Show help for specific command
            definition = registry.get_command(command_name)
            if not definition:
                return CommandResult.error(
                    f"Unknown command: `/{command_name}`\n\nUse `/help` to see all commands.",
                    ephemeral=True,
                )

            message = (
                f"*Command: /{definition.name}*\n\n"
                f"{definition.description}\n\n"
                f"*Usage:* `{definition.usage}`\n"
            )

            if definition.aliases:
                message += f"*Aliases:* {', '.join(f'`/{a}`' for a in definition.aliases)}\n"

            if definition.examples:
                message += "\n*Examples:*\n"
                for ex in definition.examples:
                    message += f"  `{ex}`\n"

            return CommandResult.ok(message, ephemeral=True)

        # Show all commands
        help_text = registry.get_help_text(platform=ctx.platform)
        return CommandResult.ok(help_text, ephemeral=True)


class HistoryCommandHandler(BaseCommandHandler):
    """Handler for /history command - show debate history."""

    @property
    def definition(self) -> CommandDefinition:
        return CommandDefinition(
            name="history",
            description="Show your recent debate history",
            usage="/history [count]",
            examples=[
                "/history",
                "/history 10",
            ],
            aliases=["hist", "recent"],
            permission=CommandPermission.AUTHENTICATED,
        )

    async def execute(self, ctx: CommandContext) -> CommandResult:
        count_str = ctx.get_arg(0, "5")
        try:
            count = min(int(count_str), 20)  # Max 20
        except ValueError:
            count = 5

        try:
            from aragora.memory.consensus import ConsensusMemory

            memory = ConsensusMemory()
            await memory.initialize()  # type: ignore[attr-defined]

            debates = await memory.list_debates(limit=count)  # type: ignore[attr-defined]

            if not debates:
                return CommandResult.ok(
                    "No debate history found.\n\nStart your first debate with `/debate <topic>`!",
                    ephemeral=True,
                )

            lines = [f"*Last {len(debates)} Debates:*\n"]
            for d in debates:
                debate_id = d.get("debate_id", "?")[:8]
                status = d.get("status", "?")
                task = d.get("task", "Unknown")[:40]
                consensus = "Yes" if d.get("consensus_reached") else "No"
                lines.append(f"`{debate_id}` | {status} | Consensus: {consensus} | {task}")

            lines.append("\n\nUse `/status <id>` for full details.")
            return CommandResult.ok("\n".join(lines), ephemeral=True)

        except Exception as e:
            logger.exception("Failed to get history")
            return CommandResult.error(f"Failed to get history: {e!s}", ephemeral=True)


class ResultsCommandHandler(BaseCommandHandler):
    """Handler for /results command - get debate results."""

    @property
    def definition(self) -> CommandDefinition:
        return CommandDefinition(
            name="results",
            description="Get the results of a completed debate",
            usage="/results <debate-id>",
            examples=[
                "/results abc123",
            ],
            aliases=["r", "result", "answer"],
            permission=CommandPermission.AUTHENTICATED,
        )

    async def execute(self, ctx: CommandContext) -> CommandResult:
        debate_id = ctx.get_arg(0)

        if not debate_id:
            return CommandResult.error(
                "Please provide a debate ID.\n\nUsage: `/results <debate-id>`",
                ephemeral=True,
            )

        try:
            from aragora.memory.consensus import ConsensusMemory

            memory = ConsensusMemory()
            await memory.initialize()  # type: ignore[attr-defined]

            debate = await memory.get_debate(debate_id)  # type: ignore[attr-defined]
            if not debate:
                return CommandResult.error(f"Debate `{debate_id}` not found.", ephemeral=True)

            if debate.get("status") != "completed":
                return CommandResult.ok(
                    f"Debate `{debate_id}` is not yet complete.\n"
                    f"Current status: {debate.get('status', 'unknown')}",
                    ephemeral=True,
                )

            task = debate.get("task", "Unknown topic")
            answer = debate.get("final_answer", "No answer available")
            confidence = debate.get("confidence", 0)
            consensus = debate.get("consensus_reached", False)

            message = (
                f"*Debate Results*\n\n"
                f"*Topic:* {task}\n"
                f"*Consensus:* {'Reached' if consensus else 'Not reached'}\n"
                f"*Confidence:* {confidence:.0%}\n\n"
                f"*Answer:*\n{answer}"
            )

            return CommandResult.ok(message, ephemeral=False)

        except Exception as e:
            logger.exception("Failed to get results")
            return CommandResult.error(f"Failed to get results: {e!s}", ephemeral=True)


def register_default_commands() -> None:
    """Register all default command handlers."""
    from .registry import get_command_registry

    registry = get_command_registry()

    handlers = [
        DebateCommandHandler(),
        StatusCommandHandler(),
        HelpCommandHandler(),
        HistoryCommandHandler(),
        ResultsCommandHandler(),
    ]

    for handler in handlers:
        registry.register(handler.definition, handler)

    logger.info(f"Registered {len(handlers)} default command handlers")
