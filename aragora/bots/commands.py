"""
Command registry and built-in commands for the bot framework.

Provides a decorator-based system for registering commands that can be
invoked from any supported platform.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from aragora.bots.base import (
    CommandContext,
    CommandResult,
    Platform,
)

logger = logging.getLogger(__name__)

# Type alias for command handlers
CommandHandler = Callable[[CommandContext], Coroutine[Any, Any, CommandResult]]


@dataclass
class BotCommand:
    """Definition of a bot command."""

    name: str
    handler: CommandHandler
    description: str = ""
    usage: str = ""
    aliases: List[str] = field(default_factory=list)
    platforms: Set[Platform] = field(default_factory=lambda: set(Platform))
    requires_args: bool = False
    min_args: int = 0
    max_args: Optional[int] = None
    admin_only: bool = False
    rate_limit: Optional[int] = None  # Max calls per minute per user
    cooldown: float = 0  # Seconds between invocations

    def __post_init__(self) -> None:
        if not self.usage and self.requires_args:
            self.usage = f"{self.name} <args>"

    def matches_platform(self, platform: Platform) -> bool:
        """Check if command is available on the given platform."""
        return platform in self.platforms

    def validate_args(self, args: List[str]) -> Optional[str]:
        """Validate argument count. Returns error message if invalid."""
        if self.requires_args and len(args) < 1:
            return f"Command '{self.name}' requires arguments. Usage: {self.usage}"
        if len(args) < self.min_args:
            return f"Command '{self.name}' requires at least {self.min_args} argument(s)."
        if self.max_args is not None and len(args) > self.max_args:
            return f"Command '{self.name}' accepts at most {self.max_args} argument(s)."
        return None


class CommandRegistry:
    """Registry for bot commands."""

    def __init__(self) -> None:
        self._commands: Dict[str, BotCommand] = {}
        self._aliases: Dict[str, str] = {}  # alias -> command name
        self._cooldowns: Dict[str, Dict[str, float]] = {}  # command -> user_id -> timestamp

    def register(self, command: BotCommand) -> None:
        """Register a command."""
        if command.name in self._commands:
            logger.warning(f"Overwriting existing command: {command.name}")

        self._commands[command.name] = command

        # Register aliases
        for alias in command.aliases:
            if alias in self._aliases:
                logger.warning(f"Alias '{alias}' already registered for '{self._aliases[alias]}'")
            self._aliases[alias] = command.name

        logger.debug(f"Registered command: {command.name}")

    def unregister(self, name: str) -> bool:
        """Unregister a command by name."""
        if name not in self._commands:
            return False

        command = self._commands.pop(name)

        # Remove aliases
        for alias in command.aliases:
            self._aliases.pop(alias, None)

        return True

    def get(self, name: str) -> Optional[BotCommand]:
        """Get a command by name or alias."""
        # Check direct name first
        if name in self._commands:
            return self._commands[name]

        # Check aliases
        if name in self._aliases:
            return self._commands.get(self._aliases[name])

        return None

    def list_for_platform(self, platform: Platform) -> List[BotCommand]:
        """Get all commands available on a platform."""
        return [cmd for cmd in self._commands.values() if cmd.matches_platform(platform)]

    def list_commands(self, platform: Optional[Platform] = None) -> List[BotCommand]:
        """List all registered commands, optionally filtered by platform."""
        if platform is not None:
            return self.list_for_platform(platform)
        return list(self._commands.values())

    def _check_cooldown(self, command: BotCommand, user_id: str) -> Optional[float]:
        """Check if user is on cooldown. Returns remaining seconds if so."""
        if command.cooldown <= 0:
            return None

        if command.name not in self._cooldowns:
            self._cooldowns[command.name] = {}

        user_cooldowns = self._cooldowns[command.name]
        import time

        now = time.time()

        if user_id in user_cooldowns:
            elapsed = now - user_cooldowns[user_id]
            if elapsed < command.cooldown:
                return command.cooldown - elapsed

        return None

    def _update_cooldown(self, command: BotCommand, user_id: str) -> None:
        """Update cooldown timestamp for a user."""
        if command.cooldown <= 0:
            return

        if command.name not in self._cooldowns:
            self._cooldowns[command.name] = {}

        import time

        self._cooldowns[command.name][user_id] = time.time()

    async def execute(self, ctx: CommandContext) -> CommandResult:
        """Execute a command from context."""
        if not ctx.args:
            return CommandResult.fail("No command specified")

        command_name = ctx.args[0].lower()
        command = self.get(command_name)

        if not command:
            return CommandResult.fail(
                f"Unknown command: {command_name}. Use 'help' to see available commands."
            )

        # Check platform availability
        if not command.matches_platform(ctx.platform):
            return CommandResult.fail(
                f"Command '{command_name}' is not available on {ctx.platform.value}."
            )

        # Check cooldown
        remaining = self._check_cooldown(command, ctx.user_id)
        if remaining is not None:
            return CommandResult.fail(
                f"Please wait {remaining:.1f} seconds before using '{command_name}' again."
            )

        # Validate arguments (excluding command name)
        args = ctx.args[1:]
        error = command.validate_args(args)
        if error:
            return CommandResult.fail(error)

        # Create new context with parsed args
        exec_ctx = CommandContext(
            message=ctx.message,
            user=ctx.user,
            channel=ctx.channel,
            platform=ctx.platform,
            args=args,
            raw_args=" ".join(args),
            metadata=ctx.metadata,
        )

        try:
            result = await command.handler(exec_ctx)
            self._update_cooldown(command, ctx.user_id)
            return result
        except Exception as e:
            logger.error(f"Command '{command_name}' failed: {e}", exc_info=True)
            return CommandResult.fail(f"Command failed: {str(e)}")

    def command(
        self,
        name: str,
        description: str = "",
        usage: str = "",
        aliases: Optional[List[str]] = None,
        platforms: Optional[Set[Platform]] = None,
        requires_args: bool = False,
        min_args: int = 0,
        max_args: Optional[int] = None,
        admin_only: bool = False,
        rate_limit: Optional[int] = None,
        cooldown: float = 0,
    ) -> Callable[[CommandHandler], CommandHandler]:
        """Decorator to register a command handler."""

        def decorator(handler: CommandHandler) -> CommandHandler:
            cmd = BotCommand(
                name=name,
                handler=handler,
                description=description,
                usage=usage,
                aliases=aliases or [],
                platforms=platforms or set(Platform),
                requires_args=requires_args,
                min_args=min_args,
                max_args=max_args,
                admin_only=admin_only,
                rate_limit=rate_limit,
                cooldown=cooldown,
            )
            self.register(cmd)
            return handler

        return decorator


# Global default registry
_default_registry: Optional[CommandRegistry] = None


def get_default_registry() -> CommandRegistry:
    """Get or create the default command registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = CommandRegistry()
        _register_builtin_commands(_default_registry)
    return _default_registry


def command(
    name: str,
    description: str = "",
    **kwargs: Any,
) -> Callable[[CommandHandler], CommandHandler]:
    """Shorthand decorator using the default registry."""
    return get_default_registry().command(name, description, **kwargs)


def _register_builtin_commands(registry: CommandRegistry) -> None:
    """Register built-in commands."""

    @registry.command(
        "help",
        description="Show available commands",
        usage="help [command]",
        aliases=["?", "commands"],
    )
    async def cmd_help(ctx: CommandContext) -> CommandResult:
        """Show help for commands."""
        if ctx.args:
            # Help for specific command
            cmd_name = ctx.args[0].lower()
            cmd = registry.get(cmd_name)
            if not cmd:
                return CommandResult.fail(f"Unknown command: {cmd_name}")

            message = f"**{cmd.name}**\n"
            if cmd.description:
                message += f"{cmd.description}\n"
            if cmd.usage:
                message += f"Usage: `{cmd.usage}`\n"
            if cmd.aliases:
                message += f"Aliases: {', '.join(cmd.aliases)}\n"

            return CommandResult.ok(message)

        # List all commands for this platform
        commands = registry.list_for_platform(ctx.platform)
        if not commands:
            return CommandResult.ok("No commands available.")

        lines = ["**Available Commands:**"]
        for cmd in sorted(commands, key=lambda c: c.name):
            desc = f" - {cmd.description}" if cmd.description else ""
            lines.append(f"- `{cmd.name}`{desc}")

        return CommandResult.ok("\n".join(lines))

    @registry.command(
        "status",
        description="Check Aragora system status",
        aliases=["ping", "health"],
    )
    async def cmd_status(ctx: CommandContext) -> CommandResult:
        """Check system health status."""
        import aiohttp

        api_base = ctx.metadata.get("api_base", "http://localhost:8080")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{api_base}/healthz", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        return CommandResult.ok("Aragora is online and healthy.")
                    else:
                        return CommandResult.ok(f"Aragora returned status {resp.status}")
        except asyncio.TimeoutError:
            return CommandResult.fail("Health check timed out.")
        except Exception as e:
            return CommandResult.fail(f"Health check failed: {str(e)}")

    @registry.command(
        "debate",
        description="Start a multi-agent debate",
        usage="debate <topic>",
        requires_args=True,
        min_args=1,
        cooldown=30,  # 30 second cooldown to prevent spam
    )
    async def cmd_debate(ctx: CommandContext) -> CommandResult:
        """Start a multi-agent debate on a topic via DecisionRouter."""
        topic = ctx.raw_args
        if not topic:
            return CommandResult.fail("Please provide a debate topic.")

        api_base = ctx.metadata.get("api_base", "http://localhost:8080")

        # Try to use DecisionRouter for unified routing with deduplication
        try:
            from aragora.core import (
                DecisionRequest,
                DecisionType,
                InputSource,
                RequestContext,
                ResponseChannel,
                get_decision_router,
            )

            # Map platform to InputSource
            platform_to_source = {
                Platform.DISCORD: InputSource.DISCORD,
                Platform.TEAMS: InputSource.TEAMS,
                Platform.SLACK: InputSource.SLACK,
                Platform.TELEGRAM: InputSource.TELEGRAM,
                Platform.WHATSAPP: InputSource.WHATSAPP,
            }
            source = platform_to_source.get(ctx.platform, InputSource.API)

            response_channel = ResponseChannel(
                platform=ctx.platform.value,
                channel_id=ctx.channel_id,
                user_id=ctx.user_id,
                thread_id=ctx.thread_id,
            )

            request_context = RequestContext(
                user_id=ctx.user_id,
                session_id=f"{ctx.platform.value}:{ctx.channel_id}",
            )

            request = DecisionRequest(
                content=topic,
                decision_type=DecisionType.DEBATE,
                source=source,
                response_channels=[response_channel],
                context=request_context,
            )

            router = get_decision_router()
            result = await router.route(request)

            if result.debate_id:
                return CommandResult.ok(
                    f"Debate started on: **{topic}**\n"
                    f"Debate ID: `{result.debate_id}`\n"
                    f"View at: {api_base.replace('http://', 'https://')}/debate/{result.debate_id}",
                    data={"debate_id": result.debate_id},
                )
            else:
                return CommandResult.fail(
                    result.error or "Failed to start debate via router"
                )

        except ImportError:
            logger.debug("DecisionRouter not available, falling back to HTTP")
        except Exception as e:
            logger.warning(f"DecisionRouter failed, falling back to HTTP: {e}")

        # Fallback to HTTP API if DecisionRouter unavailable
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{api_base}/api/debate",
                    json={
                        "question": topic,
                        "agents": "grok,anthropic-api,openai-api,deepseek",
                        "rounds": 3,
                        "metadata": {
                            "source": ctx.platform.value,
                            "channel_id": ctx.channel_id,
                            "user_id": ctx.user_id,
                            "thread_id": ctx.thread_id,
                        },
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    data = await resp.json()

                    if resp.status == 200 and data.get("success"):
                        debate_id = data.get("debate_id")
                        return CommandResult.ok(
                            f"Debate started on: **{topic}**\n"
                            f"Debate ID: `{debate_id}`\n"
                            f"View at: {api_base.replace('http://', 'https://')}/debate/{debate_id}",
                            data={"debate_id": debate_id},
                        )
                    else:
                        error = data.get("error", "Unknown error")
                        return CommandResult.fail(f"Failed to start debate: {error}")

        except asyncio.TimeoutError:
            return CommandResult.fail("Request timed out. Please try again.")
        except Exception as e:
            return CommandResult.fail(f"Failed to start debate: {str(e)}")

    @registry.command(
        "gauntlet",
        description="Run adversarial stress-test validation",
        usage="gauntlet <statement or decision>",
        requires_args=True,
        min_args=1,
        cooldown=60,  # 60 second cooldown
    )
    async def cmd_gauntlet(ctx: CommandContext) -> CommandResult:
        """Run gauntlet validation on a statement."""
        import aiohttp

        statement = ctx.raw_args
        if not statement:
            return CommandResult.fail("Please provide a statement to validate.")

        api_base = ctx.metadata.get("api_base", "http://localhost:8080")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{api_base}/api/gauntlet/run",
                    json={
                        "statement": statement,
                        "intensity": "medium",
                        "metadata": {
                            "source": ctx.platform.value,
                            "channel_id": ctx.channel_id,
                            "user_id": ctx.user_id,
                        },
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    data = await resp.json()

                    if resp.status == 200:
                        run_id = data.get("run_id")
                        return CommandResult.ok(
                            f"Gauntlet started for: **{statement[:100]}{'...' if len(statement) > 100 else ''}**\n"
                            f"Run ID: `{run_id}`\n"
                            f"Results will be posted when ready."
                        )
                    else:
                        error = data.get("error", "Unknown error")
                        return CommandResult.fail(f"Failed to start gauntlet: {error}")

        except asyncio.TimeoutError:
            return CommandResult.fail("Request timed out. Please try again.")
        except Exception as e:
            return CommandResult.fail(f"Failed to start gauntlet: {str(e)}")
