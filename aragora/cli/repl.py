"""
Aragora REPL command - Interactive debate mode.

Provides an interactive shell for running debates, querying memory,
and exploring the Aragora system.
"""

import asyncio
import logging
import readline
import sys
from pathlib import Path
from typing import Any, Optional, cast

from aragora.agents.base import AgentType

logger = logging.getLogger(__name__)

# REPL history file
HISTORY_FILE = Path.home() / ".aragora_history"

# Commands and their help text
COMMANDS = {
    "/help": "Show this help message",
    "/quit": "Exit the REPL",
    "/exit": "Exit the REPL",
    "/agents": "List available agents",
    "/stats": "Show debate statistics",
    "/history": "Show recent debates",
    "/clear": "Clear the screen",
    "/config": "Show current configuration",
    "/memory": "Query memory for a topic",
}


class AragoraREPL:
    """Interactive Aragora REPL."""

    def __init__(
        self,
        agents: str = "anthropic-api,openai-api",
        rounds: int = 3,
        db_path: str = "agora_memory.db",
    ):
        self.agents = agents
        self.rounds = rounds
        self.db_path = db_path
        self.running = True
        self._setup_readline()

    def _setup_readline(self) -> None:
        """Setup readline with history and completion."""
        try:
            if HISTORY_FILE.exists():
                readline.read_history_file(HISTORY_FILE)
            readline.set_history_length(1000)
        except (OSError, PermissionError) as e:
            logger.debug(f"Failed to load readline history: {e}")

        # Tab completion for commands
        def completer(text: str, state: int) -> Optional[str]:
            options = [cmd for cmd in COMMANDS if cmd.startswith(text)]
            if state < len(options):
                return options[state]
            return None

        readline.set_completer(completer)
        readline.parse_and_bind("tab: complete")

    def _save_history(self) -> None:
        """Save command history."""
        try:
            readline.write_history_file(HISTORY_FILE)
        except (OSError, PermissionError) as e:
            logger.debug(f"Failed to save readline history: {e}")

    def print_banner(self) -> None:
        """Print welcome banner."""
        print("\n" + "=" * 60)
        print("  ARAGORA REPL - Interactive Debate Mode")
        print("=" * 60)
        print(f"\nAgents: {self.agents}")
        print(f"Rounds: {self.rounds}")
        print("\nType a question to start a debate, or /help for commands.")
        print("=" * 60 + "\n")

    def print_help(self) -> None:
        """Print help message."""
        print("\nCommands:")
        for cmd, desc in COMMANDS.items():
            print(f"  {cmd:<12} {desc}")
        print("\nOr just type a question to start a debate.\n")

    def handle_command(self, cmd: str) -> bool:
        """Handle a REPL command. Returns True to continue, False to exit."""
        cmd = cmd.strip().lower()

        if cmd in ("/quit", "/exit"):
            print("Goodbye!")
            return False

        elif cmd == "/help":
            self.print_help()

        elif cmd == "/agents":
            self._list_agents()

        elif cmd == "/stats":
            self._show_stats()

        elif cmd == "/history":
            self._show_history()

        elif cmd == "/clear":
            print("\033[2J\033[H", end="")  # ANSI clear screen

        elif cmd == "/config":
            self._show_config()

        elif cmd.startswith("/memory "):
            query = cmd[8:].strip()
            self._query_memory(query)

        elif cmd == "/memory":
            print("Usage: /memory <query>")

        else:
            print(f"Unknown command: {cmd}")
            print("Type /help for available commands.")

        return True

    def _list_agents(self) -> None:
        """List available agents."""
        print("\nAvailable agents:")
        agents = [
            ("anthropic-api", "Anthropic Claude API"),
            ("openai-api", "OpenAI GPT API"),
            ("gemini", "Google Gemini API"),
            ("grok", "xAI Grok API"),
            ("openrouter", "OpenRouter (multi-model)"),
            ("claude", "Claude CLI (local)"),
            ("codex", "Codex CLI (local)"),
        ]
        for name, desc in agents:
            print(f"  {name:<15} {desc}")
        print()

    def _show_stats(self) -> None:
        """Show debate statistics."""
        try:
            from aragora.memory.store import CritiqueStore

            store = CritiqueStore(db_path=self.db_path)
            stats = store.get_stats()

            print("\nDebate Statistics:")
            print(f"  Total debates: {stats.get('total_debates', 0)}")
            print(f"  Total critiques: {stats.get('total_critiques', 0)}")
            print(f"  Avg confidence: {stats.get('avg_confidence', 0):.2f}")
            print()
        except Exception as e:
            print(f"Could not load stats: {e}")

    def _show_history(self) -> None:
        """Show recent debate history."""
        try:
            from aragora.memory.store import CritiqueStore

            store = CritiqueStore(db_path=self.db_path)
            # Use get_recent() which returns recent critiques - there's no get_recent_debates
            recent = store.get_recent(limit=5)

            if not recent:
                print("\nNo recent critiques found.")
                return

            print("\nRecent critiques:")
            for critique in recent:
                agent = critique.agent
                target = critique.target_agent
                severity = critique.severity
                print(f"  - {agent} -> {target} (severity: {severity:.2f})")
            print()
        except Exception as e:
            print(f"Could not load history: {e}")

    def _show_config(self) -> None:
        """Show current configuration."""
        print("\nCurrent configuration:")
        print(f"  Agents: {self.agents}")
        print(f"  Rounds: {self.rounds}")
        print(f"  Database: {self.db_path}")
        print()

    def _query_memory(self, query: str) -> None:
        """Query memory for related content."""
        try:
            from aragora.memory.store import CritiqueStore

            store = CritiqueStore(db_path=self.db_path)
            # CritiqueStore doesn't have a search method - use get_relevant patterns instead
            # which searches for patterns by issue type
            patterns = store.get_relevant(issue_type=query, limit=5)

            if not patterns:
                print(f"\nNo patterns found for: {query}")
                return

            print(f"\nPatterns related to '{query}':")
            for p in patterns:
                issue_text = p.issue_text[:80] if p.issue_text else ""
                print(f"  - [{p.issue_type}] {issue_text}...")
            print()
        except Exception as e:
            print(f"Memory query failed: {e}")

    async def run_debate(self, task: str) -> None:
        """Run a debate on the given task."""
        print(f"\nStarting debate: {task}")
        print("-" * 40)

        try:
            from aragora.agents.base import create_agent
            from aragora.debate.orchestrator import Arena, DebateProtocol
            from aragora.core import Agent, Environment
            from aragora.spectate.stream import SpectatorStream

            # Parse agents
            agent_specs = self.agents.split(",")
            roles = ["proposer", "critic", "synthesizer"]
            agents: list[Agent] = []
            for i, agent_type_str in enumerate(agent_specs):
                role = roles[i % len(roles)]
                agent = create_agent(
                    model_type=cast(AgentType, agent_type_str.strip()),
                    name=f"{agent_type_str}_{role}",
                    role=role,
                )
                agents.append(agent)

            # Setup
            env = Environment(task=task)
            protocol = DebateProtocol(rounds=self.rounds, consensus="majority")
            spectator = SpectatorStream(enabled=True, format="plain")

            arena = Arena(env, agents, protocol, spectator=spectator)
            result = await arena.run()

            print("\n" + "=" * 40)
            print("RESULT")
            print("=" * 40)
            if result.consensus_reached:
                print(f"Consensus: YES (confidence: {result.confidence:.2f})")
            else:
                print("Consensus: NO")
            print(f"\nFinal Answer:\n{result.final_answer}")
            print()

        except KeyboardInterrupt:
            print("\n\nDebate interrupted.")
        except Exception as e:
            print(f"\nDebate failed: {e}")

    def run(self) -> None:
        """Run the REPL loop."""
        self.print_banner()

        while self.running:
            try:
                line = input("aragora> ").strip()

                if not line:
                    continue

                if line.startswith("/"):
                    self.running = self.handle_command(line)
                else:
                    # Run debate
                    asyncio.run(self.run_debate(line))

            except KeyboardInterrupt:
                print("\n\nUse /quit to exit.")
            except EOFError:
                print("\nGoodbye!")
                break

        self._save_history()


def cmd_repl(args) -> None:
    """Handle 'repl' command."""
    repl = AragoraREPL(
        agents=getattr(args, "agents", "anthropic-api,openai-api"),
        rounds=getattr(args, "rounds", 3),
        db_path=getattr(args, "db", "agora_memory.db"),
    )
    repl.run()
