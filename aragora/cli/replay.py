"""
Aragora replay command - Replay stored debates.

View and replay previously recorded debate sessions.
"""

import json
import time
from pathlib import Path
from typing import Optional

from aragora.spectate.events import SpectatorEvents
from aragora.spectate.stream import SpectatorStream


def find_replay_files(directory: Optional[str] = None) -> list[Path]:
    """Find all replay files in the given directory."""
    if directory:
        base = Path(directory)
    else:
        # Check common locations
        for loc in [".aragora/replays", ".nomic/replays", "replays"]:
            if Path(loc).exists():
                base = Path(loc)
                break
        else:
            return []

    if not base.exists():
        return []

    return sorted(base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_replay(filepath: Path) -> Optional[dict]:
    """Load a replay file."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error loading replay: {e}")
        return None


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def cmd_replay(args) -> None:
    """Handle 'replay' command."""
    action = getattr(args, "action", "list")

    if action == "list":
        _list_replays(args)
    elif action == "show":
        _show_replay(args)
    elif action == "play":
        _play_replay(args)
    else:
        _list_replays(args)


def _list_replays(args) -> None:
    """List available replay files."""
    directory = getattr(args, "directory", None)
    limit = getattr(args, "limit", 10)

    replays = find_replay_files(directory)

    if not replays:
        print("\nNo replay files found.")
        print("Replays are stored when debates are recorded.")
        return

    print(f"\nRecent Debates ({len(replays)} total):")
    print("-" * 60)

    for i, filepath in enumerate(replays[:limit]):
        replay = load_replay(filepath)
        if not replay:
            continue

        # Extract metadata
        task = replay.get("task", "Unknown task")[:50]
        duration = replay.get("duration_seconds", 0)
        rounds = replay.get("rounds_used", 0)
        confidence = replay.get("confidence", 0)
        consensus = "Yes" if replay.get("consensus_reached") else "No"

        print(f"\n  [{i + 1}] {filepath.stem}")
        print(f"      Task: {task}...")
        print(f"      Rounds: {rounds} | Duration: {format_duration(duration)}")
        print(f"      Consensus: {consensus} | Confidence: {confidence:.2f}")

    if len(replays) > limit:
        print(f"\n  ... and {len(replays) - limit} more")

    print()


def _show_replay(args) -> None:
    """Show details of a specific replay."""
    replay_id = getattr(args, "id", None)
    if not replay_id:
        print("Usage: aragora replay show <id>")
        return

    # Find the replay file
    replays = find_replay_files()
    filepath = None

    for rp in replays:
        if replay_id in rp.stem or rp.stem == replay_id:
            filepath = rp
            break

    if not filepath:
        # Try as direct path
        filepath = Path(replay_id)
        if not filepath.exists():
            print(f"Replay not found: {replay_id}")
            return

    replay = load_replay(filepath)
    if not replay:
        return

    print(f"\nReplay: {filepath.stem}")
    print("=" * 60)
    print(f"\nTask: {replay.get('task', 'Unknown')}")
    print(f"Duration: {format_duration(replay.get('duration_seconds', 0))}")
    print(f"Rounds: {replay.get('rounds_used', 0)}")
    print(f"Consensus: {'Yes' if replay.get('consensus_reached') else 'No'}")
    print(f"Confidence: {replay.get('confidence', 0):.2f}")

    # Show agents
    agents = replay.get("agents", [])
    if agents:
        print(f"\nAgents: {', '.join(agents)}")

    # Show final answer
    final = replay.get("final_answer", "")
    if final:
        print("\nFinal Answer:")
        print("-" * 40)
        print(final[:500])
        if len(final) > 500:
            print("...")

    # Show message count
    messages = replay.get("messages", [])
    critiques = replay.get("critiques", [])
    print(f"\nMessages: {len(messages)} | Critiques: {len(critiques)}")
    print()


def _play_replay(args) -> None:
    """Play back a replay with spectator output."""
    replay_id = getattr(args, "id", None)
    speed = getattr(args, "speed", 1.0)

    if not replay_id:
        print("Usage: aragora replay play <id> [--speed 2.0]")
        return

    # Find the replay file
    replays = find_replay_files()
    filepath = None

    for rp in replays:
        if replay_id in rp.stem or rp.stem == replay_id:
            filepath = rp
            break

    if not filepath:
        filepath = Path(replay_id)
        if not filepath.exists():
            print(f"Replay not found: {replay_id}")
            return

    replay = load_replay(filepath)
    if not replay:
        return

    # Setup spectator for output
    spectator = SpectatorStream(enabled=True, format="plain")

    print(f"\nPlaying: {filepath.stem}")
    print(f"Speed: {speed}x")
    print("Press Ctrl+C to stop\n")
    print("=" * 60)

    try:
        # Emit debate start
        spectator.emit(
            SpectatorEvents.DEBATE_START,
            details=replay.get("task", "")[:60],
        )

        # Play through messages
        messages = replay.get("messages", [])
        for i, msg in enumerate(messages):
            # Calculate delay
            delay = 0.5 / speed

            time.sleep(delay)

            agent = msg.get("agent", "unknown")
            content = msg.get("content", "")[:80]
            round_num = msg.get("round", None)

            spectator.emit(
                SpectatorEvents.PROPOSAL,
                agent=agent,
                details=content,
                round_number=round_num,
            )

        # Play through critiques
        critiques = replay.get("critiques", [])
        for critique in critiques:
            time.sleep(0.3 / speed)

            agent = critique.get("critic", "unknown")
            content = critique.get("content", "")[:80]

            spectator.emit(
                SpectatorEvents.CRITIQUE,
                agent=agent,
                details=content,
            )

        # Emit consensus/result
        time.sleep(0.5 / speed)
        if replay.get("consensus_reached"):
            spectator.emit(
                SpectatorEvents.CONSENSUS,
                metric=replay.get("confidence", 0),
                details="Consensus reached",
            )
        else:
            spectator.emit(
                SpectatorEvents.DEBATE_END,
                details="No consensus",
            )

        print("\n" + "=" * 60)
        print("Replay complete.")

    except KeyboardInterrupt:
        print("\n\nReplay stopped.")
