"""
Spectator Mode - Real-time event streaming for debate visualization.

Provides a lightweight, synchronous, read-only event stream that visualizes
the internal state of the debate in real-time to the terminal (stdout).
Safe for TTY, encoding, and NO_COLOR environments.
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, TextIO

from .events import SpectatorEvents, EVENT_STYLES, EVENT_ASCII

logger = logging.getLogger(__name__)

# Valid event types for validation
VALID_EVENT_TYPES = frozenset([
    SpectatorEvents.DEBATE_START,
    SpectatorEvents.DEBATE_END,
    SpectatorEvents.ROUND_START,
    SpectatorEvents.ROUND_END,
    SpectatorEvents.PROPOSAL,
    SpectatorEvents.CRITIQUE,
    SpectatorEvents.REFINE,
    SpectatorEvents.VOTE,
    SpectatorEvents.JUDGE,
    SpectatorEvents.CONSENSUS,
    SpectatorEvents.CONVERGENCE,
    SpectatorEvents.CONVERGED,
    SpectatorEvents.MEMORY_RECALL,
    SpectatorEvents.SYSTEM,
    SpectatorEvents.ERROR,
])


@dataclass
class SpectatorStream:
    """
    Real-time event streamer for Aragora debates.
    Provides immediate visual feedback on the internal state of the debate.
    Safe for TTY, encoding, and NO_COLOR environments.
    """

    enabled: bool = False
    output: TextIO = field(default_factory=lambda: sys.stdout)
    format: str = "auto"  # "auto", "ansi", "plain", "json"
    show_preview: bool = True
    preview_length: int = 80

    # Runtime-detected capabilities
    _use_color: bool = field(default=False, init=False)
    _use_emoji: bool = field(default=False, init=False)

    def __post_init__(self):
        """Initialize capabilities based on environment."""
        if not self.enabled:
            return

        if self.format == "auto":
            self._detect_capabilities()
        elif self.format == "ansi":
            self._use_color = True
            self._use_emoji = True
        elif self.format == "plain":
            self._use_color = False
            self._use_emoji = False
        # json format doesn't use color/emoji

    def _detect_capabilities(self) -> None:
        """Auto-detect terminal capabilities."""
        # Check if output is a TTY
        is_tty = hasattr(self.output, 'isatty') and self.output.isatty()

        # Respect NO_COLOR standard (https://no-color.org/)
        no_color = os.environ.get("NO_COLOR") is not None

        # Check TERM for dumb terminals
        term = os.environ.get("TERM", "")
        is_dumb = term == "dumb"

        # Check encoding for emoji support
        encoding = getattr(self.output, 'encoding', 'ascii') or 'ascii'
        supports_utf8 = encoding.lower().replace('-', '') in ('utf8', 'utf16', 'utf32')

        self._use_color = is_tty and not no_color and not is_dumb
        self._use_emoji = is_tty and supports_utf8 and not is_dumb

    def emit(
        self,
        event_type: str,
        agent: str = "",
        details: str = "",
        metric: Optional[float] = None,
        round_number: Optional[int] = None,
    ) -> None:
        """
        Emit a spectator event.

        Fail-safe: any exception is caught and silently ignored
        to ensure spectating never breaks debates.
        """
        if not self.enabled:
            return

        # Validate event type
        if event_type not in VALID_EVENT_TYPES:
            logger.warning(f"Unknown spectator event type: {event_type!r}")
            # Continue anyway - don't break on unknown events

        try:
            if self.format == "json":
                self._emit_json(event_type, agent, details, metric, round_number)
            else:
                self._emit_text(event_type, agent, details, metric, round_number)
        except (KeyboardInterrupt, SystemExit):
            # Re-raise critical signals - don't swallow user interrupts
            raise
        except (MemoryError, RecursionError):
            # Re-raise resource exhaustion - indicates serious problem
            raise
        except Exception as e:
            # Swallow non-critical errors (IO, encoding, etc.) to ensure
            # spectating never breaks debates. Log at debug level.
            logger.debug(f"Spectator emit failed (non-fatal): {type(e).__name__}: {e}")

    def _emit_json(
        self,
        event_type: str,
        agent: str,
        details: str,
        metric: Optional[float],
        round_number: Optional[int],
    ) -> None:
        """Emit JSON-formatted event."""
        import json

        event = {
            "type": event_type,
            "timestamp": time.time(),
            "agent": agent or None,
            "details": details or None,
            "metric": metric,
            "round": round_number,
        }

        line = json.dumps(event)
        self._safe_print(line)

    def _emit_text(
        self,
        event_type: str,
        agent: str,
        details: str,
        metric: Optional[float],
        round_number: Optional[int],
    ) -> None:
        """Emit text-formatted event (plain or ANSI)."""
        # ANSI codes
        RESET = "\033[0m" if self._use_color else ""
        BOLD = "\033[1m" if self._use_color else ""
        DIM = "\033[2m" if self._use_color else ""

        # Get style for this event type
        icon, color_code = EVENT_STYLES.get(event_type, ("•", "\033[0m"))
        color = color_code if self._use_color else ""

        # Use ASCII fallback if no emoji support
        if not self._use_emoji:
            icon = EVENT_ASCII.get(event_type, f"[{event_type.upper()}]")

        # Build message parts
        timestamp = time.strftime("%H:%M:%S")
        parts = [f"{DIM}[{timestamp}]{RESET}"]

        if round_number is not None:
            parts.append(f"R{round_number}")

        parts.append(f"{color}{icon}")

        if agent:
            parts.append(f"{BOLD}{agent}{RESET}{color}")

        if details:
            preview = self._truncate(details) if self.show_preview else details
            if preview:
                parts.append(preview)

        if metric is not None:
            metric_str = f"({metric:.2f})" if isinstance(metric, float) else f"({metric})"
            parts.append(f"{DIM}{metric_str}{RESET}" if self._use_color else metric_str)

        if self._use_color:
            parts.append(RESET)

        line = " ".join(parts)
        self._safe_print(line)

    def _truncate(self, text: str) -> str:
        """Safely truncate content for preview."""
        if len(text) <= self.preview_length:
            return text
        return text[: self.preview_length - 3] + "..."

    def _safe_print(self, line: str) -> None:
        """Print with encoding safety."""
        try:
            print(line, file=self.output, flush=True)
        except UnicodeEncodeError:
            # Fallback to ASCII-safe version
            safe_line = line.encode('ascii', 'replace').decode('ascii')
            print(safe_line, file=self.output, flush=True)

