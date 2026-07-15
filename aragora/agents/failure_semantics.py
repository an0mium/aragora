"""Shared semantics for recognizing agent FAILURE placeholders (issue #9303).

Several layers emit human-friendly placeholder text when an agent errors out
(autonomic recovery notes, chaos-theater whimsy, timeout stubs). Those strings
are *about* a failure — they are never evidence, never an answer, and must
never support a verdict. This module is the single source of truth for
recognizing them so the CLI failure gate, receipt minting, and memory
injection all agree on what "the agent said nothing" means.

Failure modes this guards against (measured live in the 2026-07-15 dim-8
instrument run):
- a debate where every agent errored minted a PASS receipt at 80% confidence
  whose sole "response" was a chaos-theater placeholder;
- placeholder text from a failed attempt was re-injected into the next
  attempt as institutional knowledge.
"""

from __future__ import annotations

from typing import Any, Iterable

#: Substrings that mark a response as an error placeholder rather than
#: content. Mirrors the autonomic/chaos-theater emitters; keep lowercase.
AGENT_FAILURE_RESPONSE_MARKERS: tuple[str, ...] = (
    "[system: agent ",
    "[error generating proposal:",
    "[no proposals available",
    "agent timed out",
    "connection failed",
    "encountered an error",
    "encountered an unexpected situation",
    "something went wrong with",
    "needs to restart their thought process",
    "tripped over an edge case",
    "experienced a minor cognitive hiccup",
    "got confused and needs to recalibrate",
    "a wild bug appeared",
    "has achieved unexpected behavior",
    "fatal exception in",
    "error 418:",
)

#: UNMISTAKABLE placeholder signatures — never appear in real answers.
_STRONG_MARKERS: tuple[str, ...] = (
    "[system: agent ",
    "[error generating proposal:",
    "[no proposals available",
    "needs to restart their thought process",
    "tripped over an edge case",
    "experienced a minor cognitive hiccup",
    "got confused and needs to recalibrate",
    "a wild bug appeared",
    "has achieved unexpected behavior",
    "error 418:",
)

#: Generic phrases that also occur in real technical content ("the outage
#: happened because connection failed"). These only classify as placeholders
#: when the text is a one-liner (openai #9306 r2 [P2]).
_WEAK_MARKERS: tuple[str, ...] = (
    "agent timed out",
    "connection failed",
    "encountered an error",
    "encountered an unexpected situation",
    "something went wrong with",
    "fatal exception in",
)

_WEAK_MARKER_MAX_CHARS = 120


#: Placeholders are short, single-sentence stubs. A response longer than this
#: is substantive content even when it QUOTES an error string (a review
#: discussing "connection failed" is an answer, not a failure).
_MAX_PLACEHOLDER_CHARS = 500


def looks_like_agent_failure_response(text: Any) -> bool:
    """True when ``text`` is empty or reads as an error placeholder.

    Marker matching applies only to short texts (< _MAX_PLACEHOLDER_CHARS):
    real placeholders are one-liners, and the length bound prevents
    misclassifying substantive answers that merely mention an error.
    """
    if text is None:
        return True
    lowered = str(text).strip().lower()
    if not lowered:
        return True
    if len(lowered) >= _MAX_PLACEHOLDER_CHARS:
        return False
    if any(marker in lowered for marker in _STRONG_MARKERS):
        return True
    # Weak/generic phrases classify only one-liners: a concise real answer
    # ABOUT an outage must never mint NO_EVIDENCE.
    if len(lowered) <= _WEAK_MARKER_MAX_CHARS:
        return any(marker in lowered for marker in _WEAK_MARKERS)
    return False


def all_responses_are_failures(texts: Iterable[Any]) -> bool:
    """True when there is at least one response and EVERY one is a failure
    placeholder (or empty). An empty iterable returns True as well: zero
    responses is zero evidence."""
    for text in texts:
        if not looks_like_agent_failure_response(text):
            return False
    return True


__all__ = [
    "AGENT_FAILURE_RESPONSE_MARKERS",
    "looks_like_agent_failure_response",
    "all_responses_are_failures",
]
