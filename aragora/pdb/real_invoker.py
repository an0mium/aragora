"""Real :class:`ProviderInvoker` wiring for PDB Mode 3 Protocol B.

This module ships the first non-mock invoker: it dispatches findings,
critique, and synthesis calls to the real class-based agent
instances in :mod:`aragora.agents.api_agents` and coerces their
free-form text output into the structured response shapes the
:mod:`aragora.pdb.protocol` executor consumes.

Phase A scope (what this PR delivers):

- :data:`FAMILY_CLAUDE` → :class:`AnthropicAPIAgent`
- :data:`FAMILY_GPT` → :class:`OpenAIAPIAgent`
- every other family (gemini / grok / deepseek / kimi / qwen / mistral)
  is marked **unavailable** and raises :class:`ProviderUnavailableError`
  when the executor tries to invoke it. The executor treats the raise
  as a per-slot degrade per its existing contract.

Phase B (intentionally deferred): wiring the heterodox slots so the
full eight-slot panel runs on real providers, rate-limit retries,
caching, and prompt-engineering refinement. See the mission brief.

The class holds a pre-initialized agent per family (not per call) so
each invocation reuses the underlying HTTP session setup the base
:class:`APIAgent` constructs.

Token + cost tracking. Every call wraps the underlying agent in
:func:`time.monotonic` for latency and reads ``last_tokens_in`` /
``last_tokens_out`` off the agent after ``generate`` returns, then
looks up a conservative per-model rate to compute ``cost_usd``.
Unknown models log a warning and record ``0.0`` so the budget layer
still receives a valid float.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol as _TypingProtocol, Sequence

from aragora.pdb.panel_config import PDBPanelSlot
from aragora.pdb.protocol import (
    SlotCritiqueResponse,
    SlotFindingsResponse,
    SynthesisResponse,
)
from aragora.pdb.response_parser import (
    parse_critique_response,
    parse_findings_response,
    parse_synthesis_response,
)
from aragora.review.builder import PanelVote
from aragora.review.protocol import DissentPosition
from aragora.swarm.pr_review_protocol import PRReviewBinding, PRReviewFinding

__all__ = [
    "FAMILY_CLAUDE",
    "FAMILY_GPT",
    "WIRED_FAMILIES",
    "HETERODOX_FAMILIES",
    "ProviderUnavailableError",
    "RealProviderInvoker",
    "estimate_cost_usd",
]


logger = logging.getLogger(__name__)

FAMILY_CLAUDE = "claude"
FAMILY_GPT = "gpt"

WIRED_FAMILIES: frozenset[str] = frozenset({FAMILY_CLAUDE, FAMILY_GPT})
"""Families the Phase A invoker actually dispatches to real agents."""

HETERODOX_FAMILIES: frozenset[str] = frozenset(
    {"gemini", "grok", "deepseek", "kimi", "qwen", "mistral"}
)
"""Families the Phase A invoker treats as unavailable by design."""


# Conservative per-model rates (USD per 1M tokens). Kept minimal and
# self-contained rather than reusing the full billing pipeline so the
# invoker has a predictable, test-friendly cost surface.
_PRICE_PER_MTOK: Mapping[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus-4": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4-5": (0.80, 4.00),
    # OpenAI
    "gpt-5.4": (2.50, 10.00),
    "gpt-5.4-pro": (5.00, 20.00),
    "gpt-5.3": (2.50, 10.00),
    "gpt-5.3-chat-latest": (2.50, 10.00),
    "gpt-5.3-codex": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


def estimate_cost_usd(
    *,
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> float:
    """Return a best-effort USD cost for a model call.

    Unknown models log once and return ``0.0`` so the budget layer
    never crashes on an unrecognised response. Both token counts are
    normalised to ``max(0, value)`` so a malformed usage dict cannot
    produce negative cost.
    """
    tokens_in = max(0, int(tokens_in or 0))
    tokens_out = max(0, int(tokens_out or 0))
    rates = _PRICE_PER_MTOK.get(model)
    if rates is None:
        # Strip provider prefixes (``anthropic/...`` etc.) and try again.
        stripped = model.split("/", 1)[-1] if model else model
        rates = _PRICE_PER_MTOK.get(stripped)
    if rates is None:
        logger.warning(
            "pdb.real_invoker: no price entry for model %r; recording cost=0.0",
            model,
        )
        return 0.0
    in_price, out_price = rates
    cost = (tokens_in / 1_000_000.0) * in_price + (tokens_out / 1_000_000.0) * out_price
    return round(cost, 6)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProviderUnavailableError(RuntimeError):
    """Raised when a slot's family has no wired provider.

    The executor in :mod:`aragora.pdb.protocol` catches this from
    per-slot phases and records the slot in ``degrade_reasons`` /
    ``missing_slots`` without tearing down the whole brief — as long
    as the slot is optional and the heterogeneity floor still holds.
    """

    def __init__(self, slot_id: str, family: str, reason: str) -> None:
        self.slot_id = slot_id
        self.family = family
        self.reason = reason
        super().__init__(f"slot {slot_id!r} (family {family!r}) is unavailable: {reason}")


# ---------------------------------------------------------------------------
# Minimal structural type for the agents we consume
# ---------------------------------------------------------------------------


class _AgentLike(_TypingProtocol):
    """The tiny surface of :class:`APIAgent` the invoker actually touches.

    Narrowed to make tests trivial: any object with ``model``,
    ``last_tokens_in``, ``last_tokens_out`` and an async ``generate``
    method satisfies the contract.
    """

    model: str
    last_tokens_in: int
    last_tokens_out: int

    async def generate(self, prompt: str, context: Any = None, **kwargs: Any) -> str: ...


# ---------------------------------------------------------------------------
# Invoker
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _AgentCallResult:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cost_usd: float


class RealProviderInvoker:
    """Phase A :class:`aragora.pdb.protocol.ProviderInvoker` implementation.

    Parameters
    ----------
    claude:
        An :class:`AnthropicAPIAgent`-like object, or ``None`` if the
        Anthropic core slot should be treated as unavailable.
    gpt:
        An :class:`OpenAIAPIAgent`-like object, or ``None`` if the
        OpenAI core slot should be treated as unavailable.
    unavailable_slots:
        Optional override marking specific slot ids as unavailable
        regardless of family routing (used by the factory when a
        family-specific env var is absent).
    """

    def __init__(
        self,
        *,
        claude: _AgentLike | None = None,
        gpt: _AgentLike | None = None,
        unavailable_slots: frozenset[str] = frozenset(),
    ) -> None:
        self._agents: dict[str, _AgentLike | None] = {
            FAMILY_CLAUDE: claude,
            FAMILY_GPT: gpt,
        }
        self._unavailable_slots = frozenset(unavailable_slots)

    # ------------------------------------------------------------------
    # Public surface — ProviderInvoker protocol methods
    # ------------------------------------------------------------------

    def findings(
        self,
        *,
        slot: PDBPanelSlot,
        provider: str,
        prompt: str,
        binding: PRReviewBinding,
    ) -> SlotFindingsResponse:
        """Run the findings phase for a single slot."""
        self._assert_available(slot)
        agent = self._require_agent(slot)
        call = self._call_agent(agent, prompt)
        parsed = parse_findings_response(call.text, slot_id=slot.slot_id)

        top_findings = _to_findings_tuple(parsed["top_findings"], slot_id=slot.slot_id)
        confidence = float(parsed["confidence"])
        # If the response failed to parse, hold confidence at 0 so the
        # builder's weighted policy cannot treat garbage as signal.
        if not parsed["parsed"]:
            confidence = 0.0

        return SlotFindingsResponse(
            slot_id=slot.slot_id,
            provider=provider,
            model=call.model,
            position=parsed["position"],
            confidence=confidence,
            summary=parsed["summary"],
            top_findings=top_findings,
            contested_finding_ids=tuple(parsed["contested_finding_ids"]),
            reason=parsed["reason"] or parsed["summary"],
            latency_ms=call.latency_ms,
            cost_usd=call.cost_usd,
        )

    def critique(
        self,
        *,
        slot: PDBPanelSlot,
        provider: str,
        prompt: str,
        peer_findings: Mapping[str, SlotFindingsResponse],
        binding: PRReviewBinding,
    ) -> SlotCritiqueResponse:
        """Run the critique phase for a single slot."""
        self._assert_available(slot)
        agent = self._require_agent(slot)
        call = self._call_agent(agent, prompt)
        parsed = parse_critique_response(call.text, slot_id=slot.slot_id)

        confidence = float(parsed["confidence"])
        if not parsed["parsed"]:
            confidence = 0.0

        return SlotCritiqueResponse(
            slot_id=slot.slot_id,
            provider=provider,
            position=parsed["position"],
            confidence=confidence,
            reason=parsed["reason"],
            agrees_with=tuple(parsed["agrees_with"]),
            disagrees_with=tuple(parsed["disagrees_with"]),
            contested_finding_ids=tuple(parsed["contested_finding_ids"]),
            latency_ms=call.latency_ms,
            cost_usd=call.cost_usd,
        )

    def synthesize(
        self,
        *,
        synthesizer_slot: PDBPanelSlot,
        provider: str,
        prompt: str,
        votes: Sequence[PanelVote],
        binding: PRReviewBinding,
    ) -> SynthesisResponse:
        """Run the synthesis pass."""
        self._assert_available(synthesizer_slot)
        agent = self._require_agent(synthesizer_slot)
        call = self._call_agent(agent, prompt)
        parsed = parse_synthesis_response(call.text)

        # The synthesizer's position for :class:`SynthesisResponse` is
        # a derived view: we pick the majority :class:`DissentPosition`
        # across the panel votes so a malformed synthesis cannot flip
        # the brief's recommendation. The builder still computes the
        # final :class:`Recommendation` deterministically from the
        # votes; this field is informational only.
        majority_position = _majority_position(votes)
        # We don't ask the synthesizer for a confidence score in the
        # prompt; use a mid-range default when parsing succeeded so
        # the synthesizer's SYNTHESIZER-role vote does not dominate
        # the builder's weighted aggregation. A parse failure zeros
        # the confidence so the builder ignores it.
        confidence = 0.6 if parsed["parsed"] else 0.0

        return SynthesisResponse(
            slot_id=synthesizer_slot.slot_id,
            provider=provider,
            model=call.model,
            top_line=parsed["top_line"],
            validation_summary=parsed["validation_summary"],
            position=majority_position,
            confidence=confidence,
            preserved_dissent=tuple(parsed["preserved_dissent"]),
            latency_ms=call.latency_ms,
            cost_usd=call.cost_usd,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assert_available(self, slot: PDBPanelSlot) -> None:
        """Raise :class:`ProviderUnavailableError` if ``slot`` is not wired."""
        if slot.slot_id in self._unavailable_slots:
            raise ProviderUnavailableError(
                slot_id=slot.slot_id,
                family=slot.family,
                reason="slot marked unavailable by invoker factory (no API key)",
            )
        if slot.family in HETERODOX_FAMILIES:
            raise ProviderUnavailableError(
                slot_id=slot.slot_id,
                family=slot.family,
                reason="heterodox family not wired in Phase A (Claude + GPT only)",
            )
        if slot.family not in WIRED_FAMILIES:
            raise ProviderUnavailableError(
                slot_id=slot.slot_id,
                family=slot.family,
                reason=f"family {slot.family!r} has no Phase A wiring",
            )
        if self._agents.get(slot.family) is None:
            raise ProviderUnavailableError(
                slot_id=slot.slot_id,
                family=slot.family,
                reason=f"no agent instance registered for family {slot.family!r}",
            )

    def _require_agent(self, slot: PDBPanelSlot) -> _AgentLike:
        agent = self._agents.get(slot.family)
        if agent is None:
            # _assert_available runs before this, so this is defensive.
            raise ProviderUnavailableError(
                slot_id=slot.slot_id,
                family=slot.family,
                reason="agent reference is None at call time",
            )
        return agent

    def _call_agent(self, agent: _AgentLike, prompt: str) -> _AgentCallResult:
        """Invoke ``agent.generate`` synchronously with latency + cost tracking."""
        start = time.monotonic()
        try:
            text = _run_sync(agent.generate(prompt))
        except Exception:
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.warning(
                "pdb.real_invoker: agent call raised after %dms (model=%r)",
                latency_ms,
                getattr(agent, "model", "unknown"),
            )
            raise
        latency_ms = int((time.monotonic() - start) * 1000)
        tokens_in = int(getattr(agent, "last_tokens_in", 0) or 0)
        tokens_out = int(getattr(agent, "last_tokens_out", 0) or 0)
        model = str(getattr(agent, "model", "unknown"))
        cost = estimate_cost_usd(model=model, tokens_in=tokens_in, tokens_out=tokens_out)
        return _AgentCallResult(
            text=text or "",
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=cost,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _run_sync(coro: Any) -> str:
    """Run ``coro`` on a fresh event loop and return its string result.

    The worker's :func:`asyncio.to_thread` already ensures this helper
    runs on a thread distinct from the worker's event loop, so
    :func:`asyncio.run` is safe here. Returning an empty string on
    ``None`` keeps the downstream parsers happy.
    """
    result = asyncio.run(coro)
    if result is None:
        return ""
    return str(result)


def _to_findings_tuple(
    findings: Sequence[Mapping[str, Any]],
    *,
    slot_id: str,
) -> tuple[PRReviewFinding, ...]:
    """Build :class:`PRReviewFinding` instances from parsed finding dicts."""
    out: list[PRReviewFinding] = []
    for idx, raw in enumerate(findings):
        finding_id = str(raw.get("finding_id") or f"{slot_id}-F{idx + 1}")
        out.append(
            PRReviewFinding(
                finding_id=finding_id,
                category=str(raw.get("category") or "general"),
                severity=str(raw.get("severity") or "medium"),
                summary=str(raw.get("summary") or ""),
                evidence=list(raw.get("evidence") or []),
                source=f"slot:{slot_id}",
            )
        )
    return tuple(out)


def _majority_position(votes: Sequence[PanelVote]) -> DissentPosition:
    """Return the most-common :class:`DissentPosition` across votes.

    On a tie we return ``DEFER`` so the synthesizer's informational
    position reflects the panel-level uncertainty rather than an
    accidental approve.
    """
    if not votes:
        return DissentPosition.DEFER
    counts: dict[DissentPosition, int] = {}
    for vote in votes:
        counts[vote.position] = counts.get(vote.position, 0) + 1
    top = max(counts.values())
    winners = [pos for pos, n in counts.items() if n == top]
    if len(winners) == 1:
        return winners[0]
    return DissentPosition.DEFER
