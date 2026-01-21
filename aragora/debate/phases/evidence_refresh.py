"""
Evidence refresh module for debate rounds.

Handles refreshing evidence based on claims made during debate rounds.
This module is extracted from debate_rounds.py for better modularity.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:
    from aragora.core import Critique
    from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)

# Timeout for async callbacks (evidence refresh can be slow)
DEFAULT_CALLBACK_TIMEOUT = 30.0


async def _with_callback_timeout(
    coro,
    timeout: float = DEFAULT_CALLBACK_TIMEOUT,
    default=None,
):
    """Execute coroutine with timeout, returning default on timeout.

    This prevents debates from stalling indefinitely when callbacks
    like evidence refresh hang.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Callback timed out after {timeout}s, using default: {default}")
        return default


class EvidenceRefresher:
    """
    Refreshes evidence based on claims made in proposals and critiques.

    This class extracts factual claims from proposals and critiques,
    then searches for new evidence to support or refute those claims.
    The fresh evidence is injected into the context for the revision phase.

    Usage:
        refresher = EvidenceRefresher(
            refresh_callback=arena._refresh_evidence,
            hooks=arena.hooks,
            notify_spectator=arena._notify_spectator,
        )
        await refresher.refresh_for_round(ctx, round_num, partial_critiques)
    """

    def __init__(
        self,
        refresh_callback: Optional[Callable] = None,
        hooks: Optional[dict] = None,
        notify_spectator: Optional[Callable] = None,
        timeout: float = DEFAULT_CALLBACK_TIMEOUT,
    ):
        """
        Initialize the evidence refresher.

        Args:
            refresh_callback: Async callback for refreshing evidence.
                              Signature: (text: str, ctx: DebateContext, round: int) -> int
            hooks: Dictionary of event hooks
            notify_spectator: Callback for spectator notifications
            timeout: Timeout in seconds for refresh operations
        """
        self._refresh_evidence = refresh_callback
        self.hooks = hooks or {}
        self._notify_spectator = notify_spectator
        self._timeout = timeout

    async def refresh_for_round(
        self,
        ctx: "DebateContext",
        round_num: int,
        partial_critiques: List["Critique"],
    ) -> int:
        """
        Refresh evidence based on claims made in the current round.

        Extracts factual claims from proposals and critiques, then
        searches for new evidence to support or refute those claims.
        The fresh evidence is injected into the context for the revision phase.

        Args:
            ctx: The DebateContext with proposals and critiques
            round_num: Current round number
            partial_critiques: List of critiques from current/recent rounds

        Returns:
            Number of new evidence snippets found, or 0 if refresh was skipped
        """
        if not self._refresh_evidence:
            return 0

        # Only refresh evidence every other round to avoid API overload
        if round_num % 2 == 0:
            return 0

        try:
            # Collect text from proposals and recent critiques
            texts_to_analyze = []

            # Add proposal content
            for agent_name, proposal in ctx.proposals.items():
                if proposal:
                    texts_to_analyze.append(proposal[:2000])  # Limit per proposal

            # Add recent critique content
            for critique in partial_critiques[-5:]:  # Last 5 critiques
                critique_text = (
                    critique.to_prompt() if hasattr(critique, "to_prompt") else str(critique)
                )
                texts_to_analyze.append(critique_text[:1000])

            if not texts_to_analyze:
                return 0

            combined_text = "\n".join(texts_to_analyze)

            # Call the refresh callback with timeout protection
            refreshed = await _with_callback_timeout(
                self._refresh_evidence(combined_text, ctx, round_num),
                timeout=self._timeout,
                default=0,  # Return 0 snippets on timeout
            )

            if refreshed:
                logger.info(f"evidence_refreshed round={round_num} new_snippets={refreshed}")

                # Notify spectator
                if self._notify_spectator:
                    self._notify_spectator(
                        "evidence",
                        details=f"Refreshed evidence: {refreshed} new sources",
                        metric=refreshed,
                        agent="system",
                    )

                # Emit evidence refresh event
                if "on_evidence_refresh" in self.hooks:
                    self.hooks["on_evidence_refresh"](
                        round_num=round_num,
                        new_snippets=refreshed,
                    )

            return refreshed or 0

        except Exception as e:
            logger.warning(f"Evidence refresh failed for round {round_num}: {e}")
            return 0
