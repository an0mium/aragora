"""
Consensus verification logic for claim validation during voting.

This module extracts verification-related logic from ConsensusPhase,
providing:
- Verification bonus application to vote counts
- ELO updates based on verification results
- Event emission for verification results
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.debate.context import DebateContext

logger = logging.getLogger(__name__)


class ConsensusVerifier:
    """Handles claim verification during consensus phase.

    Applies verification bonuses to vote counts based on formally
    verified claims in proposals.

    Usage:
        verifier = ConsensusVerifier(
            protocol=protocol,
            elo_system=elo_system,
            verify_claims=verify_claims_callback,
            extract_debate_domain=extract_domain_callback,
        )

        vote_counts = await verifier.apply_verification_bonuses(
            ctx, vote_counts, proposals, choice_mapping
        )
    """

    def __init__(
        self,
        protocol: Any = None,
        elo_system: Any = None,
        verify_claims: Optional[Callable] = None,
        extract_debate_domain: Optional[Callable] = None,
    ):
        """Initialize the consensus verifier.

        Args:
            protocol: DebateProtocol with verification settings
            elo_system: ELO system for rating updates
            verify_claims: Callback to verify claims in a proposal
            extract_debate_domain: Callback to extract debate domain
        """
        self.protocol = protocol
        self.elo_system = elo_system
        self._verify_claims = verify_claims
        self._extract_debate_domain = extract_debate_domain

    async def apply_verification_bonuses(
        self,
        ctx: "DebateContext",
        vote_counts: Counter,
        proposals: dict[str, str],
        choice_mapping: dict[str, str],
    ) -> Counter:
        """Apply verification bonuses to vote counts for verified proposals.

        When verify_claims_during_consensus is enabled in the protocol,
        proposals with verified claims get a weight bonus. Results are
        stored in ctx.result.verification_results for feedback loop.

        Args:
            ctx: DebateContext to store verification results
            vote_counts: Current vote counts by choice
            proposals: Dict of agent_name -> proposal_text
            choice_mapping: Mapping from vote choice to canonical form

        Returns:
            Updated vote counts with verification bonuses applied
        """
        if not self.protocol or not getattr(self.protocol, "verify_claims_during_consensus", False):
            return vote_counts

        if not self._verify_claims:
            return vote_counts

        verification_bonus = getattr(self.protocol, "verification_weight_bonus", 0.2)
        verification_timeout = getattr(self.protocol, "verification_timeout_seconds", 5.0)
        result = ctx.result

        for agent_name, proposal_text in proposals.items():
            # Map agent name to canonical choice
            canonical = choice_mapping.get(agent_name, agent_name)
            if canonical not in vote_counts:
                continue

            try:
                # Verify top claims in the proposal (async with timeout)
                verification_result = await asyncio.wait_for(
                    self._verify_claims(proposal_text, limit=2), timeout=verification_timeout
                )

                # Handle both dict and int return types for backward compat
                if isinstance(verification_result, dict):
                    verified_count = verification_result.get("verified", 0)
                    disproven_count = verification_result.get("disproven", 0)
                else:
                    # Legacy: callback returns int
                    verified_count = verification_result or 0
                    disproven_count = 0

                # Store verification counts for feedback loop
                # Note: verification_results accepts both int (legacy) and dict (new format)
                if hasattr(result, "verification_results"):
                    result.verification_results[agent_name] = {  # type: ignore[assignment]
                        "verified": verified_count,
                        "disproven": disproven_count,
                    }

                bonus = 0.0
                if verified_count > 0:
                    # Apply bonus: boost votes for this proposal
                    current_count = vote_counts[canonical]
                    bonus = current_count * verification_bonus * verified_count
                    vote_counts[canonical] = current_count + bonus  # type: ignore[assignment]

                    # Store bonus for feedback loop
                    if hasattr(result, "verification_bonuses"):
                        result.verification_bonuses[agent_name] = bonus

                    logger.info(
                        f"verification_bonus agent={agent_name} "
                        f"verified={verified_count} bonus={bonus:.2f}"
                    )

                # Emit verification result event
                self._emit_verification_event(ctx, agent_name, verified_count or 0, bonus)
            except asyncio.TimeoutError:
                logger.debug(f"verification_timeout agent={agent_name}")
                if hasattr(result, "verification_results"):
                    result.verification_results[agent_name] = -1  # Timeout indicator
                self._emit_verification_event(ctx, agent_name, -1, 0.0, timeout=True)
            except Exception as e:
                logger.debug(f"verification_error agent={agent_name} error={e}")

        # Update ELO based on verification results
        await self._update_elo_from_verification(ctx)

        return vote_counts

    async def _update_elo_from_verification(self, ctx: "DebateContext") -> None:
        """Update agent ELO ratings based on verification results.

        When claims are formally verified, the authoring agent's ELO is adjusted:
        - Verified claims: ELO boost (quality reasoning)
        - Disproven claims: ELO penalty (flawed reasoning)
        - Timeouts/errors: No change

        Args:
            ctx: DebateContext with verification_results
        """
        if not self.elo_system:
            return

        result = ctx.result
        if not hasattr(result, "verification_results") or not result.verification_results:
            return

        # Extract domain from context
        domain = "general"
        if self._extract_debate_domain:
            try:
                domain = self._extract_debate_domain()
            except Exception as e:
                logger.debug(f"Failed to extract debate domain: {e}")

        # Process verification results for each agent
        for agent_name, verification_data in result.verification_results.items():
            # Handle both dict and int formats for backward compatibility
            if isinstance(verification_data, dict):
                verified_count = verification_data.get("verified", 0)
                disproven_count = verification_data.get("disproven", 0)
            else:
                # Legacy format: int value
                # Skip timeouts (indicated by -1) and errors
                if verification_data < 0:
                    continue
                verified_count = verification_data
                disproven_count = 0

            # Skip if nothing to report
            if verified_count == 0 and disproven_count == 0:
                continue

            try:
                change = self.elo_system.update_from_verification(
                    agent_name=agent_name,
                    domain=domain,
                    verified_count=verified_count,
                    disproven_count=disproven_count,
                )

                if change != 0:
                    logger.debug(
                        f"verification_elo_applied agent={agent_name} "
                        f"verified={verified_count} disproven={disproven_count} "
                        f"change={change:.1f}"
                    )
            except Exception as e:
                logger.debug(f"verification_elo_error agent={agent_name} error={e}")

    def _emit_verification_event(
        self,
        ctx: "DebateContext",
        agent_name: str,
        verified_count: int,
        bonus: float,
        timeout: bool = False,
    ) -> None:
        """Emit CLAIM_VERIFICATION_RESULT event to WebSocket.

        Args:
            ctx: DebateContext with event_emitter
            agent_name: Name of agent whose proposal was verified
            verified_count: Number of verified claims (-1 if timeout)
            bonus: Vote bonus applied
            timeout: Whether verification timed out
        """
        if not ctx.event_emitter:
            return

        try:
            from aragora.server.stream import StreamEvent, StreamEventType

            ctx.event_emitter.emit(
                StreamEvent(
                    type=StreamEventType.CLAIM_VERIFICATION_RESULT,
                    loop_id=ctx.loop_id,
                    agent=agent_name,
                    data={
                        "agent": agent_name,
                        "verified_count": verified_count,
                        "bonus_applied": bonus,
                        "timeout": timeout,
                        "debate_id": ctx.debate_id,
                    },
                )
            )
        except Exception as e:
            logger.debug(f"verification_event_error: {e}")


__all__ = ["ConsensusVerifier"]
