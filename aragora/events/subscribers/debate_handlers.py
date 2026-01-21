"""
Debate and ELO event handler mixin for CrossSubscriberManager.

Handles event flow related to debates:
- ELO → Debate: Performance updates affect team selection
- Calibration → Agent: Confidence adjustments
- Culture → Debate: Organizational patterns
- Staleness → Debate: Knowledge freshness checks
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from aragora.events.types import StreamEvent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DebateHandlersMixin:
    """Mixin providing debate and ELO event handlers."""

    # These will be set by the main class
    stats: dict
    retry_handler: Any
    circuit_breaker: Any
    _culture_cache: dict
    _culture_cache_ttl: float
    _staleness_debounce: dict
    _staleness_debounce_seconds: float

    def _is_km_handler_enabled(self, handler_name: str) -> bool:
        """Check if a handler is enabled (defined in main class)."""
        raise NotImplementedError

    def _handle_elo_to_debate(self, event: StreamEvent) -> None:
        """Handle ELO → Debate events.

        Updates team selection weights based on performance changes.
        """
        try:
            from aragora.observability.metrics.slo import check_and_record_slo
        except ImportError:
            check_and_record_slo = None

        start = time.time()

        try:
            data = event.data
            agent_name = data.get("agent_name")
            new_rating = data.get("new_rating")
            rating_change = data.get("rating_change", 0)
            domain = data.get("domain", "general")

            if not agent_name or new_rating is None:
                logger.warning("ELO → Debate: Missing agent_name or new_rating")
                return

            # Update team selector weights
            try:
                from aragora.debate.team_selector import TeamSelector

                selector = TeamSelector()
                if hasattr(selector, "update_agent_weight"):
                    selector.update_agent_weight(
                        agent=agent_name,
                        rating=new_rating,
                        change=rating_change,
                        domain=domain,
                    )
                    logger.debug(
                        f"Updated team selection weight for {agent_name}: "
                        f"{new_rating} ({'+' if rating_change >= 0 else ''}{rating_change})"
                    )
            except ImportError:
                logger.debug("TeamSelector not available for ELO updates")
            except Exception as e:
                logger.debug(f"TeamSelector update failed: {e}")

            self.stats["elo_to_debate"]["events"] += 1

            # Record SLO
            if check_and_record_slo:
                latency_ms = (time.time() - start) * 1000
                check_and_record_slo("elo_to_debate", latency_ms)

        except Exception as e:
            logger.error(f"ELO → Debate handler error: {e}")
            self.stats["elo_to_debate"]["errors"] += 1

    def _handle_calibration_to_agent(self, event: StreamEvent) -> None:
        """Handle Calibration → Agent events.

        Updates agent confidence based on calibration results.
        """
        try:
            from aragora.observability.metrics.slo import check_and_record_slo
        except ImportError:
            check_and_record_slo = None

        start = time.time()

        try:
            data = event.data
            agent_name = data.get("agent_name")
            calibration_score = data.get("calibration_score")
            overconfidence = data.get("overconfidence", 0)
            sample_size = data.get("sample_size", 0)

            if not agent_name or calibration_score is None:
                return

            # Update agent confidence scaling
            try:
                from aragora.debate.confidence import ConfidenceManager

                manager = ConfidenceManager()
                if hasattr(manager, "update_agent_calibration"):
                    manager.update_agent_calibration(
                        agent=agent_name,
                        score=calibration_score,
                        overconfidence=overconfidence,
                        sample_size=sample_size,
                    )
                    logger.debug(
                        f"Updated calibration for {agent_name}: "
                        f"score={calibration_score:.2f}, overconfidence={overconfidence:.2f}"
                    )
            except ImportError:
                logger.debug("ConfidenceManager not available")
            except Exception as e:
                logger.debug(f"Calibration update failed: {e}")

            self.stats["calibration_to_agent"]["events"] += 1

            # Record SLO
            if check_and_record_slo:
                latency_ms = (time.time() - start) * 1000
                check_and_record_slo("calibration_to_agent", latency_ms)

        except Exception as e:
            logger.error(f"Calibration → Agent handler error: {e}")
            self.stats["calibration_to_agent"]["errors"] += 1

    def _handle_culture_to_debate(self, event: StreamEvent) -> None:
        """Handle Culture → Debate events.

        Applies organizational debate patterns.
        """
        if not self._is_km_handler_enabled("culture_to_debate"):
            return

        try:
            data = event.data
            debate_id = data.get("debate_id")
            culture_hints = data.get("hints", {})
            workspace = data.get("workspace", "default")

            if not debate_id:
                return

            # Cache culture hints for the debate
            cache_key = f"{workspace}:{debate_id}"
            self._culture_cache[cache_key] = {
                "hints": culture_hints,
                "timestamp": time.time(),
            }

            logger.debug(f"Cached culture hints for debate {debate_id}")
            self.stats["culture_to_debate"]["events"] += 1

        except Exception as e:
            logger.error(f"Culture → Debate handler error: {e}")
            self.stats["culture_to_debate"]["errors"] += 1

    def _handle_mound_to_culture(self, event: StreamEvent) -> None:
        """Handle Knowledge Mound → Culture events.

        Retrieves organizational patterns for debate context.
        """
        if not self._is_km_handler_enabled("mound_to_culture"):
            return

        try:
            data = event.data
            debate_id = data.get("debate_id")
            task = data.get("task")
            workspace = data.get("workspace", "default")

            if not task:
                return

            # Async retrieval of culture patterns
            async def retrieve_culture():
                try:
                    from aragora.knowledge.mound_core import KnowledgeMound

                    mound = KnowledgeMound(workspace=workspace)
                    if hasattr(mound, "get_culture_patterns"):
                        patterns = await mound.get_culture_patterns(task)
                        if patterns and debate_id:
                            cache_key = f"{workspace}:{debate_id}"
                            self._culture_cache[cache_key] = {
                                "hints": patterns,
                                "timestamp": time.time(),
                            }
                except Exception as e:
                    logger.debug(f"Culture retrieval failed: {e}")

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(retrieve_culture())
                else:
                    loop.run_until_complete(retrieve_culture())
            except RuntimeError:
                # No event loop, skip async operation
                pass

            self.stats["mound_to_culture"]["events"] += 1

        except Exception as e:
            logger.error(f"Mound → Culture handler error: {e}")
            self.stats["mound_to_culture"]["errors"] += 1

    def _store_debate_culture(
        self,
        debate_id: str,
        task: str,
        patterns: dict,
        workspace: str = "default",
    ) -> None:
        """Store debate culture patterns for later retrieval.

        Args:
            debate_id: Debate identifier
            task: Debate task/topic
            patterns: Culture patterns to store
            workspace: Workspace identifier
        """
        cache_key = f"{workspace}:{debate_id}"
        self._culture_cache[cache_key] = {
            "hints": patterns,
            "task": task,
            "timestamp": time.time(),
        }
        logger.debug(f"Stored culture patterns for debate {debate_id}")

    def get_debate_culture_hints(self, debate_id: str, workspace: str = "default") -> dict:
        """Get cached culture hints for a debate.

        Args:
            debate_id: Debate identifier
            workspace: Workspace identifier

        Returns:
            Culture hints dict or empty dict if not found/expired
        """
        cache_key = f"{workspace}:{debate_id}"
        entry = self._culture_cache.get(cache_key)

        if not entry:
            return {}

        # Check TTL
        if time.time() - entry["timestamp"] > self._culture_cache_ttl:
            del self._culture_cache[cache_key]
            return {}

        return entry.get("hints", {})

    def _handle_staleness_to_debate(self, event: StreamEvent) -> None:
        """Handle Staleness → Debate events.

        Triggers knowledge refresh when stale data detected.
        """
        if not self._is_km_handler_enabled("staleness_to_debate"):
            return

        try:
            data = event.data
            node_ids = data.get("stale_nodes", [])
            workspace = data.get("workspace", "default")

            if not node_ids:
                return

            # Debounce to avoid refresh storms
            debounce_key = f"{workspace}:staleness"
            last_check = self._staleness_debounce.get(debounce_key, 0)
            now = time.time()

            if now - last_check < self._staleness_debounce_seconds:
                logger.debug("Staleness check debounced")
                return

            self._staleness_debounce[debounce_key] = now

            # Trigger refresh for stale nodes
            try:
                from aragora.knowledge.mound_core import KnowledgeMound

                mound = KnowledgeMound(workspace=workspace)
                if hasattr(mound, "refresh_nodes"):
                    mound.refresh_nodes(node_ids)
                    logger.debug(f"Triggered refresh for {len(node_ids)} stale nodes")
            except Exception as e:
                logger.debug(f"Staleness refresh failed: {e}")

            self.stats["staleness_to_debate"]["events"] += 1

        except Exception as e:
            logger.error(f"Staleness → Debate handler error: {e}")
            self.stats["staleness_to_debate"]["errors"] += 1

    def _handle_provenance_to_mound(self, event: StreamEvent) -> None:
        """Handle Provenance → Knowledge Mound events.

        Syncs claim provenance data for cross-debate verification.
        """
        if not self._is_km_handler_enabled("provenance_to_mound"):
            return

        try:
            data = event.data
            claim_id = data.get("claim_id")
            sources = data.get("sources", [])
            verification_status = data.get("verification_status")
            workspace = data.get("workspace", "default")

            if not claim_id:
                return

            try:
                from aragora.knowledge.mound_core import KnowledgeMound

                mound = KnowledgeMound(workspace=workspace)
                if hasattr(mound, "update_provenance"):
                    mound.update_provenance(
                        claim_id=claim_id,
                        sources=sources,
                        verification_status=verification_status,
                    )
                    logger.debug(f"Updated provenance for claim {claim_id}")
            except Exception as e:
                logger.debug(f"Provenance update failed: {e}")

            self.stats["provenance_to_mound"]["events"] += 1

        except Exception as e:
            logger.error(f"Provenance → Mound handler error: {e}")
            self.stats["provenance_to_mound"]["errors"] += 1

    def _handle_mound_to_provenance(self, event: StreamEvent) -> None:
        """Handle Knowledge Mound → Provenance events.

        Provides historical provenance data for claim verification.
        """
        if not self._is_km_handler_enabled("mound_to_provenance"):
            return

        try:
            data = event.data
            claim_text = data.get("claim_text")
            debate_id = data.get("debate_id")
            workspace = data.get("workspace", "default")

            if not claim_text:
                return

            try:
                from aragora.knowledge.mound_core import KnowledgeMound

                mound = KnowledgeMound(workspace=workspace)
                if hasattr(mound, "find_claim_provenance"):
                    provenance = mound.find_claim_provenance(claim_text)
                    if provenance:
                        logger.debug(f"Found provenance for claim in debate {debate_id}")
            except Exception as e:
                logger.debug(f"Provenance lookup failed: {e}")

            self.stats["mound_to_provenance"]["events"] += 1

        except Exception as e:
            logger.error(f"Mound → Provenance handler error: {e}")
            self.stats["mound_to_provenance"]["errors"] += 1
