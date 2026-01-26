"""
Validation and consensus event handlers.

Handles consensus ingestion and validation feedback:
- Consensus → KM: Ingest consensus content with dissent tracking
- Provenance ↔ KM: Store/query verification chains
- KM Validation Feedback: Improve source quality based on debate outcomes
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from aragora.events.types import StreamEvent

# Import metrics stubs - will be overwritten if metrics available
try:
    from aragora.server.prometheus_cross_pollination import (
        record_km_inbound_event,
        record_km_outbound_event,
    )
except ImportError:

    def record_km_inbound_event(source: str, event_type: str) -> None:
        pass

    def record_km_outbound_event(target: str, event_type: str) -> None:
        pass


logger = logging.getLogger(__name__)


class ValidationHandlersMixin:
    """Mixin providing validation and consensus event handlers."""

    # Required from parent: _is_km_handler_enabled method
    _is_km_handler_enabled: Callable[[str], bool]

    def _handle_provenance_to_mound(self, event: "StreamEvent") -> None:
        """
        Consensus reached → Store verified provenance chains.

        After debate consensus, store verified provenance chains in KM.
        """
        if not self._is_km_handler_enabled("provenance_to_mound"):
            return

        data = event.data
        debate_id = data.get("debate_id", "")
        consensus_reached = data.get("consensus_reached", False)

        if not consensus_reached:
            return

        logger.debug(f"Storing provenance chains from consensus in debate {debate_id}")

        # Record KM inbound metric
        record_km_inbound_event("provenance", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

            adapter = BeliefAdapter()

            # Store verified provenance chains
            chains = data.get("provenance_chains", [])
            for chain in chains:
                if chain.get("verified", False):
                    adapter.store_provenance(
                        chain_id=chain.get("id", ""),
                        source_id=chain.get("source_id", ""),
                        claim_ids=chain.get("claim_ids", []),
                        verified=True,
                        verification_method=chain.get("method", "consensus"),
                        debate_id=debate_id,
                    )

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Provenance→KM storage failed: {e}")

    def _handle_mound_to_provenance(self, event: "StreamEvent") -> None:
        """
        Claim verification → Query KM for verification history.

        When verifying claims, check KM for related verified chains.
        """
        if not self._is_km_handler_enabled("mound_to_provenance"):
            return

        data = event.data
        claim_id = data.get("claim_id", "")
        claim_text = data.get("claim", "")

        if not claim_text:
            return

        logger.debug(f"Querying KM for verification history: claim {claim_id}")

        # Record KM outbound metric
        record_km_outbound_event("provenance", event.type.value)

        try:
            from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

            adapter = BeliefAdapter()

            # Search for related verified claims
            related = adapter.search_similar_cruxes(
                query=claim_text,
                limit=5,
            )

            if related:
                logger.debug(f"Found {len(related)} related verified claims")

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"KM→Provenance query failed: {e}")

    def _handle_consensus_to_mound(self, event: "StreamEvent") -> None:
        """
        Consensus reached → Ingest consensus content to Knowledge Mound.

        After debate consensus, store the consensus conclusion, key claims,
        and dissenting views as knowledge nodes for organizational learning.

        Enhanced features:
        - Dissent tracking: Store dissenting views as separate nodes linked to consensus
        - Evolution tracking: Detect similar prior consensus and create supersedes links
        - Linking: Connect consensus to claims, evidence, and related knowledge
        """
        if not self._is_km_handler_enabled("consensus_to_mound"):
            return

        data = event.data
        debate_id = data.get("debate_id", "")
        consensus_reached = data.get("consensus_reached", False)

        if not consensus_reached:
            return

        topic = data.get("topic", "")
        conclusion = data.get("conclusion", "")
        confidence = data.get("confidence", 0.5)
        strength = data.get("strength", "moderate")
        key_claims = data.get("key_claims", [])
        supporting_evidence = data.get("supporting_evidence", [])
        domain = data.get("domain", "general")
        tags = data.get("tags", [])

        # Dissent data
        dissents = data.get("dissents", [])
        dissenting_agents = data.get("dissenting_agents", [])
        _dissent_ids = data.get("dissent_ids", [])  # Preserved for future linking

        # Evolution data
        supersedes = data.get("supersedes", None)
        agreeing_agents = data.get("agreeing_agents", [])
        participating_agents = data.get("participating_agents", [])

        if not topic and not conclusion:
            return

        logger.info(
            f"Ingesting consensus from debate {debate_id} to Knowledge Mound "
            f"(dissents={len(dissents)}, evolution={supersedes is not None})"
        )

        # Record KM inbound metric
        record_km_inbound_event("consensus", event.type.value)

        try:
            from aragora.knowledge.mound import get_knowledge_mound
            from aragora.knowledge.mound.types import IngestionRequest, KnowledgeSource

            mound = get_knowledge_mound()
            if not mound:
                logger.debug("Knowledge Mound not available for consensus ingestion")
                return

            # Check if mound is initialized
            if not mound.is_initialized:
                logger.debug("Knowledge Mound not initialized, skipping consensus ingestion")
                return

            # Build content from topic and conclusion
            content = f"{topic}: {conclusion}" if conclusion else topic

            # Map strength to tier
            strength_to_tier = {
                "unanimous": "glacial",  # Highly stable
                "strong": "slow",
                "moderate": "slow",
                "weak": "medium",
                "split": "medium",
                "contested": "fast",  # May change
            }
            tier = strength_to_tier.get(strength, "slow")

            # Calculate agreement ratio
            agreement_ratio = (
                len(agreeing_agents) / len(participating_agents) if participating_agents else 0.0
            )

            import asyncio

            async def ingest_consensus_with_enhancements():
                # ============================================================
                # EVOLUTION TRACKING: Check for similar prior consensus
                # ============================================================
                supersedes_node_id = None
                if supersedes:
                    # Direct supersedes reference provided
                    supersedes_node_id = f"cs_{supersedes}"
                else:
                    # Search for similar prior consensus on same topic
                    try:
                        similar_results = await mound.search(  # type: ignore[attr-defined]
                            query=topic,
                            node_types=["consensus"],
                            limit=3,
                            min_score=0.85,  # High threshold for "same topic"
                        )
                        if similar_results:
                            # Found similar prior consensus - this new one supersedes it
                            prior = similar_results[0]
                            prior_debate_id = prior.metadata.get("debate_id", "")
                            if prior_debate_id != debate_id:
                                supersedes_node_id = prior.id
                                logger.info(
                                    f"Consensus {debate_id} supersedes prior "
                                    f"consensus {prior_debate_id} on topic '{topic[:50]}...'"
                                )
                    except Exception as e:
                        logger.debug(f"Evolution tracking search failed: {e}")

                # ============================================================
                # MAIN CONSENSUS INGESTION
                # ============================================================
                request = IngestionRequest(  # type: ignore[call-arg]
                    content=content,
                    workspace_id=mound.workspace_id,
                    source_type=KnowledgeSource.CONSENSUS,
                    debate_id=debate_id,
                    node_type="consensus",
                    confidence=confidence,
                    tier=tier,
                    supersedes=supersedes_node_id,
                    metadata={
                        "debate_id": debate_id,
                        "strength": strength,
                        "topic": topic,
                        "conclusion": conclusion,
                        "domain": domain,
                        "tags": tags,
                        "key_claims_count": len(key_claims),
                        "dissent_count": len(dissents),
                        "agreement_ratio": agreement_ratio,
                        "agreeing_agents": agreeing_agents,
                        "dissenting_agents": dissenting_agents,
                        "participating_agents": participating_agents,
                        "has_dissent": len(dissents) > 0 or len(dissenting_agents) > 0,
                        "ingested_at": datetime.now().isoformat(),
                    },
                )

                result = await mound.store(request)  # type: ignore[misc]
                consensus_node_id = result.node_id

                logger.debug(
                    f"Ingested consensus {debate_id}: node_id={consensus_node_id}, "
                    f"deduplicated={result.deduplicated}, supersedes={supersedes_node_id}"
                )

                # ============================================================
                # DISSENT TRACKING: Store dissenting views
                # ============================================================
                dissent_node_ids = []
                for i, dissent in enumerate(dissents[:10]):  # Limit to 10 dissents
                    if isinstance(dissent, dict):
                        dissent_content = dissent.get("content", "")
                        dissent_type = dissent.get(
                            "type", dissent.get("dissent_type", "alternative_approach")
                        )
                        dissent_agent = dissent.get("agent_id", dissent.get("agent", "unknown"))
                        dissent_reasoning = dissent.get("reasoning", "")
                        dissent_confidence = dissent.get("confidence", 0.5)
                        acknowledged = dissent.get("acknowledged", False)
                        rebuttal = dissent.get("rebuttal", "")
                    elif isinstance(dissent, str):
                        dissent_content = dissent
                        dissent_type = "alternative_approach"
                        dissent_agent = (
                            dissenting_agents[i] if i < len(dissenting_agents) else "unknown"
                        )
                        dissent_reasoning = ""
                        dissent_confidence = 0.5
                        acknowledged = False
                        rebuttal = ""
                    else:
                        continue

                    if not dissent_content.strip():
                        continue

                    # Determine dissent importance based on type
                    dissent_importance = 0.5
                    if dissent_type == "risk_warning":
                        dissent_importance = 0.7  # Risk warnings are valuable
                    elif dissent_type == "fundamental_disagreement":
                        dissent_importance = 0.6  # Strong dissent worth preserving
                    elif dissent_type == "edge_case_concern":
                        dissent_importance = 0.55  # Edge cases inform future debates

                    dissent_request = IngestionRequest(
                        content=f"[DISSENT from {dissent_agent}] {dissent_content}",
                        workspace_id=mound.workspace_id,
                        source_type=KnowledgeSource.CONSENSUS,
                        debate_id=debate_id,
                        node_type="dissent",
                        confidence=dissent_confidence,
                        tier="medium",  # Dissents may be reconsidered
                        derived_from=[consensus_node_id] if consensus_node_id else None,
                        metadata={
                            "debate_id": debate_id,
                            "dissent_type": dissent_type,
                            "agent_id": dissent_agent,
                            "reasoning": dissent_reasoning,
                            "acknowledged": acknowledged,
                            "rebuttal": rebuttal,
                            "parent_consensus_id": consensus_node_id,
                            "dissent_index": i,
                            "topic": topic,
                            "is_risk_warning": dissent_type == "risk_warning",
                            "importance": dissent_importance,
                        },
                    )

                    dissent_result = await mound.store(dissent_request)  # type: ignore[misc]
                    if dissent_result.node_id:
                        dissent_node_ids.append(dissent_result.node_id)
                        logger.debug(
                            f"Stored dissent from {dissent_agent}: "
                            f"type={dissent_type}, node_id={dissent_result.node_id}"
                        )

                if dissent_node_ids:
                    logger.info(
                        f"Stored {len(dissent_node_ids)} dissenting views for consensus {debate_id}"
                    )

                # ============================================================
                # CLAIM LINKING: Store key claims linked to consensus
                # ============================================================
                claim_node_ids = []
                for i, claim in enumerate(key_claims[:10]):  # Limit to 10 claims
                    if isinstance(claim, str) and claim.strip():
                        claim_request = IngestionRequest(
                            content=claim,
                            workspace_id=mound.workspace_id,
                            source_type=KnowledgeSource.CONSENSUS,
                            debate_id=debate_id,
                            node_type="claim",
                            confidence=confidence * 0.9,  # Slightly lower than main consensus
                            tier=tier,
                            derived_from=[consensus_node_id] if consensus_node_id else None,
                            metadata={
                                "debate_id": debate_id,
                                "claim_index": i,
                                "parent_consensus_id": consensus_node_id,
                                "domain": domain,
                            },
                        )
                        claim_result = await mound.store(claim_request)  # type: ignore[misc]
                        if claim_result.node_id:
                            claim_node_ids.append(claim_result.node_id)

                # ============================================================
                # EVIDENCE LINKING: Store supporting evidence references
                # ============================================================
                for i, evidence in enumerate(supporting_evidence[:5]):  # Limit evidence
                    if isinstance(evidence, str) and evidence.strip():
                        evidence_request = IngestionRequest(
                            content=evidence,
                            workspace_id=mound.workspace_id,
                            source_type=KnowledgeSource.CONSENSUS,
                            debate_id=debate_id,
                            node_type="evidence",
                            confidence=confidence * 0.85,
                            tier=tier,
                            derived_from=[consensus_node_id] if consensus_node_id else None,
                            metadata={
                                "debate_id": debate_id,
                                "evidence_index": i,
                                "parent_consensus_id": consensus_node_id,
                                "supports_conclusion": True,
                            },
                        )
                        await mound.store(evidence_request)

                # ============================================================
                # UPDATE SUPERSEDED NODE (if applicable)
                # ============================================================
                if supersedes_node_id and hasattr(mound, "update_metadata"):
                    try:
                        await mound.update_metadata(
                            node_id=supersedes_node_id,
                            updates={"superseded_by": consensus_node_id},
                        )
                        logger.debug(
                            f"Marked {supersedes_node_id} as superseded by {consensus_node_id}"
                        )
                    except Exception as e:
                        logger.debug(f"Failed to update superseded node: {e}")

                # Log summary
                logger.info(
                    f"Consensus ingestion complete for debate {debate_id}: "
                    f"consensus={consensus_node_id}, claims={len(claim_node_ids)}, "
                    f"dissents={len(dissent_node_ids)}, "
                    f"supersedes={'yes' if supersedes_node_id else 'no'}"
                )

            # Run async ingestion
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(ingest_consensus_with_enhancements())
                else:
                    loop.run_until_complete(ingest_consensus_with_enhancements())
            except RuntimeError:
                # No event loop, create one
                asyncio.run(ingest_consensus_with_enhancements())

        except ImportError as e:
            logger.debug(f"Consensus→KM ingestion import failed: {e}")
        except Exception as e:
            logger.warning(f"Consensus→KM ingestion failed: {e}")

    def _handle_km_validation_feedback(self, event: "StreamEvent") -> None:
        """
        KM Validation Feedback: Improve source system quality based on debate outcomes.

        When consensus is reached, this handler:
        1. Queries KM for items that may have contributed to the debate
        2. For items from ContinuumMemory or ConsensusMemory that match the topic:
           - If consensus was reached with high confidence → positive validation
           - If consensus contradicts prior knowledge → negative validation
        3. Feeds validation back to source adapters to improve quality scores

        This creates a learning loop where KM data that proves useful in debates
        gets promoted (higher tiers, higher importance), while contradicted data
        gets demoted or flagged for review.
        """
        if not self._is_km_handler_enabled("km_validation_feedback"):
            return

        data = event.data
        debate_id = data.get("debate_id", "")
        consensus_reached = data.get("consensus_reached", False)
        confidence = data.get("confidence", 0.5)
        topic = data.get("topic", "")

        # Only process debates with clear outcomes
        if not consensus_reached or confidence < 0.5 or not topic:
            return

        logger.debug(
            f"Processing KM validation feedback for debate {debate_id}: "
            f"confidence={confidence:.2f}, topic={topic[:50]}..."
        )

        try:
            import asyncio

            from aragora.knowledge.mound import get_knowledge_mound
            from aragora.knowledge.mound.adapters.continuum_adapter import (
                ContinuumAdapter,
                KMValidationResult,
            )
            from aragora.knowledge.mound.adapters.consensus_adapter import (  # noqa: F401
                ConsensusAdapter,
            )

            mound = get_knowledge_mound()
            if not mound:
                logger.debug("Knowledge Mound not available for validation feedback")
                return

            # Check if mound is initialized
            if not mound.is_initialized:
                logger.debug("Knowledge Mound not initialized, skipping validation feedback")
                return

            async def process_validation_feedback():
                # Query KM for items that may have contributed to this debate
                try:
                    # Search for related knowledge by topic
                    results = await mound.search(
                        query=topic,
                        limit=20,
                        min_score=0.6,  # Moderate threshold for potential contributors
                    )

                    if not results:
                        logger.debug(f"No KM items found for validation feedback: {topic[:50]}")
                        return

                    continuum_validations = 0
                    consensus_validations = 0

                    for result in results:
                        node_id = (
                            result.node_id
                            if hasattr(result, "node_id")
                            else result.get("node_id", "")
                        )
                        score = (
                            result.score if hasattr(result, "score") else result.get("score", 0.0)
                        )
                        source = (
                            result.source if hasattr(result, "source") else result.get("source", "")
                        )

                        # Determine validation recommendation based on outcome
                        # High confidence + high similarity = item was useful
                        cross_debate_utility = score * confidence

                        if confidence >= 0.8 and score >= 0.7:
                            recommendation = "promote"
                        elif confidence >= 0.6 and score >= 0.5:
                            recommendation = "keep"
                        elif confidence < 0.5:
                            recommendation = "review"
                        else:
                            recommendation = "keep"

                        # Create validation result
                        validation = KMValidationResult(
                            memory_id=node_id,
                            km_confidence=confidence,
                            cross_debate_utility=cross_debate_utility,
                            validation_count=1,
                            was_supported=consensus_reached and confidence >= 0.7,
                            was_contradicted=False,  # Would need contradiction detection
                            recommendation=recommendation,
                            metadata={
                                "debate_id": debate_id,
                                "topic": topic[:100],
                                "similarity_score": score,
                                "source_type": source,
                            },
                        )

                        # Route validation to appropriate adapter
                        if node_id.startswith("cm_"):
                            # ContinuumMemory item
                            try:
                                from aragora.memory.continuum import get_continuum_memory

                                continuum = get_continuum_memory()
                                if continuum and hasattr(continuum, "_km_adapter"):
                                    adapter = continuum._km_adapter
                                    if adapter and isinstance(adapter, ContinuumAdapter):
                                        updated = await adapter.update_continuum_from_km(
                                            memory_id=node_id,
                                            km_validation=validation,
                                        )
                                        if updated:
                                            continuum_validations += 1
                            except ImportError:
                                pass
                            except Exception as e:
                                logger.debug(f"Continuum validation failed: {e}")

                        elif node_id.startswith("cs_"):
                            # Consensus item - track but consensus records are immutable
                            # Instead, update the confidence tracking for the adapter
                            consensus_validations += 1

                    if continuum_validations > 0 or consensus_validations > 0:
                        logger.info(
                            f"KM validation feedback for debate {debate_id}: "
                            f"continuum={continuum_validations}, consensus={consensus_validations}"
                        )

                        # Emit validation event for dashboard
                        try:
                            from aragora.events.types import (
                                StreamEvent,
                                StreamEventType,
                            )

                            validation_event = StreamEvent(
                                type=StreamEventType.KM_ADAPTER_VALIDATION,
                                data={
                                    "debate_id": debate_id,
                                    "topic_preview": topic[:50],
                                    "confidence": confidence,
                                    "continuum_validations": continuum_validations,
                                    "consensus_validations": consensus_validations,
                                    "total_items_reviewed": len(results),
                                },
                            )
                            # Don't dispatch to avoid recursion - just log for now
                            logger.debug(f"Validation event: {validation_event.data}")
                        except Exception as e:
                            logger.debug(f"Failed to create validation event: {e}")

                except Exception as e:
                    logger.warning(f"KM validation feedback query failed: {e}")

            # Run async validation
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(process_validation_feedback())
                else:
                    loop.run_until_complete(process_validation_feedback())
            except RuntimeError:
                asyncio.run(process_validation_feedback())

        except ImportError as e:
            logger.debug(f"KM validation feedback import failed: {e}")
        except Exception as e:
            logger.warning(f"KM validation feedback failed: {e}")
