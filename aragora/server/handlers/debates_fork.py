"""
Fork and follow-up debate operations handler mixin.

Extracted from debates.py for modularity. Provides counterfactual forking,
outcome verification, and crux-based follow-up debate creation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aragora.server.handlers.base import (
    error_response,
    json_response,
    HandlerResult,
    handle_errors,
    require_storage,
    safe_error_message,
)

if TYPE_CHECKING:
    from aragora.server.handlers.debates import DebatesHandler

logger = logging.getLogger(__name__)


class ForkOperationsMixin:
    """Mixin providing fork and follow-up operations for DebatesHandler."""

    @require_storage
    def _fork_debate(self: "DebatesHandler", handler, debate_id: str) -> HandlerResult:
        """Create a counterfactual fork of a debate at a specific branch point.

        Request body:
            {
                "branch_point": int,  # Round number to branch from
                "modified_context": str  # Optional: context for the counterfactual
            }

        Returns:
            Information about the created branch
        """
        from aragora.server.validation import FORK_REQUEST_SCHEMA, validate_against_schema

        # Read and validate request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid or missing JSON body", 400)

        validation = validate_against_schema(body, FORK_REQUEST_SCHEMA)
        if not validation.is_valid:
            return error_response(validation.error, 400)

        branch_point = body.get("branch_point", 0)
        modified_context = body.get("modified_context")

        # Get the original debate
        storage = self.get_storage()
        try:
            debate = storage.get_debate(debate_id)
            if not debate:
                return error_response(f"Debate not found: {debate_id}", 404)

            messages = debate.get("messages", [])
            if branch_point > len(messages):
                return error_response(
                    f"Branch point {branch_point} exceeds message count {len(messages)}",
                    400
                )

            # Import counterfactual module
            try:
                from aragora.debate.counterfactual import (
                    CounterfactualOrchestrator,
                    PivotClaim,
                    CounterfactualBranch,
                )
            except ImportError:
                return error_response("Counterfactual module not available", 503)

            # Create a pivot claim from the context
            import uuid as uuid_mod
            pivot = PivotClaim(
                claim_id=f"pivot-{uuid_mod.uuid4().hex[:8]}",
                statement=modified_context or f"Branch at round {branch_point}",
                author="user",
                disagreement_score=1.0,
                importance_score=1.0,
                blocking_agents=[],
                branch_reason=f"User-initiated fork at round {branch_point}",
            )

            # Create the branch record
            branch_id = f"fork-{debate_id}-r{branch_point}-{uuid_mod.uuid4().hex[:8]}"

            branch = CounterfactualBranch(
                branch_id=branch_id,
                parent_debate_id=debate_id,
                pivot_claim=pivot,
                assumption=True,  # Default to exploring the "true" branch
                messages=messages[:branch_point] if branch_point > 0 else [],
            )

            # Store the branch info
            branch_data = {
                "branch_id": branch_id,
                "parent_debate_id": debate_id,
                "branch_point": branch_point,
                "modified_context": modified_context,
                "pivot_claim": pivot.statement,
                "status": "created",
                "messages_inherited": branch_point,
            }

            # Try to store in nomic dir
            nomic_dir = self.get_nomic_dir()
            if nomic_dir:
                import json as json_mod
                branches_dir = nomic_dir / "branches"
                branches_dir.mkdir(exist_ok=True)
                branch_file = branches_dir / f"{branch_id}.json"
                with open(branch_file, "w") as f:
                    json_mod.dump(branch_data, f, indent=2)

            return json_response({
                "success": True,
                "branch_id": branch_id,
                "parent_debate_id": debate_id,
                "branch_point": branch_point,
                "messages_inherited": branch_point,
                "modified_context": modified_context,
                "status": "created",
                "message": f"Created fork '{branch_id}' from debate '{debate_id}' at round {branch_point}",
            })

        except Exception as e:
            logger.error("Failed to create fork for %s at round %s: %s: %s", debate_id, branch_point, type(e).__name__, e, exc_info=True)
            return error_response(safe_error_message(e, "create fork"), 500)

    @handle_errors("verify debate outcome")
    def _verify_outcome(self: "DebatesHandler", handler, debate_id: str) -> HandlerResult:
        """Record verification of whether a debate's winning position was correct.

        POST body:
            correct: bool - whether the winning position was actually correct
            source: str - verification source (default: "manual")

        Completes the truth-grounding feedback loop by linking positions to outcomes.
        """
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid or missing JSON body", 400)

        correct = body.get('correct', False)
        source = body.get('source', 'manual')

        # Get position tracker from context
        position_tracker = self.ctx.get("position_tracker")

        if position_tracker:
            position_tracker.record_verification(debate_id, correct, source)
            return json_response({
                "status": "verified",
                "debate_id": debate_id,
                "correct": correct,
                "source": source,
            })

        # Try to create a temporary tracker
        try:
            from aragora.agents.truth_grounding import PositionTracker
            nomic_dir = self.get_nomic_dir()
            if nomic_dir:
                db_path = nomic_dir / "aragora_positions.db"
                if db_path.exists():
                    tracker = PositionTracker(db_path=str(db_path))
                    tracker.record_verification(debate_id, correct, source)
                    return json_response({
                        "status": "verified",
                        "debate_id": debate_id,
                        "correct": correct,
                        "source": source,
                    })
            return error_response("Position tracking not configured", 503)
        except ImportError:
            return error_response("PositionTracker module not available", 503)

    @require_storage
    def _get_followup_suggestions(self: "DebatesHandler", debate_id: str) -> HandlerResult:
        """Get follow-up debate suggestions based on identified cruxes.

        Analyzes the debate's uncertainty metrics and generates suggestions
        for follow-up debates to resolve key disagreement points.

        Returns:
            List of follow-up suggestions with priority and suggested task
        """
        storage = self.get_storage()

        try:
            debate = storage.get_debate(debate_id)
            if not debate:
                return error_response(f"Debate not found: {debate_id}", 404)

            # Import uncertainty analysis components
            from aragora.uncertainty.estimator import DisagreementAnalyzer, DisagreementCrux

            # Extract cruxes from debate data
            cruxes = []

            # Check for stored uncertainty metrics
            uncertainty = debate.get("uncertainty_metrics", {})
            if uncertainty and "cruxes" in uncertainty:
                for crux_data in uncertainty["cruxes"]:
                    crux = DisagreementCrux(
                        description=crux_data.get("description", ""),
                        divergent_agents=crux_data.get("agents", []),
                        evidence_needed=crux_data.get("evidence_needed", ""),
                        severity=crux_data.get("severity", 0.5),
                    )
                    cruxes.append(crux)

            # If no stored cruxes, analyze the debate
            if not cruxes:
                from aragora.core import Vote, Message
                votes_data = debate.get("votes", [])
                messages_data = debate.get("messages", [])
                proposals = debate.get("proposals", {})

                # Convert to Vote/Message objects if needed
                votes = []
                for v in votes_data:
                    if hasattr(v, 'choice'):
                        votes.append(v)
                    else:
                        votes.append(Vote(
                            agent=v.get("agent", "unknown"),
                            choice=v.get("choice", ""),
                            confidence=v.get("confidence", 0.5),
                            reasoning=v.get("reasoning", ""),
                        ))

                messages = []
                for m in messages_data:
                    if hasattr(m, 'content'):
                        messages.append(m)
                    else:
                        messages.append(Message(
                            agent=m.get("agent", "unknown"),
                            content=m.get("content", ""),
                            role=m.get("role", "proposer"),
                            round=m.get("round", 1),
                        ))

                # Run disagreement analysis
                analyzer = DisagreementAnalyzer()
                metrics = analyzer.analyze_disagreement(messages, votes, proposals)
                cruxes = metrics.cruxes

            if not cruxes:
                return json_response({
                    "debate_id": debate_id,
                    "suggestions": [],
                    "message": "No significant disagreement cruxes identified",
                })

            # Generate follow-up suggestions
            analyzer = DisagreementAnalyzer()
            available_agents = debate.get("agents", [])
            suggestions = analyzer.suggest_followups(
                cruxes=cruxes,
                parent_debate_id=debate_id,
                available_agents=available_agents,
            )

            return json_response({
                "debate_id": debate_id,
                "suggestions": [s.to_dict() for s in suggestions],
                "count": len(suggestions),
            })

        except Exception as e:
            logger.error("Failed to get followup suggestions for %s: %s: %s", debate_id, type(e).__name__, e, exc_info=True)
            return error_response(safe_error_message(e, "get followup suggestions"), 500)

    @require_storage
    def _create_followup_debate(self: "DebatesHandler", handler, debate_id: str) -> HandlerResult:
        """Create a follow-up debate to resolve a specific crux.

        POST body:
            crux_id: str - ID of the crux to explore
            task: str (optional) - Custom task, otherwise generated from crux
            agents: list[str] (optional) - Specific agents to include

        Returns:
            Created debate metadata with parent lineage
        """
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid or missing JSON body", 400)

        crux_id = body.get("crux_id")
        custom_task = body.get("task")
        requested_agents = body.get("agents", [])

        if not crux_id and not custom_task:
            return error_response("Either crux_id or task is required", 400)

        storage = self.get_storage()

        try:
            # Get parent debate
            parent_debate = storage.get_debate(debate_id)
            if not parent_debate:
                return error_response(f"Parent debate not found: {debate_id}", 404)

            # Find the crux if crux_id provided
            crux_data = None
            if crux_id:
                uncertainty = parent_debate.get("uncertainty_metrics", {})
                for c in uncertainty.get("cruxes", []):
                    if c.get("id") == crux_id:
                        crux_data = c
                        break

                if not crux_data:
                    return error_response(f"Crux not found: {crux_id}", 404)

            # Generate task
            if custom_task:
                task = custom_task
            elif crux_data:
                # Generate task from crux description
                from aragora.uncertainty.estimator import DisagreementAnalyzer, DisagreementCrux
                analyzer = DisagreementAnalyzer()
                crux = DisagreementCrux(
                    description=crux_data.get("description", ""),
                    divergent_agents=crux_data.get("agents", []),
                    severity=crux_data.get("severity", 0.5),
                )
                task = analyzer._generate_followup_task(crux)
            else:
                return error_response("Could not generate task", 400)

            # Determine agents
            if requested_agents:
                agents = requested_agents
            elif crux_data:
                agents = crux_data.get("agents", [])
                # Add parent debate agents if not enough
                if len(agents) < 2:
                    for agent in parent_debate.get("agents", []):
                        if agent not in agents:
                            agents.append(agent)
                        if len(agents) >= 3:
                            break
            else:
                agents = parent_debate.get("agents", [])[:3]

            # Create unique ID for follow-up debate
            import time
            import hashlib
            followup_id = f"followup-{debate_id[:8]}-{int(time.time()) % 100000}"

            # Store follow-up debate metadata
            followup_data = {
                "id": followup_id,
                "task": task,
                "agents": agents,
                "parent_debate_id": debate_id,
                "crux_id": crux_id,
                "crux_description": crux_data.get("description") if crux_data else None,
                "status": "pending",
                "created_at": time.time(),
            }

            # Store in nomic dir
            nomic_dir = self.get_nomic_dir()
            if nomic_dir:
                import json as json_mod
                followups_dir = nomic_dir / "followups"
                followups_dir.mkdir(exist_ok=True)
                followup_file = followups_dir / f"{followup_id}.json"
                with open(followup_file, "w") as f:
                    json_mod.dump(followup_data, f, indent=2)

            logger.info(f"Created follow-up debate {followup_id} from parent {debate_id}")

            return json_response({
                "success": True,
                "followup_id": followup_id,
                "parent_debate_id": debate_id,
                "task": task,
                "agents": agents,
                "crux_id": crux_id,
                "status": "pending",
                "message": f"Created follow-up debate to explore: {task[:100]}",
            })

        except Exception as e:
            logger.error("Failed to create followup debate for %s: %s: %s", debate_id, type(e).__name__, e, exc_info=True)
            return error_response(safe_error_message(e, "create followup debate"), 500)


__all__ = ["ForkOperationsMixin"]
