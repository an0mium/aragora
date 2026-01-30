"""
Onboarding Orchestrator - User Journey and Activation Flows.

Provides onboarding flow management with step-based journeys,
data collection, verification, and conversion tracking.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, cast

from .models import (
    ChannelType,
    OnboardingFlow,
    OnboardingSession,
    OnboardingStep,
)

logger = logging.getLogger(__name__)


class OnboardingOrchestrator:
    """
    Orchestrator for user onboarding flows.

    Manages multi-step onboarding journeys with data collection,
    verification, branching logic, and conversion analytics.
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        """
        Initialize the onboarding orchestrator.

        Args:
            storage_path: Path for flow and session storage
        """
        self._storage_path = Path(storage_path) if storage_path else None
        self._flows: dict[str, OnboardingFlow] = {}
        self._sessions: dict[str, OnboardingSession] = {}
        self._validators: dict[str, Callable] = {}
        self._step_handlers: dict[str, Callable] = {}
        self._lock = asyncio.Lock()

        if self._storage_path:
            self._storage_path.mkdir(parents=True, exist_ok=True)

        # Register default validators
        self._register_default_validators()

    def _register_default_validators(self) -> None:
        """Register default field validators."""

        def validate_email(value: str) -> bool:
            return "@" in value and "." in value

        def validate_phone(value: str) -> bool:
            digits = "".join(c for c in value if c.isdigit())
            return len(digits) >= 10

        def validate_required(value: Any) -> bool:
            return bool(value)

        self._validators["email"] = validate_email
        self._validators["phone"] = validate_phone
        self._validators["required"] = validate_required

    # ========== Flow Management ==========

    async def create_flow(
        self,
        name: str,
        description: str = "",
        steps: list[OnboardingStep] | None = None,
        target_segment: str | None = None,
        channels: list[ChannelType] | None = None,
    ) -> OnboardingFlow:
        """
        Create a new onboarding flow.

        Args:
            name: Flow name
            description: Flow description
            steps: List of onboarding steps
            target_segment: Target user segment
            channels: Supported channels

        Returns:
            Created flow
        """
        async with self._lock:
            flow_id = str(uuid.uuid4())

            flow = OnboardingFlow(
                id=flow_id,
                name=name,
                description=description,
                steps=steps or [],
                target_segment=target_segment,
                channels=channels or [],
            )

            self._flows[flow_id] = flow
            logger.info(f"Created onboarding flow: {name}")

            return flow

    async def get_flow(self, flow_id: str) -> OnboardingFlow | None:
        """Get a flow by ID."""
        return self._flows.get(flow_id)

    async def list_flows(
        self,
        status: str | None = None,
        target_segment: str | None = None,
    ) -> list[OnboardingFlow]:
        """List flows with optional filters."""
        flows = list(self._flows.values())

        if status:
            flows = [f for f in flows if f.status == status]
        if target_segment:
            flows = [f for f in flows if f.target_segment == target_segment]

        return flows

    async def add_step(
        self,
        flow_id: str,
        name: str,
        step_type: str,
        content: dict[str, Any] | None = None,
        required: bool = True,
        validation: dict[str, Any] | None = None,
        next_step: str | None = None,
        branch_conditions: dict[str, str] | None = None,
    ) -> OnboardingStep | None:
        """
        Add a step to a flow.

        Args:
            flow_id: Target flow
            name: Step name
            step_type: Type (info, input, verification, action, decision)
            content: Step content/configuration
            required: Whether step is required
            validation: Validation rules
            next_step: Default next step ID
            branch_conditions: Conditional branching rules

        Returns:
            Created step
        """
        async with self._lock:
            flow = self._flows.get(flow_id)
            if not flow:
                return None

            step_id = str(uuid.uuid4())
            order = len(flow.steps)

            step = OnboardingStep(
                id=step_id,
                name=name,
                type=cast(Literal["info", "input", "verification", "action", "decision"], step_type),
                content=content or {},
                required=required,
                order=order,
                validation=validation,
                next_step=next_step,
                branch_conditions=branch_conditions or {},
            )

            flow.steps.append(step)
            flow.updated_at = datetime.utcnow()

            return step

    async def activate_flow(self, flow_id: str) -> OnboardingFlow | None:
        """Activate a flow for use."""
        async with self._lock:
            flow = self._flows.get(flow_id)
            if not flow:
                return None

            if not flow.steps:
                raise ValueError("Cannot activate flow with no steps")

            flow.status = "active"
            flow.updated_at = datetime.utcnow()

            logger.info(f"Activated onboarding flow: {flow.name}")
            return flow

    async def archive_flow(self, flow_id: str) -> OnboardingFlow | None:
        """Archive a flow."""
        async with self._lock:
            flow = self._flows.get(flow_id)
            if not flow:
                return None

            flow.status = "archived"
            flow.updated_at = datetime.utcnow()

            return flow

    # ========== Session Management ==========

    async def start_session(
        self,
        flow_id: str,
        user_id: str,
        channel_id: str,
        tenant_id: str | None = None,
        initial_data: dict[str, Any] | None = None,
    ) -> OnboardingSession:
        """
        Start an onboarding session for a user.

        Args:
            flow_id: Flow to use
            user_id: User ID
            channel_id: Channel ID
            tenant_id: Tenant ID
            initial_data: Pre-filled data

        Returns:
            Created session
        """
        async with self._lock:
            flow = self._flows.get(flow_id)
            if not flow:
                raise ValueError(f"Flow {flow_id} not found")

            if flow.status != "active":
                raise ValueError(f"Flow {flow_id} is not active")

            session_id = str(uuid.uuid4())

            session = OnboardingSession(
                id=session_id,
                flow_id=flow_id,
                user_id=user_id,
                channel_id=channel_id,
                tenant_id=tenant_id,
                started_at=datetime.utcnow(),
                collected_data=initial_data or {},
            )

            # Set first step
            if flow.steps:
                session.current_step = flow.steps[0].id

            self._sessions[session_id] = session
            flow.started_count += 1

            logger.info(f"Started onboarding session {session_id} for user {user_id}")

            return session

    async def get_session(self, session_id: str) -> OnboardingSession | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    async def list_sessions(
        self,
        flow_id: str | None = None,
        user_id: str | None = None,
        status: str | None = None,
        tenant_id: str | None = None,
    ) -> list[OnboardingSession]:
        """List sessions with optional filters."""
        sessions = list(self._sessions.values())

        if flow_id:
            sessions = [s for s in sessions if s.flow_id == flow_id]
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        if status:
            sessions = [s for s in sessions if s.status == status]
        if tenant_id:
            sessions = [s for s in sessions if s.tenant_id == tenant_id]

        return sessions

    async def get_current_step(
        self,
        session_id: str,
    ) -> tuple[OnboardingStep | None, dict[str, Any]]:
        """
        Get the current step for a session.

        Returns:
            Tuple of (step, context for rendering)
        """
        session = self._sessions.get(session_id)
        if not session or not session.current_step:
            return None, {}

        flow = self._flows.get(session.flow_id)
        if not flow:
            return None, {}

        step = next((s for s in flow.steps if s.id == session.current_step), None)
        if not step:
            return None, {}

        context = {
            "session_id": session_id,
            "user_id": session.user_id,
            "collected_data": session.collected_data,
            "completed_steps": session.completed_steps,
            "progress": len(session.completed_steps) / len(flow.steps) if flow.steps else 0,
        }

        return step, context

    # ========== Step Progression ==========

    async def submit_step(
        self,
        session_id: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Submit data for the current step and advance.

        Args:
            session_id: Session ID
            data: Submitted data

        Returns:
            Result with next step info or completion status
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return {"success": False, "error": "Session not found"}

            if session.status != "in_progress":
                return {"success": False, "error": f"Session is {session.status}"}

            flow = self._flows.get(session.flow_id)
            if not flow:
                return {"success": False, "error": "Flow not found"}

            current_step = next((s for s in flow.steps if s.id == session.current_step), None)
            if not current_step:
                return {"success": False, "error": "Current step not found"}

            # Validate data
            validation_result = await self._validate_step_data(current_step, data)
            if not validation_result["valid"]:
                # Track retry
                retries = session.retries.get(current_step.id, 0) + 1
                session.retries[current_step.id] = retries

                if current_step.retry_limit and retries >= current_step.retry_limit:
                    return {
                        "success": False,
                        "error": "Maximum retries exceeded",
                        "validation_errors": validation_result["errors"],
                    }

                return {
                    "success": False,
                    "error": "Validation failed",
                    "validation_errors": validation_result["errors"],
                    "retries_remaining": (
                        current_step.retry_limit - retries if current_step.retry_limit else None
                    ),
                }

            # Store collected data
            session.collected_data.update(data)
            session.step_data[current_step.id] = data
            session.completed_steps.append(current_step.id)

            # Execute step handler if registered
            handler = self._step_handlers.get(current_step.type)
            if handler:
                try:
                    await handler(session, current_step, data)
                except Exception as e:
                    logger.error(f"Step handler error: {e}")

            # Determine next step
            next_step_id = self._determine_next_step(current_step, data, flow)

            if next_step_id:
                session.current_step = next_step_id
                next_step = next((s for s in flow.steps if s.id == next_step_id), None)
                session.updated_at = datetime.utcnow()

                return {
                    "success": True,
                    "completed_step": current_step.id,
                    "next_step": {
                        "id": next_step_id,
                        "name": next_step.name if next_step else None,
                        "type": next_step.type if next_step else None,
                        "content": next_step.content if next_step else None,
                    },
                    "progress": len(session.completed_steps) / len(flow.steps),
                }
            else:
                # Flow completed
                return await self._complete_session(session, flow)

    def _determine_next_step(
        self,
        current_step: OnboardingStep,
        data: dict[str, Any],
        flow: OnboardingFlow,
    ) -> str | None:
        """Determine the next step based on conditions."""
        # Check branch conditions
        for condition_key, target_step in current_step.branch_conditions.items():
            # Simple condition format: "field:value"
            if ":" in condition_key:
                field, expected_value = condition_key.split(":", 1)
                if str(data.get(field)) == expected_value:
                    return target_step

        # Use default next step if specified
        if current_step.next_step:
            return current_step.next_step

        # Find next step by order
        current_idx = next((i for i, s in enumerate(flow.steps) if s.id == current_step.id), -1)
        if current_idx >= 0 and current_idx + 1 < len(flow.steps):
            return flow.steps[current_idx + 1].id

        return None

    async def _validate_step_data(
        self,
        step: OnboardingStep,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate step data against validation rules."""
        if not step.validation:
            return {"valid": True, "errors": []}

        errors = []

        for field, rules in step.validation.items():
            value = data.get(field)

            if isinstance(rules, str):
                rules = [rules]

            if isinstance(rules, list):
                for rule in rules:
                    validator = self._validators.get(rule)
                    if validator and not validator(value):
                        errors.append(f"Field '{field}' failed {rule} validation")
            elif isinstance(rules, dict):
                if rules.get("required") and not value:
                    errors.append(f"Field '{field}' is required")
                if rules.get("type") == "email":
                    validator = self._validators.get("email")
                    if validator and value and not validator(value):
                        errors.append(f"Field '{field}' must be a valid email")

        return {"valid": len(errors) == 0, "errors": errors}

    async def _complete_session(
        self,
        session: OnboardingSession,
        flow: OnboardingFlow,
    ) -> dict[str, Any]:
        """Complete an onboarding session."""
        session.status = "completed"
        session.completed_at = datetime.utcnow()
        session.updated_at = datetime.utcnow()
        session.current_step = None

        flow.completed_count += 1

        logger.info(f"Completed onboarding session {session.id}")

        return {
            "success": True,
            "completed": True,
            "collected_data": session.collected_data,
            "verification_status": session.verification_status,
        }

    async def abandon_session(
        self,
        session_id: str,
        reason: str = "",
    ) -> OnboardingSession | None:
        """Mark a session as abandoned."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            session.status = "abandoned"
            session.updated_at = datetime.utcnow()
            session.metadata["abandon_reason"] = reason

            flow = self._flows.get(session.flow_id)
            if flow:
                flow.abandoned_count += 1

            logger.info(f"Abandoned onboarding session {session_id}: {reason}")
            return session

    async def pause_session(self, session_id: str) -> OnboardingSession | None:
        """Pause a session for later resumption."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            session.status = "paused"
            session.updated_at = datetime.utcnow()

            return session

    async def resume_session(self, session_id: str) -> OnboardingSession | None:
        """Resume a paused session."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            if session.status != "paused":
                return session

            session.status = "in_progress"
            session.updated_at = datetime.utcnow()

            return session

    # ========== Handlers and Validators ==========

    def register_validator(
        self,
        name: str,
        validator: Callable,
    ) -> None:
        """Register a custom field validator."""
        self._validators[name] = validator
        logger.info(f"Registered validator: {name}")

    def register_step_handler(
        self,
        step_type: str,
        handler: Callable,
    ) -> None:
        """Register a handler for a step type."""
        self._step_handlers[step_type] = handler
        logger.info(f"Registered step handler: {step_type}")

    # ========== Analytics ==========

    async def get_flow_stats(self, flow_id: str) -> dict[str, Any] | None:
        """Get statistics for a flow."""
        flow = self._flows.get(flow_id)
        if not flow:
            return None

        sessions = [s for s in self._sessions.values() if s.flow_id == flow_id]

        completion_rate = flow.completed_count / flow.started_count if flow.started_count > 0 else 0
        abandon_rate = flow.abandoned_count / flow.started_count if flow.started_count > 0 else 0

        # Step completion rates
        step_stats = {}
        for step in flow.steps:
            completed = sum(1 for s in sessions if step.id in s.completed_steps)
            step_stats[step.id] = {
                "name": step.name,
                "completed": completed,
                "rate": completed / len(sessions) if sessions else 0,
            }

        return {
            "flow_id": flow_id,
            "name": flow.name,
            "status": flow.status,
            "started": flow.started_count,
            "completed": flow.completed_count,
            "abandoned": flow.abandoned_count,
            "completion_rate": completion_rate,
            "abandon_rate": abandon_rate,
            "step_stats": step_stats,
        }

    async def get_stats(self) -> dict[str, Any]:
        """Get overall onboarding statistics."""
        async with self._lock:
            active_flows = sum(1 for f in self._flows.values() if f.status == "active")

            total_started = sum(f.started_count for f in self._flows.values())
            total_completed = sum(f.completed_count for f in self._flows.values())
            total_abandoned = sum(f.abandoned_count for f in self._flows.values())

            in_progress = sum(1 for s in self._sessions.values() if s.status == "in_progress")

            return {
                "flows_total": len(self._flows),
                "flows_active": active_flows,
                "sessions_total": len(self._sessions),
                "sessions_in_progress": in_progress,
                "total_started": total_started,
                "total_completed": total_completed,
                "total_abandoned": total_abandoned,
                "overall_completion_rate": (
                    total_completed / total_started if total_started > 0 else 0
                ),
                "validators_registered": len(self._validators),
                "step_handlers_registered": len(self._step_handlers),
            }
