"""
Protocol definitions for Moltbot manager interfaces.

This module provides Protocol classes that define the expected interfaces
for moltbot managers, enabling proper type checking without circular imports.

Usage:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from aragora.extensions.moltbot.protocols import (
            GatewayProtocol,
            InboxProtocol,
            VoiceProcessorProtocol,
            CanvasManagerProtocol,
            OnboardingOrchestratorProtocol,
            CapabilityMatcherProtocol,
        )
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class GatewayProtocol(Protocol):
    """Protocol for device gateway management.

    Defines the interface for device registration, heartbeats,
    command routing, and gateway statistics.
    """

    async def register_device(
        self,
        config: Any,
        user_id: str,
        tenant_id: Optional[str] = None,
    ) -> Any:
        """Register a new device with the gateway."""
        ...

    async def get_device(self, device_id: str) -> Optional[Any]:
        """Get a device by ID."""
        ...

    async def list_devices(
        self,
        user_id: Optional[str] = None,
        device_type: Optional[str] = None,
        status: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> list[Any]:
        """List devices with optional filters."""
        ...

    async def unregister_device(self, device_id: str) -> bool:
        """Unregister a device from the gateway."""
        ...

    async def send_command(
        self,
        device_id: str,
        command: str,
        payload: Optional[dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Send a command to a device."""
        ...

    async def heartbeat(
        self,
        device_id: str,
        state: Optional[dict[str, Any]] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Process a device heartbeat."""
        ...

    async def get_stats(self) -> dict[str, Any]:
        """Get gateway statistics."""
        ...


@runtime_checkable
class InboxProtocol(Protocol):
    """Protocol for message and channel management.

    Defines the interface for channel registration, message
    sending/receiving, threading, and inbox statistics.
    """

    async def register_channel(
        self,
        config: Any,
        user_id: str,
        tenant_id: Optional[str] = None,
    ) -> Any:
        """Register a new communication channel."""
        ...

    async def get_channel(self, channel_id: str) -> Optional[Any]:
        """Get a channel by ID."""
        ...

    async def list_channels(
        self,
        user_id: Optional[str] = None,
        channel_type: Optional[Any] = None,
        tenant_id: Optional[str] = None,
    ) -> list[Any]:
        """List channels with optional filters."""
        ...

    async def unregister_channel(self, channel_id: str) -> bool:
        """Unregister a channel."""
        ...

    async def send_message(
        self,
        channel_id: str,
        user_id: str,
        content: str,
        content_type: str = "text",
        thread_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Send an outbound message through a channel."""
        ...

    async def get_message(self, message_id: str) -> Optional[Any]:
        """Get a message by ID."""
        ...

    async def list_messages(
        self,
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        direction: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Any]:
        """List messages with optional filters."""
        ...

    async def mark_read(self, message_id: str) -> Optional[Any]:
        """Mark a message as read."""
        ...

    async def get_thread(self, thread_id: str) -> list[Any]:
        """Get all messages in a thread."""
        ...

    async def get_stats(self) -> dict[str, Any]:
        """Get inbox statistics."""
        ...


@runtime_checkable
class VoiceProcessorProtocol(Protocol):
    """Protocol for voice session management.

    Defines the interface for voice session lifecycle,
    speech-to-text, text-to-speech, and transcription.
    """

    async def create_session(
        self,
        config: Any,
        user_id: str,
        channel_id: str,
        tenant_id: Optional[str] = None,
    ) -> Any:
        """Create a new voice session."""
        ...

    async def get_session(self, session_id: str) -> Optional[Any]:
        """Get a voice session by ID."""
        ...

    async def list_sessions(
        self,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        status: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> list[Any]:
        """List voice sessions with optional filters."""
        ...

    async def end_session(
        self,
        session_id: str,
        reason: str = "completed",
    ) -> Optional[Any]:
        """End a voice session."""
        ...

    async def pause_session(self, session_id: str) -> Optional[Any]:
        """Pause a voice session."""
        ...

    async def resume_session(self, session_id: str) -> Optional[Any]:
        """Resume a paused voice session."""
        ...

    async def process_audio(
        self,
        session_id: str,
        audio_data: bytes,
    ) -> dict[str, Any]:
        """Process audio data (Speech-to-Text)."""
        ...

    async def synthesize_speech(
        self,
        session_id: str,
        text: str,
    ) -> dict[str, Any]:
        """Synthesize speech from text (Text-to-Speech)."""
        ...

    async def get_transcript(self, session_id: str) -> list[dict[str, Any]]:
        """Get the full transcript for a session."""
        ...

    async def get_stats(self) -> dict[str, Any]:
        """Get voice processor statistics."""
        ...


@runtime_checkable
class CanvasManagerProtocol(Protocol):
    """Protocol for canvas collaboration management.

    Defines the interface for canvas CRUD operations,
    element management, and real-time collaboration.
    """

    async def create_canvas(
        self,
        config: Any,
        owner_id: str,
        tenant_id: Optional[str] = None,
    ) -> Any:
        """Create a new canvas."""
        ...

    async def get_canvas(self, canvas_id: str) -> Optional[Any]:
        """Get a canvas by ID."""
        ...

    async def list_canvases(
        self,
        owner_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> list[Any]:
        """List canvases with optional filters."""
        ...

    async def delete_canvas(self, canvas_id: str) -> bool:
        """Delete a canvas and all its elements."""
        ...

    async def add_element(
        self,
        canvas_id: str,
        element_type: Any,
        x: float,
        y: float,
        width: float,
        height: float,
        properties: Optional[dict[str, Any]] = None,
        layer_id: Optional[str] = None,
    ) -> Optional[Any]:
        """Add an element to a canvas."""
        ...

    async def update_element(
        self,
        canvas_id: str,
        element_id: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        rotation: Optional[float] = None,
        z_index: Optional[int] = None,
        properties: Optional[dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Update an element's properties."""
        ...

    async def remove_element(
        self,
        canvas_id: str,
        element_id: str,
    ) -> bool:
        """Remove an element from a canvas."""
        ...

    async def add_collaborator(
        self,
        canvas_id: str,
        user_id: str,
        permission: str = "view",
    ) -> bool:
        """Add a collaborator to a canvas."""
        ...

    async def remove_collaborator(
        self,
        canvas_id: str,
        user_id: str,
    ) -> bool:
        """Remove a collaborator from a canvas."""
        ...

    async def export_canvas(
        self,
        canvas_id: str,
        format: str = "json",
    ) -> dict[str, Any]:
        """Export canvas data."""
        ...


@runtime_checkable
class OnboardingOrchestratorProtocol(Protocol):
    """Protocol for onboarding flow management.

    Defines the interface for onboarding flow creation,
    session management, and step progression.
    """

    async def create_flow(
        self,
        name: str,
        description: str = "",
        is_active: bool = True,
        tenant_id: Optional[str] = None,
    ) -> Any:
        """Create a new onboarding flow."""
        ...

    async def get_flow(self, flow_id: str) -> Optional[Any]:
        """Get a flow by ID."""
        ...

    async def list_flows(
        self,
        active_only: bool = False,
        tenant_id: Optional[str] = None,
    ) -> list[Any]:
        """List flows with optional filters."""
        ...

    async def update_flow(
        self,
        flow_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[Any]:
        """Update an onboarding flow."""
        ...

    async def delete_flow(self, flow_id: str) -> bool:
        """Delete an onboarding flow."""
        ...

    async def add_step(
        self,
        flow_id: str,
        step_type: str,
        title: str,
        content: str = "",
        is_required: bool = True,
        timeout_seconds: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Add a step to a flow."""
        ...

    async def list_sessions(
        self,
        flow_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[Any]:
        """List sessions with optional filters."""
        ...

    async def start_session(
        self,
        flow_id: str,
        user_id: str,
        device_id: Optional[str] = None,
    ) -> Optional[Any]:
        """Start a new onboarding session."""
        ...

    async def get_session(self, session_id: str) -> Optional[Any]:
        """Get a session by ID."""
        ...

    async def advance_session(
        self,
        session_id: str,
        response: Optional[dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Advance session to next step."""
        ...

    async def complete_session(self, session_id: str) -> Optional[Any]:
        """Complete an onboarding session."""
        ...

    async def skip_step(self, session_id: str) -> Optional[Any]:
        """Skip current step in session."""
        ...


@runtime_checkable
class CapabilityMatcherProtocol(Protocol):
    """Protocol for device capability matching.

    Defines the interface for capability queries,
    device capability detection, and capability matrix.
    """

    async def list_capabilities(
        self,
        category: Optional[str] = None,
    ) -> list[Any]:
        """List all available capabilities."""
        ...

    async def get_capability(self, capability_name: str) -> Optional[Any]:
        """Get capability details."""
        ...

    async def get_dependents(self, capability_name: str) -> list[str]:
        """Get capabilities that depend on this one."""
        ...

    async def get_device_capabilities(self, device_id: str) -> Optional[Any]:
        """Get capabilities for a specific device."""
        ...

    async def check_capability(
        self,
        device_id: str,
        capability_name: str,
    ) -> dict[str, Any]:
        """Check if device has specific capability."""
        ...

    async def get_capability_matrix(
        self,
        tenant_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get capability matrix for devices."""
        ...


__all__ = [
    "GatewayProtocol",
    "InboxProtocol",
    "VoiceProcessorProtocol",
    "CanvasManagerProtocol",
    "OnboardingOrchestratorProtocol",
    "CapabilityMatcherProtocol",
]
